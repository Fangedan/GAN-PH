# Wasserstein GAN with gradient penalty
# Code for training of WGAN-gp
#
# MODIFICATION (4_CNNCT research):
#   Added _loss_connectivity() — a differentiable pore connectivity penalty
#   that addresses pore phase collapse. Two findings motivated this:
#
#   1. loss_vf operates on softmax probabilities, not argmax voxels.
#      A uniform pore_prob=0.3 satisfies the VF loss while argmax produces
#      zero actual pore voxels, so the GAN exploits this shortcut.
#
#   2. loss_ssa has a broken gradient path (torch.no_grad + .detach() +
#      .requires_grad_() severs the graph) and acts as monitoring only.
#      This is documented below but left unchanged to preserve training stability.
#
#   The fix adds two differentiable components to G_loss:
#     - Isolation penalty: penalizes isolated pore voxels (no pore neighbors)
#     - Face hinge loss:   penalizes solid dominating over pore at z=0 and z=63
#                         (necessary condition for through-thickness percolation)
#
#   New hyperparameter: w_conn = 50 (tunable)
#   See _loss_connectivity() docstring for full explanation.

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids Tkinter crash in background processes
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F          # NEW: needed for conv3d and relu
from torch import autograd


class Trainer():
    def __init__(self, model_g, optim_g, model_c, optim_c, model_e, epochs, device, dataloader, in_header,
                 tau_net=None, tau_targets=None):

        self.generator  = model_g       # Generator
        self.opt_g      = optim_g       # Optimizer for Generator
        self.critic     = model_c       # Critic
        self.opt_c      = optim_c       # Optimizer for Critic
        self.estimator  = model_e       # Estimator for specific surface area
        self.n_epoch    = epochs        # Number of epochs
        self.device     = device
        self.dataloader = dataloader    # Dataloader
        self.header     = in_header     # Header of input data

        # τ-net surrogate for tortuosity loss (optional; None = loss disabled)
        self.tau_net     = tau_net
        self.tau_targets = tau_targets or {}   # {'Ni': float, 'YSZ': float, 'Pore': float}
        if tau_net is not None:
            # Freeze weights — gradients flow THROUGH tau_net to g_data (no no_grad here),
            # but tau_net parameters themselves are never updated by the generator optimizer.
            for p in tau_net.parameters():
                p.requires_grad_(False)
            tau_net.eval()

        self.losses = {
            "G_loss":            [],
            "D_loss":            [],
            "gp_loss":           [],
            "Wasser_d":          [],
            "vf_loss":           [],
            "ssa_loss":          [],
            "conn_loss":         [],   # pore connectivity penalty
            "conn_ysz_loss":     [],   # YSZ min-slice density proxy
            "conn_ysz_face_loss":[],   # YSZ face density at z=0 and z=63 (run4)
            "tau_loss":          [],   # tortuosity surrogate loss (empty when tau_net=None)
        }

        # ----- Hyper Parameters ----- #
        self.w_gp        = 20      # Weight for gradient penalty
        self.w_param     = 1000    # Weight for volume fraction and specific surface area
        self.n_critic    = 1       # Number of critic iterations per generator update
        self.save_epoch  = 1       # Save model per epoch
        self.timing      = 9999    # Epoch at which SSA loss is added to G_loss
        self.latent_size = 100     # Latent noise vector size

        # NEW: Pore connectivity loss weight
        #
        # Calibration (collapsed case):
        #   face hinge  ≈ 0.475 per voxel → w_conn * 0.475 ≈ 24   (at w_conn=50)
        #   VF loss     ≈ 0.01            → w_param * 0.01 ≈ 10
        #   So connectivity contributes ~2× VF — strong enough to prevent collapse
        #   while not overwhelming the adversarial signal.
        #
        # Decrease w_conn (e.g. 20) if training becomes unstable.
        # Increase w_conn (e.g. 100) if pore still collapses after retraining.
        self.w_conn = 50

        # YSZ percolation proxy loss weight.
        # Raw loss magnitude ≤ threshold (0.1) since it's a mean of ReLU clips.
        # w_conn_ysz=200 → maximum contribution ≈ 20 to G_loss.
        # Real YSZ vf ≈ 0.20, so threshold=0.10 fires when any z-slice drops
        # below 50% of expected density — a generous safety margin.
        self.w_conn_ysz = 200

        # YSZ face-hinge loss weight (run4 addition).
        # Requires mean YSZ probability at z=0 AND z=63 faces ≥ 0.18.
        # Distinct from min-slice density (0.10 across all 64 slices): that loss barely
        # fired in run2 because YSZ met 10% mean as disconnected blobs. This loss
        # specifically targets the ENDPOINTS where taufactor's z-direction solve must
        # enter/exit. Max contribution: 200 * (2 * 0.18) = 72 to G_loss.
        self.w_conn_ysz_face = 200

        # Tortuosity surrogate loss weight.
        # If τ_pred MSE ≈ 0.25 (half a τ unit off per phase), then
        # w_tau=50 contributes ~12 to G_loss — similar scale to connectivity.
        # Tune up if tau S-values don't improve; tune down if training destabilizes.
        # tau_timing: epoch at which τ loss is added (let structure form first).
        self.w_tau      = 50
        self.tau_timing = 10

    # ── Original methods (unchanged) ──────────────────────────────────────────

    def _get_min_max(self):
        """
        Function of getting minimum and maximum value of specific surface area
        """
        path = os.path.join(self.header, "results.dat")
        df = pd.read_table(path, sep="\s+")
        mm_list = [[df["SV{}".format(i)].min(), df["SV{}".format(i)].max()] for i in range(3)]
        return mm_list

    def _sample_generator(self, vf, ssa, b_len):
        """
        Function of generating structure data
        :param vf    : volume fraction
        :param ssa   : specific surface area
        :param b_len : length of minibatch
        """
        noise  = torch.randn(b_len, self.latent_size, 4, 4, 4, device=self.device)
        g_data = self.generator(noise, vf, ssa)
        return g_data

    def _plot_losses(self, epoch, itr):
        """
        Function of plotting losses — updated to include conn_loss
        :param epoch : number of epoch
        :param itr   : number of iteration
        """
        conn     = self.losses["conn_loss"][-1]     if self.losses["conn_loss"]     else 0.0
        conn_ysz      = self.losses["conn_ysz_loss"][-1]      if self.losses["conn_ysz_loss"]      else 0.0
        conn_ysz_face = self.losses["conn_ysz_face_loss"][-1] if self.losses["conn_ysz_face_loss"] else 0.0
        tau           = self.losses["tau_loss"][-1]            if self.losses["tau_loss"]           else 0.0

        print("[{:03}/{:03}][{:03}/{:03}] G_loss: {:.04f} D_loss: {:.04f} "
              "Wasser_D: {:.04f} conn: {:.04f} conn_ysz: {:.04f} conn_ysz_face: {:.04f} tau: {:.04f}".format(
            epoch + 1, self.n_epoch, itr + 1, len(self.dataloader),
            self.losses["G_loss"][-1], self.losses["D_loss"][-1],
            self.losses["Wasser_d"][-1], conn, conn_ysz, conn_ysz_face, tau
        ))

        file = open("./log.dat", "a")
        file.write(
            "[{:03}/{:03}][{:03}/{:03}] G_loss: {:.04f} D_loss: {:.04f} "
            "Wasser_D: {:.04f} vf_loss: {:.04} ssa_loss: {:.04} "
            "gp_loss: {:.04} conn_loss: {:.04} conn_ysz_loss: {:.04} conn_ysz_face_loss: {:.04} tau_loss: {:.04}\n".format(
                epoch + 1, self.n_epoch, itr + 1, len(self.dataloader),
                self.losses["G_loss"][-1], self.losses["D_loss"][-1],
                self.losses["Wasser_d"][-1], self.losses["vf_loss"][-1],
                self.losses["ssa_loss"][-1], self.losses["gp_loss"][-1],
                conn, conn_ysz, conn_ysz_face, tau
            )
        )
        file.close()
        return

    def _gradient_penalty(self, r_data, g_data):
        """
        Function of calculating gradient penalty
        :param r_data : real structure data
        :param g_data : generated structure data
        """
        is_cuda = torch.cuda.is_available()
        b_len   = r_data.shape[0]
        epsilon = torch.rand(b_len, 1, 1, 1, 1).to(self.device)

        interpolated_img = epsilon * r_data + (1 - epsilon) * g_data
        interpolated_out = self.critic(interpolated_img)

        grads = autograd.grad(
            outputs=interpolated_out,
            inputs=interpolated_img,
            grad_outputs=(torch.ones(interpolated_out.shape).cuda()
                          if is_cuda else torch.ones(interpolated_out.shape)),
            create_graph=True,
            retain_graph=True
        )[0]

        grads       = grads.reshape([b_len, -1])
        grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        self.losses["gp_loss"].append(grad_penalty.item())
        return grad_penalty

    def _loss_vf(self, b_len, g_data, in_vf, n_size=64):
        """
        Function of calculating loss of volume fraction
        :param b_len  : length of minibatch
        :param g_data : generated structure data  (softmax probabilities)
        :param in_vf  : input volume fraction
        :param n_size : voxel size of structure data
        """
        total_loss = 0.
        mass = n_size ** 3

        for i in range(b_len):
            struc = g_data[i]
            vf    = in_vf[i]

            Ni_loss   = torch.square(torch.sum(struc[0, :, :, :]) / mass - vf[0, 0, 0, 0])
            YSZ_loss  = torch.square(torch.sum(struc[1, :, :, :]) / mass - vf[1, 0, 0, 0])
            Pore_loss = torch.square(torch.sum(struc[2, :, :, :]) / mass - vf[2, 0, 0, 0])
            total_loss += Ni_loss + YSZ_loss + Pore_loss

        vf_loss = total_loss / b_len
        self.losses["vf_loss"].append(vf_loss)
        return vf_loss

    def _phase_pickup(self, images, phase):
        """
        Function of picking up specific phase from structure data
        :param images : structure data
        :param phase  : specific phase
        """
        if phase == 'Ni':
            imgs = images[:, 0]
        elif phase == 'YSZ':
            imgs = images[:, 1]
        elif phase == 'Pore':
            imgs = images[:, 2]
        imgs = imgs.reshape(imgs.shape[0], 1, imgs.shape[1], imgs.shape[2], imgs.shape[3])
        return imgs

    def _preprocess(self, structure):
        """
        Function of preprocessing structure data for the SSA estimator
        :param structure : structure data
        """
        name       = ["Ni", "YSZ", "Pore"]
        struc_list = [self._phase_pickup(structure, i) for i in name]
        struc_input = torch.cat(struc_list, dim=0)
        return struc_input

    def _standardize(self, raw_ssa, mm_list):
        """
        Function of standardizing specific surface area
        :param raw_ssa : raw specific surface area
        :param mm_list : minimum and maximum value of specific surface area
        """
        length     = len(raw_ssa) // 3
        stand_list = []
        for i in range(3):
            raw         = raw_ssa[(i * length):((i + 1) * length)]
            stand_value = (raw - mm_list[i][0]) / (mm_list[i][1] - mm_list[i][0])
            stand_list.append(stand_value)
        stand = torch.cat(stand_list, dim=0)
        return stand

    def _loss_ssa(self, b_len, g_data, in_ssa, mm_list):
        """
        Function of calculating loss of specific surface area.

        NOTE — KNOWN BUG (monitoring only, not fixing):
        The combination of torch.no_grad() + .detach().clone() + .requires_grad_()
        severs the computational graph. The .requires_grad_() call creates a leaf
        tensor with no parents, so G_loss.backward() sends zero gradient through
        this term to the generator parameters.
        This means ssa_loss is purely a monitoring metric — it does not train
        the generator. Only loss_vf and g_loss (adversarial) actually update
        the generator weights.
        Fixing this would require removing no_grad and detach, which significantly
        changes training dynamics. Left unchanged to preserve original behavior;
        the new _loss_connectivity() addresses the key failure mode instead.

        :param b_len  : length of minibatch
        :param g_data : generated structure data
        :param in_ssa : input specific surface area
        :param mm_list: minimum and maximum value of specific surface area
        """
        self.estimator = self.estimator.eval()
        with torch.no_grad():
            g_data   = g_data.to(self.device)
            pred_raw = self.estimator(self._preprocess(g_data)).detach().clone()
            pred     = self._standardize(pred_raw, mm_list)
        true_raw = self._preprocess(in_ssa).to(self.device)
        true     = true_raw[:, 0, 0, 0, 0].detach().clone()

        loss    = torch.sum(torch.square(true - pred)).requires_grad_()
        ssa_loss = loss / b_len
        self.losses["ssa_loss"].append(ssa_loss)
        return ssa_loss

    # ── NEW METHOD ────────────────────────────────────────────────────────────

    def _loss_connectivity(self, g_data):
        """
        NEW: Differentiable pore connectivity penalty.

        Addresses the pore phase collapse where the GAN generates structures with
        near-zero actual pore voxels despite meeting the VF target, because loss_vf
        operates on softmax probabilities (not argmax). A uniform pore_prob=0.3 across
        all voxels satisfies loss_vf while argmax assigns nearly all voxels to Ni or YSZ.

        Two fully differentiable components:

        COMPONENT 1 — Isolation penalty
        --------------------------------
        For each voxel: penalize high pore probability combined with low pore
        probability in the 6 face-adjacent neighbors.

        Uses a 3D convolution with a 6-connectivity kernel to sum pore probability
        in each voxel's neighbors, then computes:

            isolation = pore_prob * (1 - neighbor_pore_sum / 6)

        This is zero when:
          - pore_prob = 0  (no pore to isolate)
          - all 6 neighbors are pore (well-connected, percolating cluster)
        This is high when:
          - pore_prob is high AND all neighbors have zero pore probability (fully isolated)

        Gradient incentivizes the generator to cluster pore voxels into connected
        networks rather than distributing them as isolated points.

        COMPONENT 2 — Face hinge loss
        --------------------------------
        At the z=0 (hydrogen entry) and z=63 (exit) faces of the electrode,
        pore must be present for gas to percolate through the structure.
        This is a necessary condition for through-thickness percolation.

        Hinge loss: penalize when solid (max of Ni and YSZ probability) exceeds
        pore probability by more than a margin at either face:

            loss_face = ReLU(solid - pore + margin)  averaged over the face

        margin=0.05 means pore must beat solid by at least 5 probability units.

        This is CRITICAL for the collapsed case: when pore≈0 everywhere, the
        isolation penalty is also ≈0 (nothing to isolate). The face hinge provides
        a non-zero gradient pointing toward increasing pore at the faces, which
        bootstraps pore into existence before the isolation penalty can take over.

        Collapsed case example:
          pore_prob=0.02, solid=0.55 → hinge = ReLU(0.55-0.02+0.05) = 0.58 (strong)
        Fixed case example:
          pore_prob=0.60, solid=0.25 → hinge = ReLU(0.25-0.60+0.05) = 0.00 (silent)

        Gradient flow:
        Both components compute differentiable operations on g_data (slicing, F.conv3d,
        element-wise multiplication, F.relu). Gradients flow back through g_data to
        the generator parameters during G_loss.backward(). The convolution kernel is
        a fixed constant tensor (requires_grad=False) so only the input gradient is
        computed, not a kernel gradient.

        :param g_data: (b, 3, 64, 64, 64) — softmax probabilities, channels=[Ni, YSZ, Pore]
        :return: scalar connectivity loss, range roughly [0, 1]
        """
        pore = g_data[:, 2:3, :, :, :]   # (b, 1, 64, 64, 64)  pore channel
        ni   = g_data[:, 0:1, :, :, :]   # (b, 1, 64, 64, 64)  Ni channel
        ysz  = g_data[:, 1:2, :, :, :]   # (b, 1, 64, 64, 64)  YSZ channel

        # ── Component 1: Isolation penalty ────────────────────────────────────
        # 6-connectivity kernel: sum pore probability in the 6 face-adjacent neighbors.
        # No center voxel (we measure neighbors only, not the voxel itself).
        kernel = torch.zeros(1, 1, 3, 3, 3, device=g_data.device)
        kernel[0, 0, 1, 1, 0] = 1   # z − 1
        kernel[0, 0, 1, 1, 2] = 1   # z + 1
        kernel[0, 0, 1, 0, 1] = 1   # y − 1
        kernel[0, 0, 1, 2, 1] = 1   # y + 1
        kernel[0, 0, 0, 1, 1] = 1   # x − 1
        kernel[0, 0, 2, 1, 1] = 1   # x + 1
        # kernel does not require grad — only pore (the input) accumulates gradients

        # neighbor_pore[b, 0, z, y, x] = sum of pore probs in 6 face neighbors
        # range: [0, 6]
        neighbor_pore      = F.conv3d(pore, kernel, padding=1)
        neighbor_pore_norm = neighbor_pore / 6.0   # normalize to [0, 1]

        # isolation: max when pore_prob=1 AND all neighbors have pore_prob=0
        # isolation = 0 when pore_prob=0 or when all neighbors are also pore
        isolation     = pore * (1.0 - neighbor_pore_norm)
        loss_isolation = isolation.mean()

        # ── Component 2: Face hinge loss ───────────────────────────────────────
        # pore must be present (and winning) at the electrode entry and exit faces.
        margin = 0.05
        solid  = torch.max(ni, ysz)   # solid phase probability at each voxel

        loss_face_z0 = F.relu(solid[:, :,  0, :, :] - pore[:, :,  0, :, :] + margin).mean()
        loss_face_z1 = F.relu(solid[:, :, -1, :, :] - pore[:, :, -1, :, :] + margin).mean()
        loss_face    = (loss_face_z0 + loss_face_z1) * 0.5

        # ── Combined ───────────────────────────────────────────────────────────
        loss_conn = loss_isolation + loss_face
        self.losses["conn_loss"].append(loss_conn.item())
        return loss_conn

    def _loss_connectivity_ysz(self, g_data):
        """
        Differentiable YSZ percolation proxy.

        YSZ must be present at every z-slice for a percolating path to exist
        from z=0 to z=63. When any z-slice has near-zero YSZ, taufactor
        returns NaN (unconverged) and the tau_net gradient carries no useful
        signal for those samples.

        Loss: for each (batch, z-slice), compute mean YSZ probability across
        the YX plane, then penalize slices below threshold via hinge/ReLU:

            loss = mean over (batch, z) of ReLU(threshold - mean_ysz_yx)

        threshold=0.10 fires when a z-slice has less than 10% mean YSZ
        probability (real YSZ vf ≈ 0.20, so this allows 50% drop before
        firing, i.e. a gentle nudge rather than a hard constraint).

        :param g_data: (B, 3, 64, 64, 64) softmax probabilities [Ni, YSZ, Pore]
        :return: scalar loss in [0, threshold]
        """
        ysz = g_data[:, 1, :, :, :]              # (B, Z, Y, X)
        slice_means = ysz.mean(dim=(2, 3))         # (B, Z) — mean YSZ per z-slice
        threshold   = 0.10
        loss = F.relu(threshold - slice_means).mean()
        self.losses["conn_ysz_loss"].append(loss.item())
        return loss

    def _loss_connectivity_ysz_face(self, g_data):
        """
        YSZ face-hinge loss (run4).

        The min-slice density loss (_loss_connectivity_ysz, threshold=0.10) barely
        fired in run2: YSZ met 10% mean density at every z-slice as disconnected blobs,
        but taufactor needs a PERCOLATING path from z=0 to z=63. This loss targets the
        two endpoint faces specifically, where YSZ must be present for any z-direction
        path to exist.

        Loss: penalise when mean YSZ probability at the entry face (z=0) or exit face
        (z=63) falls below 0.18 (90% of real YSZ vf ≈ 0.20):

            loss = mean_b[ ReLU(0.18 - mean_yx(ysz[:,0,:,:])) ]
                 + mean_b[ ReLU(0.18 - mean_yx(ysz[:,-1,:,:])) ]

        Unlike the pore face hinge ("YSZ wins over non-YSZ"), which would require
        >52.5% YSZ probability at every face voxel and can never reach zero with 20%
        vf, this density-at-face formulation fires in the regime 0.10–0.18 that the
        existing min-slice loss doesn't cover, and becomes silent once the face
        endpoints have realistic YSZ density.

        :param g_data: (B, 3, 64, 64, 64) softmax probabilities [Ni, YSZ, Pore]
        :return: scalar loss in [0, 0.36]
        """
        ysz = g_data[:, 1, :, :, :]                          # (B, Z, Y, X)
        face_z0 = ysz[:,  0, :, :].mean(dim=(1, 2))          # (B,)
        face_z1 = ysz[:, -1, :, :].mean(dim=(1, 2))          # (B,)
        threshold = 0.18
        loss = (F.relu(threshold - face_z0) + F.relu(threshold - face_z1)).mean()
        self.losses["conn_ysz_face_loss"].append(loss.item())
        return loss

    def _loss_tortuosity(self, g_data):
        """
        Differentiable tortuosity penalty via frozen τ-net surrogate.

        KEY DESIGN INVARIANT — do not break:
          NO torch.no_grad() wrapper here, NO g_data.detach().
          The τ-net parameters are frozen (requires_grad=False, set in __init__),
          but the FORWARD PASS is tracked by autograd. Gradients flow:
            tau_loss → tau_pred → g_data → generator weights  ✓
          Wrapping in no_grad() would sever this path (that's BUG 1 in _loss_ssa).

        Loss: MSE between τ-net output and mean(log(τ_real)) per phase,
        averaged over the three phases. Phases with None targets are skipped.
        Both tau_net and the targets in tau_targets operate in log(τ) space.

        YSZ gating: tau_net gives meaningless gradients when YSZ doesn't
        percolate (taufactor would return NaN). We gate YSZ tau loss to only
        the batch samples where min z-slice mean > 0.05, excluding disconnected
        samples from the MSE. This prevents noisy gradients from polluting the
        generator update when YSZ is fragmented.

        :param g_data: (b, 3, 64, 64, 64) softmax probabilities [Ni, YSZ, Pore]
        :return: scalar τ loss (torch.Tensor with grad)
        """
        phases   = [("Ni", 0), ("YSZ", 1), ("Pore", 2)]
        terms    = []
        for ph_name, ch in phases:
            target = self.tau_targets.get(ph_name)
            if target is None:
                continue
            phase_prob = g_data[:, ch:ch+1, :, :, :]    # (b, 1, 64, 64, 64)
            target_t   = torch.tensor(target, dtype=g_data.dtype,
                                      device=g_data.device)

            if ph_name == "YSZ":
                # Gate: only compute tau loss for connected samples.
                # min_slice < 0.05 means some z-slice has nearly no YSZ →
                # taufactor would not converge → tau_net gradient is noise.
                ysz_plane  = phase_prob[:, 0, :, :, :]              # (B, Z, Y, X)
                min_slice  = ysz_plane.mean(dim=(2, 3)).min(dim=1).values  # (B,)
                connected  = min_slice > 0.05                        # (B,) bool
                if not connected.any():
                    continue
                tau_pred = self.tau_net(phase_prob[connected])       # (n_conn,)
            else:
                tau_pred = self.tau_net(phase_prob)                  # (b,)

            terms.append(torch.mean((tau_pred - target_t) ** 2))

        if not terms:
            return torch.zeros(1, device=g_data.device).squeeze()

        tau_loss = sum(terms) / len(terms)
        self.losses["tau_loss"].append(tau_loss.item())
        return tau_loss

    # ── Original methods (unchanged) ──────────────────────────────────────────

    def _fixed_labels(self):
        """
        Function of generating fixed noise, volume fraction, and specific surface area
        """
        fixed_noise = torch.randn(1, self.latent_size, 4, 4, 4, device=self.device)

        fixed_vf = torch.zeros([1, 3, 4, 4, 4]).to(self.device)
        fixed_vf[0, 0, :, :, :] = 0.5058
        fixed_vf[0, 1, :, :, :] = 0.1927
        fixed_vf[0, 2, :, :, :] = 0.3015

        fixed_ssa = torch.zeros([1, 3, 4, 4, 4]).to(self.device)
        fixed_ssa[0, 0, :, :, :] = 0.1512
        fixed_ssa[0, 1, :, :, :] = 0.4985
        fixed_ssa[0, 2, :, :, :] = 0.1612

        return fixed_noise, fixed_vf, fixed_ssa

    def _oh_to_bmp(self, struc_oh):
        """
        Function of converting one-hot to bmp
        :param struc_oh : one-hot structure data
        """
        struc_oh  = struc_oh.to('cpu').numpy().copy()
        struc_arr = np.array(struc_oh)

        layer_ni   = struc_arr[0, :, :, :]
        layer_ysz  = struc_arr[1, :, :, :]
        layer_pore = struc_arr[2, :, :, :]

        struc_max = np.max(struc_oh, axis=0).squeeze()

        bool_ni  = np.array([layer_ni  == struc_max]).squeeze()
        num_ni   = bool_ni.astype(np.float32) * 255

        bool_ysz = np.array([layer_ysz == struc_max]).squeeze()
        num_ysz  = bool_ysz.astype(np.float32) * 127

        bool_pore = np.array([layer_pore == struc_max]).squeeze()
        num_pore  = bool_pore.astype(np.float32) * 0

        bmp_img = num_ni + num_ysz + num_pore
        return bmp_img

    def _plot_image(self, epoch, noise, vf, ssa, n_size=64):
        """
        Function of plotting structure data
        :param epoch  : number of epoch
        :param noise  : noise data
        :param vf     : volume fraction
        :param ssa    : specific surface area
        :param n_size : voxel size of structure data
        """
        g = self.generator.eval()
        with torch.no_grad():
            fake_structure = g(noise, vf, ssa).reshape(3, n_size, n_size, n_size).detach()

        arr_structure = self._oh_to_bmp(fake_structure)

        fig = plt.figure(figsize=(25, 25))
        for i in range(n_size):
            ax = fig.add_subplot(8, 8, i + 1)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.imshow(arr_structure[i], cmap='gray')
            fig.tight_layout()

        dir_path = "./Process_Images"
        os.makedirs(dir_path, exist_ok=True)
        fig.savefig(os.path.join(dir_path, "epoch_{}.png".format(epoch + 1)))
        plt.close("all")
        return

    def train(self):
        iters = 0
        G_loss = D_loss = torch.Tensor([0])
        f_noise, f_vf, f_ssa = self._fixed_labels()
        mm_list = self._get_min_max()

        for epoch in range(self.n_epoch):
            self.generator.train()
            self.critic.train()

            for itr, data in enumerate(self.dataloader):
                iters += 1

                r_data = data[0].to(self.device)                       # Real structure
                r_vf   = data[1][:, :3, :, :, :].to(self.device)      # Real volume fraction
                r_ssa  = data[1][:, 3:, :, :, :].to(self.device)      # Real specific surface area
                b_len  = r_data.size(0)                                # Minibatch length
                g_data = self._sample_generator(r_vf, r_ssa, b_len)   # Generated structure

                # ------ Train Critic ----- #
                self.opt_c.zero_grad()
                op_fake = self.critic(g_data.detach())
                op_real = self.critic(r_data.detach())

                wasser_d = op_real.mean() - op_fake.mean()
                loss_gp  = self._gradient_penalty(r_data, g_data)
                self.losses["Wasser_d"].append(wasser_d.item())
                self.losses["gp_loss"].append(loss_gp.item())

                D_loss = -wasser_d + self.w_gp * loss_gp
                self.losses["D_loss"].append(D_loss.item())
                D_loss.backward()
                self.opt_c.step()

                # ----- Train Generator ----- #
                if iters % self.n_critic == 0:
                    self.opt_g.zero_grad()
                    g_data    = self._sample_generator(r_vf, r_ssa, b_len)
                    loss_vf            = self._loss_vf(b_len, g_data, r_vf)
                    loss_ssa           = self._loss_ssa(b_len, g_data, r_ssa, mm_list)
                    loss_conn          = self._loss_connectivity(g_data)
                    loss_conn_ysz      = self._loss_connectivity_ysz(g_data)
                    loss_conn_ysz_face = self._loss_connectivity_ysz_face(g_data)

                    g_loss = -self.critic(g_data).mean()

                    # All connectivity losses run from epoch 0 — phase collapse
                    # happens early and must be prevented before it entrenches.
                    if epoch >= self.timing:
                        G_loss = (g_loss
                                  + self.w_param         * (loss_vf + loss_ssa)
                                  + self.w_conn          * loss_conn
                                  + self.w_conn_ysz      * loss_conn_ysz
                                  + self.w_conn_ysz_face * loss_conn_ysz_face)
                    else:
                        G_loss = (g_loss
                                  + self.w_param         * loss_vf
                                  + self.w_conn          * loss_conn
                                  + self.w_conn_ysz      * loss_conn_ysz
                                  + self.w_conn_ysz_face * loss_conn_ysz_face)

                    # τ loss: added after tau_timing epochs, only when tau_net loaded.
                    # Delayed start lets the generator form recognizable structures
                    # before the τ gradient is meaningful.
                    if self.tau_net is not None and epoch >= self.tau_timing:
                        loss_tau = self._loss_tortuosity(g_data)
                        G_loss   = G_loss + self.w_tau * loss_tau

                    self.losses["G_loss"].append(G_loss.item())
                    G_loss.backward()
                    self.opt_g.step()

                self._plot_losses(epoch, itr)

            if (epoch + 1) % self.save_epoch == 0:
                os.makedirs("./save_model", exist_ok=True)
                torch.save(
                    self.generator.state_dict(),
                    "./save_model/Generator_{:03}epoch.pth".format(epoch + 1)
                )

            self._plot_image(epoch, f_noise, f_vf, f_ssa)

        return
