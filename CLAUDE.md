# GAN-PH Project — Context for Claude

> Drop this file into the repo root (or paste into chat) to give Claude full context.
> It was written at the end of a multi-session run that retrained the GAN with
> tortuosity-aware and YSZ-connectivity-aware losses.

---

## What this project is

A **conditional WGAN-GP** that generates realistic 3-phase SOC electrode
microstructures (64³ voxels, phases: Ni=255, YSZ=127, Pore=0) matching real
FIB-SEM data. Quality is measured by **S-values** (Yu et al. 2025): a
Kolmogorov-Smirnov-based score comparing the generated distribution to real
structures on connectivity, TPB density, and tortuosity factor (τ). S ≥ 0.85
= OK, 0.70–0.85 = MARGINAL, < 0.70 = FAIL.

The project goal is a generator whose S-values are all ≥ 0.70 (ideally ≥ 0.85).

---

## Repository layout

```
GAN-PH/
├── 1_GAN/               # WGAN-GP: generator, critic, training loop
│   ├── main.py          # entry point — run from inside 1_GAN/
│   ├── training.py      # Trainer class (all loss functions live here)
│   ├── models.py        # Generator / Critic / Estimator architectures
│   ├── load.py          # data loading
│   └── save_model/      # (gitignored) Generator_NNNepoch.pth checkpoints
│
├── 2_CNN/               # Frozen SSA estimator (pre-trained, don't retrain)
│   └── save_model/model_200epoch.pth
│
├── 4_CNNCT/             # S-value analysis pipeline
│   ├── analyze.py       # computes per-structure metrics + S-value comparison
│   └── generate_structures.py  # samples N structures from generator
│
├── 5_TAU/               # Tortuosity surrogate (τ-net) pipeline — added in session
│   ├── tau_net.py           # 3D CNN: input (B,1,64,64,64) → output (B,) log(τ)
│   ├── train_tau_net.py     # trains τ-net on real structures
│   ├── compute_tau_labels.py # runs taufactor on real structures → tau_labels.csv
│   ├── tau_targets.json     # mean log(τ) per phase from real data
│   └── save_model/tau_net.pth  # (gitignored) best checkpoint, val MSE=0.0940
│
├── real_data/           # (gitignored) 101 real structure_XXXX/ folders
├── generated_data/      # (gitignored) most-recent generator output
└── CLAUDE.md            # ← this file
```

---

## Conda environment

```
conda activate ganph
```

All commands below assume this env. Run `1_GAN/main.py` from **inside** `1_GAN/`.

---

## Git branches

| Branch | What's on it |
|---|---|
| `master` | Original code + gitignore fixes + minor bug fixes |
| `tortuosity-loss` | **Checkpoint:** τ-net pipeline (5_TAU/) + first GAN retrain with tau loss |
| `feature/ysz-connectivity-loss` | **Active:** adds YSZ connectivity loss + tau gate (in progress) |
| `ni-connectivity-ablation` | Earlier experiment (Ni connectivity ablation) |

The work described below is all on `feature/ysz-connectivity-loss`.

---

## What was done (chronological)

### Step 1 — Offline τ labelling
```bash
cd 5_TAU
python compute_tau_labels.py --data ../real_data --output tau_labels.csv
```
Ran taufactor (Laplace solver, z-direction) on all 101 real structures × 3 phases.
Output: `5_TAU/tau_labels.csv`, `5_TAU/tau_targets.json`.

Key targets (log scale, what τ-net is trained on):
- log_Ni = 2.5764  (τ_Ni  = 13.37)
- log_YSZ = 3.6540 (τ_YSZ = 46.61)
- log_Pore = 0.7034 (τ_Pore = 2.02)

### Step 2 — Train τ-net surrogate
```bash
cd 5_TAU
python train_tau_net.py --data ../real_data --labels tau_labels.csv \
    --lr 0.0001 --n-aug 4 --epochs 200
```
Architecture: 3D CNN, ndf=16, input (B,1,64,64,64) → scalar log(τ).
Best val MSE = 0.0940 at epoch 60. Saved to `5_TAU/save_model/tau_net.pth`.

Key implementation details:
- Lazy TauDataset (uint8 volumes in RAM ~21 MB, augment on-the-fly)
- 4 augmentations = 4 z-rotations (z-preserving, τ invariant)
- Incremental checkpointing: saves on each val improvement

### Step 3 — First GAN retrain (tortuosity-loss branch)
```bash
cd 1_GAN
python main.py --data ../real_data --lr 0.00005 --epochs 50 \
    --tau-estimator ../5_TAU/save_model/tau_net.pth \
    --tau-targets ../5_TAU/tau_targets.json
```
- Added `_loss_tortuosity()` to training.py: MSE(τ_net(phase_prob), log_target)
- Frozen τ-net weights (requires_grad=False); NO torch.no_grad() so gradients
  flow through τ-net back to generator
- w_tau=50, activates at epoch 10

**Results after 50 epochs:**
| Metric | Before | After |
|---|---|---|
| tau_Ni | 0.649 FAIL | 0.770 MARGINAL ✓ |
| tau_YSZ | 0.484 FAIL | 0.484 FAIL ✗ |
| conn_Ni | — | 0.876 OK |
| conn_YSZ | — | 0.707 MARGINAL |
| tau_Pore | — | 0.676 FAIL |
| total_tpb | — | 0.693 FAIL |

### Step 4 — Root cause of tau_YSZ failure
5 out of 50 generated structures had conn_YSZ=0 (completely disconnected YSZ →
taufactor returns NaN). Many more had low YSZ connectivity (conn_YSZ=0.14–0.46),
causing taufactor to return unconverged lower-bound estimates (63–214).
When YSZ is disconnected, τ-net gradients carry no useful signal.

### Step 5 — Current work (feature/ysz-connectivity-loss branch)
Two fixes added to `1_GAN/training.py`:

**Fix 1 — `_loss_connectivity_ysz()`:**
```python
ysz = g_data[:, 1, :, :, :]           # (B, Z, Y, X)
slice_means = ysz.mean(dim=(2, 3))     # (B, Z) — mean YSZ per z-slice
loss = F.relu(0.10 - slice_means).mean()
```
Penalises any z-slice where mean YSZ probability < 0.10 (real YSZ vf ≈ 0.20).
Weight w_conn_ysz=200, active from epoch 0.

**Fix 2 — YSZ gate in `_loss_tortuosity()`:**
```python
ysz_plane = phase_prob[:, 0, :, :, :]              # (B, Z, Y, X)
min_slice  = ysz_plane.mean(dim=(2, 3)).min(dim=1).values  # (B,)
connected  = min_slice > 0.05
if not connected.any(): continue
tau_pred   = self.tau_net(phase_prob[connected])   # only connected samples
```
Skips disconnected batch samples from the YSZ tau MSE.

**Training in progress (as of when this file was written):**
Epoch ~35/50 complete. `tau_loss` oscillating 0.17–0.79, `conn_ysz_loss` ≈ 0.
S-value analysis not yet run on this checkpoint.

---

## How to run the full pipeline end-to-end

```bash
# 1. Generate structures
cd 4_CNNCT
python generate_structures.py \
    --training-data ../real_data \
    --output ../generated_data \
    --n 50

# 2. Compute S-values
python analyze.py \
    --input ../real_data \
    --compare ../generated_data \
    --output s_value_report.csv
# S-value summary printed to stdout; also saved to s_value_report_svalues.csv
```

For GAN training:
```bash
cd 1_GAN
conda run -n ganph --no-capture-output python -u main.py \
    --data ../real_data \
    --lr 0.00005 \
    --epochs 50 \
    --tau-estimator ../5_TAU/save_model/tau_net.pth \
    --tau-targets ../5_TAU/tau_targets.json
```
On CPU this takes ~60–90 minutes for 50 epochs.

---

## Key hyperparameters (1_GAN/training.py)

| Parameter | Value | Purpose |
|---|---|---|
| w_gp | 20 | Gradient penalty weight |
| w_param | 1000 | Volume fraction loss |
| w_conn | 50 | Pore connectivity (isolation + face hinge) |
| w_conn_ysz | 200 | YSZ percolation proxy (min z-slice mean) |
| w_tau | 50 | Tortuosity surrogate loss |
| tau_timing | 10 | Epoch at which tau+ysz-tau losses activate |
| lr | 5e-5 | Adam learning rate for both G and C |

---

## Final S-value results (all three runs)

| Metric | Baseline | Run 1 (tau loss) | Run 2 (+ YSZ conn) |
|---|---|---|---|
| tau_Ni | 0.649 FAIL | 0.770 MARGINAL | **0.818 MARGINAL** |
| tau_YSZ | 0.484 FAIL | 0.484 FAIL | 0.479 FAIL |
| tau_Pore | — | 0.676 FAIL | 0.688 FAIL |
| conn_Ni | — | 0.876 OK | 0.891 OK |
| conn_YSZ | — | 0.707 MARGINAL | 0.703 MARGINAL |
| conn_Pore | — | 0.876 OK | 0.878 OK |
| total_tpb | — | 0.693 FAIL | 0.689 FAIL |
| active_tpb | — | 0.721 MARGINAL | 0.718 MARGINAL |
| active_tpb_frac | — | 0.733 MARGINAL | 0.730 MARGINAL |

Run 2 generator: `feature/ysz-connectivity-loss`, epoch 50 checkpoint.

## What still needs work

### tau_YSZ (stuck at ~0.48 FAIL across all runs)

Root cause diagnosis: The min-slice-mean connectivity loss (`_loss_connectivity_ysz`)
barely fired during Run 2 training (conn_ysz_loss consistently < 0.001). The generator
was already producing YSZ above the 10% density threshold at every z-slice — but as
**topologically disconnected blobs**, not percolating channels. The metric measures
density per slice, not connectivity between slices. taufactor still saw fragmented YSZ
and returned high/unconverged τ values.

Approaches to try next (in rough order of likely impact):

1. **YSZ face-hinge** (fast, try first): Require YSZ to **win** at both z=0 and z=63
   faces, similar to pore face hinge but for YSZ. If YSZ is present at the entry/exit
   faces and has 20% vf throughout, percolation is much more likely.
   ```python
   ysz = g_data[:, 1:2, :, :, :]
   pore_ni = 1.0 - ysz  # "everything else"
   loss_z0 = F.relu(pore_ni[:,:,0,:,:] - ysz[:,:,0,:,:] + 0.05).mean()
   loss_z1 = F.relu(pore_ni[:,:,-1,:,:] - ysz[:,:,-1,:,:] + 0.05).mean()
   ```

2. **Retrain tau_net with more YSZ data** — currently τ_YSZ spans 11–214 in the
   real data (very wide, skewed). The surrogate may be poorly calibrated in the
   region where generated structures live. Re-check tau_labels.csv distributions.

3. **Weighted tau loss** — give YSZ 3× the weight of Ni and Pore in `_loss_tortuosity`
   since it's the hardest phase to fix.

4. **More epochs** — tau_Ni improved steadily from run to run. tau_YSZ may just need
   more training (100 epochs).

5. **Differentiable TPB proxy** (also fixes total_tpb FAIL) — TPB density is
   the density of voxels adjacent to all three phases. A differentiable proxy:
   ```python
   # voxel is near-TPB if all three phase probs are non-trivial
   near_tpb = g_data[:,0] * g_data[:,1] * g_data[:,2]  # (B, 64, 64, 64)
   tpb_density = near_tpb.mean()
   # penalise if below real mean
   loss_tpb = F.relu(target_tpb - tpb_density)
   ```

---

## Useful diagnostic commands

```bash
# Watch training live (last batch of each epoch)
Get-Content -Wait 1_GAN/log.dat | Select-String "\[026/026\]"

# Quick S-value check after generating structures
cd 4_CNNCT && python analyze.py --input ../real_data --compare ../generated_data

# Check τ-net prediction on a single structure
cd 5_TAU && python test_tau_loss.py
```

---

## Important implementation notes

- **Never wrap tau_net in torch.no_grad()** — the whole point is gradient flow
  from tau_loss through tau_net to the generator. See the long comment in
  training.py `_loss_tortuosity()`.
- **Z-axis is dim 0** of the (64,64,64) volume = dim 2 of (B,C,Z,Y,X) tensors.
  Taufactor solves along z. All augmentations in train_tau_net.py must preserve z.
- **τ-net outputs log(τ)**, not raw τ. Targets in tau_targets.json are also log-scale
  (`log_Ni`, `log_YSZ`, `log_Pore` keys). Do not mix up with the raw-scale keys.
- **conda run --no-capture-output** is needed on Windows to avoid stdout buffering
  when running in background.
