# Main script for WGAN-GP training
# Generates 3D microstructures of SOC electrodes
#
# Hyperparameters are configurable via command-line flags (argparse).
# Running with no flags reproduces the original synthetic-data behavior:
#     python main.py
# Train on real data (auto-counts the 101 structures, GPU batch size):
#     python main.py --data ../real_data --batch-size 32
# See all options:
#     python main.py --help
#
# Run this from inside 1_GAN/ (so `import load/models/training` and the
# default ../ paths resolve), exactly as before.

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.optim as optim

import load
import models
import training

# τ-net is in 5_TAU/ — add to path only when needed (see load block below)
_TAU_NET_DIR = Path(__file__).parent.parent / "5_TAU"


def count_structures(data_dir: Path) -> int:
    """Count structure_XXXX subfolders in a data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    n = sum(
        1 for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("structure_")
    )
    if n == 0:
        raise FileNotFoundError(f"No structure_XXXX folders found in {data_dir}")
    return n


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the conditional WGAN-GP on SOC electrode microstructures."
    )
    p.add_argument(
        "--data", "-d", type=Path, default=Path("../synthetic_data"),
        help="Folder with structure_XXXX/ stacks + results.dat "
             "(default: ../synthetic_data). Use ../real_data for the real run.",
    )
    p.add_argument(
        "--n-struc", "-n", type=int, default=None,
        help="Number of structures to train on. "
             "Default: auto-count the structure_XXXX folders in --data.",
    )
    p.add_argument(
        "--batch-size", "-b", type=int, default=4,
        help="Batch size (default: 4, a CPU-safe value). Raise to 32-64 on GPU.",
    )
    p.add_argument(
        "--epochs", "-e", type=int, default=50,
        help="Number of training epochs (default: 50).",
    )
    p.add_argument(
        "--lr", type=float, default=1e-4,
        help="Adam learning rate (default: 1e-4).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--estimator", type=Path,
        default=Path("../2_CNN/save_model/model_200epoch.pth"),
        help="Path to the frozen SSA estimator weights "
             "(default: ../2_CNN/save_model/model_200epoch.pth).",
    )
    p.add_argument(
        "--tau-estimator", type=Path, default=None,
        help="Path to frozen τ-net weights (5_TAU/save_model/tau_net.pth). "
             "If omitted, tortuosity loss is disabled.",
    )
    p.add_argument(
        "--tau-targets", type=Path, default=None,
        help="JSON produced by compute_tau_labels.py with mean τ per phase "
             "(5_TAU/tau_targets.json). Required when --tau-estimator is set.",
    )
    p.add_argument(
        "--save-every", type=int, default=1,
        help="Save a Generator checkpoint every N epochs (default: 1 = every epoch).",
    )
    p.add_argument(
        "--no-tpb-proxy", action="store_true", default=False,
        help="Disable near-TPB density proxy loss (target=0.002 is anode-calibrated; "
             "pass this flag for non-anode datasets).",
    )
    p.add_argument(
        "--no-ysz-density", action="store_true", default=False,
        help="Disable YSZ min-slice density and YSZ face-hinge losses "
             "(thresholds 0.10/0.18 are anode-calibrated; pass for non-anode datasets).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    n_size = 64  # voxel cube size — fixed by the architecture

    # Number of structures: explicit flag, else auto-count the data folder
    n_struc = args.n_struc if args.n_struc is not None else count_structures(args.data)

    if not args.estimator.exists():
        raise FileNotFoundError(
            f"Estimator weights not found: {args.estimator}\n"
            f"Pass --estimator <path>, or train 2_CNN first."
        )

    # ---- Device ---- #
    ngpu   = torch.cuda.device_count()
    device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    print("=" * 60)
    print(" GAN-PH  —  WGAN-GP training")
    print("=" * 60)
    print(f"  data          : {args.data}  ({n_struc} structures)")
    print(f"  batch size    : {args.batch_size}")
    print(f"  epochs        : {args.epochs}")
    print(f"  lr            : {args.lr}")
    print(f"  seed          : {args.seed}")
    print(f"  estimator     : {args.estimator}")
    print(f"  tau-estimator : {args.tau_estimator or 'disabled'}")
    print(f"  device        : {device}")
    print("=" * 60)

    load.torch_fix_seed(args.seed)

    # ---- Load training data ---- #
    x_train = load.load_structure(n_struc, n_size, args.data)
    y_train = load.get_label(n_struc, args.data)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # ---- Load pretrained SSA estimator (frozen) ---- #
    estimator = models.Estimator(in_ch=1, ndf=32).to(device)
    estimator.load_state_dict(
        torch.load(str(args.estimator), map_location=device)
    )
    estimator.eval()

    # ---- Load τ-net (optional, frozen) ---- #
    # Gradient flows THROUGH tau_net to the generator — see training._loss_tortuosity.
    # Weights are frozen in Trainer.__init__; no_grad is NOT used in the loss.
    tau_net     = None
    tau_targets = None
    if args.tau_estimator is not None:
        if not args.tau_estimator.exists():
            raise FileNotFoundError(
                f"τ-net weights not found: {args.tau_estimator}\n"
                "Train it first: cd 5_TAU && python train_tau_net.py ..."
            )
        if args.tau_targets is None or not args.tau_targets.exists():
            raise FileNotFoundError(
                "--tau-targets JSON required when --tau-estimator is set.\n"
                "Run compute_tau_labels.py to produce tau_targets.json."
            )
        sys.path.insert(0, str(_TAU_NET_DIR))
        from tau_net import TauNet   # type: ignore[import]  — runtime path
        tau_net = TauNet(ndf=16).to(device)
        tau_net.load_state_dict(
            torch.load(str(args.tau_estimator), map_location=device)
        )
        with open(args.tau_targets) as f:
            raw = json.load(f)
        # Use log-scale targets (key "log_Ni" etc.) because tau_net is trained on log(τ).
        # mean(log(τ_real)) is stored in the JSON by compute_tau_labels.py.
        # Fall back to log(raw mean) if log keys are missing (backward compat).
        import math as _math
        tau_targets = {}
        for ph in ("Ni", "YSZ", "Pore"):
            log_val = raw.get(f"log_{ph}")
            if log_val is None:
                raw_val = raw.get(ph)
                log_val = _math.log(raw_val) if raw_val else None
            if log_val is not None:
                tau_targets[ph] = float(log_val)
        print(f"  τ log-targets: { {k: f'{v:.4f}' for k, v in tau_targets.items()} }")

    # ---- Build Generator and Critic ---- #
    generator = models.Generator(latent_size=100).to(device)
    generator.apply(models.weights_init)

    critic = models.Critic().to(device)
    critic.apply(models.weights_init)

    # ---- Optimizers ---- #
    opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_c = optim.Adam(critic.parameters(),    lr=args.lr, betas=(0.5, 0.9))

    # ---- Train ---- #
    trainer = training.Trainer(
        model_g        = generator,
        optim_g        = opt_g,
        model_c        = critic,
        optim_c        = opt_c,
        model_e        = estimator,
        epochs         = args.epochs,
        device         = device,
        dataloader     = train_loader,
        in_header      = str(args.data),
        tau_net        = tau_net,
        tau_targets    = tau_targets,
        no_tpb_proxy   = args.no_tpb_proxy,
        no_ysz_density = args.no_ysz_density,
        save_every     = args.save_every,
    )
    trainer.train()


if __name__ == "__main__":
    main()
