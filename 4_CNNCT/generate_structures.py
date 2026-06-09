"""
4_CNNCT/generate_structures.py
================================
Generates synthetic 3D Ni-YSZ microstructures using the trained WGAN-GP
generator and saves them as BMP stacks in the format expected by analyze.py.

Conditioning vectors (VF, SSA) are sampled from the training data distribution
(results.dat) so the generator is asked to produce what it was trained on.
This ensures any failures in connectivity/TPB are attributable to the GAN
architecture — not to out-of-distribution conditioning.

Normalization matches load.py exactly:
  VF  → multiply by 0.01  (% to fraction)
  SSA → MinMaxScaler fitted on training dataset (same scaler as training)

Usage:
  # From inside 4_CNNCT/, using defaults (epoch 50, 50 structures):
  python generate_structures.py

  # Specific epoch:
  python generate_structures.py --epoch 30

  # Full control:
  python generate_structures.py ^
      --weights ../1_GAN/save_model/Generator_050epoch.pth ^
      --training-data ../synthetic_data ^
      --output ../generated_data ^
      --n 50

  # After generation, run the S-value comparison:
  python analyze.py --input ../synthetic_data ^
                    --compare ../generated_data ^
                    --output s_values.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn import preprocessing

# ── Import Generator from 1_GAN/models.py ────────────────────────────────────
# Works whether you run from 4_CNNCT/ or repo root
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "1_GAN"))
try:
    from models import Generator
except ImportError as e:
    print(f"ERROR: Could not import Generator from 1_GAN/models.py: {e}")
    print("Make sure you run this script from inside 4_CNNCT/ or the repo root.")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

LATENT_SIZE = 100    # must match main.py

# Generator output: channel 0=Ni, 1=YSZ, 2=Pore  →  pixel values below
# (matches load.py one-hot encoding and preprocess_dream3d.py phase values)
PHASE_PIXEL = np.array([255, 127, 0], dtype=np.uint8)


# ── Conditioning ──────────────────────────────────────────────────────────────

def load_conditioning(training_data_dir: Path) -> tuple:
    """
    Load and normalize VF + SSA from training results.dat.

    Replicates load.py get_label() exactly:
      - VF columns (VF0,VF1,VF2): multiply by 0.01  → fraction [0,1]
      - SSA columns (SV0,SV1,SV2): MinMaxScaler fitted on this dataset → [0,1]

    Returns
    -------
    vf_norm  : np.ndarray (n, 3)  — VF fractions
    ssa_norm : np.ndarray (n, 3)  — MinMax-scaled SSA
    """
    dat_path = training_data_dir / "results.dat"
    if not dat_path.exists():
        raise FileNotFoundError(
            f"results.dat not found in {training_data_dir}\n"
            f"Make sure --training-data points to the folder with results.dat"
        )

    df = pd.read_table(dat_path, sep=r"\s+")

    vf_norm  = df[["VF0", "VF1", "VF2"]].values.astype(np.float32) * 0.01

    scaler   = preprocessing.MinMaxScaler()
    ssa_norm = scaler.fit_transform(
        df[["SV0", "SV1", "SV2"]].values.astype(np.float32)
    )

    return vf_norm, ssa_norm.astype(np.float32)


def to_conditioning_tensors(vf_vec: np.ndarray,
                             ssa_vec: np.ndarray,
                             device: torch.device) -> tuple:
    """
    Expand 1D VF and SSA vectors to (1, 3, 4, 4, 4) tensors.

    The Generator forward() call is:
        model_g(noise, vf_label, ssa_label)
    where vf_label and ssa_label each have shape (batch, 3, 4, 4, 4).
    Each spatial position holds the same scalar value (broadcast fill).
    """
    vf_t = (torch.tensor(vf_vec, dtype=torch.float32)
            .view(1, 3, 1, 1, 1)
            .expand(1, 3, 4, 4, 4)
            .to(device))

    ssa_t = (torch.tensor(ssa_vec, dtype=torch.float32)
             .view(1, 3, 1, 1, 1)
             .expand(1, 3, 4, 4, 4)
             .to(device))

    return vf_t, ssa_t


# ── Output conversion ─────────────────────────────────────────────────────────

def tensor_to_voxel(output: torch.Tensor) -> np.ndarray:
    """
    Convert generator output to a (64, 64, 64) uint8 array.

    Parameters
    ----------
    output : torch.Tensor, shape (1, 3, 64, 64, 64)
        Softmax probabilities from Generator.forward()

    Returns
    -------
    np.ndarray (64, 64, 64) uint8 with values in {255=Ni, 127=YSZ, 0=Pore}
    """
    # argmax over channel dim → (64, 64, 64) with values {0, 1, 2}
    phase_idx = (output.squeeze(0)          # (3, 64, 64, 64)
                       .argmax(dim=0)       # (64, 64, 64)
                       .cpu()
                       .numpy()
                       .astype(np.uint8))
    return PHASE_PIXEL[phase_idx]           # {0→255, 1→127, 2→0}


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_structure(vol: np.ndarray, idx: int, output_dir: Path) -> None:
    """Save (64,64,64) volume as 64 grayscale BMP slices."""
    folder = output_dir / f"structure_{idx:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    for z in range(vol.shape[0]):
        cv2.imwrite(str(folder / f"slice_{z:04d}.bmp"), vol[z])


def write_results_dat(output_dir: Path,
                      vf_list: list,
                      ssa_list: list) -> None:
    """
    Write results.dat for generated structures.
    VF is converted back to % for consistency with pipeline format.
    SSA is stored as MinMax-normalized values (we don't have the inverse
    scaler, and analyze.py computes all metrics from voxels anyway).
    """
    with open(output_dir / "results.dat", "w") as f:
        f.write("VF0 VF1 VF2 SV0 SV1 SV2\n")
        for vf, ssa in zip(vf_list, ssa_list):
            row = list(vf * 100) + list(ssa)   # VF back to %
            f.write(" ".join(f"{v:.4f}" for v in row) + "\n")


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(weights_path: Path,
             training_data_dir: Path,
             output_dir: Path,
             n_structures: int,
             seed: int = 42) -> None:

    device = torch.device("cpu")
    rng    = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print(" GAN-PH Structure Generation")
    print("=" * 58)

    # ── Load conditioning distribution ────────────────────────────────────
    print(f"\nLoading conditioning from: {training_data_dir}")
    vf_norm, ssa_norm = load_conditioning(training_data_dir)
    n_train = len(vf_norm)
    print(f"  {n_train} training structures available")
    print(f"  VF range — Ni:   [{vf_norm[:,0].min():.3f}, {vf_norm[:,0].max():.3f}]")
    print(f"             YSZ:  [{vf_norm[:,1].min():.3f}, {vf_norm[:,1].max():.3f}]")
    print(f"             Pore: [{vf_norm[:,2].min():.3f}, {vf_norm[:,2].max():.3f}]")

    # ── Load generator ────────────────────────────────────────────────────
    print(f"\nLoading: {weights_path.name}")
    model_g = Generator(latent_size=LATENT_SIZE).to(device)
    model_g.load_state_dict(
        torch.load(str(weights_path), map_location=device)
    )
    model_g.eval()
    print("  Generator loaded — eval mode")

    # ── Generate ──────────────────────────────────────────────────────────
    print(f"\nGenerating {n_structures} structures → {output_dir}\n")

    vf_log, ssa_log = [], []

    with torch.no_grad():
        for i in tqdm(range(n_structures), desc="Generating"):
            # Sample conditioning from training distribution
            row = rng.integers(0, n_train)
            vf_vec  = vf_norm[row]
            ssa_vec = ssa_norm[row]

            noise       = torch.randn(1, LATENT_SIZE, 4, 4, 4, device=device)
            vf_t, ssa_t = to_conditioning_tensors(vf_vec, ssa_vec, device)

            output = model_g(noise, vf_t, ssa_t)   # (1, 3, 64, 64, 64)
            vol    = tensor_to_voxel(output)        # (64, 64, 64) uint8

            save_structure(vol, i + 1, output_dir)
            vf_log.append(vf_vec.copy())
            ssa_log.append(ssa_vec.copy())

    write_results_dat(output_dir, vf_log, ssa_log)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  Generated {n_structures} structures → {output_dir.resolve()}")
    print(f"\n  Next step — run connectivity analysis and S-value comparison:")
    print(f"\n    python analyze.py \\")
    print(f"        --input ../{training_data_dir.name} \\")
    print(f"        --compare ../{output_dir.name} \\")
    print(f"        --output s_values.csv")
    print(f"{'='*58}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate structures from trained WGAN-GP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to .pth file. If omitted, uses --epoch to locate automatically."
    )
    parser.add_argument(
        "--epoch", type=int, default=50,
        help="Epoch to load from 1_GAN/save_model/ (default: 50 = latest)"
    )
    parser.add_argument(
        "--training-data", default="../synthetic_data",
        help="Folder with training results.dat (default: ../synthetic_data)"
    )
    parser.add_argument(
        "--output", default="../generated_data",
        help="Output folder (default: ../generated_data)"
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of structures to generate (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = (
            _repo_root / "1_GAN" / "save_model"
            / f"Generator_{args.epoch:03d}epoch.pth"
        )

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    generate(
        weights_path=weights_path,
        training_data_dir=Path(args.training_data),
        output_dir=Path(args.output),
        n_structures=args.n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
