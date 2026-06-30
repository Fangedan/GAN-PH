"""
Train the tortuosity surrogate (TauNet) on real structures.

Prerequisite: run compute_tau_labels.py first to produce tau_labels.csv.

Augmentation strategy (z-preserving only):
  - 4 rotations by 90 degrees about z (swap x-y axes, z unchanged)
  - x 4 combinations of x-flip and y-flip
  = 16 augmentations per (structure, phase) pair
  Tortuosity is measured along z, so all 16 transforms leave tau invariant.
  DO NOT rotate about x or y axes -- that changes the transport direction.

Train/val split is done BY STRUCTURE INDEX before augmentation, so no
augmented version of a training structure leaks into validation.

Memory: volumes are pre-loaded as uint8 (~21 MB for 81 structures).
Augmentations are applied on-the-fly in __getitem__, avoiding the ~4 GB
tensor pre-allocation that caused the original implementation to stall.

Output:
  save_model/tau_net.pth -- trained weights

Usage (run from inside 5_TAU/):
  python train_tau_net.py --data ../real_data --labels tau_labels.csv
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / "4_CNNCT"))
from analyze import load_structure, find_structure_dirs, PHASES

from tau_net import TauNet, weights_init


# -- Augmentation -------------------------------------------------------------

def _apply_augmentation(binary_64, aug_idx):
    """
    Apply the aug_idx-th (0..15) z-preserving augmentation to a (64,64,64) array.
      aug_idx = rotation_k * 4 + flip_combo
      rotation_k in {0,1,2,3}: np.rot90 about z (axes 1,2)
      flip_combo in {0,1,2,3}: no-flip, flip-x, flip-y, flip-x+flip-y
    """
    k = aug_idx // 4
    flip_combo = aug_idx % 4
    v = np.rot90(binary_64, k, axes=(1, 2))
    if flip_combo in (1, 3):
        v = np.flip(v, axis=2)
    if flip_combo in (2, 3):
        v = np.flip(v, axis=1)
    return np.ascontiguousarray(v, dtype=np.float32)


# -- Dataset ------------------------------------------------------------------

class TauDataset(torch.utils.data.Dataset):
    """
    Lazy dataset: volumes are uint8 in memory (~21 MB total for 81 structures).
    Augmentation and float conversion happen in __getitem__, not at build time.

    Each item is a tuple: (vol_uint8, phase_val, tau_log, aug_idx)
      aug_idx == -1 means no augmentation (identity)
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vol, phase_val, tau_log, aug_idx = self.items[idx]
        binary = (vol == phase_val).astype(np.float32)   # (64, 64, 64)
        if aug_idx >= 0:
            binary = _apply_augmentation(binary, aug_idx)
        x = torch.from_numpy(binary[None])               # (1, 64, 64, 64)
        y = torch.tensor(tau_log, dtype=torch.float32)
        return x, y


def load_tau_csv(csv_path):
    """Load tau_labels.csv -> dict: structure_name -> {tau_Ni, tau_YSZ, tau_Pore}"""
    mapping = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            entry = {}
            for ph in PHASES:
                raw = row[f"tau_{ph}"]
                entry[f"tau_{ph}"] = (
                    float(raw) if raw not in ("nan", "NaN", "") else float("nan")
                )
            mapping[row["name"]] = entry
    return mapping


def build_dataset(struct_dirs, tau_by_name, n_aug, verbose=True):
    """
    Pre-load all volumes as uint8 (~262 KB each), then build an item list.
    Returns a TauDataset with lazy on-the-fly augmentation.
    n_aug: number of augmentations per (structure, phase) pair.
      1 = identity only (no augmentation)
      4 = 4 z-rotations (fast, recommended for CPU)
      16 = 4 rotations x 4 x/y-flip combos (full set)
    """
    phase_items = list(PHASES.items())   # [('Ni', 255), ('YSZ', 127), ('Pore', 0)]
    items = []
    skipped = 0

    for sd in struct_dirs:
        name = sd.name
        if name not in tau_by_name:
            skipped += 1
            continue
        tau_row = tau_by_name[name]
        vol = load_structure(sd).astype(np.uint8)   # (64, 64, 64) uint8

        for ph_name, ph_val in phase_items:
            tau_val = tau_row.get(f"tau_{ph_name}", float("nan"))
            if np.isnan(tau_val) or tau_val <= 0:
                continue
            tau_log = math.log(tau_val)   # log(tau); compresses 11-214 -> 2.4-5.4

            for aug_idx in range(n_aug):
                items.append((vol, ph_val, tau_log, aug_idx if n_aug > 1 else -1))

    if verbose:
        print(f"  items: {len(items)}  (skipped {skipped} structures not in CSV)")
    if not items:
        raise RuntimeError(
            "No valid (structure, phase) pairs found. "
            "Check that tau_labels.csv exists and has non-NaN values."
        )
    return TauDataset(items)


# -- Training loop ------------------------------------------------------------

def train(model, train_ds, val_ds, device, epochs, batch_size, lr, ckpt_path=None):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in torch.utils.data.DataLoader(val_ds, batch_size=batch_size):
                val_preds.append(model(xb.to(device)).cpu())
                val_trues.append(yb)
        val_pred = torch.cat(val_preds)
        val_true = torch.cat(val_trues)
        val_loss = criterion(val_pred, val_true).item()

        train_loss = total / len(train_ds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1:03}/{epochs}  "
                f"train MSE={train_loss:.4f}  val MSE={val_loss:.4f}",
                flush=True,
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save to disk immediately so early termination doesn't lose the best weights
            if ckpt_path is not None:
                torch.save(best_state, ckpt_path)
                print(f"  [checkpoint saved  val MSE={best_val:.4f}]", flush=True)

    print(f"Best val MSE: {best_val:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train TauNet surrogate on real structures."
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Parent folder with structure_XXXX/ dirs (e.g., ../real_data)")
    parser.add_argument("--labels", type=Path, default=Path("tau_labels.csv"),
                        help="CSV produced by compute_tau_labels.py (default: tau_labels.csv)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction of structures held out for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ndf", type=int, default=16,
                        help="Base channel width for TauNet (default: 16)")
    parser.add_argument("--n-aug", type=int, default=4,
                        help="Augmentations per (structure, phase) pair: "
                             "1=none, 4=z-rotations only (fast, default), "
                             "16=full z-preserving set")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.labels.exists():
        print(f"ERROR: {args.labels} not found.\nRun compute_tau_labels.py first.", flush=True)
        return

    struct_dirs = find_structure_dirs(args.data)
    tau_by_name = load_tau_csv(args.labels)

    # Split by structure (NOT by sample) to prevent augmentation leakage
    n = len(struct_dirs)
    n_val = max(1, int(n * args.val_frac))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    val_dirs   = [struct_dirs[i] for i in sorted(idx[:n_val])]
    train_dirs = [struct_dirs[i] for i in sorted(idx[n_val:])]

    print(f"Structures: {len(train_dirs)} train / {len(val_dirs)} val", flush=True)
    print("Pre-loading volumes (~21 MB) and building item lists ...", flush=True)

    train_ds = build_dataset(train_dirs, tau_by_name, n_aug=args.n_aug)
    val_ds   = build_dataset(val_dirs,   tau_by_name, n_aug=1)

    # Extract tau values for range reporting
    train_taus = np.array([item[2] for item in train_ds.items])
    print(
        f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}  "
        f"(log-tau range: [{train_taus.min():.2f}, {train_taus.max():.2f}])",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = TauNet(ndf=args.ndf)
    model.apply(weights_init)

    os.makedirs("save_model", exist_ok=True)
    out_path = Path("save_model/tau_net.pth")

    model = train(
        model, train_ds, val_ds,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_path=out_path,   # saves best to disk on each val improvement
    )

    # Final save (redundant if training completed normally, but ensures the file exists)
    torch.save(model.state_dict(), out_path)
    print(f"Saved -> {out_path}", flush=True)

    # Sanity check on val set
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for xb, yb in torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size):
            val_preds.append(model(xb.to(device)).cpu())
            val_trues.append(yb)
    preds = torch.cat(val_preds).numpy()
    true  = torch.cat(val_trues).numpy()
    mae_log = float(np.mean(np.abs(preds - true)))
    mae_raw = float(np.mean(np.abs(np.exp(preds) - np.exp(true))))
    print(
        f"Val MAE (log scale): {mae_log:.4f}  "
        f"Val MAE (raw tau scale): {mae_raw:.3f}  "
        f"(mean tau_true={np.exp(true).mean():.3f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
