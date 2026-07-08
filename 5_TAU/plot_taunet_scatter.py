"""
5_TAU/plot_taunet_scatter.py
=============================
Scatter plot of tau-net prediction vs taufactor label on held-out validation structures.

Reconstructs the SAME train/val split used in train_tau_net.py (same seed=42,
val_frac=0.2, structure-level split before augmentation) so no training structures
leak into the validation plot.

Output: 5_TAU/taunet_scatter.png

Usage (run from inside 5_TAU/):
    conda run -n ganph --no-capture-output python plot_taunet_scatter.py \\
        --data ../real_data --labels tau_labels.csv --model save_model/tau_net.pth
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "4_CNNCT"))
from analyze import find_structure_dirs, PHASES

from tau_net import TauNet
from train_tau_net import load_tau_csv, build_dataset


# Colours per phase for the scatter plot
PHASE_COLORS = {"Ni": "#d62728", "YSZ": "#1f77b4", "Pore": "#2ca02c"}


def reconstruct_val_dirs(struct_dirs, seed=42, val_frac=0.2):
    """
    Reproduce the exact train/val split from train_tau_net.py.
    Must match lines 243-248 of train_tau_net.py exactly:

        n = len(struct_dirs)
        n_val = max(1, int(n * args.val_frac))
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(n)
        val_dirs   = [struct_dirs[i] for i in sorted(idx[:n_val])]
        train_dirs = [struct_dirs[i] for i in sorted(idx[n_val:])]
    """
    n = len(struct_dirs)
    n_val = max(1, int(n * val_frac))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_dirs = [struct_dirs[i] for i in sorted(idx[:n_val])]
    return val_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Tau-net validation scatter plot (val-set only, same split as training)."
    )
    parser.add_argument("--data",   type=Path, default=Path("../real_data"),
                        help="Parent folder with structure_XXXX/ dirs (default: ../real_data)")
    parser.add_argument("--labels", type=Path, default=Path("tau_labels.csv"),
                        help="Per-structure tau labels CSV (default: tau_labels.csv)")
    parser.add_argument("--model",  type=Path, default=Path("save_model/tau_net.pth"),
                        help="TauNet checkpoint (default: save_model/tau_net.pth)")
    parser.add_argument("--ndf",    type=int, default=16,
                        help="TauNet base channel width used during training (default: 16)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("taunet_scatter.png"),
                        help="Output PNG path (default: taunet_scatter.png)")
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"ERROR: {args.labels} not found. Run compute_tau_labels.py first.")
        sys.exit(1)
    if not args.model.exists():
        print(f"ERROR: {args.model} not found. Run train_tau_net.py first.")
        sys.exit(1)

    print(f"Loading structures from {args.data} ...")
    struct_dirs = find_structure_dirs(args.data)
    tau_by_name = load_tau_csv(args.labels)

    val_dirs = reconstruct_val_dirs(struct_dirs, seed=args.seed, val_frac=args.val_frac)
    print(f"Val structures ({len(val_dirs)}): {[d.name for d in val_dirs[:5]]}{'...' if len(val_dirs) > 5 else ''}")

    # Build val dataset (no augmentation, as in train_tau_net.py)
    val_ds = build_dataset(val_dirs, tau_by_name, n_aug=1, verbose=False)
    print(f"Val samples: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TauNet(ndf=args.ndf)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device).eval()

    loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
    all_preds, all_trues, all_phase_vals = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(device)).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_trues.extend(yb.numpy().tolist())

    # Recover phase labels from the dataset item list
    # item = (vol_uint8, phase_val, tau_log, aug_idx)
    for item in val_ds.items:
        all_phase_vals.append(item[1])  # phase_val: 255=Ni, 127=YSZ, 0=Pore

    preds = np.array(all_preds)
    trues = np.array(all_trues)
    phase_vals = np.array(all_phase_vals)

    mse = float(np.mean((preds - trues) ** 2))
    print(f"Val MSE (log-tau): {mse:.4f}  (recorded best: ~0.094)")

    # Inverse phase-value map
    val_to_name = {v: k for k, v in PHASES.items()}

    fig, ax = plt.subplots(figsize=(7, 7))

    worst_n = 5
    worst_idx = np.argsort(np.abs(preds - trues))[-worst_n:]

    for ph_val, ph_name in sorted(val_to_name.items()):
        mask = (phase_vals == ph_val)
        if not mask.any():
            continue
        ax.scatter(
            trues[mask], preds[mask],
            label=f"{ph_name} (n={mask.sum()})",
            color=PHASE_COLORS.get(ph_name, "grey"),
            alpha=0.7, s=25, zorder=3,
        )

    # Highlight worst residuals
    ax.scatter(
        trues[worst_idx], preds[worst_idx],
        s=120, facecolors="none", edgecolors="black", linewidths=1.5,
        zorder=4, label=f"Worst {worst_n} residuals",
    )
    for i in worst_idx:
        ph_name = val_to_name.get(phase_vals[i], "?")
        ax.annotate(
            f"{ph_name}\n({math.exp(trues[i]):.1f})",
            (trues[i], preds[i]),
            textcoords="offset points", xytext=(6, 4), fontsize=7,
        )

    # y=x identity line
    lo = min(trues.min(), preds.min()) - 0.1
    hi = max(trues.max(), preds.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x (perfect)")

    ax.set_xlabel("log(τ)  — taufactor label", fontsize=12)
    ax.set_ylabel("log(τ)  — tau-net prediction", fontsize=12)
    ax.set_title(
        f"Tau-net: prediction vs label (held-out val set, n={len(preds)})\n"
        f"Val MSE = {mse:.4f}  (log-tau scale)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Secondary x-axis: raw tau scale tick labels
    tau_ticks = [1, 2, 5, 10, 20, 50, 100, 200]
    tick_locs = [math.log(t) for t in tau_ticks if lo <= math.log(t) <= hi]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels([str(t) for t in tau_ticks if lo <= math.log(t) <= hi],
                        fontsize=8)
    ax2.set_xlabel("τ (raw scale)", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved scatter plot -> {args.output}")

    # Print worst residuals table
    print(f"\nWorst {worst_n} residuals:")
    print(f"  {'idx':>4}  {'Phase':<6}  {'true log(τ)':>12}  {'pred log(τ)':>12}  {'|err|':>8}  {'tau_true':>9}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*9}")
    for i in sorted(worst_idx, key=lambda x: abs(preds[x] - trues[x]), reverse=True):
        ph = val_to_name.get(phase_vals[i], "?")
        err = abs(preds[i] - trues[i])
        tau_true = math.exp(trues[i])
        print(f"  {i:>4}  {ph:<6}  {trues[i]:>12.4f}  {preds[i]:>12.4f}  {err:>8.4f}  {tau_true:>9.2f}")


if __name__ == "__main__":
    main()
