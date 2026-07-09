"""
4_CNNCT/memo_check.py
=====================
Memorization check for GAN training runs.

For each generated structure, finds the nearest training crop by per-voxel
phase-agreement fraction, after checking all 16 z-preserving symmetries
(4 z-axis rotations × 4 XY-flip combinations — the tau-net augmentation group).

Also computes the baseline: for each training crop, the agreement with its
nearest OTHER training crop. With stride-32 overlapping crops this baseline
is naturally high; generated structures matching *above* the baseline signal
memorization rather than generalisation.

Outputs
-------
  --output CSV: per-generated-structure results + summary rows
  Prints:  distribution of gen→train agreement and baseline train→train agreement

Usage
-----
  # From repo root:
  conda run -n ganph --no-capture-output python 4_CNNCT/memo_check.py \\
      --train  cathode_crops_s1_str32 \\
      --generated generated_cathode_run0_final \\
      --output 4_CNNCT/cathode_run0_memo.csv
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


# ── Volume I/O ────────────────────────────────────────────────────────────────

def load_bmp_volume(structure_dir: Path, n_slices: int = 64) -> np.ndarray:
    """Load 64 BMP slices into a (64, 64, 64) uint8 array."""
    slices = []
    for z in range(n_slices):
        p = structure_dir / f"slice_{z:04d}.bmp"
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Slice not found: {p}")
        slices.append(img)
    return np.stack(slices, axis=0)


def load_all_volumes(data_dir: Path) -> dict[str, np.ndarray]:
    """Return {name: (64,64,64) uint8} for all structure_XXXX/ dirs in data_dir."""
    vols = {}
    for d in sorted(data_dir.glob("structure_*")):
        if d.is_dir():
            try:
                vols[d.name] = load_bmp_volume(d)
            except FileNotFoundError:
                print(f"  Warning: could not load {d.name}, skipping")
    return vols


# ── Symmetry group ────────────────────────────────────────────────────────────

def _symmetries(vol: np.ndarray):
    """
    Yield the 16 z-preserving symmetries of a ZYX array.
    4 z-axis rotations (0/90/180/270°) × 4 XY flip combinations
    (none / flip-Y / flip-X / flip-XY).
    """
    for k in range(4):
        r = np.rot90(vol, k, axes=(1, 2))
        yield r                                                      # no flip
        yield np.flip(r, axis=1).copy()                             # flip Y
        yield np.flip(r, axis=2).copy()                             # flip X
        yield np.flip(np.flip(r, axis=1), axis=2).copy()            # flip both


# ── Agreement metric ──────────────────────────────────────────────────────────

def best_agreement(gen_vol: np.ndarray, ref_vol: np.ndarray) -> float:
    """
    Maximum per-voxel phase-agreement fraction over all 16 z-preserving symmetries.

    Agreement = fraction of voxels where gen and ref have the same BMP value,
    maximised over the symmetry group so orientation differences don't penalise
    otherwise-identical microstructures.
    """
    total = gen_vol.size
    best  = 0.0
    for sym in _symmetries(gen_vol):
        frac = float(np.sum(sym == ref_vol)) / total
        if frac > best:
            best = frac
    return best


# ── Main check ────────────────────────────────────────────────────────────────

def memo_check(train_dir: Path, gen_dir: Path, output_csv: Path) -> None:
    print("=" * 64)
    print("  Memorization check")
    print("=" * 64)
    print(f"  Training crops : {train_dir}")
    print(f"  Generated      : {gen_dir}")

    train_vols = load_all_volumes(train_dir)
    gen_vols   = load_all_volumes(gen_dir)
    print(f"  Loaded {len(train_vols)} training crops, {len(gen_vols)} generated structures\n")

    train_names = sorted(train_vols.keys())
    gen_names   = sorted(gen_vols.keys())

    # ── Generated → nearest training crop ─────────────────────────────────
    print("  [1/2] Generated → nearest train agreement (16 symmetries each) ...")
    gen_rows = []
    gen_agreements: list[float] = []

    for gname in gen_names:
        gvol  = gen_vols[gname]
        best_name  = None
        best_score = -1.0
        for tname in train_names:
            score = best_agreement(gvol, train_vols[tname])
            if score > best_score:
                best_score = score
                best_name  = tname
        gen_agreements.append(best_score)
        gen_rows.append({"generated": gname,
                         "nearest_train": best_name,
                         "agreement": f"{best_score:.4f}"})
        print(f"    {gname} → {best_name}  agr={best_score:.4f}")

    # ── Train → nearest OTHER training crop (baseline) ─────────────────────
    print("\n  [2/2] Train → nearest OTHER train (baseline) ...")
    baseline_agreements: list[float] = []

    for i, tname in enumerate(train_names):
        tvol       = train_vols[tname]
        best_score = -1.0
        for j, other in enumerate(train_names):
            if i == j:
                continue
            score = best_agreement(tvol, train_vols[other])
            if score > best_score:
                best_score = score
        baseline_agreements.append(best_score)
        print(f"    {tname} → nearest other  agr={best_score:.4f}")

    # ── Summary ─────────────────────────────────────────────────────────────
    gen_arr  = np.array(gen_agreements)
    base_arr = np.array(baseline_agreements)

    delta   = gen_arr.mean() - base_arr.mean()
    verdict = ("MEMORIZATION SUSPECTED (gen mean > baseline mean + 0.05)"
               if delta > 0.05
               else "OK — within baseline (no memorization detected)")

    print("\n" + "=" * 64)
    print("  Generated → nearest train:")
    print(f"    mean={gen_arr.mean():.4f}  std={gen_arr.std():.4f}  "
          f"min={gen_arr.min():.4f}  max={gen_arr.max():.4f}")
    print("  Train → nearest OTHER train (baseline):")
    print(f"    mean={base_arr.mean():.4f}  std={base_arr.std():.4f}  "
          f"min={base_arr.min():.4f}  max={base_arr.max():.4f}")
    print(f"\n  gen_mean − baseline_mean = {delta:+.4f}")
    print(f"  Verdict: {verdict}")
    print("=" * 64)

    # ── Write CSV ────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["generated", "nearest_train", "agreement"])
        writer.writeheader()
        writer.writerows(gen_rows)
        writer.writerow({"generated": "SUMMARY_gen_mean",
                         "nearest_train": "", "agreement": f"{gen_arr.mean():.4f}"})
        writer.writerow({"generated": "SUMMARY_gen_std",
                         "nearest_train": "", "agreement": f"{gen_arr.std():.4f}"})
        writer.writerow({"generated": "SUMMARY_base_mean",
                         "nearest_train": "train_vs_train",
                         "agreement": f"{base_arr.mean():.4f}"})
        writer.writerow({"generated": "SUMMARY_base_std",
                         "nearest_train": "train_vs_train",
                         "agreement": f"{base_arr.std():.4f}"})
        writer.writerow({"generated": "SUMMARY_verdict",
                         "nearest_train": "", "agreement": verdict})
    print(f"\n  Results → {output_csv.resolve()}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Memorization check: compare generated vs training crop agreement")
    p.add_argument("--train",     required=True,
                   help="Path to training crops directory")
    p.add_argument("--generated", required=True,
                   help="Path to generated structures directory")
    p.add_argument("--output",    required=True,
                   help="Output CSV path")
    args = p.parse_args()

    memo_check(
        train_dir  = Path(args.train),
        gen_dir    = Path(args.generated),
        output_csv = Path(args.output),
    )


if __name__ == "__main__":
    main()
