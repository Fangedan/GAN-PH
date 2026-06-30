"""
Compute per-phase tortuosity labels for all structures in a data directory.

Run this ONCE on real_data before training the tau-net. It takes ~30-90 min
on CPU (one Laplace solve per phase per structure). Output:

  tau_labels.csv    — per-structure τ values; NaN when phase doesn't percolate
  tau_targets.json  — mean τ per phase across all valid structures
                      (these are the GAN loss targets, loaded by 1_GAN/main.py)

Usage (run from inside 5_TAU/):
  python compute_tau_labels.py --data ../real_data
  python compute_tau_labels.py --data ../real_data --output my_labels.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "4_CNNCT"))
from analyze import (
    load_structure, find_structure_dirs,
    compute_tortuosity, compute_phase_connectivity,
    PHASES,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute τ labels for all structures (taufactor required)."
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Parent folder containing structure_XXXX/ dirs "
                             "(e.g., ../real_data)")
    parser.add_argument("--output", type=Path, default=Path("tau_labels.csv"),
                        help="Per-structure CSV output (default: tau_labels.csv)")
    parser.add_argument("--targets-out", type=Path,
                        default=Path("tau_targets.json"),
                        help="JSON for mean τ targets (default: tau_targets.json)")
    args = parser.parse_args()

    try:
        import taufactor  # noqa: F401
    except ImportError:
        print("ERROR: taufactor not installed.\n"
              "  conda run -n ganph pip install taufactor")
        return

    struct_dirs = find_structure_dirs(args.data)
    n = len(struct_dirs)
    print(f"Found {n} structures in {args.data}")
    print("Computing tortuosity (each Laplace solve: ~10-60 s on CPU) ...")

    rows = []
    t0 = time.time()
    for idx, sd in enumerate(struct_dirs):
        print(f"  [{idx+1:03}/{n}] {sd.name} ...", end="", flush=True)
        t_s = time.time()
        vol = load_structure(sd)
        row = {"name": sd.name}
        for ph_name, ph_val in PHASES.items():
            row[f"tau_{ph_name}"] = compute_tortuosity(vol, ph_val)
        rows.append(row)
        dt = time.time() - t_s
        print(f"  τ_Ni={row['tau_Ni']:.3f}  τ_YSZ={row['tau_YSZ']:.3f}  "
              f"τ_Pore={row['tau_Pore']:.3f}  ({dt:.0f}s)")
        if idx == 0:
            eta_min = dt * (n - 1) / 60
            print(f"  → ETA ≈ {eta_min:.0f} min")

    total_min = (time.time() - t0) / 60
    print(f"\nTotal: {total_min:.1f} min")

    # Save per-structure CSV
    fieldnames = ["name", "tau_Ni", "tau_YSZ", "tau_Pore"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-structure τ → {args.output}")

    # Compute mean τ per phase (NaN-safe), save as GAN loss targets
    targets = {}
    for ph_name in PHASES:
        vals = [r[f"tau_{ph_name}"] for r in rows
                if not np.isnan(r[f"tau_{ph_name}"])]
        if vals:
            targets[ph_name] = float(np.mean(vals))
            print(f"  mean τ_{ph_name} = {targets[ph_name]:.3f}  "
                  f"(n_valid={len(vals)}/{n}, "
                  f"std={float(np.std(vals)):.3f})")
        else:
            targets[ph_name] = None
            print(f"  τ_{ph_name}: all NaN — phase never percolates, excluded from loss")

    with open(args.targets_out, "w") as f:
        json.dump(targets, f, indent=2)
    print(f"Saved τ targets → {args.targets_out}")


if __name__ == "__main__":
    main()
