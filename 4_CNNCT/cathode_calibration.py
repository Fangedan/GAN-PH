"""
4_CNNCT/cathode_calibration.py
================================
Compute raw metric distributions (VF, connectivity, TPB, DPB) for a set of
cathode crop directories.  No S-value comparison — this is a pre-training
calibration pass to establish baseline numbers and calibrate future loss
weights for cathode runs.

Intended use
------------
Run ONCE on the training crops (str32) and reference crops (str64) to get:
  - Mean VF per phase → conditions the generator via one-hot encoding
  - Mean TPB density  → calibration target for a cathode tpb-proxy loss
  - DPB densities     → calibration targets for future DPB-aware loss terms
  - Connectivity stats → baseline expectation for S-value scoring

Usage (from repo root)
----------------------
  conda run -n ganph --no-capture-output python -u 4_CNNCT/cathode_calibration.py \\
      --input  cathode_crops_s1_str32 \\
      --config cathode_s1_supercrop   \\
      --output 4_CNNCT/cathode_s1_str32_metrics.csv

  conda run -n ganph --no-capture-output python -u 4_CNNCT/cathode_calibration.py \\
      --input  cathode_crops_s1_str64 \\
      --config cathode_s1_supercrop   \\
      --output 4_CNNCT/cathode_s1_str64_metrics.csv

Outputs
-------
  --output CSV : one row per structure; final rows = MEAN / STD / MIN / MAX
  stdout       : summary table (mean ± std for each metric)

Notes
-----
- Tortuosity is SKIPPED here (slow; use analyze.py if needed).
- DPB self-check (sum(dpb)==SSA×vf) is verified for the first structure only.
- Script is read-only: it never modifies crops or model files.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# ── Import analyze.py's metric functions via the standard config mechanism ──────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze                            # noqa: E402  (same package dir)
from configs.dataset_config import get_config


def _compute_row(vol: np.ndarray) -> dict:
    """Compute all non-tortuosity metrics for a single structure volume."""
    total_vox = vol.size

    # ── Volume fractions ────────────────────────────────────────────────────────
    row = {}
    for name, val in analyze.PHASES.items():
        row[f"vf_{name}"] = float(np.sum(vol == val)) / total_vox

    # ── Connectivity (z-percolation) ────────────────────────────────────────────
    for name, val in analyze.PHASES.items():
        row[f"conn_{name}"] = analyze.compute_phase_connectivity(vol, val)

    # ── TPB densities ────────────────────────────────────────────────────────────
    total_tpb, active_tpb, active_frac = analyze.compute_tpb_densities(vol)
    row["total_tpb"]     = total_tpb
    row["active_tpb"]    = active_tpb
    row["active_tpb_frac"] = active_frac

    # ── DPB densities ────────────────────────────────────────────────────────────
    dpb = analyze.compute_dpb_densities(vol)
    row.update(dpb)

    return row


def run_calibration(input_dir: Path, output_csv: Path) -> None:
    struct_dirs = sorted(d for d in input_dir.iterdir()
                         if d.is_dir() and d.name.startswith("structure_"))
    if not struct_dirs:
        raise FileNotFoundError(f"No structure_XXXX dirs found in {input_dir}")

    print(f"\n{'='*64}")
    print(f"  Cathode calibration: {input_dir}")
    print(f"  Config: {analyze._ACTIVE_CFG.name}")
    print(f"  Structures: {len(struct_dirs)}")
    print(f"{'='*64}\n")

    rows = []
    first = True
    for d in struct_dirs:
        try:
            vol = analyze.load_structure(d)
        except Exception as e:
            print(f"  Warning: {d.name} — {e}")
            continue

        row = _compute_row(vol)
        row["structure"] = d.name
        rows.append(row)

        if first:
            # DPB self-check on first structure
            dpb_only = {k: v for k, v in row.items() if k.startswith("dpb_")}
            analyze._verify_dpb_ssa_consistency(vol, dpb_only)
            first = False

        vf_str = "  ".join(
            f"{n}={row[f'vf_{n}']:.3f}" for n in analyze.PHASES
        )
        conn_str = "  ".join(
            f"conn_{n}={row[f'conn_{n}']:.3f}" for n in analyze.PHASES
        )
        print(f"  {d.name}  [{vf_str}]  tpb={row['total_tpb']:.4f}  [{conn_str}]")

    if not rows:
        print("  No structures loaded — nothing to summarize.")
        return

    # ── Summary statistics ───────────────────────────────────────────────────────
    metric_keys = [k for k in rows[0] if k != "structure"]
    data = {k: np.array([r[k] for r in rows]) for k in metric_keys}

    print(f"\n{'='*64}")
    print(f"  Summary (N={len(rows)} structures)")
    print(f"{'='*64}")
    col_w = max(len(k) for k in metric_keys) + 2
    print(f"  {'Metric':<{col_w}}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*col_w}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for k in metric_keys:
        v = data[k]
        print(f"  {k:<{col_w}}  {v.mean():8.4f}  {v.std():8.4f}  {v.min():8.4f}  {v.max():8.4f}")

    # ── Write CSV ────────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["structure"] + metric_keys
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        # Summary rows
        for stat, fn in [("MEAN", np.mean), ("STD", np.std),
                         ("MIN", np.min),  ("MAX", np.max)]:
            summary_row = {"structure": f"SUMMARY_{stat}"}
            summary_row.update({k: f"{fn(data[k]):.6f}" for k in metric_keys})
            w.writerow(summary_row)

    print(f"\n  → {output_csv.resolve()}\n")


def main():
    p = argparse.ArgumentParser(
        description="Compute raw metric distributions for cathode crops (no S-values)")
    p.add_argument("--input",  required=True,
                   help="Directory containing structure_XXXX/ folders")
    p.add_argument("--config", default="cathode_s1_supercrop",
                   help="Dataset config name (default: cathode_s1_supercrop)")
    p.add_argument("--output", required=True,
                   help="Output CSV path")
    args = p.parse_args()

    cfg = get_config(args.config)
    analyze._apply_config(cfg)

    run_calibration(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
