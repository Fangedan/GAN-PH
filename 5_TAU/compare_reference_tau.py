"""
5_TAU/compare_reference_tau.py
================================
Compare our taufactor pipeline values against Prof. Jin's reference tau values.

Reads reference_tau.csv (filled in by user), looks up the corresponding
taufactor value from tau_labels.csv (local cache produced by compute_tau_labels.py),
and reports per-row absolute and relative error plus a summary.

If reference_tau.csv contains only the template header row (no data rows),
the script exits with a clear message.

Usage (run from inside 5_TAU/):
    conda run -n ganph --no-capture-output python compare_reference_tau.py
    conda run -n ganph --no-capture-output python compare_reference_tau.py \\
        --reference reference_tau.csv --labels tau_labels.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path


def load_reference(path: Path) -> list[dict]:
    """Load reference_tau.csv, skipping comment lines."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for r in reader:
            raw = r.get("tau_reference", "").strip()
            tau_ref = float("nan") if raw.lower() in ("nan", "") else float(raw)
            rows.append({
                "structure_id": r["structure_id"].strip(),
                "phase":        r["phase"].strip(),
                "tau_reference": tau_ref,
                "source_note":  r.get("source_note", "").strip(),
            })
    return rows


def load_labels(path: Path) -> dict:
    """Load tau_labels.csv -> dict: structure_name -> {tau_Ni, tau_YSZ, tau_Pore}"""
    mapping = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            entry = {}
            for ph in ("Ni", "YSZ", "Pore"):
                raw = row.get(f"tau_{ph}", "").strip()
                entry[f"tau_{ph}"] = (
                    float("nan") if raw.lower() in ("nan", "") else float(raw)
                )
            mapping[row["name"].strip()] = entry
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Compare taufactor pipeline values against reference tau values."
    )
    parser.add_argument("--reference", type=Path, default=Path("reference_tau.csv"),
                        help="CSV with reference values (default: reference_tau.csv)")
    parser.add_argument("--labels",    type=Path, default=Path("tau_labels.csv"),
                        help="Pipeline-computed labels (default: tau_labels.csv)")
    args = parser.parse_args()

    if not args.reference.exists():
        print(f"ERROR: {args.reference} not found.")
        print("Create it or point --reference at the correct path.")
        sys.exit(1)

    ref_rows = load_reference(args.reference)

    if not ref_rows:
        print(f"reference_tau.csv contains no data rows (only the template header).")
        print("Fill in Prof. Jin's reference values to run the comparison.")
        sys.exit(0)

    if not args.labels.exists():
        print(f"ERROR: {args.labels} not found.")
        print("Run compute_tau_labels.py first to generate per-structure tau values.")
        sys.exit(1)

    label_map = load_labels(args.labels)

    print()
    print("=" * 72)
    print(" 5_TAU/compare_reference_tau.py  --  Accuracy comparison")
    print(f" Reference file : {args.reference}  ({len(ref_rows)} rows)")
    print(f" Pipeline labels: {args.labels}  ({len(label_map)} structures)")
    print("=" * 72)
    print(
        f"\n {'Structure':<20} {'Phase':<6} {'Ref tau':>10} {'Pipeline tau':>13}"
        f" {'Abs err':>9} {'Rel err %':>10}  Source"
    )
    print(f" {'-'*20} {'-'*6} {'-'*10} {'-'*13} {'-'*9} {'-'*10}  {'-'*30}")

    abs_errs = []
    rel_errs = []
    n_missing = 0

    for row in ref_rows:
        sid   = row["structure_id"]
        phase = row["phase"]
        tau_ref = row["tau_reference"]

        pipeline_entry = label_map.get(sid)
        if pipeline_entry is None:
            print(
                f" {'  ' + sid:<20} {phase:<6} {tau_ref:>10.3f} {'N/A':>13}"
                f" {'N/A':>9} {'N/A':>10}  {row['source_note'][:30]}"
                f"  [structure not in {args.labels.name}]"
            )
            n_missing += 1
            continue

        tau_pipe = pipeline_entry.get(f"tau_{phase}", float("nan"))

        if math.isnan(tau_ref) or math.isnan(tau_pipe):
            abs_err_str = "N/A"
            rel_err_str = "N/A"
        else:
            abs_err = abs(tau_pipe - tau_ref)
            rel_err = abs_err / tau_ref * 100.0 if tau_ref != 0 else float("nan")
            abs_err_str = f"{abs_err:.4f}"
            rel_err_str = f"{rel_err:.2f}"
            abs_errs.append(abs_err)
            rel_errs.append(rel_err)

        print(
            f" {sid:<20} {phase:<6} {tau_ref:>10.3f} {tau_pipe:>13.3f}"
            f" {abs_err_str:>9} {rel_err_str:>10}  {row['source_note'][:30]}"
        )

    print()
    if abs_errs:
        print(f" Summary ({len(abs_errs)} comparable rows, {n_missing} missing):")
        print(f"   Mean absolute error : {sum(abs_errs)/len(abs_errs):.4f}")
        print(f"   Max  absolute error : {max(abs_errs):.4f}")
        print(f"   Mean relative error : {sum(rel_errs)/len(rel_errs):.2f}%")
        print(f"   Max  relative error : {max(rel_errs):.2f}%")
        print()
        print(" Interpretation:")
        mean_rel = sum(rel_errs) / len(rel_errs)
        if mean_rel < 5.0:
            print("   Mean relative error < 5% -- pipeline accuracy is GOOD.")
        elif mean_rel < 15.0:
            print("   Mean relative error 5-15% -- MARGINAL; check worst residuals.")
        else:
            print("   Mean relative error > 15% -- POOR accuracy; investigate pipeline.")
    else:
        print(" No comparable rows (all reference structures missing from tau_labels.csv).")

    print("=" * 72)


if __name__ == "__main__":
    main()
