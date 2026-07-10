"""
4_CNNCT/cathode_eval_hardened.py
=================================
Evaluation hardening for cathode run0.

Tasks
-----
1. Re-score all 4 checkpoints (ep54/108/162/216) against the full
   24-crop stride-32 reference set instead of the original 6 stride-64 crops.

2. Bootstrap S-values (500 iterations, resample ref + gen with replacement)
   and report 16th-84th percentile CI per metric per checkpoint.

3. Pseudo-ceiling: split the 24 reference crops into spatially disjoint halves
   and compute real-vs-real S per metric (see geometry note below).

Geometry note — pseudo-ceiling split
--------------------------------------
The manifest has y-origin values {0, 32, 64, 96} with cube_size=64 and stride=32.
Any pair of adjacent y-groups overlaps by 32 voxels (stride < cube_size).

Truly disjoint combinations available:
  Low  = y0 ∈ {0, 32}  → 12 crops, covers y = 0..95
  High = y0 = {96}      → 6  crops, covers y = 96..159
  max(low) = 95,  min(high) = 96  → zero shared voxels ✓

A 12-vs-12 split with NO overlap is geometrically impossible with this dataset
(stride=32, cube=64 on a 283-voxel Y axis).  The user's stated "y0 ≤ 64 vs
y0 ≥ 96, 12 vs 12, no shared voxels" is inconsistent: that cutpoint gives
18 vs 6, and y0=64 crops still share y=96..127 with y0=96 crops (32 voxels).

We report the 12-vs-6 disjoint split with a clear caveat.

Outputs
-------
  4_CNNCT/cathode_run0_hardened.csv   point estimates + 16th/84th pct CI
  4_CNNCT/cathode_run0_ceiling.csv    pseudo-ceiling S-values
  stdout: formatted summary tables
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless, per project convention

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import analyze
from configs.dataset_config import get_config

# Apply cathode config BEFORE any analyze functions are called
analyze._apply_config(get_config("cathode_s1_supercrop"))

# ── Constants ─────────────────────────────────────────────────────────────────
N_BOOT   = 500
CI_LO    = 16     # ~1-sigma below mean
CI_HI    = 84     # ~1-sigma above mean
RNG_SEED = 42

CHECKPOINTS = [
    ("ep054", REPO_ROOT / "generated_cathode_run0_ep54"),
    ("ep108", REPO_ROOT / "generated_cathode_run0_ep108"),
    ("ep162", REPO_ROOT / "generated_cathode_run0_ep162"),
    ("ep216", REPO_ROOT / "generated_cathode_run0_final"),
]

SCORED_METRICS = [
    "conn_LSCF", "conn_GDC", "conn_Pore",
    "total_tpb", "active_tpb", "active_tpb_frac",
    "dpb_LSCF_GDC", "dpb_LSCF_Pore", "dpb_perc_LSCF_Pore", "dpb_GDC_Pore",
]

REF_CSV  = SCRIPT_DIR / "cathode_s1_str32_metrics.csv"
MANIFEST = REPO_ROOT / "cathode_crops_s1_str32" / "manifest.csv"
REF_DIR  = REPO_ROOT / "cathode_crops_s1_str32"


# ── Helper functions ──────────────────────────────────────────────────────────

def _flag(s: float) -> str:
    if s >= 0.85:
        return "OK"
    if s >= 0.70:
        return "M "
    return "F "


def _resample(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(arr), len(arr))
    return arr[idx]


def bootstrap_s(ref_v: np.ndarray, gen_v: np.ndarray,
                n_iter: int = N_BOOT,
                rng: np.random.Generator = None) -> tuple:
    """Return (p16, p84) bootstrap CI for the S-value."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    boot = [
        analyze.s_value(_resample(ref_v, rng), _resample(gen_v, rng))
        for _ in range(n_iter)
    ]
    lo = float(np.nanpercentile(boot, CI_LO))
    hi = float(np.nanpercentile(boot, CI_HI))
    return lo, hi


def load_ref_csv(path: Path, metrics: list) -> dict:
    """Load reference metrics from calibration CSV into {metric: np.ndarray}.
    Skips SUMMARY_* rows appended by cathode_calibration.py."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("structure", "").startswith("SUMMARY_"):
                continue
            rows.append(row)
    result = {}
    for m in metrics:
        vals = np.array([float(r[m]) for r in rows if m in r])
        result[m] = vals
        result["_ids"] = [r.get("structure", r.get("name", "")) for r in rows]
    return result


def load_gen_metrics(gen_dir: Path, metrics: list) -> dict:
    """Run analyze_dataset on a generated directory; return {metric: np.ndarray}."""
    results = analyze.analyze_dataset(gen_dir, compute_tau=False)
    out = {}
    for m in metrics:
        vals = np.array([float(r[m]) for r in results if m in r])
        vals = vals[~np.isnan(vals)]
        out[m] = vals
    return out


def load_manifest_y() -> dict:
    """Return {struct_id: origin_y} from the str32 manifest."""
    mapping = {}
    with open(MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["struct_id"]] = int(row["origin_y"])
    return mapping


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RNG_SEED)

    # ── Load reference metrics ────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  Loading reference metrics (str32, 24 crops) from cache CSV ...")
    print("="*72)
    ref_all = load_ref_csv(REF_CSV, SCORED_METRICS)
    ref_ids = ref_all.pop("_ids", [])
    print(f"  N_ref = {len(ref_ids)} crops")

    # ── Load generated metrics ────────────────────────────────────────────────
    gen_metrics_all = {}
    for ckpt_name, gen_dir in CHECKPOINTS:
        if not gen_dir.exists():
            print(f"\n  WARNING: {gen_dir} not found — skipping {ckpt_name}")
            continue
        print(f"\nLoading {ckpt_name} from {gen_dir.name} ...")
        gen_metrics_all[ckpt_name] = load_gen_metrics(gen_dir, SCORED_METRICS)
        n = len(gen_metrics_all[ckpt_name].get("conn_LSCF", []))
        print(f"  N_gen = {n} structures")

    # ── Point estimates + bootstrap CIs ──────────────────────────────────────
    print("\n" + "="*72)
    print("  Re-scored S-values (ref = 24 str32 crops)")
    print("  Bootstrap CI: 16th–84th percentile, 500 iterations")
    print("="*72)

    hardened_rows = []
    for metric in SCORED_METRICS:
        ref_v = ref_all[metric]
        row = {"metric": metric}
        line_parts = [f"\n  {metric:<26}"]
        for ckpt_name, gen_m in gen_metrics_all.items():
            gen_v = gen_m[metric]
            s     = analyze.s_value(ref_v, gen_v)
            lo, hi = bootstrap_s(ref_v, gen_v, rng=rng)
            row[f"{ckpt_name}_s"]  = round(s,  4)
            row[f"{ckpt_name}_lo"] = round(lo, 4)
            row[f"{ckpt_name}_hi"] = round(hi, 4)
            line_parts.append(f"  {ckpt_name}: {s:.4f} [{lo:.4f}–{hi:.4f}] {_flag(s)}")
        print("".join(line_parts))
        hardened_rows.append(row)

    # Write hardened CSV
    col_order = ["metric"] + [
        f"{c}_{x}"
        for c in gen_metrics_all
        for x in ("s", "lo", "hi")
    ]
    out_path = SCRIPT_DIR / "cathode_run0_hardened.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=col_order)
        w.writeheader()
        w.writerows(hardened_rows)
    print(f"\n  Saved → {out_path.name}")

    # ── Checkpoint summary table ──────────────────────────────────────────────
    print("\n" + "="*72)
    print("  Checkpoint summary  (OK = s≥0.85, M = 0.70–0.85, F = <0.70)")
    print("="*72)
    ckpt_names = list(gen_metrics_all.keys())
    hdr = f"  {'Metric':<28}" + "".join(f"  {c:<30}" for c in ckpt_names)
    print(hdr)
    for row in hardened_rows:
        m = row["metric"]
        cols = []
        for c in ckpt_names:
            s  = row[f"{c}_s"]
            lo = row[f"{c}_lo"]
            hi = row[f"{c}_hi"]
            cols.append(f"{s:.4f}[{lo:.4f}–{hi:.4f}]{_flag(s)}")
        print(f"  {m:<28}" + "".join(f"  {v:<30}" for v in cols))

    # ── Overlap assessment (for report writing) ───────────────────────────────
    print("\n" + "="*72)
    print("  Checkpoint-to-checkpoint distinguishability assessment")
    print("  (do CI bands overlap? → 'statistically tied' if yes)")
    print("="*72)
    for i, ci in enumerate(ckpt_names):
        for j, cj in enumerate(ckpt_names):
            if j <= i:
                continue
            print(f"\n  {ci} vs {cj}:")
            tied = []
            distinct = []
            for row in hardened_rows:
                m  = row["metric"]
                lo_i = row[f"{ci}_lo"]
                hi_i = row[f"{ci}_hi"]
                lo_j = row[f"{cj}_lo"]
                hi_j = row[f"{cj}_hi"]
                # Overlap test: bands [lo_i,hi_i] and [lo_j,hi_j] overlap iff
                # not (hi_i < lo_j or hi_j < lo_i)
                overlaps = not (hi_i < lo_j or hi_j < lo_i)
                if overlaps:
                    tied.append(m)
                else:
                    distinct.append(m)
            if distinct:
                print(f"    Distinguishable (no CI overlap): {', '.join(distinct)}")
            else:
                print(f"    Distinguishable: (none)")
            print(f"    Statistically tied (CI overlap): {', '.join(tied) if tied else '(none)'}")

    # ── Pseudo-ceiling (spatially disjoint halves) ────────────────────────────
    print("\n" + "="*72)
    print("  Pseudo-ceiling: real-vs-real S-values")
    print("  Low half  (y0 ∈ {0,32},  N=12): covers y =   0..95")
    print("  High half (y0 = 96,       N=6 ): covers y =  96..159")
    print("  These halves share zero voxels (max_low=95, min_high=96).")
    print("  Caveat: within-half stride-32 crops overlap by 32 voxels,")
    print("    inflating within-half homogeneity and thus the ceiling S.")
    print("  Note: a 12-vs-12 disjoint split is geometrically impossible")
    print("    with this dataset (stride=32, cube=64 on 283-voxel Y axis).")
    print("="*72)

    y_map = load_manifest_y()
    low_ids  = {sid for sid, y in y_map.items() if y <= 32}
    high_ids = {sid for sid, y in y_map.items() if y >= 96}
    assert len(low_ids)  == 12, f"Expected 12 low crops, got {len(low_ids)}"
    assert len(high_ids) == 6,  f"Expected 6 high crops, got {len(high_ids)}"

    # Re-load reference CSV with struct IDs to split
    full_rows = []
    with open(REF_CSV, newline="") as f:
        for row in csv.DictReader(f):
            full_rows.append(row)

    low_rows  = [r for r in full_rows if r["structure"] in low_ids]
    high_rows = [r for r in full_rows if r["structure"] in high_ids]
    print(f"\n  Confirmed: N_low={len(low_rows)}, N_high={len(high_rows)}")

    ceiling_rows = []
    for metric in SCORED_METRICS:
        low_v  = np.array([float(r[metric]) for r in low_rows])
        high_v = np.array([float(r[metric]) for r in high_rows])
        s      = analyze.s_value(low_v, high_v)
        lo, hi = bootstrap_s(low_v, high_v, rng=rng)
        print(f"  {metric:<28}: {s:.4f}  [{lo:.4f}–{hi:.4f}]  {_flag(s)}")
        ceiling_rows.append({
            "metric":    metric,
            "ceiling_s": round(s,  4),
            "ci_lo":     round(lo, 4),
            "ci_hi":     round(hi, 4),
        })

    ceil_path = SCRIPT_DIR / "cathode_run0_ceiling.csv"
    with open(ceil_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "ceiling_s", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(ceiling_rows)
    print(f"\n  Saved → {ceil_path.name}")

    print("\n" + "="*72)
    print("  Done.")
    print("="*72)


if __name__ == "__main__":
    main()
