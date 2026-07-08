"""
4_CNNCT/ceiling_calibration.py
================================
Measure the real-vs-real S-value ceiling: two finite samples drawn from the
SAME real distribution score S < 1 due to KS variance (n~50), so the achievable
ceiling is unknown. This script measures it.

Method
------
  - Split the 101 real structures into disjoint halves (51/50).
  - Treat one half as "real" and the other as "generated".
  - Compute S-values for all 9 metrics.
  - Repeat N=20 random splits (fixed seed), report per-metric mean +/- std and min.

Cache strategy
--------------
  Non-tau metrics (connectivity + TPB) are computed once and saved to
  ceiling_metrics_cache.csv (gitignored). Tortuosity is reused from
  5_TAU/tau_labels.csv (pre-computed by compute_tau_labels.py) — values
  match analyze.py's compute_tortuosity convention because that script
  uses the same function. Explicitly noted in the output.

  On subsequent runs the cache is loaded directly, skipping the ~5-10 min
  compute step, making the 20 splits nearly instantaneous.

Output
------
  4_CNNCT/CEILING.md      -- per-metric table (committed)
  4_CNNCT/ceiling_bar.png -- bar chart (gitignored, regenerate as needed)

Usage (run from inside 4_CNNCT/ or the repo root):
    conda run -n ganph --no-capture-output python 4_CNNCT/ceiling_calibration.py \\
        --real ../real_data --tau-labels ../5_TAU/tau_labels.csv
"""

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import (
    find_structure_dirs,
    analyze_structure,
    load_structure,
    s_value,
    S_OK,
    S_MARGINAL,
)

METRICS = [
    "conn_Ni", "conn_YSZ", "conn_Pore",
    "total_tpb", "active_tpb", "active_tpb_frac",
    "tau_Ni", "tau_YSZ", "tau_Pore",
]

CACHE_FILE = Path(__file__).parent / "ceiling_metrics_cache.csv"
OUTPUT_MD  = Path(__file__).parent / "CEILING.md"
OUTPUT_PNG = Path(__file__).parent / "ceiling_bar.png"

TAU_LABELS_DEFAULT = Path(__file__).parent.parent / "5_TAU" / "tau_labels.csv"


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_tau_labels(path: Path) -> dict:
    """Load 5_TAU/tau_labels.csv -> dict: structure_name -> {tau_Ni, tau_YSZ, tau_Pore}"""
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


def load_cache(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            entry = {"name": row["name"]}
            for m in METRICS:
                raw = row.get(m, "").strip()
                entry[m] = float("nan") if raw.lower() in ("nan", "") else float(raw)
            rows.append(entry)
    return rows


def save_cache(rows: list[dict], path: Path) -> None:
    fieldnames = ["name"] + METRICS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Metrics cache saved -> {path}")


# ── Metric computation ────────────────────────────────────────────────────────

def compute_all_metrics(struct_dirs: list, tau_by_name: dict,
                        tau_labels_path: Path) -> list[dict]:
    """
    Compute all 9 per-structure metrics. Non-tau metrics are computed fresh;
    tau metrics are loaded from tau_labels.csv.

    This design explicitly documents the cache source: tau values come from
    5_TAU/tau_labels.csv (taufactor via analyze.compute_tortuosity), which is
    the SAME function as analyze.analyze_structure uses. No discrepancy possible.
    """
    log.info(f"Computing non-tau metrics for {len(struct_dirs)} structures ...")
    log.info(f"Tau values: reused from {tau_labels_path} (same code path as analyze.py)")
    t0 = time.time()
    rows = []
    for idx, sd in enumerate(struct_dirs):
        vol = load_structure(sd)
        r = analyze_structure(vol, compute_tau=False)   # conn + TPB only (fast)
        r["name"] = sd.name

        # Inject tau from pre-computed labels
        tau_row = tau_by_name.get(sd.name, {})
        for ph in ("Ni", "YSZ", "Pore"):
            r[f"tau_{ph}"] = tau_row.get(f"tau_{ph}", float("nan"))

        rows.append(r)
        if (idx + 1) % 20 == 0 or idx == 0:
            eta = (time.time() - t0) / (idx + 1) * (len(struct_dirs) - idx - 1)
            log.info(f"  [{idx+1:03}/{len(struct_dirs)}]  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    log.info(f"Done in {elapsed:.1f}s")
    return rows


# ── Ceiling computation ───────────────────────────────────────────────────────

def run_ceiling_splits(all_rows: list[dict], n_splits: int = 20,
                       seed: int = 42) -> dict:
    """
    Split all_rows into 51/50 halves N times; compute S-values treating one
    half as 'real' and the other as 'generated'. Return per-metric list of S-values.
    """
    rng = np.random.default_rng(seed)
    n = len(all_rows)
    n_half = n // 2   # 50 (the 'generated' half); 'real' half gets n-n_half = 51

    metric_s_values = {m: [] for m in METRICS}

    for split_i in range(n_splits):
        perm = rng.permutation(n)
        real_rows = [all_rows[i] for i in perm[n_half:]]  # 51 structures
        gen_rows  = [all_rows[i] for i in perm[:n_half]]  # 50 structures

        for m in METRICS:
            real_vals = np.array([r[m] for r in real_rows], dtype=float)
            gen_vals  = np.array([r[m] for r in gen_rows],  dtype=float)
            s = s_value(real_vals, gen_vals)
            metric_s_values[m].append(s)

    return metric_s_values


# ── Output ────────────────────────────────────────────────────────────────────

def write_ceiling_md(metric_stats: dict, n_real: int, n_splits: int,
                     tau_labels_path: Path) -> None:
    """Write CEILING.md with results table."""
    lines = [
        "# Real-vs-Real S-Value Ceiling Calibration",
        "",
        "S-values compare two *finite* samples from the same distribution.",
        f"With n≈50 vs {n_real-50} real structures, KS variance means S < 1 even",
        "when both samples are drawn from the same distribution. This file measures",
        "the achievable ceiling so run5's scores can be interpreted correctly.",
        "",
        f"**Method:** split {n_real} real structures into 51/50 halves, treat one as",
        f"'generated', compute S-values. Repeated {n_splits} times (seed=42).",
        f"Non-tau metrics computed fresh; tau reused from `{tau_labels_path.name}`",
        "(same code path as `analyze.py`, no discrepancy possible).",
        "",
        "---",
        "",
        "## Per-metric ceiling (real-vs-real)",
        "",
        "| Metric | Mean S | ± Std | Min S | Interpretation | run5 S | Gap |",
        "|--------|--------|-------|-------|---------------|--------|-----|",
    ]

    # run5 reference S-values (from RESULTS.md / CLAUDE.md)
    run5_ref = {
        "conn_Ni": 0.866, "conn_YSZ": 0.704, "conn_Pore": 0.874,
        "total_tpb": 0.698, "active_tpb": 0.720, "active_tpb_frac": float("nan"),
        "tau_Ni": 0.760, "tau_YSZ": 0.479, "tau_Pore": 0.697,
    }

    for m in METRICS:
        vals = [v for v in metric_stats[m] if not math.isnan(v)]
        if not vals:
            lines.append(f"| {m} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        mean_s = np.mean(vals)
        std_s  = np.std(vals)
        min_s  = np.min(vals)

        if mean_s >= S_OK:
            interp = "OK -- ceiling reachable"
        elif mean_s >= S_MARGINAL:
            interp = "MARGINAL -- ceiling is marginal band"
        else:
            interp = "FAIL -- ceiling below 0.70 (intrinsic limit)"

        run5_s = run5_ref.get(m, float("nan"))
        if not math.isnan(run5_s):
            gap = f"{run5_s - mean_s:+.3f}"
            run5_str = f"{run5_s:.3f}"
        else:
            gap = "N/A"
            run5_str = "N/A"

        lines.append(
            f"| {m} | {mean_s:.3f} | {std_s:.3f} | {min_s:.3f} | {interp} | {run5_str} | {gap} |"
        )

    lines += [
        "",
        "**Gap** = run5 S-value minus real-vs-real ceiling mean.",
        "Negative gap = run5 scored *below* the real-vs-real ceiling (room to improve).",
        "Positive gap = run5 scored *above* ceiling (sampling variance; run5 closer than random real halves).",
        "",
        "---",
        "",
        "## How to interpret run5 scores",
        "",
        "A metric that scores S=0.70 (FAIL) against run5 may still be good if the ceiling",
        "is S=0.75 — the generator is close to the natural sampling limit, not far from it.",
        "Conversely, a metric with ceiling S=0.95 that run5 scores S=0.70 has genuine headroom.",
        "",
        "---",
        "",
        "## Bar chart",
        "",
        f"Regenerate: `conda run -n ganph --no-capture-output python 4_CNNCT/ceiling_calibration.py`",
        "Output: `ceiling_bar.png` (gitignored — contains derived real-structure statistics).",
    ]

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"CEILING.md written -> {OUTPUT_MD}")


def write_bar_chart(metric_stats: dict, n_splits: int) -> None:
    """Bar chart: per-metric ceiling mean +/- std, with 0.85 and 0.70 lines."""
    means, stds, labels = [], [], []
    for m in METRICS:
        vals = [v for v in metric_stats[m] if not math.isnan(v)]
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals)  if vals else 0.0)
        labels.append(m.replace("_", "\n"))

    x = np.arange(len(METRICS))
    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0",
                  alpha=0.8, width=0.6, error_kw={"elinewidth": 1.5})

    ax.axhline(S_OK,       color="#2ca02c", linewidth=1.5, linestyle="--", label=f"OK threshold ({S_OK})")
    ax.axhline(S_MARGINAL, color="#d62728", linewidth=1.5, linestyle=":",  label=f"MARGINAL threshold ({S_MARGINAL})")

    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("S-value (real-vs-real ceiling)")
    ax.set_title(
        f"Real-vs-Real S-value Ceiling  (101 structures, 51/50 splits, N={n_splits})\n"
        "Error bars = +/- 1 std across splits"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    log.info(f"Bar chart saved -> {OUTPUT_PNG}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure real-vs-real S-value ceiling via random 51/50 splits."
    )
    parser.add_argument("--real",       type=Path, default=Path("../real_data"),
                        help="Folder with real structure_XXXX/ dirs (default: ../real_data)")
    parser.add_argument("--tau-labels", type=Path, default=TAU_LABELS_DEFAULT,
                        help=f"tau_labels.csv path (default: {TAU_LABELS_DEFAULT})")
    parser.add_argument("--n-splits",   type=int, default=20,
                        help="Number of random splits (default: 20)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no-cache",   action="store_true",
                        help="Recompute all metrics even if cache exists")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(" 4_CNNCT/ceiling_calibration.py  --  Real-vs-real ceiling")
    log.info("=" * 60)

    # Load or compute per-structure metrics
    if CACHE_FILE.exists() and not args.no_cache:
        log.info(f"Loading metrics from cache: {CACHE_FILE}")
        all_rows = load_cache(CACHE_FILE)
        log.info(f"  {len(all_rows)} structures loaded from cache")
        log.info("  Tau values: from 5_TAU/tau_labels.csv (cached in this file)")
    else:
        if not args.real.exists():
            log.error(f"real_data directory not found: {args.real}")
            sys.exit(1)
        if not args.tau_labels.exists():
            log.error(f"tau_labels.csv not found: {args.tau_labels}")
            log.error("Run: conda run -n ganph python 5_TAU/compute_tau_labels.py --data real_data")
            sys.exit(1)

        struct_dirs = find_structure_dirs(args.real)
        tau_by_name = load_tau_labels(args.tau_labels)
        all_rows = compute_all_metrics(struct_dirs, tau_by_name, args.tau_labels)
        save_cache(all_rows, CACHE_FILE)

    n_real = len(all_rows)
    log.info(f"Running {args.n_splits} splits on {n_real} structures ...")
    t0 = time.time()
    metric_s_values = run_ceiling_splits(all_rows, n_splits=args.n_splits, seed=args.seed)
    log.info(f"Splits done in {time.time()-t0:.1f}s")

    # Print table
    print()
    print(f"{'Metric':<22} {'Mean':>7} {'±Std':>6} {'Min':>7}  Interpretation")
    print(f"{'-'*22} {'-'*7} {'-'*6} {'-'*7}  {'-'*30}")
    for m in METRICS:
        vals = [v for v in metric_s_values[m] if not math.isnan(v)]
        if not vals:
            print(f"  {m:<20} {'N/A':>7}")
            continue
        mean_s = np.mean(vals)
        std_s  = np.std(vals)
        min_s  = np.min(vals)
        if mean_s >= S_OK:
            flag = " OK"
        elif mean_s >= S_MARGINAL:
            flag = "  ~"
        else:
            flag = " !!"
        print(f"{flag} {m:<20} {mean_s:>7.3f} {std_s:>6.3f} {min_s:>7.3f}  ", end="")
        if mean_s >= S_OK:
            print("ceiling reachable (>= 0.85)")
        elif mean_s >= S_MARGINAL:
            print("ceiling is marginal band (0.70-0.85)")
        else:
            print("intrinsic limit below 0.70 -- metric may be unachievable")

    write_ceiling_md(metric_s_values, n_real=n_real, n_splits=args.n_splits,
                     tau_labels_path=args.tau_labels)
    write_bar_chart(metric_s_values, n_splits=args.n_splits)

    log.info("Done.")


if __name__ == "__main__":
    main()
