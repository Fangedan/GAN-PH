"""
4_CNNCT/plot_results.py
========================
Generates two publication-ready figures from the connectivity analysis CSVs:

  Figure 1 — S-value bar chart
    All 9 metrics side by side, color-coded by threshold:
      Green  >= 0.85  (OK)
      Yellow  0.70-0.85  (Marginal)
      Red    < 0.70  (Fail — feedback required)

  Figure 2 — Distribution comparison
    Side-by-side histograms for conn_Pore and active_tpb,
    showing the collapse in GAN-generated structures vs training data.

Usage (from inside 4_CNNCT/):
  python plot_results.py

  # Custom CSV paths:
  python plot_results.py --real synthetic_connectivity.csv
                         --generated generated_connectivity.csv
                         --svalues s_values_svalues.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

# S-value thresholds (Yu et al. 2025)
S_OK       = 0.85
S_MARGINAL = 0.70

# Colour palette
C_OK       = "#2ecc71"   # green
C_MARGINAL = "#f39c12"   # amber
C_FAIL     = "#e74c3c"   # red
C_REAL     = "#2980b9"   # blue  — training data
C_GEN      = "#e74c3c"   # red   — generated data

# Human-readable metric labels
METRIC_LABELS = {
    "conn_Ni":        "Ni\nConnectivity",
    "conn_YSZ":       "YSZ\nConnectivity",
    "conn_Pore":      "Pore\nConnectivity",
    "total_tpb":      "Total TPB\nDensity",
    "active_tpb":     "Active TPB\nDensity",
    "active_tpb_frac":"Active TPB\nFraction",
    "tau_Ni":         "Tortuosity\nNi",
    "tau_YSZ":        "Tortuosity\nYSZ",
    "tau_Pore":       "Tortuosity\nPore",
}


def bar_colour(s):
    if np.isnan(s):
        return "#cccccc"
    elif s >= S_OK:
        return C_OK
    elif s >= S_MARGINAL:
        return C_MARGINAL
    else:
        return C_FAIL


# ── Figure 1: S-value bar chart ───────────────────────────────────────────────

def plot_svalues(svalues_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(svalues_csv, encoding="latin-1")

    # Normalise column names — the CSV has 'metric' and 's_value'
    df.columns = [c.strip().lower() for c in df.columns]
    metrics = df["metric"].tolist()
    svals   = pd.to_numeric(df["s_value"], errors="coerce").tolist()

    labels  = [METRIC_LABELS.get(m, m) for m in metrics]
    colours = [bar_colour(s) for s in svals]
    x       = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.bar(x, svals, color=colours, width=0.6,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # Threshold lines
    ax.axhline(S_OK,       color=C_OK,       linestyle="--",
               linewidth=1.2, alpha=0.8, zorder=2,
               label=f"OK threshold ({S_OK})")
    ax.axhline(S_MARGINAL, color=C_MARGINAL,  linestyle="--",
               linewidth=1.2, alpha=0.8, zorder=2,
               label=f"Marginal threshold ({S_MARGINAL})")

    # Value labels on bars
    for bar, s in zip(bars, svals):
        if not np.isnan(s):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{s:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    color="#333333", fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=C_OK,       label=f"OK  (S ≥ {S_OK})"),
        mpatches.Patch(color=C_MARGINAL, label=f"Marginal  ({S_MARGINAL} ≤ S < {S_OK})"),
        mpatches.Patch(color=C_FAIL,     label=f"Fail  (S < {S_MARGINAL})"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              framealpha=0.9, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("S-value  (Yu et al. 2025)", fontsize=12)
    ax.set_title(
        "Connectivity & Transport Fidelity: Training Data vs GAN-Generated\n"
        "S-value comparison — metrics the GAN was NOT trained to optimise",
        fontsize=12, pad=14
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2: Distribution comparison ─────────────────────────────────────────

def plot_distributions(real_csv: Path, gen_csv: Path, out_path: Path) -> None:
    real = pd.read_csv(real_csv)
    gen  = pd.read_csv(gen_csv)

    # Metrics to plot: (column, xlabel, title_suffix)
    panels = [
        ("conn_Pore",  "Pore connectivity fraction",
         "Pore Phase Connectivity\n(fraction of pore voxels in percolating network)"),
        ("active_tpb", "Active TPB density  (µm⁻²)",
         "Active Triple-Phase Boundary Density\n"
         "(TPB sites where Ni + YSZ + Pore all percolate)"),
        ("conn_Ni",    "Ni connectivity fraction",
         "Ni Phase Connectivity"),
        ("tau_Pore",   "Tortuosity factor",
         "Pore Tortuosity Factor"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (col, xlabel, title) in zip(axes, panels):
        r_vals = pd.to_numeric(real[col], errors="coerce").dropna().values
        g_vals = pd.to_numeric(gen[col],  errors="coerce").dropna().values

        # Skip panel if both datasets have no data (e.g. --no-tau was used)
        if len(r_vals) == 0 and len(g_vals) == 0:
            ax.set_visible(False)
            continue

        # Shared bin range
        all_vals = np.concatenate([r_vals, g_vals])
        if len(all_vals) == 0:
            ax.set_visible(False)
            continue
        lo, hi   = all_vals.min(), all_vals.max()
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 20)

        ax.hist(r_vals, bins=bins, color=C_REAL, alpha=0.65,
                label=f"Training data  (n={len(r_vals)})",
                edgecolor="white", linewidth=0.5)
        ax.hist(g_vals, bins=bins, color=C_GEN,  alpha=0.65,
                label=f"GAN-generated  (n={len(g_vals)})",
                edgecolor="white", linewidth=0.5)

        # Mean lines
        ax.axvline(r_vals.mean(), color=C_REAL, linestyle="--",
                   linewidth=1.5, alpha=0.9,
                   label=f"Mean training: {r_vals.mean():.3f}")
        ax.axvline(g_vals.mean(), color=C_GEN,  linestyle="--",
                   linewidth=1.5, alpha=0.9,
                   label=f"Mean generated: {g_vals.mean():.3f}")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(title, fontsize=10, pad=8)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Distribution Comparison: Training Data vs GAN-Generated Structures\n"
        "GAN + connectivity loss — pore connectivity and active TPB now match training data",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot S-value comparison and distribution figures"
    )
    parser.add_argument("--real",      default="synthetic_connectivity.csv",
                        help="CSV from analyze.py on training data")
    parser.add_argument("--generated", default="generated_connectivity.csv",
                        help="CSV from analyze.py on GAN-generated data")
    parser.add_argument("--svalues",   default="s_values_svalues.csv",
                        help="S-value report CSV from analyze.py --compare")
    parser.add_argument("--outdir",    default=".",
                        help="Output directory for figures (default: current dir)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    real_csv   = Path(args.real)
    gen_csv    = Path(args.generated)
    sval_csv   = Path(args.svalues)

    missing = [p for p in [real_csv, gen_csv, sval_csv] if not p.exists()]
    if missing:
        print("Missing input files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun analyze.py first to generate these CSVs.")
        return

    print("Generating figures...\n")

    plot_svalues(
        sval_csv,
        outdir / "fig1_svalues.png"
    )

    plot_distributions(
        real_csv, gen_csv,
        outdir / "fig2_distributions.png"
    )

    print("\nDone.")
    print(f"  fig1_svalues.png      — S-value bar chart")
    print(f"  fig2_distributions.png — Distribution comparison")
    print(f"\nAdd these to your presentation or commit them to the repo.")


if __name__ == "__main__":
    main()
