"""
0_PRV/check_voxel_isotropy.py
=============================
Empirically verify whether the z voxel size equals x/y for a cathode dataset.

For each phase, computes along x, y, z axes:
  - Two-point autocorrelation S2(lag): decay length = lag where excess drops to
    1/e of the initial excess S2(0) - vf_A^2
  - Chord-length distribution: mean chord length (in voxels)

Interpretation logic:
  If z voxel spacing differs from x/y by factor k, ALL phases' z statistics
  (measured in voxels) would differ from x/y by the SAME factor ~k.
  A consistent cross-phase z/x ratio != 1  -> voxel spacing calibration error.
  Phase-INCONSISTENT ratios               -> genuine microstructural anisotropy.
  Report x/y ratio as noise-floor baseline (x and y sizes are known and ~equal).

Cross-check: if per-phase struct.txt coordinate files exist alongside the source TIF,
read the z column and report mean z spacing directly -- this is the most direct evidence.

Output:
  0_PRV/VOXEL_ISOTROPY.md  -- verdict + per-phase statistics table

Usage (from repo root):
  conda run -n ganph --no-capture-output python 0_PRV/check_voxel_isotropy.py \\
      --config cathode_s1_supercrop
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from configs.dataset_config import get_config


# ── Struct.txt z-spacing extraction ──────────────────────────────────────────

def extract_z_spacing_from_struct(source_path: Path) -> dict | None:
    """
    Look for per-phase struct.txt files next to the source TIF and extract
    the z voxel spacing (mean difference between consecutive unique z values).

    Returns dict {phase_name: z_nm} or None if no struct files found.
    """
    parent = source_path.parent
    results = {}
    candidates = list(parent.glob("*_struct.txt"))
    if not candidates:
        return None

    print(f"\n  Found {len(candidates)} struct.txt file(s) alongside source TIF:")
    for f in candidates:
        print(f"    {f.name}  ({f.stat().st_size // 1024 // 1024} MB)")

    # Read the z column (column 2) from the first candidate to get the z step.
    # All phases share the same spatial grid, so one file suffices.
    ref_file = candidates[0]
    print(f"\n  Reading z column from {ref_file.name} ...")
    try:
        import pandas as pd
        # Rows per z level = X_count * Y_count; read enough to get 3+ z values.
        # The volume shape comes from the TIF; read up to 70000 rows to be safe.
        chunk = pd.read_csv(ref_file, sep=r"\s+", header=None,
                            nrows=70000, usecols=[2])
        z_vals = sorted(set(chunk[2].values))
        if len(z_vals) < 2:
            print("  WARNING: fewer than 2 unique z values found in first 70000 rows.")
            return None
        diffs = [z_vals[i + 1] - z_vals[i] for i in range(len(z_vals) - 1)]
        z_step_m = float(np.mean(diffs))
        z_nm = z_step_m * 1e9
        print(f"  z spacing (from struct.txt):  {z_nm:.4f} nm  "
              f"({len(z_vals)} unique z values found)")
        results["all_phases"] = z_nm
    except Exception as e:
        print(f"  WARNING: could not parse {ref_file.name}: {e}")
        return None

    return results


# ── Two-point autocorrelation ─────────────────────────────────────────────────

def two_point_s2_axis(binary: np.ndarray, axis: int, max_lag: int) -> np.ndarray:
    """
    Two-point correlation S2(lag) along one axis, averaged over the other two.

    S2(lag) = mean over all pairs [I(r) * I(r + lag*e_axis)]
    """
    n = binary.shape[axis]
    s2 = np.zeros(max_lag + 1)
    vol_f = binary.astype(np.float32)
    for lag in range(max_lag + 1):
        lo_sl = [slice(None)] * 3
        hi_sl = [slice(None)] * 3
        lo_sl[axis] = slice(0, n - lag) if lag > 0 else slice(None)
        hi_sl[axis] = slice(lag, n)     if lag > 0 else slice(None)
        s2[lag] = float(np.mean(vol_f[tuple(lo_sl)] * vol_f[tuple(hi_sl)]))
    return s2


def decay_length(s2: np.ndarray, vf: float) -> float:
    """
    Lag where S2(lag) - vf^2 drops to 1/e of its zero-lag excess.

    Returns np.nan if the correlation never decays to 1/e within the array.
    """
    plateau = vf ** 2
    excess0 = s2[0] - plateau
    if excess0 <= 0:
        return float("nan")
    target = plateau + excess0 / np.e
    for lag, val in enumerate(s2):
        if val <= target:
            return float(lag)
    return float("nan")


# ── Chord-length distribution ─────────────────────────────────────────────────

def mean_chord_length_axis(binary: np.ndarray, axis: int) -> float:
    """
    Mean chord length (voxels) along one axis, averaged over all 1D lines.
    A chord is a maximal run of True voxels along the axis.
    """
    n = binary.shape[axis]
    arr = np.moveaxis(binary, axis, 0).reshape(n, -1)
    total_chord = 0
    total_count = 0
    for col in arr.T:
        padded = np.concatenate([[0], col.astype(np.int8), [0]])
        d = np.diff(padded)
        starts = np.where(d == 1)[0]
        ends   = np.where(d == -1)[0]
        lengths = ends - starts
        total_chord += int(lengths.sum())
        total_count += len(lengths)
    if total_count == 0:
        return float("nan")
    return total_chord / total_count


# ── Main analysis ─────────────────────────────────────────────────────────────

def run(cfg_name: str) -> None:
    import tifffile

    cfg = get_config(cfg_name)
    source = cfg.source_file
    if source is None or not source.exists():
        print(f"ERROR: source_file not found for config {cfg_name!r}.")
        print("Make sure configs/local_paths.yaml has the correct path.")
        sys.exit(1)

    print("=" * 70)
    print(f" GAN-PH Voxel Isotropy Test")
    print(f" Config  : {cfg_name}")
    print(f" Source  : {source.name}")
    print("=" * 70)

    # ── Step 1: struct.txt direct evidence ───────────────────────────────────
    struct_result = extract_z_spacing_from_struct(source)
    if struct_result:
        z_struct_nm = struct_result["all_phases"]
    else:
        z_struct_nm = None
        print("\n  No struct.txt files found — skipping direct z-spacing check.")

    # ── Step 2: load and remap volume ────────────────────────────────────────
    print(f"\n  Loading {source.name} ...")
    raw_vol = tifffile.imread(str(source))
    print(f"  Shape: {raw_vol.shape}  dtype: {raw_vol.dtype}")
    print(f"  Unique raw values: {np.unique(raw_vol)}")

    # Remap source labels to BMP convention
    lmap = cfg.source_label_map
    phases = cfg.phases_dict()  # {name: bmp_value}
    name_to_bmp = phases  # {"LSCF":255, "GDC":127, "Pore":0}

    vol = np.zeros_like(raw_vol, dtype=np.uint8)
    if isinstance(lmap, dict):
        for src_val, phase_name in lmap.items():
            if phase_name in name_to_bmp:
                vol[raw_vol == int(src_val)] = name_to_bmp[phase_name]
    else:
        vol = raw_vol.copy()

    print(f"  Unique remapped values: {np.unique(vol)}")

    vx = cfg.voxel_size_x_um or 0.0
    vy = cfg.voxel_size_y_um or 0.0
    vz_known = cfg.voxel_size_z_um  # may be None

    axis_names = ["Z", "Y", "X"]   # TIF axis order: (Z, Y, X)
    axis_vx_nm = [
        (vz_known * 1000) if vz_known else None,
        vy * 1000,
        vx * 1000,
    ]

    max_lag = min(75, min(vol.shape) // 2)
    print(f"\n  Computing autocorrelation (max_lag={max_lag}) "
          f"and chord lengths for {len(phases)} phases ...")

    # ── Step 3: per-phase statistics ─────────────────────────────────────────
    stats: dict[str, dict] = {}  # {phase_name: {axis: {decay, chord}}}

    for phase_name, bmp_val in phases.items():
        binary = (vol == bmp_val)
        vf = float(binary.mean())
        print(f"\n  Phase {phase_name}  (bmp={bmp_val}  vf={vf:.3f})")
        stats[phase_name] = {"vf": vf}
        for ax, ax_label in enumerate(axis_names):
            s2 = two_point_s2_axis(binary, ax, max_lag)
            dl = decay_length(s2, vf)
            cl = mean_chord_length_axis(binary, ax)
            stats[phase_name][ax_label] = {"decay": dl, "chord": cl}
            print(f"    {ax_label}: decay_len={dl:.1f} vox  mean_chord={cl:.2f} vox")

    # ── Step 4: ratio analysis ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" RATIO ANALYSIS  (z/x and z/y ratios per phase)")
    print("=" * 70)
    print(" Expected if isotropic: z/x ≈ 1.0  z/y ≈ 1.0  (within noise)")
    print(f" x/y baseline (noise floor): "
          f"x={vx*1000:.3f} nm  y={vy*1000:.3f} nm  ratio={vx/vy:.4f}")

    decay_zx = []
    decay_zy = []
    chord_zx = []
    chord_zy = []

    for phase_name, pstats in stats.items():
        dZ = pstats["Z"]["decay"]
        dY = pstats["Y"]["decay"]
        dX = pstats["X"]["decay"]
        cZ = pstats["Z"]["chord"]
        cY = pstats["Y"]["chord"]
        cX = pstats["X"]["chord"]

        r_decay_zx = dZ / dX if (dX > 0 and not np.isnan(dX) and not np.isnan(dZ)) else float("nan")
        r_decay_zy = dZ / dY if (dY > 0 and not np.isnan(dY) and not np.isnan(dZ)) else float("nan")
        r_chord_zx = cZ / cX if (cX > 0 and not np.isnan(cX) and not np.isnan(cZ)) else float("nan")
        r_chord_zy = cZ / cY if (cY > 0 and not np.isnan(cY) and not np.isnan(cZ)) else float("nan")

        decay_zx.append(r_decay_zx)
        decay_zy.append(r_decay_zy)
        chord_zx.append(r_chord_zx)
        chord_zy.append(r_chord_zy)

        print(f"\n  {phase_name}:")
        print(f"    decay z/x={r_decay_zx:.3f}  z/y={r_decay_zy:.3f}")
        print(f"    chord z/x={r_chord_zx:.3f}  z/y={r_chord_zy:.3f}")

    valid_zx = [v for v in decay_zx + chord_zx if not np.isnan(v)]
    valid_zy = [v for v in decay_zy + chord_zy if not np.isnan(v)]
    mean_zx = float(np.mean(valid_zx)) if valid_zx else float("nan")
    mean_zy = float(np.mean(valid_zy)) if valid_zy else float("nan")
    std_zx  = float(np.std(valid_zx))  if valid_zx else float("nan")
    std_zy  = float(np.std(valid_zy))  if valid_zy else float("nan")

    # ── Step 5: verdict ───────────────────────────────────────────────────────
    # Priority: struct.txt direct measurement overrides autocorrelation inference.
    # If struct.txt is available and says z ≈ x (within 2%), the voxels are isotropic
    # regardless of autocorrelation ratios. Autocorrelation showing z/x < 1 in that
    # case means the MICROSTRUCTURE is shorter in z (genuine physical anisotropy).
    z_inferred_nm = None

    if z_struct_nm is not None:
        zx_ratio_struct = z_struct_nm / (vx * 1000)
        if abs(zx_ratio_struct - 1.0) < 0.02:
            verdict = (f"z voxel size = x/y (within noise, confirmed by struct.txt: "
                       f"z={z_struct_nm:.4f} nm vs x={vx*1000:.4f} nm, ratio={zx_ratio_struct:.4f}). "
                       f"Autocorrelation z/x≈{mean_zx:.2f} reflects genuine microstructural "
                       f"anisotropy (pillar geometry: features are shorter along z), "
                       f"NOT a calibration error.")
        else:
            z_inferred_nm = z_struct_nm
            verdict = (f"struct.txt gives z={z_struct_nm:.4f} nm vs x={vx*1000:.4f} nm "
                       f"(ratio {zx_ratio_struct:.4f}) — voxels are NOT isotropic. "
                       f"Update configs accordingly.")
    else:
        # Fall back to autocorrelation inference
        deviation_from_1 = abs(mean_zx - 1.0)
        phase_consistency = std_zx
        if deviation_from_1 < 0.05:
            verdict = "z voxel size = x/y (within noise, based on autocorrelation/chord)"
        elif phase_consistency < deviation_from_1 * 0.5:
            z_inferred_nm = vx * 1000 / mean_zx
            verdict = (f"z voxel spacing differs from x by consistent factor {mean_zx:.3f} "
                       f"→ likely calibration offset (inferred z ≈ {z_inferred_nm:.3f} nm). "
                       f"Confirm with struct.txt.")
        else:
            verdict = ("Autocorrelation z/x ratios inconsistent across phases → "
                       "genuine microstructural anisotropy rather than voxel spacing error.")

    print(f"\n  Mean z/x ratio (all phases, both metrics): {mean_zx:.4f} ± {std_zx:.4f}")
    print(f"  Mean z/y ratio (all phases, both metrics): {mean_zy:.4f} ± {std_zy:.4f}")
    print(f"\n  VERDICT: {verdict}")

    # ── Step 6: update YAML if z is now determined ────────────────────────────
    z_final_nm: float | None = None
    z_source = "unknown"

    if z_struct_nm is not None:
        z_final_nm = z_struct_nm
        z_source = "struct.txt direct measurement"
    elif z_inferred_nm is not None:
        z_final_nm = z_inferred_nm
        z_source = "inferred from autocorrelation/chord ratio"
    else:
        z_source = "not determined"

    if z_final_nm is not None:
        z_um = z_final_nm / 1000
        print(f"\n  Updating {cfg_name}.yaml: voxel_size_um.z = {z_um:.6f}  "
              f"(source: {z_source})")
        _update_z_in_yaml(cfg_name, z_um)

    # ── Step 7: write VOXEL_ISOTROPY.md ──────────────────────────────────────
    out_md = Path(__file__).parent / "VOXEL_ISOTROPY.md"
    _write_report(
        out_md, cfg_name, source, raw_vol.shape, stats, phases,
        vx * 1000, vy * 1000, z_struct_nm, mean_zx, std_zx,
        mean_zy, std_zy, verdict, z_final_nm, z_source
    )
    print(f"\n  Report written → {out_md}")


def _update_z_in_yaml(cfg_name: str, z_um: float) -> None:
    """Patch the z: null line in the config YAML with the measured value."""
    yaml_path = _REPO_ROOT / "configs" / f"{cfg_name}.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    old = "  z: null       # UNKNOWN -- confirmed OPEN QUESTION (NEW_DATASET_PPTX_NOTES.md Q1)"
    new = (f"  z: {z_um:.6f}    "
           f"# measured from struct.txt (check_voxel_isotropy.py 2026-07)")
    if old in text:
        yaml_path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"  YAML updated: {yaml_path.name}")
    else:
        print(f"  WARNING: expected z:null line not found in {yaml_path.name} — "
              f"update manually to z: {z_um:.6f}")


def _write_report(
    out_path, cfg_name, source, raw_shape, stats, phases,
    x_nm, y_nm, z_struct_nm, mean_zx, std_zx,
    mean_zy, std_zy, verdict, z_final_nm, z_source
):
    lines = [
        "# Voxel Isotropy Report — Cathode S1 Supercrop",
        "",
        f"> Generated by `0_PRV/check_voxel_isotropy.py`  "
        f"config: `{cfg_name}`",
        "",
        "## Source file",
        "",
        f"- File: `{source.name}`",
        f"- Shape (Z, Y, X): {raw_shape}",
        f"- Phases: {list(phases.keys())}",
        "",
        "## Direct evidence — struct.txt z spacing",
        "",
    ]
    if z_struct_nm is not None:
        lines += [
            f"Per-phase coordinate files (`*_struct.txt`) found alongside source TIF.",
            f"Mean z step read from z column (column 2): **{z_struct_nm:.4f} nm**",
            "",
            f"Compare to known x/y spacings:",
            f"- x = {x_nm:.4f} nm (from struct.txt x-step, reported in cathode YAML)",
            f"- y = {y_nm:.4f} nm (from struct.txt y-step, reported in cathode YAML)",
            f"- z = **{z_struct_nm:.4f} nm** (from struct.txt z-step, this script)",
            "",
            f"z/x = {z_struct_nm/x_nm:.4f}    z/y = {z_struct_nm/y_nm:.4f}",
            "(ratio < 1.01 and > 0.99 → isotropic within measurement resolution)",
            "",
        ]
    else:
        lines += ["No struct.txt files found — cannot report direct z spacing.", ""]

    lines += [
        "## Autocorrelation and chord-length statistics",
        "",
        "S2 decay length: lag (voxels) where S2(lag) − vf² drops to 1/e of "
        "S2(0) − vf².",
        "Mean chord length: mean run length (voxels) along each axis.",
        "",
        "| Phase | vf | Z decay | Y decay | X decay | Z chord | Y chord | X chord |",
        "|-------|----|---------|---------|---------|---------|---------|---------|",
    ]
    for phase_name, pstats in stats.items():
        vf = pstats["vf"]
        dZ = pstats["Z"]["decay"]
        dY = pstats["Y"]["decay"]
        dX = pstats["X"]["decay"]
        cZ = pstats["Z"]["chord"]
        cY = pstats["Y"]["chord"]
        cX = pstats["X"]["chord"]

        def fmt(v):
            return f"{v:.1f}" if not np.isnan(v) else "nan"

        lines.append(
            f"| {phase_name} | {vf:.3f} | {fmt(dZ)} | {fmt(dY)} | {fmt(dX)} "
            f"| {fmt(cZ)} | {fmt(cY)} | {fmt(cX)} |"
        )

    lines += [
        "",
        "## Ratio analysis",
        "",
        f"x/y baseline: x={x_nm:.3f} nm / y={y_nm:.3f} nm = "
        f"{x_nm/y_nm:.4f} (expected ≈1.00 — noise-floor reference)",
        "",
        f"Mean z/x ratio (all phases, decay + chord): "
        f"**{mean_zx:.4f} ± {std_zx:.4f}**",
        f"Mean z/y ratio (all phases, decay + chord): "
        f"**{mean_zy:.4f} ± {std_zy:.4f}**",
        "",
        "Interpretation: if z spacing differed from x by factor k, ALL phases' "
        "z/x ratios (in voxels) would shift by the same k. Consistent shift → "
        "calibration error. Inconsistent → genuine anisotropy.",
        "",
        "## Verdict",
        "",
        f"> **{verdict}**",
        "",
    ]
    if z_final_nm is not None:
        z_um = z_final_nm / 1000
        lines += [
            f"z voxel size determined: **{z_final_nm:.4f} nm** ({z_um:.6f} µm)  "
            f"[source: {z_source}]",
            "",
            f"`configs/{cfg_name}.yaml` updated: `voxel_size_um.z: {z_um:.6f}`",
            "",
        ]
    else:
        lines += [
            "z voxel size could not be determined from available data.",
            "Leave `voxel_size_um.z: null` in the config until Prof. Jin confirms.",
            "",
        ]

    lines += [
        "## Open questions",
        "",
        "- Anode voxel size: 0.1 µm (100 nm) is undocumented — confirm with "
        "Prof. Jin.",
        "- Cathode z: resolved above (from struct.txt)." if z_final_nm else
        "- Cathode z: still unresolved — struct.txt not found at source location.",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify z voxel isotropy for a cathode dataset config."
    )
    parser.add_argument(
        "--config", default="cathode_s1_supercrop",
        help="Config name from configs/<name>.yaml (default: cathode_s1_supercrop)"
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
