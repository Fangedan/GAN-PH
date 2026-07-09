"""
0_PRV/extract_cubes.py
======================
Extract 64^3 (or --size N) training cubes from a segmented 3D TIF file.

Reads the source TIF from configs/local_paths.yaml, remaps phase labels to the
BMP convention (Ni=255, YSZ=127, Pore=0) defined in the dataset config, then
emits structure_XXXX/ folders of slice_XXXX.bmp files exactly matching
1_GAN/load.py's expectations. Also writes results.dat and manifest.csv.

Spatial train/val split
-----------------------
The parent volume is split into a TRAIN region and a VAL region along the chosen
axis (default: longest axis) BEFORE any cropping. A guard gap of (size - stride)
voxels separates the regions so that no train crop can overlap any val crop.
Random crop-level splits are FORBIDDEN -- overlapping crops are spatially
correlated and would leak microstructure between train and val.

Usage
-----
  # From repo root (conda run required on Windows):
  conda run -n ganph --no-capture-output python 0_PRV/extract_cubes.py \\
      --config cathode_s1_supercrop \\
      --out cathode_crops_s1_str64 \\
      --size 64 --stride 64

  conda run -n ganph --no-capture-output python 0_PRV/extract_cubes.py \\
      --config cathode_s1_supercrop \\
      --out cathode_crops_s1_str32 \\
      --size 64 --stride 32

Arguments
---------
  --config      Dataset config name (default: cathode_s1_supercrop)
  --size        Cube edge length in voxels (default: 64)
  --stride      Stride for sliding window (default: size = non-overlapping)
  --out         Output directory (will be created; gitignore cathode_crops_*)
  --max-excluded-frac  Reject crop if excluded/4th-phase fraction > this (default: 0.0)
  --val-fraction       Fraction of volume reserved for validation (default: 0.2)
  --split-axis         Axis for train/val split: x | y | z | longest (default: longest)
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

# ── Repo root on sys.path (for configs and preprocess imports) ────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from configs.dataset_config import get_config, DatasetConfig

# Import SSA + VF functions from preprocess_dream3d.py (do not reimplement)
import preprocess_dream3d as _pp
from preprocess_dream3d import (
    compute_volume_fractions,
    compute_surface_area,
    write_results_dat,
)

try:
    import tifffile
except ImportError:
    print("ERROR: tifffile not installed. Run: pip install tifffile")
    sys.exit(1)


# ── Label remapping ───────────────────────────────────────────────────────────

def remap_labels(vol: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """
    Apply source_label_map from config: source pixel value → target BMP value.
    Voxels with source values not in the map are set to 255 (treated as ch0 phase)
    unless they are in exclude_values, where they remain as-is for exclusion check.

    Returns a new array in BMP convention.
    """
    source_map = cfg.source_label_map  # {source_val: phase_name}
    phases_bmp = cfg.phases_dict()     # {phase_name: bmp_value}
    exclude    = set(cfg.exclude_values)

    out = np.zeros_like(vol)
    mapped = np.zeros(vol.shape, dtype=bool)

    for src_val, phase_name in source_map.items():
        src_val = int(src_val)
        if phase_name not in phases_bmp:
            # Phase is named in label_map but not in the 3-phase BMP targets
            # (e.g. SCT in S2). Pass through with the original value so it lands
            # in exclude_values and gets caught by the exclusion check.
            mask = (vol == src_val)
            out[mask] = src_val   # keep original; must be in exclude_values
            mapped[mask] = True
            continue
        bmp_val = phases_bmp[phase_name]
        mask = (vol == src_val)
        out[mask] = bmp_val
        mapped[mask] = True

    # Excluded values: copy through as-is (exclusion check uses original values)
    for exc_val in exclude:
        mask = (vol == exc_val)
        out[mask] = exc_val
        mapped[mask] = True

    unmapped = ~mapped
    if unmapped.any():
        n = int(unmapped.sum())
        unique_unmapped = np.unique(vol[unmapped])
        warnings.warn(
            f"remap_labels: {n} voxels with values {unique_unmapped.tolist()} "
            f"are not in source_label_map or exclude_values. Treated as excluded."
        )
        for uv in unique_unmapped:
            out[vol == uv] = int(uv)  # pass through for exclusion check

    return out


# ── Spatial train/val split ───────────────────────────────────────────────────

def compute_split_boundaries(
    shape: tuple,
    split_axis: str,
    val_fraction: float,
    size: int,
    stride: int,
) -> tuple[int, int, int, int]:
    """
    Partition volume into [0, train_end) and [val_start, shape[axis]) along split_axis.
    Guard gap of (size - stride) voxels ensures no train crop overlaps any val crop.

    Returns
    -------
    axis_idx : int (0=z, 1=y, 2=x)
    train_end : int
    val_start : int
    axis_len : int
    """
    axis_map = {"z": 0, "y": 1, "x": 2}
    if split_axis == "longest":
        axis_idx = int(np.argmax(shape))
    else:
        axis_idx = axis_map[split_axis]

    axis_len = shape[axis_idx]
    guard = size - stride  # voxels between train and val (min 0 for stride=size)

    # Val region: last val_fraction of the axis, aligned to stride grid
    val_voxels_raw = int(np.round(axis_len * val_fraction))
    val_voxels = max(size, val_voxels_raw)  # at least one cube

    val_start = axis_len - val_voxels
    train_end = val_start - guard

    if train_end < size:
        raise ValueError(
            f"Volume too small for train/val split on axis {axis_idx} "
            f"(len={axis_len}, size={size}, stride={stride}, val_frac={val_fraction}). "
            f"Try a larger volume or smaller --val-fraction."
        )

    print(f"  Split axis: {'ZYX'[axis_idx]} (dim {axis_idx}, len={axis_len})")
    print(f"  Train region: [{0}, {train_end})")
    print(f"  Guard gap:    [{train_end}, {val_start})")
    print(f"  Val region:   [{val_start}, {axis_len})")

    return axis_idx, train_end, val_start, axis_len


def iter_crops(
    vol: np.ndarray,
    size: int,
    stride: int,
    axis_idx: int,
    region_start: int,
    region_end: int,
) -> list[tuple]:
    """
    Yield (origin_z, origin_y, origin_x, crop_array) for all valid crop origins
    within vol[region_start:region_end, ...] along split axis.

    Crops outside the region along split_axis are excluded.
    """
    Z, Y, X = vol.shape
    crops = []

    z_starts = range(0, Z - size + 1, stride)
    y_starts = range(0, Y - size + 1, stride)
    x_starts = range(0, X - size + 1, stride)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                origin = (z0, y0, x0)
                # Check if this crop falls in the region along the split axis
                crop_start = origin[axis_idx]
                crop_end   = crop_start + size

                # Crop must start and end within [region_start, region_end)
                if crop_start < region_start or crop_end > region_end:
                    continue

                cube = vol[z0:z0+size, y0:y0+size, x0:x0+size]
                crops.append((z0, y0, x0, cube))

    return crops


# ── Exclusion check ───────────────────────────────────────────────────────────

def excluded_fraction(cube: np.ndarray, exclude_values: list[int]) -> float:
    """Return fraction of voxels with values in exclude_values."""
    if not exclude_values:
        return 0.0
    mask = np.isin(cube, exclude_values)
    return float(mask.mean())


# ── Save structure (matches load.py exactly) ──────────────────────────────────

def save_structure(cube: np.ndarray, idx: int, out_dir: Path) -> None:
    """Save (size, size, size) volume as slice_XXXX.bmp files in structure_XXXX/."""
    folder = out_dir / f"structure_{idx:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    for z in range(cube.shape[0]):
        cv2.imwrite(str(folder / f"slice_{z:04d}.bmp"), cube[z].astype(np.uint8))


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(
    cfg: DatasetConfig,
    size: int,
    stride: int,
    out_dir: Path,
    max_excluded_frac: float,
    val_fraction: float,
    split_axis: str,
) -> dict:
    """
    Full extraction pipeline. Returns a summary dict for reporting.
    """
    # ── Resolve source file ───────────────────────────────────────────────
    src = cfg.source_file
    if src is None:
        raise FileNotFoundError(
            f"No source_file_key in config {cfg.name!r}, or configs/local_paths.yaml "
            f"is missing. Create it from configs/local_paths.yaml.example."
        )
    if not src.exists():
        raise FileNotFoundError(f"Source TIF not found: {src}")

    print(f"\n{'='*64}")
    print(f"  extract_cubes.py  [{cfg.name}]")
    print(f"{'='*64}")
    print(f"  Source   : {src.name}")
    print(f"  Config   : {cfg.name}")
    print(f"  Size     : {size}^3 voxels")
    print(f"  Stride   : {stride}")
    print(f"  Max excl : {max_excluded_frac:.2f}")
    print(f"  Val frac : {val_fraction:.2f}")

    # ── Warn if z voxel size unknown ─────────────────────────────────────
    if cfg.voxel_size_z_um is None:
        warnings.warn(
            f"Config {cfg.name!r}: z voxel size is null. Using x voxel size "
            f"({cfg.voxel_size_x_um} um) as approximation for SSA/VF calculations. "
            f"Confirm with Prof. Jin."
        )

    # Patch preprocess_dream3d module with this config's voxel size
    vox_um = cfg.voxel_size_x_um or cfg.voxel_size_um or 0.1
    _pp.VOXEL_SIZE_UM = float(vox_um)

    # ── Load TIF (tifffile returns (Z, Y, X) for multi-page TIF) ─────────
    print(f"\n  Loading TIF ...", end="", flush=True)
    with tifffile.TiffFile(str(src)) as tif:
        vol_raw = tif.asarray()
    print(f" done. shape={vol_raw.shape}, dtype={vol_raw.dtype}")

    # Verify axis order matches inspection report
    expected_shapes = {
        "cathode_s1_supercrop": (151, 283, 120),
        "cathode_s2": (274, 215, 166),
    }
    if cfg.name in expected_shapes:
        exp = expected_shapes[cfg.name]
        if vol_raw.shape != exp:
            raise ValueError(
                f"TIF shape {vol_raw.shape} != expected {exp} for {cfg.name}. "
                f"Check TIF axis order or config. Stop before proceeding."
            )

    # ── Remap labels ──────────────────────────────────────────────────────
    print(f"  Remapping labels ...", end="", flush=True)
    vol_bmp = remap_labels(vol_raw, cfg)
    print(f" done.")

    # Verify remap: check global phase fractions before splitting
    phase_dict = cfg.phases_dict()
    total = vol_bmp.size
    print(f"  Global phase fractions after remap:")
    for ph_name, bmp_val in phase_dict.items():
        frac = np.mean(vol_bmp == bmp_val) * 100
        print(f"    {ph_name} (bmp={bmp_val}): {frac:.1f}%")
    global_vf0 = float(np.mean(vol_bmp == phase_dict[list(phase_dict.keys())[0]]))
    global_vf1 = float(np.mean(vol_bmp == phase_dict[list(phase_dict.keys())[1]]))
    global_vf2 = float(np.mean(vol_bmp == phase_dict[list(phase_dict.keys())[2]]))

    # ── Compute split boundaries ──────────────────────────────────────────
    print(f"\n  Spatial train/val split:")
    axis_idx, train_end, val_start, axis_len = compute_split_boundaries(
        vol_bmp.shape, split_axis, val_fraction, size, stride
    )

    # ── Collect crops for both regions ────────────────────────────────────
    print(f"\n  Sliding window ({size}^3, stride={stride}) ...")
    train_crops = iter_crops(vol_bmp, size, stride, axis_idx, 0, train_end)
    val_crops   = iter_crops(vol_bmp, size, stride, axis_idx, val_start, axis_len)
    print(f"  Raw crops: train={len(train_crops)}, val={len(val_crops)}")

    # ── Filter by excluded fraction ────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    excl_vals = cfg.exclude_values

    results_dat_rows = []
    manifest_rows    = []
    struct_idx       = 1
    rejection_counts = {"excluded": 0}
    mean_excl_fracs  = []

    # Process train then val in order, numbering sequentially
    for region_label, crop_list in [("train", train_crops), ("val", val_crops)]:
        accepted = 0
        for z0, y0, x0, cube in crop_list:
            ef = excluded_fraction(cube, excl_vals)
            if ef > max_excluded_frac:
                rejection_counts["excluded"] += 1
                continue

            # Save structure
            save_structure(cube, struct_idx, out_dir)

            # Compute VF and SSA via preprocess_dream3d (uses patched VOXEL_SIZE_UM)
            vf0 = 100.0 * float(np.mean(cube == _pp.NI_VAL))
            vf1 = 100.0 * float(np.mean(cube == _pp.YSZ_VAL))
            vf2 = 100.0 * float(np.mean(cube == _pp.PORE_VAL))
            sv0 = compute_surface_area(cube, _pp.NI_VAL)
            sv1 = compute_surface_area(cube, _pp.YSZ_VAL)
            sv2 = compute_surface_area(cube, _pp.PORE_VAL)
            results_dat_rows.append((vf0, vf1, vf2, sv0, sv1, sv2))

            manifest_rows.append({
                "struct_id":    f"structure_{struct_idx:04d}",
                "parent_file":  src.name,
                "origin_z":     z0,
                "origin_y":     y0,
                "origin_x":     x0,
                "region":       region_label,
                "excl_frac":    f"{ef:.4f}",
            })
            mean_excl_fracs.append(ef)
            struct_idx += 1
            accepted += 1

        print(f"  {region_label.upper()}: {accepted} accepted / {len(crop_list)} total")

    n_accepted = len(results_dat_rows)

    # ── Write results.dat ──────────────────────────────────────────────────
    write_results_dat(out_dir, results_dat_rows)

    # ── Write manifest.csv ────────────────────────────────────────────────
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        fieldnames = ["struct_id", "parent_file", "origin_z", "origin_y",
                      "origin_x", "region", "excl_frac"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  Manifest: {manifest_path}")

    # ── VF drift check ────────────────────────────────────────────────────
    if n_accepted > 0:
        ph_names = list(phase_dict.keys())
        ph_vals  = list(phase_dict.values())
        global_vfs = [global_vf0 * 100, global_vf1 * 100, global_vf2 * 100]

        accepted_vfs = [[] for _ in range(3)]
        for row in results_dat_rows:
            for i in range(3):
                accepted_vfs[i].append(row[i])

        print(f"\n  VF drift check (accepted crops mean vs. parent global):")
        any_drift = False
        for i, (ph, gvf) in enumerate(zip(ph_names, global_vfs)):
            mean_crop_vf = float(np.mean(accepted_vfs[i]))
            drift = abs(mean_crop_vf - gvf)
            flag = "⚠️ DRIFT > 2pp" if drift > 2.0 else "OK"
            print(f"    {ph}: parent={gvf:.1f}%  crops={mean_crop_vf:.1f}%  "
                  f"drift={drift:.1f}pp  [{flag}]")
            if drift > 2.0:
                any_drift = True

        if any_drift:
            print("  ⚠️ VF drift >2pp detected -- rejection rule may be biasing composition.")
            print("     Consider increasing --max-excluded-frac or using a different region.")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Accepted structures: {n_accepted}")
    print(f"  Rejected (excluded frac > {max_excluded_frac:.2f}): "
          f"{rejection_counts['excluded']}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"{'='*64}")

    return {
        "accepted":     n_accepted,
        "rejected":     rejection_counts["excluded"],
        "train_crops":  len(train_crops),
        "val_crops":    len(val_crops),
        "out_dir":      str(out_dir),
        "axis_idx":     axis_idx,
        "train_end":    train_end,
        "val_start":    val_start,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract 64^3 training cubes from a segmented 3D TIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # S1 Supercrop, non-overlapping (stride=64):
  python 0_PRV/extract_cubes.py --config cathode_s1_supercrop --out cathode_crops_s1_str64

  # S1 Supercrop, overlapping (stride=32):
  python 0_PRV/extract_cubes.py --config cathode_s1_supercrop --out cathode_crops_s1_str32 --stride 32

  # S2 SCT Rectangle, stride=64, allow up to 5% excluded voxels:
  python 0_PRV/extract_cubes.py --config cathode_s2 --out cathode_crops_s2_str64 --max-excluded-frac 0.05
"""
    )
    parser.add_argument("--config", default="cathode_s1_supercrop",
                        help="Dataset config name (default: cathode_s1_supercrop)")
    parser.add_argument("--size",   type=int, default=64,
                        help="Cube edge (voxels, default: 64)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding window stride (default: same as --size)")
    parser.add_argument("--out",    required=True,
                        help="Output directory (gitignored cathode_crops_* recommended)")
    parser.add_argument("--max-excluded-frac", type=float, default=0.0,
                        help="Reject crop if excluded voxel fraction > this (default: 0.0)")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of volume for validation region (default: 0.2)")
    parser.add_argument("--split-axis", choices=["x", "y", "z", "longest"],
                        default="longest",
                        help="Axis for spatial train/val split (default: longest)")
    args = parser.parse_args()

    stride = args.stride if args.stride is not None else args.size
    cfg    = get_config(args.config)

    extract(
        cfg=cfg,
        size=args.size,
        stride=stride,
        out_dir=Path(args.out),
        max_excluded_frac=args.max_excluded_frac,
        val_fraction=args.val_fraction,
        split_axis=args.split_axis,
    )


if __name__ == "__main__":
    main()
