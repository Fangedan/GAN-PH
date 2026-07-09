"""
preprocess_dream3d.py  (enhanced)
==================================
Converts DREAM.3D-exported PNG slice stacks into the exact format
expected by the GAN-PH training pipeline (load.py).

What this script does, step by step:
  1. Reads all PNG slices from the input folder(s)
  2. Thresholds each image to clean phase values (removes anti-aliasing blur)
  3. Remaps phase pixel values:
        DREAM.3D exports:   255=Pore (white),  0=Ni (black),  127=YSZ (grey)
        load.py expects:    255=Ni  (white),   0=Pore (black), 127=YSZ (grey)
  4. Crops the black border DREAM.3D adds around each slice
  5. Depending on volume size, EITHER:
       (a) Resize each slice to 64x64 then tile in Z  [small stacks]
       (b) Keep full resolution and tile in all 3 axes [large stacks, e.g. 500^3]
  6. Saves each 64x64x64 sub-volume as 64 BMP slices
  7. Measures volume fractions and specific surface areas
  8. Writes all measurements to results.dat

New in this version:
  --multi       Accept a parent folder containing multiple slice sub-folders;
                each sub-folder is processed in turn and structures are numbered
                sequentially across all of them.
  --tile-xy     Force spatial tiling in XY (don't resize slices to 64x64).
                Recommended for inputs with slices larger than ~256x256 px.
                For a 500x500 input this yields up to 7x7 = 49 patches per
                z-slab instead of downsampling everything to one 64x64 view.
  --border      Border pixels to crop on each side (default: 24).
  --dry-run     Print what would happen without writing any files.
  --preview     Save one representative slice from the first sub-volume as
                preview.png so you can visually verify phase mapping.

Usage - single folder:
  python preprocess_dream3d.py --input path/to/png/folder --output ./real_data

Usage - multi-folder (Juan sends several stacks):
  python preprocess_dream3d.py --input path/to/parent_folder --output ./real_data --multi

Usage - large stack (500^3), spatially tile in XY:
  python preprocess_dream3d.py --input path/to/folder --output ./real_data --tile-xy

Test with synthetic BMP data (no real DREAM.3D needed):
  python preprocess_dream3d.py --input ./synthetic_data --output ./synth_check --synthetic-test

Requirements:
  pip install opencv-python numpy tqdm
"""

import os
import argparse
import sys
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.dataset_config import default as _cfg_default

# -- Configuration --------------------------------------------------------------
VOXEL_SIZE    = 64      # output structure size (64x64x64)

_pp_cfg       = _cfg_default()
_pp_ch        = _pp_cfg.channel_names()
VOXEL_SIZE_UM = _pp_cfg.voxel_size_um or 0.1    # 0.1 µm (100 nm, anode FIB-SEM)

# Phase pixel values in the OUTPUT (what load.py expects)
NI_VAL   = _pp_cfg.phase_value(_pp_ch[0])   # 255
YSZ_VAL  = _pp_cfg.phase_value(_pp_ch[1])   # 127
PORE_VAL = _pp_cfg.phase_value(_pp_ch[2])   # 0

# Phase pixel values in DREAM.3D's PNG output (what Juan sends)
DREAM3D_NI_VAL   = 0    # Ni is black in DREAM.3D output
DREAM3D_YSZ_VAL  = 127  # YSZ is grey (same in both)
DREAM3D_PORE_VAL = 255  # Pore is white in DREAM.3D output

# Naming patterns to try when scanning for input PNGs.
# We try each glob in order and use the first one that finds files.
SLICE_GLOB_PATTERNS = [
    "Slice*.png",       # DREAM.3D default:  Slice0000.png
    "slice_*.png",      # generate_training_data output: slice_0000.bmp (not used here but kept)
    "slice*.png",       # lowercase variant
    "*.png",            # fallback: any PNG, sorted alphabetically
]


# -- Input discovery ------------------------------------------------------------

def find_png_files(folder: Path) -> list:
    """Return a sorted list of PNG paths in folder, trying naming patterns in order."""
    for pattern in SLICE_GLOB_PATTERNS:
        files = sorted(folder.glob(pattern))
        if files:
            return files
    return []


def discover_input_folders(root: Path, multi: bool) -> list:
    """
    Return a list of folders to process.

    Single mode:  [root]
    Multi mode:   all immediate sub-directories of root that contain PNGs.
                  This covers the case where Juan sends multiple stacks as
                  separate folders inside one parent directory.
    """
    if not multi:
        return [root]

    sub_folders = sorted([p for p in root.iterdir() if p.is_dir()])
    valid = [p for p in sub_folders if find_png_files(p)]

    if not valid:
        raise FileNotFoundError(
            f"--multi was set but no sub-folder of {root} contains PNG files.\n"
            f"Expected structure:\n"
            f"  {root}/\n"
            f"    stack_001/   <- Slice0000.png Slice0001.png ...\n"
            f"    stack_002/\n"
            f"    ..."
        )

    print(f"[multi] Found {len(valid)} input folder(s) under {root}:")
    for p in valid:
        files = find_png_files(p)
        print(f"  {p.name}  ({len(files)} PNGs)")

    return valid


# -- Step 1: Load all PNG slices ------------------------------------------------

def load_slices(input_folder: Path) -> list:
    """Load PNG slice files from a folder, sorted by name."""
    png_files = find_png_files(input_folder)

    if not png_files:
        raise FileNotFoundError(
            f"No PNG files found in {input_folder}\n"
            f"Tried patterns: {SLICE_GLOB_PATTERNS}\n"
            f"Files present: {list(input_folder.iterdir())[:10]}"
        )

    slices = []
    for f in tqdm(png_files, desc=f"  Loading {input_folder.name}", leave=False):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read image: {f}")
        slices.append(img)

    return slices


# -- Step 2: Threshold to clean phase values ------------------------------------

def threshold_to_phases(img: np.ndarray) -> np.ndarray:
    """
    Snap every pixel to the nearest of {0, 127, 255}.
    Removes DREAM.3D anti-aliasing blur between phases.
    """
    img_int = img.astype(np.int32)
    d0   = np.abs(img_int - 0)
    d127 = np.abs(img_int - 127)
    d255 = np.abs(img_int - 255)

    out = np.zeros_like(img)
    out[d127 <= d0]                   = 127
    out[(d255 <= d0) & (d255 < d127)] = 255
    return out


# -- Step 3: Remap phase values -------------------------------------------------

def remap_phases(img: np.ndarray) -> np.ndarray:
    """
    DREAM.3D:  0=Ni,   127=YSZ, 255=Pore
    load.py:   255=Ni, 127=YSZ, 0=Pore
    """
    out = np.zeros_like(img)
    out[img == DREAM3D_YSZ_VAL]  = YSZ_VAL    # 127 -> 127
    out[img == DREAM3D_NI_VAL]   = NI_VAL     # 0   -> 255
    out[img == DREAM3D_PORE_VAL] = PORE_VAL   # 255 -> 0
    return out


# -- Step 4: Crop black border --------------------------------------------------

def crop_border(img: np.ndarray, border: int) -> np.ndarray:
    """
    Remove `border` pixels from each edge.
    Default 24 matches Juan's DREAM.3D export (content at rows/cols 24-475).
    Pass 0 to skip (useful for synthetic test data which has no border).
    """
    if border <= 0:
        return img
    return img[border:-border, border:-border]


# -- Step 5a: Resize mode (small stacks) ---------------------------------------

def resize_slice(img: np.ndarray, size: int = VOXEL_SIZE) -> np.ndarray:
    """
    Downsample to size x size by sampling the CENTER pixel of each block.
    Center sampling (not cv2.resize) because cv2's INTER_NEAREST samples at
    block boundaries -- which lands exactly on rendering artifacts when the
    input is a screenshot-style export with cell edges aligned to pixels.
    """
    rows = ((np.arange(size) + 0.5) * img.shape[0] / size).astype(int)
    cols = ((np.arange(size) + 0.5) * img.shape[1] / size).astype(int)
    return img[np.ix_(rows, cols)]


# -- Step 5b: Tile mode (large stacks, --tile-xy) ------------------------------

def tile_slice(img: np.ndarray, tile_size: int = VOXEL_SIZE) -> list:
    """
    Split a large slice into non-overlapping tile_sizextile_size patches.
    Returns a list of (row_idx, col_idx, patch) tuples.
    Partial tiles at the edges are discarded (same behaviour as extract_subvolumes).
    """
    h, w = img.shape
    tiles = []
    for r in range(h // tile_size):
        for c in range(w // tile_size):
            patch = img[r*tile_size:(r+1)*tile_size,
                        c*tile_size:(c+1)*tile_size]
            tiles.append((r, c, patch))
    return tiles


# -- Step 6: Extract sub-volumes ------------------------------------------------

def extract_subvolumes_resize(volume_3d: np.ndarray,
                              voxel_size: int = VOXEL_SIZE) -> list:
    """
    Resize mode: volume has shape (n_z, 64, 64).
    Pad in Z if needed, then slice into non-overlapping 64-slice chunks.
    """
    n_z = volume_3d.shape[0]

    # Z padding by mirroring
    if n_z < voxel_size:
        print(f"    Z ({n_z}) < {voxel_size} -- padding by mirroring")
        needed = voxel_size - n_z
        mirror = volume_3d[::-1][:needed]
        volume_3d = np.concatenate([volume_3d, mirror], axis=0)
        n_z = volume_3d.shape[0]

    return [volume_3d[i*voxel_size:(i+1)*voxel_size]
            for i in range(n_z // voxel_size)]


def extract_subvolumes_tile(volume_3d: np.ndarray,
                            voxel_size: int = VOXEL_SIZE) -> list:
    """
    Tile mode: volume has shape (n_z, H, W) where H, W >> 64.
    Extract non-overlapping voxel_size^3 cubes in all 3 dimensions.
    For a 500x500x500 input (after crop ~ 452x452x500):
        XY tiles: floor(452/64) x floor(452/64) = 7 x 7 = 49 per z-slab
        Z  slabs: floor(500/64) = 7
        Total:    7 x 49 = 343 structures
    The user note of ~49 structures refers to one z-slab of a 500^3 volume.
    """
    n_z, n_y, n_x = volume_3d.shape
    subvolumes = []

    n_z_blocks = n_z // voxel_size
    n_y_blocks = n_y // voxel_size
    n_x_blocks = n_x // voxel_size

    if n_z_blocks == 0:
        # Less than 64 slices -- pad then treat as one z-slab
        needed = voxel_size - n_z
        mirror = volume_3d[::-1][:needed]
        volume_3d = np.concatenate([volume_3d, mirror], axis=0)
        n_z_blocks = 1

    for iz in range(n_z_blocks):
        for iy in range(n_y_blocks):
            for ix in range(n_x_blocks):
                cube = volume_3d[
                    iz*voxel_size:(iz+1)*voxel_size,
                    iy*voxel_size:(iy+1)*voxel_size,
                    ix*voxel_size:(ix+1)*voxel_size
                ]
                subvolumes.append(cube)

    return subvolumes


# -- Step 7: Save sub-volume as BMP slices --------------------------------------

def save_structure(volume: np.ndarray, structure_idx: int,
                   output_dir: Path, dry_run: bool = False) -> Path:
    """Save a 64x64x64 volume as 64 grayscale BMP files."""
    folder = output_dir / f"structure_{structure_idx:04d}"
    if not dry_run:
        folder.mkdir(parents=True, exist_ok=True)
        for z in range(volume.shape[0]):
            cv2.imwrite(str(folder / f"slice_{z:04d}.bmp"),
                        volume[z].astype(np.uint8))
    return folder


# -- Step 8: Measurements -------------------------------------------------------

def compute_volume_fractions(volume: np.ndarray) -> tuple:
    total = volume.size
    return (
        100.0 * np.sum(volume == NI_VAL)   / total,
        100.0 * np.sum(volume == YSZ_VAL)  / total,
        100.0 * np.sum(volume == PORE_VAL) / total,
    )


def compute_surface_area(volume: np.ndarray, phase_value: int) -> float:
    mask = (volume == phase_value).astype(np.int8)
    surface_faces = sum(int(np.sum(np.abs(np.diff(mask, axis=ax))))
                        for ax in range(3))
    phase_vol_um3 = float(np.sum(mask)) * (VOXEL_SIZE_UM ** 3)
    if phase_vol_um3 == 0:
        return 0.0
    return float(surface_faces) * (VOXEL_SIZE_UM ** 2) / phase_vol_um3


# -- Step 9: Write results.dat --------------------------------------------------

def write_results_dat(output_dir: Path, results: list,
                      dry_run: bool = False) -> Path:
    dat_path = output_dir / "results.dat"
    if not dry_run:
        with open(dat_path, 'w') as f:
            f.write("VF0 VF1 VF2 SV0 SV1 SV2\n")
            for row in results:
                f.write(" ".join(f"{v:.4f}" for v in row) + "\n")
        print(f"  Wrote {len(results)} rows -> {dat_path}")
    else:
        print(f"  [dry-run] Would write {len(results)} rows -> {dat_path}")
    return dat_path


# -- Synthetic test: generate fake DREAM.3D PNGs from BMP stacks ---------------

def run_synthetic_test(synth_dir: Path, output_dir: Path, dry_run: bool):
    """
    Convert a generate_training_data.py BMP stack into fake DREAM.3D PNGs,
    then run the full preprocessing pipeline on them.
    This lets you verify the pipeline without Juan's real data.

    The BMP stacks use load.py conventions (255=Ni, 127=YSZ, 0=Pore).
    We REVERSE the phase mapping to simulate DREAM.3D export, then preprocess
    as normal to confirm we get back the original values.
    """
    print("\n[synthetic-test] Generating fake DREAM.3D PNGs from BMP stacks...")
    print(f"  Source: {synth_dir}")

    struct_dirs = sorted([p for p in synth_dir.iterdir()
                          if p.is_dir() and p.name.startswith("structure_")])
    if not struct_dirs:
        raise FileNotFoundError(
            f"No structure_XXXX folders found in {synth_dir}.\n"
            f"Run generate_training_data.py first to create synthetic data."
        )

    # Use just the first structure for a quick sanity check
    src = struct_dirs[0]
    tmp_png_dir = output_dir / "_synth_test_pngs"
    tmp_png_dir.mkdir(parents=True, exist_ok=True)

    bmp_files = sorted(src.glob("slice_*.bmp"))
    print(f"  Converting {len(bmp_files)} BMPs from {src.name} -> fake PNGs")

    for bmp_path in bmp_files:
        img = cv2.imread(str(bmp_path), cv2.IMREAD_GRAYSCALE)
        # Reverse the phase mapping: 255->0 (Ni->black), 0->255 (Pore->white), 127->127
        fake = np.zeros_like(img)
        fake[img == 255] = 0
        fake[img == 0]   = 255
        fake[img == 127] = 127
        # Save as Slice####.png (DREAM.3D naming)
        slice_num = int(bmp_path.stem.split("_")[1])
        png_name = f"Slice{slice_num:04d}.png"
        if not dry_run:
            cv2.imwrite(str(tmp_png_dir / png_name), fake)

    print(f"  Saved fake PNGs -> {tmp_png_dir}")
    print(f"\n[synthetic-test] Running full pipeline on fake PNGs (border=0)...")

    # Process the fake PNGs with border=0 (no DREAM.3D border in synthetic data)
    result_dir = output_dir / "_synth_test_output"
    process_one_folder(
        input_folder=tmp_png_dir,
        output_dir=result_dir,
        border=0,           # synthetic data has no border
        tile_xy=False,
        dry_run=dry_run,
        preview=True,
        structure_offset=0,
        results_list=[],    # standalone results
        write_dat=True,
    )

    print("\n[synthetic-test] Done. Check _synth_test_output/ for results.")
    print("  Phase mapping is correct if preview.png shows a structure")
    print("  similar to the original BMP slices.")


# -- Core per-folder processing -------------------------------------------------

def process_one_folder(input_folder: Path,
                       output_dir: Path,
                       border: int,
                       tile_xy: bool,
                       dry_run: bool,
                       preview: bool,
                       structure_offset: int,
                       results_list: list,
                       write_dat: bool = False) -> int:
    """
    Process a single input folder.
    Returns the number of structures extracted.
    Appends measurements to results_list.
    structure_offset: the starting index for structure numbering (for --multi).
    """
    print(f"\n-- {input_folder.name} ----------------------")
    raw_slices = load_slices(input_folder)
    h0, w0 = raw_slices[0].shape
    print(f"  Loaded {len(raw_slices)} slices  ({h0}x{w0} px each)")

    # Steps 2-4: threshold, remap, crop
    processed = []
    for img in tqdm(raw_slices, desc="  Threshold/remap/crop", leave=False):
        img = threshold_to_phases(img)
        img = remap_phases(img)
        img = crop_border(img, border)
        processed.append(img)

    h1, w1 = processed[0].shape
    n_slices = len(processed)
    print(f"  After crop: {h1}x{w1} px, {n_slices} slices")

    # Step 5: decide resize vs tile
    if tile_xy:
        # Keep full XY resolution; tile spatially in all 3 axes
        volume_3d = np.stack(processed, axis=0)
        print(f"  Tile mode: full volume {volume_3d.shape} -> tiling in all 3 axes")
        subvolumes = extract_subvolumes_tile(volume_3d)
        expected_xy = (h1 // VOXEL_SIZE) * (w1 // VOXEL_SIZE)
        expected_z  = max(1, n_slices // VOXEL_SIZE)
        print(f"  Expected: {expected_z} z-slabs x {expected_xy} XY tiles = "
              f"{expected_z * expected_xy} structures")
    else:
        # Resize each slice to 64x64 then tile in Z only
        resized = [resize_slice(img) for img in processed]
        volume_3d = np.stack(resized, axis=0)
        print(f"  Resize mode: each slice -> 64x64, volume {volume_3d.shape}")
        subvolumes = extract_subvolumes_resize(volume_3d)

    print(f"  Extracted {len(subvolumes)} sub-volume(s)")

    if len(subvolumes) == 0:
        print("  WARNING: 0 sub-volumes extracted -- input may be too small.")
        return 0

    # Phase check on middle sub-volume
    mid = subvolumes[len(subvolumes) // 2]
    total = mid.size
    ni_pct   = 100 * np.sum(mid == NI_VAL)   / total
    ysz_pct  = 100 * np.sum(mid == YSZ_VAL)  / total
    pore_pct = 100 * np.sum(mid == PORE_VAL) / total
    print(f"  Phase check (mid sub-vol): Ni={ni_pct:.1f}% YSZ={ysz_pct:.1f}% Pore={pore_pct:.1f}%")

    # Preview (save one representative slice as PNG for visual inspection)
    if preview and not dry_run:
        preview_path = output_dir / "preview.png"
        cv2.imwrite(str(preview_path), subvolumes[0][32])
        print(f"  Saved preview slice -> {preview_path}")

    # Steps 7-8: save + measure
    for i, vol in enumerate(tqdm(subvolumes, desc="  Saving structures", leave=False)):
        idx = structure_offset + i + 1
        save_structure(vol, idx, output_dir, dry_run)

        vf_ni, vf_ysz, vf_pore = compute_volume_fractions(vol)
        sv_ni   = compute_surface_area(vol, NI_VAL)
        sv_ysz  = compute_surface_area(vol, YSZ_VAL)
        sv_pore = compute_surface_area(vol, PORE_VAL)
        results_list.append((vf_ni, vf_ysz, vf_pore, sv_ni, sv_ysz, sv_pore))

    if write_dat:
        write_results_dat(output_dir, results_list, dry_run)

    return len(subvolumes)


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DREAM.3D PNG slices for the GAN-PH pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single folder (original behaviour):
  python preprocess_dream3d.py --input ./juan_data --output ./real_data

  # Multiple stacks in one parent folder:
  python preprocess_dream3d.py --input ./juan_parent --output ./real_data --multi

  # Large 500x500x500 stack -- tile spatially in XY:
  python preprocess_dream3d.py --input ./juan_data --output ./real_data --tile-xy

  # Dry run (no files written -- just see what would happen):
  python preprocess_dream3d.py --input ./juan_data --output ./real_data --dry-run

  # Test pipeline against your synthetic BMP data:
  python preprocess_dream3d.py --input ./synthetic_data --output ./test_out --synthetic-test
        """
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input folder (or parent folder when using --multi)")
    parser.add_argument("--output", "-o", default="./real_data",
                        help="Output folder (default: ./real_data)")
    parser.add_argument("--multi", action="store_true",
                        help="Process all sub-folders of --input as separate stacks")
    parser.add_argument("--tile-xy", action="store_true",
                        help="Tile spatially in XY instead of resizing to 64x64 "
                             "(recommended for slices > ~256px wide)")
    parser.add_argument("--border", type=int, default=24,
                        help="Pixels to crop from each edge (default: 24). "
                             "Use 0 for synthetic/test data with no border.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing any files")
    parser.add_argument("--preview", action="store_true",
                        help="Save preview.png (slice 32 of first sub-volume) "
                             "for visual verification of phase mapping")
    parser.add_argument("--synthetic-test", action="store_true",
                        help="Generate fake DREAM.3D PNGs from a BMP stack "
                             "produced by generate_training_data.py, then run "
                             "the full pipeline to verify correctness")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  DREAM.3D -> GAN-PH Preprocessing Pipeline  (enhanced)")
    print("=" * 62)
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_dir}")
    print(f"  Multi   : {args.multi}")
    print(f"  Tile-XY : {args.tile_xy}")
    print(f"  Border  : {args.border} px")
    print(f"  Dry-run : {args.dry_run}")
    print()

    # -- Synthetic-test mode ---------------------------------------------------
    if args.synthetic_test:
        run_synthetic_test(input_path, output_dir, args.dry_run)
        return

    # -- Discover input folders ------------------------------------------------
    input_folders = discover_input_folders(input_path, args.multi)

    # -- Process each folder ---------------------------------------------------
    all_results  = []
    struct_count = 0

    for folder in input_folders:
        n = process_one_folder(
            input_folder=folder,
            output_dir=output_dir,
            border=args.border,
            tile_xy=args.tile_xy,
            dry_run=args.dry_run,
            preview=args.preview,
            structure_offset=struct_count,
            results_list=all_results,
            write_dat=False,        # write once at the end
        )
        struct_count += n

    # -- Write combined results.dat --------------------------------------------
    print("\nWriting results.dat...")
    dat_path = write_results_dat(output_dir, all_results, args.dry_run)

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  Done!")
    print(f"  Structures generated : {struct_count}")
    print(f"  Output folder        : {output_dir}")
    print(f"  Label file           : {dat_path}")

    if all_results:
        vf_ni, vf_ysz, vf_pore, sv_ni, sv_ysz, sv_pore = all_results[0]
        print()
        print("  First structure properties:")
        print(f"    Ni   VF={vf_ni:.1f}%   SSA={sv_ni:.2f} um^-1")
        print(f"    YSZ  VF={vf_ysz:.1f}%  SSA={sv_ysz:.2f} um^-1")
        print(f"    Pore VF={vf_pore:.1f}% SSA={sv_pore:.2f} um^-1")

    print()
    print("  Next steps:")
    print(f"    1. Set n_struc in 2_CNN/main.py and 1_GAN/main.py to: {struct_count}")
    print(f"    2. Set in_header / Input_header to: {output_dir.resolve()}")
    print( "    3. Run: python 2_CNN/main.py")
    print( "    4. Run: python 1_GAN/main.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
