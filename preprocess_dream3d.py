"""
preprocess_dream3d.py
=====================
Converts DREAM.3D-exported PNG slice stacks into the exact format
expected by the GAN-PH training pipeline (load.py).

What this script does, step by step:
  1. Reads all PNG slices from the input folder (named Slice0000.png etc.)
  2. Thresholds each image to clean phase values (removes anti-aliasing blur)
  3. Remaps phase pixel values to match load.py:
        DREAM.3D exports:   255=Pore (white),  0=Ni (black),  127=YSZ (grey)
        load.py expects:    255=Ni  (white),   0=Pore (black), 127=YSZ (grey)
     So: 255→0, 0→255, 127→127
  4. Crops out the black border DREAM.3D adds around each slice
  5. Resizes the cropped slice from ~452x452 down to 64x64
  6. Extracts one or more 64x64x64 sub-volumes from the full stack
  7. Saves each sub-volume as 64 BMP slices in structure_XXXX/ folders
  8. Measures volume fractions (by counting voxels) and specific surface
     areas (by counting boundary faces) for each sub-volume
  9. Writes all measurements to results.dat

Output layout:
  output_dir/
    results.dat
    structure_0001/
      slice_0000.bmp  ...  slice_0063.bmp
    structure_0002/
      ...

Usage:
  python preprocess_dream3d.py --input path/to/png/folder --output ./real_data

Requirements:
  pip install opencv-python numpy tqdm
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────────
VOXEL_SIZE    = 64          # output structure size (64x64x64)
VOXEL_SIZE_UM = 0.1         # physical size of each voxel in micrometers (100 nm)

# Phase pixel values in the OUTPUT (what load.py expects)
NI_VAL   = 255
YSZ_VAL  = 127
PORE_VAL = 0

# Phase pixel values in DREAM.3D's PNG output (what Juan sent)
DREAM3D_NI_VAL   = 0    # Ni is black in DREAM.3D output
DREAM3D_YSZ_VAL  = 127  # YSZ is grey (same in both)
DREAM3D_PORE_VAL = 255  # Pore is white in DREAM.3D output


# ── Step 1: Load all PNG slices ────────────────────────────────────────────────
def load_slices(input_folder):
    """Load all Slice*.png files from a folder, sorted by name."""
    input_path = Path(input_folder)
    png_files = sorted(input_path.glob('Slice*.png'))

    if len(png_files) == 0:
        raise FileNotFoundError(
            f"No files matching 'Slice*.png' found in {input_folder}\n"
            f"Make sure the folder contains files named like Slice0000.png, Slice0001.png, etc."
        )

    print(f"Found {len(png_files)} PNG slices in {input_folder}")

    slices = []
    for f in tqdm(png_files, desc="Loading PNGs"):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read image: {f}")
        slices.append(img)

    return slices


# ── Step 2: Threshold to clean phase values ────────────────────────────────────
def threshold_to_phases(img):
    """
    DREAM.3D exports PNG with anti-aliasing blur between phases.
    This snaps every pixel to the nearest of {0, 127, 255}.

    For each pixel, we compute distance to each phase value and
    assign it to the closest one.
    """
    img_int = img.astype(np.int32)
    d0   = np.abs(img_int - 0)
    d127 = np.abs(img_int - 127)
    d255 = np.abs(img_int - 255)

    out = np.zeros_like(img)
    out[d127 <= d0]                    = 127   # closer to YSZ than to black
    out[(d255 <= d0) & (d255 < d127)]  = 255   # closer to white than anything
    # Pixels already 0 from initialization = black = Ni (in DREAM.3D space)

    return out


# ── Step 3: Remap phase values ─────────────────────────────────────────────────
def remap_phases(img):
    """
    Remap from DREAM.3D conventions to load.py conventions:
        DREAM.3D:  0=Ni, 127=YSZ, 255=Pore
        load.py:   255=Ni, 127=YSZ, 0=Pore

    So: black pixels (Ni) become white, white pixels (Pore) become black,
    and grey pixels (YSZ) stay grey.
    """
    out = np.zeros_like(img)
    out[img == DREAM3D_YSZ_VAL]  = YSZ_VAL   # 127 → 127 (no change)
    out[img == DREAM3D_NI_VAL]   = NI_VAL    # 0   → 255
    out[img == DREAM3D_PORE_VAL] = PORE_VAL  # 255 → 0
    return out


# ── Step 4: Crop black border ──────────────────────────────────────────────────
def crop_border(img):
    """
    DREAM.3D adds a black border around the content.
    After remapping, Ni is white (255) and Pore is black (0) — but the
    border is all-black. We detect the border by finding the bounding box
    of pixels that are NOT zero in the ORIGINAL (pre-remap) image, since
    the border is 0 before any remapping.

    We use a fixed crop of 24 pixels on each side based on inspection
    of Juan's data (content starts at row/col 24, ends at 475).
    """
    BORDER = 24
    cropped = img[BORDER:-BORDER, BORDER:-BORDER]
    return cropped


# ── Step 5: Resize to 64x64 ───────────────────────────────────────────────────
def resize_slice(img, size=VOXEL_SIZE):
    """
    Resize a slice to size x size using nearest-neighbor interpolation.
    Nearest-neighbor is critical here — we must not introduce new pixel
    values. Linear or cubic interpolation would create intermediate values
    that aren't valid phase labels.
    """
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized


# ── Step 6: Extract 64x64x64 sub-volumes ──────────────────────────────────────
def extract_subvolumes(volume_3d, voxel_size=VOXEL_SIZE):
    """
    Given a 3D numpy array of shape (n_slices, height, width), extract
    as many non-overlapping 64x64x64 sub-volumes as possible.

    Since Juan's data has 50 slices (less than 64), we pad in the z
    direction by mirroring the stack. This is a standard technique
    for volumetric data augmentation.

    Returns a list of 3D numpy arrays, each of shape (64, 64, 64).
    """
    n_z, n_y, n_x = volume_3d.shape

    # Pad z dimension if needed by mirroring
    if n_z < voxel_size:
        print(f"  Z dimension ({n_z}) < {voxel_size} — padding by mirroring slices")
        needed = voxel_size - n_z
        # Mirror: append reversed slices from the end
        mirror = volume_3d[::-1][:needed]
        volume_3d = np.concatenate([volume_3d, mirror], axis=0)
        n_z = volume_3d.shape[0]
        print(f"  Padded to {n_z} slices")

    subvolumes = []

    # Extract non-overlapping cubes in all three dimensions
    n_z_cubes = n_z // voxel_size
    n_y_cubes = n_y // voxel_size
    n_x_cubes = n_x // voxel_size

    for iz in range(n_z_cubes):
        for iy in range(n_y_cubes):
            for ix in range(n_x_cubes):
                z0, z1 = iz * voxel_size, (iz+1) * voxel_size
                y0, y1 = iy * voxel_size, (iy+1) * voxel_size
                x0, x1 = ix * voxel_size, (ix+1) * voxel_size
                cube = volume_3d[z0:z1, y0:y1, x0:x1]
                subvolumes.append(cube)

    return subvolumes


# ── Step 7: Save sub-volume as BMP slices ──────────────────────────────────────
def save_structure(volume, structure_idx, output_dir):
    """
    Save a 64x64x64 volume as 64 grayscale BMP files.
    File naming: structure_0001/slice_0000.bmp ... slice_0063.bmp
    """
    folder = Path(output_dir) / f"structure_{structure_idx:04d}"
    folder.mkdir(parents=True, exist_ok=True)

    for z in range(volume.shape[0]):
        slice_img = volume[z].astype(np.uint8)
        path = folder / f"slice_{z:04d}.bmp"
        cv2.imwrite(str(path), slice_img)


# ── Step 8a: Measure volume fractions ──────────────────────────────────────────
def compute_volume_fractions(volume):
    """
    Count voxels of each phase and divide by total.
    Returns (vf_ni, vf_ysz, vf_pore) as percentages.
    """
    total = volume.size
    vf_ni   = 100.0 * np.sum(volume == NI_VAL)   / total
    vf_ysz  = 100.0 * np.sum(volume == YSZ_VAL)  / total
    vf_pore = 100.0 * np.sum(volume == PORE_VAL) / total
    return vf_ni, vf_ysz, vf_pore


# ── Step 8b: Measure specific surface area ────────────────────────────────────
def compute_surface_area(volume, phase_value):
    """
    Estimate specific surface area (SSA) of a phase using voxel face counting.

    For each pair of adjacent voxels, if one is the target phase and the
    other is not, that shared face is a surface face. We count these in
    all three axis directions (x, y, z).

    SSA = surface_area_um2 / phase_volume_um3  [units: 1/um]
    """
    mask = (volume == phase_value).astype(np.int8)

    surface_faces = 0
    for axis in range(3):
        # np.diff computes mask[i+1] - mask[i] along the axis
        # A difference of ±1 means a boundary between phase and non-phase
        diff = np.diff(mask, axis=axis)
        surface_faces += int(np.sum(np.abs(diff)))

    phase_volume_um3  = float(np.sum(mask)) * (VOXEL_SIZE_UM ** 3)
    surface_area_um2  = float(surface_faces) * (VOXEL_SIZE_UM ** 2)

    if phase_volume_um3 == 0:
        return 0.0

    return surface_area_um2 / phase_volume_um3


# ── Step 9: Write results.dat ──────────────────────────────────────────────────
def write_results_dat(output_dir, results):
    """
    Write the label file that load.py reads.
    Format: one row per structure, columns: VF0 VF1 VF2 SV0 SV1 SV2
    Where VF = volume fraction (%), SV = specific surface area (1/um).
    """
    dat_path = Path(output_dir) / "results.dat"
    with open(dat_path, 'w') as f:
        f.write("VF0 VF1 VF2 SV0 SV1 SV2\n")
        for row in results:
            f.write(" ".join(f"{v:.4f}" for v in row) + "\n")
    print(f"\nWrote {len(results)} rows to {dat_path}")
    return dat_path


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DREAM.3D PNG slices for GAN-PH training pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to folder containing Slice0000.png ... SliceXXXX.png"
    )
    parser.add_argument(
        "--output", "-o",
        default="./real_data",
        help="Path to output folder (default: ./real_data)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DREAM.3D → GAN-PH Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input  : {args.input}")
    print(f"Output : {output_dir}")
    print()

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("Step 1: Loading PNG slices...")
    raw_slices = load_slices(args.input)
    print(f"  Loaded {len(raw_slices)} slices, each {raw_slices[0].shape[0]}x{raw_slices[0].shape[1]}")

    # ── Steps 2-5: Process each slice ─────────────────────────────────────────
    print("\nSteps 2-5: Thresholding, remapping, cropping, resizing...")
    processed_slices = []
    for img in tqdm(raw_slices, desc="Processing slices"):
        img = threshold_to_phases(img)   # Step 2: clean anti-aliasing
        img = remap_phases(img)          # Step 3: fix phase values
        img = crop_border(img)           # Step 4: remove black border
        img = resize_slice(img)          # Step 5: resize to 64x64
        processed_slices.append(img)

    # Verify phase values after processing
    sample = processed_slices[len(processed_slices)//2]
    print(f"\n  Sample slice phase distribution (middle slice):")
    for v, name in [(NI_VAL,'Ni'), (YSZ_VAL,'YSZ'), (PORE_VAL,'Pore')]:
        pct = 100 * np.sum(sample == v) / sample.size
        print(f"    {name} ({v}): {pct:.1f}%")

    # ── Step 6: Build 3D volume and extract sub-volumes ───────────────────────
    print("\nStep 6: Building 3D volume and extracting sub-volumes...")
    volume_3d = np.stack(processed_slices, axis=0)
    print(f"  Full volume shape: {volume_3d.shape}  (z, y, x)")

    subvolumes = extract_subvolumes(volume_3d)
    print(f"  Extracted {len(subvolumes)} sub-volume(s) of shape {subvolumes[0].shape}")

    # ── Steps 7-9: Save and measure ───────────────────────────────────────────
    print(f"\nSteps 7-8: Saving BMP stacks and measuring properties...")
    results = []

    for i, vol in enumerate(tqdm(subvolumes, desc="Saving structures")):
        # Save BMP slices
        save_structure(vol, i + 1, output_dir)

        # Measure properties
        vf_ni, vf_ysz, vf_pore = compute_volume_fractions(vol)
        sv_ni   = compute_surface_area(vol, NI_VAL)
        sv_ysz  = compute_surface_area(vol, YSZ_VAL)
        sv_pore = compute_surface_area(vol, PORE_VAL)

        results.append((vf_ni, vf_ysz, vf_pore, sv_ni, sv_ysz, sv_pore))

    # Write results.dat
    print("\nStep 9: Writing results.dat...")
    dat_path = write_results_dat(output_dir, results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Done! Summary:")
    print(f"  Structures generated : {len(subvolumes)}")
    print(f"  Output folder        : {output_dir}")
    print(f"  Label file           : {dat_path}")
    print()
    print("  First structure properties:")
    vf_ni, vf_ysz, vf_pore, sv_ni, sv_ysz, sv_pore = results[0]
    print(f"    Ni   VF={vf_ni:.1f}%   SSA={sv_ni:.2f} μm⁻¹")
    print(f"    YSZ  VF={vf_ysz:.1f}%   SSA={sv_ysz:.2f} μm⁻¹")
    print(f"    Pore VF={vf_pore:.1f}%  SSA={sv_pore:.2f} μm⁻¹")
    print()
    print("Next steps:")
    print("  1. Set n_struc in 2_CNN/main.py and 1_GAN/main.py to:", len(subvolumes))
    print("  2. Set in_header / Input_header to:", output_dir.resolve())
    print("  3. Run: python 2_CNN/main.py")
    print("  4. Run: python 1_GAN/main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
