"""
Synthetic Ni-YSZ microstructure generator
Produces BMP stacks + results.dat in the exact format expected by GAN-PH/1_GAN/load.py

Phase labels:  Ni=255, YSZ=127, Pore=0
Output layout:
  output_dir/
    results.dat
    structure_0001/
      slice_0000.bmp ... slice_0063.bmp
    structure_0002/
      ...
"""

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
N_STRUCTURES  = 50        # number of structures to generate (paper used 17550;
                          # 50 is enough to test the full pipeline quickly)
VOXEL_SIZE    = 64        # each structure is 64x64x64
OUTPUT_DIR    = Path("./synthetic_data")

# Volume fraction targets: each row is [Ni%, YSZ%, Pore%]
# We vary them to give the GAN something to learn from
VF_CONFIGS = [
    (48, 19, 33),   # Ni-heavy  (≈ Ni:YSZ 70:30)
    (38, 41, 21),   # balanced  (≈ Ni:YSZ 50:50)
    (25, 55, 20),   # YSZ-heavy (≈ Ni:YSZ 30:70)
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def place_spheres(volume, phase_value, target_vf, radius_range=(2, 5), max_attempts=5000):
    """Pack random spheres of given phase into volume until target VF is reached."""
    size = volume.shape[0]
    target_voxels = int(target_vf * size**3)
    placed = 0
    attempts = 0

    while placed < target_voxels and attempts < max_attempts:
        r  = np.random.randint(*radius_range)
        cx = np.random.randint(r, size - r)
        cy = np.random.randint(r, size - r)
        cz = np.random.randint(r, size - r)

        # Build sphere mask
        zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
        sphere = (xx**2 + yy**2 + zz**2) <= r**2

        # Only place in pore space (value==0) to avoid overwriting
        region = volume[cz-r:cz+r+1, cy-r:cy+r+1, cx-r:cx+r+1]
        mask   = sphere & (region == 0)
        region[mask] = phase_value
        placed += mask.sum()
        attempts += 1

    return volume


def compute_volume_fractions(volume):
    """Return [VF_Ni, VF_YSZ, VF_Pore] as percentages."""
    total = volume.size
    vf_ni  = 100 * np.sum(volume == 255) / total
    vf_ysz = 100 * np.sum(volume == 127) / total
    vf_por = 100 * np.sum(volume ==   0) / total
    return vf_ni, vf_ysz, vf_por


def compute_surface_area(volume, phase_value, voxel_size_um=0.1):
    """
    Estimate specific surface area (SSA) of a phase using voxel face counting.
    SSA = surface_voxel_faces / phase_volume  [units: 1/um]
    """
    mask = (volume == phase_value).astype(np.uint8)
    # Count faces shared with a different phase by shifting in each direction
    surface_faces = 0
    for axis in range(3):
        diff = np.diff(mask, axis=axis)
        surface_faces += np.sum(np.abs(diff))

    phase_volume_um3 = np.sum(mask) * (voxel_size_um ** 3)
    surface_area_um2 = surface_faces * (voxel_size_um ** 2)

    if phase_volume_um3 == 0:
        return 0.0
    return surface_area_um2 / phase_volume_um3


def generate_structure(target_ni_pct, target_ysz_pct):
    """Generate one 64^3 synthetic Ni-YSZ-Pore volume."""
    volume = np.zeros((VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE), dtype=np.uint8)

    # Place Ni spheres first, then YSZ into remaining pore space
    place_spheres(volume, 255, target_ni_pct  / 100, radius_range=(2, 5))
    place_spheres(volume, 127, target_ysz_pct / 100, radius_range=(2, 4))
    # Remaining voxels stay 0 (pore)
    return volume


def save_structure(volume, structure_idx, output_dir):
    """Save 64 BMP slices for one structure."""
    folder = output_dir / f"structure_{structure_idx:04d}"
    folder.mkdir(parents=True, exist_ok=True)

    for z in range(VOXEL_SIZE):
        slice_img = volume[z].astype(np.uint8)
        path = folder / f"slice_{z:04d}.bmp"
        cv2.imwrite(str(path), slice_img)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Generating {N_STRUCTURES} synthetic structures → {OUTPUT_DIR}\n")

    for i in tqdm(range(N_STRUCTURES)):
        # Cycle through VF configs with some random jitter
        base_ni, base_ysz, _ = VF_CONFIGS[i % len(VF_CONFIGS)]
        ni_pct  = base_ni  + np.random.uniform(-3, 3)
        ysz_pct = base_ysz + np.random.uniform(-3, 3)
        ni_pct  = np.clip(ni_pct,  5, 70)
        ysz_pct = np.clip(ysz_pct, 5, 70)

        volume = generate_structure(ni_pct, ysz_pct)
        save_structure(volume, i + 1, OUTPUT_DIR)

        # Compute actual properties from the generated volume
        vf_ni, vf_ysz, vf_por = compute_volume_fractions(volume)
        sv_ni  = compute_surface_area(volume, 255)
        sv_ysz = compute_surface_area(volume, 127)
        sv_por = compute_surface_area(volume,   0)

        results.append((vf_ni, vf_ysz, vf_por, sv_ni, sv_ysz, sv_por))

    # Write results.dat
    dat_path = OUTPUT_DIR / "results.dat"
    with open(dat_path, "w") as f:
        f.write("VF0 VF1 VF2 SV0 SV1 SV2\n")
        for row in results:
            f.write(" ".join(f"{v:.4f}" for v in row) + "\n")

    print(f"\nDone!")
    print(f"  Structures : {OUTPUT_DIR}/structure_0001 ... structure_{N_STRUCTURES:04d}")
    print(f"  Labels     : {dat_path}")
    print(f"\nFirst 3 rows of results.dat:")
    for row in results[:3]:
        print(" ", " ".join(f"{v:.4f}" for v in row))


if __name__ == "__main__":
    np.random.seed(42)
    main()
