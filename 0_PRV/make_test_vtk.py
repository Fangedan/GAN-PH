"""
make_test_vtk.py
================
Builds a legacy VTK volume file (STRUCTURED_POINTS, cell data 'Phases' = 1/2/3)
from a structure folder of grayscale BMP slices (slice_0000.bmp ...), i.e. it
fabricates the kind of file DREAM.3D hands to Juan -- but from a structure
whose ground truth YOU already know.

Run with your normal ganph python (numpy + pillow only):

  python make_test_vtk.py ../synthetic_data/structure_0001 -o test_structure.vtk

Mapping (matches Juan's DREAM.3D phase numbering, where phase 1 = Ni):
  BMP 255 (white, Ni)   -> Phase 1   (renders BLACK in the export)
  BMP 127 (gray,  YSZ)  -> Phase 2   (renders GREY)
  BMP 0   (black, Pore) -> Phase 3   (renders WHITE)
NOTE this is inverted vs the GAN-PH BMP grayscale (Ni=255). That inversion is
intentional: Juan's PNGs have black=Ni / white=Pore, and preprocess_dream3d.py
flips it back. The test VTK must mimic Juan's convention for the end-to-end
test to come out right.

Geometry: the volume is written with physical size 25.0 (microns) per side by
default, so a 64^3 structure gets spacing 25/64 ~= 0.3906 -- this reproduces
the 0-25 bounds implied by Juan's keyframes (0.2 ... 24.8).

Axis convention: BMP slice index -> VTK X axis, BMP column -> Y, BMP row -> Z.
So ParaView slicing along +X should reproduce slice_0000 ... slice_0063 in
order. (verify_slices.py double-checks orientation regardless.)
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

GRAY_TO_PHASE = {255: 1, 127: 2, 0: 3}  # Ni->1, YSZ->2, Pore->3 (Juan's convention)


def load_structure(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        sys.exit("No .bmp/.png slices found in %s" % folder)
    slices = []
    for f in files:
        a = np.array(Image.open(f).convert("L"))
        slices.append(a)
    vol = np.stack(slices, axis=0)  # (n_slice, rows, cols)
    print("Loaded %d slices of %dx%d from %s" % (vol.shape + (folder,)))

    # Snap to the three canonical gray levels, then map to phases 1/2/3
    levels = np.array([0, 127, 255])
    snapped = levels[np.argmin(np.abs(vol[..., None].astype(int) - levels), axis=-1)]
    n_off = int((snapped != vol).sum())
    if n_off:
        print("NOTE: %d pixels were not exactly 0/127/255; snapped to nearest." % n_off)
    phases = np.zeros_like(snapped, dtype=np.int32)
    for g, p in GRAY_TO_PHASE.items():
        phases[snapped == g] = p
    return phases


def write_legacy_vtk(phases, out_path, physical_size):
    n_slice, n_row, n_col = phases.shape
    nx, ny, nz = n_slice, n_col, n_row          # slice->X, col->Y, row->Z
    sx = physical_size / nx
    sy = physical_size / ny
    sz = physical_size / nz

    # Cell array ordering in VTK: x fastest, then y, then z.
    # cell value at (x=i, y=j, z=k) = phases[slice=i, row=k, col=j]
    data = np.transpose(phases, (1, 2, 0))       # (row=z, col=y, slice=x)
    flat = data.reshape(-1)                      # z slowest, then y, then x  ✓

    with open(out_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("GAN-PH test microstructure\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write("DIMENSIONS %d %d %d\n" % (nx + 1, ny + 1, nz + 1))  # points = cells+1
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING %.8f %.8f %.8f\n" % (sx, sy, sz))
        f.write("CELL_DATA %d\n" % flat.size)
        f.write("SCALARS Phases int 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, flat.reshape(-1, 16), fmt="%d")
    print("Wrote %s  (cells %dx%dx%d, bounds 0-%.4g, spacing %.6f)"
          % (out_path, nx, ny, nz, physical_size, sx))

    # Print ground truth for later comparison
    total = phases.size
    for p, name in [(1, "Ni"), (2, "YSZ"), (3, "Pore")]:
        print("  ground-truth VF %-4s (phase %d): %.4f"
              % (name, p, (phases == p).sum() / total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structure", help="Folder containing slice_*.bmp")
    ap.add_argument("-o", "--output", default="test_structure.vtk")
    ap.add_argument("--size", type=float, default=25.0,
                    help="Physical edge length (default 25.0, matching Juan's bounds)")
    args = ap.parse_args()
    phases = load_structure(args.structure)
    write_legacy_vtk(phases, args.output, args.size)


if __name__ == "__main__":
    main()
