"""
5_TAU/read_vtk_volume.py
========================
Parse an ASCII VTK rectilinear grid (CELL_DATA with integer phase labels) and
load it into our canonical (Z, Y, X) NumPy array.

Port of: 5_TAU/reference_matlab/readVTKRectilinearCellData.m

CRITICAL AXIS TRAP — documentation
------------------------------------
MATLAB ``reshape(data, cellDims)`` where ``cellDims = dims - 1 = [nx-1, ny-1, nz-1]``
is column-major: the FIRST dimension (x) varies fastest.  This produces
``vol(ix, iy, iz)`` in 1-based MATLAB indexing.

Two NumPy methods both recover the same (Z, Y, X) array:

  Method A — C-order with transposed shape
    arr = data.reshape((nz-1, ny-1, nx-1), order='C')
    Last index (x) varies fastest ✓

  Method B — F-order (MATLAB-faithful) then transpose
    arr_F = data.reshape((nx-1, ny-1, nz-1), order='F')   ix varies fastest ✓
    arr   = arr_F.T                                         → arr[iz, iy, ix]

This module computes both and asserts voxel-for-voxel equality so any future
change to the reader cannot silently reintroduce an axis swap.

Usage (standalone):
    conda run -n ganph --no-capture-output python 5_TAU/read_vtk_volume.py \\
        5_TAU/reference_matlab/microstructure_real.vtk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def read_vtk_rectilinear_cell_data(path: str | Path) -> dict:
    """
    Parse an ASCII VTK rectilinear grid with CELL_DATA integer labels.

    Mirrors readVTKRectilinearCellData.m lines 1-84, including the
    DIMENSIONS → cellDims = dims - 1 and column-major reshape.

    Returns
    -------
    dict with keys:
      volume    : np.ndarray (Z, Y, X) int32 — phase labels per voxel
      spacing   : tuple (dx, dy, dz) float   — mean voxel sizes (same units as coords)
      x, y, z   : np.ndarray float           — node coordinate arrays
      dims      : tuple (nx, ny, nz)         — node counts from DIMENSIONS header
      cell_dims : tuple (nx-1, ny-1, nz-1)  — cell counts per axis
    """
    path = Path(path)
    text = path.read_text(encoding="ascii", errors="replace")
    lines = text.splitlines()

    dims_nodes: tuple[int, int, int] | None = None
    x_coords: np.ndarray | None = None
    y_coords: np.ndarray | None = None
    z_coords: np.ndarray | None = None
    num_cells: int | None = None
    data_start_line: int | None = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        upper = line.upper()

        if upper.startswith("DIMENSIONS"):
            parts = line.split()
            dims_nodes = (int(parts[1]), int(parts[2]), int(parts[3]))

        elif upper.startswith("X_COORDINATES"):
            parts = line.split()
            count = int(parts[1])
            tokens: list[str] = []
            i += 1
            while len(tokens) < count and i < len(lines):
                tokens.extend(lines[i].split())
                i += 1
            x_coords = np.array([float(t) for t in tokens[:count]])
            continue

        elif upper.startswith("Y_COORDINATES"):
            parts = line.split()
            count = int(parts[1])
            tokens = []
            i += 1
            while len(tokens) < count and i < len(lines):
                tokens.extend(lines[i].split())
                i += 1
            y_coords = np.array([float(t) for t in tokens[:count]])
            continue

        elif upper.startswith("Z_COORDINATES"):
            parts = line.split()
            count = int(parts[1])
            tokens = []
            i += 1
            while len(tokens) < count and i < len(lines):
                tokens.extend(lines[i].split())
                i += 1
            z_coords = np.array([float(t) for t in tokens[:count]])
            continue

        elif upper.startswith("CELL_DATA"):
            parts = line.split()
            num_cells = int(parts[1])

        elif upper.startswith("LOOKUP_TABLE") and num_cells is not None:
            data_start_line = i + 1
            break

        i += 1

    if dims_nodes is None:
        raise ValueError("Could not find DIMENSIONS in VTK file.")
    if num_cells is None:
        raise ValueError("Could not find CELL_DATA in VTK file.")
    if data_start_line is None:
        raise ValueError("Could not find LOOKUP_TABLE line in VTK file.")

    nx_n, ny_n, nz_n = dims_nodes
    nx_c, ny_c, nz_c = nx_n - 1, ny_n - 1, nz_n - 1  # cell counts

    # Read integer data from remaining lines
    data_tokens: list[str] = []
    for line in lines[data_start_line:]:
        data_tokens.extend(line.split())
        if len(data_tokens) >= num_cells:
            break
    data = np.array([int(t) for t in data_tokens[:num_cells]], dtype=np.int32)

    if data.size != num_cells:
        raise ValueError(f"Expected {num_cells} cell values, read {data.size}.")
    if data.size != nx_c * ny_c * nz_c:
        raise ValueError(
            f"CELL_DATA count {data.size} != (dims-1) product {nx_c * ny_c * nz_c}."
        )

    # ── AXIS PROOF ────────────────────────────────────────────────────────────
    # Method A: C-order shape (nz_c, ny_c, nx_c) → arr_A[iz, iy, ix]
    # The last index (x) varies fastest, matching MATLAB column-major where
    # the first dimension (x) varies fastest.
    arr_A = data.reshape((nz_c, ny_c, nx_c), order="C")

    # Method B: F-order shape (nx_c, ny_c, nz_c) → arr_F[ix, iy, iz] (MATLAB-faithful)
    # Transpose → arr_B[iz, iy, ix]
    arr_F = data.reshape((nx_c, ny_c, nz_c), order="F")
    arr_B = arr_F.T

    if not np.array_equal(arr_A, arr_B):
        raise AssertionError(
            "AXIS TRAP: Method A (C-order) and Method B (F-order + T) disagree. "
            "The axis mapping is broken."
        )

    volume = arr_A  # canonical (Z, Y, X)

    dx = float(np.mean(np.diff(x_coords))) if x_coords is not None and len(x_coords) > 1 else float("nan")
    dy = float(np.mean(np.diff(y_coords))) if y_coords is not None and len(y_coords) > 1 else float("nan")
    dz = float(np.mean(np.diff(z_coords))) if z_coords is not None and len(z_coords) > 1 else float("nan")

    return {
        "volume":    volume,
        "spacing":   (dx, dy, dz),
        "x":         x_coords,
        "y":         y_coords,
        "z":         z_coords,
        "dims":      dims_nodes,
        "cell_dims": (nx_c, ny_c, nz_c),
    }


def report(vtk_data: dict, path: str | Path | None = None) -> None:
    """Print a concise summary of a loaded VTK volume."""
    vol = vtk_data["volume"]
    dx, dy, dz = vtk_data["spacing"]
    nz_c, ny_c, nx_c = vol.shape
    nx_n, ny_n, nz_n = vtk_data["dims"]
    total = vol.size

    print()
    print("=" * 64)
    if path:
        print(f"VTK file : {Path(path).name}")
    print(f"  Node DIMENSIONS  : {nx_n} x {ny_n} x {nz_n}  (nx x ny x nz)")
    print(f"  Cell counts      : {nx_c} x {ny_c} x {nz_c}  (X x Y x Z)")
    print(f"  volume.shape     : {vol.shape}  (Z, Y, X)")
    print(f"  Spacing          : dx={dx:.6g}  dy={dy:.6g}  dz={dz:.6g}")
    tol = 1e-6 * max(abs(dx), abs(dy), abs(dz))
    iso = "YES" if abs(dx - dy) < tol and abs(dx - dz) < tol else "NO"
    print(f"  Isotropic        : {iso}")

    phase_ids = sorted(np.unique(vol).tolist())
    print(f"  Phase IDs        : {phase_ids}")
    for pid in phase_ids:
        count = int((vol == pid).sum())
        print(f"    Phase {pid:3d}: {count:9d} voxels  ({100.0 * count / total:.2f}%)")

    print()
    print("  Per-axis mean VF profiles (min / max / std across slices):")
    for axis_name, sum_axes in [("Z (transport axis)", (1, 2)), ("Y", (0, 2)), ("X", (0, 1))]:
        for pid in phase_ids:
            prof = (vol == pid).mean(axis=sum_axes)
            print(
                f"    {axis_name:<18} phase {pid:2d}: "
                f"min={prof.min():.4f}  max={prof.max():.4f}  std={prof.std():.4f}"
            )

    print()
    min_dim = min(nx_c, ny_c, nz_c)
    can_64 = min_dim >= 64
    print(f"  Parent-volume check (64^3 sub-cubes):")
    print(f"    Minimum cell dimension: {min_dim} {'>=': <2} 64  --> {'YES' if can_64 else 'NO, CANNOT extract 64^3 crops'}")
    if can_64:
        n = ((nx_c - 64) // 64 + 1) * ((ny_c - 64) // 64 + 1) * ((nz_c - 64) // 64 + 1)
        print(f"    Non-overlapping 64^3 crops at stride 64: ~{n}")
    else:
        print(f"    Z dimension = {nz_c} < 64 — no 64^3 crop fits along the transport axis.")
    print("=" * 64)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Read and summarize a VTK rectilinear grid (CELL_DATA, ASCII)."
    )
    parser.add_argument("vtk", help="Path to .vtk file")
    args = parser.parse_args()

    data = read_vtk_rectilinear_cell_data(args.vtk)
    report(data, path=args.vtk)
    print("\nAxis proof: Method A (C-order) == Method B (F-order + T)  PASSED")
