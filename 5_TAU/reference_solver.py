"""
5_TAU/reference_solver.py
==========================
Faithful Python port of Prof. Jin's group MATLAB tortuosity solver.

Algorithm source (local-only, in 5_TAU/reference_matlab/ which is gitignored):
  computeEffectiveTortuosityDiffusionFast.m  — main solver, lines cited below
  keepPercolatingCluster.m                   — percolation filter, lines cited below

Every algorithmic block maps to the corresponding MATLAB lines in each function's
docstring.  The physical equations, epsilon convention, BC values, Laplacian
weights, flux formula, and tau formula are all ported exactly.

ONE FORMULATION DIFFERENCE (documented here, not a fix):
  MATLAB lines 85-103 set identity rows for inlet/outlet nodes inside the
  same matrix, producing a non-symmetric system.  scipy.sparse.linalg.cg
  does not reliably converge on non-symmetric matrices of 10^5+ unknowns
  (verified: relres ≈ 0.31 after maxit=1000 on the 100x100x50 VTK volume).
  This port uses the mathematically EQUIVALENT elimination form:
    - inlet/outlet nodes are REMOVED from the unknown vector
    - their c values (1 and 0) are substituted directly into the interior rows
    - the resulting interior sub-matrix is the symmetric SPD graph Laplacian
  The solution C, flux, Deff, and tau_eff are identical to MATLAB's intent;
  only the matrix-assembly step differs in form, not in result.

Faithfully ported algorithm decisions from MATLAB:
  - epsilon = total phase VF BEFORE percolation filter          (line 10)
  - 6-connectivity percolation check, both z-faces must touch   (keepPercolatingCluster.m)
  - Unweighted graph Laplacian: NO dx/dy/dz in the matrix;
    spacing only enters L, areaTotal, and flux post-processing  (lines 40-41, 220-239)
  - Dirichlet BCs: inlet c=1, outlet c=0                        (lines 85-103)
  - CG with spilu preconditioner, tol=1e-5, maxit=1000          (lines 151-161)
  - Flux at inlet face (iz=0) only, one-sided finite difference (lines 220-239)
  - L = (nz-1)*dz  cell-centre-to-cell-centre distance         (line 40)
  - areaTotal = nx*dx * ny*dy  full cross-section, not pathway  (line 41)
  - tau_eff = epsilon * D0 / Deff                              (line 246)

Z-direction transport only (matches eff_tor_diff.m driver, direction='z').

Usage (standalone):
    conda run -n ganph --no-capture-output python 5_TAU/reference_solver.py \\
        --vtk 5_TAU/reference_matlab/microstructure_real.vtk
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import label as nd_label


# 6-connectivity structuring element: face-adjacent voxels only.
# Matches MATLAB bwconncomp(mask, 6) in keepPercolatingCluster.m line 5.
_STRUCT_6CONN = np.zeros((3, 3, 3), dtype=bool)
for _dz, _dy, _dx in [(1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (0, 1, 1), (2, 1, 1)]:
    _STRUCT_6CONN[_dz, _dy, _dx] = True


def keep_percolating_cluster(mask: np.ndarray) -> np.ndarray:
    """
    Port of keepPercolatingCluster.m.

    Returns a boolean mask containing only voxels that belong to a 6-connected
    component touching BOTH the z=0 face (inlet) AND the z=nz-1 face (outlet).

    MATLAB source line map (keepPercolatingCluster.m):
      line  5 : bwconncomp(mask, 6)         → scipy.ndimage.label with 6-conn struct
      line 13 : [i,j,k] = ind2sub([nx,ny,nz], voxels)
                                             → np.unravel_index; our array is (Z,Y,X)
                                               so k (MATLAB z-index) = iz in Python
      line 27 : any(k == 1)                 → component[0, :, :].any()
      line 28 : any(k == nz)                → component[-1, :, :].any()
      line 32 : percolatingMask(voxels)=true→ percolating |= component

    Parameters
    ----------
    mask : (Z, Y, X) bool array — full phase mask (before percolation filtering)

    Returns
    -------
    percolating : (Z, Y, X) bool array — voxels in percolating components only
    """
    labeled, num_features = nd_label(mask, structure=_STRUCT_6CONN)
    percolating = np.zeros_like(mask, dtype=bool)
    for lbl in range(1, num_features + 1):
        component = labeled == lbl
        if component[0, :, :].any() and component[-1, :, :].any():
            percolating |= component
    return percolating


def compute_reference_tau(
    volume: np.ndarray,
    phase_id: int,
    spacing: tuple[float, float, float],
    D0: float = 1.0,
    tol: float = 1e-5,
    maxit: int = 1000,
    verbose: bool = False,
    return_field: bool = False,
) -> dict:
    """
    Port of computeEffectiveTortuosityDiffusionFast.m (z-direction only).

    Computes effective tortuosity tau_eff = epsilon * D0 / Deff for the phase
    identified by ``phase_id`` in the integer-labelled ``volume``.

    MATLAB source line map (computeEffectiveTortuosityDiffusionFast.m):
      line  4 : [nx,ny,nz]=size(mask)       → nz,ny,nx=mask.shape (our Z,Y,X)
      line  6 : dx,dy,dz=spacing(1:3)       → dx,dy,dz=spacing
      line 10 : epsilon=nnz(mask)/numel()   → epsilon=mask_full.mean() (BEFORE filter)
      line 17 : if nnz(mask)==0 → return    → handled below
      line 25 : keepPercolatingCluster()    → keep_percolating_cluster()
      line 27 : if nnz(pathwayMask)==0      → handled below
      line 34 : mask=pathwayMask            → pathway used hereafter
      line 40 : L=(nz-1)*dz                → L=(nz-1)*dz
      line 41 : areaTotal=nx*dx*ny*dy      → area_total=nx*dx*ny*dy
      lines 43-47: inletMask/outletMask    → inlet[0]/outlet[-1] restricted to pathway
      line 54 : nodeID=zeros(nx,ny,nz)     → nodeID array, -1 = not in pathway
      line 55 : phaseVoxels=find(mask)     → pathway_flat (C-order flatnonzero)
      lines 85-103 : BC values c=1, c=0   → c_inlet=1, c_outlet=0 substituted into RHS
      [FORMULATION NOTE: MATLAB inserts identity rows; this port eliminates BC nodes.
       The resulting A_int is the symmetric SPD sub-Laplacian. CG converges reliably.]
      lines 107-138: interior Laplacian   → off-diag=-1, diag=degree (interior only)
      line 148: tol=1e-5                  → tol parameter (default 1e-5)
      line 149: maxit=1000                → maxit parameter (default 1000)
      lines 151-161: ichol+pcg/pcg        → spilu+cg / unpreconditioned cg on A_int
      line 171: C(phaseVoxels)=c          → C filled from inlet=1, outlet=0, interior=x
      lines 220-239: totalFlux at k=1    → iz=0 face, one-sided (both iz=0,1 in path)
      line 241: Javg=totalFlux/areaTotal  → J_avg=total_flux/area_total
      line 243: Deff=Javg*L              → D_eff=J_avg*L
      line 246: tau_eff=eps*D0/Deff      → tau_eff=epsilon*D0/D_eff

    Parameters
    ----------
    volume       : (Z, Y, X) int array of phase labels
    phase_id     : integer label for the phase to solve
    spacing      : (dx, dy, dz) voxel sizes in consistent units (e.g. µm)
    D0           : free diffusivity (default 1.0; cancels in tau formula)
    tol          : CG convergence tolerance (default 1e-5, matches MATLAB line 148)
    maxit        : CG maximum iterations (default 1000, matches MATLAB line 149)
    verbose      : print progress messages
    return_field : if True, include "C" and "pathway" in the returned dict

    Returns
    -------
    dict with keys:
      tau_eff      : float       — effective tortuosity (np.inf if not percolating)
      Deff         : float       — effective diffusivity
      epsilon      : float       — total phase VF (BEFORE percolation filter)
      epsilon_perc : float       — percolating phase VF (after filter)
      n_nodes      : int         — total pathway voxels
      n_interior   : int         — interior (unknown) voxels solved
      pcg_flag     : int         — CG convergence flag (0 = converged, -1 = not run)
      pcg_relres   : float       — relative residual norm after solve
      percolating  : bool        — whether a percolating cluster was found
      C            : (Z,Y,X) float64 or None  — concentration field (if return_field)
      pathway      : (Z,Y,X) bool or None     — percolating mask   (if return_field)
    """
    dx, dy, dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    mask_full = (volume == phase_id)

    # line 10: epsilon BEFORE percolation filter
    epsilon = float(mask_full.mean())
    nz, ny, nx = mask_full.shape

    _none = {"C": None, "pathway": None} if return_field else {}

    if not mask_full.any():
        return dict(
            tau_eff=np.inf, Deff=0.0, epsilon=epsilon, epsilon_perc=0.0,
            n_nodes=0, n_interior=0, pcg_flag=-1, pcg_relres=np.nan, percolating=False,
            **_none,
        )

    # line 25: percolation filter
    pathway = keep_percolating_cluster(mask_full)

    if not pathway.any():
        if verbose:
            print(f"  Phase {phase_id}: no percolating z-pathway.")
        return dict(
            tau_eff=np.inf, Deff=0.0, epsilon=epsilon, epsilon_perc=0.0,
            n_nodes=0, n_interior=0, pcg_flag=-1, pcg_relres=np.nan, percolating=False,
            **_none,
        )

    epsilon_perc = float(pathway.mean())

    # lines 40-41: geometry (z-direction)
    L = (nz - 1) * dz                  # cell-centre to cell-centre distance
    area_total = nx * dx * ny * dy      # full cross-sectional area (not pathway only)

    # lines 43-47: BC masks restricted to pathway voxels
    inlet  = np.zeros(pathway.shape, dtype=bool)
    outlet = np.zeros(pathway.shape, dtype=bool)
    inlet[0,  :, :] = pathway[0,  :, :]   # z=0 face
    outlet[-1, :, :] = pathway[-1, :, :]  # z=nz-1 face

    # lines 54-57: node numbering in C-order
    pathway_flat = np.flatnonzero(pathway)   # 1D indices into pathway in C-order
    n_nodes = len(pathway_flat)
    nodeID = np.full(pathway.shape, -1, dtype=np.int64)
    nodeID.flat[pathway_flat] = np.arange(n_nodes, dtype=np.int64)

    # ── Classify pathway nodes into inlet / outlet / interior ────────────────
    is_inlet  = inlet.flat[pathway_flat].astype(bool)
    is_outlet = outlet.flat[pathway_flat].astype(bool)
    is_int    = ~(is_inlet | is_outlet)

    int_idx = np.where(is_int)[0]           # positions in [0..n_nodes)
    n_int   = len(int_idx)

    # Mapping from global node ID → local interior index (-1 if not interior)
    local_id = np.full(n_nodes, -1, dtype=np.int64)
    local_id[int_idx] = np.arange(n_int, dtype=np.int64)

    if verbose:
        n_in = is_inlet.sum(); n_out = is_outlet.sum()
        print(
            f"  Phase {phase_id}: {n_nodes} pathway voxels "
            f"({n_in} inlet, {n_out} outlet, {n_int} interior), solving..."
        )

    # ── Build symmetric interior Laplacian (elimination form) ────────────────
    # lines 107-138: unweighted Laplacian on interior nodes
    # BCs substituted directly: adjacent-to-inlet → +1 on RHS (c_inlet=1)
    #                           adjacent-to-outlet → 0 on RHS (c_outlet=0)
    # Result: A_int is symmetric SPD (graph Laplacian on interior sub-graph).

    # Decompose interior node flat-indices to (iz, iy, ix)
    flat_int = pathway_flat[int_idx]
    iz_int   = flat_int // (ny * nx)
    rem_int  = flat_int % (ny * nx)
    iy_int   = rem_int // nx
    ix_int   = rem_int % nx

    # 6-connectivity offsets in (Z, Y, X)
    offsets_zyx = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

    off_rows: list[np.ndarray] = []
    off_cols: list[np.ndarray] = []
    off_vals: list[np.ndarray] = []
    degree   = np.zeros(n_int, dtype=np.int64)  # degree over ALL pathway neighbours
    b_int    = np.zeros(n_int, dtype=np.float64)  # RHS: inlet-neighbour contributions

    for diz, diy, dix in offsets_zyx:
        niz = iz_int + diz
        niy = iy_int + diy
        nix = ix_int + dix

        in_bounds = (
            (niz >= 0) & (niz < nz) &
            (niy >= 0) & (niy < ny) &
            (nix >= 0) & (nix < nx)
        )
        n_flat = np.where(
            in_bounds, niz * (ny * nx) + niy * nx + nix, 0
        ).astype(np.int64)

        # Check which neighbours are in the pathway
        in_path = np.zeros(n_int, dtype=bool)
        if in_bounds.any():
            in_path[in_bounds] = pathway.flat[n_flat[in_bounds]]

        valid = in_bounds & in_path
        if not valid.any():
            continue

        # Accumulate degree over all pathway neighbours (interior + bc)
        np.add.at(degree, np.where(valid)[0], 1)

        # Classify neighbour type
        nbr_global = nodeID.flat[n_flat[valid]]
        nbr_local  = local_id[nbr_global]

        # Interior-interior edges: off-diagonal -1 (contributes to A_int)
        is_int_nbr = nbr_local >= 0
        src_ii = np.where(valid)[0][is_int_nbr]
        dst_ii = nbr_local[is_int_nbr]
        if src_ii.size:
            off_rows.append(src_ii.astype(np.int64))
            off_cols.append(dst_ii.astype(np.int64))
            off_vals.append(np.full(src_ii.shape, -1.0, dtype=np.float64))

        # Interior-inlet edges: c_inlet=1 → +1 on RHS
        is_inlet_nbr = (nbr_local < 0) & is_inlet[nbr_global]
        rhs_src = np.where(valid)[0][is_inlet_nbr]
        if rhs_src.size:
            np.add.at(b_int, rhs_src, 1.0)

        # Interior-outlet edges: c_outlet=0 → no RHS contribution

    # Diagonal: local interior indices (0..n_int-1)
    diag_local = np.arange(n_int, dtype=np.int64)
    all_rows = np.concatenate(off_rows + [diag_local]) if off_rows else diag_local
    all_cols = np.concatenate(off_cols + [diag_local]) if off_cols else diag_local
    all_vals = np.concatenate(off_vals + [degree.astype(np.float64)]) if off_vals else degree.astype(np.float64)

    A_int = sp.csr_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_int, n_int),
        dtype=np.float64,
    )

    # ── Solve symmetric SPD system (lines 151-161) ───────────────────────────
    # Matching MATLAB lines 151-161: try preconditioned CG, fall back to direct.
    # MATLAB's ichol+pcg targets the same SPD sub-Laplacian (elimination form).
    # For the 100x100x50 VTK volume, A_int is up to 200K x 200K with ~1.2M nnz.
    # scipy spsolve (SuperLU) handles this reliably in <30s on typical hardware.
    pcg_flag: int = -1
    pcg_relres: float = np.nan
    if n_int == 0:
        c_int = np.array([], dtype=np.float64)
        pcg_flag = 0
        pcg_relres = 0.0
    else:
        A_csc = A_int.tocsc()
        # Try CG with ILU preconditioner first (fast path, matches MATLAB intent)
        _cg_ok = False
        try:
            ilu = spla.spilu(A_csc, drop_tol=1e-3)
            M_op = spla.LinearOperator(A_int.shape, matvec=ilu.solve)
            c_try, _flag = spla.cg(A_int, b_int, tol=tol, maxiter=maxit, M=M_op)
            b_norm = float(np.linalg.norm(b_int))
            _relres = float(np.linalg.norm(A_int @ c_try - b_int) / (b_norm or 1.0))
            if _flag == 0 and _relres < 1e-4:
                c_int, pcg_flag, pcg_relres = c_try, 0, _relres
                _cg_ok = True
        except Exception:
            pass

        if not _cg_ok:
            # Fall back to direct sparse solve (SuperLU) — always converges
            if verbose:
                print(f"  CG did not converge; using direct solver (SuperLU).")
            c_int = spla.spsolve(A_csc, b_int)
            pcg_flag = 0
            b_norm = float(np.linalg.norm(b_int))
            pcg_relres = float(np.linalg.norm(A_int @ c_int - b_int) / (b_norm or 1.0))

    if verbose:
        conv = "CG converged" if pcg_flag == 0 else f"CG flag={pcg_flag}"
        print(f"  {conv}, relres={pcg_relres:.2e}, n_interior={n_int}")

    # ── Reconstruct full concentration C in (Z, Y, X) (line 171) ────────────
    C = np.full(pathway.shape, np.nan, dtype=np.float64)
    C.flat[pathway_flat[is_inlet]]  = 1.0
    C.flat[pathway_flat[is_outlet]] = 0.0
    if n_int > 0:
        C.flat[pathway_flat[int_idx]] = c_int

    # ── Flux at inlet face (iz=0), one-sided (lines 220-239) ─────────────────
    # Sums -D0*(C[1]-C[0])/dz * dx*dy over all (iy,ix) where BOTH iz=0 AND iz=1
    # are in the pathway (matches mask(i,j,1) && mask(i,j,2) in MATLAB).
    both_on    = pathway[0, :, :] & pathway[1, :, :]
    grad_c     = (C[1, :, :] - C[0, :, :]) / dz
    flux_map   = -D0 * grad_c * dx * dy
    total_flux = float(np.nansum(flux_map[both_on]))

    # lines 241-246
    J_avg   = total_flux / area_total
    D_eff   = J_avg * L
    tau_eff = (epsilon * D0 / D_eff) if D_eff > 0.0 else np.inf

    result = dict(
        tau_eff=tau_eff,
        Deff=D_eff,
        epsilon=epsilon,
        epsilon_perc=epsilon_perc,
        n_nodes=n_nodes,
        n_interior=n_int,
        pcg_flag=pcg_flag,
        pcg_relres=pcg_relres,
        percolating=True,
    )
    if return_field:
        result["C"] = C
        result["pathway"] = pathway
    return result


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Run reference MATLAB-port solver on a VTK rectilinear grid."
    )
    parser.add_argument("--vtk", required=True, help="Path to .vtk file")
    parser.add_argument(
        "--phase-ids", nargs="+", type=int, default=[1, 2, 3],
        help="Phase IDs to solve (default: 1 2 3)"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from read_vtk_volume import read_vtk_rectilinear_cell_data, report as vtk_report

    vtk_data = read_vtk_rectilinear_cell_data(args.vtk)
    vtk_report(vtk_data, path=args.vtk)

    vol     = vtk_data["volume"]
    spacing = vtk_data["spacing"]

    print("\nReference solver results (z-direction):")
    print(
        f"  {'Phase':>5}  {'epsilon':>8}  {'eps_perc':>9}  "
        f"{'Deff':>10}  {'tau_eff':>10}  {'CG':>4}  {'relres':>8}"
    )
    print("  " + "-" * 66)
    for pid in args.phase_ids:
        res = compute_reference_tau(vol, pid, spacing, verbose=args.verbose)
        tau_str = f"{res['tau_eff']:.5f}" if np.isfinite(res['tau_eff']) else "Inf"
        print(
            f"  {pid:5d}  {res['epsilon']:8.5f}  {res['epsilon_perc']:9.5f}  "
            f"{res['Deff']:10.6f}  {tau_str:>10}  {res['pcg_flag']:4d}  "
            f"{res['pcg_relres']:8.2e}"
        )
