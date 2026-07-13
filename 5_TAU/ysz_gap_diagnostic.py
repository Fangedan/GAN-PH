"""
5_TAU/ysz_gap_diagnostic.py
============================
Layer B follow-up: diagnose the YSZ 9.59% gap between reference solver and taufactor.

TASK B — Interior-plane flux diagnostic
  For the YSZ phase of the VTK volume, compute the reference solver's total z-flux
  at every cut plane k→k+1 (sum over cells where both k and k+1 are in the
  percolating pathway).  If all planes agree to ~solver residual (expected from
  discrete Laplacian conservation), the "one-sided inlet flux underestimates"
  hypothesis is FALSE.  If planes disagree materially, quantify the effect.

TASK C — taufactor convergence tightening
  Re-run taufactor on the YSZ binary with iter_limit=100000, conv_crit=0.001.
  Compare against the default (iter_limit=10000, conv_crit=0.01, tau=5.318).
  Moving toward 4.853 → convergence artifact.  Staying near 5.318 → genuine
  formulation/BC difference.

Run (from repo root):
    conda run -n ganph --no-capture-output python 5_TAU/ysz_gap_diagnostic.py \\
        --vtk 5_TAU/reference_matlab/microstructure_real.vtk
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from reference_solver import compute_reference_tau
from read_vtk_volume import read_vtk_rectilinear_cell_data


# ── Task B: per-plane flux diagnostic ─────────────────────────────────────────

def task_b_plane_flux(volume: np.ndarray, spacing: tuple, phase_id: int = 2) -> dict:
    """
    For the given phase, solve the reference Laplacian and compute total z-flux
    at EVERY cut plane k→k+1.  Both cells in a pair must be in the percolating
    pathway (same mask as the inlet-face formula in reference_solver.py).

    Returns a dict with per-plane fluxes and summary statistics.
    """
    dx, dy, dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    print(f"\n  [Task B] Reference solver for phase {phase_id} (requesting field)...")
    res = compute_reference_tau(
        volume, phase_id, spacing, D0=1.0, verbose=True, return_field=True
    )

    if not res["percolating"]:
        print(f"  Phase {phase_id} is not percolating — cannot compute plane fluxes.")
        return {}

    C       = res["C"]        # (Z, Y, X), float64; NaN outside pathway
    pathway = res["pathway"]  # (Z, Y, X), bool
    nz      = C.shape[0]

    print(f"\n  tau_ref = {res['tau_eff']:.6f}  (inlet flux, L=(nz-1)*dz formula)")
    print(f"  relres  = {res['pcg_relres']:.2e}  (direct solve, essentially machine-exact)")

    # Flux at every cut plane k → k+1
    fluxes: list[float] = []
    for k in range(nz - 1):
        both_on = pathway[k, :, :] & pathway[k + 1, :, :]
        # One-sided finite difference: same formula as reference_solver.py inlet flux
        grad_c    = (C[k + 1, :, :] - C[k, :, :]) / dz
        flux_map  = -1.0 * grad_c * dx * dy   # D0=1
        plane_flux = float(np.nansum(flux_map[both_on]))
        fluxes.append(plane_flux)

    fluxes_arr = np.array(fluxes)
    inlet_flux = fluxes_arr[0]

    # Relative deviation of each plane w.r.t. the inlet plane
    rel_dev = (fluxes_arr - inlet_flux) / abs(inlet_flux)

    # Summary statistics
    rel_max = float(np.max(np.abs(rel_dev)))
    rel_std = float(np.std(rel_dev))

    # Does mean-over-planes change the effective tau?
    mean_flux = float(np.mean(fluxes_arr))
    L         = (nz - 1) * dz
    area_total = volume.shape[2] * dx * volume.shape[1] * dy  # nx*dx * ny*dy
    epsilon   = res["epsilon"]
    Deff_inlet = (inlet_flux / area_total) * L
    Deff_mean  = (mean_flux  / area_total) * L
    tau_inlet  = (epsilon / Deff_inlet) if Deff_inlet > 0 else math.inf
    tau_mean   = (epsilon / Deff_mean)  if Deff_mean  > 0 else math.inf

    # Conserved count: planes within 1e-8 * inlet_flux of the inlet (machine-precision test)
    n_conserved = int(np.sum(np.abs(rel_dev) < 1e-8))

    return {
        "fluxes":       fluxes_arr,
        "inlet_flux":   inlet_flux,
        "mean_flux":    mean_flux,
        "rel_dev":      rel_dev,
        "rel_max":      rel_max,
        "rel_std":      rel_std,
        "n_conserved":  n_conserved,
        "tau_inlet":    tau_inlet,
        "tau_mean":     tau_mean,
        "tau_ref":      res["tau_eff"],
        "n_planes":     nz - 1,
        "epsilon":      epsilon,
    }


def print_task_b(result: dict) -> None:
    if not result:
        return
    print()
    print("=" * 72)
    print(" TASK B — Interior-plane flux diagnostic (YSZ, VTK 100x100x50)")
    print("=" * 72)

    fluxes = result["fluxes"]
    rel_dev = result["rel_dev"]
    n = result["n_planes"]

    print(f"\n  Inlet-face (k=0→1) flux  : {result['inlet_flux']:.8f}")
    print(f"  Mean over all {n} planes  : {result['mean_flux']:.8f}")
    print(f"  Max |rel deviation|       : {result['rel_max']:.2e}")
    print(f"  Std of rel deviations     : {result['rel_std']:.2e}")
    print(f"  Planes within 1e-8 of inlet: {result['n_conserved']}/{n}")

    print()
    print("  Per-plane fluxes (k→k+1):")
    print(f"  {'k':>4}  {'flux':>14}  {'rel_dev':>12}  {'cells_both'}")
    print(f"  {'—'*4}  {'—'*14}  {'—'*12}  {'—'*10}")
    for k, (f, rd) in enumerate(zip(fluxes, rel_dev)):
        # mark inlet and outlet planes
        tag = " ← INLET" if k == 0 else (" ← OUTLET-1" if k == n - 1 else "")
        print(f"  {k:>4}  {f:>14.8f}  {rd:>+12.4e}{tag}")

    print()
    print("  tau from inlet flux  (current reference_solver formula) :",
          f"{result['tau_inlet']:.6f}")
    print("  tau from mean flux   (mean over all planes)             :",
          f"{result['tau_mean']:.6f}")
    print(f"  tau_ref (returned by reference_solver)                  : {result['tau_ref']:.6f}")

    if result["rel_max"] < 1e-6:
        print()
        print("  VERDICT: All planes agree to < 1e-6 (solver residual ~1e-14).")
        print("  Discrete Laplacian conservation is CONFIRMED.")
        print("  The 'one-sided inlet flux underestimates' hypothesis is FALSE.")
        print("  Inlet flux == mean flux == any cut-plane flux.")
        print("  The 9.6% gap is NOT attributable to flux measurement choice.")
    else:
        print()
        print(f"  VERDICT: Planes vary by up to {result['rel_max']*100:.3f}%.")
        print("  Flux is NOT conserved across planes to solver-residual precision.")
        delta_tau = result["tau_mean"] - result["tau_ref"]
        print(f"  Switching to mean flux changes tau by {delta_tau:+.4f} ({delta_tau/result['tau_ref']*100:+.2f}%).")


# ── Task C: taufactor convergence tightening ──────────────────────────────────

def task_c_taufactor_convergence(volume: np.ndarray, phase_id: int = 2) -> dict:
    """
    Run taufactor on the YSZ binary with default and tight convergence criteria.
    Returns a dict with tau values and iteration counts.
    """
    try:
        import taufactor as tauf
    except ImportError:
        print("  taufactor not available.")
        return {}

    binary = (volume == phase_id).astype(np.float32)

    print(f"\n  [Task C] taufactor default  (iter_limit=10000,  conv_crit=0.01)...")
    sol_def = tauf.Solver(binary, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol_def.solve(iter_limit=10000, conv_crit=0.01, verbose=False)
    tau_def  = float(np.asarray(sol_def.tau).flat[0])
    iter_def = int(sol_def.iter)
    conv_def = bool(sol_def.converged)

    # flux variation at default convergence (sanity check)
    fl_1d = sol_def.flux_1d[0]  # shape (Nx-1,) after solve
    fl_range_def = float((np.max(fl_1d) - np.min(fl_1d)) / np.max(fl_1d)) if np.max(fl_1d) > 0 else float("nan")

    print(f"  [Task C] taufactor strict   (iter_limit=100000, conv_crit=0.001)...")
    sol_str = tauf.Solver(binary, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol_str.solve(iter_limit=100000, conv_crit=0.001, verbose=False)
    tau_str  = float(np.asarray(sol_str.tau).flat[0])
    iter_str = int(sol_str.iter)
    conv_str = bool(sol_str.converged)

    fl_1d_str = sol_str.flux_1d[0]
    fl_range_str = float((np.max(fl_1d_str) - np.min(fl_1d_str)) / np.max(fl_1d_str)) if np.max(fl_1d_str) > 0 else float("nan")

    return {
        "tau_default": tau_def, "iter_default": iter_def, "conv_default": conv_def,
        "fl_range_default": fl_range_def,
        "tau_strict":  tau_str, "iter_strict":  iter_str, "conv_strict":  conv_str,
        "fl_range_strict": fl_range_str,
    }


def print_task_c(result: dict, tau_ref: float = 4.853) -> None:
    if not result:
        return
    print()
    print("=" * 72)
    print(" TASK C — taufactor convergence tightening (YSZ, VTK 100x100x50)")
    print("=" * 72)

    tau_def = result["tau_default"]
    tau_str = result["tau_strict"]

    print(f"\n  Default  (iter_limit=10000,  conv_crit=0.01 ): "
          f"tau={tau_def:.5f}  iters={result['iter_default']}  "
          f"converged={result['conv_default']}  "
          f"fl_range={result['fl_range_default']:.4f}")
    print(f"  Strict   (iter_limit=100000, conv_crit=0.001): "
          f"tau={tau_str:.5f}  iters={result['iter_strict']}  "
          f"converged={result['conv_strict']}  "
          f"fl_range={result['fl_range_strict']:.4f}")
    print(f"\n  Reference solver tau_ref    : {tau_ref:.5f}")
    print(f"  Default  gap vs ref         : {(tau_def-tau_ref)/tau_ref*100:+.2f}%")
    print(f"  Strict   gap vs ref         : {(tau_str-tau_ref)/tau_ref*100:+.2f}%")
    print(f"  Delta between default/strict: {(tau_str-tau_def)/tau_def*100:+.2f}%")

    if abs(tau_str - tau_def) / tau_def < 0.005:
        print()
        print("  VERDICT: Tightening convergence changes tau by <0.5%.")
        print("  Convergence artifact does NOT explain the 9.6% gap.")
        print("  The gap is a real formulation difference (BC convention).")
    elif abs(tau_str - tau_ref) / tau_ref < abs(tau_def - tau_ref) / tau_ref:
        print()
        moved = abs(tau_def - tau_str) / abs(tau_def - tau_ref) * 100
        print(f"  VERDICT: Tightening convergence moves tau {moved:.0f}% of the way toward tau_ref.")
        print("  Convergence artifact explains part of the 9.6% gap.")
    else:
        print()
        print("  VERDICT: Tightening convergence moves tau AWAY from tau_ref.")
        print("  The gap is a genuine formulation/BC difference, not a convergence artifact.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="YSZ gap diagnostic: Task B (plane fluxes) + Task C (taufactor convergence)."
    )
    parser.add_argument(
        "--vtk",
        default=str(_here / "reference_matlab" / "microstructure_real.vtk"),
        help="Path to VTK file",
    )
    parser.add_argument(
        "--phase-id", type=int, default=2,
        help="Phase ID to diagnose (default: 2 = YSZ)",
    )
    parser.add_argument("--skip-b", action="store_true", help="Skip Task B")
    parser.add_argument("--skip-c", action="store_true", help="Skip Task C")
    args = parser.parse_args()

    vtk_path = Path(args.vtk)
    if not vtk_path.exists():
        print(f"VTK not found: {vtk_path}")
        print("(5_TAU/reference_matlab/ is gitignored; copy the file locally.)")
        sys.exit(1)

    print()
    print("=" * 72)
    print(" 5_TAU/ysz_gap_diagnostic.py — YSZ 9.59% gap investigation")
    print("=" * 72)

    vtk_data = read_vtk_rectilinear_cell_data(vtk_path)
    volume   = vtk_data["volume"]
    spacing  = vtk_data["spacing"]

    b_result: dict = {}
    if not args.skip_b:
        b_result = task_b_plane_flux(volume, spacing, phase_id=args.phase_id)
        print_task_b(b_result)

    c_result: dict = {}
    tau_ref_for_c = b_result.get("tau_ref", 4.85258)
    if not args.skip_c:
        c_result = task_c_taufactor_convergence(volume, phase_id=args.phase_id)
        print_task_c(c_result, tau_ref=tau_ref_for_c)

    print()
    print("=" * 72)
    print(" COMBINED VERDICT")
    print("=" * 72)
    if b_result and c_result:
        b_conserved = b_result.get("rel_max", 1.0) < 1e-6
        c_artifact  = abs(c_result["tau_strict"] - c_result["tau_default"]) / c_result["tau_default"] >= 0.005
        if b_conserved and not c_artifact:
            print()
            print("  Flux is conserved across all planes (Task B): YES")
            print("  Convergence artifact explains the gap (Task C): NO")
            print()
            print("  CONCLUSION: The 9.6% gap is a genuine BC convention difference.")
            print("  Both solvers are correctly converged to their own Laplacian problem.")
            print("  The reference uses Dirichlet BCs at actual inlet/outlet cells;")
            print("  taufactor uses ghost-layer BCs one cell outside the domain.")
            print("  For well-connected phases (Ni, Pore) this difference is <0.5%.")
            print("  For YSZ (34 outlet vs 2683 inlet cells), the extra ghost-layer")
            print("  resistance at the narrow outlet amplifies the BC difference to 9.6%.")
        elif b_conserved and c_artifact:
            print()
            print("  Flux is conserved (Task B): YES — one-sided flux explanation ruled out.")
            print("  Convergence artifact explains part of the gap (Task C): YES")
            print("  Remaining gap after tight convergence is BC convention difference.")
        else:
            print("  (see individual task verdicts above)")
    print()


if __name__ == "__main__":
    main()
