"""
5_TAU/cross_validate_tau.py
============================
Cross-validate the reference MATLAB-port solver against taufactor.

Validation structure
--------------------
Section A — Analytic cases (both solvers, known ground truth)
  A1. Dense 10x10x10 cube          → tau = 1.000 (both)
  A2. Straight z-channels           → tau = 1.000 (both)
  A3. Phase severed at z=5 (wall)  → non-percolating (Inf / NaN)

Section B — Real VTK volume (reference_matlab/microstructure_real.vtk)
  For phases 1, 2, 3:
    reference_solver  → tau_ref
    taufactor          → tau_tf
  Verdict per phase: MATCH (<=2%), CLOSE (2-10%), DIVERGE (>10%)
  Note: checks both epsilon convention and Laplacian solve agreement.

Section C — 64x64x64 sub-array check
  Documents whether the VTK volume can supply 64^3 sub-cubes for
  comparison with tau_labels.csv (answer: NO — Z-dimension = 50 < 64).

Section D — tau_labels.csv context
  If tau_labels.csv exists locally, shows where the VTK reference
  solver values fall within the tau distribution from 64^3 training crops.

Verdict thresholds (Section B):
  |tau_ref - tau_tf| / tau_ref <= 0.02  →  MATCH   (algorithmic agreement)
  0.02 < ratio <= 0.10                  →  CLOSE   (minor formulation difference)
  ratio > 0.10                          →  DIVERGE (investigate before trusting labels)

Usage (run from repo root):
    conda run -n ganph --no-capture-output python 5_TAU/cross_validate_tau.py \\
        --vtk 5_TAU/reference_matlab/microstructure_real.vtk \\
        --phase-ids 1 2 3

Output: 5_TAU/reference_tau.csv  (reference_solver values for all phases)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import numpy as np

# Path setup: allow importing from sibling directories
_here = Path(__file__).parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / "4_CNNCT"))

from reference_solver import compute_reference_tau
from read_vtk_volume import read_vtk_rectilinear_cell_data, report as vtk_report

import matplotlib
matplotlib.use("Agg")


# ── taufactor availability ─────────────────────────────────────────────────────

def _taufactor_solve(binary: np.ndarray) -> float:
    """
    Run taufactor on a float32 binary (0/1) volume.
    Solves along axis=0 (z-direction).
    Returns tau, or np.inf if no percolating path, or np.nan on error.
    """
    try:
        import taufactor as tauf
    except ImportError:
        return float("nan")

    binary_f = binary.astype(np.float32)
    # Quick percolation screen: any connected path z=0 to z=-1?
    from scipy.ndimage import label as nd_label
    struct = np.zeros((3, 3, 3), dtype=bool)
    for d in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]:
        struct[d] = True
    labeled, _ = nd_label(binary_f > 0.5, structure=struct)
    percolates = False
    for lbl in np.unique(labeled):
        if lbl == 0:
            continue
        comp = labeled == lbl
        if comp[0,:,:].any() and comp[-1,:,:].any():
            percolates = True
            break
    if not percolates:
        return np.inf

    try:
        solver = tauf.Solver(binary_f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver.solve()
        val = float(np.asarray(solver.tau).flat[0])
        return val if math.isfinite(val) else np.inf
    except Exception as exc:
        warnings.warn(f"taufactor solve failed: {exc}")
        return float("nan")


def _verdict(tau_ref: float, tau_tf: float) -> str:
    """Classify agreement between two tau values."""
    if not math.isfinite(tau_ref) or not math.isfinite(tau_tf):
        if (not math.isfinite(tau_ref)) and (not math.isfinite(tau_tf)):
            return "NON-PERC (both)"
        return f"MISMATCH (ref={'Inf' if not math.isfinite(tau_ref) else f'{tau_ref:.3f}'}, tf={'Inf' if not math.isfinite(tau_tf) else f'{tau_tf:.3f}'})"
    rel = abs(tau_ref - tau_tf) / (tau_ref + 1e-300)
    if rel <= 0.02:
        return f"MATCH   ({rel*100:.2f}%)"
    if rel <= 0.10:
        return f"CLOSE   ({rel*100:.2f}%)"
    return f"DIVERGE ({rel*100:.2f}%)"


# ── Section A: Analytic cases ─────────────────────────────────────────────────

def analytic_cases(verbose: bool = False) -> list[dict]:
    """Run analytic validation cases on both solvers."""
    spacing = (0.1, 0.1, 0.1)  # isotropic 0.1 µm — arbitrary, cancels in tau

    results = []

    # A1: Dense 10x10x10 cube, all phase 1
    vol1 = np.ones((10, 10, 10), dtype=np.int32)
    res_ref = compute_reference_tau(vol1, 1, spacing, verbose=verbose)
    bin1 = (vol1 == 1).astype(np.float32)
    tau_tf = _taufactor_solve(bin1)
    tau_ref = res_ref["tau_eff"]
    results.append({
        "label": "A1 dense 10x10x10 cube",
        "expected": "tau=1.000 (both)",
        "tau_ref": tau_ref, "tau_tf": tau_tf,
        "pass_ref": math.isfinite(tau_ref) and abs(tau_ref - 1.0) < 0.01,
        "pass_tf":  math.isfinite(tau_tf)  and abs(tau_tf  - 1.0) < 0.01,
        "verdict": _verdict(tau_ref, tau_tf),
    })

    # A2: Straight z-channels in 10x10x10 (every 3rd x,y position)
    vol2 = np.zeros((10, 10, 10), dtype=np.int32)
    vol2[:, ::3, ::3] = 1   # columns spanning all z
    res_ref = compute_reference_tau(vol2, 1, spacing, verbose=verbose)
    bin2 = (vol2 == 1).astype(np.float32)
    tau_tf = _taufactor_solve(bin2)
    tau_ref = res_ref["tau_eff"]
    results.append({
        "label": "A2 straight z-channels (stride 3)",
        "expected": "tau=1.000 (both)",
        "tau_ref": tau_ref, "tau_tf": tau_tf,
        "pass_ref": math.isfinite(tau_ref) and abs(tau_ref - 1.0) < 0.01,
        "pass_tf":  math.isfinite(tau_tf)  and abs(tau_tf  - 1.0) < 0.01,
        "verdict": _verdict(tau_ref, tau_tf),
    })

    # A3: Wall severing phase 1 at z=5 — no percolation
    vol3 = np.ones((10, 10, 10), dtype=np.int32)
    vol3[5, :, :] = 2   # blocking layer of phase 2
    res_ref = compute_reference_tau(vol3, 1, spacing, verbose=verbose)
    bin3 = (vol3 == 1).astype(np.float32)
    tau_tf = _taufactor_solve(bin3)
    tau_ref = res_ref["tau_eff"]
    ref_ok = not math.isfinite(tau_ref)   # expect Inf
    tf_ok  = not math.isfinite(tau_tf)    # expect Inf (or NaN)
    results.append({
        "label": "A3 wall at z=5, no percolation",
        "expected": "non-percolating (both)",
        "tau_ref": tau_ref, "tau_tf": tau_tf,
        "pass_ref": ref_ok,
        "pass_tf":  tf_ok,
        "verdict": "NON-PERC (both)" if (ref_ok and tf_ok) else f"FAIL ref={'Inf' if ref_ok else f'{tau_ref:.3f}'} tf={'non-perc' if tf_ok else f'{tau_tf:.3f}'}",
    })

    return results


# ── Section B: Real VTK volume ─────────────────────────────────────────────────

def vtk_comparison(vtk_path: Path, phase_ids: list[int], verbose: bool = False) -> list[dict]:
    """Run both solvers on real VTK microstructure and compare."""
    vtk_data = read_vtk_rectilinear_cell_data(vtk_path)
    vtk_report(vtk_data, path=vtk_path)

    vol     = vtk_data["volume"]
    spacing = vtk_data["spacing"]

    results = []
    for pid in phase_ids:
        print(f"\n  Phase {pid} — reference solver...")
        res_ref = compute_reference_tau(vol, pid, spacing, verbose=verbose)

        binary = (vol == pid).astype(np.float32)
        print(f"  Phase {pid} — taufactor...")
        tau_tf = _taufactor_solve(binary)

        tau_ref = res_ref["tau_eff"]
        results.append({
            "phase_id":     pid,
            "epsilon":      res_ref["epsilon"],
            "epsilon_perc": res_ref["epsilon_perc"],
            "n_nodes":      res_ref["n_nodes"],
            "tau_ref":      tau_ref,
            "tau_tf":       tau_tf,
            "pcg_flag":     res_ref["pcg_flag"],
            "pcg_relres":   res_ref["pcg_relres"],
            "percolating":  res_ref["percolating"],
            "verdict":      _verdict(tau_ref, tau_tf),
            "dims":         vtk_data["cell_dims"],
            "spacing":      spacing,
        })

    return results


# ── Section C: 64x64x64 sub-array check ──────────────────────────────────────

def subarray_check(vtk_results: list[dict]) -> None:
    """Report whether the VTK volume supports 64x64x64 sub-array extraction."""
    if not vtk_results:
        return
    nx_c, ny_c, nz_c = vtk_results[0]["dims"]
    print()
    print("Section C — 64^3 sub-array check")
    print(f"  VTK cell dimensions: {nx_c} x {ny_c} x {nz_c}  (X x Y x Z)")
    min_dim = min(nx_c, ny_c, nz_c)
    if min_dim >= 64:
        n = ((nx_c-64)//64+1) * ((ny_c-64)//64+1) * ((nz_c-64)//64+1)
        print(f"  Min dimension {min_dim} >= 64 — can extract ~{n} non-overlapping 64^3 sub-cubes.")
        print("  Run sub-array validation by extracting these cubes and comparing both solvers.")
    else:
        print(f"  Min dimension {min_dim} < 64 — NO 64^3 sub-cubes fit.")
        print(f"  Z = {nz_c} < 64: the VTK volume is not the parent of the 64^3 anode training crops.")
        print("  The reference_solver and taufactor are compared on the full 100x100x50 volume only.")


# ── Section D: tau_labels.csv context ─────────────────────────────────────────

def tau_labels_context(vtk_results: list[dict], labels_path: Path) -> None:
    """
    Show where the VTK reference-solver tau values fall within the
    tau distribution from 64^3 training crops in tau_labels.csv.

    Phase ID mapping assumed (per CLAUDE.md): 1=Ni, 2=YSZ, 3=Pore.
    Note: the VTK is the FULL 100x100x50 volume; tau_labels.csv has 64^3 sub-cubes.
    Differences in tau between the full volume and sub-cubes are expected.
    """
    phase_map = {1: "Ni", 2: "YSZ", 3: "Pore"}

    if not labels_path.exists():
        print()
        print("Section D — tau_labels.csv context")
        print(f"  {labels_path} not found (gitignored; run compute_tau_labels.py locally).")
        print("  Cannot compare VTK reference-solver values to 64^3 crop distribution.")
        return

    # Load tau_labels.csv
    dist: dict[str, list[float]] = {"Ni": [], "YSZ": [], "Pore": []}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for ph in ("Ni", "YSZ", "Pore"):
                raw = row.get(f"tau_{ph}", "").strip()
                if raw and raw.lower() not in ("nan", ""):
                    try:
                        dist[ph].append(float(raw))
                    except ValueError:
                        pass

    print()
    print("Section D — tau_labels.csv context (64^3 crop distribution vs VTK full volume)")
    print(
        f"  {'Phase':<6}  {'N crops':>7}  {'mean':>7}  {'std':>6}  {'min':>7}  {'max':>7}  "
        f"{'VTK ref-solver':>16}  {'in range?':>10}"
    )
    print("  " + "-" * 80)

    for res in vtk_results:
        pid = res["phase_id"]
        ph  = phase_map.get(pid, f"Phase{pid}")
        tau_r = res["tau_ref"]
        vals = dist.get(ph, [])
        if not vals:
            in_range = "N/A"
            mu_str = std_str = mn_str = mx_str = "N/A"
            n_str = "0"
        else:
            mu = np.mean(vals)
            sd = np.std(vals)
            mn = np.min(vals)
            mx = np.max(vals)
            mu_str = f"{mu:.3f}"
            std_str = f"{sd:.3f}"
            mn_str = f"{mn:.3f}"
            mx_str = f"{mx:.3f}"
            n_str = str(len(vals))
            if math.isfinite(tau_r):
                in_range = "YES" if mn <= tau_r <= mx else "OUTSIDE"
            else:
                in_range = "N/A (non-perc)"
        tau_r_str = f"{tau_r:.4f}" if math.isfinite(tau_r) else "Inf"
        print(
            f"  {ph:<6}  {n_str:>7}  {mu_str:>7}  {std_str:>6}  {mn_str:>7}  {mx_str:>7}  "
            f"{tau_r_str:>16}  {in_range:>10}"
        )


# ── Write reference_tau.csv ───────────────────────────────────────────────────

def write_reference_csv(vtk_results: list[dict], out_path: Path) -> None:
    """
    Write reference_tau.csv with the MATLAB-port solver values for all phases.
    Fills the file that compare_reference_tau.py reads for pipeline accuracy checks.
    """
    phase_map = {1: "Ni", 2: "YSZ", 3: "Pore"}
    fieldnames = ["structure_id", "phase", "tau_reference", "source_note"]

    with open(out_path, "w", newline="") as f:
        f.write("# Reference tortuosity values from MATLAB-port solver (reference_solver.py)\n")
        f.write("# Source: 5_TAU/reference_matlab/microstructure_real.vtk (gitignored)\n")
        f.write(
            "# Phase ID mapping: 1=Ni, 2=YSZ, 3=Pore  (matches anode eff_tor_diff.m driver)\n"
        )
        f.write("#\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in vtk_results:
            pid = res["phase_id"]
            ph  = phase_map.get(pid, f"phase{pid}")
            tau_r = res["tau_ref"]
            tau_str = f"{tau_r:.6f}" if math.isfinite(tau_r) else "nan"
            writer.writerow({
                "structure_id": "microstructure_real_vtk",
                "phase":        ph,
                "tau_reference": tau_str,
                "source_note":  (
                    f"MATLAB-port solver; vtk 100x100x50 cells; "
                    f"pcg_flag={res['pcg_flag']}; relres={res['pcg_relres']:.2e}"
                ),
            })
    print(f"\n  Wrote {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate reference MATLAB-port solver against taufactor."
    )
    parser.add_argument(
        "--vtk",
        default=str(_here / "reference_matlab" / "microstructure_real.vtk"),
        help="Path to VTK file (default: 5_TAU/reference_matlab/microstructure_real.vtk)",
    )
    parser.add_argument(
        "--phase-ids", nargs="+", type=int, default=[1, 2, 3],
        help="Phase IDs in VTK to evaluate (default: 1 2 3)"
    )
    parser.add_argument(
        "--labels",
        default=str(_here / "tau_labels.csv"),
        help="Path to tau_labels.csv (default: 5_TAU/tau_labels.csv)",
    )
    parser.add_argument(
        "--out",
        default=str(_here / "reference_tau.csv"),
        help="Output path for reference_tau.csv (default: 5_TAU/reference_tau.csv)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    vtk_path    = Path(args.vtk)
    labels_path = Path(args.labels)
    out_path    = Path(args.out)

    # ── Header ──
    print()
    print("=" * 72)
    print(" 5_TAU/cross_validate_tau.py  --  MATLAB-port vs taufactor")
    print("=" * 72)

    # ── Section A: Analytic cases ──
    print()
    print("Section A — Analytic cases")
    print(
        f"  {'Case':<35}  {'tau_ref':>9}  {'tau_tf':>9}  "
        f"{'ref':>5}  {'tf':>5}  {'verdict'}"
    )
    print("  " + "-" * 80)
    a_results = analytic_cases(verbose=args.verbose)
    all_analytic_pass = True
    for r in a_results:
        tr = f"{r['tau_ref']:.5f}" if math.isfinite(r['tau_ref']) else "Inf"
        tt = f"{r['tau_tf']:.5f}"  if math.isfinite(r['tau_tf'])  else ("Inf" if math.isinf(r['tau_tf']) else "nan")
        ref_mark = "PASS" if r["pass_ref"] else "FAIL"
        tf_mark  = "PASS" if r["pass_tf"]  else "FAIL"
        if not r["pass_ref"] or not r["pass_tf"]:
            all_analytic_pass = False
        print(f"  {r['label']:<35}  {tr:>9}  {tt:>9}  {ref_mark:>5}  {tf_mark:>5}  {r['verdict']}")
    print()
    if all_analytic_pass:
        print("  All analytic cases PASSED.")
    else:
        print("  WARNING: One or more analytic cases FAILED.")
        print("  Investigate before trusting Section B results.")

    # ── Section B: VTK comparison ──
    print()
    print("Section B — VTK volume comparison")
    if not vtk_path.exists():
        print(f"  VTK not found: {vtk_path}")
        print("  (5_TAU/reference_matlab/ is gitignored; copy the file locally.)")
        vtk_results = []
    else:
        vtk_results = vtk_comparison(vtk_path, args.phase_ids, verbose=args.verbose)

        print()
        print("  Results summary:")
        print(
            f"  {'Phase':>5}  {'epsilon':>8}  {'eps_perc':>9}  {'tau_ref':>10}  "
            f"{'tau_tf':>10}  {'CG':>4}  {'relres':>8}  verdict"
        )
        print("  " + "-" * 85)
        for r in vtk_results:
            tr = f"{r['tau_ref']:.5f}"  if math.isfinite(r['tau_ref']) else "Inf"
            tt = f"{r['tau_tf']:.5f}"   if math.isfinite(r['tau_tf'])  else ("Inf" if math.isinf(r['tau_tf']) else "nan")
            print(
                f"  {r['phase_id']:5d}  {r['epsilon']:8.5f}  {r['epsilon_perc']:9.5f}  "
                f"{tr:>10}  {tt:>10}  {r['pcg_flag']:4d}  {r['pcg_relres']:8.2e}  {r['verdict']}"
            )

        # Write reference_tau.csv
        write_reference_csv(vtk_results, out_path)

    # ── Section C: Sub-array check ──
    subarray_check(vtk_results)

    # ── Section D: tau_labels.csv context ──
    tau_labels_context(vtk_results, labels_path)

    # ── Summary ──
    print()
    print("=" * 72)
    if vtk_results:
        verdicts = [r["verdict"] for r in vtk_results]
        n_match  = sum(1 for v in verdicts if v.startswith("MATCH"))
        n_close  = sum(1 for v in verdicts if v.startswith("CLOSE"))
        n_div    = sum(1 for v in verdicts if v.startswith("DIVERGE"))
        n_np     = sum(1 for v in verdicts if "NON-PERC" in v or "MISMATCH" in v)
        print(
            f" Section B: {n_match} MATCH, {n_close} CLOSE, {n_div} DIVERGE, "
            f"{n_np} non-percolating"
        )
        if n_div > 0:
            print(" WARNING: DIVERGE phases — check epsilon convention and Laplacian build.")
    print(f" Analytic: {'ALL PASS' if all_analytic_pass else 'SOME FAIL'}")
    print(f" reference_tau.csv: {out_path}")
    print(" VALIDATION.md: update with these results after review.")
    print("=" * 72)


if __name__ == "__main__":
    main()
