"""
5_TAU/validate_taufactor.py
============================
Analytic accuracy tests for the taufactor pipeline.

PRECISION vs ACCURACY distinction
----------------------------------
  Precision (prior label audit): strict re-solves reproduce tau_labels.csv values
    — tested by re-running compute_tau_labels.py on a subset and comparing.
  Accuracy (this suite): does the pipeline produce TRUE tau on structures
    whose ground-truth tau is analytically known?

Run from inside 5_TAU/ (or the repo root):
    conda run -n ganph --no-capture-output python 5_TAU/validate_taufactor.py

If any case FAILS, the 5_TAU pipeline's accuracy is suspect.
STOP and report before using tau_labels.csv or tau_net.pth for any analysis.

Five analytic cases
-------------------
  1. Fully dense 64^3 cube               -> tau = 1.000 (unimpeded straight diffusion)
  2. Straight square channels along z    -> tau = 1.000 (no tortuosity by construction)
  3. Solid wall perpendicular to z       -> NaN  (no percolating path)
  4. AXIS CHECK: channels along x not z -> NaN  (taufactor solves along z; catches wrong-axis bug)
  5. CONVENTION CHECK: same channels,
     Pore encoding (val=0) vs Ni (val=255) -> tau=1.0 for correct call;
                                             NaN for NI_VAL call on absent phase.
"""

import math
import sys
from pathlib import Path

import numpy as np

# Same sys.path injection as compute_tau_labels.py
# Both use analyze.py functions as the single pipeline entry point.
sys.path.insert(0, str(Path(__file__).parent.parent / "4_CNNCT"))
from analyze import (
    compute_tortuosity,
    compute_phase_connectivity,
    NI_VAL,
    YSZ_VAL,
    PORE_VAL,
)

DEVICE = "cpu"  # CPU-only on this machine; passed to taufactor.Solver


# ── Case helpers ──────────────────────────────────────────────────────────────

def case1_dense_cube():
    """
    Fully dense 64^3 single-phase cube.

    Every voxel is the conducting phase => no transport path is blocked.
    The Laplace solution is a uniform gradient across z => tau = 1.000.
    """
    vol = np.full((64, 64, 64), NI_VAL, dtype=np.uint8)
    result = compute_tortuosity(vol, NI_VAL)
    tol = 1e-3
    passed = (not math.isnan(result)) and abs(result - 1.0) < tol
    return passed, f"tau={result:.6f}  (expected 1.000 +/- {tol})"


def case2_straight_z_channels():
    """
    Straight square channels perfectly aligned with z (the solve axis).

    Channel pattern: Ni columns at stride-4 positions in (y, x), spanning
    all 64 z-slices. By construction these channels have zero tortuosity =>
    tau must equal 1.000.
    """
    vol = np.zeros((64, 64, 64), dtype=np.uint8)
    vol[:, ::4, ::4] = NI_VAL   # columns z=0..63, every 4th (y, x)

    result = compute_tortuosity(vol, NI_VAL)
    tol = 0.01
    passed = (not math.isnan(result)) and abs(result - 1.0) < tol
    return passed, f"tau={result:.6f}  (expected 1.000 +/- {tol})"


def case3_blocked_wall():
    """
    Solid Pore wall perpendicular to z at z=32.

    Ni exists at z=0..31 and z=33..63 but is fully severed at z=32.
    No Ni component can percolate from z=0-face to z=63-face.
    compute_tortuosity must return NaN via the connectivity check
    (analyze.py line: 'if conn == 0.0: return float("nan")').
    """
    vol = np.full((64, 64, 64), NI_VAL, dtype=np.uint8)
    vol[32, :, :] = PORE_VAL  # complete z=32 slab blocks percolation

    result = compute_tortuosity(vol, NI_VAL)
    passed = math.isnan(result)
    return passed, f"tau={result}  (expected NaN -- Pore wall at z=32 severs Ni)"


def case4_axis_check():
    """
    AXIS CHECK: channels aligned along x (axis 2), NOT along z (axis 0).

    These channels span the full x extent at fixed (z, y) positions.
    They would have tau=1.0 if solved along x, but taufactor solves along z.
    Along z these channels do not percolate (none spans z=0 to z=63), so
    the pipeline must return NaN.

    A non-NaN result here would indicate the pipeline is solving along
    the wrong axis -- a critical correctness failure.
    """
    vol = np.zeros((64, 64, 64), dtype=np.uint8)
    for z_pos in [16, 32, 48]:           # three z-levels (not spanning z=0..63)
        for y_pos in range(0, 64, 4):
            vol[z_pos, y_pos, :] = NI_VAL  # full x span, fixed z

    conn = compute_phase_connectivity(vol, NI_VAL)
    result = compute_tortuosity(vol, NI_VAL)
    passed = math.isnan(result)
    return passed, (
        f"conn_z={conn:.3f}, tau={result}  "
        f"(expected NaN -- channels run along x, not z)"
    )


def case5_convention_check():
    """
    CONVENTION CHECK: same straight-z channel geometry as case 2, but encoded
    with Pore (val=0) as the conducting phase.

    GAN/BMP convention: Ni=255, YSZ=127, Pore=0.
    The conducting channels are at PORE_VAL (0); the background is YSZ (127).

    Correct call -- compute_tortuosity(vol, PORE_VAL):
        Pipeline extracts binary=(vol==0), finds percolating columns, tau~=1.0.
        PASS: pipeline correctly identifies the conducting phase by its value.

    Inverted call -- compute_tortuosity(vol, NI_VAL):
        NI_VAL=255 does not exist in this volume => conn=0 => NaN.
        This documents the failure mode: if a caller assumes Ni=255 but the
        volume has Ni encoded as a different value, the pipeline returns NaN
        without raising an error. The GAN/BMP convention must match the call.

    Expected outcome: correct call PASS + inverted call gives NaN (expected behavior).
    The test FAILS only if the correct PORE_VAL call returns wrong tau -- that
    would mean the pipeline cannot handle a 0-valued conducting phase.
    """
    vol = np.full((64, 64, 64), YSZ_VAL, dtype=np.uint8)  # non-conducting background
    vol[:, ::4, ::4] = PORE_VAL  # conducting Pore channels along z

    tau_pore = compute_tortuosity(vol, PORE_VAL)
    tol = 0.01
    correct_ok = (not math.isnan(tau_pore)) and abs(tau_pore - 1.0) < tol

    tau_ni_wrong = compute_tortuosity(vol, NI_VAL)
    inverted_nan = math.isnan(tau_ni_wrong)

    passed = correct_ok
    return passed, (
        f"PORE_VAL call: tau={tau_pore:.6f} (expected ~1.000, {'PASS' if correct_ok else 'FAIL'}); "
        f"NI_VAL call (inverted): tau={tau_ni_wrong} "
        f"({'NaN as expected' if inverted_nan else 'WARNING: non-NaN -- inversion not caught'})"
    )


# ── Runner ────────────────────────────────────────────────────────────────────

CASES = [
    (1, "Dense 64^3 cube -> tau=1.000",                       case1_dense_cube),
    (2, "Straight z-channels -> tau=1.000",                   case2_straight_z_channels),
    (3, "Blocked wall at z=32 -> NaN",                        case3_blocked_wall),
    (4, "AXIS CHECK: x-channels -> NaN (solve axis is z)",    case4_axis_check),
    (5, "CONVENTION: Pore(val=0) channels -> tau=1.0 / NaN",  case5_convention_check),
]


def main():
    try:
        import taufactor  # noqa: F401
    except ImportError:
        print("ERROR: taufactor not installed. Run: pip install taufactor")
        sys.exit(1)

    print()
    print("=" * 72)
    print(" 5_TAU/validate_taufactor.py  --  Analytic accuracy validation")
    print()
    print(" PRECISION = re-solves reproduce labels (prior label audit)")
    print(" ACCURACY  = pipeline produces true tau on known structures (this)")
    print("=" * 72)

    results = []
    any_failed = False

    for case_id, description, fn in CASES:
        print(f"\nCase {case_id}: {description}")
        try:
            passed, detail = fn()
        except Exception as exc:
            passed = False
            detail = f"EXCEPTION: {exc}"
        status = "PASS" if passed else "FAIL"
        if not passed:
            any_failed = True
        results.append((case_id, description, status, detail))
        print(f"  [{status}] {detail}")

    print()
    print("=" * 72)
    print(f" Summary: {sum(1 for r in results if r[2] == 'PASS')}/{len(results)} PASS")
    print("=" * 72)
    print(f" {'Case':<6}  {'Status':<6}  Description")
    print(f" {'-'*6}  {'-'*6}  {'-'*50}")
    for case_id, desc, status, _ in results:
        marker = "OK" if status == "PASS" else "!!"
        print(f" [{marker}]  {case_id:<4}  {status:<6}  {desc}")
    print("=" * 72)

    if any_failed:
        print()
        print("FAILURE -- 5_TAU pipeline accuracy is suspect.")
        print("Do NOT use tau_labels.csv or tau_net.pth until root cause is resolved.")
        sys.exit(1)
    else:
        print()
        print("All cases PASSED.")
        print("Pipeline correctly reports tau=1.0 for ideal channels, NaN for blocked/")
        print("non-percolating structures, and handles the Pore(0) encoding.")
        print()
        print("Remaining accuracy gap: real-structure validation requires reference")
        print("values from Prof. Jin. See 5_TAU/reference_tau.csv (template) and")
        print("5_TAU/compare_reference_tau.py once values are available.")


if __name__ == "__main__":
    main()
