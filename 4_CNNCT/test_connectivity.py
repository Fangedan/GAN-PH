"""
test_connectivity.py
=====================
Standalone test suite for 4_CONNECTIVITY/analyze.py.

Follows the same pattern as test_preprocess.py — generates all synthetic
inputs internally, verifies outputs against exact expected values, and
requires no real DREAM.3D data.

Run with:
    python test_connectivity.py            # run all, clean up outputs
    python test_connectivity.py --keep     # run all, keep test volumes

Tests:
  1. Solid cube       – single phase fills the entire volume → connectivity = 1.0
  2. Half cube        – phase only in z=0..31, gap at z=32..63 → connectivity = 0.0
  3. Two pillars      – Ni and Pore pillars span full z, YSZ is isolated →
                        Ni=1.0, Pore=1.0, YSZ=0.0
  4. Three-phase slab – all three phases present as full-z columns →
                        all connectivity = 1.0, active_tpb > 0
  5. Disconnected Ni  – Ni voxels are isolated (checkerboard) →
                        Ni connectivity = 0.0
  6. S-value known    – two identical distributions → S ≈ 1.0;
                        maximally different → S ≈ low value
  7. Active TPB       – structure where Ni is disconnected: active_tpb < total_tpb
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Allow running from repo root OR from inside 4_CONNECTIVITY/
sys.path.insert(0, str(Path(__file__).parent))
try:
    from analyze import (
        compute_phase_connectivity,
        compute_tpb_densities,
        s_value,
        NI_VAL, YSZ_VAL, PORE_VAL,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / "4_CONNECTIVITY"))
    from analyze import (
        compute_phase_connectivity,
        compute_tpb_densities,
        s_value,
        NI_VAL, YSZ_VAL, PORE_VAL,
    )

# ── Test helpers ──────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def _check(test_name: str, condition: bool, msg: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ PASS  {test_name}")
        PASS += 1
    else:
        print(f"  ✗ FAIL  {test_name}" + (f" — {msg}" if msg else ""))
        FAIL += 1


def _approx_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _make_vol(fill_val: int = PORE_VAL) -> np.ndarray:
    """Return a 64x64x64 volume filled with fill_val."""
    return np.full((64, 64, 64), fill_val, dtype=np.uint8)


# ── Test 1: Solid single-phase cube ──────────────────────────────────────────

def test_solid_cube():
    """
    Volume = all Ni.
    Expected: Ni connectivity = 1.0, YSZ = 0.0, Pore = 0.0
    """
    print("\nTest 1: Solid single-phase cube (all Ni)")

    vol = _make_vol(NI_VAL)

    c_ni   = compute_phase_connectivity(vol, NI_VAL)
    c_ysz  = compute_phase_connectivity(vol, YSZ_VAL)
    c_pore = compute_phase_connectivity(vol, PORE_VAL)

    _check("Ni connectivity == 1.0",
           _approx_eq(c_ni, 1.0),
           f"got {c_ni}")
    _check("YSZ connectivity == 0.0 (absent)",
           _approx_eq(c_ysz, 0.0),
           f"got {c_ysz}")
    _check("Pore connectivity == 0.0 (absent)",
           _approx_eq(c_pore, 0.0),
           f"got {c_pore}")


# ── Test 2: Phase only in first half (no z-percolation) ──────────────────────

def test_half_cube_no_percolation():
    """
    Ni only in z=0..31 (first 32 slices), Pore everywhere else.
    Ni does NOT reach z=63 → connectivity = 0.0
    """
    print("\nTest 2: Ni half-cube (z=0..31 only) — should NOT percolate")

    vol = _make_vol(PORE_VAL)
    vol[:32, :, :] = NI_VAL   # Ni in first half only

    c_ni = compute_phase_connectivity(vol, NI_VAL)

    _check("Ni connectivity == 0.0 (doesn't reach z=63)",
           _approx_eq(c_ni, 0.0),
           f"got {c_ni}")

    # Pore spans z=32..63 but NOT z=0 → also 0.0
    c_pore = compute_phase_connectivity(vol, PORE_VAL)
    _check("Pore connectivity == 0.0 (doesn't reach z=0)",
           _approx_eq(c_pore, 0.0),
           f"got {c_pore}")


# ── Test 3: Two phases, one isolated ─────────────────────────────────────────

def test_two_percolating_one_isolated():
    """
    Ni columns in x=0..31 span full z.
    Pore fills x=33..63, full z.
    YSZ is a thin isolated slab at x=32, z=20..40 only (does NOT reach z=0 or z=63).

    Expected:
      Ni   connectivity = 1.0
      Pore connectivity = 1.0
      YSZ  connectivity = 0.0
    """
    print("\nTest 3: Two percolating phases, one isolated YSZ")

    vol = _make_vol(PORE_VAL)
    vol[:, :, :32] = NI_VAL                # Ni column, full z
    vol[20:41, :, 32] = YSZ_VAL            # YSZ slab, z=20..40 only
    # x=33..63 stays Pore (full z)

    c_ni   = compute_phase_connectivity(vol, NI_VAL)
    c_pore = compute_phase_connectivity(vol, PORE_VAL)
    c_ysz  = compute_phase_connectivity(vol, YSZ_VAL)

    _check("Ni connectivity == 1.0",
           _approx_eq(c_ni, 1.0),
           f"got {c_ni}")
    _check("Pore connectivity == 1.0",
           _approx_eq(c_pore, 1.0),
           f"got {c_pore}")
    _check("YSZ connectivity == 0.0 (isolated, doesn't span z)",
           _approx_eq(c_ysz, 0.0),
           f"got {c_ysz}")


# ── Test 4: All three phases percolating → active TPB > 0 ────────────────────

def test_three_phase_active_tpb():
    """
    Three vertical columns spanning full z=0..63:
      x=0..20:  Ni
      x=21..42: YSZ
      x=43..63: Pore

    All three phases percolate. Ni-YSZ and YSZ-Pore interfaces exist.
    But there is no location adjacent to ALL THREE simultaneously
    unless the column boundaries are adjacent — which they are at x=20/21
    and x=42/43, but not at the same location.

    So total_tpb > 0 requires a voxel adjacent to all three phases.
    Column boundary at x=21 is adjacent to Ni (x=20) and YSZ (x=21/22),
    but not Pore (nearest is x=43).
    → total_tpb should be 0 for this strict 3-column geometry.

    We use a different structure: alternating thin columns of width 1:
      Columns x = 0,3,6,... → Ni
      Columns x = 1,4,7,... → YSZ
      Columns x = 2,5,8,... → Pore

    Now every voxel is adjacent to all 3 phases → total_tpb should be
    close to 1 (normalized by volume). And since all phases percolate,
    active_tpb == total_tpb.
    """
    print("\nTest 4: Three-phase alternating columns — all percolate, active_tpb > 0")

    vol = np.zeros((64, 64, 64), dtype=np.uint8)
    for x in range(64):
        if x % 3 == 0:
            vol[:, :, x] = NI_VAL
        elif x % 3 == 1:
            vol[:, :, x] = YSZ_VAL
        else:
            vol[:, :, x] = PORE_VAL

    c_ni   = compute_phase_connectivity(vol, NI_VAL)
    c_ysz  = compute_phase_connectivity(vol, YSZ_VAL)
    c_pore = compute_phase_connectivity(vol, PORE_VAL)

    _check("Ni connectivity == 1.0 (full z columns)",
           _approx_eq(c_ni, 1.0),
           f"got {c_ni}")
    _check("YSZ connectivity == 1.0",
           _approx_eq(c_ysz, 1.0),
           f"got {c_ysz}")
    _check("Pore connectivity == 1.0",
           _approx_eq(c_pore, 1.0),
           f"got {c_pore}")

    total_tpb, active_tpb, active_frac = compute_tpb_densities(vol)

    _check("total_tpb > 0 (phases are adjacent)",
           total_tpb > 0,
           f"got {total_tpb:.4f}")
    _check("active_tpb > 0 (all phases percolate)",
           active_tpb > 0,
           f"got {active_tpb:.4f}")
    _check("active_tpb ≈ total_tpb (all phases percolate → all TPB active)",
           _approx_eq(active_frac, 1.0, tol=0.01),
           f"active_frac={active_frac:.4f}, "
           f"total={total_tpb:.4f}, active={active_tpb:.4f}")


# ── Test 5: Isolated (checkerboard) Ni → connectivity = 0.0 ──────────────────

def test_disconnected_ni():
    """
    Ni voxels placed only at positions where (x+y+z) is even — like a
    3D checkerboard. Each Ni voxel is surrounded on all 6 faces by non-Ni.
    → No Ni voxel is 6-connected to another Ni voxel.
    → Ni connectivity = 0.0

    Pore fills the rest (percolates).
    """
    print("\nTest 5: Disconnected (checkerboard) Ni — connectivity = 0.0")

    z, y, x = np.mgrid[0:64, 0:64, 0:64]
    vol = np.where((x + y + z) % 2 == 0, NI_VAL, PORE_VAL).astype(np.uint8)

    c_ni = compute_phase_connectivity(vol, NI_VAL)

    _check("Ni connectivity == 0.0 (fully isolated voxels)",
           _approx_eq(c_ni, 0.0),
           f"got {c_ni}")


# ── Test 6: Active TPB < Total TPB when Ni is disconnected ───────────────────

def test_active_tpb_less_than_total_when_ni_disconnected():
    """
    Structure: thin YSZ slab (full z) + thin Pore slab (full z) + isolated Ni.

    vol[:, :, 0..20]  = YSZ  (percolates)
    vol[:, :, 21..43] = Pore (percolates)
    vol[:, :, 44..63] = all Pore, but with isolated Ni voxels at x=44, even y/z

    YSZ-Pore interface exists → total_tpb > 0 at any column adjacent to Ni.
    But Ni is isolated → active_tpb should be lower than total_tpb.

    Actually let's make a cleaner test:
    - YSZ columns at x % 3 == 1 span full z
    - Pore fills x % 3 == 2, full z
    - Ni is ONLY at z=20..40, x % 3 == 0 (doesn't reach z=0 or z=63)

    → YSZ percolates, Pore percolates, Ni does NOT
    → total_tpb > 0 (three phases are adjacent at some voxels)
    → active_tpb should be 0 (Ni is never part of a percolating network
       so no TPB site has all three adjacent phases percolating)
    """
    print("\nTest 6: active_tpb = 0 when Ni is isolated (doesn't percolate)")

    vol = np.zeros((64, 64, 64), dtype=np.uint8)
    for x in range(64):
        if x % 3 == 0:
            # Ni: only in z=20..40, does NOT span z=0 to z=63
            vol[20:41, :, x] = NI_VAL
            # Rest of this column is Pore
            vol[:20, :, x] = PORE_VAL
            vol[41:, :, x] = PORE_VAL
        elif x % 3 == 1:
            vol[:, :, x] = YSZ_VAL    # full z
        else:
            vol[:, :, x] = PORE_VAL   # full z

    c_ni  = compute_phase_connectivity(vol, NI_VAL)
    c_ysz = compute_phase_connectivity(vol, YSZ_VAL)

    _check("Ni connectivity == 0.0 (doesn't span z)",
           _approx_eq(c_ni, 0.0),
           f"got {c_ni}")
    _check("YSZ connectivity == 1.0",
           _approx_eq(c_ysz, 1.0),
           f"got {c_ysz}")

    total_tpb, active_tpb, active_frac = compute_tpb_densities(vol)

    _check("total_tpb > 0 (Ni/YSZ/Pore are adjacent somewhere)",
           total_tpb > 0,
           f"got {total_tpb:.4f}")
    _check("active_tpb == 0.0 (Ni doesn't percolate → no active TPB)",
           _approx_eq(active_tpb, 0.0, tol=1e-9),
           f"got {active_tpb:.6f}")
    _check("active_tpb_frac == 0.0",
           _approx_eq(active_frac, 0.0, tol=1e-9),
           f"got {active_frac:.6f}")


# ── Test 7: S-value edge cases ────────────────────────────────────────────────

def test_s_value():
    """
    S-value (Yu et al. 2025):
      - Two identical distributions → S ≈ 1.0
      - Distribution of all 0.9 vs all 0.1 → S should be < 0.70
    """
    print("\nTest 7: S-value computation")

    rng = np.random.default_rng(42)

    # Identical distributions → S = 1.0
    d = rng.uniform(0.2, 0.8, size=50)
    s_identical = s_value(d, d)
    _check("S-value(d, d) ≈ 1.0 for identical distributions",
           _approx_eq(s_identical, 1.0, tol=1e-6),
           f"got {s_identical:.6f}")

    # Maximally separated: all-low vs all-high → S should be well below 0.70
    low  = np.full(50, 0.05)
    high = np.full(50, 0.95)
    s_far = s_value(low, high)
    _check("S-value(low, high) < 0.70 (maximally different)",
           s_far < 0.70,
           f"got {s_far:.4f}")

    # Nearby distributions → S should be >= 0.85
    d1 = rng.normal(0.5, 0.05, size=100)
    d2 = rng.normal(0.51, 0.05, size=100)   # tiny mean shift
    s_near = s_value(d1, d2)
    _check("S-value(similar distributions) >= 0.85",
           s_near >= 0.85,
           f"got {s_near:.4f}")


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    parser = argparse.ArgumentParser(
        description="Test suite for 4_CONNECTIVITY/analyze.py"
    )
    parser.add_argument("--keep", action="store_true",
                        help="Keep any temporary outputs after tests complete")
    args = parser.parse_args()

    print("=" * 62)
    print(" GAN-PH Connectivity Analysis — Test Suite")
    print("=" * 62)

    test_solid_cube()
    test_half_cube_no_percolation()
    test_two_percolating_one_isolated()
    test_three_phase_active_tpb()
    test_disconnected_ni()
    test_active_tpb_less_than_total_when_ni_disconnected()
    test_s_value()

    print(f"\n{'='*62}")
    print(f" Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
    print(f"{'='*62}")

    if FAIL > 0:
        print("\n Some tests FAILED. Check output above for details.")
        sys.exit(1)
    else:
        print("\n All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
