"""
test_preprocess.py
==================
Standalone test harness for preprocess_dream3d.py.
Covers four scenarios without requiring real data:

  1. RESIZE mode  — small stack  (50 slices, 500x500 px → 1 structure)
  2. TILE-XY mode — large stack  (64 slices, 500x500 px → 49 structures)
  3. MULTI mode   — two stacks   (separate folders → sequential numbering)
  4. PHASE CHECK  — verifies phase values survive the full round-trip

Run from your GAN-PH root:
  conda activate ganph
  python test_preprocess.py

Outputs land in ./test_outputs/ (auto-deleted at the end unless --keep is passed).

Usage:
  python test_preprocess.py           # run all tests, delete outputs
  python test_preprocess.py --keep    # run all tests, KEEP outputs for inspection
"""

import argparse
import shutil
import subprocess
import sys
import numpy as np
import cv2
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────────

BASE = Path("./test_outputs")

PASS = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"


def header(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check(condition: bool, msg: str):
    icon = PASS if condition else FAIL
    print(f"  {icon}  {msg}")
    if not condition:
        raise AssertionError(f"FAILED: {msg}")


def make_fake_dream3d_slices(dest: Path, n_slices: int, slice_size: int,
                              border: int = 24, phases: tuple = (0.40, 0.30, 0.30)):
    """
    Generate n_slices fake DREAM.3D PNGs in dest/.
    Phases: (ni_fraction, ysz_fraction, pore_fraction) — must sum to 1.
    Phase mapping (DREAM.3D convention): 0=Ni, 127=YSZ, 255=Pore.
    The border pixels are set to 0 (black).
    """
    dest.mkdir(parents=True, exist_ok=True)
    ni_f, ysz_f, _ = phases
    content_size = slice_size - 2 * border

    rng = np.random.default_rng(seed=42)
    for i in range(n_slices):
        # Draw random phase assignment for each pixel in content area
        flat = rng.random(content_size * content_size)
        content = np.where(flat < ni_f, 0,
                  np.where(flat < ni_f + ysz_f, 127, 255)).astype(np.uint8)
        content = content.reshape(content_size, content_size)

        img = np.zeros((slice_size, slice_size), dtype=np.uint8)
        img[border:border+content_size, border:border+content_size] = content

        cv2.imwrite(str(dest / f"Slice{i:04d}.png"), img)


def count_structures(output_dir: Path) -> int:
    return len(list(output_dir.glob("structure_*")))


def count_results_rows(output_dir: Path) -> int:
    dat = output_dir / "results.dat"
    if not dat.exists():
        return 0
    lines = dat.read_text().strip().splitlines()
    return len(lines) - 1  # subtract header row


def run_preprocess(args_list: list) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "preprocess_dream3d.py"] + args_list
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  STDOUT:", result.stdout[-1000:])
        print("  STDERR:", result.stderr[-1000:])
    return result


# ── Test 1: Resize mode (small stack) ─────────────────────────────────────────

def test_resize_mode():
    header("Test 1 — RESIZE mode (50 slices, 500x500 px → 1 structure)")

    src = BASE / "t1_input"
    out = BASE / "t1_output"

    print("  Generating 50 fake DREAM.3D slices (500x500 px)...")
    make_fake_dream3d_slices(src, n_slices=50, slice_size=500, border=24)

    r = run_preprocess(["--input", str(src), "--output", str(out),
                        "--border", "24", "--preview"])

    check(r.returncode == 0, "Script exited successfully")

    n_structs = count_structures(out)
    n_rows    = count_results_rows(out)

    # 50 slices < 64 → padded to 64 → 1 z-slab; XY resized to 64 → 1 structure
    check(n_structs == 1, f"Expected 1 structure, got {n_structs}")
    check(n_rows == 1,    f"results.dat has {n_rows} row(s), expected 1")

    # Verify BMP count inside the structure folder
    bmp_count = len(list((out / "structure_0001").glob("*.bmp")))
    check(bmp_count == 64, f"Expected 64 BMP slices, got {bmp_count}")

    check((out / "preview.png").exists(), "preview.png was created")

    print("  Test 1 PASSED")


# ── Test 2: Tile-XY mode (large stack) ────────────────────────────────────────

def test_tile_xy_mode():
    header("Test 2 — TILE-XY mode (64 slices, 500x500 px → ~49 structures)")

    src = BASE / "t2_input"
    out = BASE / "t2_output"

    print("  Generating 64 fake DREAM.3D slices (500x500 px)...")
    make_fake_dream3d_slices(src, n_slices=64, slice_size=500, border=24)

    r = run_preprocess(["--input", str(src), "--output", str(out),
                        "--border", "24", "--tile-xy"])

    check(r.returncode == 0, "Script exited successfully")

    n_structs = count_structures(out)
    n_rows    = count_results_rows(out)

    # After crop: 500 - 2*24 = 452 px.  floor(452/64) = 7.  7x7 = 49 XY tiles.
    # 64 z-slices → 1 z-slab.  Total = 49.
    expected = 49
    check(n_structs == expected, f"Expected {expected} structures, got {n_structs}")
    check(n_rows == expected,    f"results.dat has {n_rows} row(s), expected {expected}")

    print(f"  Test 2 PASSED  ({n_structs} structures)")


# ── Test 3: Multi mode ────────────────────────────────────────────────────────

def test_multi_mode():
    header("Test 3 — MULTI mode (2 folders, 50 slices each → 2 structures total)")

    parent = BASE / "t3_parent"
    out    = BASE / "t3_output"

    print("  Generating 2 sub-folders of fake slices...")
    for name in ("stack_A", "stack_B"):
        make_fake_dream3d_slices(parent / name, n_slices=50, slice_size=500, border=24)

    r = run_preprocess(["--input", str(parent), "--output", str(out),
                        "--border", "24", "--multi"])

    check(r.returncode == 0, "Script exited successfully")

    n_structs = count_structures(out)
    n_rows    = count_results_rows(out)

    # 2 folders × 1 structure each = 2
    check(n_structs == 2, f"Expected 2 structures, got {n_structs}")
    check(n_rows == 2,    f"results.dat has {n_rows} row(s), expected 2")

    # Structures must be numbered sequentially: 0001, 0002 (not two 0001s)
    names = sorted([p.name for p in out.glob("structure_*")])
    check(names == ["structure_0001", "structure_0002"],
          f"Sequential numbering correct: {names}")

    print("  Test 3 PASSED")


# ── Test 4: Phase round-trip check ────────────────────────────────────────────

def test_phase_roundtrip():
    header("Test 4 — Phase round-trip (DREAM.3D → preprocess → BMP)")

    src = BASE / "t4_input"
    out = BASE / "t4_output"

    print("  Creating single-phase slices for exact pixel verification...")

    # Create a slice that is 100% Ni in DREAM.3D space (pixel=0)
    # After remap it must be 100% 255 in load.py space
    src.mkdir(parents=True, exist_ok=True)
    content_size = 500 - 2*24   # = 452

    # Slice 0: pure Ni   (DREAM.3D pixel = 0)
    img0 = np.zeros((500, 500), dtype=np.uint8)
    cv2.imwrite(str(src / "Slice0000.png"), img0)

    # Slice 1: pure YSZ  (DREAM.3D pixel = 127)
    img1 = np.full((500, 500), 127, dtype=np.uint8)
    cv2.imwrite(str(src / "Slice0001.png"), img1)

    # Pad to 64 slices by repeating
    for i in range(2, 64):
        cv2.imwrite(str(src / f"Slice{i:04d}.png"),
                    img0 if i % 2 == 0 else img1)

    r = run_preprocess(["--input", str(src), "--output", str(out), "--border", "24"])
    check(r.returncode == 0, "Script exited successfully")

    # Load the output BMP for slice 0 and verify pixels
    bmp_path = out / "structure_0001" / "slice_0000.bmp"
    check(bmp_path.exists(), f"Output BMP exists: {bmp_path}")

    bmp = cv2.imread(str(bmp_path), cv2.IMREAD_GRAYSCALE)
    unique_vals = np.unique(bmp)

    # Slice 0 was pure Ni (DREAM3D=0), after remap should be 255
    check(set(unique_vals) == {255},
          f"Slice 0 (pure Ni) remapped to 255 only — unique vals: {unique_vals}")

    print("  Test 4 PASSED")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test harness for preprocess_dream3d.py")
    parser.add_argument("--keep", action="store_true",
                        help="Keep test output files after tests complete")
    args = parser.parse_args()

    print("=" * 60)
    print("  preprocess_dream3d.py  —  Test Suite")
    print("=" * 60)

    BASE.mkdir(parents=True, exist_ok=True)

    passed = []
    failed = []

    for name, fn in [
        ("resize_mode",     test_resize_mode),
        ("tile_xy_mode",    test_tile_xy_mode),
        ("multi_mode",      test_multi_mode),
        ("phase_roundtrip", test_phase_roundtrip),
    ]:
        try:
            fn()
            passed.append(name)
        except Exception as e:
            print(f"\n  {FAIL}  {name} FAILED: {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    print(f"  Results:  {len(passed)} passed  /  {len(failed)} failed")
    if failed:
        print(f"  Failed:   {', '.join(failed)}")
    print("=" * 60)

    # ── Cleanup guide ─────────────────────────────────────────────────────────
    if not args.keep:
        print(f"\nCleaning up {BASE} ...")
        shutil.rmtree(BASE, ignore_errors=True)
        print("  Deleted test_outputs/")
    else:
        print(f"\nTest outputs kept at: {BASE.resolve()}")
        print()
        print("Manual cleanup commands (Windows CMD):")
        print(f"  rmdir /s /q {BASE}")
        print()
        print("PowerShell / bash equivalent:")
        print(f"  Remove-Item -Recurse -Force {BASE}")
        print(f"  rm -rf {BASE}")
        print()
        print("What's inside test_outputs/:")
        print("  t1_output/  — resize-mode run (1 structure)")
        print("  t2_output/  — tile-XY run     (49 structures)")
        print("  t3_output/  — multi-mode run  (2 structures)")
        print("  t4_output/  — phase check     (1 structure)")
        print()
        print("To delete only intermediate input data and keep outputs:")
        print("  del /s /q test_outputs\\t*_input   (CMD)")
        print("  rm -rf test_outputs/t*_input       (bash)")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
