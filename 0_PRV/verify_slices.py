"""
verify_slices.py
================
Validates the PNG stack produced by paraview_slice_export.py against the
ground-truth structure folder that made the test VTK.

Run with your normal ganph python (numpy + pillow only):

  python verify_slices.py --slices slices_out --original ../synthetic_data/structure_0001

Checks performed:
  1. COUNT       expected number of images exist (default 70)
  2. RESOLUTION  every image is the expected square size (default 500x500)
  3. COLORS      pixels are essentially only black/gray/white (+ background);
                 reports % of ambiguous (anti-aliased) pixels
  4. VOLUME FR.  per-phase volume fractions of the exported stack vs. the
                 original BMP structure (should agree within ~1-2%)
  5. ORIENTATION compares exported slices to ground-truth slices under
                 identity / left-right flip / up-down flip / both, and tells
                 you which transform matches -> tells you if the camera
                 direction in the ParaView script mirrors the data
  6. ORDER       confirms slice 0000 corresponds to the x~0 end (not reversed)

Exit code 0 = all good, 1 = something to fix before giving it to Juan.
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

LEVELS = np.array([0, 127, 255])
# Ground-truth BMPs use the GAN-PH convention: Pore=0, YSZ=127, Ni=255.
# Exported PNGs use Juan's rendering: Ni=black(0), YSZ=grey, Pore=white(255).
# So exported levels must be inverted (0<->255) before comparing to BMPs.
PHASE_NAMES = {0: "Pore", 127: "YSZ", 255: "Ni"}
INVERT = {0: 255, 127: 127, 255: 0}


def load_png_as_phases(path, color_tol=28):
    """Load an exported PNG, strip background, return (64-ish grid of gray levels, stats)."""
    rgb = np.array(Image.open(path).convert("RGB")).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chroma = rgb.max(-1) - rgb.min(-1)            # phases are neutral gray
    gray = rgb.mean(-1)
    dist = np.abs(gray[..., None] - LEVELS).min(-1)
    is_phase = (chroma <= 12) & (dist <= color_tol)

    if not is_phase.any():
        return None, None, 1.0
    rows = np.where(is_phase.any(axis=1))[0]
    cols = np.where(is_phase.any(axis=0))[0]
    crop_gray = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    crop_phase = is_phase[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    ambiguous = 1.0 - crop_phase.mean()           # anti-aliased / background inside bbox
    snapped = LEVELS[np.abs(crop_gray[..., None] - LEVELS).argmin(-1)]
    return snapped, crop_phase, ambiguous


def downsample_nearest(img, n):
    rows = (np.arange(n) + 0.5) / n * img.shape[0]
    cols = (np.arange(n) + 0.5) / n * img.shape[1]
    return img[rows.astype(int)][:, cols.astype(int)]


def load_original(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.bmp"))) or \
            sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        sys.exit("No ground-truth slices found in %s" % folder)
    vol = np.stack([np.array(Image.open(f).convert("L")) for f in files], 0)
    snapped = LEVELS[np.abs(vol[..., None].astype(int) - LEVELS).argmin(-1)]
    return snapped  # (n_slices, rows, cols)


TRANSFORMS = [
    ("identity",            lambda a: a),
    ("flip left-right",     lambda a: a[:, ::-1]),
    ("flip up-down",        lambda a: a[::-1, :]),
    ("rot180 (both flips)", lambda a: a[::-1, ::-1]),
    ("transpose",           lambda a: a.T),
    ("transpose+fliplr",    lambda a: a.T[:, ::-1]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True, help="Folder of exported PNGs")
    ap.add_argument("--original", required=True, help="Ground-truth structure folder")
    ap.add_argument("--frames", type=int, default=70)
    ap.add_argument("--resolution", type=int, default=500)
    ap.add_argument("--start", type=float, default=None,
                    help="First slice X used in export (default: auto, like the export script)")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--size", type=float, default=25.0,
                    help="Physical size used in make_test_vtk.py")
    ap.add_argument("--no-invert", action="store_true",
                    help="Skip the black<->white inversion. Use when --original is "
                         "another ParaView PNG export (e.g. Juan's old manual slices) "
                         "instead of GAN-PH convention BMPs.")
    args = ap.parse_args()

    ok = True
    pngs = sorted(glob.glob(os.path.join(args.slices, "*.png")))

    # 1 -- count
    print("[1] COUNT: found %d PNGs (expected %d)" % (len(pngs), args.frames))
    if len(pngs) != args.frames:
        ok = False

    # 2 -- resolution
    sizes = {Image.open(p).size for p in pngs[:5]} | {Image.open(pngs[-1]).size} if pngs else set()
    print("[2] RESOLUTION: %s (expected (%d, %d))" % (sizes, args.resolution, args.resolution))
    if sizes != {(args.resolution, args.resolution)}:
        ok = False

    original = load_original(args.original)      # (N, rows, cols)
    n_orig = original.shape[0]
    grid = original.shape[1]

    # Map exported frame index -> original slice index via physical positions
    step = args.size / args.frames
    start = args.start if args.start is not None else 0.5 * step
    end = args.end if args.end is not None else args.size - 0.5 * step
    voxel = args.size / n_orig
    positions = np.linspace(start, end, args.frames)
    orig_index = np.clip((positions / voxel).astype(int), 0, n_orig - 1)

    # 3 + 4 -- colors and volume fractions
    counts = {0: 0, 127: 0, 255: 0}
    ambig_total, frames_ok = [], 0
    exported = {}
    for i, p in enumerate(pngs):
        snapped, mask, ambiguous = load_png_as_phases(p)
        if snapped is None:
            print("    !! %s: no phase pixels found at all" % os.path.basename(p))
            ok = False
            continue
        ambig_total.append(ambiguous)
        small = downsample_nearest(snapped, grid)
        if not args.no_invert:   # exported black=Ni -> BMP convention Ni=255
            small = np.vectorize(INVERT.get)(small)
        exported[i] = small
        for lv in LEVELS:
            counts[lv] += int((small == lv).sum())
        frames_ok += 1

    total = sum(counts.values()) or 1
    print("[3] COLORS: avg %.2f%% ambiguous pixels per frame (anti-aliasing/background bleed)"
          % (100 * np.mean(ambig_total) if ambig_total else 100))
    if ambig_total and np.mean(ambig_total) > 0.05:
        print("    !! >5%% ambiguous -- check FXAA/background settings")
        ok = False

    print("[4] VOLUME FRACTIONS (exported vs ground truth):")
    truth_total = original.size
    worst = 0.0
    for lv in LEVELS:
        vf_exp = counts[lv] / total
        vf_tru = (original == lv).sum() / truth_total
        diff = abs(vf_exp - vf_tru)
        worst = max(worst, diff)
        flag = "" if diff < 0.02 else "  <-- !!"
        print("      %-12s exported %.4f | truth %.4f | diff %.4f%s"
              % (PHASE_NAMES[lv], vf_exp, vf_tru, diff, flag))
    if worst >= 0.02:
        ok = False

    # 5 -- orientation, using a few probe frames
    print("[5] ORIENTATION (which transform maps export -> ground truth):")
    probe_frames = [f for f in (args.frames // 4, args.frames // 2, 3 * args.frames // 4)
                    if f in exported]
    votes = {}
    for f in probe_frames:
        truth_slice = original[orig_index[f]]
        scores = {}
        for name, T in TRANSFORMS:
            t = T(exported[f])
            if t.shape != truth_slice.shape:
                continue
            scores[name] = (t == truth_slice).mean()
        best = max(scores, key=scores.get)
        votes[best] = votes.get(best, 0) + 1
        print("      frame %04d (orig slice %d): best = %-20s agreement %.1f%%"
              % (f, orig_index[f], best, 100 * scores[best]))
        if scores[best] < 0.90:
            print("      !! low agreement -- wrong slice axis, ordering, or geometry")
            ok = False
    if votes:
        winner = max(votes, key=votes.get)
        if winner != "identity":
            print("      NOTE: images are consistently '%s' relative to ground truth." % winner)
            print("      Not fatal -- but preprocess_dream3d.py must apply the same transform,")
            print("      or flip the camera in paraview_slice_export.py.")

    # 6 -- ordering (first frame should match the x~0 end better than the far end)
    if 0 in exported and probe_frames:
        first = exported[0]
        cands = {name: T(first) for name, T in TRANSFORMS if T(first).shape == original[0].shape}
        agree_lo = max((c == original[orig_index[0]]).mean() for c in cands.values())
        agree_hi = max((c == original[-1]).mean() for c in cands.values())
        print("[6] ORDER: frame 0 vs first slice %.1f%% | vs last slice %.1f%%"
              % (100 * agree_lo, 100 * agree_hi))
        if agree_hi > agree_lo:
            print("      !! stack appears REVERSED (frame 0 = far end)")
            ok = False

    print("\n%s" % ("ALL CHECKS PASSED -- safe to send to Juan." if ok
                    else "PROBLEMS FOUND -- fix before sending to Juan."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
