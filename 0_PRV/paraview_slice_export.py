"""
paraview_slice_export.py
========================
Automates Juan's manual ParaView 6.1.1 workflow for turning a microstructure
VTK volume (from DREAM.3D or similar) into a stack of slice PNG images,
ready for preprocess_dream3d.py in the GAN-PH pipeline.

Replicates, step for step, the process in
"Steps to generate microstructure slices using ParaView 6.1.1":

  1.  Open VTK file, Apply
  2.  Color by 'Phases', representation Outline -> Surface
  3.  Color map: "Interpret Values As Categories", phases 1/2/3 ->
      Black / Grey / White
  4.  Add Slice filter: Origin X = 0, Normal = +X, Show Plane disabled
  5.  Camera Parallel Projection enabled
  6.  View from +X, then "Zoom Closest To Data"
  7.  Color legend hidden, Light Kit disabled, Orientation Axes hidden
  8.  Animation track on Slice1 "Slice Offset Values" (ContourValues),
      Ramp keyframes: t=0 -> START (Juan: 0.2), t=1 -> END (Juan: 24.8)
  9.  Number of frames (default 64 = one per voxel; Juan used 70 manually,
      which over-samples a 64-voxel volume and truncates downstream)
  10. File > Save Animation (512x512 borderless; Juan's manual was 500x500
      with a background border -- exact framing supersedes that)

USAGE (must be run with ParaView's pvpython, NOT regular python):

  Windows:
    "C:\\Program Files\\ParaView 6.1.1\\bin\\pvpython.exe" paraview_slice_export.py m_1.vtk -o slices_out

  Reproduce Juan's exact numbers explicitly:
    pvpython paraview_slice_export.py m_1.vtk -o slices_out --start 0.2 --end 24.8 --frames 70

  By default --start/--end are computed automatically from the data bounds
  (first/last slice positions are half a slice-step inside the volume, which
  is exactly how Juan's 0.2 and 24.8 arise for a 0-25 micron volume at 70
  frames). --frames defaults to 70.

  Batch a whole folder of VTK files:
    pvpython paraview_slice_export.py vtk_folder/ -o all_slices

Output: <outdir>/<name>.0000.png ... <name>.0069.png  (500x500 by default)
"""

import argparse
import os
import sys
import glob

from paraview.simple import (  # noqa: pvpython-only import
    OpenDataFile, Show, Hide, Render, Slice, ColorBy,
    GetActiveViewOrCreate, GetColorTransferFunction, GetAnimationScene,
    GetAnimationTrack, CompositeKeyFrame, SaveAnimation, Delete,
    ResetCamera, UpdatePipeline,
)
import paraview.simple as pvs


def hide_slice_widget(slc):
    """'Show Plane' off. The function name changed across ParaView versions:
    Hide3DWidgets (<= 5.12) -> HideInteractiveWidgets (5.13 / 6.x)."""
    fn = getattr(pvs, "HideInteractiveWidgets", None) or \
         getattr(pvs, "Hide3DWidgets", None)
    if fn is None:
        return  # widgets aren't rendered in offscreen pvpython anyway
    try:
        fn(proxy=slc.SliceType)
    except Exception:
        try:
            fn(slc.SliceType)
        except Exception:
            pass  # cosmetic only -- never fail the export over this


# Phase value -> color, matching Juan's manual coloring (1-Black, 2-Grey, 3-White).
# In Juan's DREAM.3D data phase 1 = Ni, so exported PNGs have black=Ni and
# white=Pore -- inverted vs the GAN-PH BMP grayscale. preprocess_dream3d.py
# performs that inversion; do NOT change these colors to "fix" it.
PHASE_COLORS = [
    (0.0, 0.0, 0.0),                       # Black  (Pore)
    (127.0 / 255, 127.0 / 255, 127.0 / 255),  # Grey   (YSZ)
    (1.0, 1.0, 1.0),                       # White  (Ni)
]


def find_phase_array(reader, preferred="Phases"):
    """Return (association, name, (min, max)) of the phase array."""
    for assoc, info in (("CELLS", reader.GetCellDataInformation()),
                        ("POINTS", reader.GetPointDataInformation())):
        arr = info.GetArray(preferred)
        if arr is not None:
            return assoc, preferred, arr.GetComponentRange(0)
    # Fallback: first available array (cell data preferred, like DREAM.3D exports)
    for assoc, info in (("CELLS", reader.GetCellDataInformation()),
                        ("POINTS", reader.GetPointDataInformation())):
        if info.GetNumberOfArrays() > 0:
            arr = info.GetArray(0)
            print("WARNING: array '%s' not found, using '%s' (%s) instead"
                  % (preferred, arr.GetName(), assoc))
            return assoc, arr.GetName(), arr.GetComponentRange(0)
    raise RuntimeError("No data arrays found in the VTK file.")


def setup_categorical_lut(array_name, value_range):
    """'Interpret Values As Categories' with Black/Grey/White per phase."""
    lo, hi = int(round(value_range[0])), int(round(value_range[1]))
    values = list(range(lo, hi + 1))
    if len(values) != 3:
        print("WARNING: expected 3 phase values, found %d (%s). "
              "Coloring will spread Black->White evenly." % (len(values), values))

    lut = GetColorTransferFunction(array_name)
    lut.InterpretValuesAsCategories = 1

    annotations = []
    colors = []
    n = max(len(values) - 1, 1)
    for i, v in enumerate(values):
        annotations += [str(v), str(v)]
        if len(values) == 3:
            c = PHASE_COLORS[i]
        else:  # even grayscale ramp fallback
            g = float(i) / n
            c = (g, g, g)
        colors += list(c)
    lut.Annotations = annotations
    lut.IndexedColors = colors
    lut.NanColor = [1.0, 0.0, 1.0]  # magenta = something went wrong, easy to spot
    return lut


def look_down_positive_x(view):
    """Equivalent of the GUI '+X' camera button, then 'Zoom Closest To Data'."""
    try:
        pvs.ResetActiveCameraToPositiveX()
    except AttributeError:
        # Manual fallback for older APIs
        fp = view.CameraFocalPoint[:]
        view.CameraPosition = [fp[0] + 1.0, fp[1], fp[2]]
        view.CameraFocalPoint = fp
        view.CameraViewUp = [0.0, 0.0, 1.0]
    # "Zoom Closest To Data" (tight fit). Fall back to a normal reset.
    try:
        view.ResetCamera(True)
    except TypeError:
        ResetCamera(view)


def process_one(vtk_path, out_dir, args):
    # paraview.simple auto-resets the camera on the FIRST Render() call,
    # silently discarding any camera state set before it. Disable that, and
    # belt-and-suspenders: re-assert the framing after the first render too.
    try:
        pvs._DisableFirstRenderCameraReset()
    except Exception:
        pass
    name = args.name or os.path.splitext(os.path.basename(vtk_path))[0]
    print("=== %s -> %s/ (%d frames) ===" % (vtk_path, out_dir, args.frames))

    # --- Step 1: open VTK file + Apply ---------------------------------
    reader = OpenDataFile(vtk_path)
    if reader is None:
        raise RuntimeError("ParaView could not open: %s" % vtk_path)
    UpdatePipeline(proxy=reader)

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [args.resolution, args.resolution]

    # --- Steps 2-3: Surface representation, color by Phases, categories
    assoc, array_name, rng = find_phase_array(reader, args.array)
    setup_categorical_lut(array_name, rng)

    bounds = reader.GetDataInformation().GetBounds()
    xmin, xmax = bounds[0], bounds[1]
    ycenter = 0.5 * (bounds[2] + bounds[3])
    zcenter = 0.5 * (bounds[4] + bounds[5])

    # --- Step 4: Slice filter, Origin X=0, Normal=+X, no plane widget --
    slc = Slice(Input=reader)
    slc.SliceType = "Plane"
    slc.SliceType.Origin = [0.0, ycenter, zcenter]
    slc.SliceType.Normal = [1.0, 0.0, 0.0]
    slc.SliceOffsetValues = [args.start if args.start is not None else xmin]

    Hide(reader, view)
    disp = Show(slc, view)
    disp.Representation = "Surface"
    ColorBy(disp, (assoc, array_name))
    disp.SetScalarBarVisibility(view, False)        # step 7: hide color legend
    hide_slice_widget(slc)                          # "Show Plane" disabled

    # --- Steps 5-7: camera + render cleanup ----------------------------
    view.CameraParallelProjection = 1
    view.OrientationAxesVisibility = 0              # hide orientation axes
    view.UseLight = 0                               # disable light kit
    try:
        view.UseFXAA = 0                            # keep phase edges crisp
    except AttributeError:
        pass
    # Disable multisample anti-aliasing if the API allows it: with exact-fit
    # framing, cell boundaries align to pixel boundaries and MSAA paints
    # blended artifact lines exactly there. (Downstream samplers must use
    # block CENTERS regardless -- see preprocess resize_slice / verifier.)
    for attempt in ("MultiSamples",):
        try:
            setattr(view, attempt, 0)
            break
        except Exception:
            pass
    if args.background:
        view.UseColorPaletteForBackground = 0
        view.BackgroundColorMode = "Single Color"
        view.Background = args.background

    look_down_positive_x(view)
    # Exact-fit framing: parallel scale = half the data's vertical extent, so
    # the slice fills the viewport edge to edge. No background border, and at
    # --resolution 512 each voxel maps to exactly 8x8 px. This removes the
    # sub-pixel framing nondeterminism of plain ResetCamera (measured: content
    # box drifted between 354 and 360 px across runs), which leaked background
    # into downstream fixed-border cropping.
    half = 0.5 * max(bounds[3] - bounds[2], bounds[5] - bounds[4])
    view.CameraParallelScale = half
    Render(view)
    # Re-assert after the first render in case an automatic reset fired anyway
    view.CameraParallelScale = half
    Render(view)

    # --- Steps 8-9: animate Slice Offset Values (ContourValues) --------
    # Default start/end: half a slice-step inside the volume on each side,
    # which reproduces Juan's 0.2 / 24.8 for a 0-25 um volume at 70 frames.
    step = (xmax - xmin) / float(args.frames)
    start = args.start if args.start is not None else xmin + 0.5 * step
    end = args.end if args.end is not None else xmax - 0.5 * step
    print("  slicing X from %.4f to %.4f in %d frames" % (start, end, args.frames))

    scene = GetAnimationScene()
    scene.PlayMode = "Sequence"
    scene.StartTime = 0.0
    scene.EndTime = 1.0
    scene.NumberOfFrames = args.frames

    track = GetAnimationTrack("ContourValues", index=-1, proxy=slc)
    kf0 = CompositeKeyFrame(KeyTime=0.0, KeyValues=[start], Interpolation="Ramp")
    kf1 = CompositeKeyFrame(KeyTime=1.0, KeyValues=[end])
    track.KeyFrames = [kf0, kf1]

    # --- Step 10: Save Animation as PNG sequence -----------------------
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_pattern = os.path.join(out_dir, name + ".png")
    SaveAnimation(out_pattern, view,
                  ImageResolution=[args.resolution, args.resolution],
                  FrameWindow=[0, args.frames - 1],
                  SuffixFormat=".%04d")
    print("  saved %d images: %s" % (args.frames,
          os.path.join(out_dir, name + ".0000.png ...")))

    # Clean up so batch mode starts fresh for the next file
    Delete(track)
    Delete(kf0)
    Delete(kf1)
    Delete(slc)
    Delete(reader)


def main():
    p = argparse.ArgumentParser(
        description="Automate ParaView slice-image export for GAN-PH "
                    "(run with pvpython, not python).")
    p.add_argument("input", help="VTK file, or a folder containing .vtk files")
    p.add_argument("-o", "--output", default="slices",
                   help="Output folder (default: ./slices). In batch mode a "
                        "subfolder is created per VTK file.")
    p.add_argument("--name", default=None,
                   help="Image filename prefix (default: VTK filename)")
    p.add_argument("--frames", type=int, default=64,
                   help="Number of slices/images to generate (default: 64 = one per "
                        "voxel of a 64^3 structure; Juan's manual process used 70)")
    p.add_argument("--start", type=float, default=None,
                   help="First slice X position (default: auto from bounds; "
                        "Juan used 0.2)")
    p.add_argument("--end", type=float, default=None,
                   help="Last slice X position (default: auto from bounds; "
                        "Juan used 24.8)")
    p.add_argument("--resolution", type=int, default=512,
                   help="Square image resolution in pixels (default: 512 = exactly "
                        "8 px per voxel of a 64^3 structure, borderless)")
    p.add_argument("--array", default="Phases",
                   help="Name of the phase array (default: Phases)")
    p.add_argument("--background", type=float, nargs=3, default=None,
                   metavar=("R", "G", "B"),
                   help="Force a solid background color, 0-1 floats "
                        "(default: keep ParaView's default)")
    args = p.parse_args()

    if os.path.isdir(args.input):
        vtks = sorted(glob.glob(os.path.join(args.input, "*.vtk")))
        if not vtks:
            sys.exit("No .vtk files found in %s" % args.input)
        for v in vtks:
            sub = os.path.join(args.output,
                               os.path.splitext(os.path.basename(v))[0])
            process_one(v, sub, args)
    else:
        process_one(args.input, args.output, args)

    print("Done.")


if __name__ == "__main__":
    main()
