# 0_PRV — VTK → slice-image export

This stage runs **before** `preprocess_dream3d.py`, and it runs on **Juan's
machine**, not in the `ganph` conda env. It automates the manual ParaView
workflow described in *"Steps to generate microstructure slices using
ParaView 6.1.1"*: open a microstructure VTK volume, color the three phases
black/grey/white, sweep a slice plane along X, and save one PNG per slice.

```
DREAM.3D / other tool          this folder                  rest of pipeline
        │                          │                              │
   structure.vtk ──► paraview_slice_export.py ──► PNG stack ──► preprocess_dream3d.py ──► ...
```

## Files

| File | Runs with | Purpose |
|------|-----------|---------|
| `paraview_slice_export.py` | `pvpython` (ParaView 6.1.1) | **The deliverable.** Exports a VTK volume as 64 × 512×512 borderless PNG slices (one per voxel, 8 px/voxel). Only file Juan needs (plus the .bat). |
| `run_slice_export.bat` | Windows (drag-and-drop) | Wrapper so Juan can drag a `.vtk` file (or folder of them) onto it instead of typing commands. Must sit next to the .py file. |
| `make_test_vtk.py` | `ganph` python | Test harness: converts one of our BMP structures into a DREAM.3D-style test VTK with known ground truth. |
| `verify_slices.py` | `ganph` python | Test harness: validates an exported PNG stack (count, resolution, colors, volume fractions, orientation, slice order). |

## Phase conventions (IMPORTANT — read before "fixing" anything)

Two different conventions coexist by design:

| Phase | VTK value (Juan/DREAM.3D) | Exported PNG color | GAN-PH BMP gray |
|-------|---------------------------|--------------------|-----------------|
| Ni    | 1 | **black (0)**   | **255 (white)** |
| YSZ   | 2 | grey (127)      | 127             |
| Pore  | 3 | **white (255)** | **0 (black)**   |

Exported PNGs are **black↔white inverted** relative to our BMPs.
`preprocess_dream3d.py` performs that inversion (black→Ni→255). Do NOT
change the colors in `paraview_slice_export.py` — they replicate Juan's
manual coloring (1-Black, 2-Grey, 3-White) and the whole chain was verified
end-to-end this way. `verify_slices.py` inverts automatically when comparing
against BMP ground truth; pass `--no-invert` when comparing PNG-to-PNG
(e.g. against Juan's old manual exports).

## Juan's quick start (no pipeline / conda needed)

1. Put `paraview_slice_export.py` and `run_slice_export.bat` in the same
   folder anywhere on the PC.
2. If ParaView is not at `C:\Program Files\ParaView 6.1.1\` or
   `C:\ParaView\`, open the .bat in Notepad and fix the `PVPYTHON` path.
3. **Drag a `.vtk` file onto `run_slice_export.bat`.** Slices appear in a
   new folder `<vtkname>_slices` next to the VTK. Dragging a folder of
   `.vtk` files batches all of them.
4. Zip the output folder(s) and send them over.

Equivalent command line, with all defaults shown:

```
"C:\Program Files\ParaView 6.1.1\bin\pvpython.exe" paraview_slice_export.py ^
    m_1.vtk -o m_1_slices --frames 64 --resolution 512 --array Phases
```

`--start` / `--end` default to half a slice-step inside the volume bounds
(reproduces the 0.2 → 24.8 keyframes from the manual workflow for a 0–25 µm
volume). A deprecation WARN about ".%04d printf format" during save is
harmless.

## Testing (our side) — performed & passing on ParaView 6.1.1, 2026-06

```bash
# 1. fabricate a test VTK from a structure we already know the answer to
python make_test_vtk.py ..\synthetic_data\structure_0001 -o test_structure.vtk

# 2. run the export exactly the way Juan will
C:\ParaView\bin\pvpython.exe paraview_slice_export.py test_structure.vtk -o test_slices

# 3. verify the PNGs against ground truth (exit code 0 = safe to ship)
python verify_slices.py --slices test_slices --original ..\synthetic_data\structure_0001

# 4. end-to-end: preprocess must report VFs matching step 1's printout
python ..\preprocess_dream3d.py --input test_slices --dry-run --border 0
```

When Juan re-exports the structure he originally sent (same VTK), close the
loop with:

```bash
python verify_slices.py --slices <his_new_output> --original <his_old_png_folder> --no-invert
```

Expected result: identity, ~100% agreement.

## Known quirks (verified, not bugs)

- **Border:** script exports are **borderless** (camera parallel scale pinned
  to data extent) → use `--border 0` in `preprocess_dream3d.py`. Juan's old
  manual exports have a background border (default 24). Never trust a fixed
  border on screenshot-style exports: plain ResetCamera framing drifts by a
  few px between runs, and background pixels (gray ≈94) silently snap to YSZ
  in threshold_to_phases.
- **Up-down flip:** exported images are vertically mirrored vs the raw voxel
  array (image row 0 = top, ParaView +Z = up). Juan's manual exports have the
  same property; preprocess was built around it. Consistent mirror — all
  pipeline metrics (VF, SSA, connectivity, TPB, tortuosity) are invariant.
- **ParaView API:** `Hide3DWidgets` was renamed `HideInteractiveWidgets` in
  6.x; the script handles both. Tested against 6.1.1 specifically — retest
  before assuming other versions work.
- **Frames = 64, not 70:** Juan's manual process used 70 frames, but 70
  frames over a 64-voxel volume means preprocess (which cubes the first 64
  slices) silently drops the last ~9% of the structure and double-samples a
  few slices — measured as a ~2.6% YSZ bias on test data. 64 frames = exact
  one-per-voxel sampling. Use `--frames 70` only to replicate Juan's manual
  output for comparison. NOTE: real_data built from Juan's manual 70-slice
  export carries this artifact; regenerate it from his VTK before retraining.
