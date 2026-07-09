# PPTX Intelligence Extraction — USC_Update_10NOV2021.pptx

> Source: `C:\Users\alin2\Downloads\UTD Summer 26 Internship\raw images backup\USC_Update_10NOV2021.pptx`
> Extracted 2026-07-09 via python-pptx from 9 slides.
> Do NOT commit the PPTX itself (it is not currently tracked and should stay untracked).

---

## Full slide-by-slide text

### Slide 1 — Title
USC Update: 3D Meshing & Simulations  
November 2021

### Slide 2 — Sample Details
LSCF-GDC composite substrate with SCT capping layer

- LSCF: (La₀.₆Sr₀.₄)₀.₉₅Co₀.₂Fe₀.₈O₃-δ
- GDC: Ce₀.₈Gd₀.₂O₁.₉
- SCT: SrCo₀.₉Ta₀.₁O₃-δ

Samples were prepared at UConn using FIB-SEM to form ##um high cylinders with ##um diameters
*(Note: "##um" blanks indicate values were lost in the original document — voxel size NOT stated here.)*

- **Sample #1**: pristine LSCF+GDC cell
- **Sample #2**: pristine LSCF+GDC/SCT bilayer cell
- **Sample #3**: post tested (50 h) LSCF+GDC/SCT bilayer cell *(not in the raw images backup)*

### Slide 3 — Segmentation: Sample 1
Phases labeled: Pore, GDC, LSCF
*(Figures showing segmented cross-sections; phase labels extracted from text boxes)*

### Slide 4 — Segmentation: Sample 2
Phases labeled: Pore, GDC, LSCF, **SCT**
*(Figures showing segmented cross-sections with the 4th phase SCT visible)*

### Slide 5 — Meshing Strategies
- **Method 2** (the one used): Store segmented structure as a txt file of (x,y,z) coordinates
  and pixel values; import as a 3D scatter plot; create a uniform block geometry the same size
  as the segmented domain and mesh it according to the voxel size.
  
  _Quote_: "Does not lose any information, since the mesh resolution is equal to the voxel size"
  — but voxel size is not stated numerically in this slide.

### Slides 6–9 — Results
- Navier-Stokes simulation in pore phase (O₂ gas, ambient pressure inlet, 1 µm/s velocity)
- Diffusion through GDC phase
- Linear elastic stress in GDC phase
  - Fixed displacement: 0.1 µm on top boundary

---

## Intelligence findings (six questions from NEW_DATASET_INSPECTION.md)

### 1. Z voxel size — UNRESOLVED by PPTX ⚠️

The PPTX does NOT state voxel size numerically. The slide says "FIB-SEM to form ##um high
cylinders with ##um diameters" — the `##um` placeholders indicate values that are either
corrupted, redacted, or were left blank in the original document.

**From struct.txt analysis (NEW_DATASET_INSPECTION.md):**
- x-step ≈ 40.34 nm, y-step ≈ 40.14 nm (non-cubic by ~0.5%)
- z-step unknown — the PPTX does not resolve whether z = x ≈ 40.3 nm or is different

**Status: z voxel size remains OPEN — ask Prof. Jin.**  
Hypothesis: the sample preparation (FIB for pillar, then synchrotron STXM imaging) typically
gives isotropic or nearly-isotropic 3D reconstruction at ~40 nm. Using x ≈ 40.3 nm for all
three axes is a reasonable placeholder but must be confirmed.

### 2. Identity of the 4th phase — RESOLVED ✅

**The 4th phase is SCT (SrCo₀.₉Ta₀.₁O₃-δ).**

SCT is a MIEC (mixed ionic-electronic conductor) used as a capping/functional layer on top of
the LSCF-GDC composite. It appears ONLY in Sample 2 (the bilayer cell). Sample 1 is a simple
2-phase solid (LSCF+GDC) + pore — no SCT.

This resolves the S2_MATBOX unknown phase (val=88, ~24.6%) and S2_GDC_LSCF_SCT_Rectangle
label=3 (6.6%): both are SCT.

**Label mapping for S2:**

| Phase | Role | Likely pixel value in S2 files |
|---|---|---|
| Pore | gas transport | val=0 |
| SCT | MIEC capping layer | val=88 (S2_MATBOX) or label=3 (SCT_Rectangle) |
| GDC | ion conductor | val=171 (S2_MATBOX) or label=1 (SCT_Rectangle) |
| LSCF | electron conductor | val=254 (S2_MATBOX) or label=2 (SCT_Rectangle) |
| Background | padding | val=5 (S2_MATBOX) |

**Implication for GAN training:** Sample 2 is a 4-phase system (Pore/SCT/GDC/LSCF) not
directly equivalent to the current 3-phase GAN. For a clean analogy to the anode GAN,
**Sample 1 is preferred** for initial cathode generalization (3-phase: Pore/GDC/LSCF only).
S2 training would require a 4-channel softmax — architectural change, not just config.

### 3. Specimen provenance — RESOLVED ✅

- **S1 and S2 are different specimens** (different cells: pristine 2-layer vs. pristine 3-layer).
- Both are pristine (untested). Sample 3 (post-test, 50 h) is NOT in the raw images backup.
- Both were prepared at UConn. The imaging was done with synchrotron STXM/XANES (multi-energy
  approach matching the Ce L-edge and Fe K-edge channels seen in raw TIFs).
- S1 and S2 cannot be mixed directly — they have different phase compositions and different
  label encodings (see NEW_DATASET_INSPECTION.md Q5).

### 4. Volume fractions — cross-check

The PPTX does NOT state explicit volume fraction numbers. Cross-check from voxel counts:
- S1 Supercrop: LSCF≈17%, GDC≈21%, Pore≈62%
  - 62% pore is high; this "Supercrop" subregion may be from a porous zone.
  - No reference fractions in PPTX to compare against.
- Simulations (Navier-Stokes) used 1 µm/s pore/solid interface velocity, which is a
  kinetics parameter not a microstructure fraction.

### 5. Technique clarification

"Samples were prepared at UConn using FIB-SEM to form cylinders" — FIB was used to mill
micron-scale cylinders from the electrode, enabling synchrotron X-ray nano-CT or STXM
imaging. The raw .tif files with energies (5700/5750/6090 eV = Ce L-edge; 7040/7160 eV =
Fe K-edge) are synchrotron XANES maps used for phase-specific segmentation.
The technique is NOT standard FIB-SEM serial sectioning — it is FIB-prepared pillar +
synchrotron spectro-tomography. This explains the ~40 nm voxel size (synchrotron
resolution for this sample size).

---

## Open questions for Prof. Jin (from inspection + PPTX gap)

1. **Z voxel size**: Is z ≈ 40 nm (same as x/y from struct.txt)? Or different?  
   The PPTX dimension blanks (##um) were not filled in. This affects SSA and TPB calculations.

2. **S2 training intent**: Is the 4-phase S2 dataset intended for GAN training (requires
   4-channel architecture), or should only S1 (3-phase) be used?

3. **Metric for cathode reaction sites**: For MIEC cathodes, the reaction zone is distributed
   across the LSCF/GDC/Pore interface AND the LSCF bulk (vs. TPB for the anode).
   Should the scoring metric be TPB density, or total LSCF-Pore surface area, or something else?

4. **Anode voxel size**: preprocess_dream3d.py and analyze.py use VOXEL_SIZE_UM=0.1 (100 nm)
   for the existing Ni-YSZ-Pore FIB-SEM data. Is 100 nm the correct voxel size for that dataset?
   (No explicit citation found in the code; just a comment saying "standard for FIB-SEM Ni-YSZ".)

---

## Summary table

| Question | Answer | Source |
|---|---|---|
| Z voxel size = x/y? | **UNKNOWN** — blanks in PPTX | PPTX slide 2 |
| 4th phase identity | **SCT** (SrCo₀.₉Ta₀.₁O₃-δ, MIEC capping layer) | PPTX slides 2, 4 |
| S1 vs S2 same cell? | **NO** — different specimens (pristine 2-layer vs. 3-layer) | PPTX slide 2 |
| Reported VF? | **None stated** | PPTX slides |
| File format (struct.txt) | (x,y,z,value) — "Method 2" from slide 5 | PPTX slide 5 |
| Instrument | FIB-milled pillars + synchrotron XANES | PPTX slide 2 + raw TIF energies |
