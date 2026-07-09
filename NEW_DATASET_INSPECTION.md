# New Dataset Inspection — LSCF-GDC Oxygen Electrode (Synchrotron XANES)

> Reconnaissance only. No files copied into repo. No commits.
> Generated 2026-07-09. PNG slices: `new_dataset_slices.png`, `new_dataset_matbox.png` (repo root, gitignored).
> Source directory: `C:\Users\alin2\Downloads\UTD Summer 26 Internship\raw images backup`

---

## TL;DR

The dataset contains a **SOFC cathode** (oxygen electrode) microstructure with phases
**LSCF + GDC + Pore** — directly analogous to the current GAN's Ni + YSZ + Pore system.
Pre-segmented TIF stacks are present and usable for GAN training without a segmentation step.

**Critical issues before training can proceed:**
1. ⚠️ Voxel size is non-cubic (~40.3 nm × ~40.1 nm) — current GAN assumes isotropic voxels.
2. ⚠️ Phase label encoding differs across files — no single consistent convention (see Q4).
3. ⚠️ Total usable 64³ crop count is low (~30–80 depending on volume chosen) vs. 101 in the current dataset.
4. ⚠️ Multiple distinct samples (S1 vs S2) with different volumes, energies, and conventions.

---

## Q1 — Structure: directory tree, file formats, naming convention

```
raw images backup/          (single flat directory, no subdirectories)
│
│   ── Text-format binary phase maps (tab-separated, (x,y,z,phase01)) ──
├── GDC_struct.txt          (205 MB)
├── LSCF_struct.txt         (205 MB)
├── Pore-LSCF_GDC_struct.txt (205 MB)
│
│   ── Sample S1: raw XANES float32 stacks (3 Ce L-edge energies) ──
├── S1_5700eV_raw.tif       (941 MB)
├── S1_5750eV_raw.tif       (941 MB)
├── S1_6090eV_raw.tif       (941 MB)
├── S1_MATBOX.tif           (20 MB)   ← segmented, 4-label
│
│   ── Sample S2: aligned XANES stacks (2 Ce L-edge + 2 Fe K-edge) ──
├── S2_5700eV_Aligned.tif   (146 MB, float32 [0,255])
├── S2_5750eV_Aligned.tif   (146 MB, float32 [0,255])
├── S2_7040eV_Aligned.tif   (37 MB, uint8)
├── S2_7160eV.tif           (37 MB, uint8)
├── S2_MATBOX.tif           (16 MB)   ← segmented, 5-label
├── S2_GDC_LSCF_SCT_Rectangle.tif (10 MB) ← segmented, integer labels 0-3
│
│   ── "Supercrop" subvolumes (cropped from one of the above) ──
├── Segmented_GDC_Supercrop.tif      (5.2 MB)  ← binary: GDC=105, other=0
├── Segmented_LSCF_Supercrop.tif     (5.2 MB)  ← binary: LSCF=255, other=0
├── Segmented_LSCF_GDC_Supercrop.tif (5.2 MB)  ← 3-phase: LSCF=255, GDC=105, Pore=0
├── Segmented_Phases_Supercrop.txt   (150 MB)  ← text-format of same volume
│
└── USC_Update_10NOV2021.pptx        (39 MB)   ← context slides from USC (Nov 2021)
```

**There are no subdirectories** — contrast with the current dataset's `real_data/structure_XXXX/` layout.
There is **no naming convention** analogous to `structure_XXXX`. The files represent two distinct
samples (S1, S2) and several different representations (raw, segmented, cropped) of each.

---

## Q2 — Geometry: dimensions, voxel size, 64³ crop count

### Segmented volumes (usable for GAN training)

| File | Shape (Z×Y×X) | Usable? | Non-overlapping 64³ crops |
|---|---|---|---|
| Segmented_LSCF_GDC_Supercrop.tif | **151 × 283 × 120** | Yes | 2×4×1 = **8** (Z=120 limits to one 64-cube deep) |
| S2_GDC_LSCF_SCT_Rectangle.tif | **274 × 215 × 166** | Yes (4-label) | 4×3×2 = **24** |
| S1_MATBOX.tif | **228 × 287 × 310** | Partial (47.8% background) | ~3×4×4 = **~48** excluding bg region |
| S2_MATBOX.tif | **304 × 229 × 227** | Partial (41.0% background) | ~4×3×3 = **~36** excluding bg region |

With stride-32 overlapping crops: Supercrop → ~42, SCT_Rectangle → ~140.
Even at maximum overlap, this is a small dataset (comparable to the current 101 structures).

### Voxel size (from `_struct.txt` coordinate columns)

- x-axis step: **4.034 nm × 10 = 40.34 nm/voxel**
- y-axis step: **4.014 nm × 10 = 40.14 nm/voxel**
- z-axis step: not directly read but likely similar (~40 nm)
- **NON-CUBIC voxels** — x and y differ by ~0.5%. The current GAN assumes cubic voxels.

At 40 nm/voxel, a 64³ crop spans **64 × 40 nm = 2.56 µm per side**.
The existing Ni-YSZ-Pore dataset (FIB-SEM, ~5-25 nm/voxel) likely has smaller physical voxels
and a different representative volume element scale. This discrepancy affects generalizability.

### Raw XANES volumes (not directly trainable)

| File | Shape | Dtype | Notes |
|---|---|---|---|
| S1_5700/5750/6090eV_raw.tif | 480 × 700 × 700 | float32 | log-attenuation, values ~±0.01 |
| S2_5700/5750eV_Aligned.tif | 373 × 310 × 316 | float32 | rescaled to [0,255] |
| S2_7040/7160eV.tif | 373 × 310 × 316 | uint8 | |

S1 raw volumes are very large (941 MB each) — not practical to load for GAN training.

### Text-format struct files

`GDC_struct.txt`, `LSCF_struct.txt`, `Pore-LSCF_GDC_struct.txt`: each ~5,002,888 lines.
Format: `x  y  z  phase_value` where x is the fast axis (120 unique x-values per y-row),
voxel step ~40.3 nm. Phase value is 0.0000 or 1.0000 (binary indicator per phase).
Exact volume dimensions unclear (total lines don't divide cleanly into a known shape);
likely correspond to the Supercrop (151×283×120 = 5,127,960 — close but check carefully).

---

## Q3 — Segmented or raw? ⚠️ CRITICAL

**BOTH are present — segmented files exist and are directly usable.**

The segmented TIF files (`Segmented_LSCF_GDC_Supercrop.tif`, `S1_MATBOX.tif`,
`S2_MATBOX.tif`, `S2_GDC_LSCF_SCT_Rectangle.tif`) contain **discrete integer phase labels**.
No segmentation step is required to use these for GAN training.

The raw XANES stacks (`S1_5700eV_raw.tif`, etc.) are **pre-segmentation floating-point images**
used to generate the segmented outputs. They could in principle support re-segmentation
with a different algorithm, but that is not needed if the existing segmented files are accepted.

**Recommendation**: Use the segmented TIFs directly. Start with `S2_GDC_LSCF_SCT_Rectangle.tif`
(cleanest label encoding: integer 0-3) or `Segmented_LSCF_GDC_Supercrop.tif`
(matches current GAN color convention most closely: val=255=LSCF, val=105=GDC, val=0=Pore).

---

## Q4 — Phases: count, identities, label encoding

### Material system

This is a **SOFC cathode (oxygen electrode)**: LSCF + GDC + Pore.
- LSCF = La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃ (electronic conductor → analogous to **Ni** in current GAN)
- GDC = Gd₀.₁Ce₀.₉O₂₋δ (ionic conductor → analogous to **YSZ** in current GAN)
- Pore (open porosity → same role)

**This is a different electrode than the current GAN** (which models the fuel electrode/anode).
The physics is analogous but the materials and phase fractions differ significantly.

### Phase label encoding (varies by file — must standardize)

| File | LSCF | GDC | Pore | Background | Notes |
|---|---|---|---|---|---|
| Segmented_LSCF_GDC_Supercrop.tif | **255** | **105** | **0** | — | Closest to GAN/BMP convention |
| Segmented_LSCF_Supercrop.tif | **255** | 0 (masked) | 0 | — | Binary LSCF mask |
| Segmented_GDC_Supercrop.tif | 0 (masked) | **105** | 0 | — | Binary GDC mask |
| S1_MATBOX.tif | **255** | **110** | **0** | **5** | GDC encoded as 110 (≠105 in Supercrop) |
| S2_MATBOX.tif | **254** | **171** | **0** | **5** | All values shifted; 4th val=88 unknown |
| S2_GDC_LSCF_SCT_Rectangle.tif | **2?** | **1?** | **0** | — | 4 labels (0-3); identities inferred |

**Current GAN/BMP convention**: Ni=255, YSZ=127, Pore=0.
The new dataset does NOT match — YSZ=127 ≠ GDC=105 or 110. A preprocessing remap is required.
DREAM.3D inverted (Ni=0, YSZ=128, Pore=255) also does not match.

### Phase volume fractions

| Volume | LSCF/electron. | GDC/ionic | Pore | 4th phase |
|---|---|---|---|---|
| Segmented_LSCF_GDC_Supercrop (excl. bg) | 17.1% | 21.2% | **61.7%** | — |
| S2_GDC_LSCF_SCT_Rectangle | 11.1% (label=2) | 29.3% (label=1) | **53.0%** (label=0) | 6.6% (label=3) |
| S1_MATBOX (excl. val=5 bg) | 27.6% | 31.2% | 41.2% | — |
| S2_MATBOX (excl. val=5 bg) | 8.0% | 22.1% | 36.3% | 15.9% (val=88) |

The Supercrop has unusually high pore fraction (62%) — this may reflect the specific region cropped.
The S1_MATBOX after excluding background (41% pore) is more physically reasonable for a cathode.

**Unknown 4th phase** in S2_MATBOX (val=88, 16%) and S2_GDC_LSCF_SCT_Rectangle (label=3, 7%):
possibly Pt current collector, carbon interlayer, or resin embedding material. Prof. Jin confirmation needed.

---

## Q5 — Consistency

- **No OneDrive placeholder stubs**: all files have `Archive` attribute and realistic sizes. No `.cloudc` stubs observed.
- **S1 raw TIFs are byte-identical in size**: 940,883,658–940,883,659 bytes each — consistent 3-energy acquisition.
- **Supercrop files are identical in shape** (151×283×120 for all three GDC/LSCF/combined) — these are clearly co-registered binary masks of the same volume.
- **S1 and S2 are DIFFERENT specimens**: different raw dimensions (700×700 vs 310×316), different energy channels, different label encodings in MATBOX files. They cannot be naively mixed without alignment/calibration.
- **GDC label inconsistency between files from the same sample**: Supercrop uses GDC=105, S1_MATBOX uses GDC=110. Same sample, different values — likely from different pipeline runs. Must pick one and remap the other.
- **S2_MATBOX has 5 unique values** vs S1_MATBOX's 4: the extra val=88 (16% of S2 volume) has no counterpart in S1. Its identity is unknown.
- No corrupt files detected (all opened without error). No non-image non-PPTX files.

---

## Q6 — Metadata

| Source | Finding |
|---|---|
| Energy values (5700, 5750, 6090 eV) | Ce L-edge XANES — confirms GDC phase (Ce³⁺/Ce⁴⁺ redox) |
| Energy values (7040, 7160 eV) | Fe K-edge XANES — confirms LSCF phase (Fe in perovskite) |
| Technique | Synchrotron STXM or nanotomography (multi-energy → element-sensitive maps) |
| PPTX filename | `USC_Update_10NOV2021.pptx` — USC collaboration, November 2021 data |
| Folder name | `UTD Summer 26 Internship` — dataset shared for 2026 internship project |
| Voxel size (struct.txt) | ~40.3 nm × 40.1 nm (non-cubic); likely ~40 nm isotropic per Prof. Jin's lab setup |
| PPTX content | Could not extract (python-pptx not in ganph env); likely has instrument/sample details |

**Action item**: Extract PPTX text (`pip install python-pptx`) to confirm:
- Exact voxel size (and whether z-step = x/y-step)
- Phase identity of S2_MATBOX val=88 and SCT_Rectangle label=3
- Whether S1 and S2 are from the same electrode specimen (different regions) or different samples
- Whether the "Aligned" S2 stacks have been registered to the segmented MATBOX

---

## Design implications for pipeline generalization

| Issue | Severity | Mitigation |
|---|---|---|
| Different electrode type (cathode vs anode) | LOW — same 3-phase topology | Remap LSCF→Ni role, GDC→YSZ role |
| Non-cubic voxels (40.34 × 40.14 nm) | MEDIUM | Either accept isotropic approximation (~0.5% error) or add voxel-size conditioning |
| Phase label encoding mismatch | HIGH — must fix before use | Remap to Ni=255, YSZ=127, Pore=0 convention per file |
| Unknown 4th phase in S2 files | MEDIUM | Consult Prof. Jin; treat as Pore or mask it out |
| Small 64³ crop count (~8–140 depending on volume) | HIGH | Stride-32 overlapping crops recommended; may need multiple volumes combined |
| Pore fraction mismatch (41–62% vs typical anode ~30%) | MEDIUM | Consider phase-fraction conditioning (already in GAN via volume fraction loss) |
| S1 vs S2 consistency (different encodings, dimensions) | HIGH | Choose one sample for initial training; avoid mixing until alignment is verified |

**Recommended starting point**: `Segmented_LSCF_GDC_Supercrop.tif` (151×283×120) —
cleanest 3-phase encoding, no background padding to handle. Remap 255→255, 105→127, 0→0
to match GAN/BMP convention. Use stride-32 to get ~42 crops. Small dataset, but clean.
Upgrade to `S1_MATBOX.tif` (after masking val=5 background) for more crops (~48).
