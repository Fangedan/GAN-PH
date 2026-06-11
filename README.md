# Advanced 3D Microstructure Generation of SOC Electrodes
### Conditional Wasserstein GAN + Connectivity-Aware Training + Topological Validation

> **Interactive Demo** — [Live SOC Electrode Simulation →](https://fangedan.github.io/GAN-PH)
>
> A 3D visualization of the electrochemical process inside a Ni-YSZ solid oxide cell electrode, with animated particle flows and a fuel cell / electrolysis mode toggle.

---

## Overview

This repository implements a machine-learning pipeline for **generating, optimizing, and validating** realistic 3D microstructures of solid oxide cell (SOC) electrodes. A conditional Wasserstein GAN (WGAN-GP) generates synthetic Ni-YSZ porous microstructures while controlling **volume fraction** and **specific surface area**, and — added in this fork — a **differentiable connectivity loss** that forces the generated pore phase to actually percolate. Two independent validation layers (persistent homology and a connectivity / triple-phase-boundary analysis module) confirm the structures are not just statistically correct but physically transport-viable.

The original architecture controls *composition* (how much of each material is present). This fork extends it to also control *transport* (whether the electrode can actually conduct electrons/ions and move gas) — the properties that determine real electrochemical performance.

![graphical_abstract](https://github.com/user-attachments/assets/70e6b04b-8bc0-4fab-9fa0-016094c8eac4)

Based on the paper:
> Yamatoko et al., *"Advanced 3D Microstructure Generation of Solid Oxide Cell Electrodes Using Conditional Generative Adversarial Network and Validation Using Nonintuitive Topological Characteristics"*, Advanced Intelligent Discovery, 2025. DOI: [10.1002/aidi.202500010](https://doi.org/10.1002/aidi.202500010)

---

## What's new in this fork

| Addition | What it does |
|----------|--------------|
| **`0_PRV/`** — ParaView automation | Replaces a manual ParaView workflow with one script (`paraview_slice_export.py`) that turns a DREAM.3D `.vtk` volume into a slice-image stack. Verified lossless end-to-end against ground truth. |
| **`4_CNNCT/`** — connectivity & TPB analysis | New module measuring pore/Ni/YSZ percolation, active triple-phase-boundary (TPB) density, tortuosity, and Yu et al. S-values — the transport descriptors the original pipeline never checked. |
| **Differentiable connectivity loss** (`1_GAN/training.py`) | A new generator loss term (isolation penalty + face hinge) that fixes systematic pore collapse. Pore-connectivity S-value improved **0.48 → 0.90**, active-TPB **0.59 → 0.86** (both FAIL → OK). |
| **`preprocess_dream3d.py`** | Converts DREAM.3D PNG exports into the BMP voxel-stack format the GAN expects. Replaces an existing MATLAB workflow. |

---

## Repository Structure

```
GAN-PH/
├── 0_PRV/                      # ParaView VTK → slice-image automation (runs on the data provider's machine)
│   ├── paraview_slice_export.py  # pvpython script: .vtk volume → 64 PNG slices
│   ├── run_slice_export.bat       # drag-and-drop wrapper (no terminal needed)
│   ├── make_test_vtk.py           # build a ground-truth test VTK from a known structure
│   ├── verify_slices.py           # validate exported PNGs against ground truth
│   ├── diagnose_vf.py             # forensic tool for volume-fraction drift
│   └── README.md                  # usage + phase/format conventions
│
├── preprocess_dream3d.py       # DREAM.3D PNG → BMP voxel-stack pipeline
├── test_preprocess.py          # test suite for the preprocessor (4 tests)
│
├── 2_CNN/                      # CNN specific-surface-area estimator (trained first, frozen surrogate)
│   ├── main.py · trainer.py · model.py · load.py · analysis.py
│
├── 1_GAN/                      # Conditional WGAN-GP training
│   ├── main.py                 # entry point — configure and launch training
│   ├── training.py             # training loop + differentiable connectivity loss
│   ├── models.py               # Generator and Critic architectures
│   ├── load.py                 # data loader for BMP stacks
│   └── analysis.py             # loss-curve plotting
│
├── 3_PH/                       # Persistent-homology topological validation
│   ├── 1_PD.py · 2_PI.py · 3_PCA.py
│
├── 4_CNNCT/                    # Connectivity / TPB / tortuosity analysis (new)
│   ├── analyze.py              # connectivity, active TPB, tortuosity, S-values
│   ├── generate_structures.py  # sample N structures from a trained generator
│   ├── plot_results.py         # S-value bar chart + distribution histograms
│   └── test_connectivity.py    # 23-test suite for the analysis functions
│
├── generate_training_data.py   # synthetic sphere-packed data generator (for testing without real data)
└── docs/
    └── index.html              # interactive SOC simulation (GitHub Pages)
```

---

## Pipeline

The full pipeline runs in five stages. Stage 0 prepares slice images from raw volumes; stages 1–4 train, generate, and validate.

```
                 DREAM.3D .vtk volume
                          │
        ┌─────────────────┴─────────────────┐
        │  0_PRV/paraview_slice_export.py    │   (automates manual ParaView slicing)
        └─────────────────┬─────────────────┘
                          ▼
              PNG slice stack
                          │
        ┌─────────────────┴─────────────────┐
        │      preprocess_dream3d.py         │   (PNG → 64³ BMP stacks + results.dat)
        └─────────────────┬─────────────────┘
                          ▼
              BMP stacks + results.dat
                          │
        2_CNN/main.py     ▼   train frozen SSA estimator (surrogate)
                          │
        1_GAN/main.py     ▼   train WGAN-GP  (+ connectivity loss)
                          │
        3_PH/{1_PD,2_PI,3_PCA}.py   topological validation (persistence diagrams → images → PCA)
                          │
        4_CNNCT/{generate_structures,analyze,plot_results}.py   transport validation (connectivity, TPB, S-values)
```

If no real DREAM.3D data is available, `generate_training_data.py` produces synthetic sphere-packed BMP stacks in the correct format for testing the pipeline.

---

## Data Format

All training data must follow this structure:

```
your_data_folder/
├── results.dat                 # Labels: VF0 VF1 VF2 SV0 SV1 SV2 (one row per structure)
├── structure_0001/
│   ├── slice_0000.bmp          # 64×64 grayscale BMP
│   ├── slice_0001.bmp
│   └── ... slice_0063.bmp
├── structure_0002/
│   └── ...
```

**Phase pixel values (GAN/BMP convention):**

| Phase | Pixel value | Role |
|-------|------------|------|
| Ni    | 255 (white) | Electron conductor |
| YSZ   | 127 (grey)  | Ion conductor |
| Pore  | 0   (black) | Gas transport |

> **Note on conventions:** DREAM.3D / ParaView number phases the other way (Ni = phase 1, rendered black; Pore = phase 3, rendered white). `preprocess_dream3d.py` performs the black↔white inversion automatically. See `0_PRV/README.md` for the full conventions table.

**results.dat columns:** `VF0 VF1 VF2 SV0 SV1 SV2`
- VF = volume fraction (%) for Ni, YSZ, Pore
- SV = specific surface area (µm⁻¹) for Ni, YSZ, Pore

---

## Stage 0 — ParaView slice export (`0_PRV/`)

Microstructure volumes from DREAM.3D arrive as `.vtk` files. The original workflow required manually slicing each volume into images through the ParaView GUI — a multi-step process repeated per structure. `0_PRV/paraview_slice_export.py` automates the entire workflow with ParaView's bundled `pvpython`.

```bash
# Run with ParaView 6.1.1's pvpython (not the conda python)
"C:\Program Files\ParaView 6.1.1\bin\pvpython.exe" paraview_slice_export.py m_1.vtk -o m_1_slices

# Or on Windows, drag a .vtk file (or a folder of them) onto run_slice_export.bat
```

Output: 64 borderless 512×512 PNG slices per volume (exactly 8 px per voxel of a 64³ structure), ready for `preprocess_dream3d.py --border 0`.

**Verified lossless.** Because the script can't be run on the data provider's machine ahead of time, it ships with a ground-truth test harness: `make_test_vtk.py` builds a VTK from a structure whose every voxel is known, `verify_slices.py` checks the export against it (frame count, resolution, color purity, per-phase volume fractions, orientation, slice order), and the full round trip reproduces the source volume fractions exactly. See `0_PRV/README.md` for details and the conventions table.

---

## Stage 0.5 — DREAM.3D preprocessing (`preprocess_dream3d.py`)

Converts DREAM.3D / ParaView PNG slice exports into the BMP voxel-stack format the GAN expects. Handles thresholding, phase remapping (inverted conventions), border cropping, resize-or-tile to 64×64, z-padding, and writing `results.dat`.

### Basic usage

```bash
python preprocess_dream3d.py --input path/to/png/folder --output ./real_data
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--multi` | off | Process all sub-folders of `--input` as separate stacks. Structures are numbered sequentially across all folders. |
| `--tile-xy` | off | For large inputs (e.g. 500×500×500), tile spatially in all three axes instead of resizing to 64×64. A 500×500 slice (cropped to 452×452) yields ⌊452/64⌋² = 49 patches per z-slab. |
| `--border N` | 24 | Pixels to crop from each edge before resizing. Use `0` for borderless `0_PRV` exports; `24` matches older bordered exports. |
| `--dry-run` | off | Print what would happen without writing any files. |
| `--preview` | off | Save `preview.png` (slice 32 of the first sub-volume) for visual verification of phase mapping. |
| `--synthetic-test` | off | Round-trip test: reverse-maps an existing BMP stack to fake DREAM.3D PNGs, then runs the full pipeline, verifying correctness without real data. |

> Downsampling uses **block-center sampling**, not interpolation — categorical phase labels must never be blended (a blurred Ni/Pore boundary would read as YSZ). This keeps volume fractions exact through the resize.

### Large dataset examples

```bash
# Multiple stacks in one parent folder
python preprocess_dream3d.py --input ./juan_parent --output ./real_data --multi

# Large 500x500x500 stack — tile spatially to get ~49 structures per z-slab
python preprocess_dream3d.py --input ./juan_data --output ./real_data --tile-xy

# Dry run first, then preview
python preprocess_dream3d.py --input ./juan_data --output ./real_data --tile-xy --dry-run
python preprocess_dream3d.py --input ./juan_data --output ./real_data --tile-xy --preview
```

---

## Stage 1 — Connectivity-aware GAN training (`1_GAN/`)

The conditional WGAN-GP generates a 64³ cube where each voxel is a softmax over (Ni, YSZ, Pore); the final structure is the per-voxel argmax. Training is conditioned on volume fraction and specific surface area via a frozen CNN surrogate (`2_CNN/`).

### The connectivity loss

The original generator optimized only composition (VF, SSA), both measured on the softmax probabilities. Because the structure is built from the *argmax*, the generator could satisfy the volume-fraction target with uniform pore-probability everywhere while never letting pore win a single voxel — driving the pore phase to collapse. The result: structures with correct composition but no connected pore network, and therefore almost no active reaction sites.

`_loss_connectivity()` adds two fully differentiable terms to the generator loss:

- **Isolation penalty** — a 3D convolution with a 6-connectivity kernel counts each voxel's face-adjacent pore neighbors; `pore_prob × (1 − neighbor_pore/6)` penalizes isolated pore voxels and rewards clustered, connected ones.
- **Face hinge** — `ReLU(solid − pore + 0.05)` at the z = 0 and z = 63 faces penalizes solid dominating pore at the electrode entry/exit. Critically, this term still produces a gradient when pore has collapsed to zero (when the isolation penalty has none), bootstrapping the pore phase back into existence.

```
G_loss = g_loss + 1000 × (loss_vf + loss_ssa) + 50 × loss_connectivity
```

---

## Stage 2–4 — Validation

**`3_PH/` — Topological validation.** Persistence diagrams → persistence images → PCA, comparing generated and real structures on hidden topological characteristics, per phase (Ni, YSZ, Pore).

**`4_CNNCT/` — Transport validation.** `analyze.py` computes:
- **Phase connectivity** — `scipy` connected-component labeling, checking whether a component spans z = 0 → z = 63 (percolation through the electrode thickness).
- **Active TPB density** (µm⁻²) — triple-phase-boundary sites where Ni, YSZ, and Pore all percolate (the electrochemically active ones).
- **Tortuosity** — via `taufactor` (requires percolating phases to converge).
- **S-values** — Yu et al. 2025 distribution-similarity score: `S ≥ 0.85` OK, `0.70–0.85` Marginal, `< 0.70` Fail.

```bash
# Analyze a folder of structures
python 4_CNNCT/analyze.py --input ../synthetic_data --output synthetic_connectivity.csv --no-tau

# Compare generated vs training distributions (S-values)
python 4_CNNCT/analyze.py --input ../synthetic_data --compare ../generated_data --output s_values.csv --no-tau
```

---

## Key Results

Connectivity loss, evaluated on 50 generated structures vs 50 synthetic training structures (S-values, Yu et al. 2025):

| Metric | Before | After | |
|--------|--------|-------|---|
| Pore connectivity | 0.480 | **0.896** | FAIL → OK |
| Active TPB density | 0.592 | **0.863** | FAIL → OK |
| Total TPB density | 0.610 | **0.867** | FAIL → OK |
| YSZ connectivity | 0.773 | **0.889** | MARGINAL → OK |

- Before the fix, **47 of 50** generated structures had zero percolating pore; active TPB was ~60× below training data (0.009 vs 0.526 µm⁻²).
- After the fix, all 50 structures have pore connectivity > 0.85; active-TPB mean rose to 0.575 µm⁻², matching the training distribution.

The base model additionally:
- Generates 64×64×64 structures visually indistinguishable from real ones, controlling volume fraction and specific surface area independently.
- Captures hidden topological characteristics validated via persistent-homology PCA.
- Scales to larger output sizes (96³, 128³, 256³) without retraining.

> Current results are on synthetic sphere-packed data; validation on real DREAM.3D structures is in progress.

---

## Testing

```bash
python test_preprocess.py              # preprocessor: 4 end-to-end tests
python 4_CNNCT/test_connectivity.py    # analysis module: 23 tests
# 0_PRV harness: see 0_PRV/README.md (make_test_vtk.py → verify_slices.py)
```

| Suite | Coverage |
|-------|----------|
| `test_preprocess.py` | resize mode, tile-XY mode, multi-folder mode, pixel-level phase round-trip |
| `4_CNNCT/test_connectivity.py` | solid cube, disconnected checkerboard, three-phase active TPB, S-value edge cases, face percolation |
| `0_PRV/verify_slices.py` | frame count, resolution, color purity, volume fractions, orientation, slice order |

---

## Installation

```bash
git clone https://github.com/Fangedan/GAN-PH.git
cd GAN-PH

conda create -n ganph python=3.10
conda activate ganph

# PyTorch (CPU)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Other dependencies
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.0 scikit-learn==1.4.2 matplotlib==3.9.0
pip install opencv-python tqdm torchsummary pyvista homcloud==4.4.1 seaborn taufactor
```

> **GPU:** replace the PyTorch line with the CUDA build:
> `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121`
>
> **ParaView (Stage 0 only):** `0_PRV` requires a separate [ParaView 6.1.1](https://www.paraview.org/download/) install and uses its bundled `pvpython` — it is not part of the conda environment.

---

## Usage

### Step 0 — (Optional) Export slices from .vtk volumes
```bash
"C:\Program Files\ParaView 6.1.1\bin\pvpython.exe" 0_PRV/paraview_slice_export.py volume.vtk -o slices
```

### Step 1 — Prepare training data
```bash
# Option A: synthetic data for testing
python generate_training_data.py            # → ./synthetic_data/

# Option B: real DREAM.3D data
python preprocess_dream3d.py --input path/to/pngs --output ./real_data            # single folder
python preprocess_dream3d.py --input path/to/parent --output ./real_data --multi  # multiple stacks
python preprocess_dream3d.py --input path/to/pngs --output ./real_data --tile-xy  # large stack
```
After running, set `n_struc` in `2_CNN/main.py` and `1_GAN/main.py` to the structure count reported at the end of the script.

### Step 2 — Train the CNN surface-area estimator
```bash
cd 2_CNN        # edit main.py: set Input_header and n_struc
python main.py
```

### Step 3 — Train the GAN
```bash
cd ../1_GAN     # edit main.py: set in_header, n_struc, path to CNN weights
python main.py
```

### Step 4 — Validate
```bash
cd ../3_PH      # topological validation
python 1_PD.py && python 2_PI.py && python 3_PCA.py

cd ../4_CNNCT   # transport validation
python generate_structures.py
python analyze.py --input ../generated_data --output generated_connectivity.csv --no-tau
python plot_results.py
```

---

## Acknowledgements

Original code and research by Yamatoko et al., Kyoto University / AGH University of Krakow.
This fork is maintained by **Andrew Lin** (CS Intern, UTD Lab — Prof. Xinfang Jin) — DREAM.3D preprocessing pipeline (`preprocess_dream3d.py`), ParaView slice-export automation (`0_PRV/`), connectivity/TPB analysis module (`4_CNNCT/`), the differentiable connectivity loss, test suites, and the interactive simulation.
