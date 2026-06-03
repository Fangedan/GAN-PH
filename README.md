# Advanced 3D Microstructure Generation of SOC Electrodes
### Conditional Wasserstein GAN + Persistent Homology Validation

> **Interactive Demo** — [Live SOC Electrode Simulation →](https://fangedan.github.io/GAN-PH)
> 
> A 3D visualization of the electrochemical process inside a Ni-YSZ solid oxide cell electrode, with animated particle flows and fuel cell / electrolysis mode toggle.

---

## Overview

This repository implements a machine learning pipeline for generating and validating realistic 3D microstructures of solid oxide cell (SOC) electrodes. A conditional Wasserstein GAN (WGAN-GP) is trained to generate synthetic Ni-YSZ porous microstructures while precisely controlling two key structural properties: **volume fraction** and **specific surface area**. Persistent homology analysis validates that the generated structures capture not just explicit statistics but also hidden topological characteristics of real electrode microstructures.

![graphical_abstract](https://github.com/user-attachments/assets/70e6b04b-8bc0-4fab-9fa0-016094c8eac4)

Based on the paper:
> Yamatoko et al., *"Advanced 3D Microstructure Generation of Solid Oxide Cell Electrodes Using Conditional Generative Adversarial Network and Validation Using Nonintuitive Topological Characteristics"*, Advanced Intelligent Discovery, 2025. DOI: [10.1002/aidi.202500010](https://doi.org/10.1002/aidi.202500010)

---

## Repository Structure

```
GAN-PH/
├── 1_GAN/                      # Conditional WGAN-GP training
│   ├── main.py                 # Entry point — configure and launch training
│   ├── training.py             # Training loop (critic + generator updates)
│   ├── models.py               # Generator and Critic network architectures
│   ├── load.py                 # Data loader for BMP stacks
│   └── analysis.py             # Loss curve plotting
│
├── 2_CNN/                      # CNN surface area estimator (trained first)
│   ├── main.py                 # Entry point
│   ├── trainer.py              # Training loop
│   ├── model.py                # CNN architecture
│   ├── load.py                 # Data loader
│   └── analysis.py             # Loss curve plotting
│
├── 3_PH/                       # Persistent homology validation
│   ├── 1_PD.py                 # Compute persistence diagrams
│   ├── 2_PI.py                 # Convert diagrams to persistence images
│   └── 3_PCA.py                # PCA comparison and plotting
│
├── generate_training_data.py   # Synthetic data generator (sphere-packing)
├── preprocess_dream3d.py       # DREAM.3D PNG → BMP pipeline (core deliverable)
└── docs/
    └── index.html              # Interactive SOC simulation (GitHub Pages)
```

---

## Pipeline

The full pipeline runs in six steps:

```
DREAM.3D PNGs  →  preprocess_dream3d.py  →  BMP stacks + results.dat
                                                      ↓
                                            2_CNN/main.py  (train SSA estimator)
                                                      ↓
                                            1_GAN/main.py  (train WGAN-GP)
                                                      ↓
                                            3_PH/1_PD.py   (persistence diagrams)
                                            3_PH/2_PI.py   (persistence images)
                                            3_PH/3_PCA.py  (PCA validation plots)
```

If no real DREAM.3D data is available, `generate_training_data.py` produces synthetic BMP stacks in the correct format for testing the pipeline.

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

**Phase pixel values:**

| Phase | Pixel value | Role |
|-------|------------|------|
| Ni    | 255 (white) | Electron conductor |
| YSZ   | 127 (grey)  | Ion conductor |
| Pore  | 0   (black) | Gas transport |

**results.dat columns:** `VF0 VF1 VF2 SV0 SV1 SV2`
- VF = volume fraction (%) for Ni, YSZ, Pore
- SV = specific surface area (μm⁻¹) for Ni, YSZ, Pore

---

## DREAM.3D Preprocessing

If your data comes from DREAM.3D (PNG slices), use the preprocessing script to convert it:

```bash
python preprocess_dream3d.py --input path/to/png/folder --output ./real_data
```

This handles thresholding, phase remapping, border cropping, resizing to 64×64, z-padding to 64 slices, and writing `results.dat` automatically.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Fangedan/GAN-PH.git
cd GAN-PH

# Create conda environment
conda create -n ganph python=3.10
conda activate ganph

# Install PyTorch (CPU)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install other dependencies
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.0 scikit-learn==1.4.2 matplotlib==3.9.0
pip install opencv-python tqdm torchsummary pyvista homcloud==4.4.1 seaborn
```

> **Note:** If you have an NVIDIA GPU, replace the PyTorch install with the CUDA version:
> `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121`

---

## Usage

### Step 1 — Prepare training data

**Option A:** Generate synthetic data for testing:
```bash
python generate_training_data.py
# Outputs to ./synthetic_data/
```

**Option B:** Preprocess real DREAM.3D data:
```bash
python preprocess_dream3d.py --input path/to/pngs --output ./real_data
```

### Step 2 — Train the CNN surface area estimator

```bash
cd 2_CNN
# Edit main.py: set Input_header and n_struc to match your data folder
python main.py
```

### Step 3 — Train the GAN

```bash
cd ../1_GAN
# Edit main.py: set in_header, n_struc, and path to CNN weights
python main.py
```

### Step 4 — Run topological validation

```bash
cd ../3_PH
python 1_PD.py    # Compute persistence diagrams
python 2_PI.py    # Convert to persistence images (run once per phase: Ni, YSZ, Pore)
python 3_PCA.py   # PCA plots for all three phases
```

---

## Key Results

After training on real Ni-YSZ microstructure data, the model:
- Generates 64×64×64 structures visually indistinguishable from real ones
- Controls volume fraction and specific surface area independently
- Captures hidden topological characteristics validated via persistent homology PCA
- Scales to larger output sizes (96³, 128³, 256³) without retraining

---

## Acknowledgements

Original code and research by Yamatoko et al., Kyoto University / AGH University of Krakow.  
This fork is maintained by Andrew Lin (CS Intern, UTD Lab) — Python preprocessing pipeline and interactive simulation.
