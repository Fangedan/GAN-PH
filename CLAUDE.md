# GAN-PH Project — Context for Claude

> Drop this file into the repo root (or paste into chat) to give Claude full context.
> It was written at the end of a multi-session run that retrained the GAN with
> tortuosity-aware and YSZ-connectivity-aware losses.

---

## What this project is

A **conditional WGAN-GP** that generates realistic 3-phase SOC electrode
microstructures (64³ voxels, phases: Ni=255, YSZ=127, Pore=0) matching real
FIB-SEM data. Quality is measured by **S-values** (Yu et al. 2025): a
Kolmogorov-Smirnov-based score comparing the generated distribution to real
structures on connectivity, TPB density, and tortuosity factor (τ). S ≥ 0.85
= OK, 0.70–0.85 = MARGINAL, < 0.70 = FAIL.

The project goal is a generator whose S-values are all ≥ 0.70 (ideally ≥ 0.85).

---

## Repository layout

```
GAN-PH/
├── 1_GAN/               # WGAN-GP: generator, critic, training loop
│   ├── main.py          # entry point — run from inside 1_GAN/
│   ├── training.py      # Trainer class (all loss functions live here)
│   ├── models.py        # Generator / Critic / Estimator architectures
│   ├── load.py          # data loading
│   └── save_model/      # (gitignored) Generator_NNNepoch.pth checkpoints
│
├── 2_CNN/               # Frozen SSA estimator (pre-trained, don't retrain)
│   └── save_model/model_200epoch.pth
│
├── 4_CNNCT/             # S-value analysis pipeline
│   ├── analyze.py       # computes per-structure metrics + S-value comparison
│   └── generate_structures.py  # samples N structures from generator
│
├── 5_TAU/               # Tortuosity surrogate (τ-net) pipeline — added in session
│   ├── tau_net.py           # 3D CNN: input (B,1,64,64,64) → output (B,) log(τ)
│   ├── train_tau_net.py     # trains τ-net on real structures
│   ├── compute_tau_labels.py # runs taufactor on real structures → tau_labels.csv
│   ├── tau_targets.json     # mean log(τ) per phase from real data
│   └── save_model/tau_net.pth  # (gitignored) best checkpoint, val MSE=0.0940
│
├── real_data/           # (gitignored) 101 real structure_XXXX/ folders
├── generated_data/      # (gitignored) most-recent generator output
└── CLAUDE.md            # ← this file
```

---

## Conda environment

```
conda activate ganph
```

All commands below assume this env. Run `1_GAN/main.py` from **inside** `1_GAN/`.

---

## Git branches

| Branch | What's on it |
|---|---|
| `master` | Original code + gitignore fixes + minor bug fixes |
| `tortuosity-loss` | **Checkpoint:** τ-net pipeline (5_TAU/) + first GAN retrain with tau loss |
| `feature/ysz-connectivity-loss` | **Active:** adds YSZ connectivity loss + tau gate (in progress) |
| `ni-connectivity-ablation` | Earlier experiment (Ni connectivity ablation) |

The work described below is all on `feature/ysz-connectivity-loss`.

---

## What was done (chronological)

### Step 1 — Offline τ labelling
```bash
cd 5_TAU
python compute_tau_labels.py --data ../real_data --output tau_labels.csv
```
Ran taufactor (Laplace solver, z-direction) on all 101 real structures × 3 phases.
Output: `5_TAU/tau_labels.csv`, `5_TAU/tau_targets.json`.

Key targets (log scale, what τ-net is trained on):
- log_Ni = 2.5764  (τ_Ni  = 13.37)
- log_YSZ = 3.6540 (τ_YSZ = 46.61)
- log_Pore = 0.7034 (τ_Pore = 2.02)

### Step 2 — Train τ-net surrogate
```bash
cd 5_TAU
python train_tau_net.py --data ../real_data --labels tau_labels.csv \
    --lr 0.0001 --n-aug 4 --epochs 200
```
Architecture: 3D CNN, ndf=16, input (B,1,64,64,64) → scalar log(τ).
Best val MSE = 0.0940 at epoch 60. Saved to `5_TAU/save_model/tau_net.pth`.

Key implementation details:
- Lazy TauDataset (uint8 volumes in RAM ~21 MB, augment on-the-fly)
- 4 augmentations = 4 z-rotations (z-preserving, τ invariant)
- Incremental checkpointing: saves on each val improvement

### Step 3 — First GAN retrain (tortuosity-loss branch)
```bash
cd 1_GAN
python main.py --data ../real_data --lr 0.00005 --epochs 50 \
    --tau-estimator ../5_TAU/save_model/tau_net.pth \
    --tau-targets ../5_TAU/tau_targets.json
```
- Added `_loss_tortuosity()` to training.py: MSE(τ_net(phase_prob), log_target)
- Frozen τ-net weights (requires_grad=False); NO torch.no_grad() so gradients
  flow through τ-net back to generator
- w_tau=50, activates at epoch 10

**Results after 50 epochs:**
| Metric | Before | After |
|---|---|---|
| tau_Ni | 0.649 FAIL | 0.770 MARGINAL ✓ |
| tau_YSZ | 0.484 FAIL | 0.484 FAIL ✗ |
| conn_Ni | — | 0.876 OK |
| conn_YSZ | — | 0.707 MARGINAL |
| tau_Pore | — | 0.676 FAIL |
| total_tpb | — | 0.693 FAIL |

### Step 4 — Root cause of tau_YSZ failure
5 out of 50 generated structures had conn_YSZ=0 (completely disconnected YSZ →
taufactor returns NaN). Many more had low YSZ connectivity (conn_YSZ=0.14–0.46),
causing taufactor to return unconverged lower-bound estimates (63–214).
When YSZ is disconnected, τ-net gradients carry no useful signal.

### Step 5 — Current work (feature/ysz-connectivity-loss branch)
Two fixes added to `1_GAN/training.py`:

**Fix 1 — `_loss_connectivity_ysz()`:**
```python
ysz = g_data[:, 1, :, :, :]           # (B, Z, Y, X)
slice_means = ysz.mean(dim=(2, 3))     # (B, Z) — mean YSZ per z-slice
loss = F.relu(0.10 - slice_means).mean()
```
Penalises any z-slice where mean YSZ probability < 0.10 (real YSZ vf ≈ 0.20).
Weight w_conn_ysz=200, active from epoch 0.

**Fix 2 — YSZ gate in `_loss_tortuosity()`:**
```python
ysz_plane = phase_prob[:, 0, :, :, :]              # (B, Z, Y, X)
min_slice  = ysz_plane.mean(dim=(2, 3)).min(dim=1).values  # (B,)
connected  = min_slice > 0.05
if not connected.any(): continue
tau_pred   = self.tau_net(phase_prob[connected])   # only connected samples
```
Skips disconnected batch samples from the YSZ tau MSE.

**Training in progress (as of when this file was written):**
Epoch ~35/50 complete. `tau_loss` oscillating 0.17–0.79, `conn_ysz_loss` ≈ 0.
S-value analysis not yet run on this checkpoint.

---

## How to run the full pipeline end-to-end

```bash
# 1. Generate structures
cd 4_CNNCT
python generate_structures.py \
    --training-data ../real_data \
    --output ../generated_data \
    --n 50

# 2. Compute S-values
python analyze.py \
    --input ../real_data \
    --compare ../generated_data \
    --output s_value_report.csv
# S-value summary printed to stdout; also saved to s_value_report_svalues.csv
```

For GAN training:
```bash
cd 1_GAN
conda run -n ganph --no-capture-output python -u main.py \
    --data ../real_data \
    --lr 0.00005 \
    --epochs 50 \
    --tau-estimator ../5_TAU/save_model/tau_net.pth \
    --tau-targets ../5_TAU/tau_targets.json
```
On CPU this takes ~60–90 minutes for 50 epochs.

---

## Key hyperparameters (1_GAN/training.py)

| Parameter | Value | Purpose |
|---|---|---|
| w_gp | 20 | Gradient penalty weight |
| w_param | 1000 | Volume fraction loss |
| w_conn | 50 | Pore connectivity (isolation + face hinge) |
| w_conn_ysz | 200 | YSZ percolation proxy (min z-slice mean) |
| w_tau | 50 | Tortuosity surrogate loss |
| tau_timing | 10 | Epoch at which tau+ysz-tau losses activate |
| lr | 5e-5 | Adam learning rate for both G and C |

---

## What still needs work

After the current run completes, run S-value analysis and check:

1. **tau_YSZ** — was FAIL (0.484). The connectivity loss + tau gate should improve
   this. If still FAIL, consider:
   - Increasing w_conn_ysz (try 400)
   - Increasing w_tau for YSZ specifically (modify _loss_tortuosity to weight YSZ more)
   - Training more epochs (100 instead of 50)

2. **total_tpb** — was FAIL (0.693). TPB density depends on Ni-YSZ-Pore triple
   phase boundaries. No explicit loss for this yet. If it stays FAIL, consider
   adding a differentiable TPB proxy (count voxels adjacent to all three phases).

3. **tau_Pore** — was FAIL (0.676). Pore tortuosity. Similar issue to YSZ —
   may need a pore connectivity loss (already has face hinge but no min-slice loss).

4. **conn_YSZ** — was MARGINAL (0.707). Should improve with YSZ connectivity loss.

---

## Useful diagnostic commands

```bash
# Watch training live (last batch of each epoch)
Get-Content -Wait 1_GAN/log.dat | Select-String "\[026/026\]"

# Quick S-value check after generating structures
cd 4_CNNCT && python analyze.py --input ../real_data --compare ../generated_data

# Check τ-net prediction on a single structure
cd 5_TAU && python test_tau_loss.py
```

---

## Important implementation notes

- **Never wrap tau_net in torch.no_grad()** — the whole point is gradient flow
  from tau_loss through tau_net to the generator. See the long comment in
  training.py `_loss_tortuosity()`.
- **Z-axis is dim 0** of the (64,64,64) volume = dim 2 of (B,C,Z,Y,X) tensors.
  Taufactor solves along z. All augmentations in train_tau_net.py must preserve z.
- **τ-net outputs log(τ)**, not raw τ. Targets in tau_targets.json are also log-scale
  (`log_Ni`, `log_YSZ`, `log_Pore` keys). Do not mix up with the raw-scale keys.
- **conda run --no-capture-output** is needed on Windows to avoid stdout buffering
  when running in background.
