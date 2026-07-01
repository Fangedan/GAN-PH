# GAN-PH Project — Context for Claude

> Drop this file into the repo root (or paste into chat to give Claude full context.
> Updated at the end of a session that ran experiments run3 (distribution-matching,
> failed) and run4 (YSZ face-hinge, currently training on ysz-face-hinge branch).

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
├── 5_TAU/               # Tortuosity surrogate (τ-net) pipeline
│   ├── tau_net.py           # 3D CNN: input (B,1,64,64,64) → output (B,) log(τ)
│   ├── train_tau_net.py     # trains τ-net on real structures
│   ├── compute_tau_labels.py # runs taufactor on real structures → tau_labels.csv
│   ├── tau_targets.json     # mean + std log(τ) per phase from real data
│   ├── RESULTS.md           # per-run S-value table (source of truth for experiments)
│   └── save_model/tau_net.pth  # (gitignored) best checkpoint, val MSE=0.0940
│
├── real_data/           # (gitignored) 101 real structure_XXXX/ folders
├── generated_data/      # (gitignored) most-recent generator output
└── CLAUDE.md            # this file
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
| `tortuosity-loss` | run1: τ-net pipeline + first GAN retrain with tau loss |
| `feature/ysz-connectivity-loss` | run2: adds YSZ min-slice density loss + tau gate |
| `tau-ysz-diagnosis` | run3: distribution-matching loss (FAILED — abandoned) |
| `ysz-face-hinge` | **Active:** run4 — YSZ face density at endpoints (built on run2) |
| `ni-connectivity-ablation` | Earlier experiment (Ni connectivity ablation) |

---

## Full experiment history

### run0 — Baseline (master, commit 66b3884)
No tau loss. tau_Ni=0.649 FAIL, tau_YSZ=0.484 FAIL.

### run1 — Tau loss added (tortuosity-loss, commit 633644c)
Added `_loss_tortuosity()`: MSE(τ_net(phase_prob), log_target) for Ni/YSZ/Pore.
w_tau=50, activates at epoch 10. Result: tau_Ni improved to 0.770 MARGINAL.
tau_YSZ stuck at 0.484 because disconnected YSZ → taufactor NaN → tau_net gradient useless.

### run2 — YSZ connectivity proxy (feature/ysz-connectivity-loss, commit a9e1b69)
Added `_loss_connectivity_ysz()`: ReLU(0.10 - mean_ysz_per_z_slice).mean() × w=200.
Added YSZ gate in `_loss_tortuosity()`: skip samples where min z-slice mean < 0.05.
Result: tau_Ni→0.818 MARGINAL. tau_YSZ still 0.479 FAIL — density loss barely fired
(<0.001) because YSZ blobs satisfied 10% density threshold without being topologically connected.

### run3 — Distribution-matching loss (tau-ysz-diagnosis, commit ad58eb2) — FAILED
Replaced MSE-to-mean with mean_loss + std_loss. Key bug: run3 accidentally omitted
`--data ../real_data` and trained on synthetic_data (only 13 batches/epoch vs 26 for
real data). Additionally, batch=4 std gradient was too noisy (3 DOF). Generator
mode-collapsed: W_D=900 at epoch 50, all 50 generated structures returned τ=NaN.
tau_Ni=0.577 FAIL (regression). ABANDONED.

DO NOT use distribution-matching std loss with batch=4. If retried, accumulate
τ-net predictions over N≥50 steps with a running FIFO buffer before computing std.

### run4 — YSZ face-hinge (ysz-face-hinge, commit c9d0e9a) — IN PROGRESS
Built on run2 code. Added `_loss_connectivity_ysz_face()`: requires mean YSZ
probability at z=0 and z=63 faces ≥ 0.18 (vs existing 0.10 threshold across all
slices). w_conn_ysz_face=200, active from epoch 0.

Physics: if YSZ is present at both entry/exit faces and has ≥10% density per slice,
taufactor's z-direction solve has a path to find. The 0.10 threshold barely fired
because blobs satisfied density without connecting the endpoints.

---

## S-value table (all runs)

| run | tau_Ni | tau_YSZ | tau_Pore | conn_Ni | conn_YSZ | conn_Pore | total_tpb | active_tpb |
|---|---|---|---|---|---|---|---|---|
| run0 | 0.649 F | 0.484 F | — | — | — | — | — | — |
| run1 | 0.770 M | 0.484 F | 0.676 F | 0.876 OK | 0.707 M | 0.876 OK | 0.693 F | 0.721 M |
| run2 | 0.818 M | 0.479 F | 0.688 F | 0.891 OK | 0.703 M | 0.878 OK | 0.689 F | 0.718 M |
| run3 | 0.577 F | 0.475 F | 0.633 F | 0.795 M | 0.701 M | 0.812 M | 0.723 M | 0.743 M |
| run4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Full details in `5_TAU/RESULTS.md`.

---

## Key hyperparameters (1_GAN/training.py on ysz-face-hinge)

| Parameter | Value | Purpose |
|---|---|---|
| w_gp | 20 | Gradient penalty weight |
| w_param | 1000 | Volume fraction loss |
| w_conn | 50 | Pore connectivity (isolation + face hinge) |
| w_conn_ysz | 200 | YSZ min-slice density (threshold=0.10, all z) |
| w_conn_ysz_face | 200 | YSZ face density (threshold=0.18, z=0 and z=63 only) |
| w_tau | 50 | Tortuosity surrogate loss (MSE to log mean) |
| tau_timing | 10 | Epoch at which tau loss activates |
| lr | 5e-5 | Adam learning rate for both G and C |

---

## How to run the full pipeline end-to-end

```bash
# 1. Train GAN (from inside 1_GAN/)
conda run -n ganph --no-capture-output python -u main.py \
    --data ../real_data \
    --lr 0.00005 \
    --epochs 50 \
    --tau-estimator ../5_TAU/save_model/tau_net.pth \
    --tau-targets ../5_TAU/tau_targets.json
# On CPU ~90-100 minutes for 50 epochs (26 batches/epoch with 101 real structures, batch=4)

# 2. Generate structures (from inside 4_CNNCT/)
conda run -n ganph --no-capture-output python -u generate_structures.py \
    --training-data ../real_data \
    --output ../generated_data \
    --n 50

# 3. Compute S-values (from inside 4_CNNCT/)
conda run -n ganph --no-capture-output python -u analyze.py \
    --input ../real_data \
    --compare ../generated_data \
    --output s_values_runN.csv
# S-value summary printed to stdout; also saved to s_values_runN_svalues.csv
```

Watch training live:
```powershell
Get-Content -Wait 1_GAN/log.dat | Select-String "\[026/026\]"
```

---

## tau_targets.json format

```json
{
  "Ni": 13.37,         "log_Ni": 2.5764,   "std_log_Ni": 0.1816,
  "YSZ": 46.61,        "log_YSZ": 3.6540,  "std_log_YSZ": 0.5660,
  "Pore": 2.02,        "log_Pore": 0.7034, "std_log_Pore": 0.0319
}
```

`main.py` reads `log_*` keys (log-scale targets). `std_log_*` keys are computed
by `compute_tau_labels.py` and available for future use. Do NOT use raw keys for
training — tau_net outputs log(τ), not raw τ.

---

## What still needs work (after run4 results are in)

### If run4 improved tau_YSZ: next is Task 4 — TPB proxy
```python
# Near-TPB: voxel where all three phases have non-trivial probability
near_tpb = g_data[:,0] * g_data[:,1] * g_data[:,2]   # (B, 64, 64, 64)
loss_tpb = F.relu(target_tpb - near_tpb.mean())
```
total_tpb and active_tpb have been stuck at FAIL/MARGINAL across all runs.

### If run4 did NOT improve tau_YSZ: consider next alternatives
- Increase w_conn_ysz_face (try 500) to force more YSZ at faces
- Raise face threshold from 0.18 to 0.20 (full real vf)
- Try more epochs (100) — tau_Ni improved consistently with more training
- Weighted tau loss: give YSZ 3× weight in `_loss_tortuosity()`

### Task 5 — SSA gradient bug (BUG 1)
`_loss_ssa` severs gradients: `torch.no_grad() + .detach() + .requires_grad_()`.
Fix: remove all three and let gradients flow through estimator naturally.
Only attempt this after tau metrics are in better shape (avoid confounds).

---

## Important implementation notes

- **Never wrap tau_net in torch.no_grad()** — gradients must flow tau_loss → tau_net → g_data → generator.
- **Z-axis is dim 2** of (B,C,Z,Y,X) tensors. Taufactor solves along z.
- **tau_net outputs log(τ)** — targets in tau_targets.json are also log-scale.
- **conda run --no-capture-output** is required on Windows to avoid stdout encoding errors.
- **Always pass --data ../real_data** when training — omitting it uses synthetic_data
  (only 13 batches/epoch, different distribution). This bug caused run3 failure.
- log.dat accumulates across all training runs. To identify current run's entries,
  grep for a column that is unique to that run (e.g. `conn_ysz_face_loss` for run4).
- The `generate_structures.py` default loads `Generator_050epoch.pth` — make sure
  you're on the right branch before generating.
