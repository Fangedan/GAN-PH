# GAN-PH Project — Context for Claude

> Updated at the end of a session that ran experiments run5 (tpb-proxy, in progress)
> and set up run6 (run6-weighted-tau-ysz) to run automatically overnight.
> The overnight pipeline is managed by `overnight_pipeline.sh`.

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
├── generated_data/      # (gitignored) run5 evaluation output (50 structures)
├── generated_data_run6/ # (gitignored) run6 evaluation output (50 structures)
├── overnight_pipeline.sh  # automated pipeline: wait run5 → eval → train run6 → eval
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
| `feature/ysz-connectivity-loss` | run2: YSZ min-slice density loss + tau gate |
| `tau-ysz-diagnosis` | run3: distribution-matching loss (FAILED — abandoned) |
| `ysz-face-hinge` | run4: YSZ face density at endpoints (built on run2) |
| `tpb-proxy` | **Current:** run5 — near-TPB density proxy loss (built on ysz-face-hinge) |
| `run6-weighted-tau-ysz` | run6: weighted tau loss YSZ=3×, Ni=1×, Pore=1× (built on tpb-proxy) |
| `ni-connectivity-ablation` | Earlier experiment (Ni connectivity ablation) |

---

## Full experiment history

### run0 — Baseline (master, 66b3884)
No tau loss. tau_Ni=0.649 FAIL, tau_YSZ=0.484 FAIL.

### run1 — Tau loss (tortuosity-loss, 633644c)
Added `_loss_tortuosity()`: MSE(τ_net(phase_prob), log_target) for Ni/YSZ/Pore.
w_tau=50, activates at epoch 10. tau_Ni→0.770 M, tau_YSZ stuck 0.484 F.

### run2 — YSZ min-slice density (feature/ysz-connectivity-loss, a9e1b69)
Added `_loss_connectivity_ysz()`: ReLU(0.10 - mean_ysz_per_z_slice).mean() × w=200.
Added YSZ gate in `_loss_tortuosity()`. tau_Ni→0.818 M. tau_YSZ 0.479 F — density
loss barely fired (<0.001), blobs satisfied threshold without topological percolation.

### run3 — Distribution-matching loss (tau-ysz-diagnosis, ad58eb2) — FAILED
CRITICAL: accidentally trained on synthetic_data (13 batches vs 26). Generator
mode-collapsed, W_D=900, all 50 τ=NaN. DO NOT USE std loss with batch=4.

### run4 — YSZ face-hinge (ysz-face-hinge, c9d0e9a)
Added `_loss_connectivity_ysz_face()`: ReLU(0.18 - mean_ysz_face_z0) + same for z63.
w_conn_ysz_face=200. tau_YSZ→0.481 FAIL (no change). tau_Ni→0.755 M (regression
from run2's 0.818, face loss competes with tau gradient).

### run5 — TPB proxy loss (tpb-proxy, 5dad25a) — IN PROGRESS
Added `_loss_tpb_proxy()`: near_tpb = Ni_prob × YSZ_prob × Pore_prob, targets 0.002.
w_tpb=1000. Root cause of total_tpb FAIL: generated std=0.069 vs real std=0.017
(distribution width problem, not mean mismatch). tpb_loss started firing at epoch 19
(value 0.00063). tau_loss decreasing well (1.95 at epoch 19). Results pending.

Training command used:
```bash
cd 1_GAN
conda run -n ganph --no-capture-output python -u main.py \
    --data ../real_data --lr 0.00005 --epochs 50 \
    --tau-estimator ../5_TAU/save_model/tau_net.pth \
    --tau-targets ../5_TAU/tau_targets.json > run5_output.log 2>&1
```

### run6 — Weighted tau YSZ 3× (run6-weighted-tau-ysz, 89a42c5) — PLANNED
Modified `_loss_tortuosity()`: phases = [("Ni",0,1.0), ("YSZ",1,3.0), ("Pore",2,1.0)].
Loss = sum(weight × MSE) / sum(weights). YSZ gets 3× gradient signal within same
w_tau=50 budget. Will run automatically after run5 via overnight_pipeline.sh.
Checkpoint will land in `/c/Users/alin2/GAN-PH-run6/1_GAN/save_model/Generator_050epoch.pth`
(git worktree, NOT the main repo).

---

## S-value table (all completed runs)

| run | tau_Ni | tau_YSZ | tau_Pore | conn_Ni | conn_YSZ | conn_Pore | total_tpb | active_tpb |
|---|---|---|---|---|---|---|---|---|
| run0 | 0.649 F | 0.484 F | — | — | — | — | — | — |
| run1 | 0.770 M | 0.484 F | 0.676 F | 0.876 OK | 0.707 M | 0.876 OK | 0.693 F | 0.721 M |
| run2 | 0.818 M | 0.479 F | 0.688 F | 0.891 OK | 0.703 M | 0.878 OK | 0.689 F | 0.718 M |
| run3 | 0.577 F | 0.475 F | 0.633 F | 0.795 M | 0.701 M | 0.812 M | 0.723 M | 0.743 M |
| run4 | 0.755 M | 0.481 F | 0.677 F | 0.880 OK | 0.705 M | 0.872 OK | 0.694 F | 0.720 M |
| run5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| run6 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Best baseline: run2 (tau_Ni=0.818 M). Full details in `5_TAU/RESULTS.md`.

---

## Key hyperparameters (1_GAN/training.py, as of tpb-proxy)

| Parameter | Value | Purpose |
|---|---|---|
| w_gp | 20 | Gradient penalty weight |
| w_param | 1000 | Volume fraction loss |
| w_conn | 50 | Pore connectivity (isolation + face hinge) |
| w_conn_ysz | 200 | YSZ min-slice density (threshold=0.10, all z) |
| w_conn_ysz_face | 200 | YSZ face density (threshold=0.18, z=0 and z=63 only) |
| w_tpb | 1000 | Near-TPB density proxy (target=0.002) |
| w_tau | 50 | Tortuosity surrogate loss |
| tau_timing | 10 | Epoch at which tau+tpb losses activate |
| timing | 9999 | Epoch at which SSA loss is added (never in 50 epochs, SSA bug) |
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
# On CPU ~90-100 minutes for 50 epochs (26 batches/epoch, batch=4)

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

`main.py` reads `log_*` keys. Do NOT use raw keys — tau_net outputs log(τ).

---

## Overnight pipeline (overnight_pipeline.sh)

Runs automatically: wait for run5 epoch 50 → eval → create run6 worktree → train run6 → eval.
Key output files (written to 1_GAN/):
- `run5_svalues.log` — stdout from analyze.py (run5 S-value summary)
- `run6_output.log` — run6 training stdout (mirrored from worktree)
- `run6_svalues.log` — stdout from analyze.py (run6 S-value summary)

Pipeline progress is also written to `pipeline.log` in the repo root.

For run6, a git worktree is created at `/c/Users/alin2/GAN-PH-run6` (Windows path:
`C:\Users\alin2\GAN-PH-run6`). Run6 checkpoint ends up in `GAN-PH-run6/1_GAN/save_model/`
and is also copied to `1_GAN/save_model_run6/Generator_050epoch.pth` for evaluation.

---

## What still needs work (after run5/run6 results)

### If run5 improved total_tpb (main goal of run5)
Expected result: total_tpb S-value improves from 0.694 F toward 0.70+.
tau_Ni should stay near 0.75–0.82 M.

### If run6 improved tau_YSZ (main goal of run6)
tau_YSZ stuck at 0.479–0.484 F across all runs 0–4. The 3× gradient weight
is the next most promising fix short of a fundamentally different approach.

### Task 5 — SSA gradient bug (after tau metrics are addressed)
`_loss_ssa` severs gradients: `torch.no_grad() + .detach() + .requires_grad_()`.
Note: SSA loss is ALSO disabled by `timing=9999` (never added to G_loss in 50 epochs).
Fix requires both: (1) remove the no_grad/detach, and (2) set timing to a real epoch
(e.g. timing=10 so SSA trains after the generator has formed recognizable structures).
Only attempt after tau metrics are improved — avoid too many simultaneous changes.

### Task 6 — Branch hygiene
- Update RESULTS.md with run5 and run6 S-values once pipeline finishes
- Decide what to merge to master (probably tpb-proxy after run5 results confirm no regressions)
- Tag important checkpoints in the git history

---

## Important implementation notes

- **Never wrap tau_net in torch.no_grad()** — gradients must flow tau_loss → tau_net → g_data → generator.
- **Z-axis is dim 2** of (B,C,Z,Y,X) tensors. Taufactor solves along z.
- **tau_net outputs log(τ)** — targets in tau_targets.json are log-scale.
- **conda run --no-capture-output** is required on Windows to avoid stdout encoding errors.
- **Always pass --data ../real_data** when training. Omitting uses synthetic_data
  (only 13 batches/epoch, different distribution). This caused run3's failure.
- log.dat accumulates across ALL training runs. To detect current run's entries,
  grep for a column unique to that run (e.g. `tpb_loss` for run5/run6).
- The overnight pipeline uses `git worktree` for run6 so log.dat is not clobbered
  by a branch switch. Run6 trains in `C:\Users\alin2\GAN-PH-run6\`.
- matplotlib.use('Agg') must be set before pyplot import — avoids TkAgg crash in
  background processes on Windows.
