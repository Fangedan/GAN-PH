# GAN-PH Project — Context for Claude

> Updated after runs 0–14. All runs complete. No training currently in progress.
> SSA differentiable loss permanently abandoned. tau_Ni/tau_Pore tradeoff is fundamental.
> Best config: run5 (tpb-proxy, 50 epochs). Next steps require architectural changes.

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
├── generated_data_run8/ # (gitignored) run8 evaluation output (50 structures)
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
| `tpb-proxy` | run5 (50 ep) + run8 (100 ep) + run10 (65 ep) — near-TPB density proxy loss |
| `run6-weighted-tau-ysz` | run6: weighted tau loss YSZ=3×, Ni=1×, Pore=1× (built on tpb-proxy) |
| `run7a-ssa-fix` | run7a: SSA gradient fix attempt 1 (FAILED — gradient explosion) |
| `run7b-ssa-fix` | run7b: SSA gradient fix attempt 2 (FAILED — broadcasting + OOD) |
| `ni-connectivity-ablation` | Earlier experiment (Ni connectivity ablation) |
| `run9-tpb-on-run2` | run9: run2 base + tpb_proxy, no face-hinge |
| `run11-half-face-hinge` | run11: w_conn_ysz_face=100 (half) — SEVERE REGRESSION |
| `run12-lower-tau-weight` | run12: w_tau=20 globally — tau_Ni crashed |
| `run13-ni-tau-only` | run13: tau loss Ni-only — tau loss over-converged → KS FAIL |
| `run14-weighted-pore-tau` | **Latest:** per-phase tau weights Pore=0.05× — tau_Pore improved, tau_Ni crashed |

---

## Full experiment history

### run0 — Baseline (master, 66b3884)
No tau loss. tau_Ni=0.649 F, tau_YSZ=0.484 F.

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

### run5 — TPB proxy loss (tpb-proxy, 5dad25a) — DONE
Added `_loss_tpb_proxy()`: near_tpb = Ni_prob × YSZ_prob × Pore_prob, targets 0.002.
w_tpb=1000. tau_Ni=0.760 M, tau_YSZ=0.479 F, tau_Pore=0.697 F, total_tpb=0.698 F.
tpb_loss reached 0.0017 by epoch 50. tau_loss still converging at epoch 50 (0.31).

### run6 — Weighted tau YSZ 3× (run6-weighted-tau-ysz, 89a42c5) — DONE
Modified `_loss_tortuosity()`: phases = [("Ni",0,1.0), ("YSZ",1,3.0), ("Pore",2,1.0)].
tau_Ni crashed 0.760→0.681 F (REGRESSION). tau_YSZ: 0.479→0.459 F (WORSE). ABANDONED.
total_tpb improved (0.698→0.718 M). tau_Pore improved (0.697→0.730 M).

### run7 — Skipped
Built on run6 (3× YSZ weight) — inherits tau_Ni regression. Not trained.

### run7a — SSA gradient fix attempt 1 (run7a-ssa-fix, 525e694) — FAILED
Removed no_grad/detach from `_loss_ssa`, froze estimator weights, timing=10.
G_loss exploded to 4.5B at epoch 11. Root cause: true was in raw SSA units (~1e7),
pred standardized to [0,1]. With w_param=1000, contribution ≈ 1.9e10.

### run7b — SSA gradient fix attempt 2 (run7b-ssa-fix, 21d25e1) — FAILED
Added standardization of true using mm_list, separate w_ssa=50.
G_loss still exploded to 344M at epoch 11. Root causes:
1. `true` is [3B], `pred` is [3B,1] → broadcasting gives [3B,3B] cross-product (wrong)
2. Estimator sees OOD inputs (soft probs, not binary voxels) → pred outside mm_list range
3. Estimator CNN Jacobian amplifies any gradient routed through it
**SSA differentiable loss is permanently abandoned** — see important notes below.

### run8 — Extended training 100 epochs (tpb-proxy, 5dad25a) — DONE
Same as run5 but 100 epochs. tau_Pore improved (0.697→0.718 M) — only metric that helped.
All other metrics regressed vs run5. Over-training reduces microstructure diversity.
tau_Ni=0.718 M, tau_YSZ=0.459 F, tau_Pore=0.718 M, conn_Ni=0.836 M, total_tpb=0.670 F.

### run9 — run2 base + tpb_proxy, no face-hinge (run9-tpb-on-run2, 94c610c) — DONE
Added tpb_proxy to run2 base (feature/ysz-connectivity-loss), omitting face-hinge.
tau_Ni=0.774 M (better than run5's 0.760 — face-hinge competing with tau gradient confirmed).
total_tpb=0.667 F (WORSE than run5's 0.698 F) — face-hinge adds diversity that helps KS metrics.
tau_Pore=0.699 F (borderline, 0.001 below threshold).

### run10 — Extended training 65 epochs (tpb-proxy, 5dad25a) — DONE
Hypothesis: 65ep sweet spot between run5 (50ep) and run8 (100ep).
FAILED: 65 epochs is WORSE than BOTH 50 and 100 epochs across nearly all metrics.
tau_Ni=0.716 M, tau_YSZ=0.461 F, tau_Pore=0.657 F, total_tpb=0.653 F.
Non-monotonic dynamics: tau loss enters a "valley" at ep60-75 (tau_Ni→0.566 F).

### run11 — Halved face-hinge weight (run11-half-face-hinge, ace571a) — DONE
w_conn_ysz_face: 200→100. SEVERE REGRESSION: tau_Ni=0.682 F (was 0.760 M).
Face-hinge at 200 is a load-bearing equilibrium, not a competing bottleneck.

### run12 — Lower global tau weight (run12-lower-tau-weight, 7d90c1e) — DONE
w_tau: 50→20 (all phases). tau_Pore improved 0.697→0.749 M (Pore over-constraint reduced).
tau_Ni crashed 0.760→0.550 F. Phase-specific finding: Pore tau over-constrains its narrow
natural distribution (std_log=0.0319); Ni tau needs the full signal strength.

### run13 — Ni-only tau loss (run13-ni-tau-only, 53670ff) — DONE
Applied tau loss to Ni phase only (removed YSZ+Pore). tau_Pore=0.819 M (best ever!).
But tau_Ni still crashed to 0.560 F. Root cause: with Ni-only, the tau loss FULLY
converges (tau_loss: 0.31→0.006 at epoch 50) → all Ni samples same tau → KS FAIL.
KEY INSIGHT: in run5, the 3-phase gradient competition prevents any single phase from
fully converging (natural adversarial equilibrium). Removing phases breaks this.

### run14 — Per-phase tau weights Pore=0.05× (run14-weighted-pore-tau, f8e6d63) — DONE
Ni=1.0, YSZ=1.0, Pore=0.05 per-phase tau weights. Hypothesis: 3-phase gradient
competition protects tau_Ni diversity while 0.05× Pore weight barely constrains Pore.
FAILED: tau_Ni=0.580 F (crashed worse than run12). tau_Pore=0.772 M (best ever!).
Root cause: the Ni/Pore tradeoff is FUNDAMENTAL — improving one always crashes the other
through the softmax sum-to-1 constraint. No weight tuning can escape this.
tau_loss=0.357 at epoch 50 (healthy partial convergence, same as run5's 0.31).
Final: tau_Ni=0.580 F, tau_YSZ=0.461 F, tau_Pore=0.772 M, conn_Ni=0.762 M,
conn_YSZ=0.678 F, conn_Pore=0.892 OK, total_tpb=0.657 F, active_tpb=0.675 F.

---

## S-value table (all completed runs)

| run | tau_Ni | tau_YSZ | tau_Pore | conn_Ni | conn_YSZ | conn_Pore | total_tpb | active_tpb |
|---|---|---|---|---|---|---|---|---|
| run0 | 0.649 F | 0.484 F | — | — | — | — | — | — |
| run1 | 0.770 M | 0.484 F | 0.676 F | 0.876 OK | 0.707 M | 0.876 OK | 0.693 F | 0.721 M |
| run2 | 0.818 M | 0.479 F | 0.688 F | 0.891 OK | 0.703 M | 0.878 OK | 0.689 F | 0.718 M |
| run3 | 0.577 F | 0.475 F | 0.633 F | 0.795 M | 0.701 M | 0.812 M | 0.723 M | 0.743 M |
| run4 | 0.755 M | 0.481 F | 0.677 F | 0.880 OK | 0.705 M | 0.872 OK | 0.694 F | 0.720 M |
| run5 | 0.760 M | 0.479 F | 0.697 F | 0.866 OK | 0.704 M | 0.874 OK | 0.698 F | 0.720 M |
| run6 | 0.681 F | 0.459 F | 0.730 M | 0.820 M | 0.682 F | 0.884 OK | 0.718 M | 0.694 F |
| run7a | FAILED | — | — | — | — | — | — | — |
| run7b | FAILED | — | — | — | — | — | — | — |
| run8 | 0.718 M | 0.459 F | 0.718 M | 0.836 M | 0.677 F | 0.880 OK | 0.670 F | 0.676 F |
| run9 | 0.774 M | 0.464 F | 0.699 F | 0.881 OK | 0.681 F | 0.882 OK | 0.667 F | 0.684 F |
| run10 | 0.716 M | 0.461 F | 0.657 F | 0.833 M | 0.685 F | 0.859 OK | 0.653 F | 0.682 F |
| run11 | 0.682 F | 0.468 F | 0.686 F | 0.811 M | 0.693 F | 0.861 OK | 0.648 F | 0.697 F |
| run12 | 0.550 F | 0.473 F | 0.749 M | 0.745 M | 0.693 F | 0.891 OK | 0.676 F | 0.697 F |
| run13 | 0.560 F | 0.462 F | 0.819 M | 0.740 M | 0.696 F | 0.899 OK | 0.601 F | 0.675 F |
| run14 | 0.580 F | 0.461 F | 0.772 M | 0.762 M | 0.678 F | 0.892 OK | 0.657 F | 0.675 F |

Best tau_Ni: run2 (0.818 M). Best overall balance: run5 (50 ep, tpb-proxy). Best tau_Pore: run14 (0.772 M). Full details in `5_TAU/RESULTS.md`.

---

## Key hyperparameters (1_GAN/training.py, as of run14-weighted-pore-tau branch)

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
| timing | 9999 | Epoch at which SSA loss is added (never fires — SSA abandoned) |
| lr | 5e-5 | Adam learning rate for both G and C |

---

## How to run the full pipeline end-to-end

```bash
# 1. Train GAN (from inside 1_GAN/)
conda run -n ganph --no-capture-output python -u main.py \
    --data ../real_data \
    --lr 0.00005 \
    --epochs 100 \
    --tau-estimator ../5_TAU/save_model/tau_net.pth \
    --tau-targets ../5_TAU/tau_targets.json
# On CPU ~90-100 minutes per 50 epochs (26 batches/epoch, batch=4)

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

Watch training live (every-last-batch-per-epoch lines):
```powershell
Get-Content -Wait 1_GAN/runN_output.log | Select-String "\[026/026\]"
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

## What still needs work (after run14 — all experiments paused)

### Fundamental Ni/Pore tortuosity tradeoff (UNRESOLVED, 14 runs)
tau_Ni and tau_Pore are inversely coupled through the softmax sum-to-1 constraint.
Improving Pore tau always crashes Ni tau. No weight tuning can escape this.

| Config | tau_Ni | tau_Pore | Both ≥0.70? |
|---|---|---|---|
| run5 (baseline) | 0.760 M | 0.697 F | NO (Pore barely fails) |
| run12 (w_tau=20) | 0.550 F | 0.749 M | NO (Ni crashes) |
| run13 (Ni-only) | 0.560 F | 0.819 M | NO (Ni crashes) |
| run14 (Pore=0.05×) | 0.580 F | 0.772 M | NO (Ni crashes) |

**Run5 (50 epochs, tpb-proxy, balanced tau weights) is the best achievable config**
with the current MSE-based tau loss. It gets tau_Ni=0.760 M (passes) and
tau_Pore=0.697 F (just fails by 0.003).

### tau_YSZ — stuck at 0.459–0.484 across ALL 14 runs
No density/face/tau-loss approach has moved this. The problem is topological:
YSZ connectivity is non-local; density-based losses satisfy local density without
forming through-thickness percolation. Generated tau_YSZ=10–26 vs real mean 47.

Possible future approaches (require architectural changes):
1. **Differentiable flood-fill**: iterative 3D max-pool along z from z=0 face,
   check YSZ probability propagates to z=63. O(64) conv ops. Not yet implemented.
2. **Accept tau_YSZ FAIL**: it has never improved; it may be unfixable with current GAN.
3. **Longer training**: run8 (100 ep) showed tau_YSZ doesn't improve with more epochs.

### Key insight: tau loss convergence dynamics
- At w_tau=50, 50 epochs is the sweet spot: tau_loss≈0.31 (partial convergence, healthy diversity)
- At epochs 60–75: tau loss fully converges → all samples same τ → KS variance FAIL
- At epoch 100: adversarial signal partially restores diversity (tau_Ni=0.718 M) but TPB metrics suffer
- 3-phase tau gradient competition in run5 prevents any single phase from over-converging
- Removing or strongly downweighting any phase breaks this equilibrium → fast convergence → FAIL
- **Do not train past 50 epochs with current loss config without epoch sweep analysis**

### Task — SSA differentiable loss (ABANDONED — do not retry without architectural changes)
The SSA estimator (2_CNN) was trained on binary voxel structures, not probability maps.
Feeding soft GAN outputs causes OOD estimator behavior. Additionally `true` [3B] and
`pred` [3B,1] shapes broadcast to [3B,3B] in `torch.square(true - pred)`.
Fix requires: retraining estimator on probability maps, OR Gumbel-softmax /
straight-through to get approximately-binary inputs while keeping gradients.
Do NOT attempt again without one of those changes. (timing=9999, never fires)

### Branch hygiene (for next session)
- Decide what to merge to master (run5/tpb-proxy is cleanest working baseline)
- Tag run5 checkpoint as best-known-good in git history
- RESULTS.md is fully updated through run14 on run14-weighted-pore-tau branch

---

## Important implementation notes

- **Never wrap tau_net in torch.no_grad()** — gradients must flow tau_loss → tau_net → g_data → generator.
- **Z-axis is dim 2** of (B,C,Z,Y,X) tensors. Taufactor solves along z.
- **tau_net outputs log(τ)** — targets in tau_targets.json are log-scale.
- **conda run --no-capture-output** is required on Windows to avoid stdout encoding errors.
- **Always pass --data ../real_data** when training. Omitting uses synthetic_data
  (only 13 batches/epoch, different distribution). This caused run3's failure.
- log.dat accumulates across ALL training runs. To detect current run's entries,
  grep for a column unique to that run (e.g. `tpb_loss` for run5/run6/run8).
- matplotlib.use('Agg') must be set before pyplot import — avoids TkAgg crash in
  background processes on Windows.
- **SSA estimator is MONITORING-ONLY** (timing=9999). Do not enable as a loss
  without fixing: (a) estimator retrained on prob maps, (b) shape broadcast bug
  (need to squeeze pred or unsqueeze true so shapes are [3B] vs [3B]).
