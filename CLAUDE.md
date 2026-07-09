# GAN-PH Project — Context for Claude

> Updated after runs 0–14 + Prof. Jin scope decision (2026-07). All runs complete. No training currently in progress.
> SSA differentiable loss permanently abandoned. tau_Ni/tau_Pore tradeoff is fundamental.
> Best config: run5 (tpb-proxy, 50 epochs).
> **SCOPE CHANGE:** Tortuosity (all phases) is DESCOPED as a success criterion — see section below.

---

## What this project is

A **conditional WGAN-GP** that generates realistic 3-phase SOC electrode
microstructures (64³ voxels, phases: Ni=255, YSZ=127, Pore=0) matching real
FIB-SEM data. Quality is measured by **S-values** (Yu et al. 2025): a
Kolmogorov-Smirnov-based score comparing the generated distribution to real
structures. S ≥ 0.85 = OK, 0.70–0.85 = MARGINAL, < 0.70 = FAIL.

**Scored criteria (pass/fail):** connectivity (conn_Ni, conn_YSZ, conn_Pore),
TPB density (total_tpb, active_tpb, active_tpb_frac). YSZ quality is scored
by conn_YSZ (percolation fraction), not tortuosity.

**Informational only (not scored):** tau_Ni, tau_YSZ, tau_Pore — computed and
reported by analyze.py but excluded from pass/fail. See SCOPE DECISION (2026-07)
section below. Real-vs-real ceiling for tau_YSZ = 0.658 (intrinsically below
MARGINAL — no generator can achieve OK on this metric with this dataset).

The project goal is a generator whose **scored** S-values are all ≥ 0.70
(ideally ≥ 0.85). Tau S-values are monitored but not success-gated.

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

## Key hyperparameters (1_GAN/training.py, run5 config — equal tau weights across all phases)

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

### Branch hygiene
- master carries run5 training config + full run0–14 documentation (resolved 2026-07)
- Tag `best-known-good-run5` on master points to the post-merge commit
- run5 weights (*.pth) are gitignored and live locally; run2 weights also intact locally (historical tau_Ni record)

---

## SCOPE DECISION (2026-07)

**Prof. Jin (Week 5 meeting + written follow-up) confirmed tortuosity as a whole (all three phases: tau_Ni, tau_YSZ, tau_Pore) is NOT a critical success criterion for this project.**

This is a **descope, not a solve**. 14 runs demonstrated the metric was fundamentally difficult to optimize; the decision reflects scientific priorities, not a claim that tau was achieved.

### What changes
- **tau_* S-values are now informational/diagnostic only.** They are still computed and reported, but excluded from pass/fail scoring. A reader seeing tau values in the output should treat them as monitoring data, not graded criteria.
- **The distribution-matching retry for tau_Ni is DROPPED.** No further experiments aimed at improving tau S-values are planned.
- **Pareto frontier axes are now: active TPB density vs phase connectivity/percolation.** The tau axes are removed from the success frontier.
- **YSZ scoring moves from tau_YSZ to conn_YSZ (percolation fraction).** tau_YSZ was pinned at 0.459–0.484 FAIL across all 14 runs and is unchanged by the descope — it remains documented as a known-stuck metric.

### What does NOT change (findings remain on record)
- **run2's tau_Ni 0.818 M** (best Ni tortuosity) is a methods finding, not a scored victory — it demonstrates what the tau loss could achieve for one phase under the best conditions, but the metric is no longer success-gated.
- **run13's tau_Pore 0.819 M** (best Pore tortuosity) is similarly a methods finding.
- **The fundamental Ni/Pore softmax coupling** (documented runs 8–14) is a genuine architectural finding — improving Pore tau always trades off against Ni tau through the softmax sum-to-1 constraint. It remains documented even though neither metric is now success-gated.
- **tau_YSZ stuck at 0.459–0.484 FAIL across all 14 runs** is still documented. Descoping it does not mean it was fixed.

### Descope ≠ solved
The tau metrics were descoped because they were not achievable with the current architecture and they are not the primary physical quantity Prof. Jin needs to assess. The underlying reason tau_YSZ, tau_Ni, and tau_Pore were hard — softmax coupling, non-local percolation topology, convergence dynamics — is unchanged and documented in RESULTS.md.

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

---

## CATHODE DATASET (2026-07)

> Phase 1 of dataset generalisation: config system + cube extraction (branch feature/dataset-configs).
> Commits: Task1 PPTX extraction, Task2 config system, Task3 cube extraction, Task4 de-hardcode,
> Task5 docs. No training runs — extraction only.

### Dataset identity

LSCF+GDC+Pore 3-phase SOC **cathode** microstructure, analogous to the anode Ni-YSZ-Pore but for
the cathode side. Source: Dr. Xinfang Jin's group (UTD), synchrotron XANES at Ce L-edge + Fe K-edge,
FIB-milled pillar geometry. Voxel size ~40 nm (x/y measured from coordinates: x=0.04034 µm,
y=0.04014 µm). Segmented TIF, phase labels {255: LSCF, 105: GDC, 0: Pore} after remapping.

**Phase correspondence:**

| Channel | Anode | Cathode (S1) | BMP value |
|---------|-------|--------------|-----------|
| 0 | Ni (electron_conductor) | LSCF (electron_conductor) | 255 |
| 1 | YSZ (ion_conductor) | GDC (ion_conductor) | 127 |
| 2 | Pore (gas) | Pore (gas) | 0 |

### Specimens: S1 and S2

Two specimens were analysed from the PPTX (NEW_DATASET_PPTX_NOTES.md); only S1 is used.

- **S1 (Supercrop):** Pristine LSCF+GDC+Pore bilayer. File: `Segmented_LSCF_GDC_Supercrop.tif`,
  shape (151, 283, 120), dtype uint8. Source labels: {255:LSCF, 105:GDC, 0:Pore}. Config:
  `cathode_s1_supercrop` — the sole cathode training source.
- **S2:** Pristine LSCF+GDC+SCT trilayer. **DESCOPED 2026-07 per Prof. Jin** — SCT is an extra
  film layer not representative of the target cathode. Config stub `cathode_s2` retained for
  label-map documentation only; do NOT pass to training pipeline.
  - 4th phase SCT = SrCo₀.₉Ta₀.₁O₃-δ (mixed ionic-electronic conductor capping layer, MIEC).
    Identified from PPTX slides 2 and 4.

### Config system layout

```
configs/
├── dataset_config.py          # loader: get_config(name) / default() → anode
├── anode_niysz.yaml           # Ni-YSZ-Pore, voxel=0.1 µm, existing pipeline
├── cathode_s1_supercrop.yaml  # LSCF-GDC-Pore, voxel ~40 nm, training source
├── cathode_s2.yaml            # DESCOPED stub (label-map docs only)
├── local_paths.yaml           # (gitignored) machine-specific TIF paths
└── local_paths.yaml.example   # template for other machines
```

- `default()` returns `anode_niysz` — all existing scripts are byte-identical with no flag.
- To switch: `python analyze.py --dataset-config cathode_s1_supercrop ...`
- New datasets = new YAML file, no code changes (per Prof. Jin directive).

### Cube extraction results (0_PRV/extract_cubes.py)

Run on `Segmented_LSCF_GDC_Supercrop.tif` (151×283×120 voxels):

| Config | stride | Accepted | Val | Notes |
|--------|--------|----------|-----|-------|
| S1 str64 | 64 | 6 | 0 | VF drift: GDC +3.7pp, Pore +4.5pp ⚠ |
| S1 str32 | 32 | 24 | 0 | VF drift: LSCF +3.6pp, GDC −5.6pp ⚠ |
| S2 str64/str32 | — | SKIPPED | — | S2 descoped |

Val = 0 in all runs: geometric limitation — Supercrop too small for a val split on stride grid
(longest axis Y=283 voxels; no stride-aligned start at or after y≥219 fits a 64-voxel crop).
VF drift >2pp is expected from spatial inhomogeneity in a small volume. Augmentation
(z-preserving rotations, same as `train_tau_net.py`) is essential before any training run.

Output: `cathode_crops_str64/` and `cathode_crops_str32/` (gitignored). Each structure_XXXX/
contains 64 × (64,64) uint8 BMP slices with values {0,127,255}; `results.dat` and `manifest.csv`.

### Prof. Jin directives — Phase 2 answers (2026-07)

**(a) S2 descoped:** S2 is fully descoped (extra SCT film layer, not representative of
target cathode). S1 Supercrop is the sole cathode training source. Already reflected in
`cathode_s2.yaml` (stub, marked DESCOPED). No further S2 work planned.

**(b) Metric policy — all boundary types (2026-07):** For the LSCF/GDC cathode, track
ALL boundary types: total TPB, active TPB, AND double-phase-boundary (DPB) interface areas.
Oxygen reacts at both TPB and two-phase interfaces in this MIEC cathode system. DPB is ADDED
alongside existing TPB metrics (nothing replaced). Per-phase SSA remains important.
Implementation: `compute_dpb_densities()` in `4_CNNCT/analyze.py`; cathode config has
`dpb_informational: false` and `reaction_pairs: [[LSCF, Pore]]`; anode config has
`dpb_informational: true` (DPB computed but not scored for anode).
Self-check: sum(dpb_A_B for B≠A) == SSA_A × vf_A holds exactly for all phases (verified).

**(c) Voxel size verification directive:** z voxel size is EXPECTED to equal x/y (~40 nm)
but must be verified empirically, not assumed. **RESOLVED by `0_PRV/check_voxel_isotropy.py`:**
- Direct evidence from struct.txt: z = 40.2665 nm; x = 40.340 nm; y = 40.140 nm
- z/x = 0.9982 → voxels are isotropic within 0.2% (confirmed)
- Autocorrelation z/x ≈ 0.80 reflects genuine microstructural anisotropy (FIB pillar
  geometry: structures shorter along the pillar axis = z), NOT a calibration error
- `cathode_s1_supercrop.yaml`: voxel_size_um.z updated from null → 0.040267 µm
- Full report: `0_PRV/VOXEL_ISOTROPY.md`

**(d) More cathode volumes:** Prof. Jin is asking the data provider for additional cathode
volumes. New volumes should be added as new config files (`configs/<name>.yaml`); no code
changes required. The config system is designed for this.

### Open questions (remaining after Phase 2)

- **Anode voxel size**: 0.1 µm (100 nm) is still undocumented — no source found in any
  file. Confirm with Prof. Jin (open question from Phase 1).

### tau-net and SSA surrogate validity

The existing `tau_net.pth` and `2_CNN` SSA estimator were trained on **anode** data (Ni/YSZ/Pore).
They are **NOT valid for cathode training**. Do not pass cathode structures to either surrogate
without retraining on cathode data. The cathode GAN training loss will need new surrogate models
or different auxiliary losses.

### Gitignore additions (Phase 1)

```
configs/local_paths.yaml      # machine-specific TIF paths — never commit
cathode_crops_*/              # extraction output dirs — gitignored like generated_data/
new_dataset_slices.png        # inspection images from Phase 0 recon
new_dataset_matbox.png
```

---

## CATHODE CALIBRATION (2026-07, pre-run0)

> Run by `4_CNNCT/cathode_calibration.py` on the S1 Supercrop crops before any GAN training.
> Establishes baseline metric distributions and real-vs-real S-value ceilings.
> Raw CSVs: `4_CNNCT/cathode_s1_str32_metrics.csv` and `cathode_s1_str64_metrics.csv` (gitignored).

### Metric distributions (str32 training crops, N=24)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| vf_LSCF | 0.207 | 0.098 | 0.072 | 0.490 |
| vf_GDC | 0.156 | 0.050 | 0.099 | 0.309 |
| vf_Pore | 0.637 | 0.102 | 0.411 | 0.795 |
| conn_LSCF | 0.320 | 0.356 | 0.000 | 0.979 |
| conn_GDC | 0.267 | 0.354 | 0.000 | 0.853 |
| conn_Pore | 0.999 | 0.000 | 0.998 | 1.000 |
| total_tpb (µm⁻²) | 0.481 | 0.085 | 0.313 | 0.660 |
| active_tpb (µm⁻²) | 0.024 | 0.061 | 0.000 | 0.239 |
| active_tpb_frac | 0.043 | 0.110 | 0.000 | 0.408 |
| dpb_LSCF_GDC (µm⁻¹) | 0.387 | 0.079 | 0.206 | 0.577 |
| dpb_LSCF_Pore (µm⁻¹) | 2.547 | 0.496 | 1.473 | 3.246 |
| dpb_GDC_Pore (µm⁻¹) | 1.806 | 0.335 | 1.276 | 2.489 |
| dpb_perc_LSCF_Pore (µm⁻¹) | 0.746 | 0.893 | 0.000 | 2.525 |

**Key findings:**
- conn_LSCF and conn_GDC are HIGHLY variable (std ≈ mean) and often 0 in individual crops.
  This is an intrinsic property of the small S1 Supercrop — spatial inhomogeneity means
  some 64³ sub-volumes are LSCF-dominant, others GDC-dominant.
- active_tpb_frac mean=0.043 (4.3%) is very low because conn_LSCF and conn_GDC are often 0.
  Active TPB requires ALL THREE phases to percolate simultaneously.
- Pore connectivity is essentially perfect (0.999) — the generic pore connectivity loss
  (`w_conn=50`) should be sufficient to reproduce this in the generator.

### Real-vs-real S-value ceiling (str32 as "generated", str64 as reference, N_ref=6)

| Metric | Ceiling S-value | Interpretation |
|---|---|---|
| conn_LSCF | 0.927 | OK |
| conn_GDC | 0.861 | OK |
| conn_Pore | 0.903 | OK |
| total_tpb | 0.907 | OK |
| active_tpb | 0.893 | OK |
| active_tpb_frac | 0.887 | OK |
| dpb_LSCF_GDC | 0.898 | OK |
| dpb_LSCF_Pore | 0.855 | OK |
| dpb_perc_LSCF_Pore | 0.910 | OK |
| dpb_GDC_Pore | 0.920 | OK |

**ALL scored metrics have ceiling ≥ 0.855 (OK).** Unlike the anode where tau_YSZ was
intrinsically stuck at 0.658 (real-vs-real, below MARGINAL), no cathode metric is
fundamentally limited by the dataset. A well-trained generator can in principle achieve
OK on every scored metric. (N_ref=6 makes these ceilings noisy — treat as indicative.)

### Calibration targets for future cathode loss terms

- **TPB proxy target (cathode run1+):** The anode target 0.002 (dimensionless probability
  product) was calibrated on anode VFs (~35/30/35 LSCF/GDC/Pore). Cathode VFs are very
  different (21/16/64). Do NOT reuse 0.002. The cathode tpb_proxy target must be
  calibrated empirically by running the proxy on real cathode structures passed through
  the generator's preprocessing. TBD after run0.
- **DPB loss targets (if added):** dpb_LSCF_Pore=2.547 µm⁻¹, dpb_GDC_Pore=1.806 µm⁻¹.
  These are the mean cathode values to target (not yet implemented as loss terms).
- **GDC connectivity loss (analogous to anode YSZ):** conn_GDC mean=0.267 in training crops.
  If the generator collapses GDC connectivity, a min-slice density loss (similar to anode's
  w_conn_ysz) may be needed. Threshold calibration: GDC mean z-slice density = vf_GDC ≈
  0.156; threshold should be ~0.08–0.10 (similar proportion as anode's 0.10 out of ~0.30).

---

## CATHODE RUN0 (2026-07, branch feature/cathode-run0)

> First GAN training run on S1 Supercrop cathode data. Baseline — no cathode-specific losses.
> G_loss = WGAN-GP + 1000×vf + 50×conn(pore). Tau, TPB proxy, YSZ-density all OFF.
> 216 epochs, 6 batches/epoch, checkpoints at 54/108/162/216. ~1296 G-steps.
> Pipeline: `cathode_run0_pipeline.ps1`. Report: `1_GAN/CATHODE_RUN0_REPORT.md`.
> Status: **TRAINING IN PROGRESS** (launched 2026-07-09 14:30, expected done ~16:10).

### Loss configuration

| Term | Weight | Status |
|---|---|---|
| WGAN critic | — | ON |
| VF loss | 1000 | ON |
| Pore connectivity (generic) | 50 | ON |
| YSZ/GDC min-slice density | 200 | OFF (`--no-ysz-density`) |
| YSZ/GDC face-hinge | 200 | OFF (`--no-ysz-density`) |
| Near-TPB proxy | 1000 | OFF (`--no-tpb-proxy`) |
| τ loss | 50 | OFF (no `--tau-estimator`, cathode tau-net not trained) |
| SSA (anode estimator) | 1000 | MONITORING ONLY (timing=9999) |

### S-value results (ep216 final, 50 structures, scored metrics)

| Metric | S-value | |
|---|---|---|
| conn_LSCF | 0.779 | MARGINAL |
| conn_GDC | 0.870 | OK |
| conn_Pore | 0.859 | OK |
| total_tpb | 0.802 | MARGINAL |
| active_tpb | 0.934 | OK |
| active_tpb_frac | 0.852 | OK |
| dpb_LSCF_GDC | 0.734 | MARGINAL |
| dpb_LSCF_Pore | 0.859 | OK |
| dpb_perc_LSCF_Pore | 0.786 | MARGINAL |
| dpb_GDC_Pore | 0.817 | MARGINAL |

**5 OK, 5 MARGINAL, 0 FAIL.** Strong baseline — no scored metric below 0.70 with zero cathode-specific
auxiliary losses. Compare: anode run0 had tau_Ni=0.649 F, tau_YSZ=0.484 F with the same minimal setup.

**Key findings:**
- conn_LSCF (0.779 M) is the primary gap — oscillates, reached 0.901 OK at ep108 but degraded.
- total_tpb (0.802 M) improving monotonically across epochs; a calibrated cathode tpb_proxy would push to OK.
- conn_GDC (0.870 OK) is surprisingly well-matched — cathode GDC is more learnable than anode YSZ.
- Memorization: gen mean 0.562 < baseline 0.593 → no memorization, generator generalizes.
- Best checkpoint: ep108 (7 OK, 2 M, 1 F) vs ep216 (5 OK, 5 M, 0 F) — tradeoff documented in report.

Full details in `1_GAN/CATHODE_RUN0_REPORT.md`.

---
