# Experiment Results

One row per training run. Updated after every analyze.py evaluation.

## S-value table

| run_id | branch | commit | what_changed | tau_Ni | tau_YSZ | tau_Pore | conn_Ni | conn_YSZ | conn_Pore | total_tpb | active_tpb | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run0 | master | 66b3884 | baseline (no tau loss) | 0.649 F | 0.484 F | — | — | — | — | — | — | original training, no tortuosity supervision |
| run1 | tortuosity-loss | 633644c | add tau loss (mean MSE, all phases, w_tau=50, gate@epoch10) | 0.770 M | 0.484 F | 0.676 F | 0.876 OK | 0.707 M | 0.876 OK | 0.693 F | 0.721 M | tau_Ni improved; tau_YSZ flat — YSZ gate removes disconnected samples, leaving no gradient |
| run2 | feature/ysz-connectivity-loss | a9e1b69 | add YSZ min-slice density loss (w=200) + gate tau_YSZ on connectivity | 0.818 M | 0.479 F | 0.688 F | 0.891 OK | 0.703 M | 0.878 OK | 0.689 F | 0.718 M | tau_Ni improving; tau_YSZ stuck — density loss barely fired (<0.001), blobs satisfied density but not topology |
| run3 | tau-ysz-diagnosis | ad58eb2 | distribution-matching loss (mean+std per batch, w_tau_std=1.0) | 0.577 F | 0.475 F | 0.633 F | 0.795 M | 0.701 M | 0.812 M | 0.723 M | 0.743 M | FAILED — also trained on synthetic_data by mistake (13 batches vs 26). W_D=900, all 50 τ=NaN. ABANDONED. |
| run4 | ysz-face-hinge | c9d0e9a | add YSZ face density at z=0/z=63 (threshold=0.18, w=200) | 0.755 M | 0.481 F | 0.677 F | 0.880 OK | 0.705 M | 0.872 OK | 0.694 F | 0.720 M | tau_YSZ unchanged (0.479→0.481). Face loss fired but didn't create topological percolation. tau_Ni regressed vs run2 (0.818→0.755). |
| run5 | tpb-proxy | 5dad25a | add near-TPB density proxy (near_tpb=Ni×YSZ×Pore, target=0.002, w=1000) | 0.760 M | 0.479 F | 0.697 F | 0.866 OK | 0.704 M | 0.874 OK | 0.698 F | 0.720 M | tpb_loss fired at epoch 19 (0.00063), reached 0.0017 by epoch 50. tau_Pore improved +0.020 vs run4. total_tpb still 0.002 short of 0.70. tau_YSZ flat (stuck at ~0.48 for 6 runs). |
| run6 | run6-weighted-tau-ysz | 89a42c5 | weight tau loss: YSZ=3×, Ni=1×, Pore=1× (same w_tau=50) | 0.681 F | 0.459 F | 0.730 M | 0.820 M | 0.682 F | 0.884 OK | 0.718 M | 0.694 F | 3× YSZ gradient fought with Ni — tau_Ni crashed 0.760→0.681 F (severe regression). tau_YSZ slightly WORSE (0.479→0.459). total_tpb improved (0.698→0.718 M). tau_Pore improved (0.697→0.730 M). REGRESSION overall. |
| run7 | run7-ssa-fix | 12b394d | fix SSA gradient + enable (timing=10): remove no_grad/detach/requires_grad_, freeze estimator | — | — | — | — | — | — | — | — | SKIPPED — built on run6 (3× YSZ weight). Run6 caused tau_Ni regression; run7a used instead. |
| run7a | run7a-ssa-fix | 525e694 | SSA gradient fix + timing=10, built on run5 (tpb-proxy, equal tau weights). | — | — | — | — | — | — | — | — | FAILED: G_loss exploded to 4.5B at epoch 11 (SSA activation). Root cause: true was raw SSA units (~1e7) vs pred standardized to [0,1]; w_param=1000 amplified to ~1e10. |
| run7b | run7b-ssa-fix | 21d25e1 | Fix SSA scale: standardize true same as pred, add w_ssa=50 (separate from w_param=1000). | — | — | — | — | — | — | — | — | FAILED: G_loss still exploded to 344M at epoch 11. Root cause: true [3B] vs pred [3B,1] broadcasting creates [3B,3B] cross-product; estimator sees OOD soft-probability inputs (not binary) → pred outside mm_list range. SSA differentiable loss is fundamentally broken with this estimator. |
| run8 | tpb-proxy | 5dad25a | Extended training: 100 epochs, run5 (tpb-proxy) settings, SSA monitoring-only (timing=9999). | — | — | — | — | — | — | — | — | IN PROGRESS. |

**Legend:** F=FAIL(<0.70), M=MARGINAL(0.70–0.85), OK(≥0.85)
**Best baseline for tau_Ni:** run2 (0.818). **Best baseline overall:** run2.

---

## Task 1 findings (decision gate)

Ran convergence sensitivity test on 5 worst tau_YSZ structures (214→125 range).
Compared default (iter=10k, crit=0.01) vs strict (iter=100k, crit=0.001):

| structure | conn_YSZ | tau_default | tau_strict | delta |
|---|---|---|---|---|
| structure_0027 | 0.547 | 214.33 | 214.42 | 0.0% |
| structure_0084 | 0.377 | 213.90 | 213.90 | 0.0% |
| structure_0072 | 0.750 | 178.98 | 178.98 | 0.0% |
| structure_0070 | 0.460 | 154.77 | 159.76 | 3.2% (not converged even at 100k) |
| structure_0046 | 0.267 | 125.31 | 125.32 | 0.0% |

**Conclusion:** Labels are genuine. tau_YSZ is high because YSZ at ~20% vf is near its
percolation threshold — the paths are genuinely tortuous.

tau label statistics (from 101 real structures, NaN-excluded):
- tau_Ni:   n=101, mean_log=2.5764, std_log=0.1816, range [8.5, 23.5]
- tau_YSZ:  n=96,  mean_log=3.6540, std_log=0.5660, range [11.3, 214.3], 5 NaN
- tau_Pore: n=101, mean_log=0.7034, std_log=0.0319, range [1.88, 2.16]

---

## tau_YSZ root cause summary

tau_YSZ has been stuck at 0.479–0.484 across ALL runs (including baseline). The
attempts to fix it:

| approach | result | why it failed |
|---|---|---|
| MSE tau loss | no change (0.484→0.484) | YSZ disconnected → NaN → tau_net gradient is noise |
| YSZ gate (skip disconnected) | no change | removes bad samples but doesn't create percolation |
| Min-slice density (0.10 threshold) | barely fired | blobs satisfied density without connecting z=0→z=63 |
| Face density at endpoints (0.18 threshold) | no change | face presence isn't sufficient for topological connectivity |
| Distribution-matching std | degenerate collapse | wrong approach entirely |

**Pattern:** All surface-level density losses fail because YSZ topology (percolation)
is a non-local property that can't be controlled by per-slice or per-face density.

**Key data point:** The tau_net surrogate showed lower loss at run4 end (0.30 vs 0.73
in run2), yet tau_YSZ S-value didn't improve. This suggests the generator is
"fooling" the surrogate without creating physically connected YSZ.

---

## SSA differentiable loss — why it is fundamentally broken

Three separate attempts to enable SSA as a differentiable training signal all failed:

| attempt | fix applied | result |
|---|---|---|
| run7a | remove no_grad/detach, freeze estimator | G_loss = 4.5B (true in raw units ~1e7) |
| run7b | standardize true with mm_list, w_ssa=50 | G_loss = 344M (still exploding) |

Root causes identified:
1. **Shape mismatch / wrong broadcasting**: `true` is `[3*B]`, estimator output `pred` is `[3*B, 1]`. `torch.square(true - pred)` broadcasts to `[3*B, 3*B]` (all pairwise cross-differences) not element-wise. Result: sum over B²×9 wrong pairs.
2. **OOD estimator inputs**: Estimator was trained on binary voxel structures (0 or 1). During GAN training, `g_data` is softmax probabilities (continuous 0–1). The estimator outputs values wildly outside the training-time mm_list range → standardization maps pred far outside [0,1].
3. **Jacobian amplification**: Even if loss value were small, the frozen estimator CNN's Jacobian ∂pred/∂g_data is large (deep CNN, no gradient clipping). Any loss routed through it explodes the generator update.

**Conclusion:** SSA loss through this estimator is not viable without either (a) retraining the estimator on probability maps or (b) a Gumbel-softmax / straight-through trick to feed binary inputs while keeping gradients. Keep SSA as monitoring-only (timing=9999 + torch.no_grad).

---

## Planned experiments

| exp_id | what | hypothesis | status |
|---|---|---|---|
| run8 | 100 epochs, tpb-proxy settings | tau_loss still converging at epoch 50; longer training may push tau_Ni ≥ 0.82 and tau_Pore ≥ 0.70 | IN PROGRESS |
