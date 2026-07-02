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
| run8 | tpb-proxy | 5dad25a | Extended training: 100 epochs, run5 (tpb-proxy) settings, SSA monitoring-only (timing=9999). | 0.718 M | 0.459 F | 0.718 M | 0.836 M | 0.677 F | 0.880 OK | 0.670 F | 0.676 F | 100 epochs over-converges auxiliary losses → reduced diversity. tau_Pore is the only metric that improves vs run5 (0.697→0.718 M). All other metrics regress. |
| run9 | run9-tpb-on-run2 | 94c610c | run2 base + tpb_proxy (w=1000, target=0.002), NO face-hinge. 50 epochs. | 0.774 M | 0.464 F | 0.699 F | 0.881 OK | 0.681 F | 0.882 OK | 0.667 F | 0.684 F | Without face-hinge: tau_Ni recovers vs run5 (0.760→0.774). But total_tpb WORSE (0.698→0.667 F). Face-hinge adds diversity that helps KS metrics even though it competes with tau gradient. tau_Pore 0.001 below 0.70 threshold. |
| run10 | tpb-proxy | 5dad25a | Extended training: 65 epochs, same as run5 but 15 more epochs. Hypothesis: 65ep sweet spot between run5 (50ep) and run8 (100ep). | 0.716 M | 0.461 F | 0.657 F | 0.833 M | 0.685 F | 0.859 OK | 0.653 F | 0.682 F | REGRESSION vs run5 on all metrics. 65 epochs is WORSE than both 50 and 100 epochs — non-monotonic training dynamics. 50 epochs remains the sweet spot for tpb-proxy. |
| run11 | run11-half-face-hinge | ace571a | tpb-proxy, 50 epochs, w_conn_ysz_face=100 (halved from 200). Hypothesis: face-hinge at 200 competes with tau gradient causing tau_Ni regression. | 0.682 F | 0.468 F | 0.686 F | 0.811 M | 0.693 F | 0.861 OK | 0.648 F | 0.697 F | SEVERE REGRESSION. tau_Ni=0.682 F (was 0.760 M). Face-hinge at w=200 is a stable equilibrium, not a bottleneck — reducing it disrupts the balance catastrophically. |
| run12 | run12-lower-tau-weight | 7d90c1e | tpb-proxy, 50 epochs, w_tau=20 (global, all phases). Hypothesis: tau over-convergence at epoch 65-75 causes valley; lower weight keeps tau unconverged. | 0.550 F | 0.473 F | 0.749 M | 0.745 M | 0.693 F | 0.891 OK | 0.676 F | 0.697 F | Phase-specific finding: tau_Pore improved 0.697→0.749 M (Pore over-constraint removed) but tau_Ni crashed 0.760→0.550 F (Ni tau signal too weak). Real Pore tau is very narrow (std_log=0.0319); MSE loss collapses Pore variance. |
| run13 | run13-ni-tau-only | 53670ff | tpb-proxy, 50 epochs, tau loss Ni-only (YSZ+Pore disabled). Hypothesis: keep Ni tau signal, remove Pore over-constraint. | 0.560 F | 0.462 F | 0.819 M | 0.740 M | 0.696 F | 0.899 OK | 0.601 F | 0.675 F | tau_Pore=0.819 M (best ever!). But tau_Ni still crashed to 0.560 F. Root cause: with only 1 phase, tau loss fully converges (0.31→0.006) → all Ni samples same tortuosity → KS FAIL. In run5, 3-phase competition prevents any single phase from fully converging. |
| run14 | run14-weighted-pore-tau | 2d21b73 | tpb-proxy, 50 epochs, per-phase tau weights: Ni=1.0, YSZ=1.0, Pore=0.05. Restore 3-phase gradient competition to prevent tau_Ni over-convergence; Pore barely constrained. | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | IN PROGRESS — hypothesis: tau_Ni recovers to ~0.760 M (multi-phase competition preserved) AND tau_Pore improves toward 0.75 M (Pore barely constrained). First attempt to get both simultaneously ≥0.70. |

**Legend:** F=FAIL(<0.70), M=MARGINAL(0.70–0.85), OK(≥0.85)
**Best baseline for tau_Ni:** run2 (0.818 M). **Best overall (tpb-proxy branch):** run5 — 50 epochs. **Best tau_Pore:** run13 (0.819 M). **Best tau_Pore + tau_Ni simultaneously:** TBD (run14 in progress).

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

## Epoch sweep findings (tpb-proxy branch)

| epochs | tau_Ni | tau_YSZ | tau_Pore | conn_Ni | total_tpb | active_tpb |
|--------|--------|---------|---------|---------|-----------|------------|
| 50 (run5) | 0.760 M | 0.479 F | 0.697 F | 0.866 OK | 0.698 F | 0.720 M |
| 65 (run10) | 0.716 M | 0.461 F | 0.657 F | 0.833 M | 0.653 F | 0.682 F |
| 100 (run8) | 0.718 M | 0.459 F | 0.718 M | 0.836 M | 0.670 F | 0.676 F |

**Finding:** 50 epochs is unambiguously the sweet spot. 65 epochs is the worst across nearly all metrics — training goes through a "valley" where the auxiliary losses have over-converged but the adversarial diversity hasn't recovered. 100 epochs partially recovers tau_Pore (0.697→0.718 M) but sacrifices conn_Ni and active_tpb. No epoch count between 50 and 100 produces a better overall profile than 50 epochs.

---

## Planned experiments

| exp_id | what | hypothesis | status |
|---|---|---|---|
| run8 | 100 epochs, tpb-proxy settings | tau_loss still converging at epoch 50 | DONE — tau_Pore only metric that improved |
| run9 | run2 base + tpb_proxy, no face-hinge, 50 epochs | run2 had best tau_Ni; tpb_proxy adds TPB signal | DONE — tau_Ni up (0.760→0.774) but total_tpb down (0.698→0.667) |
| run10 | tpb-proxy, 65 epochs | sweet spot between 50 and 100 | DONE — REGRESSION, 50ep remains best |
| run11 | tpb-proxy, 50 epochs, w_conn_ysz_face=100 (half) | face-hinge at 200 competes with tau gradient | DONE — SEVERE REGRESSION, tau_Ni=0.682 F. Face-hinge at 200 is load-bearing. |
| run8@60 | tpb-proxy checkpoint at epoch 60 (no new training) | find tau_Pore crossover point | DONE — tau_Ni=0.603 F, tau_Pore=0.683 F, tau_YSZ=0.509 F (best YSZ ever!). Valley confirmed. |
| run8@75 | tpb-proxy checkpoint at epoch 75 (no new training) | where tau_Pore first hits 0.70 | DONE — tau_Ni=0.566 F, tau_Pore=0.725 M. tau_Pore crosses 0.70 only when tau_Ni is deep FAIL. |
| run12 | tpb-proxy, 50 epochs, w_tau=20 (global) | prevent tau over-convergence via lower weight | DONE — tau_Pore 0.697→0.749 M but tau_Ni 0.760→0.550 F. Phase-specific finding: Pore tau over-constrains variance. |
| run13 | tpb-proxy, 50 epochs, Ni-only tau loss | remove Pore tau over-constraint entirely | DONE — tau_Pore=0.819 M (best ever!), tau_Ni=0.560 F (still crashed). Root cause: Ni-only causes full Ni tau convergence (tau_loss: 0.31→0.006) → variance collapse. 3-phase competition in run5 was protective. |
| run14 | tpb-proxy, 50 epochs, Ni=1.0 YSZ=1.0 Pore=0.05 tau weights | restore 3-phase gradient competition (prevents Ni over-convergence) while barely constraining Pore (prevents Pore variance collapse) | IN PROGRESS — if hypothesis correct: tau_Ni~0.760 M AND tau_Pore~0.75 M simultaneously for first time |
