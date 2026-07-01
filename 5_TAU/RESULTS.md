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

## Planned experiments

| exp_id | what | hypothesis | status |
|---|---|---|---|
| run3 | distribution-matching loss | broader spread → better KS | DONE — FAILED |
| run4 | YSZ face-hinge at z=0/z=63 | face presence → percolation | DONE — no tau_YSZ improvement |
| run5 | TPB proxy loss | total_tpb stuck at 0.69-0.72 F; near_tpb = Ni×YSZ×Pore | NEXT |
| run6 | Weighted tau loss (YSZ 3×) | more gradient signal to YSZ | PLANNED |
