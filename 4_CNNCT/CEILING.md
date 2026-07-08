# Real-vs-Real S-Value Ceiling Calibration

S-values compare two *finite* samples from the same distribution.
With n≈50 vs 51 real structures, KS variance means S < 1 even
when both samples are drawn from the same distribution. This file measures
the achievable ceiling so run5's scores can be interpreted correctly.

**Method:** split 101 real structures into 51/50 halves, treat one as
'generated', compute S-values. Repeated 20 times (seed=42).
Non-tau metrics computed fresh; tau reused from `tau_labels.csv`
(same code path as `analyze.py`, no discrepancy possible).

---

## Per-metric ceiling (real-vs-real)

| Metric | Mean S | ± Std | Min S | Interpretation | run5 S | Gap |
|--------|--------|-------|-------|---------------|--------|-----|
| conn_Ni | 0.934 | 0.017 | 0.894 | OK -- ceiling reachable | 0.866 | -0.068 |
| conn_YSZ | 0.928 | 0.025 | 0.859 | OK -- ceiling reachable | 0.704 | -0.224 |
| conn_Pore | 0.944 | 0.016 | 0.913 | OK -- ceiling reachable | 0.874 | -0.070 |
| total_tpb | 0.942 | 0.014 | 0.908 | OK -- ceiling reachable | 0.698 | -0.244 |
| active_tpb | 0.930 | 0.020 | 0.870 | OK -- ceiling reachable | 0.720 | -0.210 |
| active_tpb_frac | 0.927 | 0.023 | 0.856 | OK -- ceiling reachable | N/A | N/A |
| tau_Ni | 0.864 | 0.038 | 0.791 | OK -- ceiling reachable | 0.760 | -0.104 |
| tau_YSZ | 0.658 | 0.057 | 0.594 | FAIL -- ceiling below 0.70 (intrinsic limit) | 0.479 | -0.179 |
| tau_Pore | 0.945 | 0.014 | 0.908 | OK -- ceiling reachable | 0.697 | -0.248 |

**Gap** = run5 S-value minus real-vs-real ceiling mean.
Negative gap = run5 scored *below* the real-vs-real ceiling (room to improve).
Positive gap = run5 scored *above* ceiling (sampling variance; run5 closer than random real halves).

---

## Key findings

**tau_YSZ ceiling = 0.658 (FAIL)**
Even real structures compared against *other* real structures only score 0.658 on tau_YSZ.
This means the FAIL label for tau_YSZ is intrinsic to the real dataset's variance, not a
generator failure. No GAN can score OK on tau_YSZ against this dataset because the metric's
S-value ceiling is below MARGINAL. This independently validates the 2026-07 scope decision
to descope tau_YSZ (and tau as a whole) from success criteria.

**conn_YSZ ceiling = 0.928 (OK), run5 = 0.704 (Marginal), gap = -0.224**
Unlike tau_YSZ, conn_YSZ has a reachable OK ceiling. run5 is well below it. The YSZ
re-scope (Task 5) promotes conn_YSZ to the primary YSZ criterion — this gap quantifies the
genuine improvement opportunity.

**active_tpb ceiling = 0.930 (OK), run5 = 0.720 (Marginal), gap = -0.210**
Active TPB is the primary electrochemical performance metric after the descope.
The ceiling is reachable; run5 is 0.21 below it.

**conn_Ni / conn_Pore ceiling ≈ 0.93-0.94, run5 ≈ 0.87**
These are already in the OK band for run5; gap ~0.07. Small but genuine headroom.

---

## How to interpret run5 scores

A metric that scores S=0.70 (FAIL) against run5 may still be good if the ceiling
is S=0.75 — the generator is close to the natural sampling limit, not far from it.
Conversely, a metric with ceiling S=0.95 that run5 scores S=0.70 has genuine headroom.

---

## Bar chart

Regenerate: `conda run -n ganph --no-capture-output python 4_CNNCT/ceiling_calibration.py`
Output: `ceiling_bar.png` (gitignored — contains derived real-structure statistics).
