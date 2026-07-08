# 5_TAU Validation Report

Covers the accuracy and precision of the taufactor tortuosity pipeline,
per Prof. Jin's Week 5 directive.

---

## Precision vs accuracy framing

| Term | Definition | Status |
|------|-----------|--------|
| **Precision** | Strict re-solves reproduce `tau_labels.csv` values on the same structures | Verified (2026-07): convergence sensitivity test on 5 worst tau_YSZ structures; max delta 3.2% on one non-converged case, 0.0% on the rest. See `RESULTS.md § Task 1 findings`. |
| **Accuracy** | Pipeline produces TRUE tau where ground truth is analytically known | Verified by `validate_taufactor.py` (5/5 analytic cases PASS). **Accuracy on real FIB-SEM structures requires reference values from Prof. Jin** — see `reference_tau.csv` and `compare_reference_tau.py`. |

The prior label audit tested **precision**. Prof. Jin's directive asks for **accuracy**. These are different questions: a solver can be perfectly self-consistent (same answer every time) and still be systematically wrong if the boundary conditions, phase labeling, or solve axis are misconfigured.

---

## Analytic case results (`validate_taufactor.py`)

Run: 2026-07 on GAN-PH machine (CPU, taufactor installed in `ganph` conda env).

| Case | Description | Expected | Result | Status |
|------|-------------|----------|--------|--------|
| 1 | Fully dense 64³ Ni cube | tau = 1.000 ± 0.001 | 1.000000 | **PASS** |
| 2 | Straight Ni columns along z (stride 4) | tau = 1.000 ± 0.010 | 1.000000 | **PASS** |
| 3 | Pore wall at z=32 severs Ni | NaN | NaN | **PASS** |
| 4 | AXIS CHECK: Ni lines along x (not z) | NaN | NaN | **PASS** |
| 5 | CONVENTION: Pore(val=0) channels | tau≈1.0 (correct call); NaN (NI_VAL call) | 1.000000 / NaN | **PASS** |

**Conclusions from analytic cases:**

- The solver axis is **z (axis 0)** — confirmed by Case 4. taufactor solves along the electrode thickness direction, consistent with the physical problem.
- The percolation check in `analyze.py` (`conn == 0.0 → return NaN`) correctly catches non-percolating phases before calling taufactor — no undefined-behavior solve on disconnected phases.
- Phase labeling works for all three values (0, 127, 255) — the `(vol == phase_val).astype(float32)` extraction is convention-agnostic as long as the caller passes the correct `phase_val`. Swapping phase values without updating calls gives NaN (graceful failure, not silent wrong answer) when the target phase is absent.

**taufactor CUDA warning:** on CPU-only machines, taufactor emits
`UserWarning: CUDA not available, defaulting device to cpu`.
This is expected behavior and does not affect results. To suppress it, pass
`device='cpu'` explicitly to `taufactor.Solver(...)` — a one-line change
to `analyze.py:compute_tortuosity` if desired.

---

## Reference comparison (pending Prof. Jin values)

**Status: PENDING.** `reference_tau.csv` contains only the template header row.

When Prof. Jin provides reference tortuosity values:
1. Fill in `reference_tau.csv` with the structure ID, phase, reference tau, and source note.
2. Run:
   ```bash
   conda run -n ganph --no-capture-output python compare_reference_tau.py
   ```
3. The script looks up matching rows from `tau_labels.csv` (computed by `compute_tau_labels.py`)
   and reports per-row absolute and relative error.

**Accuracy thresholds (proposed):**
- Mean relative error < 5%  → GOOD; pipeline is accurate on real structures.
- Mean relative error 5–15% → MARGINAL; check worst residuals for systematic bias.
- Mean relative error > 15% → POOR; investigate phase labeling, solve axis, or boundary conditions.

---

## Tau-net scatter plot (`plot_taunet_scatter.py`)

Plots tau-net prediction vs taufactor label on the held-out validation set.
Reconstructs the exact train/val split from `train_tau_net.py` (seed=42, val_frac=0.2,
structure-level split before augmentation) to prevent data leakage.

**To generate the scatter plot** (requires `real_data/` and `tau_labels.csv` locally):
```bash
conda run -n ganph --no-capture-output python plot_taunet_scatter.py \
    --data ../real_data --labels tau_labels.csv --model save_model/tau_net.pth
```

Output: `taunet_scatter.png` (gitignored — contains real structure data).

**Recorded val MSE: 0.094** (log-tau scale, from `train_tau_net.py` training run).
The scatter plot should reproduce this MSE on the same val split. A significant
discrepancy would indicate the split is not being reconstructed correctly.

---

## What this validation does NOT cover

1. **SSA estimator (2_CNN):** The SSA pipeline accuracy has not been validated. The SSA
   loss is permanently disabled (timing=9999) so this is informational only.
2. **Tau-net accuracy on generated structures:** tau-net was trained on real binary
   structures. When applied to the GAN's soft-probability outputs during training, it sees
   OOD inputs. This is acceptable for a training signal but means tau-net predictions on
   generated structures are less accurate than on real ones.
3. **tau_YSZ accuracy:** 5 of 101 real structures have tau_YSZ=NaN (YSZ non-percolating).
   The tau_labels.csv values for the remaining 96 are confirmed precise (convergence
   sensitivity test), but accuracy against a physical reference is pending.
