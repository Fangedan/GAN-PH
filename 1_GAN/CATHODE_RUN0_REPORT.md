# Cathode Run 0 — Morning Report

> Branch: feature/cathode-run0  
> Dataset: S1 Supercrop (LSCF/GDC/Pore, 64³ voxels, ~40 nm voxel)  
> Training set: 24 crops (stride-32 TRAIN region)  
> Reference set: 6 crops (stride-64, same TRAIN region — val split geometrically impossible)  
> Config: cathode_s1_supercrop  

---

## Step budget

| Quantity | Value |
|---|---|
| Training crops | 24 |
| Batch size | 4 |
| Batches/epoch | 6 |
| Epochs | 216 |
| Total G-steps | 1296 |
| Anode run5 ref | 1300 G-steps (26 × 50) |
| Checkpoints | epochs 54 / 108 / 162 / 216 |

---

## Loss configuration

| Term | Weight | Status |
|---|---|---|
| WGAN critic | — | ON |
| VF loss | 1000 | ON (generic) |
| Pore connectivity (generic) | 50 | ON |
| YSZ min-slice density | 200 | OFF (`--no-ysz-density`) |
| YSZ face-hinge | 200 | OFF (`--no-ysz-density`) |
| Near-TPB proxy | 1000 | OFF (`--no-tpb-proxy`) |
| τ loss | 50 | OFF (no `--tau-estimator`) |
| SSA (anode estimator) | 1000 | MONITORING ONLY (timing=9999) |

---

## S-value scorecard

> Fill after run. S ≥ 0.85 = OK, 0.70–0.85 = MARGINAL, < 0.70 = FAIL.
> Tau is informational only (cathode config: `tau_scored: false`).
> Reference population: 6 stride-64 crops (small N → KS test has low power; treat as indicative only).

### Checkpoint progression

| Epoch | conn_LSCF | conn_GDC | conn_Pore | total_tpb | active_tpb |
|---|---|---|---|---|---|
| 54   | — | — | — | — | — |
| 108  | — | — | — | — | — |
| 162  | — | — | — | — | — |
| **216 (final)** | — | — | — | — | — |

### DPB metrics (final, cathode-specific, scored)

| Metric | S-value |
|---|---|
| dpb_LSCF_GDC | — |
| dpb_LSCF_Pore | — |
| dpb_GDC_Pore | — |
| dpb_perc_LSCF_Pore | — |

### Tau metrics (final, informational only)

| Metric | S-value |
|---|---|
| tau_LSCF | — |
| tau_GDC  | — |
| tau_Pore | — |

---

## Memorization check

> Fill after run. Source: `4_CNNCT/cathode_run0_memo.csv`.
> Expected: gen→train mean ≈ train→train baseline (overlapping crops have high baseline).

| Metric | Value |
|---|---|
| gen → nearest train (mean agreement) | — |
| gen → nearest train (std) | — |
| train → nearest OTHER train baseline (mean) | — |
| train → nearest OTHER train baseline (std) | — |
| gen_mean − baseline_mean | — |
| Verdict | — |

---

## Key observations

> Fill after run.

- [ ] Pore connectivity converged? (baseline check — run0 with generic pore loss only)
- [ ] LSCF connectivity?
- [ ] GDC percolation? (historically hardest: anode YSZ was stuck 0.46–0.48 across 14 runs)
- [ ] Memorization verdict passes? (gen_mean − baseline_mean < 0.05)
- [ ] Any loss divergence / NaN in training log?

---

## Next steps

> Fill after run.

- If conn_Pore fails: consider increasing w_conn or adding GDC density loss
- If LSCF/GDC connectivity fails: analogous to anode YSZ — may need differentiable flood-fill
- If memorization detected: reduce overlap (use stride-64 training set) or add augmentation
- If total_tpb fails: cathode-calibrated tpb_proxy loss (need to measure cathode TPB first)

---

## Notes on dataset limitations

- **0 val crops**: Y=283 voxels, val_start=219 — no stride-aligned crop fits in [219, 283).
  Reference population is 6 stride-64 TRAIN crops (non-overlapping but not held-out).
  KS test with N_ref=6 has low statistical power — S-values are indicative, not definitive.
- **VF drift**: stride-32 crops show GDC −5.6pp drift vs parent volume (spatial inhomogeneity).
  Generator is conditioned on crop VFs, not parent VF. Monitor generated vs training VFs.
- **No tau surrogate**: anode tau_net.pth was trained on Ni/YSZ/Pore — invalid for LSCF/GDC/Pore.
  Tau values computed by taufactor (the actual solver), not the surrogate. Informational only.
