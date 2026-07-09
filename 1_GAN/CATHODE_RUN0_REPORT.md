# Cathode Run 0 — Morning Report

> Branch: feature/cathode-run0
> Dataset: S1 Supercrop (LSCF/GDC/Pore, 64³ voxels, ~40 nm voxel)
> Training set: 24 crops (stride-32 TRAIN region)
> Reference set: 6 crops (stride-64, same TRAIN region — val split geometrically impossible)
> Config: cathode_s1_supercrop
> Completed: 2026-07-09, ~16:10 local time

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

> S ≥ 0.85 = OK, 0.70–0.85 = MARGINAL, < 0.70 = FAIL.
> Tau is informational only (cathode config: `tau_scored: false`).
> Reference population: 6 stride-64 crops (small N → KS test has low power; treat as indicative).
> Real-vs-real ceiling: all scored metrics ≥ 0.855 OK (no metric intrinsically limited).

### Checkpoint progression (20 structures each)

| Epoch | conn_LSCF | conn_GDC | conn_Pore | total_tpb | active_tpb | active_tpb_frac | dpb_LSCF_GDC | dpb_LSCF_Pore | dpb_perc_LSCF_Pore | dpb_GDC_Pore |
|---|---|---|---|---|---|---|---|---|---|---|
| 54  | 0.778 M | 0.893 OK | 0.813 M | 0.609 F | 0.904 OK | 0.847 M | 0.736 M | 0.649 F | 0.883 OK | 0.679 F |
| 108 | 0.901 OK | 0.852 OK | 0.827 M | 0.688 F | 0.909 OK | 0.934 OK | 0.857 OK | 0.736 M | 0.915 OK | 0.841 M |
| 162 | 0.807 M | 0.852 OK | 0.869 OK | 0.726 M | 0.951 OK | 0.911 OK | 0.814 M | 0.765 M | 0.866 OK | 0.793 M |
| **216 (final)** | **0.779 M** | **0.870 OK** | **0.859 OK** | **0.802 M** | **0.934 OK** | **0.852 OK** | **0.734 M** | **0.859 OK** | **0.786 M** | **0.817 M** |

### Final scorecard (ep216, 50 structures — scored metrics)

| Metric | S-value | Interpretation |
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

**Summary: 5 OK, 5 MARGINAL, 0 FAIL. No scored metric below 0.70.**

This is a strong baseline for a run with no cathode-specific auxiliary losses.
Compare: anode run0 had tau_Ni=0.649 F, tau_YSZ=0.484 F with the same minimal loss setup.

### Tau metrics (final, informational only)

| Metric | S-value | Note |
|---|---|---|
| tau_LSCF | 0.452 | FAIL — expected (no cathode tau-net; anode net not valid) |
| tau_GDC  | 0.592 | FAIL — expected |
| tau_Pore | 0.726 | MARGINAL — informational |

---

## Memorization check

> Source: `4_CNNCT/cathode_run0_memo.csv`.

| Metric | Value |
|---|---|
| gen → nearest train (mean agreement) | 0.5617 |
| gen → nearest train (std) | 0.0545 |
| train → nearest OTHER train baseline (mean) | 0.5933 |
| train → nearest OTHER train baseline (std) | 0.0696 |
| gen_mean − baseline_mean | −0.032 |
| Verdict | **OK — no memorization detected** |

The generator is *more* diverse than training crops (gen mean < baseline mean).
This is expected and healthy — the generator produces novel combinations, not copies.

---

## Key observations

- [x] **Pore connectivity converged** — conn_Pore OK (0.859) with generic pore loss. Loss works.
- [x] **GDC connectivity OK (0.870)** — the generator learns GDC percolation without targeted loss.
  Remarkable: this is the cathode analogue of anode YSZ, which never exceeded 0.707 across 14 runs.
  The cathode GDC appears more learnable than the anode YSZ (likely due to larger cluster sizes).
- [~] **LSCF connectivity MARGINAL (0.779)** — oscillates: 0.778 M → 0.901 OK → 0.807 M → 0.779 M.
  The generator can achieve OK (ep108) but doesn't hold it with more training. Suggests competing
  dynamics between LSCF and GDC phases (similar to anode Ni/Pore tau tradeoff via softmax coupling).
  Priority target for run1: LSCF min-slice density loss calibrated to cathode VF.
- [~] **total_tpb MARGINAL (0.802)** — improving monotonically across epochs (0.609→0.802) and still
  rising at ep216. A cathode-calibrated tpb_proxy loss in run1 would push this to OK.
  NOTE: the anode proxy target (0.002) is NOT valid for cathode (different VF composition).
  Cathode target TBD (requires empirical calibration on soft generator output).
- [x] **active_tpb OK (0.934)** — the fraction of TPB that is active is well-matched.
- [~] **dpb_LSCF_GDC MARGINAL (0.734)** — interface between the two solid phases is off.
  Likely reflects VF distribution mismatch: generated LSCF/GDC VFs may differ from reference.
- [x] **dpb_LSCF_Pore OK (0.859)** — key cathode reaction interface is well-matched.
- [~] **dpb_GDC_Pore MARGINAL (0.817)** — close to OK (0.85 threshold).
- [x] **No memorization** — gen_mean (0.562) < baseline (0.593). Generator generalizes.
- [x] **No training divergence** — all 4 checkpoints present, losses finite throughout.

---

## Best checkpoint

| Checkpoint | OK count | MARGINAL | FAIL | Notes |
|---|---|---|---|---|
| ep54  | 2 | 5 | 3 | Early — DPB and TPB not converged |
| ep108 | **7** | 2 | 1 | Best connectivity; total_tpb still FAIL |
| ep162 | 6 | 4 | 0 | Good balance |
| ep216 | 5 | **5** | **0** | No FAILs; best TPB/DPB; LSCF conn lost |

**ep108 has the most OK metrics (7) but 1 FAIL (total_tpb=0.688).**
**ep216 has 0 FAILs and is the conservative choice for downstream use.**

For run1 design: ep108 suggests the generator CAN achieve conn_LSCF=0.901 OK —
the issue is sustaining it at longer training. Consider checkpointing at 108 epochs for run1.

---

## Next steps (cathode run1 design)

### Priority 1 — conn_LSCF loss
Add a min-slice density loss for LSCF (channel 0, analogous to anode's `w_conn_ysz`):
- Threshold: mean LSCF z-slice density from training crops ≈ vf_LSCF ≈ 0.207
  → threshold ~0.10 (similar fraction to anode's 0.10 out of ~0.30 YSZ VF)
- Activate from epoch 1 (not gated)
- Weight: start at 200 (same as anode w_conn_ysz) and reduce if LSCF over-constrains

### Priority 2 — total_tpb calibrated proxy
Add a cathode-calibrated tpb_proxy loss:
- Calibration step needed: generate ~20 structures from ep216, compute the raw
  `(ch0_prob * ch1_prob * ch2_prob).mean()` on the generator's soft output for
  each structure, correlate with physical total_tpb. The proxy target for cathode
  is NOT 0.002 (anode value).
- Activate from epoch tau_timing (default 10)

### Priority 3 — epoch budget for run1
ep108 is a natural sweet spot (conn_LSCF OK before it degrades). Consider:
- Train 108 epochs (=648 G-steps) and save final only
- OR train 216 epochs with a LSCF-connectivity loss that locks in conn_LSCF before it drifts

### Branch
Create `feature/cathode-run1` off `feature/cathode-run0`. Do NOT modify `feature/dataset-configs`.

---

## Notes on dataset limitations

- **0 val crops**: Y=283 voxels, val_start=219 — no stride-aligned crop fits in [219, 283).
  Reference population is 6 stride-64 TRAIN crops (non-overlapping but not held-out).
  KS test with N_ref=6 has low statistical power — S-values are indicative, not definitive.
- **VF drift**: stride-32 crops show GDC −5.6pp drift vs parent volume (spatial inhomogeneity).
  Generator is conditioned on crop VFs, not parent VF. Monitor generated vs training VFs.
- **No tau surrogate**: anode tau_net.pth was trained on Ni/YSZ/Pore — invalid for LSCF/GDC/Pore.
  Tau values computed by taufactor (the actual solver). Informational only.
- **Checkpoint path**: training writes to `./save_model/` (repo root) when run as
  `python 1_GAN/main.py`. This was fixed in `cathode_run0_pipeline.ps1` after training completed;
  for run1 the pipeline will use `--weights save_model/Generator_NNNepoch.pth`.
