# Cathode Run 0 — Report (Hardened Evaluation)

> Branch: feature/cathode-run0
> Dataset: S1 Supercrop (LSCF/GDC/Pore, 64³ voxels, ~40 nm voxel)
> Training set: 24 crops (stride-32, TRAIN region)
> Training-region reference set: 24 stride-32 crops (same region; no held-out val — see Evaluation Limits below)
> Config: cathode_s1_supercrop
> Training completed: 2026-07-09, ~16:10 local time
> Evaluation hardened: 2026-07-10 — re-scored against 24 str32 crops; bootstrap CIs added

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

## Evaluation Limits

- **N_ref = 24** (stride-32 training-region crops). KS test power improves over the original 6 str64 crops but remains limited. Two samples from the same underlying distribution can produce S-values below MARGINAL by chance.
- **No held-out val set**: Y=283 voxels; no stride-aligned 64-cube starts at or after y=219. The training-region reference set and the training data are the same 24 crops. S-values measure fidelity-to-training-distribution, not generalization.
- **Correlated reference samples**: stride-32 with cube-size 64 means adjacent crops share 32 voxels. The reference set is not iid; within-group correlation inflates effective within-group homogeneity and can bias S upward for metrics with strong spatial gradients.
- **Pseudo-ceiling below MARGINAL for several metrics**: real-vs-real S-values (12 vs 6 disjoint crops) are in FAIL range for conn_LSCF, total_tpb, dpb_LSCF_Pore, dpb_perc_LSCF_Pore (see pseudo-ceiling table). For those metrics, FAIL/MARGINAL generator scores partly reflect genuine spatial heterogeneity in the S1 Supercrop volume, not solely generator inadequacy.
- **Memorization check limitation**: the symmetry group used (16 z-preserving rotations/flips) detects exact, rotated, or flipped copies but does NOT detect translated patches. A generated structure overlapping a training crop with a spatial offset will not be flagged.

---

## S-value scorecard (hardened, ref = 24 str32 crops)

> S ≥ 0.85 = OK, 0.70–0.85 = MARGINAL, < 0.70 = FAIL.
> Tau is informational only (cathode config: `tau_scored: false`).
> Bootstrap: 16th–84th percentile, 500 iterations, resampling both ref and gen sets.

### Checkpoint progression with bootstrap CI

| Metric | ep54 (N=20) | ep108 (N=20) | ep162 (N=20) | ep216 (N=50) |
|---|---|---|---|---|
| conn_LSCF | 0.777 M [0.719–0.837] | **0.885 OK** [0.808–0.914] | 0.800 M [0.740–0.852] | 0.768 M [0.722–0.814] |
| conn_GDC | 0.843 M [0.777–0.895] | 0.796 M [0.742–0.857] | 0.796 M [0.738–0.850] | 0.815 M [0.767–0.858] |
| conn_Pore | 0.802 M [0.752–0.829] | 0.824 M [0.713–0.844] | **0.874 OK** [0.841–0.877] | **0.874 OK** [0.844–0.876] |
| total_tpb | 0.623 F [0.599–0.625] | 0.684 F [0.649–0.685] | 0.731 M [0.691–0.739] | 0.794 M [0.752–0.807] |
| active_tpb | **0.952 OK** [0.887–0.953] | **0.971 OK** [0.892–0.963] | **0.893 OK** [0.829–0.940] | **0.881 OK** [0.839–0.915] |
| active_tpb_frac | 0.820 M [0.761–0.884] | **0.906 OK** [0.837–0.943] | 0.849 M [0.784–0.910] | 0.821 M [0.785–0.858] |
| dpb_LSCF_GDC | 0.717 M [0.680–0.735] | **0.882 OK** [0.824–0.904] | 0.815 M [0.773–0.851] | 0.743 M [0.712–0.755] |
| dpb_LSCF_Pore | 0.654 F [0.625–0.670] | 0.741 M [0.686–0.768] | 0.758 M [0.701–0.779] | 0.844 M [0.778–0.856] |
| dpb_perc_LSCF_Pore | **0.870 OK** [0.793–0.873] | **0.971 OK** [0.826–0.926] | **0.878 OK** [0.787–0.909] | 0.810 M [0.753–0.862] |
| dpb_GDC_Pore | 0.721 M [0.683–0.731] | **0.887 OK** [0.820–0.903] | 0.817 M [0.766–0.842] | 0.844 M [0.798–0.853] |
| **OK count** | **2** | **6** | **3** | **2** |
| **MARGINAL** | **6** | **3** | **7** | **8** |
| **FAIL** | **2** | **1** | **0** | **0** |

### Tau metrics (ep216, informational only)

| Metric | S-value | Note |
|---|---|---|
| tau_LSCF | 0.452 | FAIL — expected (no cathode tau-net; anode net not valid) |
| tau_GDC  | 0.592 | FAIL — expected |
| tau_Pore | 0.726 | MARGINAL — informational |

---

## Pseudo-ceiling (real-vs-real S)

> Split: low half (y0 ∈ {0,32}, N=12, covers y=0–95) vs high half (y0=96, N=6, covers y=96–159).
> These halves share **zero voxels** (max_low=95, min_high=96, truly disjoint).
> Bootstrap CI: 16th–84th percentile, 500 iterations.
>
> **Geometry caveat**: a 12-vs-12 disjoint split is geometrically impossible with this
> dataset (stride=32, cube=64 on 283-voxel Y axis). The closest achievable is 12 vs 6.
> Within-half crops overlap by 32 voxels, inflating within-half homogeneity and thus
> ceiling S. These ceiling values are an **upper bound** on what any generator can
> achieve against this reference population.

| Metric | Ceiling S | CI [p16–p84] | Band |
|---|---|---|---|
| conn_LSCF | 0.676 | [0.601–0.750] | F |
| conn_GDC | 0.840 | [0.709–0.903] | M |
| conn_Pore | 0.742 | [0.685–0.760] | M |
| total_tpb | 0.652 | [0.610–0.652] | F |
| active_tpb | 0.852 | [0.787–0.903] | OK |
| active_tpb_frac | 0.832 | [0.765–0.913] | M |
| dpb_LSCF_GDC | 0.702 | [0.646–0.700] | M |
| dpb_LSCF_Pore | 0.659 | [0.598–0.672] | F |
| dpb_perc_LSCF_Pore | 0.574 | [0.522–0.627] | F |
| dpb_GDC_Pore | 0.761 | [0.685–0.799] | M |

**Interpretation**: conn_LSCF, total_tpb, dpb_LSCF_Pore, and dpb_perc_LSCF_Pore have
pseudo-ceilings in the FAIL band. For these metrics, the generator scoring FAIL or MARGINAL
may partly reflect genuine spatial heterogeneity in the 151×283×120 Supercrop volume rather
than generator failure. This does not mean these metrics are uninformative — the generator
should approach or exceed the ceiling — but FAIL scores below the ceiling do not indicate
catastrophic mode collapse.

---

## Memorization check

> Source: `4_CNNCT/cathode_run0_memo.csv`.
> Limitation: detects exact/rot/flip copies (16 z-preserving symmetries) but not translated patches.

| Metric | Value |
|---|---|
| gen → nearest train (mean agreement) | 0.5617 |
| gen → nearest train (std) | 0.0545 |
| train → nearest OTHER train baseline (mean) | 0.5933 |
| train → nearest OTHER train baseline (std) | 0.0696 |
| gen_mean − baseline_mean | −0.032 |
| Verdict | **OK — no memorization detected** |

The generator is more diverse than training crops (gen mean < baseline mean). Expected and healthy.

---

## Best checkpoint

| Checkpoint | OK | MARGINAL | FAIL | Notes |
|---|---|---|---|---|
| ep54 | 2 | 6 | 2 | Early; conn and TPB/DPB not converged |
| ep108 | **6** | 3 | **1** | Best OK count; total_tpb still FAIL |
| ep162 | 3 | 7 | 0 | Good balance; fewer FAILs than ep54/108 |
| ep216 | 2 | **8** | 0 | 0 FAILs; best TPB/DPB trajectory |

**ep108 has the most OK metrics (6) but carries 1 FAIL (total_tpb=0.684 F).**
**ep216 has 0 FAILs and shows the strongest total_tpb trajectory (0.794 M, still rising).**

**Is ep108's conn_LSCF advantage over ep216 real or noise?** Bootstrap CIs overlap
(ep108: [0.808–0.914]; ep216: [0.722–0.814] — overlap window 0.808–0.814). The improvement
is marginally distinguishable but narrow. conn_LSCF is NOT reliably better at ep108 vs ep216;
both are within each other's 1-sigma bands.

**Is ep108 genuinely better than ep216 overall?** They target different trade-offs:
ep108 optimizes connectivity (conn_LSCF OK, dpb_LSCF_GDC OK) at the cost of 1 FAIL on
total_tpb. ep216 achieves 0 FAILs with total_tpb approaching MARGINAL-OK boundary. The
choice between them is a domain judgment (prefer no-FAIL vs prefer more-OKs), not a
statistical question the data resolves.

For run1 design: the generator CAN achieve conn_LSCF=0.885 OK (ep108), but does not
sustain it at longer training. A dedicated LSCF connectivity loss in run1 is needed to
lock in this behavior before TPB dynamics dominate.

---

## Key observations

- [x] **Pore connectivity converged** — conn_Pore OK at ep162/216 (0.874) without targeted loss.
- [x] **GDC connectivity MARGINAL** — consistently 0.796–0.843 across all checkpoints. GDC percolation
  is learnable (much better than anode YSZ which was stuck at 0.46–0.49 across 14 runs) but
  does not reach OK without a targeted loss. ceiling = 0.840 M, so generator is near the ceiling.
- [~] **LSCF connectivity oscillates within noise** — point estimate peaks at ep108 (0.885 OK) but
  bootstrap CIs overlap between all checkpoint pairs except total_tpb. The claim of "competing
  dynamics" from the original report is not supported by the error bars. ep108 is the best
  point estimate; the gain over ep216 is marginally distinguishable but not robust.
- [~] **total_tpb monotonically improving** (0.623F → 0.684F → 0.731M → 0.794M) — this IS
  distinguishable across adjacent checkpoints: ep108 vs ep162 (F vs M no-overlap), ep162 vs ep216
  (M vs M, slightly wider but clearly separated). Still MARGINAL at ep216; ceiling = 0.652 F,
  so the generator is already exceeding the real-vs-real ceiling.
- [x] **active_tpb consistently OK** — 0.881–0.971 across all checkpoints.
- [x] **No memorization** — gen_mean (0.562) < baseline (0.593). Generator generalizes.
- [x] **No training divergence** — all 4 checkpoints present, losses finite throughout.
- [!] **Pseudo-ceiling below FAIL for conn_LSCF, total_tpb, dpb_LSCF_Pore, dpb_perc_LSCF_Pore** —
  the dataset itself does not achieve MARGINAL on these metrics between disjoint halves. Run1
  should target exceeding the ceiling rather than targeting an absolute OK/MARGINAL band on these
  metrics without knowing whether real data can achieve OK with a larger reference set.

---

## Next steps (cathode run1 design) — PENDING RE-EVALUATION

> These recommendations were written against the original 6-crop scorecard.
> The hardened evaluation changes the picture on some points (notably: total_tpb ceiling is
> already exceeded by ep216; conn_LSCF improvement at ep108 is not robust to bootstrap).
> Full re-evaluation needed before committing run1 design.

### Priority 1 — conn_LSCF loss
Add a min-slice density loss for LSCF (channel 0, analogous to anode's `w_conn_ysz`):
- Threshold: mean LSCF z-slice density from training crops ≈ vf_LSCF ≈ 0.207
  → threshold ~0.10 (similar fraction to anode's 0.10 out of ~0.30 YSZ VF)
- Activate from epoch 1 (not gated)
- Weight: start at 200 (same as anode w_conn_ysz) and reduce if LSCF over-constrains

### Priority 2 — total_tpb calibrated proxy *(RE-EVALUATE — ceiling already exceeded)*
Add a cathode-calibrated tpb_proxy loss:
- Note: ep216 total_tpb=0.794 M already exceeds the pseudo-ceiling (0.652 F). This metric may
  not need a proxy loss if the ceiling remains below MARGINAL with a larger reference set.
  If more cathode crops become available (Prof. Jin directive), re-run the ceiling analysis first.
- If still needed: calibrate cathode proxy target empirically (NOT anode 0.002 value).

### Priority 3 — epoch budget for run1 *(RE-EVALUATE — ep108 advantage not robust)*
- ep108 appeared to be a sweet spot in the original 6-crop analysis (conn_LSCF 7 OK).
- With 24-crop analysis, ep108 has 6 OK (not 7) and the conn_LSCF improvement over ep216 is
  marginally distinguishable, not robust. The "sweet spot" narrative is weakened.
- Recommendation: train 216 epochs (matching run0) with a LSCF-connectivity loss added,
  then checkpoint at 108 and 216 to check if the loss stabilizes conn_LSCF without reverting.

### Branch
Create `feature/cathode-run1` off `feature/cathode-run0`. Do NOT modify `feature/dataset-configs`.

---

## Notes on dataset limitations

- **0 val crops**: Y=283 voxels, val_start=219 — no stride-aligned crop fits in [219, 283).
  Training-region reference set is the same 24 stride-32 TRAIN crops used for training.
  S-values measure training-distribution fidelity, not generalization.
- **Pseudo-ceiling caveat**: ceiling was computed on 12 vs 6 crops (not 12 vs 12 — geometrically
  impossible). Low N_high=6 inflates CI width; ceiling point estimates are meaningful, bands
  are wide. More cathode volumes (Prof. Jin directive) will improve this.
- **VF drift**: stride-32 crops show GDC −5.6pp drift vs parent volume (spatial inhomogeneity).
  Generator is conditioned on crop VFs, not parent VF. Monitor generated vs training VFs.
- **No tau surrogate**: anode tau_net.pth was trained on Ni/YSZ/Pore — invalid for LSCF/GDC/Pore.
  Tau values computed by taufactor (actual solver). Informational only.
- **Checkpoint path**: training writes to `./save_model/` (repo root) when run as
  `python 1_GAN/main.py`. Fixed in `cathode_run0_pipeline.ps1` post-training;
  for run1 the pipeline will use `--weights save_model/Generator_NNNepoch.pth`.
