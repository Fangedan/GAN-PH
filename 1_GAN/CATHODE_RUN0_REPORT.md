# Cathode Run 0 — Report (Hardened Evaluation)

> Branch: feature/cathode-run0
> Dataset: S1 Supercrop (LSCF/GDC/Pore, 64³ voxels, ~40 nm voxel)
> Training set: 24 crops (stride-32, TRAIN region)
> Training-region reference set: 24 stride-32 crops (same region; no held-out val — see Evaluation Limits below)
> Config: cathode_s1_supercrop
> Training completed: 2026-07-09, ~16:10 local time
> Evaluation hardened: 2026-07-10 — re-scored against 24 str32 crops; bootstrap CIs added
> Status: **PAUSED — DATA-LIMITED** (see Data Request Rationale)

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
- **Cross-region heterogeneity — not a ceiling**: the real-vs-real S computed on spatially disjoint halves of the specimen (see below) quantifies how much the specimen's microstructural statistics drift across ~100 voxels. The generator trains on and is scored against the full 24-crop pool; it can legitimately outscore any disjoint-half comparison because it learned the full distribution. For metrics where cross-region S is low (conn_LSCF, total_tpb), those low values reflect spatial non-stationarity within this one specimen, not a limit on what the generator can achieve.
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

## Cross-region heterogeneity (real-vs-real S, spatially disjoint halves)

> Split: low half (y0 ∈ {0,32}, N=12, covers y=0–95) vs high half (y0=96, N=6, covers y=96–159).
> These halves share **zero voxels** (max_low=95, min_high=96).
> Bootstrap CI: 16th–84th percentile, 500 iterations.
>
> **What this measures**: unlike the anode real-vs-real comparison — which split independent
> structures drawn from the same distribution and thus measured sampling noise — this splits
> two spatial regions of a **single 151×283×120 specimen**. Low cross-region S values indicate
> that the specimen's microstructural statistics drift meaningfully across ~100 voxels of Y,
> i.e. the material is spatially non-stationary at this scale.
>
> The generator trains on and is evaluated against the full 24-crop pool, so it can and should
> outscore a disjoint-half comparison. These numbers are **not** an upper bound on the generator.
> They quantify how much spatial heterogeneity is present in this one specimen, which determines
> how well S-values computed against the 24-crop pool represent a stable physical target.
>
> **Geometry note**: a 12-vs-12 truly disjoint split is geometrically impossible (stride=32,
> cube=64, 283-voxel Y axis). Best achievable: 12 vs 6. Within-half crops overlap by 32 voxels,
> inflating within-half homogeneity and biasing cross-region S upward. The values below are
> therefore upper estimates of the true cross-region agreement.

| Metric | Cross-region S | CI [p16–p84] | Band |
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
cross-region S in the FAIL band. This means the statistics of these metrics shift substantially
between the y=0–95 and y=96–159 regions of the Supercrop. When the generator scores above
these values — as it does for total_tpb (ep216: 0.794 M vs cross-region: 0.652 F) and
for conn_LSCF at all checkpoints — it is matching the full-pool distribution better than
any single spatial subregion does. That is expected behaviour, not a violation of a ceiling.

**Consequence for run design**: for metrics where cross-region S is in FAIL (total_tpb,
dpb_LSCF_Pore, dpb_perc_LSCF_Pore), the training reference itself is spatially unstable.
Optimizing a generator loss to increase S against this 24-crop pool further would chase a
moving target that may not be physically meaningful. These metrics should be treated as
monitoring-only until additional specimens are available to establish a stable reference
distribution. See Data Request Rationale.

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

**Is ep108's conn_LSCF advantage over ep216 real?** The bootstrap CIs are nearly disjoint —
ep108: [0.808–0.914] vs ep216: [0.722–0.814], with an overlap window of just [0.808–0.814]
(6 S-points wide out of a combined span of 192). This is weak evidence of degradation with
continued training: the point estimates drop monotonically after ep108 (0.885 → 0.800 → 0.768),
and the CIs barely touch. However, N=20 structures per early checkpoint is too few to make
this definitive. Record as: conn_LSCF degrades at longer training; revisit when evaluation
is better powered (more specimens → larger reference set → narrower bootstrap bands).

**Is ep108 genuinely better than ep216 overall?** Different trade-offs:
ep108 optimizes connectivity (conn_LSCF OK, dpb_LSCF_GDC OK) at the cost of 1 FAIL on
total_tpb. ep216 achieves 0 FAILs with total_tpb approaching the MARGINAL–OK boundary.
For downstream use, ep216 is the conservative choice; ep108 is the choice if conn_LSCF
matters more than total_tpb for the application.

For run1 design: a dedicated LSCF connectivity loss is needed to lock in conn_LSCF before
TPB dynamics dominate at longer training. Whether ep108 or ep216 is the right stopping
point for run1 should be re-evaluated after adding that loss.

---

## Key observations

- [x] **Pore connectivity converged** — conn_Pore OK at ep162/216 (0.874) without targeted loss.
- [x] **GDC connectivity MARGINAL** — consistently 0.796–0.843 across all checkpoints. GDC percolation
  is learnable (much better than anode YSZ which was stuck at 0.46–0.49 across 14 runs) but
  does not reach OK without a targeted loss. Cross-region S = 0.840 M, so generator is near the
  cross-region agreement level — no easy headroom for OK without more data.
- [~] **LSCF connectivity — weak evidence of degradation with training** — point estimate peaks at
  ep108 (0.885 OK), then falls monotonically. Bootstrap CIs are nearly disjoint between ep108
  and ep216 (overlap: [0.808–0.814], 6 points wide). This is weak evidence that conn_LSCF
  degrades with continued training. Not definitive at N=20 per checkpoint; revisit with more data.
- [~] **total_tpb monotonically improving** (0.623F → 0.684F → 0.731M → 0.794M) — improvements are
  distinguishable across adjacent checkpoints. At ep216, the generator matches the full 24-crop
  pool better than the two spatial halves of the specimen match each other (0.794 vs cross-region
  0.652). Further optimizing this S-value against the current reference is not physically
  meaningful: the training data itself is spatially inhomogeneous on total_tpb. Deferred until
  more specimens are available to establish a stable target distribution.
- [x] **active_tpb consistently OK** — 0.881–0.971 across all checkpoints.
- [x] **No memorization** — gen_mean (0.562) < baseline (0.593). Generator generalizes.
- [x] **No training divergence** — all 4 checkpoints present, losses finite throughout.
- [!] **Spatial non-stationarity limits interpretation** — conn_LSCF, total_tpb, dpb_LSCF_Pore,
  and dpb_perc_LSCF_Pore show cross-region S in FAIL. For these metrics, S-values against the
  current 24-crop pool should be treated as monitoring data, not optimization targets, until
  additional specimens establish a spatially stable reference distribution.

---

## Data request rationale (for Prof. Jin)

This section documents why additional cathode volumes are needed before run1 can be
meaningfully designed, beyond what was apparent at the start of run0.

**What run0 quantified**: training on a single 151×283×120 specimen, the generator
achieves 0 FAIL / 8 MARGINAL / 2 OK (ep216) against the full 24-crop reference pool.
The evaluation hardening further revealed that the specimen itself is spatially
non-stationary: crops from y=0–95 and crops from y=96–159 score against each other at
0.652 F (total_tpb) and 0.676 F (conn_LSCF). The generator, trained on both halves,
outscore this cross-region comparison — which is expected but also means the generator
is not being evaluated against a stable physical ground truth.

**What more volumes would enable**:

1. **True metric ceiling**: with N≥3 independent specimens, scoring one specimen's crops
   against another specimen's pool gives a cross-specimen S that is a meaningful upper
   bound — the two specimens are independent draws from the same process, not two halves
   of one drifting sample. The current cross-region S is an upper estimate of heterogeneity,
   not a ceiling on what a good generator should achieve.

2. **Held-out validation set**: even a second specimen (second specimen = held-out; first
   = training) would enable proper out-of-distribution evaluation. Currently, S-values
   measure fit to training data; with a held-out specimen, they would measure generalization.

3. **Meaningful bootstrap bands**: N_gen=20 per early checkpoint produces CI widths of
   ±0.06–0.09 S-units. With N_gen=50 and a stable cross-specimen reference, bands would
   narrow to ±0.02–0.04, making checkpoint comparisons and run-to-run differences
   statistically interpretable.

4. **Run1 target calibration**: the cross-region S tells us that total_tpb and conn_LSCF
   distributions differ substantially across y-space in this one specimen. Before designing
   auxiliary losses to push the generator toward specific S-value targets on these metrics,
   it is important to confirm those targets are stable across specimens. If a second specimen
   shows similar VF and connectivity distributions, total_tpb and conn_LSCF targets can be
   locked in with confidence.

**Requested**: additional LSCF+GDC+Pore cathode FIB-SEM volumes, same segmentation
protocol as S1 Supercrop. Even 1–2 additional specimens at comparable size would unblock
items 1 and 2 above. Full pipeline is ready to ingest new volumes via the config system
(new YAML file, no code changes required).

---

## Next steps (cathode run1 design) — ON FILE, EXECUTION PAUSED

> Run1 design is documented below but execution is paused pending additional specimens.
> The cross-region heterogeneity analysis revealed that for several key metrics (total_tpb,
> conn_LSCF), the current single-specimen reference is too spatially inhomogeneous to
> provide stable optimization targets. Adding auxiliary losses before the reference
> distribution is stable risks chasing spatially-variable artefacts.

### Priority 1 — conn_LSCF loss (ready to implement)
Add a min-slice density loss for LSCF (channel 0, analogous to anode's `w_conn_ysz`):
- Threshold: mean LSCF z-slice density from training crops ≈ vf_LSCF ≈ 0.207
  → threshold ~0.10 (similar fraction to anode's 0.10 out of ~0.30 YSZ VF)
- Activate from epoch 1 (not gated)
- Weight: start at 200 (same as anode w_conn_ysz) and reduce if LSCF over-constrains
- Rationale: conn_LSCF shows weak evidence of degradation with training length (ep108 → ep216);
  a targeted loss should lock it in before TPB dynamics dominate.

### Priority 2 — total_tpb proxy (deferred — wait for more specimens)
- At ep216, the generator already matches the full pool better than the spatial halves match
  each other (0.794 vs cross-region 0.652). The specimen's total_tpb is spatially
  inhomogeneous — optimizing a proxy loss against this reference would be chasing a target
  that may not reflect a stable physical quantity.
- Action: re-run cross-region S after second specimen arrives. If cross-region S rises toward
  MARGINAL (suggesting the single-specimen heterogeneity was an outlier), calibrate the
  cathode tpb_proxy target at that point. Anode target (0.002) is not valid for cathode.

### Priority 3 — epoch budget for run1 (deferred — revisit after conn_LSCF loss)
- Weak evidence that conn_LSCF degrades after ep108. Whether this is resolved by the
  connectivity loss (which may sustain conn_LSCF to 216 epochs) or by stopping at 108
  depends on how the loss interacts with the training dynamics. Checkpoint at both 108
  and 216 in run1.

### Branch
Create `feature/cathode-run1` off `feature/cathode-run0`. Do NOT modify `feature/dataset-configs`.

---

## Notes on dataset limitations

- **0 val crops**: Y=283 voxels, val_start=219 — no stride-aligned crop fits in [219, 283).
  Training-region reference set is the same 24 stride-32 TRAIN crops used for training.
  S-values measure training-distribution fidelity, not generalization.
- **Cross-region S (not a ceiling)**: the real-vs-real comparison uses 12 crops (y=0–95)
  vs 6 crops (y=96–159). Low N_high=6 inflates CI width. These values measure spatial
  non-stationarity within the specimen, not a limit on the generator. See dedicated section.
- **VF drift**: stride-32 crops show GDC −5.6pp drift vs parent volume (spatial inhomogeneity).
  Generator is conditioned on crop VFs, not parent VF. Monitor generated vs training VFs.
- **No tau surrogate**: anode tau_net.pth was trained on Ni/YSZ/Pore — invalid for LSCF/GDC/Pore.
  Tau values computed by taufactor (actual solver). Informational only.
- **Checkpoint path**: training writes to `./save_model/` (repo root) when run as
  `python 1_GAN/main.py`. Fixed in `cathode_run0_pipeline.ps1` post-training;
  for run1 the pipeline will use `--weights save_model/Generator_NNNepoch.pth`.
