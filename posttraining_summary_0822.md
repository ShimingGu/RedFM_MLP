# Post-training summary — 2026-08-22 snapshot

## Executive summary

This is a fixed snapshot taken while GLM-5.2 QLoRA→RLVR was still running. The
unfinished GLM RLVR arm is excluded from all rankings and conclusions below.
The document is intentionally structured so a later `posttraining_summary_0823.md`
can add the two GLM RLVR rows without changing the supervised-method sections.

The current conclusions are:

1. **Qwen benefits strongly from direct encoder adaptation.** QLoRA and DoRA
   are effectively tied and are much stronger than IA3 or a residual adapter.
2. **RLVR does not change the Qwen conclusion.** Relative to QLoRA-SFT, it makes
   NMAD and the catastrophic-outlier fraction only slightly better while making
   cross-entropy materially worse.
3. **Aion encoder PEFT changes little on the matched cohorts.** For photometry,
   DoRA is the best encoder arm, but its gain over the frozen attentive-head
   control is only about 1% in NMAD. QLoRA and IA3 are similarly close to the
   control.
4. **Post-encoder residual adaptation is not competitive for Aion.** It is
   clearly harmful for both input modes, especially on the image-selected arm.
5. **Aion QLoRA→RLVR also leaves the conclusion unchanged.** The photometry arm
   gets tiny NMAD/outlier gains but worse cross-entropy; the image arm becomes
   slightly worse on all three headline metrics.
6. **The current GLM-5.2 checkpoint does not benefit from direct PEFT on the
   full photometry cohort.** QLoRA, IA3, and DoRA are nearly indistinguishable
   and all substantially worse than the frozen-head control. The residual
   adapter is the only GLM arm that does not degrade the full cohort, and its
   improvement is small.
7. **GLM morphology conclusions are provisional.** Only 429 objects have all
   required morphology fields (86 train, 21 validation, 322 test). The residual
   adapter is best within that small cohort, but the absolute metrics are poor
   and should not be generalized.

## Scope and interpretation

- Reported values are final **test** metrics from each `result.pt`.
- Lower cross-entropy, NMAD, and catastrophic-outlier fraction are better.
- Parenthesized percentages are relative changes from the row's matched
  baseline; negative values are improvements.
- QLoRA/DoRA/IA3 are compared with their frozen encoder plus learned head.
- Residual adapters are compared with their matched frozen cached-embedding
  head.
- RLVR is compared with its completed QLoRA-SFT source policy, not with the
  frozen baseline.
- Results should be compared across methods within a cohort. In particular,
  Aion photometry and Aion photometry+images use different retained cohorts, so
  their absolute metrics are not a controlled measurement of the value of
  images.

## Plot completeness

All expected plots for completed runs existed, were non-empty, and decoded as
valid images at the time of this snapshot.

| Family | Completed comparisons | Expected plot types | Valid plots |
|---|---:|---|---:|
| Qwen | 5 | loss, scatter, PIT, N(z), tomographic N(z) | 25/25 |
| Aion | 10 (5 methods × 2 modes) | loss, scatter, PIT, N(z), tomographic N(z) | 50/50 |
| GLM-5.2, excluding pending RLVR | 8 (4 methods × 2 modes) | loss, scatter, PIT, N(z), tomographic N(z) | 40/40 |
| **Total** | **23** |  | **115/115** |

Primary output roots:

- Qwen: `/arc/home/gsm/aion_output/figures/qwen-*posttraining*`
- Aion: `/arc/home/gsm/aion_output/figures/aion-*-e10` and
  `/arc/home/gsm/aion_output/figures/aion-qlora_rlvr-e1`
- GLM-5.2: `/arc/home/gsm/aion_output/figures/glm52-posttraining-e10`

## Qwen results

Matched frozen-head baseline: cross-entropy 4.5542, NMAD 0.1564,
catastrophic-outlier fraction 0.3655. Test cohort: 225,000 objects.

| Method | Cross-entropy | NMAD | Catastrophic outliers |
|---|---:|---:|---:|
| QLoRA | **3.9301 (-13.7%)** | **0.0729 (-53.4%)** | **0.1816 (-50.3%)** |
| DoRA | 3.9327 (-13.6%) | 0.0733 (-53.1%) | 0.1817 (-50.3%) |
| IA3 | 4.3949 (-3.5%) | 0.1267 (-19.0%) | 0.2923 (-20.0%) |
| Residual adapter | 4.4607 (-2.1%) | 0.1385 (-11.4%) | 0.3230 (-11.6%) |
| QLoRA→RLVR vs QLoRA | 4.2122 (+7.2%) | 0.0724 (-0.8%) | 0.1767 (-2.7%) |

QLoRA is the simplest winner, although DoRA is statistically and practically
very close in these point estimates. QLoRA→RLVR trades worse distributional
fit for very small point-estimate/outlier gains and therefore does not alter
the supervised ranking.

## Aion results

### Native photometry

Encoder-method baseline: cross-entropy 4.0271, NMAD 0.1090, outlier fraction
0.2864. The separately matched mean-vector baseline for the residual adapter is
4.1108, 0.1129, and 0.2941. Test cohort: 42,216 objects.

| Method | Cross-entropy | NMAD | Catastrophic outliers |
|---|---:|---:|---:|
| DoRA | **4.0203 (-0.2%)** | **0.1077 (-1.1%)** | 0.2861 (-0.1%) |
| QLoRA | 4.0234 (-0.1%) | 0.1081 (-0.9%) | **0.2859 (-0.2%)** |
| IA3 | 4.0269 (-0.0%) | 0.1089 (-0.1%) | 0.2862 (-0.1%) |
| Residual adapter | 4.2977 (+4.5%) | 0.1465 (+29.8%) | 0.3563 (+21.2%) |
| QLoRA→RLVR vs QLoRA | 4.1420 (+2.9%) | 0.1067 (-1.2%) | 0.2832 (-1.0%) |

The encoder PEFT arms are all close to the frozen attentive-head result. DoRA
has the best supervised point estimate, but the effect size is small. RLVR again
adds a minor NMAD/outlier improvement while reducing probabilistic quality.

### Native photometry plus HSC images

Encoder-method baseline: cross-entropy 4.5949, NMAD 0.2253, outlier fraction
0.5080. The separately matched mean-vector baseline for the residual adapter is
4.7146, 0.2628, and 0.5660. Test cohort: 3,173 objects.

| Method | Cross-entropy | NMAD | Catastrophic outliers |
|---|---:|---:|---:|
| QLoRA | **4.5936 (-0.0%)** | **0.2234 (-0.8%)** | **0.5052 (-0.6%)** |
| IA3 | 4.5948 (+0.0%) | 0.2254 (+0.1%) | 0.5080 (+0.0%) |
| DoRA | 4.5973 (+0.1%) | 0.2267 (+0.6%) | 0.5093 (+0.2%) |
| Residual adapter | 5.2428 (+11.2%) | 0.4280 (+62.8%) | 0.8096 (+43.0%) |
| QLoRA→RLVR vs QLoRA | 4.6089 (+0.3%) | 0.2257 (+1.0%) | 0.5077 (+0.5%) |

QLoRA is the only encoder arm with a consistent, though small, improvement on
this cohort. Neither RLVR nor a post-encoder residual adapter improves it.

## GLM-5.2-0.8B-A0.8B results

This local checkpoint is an architecture-test checkpoint. Its routed expert
parameters are not adapted by the current PEFT targets, so these results should
not be generalized to a full GLM-5.2 checkpoint without a matched rerun.

### Photometry

Frozen-head baseline: cross-entropy 4.9128, NMAD 0.2454, outlier fraction
0.5374. Test cohort: 225,000 objects.

| Method | Cross-entropy | NMAD | Catastrophic outliers |
|---|---:|---:|---:|
| Residual adapter | **4.8977 (-0.3%)** | **0.2443 (-0.5%)** | **0.5349 (-0.5%)** |
| QLoRA | 5.1947 (+5.7%) | 0.3692 (+50.4%) | 0.6670 (+24.1%) |
| DoRA | 5.1938 (+5.7%) | 0.3692 (+50.4%) | 0.6672 (+24.2%) |
| IA3 | 5.1934 (+5.7%) | 0.3704 (+50.9%) | 0.6689 (+24.5%) |

The three direct PEFT methods converge to essentially the same degraded result.
This is stronger evidence of a target/checkpoint limitation than a meaningful
choice among QLoRA, DoRA, and IA3. The residual adapter preserves the frozen
representation and gives only a marginal gain.

### Photometry plus catalogue morphology

Frozen-head baseline: cross-entropy 5.7040, NMAD 0.5271, outlier fraction
0.9752. Test cohort: 322 objects; total usable cohort: 429.

| Method | Cross-entropy | NMAD | Catastrophic outliers |
|---|---:|---:|---:|
| Residual adapter | **5.5979 (-1.9%)** | **0.4794 (-9.0%)** | **0.9099 (-6.7%)** |
| QLoRA | 5.6423 (-1.1%) | 0.5203 (-1.3%) | 0.9689 (-0.6%) |
| DoRA | 5.6421 (-1.1%) | 0.5203 (-1.3%) | 0.9689 (-0.6%) |
| IA3 | 5.6639 (-0.7%) | 0.5193 (-1.5%) | 0.9720 (-0.3%) |

The residual adapter is numerically best, but this arm is too small and too
outlier-dominated for a strong scientific conclusion.

## GLM RLVR — intentionally incomplete in this snapshot

At snapshot time, the photometry arm had finished caching its 60,000-policy
reference cohort and had entered RLVR optimization. No GLM RLVR result is used
above. Based on the completed Qwen and Aion experiments, the working hypothesis
is that RLVR may move NMAD/outlier metrics slightly while leaving the supervised
method ranking unchanged.

<!-- GLM_RLVR_RESULTS_START -->
| Mode | Source baseline | Cross-entropy | NMAD | Catastrophic outliers | Status |
|---|---|---:|---:|---:|---|
| Photometry | GLM QLoRA | — | — | — | Running on 2026-08-22 |
| Photometry+morphology | GLM QLoRA | — | — | — | Waiting for photometry arm |
<!-- GLM_RLVR_RESULTS_END -->

To extend this report later:

1. Copy it to `posttraining_summary_0823.md`.
2. Read the two results from
   `glm52-posttraining-e10/{photometry,photometry_morphology}/rlvr/result.pt`.
3. Compare each RLVR row against the QLoRA result from the same mode.
4. Replace only the table between the `GLM_RLVR_RESULTS` markers and the short
   hypothesis paragraph above it.
5. Confirm five new plots per mode. A complete GLM RLVR run increases the plot
   count from 115 to 125.

## Bottom line

The stable conclusion before GLM RLVR completes is that **Qwen QLoRA/DoRA are
the only large, clear encoder-post-training wins in this experiment set**.
Aion is already close to its attainable result with a frozen encoder and
attentive head, while the tested GLM architecture checkpoint is harmed by the
current direct PEFT configuration. Existing Qwen and Aion RLVR runs adjust the
metric trade-off but do not change those conclusions.
