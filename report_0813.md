# Recent 10-epoch Qwen and AION results (2026-08-13)

## Scope and completion check

This report covers directories matching:

- `/arc/home/gsm/aion_output/figures/qwen-*-e10`
- `/arc/home/gsm/aion_output/figures/aion-*-e10`

It also includes `/arc/home/gsm/aion_output/figures/qwen-qlora_rlvr-e1`, because this is the one-epoch RLVR continuation of the completed 10-epoch Qwen QLoRA run. Its own directory suffix is `e1`, so it is missed by the literal `qwen-*-e10` glob even though it is part of the same post-training comparison.

The recency window is the 10 days ending 2026-08-13 (2026-08-03 through 2026-08-13, UTC). A run is counted as finished only when it has a final result tensor, a run-level JSON summary with `final_metrics`, and a terminal `saved .../result.pt` log entry. File modification time alone is not treated as completion evidence.

Four literally matching experiment directories and one linked Qwen continuation finished in the window:

| Finished (UTC) | Directory | Completed evaluations |
|---|---|---|
| 2026-08-05 10:58 | `qwen-qwen_posttraining_comparison-e10` | Frozen Qwen baseline; QLoRA |
| 2026-08-10 23:28 | `qwen-residual_embedding_adapter-e10` | Residual embedding adapter |
| 2026-08-10 23:28 | `qwen-qlora_rlvr-e1` | One-epoch RLVR continuation from Qwen QLoRA-e10 |
| 2026-08-12 22:02 | `aion-residual_embedding_adapter-e10` | Head-only baseline, residual embedding adapter, and one-epoch RLVR, each on photometry and photometry+images |
| 2026-08-13 01:07 | `aion-qlora-e10` | Head-only baseline, attentive head-only, and QLoRA, each on photometry and photometry+images |

The two AION head-only baselines are duplicated across the AION experiment directories and have identical metrics. In total, the five included directories contain 16 completed result entries representing 14 distinct method/input evaluations.

Two other matching directories were active but not finished when inspected and are therefore excluded from the result comparisons:

| Directory | Last recorded progress | Missing completion evidence |
|---|---:|---|
| `qwen-dora_posttraining-e10` | 36,200 / 37,500 updates (epoch 10/10) | No final epoch metric, result tensor, run JSON, or comparison figures |
| `qwen-ia3_posttraining-e10` | 34,200 / 37,500 updates (epoch 10/10) | No final epoch metric, result tensor, run JSON, or comparison figures |

## Evaluation context

The Qwen and AION numbers are not directly comparable because they use different catalogues, selections, and split sizes.

| Family/input | Representation and model | Total | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| Qwen | Qwen3.5-4B-Base, last-token pooling, 11 magnitudes, no morphology/images | 300,000 | 60,000 | 15,000 | 225,000 |
| AION photometry | `polymathic-ai/aion-base`, mean encoder-token pooling, native HSC grizy magnitude tokens | 56,288 | 11,258 | 2,814 | 42,216 |
| AION photometry+images | AION magnitude tokens plus five 96x96 HSC image bands | 4,231 | 846 | 212 | 3,173 |

The AION image subset is much smaller because 49,966 of 54,197 complete image assignments failed the 0.9 cutout-coverage requirement. Therefore, the difference between AION photometry and photometry+images is confounded by sample selection and training-set size; it is not a controlled estimate of the value of images.

Metrics below use the final JSON summaries. Lower is better for cross-entropy (CE), CRPS, NMAD, and catastrophic-outlier fraction. Bias and median bias are best near zero. The nominal target for p16-p84 coverage is approximately 68%, and the ideal PIT mean is 0.5 (although PIT shape, shown in the saved figures, contains more information than its mean). Mean log score is omitted because it is equal to CE to the reported precision in these runs.

## Test-set results

### Qwen: magnitudes-only

| Method | CE | CRPS | NMAD | Outliers | Bias | Median bias | Coverage | PIT mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 4.5542 | 0.3297 | 0.1564 | 36.55% | 0.0095 | -0.0048 | 66.46% | 0.518 |
| Residual embedding adapter | 4.4607 | 0.3107 | 0.1385 | 32.30% | 0.0199 | -0.0003 | 68.83% | 0.510 |
| QLoRA | **3.9301** | **0.2249** | 0.0729 | 18.16% | 0.0109 | **0.0006** | 69.09% | 0.520 |
| QLoRA + **RLVR** | 4.2122 | 0.2462 | **0.0724** | **17.67%** | **0.0041** | -0.0028 | 53.68% | 0.534 |

QLoRA is the strongest balanced Qwen result. Relative to frozen Qwen, it reduces NMAD by 53.4%, the outlier fraction by 50.3%, CRPS by 31.8%, and CE by 13.7%. The residual embedding adapter gives a smaller but consistent gain: NMAD falls 11.4%, outliers 11.6%, CRPS 5.8%, and CE 2.1%. QLoRA has slightly larger mean bias than frozen Qwen, but its median bias is much closer to zero.

The one-epoch RLVR continuation optimizes a different tradeoff. Relative to QLoRA, it improves NMAD by 0.8%, outliers by 2.7%, and absolute mean bias by 62.1%, while worsening CE by 7.2% and CRPS by 9.4%. Its p16-p84 coverage drops by 15.4 percentage points to only 53.68%, indicating substantially overconfident/under-dispersed predictive distributions. RLVR therefore gives the best Qwen point-error/outlier figures, but QLoRA remains clearly better as a calibrated probabilistic redshift estimator. The expected verifier reward is 0.2276 on test and 0.2264 on validation.

### AION: photometry

| Method | CE | CRPS | NMAD | Outliers | Bias | Median bias | Coverage | PIT mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Head only | 4.1108 | 0.2430 | 0.1129 | 29.41% | -0.0068 | -0.0070 | 70.75% | 0.536 |
| Attentive head only | 4.0271 | 0.2376 | 0.1090 | 28.64% | **0.0002** | -0.0013 | **68.59%** | 0.521 |
| **QLoRA** | **4.0234** | **0.2373** | **0.1081** | **28.59%** | -0.0003 | **-0.0010** | 68.80% | 0.522 |
| Residual embedding adapter | 4.2977 | 0.2679 | 0.1465 | 35.63% | 0.0136 | 0.0060 | 77.59% | **0.501** |
| RLVR after embedding adapter | 4.3059 | 0.2697 | 0.1474 | 35.82% | 0.0021 | -0.0011 | 74.09% | 0.518 |

QLoRA gives the best predictive metrics, but its advantage over the attentive head is very small. Against the ordinary head-only baseline, QLoRA reduces NMAD by 4.3%, outliers by 2.8%, CRPS by 2.4%, and CE by 2.1%. The attentive head achieves nearly the same gains and the smallest absolute mean bias. Both also move coverage closer to the nominal 68% target.

The residual embedding adapter is materially worse than head-only on this input: NMAD is 29.8% higher, outliers 21.2% higher, and CRPS 10.2% higher. One epoch of RLVR does not recover that loss; it slightly worsens the main predictive metrics, although it reduces the adapter's mean bias from 0.0136 to 0.0021.

The head-only per-field breakdown shows a substantial tract effect. On the test split, tract 9569 has NMAD 0.0996 and 25.02% outliers, whereas tract 9570 has NMAD 0.1286 and 33.63% outliers. Per-field metrics were not included in the adapted-method JSON summaries.

### AION: photometry+images

| Method | CE | CRPS | NMAD | Outliers | Bias | Median bias | Coverage | PIT mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Head only | 4.7146 | 0.2946 | 0.2628 | 56.60% | -0.0062 | -0.0183 | 68.33% | 0.525 |
| Attentive head only | 4.5949 | 0.2680 | 0.2253 | 50.80% | 0.0075 | -0.0027 | 75.86% | **0.505** |
| **QLoRA** | **4.5936** | **0.2673** | **0.2234** | **50.52%** | 0.0063 | -0.0039 | 75.89% | 0.506 |
| Residual embedding adapter | 5.2428 | 0.7377 | 0.4280 | 80.96% | 0.5125 | 0.4790 | 69.40% | 0.272 |
| RLVR after embedding adapter | 5.2394 | 0.7336 | 0.4297 | 80.84% | 0.5076 | 0.4752 | 69.43% | 0.273 |

QLoRA again gives the best predictive metrics, narrowly ahead of the attentive head. Relative to head-only, QLoRA reduces NMAD by 15.0%, outliers by 10.7%, CRPS by 9.3%, and CE by 2.6%. The attentive head is within 0.0019 NMAD and 0.13 percentage points of outlier fraction, so the extra encoder adaptation produces only a marginal gain on this small split.

The image-mode residual adapter has a clear failure mode: mean bias is about +0.51, roughly 81% of objects are catastrophic outliers, CRPS is 0.738, and PIT mean collapses to 0.272. RLVR changes these figures only marginally and does not repair the failure. This should be treated as an unstable or misconfigured training result, not as evidence about residual adapters in general.

## Validation-set summary

The validation rankings agree with the test rankings, with no sign that the winning test result came from a large validation/test reversal.

| Family/input | Method | CE | CRPS | NMAD | Outliers |
|---|---|---:|---:|---:|---:|
| Qwen | Frozen | 4.5660 | 0.3326 | 0.1563 | 36.78% |
| Qwen | Residual embedding adapter | 4.4741 | 0.3139 | 0.1396 | 32.63% |
| Qwen | QLoRA | **3.9305** | **0.2271** | 0.0738 | 18.39% |
| Qwen | QLoRA + **RLVR** | 4.2157 | 0.2484 | **0.0727** | **18.08%** |
| AION photometry | Head only | 4.0753 | 0.2377 | 0.1100 | 28.61% |
| AION photometry | Attentive head only | **3.9946** | 0.2316 | 0.1083 | **27.61%** |
| AION photometry | QLoRA | 3.9957 | **0.2309** | **0.1083** | 28.00% |
| AION photometry | Residual embedding adapter | 4.2845 | 0.2623 | 0.1461 | 34.08% |
| AION photometry | RLVR | 4.2883 | 0.2633 | 0.1485 | 34.65% |
| AION photometry+images | Head only | 4.7245 | 0.2745 | 0.2766 | 58.49% |
| AION photometry+images | Attentive head only | **4.5885** | 0.2506 | 0.2136 | **49.06%** |
| AION photometry+images | QLoRA | 4.5887 | **0.2493** | **0.2135** | **49.06%** |
| AION photometry+images | Residual embedding adapter | 5.2520 | 0.7415 | 0.4290 | 82.55% |
| AION photometry+images | RLVR | 5.2486 | 0.7372 | 0.4289 | 82.55% |

## Main conclusions

1. **Qwen QLoRA is the strongest balanced probabilistic result; RLVR slightly wins NMAD and outlier rate.** QLoRA roughly halves both metrics relative to frozen Qwen and has the best CE/CRPS. RLVR trims NMAD and outliers a little further, but its 53.68% interval coverage is severely low.
2. **For AION, QLoRA wins, but attentive head-only is almost as good.** On both input modes the difference between those methods is tiny, suggesting that much of the available gain comes from a better aggregation/head rather than encoder adaptation.
3. **AION residual embedding adaptation is unsuccessful in these runs.** It degrades photometry-only performance and catastrophically fails on the image subset. The one-epoch RLVR continuation does not correct it.
4. **The image experiment is data-limited and selection-confounded.** Its training split contains only 846 objects versus 11,258 for photometry-only, so its poorer absolute scores cannot be attributed cleanly to the addition of image tokens.
5. **Do not rank Qwen against AION from these tables.** The Qwen run uses a different catalogue and 225,000-object test set, while AION uses 42,216 or 3,173 test objects after different filtering.

## Result and figure locations

Each completed comparison has a JSON summary and five diagnostic figures (loss, scatter, PIT, N(z), and tomographic N(z)):

- Qwen frozen vs QLoRA: `/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison-e10/run.json` and `qwen-qwen_posttraining_comparison_{loss,scatter,pit,nz,nztomo}.jpeg`
- Qwen residual adapter: `/arc/home/gsm/aion_output/figures/qwen-residual_embedding_adapter-e10/embedding_adapter_run.json` and `qwen_embedding_adapter_comparison_{loss,scatter,pit,nz,nztomo}.jpeg`
- Qwen RLVR after QLoRA: `/arc/home/gsm/aion_output/figures/qwen-qlora_rlvr-e1/rlvr_run.json` and `qwen_rlvr_comparison_{loss,scatter,pit,nz,nztomo}.jpeg`
- AION QLoRA, photometry: `/arc/home/gsm/aion_output/figures/aion-qlora-e10/photometry/qlora_run.json` and `aion_photometry_qlora_comparison_{loss,scatter,pit,nz,nztomo}.jpeg`
- AION QLoRA, photometry+images: `/arc/home/gsm/aion_output/figures/aion-qlora-e10/photometry_images/qlora_run.json` and `aion_photometry_images_qlora_comparison_{loss,scatter,pit,nz,nztomo}.jpeg`
- AION residual adapter and RLVR, photometry: `/arc/home/gsm/aion_output/figures/aion-residual_embedding_adapter-e10/photometry/{embedding_adapter,rlvr}_run.json` and the corresponding comparison JPEGs
- AION residual adapter and RLVR, photometry+images: `/arc/home/gsm/aion_output/figures/aion-residual_embedding_adapter-e10/photometry_images/{embedding_adapter,rlvr}_run.json` and the corresponding comparison JPEGs

All underlying metric values in this report were transcribed from those run JSON files; relative changes were calculated from them. Completion status was cross-checked against the logs and result artifacts.
