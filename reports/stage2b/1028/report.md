# Stage 2B — numerical set encoder and differentiable SED pilot

This is the executed small paired HSC+Legacy source pilot. The full independent Legacy training cache is unavailable, so the definitive population-scale mixed-source experiment remains pending. Stage 2A completed its physical checks and baseline evaluation first; handoff.json and the Stage 2B correctness tests pass. No Qwen/LLM training or Stage 2C was performed.

## Source and target results

| Condition | N | NMAD | Outliers | CE | CRPS | Coverage | Median z | Narrow fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_N0_HSC | 512 | 0.0811 | 0.0840 | 3.5194 | 0.0685 | 0.7578 | 0.2550 | 0.2852 |
| real_N0_Legacy | 512 | 0.0967 | 0.1094 | 3.6369 | 0.0812 | 0.6777 | 0.2475 | 0.3027 |
| real_N1_HSC | 512 | 0.0386 | 0.0254 | 3.0991 | 0.0431 | 0.7734 | 0.2775 | 0.2070 |
| real_N1_Legacy | 512 | 0.0359 | 0.0117 | 2.9806 | 0.0350 | 0.7715 | 0.2925 | 0.1934 |
| real_N2_HSC | 512 | 0.0492 | 0.0352 | 3.2033 | 0.0489 | 0.7988 | 0.2850 | 0.2285 |
| real_N2_Legacy | 512 | 0.0409 | 0.0117 | 2.9973 | 0.0373 | 0.8281 | 0.2925 | 0.2031 |
| real_N1_summary_HSC | 512 | 0.0429 | 0.0273 | 3.1009 | 0.0465 | 0.7852 | 0.3075 | 0.2734 |
| real_N1_summary_Legacy | 512 | 0.0400 | 0.0176 | 2.9685 | 0.0386 | 0.8418 | 0.3225 | 0.2793 |
| real_N1_shuffled_HSC | 512 | 0.0441 | 0.0332 | 3.1264 | 0.0446 | 0.7461 | 0.3225 | 0.2285 |
| real_N1_shuffled_Legacy | 512 | 0.0370 | 0.0117 | 2.9645 | 0.0352 | 0.7969 | 0.3225 | 0.1875 |
| real_N1_brightness_HSC | 512 | 0.0385 | 0.0254 | 3.0915 | 0.0421 | 0.7812 | 0.2775 | 0.2051 |
| real_N1_brightness_Legacy | 512 | 0.0358 | 0.0098 | 2.9927 | 0.0351 | 0.7676 | 0.2925 | 0.2188 |
| real_N1_noise_HSC | 512 | 0.0366 | 0.0449 | 3.1581 | 0.0553 | 0.7969 | 0.2625 | 0.2812 |
| real_N1_noise_Legacy | 512 | 0.0392 | 0.0391 | 3.1217 | 0.0515 | 0.7832 | 0.2625 | 0.2852 |
| real_N2_noise_HSC | 512 | 0.0402 | 0.0469 | 3.2062 | 0.0538 | 0.7324 | 0.3225 | 0.2734 |
| real_N2_noise_Legacy | 512 | 0.0367 | 0.0332 | 3.0523 | 0.0468 | 0.7559 | 0.2775 | 0.2129 |
| CLAUDS_N0_full | 256 | 0.2200 | 0.7422 | 9.5743 | 0.7275 | 0.1914 | 1.0125 | 0.8047 |
| CLAUDS_N0_HSC | 256 | 0.2243 | 0.7422 | 9.4697 | 0.7067 | 0.2031 | 1.0125 | 0.8555 |
| CLAUDS_N0_wrong_response | 256 | 0.2076 | 0.7227 | 9.4599 | 0.7232 | 0.2148 | 1.0125 | 0.7930 |
| CLAUDS_N1_full | 256 | 0.2206 | 0.7227 | 9.0734 | 0.7148 | 0.2266 | 1.0125 | 0.4414 |
| CLAUDS_N1_HSC | 256 | 0.2207 | 0.6953 | 8.8272 | 0.6798 | 0.2422 | 1.0275 | 0.5586 |
| CLAUDS_N1_wrong_response | 256 | 0.2166 | 0.7422 | 9.3983 | 0.7289 | 0.2148 | 1.0125 | 0.4180 |
| CLAUDS_N2_full | 256 | 0.2278 | 0.7109 | 9.2064 | 0.6912 | 0.2227 | 1.0125 | 0.3945 |
| CLAUDS_N2_HSC | 256 | 0.2278 | 0.7109 | 9.3957 | 0.6848 | 0.2383 | 1.0125 | 0.4492 |
| CLAUDS_N2_wrong_response | 256 | 0.2102 | 0.7305 | 9.4893 | 0.7311 | 0.1992 | 1.0125 | 0.5234 |
| CLAUDS_N2_neural_only | 256 | 0.2220 | 0.7266 | 9.3384 | 0.7060 | 0.1992 | 1.0500 | 0.4844 |
| CLAUDS_N1_summary_full | 256 | 0.2551 | 0.6875 | 7.7906 | 0.6435 | 0.4062 | 1.0275 | 0.6680 |
| CLAUDS_N1_summary_HSC | 256 | 0.2198 | 0.6641 | 7.1669 | 0.5805 | 0.5469 | 1.0575 | 0.5352 |
| CLAUDS_N1_shuffled_full | 256 | 0.2275 | 0.6992 | 9.2223 | 0.6738 | 0.2305 | 1.0575 | 0.7695 |
| CLAUDS_N1_shuffled_HSC | 256 | 0.2198 | 0.6875 | 9.0664 | 0.6675 | 0.2422 | 1.0425 | 0.6719 |
| CLAUDS_N1_brightness_full | 256 | 0.2306 | 0.7070 | 7.9731 | 0.6423 | 0.3906 | 1.0125 | 0.4727 |
| CLAUDS_N1_brightness_HSC | 256 | 0.2664 | 0.6406 | 7.3738 | 0.5687 | 0.5117 | 1.0275 | 0.4961 |
| CLAUDS_N1_noise_full | 256 | 0.2051 | 0.8789 | 9.3301 | 0.9248 | 0.1602 | 0.2925 | 0.3672 |
| CLAUDS_N1_noise_HSC | 256 | 0.2303 | 0.8398 | 9.1776 | 0.9007 | 0.1719 | 0.5775 | 0.3281 |
| CLAUDS_N2_noise_full | 256 | 0.2106 | 0.7930 | 9.1098 | 0.8439 | 0.2305 | 0.7575 | 0.2070 |
| CLAUDS_N2_noise_HSC | 256 | 0.1935 | 0.8359 | 9.1696 | 0.8517 | 0.1797 | 0.7725 | 0.1914 |

## Architecture and controls

N0 is a padded numerical MLP with flux/error/mask/quality and sampled responses. N1 encodes each numeric response (256→16→16), shares a band MLP (22→64→64), pools by masked mean, and predicts 300 bins (65→256→300). N2 has exactly the same encoder/head plus a 64→12 positive simplex SED-coefficient predictor. It projects the verified 12 FSPS bases through K[filter,z,component], profiles a nonnegative amplitude at every candidate z, and combines bounded neural logits with a source-fixed tempered profile score. No arbitrary per-filter flux decoder is learned. Exact counts, runtime and peak GPU allocation are in each history JSON; optimizer and label budgets are matched for primary pilots, with any source-only underfit extension recorded.

For N2, logits = 4*tanh(raw/4) + 0.2*(-0.5*(chi2−min chi2)/Nvalid). Effective variance is measured sigma² plus the Stage-2A source-selected fractional-error term applied to max(observed flux,0). The score is a tempered profile pseudo-likelihood, not a marginalized physical posterior. Candidate shape RMS rescaling is an exact amplitude reparameterization. CE plus 0.1*log(1+reduced true-z chi2) and 1e-4*sum(w²) trains coefficients; true source z indexes the training loss only and is absent from inference inputs. The explicit positive beta and bounded correction prevent silent removal of the physical score. Neural-only ablation is separately labelled.

The response grid covers the complete registered UV through W2 wavelength range. Full curves, compact four-number response summaries, and a fixed shuffled-response training control are compared. Post-hoc response corruption is a diagnostic, not proof of physics. All primary models remove common amplitude using the median absolute valid flux; brightness is supplied only to N1_brightness. N1_noise/N2_noise are separate source-only incremental-variance augmentations with error multipliers 1,3,10,30. Common dimming is a normalization symmetry, not a realistic high-z population. EBV value and availability are saved separately and excluded from the primary neural representation. Missing numerical storage is masked before aggregation and likelihood; no unavailable flux is treated as a precise zero.

## Splits, support and gates

The real pilot uses 1,024 training and 512 validation galaxies, each with two actual survey views (2,048/1,024 observations), deterministic identity hashes and survey-balanced counts. Original HSC and Legacy source test assignments are excluded. Validation is the union of both original validation memberships, so no validation galaxy enters training through its other survey. Identity, imaging group, positional and target overlaps are audited. See dataset_audit.json. The paired population mostly lies below z=1, has very few z>1 objects and none at z≥2; brightness/noise augmentation supplies no missing high-z labels.

Gate results: {'gate1': True, 'gate2': {'N0': True, 'N1': True, 'N2': True}, 'gate3': False, 'synthetic_new_filters_N1': {'bias': 0.15812700986862183, 'median_bias': 0.027182750403881073, 'nmad': 0.4335673451423645, 'catastrophic_outlier_fraction': 0.765625, 'cross_entropy': 5.703572750091553, 'mean_log_score': 5.703571796417236, 'mean_crps': 0.6619983911514282, 'p16_p84_coverage': 0.78125, 'pit_mean': 0.5049750804901123, 'n': 64, 'predicted_median': 2.242499828338623, 'predicted_p05': 2.242499828338623, 'predicted_p95': 2.242499828338623, 'predicted_std': 2.384185791015625e-07, 'modal_bin_fraction': 1.0, 'unique_median_bins': 1, 'concentration_width_003': 1.0, 'ce': 5.703572750091553, 'outlier_fraction': 0.765625, 'crps': 0.6619983911514282, 'coverage_68': 0.78125}, 'synthetic_new_filters_N2': {'bias': -0.08382819592952728, 'median_bias': -0.2121691256761551, 'nmad': 0.2662122845649719, 'catastrophic_outlier_fraction': 0.84375, 'cross_entropy': 5.696861267089844, 'mean_log_score': 5.696861743927002, 'mean_crps': 0.673380434513092, 'p16_p84_coverage': 0.78125, 'pit_mean': 0.6217157244682312, 'n': 64, 'predicted_median': 1.3874999284744263, 'predicted_p05': 1.1197500228881836, 'predicted_p95': 2.7472500801086426, 'predicted_std': 0.5650582313537598, 'modal_bin_fraction': 0.0625, 'unique_median_bins': 42, 'concentration_width_003': 0.125, 'ce': 5.696861267089844, 'outlier_fraction': 0.84375, 'crps': 0.673380434513092, 'coverage_68': 0.78125}, 'expanded_seeds': [], 'full_population_expansion': False, 'reason': 'Full original independent Legacy training cache unavailable; only audited paired-view pilot. Further expansion requires interpretable gains, not target-label tuning.'}

The synthetic experiment reuses the Stage-1 disjoint SED-instance splits, source filters, held-out CLAUDS/new-filter observations and entirely held-out double-break/emission family. Its 512 source labels are distinct from real training. The same 300-bin head is used for all new models. synthetic_metrics.csv separates systems, families, missing/noisy/wrong-response variants; paired JSONs quantify identical-latent filter changes and the known +0.3 redshift change. Stage 1’s physical baseline remains a context result with a different template bank and 90-bin PDF grid; CE must not be compared across grids without rebinning. The Stage-1 observation model uses simplified IGM/dust, so it does not exactly match the Stage-2A FSPS/Inoue14 physical basis.

## Scientific interpretation

N1 versus N0 isolates the tested shared-set architecture under a small supervised budget; N2 versus N1 adds the explicit SED decoder and reconstruction objective. Equal parameter counts do not imply equal optimization or effective capacity. Use source convergence and the frozen synthetic gate before interpreting target differences. No target result selected any checkpoint, response encoding, error floor, augmentation or training budget.

CLAUDS labels are Phosphoros ZPHOT and share broad template-fitting assumptions with the physical baseline. Object-dependent predictions, useful outlier rates and calibrated PDFs must be considered together. Accuracy against this reference is exploratory agreement, not independent spectroscopic confirmation. Underfit source models, missing high-z support, brightness/noise shift, unproven historical extinction equivalence, nominal HSC r/i curves and template mismatch limit attribution. A numerical or physical gain cannot establish a foundation-model/pretraining advantage; Stage 2C has not been initiated.

## Reproduction

Run scripts/stage2_numerical.py tests, prepare and run, using reproduce.sh and requirements-executed.txt. The source protocol is saved before training; models_locked.json is written before opening CLAUDS labels. Per-object PDFs, N2 per-z chi-square/SED weights, histories, hashes, controls, source/target conditional metrics and plots are separate from Stage 2A.

## Answers supported by this pilot

The shared set architecture improves real-source learning in this seed: HSC/Legacy NMAD goes from 0.0811/0.0967 for N0 to 0.0386/0.0359 for N1. N2 gives 0.0492/0.0409 and a worse balanced source CE than N1. This does not establish an additional physical-decoder gain.

N0 full CLAUDS: outliers 74.2%, CE 9.574, 68% coverage 19.1%; 80.5% of medians lie within a width-0.03 interval. N1 full CLAUDS: outliers 72.3%, CE 9.073, 68% coverage 22.7%; 44.1% of medians lie within a width-0.03 interval. N2 full CLAUDS: outliers 71.1%, CE 9.206, 68% coverage 22.3%; 39.5% of medians lie within a width-0.03 interval. These concentrated, poorly calibrated predictions are not credible true-redshift transfer. N2 slightly lowers exploratory outliers but worsens CE against N1; that mixed result does not support a physical-generalisation claim. Compact/shuffled-response, brightness and noise controls are all displayed, not selected using target performance.

All five small synthetic fits select update zero by source-validation CE: training did not improve held-out synthetic validation loss. Their random-initialization predictions are explicitly retained as selected by the protocol, not presented as successfully trained synthetic regressors. The 300-bin, 512-label pilot therefore supplies no reliable synthetic architecture ranking. N2 has 84.4% outliers on strictly held-out filters versus 76.6% for N1. No expensive expansion or extra seeds are justified by Gate 3; a fresh source-only protocol addressing this synthetic learning failure would be needed before scaling.

The real target has 69/256 ZPHOT references at z≥2, while the eligible source training population contains none. The faint/noisy population and UV/NIR bands are also shifted. These facts, plus the failed synthetic learning check and physical-model mismatch, prevent clean attribution of real transfer failures to architecture alone. Noise augmentation did not supply high-z labels and did not rescue transfer.

There is no evidence here that reintroducing Qwen is the necessary next step. Resolve synthetic numerical learnability, obtain independent high-z spectroscopy and verify catalogue processing first. This conclusion leaves foundation-model/pretraining claims untested. No Stage 2C run was started.

An empty-set decoder bug was found during the first fixed target inference. The template is now masked before shape normalization, and the squared norm is floored before sqrt. Dedicated finite-output and gradient tests pass. All checkpoint hashes and scientific settings are identical to the lock saved before that inference; no target labels informed the repair. See empty_set_fix_audit.json.

| Model | Domain | Parameters | Updates | Best update | Seconds | Peak GPU MiB |
|---|---|---:|---:|---:|---:|---:|
| N0 | real | 113900 | 1000 | 900 | 4.8 | 72.3 |
| N1_brightness | real | 104012 | 1000 | 1000 | 8.7 | 70.7 |
| N1_noise | real | 104012 | 1000 | 850 | 8.7 | 70.7 |
| N1 | real | 104012 | 1000 | 1000 | 9.0 | 70.7 |
| N1_shuffled | real | 104012 | 1000 | 950 | 9.1 | 70.7 |
| N1_summary | real | 104012 | 1000 | 550 | 8.7 | 70.7 |
| N2_noise | real | 104792 | 1000 | 1000 | 17.4 | 90.8 |
| N2 | real | 104792 | 1000 | 950 | 17.0 | 90.8 |
| N0 | synthetic | 113900 | 800 | 0 | 3.6 | 70.5 |
| N1 | synthetic | 104012 | 800 | 0 | 7.2 | 69.2 |
| N1_shuffled | synthetic | 104012 | 800 | 0 | 6.9 | 69.2 |
| N1_summary | synthetic | 104012 | 800 | 0 | 6.9 | 69.2 |
| N2 | synthetic | 104792 | 800 | 0 | 13.4 | 84.5 |

Stage-1 physics remains useful synthetic context; it used a different controlled template bank and 90-bin grid. Its point/calibration metrics are reproduced in stage1_physics_context.json without an invalid cross-grid CE comparison. Synthetic redshift/S/N and reconstruction diagnostics are separate CSVs; scale tests pass.
