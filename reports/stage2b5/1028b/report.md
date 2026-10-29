# Stage 2B.5 — synthetic learnability and decoder audit

Completed Stage2A.5 before this stage. Original scripts, Stage1 truth/splits and all prior reports/checkpoints remain unchanged. No real CLAUDS labels, source test photometry, Qwen training or population-scale real training are used. This is a diagnostic experiment, not a rerun of the registered Stage2B benchmark.

## Exact update-zero reproduction

| Arm | Train CE initial → final | Validation CE initial → final | Selected update | Parameter change L2 |
|---|---:|---:|---:|---:|
| N0 | 5.717 → 4.434 | 5.690 → 7.569 | 0 | 21.71 |
| N1 | 5.699 → 2.135 | 5.694 → 7.431 | 0 | 39.78 |
| N2 | 5.814 → 2.184 | 5.835 → 7.529 | 0 | 38.41 |
| N1_summary | 5.699 → 2.547 | 5.694 → 7.744 | 0 | 40.94 |
| N1_shuffled | 5.699 → 1.931 | 5.694 → 7.450 | 0 | 41.39 |

All five initial validation CEs and update-zero selections reproduce within1e-4. Training gradients reach the heads/encoders and coefficient predictor; parameter changes are nonzero and losses decrease. The failure is poor validation generalisation while fitting training labels, not an optimizer that failed to run. Histories save gradient norms, reconstruction/CE gradients, parameter/head changes, entropy, occupied-probability bins and initial/final logits. AdamW includes all intended parameters and weight decay; no scheduler was configured in the original experiment. Training/eval mode and label indices are consistent.

The original source training has 128 independent SEDs, 256 redshift draws, and only 166/300 occupied classes. 33.3% of source-validation labels fall in bins with no direct training label. Randomly initialized independent fine-bin logits can learn empirical class occupancy and memorize SEDs while becoming overconfident on validation. This is contributory evidence; the fixed source-only optimization screen does not establish that sparsity alone explains every failure.

## Elementary repairs and learnability

The original padded N0 receives random band-to-slot assignments. On a fixed tiny spectrum family it collapsed to a constant predictor, whereas N1 and the independent known-family physical oracle learned. A new source-only N0 canonicalizes bands by numerical wavelength before flattening. Failed N0 attempts remain under *_unsorted and elementary_gate_before_N0_canonicalization.json. This restores a stable conventional MLP representation; it does not alter the original replay. N1/N2 retain shared-set invariance. A cache-key collision discovered before downstream experiments was also repaired by including requested object count in matched-dataset filenames; completed exact replay outputs were retained.

L0 has independent tiny-jitter SED parameters, one prominent4000A break, uniform z0.1–1.1, no noise/dust and HSC curves. Positive1% nominal sigma supplies finite numerical weights; it is not a claim that random noise was added. Both N0 and N1 pass the declared validation gate, and an additional seed is reported. Tiny-overfit tests are debugging evidence, not generalisation. N2’s tiny matched task reduces CE but may retain ambiguous/wrong medians; that limitation is preserved.

| New source experiment | N | NMAD | Outliers | CE | CRPS | Coverage | Concentration |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0_N0 | 256 | 0.0227 | 0.0000 | 2.6398 | 0.0335 | 0.9414 | 0.1211 |
| L0_N1 | 256 | 0.0169 | 0.0000 | 1.7827 | 0.0204 | 0.6484 | 0.1016 |
| L0_N0_seed23 | 256 | 0.0129 | 0.0000 | 2.3985 | 0.0224 | 0.9414 | 0.0977 |
| L0_N1_seed23 | 256 | 0.0692 | 0.3281 | 2.6980 | 0.1035 | 0.2891 | 0.2930 |
| L1_N0 | 256 | 0.0239 | 0.0039 | 2.8682 | 0.0363 | 0.9844 | 0.1055 |
| L1_N1 | 256 | 0.1235 | 0.3555 | 3.5286 | 0.1207 | 0.5430 | 0.1758 |
| L2_luminosity_N0 | 256 | 0.0184 | 0.0156 | 2.8019 | 0.0366 | 0.9688 | 0.1172 |
| L2_luminosity_N1 | 256 | 0.0183 | 0.0391 | 2.6847 | 0.0483 | 0.9727 | 0.1406 |
| L2_dust_N0 | 256 | 0.0289 | 0.0156 | 2.8185 | 0.0353 | 0.9805 | 0.0938 |
| L2_dust_N1 | 256 | 0.0153 | 0.0195 | 2.7321 | 0.0465 | 0.9688 | 0.1211 |
| L2_noise_N0 | 256 | 0.0380 | 0.0000 | 2.9478 | 0.0399 | 0.9180 | 0.1211 |
| L2_noise_N1 | 256 | 0.0241 | 0.0000 | 2.6759 | 0.0293 | 0.9688 | 0.1016 |
| L2_missing_N0 | 256 | 0.0408 | 0.0781 | 3.2223 | 0.0626 | 0.8398 | 0.1250 |
| L2_missing_N1 | 256 | 0.0365 | 0.0898 | 3.0321 | 0.0544 | 0.7773 | 0.1172 |
| L2_weak_N0 | 256 | 0.2413 | 0.5391 | 4.2421 | 0.1685 | 0.7109 | 1.0000 |
| L2_weak_N1 | 256 | 0.2413 | 0.5391 | 4.2423 | 0.1685 | 0.7109 | 1.0000 |
| L3_N0 | 512 | 0.0626 | 0.0547 | 3.3266 | 0.0648 | 0.7910 | 0.1074 |
| L3_N1 | 512 | 0.0378 | 0.0254 | 2.9238 | 0.0437 | 0.8789 | 0.1562 |
| L3_N2_mismatched | 512 | 0.0366 | 0.0215 | 2.9016 | 0.0424 | 0.8730 | 0.1289 |
| matched_N0 | 512 | 0.1503 | 0.3320 | 4.0635 | 0.1329 | 0.7363 | 0.1367 |
| matched_N1 | 512 | 0.0859 | 0.1523 | 3.6087 | 0.0875 | 0.7930 | 0.2168 |
| matched_N2 | 512 | 0.0914 | 0.1797 | 3.7208 | 0.0967 | 0.8105 | 0.1230 |
| L0_label_shuffled | 256 | 0.2549 | 0.5508 | 4.2556 | 0.1737 | 0.6445 | 1.0000 |
| L4_N0_source_validation | 96 | 0.4856 | 0.7292 | 5.6233 | 0.6755 | 0.6146 | 0.1771 |
| L4_N1_source_validation | 96 | 0.4695 | 0.7083 | 5.6213 | 0.6702 | 0.6354 | 0.8125 |
| L4_N2_source_validation | 96 | 0.3540 | 0.6562 | 5.4947 | 0.6329 | 0.6562 | 0.1458 |
| identifiable_N0 | 512 | 0.0139 | 0.0000 | 2.5319 | 0.0225 | 1.0000 | 0.1230 |
| identifiable_N1 | 512 | 0.0248 | 0.1016 | 2.7105 | 0.0400 | 0.9727 | 0.1152 |
| identifiable_N2 | 512 | 0.0000 | 0.0000 | 2.1936 | 0.0135 | 1.0000 | 0.0957 |

L1 broadens independent slope/break parameters; L2 separately examines amplitude, dust, noise, missingness and weak features; L3 provides actual HSC/Legacy paired views of identical latent instances. Each new source stage uses1024 unique training SEDs and256 independent validation SEDs, not duplicated observations relabelled as independent objects. L3/matched stages have two views per SED. Task-specific seeds differ; cross-task differences are not matched causal effects. The pure amplitude symmetry is tested independently. Known-intrinsic-spectrum likelihoods in source_identifiability_oracles.csv show how weak features can remain ambiguous even with more intrinsic information than the networks possess.

Four fixed source-only optimization candidates test original hard300 CE, lower learning rate, Gaussian-neighbour CE (sigma3 bins), and a30-output head mapped to300 equal-bin masses. The chosen protocol uses lowest original source-validation hard CE. It gains only modestly on the original benchmark; L4 uses this new labelled protocol and does not overwrite the original. The source-validation set is reused for this small diagnostic screen and is not an unbiased confirmation set. Original strict held-out filters/families are evaluated only for L4 models with >0.3 source-validation CE gain and nonzero selected update; skipped arms are not assigned an architectural ranking.

## Physics audit and controlled unseen responses

The known-mixture oracle recovers the injected grid redshift in 100.0% of128 noiseless HSC/Legacy observations. Both known-mixture and NNLS true-z chi-square approach zero. Free12-component fits with5 observed bands often have a continuum of exact fits across z; an arbitrary minimum on this plateau is not numerical failure. Learned mixtures have less intrinsic information than the privileged known-mixture oracle.

Random12-component mixtures test a deliberately underdetermined inverse problem. A second near-single-family FSPS0 +0–5% FSPS1 control makes the matched physics task identifiable, with1024/256 independent mixtures and the same source-selected optimizer. All three models have the same labels and updates. Models are locked before evaluating new responses. This distinguishes physical-decoder learnability from coefficient degeneracy.

N2 keeps the explicit original bounded neural correction plus tempered physical profile score and reconstruction loss. Stage2A.5’s verified kernel is copied into new models; exact replay alone uses the old kernel. Gradient probes quantify CE/reconstruction norms and cosine alignment, and physical/neural logit ranges. Neural-only and learned-physical-only ablations are reported. Effective-error audit explicitly distinguishes1% nominal measurement errors in noiseless matched controls from the decoder’s additional10% observed-flux TEF. The latter often dominates and is a pseudo-likelihood weighting prescription, not a fully specified generative noise distribution. A broadened score or wider prediction distribution is not by itself evidence of physics benefit.

| Controlled domain | Model | Unseen response system | NMAD | Outliers | CE | Coverage |
|---|---|---|---:|---:|---:|---:|
| identifiable | N0 | full | 0.2911 | 0.7148 | 8.5721 | 0.1797 |
| identifiable | N0 | new_only | 0.2792 | 0.7148 | 9.1704 | 0.1250 |
| identifiable | N1 | full | 0.1323 | 0.4297 | 6.8074 | 0.5000 |
| identifiable | N1 | new_only | 0.1782 | 0.5625 | 7.3031 | 0.4844 |
| identifiable | N2 | full | 0.1235 | 0.3008 | 6.2895 | 0.2969 |
| identifiable | N2 | new_only | 0.2176 | 0.5078 | 6.1734 | 0.0938 |
| identifiable | NNLS_physical_only | full | 0.0000 | 0.0000 | 0.4838 | 1.0000 |
| identifiable | NNLS_physical_only | new_only | 0.0255 | 0.0000 | 2.2081 | 1.0000 |
| analytic | N0 | full | 0.1999 | 0.6367 | 6.7438 | 0.4922 |
| analytic | N0 | new_only | 0.1983 | 0.6367 | 6.6072 | 0.4609 |
| analytic | N1 | full | 0.3381 | 0.6992 | 7.9084 | 0.1680 |
| analytic | N1 | new_only | 0.3406 | 0.7109 | 8.5464 | 0.1602 |
| analytic | N2 | full | 0.2006 | 0.5156 | 6.8799 | 0.2188 |
| analytic | N2 | new_only | 0.2153 | 0.5859 | 7.6201 | 0.1797 |
| analytic | NNLS_physical_only | full | 0.0313 | 0.0195 | 7.6156 | 0.1484 |
| analytic | NNLS_physical_only | new_only | 0.2897 | 0.6133 | 6.4775 | 0.5312 |
| matched | N0 | full | 0.2813 | 0.5820 | 4.8695 | 0.6328 |
| matched | N0 | new_only | 0.2782 | 0.5820 | 4.9894 | 0.3906 |
| matched | N1 | full | 0.1821 | 0.4180 | 5.0189 | 0.4844 |
| matched | N1 | new_only | 0.1937 | 0.5273 | 6.7017 | 0.4023 |
| matched | N2 | full | 0.1716 | 0.4570 | 4.6481 | 0.5156 |
| matched | N2 | new_only | 0.1793 | 0.5078 | 4.6174 | 0.5508 |
| matched | NNLS_physical_only | full | 0.0000 | 0.0000 | 0.5516 | 1.0000 |
| matched | NNLS_physical_only | new_only | 0.1652 | 0.4492 | 3.1095 | 0.6719 |
| identifiable | NNLS_same_TEF | full | 0.0133 | 0.0000 | 2.3909 | 1.0000 |
| identifiable | NNLS_same_TEF | new_only | 0.2039 | 0.4062 | 4.3573 | 1.0000 |
| analytic | NNLS_same_TEF | full | 0.0256 | 0.0000 | 3.7311 | 0.4570 |
| analytic | NNLS_same_TEF | new_only | 0.2945 | 0.5508 | 4.6901 | 0.7656 |
| matched | NNLS_same_TEF | full | 0.0219 | 0.0000 | 2.8456 | 1.0000 |
| matched | NNLS_same_TEF | new_only | 0.2518 | 0.6602 | 4.5866 | 0.7227 |

These controlled held views use independent source-validation SEDs reobserved through actual CLAUDS-like/full or strictly new UV/NIR curves; they are not real CLAUDS galaxies. Noise, missingness, wrong metadata and +0.1 redshift counterfactuals use locked models. New controlled sets are separately identified from original Stage1 strict evaluation membership. N2 has external FSPS prior information through the full response kernel, while N1 learns a finite response encoding; this information advantage is explicit. Known-mixture oracles have additional privileged latent information.

Paired SED-unit bootstrap results (N2 minus N1 outlier fractions; negative favours N2):

[{'domain': 'identifiable', 'system': 'full', 'unique_SED_units': 256, 'N2_minus_N1_outlier_change': -0.12890625, 'paired_SED_bootstrap_CI95': [-0.2109375, -0.046875], 'replicates': 300}, {'domain': 'identifiable', 'system': 'new_only', 'unique_SED_units': 256, 'N2_minus_N1_outlier_change': -0.0546875, 'paired_SED_bootstrap_CI95': [-0.16015625, 0.06845703124999991], 'replicates': 300}, {'domain': 'analytic', 'system': 'full', 'unique_SED_units': 256, 'N2_minus_N1_outlier_change': -0.18359375, 'paired_SED_bootstrap_CI95': [-0.29501953125, -0.08779296875000009], 'replicates': 300}, {'domain': 'analytic', 'system': 'new_only', 'unique_SED_units': 256, 'N2_minus_N1_outlier_change': -0.125, 'paired_SED_bootstrap_CI95': [-0.23046875, -0.03701171875000009], 'replicates': 300}, {'domain': 'matched', 'system': 'full', 'unique_SED_units': 256, 'N2_minus_N1_outlier_change': 0.0390625, 'paired_SED_bootstrap_CI95': [-0.01171875, 0.09189453124999991], 'replicates': 300}, {'domain': 'matched', 'system': 'new_only', 'unique_SED_units': 256, 'N2_minus_N1_outlier_change': -0.01953125, 'paired_SED_bootstrap_CI95': [-0.0703125, 0.02734375], 'replicates': 300}]

No bootstrap treats repeated survey/noise views as independent SEDs. A physical advantage can be claimed only in the particular controlled conditions supported by these intervals, oracle adequacy and source convergence, not as general real-population transfer.

## Direct conclusions and future components

1. Original update-zero selection reflects decreasing training loss with worsening validation CE, sparse fine-bin supervision and SED ambiguity/mismatch. It does not mean weights or gradients were absent. The fixed small optimization screen yields limited rescue on the original benchmark; no single cause is proven sufficient.

2. N0 and N1 can learn an identifiable elementary source problem; N0 needs stable canonical numeric input slots. N1’s shared aggregation passes permutation checks. Later weak-feature failures must be distinguished from optimization failures.

3. The physical kernel, amplitude profile and NNLS recover the expected true-z fit when spectral assumptions match. Unknown mixtures can be non-identifiable. Empty/masked sets, negative fluxes, scaling symmetry and finite gradients pass. No true redshift enters inference; source z is used only to index a training reconstruction loss or explicitly labelled oracle diagnostic.

4. The decoder is numerically usable as a tested, conditional physical constraint. Controlled source/held-response results above delimit any benefit. Neither a good elementary fit nor a wider target distribution establishes general physical transfer; mismatched analytic/CWW versus FSPS families and bounded neural corrections remain risks.

5. Training data quantity and class resolution, intrinsic non-identifiability, stable input architecture, physical misspecification and source-noise assumptions all matter. The new ladder and original benchmark are separate experiments; sample/parameter budgets, convergence and extra template information are reported rather than hidden.

6. No population-scale real experiment is justified here. Real-source high-z spectroscopy, an independent Legacy training population, trustworthy catalogue/calibration semantics and independently verified target redshifts are still required.

7. No result demonstrates pretrained Qwen astrophysical knowledge or its necessity. Establish interpretable numerical/physical controls and independent real confirmation before any Stage2C comparison. No Qwen/large-backbone training was started.

## Reproduction

Run stage2a5_audit.py replay/kernels/analyze/details first, then stage2b5_learnability.py prepare/run and report_stage2_1028b.py controls/finalize. Original source splits and synthetic truth are immutable. See protocols, per-object PDFs, source/held metrics, gradients, exact runtime dependencies, model budgets, protected hashes and separate manifests.

## Quantified physical-score attribution

There is conditional synthetic evidence for a decoder benefit after source learning is valid: in the near-single-family matched test, full held-response outliers decrease from42.97% for N1 to30.08% for N2. The paired256-SED bootstrap gives N2−N1 =−12.89 percentage points (95% interval−21.09 to−4.69). The strictly new-only interval crosses zero, and arbitrary-mixture controls do not establish an improvement. Analytic-domain gains also leave high absolute error. These are conditional-on-checkpoint intervals, not uncertainty over training seeds; there is no general or real-CLAUDS transfer claim.

The independent full-response NNLS baseline reaches0% outliers in the identifiable full-band matched test, both with measured errors alone and with the same10% TEF. N2 therefore leaves recoverable physical information unused. The measurement-only and same-TEF baselines are distinguished explicitly; no noise-model information advantage is hidden. Coefficient recovery is good for the near-single-family case (median L1 error about0.05), but poor for random12-component mixtures (about1.3–1.4), consistent with underdetermination from five source bands.

The explicit score is Gaussian profile log likelihood divided by Nvalid/beta: temperature25 for five bands and55 for eleven bands, in addition to TEF broadening. In the identifiable source control its median physical-logit range is3.55 versus6.82 for the bounded neural term; the untempered physical range is88.86. About80.8% of original synthetic source bands have TEF variance larger than measured variance (median model fraction84.5%). This is deliberate score tempering, not a hidden arithmetic bug, but its effect is physically important: on identical frozen source predictions, physical-only outliers are50.0% under original tempering and0.59% without it.

Source-only fixed scale ablations follow. They were derived from the mathematical score and do not change saved checkpoints or hyperparameters; held-response and real-target evaluations were not rerun under these alternatives. Removing tempering is not declared a general fix, since mismatched physics can then dominate a correct neural prediction.

| Source domain | Frozen score diagnostic | Outliers | CE | CRPS |
|---|---|---:|---:|---:|
| identifiable | neural_only | 0.0000 | 2.2546 | 0.0174 |
| identifiable | physical_only_original_tempering | 0.5000 | 4.3296 | 0.4281 |
| identifiable | physical_only_untempered | 0.0059 | 2.1897 | 0.0729 |
| identifiable | combined_original | 0.0000 | 2.1936 | 0.0135 |
| identifiable | combined_untempered_diagnostic | 0.0000 | 1.7963 | 0.0087 |
| analytic | neural_only | 0.0176 | 2.9663 | 0.0501 |
| analytic | physical_only_original_tempering | 0.7637 | 4.7390 | 0.5407 |
| analytic | physical_only_untempered | 0.2031 | 5.4561 | 0.2556 |
| analytic | combined_original | 0.0215 | 2.9016 | 0.0424 |
| analytic | combined_untempered_diagnostic | 0.0527 | 4.4424 | 0.0824 |
| matched | neural_only | 0.2715 | 3.9118 | 0.1189 |
| matched | physical_only_original_tempering | 0.5898 | 4.6369 | 0.4117 |
| matched | physical_only_untempered | 0.2324 | 7.4193 | 0.1512 |
| matched | combined_original | 0.1797 | 3.7208 | 0.0967 |
| matched | combined_untempered_diagnostic | 0.1895 | 7.1930 | 0.1137 |

## Source-only convergence limitation

The2000-update identifiable N1/N2 source curves were still decreasing, so an additional fixed4000-update diagnostic checks optimization confounding on source validation only. No held-response or real-target predictions are reevaluated with these later checkpoints. The earlier held-response gain is therefore conditional on that fixed training budget and does not establish an architecture ranking after matched convergence.

| Source-only longer-budget model | NMAD | Outliers | CE |
|---|---:|---:|---:|
| identifiable_source_convergence_N1 | 0.0000 | 0.0000 | 2.2729 |
| identifiable_source_convergence_N2 | 0.0000 | 0.0000 | 2.1047 |
