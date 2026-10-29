# Stage 2B.6 — matched convergence and physical-score trust

Stage2A.6 completed before this stage. Its constrained family is a verified diagnostic, not a trusted real-galaxy likelihood. Fresh intrinsic SED identities are split before all views and noise into training, gate-calibration, source validation and held evaluation; no prior inspected synthetic truth is reused as new confirmation. Real target labels and real source test data are never read.

## Source convergence gates

{'N1_seed11': {'validation_CE_improvement': 2.750368118286133, 'best_validation_CE': 2.9482879638671875, 'source_plateau': None, 'converged': False, 'meaningful_learning': True}, 'N2_original_seed11': {'validation_CE_improvement': 2.4806747436523438, 'best_validation_CE': 2.921494483947754, 'source_plateau': 4400, 'converged': True, 'meaningful_learning': True}, 'N2_constrained_seed11': {'validation_CE_improvement': 60.32481002807617, 'best_validation_CE': 16.921772003173828, 'source_plateau': None, 'converged': False, 'meaningful_learning': False}, 'N1_seed23': {'validation_CE_improvement': 2.729438543319702, 'best_validation_CE': 2.972264051437378, 'source_plateau': None, 'converged': False, 'meaningful_learning': True}, 'N2_original_seed23': {'validation_CE_improvement': 2.456472158432007, 'best_validation_CE': 2.918107748031616, 'source_plateau': None, 'converged': False, 'meaningful_learning': True}, 'N2_constrained_seed23': {'validation_CE_improvement': 61.05458641052246, 'best_validation_CE': 16.54648780822754, 'source_plateau': None, 'converged': False, 'meaningful_learning': False}, 'N0_seed11': {'validation_CE_improvement': 2.5031142234802246, 'best_validation_CE': 3.201267719268799, 'source_plateau': None, 'converged': False, 'meaningful_learning': True}, 'N1_kernel_seed11': {'validation_CE_improvement': 2.6748194694519043, 'best_validation_CE': 3.023801326751709, 'source_plateau': 4800, 'converged': True, 'meaningful_learning': True}, 'N2_G1_seed11': {'validation_CE_improvement': 38.481356620788574, 'best_validation_CE': 4.059548377990723, 'source_plateau': None, 'converged': False, 'meaningful_learning': True}, 'N2_G1_seed23': {'validation_CE_improvement': 38.88363838195801, 'best_validation_CE': 3.9742183685302734, 'source_plateau': None, 'converged': False, 'meaningful_learning': True}}

All primary N1/N2 arms use paired initialization seeds, the same labeled examples, optimizer and5000-update cap. Common2000/4000/5000 snapshots and first predeclared source plateau are retained. A missing plateau means matched-convergence superiority is inconclusive, even if equal-budget metrics differ. Selected checkpoints minimize source-validation CE only. N1_kernel consumes complete basis-response arrays as an information-access control; it has additional input parameters and is not an identical-capacity proof. N0 is the canonical padded baseline.

## Fixed held-data outcomes

| Model | Domain | Responses | NMAD | Outliers | CE | CRPS | Coverage |
|---|---|---|---:|---:|---:|---:|---:|
| N1_seed11 | M1 | full | 0.3445 | 0.6328 | 9.2220 | 0.4435 | 0.1562 |
| N1_seed11 | M1 | new_only | 0.3445 | 0.6328 | 9.2598 | 0.4422 | 0.1562 |
| N1_seed11 | M2 | full | 0.2886 | 0.6797 | 9.6453 | 0.4509 | 0.0859 |
| N1_seed11 | M2 | new_only | 0.2886 | 0.6953 | 9.8295 | 0.4505 | 0.0859 |
| N1_seed11 | M3 | full | 0.2771 | 0.7734 | 10.1762 | 0.4971 | 0.0391 |
| N1_seed11 | M3 | new_only | 0.2853 | 0.8047 | 10.3383 | 0.5059 | 0.0391 |
| N1_seed11 | M4 | full | 0.3040 | 0.7734 | 9.8143 | 0.4979 | 0.0859 |
| N1_seed11 | M4 | new_only | 0.3051 | 0.7734 | 9.8595 | 0.4983 | 0.0859 |
| N2_original_seed11 | M1 | full | 0.1593 | 0.5781 | 5.8462 | 0.2308 | 0.4531 |
| N2_original_seed11 | M1 | new_only | 0.1712 | 0.6641 | 5.3576 | 0.3010 | 0.1953 |
| N2_original_seed11 | M2 | full | 1.2903 | 0.7812 | 5.7977 | 1.1667 | 0.6484 |
| N2_original_seed11 | M2 | new_only | 0.1914 | 0.4531 | 4.8932 | 0.2427 | 0.9375 |
| N2_original_seed11 | M3 | full | 0.7760 | 0.9141 | 7.1961 | 1.8585 | 0.1016 |
| N2_original_seed11 | M3 | new_only | 0.2561 | 0.7188 | 6.8856 | 0.4298 | 0.4766 |
| N2_original_seed11 | M4 | full | 0.1047 | 0.6875 | 5.9166 | 0.2484 | 0.5156 |
| N2_original_seed11 | M4 | new_only | 0.1261 | 0.8047 | 5.6984 | 0.3771 | 0.2578 |
| N1_seed23 | M1 | full | 0.3420 | 0.6172 | 8.9662 | 0.4192 | 0.1953 |
| N1_seed23 | M1 | new_only | 0.3445 | 0.6172 | 9.1042 | 0.4347 | 0.1797 |
| N1_seed23 | M2 | full | 0.2845 | 0.6719 | 9.2301 | 0.4227 | 0.1328 |
| N1_seed23 | M2 | new_only | 0.2865 | 0.6719 | 9.5306 | 0.4390 | 0.1094 |
| N1_seed23 | M3 | full | 0.2812 | 0.7734 | 9.9486 | 0.4797 | 0.0625 |
| N1_seed23 | M3 | new_only | 0.2833 | 0.7891 | 10.1809 | 0.4870 | 0.0547 |
| N1_seed23 | M4 | full | 0.3018 | 0.7656 | 9.8337 | 0.4713 | 0.1172 |
| N1_seed23 | M4 | new_only | 0.3141 | 0.7734 | 9.8186 | 0.4878 | 0.0859 |
| N2_original_seed23 | M1 | full | 0.1796 | 0.5078 | 4.4580 | 0.1585 | 0.5938 |
| N2_original_seed23 | M1 | new_only | 0.2892 | 0.4844 | 5.0966 | 0.3057 | 0.3438 |
| N2_original_seed23 | M2 | full | 0.2743 | 0.5469 | 5.1819 | 0.3864 | 0.5625 |
| N2_original_seed23 | M2 | new_only | 0.6481 | 0.6797 | 5.4500 | 0.6737 | 0.5938 |
| N2_original_seed23 | M3 | full | 0.5363 | 0.7422 | 6.5741 | 0.7351 | 0.3047 |
| N2_original_seed23 | M3 | new_only | 0.4518 | 0.7891 | 7.0865 | 0.6296 | 0.1406 |
| N2_original_seed23 | M4 | full | 0.1443 | 0.5234 | 4.8331 | 0.1769 | 0.4922 |
| N2_original_seed23 | M4 | new_only | 0.2683 | 0.5547 | 5.0171 | 0.3272 | 0.4219 |
| N0_seed11 | M1 | full | 0.3258 | 0.6172 | 8.6825 | 0.3959 | 0.2188 |
| N0_seed11 | M1 | new_only | 0.3420 | 0.6172 | 8.9862 | 0.4179 | 0.1797 |
| N0_seed11 | M2 | full | 0.2961 | 0.6641 | 9.1436 | 0.4097 | 0.1484 |
| N0_seed11 | M2 | new_only | 0.2865 | 0.6719 | 9.3407 | 0.4262 | 0.1172 |
| N0_seed11 | M3 | full | 0.2812 | 0.7578 | 9.9838 | 0.4724 | 0.0625 |
| N0_seed11 | M3 | new_only | 0.2833 | 0.7812 | 10.0685 | 0.4853 | 0.0469 |
| N0_seed11 | M4 | full | 0.2926 | 0.7500 | 9.7098 | 0.4491 | 0.1172 |
| N0_seed11 | M4 | new_only | 0.2923 | 0.7656 | 9.8360 | 0.4681 | 0.1016 |
| N1_kernel_seed11 | M1 | full | 0.0567 | 0.1641 | 3.9708 | 0.0735 | 0.5781 |
| N1_kernel_seed11 | M1 | new_only | 0.1243 | 0.4531 | 7.0687 | 0.2099 | 0.2500 |
| N1_kernel_seed11 | M2 | full | 0.1783 | 0.4453 | 5.6205 | 0.1780 | 0.3359 |
| N1_kernel_seed11 | M2 | new_only | 0.1966 | 0.5859 | 7.7887 | 0.2660 | 0.2188 |
| N1_kernel_seed11 | M3 | full | 0.1265 | 0.2422 | 4.8638 | 0.1456 | 0.4844 |
| N1_kernel_seed11 | M3 | new_only | 0.2173 | 0.4922 | 6.7641 | 0.2364 | 0.2656 |
| N1_kernel_seed11 | M4 | full | 0.0746 | 0.1797 | 4.0821 | 0.0812 | 0.4922 |
| N1_kernel_seed11 | M4 | new_only | 0.1077 | 0.4297 | 6.5248 | 0.1801 | 0.3203 |
| N2_G1_seed11 | M1 | full | 0.0240 | 0.0000 | 18.5532 | 0.0319 | 0.3516 |
| N2_G1_seed11 | M1 | new_only | 0.0956 | 0.0625 | 63.0301 | 0.1162 | 0.0547 |
| N2_G1_seed11 | M2 | full | 0.1828 | 0.4688 | 7.8984 | 0.2353 | 0.2812 |
| N2_G1_seed11 | M2 | new_only | 0.1857 | 0.4531 | 7.4138 | 0.3144 | 0.2734 |
| N2_G1_seed11 | M3 | full | 0.1732 | 0.3984 | 6.9493 | 0.3544 | 0.3984 |
| N2_G1_seed11 | M3 | new_only | 0.3046 | 0.6328 | 14.4704 | 0.5943 | 0.3359 |
| N2_G1_seed11 | M4 | full | 0.0137 | 0.0000 | 20.4871 | 0.0342 | 0.2812 |
| N2_G1_seed11 | M4 | new_only | 0.0904 | 0.1562 | 94.0020 | 0.3530 | 0.1953 |
| G2_seed11 | M1 | full | 0.0948 | 0.3672 | 5.2966 | 0.1912 | 0.5859 |
| G2_seed11 | M1 | new_only | 0.1525 | 0.2656 | 14.4446 | 0.1503 | 0.1016 |
| G2_seed11 | M2 | full | 0.2598 | 0.5078 | 6.4745 | 0.2274 | 0.3828 |
| G2_seed11 | M2 | new_only | 0.1815 | 0.4141 | 6.0858 | 0.1840 | 0.3672 |
| G2_seed11 | M3 | full | 0.1453 | 0.3438 | 5.7886 | 0.1749 | 0.3906 |
| G2_seed11 | M3 | new_only | 0.2884 | 0.6094 | 8.2049 | 0.3682 | 0.4141 |
| G2_seed11 | M4 | full | 0.1548 | 0.3984 | 5.4870 | 0.2052 | 0.4766 |
| G2_seed11 | M4 | new_only | 0.1669 | 0.3203 | 37.3541 | 0.3870 | 0.1328 |
| N2_G1_seed23 | M1 | full | 0.0120 | 0.0000 | 8.7482 | 0.0173 | 0.5547 |
| N2_G1_seed23 | M1 | new_only | 1.0132 | 0.7031 | 1668.3119 | 1.8610 | 0.0859 |
| N2_G1_seed23 | M2 | full | 0.1976 | 0.4688 | 7.9686 | 0.2287 | 0.2500 |
| N2_G1_seed23 | M2 | new_only | 0.7811 | 0.8906 | 63.0901 | 1.9433 | 0.1250 |
| N2_G1_seed23 | M3 | full | 0.1775 | 0.3906 | 6.7717 | 0.2517 | 0.2891 |
| N2_G1_seed23 | M3 | new_only | 0.3217 | 0.7344 | 12.2298 | 0.5734 | 0.7422 |
| N2_G1_seed23 | M4 | full | 0.0137 | 0.0000 | 11.9917 | 0.0238 | 0.4219 |
| N2_G1_seed23 | M4 | new_only | 1.6644 | 0.6406 | 1089.5425 | 1.5883 | 0.1641 |
| G2_seed23 | M1 | full | 0.2692 | 0.5703 | 7.6656 | 0.2859 | 0.3906 |
| G2_seed23 | M1 | new_only | 1.0901 | 0.7031 | 230.2959 | 1.7566 | 0.0781 |
| G2_seed23 | M2 | full | 0.2480 | 0.4766 | 8.2149 | 0.2423 | 0.3281 |
| G2_seed23 | M2 | new_only | 0.9853 | 0.8516 | 16.5144 | 1.4594 | 0.2266 |
| G2_seed23 | M3 | full | 0.1652 | 0.3828 | 6.7559 | 0.2452 | 0.3047 |
| G2_seed23 | M3 | new_only | 0.2759 | 0.7266 | 9.5593 | 0.3627 | 0.8203 |
| G2_seed23 | M4 | full | 0.2259 | 0.6250 | 8.1178 | 0.3116 | 0.2656 |
| G2_seed23 | M4 | new_only | 1.5857 | 0.6328 | 228.5359 | 1.5480 | 0.1250 |
| P12 | M1 | full | 0.0000 | 0.1328 | 1.0384 | 0.0726 | 0.8281 |
| P12 | M1 | new_only | 0.2677 | 1.0000 | 4.8573 | 0.7506 | 0.0078 |
| P12 | M2 | full | 0.0132 | 0.0000 | 1.6463 | 0.0145 | 0.8438 |
| P12 | M2 | new_only | 0.2326 | 0.4922 | 3.6261 | 0.2251 | 0.6484 |
| P12 | M3 | full | 0.0475 | 0.0234 | 10.2139 | 0.0625 | 0.2109 |
| P12 | M3 | new_only | 0.2729 | 0.5781 | 13.7279 | 0.2585 | 0.4844 |
| P12 | M4 | full | 0.0222 | 0.2188 | 4.4256 | 0.1484 | 0.4688 |
| P12 | M4 | new_only | 0.2644 | 0.9922 | 5.4179 | 0.7761 | 0.0234 |
| P2 | M1 | full | 0.0000 | 0.0000 | 0.1779 | 0.0008 | 0.9844 |
| P2 | M1 | new_only | 0.0107 | 0.0000 | 1.1809 | 0.0064 | 0.8828 |
| P2 | M2 | full | 0.1714 | 0.7344 | 460.3546 | 0.4616 | 0.0000 |
| P2 | M2 | new_only | 0.1982 | 0.7969 | 143.4254 | 0.8524 | 0.0312 |
| P2 | M3 | full | 0.2468 | 0.4844 | 108.1072 | 1.3852 | 0.1094 |
| P2 | M3 | new_only | 1.1583 | 0.8047 | 89.5653 | 1.3666 | 0.0469 |
| P2 | M4 | full | 0.0000 | 0.0000 | 13.4836 | 0.0110 | 0.5703 |
| P2 | M4 | new_only | 0.0148 | 0.0000 | 2.7092 | 0.0142 | 0.5625 |
| P1 | M1 | full | 0.0340 | 0.3672 | 235.4079 | 0.4848 | 0.2109 |
| P1 | M1 | new_only | 0.3217 | 0.5547 | 13.0138 | 0.5549 | 0.4844 |
| P1 | M2 | full | 0.1380 | 0.3984 | 86.0322 | 0.3467 | 0.1406 |
| P1 | M2 | new_only | 0.2721 | 0.6016 | 21.2189 | 0.5451 | 0.1719 |
| P1 | M3 | full | 0.0829 | 0.1875 | 27.4608 | 0.1160 | 0.1094 |
| P1 | M3 | new_only | 0.2124 | 0.7500 | 31.9916 | 0.4593 | 0.1172 |
| P1 | M4 | full | 0.0238 | 0.3203 | 227.9979 | 0.3870 | 0.1562 |
| P1 | M4 | new_only | 0.3348 | 0.5234 | 14.7316 | 0.4548 | 0.3516 |
| PM | M1 | full | 0.0000 | 0.0000 | 0.1754 | 0.0008 | 0.9844 |
| PM | M1 | new_only | 0.0107 | 0.0000 | 1.1543 | 0.0062 | 0.8906 |
| PM | M2 | full | 0.1714 | 0.7344 | 460.5392 | 0.4615 | 0.0000 |
| PM | M2 | new_only | 0.2146 | 0.7969 | 144.1406 | 0.8894 | 0.0312 |
| PM | M3 | full | 0.2360 | 0.4766 | 109.0050 | 1.3918 | 0.1016 |
| PM | M3 | new_only | 1.2599 | 0.7969 | 90.3528 | 1.3968 | 0.0234 |
| PM | M4 | full | 0.0000 | 0.0000 | 13.4820 | 0.0110 | 0.5703 |
| PM | M4 | new_only | 0.0154 | 0.0000 | 2.6714 | 0.0137 | 0.5703 |

## Physical and trust definitions

Original N2 uses a predicted global12-component mixture,10% TEF and0.2/Nvalid weighting. Constrained N2 uses a predicted globalD2 mixture, measured errors and a fixed0.2 profile coefficient. Their matched encoder/head architecture is identical; the decoder coefficient count differs. Fixed ablations separately vary error model, temperature and SED score on the same frozen source-calibration predictions, so cross-arm differences are not falsely assigned to a single factor. Independent P12/P2 baselines instead optimize coefficients at every candidate z; proper PM integrates the documented A.6 priors. This is more fitting freedom than a single predicted global mixture.

G1 uses only independently calculated constrained adequacy:0.2/[1+(q/3)^2], q=min_z chi2_D2/max(Nvalid−2,1), zero when fewer than3 measurements exist. This proxy is not a calibrated chi-square probability; constraints, redshift searching and non-identifiability limit degrees-of-freedom interpretation. A near-zero unrestrictedD12 chi-square never automatically grants trust. Rule features are detached from neural parameters, preventing a network from intentionally ruining reconstruction to switch off physics.

A tiny learned G2 is introduced only if fixed source controls justify trust heterogeneity. It sees4 observed-data adequacy/count/SNR/disagreement features, not true redshift, SED identity, coefficients or mismatch class. Its5 parameters train on the independent source-calibration partition with a regularized smooth worst-domain CE objective. Domain is used for balancing/conditional analysis only, never as a deployable input. Gate collapse, conditional outcomes and source-validation diagnostics are reported; ignoring all hard objects is not a claimed physical improvement.

The full likelihood cache derives solely from observed f/sigma, masks and actual response metadata. It includes the eligible normalized marginal model, keeps negative measurements and never inserts missing EBV. All profiles use free nonnegative amplitude. Gaussian constants cancel for profile z scores but are retained for proper evidence. A temperature does not repair a wrong minimum. Source template mismatch, learned coefficient inadequacy, score flattening and probability calibration are separate diagnoses.

## Interpretation and limitations

Paired SED-level bootstrap is computed per seed and domain; two independently initialized seeds supply separate training variability. Held noise/missingness, response corruption and known+0.1 counterfactuals are saved. Wrong-metadata sensitivity is not proof of physical reasoning. Matched spectral priors constitute additional external information for N2 and physical-only baselines; the full-kernel N1 control documents, but cannot erase, information/capacity differences.

Real-source plausibility uses only existing paired confirmation galaxies in HSC5, Legacy3, Legacy5 and HSC+WISE views; observed trust features are calculated before spectroscopic evaluation. This does not validate a synthetic gate on real data. High-z real supervision and an independent Legacy training population remain unavailable. Unknown extinction histories, HSC exposure mixtures and cmodel/Tractor compatibility remain.

No population-scale real run, CLAUDS-label calibration or pretrained Qwen/Stage2C experiment is justified solely by these controlled tests. Any useful effect must survive matched source convergence, both seeds, new-only responses and SED mismatch. Negative or inconclusive gates remain explicit. Exact source histories, physical scores, fitted priors, bootstrap units, runtime and hashes accompany this report.

## Direct empirical answers and decision limits

1. Equal-budget paired seeds and close source-loss comparisons are executed. Full matched convergence is assessed using both first and final-window plateaus in scientific_convergence_assessment.json. If either N1 or its N2 comparator lacks the registered source plateau, superiority over a properly converged N1 is inconclusive. A large initial-to-final CE drop alone does not pass learnability when CE remains worse than a uniform predictor; forcedD2 arms that fail this are not evaluated on held data.

2. Conditional new-only and full-response effects must be read per domain and seed, not pooled. The table below reports the training-seed range separately from paired-SED bootstrap intervals. Marginalization and constrained physics are not presumed better merely because their PDFs are sharper.

| Model | Domain | Responses | Mean outliers | Seed range |
|---|---|---|---:|---:|
| N1 | M1 | full | 0.6250 | 0.6172–0.6328 |
| N1 | M1 | new_only | 0.6250 | 0.6172–0.6328 |
| N1 | M2 | full | 0.6758 | 0.6719–0.6797 |
| N1 | M2 | new_only | 0.6836 | 0.6719–0.6953 |
| N1 | M3 | full | 0.7734 | 0.7734–0.7734 |
| N1 | M3 | new_only | 0.7969 | 0.7891–0.8047 |
| N1 | M4 | full | 0.7695 | 0.7656–0.7734 |
| N1 | M4 | new_only | 0.7734 | 0.7734–0.7734 |
| N2_original | M1 | full | 0.5430 | 0.5078–0.5781 |
| N2_original | M1 | new_only | 0.5742 | 0.4844–0.6641 |
| N2_original | M2 | full | 0.6641 | 0.5469–0.7812 |
| N2_original | M2 | new_only | 0.5664 | 0.4531–0.6797 |
| N2_original | M3 | full | 0.8281 | 0.7422–0.9141 |
| N2_original | M3 | new_only | 0.7539 | 0.7188–0.7891 |
| N2_original | M4 | full | 0.6055 | 0.5234–0.6875 |
| N2_original | M4 | new_only | 0.6797 | 0.5547–0.8047 |
| N2_G1 | M1 | full | 0.0000 | 0.0000–0.0000 |
| N2_G1 | M1 | new_only | 0.3828 | 0.0625–0.7031 |
| N2_G1 | M2 | full | 0.4688 | 0.4688–0.4688 |
| N2_G1 | M2 | new_only | 0.6719 | 0.4531–0.8906 |
| N2_G1 | M3 | full | 0.3945 | 0.3906–0.3984 |
| N2_G1 | M3 | new_only | 0.6836 | 0.6328–0.7344 |
| N2_G1 | M4 | full | 0.0000 | 0.0000–0.0000 |
| N2_G1 | M4 | new_only | 0.3984 | 0.1562–0.6406 |
| G2 | M1 | full | 0.4688 | 0.3672–0.5703 |
| G2 | M1 | new_only | 0.4844 | 0.2656–0.7031 |
| G2 | M2 | full | 0.4922 | 0.4766–0.5078 |
| G2 | M2 | new_only | 0.6328 | 0.4141–0.8516 |
| G2 | M3 | full | 0.3633 | 0.3438–0.3828 |
| G2 | M3 | new_only | 0.6680 | 0.6094–0.7266 |
| G2 | M4 | full | 0.5117 | 0.3984–0.6250 |
| G2 | M4 | new_only | 0.4766 | 0.3203–0.6328 |
| N1_calibrated | M1 | full | 0.6250 | 0.6172–0.6328 |
| N1_calibrated | M1 | new_only | 0.6250 | 0.6172–0.6328 |
| N1_calibrated | M2 | full | 0.6758 | 0.6719–0.6797 |
| N1_calibrated | M2 | new_only | 0.6836 | 0.6719–0.6953 |
| N1_calibrated | M3 | full | 0.7734 | 0.7734–0.7734 |
| N1_calibrated | M3 | new_only | 0.7969 | 0.7891–0.8047 |
| N1_calibrated | M4 | full | 0.7695 | 0.7656–0.7734 |
| N1_calibrated | M4 | new_only | 0.7734 | 0.7734–0.7734 |
| G0_calibrated | M1 | full | 0.6094 | 0.5938–0.6250 |
| G0_calibrated | M1 | new_only | 0.5469 | 0.5000–0.5938 |
| G0_calibrated | M2 | full | 0.4922 | 0.4766–0.5078 |
| G0_calibrated | M2 | new_only | 0.5234 | 0.4219–0.6250 |
| G0_calibrated | M3 | full | 0.3633 | 0.3438–0.3828 |
| G0_calibrated | M3 | new_only | 0.6250 | 0.5312–0.7188 |
| G0_calibrated | M4 | full | 0.6719 | 0.6250–0.7188 |
| G0_calibrated | M4 | new_only | 0.6367 | 0.5391–0.7344 |
| PM_calibrated | M1 | full | 0.6094 | 0.5938–0.6250 |
| PM_calibrated | M1 | new_only | 0.5469 | 0.5000–0.5938 |
| PM_calibrated | M2 | full | 0.4922 | 0.4766–0.5078 |
| PM_calibrated | M2 | new_only | 0.5234 | 0.4219–0.6250 |
| PM_calibrated | M3 | full | 0.3633 | 0.3438–0.3828 |
| PM_calibrated | M3 | new_only | 0.6250 | 0.5312–0.7188 |
| PM_calibrated | M4 | full | 0.6719 | 0.6250–0.7188 |
| PM_calibrated | M4 | new_only | 0.6367 | 0.5391–0.7344 |

3. Fixed frozen-model source controls disentangle10% TEF, the earlier Nvalid/0.2 temperature, global trust coefficient and SED class. Stronger physics can help M1 while imposing enormous CE under SED or coherent measurement mismatch. Temperature changes probability weighting but cannot repair a wrong physical minimum. Full tables preserve these negative cases.

4. G1 is an observed-data adequacy rule, not an oracle. G2 uses the same calibration labels as explicitly supplied scalar N1-temperature and global-physics controls; its extra supervised calibration exposure is documented. Gate collapse and unsupported count/SNR extrapolation are limitations, not successful physical reasoning. The ORACLE_domain_class arm uses privileged mismatch labels and is an upper-bound diagnostic only, never a deployable model.

5. The properDM score is numerically converged, but the Stage2A.6 real-source family mismatch persists. Proper priors change latent-volume weighting; they cannot synthesize missing spectra. Profile-only and normalized marginal scores, per-z coefficient fitting and a global neural mixture remain distinct baselines. The source-selected calibration controls never consult held metrics.

6. The limited real-source trust check is numerical plausibility only. It uses observed features before spectroscopic evaluation and never treats a zeroD12 minimum as sufficient evidence of reliability. Results: [{'system': 'HSC5', 'n': 128, 'median_G1_weight': 8.465035634328827e-08, 'fraction_low_G1_weight_below001': 0.984375, 'D12_near_zero_fraction': 0.828125, 'mean_G1_given_zero_D12': 0.0016687594632095262, 'D2_outliers': 0.9296875, 'real_z_ge2': 0}, {'system': 'HSC5_W1W2', 'n': 128, 'median_G1_weight': 2.1422241713759095e-08, 'fraction_low_G1_weight_below001': 0.9921875, 'D12_near_zero_fraction': 0.0078125, 'mean_G1_given_zero_D12': 8.474686189045147e-09, 'D2_outliers': 0.6640625, 'real_z_ge2': 0}, {'system': 'Legacy3', 'n': 128, 'median_G1_weight': 8.813287522299306e-06, 'fraction_low_G1_weight_below001': 0.75, 'D12_near_zero_fraction': 1.0, 'mean_G1_given_zero_D12': 0.04706314620076638, 'D2_outliers': 0.9921875, 'real_z_ge2': 0}, {'system': 'Legacy5', 'n': 128, 'median_G1_weight': 7.045061889656843e-07, 'fraction_low_G1_weight_below001': 0.9375, 'D12_near_zero_fraction': 0.2578125, 'mean_G1_given_zero_D12': 0.008310956232821631, 'D2_outliers': 0.84375, 'real_z_ge2': 0}]

7. No Qwen, Stage2C or population-scale real experiment is justified by these tests alone. Require sustained source convergence, conditional mismatch robustness on both seeds, useful new-response effects, high-z spectroscopy and viable independent Legacy training data. Synthetic CWW instances have independent continuous slope/dust realizations with base-template sharing; bootstrap uncertainty is conditional on this finite held-family library, not a population-level library uncertainty.

A float32 subnormal-probability issue made the first Jensen–Shannon trust-feature calculation nonfinite. It stopped gate fitting before any held evaluation. Log-space float64 evaluation repairs the intended diagnostic without changing any source spectra, physical scores or trained neural weights. Invalid gate attempts and the unrevised feature values are preserved separately; all final features are finite. See trust_feature_precision_fix.json.

## Runtime and parameter budgets

Peak GPU below is the process peak observed up to that arm, not an isolated per-model allocation. CPU forward timing is separately saved in inference_runtime.csv and explicitly excludes shared observed-physics cache construction.

| Model | Parameters | Updates | Best update | First plateau | Final window stable | Seconds | Session-peak GPU MiB |
|---|---:|---:|---:|---:|---|---:|---:|
| N0_seed11 | 113900 | 5000 | 5000 | None | False | 28.8 | 107.4 |
| N1_kernel_seed11 | 157516 | 5000 | 5000 | 4800 | True | 50.0 | 112.9 |
| N1_seed11 | 104012 | 5000 | 4800 | None | False | 50.8 | 77.3 |
| N1_seed23 | 104012 | 5000 | 4800 | None | False | 51.4 | 107.4 |
| N2_G1_seed11 | 104142 | 5000 | 5000 | None | False | 84.6 | 112.9 |
| N2_G1_seed23 | 104142 | 5000 | 4200 | None | False | 86.0 | 112.9 |
| N2_constrained_seed11 | 104142 | 5000 | 4800 | None | False | 89.2 | 107.4 |
| N2_constrained_seed23 | 104142 | 5000 | 5000 | None | False | 88.6 | 107.4 |
| N2_original_seed11 | 104792 | 5000 | 4600 | 4400 | False | 89.4 | 107.4 |
| N2_original_seed23 | 104792 | 5000 | 5000 | None | False | 89.1 | 107.4 |

## Quantitative decision summary

Neither primary N1 seed nor the original/adaptive N2 seeds satisfies the registered final-window source plateau within5000 updates. Thus the comparison to a fully converged N1 remains inconclusive; common-budget and first-plateau results must not be promoted to matched-convergence superiority. The forced D2 arm remains worse than uniform source CE and correctly fails held-evaluation eligibility.

On fresh M1 full responses, G1 has0/128 outliers in both seeds, but CE is18.55/8.75. This is a direct warning that an apparently excellent coarse point statistic can coexist with very poor PDFs. On strictly new-only filters, G1 outliers are6.25%/70.31%, so its benefit is not stable across training seeds. G2 full-response outliers are36.72%/57.03% and its average matched full-response gate is approximately0.00052/0.000088: the learned trust mechanism does not preserve the matched full-response point advantage. Small weights alone are not a proof that physics is inactive, since physical-score ranges vary.

The independently fitted D2 and properDM baselines both achieve0% M1 outliers on full and new-only responses. With full responses, their CE is about0.18; both retain extreme failures under M2/M3 mismatch and coherent M4 errors. The full-kernel N1 information control gives16.41% M1 full-response outliers versus63.28% for ordinary N1 seed11, despite slightly worse source CE. This control has additional parameters and explicit external template information; the earlier decoder gain cannot be attributed solely to architecture under identical information.

Consequently the conditional exact-family physics gate works as a positive control, but the combined requirements of source convergence, new-filter robustness, probability calibration and mismatch resilience are not established. Proper marginalisation is not a general repair for a wrong SED family or an unrecognised response shift. No real-survey expansion or pretrained-Qwen claim is warranted.
