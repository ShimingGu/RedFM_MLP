# Stage 2A.5 — HSC/Legacy physical root-cause audit

Archived P0 replay passes bit-exact PDF checks on 16 deterministic validation objects per source before interpretation. Original Stage2A/2B scripts and results are preserved. No CLAUDS labels or source test photometry are read.

## Matched source comparisons

| Cohort | Configuration | NMAD | Outliers | CE | Coverage | Median zspec chi2 | Median minimum chi2 | Wrong minimum >2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HSC_train | original | 0.3684 | 0.8438 | 781.6142 | 0.0625 | 279.9948 | 0.0000 | 0.6406 |
| HSC_train | accurate | 0.3684 | 0.8438 | 781.6041 | 0.0625 | 280.6531 | 0.0000 | 0.6719 |
| HSC_train | accurate_P2 | 0.2415 | 0.9844 | 4.7872 | 0.1953 | 0.0355 | 0.0000 | 0.6328 |
| HSC_val | original | 0.3434 | 0.8008 | 886.0064 | 0.0566 | 310.6649 | 0.0000 | 0.6387 |
| HSC_val | accurate | 0.3412 | 0.8027 | 885.0122 | 0.0547 | 311.0961 | 0.0000 | 0.6309 |
| HSC_val | accurate_P2 | 0.2572 | 0.9746 | 4.7835 | 0.2324 | 0.0495 | 0.0000 | 0.5977 |
| Legacy_val | original | 0.0697 | 0.1309 | 52.4799 | 0.5371 | 3.0866 | 0.4948 | 0.0098 |
| Legacy_val | accurate | 0.0697 | 0.1309 | 52.4665 | 0.5371 | 3.0867 | 0.4950 | 0.0098 |
| Legacy_val | accurate_P2 | 0.1010 | 0.1719 | 4.1196 | 0.8691 | 0.2980 | 0.0498 | 0.0078 |
| paired_train_HSC | original | 0.4123 | 0.7812 | 553.9669 | 0.0938 | 257.8183 | 0.0000 | 0.5781 |
| paired_train_HSC | accurate | 0.4123 | 0.7969 | 548.6127 | 0.0938 | 258.1374 | 0.0000 | 0.5938 |
| paired_train_HSC | accurate_P2 | 0.3147 | 0.9219 | 4.6728 | 0.2656 | 0.0517 | 0.0000 | 0.5156 |
| paired_train_Legacy | original | 0.0425 | 0.0625 | 20.8967 | 0.3594 | 14.0252 | 3.0001 | 0.0000 |
| paired_train_Legacy | accurate | 0.0425 | 0.0625 | 20.8926 | 0.3594 | 13.9568 | 2.9968 | 0.0000 |
| paired_train_Legacy | accurate_P2 | 0.0993 | 0.2812 | 3.7379 | 0.8750 | 0.1264 | 0.0293 | 0.0000 |
| paired_val_HSC | original | 0.3208 | 0.8672 | 986.7963 | 0.0312 | 488.2608 | 0.0000 | 0.7656 |
| paired_val_HSC | accurate | 0.3208 | 0.8672 | 986.2215 | 0.0312 | 488.0790 | 0.0000 | 0.7422 |
| paired_val_HSC | accurate_P2 | 0.2080 | 0.9922 | 4.7740 | 0.1641 | 0.0686 | 0.0000 | 0.7344 |
| paired_val_Legacy | original | 0.0311 | 0.0703 | 29.9600 | 0.2812 | 17.0206 | 3.4283 | 0.0000 |
| paired_val_Legacy | accurate | 0.0307 | 0.0703 | 29.9596 | 0.2812 | 17.0130 | 3.4285 | 0.0000 |
| paired_val_Legacy | accurate_P2 | 0.0786 | 0.1406 | 3.5811 | 0.8594 | 0.1415 | 0.0253 | 0.0000 |
| HSC_train | extended | 0.2900 | 0.9141 | 648.2936 | 0.1016 | 111.0414 | 0.0000 | 0.7109 |
| HSC_train | source_prior | 0.1876 | 0.5547 | 778.9501 | 0.1562 | 280.6531 | 0.0000 | 0.6719 |
| HSC_train | foreground | 0.4230 | 0.8281 | 801.8060 | 0.0703 | 295.8460 | 0.0000 | 0.6562 |
| HSC_train | r2i2_sensitivity | 0.3430 | 0.8438 | 916.4104 | 0.0703 | 397.9054 | 0.0000 | 0.6250 |
| HSC_val | extended | 0.2955 | 0.8887 | 687.2680 | 0.0762 | 117.2560 | 0.0000 | 0.7109 |
| HSC_val | source_prior | 0.2286 | 0.5605 | 882.1686 | 0.1133 | 311.0961 | 0.0000 | 0.6309 |
| HSC_val | foreground | 0.3992 | 0.7969 | 907.8430 | 0.0605 | 335.5890 | 0.0000 | 0.6523 |
| HSC_val | r2i2_sensitivity | 0.3541 | 0.7930 | 1031.4316 | 0.0605 | 393.4513 | 0.0000 | 0.6367 |
| Legacy_val | extended | 0.0675 | 0.1348 | 51.1979 | 0.5801 | 1.9909 | 0.2939 | 0.0098 |
| Legacy_val | source_prior | 0.0846 | 0.2891 | 52.9032 | 0.3633 | 3.0867 | 0.4950 | 0.0098 |
| Legacy_val | foreground | 0.0752 | 0.1387 | 47.8571 | 0.5234 | 3.1304 | 0.4835 | 0.0098 |

## Numerical audit

{'all_21_filters': True, 'all_12_templates': True, 'redshifts': [0.0075, 0.1, 0.42, 1.0, 2.0, 3.0, 4.0, 4.4925], 'original_relative_difference_quantiles': [3.512871002903029e-06, 0.0001273472540714663, 0.0015095535054234302, 0.00360252700418923], 'dense_half_A_max_relative_difference': 0.0008611087453289994, 'order3_vs5_pass': True, 'constant_fnu': [{'filter': 'BASS.g', 'constant_fnu': 1.0}, {'filter': 'BASS.r', 'constant_fnu': 1.0}, {'filter': 'DECam.g', 'constant_fnu': 1.0}, {'filter': 'DECam.r', 'constant_fnu': 1.0}, {'filter': 'DECam.z', 'constant_fnu': 1.0}, {'filter': 'HSC.g', 'constant_fnu': 1.0}, {'filter': 'HSC.i', 'constant_fnu': 1.0}, {'filter': 'HSC.i2', 'constant_fnu': 1.0}, {'filter': 'HSC.r', 'constant_fnu': 1.0}, {'filter': 'HSC.r2', 'constant_fnu': 1.0}, {'filter': 'HSC.y', 'constant_fnu': 1.0}, {'filter': 'HSC.z', 'constant_fnu': 1.0}, {'filter': 'MegaCam.u', 'constant_fnu': 1.0}, {'filter': 'MegaCam.u_star', 'constant_fnu': 1.0}, {'filter': 'MzLS.z', 'constant_fnu': 1.0}, {'filter': 'VIRCAM.H', 'constant_fnu': 1.0}, {'filter': 'VIRCAM.J', 'constant_fnu': 1.0}, {'filter': 'VIRCAM.Ks', 'constant_fnu': 1.0}, {'filter': 'VIRCAM.Y', 'constant_fnu': 1.0}, {'filter': 'WISE.W1', 'constant_fnu': 1.0}, {'filter': 'WISE.W2', 'constant_fnu': 1.0}], 'AB_nanomaggy_zero_point': 22.5, 'unit_conversion': 0.275422870333817, 'seconds': 23.683783531188965, 'passed': True, 'interpretation': 'Old conservative interpolation and coarse filter quadrature differ from full convolution; significance is evaluated in measured-sigma units, not a universal flux tolerance'}

The improved convolution integrates piecewise-linear fnu and photon throughput on their union of knots using three-point Gauss–Legendre quadrature. Five-point quadrature and an independently sampled 0.5A trapezoid verify convergence. It uses all 21 actual responses, every original FSPS spectrum, the same IGM evaluated at native SED knots, no new extinction screen, and the original amplitude-free normalization. Differences are evaluated against actual sigma in numerical_significance.csv. No universal relative-flux tolerance is substituted for observational significance.

NNLS, EAZY NNLS and independent BVLS agree on fixed low/high-z cases in double precision. Column order, inverse-variance weighting, signed fluxes and invalid uncertainties are explicitly tested. Large condition numbers/underdetermined 5-band 12-template fits are recorded, not treated as evidence of unique redshift. PDF masses use the original bin widths and grid.

HDF cmodel magnitude/uncertainty conversion and grizy mapping reproduce the archived source values independently. Those errors are first-order propagated magnitude errors, not an independent signed-flux error model. HSC formal relative errors are exceptionally small. Tractor measurements have different profile/calibration semantics and usually lower optical/NIR precision. Exact foreground correction history and exposure-level r/r2,i/i2 weights remain unknown; no arbitrary zero-point correction is fitted. See photometry_audit.json.

## Attribution and decoder handoff

The table separates a real convolution correction from template extension, effective-error reweighting, documented filter/extinction sensitivities and a source-derived prior. Compare changes in continuous-zspec residuals, global minima and delta chi2, not only PDF median or CE. Original-configuration continuous-zspec residuals use the original EAZY convolution; refined configurations use independent Gauss convolution. Each retains its own grid-global chi2 and minimum residuals. This deliberately avoids interpreting nearest-bin rounding as template failure.

Same-galaxy paired views control intrinsic population and spectroscopic redshift. The optical-only Legacy diagnostic removes W1/W2 on identical objects. Joint fits assume independent errors and shared SED/amplitude only for diagnosis; overlapping calibration/morphology errors violate strict independence. Cross-view SED checks allow a separate amplitude and expose wavelength-dependent shape inconsistencies.

The corrected full-response kernel and tested masked NNLS are suitable numerical infrastructure for Stage2B.5. Measured-error P0 is not automatically a reliable decoder constraint: template incompleteness, cmodel/Tractor systematics, formal error scale and high-z colour degeneracy must be evaluated with the results above. A source prior can suppress a wrong high-z solution without repairing its likelihood minimum. Error broadening can improve coverage without restoring true-z identifiability. No single combined correction should be called a universal causal solution or proof of CLAUDS transfer.

Per-object arrays contain continuous-zspec and grid-global chi2, coefficients, active sets recoverable from coefficients, residuals, matrix rank and photometric flags from immutable input arrays. Reduced chi2 is meaningful only where Nvalid−rank>0. All chi2 curves, exact inputs, raw residuals and predetermined representative-selection rules are saved. Alternative grid minima and all modes remain in the full chi2 arrays. Reproduce with scripts/stage2a5_audit.py replay, kernels, analyze.

## Direct answers from fixed source confirmation

On 512 deterministic HSC validation objects, original outliers are 80.1%; accurate convolution gives 80.3%. The same correction gives Legacy outliers 13.1%. Thus convolution alone does not explain the HSC/Legacy accuracy discrepancy. Its sigma-scaled effects are measured rather than assumed negligible. The earlier 1.40% point-sampling difference is superseded by the full all-template/all-filter audit.

At continuous zspec the HSC median chi2 is 311.10, versus its global median minimum 0.00. The fraction of low-z source objects with a wrong global minimum above z=2 is 63.1%. This identifies a likelihood/SED adequacy and colour-degeneracy issue, rather than a mere median-PDF rounding artifact.

The fixed source-selected 10% error term gives 97.5% outliers and 23.2% coverage; the fixed FSPS+CWW basis gives 88.9% outliers and median true-z chi2 117.26. These diagnose error semantics and template coverage separately. Numerical, library, filter, extinction and prior effects overlap and are not additive causal percentages.

For 128 identical validation galaxies, accurate HSC/Legacy outliers are 86.7%/7.0%; Legacy optical-only gives 97.7%. Population differences alone cannot explain a discrepancy that persists on matched objects. Additional WISE coverage, model-family convex-cone coverage, formal errors and profile-dependent calibration jointly affect identifiability. Per-band residual signs and cross-view reconstructions are saved; no unique observational calibration cause is established.

The verified accurate convolution and NNLS are reusable. The measured-error FSPS profile alone is not validated as a universally trustworthy N2 constraint: bad true-z fits and wrong high-z minima require matched-physics oracle checks and explicit misspecification tests. A source prior changes posterior weights, not physical minima. Unknown foreground histories, r/i exposure mixtures, magnitude-derived uncertainties and cmodel/Tractor profile semantics remain explicit. No CLAUDS inference or transfer claim is made.

At accurate HSC validation minima, median chi-square is effectively zero despite median continuous true-z chi-square311.1. The median fraction of redshift bins fitting to chi-square<1e-8 is 8.7%. With12 coefficients and5 measurements, exact-fit plateaus and their redshift extent matter: an arbitrary argmin inside them is not a unique physical solution. Formal i-band relative sigma is about0.07%; small SED/measurement discrepancies can therefore strongly suppress the true-redshift likelihood. Legacy W1/W2 supplies crucial additional wavelength leverage; the matched optical-only comparison supports this explanation, but does not identify a unique calibration or template defect.
