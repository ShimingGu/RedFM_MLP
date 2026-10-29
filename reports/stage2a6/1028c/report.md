# Stage 2A.6 — SED freedom, evidence volume and identifiability

All choices use source data or the pinned library only. No target labels, real source test objects or prior outputs are modified. Sixty-four paired source-training galaxies support development,64 distinct validation galaxies and128 distinct confirmation galaxies retain all HSC/Legacy views together. Templates and the verified21-filter union-knot integrator are unchanged.

## Fixed confirmation and band-coverage comparisons

| Condition | Model | NMAD | Outliers | CE | CRPS | Coverage | Competitive-bin fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| bands_HSC3_grz | D12 | 0.2583 | 0.9531 | 53.0282 | 1.6420 | 0.2109 | 0.3600 |
| bands_HSC3_grz | D1_profile | 0.4550 | 0.5781 | 4397.3853 | 1.2151 | 0.0625 | 0.0033 |
| bands_HSC3_grz | D1_template_marg | 0.4550 | 0.5781 | 4397.3843 | 1.2151 | 0.0625 | 0.0033 |
| bands_HSC3_grz | D1_evidence | 0.5068 | 0.5859 | 4399.3994 | 1.3065 | 0.0469 | 0.0033 |
| bands_HSC3_grz | D2 | 0.7100 | 0.9531 | 32121.9590 | 2.1461 | 0.0469 | 0.0033 |
| bands_HSC3_grz | D3 | 0.6861 | 0.7422 | 3184.9749 | 1.2107 | 0.1250 | 0.1050 |
| bands_HSC3_grz | DM | 0.6979 | 0.9609 | 32121.1211 | 2.1686 | 0.0391 | 0.0033 |
| bands_HSC3_grz | DM_alpha05 | 0.6935 | 0.9766 | 32125.5645 | 2.2534 | 0.0000 | 0.0033 |
| confirmation_HSC_tef0.0 | D12 | 0.2877 | 0.8359 | 381.1262 | 1.9735 | 0.0547 | 0.1017 |
| confirmation_HSC_tef0.0 | D1_profile | 0.0686 | 0.1797 | 6001.0674 | 0.3165 | 0.0000 | 0.0033 |
| confirmation_HSC_tef0.0 | D1_template_marg | 0.0686 | 0.1797 | 6001.0669 | 0.3165 | 0.0000 | 0.0033 |
| confirmation_HSC_tef0.0 | D1_evidence | 0.0686 | 0.1797 | 6001.4868 | 0.3178 | 0.0000 | 0.0033 |
| confirmation_HSC_tef0.0 | D2 | 0.4393 | 0.9297 | 48021.6523 | 2.3890 | 0.0078 | 0.0033 |
| confirmation_HSC_tef0.0 | D3 | 0.1131 | 0.4219 | 5294.9590 | 0.7239 | 0.0078 | 0.0033 |
| confirmation_HSC_tef0.0 | DM | 0.4150 | 0.9297 | 48019.9492 | 2.4069 | 0.0000 | 0.0033 |
| confirmation_HSC_tef0.0 | DM_alpha05 | 0.4194 | 0.9297 | 48025.2852 | 2.4136 | 0.0000 | 0.0033 |
| confirmation_HSC_tef0.1 | D12 | 0.2817 | 0.9531 | 4.7373 | 1.3471 | 0.2031 | 0.3533 |
| confirmation_HSC_tef0.1 | D1_profile | 0.9658 | 0.7500 | 4.1321 | 0.7797 | 0.6250 | 0.0367 |
| confirmation_HSC_tef0.1 | D1_template_marg | 0.5348 | 0.6562 | 4.0237 | 0.7528 | 0.6406 | 0.0367 |
| confirmation_HSC_tef0.1 | D1_evidence | 0.2596 | 0.9609 | 8.7187 | 2.8656 | 0.0312 | 0.0433 |
| confirmation_HSC_tef0.1 | D2 | 0.4996 | 0.9141 | 9.6468 | 1.8065 | 0.1719 | 0.0317 |
| confirmation_HSC_tef0.1 | D3 | 0.3799 | 0.7109 | 4.2692 | 0.7524 | 0.3438 | 0.1000 |
| confirmation_HSC_tef0.1 | DM | 0.3903 | 0.9219 | 10.5330 | 2.1466 | 0.1484 | 0.0417 |
| confirmation_HSC_tef0.1 | DM_alpha05 | 0.2698 | 0.9688 | 13.6694 | 2.7876 | 0.1250 | 0.0367 |
| bands_Legacy4_W1 | D12 | 0.0493 | 0.0469 | 37.8133 | 0.0586 | 0.4141 | 0.0200 |
| bands_Legacy4_W1 | D1_profile | 0.1088 | 0.2812 | 677.1972 | 0.1422 | 0.0234 | 0.0033 |
| bands_Legacy4_W1 | D1_template_marg | 0.1088 | 0.2812 | 677.1948 | 0.1424 | 0.0234 | 0.0033 |
| bands_Legacy4_W1 | D1_evidence | 0.1079 | 0.2734 | 677.2318 | 0.1428 | 0.0234 | 0.0033 |
| bands_Legacy4_W1 | D2 | 0.5615 | 0.8438 | 22124.1992 | 1.9387 | 0.0078 | 0.0033 |
| bands_Legacy4_W1 | D3 | 0.0738 | 0.2344 | 2116.3845 | 0.1443 | 0.0703 | 0.0033 |
| bands_Legacy4_W1 | DM | 0.5615 | 0.8438 | 22124.1328 | 1.9385 | 0.0078 | 0.0033 |
| bands_Legacy4_W1 | DM_alpha05 | 0.5615 | 0.8438 | 22125.4199 | 1.9386 | 0.0078 | 0.0033 |
| bands_HSC5 | D12 | 0.2877 | 0.8359 | 381.1262 | 1.9735 | 0.0547 | 0.1017 |
| bands_HSC5 | D1_profile | 0.0686 | 0.1797 | 6001.0674 | 0.3165 | 0.0000 | 0.0033 |
| bands_HSC5 | D1_template_marg | 0.0686 | 0.1797 | 6001.0669 | 0.3165 | 0.0000 | 0.0033 |
| bands_HSC5 | D1_evidence | 0.0686 | 0.1797 | 6001.4868 | 0.3178 | 0.0000 | 0.0033 |
| bands_HSC5 | D2 | 0.4393 | 0.9297 | 48021.6523 | 2.3890 | 0.0078 | 0.0033 |
| bands_HSC5 | D3 | 0.1131 | 0.4219 | 5294.9590 | 0.7239 | 0.0078 | 0.0033 |
| bands_HSC5 | DM | 0.4150 | 0.9297 | 48019.9492 | 2.4069 | 0.0000 | 0.0033 |
| bands_HSC5 | DM_alpha05 | 0.4194 | 0.9297 | 48025.2852 | 2.4136 | 0.0000 | 0.0033 |
| bands_HSC3_gri | D12 | 0.2873 | 0.9453 | 104.2587 | 1.6719 | 0.2188 | 0.3600 |
| bands_HSC3_gri | D1_profile | 0.8245 | 0.6172 | 2669.6262 | 1.4326 | 0.0391 | 0.0033 |
| bands_HSC3_gri | D1_template_marg | 0.8245 | 0.6172 | 2669.6282 | 1.4326 | 0.0391 | 0.0033 |
| bands_HSC3_gri | D1_evidence | 1.1456 | 0.6406 | 2672.0708 | 1.4942 | 0.0469 | 0.0033 |
| bands_HSC3_gri | D2 | 1.5492 | 0.9062 | 23754.1855 | 1.9053 | 0.0469 | 0.0033 |
| bands_HSC3_gri | D3 | 0.3095 | 0.6016 | 2298.6665 | 0.9142 | 0.1562 | 0.0667 |
| bands_HSC3_gri | DM | 1.3880 | 0.8984 | 23753.9453 | 1.9196 | 0.0391 | 0.0033 |
| bands_HSC3_gri | DM_alpha05 | 1.2232 | 0.9297 | 23757.5020 | 2.0166 | 0.0078 | 0.0033 |
| bands_HSC5_W1W2 | D12 | 0.0444 | 0.0781 | 1099.5150 | 0.0716 | 0.0469 | 0.0033 |
| bands_HSC5_W1W2 | D1_profile | 0.0598 | 0.0781 | 6169.7266 | 0.0832 | 0.0000 | 0.0033 |
| bands_HSC5_W1W2 | D1_template_marg | 0.0598 | 0.0781 | 6169.7266 | 0.0832 | 0.0000 | 0.0033 |
| bands_HSC5_W1W2 | D1_evidence | 0.0598 | 0.0781 | 6169.6758 | 0.0832 | 0.0000 | 0.0033 |
| bands_HSC5_W1W2 | D2 | 0.2026 | 0.6641 | 70380.6875 | 0.7603 | 0.0000 | 0.0033 |
| bands_HSC5_W1W2 | D3 | 0.0759 | 0.1797 | 7982.8369 | 0.1465 | 0.0078 | 0.0033 |
| bands_HSC5_W1W2 | DM | 0.2026 | 0.6641 | 70380.4062 | 0.7601 | 0.0000 | 0.0033 |
| bands_HSC5_W1W2 | DM_alpha05 | 0.2026 | 0.6641 | 70381.0234 | 0.7601 | 0.0000 | 0.0033 |
| confirmation_Legacy_tef0.0 | D12 | 0.0359 | 0.0391 | 37.6730 | 0.0506 | 0.2969 | 0.0067 |
| confirmation_Legacy_tef0.0 | D1_profile | 0.0761 | 0.2266 | 631.2063 | 0.1274 | 0.0234 | 0.0033 |
| confirmation_Legacy_tef0.0 | D1_template_marg | 0.0761 | 0.2266 | 631.2067 | 0.1277 | 0.0234 | 0.0033 |
| confirmation_Legacy_tef0.0 | D1_evidence | 0.0789 | 0.2266 | 631.2408 | 0.1282 | 0.0234 | 0.0033 |
| confirmation_Legacy_tef0.0 | D2 | 0.5677 | 0.8438 | 23073.8281 | 1.8635 | 0.0078 | 0.0033 |
| confirmation_Legacy_tef0.0 | D3 | 0.0739 | 0.2344 | 2183.2695 | 0.1446 | 0.0703 | 0.0033 |
| confirmation_Legacy_tef0.0 | DM | 0.5677 | 0.8438 | 23073.3242 | 1.8635 | 0.0078 | 0.0033 |
| confirmation_Legacy_tef0.0 | DM_alpha05 | 0.5677 | 0.8438 | 23074.8652 | 1.8633 | 0.0078 | 0.0033 |
| bands_HSC5_W1 | D12 | 0.0412 | 0.0859 | 1068.8130 | 0.0729 | 0.0781 | 0.0033 |
| bands_HSC5_W1 | D1_profile | 0.0642 | 0.1094 | 6044.8066 | 0.0960 | 0.0000 | 0.0033 |
| bands_HSC5_W1 | D1_template_marg | 0.0642 | 0.1094 | 6044.8062 | 0.0960 | 0.0000 | 0.0033 |
| bands_HSC5_W1 | D1_evidence | 0.0642 | 0.1094 | 6044.7544 | 0.0960 | 0.0000 | 0.0033 |
| bands_HSC5_W1 | D2 | 0.2145 | 0.6641 | 64962.4844 | 0.7584 | 0.0000 | 0.0033 |
| bands_HSC5_W1 | D3 | 0.0759 | 0.1719 | 7639.5586 | 0.1463 | 0.0078 | 0.0033 |
| bands_HSC5_W1 | DM | 0.2145 | 0.6641 | 64962.3125 | 0.7584 | 0.0000 | 0.0033 |
| bands_HSC5_W1 | DM_alpha05 | 0.2145 | 0.6641 | 64962.8555 | 0.7584 | 0.0000 | 0.0033 |
| bands_Legacy3 | D12 | 0.2339 | 0.9844 | 19.6968 | 1.7028 | 0.1953 | 0.3683 |
| bands_Legacy3 | D1_profile | 0.3535 | 0.5469 | 459.0287 | 0.9731 | 0.1094 | 0.0033 |
| bands_Legacy3 | D1_template_marg | 0.3535 | 0.5469 | 459.0283 | 0.9732 | 0.1094 | 0.0033 |
| bands_Legacy3 | D1_evidence | 0.5948 | 0.6172 | 460.9980 | 1.3500 | 0.0469 | 0.0033 |
| bands_Legacy3 | D2 | 0.3288 | 0.9922 | 8244.6738 | 2.7804 | 0.0469 | 0.0033 |
| bands_Legacy3 | D3 | 0.7650 | 0.8438 | 618.1146 | 1.5312 | 0.1484 | 0.1150 |
| bands_Legacy3 | DM | 0.2959 | 0.9766 | 8243.9727 | 2.8076 | 0.0391 | 0.0033 |
| bands_Legacy3 | DM_alpha05 | 0.2959 | 1.0000 | 8249.2393 | 2.8787 | 0.0000 | 0.0033 |
| bands_Legacy5 | D12 | 0.0359 | 0.0391 | 37.6730 | 0.0506 | 0.2969 | 0.0067 |
| bands_Legacy5 | D1_profile | 0.0761 | 0.2266 | 631.2063 | 0.1274 | 0.0234 | 0.0033 |
| bands_Legacy5 | D1_template_marg | 0.0761 | 0.2266 | 631.2067 | 0.1277 | 0.0234 | 0.0033 |
| bands_Legacy5 | D1_evidence | 0.0789 | 0.2266 | 631.2408 | 0.1282 | 0.0234 | 0.0033 |
| bands_Legacy5 | D2 | 0.5677 | 0.8438 | 23073.8281 | 1.8635 | 0.0078 | 0.0033 |
| bands_Legacy5 | D3 | 0.0739 | 0.2344 | 2183.2695 | 0.1446 | 0.0703 | 0.0033 |
| bands_Legacy5 | DM | 0.5677 | 0.8438 | 23073.3242 | 1.8635 | 0.0078 | 0.0033 |
| bands_Legacy5 | DM_alpha05 | 0.5677 | 0.8438 | 23074.8652 | 1.8633 | 0.0078 | 0.0033 |
| confirmation_Legacy_tef0.1 | D12 | 0.0897 | 0.1875 | 3.5493 | 0.0782 | 0.9062 | 0.0833 |
| confirmation_Legacy_tef0.1 | D1_profile | 0.0623 | 0.1875 | 4.2724 | 0.1101 | 0.7734 | 0.0167 |
| confirmation_Legacy_tef0.1 | D1_template_marg | 0.0616 | 0.1875 | 4.2294 | 0.1098 | 0.7812 | 0.0167 |
| confirmation_Legacy_tef0.1 | D1_evidence | 0.0898 | 0.2344 | 4.3271 | 0.1237 | 0.6641 | 0.0167 |
| confirmation_Legacy_tef0.1 | D2 | 0.9537 | 0.9219 | 28.7148 | 1.5310 | 0.1797 | 0.0133 |
| confirmation_Legacy_tef0.1 | D3 | 0.0514 | 0.2578 | 5.2447 | 0.1114 | 0.3438 | 0.0333 |
| confirmation_Legacy_tef0.1 | DM | 0.8875 | 0.8984 | 28.4626 | 1.4976 | 0.1875 | 0.0100 |
| confirmation_Legacy_tef0.1 | DM_alpha05 | 1.3475 | 0.9375 | 29.5745 | 1.7170 | 0.1562 | 0.0133 |

## Model and probability definitions

D12 replays unrestricted NNLS; c=A*w, A=sum(c), w on the simplex is a reparameterization, not extra identifiability. D1 profiles one discrete library template or averages over a fixed uniform template prior. D2 uses the library-only minimum/maximum4000A-break spectra; D3 adds a spectral-coverage endpoint selected without redshift-error fitting. Reduced freedom can exclude the true spectrum; true-z residual adequacy is reported alongside point error.

DM uses D2 with normalized Beta(1,1) coefficients and a proper half-normal nonnegative amplitude prior. The amplitude integral is analytic; the shape integral uses adaptive peak/boundary-aware quadrature and independent checks. DM_alpha05 fixes the Beta(0.5,0.5) sensitivity. D1_evidence uses the same amplitude prior and normalized1/12 discrete probabilities. Only these proper marginal calculations are called evidence. Profile curves, including D1_template_marg with profiled amplitude, remain explicitly pseudo-likelihoods and are not compared as normalized model evidences.

The broad amplitude scale is1000 times the95th percentile of absolute training-pair flux. It is a proper observer-frame normalization prior, not an intrinsic luminosity prior. Distance/(1+z) factors are absorbed into this explicitly chosen amplitude coordinate; a luminosity prior would require a different redshift-dependent transformation. A proper amplitude prior can alter redshift weighting. Uniform redshift density and actual bin widths are used for every PDF; no source-population redshift prior is added.

Measured errors and the earlier fixed10% observed-flux TEF are separate likelihood conditions, not retuned. Missing/invalid errors remove likelihood terms; signed finite flux remains. Unknown extinction is never filled with a measured zero. Numerical constant terms are retained for proper evidences. Quadrature error is tracked as posterior total variation; inaccurate marginal products are ineligible for decoder use.

## Identifiability, adequacy and attribution

Full chi-square/evidence grids, continuous-zspec residuals, coefficient amplitudes, active matrix rank, competitive modes and plateau widths are saved. Under nonnegative constraints, rank is a local diagnostic; Nbands minus nominal12 is not used as a reduced-chi-square denominator. Raw zspec chi-square>25 is reported as an incompatibility diagnostic, not a universal calibrated chi-square p-value. Marginal modes are evidence modes, not chi-square minima.

The fixed band experiment keeps exactly the same128 galaxies. HSC optical subsets isolate band count within one pipeline; Legacy grz/grzW1/grzW1W2 adds actual audited WISE leverage; equal-five-band HSC versus Legacy remains confounded by filters, errors and measurement pipelines. HSC+WISE mixes cmodel and Tractor apertures and assumes independent calibration/errors only as a diagnostic. No fabricated WISE data or photometric offset correction is used. Joint HSC+Legacy inference is not promoted to primary evidence because measurement compatibility remains unresolved.

Single/two/three/twelve-component and deliberately mismatched simulations use actual filters, known z and explicit noiseless/noisy observations. Low fit residual is not redshift recovery. Paired bootstrap units are galaxies, with correlated survey views kept together. A narrow constrained posterior is not automatically physically adequate.

## Decoder handoff

The numerically checked D2 profile, proper DM (if convergence passes), D12 reference and both error prescriptions are retained. No model is declared universally trustworthy from aggregate NMAD. Stage2B.6 must test matched and mismatched spectra, source convergence and observed-data trust diagnostics before any generalisation claim. Allowed trust inputs exclude true z, true coefficients, family identity and mismatch labels. See stage2a6_handoff.json for precise paths, priors, units and limitations.

## Empirical answers

1. Excessive flexibility contributes to wrong exact-fit plateaus, but is not a sufficient explanation or a universal remedy. On128 independent confirmation galaxies, HSC D12 has 83.6% outliers; single-template D1 reduces this to 18.0%. However median true-z chi-square rises from 228.1 to 4654.4, with essentially no68% coverage. The point improvement is not a validated physical likelihood.

2. The fixed D2 basis performs poorly on real HSC (93.0% outliers, median true-z chi-square 77977.2). D3 and the10% TEF sensitivity do not supply a model that simultaneously restores useful discrimination and measured-error adequacy. Restricting SED freedom can remove valid spectra as well as wrong solutions.

3. Proper D2 coefficient/amplitude marginalisation is numerically converged but does not repair its family mismatch: DM outliers are 93.0%, with CE 48019.9, versus D2 profile CE 48021.7. Volume weighting cannot reconstruct a spectrum outside the family. Proper amplitude and normalized coefficient priors can change weighting; this is not an arbitrary chi-square temperature or a claim of prior-independent evidence superiority.

4. For identical galaxies, Legacy grz/grzW1/grzW1W2 D12 outliers are 98.4%/4.7%/3.9%. Actual HSC+WISE reduces HSC5 outliers from 83.6% to 7.8%. This strongly supports wavelength leverage beyond raw optical band count, while additional constraints, error differences and cmodel/Tractor compatibility remain confounded. HSC+WISE CE remains very poor despite good point statistics.

5/6. No newly tested score is declared trustworthy for arbitrary real galaxies. The original, D1 and the best-characterized low-dimensional numerical models are handed off as separate diagnostics. Stage2B.6 may use exact-family matched simulations and observed-data trust rules to test their conditional usefulness; it must explicitly suppress or report misspecification rather than assume source point gains solve it. No real target labels or test data are used.
