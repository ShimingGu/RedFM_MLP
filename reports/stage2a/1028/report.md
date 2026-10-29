# Stage 2A — physical SED baseline, 1028

Executed EAZY 0.8.7 NNLS on the actual full response curves with 12 pinned FSPS QSF templates and Inoue14 IGM. P0 is a normalized profile-likelihood pseudo-posterior with free nonnegative template amplitudes, not EAZY’s fully configured posterior. No population prior or target-label selection was used.

## Results

| Condition | N | NMAD | Outliers | CE | CRPS | 68% coverage | Median z | Narrow-interval fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HSC_P0 | 5683 | 0.3650 | 0.8024 | 929.2894 | 1.8839 | 0.0591 | 3.1575 | 0.1534 |
| Legacy_P0 | 15053 | 0.0702 | 0.1467 | 35.8797 | 0.1215 | 0.5406 | 0.4575 | 0.0812 |
| HSC_P2_confirmation | 256 | 0.4183 | 0.9102 | 4.6519 | 1.2518 | 0.4141 | 3.2175 | 0.1016 |
| HSC_Pext_screen | 256 | 1.3791 | 0.5977 | 478.0431 | 1.4891 | 0.1602 | 3.1050 | 0.0977 |
| Legacy_P2_confirmation | 256 | 0.1018 | 0.1758 | 3.9474 | 0.1404 | 0.8711 | 0.4125 | 0.2148 |
| Legacy_Pext_screen | 256 | 0.0672 | 0.1523 | 49.9488 | 0.1328 | 0.4883 | 0.3525 | 0.1133 |
| CLAUDS_P0_full | 256 | 0.1602 | 0.4375 | 28659.7480 | 0.4674 | 0.5391 | 1.7925 | 0.0625 |
| CLAUDS_P0_HSC | 256 | 0.2298 | 0.5312 | 728.4473 | 0.5777 | 0.6797 | 2.0550 | 0.0742 |
| CLAUDS_P2_full | 256 | 0.1442 | 0.4141 | 5.9088 | 0.4244 | 0.7617 | 1.7775 | 0.0547 |
| CLAUDS_P2_HSC | 256 | 0.2762 | 0.6055 | 5.4302 | 0.5213 | 0.8555 | 2.1075 | 0.0938 |
| CLAUDS_Pext_full | 256 | 0.1573 | 0.4336 | 27837.1758 | 0.4643 | 0.5156 | 1.7925 | 0.0547 |
| CLAUDS_Pext_HSC | 256 | 0.2362 | 0.5273 | 714.9784 | 0.5927 | 0.6680 | 2.0400 | 0.0781 |

## Scientific assessment

A/B: P0 full-band predictions occupy 153 median bins, span 0.218–3.596 at the 5th–95th percentiles, and have 43.8% ZPHOT outliers. Compare this directly with the historical 96–97% low-z failure; a broad distribution alone is not success.

C: NMAD=0.1602, CRPS=0.4674, CE=28659.748, and nominal 68% coverage=53.9%. These separately expose point error and PDF miscalibration.

D: Actual HSC-only outliers=53.1% and NMAD=0.2298, versus full-band 43.8% and 0.1602. Extra wavelength coverage must be judged from both statistics.

E: The fixed independent extinction correction gives 43.4% outliers and NMAD=0.1573. P2 and all quality/completeness/support strata are separately tabulated. A population prior was deliberately omitted; no claim of prior robustness is made.

F: The response integration, masking and free-amplitude likelihood are suitable verified infrastructure for a small Stage 2B pilot. The observed accuracy and calibration do not establish true-redshift transfer; template mismatch, processing differences and genuine degeneracies remain possible explanations.

## Inputs and interpretation

Full existing HSC validation and recovered Legacy validation are evaluated (audit: {'hsc_rows': 28127, 'hsc_train': 22444, 'hsc_val': 5683, 'legacy_original_validation': 15053, 'clauds_queries': 256, 'filters': ['BASS.g', 'BASS.r', 'DECam.g', 'DECam.r', 'DECam.z', 'HSC.g', 'HSC.i', 'HSC.i2', 'HSC.r', 'HSC.r2', 'HSC.y', 'HSC.z', 'MegaCam.u', 'MegaCam.u_star', 'MzLS.z', 'VIRCAM.H', 'VIRCAM.J', 'VIRCAM.Ks', 'VIRCAM.Y', 'WISE.W1', 'WISE.W2'], 'source_hsc_target_overlap': {'radius_arcsec': 1.0, 'source_objects_excluded': 0, 'excluded_source_ids': [], 'minimum_retained_arcsec': 6.922833364857496}, 'source_legacy_target_overlap': {'radius_arcsec': 1.0, 'source_objects_excluded': 0, 'excluded_source_ids': [], 'minimum_retained_arcsec': 130.84893772037128}, 'source_split_ranges': 'Existing source cohorts already require 0<=z<=4.5; no high-z labels are clipped', 'outside_grid_source_labels': 0, 'target_labels_read': False}). The source screening precedes full fits. Optional fractional template-error candidates 0, 0.03, 0.10 are selected on 256 calibration-half objects per source, balanced by survey; complementary confirmation halves are reserved. Exact calibration selection is in settings_locked.json. All source overlaps with the fixed 256 targets are excluded within 1 arcsec. Repeated source identities use the same calibration half and are reported as shared, not independent.

HSC uses archived grizy cmodel fluxes inferred from magnitudes, with magnitude-limit censoring inherited from the loader. Legacy uses native signed Tractor grzW1W2 fluxes and inverse-variance errors; releases 9010/9012 use DECam and 9011 uses BASS/MzLS, with WISE W1/W2 retained. CLAUDS uses hscPipe cmodel flux in microJy converted by 10^((22.5−23.9)/2.5), including its actual flags and signed values. Nonfinite flux, nonpositive or missing uncertainty and unavailable filter IDs are masked. Fewer than three usable bands yields an explicitly flagged uniform PDF. Quality flags are reported, not silently promoted to missing observations. Source quality/membership remains the archived spectroscopic selection.

The primary configuration uses photometry as loaded without an additional Galactic screen and does not require EBV. The predeclared Pext sensitivity applies G23(Rv=3.1) with independent SFD EBV (source catalogue metadata; target official IRSA DUST queries), using flat-fnu band attenuation. It is an approximate band correction, not proof that historical source and target processing match. Unavailable EBV stays unavailable, never a measured zero. HSC r/r2 and i/i2 exposure mixtures remain a nominal-r/i approximation. These delimited uncertainties do not invalidate the tested interface.

CLAUDS ZPHOT comes from the hscPipe–Phosphoros catalogue. Its template fitting and EAZY share broad stellar-population/template assumptions, though not identical libraries or priors; agreement is not independent redshift truth. No independent spectroscopic column was present in this local catalogue. Existing source-target positional coincidences are excluded from calibration rather than used to tune transfer. See the [catalogue paper](https://arxiv.org/abs/2301.13750), [EAZY method](https://arxiv.org/abs/0807.1533), and [IRSA API](https://irsa.ipac.caltech.edu/applications/DUST/docs/dustProgramInterface.html).

Full UV/NIR versus actual HSC-only measurements, support, completeness, quality, magnitude, S/N and redshift strata are in conditional_metrics.csv. Point-distribution concentration is reported to distinguish object-dependent inference from a constant predictor; uncertainty is assessed with CE, CRPS, interval coverage and PIT, not NMAD alone. Small support strata have high sampling uncertainty. Stress cases use the exact Stage-0.5 identities and archived noise realizations; common amplitude/error scaling is a likelihood symmetry and simulated dimming does not create high-redshift galaxies.

## Reproduction and Stage 2B handoff

Run scripts/stage2_data.py, then scripts/stage2_physics.py build and evaluate in the recorded environment. Per-object profile chi-square, normalized PDFs, flags, kernel, filter translations, dependency versions and hashes are saved beside this report. The kernel has [filter,300 redshift bins,12 templates] and integrates all actual UV through W2 responses. Source test data and historical weights/results are untouched.

Stage 2B can reuse this verified signed-flux/masked-error interface and physical kernel. Retain P0, the ordinary numerical MLP and the shared-band neural-only encoder as controls. Historical extinction equivalence, nominal HSC filters, source redshift coverage and template incompleteness remain scientific limitations. The complete independent Legacy training cache is missing; the available audited HSC–Legacy same-galaxy multi-view pilot must not be described as the definitive population-scale mixed-source experiment.

## Numerical and failure audit

The constant-fnu test agrees to 2.3e-16. Independent Stage-1 photon integration at z=0.42 agrees within 1.40% across all registered responses; the difference reflects conservative spectral interpolation versus point interpolation on throughput knots. This is a measured approximation, not exact equality. The known-mixture chi-square is 1.63e-27.

physics_P0_full: 9/256 unsupported (5 with no bands); usable-band histogram {'0': 5, '1': 2, '2': 2, '3': 1, '4': 3, '5': 52, '6': 68, '7': 65, '9': 5, '10': 14, '11': 39}. Median CE=5.704, 95th percentile=75.219, maximum=7329296.500. The large aggregate P0 CE is driven by highly overconfident incompatible objects; no probability floor hides those failures. See fit_diagnostics.json and per-object PDFs.

physics_P2_full: 9/256 unsupported (5 with no bands); usable-band histogram {'0': 5, '1': 2, '2': 2, '3': 1, '4': 3, '5': 52, '6': 68, '7': 65, '9': 5, '10': 14, '11': 39}. Median CE=4.843, 95th percentile=11.490, maximum=107.436. The large aggregate P0 CE is driven by highly overconfident incompatible objects; no probability floor hides those failures. See fit_diagnostics.json and per-object PDFs.

On the exact Stage-0.5 matched objects, 7-mag common dimming changes the P0 PDF by at most 0; point predictions are unchanged. This symmetry does not compensate for the poor source point estimates or create high-redshift supervision. Error-only broadening and noisy measurements have separate, often poor outcomes in stress_metrics.csv.
