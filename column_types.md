# CLAUDS catalogue column types

Source: `data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits` (235 columns).

## Summary

| Group | FITS columns | Scalar dimensions | Default IoTFM status |
|---|---:|---:|---|
| Identity and sky/tile location | 5 | 5 | Excluded |
| Photometric flux measurements | 55 | 55 | Included |
| Photometric flux uncertainties | 55 | 55 | Included |
| Kron radius / size proxies | 11 | 11 | Included |
| Per-band detection and quality flags | 77 | 77 | Included |
| Cross-band morphology, masks, and star classification | 11 | 17 | Included |
| Photo-z estimates, intervals, fit statistics, and likelihoods | 21 | 21 | Excluded as target leakage |
| **Total** | **235** | **241** | **209 included columns** |

The scalar-dimension total exceeds the FITS-column total because
`FLAG_FIELD_BINARY` is one FITS column containing seven values.

## 1. Identity and location metadata (5 columns)

- `ID`: catalogue object identifier.
- `RA`, `DEC`: celestial coordinates.
- `tract`, `patch`: HSC processing/tile location.

These can encode survey footprint, depth, calibration, selection, and sample
construction rather than galaxy physics. The default experiment excludes
`ID` separately from the four location columns. They should be tested only as
explicit leakage/selection-function ablations.

## 2. Direct photometric flux measurements (55 columns)

For each of 11 bands, the catalogue supplies five flux estimators:

- `FLUX_APER_2_*`: fixed-aperture flux, aperture definition 2.
- `FLUX_APER_3_*`: fixed-aperture flux, aperture definition 3.
- `FLUX_PSF_*`: point-spread-function fitted flux.
- `FLUX_KRON_*`: adaptive Kron flux.
- `FLUX_CMODEL_*`: composite-model flux.

Bands/instruments:

- HSC: `HSC-G`, `HSC-R`, `HSC-I`, `HSC-Z`, `HSC-Y`.
- MegaCam: `MegaCam-u`, `MegaCam-uS` (u-star).
- VIRCAM: `VIRCAM-Y`, `VIRCAM-J`, `VIRCAM-H`, `VIRCAM-Ks`.

These are the primary astrophysical signal. Multiple estimators in the same
band are highly correlated, but their differences can contain morphology,
blending, and aperture information.

## 3. Photometric uncertainties (55 columns)

Every flux estimator above has a matching `FLUXERR_*` column. These describe
measurement noise and depth and allow the model to distinguish a faint secure
measurement from a weak or unreliable one. They also carry observing-condition
and selection-function information.

## 4. Size and morphology proxies (11 columns)

- `RADIUS_KRON_<band>` for all 11 bands.

Kron radii encode apparent angular extent and measurement behavior. They may
help distinguish compact sources, extended galaxies, blends, and low-redshift
objects, but can also be unstable when the corresponding detection is poor.

## 5. Per-band detection and quality flags (77 columns)

Seven binary fields are repeated for every band:

- `hasBadPhotometry_*`
- `isDuplicated_*`
- `isNoData_*`
- `isSky_*`
- `isParent_*`
- `notObserved_*`
- `isClean_*`

These are measurement-state metadata, not direct galaxy measurements. They
make missingness, deblending, coverage, and pipeline quality explicit. They are
scientifically legitimate for a catalogue-performance model, but should be
ablated if the goal is to measure information from galaxy photometry alone.

## 6. Cross-band morphology, masks, and star classification (11 columns)

### Compactness and morphology

- `isCompact`
- `isCompact_HSC-G`, `isCompact_HSC-R`, `isCompact_HSC-I`,
  `isCompact_HSC-Z`, `isCompact_HSC-Y`

### Footprint and masking

- `FLAG_FIELD_BINARY`: one seven-element flag vector.
- `isOutsideMask`

### Stellar/template classification

- `Likelihood-Log_star`
- `isStarTemp`
- `isStar`

These can be useful for rejecting stars and characterizing morphology, but may
also encode upstream pipeline decisions. `isStar` and related fields deserve a
separate ablation because they could make the task easier through catalogue
classification rather than through learned photometric structure.

## 7. Photo-z products and direct target leakage (21 columns)

There are three seven-column photo-z product families:

- NIR: `ZPHOT_NIR`, `Z_LOW68_NIR`, `Z_HIGH68_NIR`, `Z_CHI_NIR`,
  `Z_PEAK_NIR`, `Posterior-Log_NIR`, `Likelihood-Log_NIR`.
- Six-band: `ZPHOT_6B`, `Z_LOW68_6B`, `Z_HIGH68_6B`, `Z_CHI_6B`,
  `Z_PEAK_6B`, `Posterior-Log_6B`, `Likelihood-Log_6B`.
- Main: `ZPHOT`, `Z_LOW68`, `Z_HIGH68`, `Z_CHI`, `Z_PEAK`,
  `Posterior-Log`, `Likelihood-Log`.

`ZPHOT` is the present training/evaluation target. All 21 columns are derived
from photo-z inference and must remain excluded from model inputs; including
any of them would introduce direct or near-direct target leakage.

## Recommended ablation groups

To locate why the whole-catalogue experiment behaves differently from the
magnitude-only control, add groups cumulatively in this order:

1. One consistent total-flux estimator across all 11 bands (for example,
   `FLUX_CMODEL_*`).
2. Matching `FLUXERR_CMODEL_*` uncertainties.
3. Other flux estimators (`APER_2`, `APER_3`, `PSF`, and `KRON`).
4. `RADIUS_KRON_*` and compactness fields.
5. Per-band quality/detection flags.
6. Mask/field flags.
7. Stellar/template-classification fields.
8. Location fields only as an explicit final ablation; keep `ID` off.

This sequence separates useful photometric depth and morphology from redundant
measurements and pipeline-state metadata without changing the target or split.
