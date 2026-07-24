# Multiband morphology tabular catalogue target

## Status

This document records the agreed design. Implementation was authorized on
2026-07-24 and is now provided by
`aion_magnitude.multiband_morphology_catalogue` plus
`scripts/create_multiband_morphology_catalogue.sh`. The full catalogue run has
not been launched or monitored.

## Objective

Create a new CLAUDS/HSC tabular catalogue with independently named morphology
measurements for the `u`, `g`, `r`, `i`, `z`, and `y` images.

Band suffixes are required so that the existing `u/uS` morphology is not
confused with morphology measured from the actual HSC `grizy` images.

## Image sources

- `u`: existing CLAUDS `u/uS` image tiles.
- `g`, `r`, `i`, `z`, `y`: HSC PDR3 Deep/UltraDeep images read directly from
  `/arc/projects/ots/pdr3_dud/`.
- Do not copy the PDR3 image data to `/scratch`.
- A small cached manifest containing filenames, bands, WCS footprints, and
  other header metadata is allowed; it is not an image-data copy.

The locally inspected PDR3 files are approximately 1,500 patches per HSC
band. Each patch is 4100×4100 pixels and contains a science image, mask, and
variance plane.

## Output catalogue

Build the new catalogue from the original Phosphoros catalogue rather than
appending to the current unsuffixed morphology catalogue. This avoids retaining
ambiguous fields such as `p_spiral` alongside the new band-specific fields.

Use a distinct output name such as:

```text
COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits
```

The creator must preserve the original catalogue row order and object IDs.

## Columns

For every band `x` in `u`, `g`, `r`, `i`, `z`, and `y`, create:

```text
p_spiral_x
p_bar_x
p_elliptical_type_x
axis_ellipticity_x
concentration_C_x
asymmetry_A_x
possible_morphological_mismatch_x
surface_brightness_24_x
surface_brightness_96_x
mean_per_sqarcsec_12_x
mean_per_sqarcsec_24_x
morphology_available_x
```

### Morphology probabilities

`p_spiral_x`, `p_bar_x`, and `p_elliptical_type_x` must describe the spatial
appearance of band `x` itself.

Do not generate these values by using a `u` spatial template scaled by grizy
catalogue flux ratios. No other band's image or catalogue amplitude should be
mixed into a band-specific morphology probability.

The current multi-band Galaxy10 AION head should not simply be applied to a
single band: the earlier single-band trial collapsed toward almost universally
elliptical predictions. The intended implementation should train or load a
separately validated band-isolated AION morphology head. Its training examples
should expose only one image band at a time, and its probabilities should be
temperature-calibrated before catalogue inference.

The three requested probabilities retain the existing Galaxy10 collapse
semantics unless subsequent validation establishes a better label target.

### Direct pixel morphology

Measure `axis_ellipticity_x`, `concentration_C_x`, and `asymmetry_A_x` from
the background-subtracted 96×96 cutout for band `x`.

Use the same mathematical definitions as the existing morphology catalogue.
For HSC, use the science, mask, and variance planes to reject unusable pixels.

`possible_morphological_mismatch_x` compares `p_elliptical_type_x` with
`axis_ellipticity_x` using the existing diagnostic definition and threshold.
It remains a warning flag, not a physical error label.

### Background subtraction

Estimate a robust local background independently for every object and band,
using valid boundary/background pixels and sigma clipping or an equivalent
robust estimator.

Subtract this background before calculating direct morphology and brightness
columns. Preserve signed residual values; do not clip negative pixels, which
would bias faint objects toward positive brightness.

### Background-subtracted total galaxy brightness

These columns are background-subtracted sums over valid pixels and are
intended to track the total observed luminosity of the galaxy within each
cutout:

```text
surface_brightness_24_x = sum of central 24×24 valid pixels after background subtraction
surface_brightness_96_x = sum of full 96×96 valid pixels after background subtraction
```

They are integrated cutout/aperture sums, despite the historical
`surface_brightness` column name.

### Unsubtracted mean per square arcsecond

Calculate:

```text
mean_per_sqarcsec_12_x = mean of raw valid central 12×12 pixels / pixel_area_arcsec2
mean_per_sqarcsec_24_x = mean of raw valid central 24×24 pixels / pixel_area_arcsec2
```

Do not subtract the estimated local background from the
`mean_per_sqarcsec_*` columns. They intentionally describe the observed
luminosity surface density in the image, including its local background
level, rather than galaxy-only integrated luminosity.

Derive `pixel_area_arcsec2` from the local WCS pixel-area matrix. For square
pixels with scale `s` arcsec/pixel, the area is `s²`. Using the WCS determinant
also supports rotated or slightly non-square pixels.

Record the pixel-area calculation and photometric units in catalogue metadata.
Pixel-area normalization alone does not correct different photometric zero
points, so each band must be converted to a documented, consistent flux unit
before cross-band brightness comparisons.

## Quality and availability

Set all derived values for a band to NaN when its image is absent, its centre
falls outside usable coverage, too many pixels are masked, or its signal is
otherwise inadequate.

Set `morphology_available_x=True` only when the required probability and pixel
measurements for band `x` are valid.

Record useful diagnostic counts per band in the run metadata:

- assigned catalogue rows;
- rows with valid 96×96 coverage;
- rows rejected by mask or variance quality;
- rows with valid AION probabilities;
- rows with complete output features.

## PSF and interpretation

The first implementation does not need to PSF-match the bands, but the
catalogue metadata must state that the HSC coadds have band-dependent seeing.
Consequently, differences between band morphology columns can reflect both
real wavelength-dependent morphology and PSF/resolution differences.

The probability columns describe morphology visible in the supplied cutout,
not guaranteed intrinsic morphology.

## Implementation requirements

The eventual creator should:

1. Read all image data in place from CLAUDS and PDR3 storage.
2. Build/reuse a compact WCS footprint manifest.
3. Assign catalogue positions to candidate patches per band.
4. Open and process one patch at a time rather than loading the survey.
5. Extract centred 96×96 cutouts and central 24×24 regions.
6. Use HSC mask and variance planes for pixel-quality decisions.
7. Cache outputs in resumable, per-band memmaps.
8. Refuse to write the final FITS table while assigned rows remain
   unprocessed.
9. Write atomically and verify row count, object order, column presence, and
   finite-value/availability consistency.
10. Provide CLI controls for band selection, cache/output paths, batch size,
    device, row limits, resume, and force-rebuild behavior.

The implementation may include a shell launcher, but the full job is to be
run later by the user. Development validation should be limited to unit tests,
header/WCS checks, and a very small row or patch smoke test.
