# Input and unit audit

The source loader reads `HSC_metadata/{g,r,i,z,y}_cmodel_mag` and
`*_cmodel_magsigma` from the published GalaxiesML-Spectra v2 HDF5. The package
maps each valid AB magnitude m to `10**((22.5-m)/2.5)` nMgy and its measured
magnitude uncertainty to `f*ln(10)/2.5*sigma_m`. The original selection requires
finite `0<m<40`, `0<sigma_m<10`, at least three valid bands, and DESI quality
cuts. Invalid pairs become NaN. This magnitude-only source cannot preserve or
recover original nonpositive fluxes. Stage 0 independently reconstructs all
22,444 training rows bit exactly against the archived float32 feature matrix.
Source image and spectral arrays, redshift arrays in the HDF5, and held-out test
photometry are never read. Root IDs are used only to locate training rows.

CLAUDS reads the actual `FLUX_CMODEL_HSC-[G,R,I,Z,Y]` and matching `FLUXERR` fields
specified in band_audit.csv. Its measured errors scale linearly with flux:

```
f_nMgy = 10**((22.5-23.9)/2.5) * f_microJy
       = 0.275422870333817 * f_microJy
m_AB   = 23.9 - 2.5*log10(f_microJy)       # positive flux only
m_old  = 23.0 - 2.5*log10(f_microJy) = m_AB - 0.9
```

The historical optional conversion was too large by `10**0.36=2.29087`.
All native archived serializers remain unchanged. Actual finite signed/zero
fluxes survive. Non-finite/-99 flux or `notObserved`/`isNoData` means missing;
nonpositive/non-finite error means missing error. The four raw flag arrays are
verified against the archive. `hasBadPhotometry` and `isClean` are recorded
without introducing a quality cut. No uncertainty floor or imputation is added.

Both sides use HSC grizy nominal system identifiers. The registry acknowledges
unresolved r/r2 and i/i2 mixtures. cModel describes a profile-fitting measurement,
not aperture photometry; matching that name and filter system does not prove
identical extraction, deblending, calibration, selection, or pipeline versions.
The CLAUDS publication documents forced measurements across bands, microJy
cModel fluxes, and an extinction-uncorrected hscPipe/Phosphoros path. The source
HDF5 exposes cModel magnitudes but no definitive per-column pipeline/extinction
attributes or corresponding imaging-quality flags. No new extinction correction
is applied; source `MW_EBV` is as supplied, target EBV remains unavailable.
The exact local processing histories remain unresolved. See
[Desprez et al. (2023), sections 3.1/4.1 and Table E.2](https://arxiv.org/pdf/2301.13750)
and the [GalaxiesML-Spectra publisher record](https://zenodo.org/records/16989593).

The source field order is interleaved g flux/error, r flux/error, i flux/error,
z flux/error, y flux/error, then MW_EBV. Five decimals, `NA`, the prefix and schema
are reproduced by the existing serializer. It appends no prediction/query/EOS
token; the collator applies the existing tokenizer and last nonpadding token is
pooled. Final numeric EBV in source becomes `NA` in the main target diagnostic;
omission also changes the pooled field. Full tokenizer IDs and final tokens for
photometrically preselected examples are in inputs/. All records fit 512 tokens.

`source_exact_*` uses the original source survey identifier embedded in `schema=`
and original PDR2 field names as **counterfactual diagnostic aliases**. These
are not an assertion of CLAUDS provenance. `target_schema_na` changes only the
schema survey name to CLAUDS, leaving the numerical observations fixed.
There is no independent `survey=` field in the original source schema.
These aliases must not be adopted as a scientifically truthful production
representation without explicit provenance. No missing target EBV is filled.
