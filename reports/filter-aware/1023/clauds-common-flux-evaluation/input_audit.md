# Input conversion audit

The catalogue publication specifies AB photometry and **microJy** for every hscPipe `FLUX_CMODEL`/`FLUXERR_CMODEL` band, including VIRCAM (Table E.2). Its hscPipe/Phosphoros flux treatment is uncorrected for Galactic extinction (section 4.1); the separate SExtractor correction statement is not applicable here. The local FITS header has no definitive per-column unit or cmodel extinction-history annotation, so this identification rests on the matching catalogue provenance and schema. [Desprez et al. (2023)](https://arxiv.org/pdf/2301.13750)

The existing loader constructs historical native magnitudes as `23 - 2.5 log10(raw_flux)`. Those AB-labelled values are consequently 0.9 mag brighter than the physical microJy AB convention `23.9 - 2.5 log10(raw_flux)`. The old optional flux helper also assumed zero point 23. **The native serializer and all archived native strings were preserved.** Only the optional common-flux conversion was corrected:

```text
m_AB = 23.9 - 2.5 log10(f_microJy)
f_nMgy = 10**((22.5 - m_AB)/2.5)
       = f_microJy * 10**((22.5 - 23.9)/2.5)
       = 0.2754228703 * f_microJy
sigma_nMgy = 0.2754228703 * measured_sigma_microJy
```

The former scale was 0.6309573445, an overestimate by a factor 2.29087. No Vega-to-AB offset is applied to the NIR flux columns. For positive measurements, the flux transform agrees with the specified magnitude formula. Direct linear error scaling agrees with the first-order magnitude Jacobian `sigma_f = f ln(10)/2.5 sigma_m`; missing errors are not manufactured. No extra Phosphoros uncertainty rescaling or error floor is introduced.

230/256 queries have at least one originally missing/imputed magnitude. The flux interface never exponentiates those imputed entries. It reads raw cmodel flux/error columns, preserves finite signed/zero measurements, and uses `NA` for non-finite/-99 fluxes or measurements flagged `notObserved`/`isNoData`. Non-finite or nonpositive uncertainties become `sigma=NA`. The original masks and `hasBadPhotometry`/`isClean` flags are retained in the sidecar; no new quality cut is imposed. No finite non-sentinel measurement in these queries was additionally excluded by the two availability flags. 5/256 galaxies have no usable flux in any supported band; they remain in the fixed cohort.

No new extinction correction, EBV value, zero-point fitting, imputation, object selection or label information enters the strings. Target `EBV=NA` is retained; source training used its as-catalogued HSC magnitudes transformed to nanomaggies and supplied source MW_EBV. The missing target EBV and incompletely documented local extinction history remain limitations, not quantities inferred from ZPHOT.

This is **not a pure numerical ablation**: `unit` and `sigma_unit` change from `AB_mag` to `AB_nanomaggy`; `mag` becomes `flux`; real flux errors replace `sigma=NA` when available; unobserved imputed entries become explicitly missing, while real negative fluxes become usable observations. Survey, measurement tag, physical filter identities, band order, `extinction=as_loaded`, EBV convention and five-decimal formatting remain fixed. Level 2 retains the original pivot/equivalent-width metadata.

## Bands and tokenisation

Configuration I removes only non-HSC records, retaining the actual Subaru/HSC g,r,i,z,y identifiers and corresponding measurements in source order. Configuration II retains MegaCam u/u_star, HSC grizy and VIRCAM Y/J/H/Ks, with missing bands still explicit. No positional substitutions or invented photometry occur. HSC r/r2 and i/i2 exposure mixtures remain the original registry approximations.

HSC valid-band availability is {0: 6, 1: 3, 4: 10, 5: 237}; 237/256 have all five HSC bands. The per-band raw flux, uncertainty, mask and imputation counts are in [conversion_audit.json](conversion_audit.json). Actual strings, tokens and ordinary/missing examples are in [inputs/](inputs/).

43/256 full-band Level-2 flux records exceed the original 512-token limit; the maximum is 528. Before any new predictions, a single common limit `max(512, largest complete record)=528` was fixed for **every condition**. No observations were truncated. Native records fit below both limits and reproduced archived predictions exactly.

| Condition | Minimum tokens | Median | Maximum | Truncated |
|---|---:|---:|---:|---:|
| A_native_full | 140 | 140 | 140 | 0 |
| B_native_full | 243 | 243 | 243 | 0 |
| B_flux_full | 170 | 248 | 309 | 0 |
| B_flux_hsc | 92 | 152 | 157 | 0 |
| C_native_full | 462 | 462 | 462 | 0 |
| C_flux_full | 389 | 467 | 528 | 0 |
| C_flux_hsc | 190 | 250 | 255 | 0 |
