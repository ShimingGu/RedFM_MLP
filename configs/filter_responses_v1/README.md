# Response curves used by filter_registry_v1

Original ECSV/PAR files are vendored unchanged. CSV files contain their wavelength grids in nm and nonnegative relative responses. The source URLs, pinned commits and SHA256 checksums of both forms are in `../filter_registry_v1.json`. Upstream licenses are included here.

The registry builder is `scripts/build_filter_registry.py`. Training and serialization use this offline snapshot; they never query a mutable remote service.

For relative photon-counting system response R(lambda), the definitions are:

- Pivot wavelength: sqrt[integral R(lambda) lambda d(lambda) / integral R(lambda)/lambda d(lambda)].
- Rectangular equivalent width: integral R(lambda) d(lambda) / max R(lambda).

Both are evaluated by trapezoidal integration in nm, independent of the observed galaxy's SED. Width is not FWHM. Negative response samples due to measured scan noise are clipped at zero, with their count recorded. No additional wavelength factor, atmosphere, or object-SED weighting is applied. Relative throughput normalization cancels in both summaries.

The speclite curves supply HSC, MegaCam, DECam, BASS, MzLS and WISE. The sedpy VIRCAM curves combine ESO filter, detector, mirror and atmosphere components. Their original headers preserve the component provenance. WISE RSRs are response curves; catalogue WISE fluxes remain on the AB nanomaggy system, without inserting a Vega conversion.

Here R denotes throughput/QE per incident photon. For an incident energy spectrum f_lambda, detected counts are proportional to the integral of f_lambda * lambda * R(lambda). WISE also distributes lambda*R curves under the name “photon-counting RSR”; these must not receive another lambda factor. The vendored `wise2010` responses are the Wright et al. QE-based curves, so the stated pivot formula uses them directly. See the [WISE calibration supplement, section v](https://irsa.ipac.caltech.edu/data/WISE/docs/release/Prelim/expsup/sec4_3g.html).

These are nominal system responses, not reconstructed responses for every object/exposure. HSC r/r2 and i/i2 mixtures are unknown in the existing feature arrays; original r/i responses are explicit representative approximations. DECam DR1 curves represent Legacy DR9 southern photometry approximately. DR9 RELEASE 9010/9012 maps to DECam grz, while 9011 maps to BASS gr and MzLS z; unknown releases fail rather than receiving guessed optical filters.

CLAUDS names were checked against the real FITS columns and the release paper: u/u_star are MegaCam MP9302/MP9301; grizy are HSC; uppercase Y and JHKs are VIRCAM. Lowercase HSC y and uppercase VIRCAM Y are distinct filters.

Identification references:

- [CLAUDS/HSC catalogue release](https://arxiv.org/abs/2301.13750)
- [CLAUDS u and u-star filter definitions](https://doi.org/10.1093/mnras/stz2522)
- [Legacy release-to-camera mapping](https://www.legacysurvey.org/release/)
- [Legacy DR9 units](https://www.legacysurvey.org/dr9/catalogs/)
- [MegaCam total response conventions](https://www.cadc-ccda.hia.nrc-cnrc.gc.ca/en/megapipe/docs/filt.html)
- [speclite filter provenance](https://speclite.readthedocs.io/en/latest/filters.html)

SVO endpoints were unavailable during this work; no metadata were invented to fill that gap. Physical summaries were calculated directly from the pinned response curves above.
