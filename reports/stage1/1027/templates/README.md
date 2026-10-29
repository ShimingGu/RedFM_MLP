These four extended empirical CWW galaxy SEDs are vendored unchanged from
gbrammer/eazy-photoz at the commit recorded in manifest.json. The first column
is wavelength in Angstrom; the second is relative f_lambda. The forward model
converts to an Lnu shape, applies the explicitly saved continuum/dust parameters,
and normalizes at 5500 Angstrom. No real CLAUDS data determine these choices.

The full continuous emergent-SED instances are split by identity. Empirical base
templates are shared across the primary instance splits; the separate stronger
test holds out the entire double-break/emission spectral family. This distinction
is retained in the scientific report.
