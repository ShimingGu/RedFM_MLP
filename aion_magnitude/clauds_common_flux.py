"""Fixed observational interface for the CLAUDS common-flux diagnostic."""
from __future__ import annotations

import numpy as np

from .filter_photometry import CLAUDS_BANDS, serialize_photometry

HSC_BANDS = ("g", "r", "i", "z", "y")


def serialize_clauds_common_flux(features, flux, sigma, *, level, band_configuration="full", registry=None):
    """Keep native Level-1/2 structure; exclude bands only by their identities.

    ``flux`` and ``sigma`` already are masked nanomaggies in CLAUDS_BANDS order.
    NA is the established explicit missing-value convention. No imputed
    magnitude is used to supply a missing measurement or uncertainty.
    """
    flux, sigma = np.asarray(flux, float), np.asarray(sigma, float)
    if flux.shape != (len(CLAUDS_BANDS),) or sigma.shape != flux.shape:
        raise ValueError("Eleven named CLAUDS flux/error measurements required")
    if np.any(np.isfinite(sigma) & (sigma <= 0)):
        raise ValueError("Missing uncertainties must be NaN, never zero or negative")
    if level not in (1, 2) or band_configuration not in ("full", "hsc"):
        raise ValueError("Use Level 1/2 and predetermined full/hsc configurations")
    text = serialize_photometry(features, catalogue="clauds", level=level,
                                numerics="flux", registry=registry,
                                clauds_flux=np.stack([flux, sigma], axis=-1))
    if band_configuration == "hsc":
        keys = {f"filter=HSC.{band}" for band in HSC_BANDS}
        text = "\n".join(line for line in text.splitlines()
                         if not line.startswith("filter=") or line.split()[0] in keys)
    return text
