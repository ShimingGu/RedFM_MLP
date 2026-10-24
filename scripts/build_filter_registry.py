#!/usr/bin/env python3
"""Vendor pinned public response curves and compute SED-independent summaries."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np
from astropy.table import Table
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]


def response_summary(wavelength_nm, response):
    x, y = np.asarray(wavelength_nm, float), np.asarray(response, float)
    if (x.ndim != 1 or x.shape != y.shape or len(x) < 3
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(x <= 0) or np.any(np.diff(x) <= 0)):
        raise ValueError("Finite, positive, increasing response wavelengths required")
    # Measured scan noise can be slightly negative outside the passband.
    clipped = int(np.count_nonzero(y < 0))
    y = np.maximum(y, 0)
    if y.max() <= 0:
        raise ValueError("Response must contain positive throughput")
    return dict(
        pivot_nm=float(np.sqrt(np.trapezoid(y * x, x) / np.trapezoid(y / x, x))),
        width_nm=float(np.trapezoid(y, x) / y.max()),
        negative_response_samples_clipped=clipped,
    )


def specifications():
    rows = []
    def add(key, name, instrument, note="Nominal system response; not an object-specific throughput."):
        rows.append((key, "desihub/speclite", f"speclite/data/filters/{name}.ecsv", instrument, note))
    for b in ("g", "r", "i", "z", "y", "r2", "i2"):
        add(f"HSC.{b}", f"hsc2017-{b}", "Subaru/HSC",
            "HSC 2017 nominal total response. r/r2 and i/i2 exposure mixtures are not available in the existing feature cache; r and i are explicitly representative approximations.")
    for b, name in (("u", "u"), ("u_star", "ustar")):
        add(f"MegaCam.{b}", f"cfht_megacam-{name}", "CFHT/MegaCam",
            f"CFHT filter MP{'9302' if b == 'u' else '9301'}; average system response including nominal atmosphere, not a spatially resolved response.")
    for b in ("g", "r", "z"):
        add(f"DECam.{b}", f"decamDR1-{b}", "CTIO/DECam",
            "DECam DR1 total response at airmass 1.2; documented representative response for Legacy DR9 southern DECam photometry, not an exact per-exposure DR9 curve.")
    for b in ("g", "r"):
        add(f"BASS.{b}", f"BASS-{b}", "Bok/90Prime")
    add("MzLS.z", "MzLS-z", "Mayall/Mosaic-3")
    for b in ("W1", "W2"):
        add(f"WISE.{b}", f"wise2010-{b}", "WISE",
            "Wright et al. 2010 weighted mean relative spectral response; speclite zeros values below 1e-4 of maximum. AB flux convention comes from the catalogue, not the WISE Vega zero point.")
    for b in ("Y", "J", "H", "Ks"):
        rows.append((f"VIRCAM.{b}", "bd-j/sedpy", f"sedpy/data/filters/vista_vircam_{b}.par", "VISTA/VIRCAM",
                     "ESO component curves combined by sedpy: filter, QE, mirrors and model atmosphere (airmass 1, PWV 1mm); nominal response, not object-specific."))
    return rows


def build(output, revisions=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Registry already exists; choose a new version/path")
    commits = {}
    for repo in ("desihub/speclite", "bd-j/sedpy"):
        if revisions and repo in revisions:
            commits[repo] = revisions[repo]
            continue
        with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/commits/main", timeout=30) as f:
            commits[repo] = json.load(f)["sha"]
    curve_dir = output.parent / "filter_responses_v1"
    curve_dir.mkdir(parents=True, exist_ok=True)
    def get(spec):
        key, repo, path, instrument, note = spec
        url = f"https://raw.githubusercontent.com/{repo}/{commits[repo]}/{path}"
        raw = urllib.request.urlopen(url, timeout=40).read()
        suffix = Path(path).suffix
        original = curve_dir / f"{key}{suffix}"
        original.write_bytes(raw)
        if suffix == ".ecsv":
            t = Table.read(io.BytesIO(raw), format="ascii.ecsv")
            x = t["wavelength"].quantity.to_value(u.nm)
            y = np.asarray(t["response"], float)
            upstream_meta = t.meta
        else:
            x, y = np.loadtxt(io.BytesIO(raw), unpack=True)
            x = x / 10  # sedpy file header: Angstrom
            y = y / 100  # sedpy VIRCAM file header: transmission in percent
            upstream_meta = {"header": "\n".join(raw.decode().splitlines()[:25])}
        clean = curve_dir / f"{key}.csv"
        np.savetxt(clean, np.column_stack([x, np.maximum(y, 0)]), delimiter=",", header="wavelength_nm,response", comments="", fmt="%.12g")
        return key, dict(instrument=instrument, filter_id=key, source_url=url,
                         source_commit=commits[repo], original_path=str(original.relative_to(output.parent)),
                         original_sha256=hashlib.sha256(raw).hexdigest(),
                         curve_path=str(clean.relative_to(output.parent)),
                         curve_sha256=hashlib.sha256(clean.read_bytes()).hexdigest(),
                         upstream_metadata=upstream_meta, approximation=note,
                         wavelength_unit="nm", **response_summary(x, y))
    with ThreadPoolExecutor(max_workers=6) as pool:
        filters = dict(pool.map(get, specifications()))
    registry = dict(
        version="filter_registry_v1_20261023", retrieved_utc="2026-10-23",
        definitions={
            "response": "Dimensionless relative system throughput R(lambda) for photon-counting response, as distributed by speclite/sedpy; do not multiply by lambda again before computing these summaries.",
            "pivot_nm": "sqrt(integral R(lambda)*lambda dlambda / integral R(lambda)/lambda dlambda)",
            "width_nm": "integral R(lambda) dlambda / max R(lambda) (rectangular equivalent width, not FWHM)",
            "integration": "numpy trapezoid on vendored sample grid in nm; negative scan-noise response clipped to zero; no SED weights, no extra atmosphere or red-leak removal",
        },
        identification_sources=[
            "https://arxiv.org/abs/2301.13750",
            "https://doi.org/10.1093/mnras/stz2522",
            "https://www.legacysurvey.org/release/",
            "https://www.legacysurvey.org/dr9/catalogs/",
            "https://www.cadc-ccda.hia.nrc-cnrc.gc.ca/en/megapipe/docs/filt.html",
            "https://speclite.readthedocs.io/en/latest/filters.html",
        ], filters=filters,
    )
    output.write_text(json.dumps(registry, indent=2, sort_keys=True, default=str) + "\n")
    for key, f in filters.items():
        print(key, f"pivot={f['pivot_nm']:.3f} nm width={f['width_nm']:.3f} nm")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "configs/filter_registry_v1.json")
    p.add_argument("--revisions-from", type=Path, help="Reproduce commits recorded in an existing registry into a new output location")
    args = p.parse_args()
    revisions = None
    if args.revisions_from:
        old = json.loads(args.revisions_from.read_text())
        revisions = {"desihub/speclite": old["filters"]["HSC.g"]["source_commit"],
                     "bd-j/sedpy": old["filters"]["VIRCAM.Y"]["source_commit"]}
    build(args.output, revisions)
