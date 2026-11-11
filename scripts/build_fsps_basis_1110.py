#!/usr/bin/env python3
"""Build a modest, cached FSPS stellar-population basis for SED completion."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
SPS_HOME = OUT / "physics/fsps"
ZMET = [1, 5, 8, 10, 12]
AGES = [0.03, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0, 8.0, 11.0]
AV = [0.0, 0.3, 0.7, 1.2]
NEB = [0, 1]


def main():
    os.environ["SPS_HOME"] = str(SPS_HOME)
    import fsps
    import sys

    sys.path.insert(0, str(ROOT))
    from aion_magnitude.redshift_anchor_physics import attenuation

    out = OUT / "data/physics"
    out.mkdir(parents=True, exist_ok=True)
    wave = None
    records = []
    started = time.time()
    for zmet in ZMET:
        for neb in NEB:
            sp = fsps.StellarPopulation(
                compute_vega_mags=False,
                sfh=0,
                zmet=zmet,
                add_dust_emission=False,
                add_neb_emission=bool(neb),
            )
            z_actual = float(sp.zlegend[zmet - 1])
            for age in AGES:
                w, lum = sp.get_spectrum(tage=float(age), peraa=True)
                w = np.asarray(w, dtype=np.float64)
                lum = np.asarray(lum, dtype=np.float64)
                if wave is None:
                    wave = w
                else:
                    assert np.array_equal(wave, w)
                for av in AV:
                    flux = lum * attenuation(wave, av)
                    records.append((zmet, z_actual, age, neb, av, flux))
                print(f"zmet={zmet} Z={z_actual:.3e} age={age} neb={neb} av=done elapsed={time.time()-started:.1f}s", flush=True)
    flux = np.asarray([r[-1] for r in records], dtype=np.float64)
    meta = np.asarray([r[:-1] for r in records], dtype=np.float64)
    mask = (wave >= 4000) & (wave <= 6000)
    scale = np.median(flux[:, mask], axis=1)
    flux = flux / scale[:, None]
    np.savez_compressed(
        out / "fsps_basis.npz",
        wave=wave.astype(np.float64),
        flux=flux.astype(np.float32),
        zmet=meta[:, 0].astype(np.int16),
        metallicity=meta[:, 1].astype(np.float32),
        age_gyr=meta[:, 2].astype(np.float32),
        nebular=meta[:, 3].astype(np.int8),
        av=meta[:, 4].astype(np.float32),
        normalization_scale=scale.astype(np.float32),
    )
    provenance = dict(
        created_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        fsps_version=fsps.__version__,
        sps_home=str(SPS_HOME),
        fsps_commit=(OUT / "physics/fsps_commit.txt").read_text().strip(),
        stellar_library="C3K_LR (FSPS default; C3K v2.3 spectra at R~100, ATLAS12 outside 3000-25000 A)",
        wavelength_grid=dict(min_angstrom=float(wave.min()), max_angstrom=float(wave.max()), n=len(wave)),
        library_coverage=dict(
            c3k_lr="100 A - 100 micron output coverage; C3K v2.3 native 3000-25000 A; ATLAS12 opacity-binned spectra outside that range",
            desi_spectra="3600-9824 A observed; rest coverage depends on redshift and is recorded per object",
            note="rest-frame 1000-3000 A is a physical-model prior for low-z sources, not a measured spectrum",
        ),
        parameters=dict(zmet=ZMET, age_gyr=AGES, av=AV, nebular=NEB),
        n_templates=int(len(records)),
        normalization="median L_lambda over rest 4000-6000 A set to unity; fit amplitude carries the absolute scale",
        dust="F99 A(lambda)/A(V), R_V=3.1, clamped to 0.0912-3 micron; applied in rest frame",
        nebular="FSPS default nebular emission for instantaneous-burst SSPs; on/off templates allow a nonnegative mixture",
        warning="FSPS is a physical prior, not a measurement; replaced DESI spectral pixels and positive IVAR are not treated as unmodified truth",
    )
    (out / "fsps_basis.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("WROTE", out / "fsps_basis.npz", flux.shape, flush=True)


if __name__ == "__main__":
    main()
