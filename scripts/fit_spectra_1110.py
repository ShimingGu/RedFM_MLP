#!/usr/bin/env python3
"""Fit permitted low-z source and high-z anchor spectra with the FSPS basis."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
HDF5 = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/GalaxiesMLSpec/DESI_HSC_64x64_v2.hdf5")
sys.path.insert(0, str(ROOT))

from aion_magnitude.redshift_anchor_fit import fit_object  # noqa: E402
from aion_magnitude.redshift_anchor_physics import CModelNoise, load_hsc_filters, read_spectrum, spectrum_hdf5_index  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kind", choices=["lowz", "anchor", "all"], default="all")
    ap.add_argument("--draws", type=int, default=16)
    args = ap.parse_args()

    basis = read(OUT / "data/physics/fsps_basis.npz")
    filters = load_hsc_filters(ROOT)
    lowz = read(OUT / "data/frozen/lowz_fit_subset.npz")
    anchors = read(OUT / "data/frozen/anchors.npz")
    lowz_fit = anchors["role"] == "fitting"
    paired_path = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design/paired_split/paired_cohort.npz")
    paired = read(paired_path)
    trows = np.flatnonzero(paired["category"] == "T")
    noise = CModelNoise(paired["hsc_features"][trows], paired["hsc_valid"][trows])

    import h5py

    h5 = h5py.File(HDF5, "r")
    ids, order = spectrum_hdf5_index(h5)
    row_of = {int(t): int(r) for t, r in zip(ids, order)}
    ledger_path = OUT / "protocol/ledgers/spectrum_access.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    if args.kind in ("lowz", "all"):
        for i in range(len(lowz["targetid"])):
            tasks.append(dict(kind="lowz", i=i, targetid=int(lowz["targetid"][i]), z=float(lowz["z"][i]), features=lowz["features"][i], valid=np.ones(5, bool), role="source_training"))
    if args.kind in ("anchor", "all"):
        for i in np.flatnonzero(lowz_fit):
            tasks.append(dict(kind="anchor", i=int(i), targetid=int(anchors["targetid"][i]), z=float(anchors["z"][i]), features=anchors["features"][i], valid=anchors["hsc_valid"][i].astype(bool), role="anchor_fitting", stratum=str(anchors["stratum"][i])))
    if args.limit:
        tasks = tasks[: args.limit]

    started = time.time()
    completed = 0
    skipped = 0
    for task in tasks:
        folder = OUT / "data/fits" / ("lowz" if task["kind"] == "lowz" else "anchor")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{task['targetid']}.npz"
        if path.exists():
            old = read(path)
            if int(old["targetid"]) == task["targetid"] and abs(float(old["z"]) - task["z"]) < 1e-6:
                skipped += 1
                continue
        row = row_of.get(task["targetid"])
        if row is None:
            raise RuntimeError(f"Permitted targetid missing from spectrum file: {task['targetid']}")
        wave, flux, ivar = read_spectrum(h5, row)
        with ledger_path.open("a") as fh:
            fh.write(json.dumps(dict(utc=now(), targetid=task["targetid"], role=task["role"], hdf5_row=row, wave_first=float(wave[0]), wave_last=float(wave[-1]), wave_sha256=sha_array(wave), flux_sha256=sha_array(flux), ivar_sha256=sha_array(ivar))) + "\n")
        flux_phot = task["features"][0:10:2].astype(np.float64)
        sigma_phot = task["features"][1:10:2].astype(np.float64)
        rng = np.random.default_rng(1110 + int(task["targetid"]) % (2 ** 32))
        fit = fit_object(
            basis=basis,
            filters=filters,
            z=task["z"],
            ebv=float(task["features"][10]),
            wave=wave,
            flux=flux,
            ivar=ivar,
            phot_flux=flux_phot,
            phot_sigma=sigma_phot,
            valid_bands=task["valid"],
            rng=rng,
            n_draws=args.draws,
        )
        if fit is None:
            print("FIT_FAILED", task["targetid"], flush=True)
            continue
        payload = dict(
            targetid=np.int64(task["targetid"]),
            z=np.float32(task["z"]),
            role=np.array(task["role"]),
            kind=np.array(task["kind"]),
            features=task["features"].astype(np.float32),
            hsc_valid=task["valid"].astype(bool),
            fit_quality=np.array(json.dumps(fit["quality"])),
            coverage_fraction=np.float32(fit["coverage_fraction"]),
            rest_lo=np.float32(fit["rest_lo"]),
            rest_hi=np.float32(fit["rest_hi"]),
            n_spec_bins=np.int32(fit["n_spec_bins"]),
            median_snr=np.float32(fit["median_snr"]),
            mean_age=np.float32(fit["mean_age"]),
            mean_zmet=np.float32(fit["mean_zmet"]),
            mean_av=np.float32(fit["mean_av"]),
            mean_neb=np.float32(fit["mean_neb"]),
            effective_templates=np.float32(fit["effective_templates"]),
            a_spec=np.float64(fit["a_spec"]),
            a_tot=np.float64(fit["a_tot"]),
            w=fit["w"].astype(np.float32),
            w_draws=fit["w_draws"].astype(np.float32),
            cal=fit["cal"].astype(np.float32),
            phot_dered=fit["phot_dered"].astype(np.float32),
            phot_sigma_dered=fit["phot_sigma_dered"].astype(np.float32),
            spec_wave=fit["spec_wave"].astype(np.float32),
            spec_flux=fit["spec_flux"].astype(np.float32),
            spec_sigma=fit["spec_sigma"].astype(np.float32),
            cal_mask=fit["cal_mask"].astype(bool),
        )
        if task["kind"] == "anchor":
            payload["stratum"] = np.array(task["stratum"])
        tmp = path.with_name(path.name + ".partial.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
        completed += 1
        if completed % 20 == 0:
            elapsed = time.time() - started
            total = len(tasks)
            done = completed + skipped
            rate = completed / max(elapsed, 1e-9)
            write_json(OUT / "status/fit_status.json", dict(
                updated_utc=now(), stage="fitting_spectra", completed=done, total=total,
                lowz_completed=len(list((OUT / "data/fits/lowz").glob("*.npz"))),
                anchor_completed=len(list((OUT / "data/fits/anchor").glob("*.npz"))),
                elapsed_seconds=elapsed, rate_per_second=rate,
                eta_seconds=(total - done) / rate if rate > 0 else None,
                pid=__import__("os").getpid(),
            ))
            print(f"fit {done}/{total} elapsed={elapsed:.1f}s eta={(total-done)/rate if rate>0 else -1:.0f}s", flush=True)
    write_json(OUT / "status/fit_status.json", dict(
        updated_utc=now(), stage="fit_complete", completed=completed + skipped, total=len(tasks),
        lowz_completed=len(list((OUT / "data/fits/lowz").glob("*.npz"))),
        anchor_completed=len(list((OUT / "data/fits/anchor").glob("*.npz"))),
        elapsed_seconds=time.time() - started, pid=__import__("os").getpid(),
    ))
    print("FIT_COMPLETE", completed, "skipped", skipped, "of", len(tasks), flush=True)


if __name__ == "__main__":
    main()
