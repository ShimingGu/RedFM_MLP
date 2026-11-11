#!/usr/bin/env python3
"""Calibrate the physical template comparator on budgeted development photometry.

No spectra are read.  Weights and the calibration model are written before the
fit.  The calibration uses the 64 development anchors (parent-stratum weights)
and the 3745 low-z V objects (uniform weights), with equal component weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
H1 = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-hsc-pilot-zcut0p6")
sys.path.insert(0, str(ROOT))

_CACHE = None
_PIVOTS = None
_ZGRID = None


def _init(cache_path, pivots):
    global _CACHE, _PIVOTS, _ZGRID
    _CACHE = dict(np.load(cache_path, allow_pickle=False))
    _PIVOTS = np.asarray(pivots, dtype=np.float64)
    _ZGRID = _CACHE["z_grid"].astype(np.float64)


def _pdf(task):
    from aion_magnitude.redshift_anchor_comparator import template_pdf

    tid, z, features, group, stratum = task
    p = template_pdf(features, _CACHE, _ZGRID, pivots=_PIVOTS)
    return tid, z, p.astype(np.float32), group, stratum


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def object_arrays(pz, z):
    from aion_magnitude.redshift_anchor_calibration import CENTERS, z_bin

    pz = np.asarray(pz, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    cdf = np.cumsum(pz, axis=1)
    p16 = CENTERS[np.argmax(cdf >= 0.16, axis=1)]
    p50 = CENTERS[np.argmax(cdf >= 0.50, axis=1)]
    p84 = CENTERS[np.argmax(cdf >= 0.84, axis=1)]
    delta = (p50 - z) / (1.0 + z)
    ce = -np.log(np.clip(pz[np.arange(len(z)), z_bin(z)], 1e-300, None))
    crps = np.sum((cdf - (CENTERS[None, :] >= z[:, None]).astype(float)) ** 2, axis=1) * np.mean(np.diff(CENTERS))
    return dict(delta=delta, ce=ce, crps=crps, coverage=((z >= p16) & (z <= p84)).astype(float), p50=p50, width=(p84 - p16))


def summarize(pz, z, weights=None):
    a = object_arrays(pz, z)
    w = np.ones(len(z), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    delta = a["delta"]
    med = float(np.median(delta))
    return dict(
        n=int(len(z)),
        mae=float(np.sum(w * np.abs(delta))),
        bias=float(np.sum(w * delta)),
        median_bias=float(med),
        outlier=float(np.sum(w * (np.abs(delta) > 0.15))),
        nmad=float(1.48 * np.median(np.abs(delta - med))),
        ce=float(np.sum(w * a["ce"])),
        crps=float(np.sum(w * a["crps"])),
        coverage68=float(np.sum(w * a["coverage"])),
        mean_width68=float(np.sum(w * a["width"])),
    )


def main():
    from aion_magnitude.redshift_anchor_calibration import apply_calibration, fit_calibration, kish_ess
    from aion_magnitude.redshift_anchor_physics import load_hsc_filters

    filters = load_hsc_filters(ROOT)
    pivots = np.array([filters[b]["pivot_nm"] * 10.0 for b in "grizy"])
    anchors = read(OUT / "data/frozen/anchors.npz")
    lowv = read(H1 / "splits/lowz_only_validation.npz")
    paired = read("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design/paired_split/paired_cohort.npz")
    feat_of = {int(t): paired["hsc_features"][i] for i, t in enumerate(paired["targetid"])}
    dev = anchors["role"] == "development"
    dev_tid = anchors["targetid"][dev]
    dev_z = anchors["z"][dev]
    dev_stratum = anchors["stratum"][dev].astype(str)
    low_tid = lowv["targetid"]
    low_z = lowv["z"].astype(np.float64)

    cache_path = OUT / "data/physics/template_band_flux.npz"
    pz_path = OUT / "data/comparator/calibration_pz.npz"
    config_path = OUT / "protocol/configs/comparator_calibration.json"
    oracle = json.loads((OUT / "protocol/ledgers/oracle_screening.json").read_text())
    config = dict(
        version="comparator_calibration_1110_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        model=dict(temperature=True, n_prior=4, basis="Legendre P0..P3 on x=2z/4.5-1; prior=exp(sum a_k P_k); p_cal ~ p^(1/T)*prior", l2=1e-3),
        weights=dict(
            components=dict(dev_anchor=0.5, lowz_V=0.5),
            dev_anchor="within component, w_stratum = parent stratum count / development count; normalized to sum one",
            lowz_V="uniform within component; normalized to sum one",
            effective_sample_size="Kish 1/sum(w^2)",
        ),
        data=dict(dev_anchor=int(dev.sum()), lowz_V=int(len(low_z)), pz_cache=str(pz_path), pz_cache_sha256=None),
        firewall="development photometry only; no development spectra; no P/E data",
    )
    if config_path.exists():
        old = json.loads(config_path.read_text())
        assert old["version"] == config["version"] and old["weights"] == config["weights"]
        config = old
    else:
        write_json(config_path, config)

    if pz_path.exists():
        cache = read(pz_path)
        assert len(cache["targetid"]) == len(dev_tid) + len(low_tid)
        print("PZ_CACHE_PRESENT", len(cache["targetid"]), flush=True)
    else:
        tasks = [(int(t), float(z), feat_of[int(t)], "dev_anchor", str(s)) for t, z, s in zip(dev_tid, dev_z, dev_stratum)]
        tasks += [(int(t), float(z), feat_of[int(t)], "lowz_V", "") for t, z in zip(low_tid, low_z)]
        result = []
        status_path = OUT / "status/comparator_calibration.json"
        with ProcessPoolExecutor(max_workers=4, initializer=_init, initargs=(str(cache_path), pivots)) as pool:
            for j, item in enumerate(pool.map(_pdf, tasks, chunksize=25)):
                result.append(item)
                if j % 100 == 0:
                    write_json(status_path, dict(utc=datetime.now(timezone.utc).isoformat(), stage="computing_comparator_pdf", completed=j, total=len(tasks), pid=os.getpid()))
                    print("PDF", j, len(tasks), flush=True)
        tid = np.array([r[0] for r in result], dtype=np.int64)
        z = np.array([r[1] for r in result], dtype=np.float64)
        pz = np.stack([r[2] for r in result]).astype(np.float32)
        group = np.array([r[3] for r in result], dtype=str)
        stratum = np.array([r[4] for r in result], dtype=str)
        pz_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = pz_path.with_name(pz_path.name + ".partial.npz")
        np.savez_compressed(tmp, targetid=tid, z=z, pz=pz, group=group, stratum=stratum)
        tmp.replace(pz_path)
        cache = read(pz_path)
        print("PZ_CACHE_WRITTEN", len(tid), flush=True)

    dev_pz = cache["pz"][cache["group"] == "dev_anchor"]
    low_pz = cache["pz"][cache["group"] == "lowz_V"]
    dev_z = cache["z"][cache["group"] == "dev_anchor"]
    low_z = cache["z"][cache["group"] == "lowz_V"]
    dev_stratum = cache["stratum"][cache["group"] == "dev_anchor"].astype(str)
    assert len(dev_pz) == len(dev_tid) and len(low_pz) == len(low_tid)

    # Predeclared weights.
    parent_counts = {f"[{x['z_low']},{x['z_high']})": int(x["available"]) for x in oracle["strata"]}
    dev_counts = {s: int((dev_stratum == s).sum()) for s in set(dev_stratum)}
    w_dev_component = np.array([parent_counts[s] / dev_counts[s] for s in dev_stratum], dtype=np.float64)
    w_dev_component /= w_dev_component.sum()
    w_low_component = np.ones(len(low_z), dtype=np.float64) / len(low_z)
    w_dev = 0.5 * w_dev_component
    w_low = 0.5 * w_low_component
    weights = np.concatenate([w_dev, w_low])
    z_all = np.concatenate([dev_z, low_z])
    pz_all = np.concatenate([dev_pz, low_pz], axis=0)
    assert abs(weights.sum() - 1.0) < 1e-12

    base = dict(
        dev_anchor=summarize(dev_pz, dev_z),
        lowz_V=summarize(low_pz, low_z),
        combined_weighted=summarize(pz_all, z_all, weights),
        weights=dict(dev_component_sum=float(w_dev.sum()), lowz_component_sum=float(w_low.sum()), ess_dev=kish_ess(w_dev_component), ess_lowz=kish_ess(w_low_component), ess_combined=kish_ess(weights)),
    )
    temp_fit = fit_calibration(pz_all, z_all, weights, n_prior=0, temperature_only=True)
    full_fit = fit_calibration(pz_all, z_all, weights, n_prior=4, temperature_only=False)
    temp_pz = apply_calibration(pz_all, temp_fit["temperature"], None)
    full_pz = apply_calibration(pz_all, full_fit["temperature"], full_fit["prior_coeffs"])

    def split_summary(pz_cal):
        return dict(
            dev_anchor=summarize(pz_cal[: len(dev_z)], dev_z),
            lowz_V=summarize(pz_cal[len(dev_z):], low_z),
            combined_weighted=summarize(pz_cal, z_all, weights),
        )

    # Deterministic 5-fold by targetid hash; fit on four folds and report held-out weighted CE/MAE.
    folds = np.array([int(t) % 5 for t in cache["targetid"]])
    cv = []
    for k in range(5):
        train = folds != k
        test = ~train
        if train.sum() == 0 or test.sum() == 0:
            continue
        wtr = weights[train].copy()
        wtr /= wtr.sum()
        fit_k = fit_calibration(pz_all[train], z_all[train], wtr, n_prior=4)
        p_k = apply_calibration(pz_all[test], fit_k["temperature"], fit_k["prior_coeffs"])
        cv.append(dict(fold=k, n_train=int(train.sum()), n_test=int(test.sum()), fit=fit_k, heldout=summarize(p_k, z_all[test], weights[test])))

    calibrator = dict(
        version="comparator_calibrator_1110_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        config=str(config_path),
        config_sha256=sha(config_path),
        pz_cache=str(pz_path),
        pz_cache_sha256=sha(pz_path),
        model=config["model"],
        weights=config["weights"],
        effective_sample_size=base["weights"],
        uncalibrated=base,
        temperature_only=dict(fit=temp_fit, metrics=split_summary(temp_pz)),
        full=dict(fit=full_fit, metrics=split_summary(full_pz)),
        cross_validation=cv,
        selected="full" if split_summary(full_pz)["combined_weighted"]["ce"] <= split_summary(temp_pz)["combined_weighted"]["ce"] else "temperature_only",
        selected_parameters=full_fit if split_summary(full_pz)["combined_weighted"]["ce"] <= split_summary(temp_pz)["combined_weighted"]["ce"] else temp_fit,
        note="Calibration uses development photometry only; no development spectra, no P/E. Metrics are in-sample except the 5-fold held-out entries.",
    )
    path = OUT / "evaluation/comparator_calibrator.json"
    config["data"]["pz_cache_sha256"] = calibrator["pz_cache_sha256"]
    write_json(config_path, config)
    calibrator["config_sha256"] = sha(config_path)
    write_json(path, calibrator)
    write_json(OUT / "status/comparator_calibration.json", dict(utc=datetime.now(timezone.utc).isoformat(), stage="complete", calibrator=str(path), selected=calibrator["selected"], n=int(len(z_all))))
    print(json.dumps(dict(selected=calibrator["selected"], uncalibrated=base["combined_weighted"], full=calibrator["full"]["metrics"]["combined_weighted"], temperature_only=calibrator["temperature_only"]["metrics"]["combined_weighted"], ess=base["weights"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
