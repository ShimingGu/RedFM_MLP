#!/usr/bin/env python3
"""Dev-only temperature + smooth-prior calibration of the frozen main model.

This script never reads P or E labels.  It is intended to be run by the post
supervisor after training and checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
H1 = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-hsc-pilot-zcut0p6")
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def weighted_summary(pz, z, w):
    from aion_magnitude.redshift_anchor_calibration import CENTERS, z_bin

    pz = np.asarray(pz, float); z = np.asarray(z, float); w = np.asarray(w, float); w = w / w.sum()
    cdf = np.cumsum(pz, axis=1)
    p16 = CENTERS[np.argmax(cdf >= .16, axis=1)]; p50 = CENTERS[np.argmax(cdf >= .50, axis=1)]; p84 = CENTERS[np.argmax(cdf >= .84, axis=1)]
    delta = (p50 - z) / (1 + z)
    ce = -np.log(np.clip(pz[np.arange(len(z)), z_bin(z)], 1e-300, None))
    crps = np.sum((cdf - (CENTERS[None, :] >= z[:, None])) ** 2, axis=1) * np.mean(np.diff(CENTERS))
    med = float(np.median(delta))
    return dict(n=int(len(z)), mae=float(np.sum(w * np.abs(delta))), bias=float(np.sum(w * delta)), median_bias=med, outlier=float(np.sum(w * (np.abs(delta) > .15))), nmad=float(1.48 * np.median(np.abs(delta - med))), ce=float(np.sum(w * ce)), crps=float(np.sum(w * crps)), coverage68=float(np.sum(w * ((z >= p16) & (z <= p84)))), mean_width68=float(np.sum(w * (p84 - p16))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps(dict(stage="gated", message="main calibration is implemented but not executed; pass --execute after frozen selection"), indent=2))
        return
    import torch
    from aion_magnitude.qwen_posttraining import QwenPosttrainingConfig, create_qlora_photoz_model
    from aion_magnitude.redshift_anchor_calibration import apply_calibration, fit_calibration, kish_ess
    from aion_magnitude.redshift_anchor_evaluation import predict_texts

    selection = json.loads((OUT / "evaluation/model_selection.json").read_text())
    state_path = OUT / "training/main/selected_frozen/trainable_state.pt"
    assert sha(state_path) == selection["selected_state_sha256"]
    h1cfg = json.loads((H1 / "protocol/configs/H1-A.json").read_text())
    training = dict(h1cfg["training"], z_min=0.0, z_max=4.5, n_z_bins=300, seed=1110, device="cuda")
    config = QwenPosttrainingConfig(**training).normalized()
    model, tokenizer, device = create_qlora_photoz_model(config, use_dora=False)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    current = dict(model.named_parameters())
    assert set(state) == {k for k, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for k, v in state.items():
            current[k].copy_(v.to(current[k].device))
    model.eval()

    val = read(OUT / "data/training/validation.npz")
    texts = [str(x) for x in val["text"]]
    pred = predict_texts(model, tokenizer, device, texts)
    z = val["z"].astype(np.float64)
    group = val["group"].astype(str)
    stratum = val["stratum"].astype(str)
    low = group == "lowz_V"
    dev = group == "dev_anchor"
    # Predeclared weights: equal components; dev within-stratum population weights.
    oracle = json.loads((OUT / "protocol/ledgers/oracle_screening.json").read_text())
    parent = {f"[{x['z_low']},{x['z_high']})": int(x["available"]) for x in oracle["strata"]}
    w_dev = np.array([parent[s] / max(1, int((stratum[dev] == s).sum())) for s in stratum[dev]], dtype=float)
    w_dev /= w_dev.sum()
    w_low = np.ones(low.sum()) / low.sum()
    w = np.concatenate([0.5 * w_dev, 0.5 * w_low])
    z_all = np.concatenate([z[dev], z[low]])
    pz_all = np.concatenate([pred["pz"][dev], pred["pz"][low]], axis=0)
    fit = fit_calibration(pz_all, z_all, w, n_prior=4)
    pz_cal = apply_calibration(pz_all, fit["temperature"], fit["prior_coeffs"])
    config_meta = dict(
        version="main_calibration_1110_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        model_selection=str(OUT / "evaluation/model_selection.json"), selection_sha256=sha(OUT / "evaluation/model_selection.json"),
        weights=dict(components=dict(dev_anchor=.5, lowz_V=.5), dev_anchor="parent stratum count / dev count", lowz_V="uniform", ess=dict(dev=kish_ess(w_dev), lowz=kish_ess(w_low), combined=kish_ess(w))),
        model=dict(temperature=True, n_prior=4, basis="Legendre P0..P3 on x=2z/4.5-1", l2=1e-3),
        firewall="dev-only; no P/E; no development spectra",
        state_sha256=selection["selected_state_sha256"],
    )
    cfg_path = OUT / "protocol/configs/main_calibration.json"
    if not cfg_path.exists():
        write(cfg_path, config_meta)
    result = dict(
        version="main_calibrator_1110_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        config=str(cfg_path), config_sha256=sha(cfg_path), selected_position=selection["selected_position"],
        selected_state_sha256=selection["selected_state_sha256"], fit=fit,
        uncalibrated=dict(dev_anchor=weighted_summary(pred["pz"][dev], z[dev], w_dev), lowz_V=weighted_summary(pred["pz"][low], z[low], w_low), combined=weighted_summary(pz_all, z_all, w)),
        calibrated=dict(dev_anchor=weighted_summary(pz_cal[:dev.sum()], z[dev], w_dev), lowz_V=weighted_summary(pz_cal[dev.sum():], z[low], w_low), combined=weighted_summary(pz_cal, z_all, w)),
        firewall="dev-only; no P/E labels were opened",
    )
    write(OUT / "evaluation/main_calibrator.json", result)
    write(OUT / "status/main_calibration.json", dict(utc=datetime.now(timezone.utc).isoformat(), stage="complete", path=str(OUT / "evaluation/main_calibrator.json")))
    print(json.dumps(dict(fit=fit, uncalibrated=result["uncalibrated"]["combined"], calibrated=result["calibrated"]["combined"]), indent=2))


if __name__ == "__main__":
    main()
