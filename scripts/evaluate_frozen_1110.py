#!/usr/bin/env python3
"""Gated frozen evaluation and reporting entry point for the 1110 campaign.

Actions:
  self-test   validate metrics/contrast code on archived H1 P predictions
  freeze      create the P/E evaluation gate from frozen identities (explicit)
  h1-e        evaluate fixed H1 A/B/C/F on E (requires gate)
  main-p-e    evaluate the frozen main model on P and E (requires gate)
  report      metrics, paired bootstrap, CSV/JSON/figures/report from saved predictions
"""
from __future__ import annotations

import argparse
import csv
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

from aion_magnitude.redshift_anchor_evaluation import (  # noqa: E402
    fixed_strata,
    load_h1_predictions,
    load_p_texts,
    load_prediction_chunks,
    metrics,
    paired_contrast,
    serialize_e_membership,
    sha,
    stratified_bootstrap_indices,
    write_json,
)
from aion_magnitude.redshift_anchor_calibration import apply_calibration  # noqa: E402
from aion_magnitude.redshift_anchor_evaluation import calibrate_prediction  # noqa: E402

PRIMARY = ("[0.8,1)", "[1,1.5)")


def now():
    return datetime.now(timezone.utc).isoformat()


def require_gate():
    path = OUT / "evaluation/eval_gate.json"
    if not path.exists():
        raise SystemExit("No eval_gate.json; frozen identities and criteria must be gated before P/E inference")
    gate = json.loads(path.read_text())
    assert gate.get("authorized") is True
    return gate


def e_cohort():
    e = serialize_e_membership(OUT / "data/frozen/E_membership.npz", ROOT)
    ledger = json.loads((OUT / "protocol/ledgers/E_membership.json").read_text())
    assert hashlib.sha256(np.ascontiguousarray(e["targetid"]).tobytes()).hexdigest() == ledger["membership_sha256"]
    return e


def freeze(args):
    if not args.execute:
        print(json.dumps(dict(stage="gated", message="freeze requires explicit --execute; no P/E labels are read here"), indent=2))
        return
    selection = json.loads((OUT / "evaluation/model_selection.json").read_text())
    calibrator = json.loads((OUT / "evaluation/main_calibrator.json").read_text())
    state = OUT / "training/main/selected_frozen/trainable_state.pt"
    assert sha(state) == selection["selected_state_sha256"]
    assert calibrator["model_selection"] == str(OUT / "evaluation/model_selection.json")
    assert calibrator["selection_sha256"] == sha(OUT / "evaluation/model_selection.json")
    assert calibrator["config_sha256"] == sha(OUT / "protocol/configs/main_calibration.json")
    assert int(calibrator["selected_position"]) == int(selection["selected_position"])
    e_ledger = json.loads((OUT / "protocol/ledgers/E_membership.json").read_text())
    h1_manifest = json.loads((H1 / "manifest.json").read_text())
    gate = dict(
        created_utc=now(), authorized=True,
        model=dict(state=str(state), sha256=sha(state), selection=str(OUT / "evaluation/model_selection.json"), selection_sha256=sha(OUT / "evaluation/model_selection.json")),
        calibrator=dict(path=str(OUT / "evaluation/main_calibrator.json"), sha256=sha(OUT / "evaluation/main_calibrator.json")),
        E=dict(path=str(OUT / "data/frozen/E_membership.npz"), sha256=sha(OUT / "data/frozen/E_membership.npz"), membership_sha256=e_ledger["membership_sha256"], n=e_ledger["n"]),
        P=dict(path=str(H1 / "evaluation/locked_P.npz"), sha256=h1_manifest["P_sha256"][str(H1 / "evaluation/locked_P.npz")], n=61696),
        H1_references=["H1-A", "H1-B", "H1-C", "H1-F"],
        criteria=dict(primary_bins=PRIMARY, mae=[0.055, 0.090], abs_bias=[0.040, 0.060], outlier=[0.050, 0.200], nmad=[0.055, 0.065], ce=[4.00, 4.70], crps=[0.090, 0.150], coverage68=[0.60, 0.76], bootstrap_draws=1000),
        prohibitions=["no P/E retuning", "no new model selection on P/E", "no H1-B/C/E/F initialisation of the main model", "E and P labels remain evaluation-only"],
        note="Gate freezes identities and criteria only; no P/E inference is performed by this action.",
    )
    write_json(OUT / "evaluation/eval_gate.json", gate)
    print(json.dumps(gate, indent=2))


def build_h1(arm):
    import torch
    from aion_magnitude.highz_h1_head import replace_head
    from aion_magnitude.matched_training import build_fresh_model

    gate = json.loads((H1 / "evaluation/final_checkpoints_frozen.json").read_text())
    state = Path(gate["models"][arm]["primary_directory"]) / "trainable_state.pt"
    assert sha(state) == gate["models"][arm]["primary_trainable_state_sha256"]
    cfg = json.loads((H1 / "protocol/configs" / f"{arm}.json").read_text())
    model, tok, device = build_fresh_model(cfg)
    if cfg.get("head") == "mixture":
        replace_head(model)
    saved = torch.load(state, map_location="cpu", weights_only=True)
    current = dict(model.named_parameters())
    assert set(saved) == {k for k, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for k, v in saved.items():
            current[k].copy_(v.to(current[k].device))
    model.eval()
    return model, tok, device, state


def build_main():
    import torch
    from aion_magnitude.qwen_posttraining import QwenPosttrainingConfig, create_qlora_photoz_model
    from aion_magnitude.redshift_anchor_evaluation import predict_texts  # noqa: F401

    h1cfg = json.loads((H1 / "protocol/configs/H1-A.json").read_text())
    training = dict(h1cfg["training"], z_min=0.0, z_max=4.5, n_z_bins=300, seed=1110, device="cuda")
    cfg = QwenPosttrainingConfig(**training).normalized()
    model, tok, device = create_qlora_photoz_model(cfg, use_dora=False)
    state = OUT / "training/main/selected_frozen/trainable_state.pt"
    saved = torch.load(state, map_location="cpu", weights_only=True)
    current = dict(model.named_parameters())
    assert set(saved) == {k for k, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for k, v in saved.items():
            current[k].copy_(v.to(current[k].device))
    model.eval()
    return model, tok, device


def h1_e(args):
    gate = require_gate()
    e = e_cohort()
    assert sha(OUT / "data/frozen/E_membership.npz") == gate["E"]["sha256"]
    from aion_magnitude.redshift_anchor_evaluation import export_predictions, predict_texts
    cohort = e
    for arm in args.arms:
        if arm not in gate["H1_references"]:
            raise SystemExit(f"{arm} not frozen as an E reference")
        model, tok, device, state = build_h1(arm)
        folder = OUT / "evaluation/E" / arm
        complete = export_predictions(folder, cohort, lambda texts: predict_texts(model, tok, device, texts))
        write_json(folder / "source.json", dict(arm=arm, state=str(state), state_sha256=sha(state), cohort_sha256=cohort["features_sha256"], complete=complete, utc=now()))
        del model
        print("H1_E_COMPLETE", arm, flush=True)


def main_p_e(args):
    gate = require_gate()
    assert sha(OUT / "training/main/selected_frozen/trainable_state.pt") == gate["model"]["sha256"]
    assert sha(OUT / "evaluation/main_calibrator.json") == gate["calibrator"]["sha256"]
    import torch
    from aion_magnitude.redshift_anchor_evaluation import export_predictions, predict_texts
    model, tok, device = build_main()
    cal = json.loads((OUT / "evaluation/main_calibrator.json").read_text())
    temp, coeffs = cal["fit"]["temperature"], cal["fit"]["prior_coeffs"]

    def predictor(texts):
        p = predict_texts(model, tok, device, texts)
        return calibrate_prediction(p, temp, coeffs)

    for population, cohort in (("P", load_p_texts(H1)), ("E", e_cohort())):
        folder = OUT / "evaluation/final" / population
        complete = export_predictions(folder, cohort, predictor)
        write_json(folder / "source.json", dict(model="main", population=population, state_sha256=sha(OUT / "training/main/selected_frozen/trainable_state.pt"), calibrator_sha256=sha(OUT / "evaluation/main_calibrator.json"), cohort_sha256=cohort.get("features_sha256", sha(H1 / "evaluation/locked_P.npz")), complete=complete, utc=now()))
        print("MAIN_COMPLETE", population, flush=True)
    del model


def self_test():
    cohort = load_p_texts(H1)
    rows = np.flatnonzero((cohort["z"] >= 1.0) & (cohort["z"] < 1.5))
    out = {}
    preds = {}
    for arm in ("H1-B", "H1-C"):
        pred = load_h1_predictions(H1, arm, cohort)
        preds[arm] = pred
        out[arm] = metrics(pred, rows)
    draws = np.load("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design/evaluation_policy/bootstrap_indices.npy", mmap_mode="r")
    assert draws.shape == (1000, 61696)
    contrast = paired_contrast(preds["H1-B"], preds["H1-C"], rows, draws)
    expected = dict(mae_b=0.14256, outlier_b=0.35598, mae_c=0.14199, outlier_c=0.34783)
    assert abs(out["H1-B"]["mae"] - expected["mae_b"]) < 1e-5
    assert abs(out["H1-C"]["mae"] - expected["mae_c"]) < 1e-5
    assert abs(out["H1-B"]["outlier"] - expected["outlier_b"]) < 1e-5
    assert abs(out["H1-C"]["outlier"] - expected["outlier_c"]) < 1e-5
    check = dict(created_utc=now(), population="H1 locked P [1,1.5)", rows=int(len(rows)), metrics=out, contrast_H1C_minus_H1B=contrast, expected=expected, passed=True)
    write_json(OUT / "protocol/checks_downstream.json", check)
    print(json.dumps(dict(passed=True, n=len(rows), mae_b=out["H1-B"]["mae"], mae_c=out["H1-C"]["mae"], contrast_mae=contrast["point"]["mae"], contrast_outlier=contrast["point"]["outlier"]), indent=2))


def report(args):
    """Metrics/CSV/plots/report from already exported frozen predictions."""
    require_gate()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    populations = {}
    populations["P"] = load_p_texts(H1)
    populations["E"] = e_cohort()
    models, preds = {}, {}
    for arm in ("H1-A", "H1-B", "H1-C", "H1-F"):
        folder = OUT / "evaluation/E" / arm
        if (folder / "complete.json").exists():
            preds.setdefault(f"h1_{arm}", {})["E"] = load_prediction_chunks(folder, populations["E"])
        try:
            preds.setdefault(f"h1_{arm}", {})["P"] = load_h1_predictions(H1, arm, populations["P"])
        except FileNotFoundError:
            pass
    for population in ("P", "E"):
        folder = OUT / "evaluation/final" / population
        if (folder / "complete.json").exists():
            preds.setdefault("main", {})[population] = load_prediction_chunks(folder, populations[population])
    result = dict(created_utc=now(), populations={}, contrasts={}, models={})
    outdir = OUT / "evaluation/final_metrics"; outdir.mkdir(parents=True, exist_ok=True)
    for pop, cohort in populations.items():
        z = cohort["z"]; result["populations"][pop] = {}
        strata = fixed_strata(z)
        strata.update({"primary_0.8_1.0": np.flatnonzero((z >= .8) & (z < 1)), "primary_1.0_1.5": np.flatnonzero((z >= 1) & (z < 1.5)), "all": np.arange(len(z))})
        for name, pred in [(n, p[pop]) for n, p in preds.items() if pop in p]:
            result["models"].setdefault(name, {})[pop] = {label: metrics(pred, rows) for label, rows in strata.items()}
        with (outdir / f"{pop}_metrics.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["model", "stratum", "n", "mae", "bias", "median_bias", "nmad", "outlier", "ce", "crps", "coverage68", "coverage95", "pit_mean", "mean_width68"])
            for name in result["models"]:
                for label, m in result["models"][name].get(pop, {}).items():
                    w.writerow([name, label, m["n"], m["mae"], m["bias"], m["median_bias"], m["nmad"], m["outlier"], m["ce"], m["crps"], m["coverage68"], m["coverage95"], m["pit_mean"], m["mean_width68"]])
        # Bootstrap contrasts on the primary bins.
        rows = strata["primary_1.0_1.5"]
        if pop == "P":
            draws = np.load("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design/evaluation_policy/bootstrap_indices.npy", mmap_mode="r")
        else:
            bpath = OUT / "evaluation/E_bootstrap_indices.npy"
            if not bpath.exists():
                idx = stratified_bootstrap_indices(z, draws=1000, seed=1110); np.save(bpath, idx)
                write_json(OUT / "evaluation/E_bootstrap.json", dict(path=str(bpath), sha256=sha(bpath), draws=1000, seed=1110, strata="fixed 0-4.5 bins"))
            draws = np.load(bpath, mmap_mode="r")
        refs = [n for n in result["models"] if n.startswith("h1_")]
        for ref in refs:
            if "main" in result["models"] and pop in result["models"]["main"]:
                c = paired_contrast(preds[ref][pop], preds["main"][pop], rows, draws)
                result["contrasts"].setdefault(pop, {})[f"main_minus_{ref}"] = c
    write_json(outdir / "results.json", result)
    # Recovery and calibration figures.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for j, pop in enumerate(("P", "E")):
        cohort = populations[pop]; z = cohort["z"]
        if "main" in preds and pop in preds["main"]:
            pred = preds["main"][pop]
            axes[0, j].hexbin(z, pred["z_p50"], gridsize=60, mincnt=1, bins="log")
            axes[0, j].plot([0, 2], [0, 2], "k:", lw=1)
            axes[0, j].set(title=f"{pop}: main calibrated", xlabel="spec-z", ylabel="z_p50", xlim=(0, 2), ylim=(0, 2))
            cdf = np.cumsum(pred["pz"], axis=1)
            pit = cdf[np.arange(len(z)), np.clip(np.searchsorted(np.linspace(0, 4.5, 301), z, side="right") - 1, 0, 299)]
            axes[1, j].hist(pit, np.linspace(0, 1, 21), density=True, histtype="step")
            axes[1, j].set(title=f"{pop}: PIT", xlabel="PIT", ylabel="density")
    fig.tight_layout(); fig.savefig(outdir / "recovery_calibration.png", dpi=150); plt.close(fig)

    report_dir = OUT / "reports"; report_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# RedFM 1110 anchor-recovery report", "",
             f"Generated {now()}. Metrics are frozen prediction exports; no P/E retuning was performed.", "",
             "Primary bins are [0.8,1.0) and [1.0,1.5); CE is native-grid categorical NLL, CRPS in redshift units, NMAD uses the 1.48 lower-median convention.", ""]
    for pop in ("P", "E"):
        if pop not in result.get("models", {}).get("main", {}):
            continue
        lines += [f"## Population {pop}", "", "| model | stratum | n | MAE | bias | outlier | NMAD | CE | CRPS | cov68 | width68 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, cells in result["models"].items():
            cell = cells.get(pop, {}).get("primary_1.0_1.5")
            if not cell:
                continue
            lines.append(f"| {name} | [1,1.5) | {cell['n']} | {cell['mae']:.5f} | {cell['bias']:.5f} | {cell['outlier']:.3f} | {cell['nmad']:.5f} | {cell['ce']:.4f} | {cell['crps']:.4f} | {cell['coverage68']:.3f} | {cell['mean_width68']:.4f} |")
        lines += ["", "Paired contrasts (main minus reference, same bootstrap draws):", ""]
        for name, c in result.get("contrasts", {}).get(pop, {}).items():
            lines.append(f"- {name}: MAE {c['point']['mae']:+.5f} [{c['ci95']['mae'][0]:+.5f}, {c['ci95']['mae'][1]:+.5f}], outlier {c['point']['outlier']:+.3f} [{c['ci95']['outlier'][0]:+.3f}, {c['ci95']['outlier'][1]:+.3f}]")
        lines.append("")
    lines += ["## Interpretation constraints", "",
              "- The 256-object high-z budget and the 64-object development split limit precision, especially at [1,1.5).",
              "- Synthetic oversampling is a learning device, not a target n(z). P and E are distinct populations; E includes the parent catalogue's broader DESI selection.",
              "- No result here establishes foundation-pretraining causality, seed robustness, or equal information per label.",
              "- The role-split sorting limitation and high-z fitting coverage are documented in logs/decision_log.md.", ""]
    (report_dir / "redfm_1110_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(models=list(result["models"]), contrasts=list(result["contrasts"]), report=str(report_dir / "redfm_1110_report.md")), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["self-test", "freeze", "h1-e", "main-p-e", "report"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--arms", nargs="+", default=["H1-A", "H1-B", "H1-C", "H1-F"])
    args = ap.parse_args()
    if args.action == "self-test":
        self_test()
    elif args.action == "freeze":
        freeze(args)
    elif args.action == "h1-e":
        h1_e(args)
    elif args.action == "main-p-e":
        main_p_e(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
