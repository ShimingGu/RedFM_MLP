#!/usr/bin/env python3
"""Write the phase-1 campaign manifest and overall status."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with __import__("numpy").load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    frozen = json.loads((OUT / "manifest.json").read_text())
    synth = json.loads((OUT / "data/synthetic/synthetic.json").read_text())
    train = json.loads((OUT / "data/training/training_data.json").read_text())
    access = json.loads((OUT / "protocol/ledgers/spectrum_access_summary.json").read_text())
    fit_status = json.loads((OUT / "status/fit_status.json").read_text())
    train_status = json.loads((OUT / "training/main/status.json").read_text()) if (OUT / "training/main/status.json").exists() else {}
    ckpt = sorted((OUT / "training/main/checkpoints").glob("resume-slot*.pt"))
    manifest = dict(
        version="redshift-anchor-1110-phase1-v2",
        created_utc=datetime.now(timezone.utc).isoformat(),
        worktree="/scratch/RedFM_MLP/redshift-anchor-1110-worktree",
        branch="redshift/anchors-1110",
        base_commit="0fb1e01360759aa840fe52cec176e68467462c78",
        output_root=str(OUT),
        frozen_inputs=dict(path=str(OUT / "manifest.json"), sha256=sha(OUT / "manifest.json"), anchors=256, fitting=192, development=64, lowz_source=10489, lowz_spectra=2048, E_objects=frozen["E_objects"]),
        spectra=dict(summary=access, ledger=str(OUT / "protocol/ledgers/spectrum_access.jsonl")),
        fits=dict(status=fit_status, lowz=int(len(list((OUT / "data/fits/lowz").glob("*.npz")))), anchor=int(len(list((OUT / "data/fits/anchor").glob("*.npz")))), pre_age_correction=str(OUT / "data/fits_pre_age_correction")),
        synthetic=dict(path=str(OUT / "data/synthetic/synthetic.npz"), sha256=synth["sha256"], n=synth["n"], parent_counts=synth["parent_counts"], z_counts=synth["z_counts"], proposal=synth["proposal"]),
        training_data=dict(meta=str(OUT / "data/training/training_data.json"), sha256=sha(OUT / "data/training/training_data.json"), pool=train["hashes"]["pool.npz"], sequence=train["hashes"]["sequence.npz"], validation=train["hashes"]["validation.npz"], component_counts=train["sequence_counts"], presentations=train["presentations"]),
        comparator=dict(cache=str(OUT / "data/physics/template_band_flux.npz"), sha256=sha(OUT / "data/physics/template_band_flux.npz"), dev=str(OUT / "evaluation/comparator_dev.json"), dev_sha256=sha(OUT / "evaluation/comparator_dev.json")),
        comparator_calibration=dict(path=str(OUT / "evaluation/comparator_calibrator.json"), sha256=sha(OUT / "evaluation/comparator_calibrator.json"), config=str(OUT / "protocol/configs/comparator_calibration.json"), pz_cache=str(OUT / "data/comparator/calibration_pz.npz"), pz_cache_sha256=sha(OUT / "data/comparator/calibration_pz.npz")),
        direct_comparator=dict(data=str(OUT / "data/training_direct/training_data.json"), data_sha256=sha(OUT / "data/training_direct/training_data.json"), config=str(OUT / "protocol/configs/direct_comparator.json") if (OUT / "protocol/configs/direct_comparator.json").exists() else None, launched=any((OUT / "training/direct").glob("checkpoints/resume-slot*.pt"))),
        downstream=dict(audit=str(OUT / "protocol/downstream_audit.json"), audit_sha256=sha(OUT / "protocol/downstream_audit.json"), self_test=str(OUT / "protocol/checks_downstream.json"), self_test_sha256=sha(OUT / "protocol/checks_downstream.json"), eval_gate_created=(OUT / "evaluation/eval_gate.json").exists()),
        checks=dict(physics=sha(OUT / "protocol/checks_physics.json"), preflight=sha(OUT / "protocol/checks_training_preflight.json"), synthetic_colors=sha(OUT / "protocol/checks_synthetic_colors.json"), firewall=sha(OUT / "protocol/checks_firewall.json"), e_serialization=sha(OUT / "protocol/checks_e_serialization.json"), supervisor=sha(OUT / "protocol/checks_supervisor.json")),
        first_validation=json.loads((OUT / "status/validation_summary.json").read_text()) if (OUT / "status/validation_summary.json").exists() else None,
        development_analysis=str(OUT / "status/development_analysis_20480.json") if (OUT / "status/development_analysis_20480.json").exists() else None,
        development_analysis_51200=str(OUT / "status/development_analysis_51200.json") if (OUT / "status/development_analysis_51200.json").exists() else None,
        implementation_audit_51200=str(OUT / "status/implementation_audit_51200.json") if (OUT / "status/implementation_audit_51200.json").exists() else None,
        selection_penalty_audit_51200=str(OUT / "status/selection_penalty_audit_51200.json") if (OUT / "status/selection_penalty_audit_51200.json").exists() else None,
        development_analysis_markdown=str(OUT / "logs/development_analysis_20480.md") if (OUT / "logs/development_analysis_20480.md").exists() else None,
        selection_penalty_audit=str(OUT / "status/selection_penalty_audit.json") if (OUT / "status/selection_penalty_audit.json").exists() else None,
        revision_option=dict(markdown=str(OUT / "logs/revision_option_A.md") if (OUT / "logs/revision_option_A.md").exists() else None, config=str(OUT / "protocol/configs/revision_A_1110.json") if (OUT / "protocol/configs/revision_A_1110.json").exists() else None, feasibility=str(OUT / "status/revision_feasibility.json") if (OUT / "status/revision_feasibility.json").exists() else None, launched=False),
        training=dict(config=str(OUT / "protocol/configs/main.json"), config_sha256=sha(OUT / "protocol/configs/main.json"), status=train_status, checkpoints=[dict(path=str(p), sha256=sha(p), position=__import__("torch").load(p, map_location="cpu", weights_only=False)["position"]) for p in ckpt]),
        environment=dict(path=str(OUT / "protocol/environment.json"), sha256=sha(OUT / "protocol/environment.json")),
        provenance=dict(path=str(OUT / "protocol/provenance.json"), sha256=sha(OUT / "protocol/provenance.json")),
        decision_log=str(OUT / "logs/decision_log.md"),
        firewall=dict(P_sealed=True, E_sealed=True, development_spectra_read=False, highz_budget=256),
    )
    (OUT / "protocol/campaign_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    overall = dict(
        updated_utc=datetime.now(timezone.utc).isoformat(),
        stage="main_training_running",
        isolation="complete",
        frozen_data="complete",
        physics_environment="complete",
        spectra_fit=dict(complete=2240, lowz=2048, anchor=192),
        synthetic=dict(complete=synth["n"], parents=synth["parents"]),
        training_data="complete",
        main_training=train_status,
        first_validation=str(OUT / "status/validation_summary.json") if (OUT / "status/validation_summary.json").exists() else None,
        development_analysis=str(OUT / "status/development_analysis_20480.json") if (OUT / "status/development_analysis_20480.json").exists() else None,
        development_analysis_51200=str(OUT / "status/development_analysis_51200.json") if (OUT / "status/development_analysis_51200.json").exists() else None,
        implementation_audit_51200=str(OUT / "status/implementation_audit_51200.json") if (OUT / "status/implementation_audit_51200.json").exists() else None,
        selection_penalty_audit_51200=str(OUT / "status/selection_penalty_audit_51200.json") if (OUT / "status/selection_penalty_audit_51200.json").exists() else None,
        development_analysis_markdown=str(OUT / "logs/development_analysis_20480.md") if (OUT / "logs/development_analysis_20480.md").exists() else None,
        selection_penalty_audit=str(OUT / "status/selection_penalty_audit.json") if (OUT / "status/selection_penalty_audit.json").exists() else None,
        revision_option=str(OUT / "logs/revision_option_A.md") if (OUT / "logs/revision_option_A.md").exists() else None,
        supervisor=str(OUT / "status/supervisor.json"),
        posttrain=str(OUT / "status/posttrain.json") if (OUT / "status/posttrain.json").exists() else None,
        report_supervisor=str(OUT / "status/report_supervisor.json") if (OUT / "status/report_supervisor.json").exists() else None,
        gpu_lease="MIG-ed1476fa-775b-52cc-85d6-6b7ad1ef2323",
        blocked="none for training; final P/E evaluation intentionally not executed",
        notes=["age-of-Universe consistency correction applied to fits before training", "development/P/E data never used in optimisation", "comparator development reference shows positive bias and is not used for training", "role-split implementation sorted selected anchors by z before assigning fitting/development; disclosed in decision_log.md; no redraw after spectra access", "first 10240 validation is head-only: low-z V MAE .0316 but dev [1,1.5) MAE .158; joint training and later validation pending"],
    )
    (OUT / "status/overall.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
