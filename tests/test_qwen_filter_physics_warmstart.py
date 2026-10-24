"""Level-2 isolation, registry verification, ordered scheduling and full reporting."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import qwen_filter_physics_warmstart as physics
from aion_magnitude import filter_photometry as fp
from aion_magnitude.metrics import predict_photoz_from_logits, summarize_pdf_metrics
from aion_magnitude.qwen_continued_sft import ContinuedSFTProtocol
from aion_magnitude.utils import make_redshift_grid
from dataclasses import asdict


@pytest.mark.parametrize("kind,release", [("galaxiesml-spec", None), ("clauds", None), ("desi-ls", 9010), ("desi-ls", 9011)])
def test_metadata_is_the_only_change_and_is_bound_to_the_named_band(kind, release):
    values = {}
    for i, (_, value, error) in enumerate(fp.feature_columns(kind)):
        values[value] = -2.5 if i == 0 else 20 + i
        if error: values[error] = np.nan if i == 1 else .1
    if kind != "clauds": values["MW_EBV"] = .03
    registry = fp.load_registry()
    kwargs = dict(catalogue=kind, legacy_release=release, registry=registry)
    one = fp.serialize_photometry(values, level=1, **kwargs)
    two = fp.serialize_photometry(values, level=2, **kwargs)
    audit = physics.audit_metadata_delta([one], [two], registry)
    assert audit["only_pivot_and_width_added"] and audit["n"] == 1
    first = two.splitlines()[1].split()[0].removeprefix("filter=")
    pivot = f"pivot_nm={registry['filters'][first]['pivot_nm']:.3f}"
    with pytest.raises(ValueError, match="Incorrect physical metadata"):
        physics.audit_metadata_delta([one], [two.replace(pivot, "pivot_nm=1.000")], registry)
    with pytest.raises(ValueError, match="beyond pivot/width"):
        physics.audit_metadata_delta([one], [two.replace("sigma=NA", "sigma=0.00000") if "sigma=NA" in two else two.replace("EBV=0.03000", "EBV=0.04000")], registry)


def test_registry_integrals_and_documented_approximations():
    registry = fp.load_registry(verify_curves=True)
    audit = physics.audit_registry(registry)
    assert len(audit["filters"]) == 21 and audit["all_integrals_reproduced"]
    assert "r/r2" in audit["filters"]["HSC.r"]["approximation"]
    assert "i/i2" in audit["filters"]["HSC.i"]["approximation"]
    assert audit["filters"]["WISE.W1"]["pivot_nm"] > 3000
    registry["filters"]["HSC.g"]["width_nm"] *= 2
    with pytest.raises(ValueError, match="do not reproduce"): physics.audit_registry(registry)


def test_waiting_for_B_preserves_order_and_handles_pid_reuse(monkeypatch):
    original = dict(pid=1234, start_ticks="100")
    replies = iter([original, original, dict(pid=1234, start_ticks="999")])
    waiting, sleeps = [], []
    monkeypatch.setattr(physics, "process_identity", lambda _:next(replies))
    monkeypatch.setattr(physics.time, "sleep", lambda seconds:sleeps.append(seconds))
    physics.wait_for_predecessor(original, waiting.append)
    assert len(waiting) == 2 and sleeps == [30, 30]
    assert all("Experiment B" in message for message in waiting)


def test_process_identity_reads_linux_start_ticks():
    identity = physics.process_identity(os.getpid())
    assert identity["pid"] == os.getpid() and identity["start_ticks"].isdigit()
    assert physics.process_identity(999999999) is None


def evaluation():
    edges, centers = make_redshift_grid(0, 4.5, 5)
    logits = torch.tensor([[3.,1.,0.,0.,0.], [0.,3.,1.,0.,0.]])
    result = dict(logits=logits, z_spec=torch.tensor([.4, 1.4]), object_id=["val-a", "val-b"], redshift_edges=edges, redshift_centers=centers)
    result.update(predict_photoz_from_logits(logits, centers=centers))
    return result


def test_completed_report_compares_original_A_B_C_with_costs(tmp_path, monkeypatch):
    out, a_dir, b_dir, fast = [tmp_path / name for name in ("level2", "level0", "level1", "fast")]
    for directory in (out, a_dir, b_dir, fast): directory.mkdir()
    monkeypatch.setattr(physics.control, "FAST", fast)
    prediction = evaluation(); metrics = summarize_pdf_metrics(prediction)
    status = dict(phase="complete", global_update=10, max_updates=2806, best_update=0,
        best_metrics=metrics, zero_update_metrics=metrics, elapsed_seconds=5, total_run_wall_seconds=12,
        peak_allocated_gib=5.5, validation_history=[dict(update=0, **metrics), dict(update=10, **metrics)],
        training_history=[dict(update=1, loss=3.), dict(update=10, loss=2.9)])
    manifest = dict(control_dir=str(a_dir), band_control_dir=str(b_dir), reference_metrics=metrics,
        source_config=dict(max_length=512), token_audit=dict(max=260), targets={kind:dict(token_audit=dict(max=490)) for kind in ("desi-ls", "clauds")})
    physics.write_json(out / "manifest.json", manifest)
    physics.write_json(out / "serializer_change.json", dict(original_level0_metrics=metrics, level2_zero_update_metrics=metrics))
    physics.write_json(b_dir / "serializer_change.json", dict(level1_zero_update_metrics=metrics))
    for directory in (out, a_dir, b_dir):
        physics.write_json(directory / "source_summary.json", {key:value for key,value in status.items() if key != "total_run_wall_seconds"})
        physics.write_json(directory / "summary.json", status)
    for name in ("reference_validation.pt", "zero_update_validation.pt", "best_validation.pt"): torch.save(prediction, out / name)
    targets = {kind:dict(metrics=metrics, query_ids=prediction["object_id"]) for kind in ("desi-ls", "clauds")}
    physics.write_json(fast / "summary.json", dict(targets=targets))
    for directory in (out, a_dir, b_dir):
        physics.write_json(directory / "transfer_summary.json", dict(targets=targets))
        for kind in targets:
            np.savez(directory / f"{kind}_selected_predictions.npz", z_spec=prediction["z_spec"].numpy(), z_p50=prediction["z_p50"].numpy())
    physics.plots(out); physics.render_report(out, status)
    report = (out / "report.md").read_text()
    assert "C minus B" in report and "selected Level 0 A" in report and "selected Level 1 B" in report
    assert "original archived Level 0" in report
    assert "Total GPU-run wall time" in report and "0.003" in report
    assert "fixed filter set" in report and "shuffled or incorrect metadata" in report
    assert "r/r2" in report and "i/i2" in report and "ZPHOT, not spectroscopy" in report
    for name in ("learning_curves.png", "serializer_predictions.png", "transfer_predictions.png"):
        assert (out / name).stat().st_size > 1000


def test_execution_evaluates_level2_before_training_and_transfer_after_selection(tmp_path, monkeypatch):
    reference = evaluation(); metrics = summarize_pdf_metrics(reference)
    torch.save(reference, tmp_path / "reference_validation.pt")
    physics.band.write_inputs(tmp_path / "canonical_replay_inputs.jsonl", reference["object_id"], ["native0-a", "native0-b"])
    data = dict(targetid=np.array(["train", "val-a", "val-b"]), z=np.array([.8, .4, 1.4], dtype=np.float32), split=np.array(["train", "val", "val"]))
    texts = ["level2-train", "level2-val-a", "level2-val-b"]
    manifest = dict(protocol=asdict(ContinuedSFTProtocol()), source_config=dict(max_length=512), initial_snapshot_sha256={})
    monkeypatch.setattr(physics, "load_prepared", lambda _:(manifest, data, texts))
    monkeypatch.setattr(physics, "render_report", lambda *args:None)
    monkeypatch.setattr(physics, "plots", lambda *args:None)
    @contextmanager
    def reservation(*args, **kwargs): yield dict(uuid="selected", memory_before_gib=dict(used=0, free=10))
    monkeypatch.setattr(physics.band, "reserve_idle_gpu", reservation)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-selected")
    monkeypatch.setattr(physics.torch.cuda, "reset_peak_memory_stats", lambda:None)
    monkeypatch.setattr(physics.torch.cuda, "max_memory_allocated", lambda:0)
    monkeypatch.setattr(physics, "set_random_seed", lambda _:None)
    events = []
    monkeypatch.setattr(physics.control, "restore_model", lambda *args:(object(), object()))
    def evaluate(model, tokenizer, rows, input_texts, config):
        events.append("canonical" if input_texts[0].startswith("native0") else "zero-level2")
        assert list(rows["targetid"]) == reference["object_id"]
        return reference
    monkeypatch.setattr(physics.control.pilot, "evaluate", evaluate)
    def train(**kwargs):
        events.append("train-and-select")
        assert kwargs["initial_evaluation"] is reference
        assert kwargs["protocol"] == ContinuedSFTProtocol()
        return dict(best_metrics=metrics, zero_update_metrics=metrics)
    monkeypatch.setattr(physics, "continue_sft", train)
    monkeypatch.setattr(physics.band, "transfer", lambda *args:events.append("fixed-target-inference"))
    physics.run(SimpleNamespace(output_dir=tmp_path, resume=False, after_pid=None, after_start_ticks=None, gpu="MIG-selected", wait_for_gpu=True))
    assert events == ["canonical", "zero-level2", "train-and-select", "fixed-target-inference"]
    change = json.loads((tmp_path / "serializer_change.json").read_text())
    assert "level2_zero_update_metrics" in change and change["same_weights"]
    assert json.loads((tmp_path / "summary.json").read_text())["phase"] == "complete"
