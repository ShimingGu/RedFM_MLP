"""Matched warm starts, safe GPU queueing and the final Level-1 report path."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import qwen_filter_band_warmstart as band
from aion_magnitude.metrics import predict_photoz_from_logits, summarize_pdf_metrics
from aion_magnitude.qwen_continued_sft import ContinuedSFTProtocol
from aion_magnitude.utils import make_redshift_grid

DEVICE = "a9b1fb4a-5c28-5b03-9686-f831c6facf25"


def matched_manifest():
    return dict(source_run=str(band.control.SOURCE), model="Qwen3.5-4B-Base", model_source={"sha256":"base"},
        source_config={"n_z_bins":300}, adapter_config={"r":8}, protocol=asdict(ContinuedSFTProtocol()),
        initial_snapshot_sha256={"adapter/adapter_model.safetensors":"original", "photoz_head.pt":"trained"},
        cohort_sha256="all-values-labels-and-masks", train_rows=22444, val_rows=5683,
        query_manifest_sha256="audited-queries")


@pytest.mark.parametrize("key,value", [
    ("initial_snapshot_sha256", {"adapter/adapter_model.safetensors":"A-continued"}),
    ("cohort_sha256", "changed-labels"), ("source_config", {"n_z_bins":200}),
    ("query_manifest_sha256", "different-targets"), ("adapter_config", {"r":16}),
])
def test_rejects_unmatched_checkpoint_data_or_protocol(key, value):
    original = matched_manifest()
    band.check_match(original, original)
    changed = deepcopy(original); changed[key] = value
    with pytest.raises(ValueError, match=key): band.check_match(changed, original)


def test_gpu_wait_respects_lock_and_busy_memory_then_holds_reservation(tmp_path, monkeypatch):
    lock = tmp_path / f"{DEVICE}.lock"
    owner = lock.open("a+"); fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queries, waiting, sleeps = [], [], []
    def memory(device):
        queries.append(device)
        return dict(used=3 if len(queries) == 1 else .014, free=7 if len(queries) == 1 else 10.7, total=10.75)
    def on_wait(reason):
        waiting.append(reason)
        if len(waiting) == 1:
            # A lock alone must block B even if another job has low GPU memory.
            assert queries == []
            owner.close()
    monkeypatch.setattr(band, "memory_info", memory)
    monkeypatch.setattr(band.time, "sleep", lambda seconds:sleeps.append(seconds))
    with band.reserve_idle_gpu("MIG-" + DEVICE, wait=True, on_wait=on_wait, lock_root=tmp_path) as gpu:
        assert gpu["uuid"] == DEVICE
        with lock.open("a+") as competitor:
            with pytest.raises(BlockingIOError): fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lock.open("a+") as competitor:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert len(waiting) == 2 and sleeps == [30, 30]


def test_gpu_queue_never_retries_training_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(band, "memory_info", lambda _:dict(used=.014, free=10.7, total=10.75))
    with pytest.raises(ValueError, match="training failed"):
        with band.reserve_idle_gpu(DEVICE, wait=True, on_wait=lambda _:pytest.fail("Unexpected retry"), lock_root=tmp_path):
            raise ValueError("training failed")
    with (tmp_path / f"{DEVICE}.lock").open("a+") as competitor:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_no_wait_refuses_reserved_slice_without_querying_gpu(tmp_path, monkeypatch):
    owner = (tmp_path / f"{DEVICE}.lock").open("a+")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(band, "memory_info", lambda _:pytest.fail("Must not query a reserved device"))
    try:
        with pytest.raises(RuntimeError, match="reserved"):
            with band.reserve_idle_gpu(DEVICE, wait=False, on_wait=lambda _:None, lock_root=tmp_path): pass
    finally: owner.close()


def evaluation():
    edges, centers = make_redshift_grid(0, 4.5, 5)
    logits = torch.tensor([[3.,1.,0.,0.,0.], [0.,3.,1.,0.,0.]])
    result = dict(logits=logits, z_spec=torch.tensor([.4, 1.4]), object_id=["val-a", "val-b"],
                  redshift_edges=edges, redshift_centers=centers)
    result.update(predict_photoz_from_logits(logits, centers=centers))
    return result


def test_canonical_replay_slicing_preserves_grid_and_identity():
    original = evaluation()
    subset = band.subset_evaluation(original, 1)
    assert subset["object_id"] == ["val-a"]
    assert subset["z_spec"].shape == (1,)
    torch.testing.assert_close(subset["redshift_edges"], original["redshift_edges"], rtol=0, atol=0)


def test_completed_report_and_plots_include_matched_control(tmp_path):
    out = tmp_path / "level1"; out.mkdir()
    a_dir = tmp_path / "level0"; a_dir.mkdir()
    prediction = evaluation(); metrics = summarize_pdf_metrics(prediction)
    status = dict(phase="complete", global_update=10, max_updates=2806, best_update=0,
        best_metrics=metrics, zero_update_metrics=metrics, elapsed_seconds=5,
        validation_history=[dict(update=0, **metrics), dict(update=10, **metrics)],
        training_history=[dict(update=1, loss=3.0), dict(update=10, loss=2.9)])
    manifest = dict(control_dir=str(a_dir), token_audit=dict(n=28127, max=174, max_seq_length=512), reference_metrics=metrics)
    band.write_json(out / "manifest.json", manifest)
    band.write_json(out / "serializer_change.json", dict(original_level0_metrics=metrics,
        level1_zero_update_metrics=metrics, delta_level1_minus_level0={key:0 for key in band.control.METRICS}))
    band.write_json(out / "source_summary.json", status); band.write_json(a_dir / "source_summary.json", status)
    for filename in ("reference_validation.pt", "zero_update_validation.pt", "best_validation.pt"):
        torch.save(prediction, out / filename)
    targets = {kind:dict(metrics=metrics, query_ids=["val-a", "val-b"]) for kind in ("desi-ls", "clauds")}
    band.write_json(out / "transfer_summary.json", dict(targets=targets))
    band.write_json(a_dir / "transfer_summary.json", dict(targets=targets))
    import numpy as np
    for directory in (out, a_dir):
        for kind in targets:
            np.savez(directory / f"{kind}_selected_predictions.npz", z_spec=prediction["z_spec"].numpy(), z_p50=prediction["z_p50"].numpy())
    band.plots(out); band.render_report(out, status)
    report = (out / "report.md").read_text()
    assert "Original weights / Level 0" in report
    assert "Original weights / Level 1" in report
    assert "selected Level 0 A" in report and "selected Level 1 B" in report
    assert "CLAUDS uses ZPHOT" in report
    assert "B minus A" in report
    for filename in ("learning_curves.png", "serializer_predictions.png", "transfer_predictions.png"):
        assert (out / filename).stat().st_size > 1000
