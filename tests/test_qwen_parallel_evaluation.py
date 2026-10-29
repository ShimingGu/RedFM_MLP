"""Parallel evaluation preserves serial batches, identities, metrics and RNG draws."""
from types import SimpleNamespace
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from aion_magnitude.metrics import predict_photoz_from_logits, summarize_pdf_metrics
from aion_magnitude.qwen_parallel_evaluation import EvaluationPool, ProgressLoader, assemble_evaluations, batch_partitions
from aion_magnitude.qwen_posttraining import TextRedshiftDataset
from aion_magnitude.utils import make_redshift_grid


@pytest.mark.parametrize("n,batch_size,workers", [(5683,8,2), (32,8,2), (256,8,2), (1,8,2), (17,8,3)])
def test_partitions_keep_original_batch_grouping(n, batch_size, workers):
    parts = batch_partitions(n, batch_size, workers)
    assert sorted(i for part in parts for i in part) == list(range(n))
    for part in parts:
        for start in range(0, len(part), batch_size):
            group = part[start:start+batch_size]
            assert group[0] % batch_size == 0
            assert group == list(range(group[0], min(group[0]+batch_size, n)))


def test_reassembly_matches_serial_metrics_exactly_and_rejects_wrong_ids():
    edges, centers = make_redshift_grid(0, 4.5, 7)
    ids = [f"val-{i}" for i in range(19)]
    labels = torch.linspace(.1, 4, 19)
    logits = torch.arange(19*7, dtype=torch.float32).reshape(19,7) / 19
    parts = batch_partitions(19, 8, 2)
    evaluations = [dict(logits=logits[rows], z_spec=labels[rows], object_id=[ids[i] for i in rows],
        redshift_edges=edges, redshift_centers=centers) for rows in parts]
    combined = assemble_evaluations(evaluations, parts, ids, labels, edges, centers)
    serial = dict(logits=logits, z_spec=labels, object_id=ids, redshift_edges=edges, redshift_centers=centers)
    serial.update(predict_photoz_from_logits(logits, centers=centers))
    torch.testing.assert_close(combined["logits"], logits, rtol=0, atol=0)
    assert summarize_pdf_metrics(combined) == summarize_pdf_metrics(serial)
    evaluations[0]["object_id"][0] = "wrong"
    with pytest.raises(ValueError, match="identity/order"):
        assemble_evaluations(evaluations, parts, ids, labels, edges, centers)


def test_reassembly_rejects_duplicate_rows():
    edges, centers = make_redshift_grid(0, 1, 2)
    with pytest.raises(ValueError, match="exactly once"):
        assemble_evaluations([{},{}], [[0],[0]], ["one","two"], torch.zeros(2), edges, centers)


def test_evaluation_keeps_main_cpu_rng_draw_and_syncs_frozen_selected_weights():
    pool = EvaluationPool.__new__(EvaluationPool)
    pool.manifest = dict(source_config=dict(max_length=512))
    model = nn.Linear(2, 3)
    pool.parameter_names = [name for name, _ in model.named_parameters()]
    model.requires_grad_(False)
    dataset = TextRedshiftDataset(["a","b"], [.1,.2], ["one","two"])
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda x:x)
    expected_state = {name:value.detach().clone() for name,value in model.named_parameters()}
    def evaluate_state(state, actual_dataset, **kwargs):
        assert actual_dataset is dataset and kwargs["batch_size"] == 2
        for name in state:
            torch.testing.assert_close(state[name], expected_state[name], rtol=0, atol=0)
        return "result"
    pool.evaluate_state = evaluate_state
    torch.manual_seed(1023); iter(loader); expected_rng = torch.get_rng_state()
    torch.manual_seed(1023)
    edges, centers = make_redshift_grid(0,1,3)
    assert pool.evaluate(model, loader, device=torch.device("cpu"), redshift_edges=edges, redshift_centers=centers) == "result"
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert not model.training


def test_progress_loader_preserves_batches_and_rng_and_counts_completed_rows():
    dataset = TextRedshiftDataset([str(i) for i in range(19)], torch.arange(19), list(range(19)))
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
        collate_fn=lambda rows: dict(z_spec=[r['z_spec'] for r in rows], rows=rows))
    torch.manual_seed(42)
    expected = [batch['z_spec'] for batch in loader]
    expected_rng = torch.get_rng_state()
    torch.manual_seed(42)
    progress = []
    actual = []
    for batch in ProgressLoader(loader, progress.append, every_batches=2):
        actual.append(batch.pop('z_spec'))  # The real evaluator also mutates the batch.
    assert actual == expected
    assert progress == [0, 8, 16, 19]
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['advancing', 'stalled', 'stale_request', 'one_shard_stalled'])
def test_evaluation_deadline_tracks_each_requests_actual_progress(tmp_path, monkeypatch, mode):
    import aion_magnitude.qwen_parallel_evaluation as evaluation
    pool = EvaluationPool.__new__(EvaluationPool)
    count = 2 if mode == 'one_shard_stalled' else 1
    pool.workers = [tmp_path / str(i) for i in range(count)]
    for path in pool.workers: path.mkdir()
    pool.children = [SimpleNamespace(poll=lambda:None) for _ in pool.workers]
    results = [path / 'current-result.pt' for path in pool.workers]
    progress = [path / 'progress.json' for path in pool.workers]
    clock = [0.]
    def tick(seconds):
        clock[0] += seconds
        for i, path in enumerate(progress):
            rows = 0 if mode == 'stalled' or (mode == 'one_shard_stalled' and i == 0) else round(clock[0]*10)
            path.write_text(json.dumps(dict(request='old-result.pt' if mode == 'stale_request' else results[i].name, completed_rows=rows)))
        if clock[0] >= 3 and mode == 'advancing': results[0].touch()
    monkeypatch.setattr(evaluation.time, 'monotonic', lambda:clock[0])
    monkeypatch.setattr(evaluation.time, 'sleep', tick)
    if mode == 'advancing':
        pool._wait(results, timeout=1, progress_paths=progress)
        assert clock[0] >= 3  # A fixed whole-request timeout would have failed.
    else:
        with pytest.raises(TimeoutError, match='no progress'):
            pool._wait(results, timeout=1, progress_paths=progress)
        assert clock[0] < 2


def test_evaluation_reports_dead_worker_before_idle_timeout(tmp_path):
    pool = EvaluationPool.__new__(EvaluationPool)
    pool.workers = [tmp_path]
    pool.children = [SimpleNamespace(poll=lambda:1, returncode=1)]
    with pytest.raises(RuntimeError, match='exited'):
        pool._wait([tmp_path/'missing-result.pt'])


@pytest.mark.parametrize("failed", [False, True])
def test_pipeline_runs_only_one_arm_at_a_time_and_stops_on_failure(tmp_path, monkeypatch, failed):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    import qwen_filter_parallel as runner
    arms = {}
    for arm, level in zip("ABC", range(3)):
        out = tmp_path / f"level{level}"; out.mkdir()
        arms[arm] = (SimpleNamespace(load_prepared=lambda _:None), out)
    (arms["A"][1] / "latest.pt").touch()
    monkeypatch.setattr(runner, "ARMS", arms)
    monkeypatch.setattr(runner, "BASE", tmp_path)
    monkeypatch.setattr(runner.signal, "signal", lambda *args:None)
    events = []
    class Child:
        def __init__(self, command, **kwargs):
            self.arm = command[command.index("--arm")+1]
            assert ("--resume" in command) == (self.arm == "A")
            self.pid = 100 + ord(self.arm)
            events.append("start-" + self.arm)
        def wait(self):
            events.append("finish-" + self.arm)
            if not failed:
                (arms[self.arm][1] / "summary.json").write_text('{}')
                return 0
            return 1
    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    args = SimpleNamespace(gpu="training", eval_gpus="extra1,extra2")
    if failed:
        with pytest.raises(RuntimeError, match="refusing to start"): runner.pipeline(args)
        assert events == ["start-A", "finish-A"]
        assert json.loads((tmp_path / "pipeline.json").read_text())["phase"] == "failed"
    else:
        runner.pipeline(args)
        assert events == ["start-A", "finish-A", "start-B", "finish-B", "start-C", "finish-C"]
        assert json.loads((tmp_path / "pipeline.json").read_text())["phase"] == "complete"
