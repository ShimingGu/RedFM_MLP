"""Evaluation replicas on separate MIG slices; optimizer and batches stay unchanged."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .metrics import predict_photoz_from_logits, redshift_cross_entropy_loss
from .qwen_continued_sft import atomic_torch_save, write_json


def batch_partitions(n, batch_size, workers):
    """Keep original complete batches together so padding and predictions match."""
    if n < 1 or batch_size < 1 or workers < 1:
        raise ValueError("Positive row, batch and worker counts are required")
    parts = [[] for _ in range(min(workers, (n+batch_size-1)//batch_size))]
    for batch, start in enumerate(range(0, n, batch_size)):
        parts[batch % len(parts)].extend(range(start, min(start+batch_size, n)))
    return parts


def assemble_evaluations(parts, indices, object_ids, labels, edges, centers):
    """Reassemble original query order before computing any global metrics."""
    n = len(object_ids)
    if len(parts) != len(indices) or sorted(i for rows in indices for i in rows) != list(range(n)):
        raise ValueError("Evaluation shards must cover every query exactly once")
    logits = torch.empty((n, len(centers)), dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32).cpu()
    edges, centers = edges.detach().cpu(), centers.detach().cpu()
    for result, rows in zip(parts, indices, strict=True):
        if result["object_id"] != [object_ids[i] for i in rows]:
            raise ValueError("Evaluation shard identity/order changed")
        torch.testing.assert_close(result["z_spec"], labels[rows], rtol=0, atol=0)
        for key, expected in [("redshift_edges", edges), ("redshift_centers", centers)]:
            torch.testing.assert_close(result[key], expected, rtol=0, atol=0)
        logits[rows] = result["logits"]
    evaluation = dict(logits=logits, z_spec=labels, object_id=list(object_ids), redshift_edges=edges,
        redshift_centers=centers, loss=redshift_cross_entropy_loss(logits, labels, edges=edges).item())
    evaluation.update(predict_photoz_from_logits(logits, centers=centers))
    return evaluation


class ProgressLoader:
    """Report completed batches without altering their contents, grouping or RNG."""

    def __init__(self, loader, publish, *, every_batches=64, cancelled=None):
        self.loader = loader
        self.publish = publish
        self.every_batches = every_batches
        self.cancelled = cancelled

    def __iter__(self):
        completed = 0
        self.publish(completed)
        for index, batch in enumerate(self.loader, 1):
            if self.cancelled is not None and self.cancelled():
                raise RuntimeError("Evaluation cancelled: parent exited or requested shutdown")
            count = len(batch["z_spec"])
            yield batch
            # The evaluator has finished this forward pass and copied its logits to CPU.
            completed += count
            if index == 1 or index % self.every_batches == 0:
                self.publish(completed)
        self.publish(completed)


class EvaluationPool:
    """File-based CPU tensor transport avoids CUDA peer access/NCCL between MIGs."""
    def __init__(self, output_dir, manifest, devices):
        self.out = Path(output_dir)
        self.folder = self.out / "evaluation_workers"
        self.folder.mkdir(exist_ok=True)
        self.manifest = manifest
        self.devices = list(devices)
        if len(set(self.devices)) != len(self.devices) or not self.devices:
            raise ValueError("Distinct evaluation devices required")
        self.children, self.workers, self.parameter_names = [], [], None
        self.sequence = 0
        self.session = str(time.time_ns())
        root = Path(__file__).resolve().parents[1]
        for index, device in enumerate(self.devices):
            worker = self.folder / f"session-{self.session}-worker-{index}"
            worker.mkdir()
            (worker / "initial_model").symlink_to((self.out / "initial_model").resolve(), target_is_directory=True)
            write_json(worker / "manifest.json", manifest)
            with (worker / "worker.log").open("a") as log:
                child = subprocess.Popen([sys.executable, str(root / "notebooks/qwen_filter_parallel.py"),
                    "--stage", "worker", "--output-dir", str(worker.resolve()), "--parent-pid", str(os.getpid())],
                    cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES=device), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            self.children.append(child); self.workers.append(worker)
        self.history = []

    def wait_ready(self):
        self._wait([folder / "ready.json" for folder in self.workers])

    def _wait(self, paths, timeout=3600, *, progress_paths=None):
        """Startup has a fixed deadline; evaluation times out only if a shard stalls.

        A progressing 225k-object evaluation may legitimately take over an hour.
        Only an increase in completed rows for this exact request extends its
        deadline. Other workers and stale/repeated heartbeats cannot mask a stall.
        """
        started = time.monotonic()
        last_activity = [started] * len(paths)
        completed = [-1] * len(paths)
        if progress_paths is not None and len(progress_paths) != len(paths):
            raise ValueError("Each result needs its own progress path")
        while not all(path.exists() for path in paths):
            for child, folder in zip(self.children, self.workers, strict=True):
                if child.poll() is not None:
                    raise RuntimeError(f"Evaluation worker exited ({child.returncode}); see {folder / 'worker.log'}")
            current = time.monotonic()
            for index, path in enumerate(paths):
                if path.exists():
                    continue
                if progress_paths is not None:
                    try:
                        progress = json.loads(progress_paths[index].read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        progress = {}
                    rows = progress.get("completed_rows", -1)
                    if (progress.get("request") == path.name and isinstance(rows, int)
                            and rows > completed[index]):
                        completed[index] = rows
                        last_activity[index] = current
                if current - last_activity[index] > timeout:
                    raise TimeoutError(
                        f"Evaluation worker timed out: no progress for {timeout:g}s; "
                        f"completed_rows={completed[index]}, awaiting {path}; "
                        f"see {path.parent / 'worker.log'}"
                    )
            time.sleep(.2)

    def evaluate_state(self, state, dataset, *, batch_size, max_length, edges, centers):
        self.wait_ready()
        self.sequence += 1
        prefix = f"request-{self.session}-{self.sequence}"
        started = time.monotonic()
        state_path = self.folder / f"{prefix}-state.pt"
        atomic_torch_save(state, state_path)
        indices = batch_partitions(len(dataset), batch_size, len(self.workers))
        result_paths = []
        for worker, rows in zip(self.workers, indices):
            result_path = worker / f"{prefix}-result.pt"
            atomic_torch_save(dict(state_path=str(state_path.resolve()),
                texts=[dataset.texts[i] for i in rows], labels=dataset.redshifts[rows],
                object_ids=[dataset.object_ids[i] for i in rows], batch_size=batch_size, max_length=max_length,
                redshift_edges=edges.detach().cpu(), redshift_centers=centers.detach().cpu(),
                result_path=str(result_path.resolve())), worker / "request.pt")
            result_paths.append(result_path)
        self._wait(result_paths, progress_paths=[p.parent / "progress.json" for p in result_paths])
        parts = [torch.load(path, map_location="cpu", weights_only=False) for path in result_paths]
        evaluation = assemble_evaluations(parts, indices, dataset.object_ids, dataset.redshifts, edges, centers)
        record = dict(sequence=self.sequence, n=len(dataset), batch_size=batch_size, workers=len(indices),
            seconds=time.monotonic()-started, same_original_batch_grouping=True,
            worker_peak_allocated_gib=[part["worker_peak_allocated_gib"] for part in parts])
        self.history.append(record)
        write_json(self.out / "evaluation_acceleration.json", dict(training_gpus=1, evaluation_gpus=self.devices,
            optimization_unchanged=True, effective_batch_size=16, transport="CPU checkpoint files; no MIG peer access",
            evaluations=self.history))
        for path in result_paths: path.unlink()
        state_path.unlink()
        return evaluation

    def evaluate(self, model, loader, *, device, redshift_edges, redshift_centers):
        model.eval()
        # Match DataLoader's one CPU RNG draw at iterator creation in the serial evaluator.
        iter(loader)
        parameters = dict(model.named_parameters())
        if self.parameter_names is None:
            self.parameter_names = [name for name, value in parameters.items() if value.requires_grad]
        if not self.parameter_names: raise ValueError("No adapter/head state to synchronize")
        state = {name:parameters[name].detach().cpu().clone() for name in self.parameter_names}
        return self.evaluate_state(state, loader.dataset, batch_size=loader.batch_size,
            max_length=self.manifest["source_config"]["max_length"], edges=redshift_edges, centers=redshift_centers)

    def close(self):
        for child, folder in zip(self.children, self.workers, strict=True):
            if child.poll() is None: (folder / "shutdown").touch()
        for child in self.children:
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()


def worker(output_dir, parent_pid):
    """One model replica, one exclusive slice; no optimizer or training calls."""
    from torch.utils.data import DataLoader
    import qwen_filter_warmstart as control
    from .qwen_posttraining import TextRedshiftDataset, _load_trainable_state, evaluate_text_photoz_model, make_text_collator

    out = Path(output_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    with control.pilot.reserve_one_gpu() as gpu:
        if gpu["memory_before_gib"]["used"] > .2: raise RuntimeError("Evaluation worker needs an idle slice")
        model, tokenizer = control.restore_model(out, manifest)
        model.requires_grad_(False); model.eval()
        write_json(out / "ready.json", dict(pid=os.getpid(), gpu=gpu, inference_only=True))
        while os.getppid() == parent_pid and not (out / "shutdown").exists():
            request = out / "request.pt"
            if not request.exists(): time.sleep(.2); continue
            payload = torch.load(request, map_location="cpu", weights_only=False)
            request.unlink()
            _load_trainable_state(model, torch.load(payload["state_path"], map_location="cpu", weights_only=True))
            dataset = TextRedshiftDataset(payload["texts"], payload["labels"], payload["object_ids"])
            loader = DataLoader(dataset, batch_size=payload["batch_size"], shuffle=False,
                collate_fn=make_text_collator(tokenizer, max_length=payload["max_length"]))
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            def publish(completed):
                write_json(out / "progress.json", dict(
                    request=Path(payload["result_path"]).name, completed_rows=completed,
                    total_rows=len(dataset), seconds=time.monotonic()-started, pid=os.getpid(),
                    updated_unix=time.time(), batch_size=payload["batch_size"],
                ))
            progress_loader = ProgressLoader(loader, publish,
                cancelled=lambda: os.getppid() != parent_pid or (out / "shutdown").exists())
            result = evaluate_text_photoz_model(model, progress_loader, device=torch.device("cuda"),
                redshift_edges=payload["redshift_edges"].to("cuda"), redshift_centers=payload["redshift_centers"])
            result["worker_peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
            atomic_torch_save(result, payload["result_path"])
