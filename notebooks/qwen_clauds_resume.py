#!/usr/bin/env python3
"""Resume an existing CLAUDS adapter run with a separate evaluation slice."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

# Imports can initialize the CUDA driver, which captures visibility once.
visibility_parser = argparse.ArgumentParser(add_help=False)
visibility_parser.add_argument("--gpu")
visibility, _ = visibility_parser.parse_known_args()
if visibility.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = visibility.gpu

import torch

from aion_magnitude import qwen_posttraining as training
from aion_magnitude.qwen_continued_sft import write_json
from aion_magnitude.qwen_parallel_evaluation import EvaluationPool, worker
import qwen_dora_rlvr_posttraining as clauds
import qwen_filter_warmstart as control


class AdapterEvaluationPool(EvaluationPool):
    """Reuse the existing file transport with the checkpoint's model config."""

    def __init__(self, output, config, use_dora, device):
        self.out = output
        self.folder = output / "evaluation_workers"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest = dict(source_config=dict(max_length=config.max_length))
        self.devices = [device]
        self.children, self.workers, self.parameter_names = [], [], None
        self.sequence, self.session, self.history = 0, str(time.time_ns()), []
        folder = self.folder / f"session-{self.session}-worker-0"
        folder.mkdir()
        write_json(folder / "manifest.json", dict(config=vars(config), use_dora=use_dora))
        with (folder / "worker.log").open("a") as log:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker",
                 "--output-dir", str(folder.resolve()), "--parent-pid", str(os.getpid())],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            )
        self.children.append(child)
        self.workers.append(folder)


def run_worker(args):
    def restore(output, manifest):
        model, tokenizer, _ = training.create_qlora_photoz_model(
            training.QwenPosttrainingConfig(**manifest["config"]).normalized(),
            use_dora=manifest["use_dora"],
        )
        return model, tokenizer

    control.restore_model = restore
    worker(args.output_dir, args.parent_pid)


def prepared_run(args):
    checkpoints = training._checkpoint_files(args.checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError("Resume requires an existing periodic checkpoint")
    checkpoint = checkpoints[-1]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if bool(saved.get("use_dora", False)) != (args.method == "dora"):
        raise ValueError("Checkpoint method differs from the requested method")
    config = training.QwenPosttrainingConfig(**saved["config"]).normalized()
    prepared = json.loads((args.output_dir / "prepared.json").read_text())
    if prepared["input_representation"] != "magnitudes_only":
        raise ValueError("This launcher expects the existing magnitudes-only CLAUDS runs")
    rows = prepared["rows"]
    total = sum(rows.values())
    source_args = clauds.comparison.build_parser().parse_args([
        "--stage", "prepare", "--catalogue", prepared["catalogue"],
        "--max-rows", str(prepared["max_rows"]), "--seed", str(config.seed),
        "--train-fraction", str(rows["train"] / total),
        "--val-fraction", str(rows["val"] / total),
        "--test-fraction", str(rows["test"] / total),
    ])
    train, val, test, bounds = clauds._text_datasets(source_args)
    if (len(train), len(val), len(test)) != tuple(saved["dataset_sizes"]):
        raise ValueError("Rebuilt cohort sizes differ from the saved run")
    if tuple(bounds) != (config.z_min, config.z_max):
        raise ValueError("Rebuilt redshift grid bounds differ from the saved run")
    if config.seed != prepared["sample_seed"] or prepared["sample_mode"] != "random":
        raise ValueError("Rebuilt sampling protocol differs from the saved run")
    if not saved["optimizer"]["state"] or not saved["scheduler"]:
        raise ValueError("Checkpoint lacks optimizer or scheduler state")
    if not all(torch.isfinite(value).all() for value in saved["trainable_state"].values()):
        raise ValueError("Checkpoint has non-finite adapter/head weights")
    result = args.output_dir / args.method / "result.pt"
    if result.exists():
        raise FileExistsError(f"Completed result already exists: {result}")
    print(json.dumps(dict(method=args.method, checkpoint=str(checkpoint),
                          resume_update=saved["global_update"], dataset_sizes=saved["dataset_sizes"],
                          training_gpu=args.gpu, evaluation_gpu=args.eval_gpu)), flush=True)
    return config, saved, (train, val, test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qlora", "dora"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--eval-gpu")
    parser.add_argument("--check-only", action="store_true", help="Validate saved state and cohort on CPU; do not launch training")
    parser.add_argument("--probe", action="store_true", help="Compare serial and remote inference on 32 validation objects; do not train")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return
    if not all((args.method, args.checkpoint_dir, args.gpu, args.eval_gpu)):
        parser.error("--method, --checkpoint-dir, --gpu and --eval-gpu are required")
    if args.gpu == args.eval_gpu:
        parser.error("Select distinct training and evaluation slices")
    config, saved, datasets = prepared_run(args)
    if args.check_only:
        return
    output = args.output_dir / args.method
    with control.pilot.reserve_one_gpu():
        pool = AdapterEvaluationPool(output, config, args.method == "dora", args.eval_gpu)
        try:
            pool.wait_ready()
            if args.probe:
                from torch.utils.data import DataLoader
                from aion_magnitude.utils import make_redshift_grid
                val = datasets[1]
                subset = training.TextRedshiftDataset(val.texts[:32], val.redshifts[:32], val.object_ids[:32])
                model, tokenizer, device = training.create_qlora_photoz_model(config, use_dora=args.method == "dora")
                loader = DataLoader(subset, batch_size=config.eval_batch_size, shuffle=False,
                                    collate_fn=training.make_text_collator(tokenizer, max_length=config.max_length))
                training._load_trainable_state(model, saved["trainable_state"])
                edges, centers = make_redshift_grid(config.z_min, config.z_max, config.n_z_bins)
                kwargs = dict(device=device, redshift_edges=edges.to(device), redshift_centers=centers)
                reference = training.evaluate_text_photoz_model(model, loader, **kwargs)
                actual = pool.evaluate(model, loader, **kwargs)
                replay = control.verify_replay(actual, reference)
                write_json(output / "evaluation_replay.json", replay)
                print(json.dumps(replay), flush=True)
                return
            original = training.evaluate_text_photoz_model
            training.evaluate_text_photoz_model = pool.evaluate
            try:
                write_json(output / "resume_execution.json", dict(
                    training_gpu=args.gpu, evaluation_gpu=args.eval_gpu,
                    resumed_update=saved["global_update"], optimizer_gpus=1,
                    evaluation_batch_grouping_preserved=True, configuration_from_checkpoint=True,
                ))
                result = training.train_qlora_photoz(
                    train_dataset=datasets[0], val_dataset=datasets[1], test_dataset=datasets[2],
                    output_dir=output, config=config, checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=100, resume=True, use_dora=args.method == "dora",
                )
                clauds._write_summary(
                    argparse.Namespace(output_dir=args.output_dir, qwen_model=config.model_path),
                    result, method=args.method,
                )
                print(f"saved {output / 'result.pt'}", flush=True)
            finally:
                training.evaluate_text_photoz_model = original
        finally:
            pool.close()


if __name__ == "__main__":
    main()
