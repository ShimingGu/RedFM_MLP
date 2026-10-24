#!/usr/bin/env python3
"""Sequential A -> B -> C with one training slice and two evaluation replicas."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from aion_magnitude.qwen_continued_sft import write_json, verify_replay
from aion_magnitude.qwen_parallel_evaluation import EvaluationPool, worker
from aion_magnitude.qwen_posttraining import TextRedshiftDataset, make_text_collator
from aion_magnitude.utils import make_redshift_grid
import qwen_filter_warmstart as control
import qwen_filter_band_warmstart as band
import qwen_filter_physics_warmstart as physics

BASE = ROOT / "filter-aware-runs/1023/hsc-warmstart"
ARMS = {"A":(control, BASE / "level0"), "B":(band, BASE / "level1"), "C":(physics, BASE / "level2")}


def selections(args):
    extras = args.eval_gpus.split(",")
    if len(extras) != 2 or len(set([args.gpu]+extras)) != 3:
        raise ValueError("Select one training GPU and two distinct evaluation GPUs")
    return extras


def resource_note(out, args):
    return ("\n## Additional evaluation GPUs\n\n"
        f"Training uses `{args.gpu}`. Validation and selected-model transfer inference are split over "
        f"`{args.eval_gpus}` using two independent frozen replicas, with synchronized adapter/head weights "
        "and the original batch-of-eight grouping and object order. Training, effective batch 16, optimizer, "
        "RNG, source selection and independent original-checkpoint initialization are unchanged. "
        "A → B → C executes sequentially. This accelerates evaluation, not optimizer updates. "
        "`evaluation_acceleration.json` records per-evaluation timings and both workers' peak allocated memory; "
        "the original training memory figure covers the training GPU only. `execution_backend.json` records code and devices.\n")


def run_arm(args):
    extras = selections(args)
    module, out = ARMS[args.arm]
    manifest, _, _ = module.load_prepared(out)
    if args.output_dir is not None and args.output_dir.resolve() != out.resolve():
        raise ValueError("Arm/output directory mismatch")
    original_restore = control.restore_model
    original_continue = module.continue_sft
    pool = None
    def restore(output, contract):
        nonlocal pool
        model, tokenizer = original_restore(output, contract)
        pool = EvaluationPool(output, contract, extras)
        pool.parameter_names = [name for name, value in model.named_parameters() if value.requires_grad]
        pool.wait_ready()
        return model, tokenizer
    def evaluate(model, loader, **kwargs):
        if pool is None: raise RuntimeError("Evaluation replicas were not initialized")
        return pool.evaluate(model, loader, **kwargs)
    def continued(**kwargs):
        return original_continue(**dict(kwargs, evaluator=evaluate))
    def evaluate_texts(model, tokenizer, data, texts, config):
        dataset = TextRedshiftDataset(texts, data["z"], [str(x) for x in data["targetid"]])
        loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=make_text_collator(tokenizer, max_length=config["max_length"]))
        edges, centers = make_redshift_grid(config["z_min"], config["z_max"], config["n_z_bins"])
        return evaluate(model, loader, device=torch.device("cuda"), redshift_edges=edges, redshift_centers=centers)
    for target in (control, band, physics):
        original_render = target.render_report
        def report(output, status, original=original_render):
            original(output, status)
            with (output / "report.md").open("a") as handle: handle.write(resource_note(output, args))
        target.render_report = report
    control.restore_model = restore
    module.continue_sft = continued
    control.evaluate_text_photoz_model = evaluate
    control.pilot.evaluate = evaluate_texts
    write_json(out / "execution_backend.json", dict(assigned_utc=control.now(), arm=args.arm,
        training_gpu=args.gpu, evaluation_gpus=extras, optimizer_updates_parallel=False,
        evaluation_parallel=True, original_batch_grouping_preserved=True, protocol_unchanged=True,
        resumed=args.resume, code_sha256={str(p.relative_to(ROOT)):control.pilot.sha256(p) for p in
            [Path(__file__), ROOT / "aion_magnitude/qwen_parallel_evaluation.py"]}))
    run_args = SimpleNamespace(output_dir=out, gpu=args.gpu, wait_for_gpu=True, resume=args.resume,
                              after_pid=None, after_start_ticks=None)
    try: module.run(run_args)
    except Exception as error:
        write_json(out / "failure.json", dict(utc=control.now(), error=repr(error), execution_backend="parallel_evaluation"))
        status = json.loads((out / "status.json").read_text()); status["phase"] = "failed"
        write_json(out / "status.json", status); module.render_report(out, status)
        raise
    finally:
        if pool is not None: pool.close()


def probe(args):
    extras = selections(args)
    manifest, data, texts = control.load_prepared(ARMS["A"][1])
    out = BASE / "parallel-validation-probe"
    out.mkdir(exist_ok=True)
    if not (out / "initial_model").exists(): (out / "initial_model").symlink_to(ARMS["A"][1] / "initial_model", target_is_directory=True)
    rows = np.flatnonzero(data["split"] == "val")[:32]
    dataset = TextRedshiftDataset([texts[i] for i in rows], data["z"][rows], [str(x) for x in data["targetid"][rows]])
    state = torch.load(ARMS["A"][1] / "initial_trainable_state.pt", map_location="cpu", weights_only=True)
    reference = band.subset_evaluation(torch.load(ARMS["A"][1] / "reference_validation.pt", map_location="cpu", weights_only=False), 32)
    pool = EvaluationPool(out, manifest, extras)
    try:
        result = pool.evaluate_state(state, dataset, batch_size=8, max_length=manifest["source_config"]["max_length"],
            edges=reference["redshift_edges"], centers=reference["redshift_centers"])
        replay = verify_replay(result, reference)
        write_json(out / "replay.json", replay)
        print(json.dumps(replay, indent=2), flush=True)
    finally: pool.close()


def pipeline(args):
    selections(args)
    with (BASE / "pipeline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = None
        def stop(signum, frame):
            if active is not None and active.poll() is None: os.killpg(active.pid, signum)
            raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        for arm, (module, out) in ARMS.items():
            if (out / "summary.json").exists(): continue
            module.load_prepared(out)
            command = [sys.executable, str(Path(__file__).resolve()), "--stage", "run", "--arm", arm,
                       "--gpu", args.gpu, "--eval-gpus", args.eval_gpus]
            if (out / "latest.pt").exists(): command.append("--resume")
            previous = out / "job.json"
            if previous.exists(): shutil.copy2(previous, out / f"job-before-parallel-{time.time_ns()}.json")
            with (out / "run.log").open("a") as log:
                active = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(previous, dict(pid=active.pid, launched_utc=control.now(), gpu=args.gpu, evaluation_gpus=args.eval_gpus.split(","),
                command=command, supervisor_pid=os.getpid(), sequence="A -> B -> C"))
            write_json(BASE / "pipeline.json", dict(phase="running", active_arm=arm, active_pid=active.pid,
                supervisor_pid=os.getpid(), training_gpu=args.gpu, evaluation_gpus=args.eval_gpus.split(","), sequence="A -> B -> C"))
            result = active.wait()
            if result != 0 or not (out / "summary.json").exists():
                write_json(BASE / "pipeline.json", dict(phase="failed", active_arm=arm, returncode=result,
                    supervisor_pid=os.getpid(), message="Later arms were not started; inspect the active arm run.log"))
                raise RuntimeError(f"Experiment {arm} failed; refusing to start the next arm")
        write_json(BASE / "pipeline.json", dict(phase="complete", sequence="A -> B -> C", finished_utc=control.now()))


def launch_pipeline(args):
    selections(args)
    job_file = BASE / "pipeline-job.json"
    if job_file.exists():
        old = json.loads(job_file.read_text())
        if physics.process_identity(old["pid"]) is not None: raise RuntimeError("Pipeline is already running")
    # The old per-arm launchers must be stopped before relinquishing A's GPU.
    for arm, (_, out) in ARMS.items():
        job_file_arm = out / "job.json"
        if job_file_arm.exists():
            old = json.loads(job_file_arm.read_text())
            if physics.process_identity(old["pid"]) is not None: raise RuntimeError(f"Original {arm} launcher {old['pid']} still exists")
    for arm in ("B", "C"):
        module, out = ARMS[arm]
        status = dict(phase="queued", gpu=args.gpu, queue_reason="Managed sequential pipeline: waiting for all earlier arms to complete successfully",
                      updated_utc=control.now(), test_evaluated=False)
        write_json(out / "status.json", status); module.render_report(out, status)
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", "pipeline", "--gpu", args.gpu, "--eval-gpus", args.eval_gpus]
    with (BASE / "pipeline.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(job_file, dict(pid=child.pid, launched_utc=control.now(), command=command,
        training_gpu=args.gpu, evaluation_gpus=args.eval_gpus.split(","), sequence="A -> B -> C"))
    print(json.dumps(dict(pid=child.pid, log=str(BASE / "pipeline.log"))), flush=True)


def migrate(args):
    """Switch only after a valid new checkpoint and two idle, unlocked slices."""
    extras = selections(args)
    sys.path.insert(0, str(ROOT / "scripts"))
    from qwen_survey_gpu_guard import memory_info
    expected = {arm:json.loads((out / "job.json").read_text()) for arm, (_, out) in ARMS.items()}
    identities = {arm:physics.process_identity(job["pid"]) for arm,job in expected.items()}
    if any(identity is None for identity in identities.values()):
        raise RuntimeError("An original launcher already exited; inspect the queue before migration")
    checkpoint = ARMS["A"][1] / "latest.pt"
    initial = torch.load(checkpoint, map_location="cpu", weights_only=False)
    minimum = args.min_update or initial["state"]["global_update"]+50
    original_protocol = json.loads((ARMS["A"][1] / "manifest.json").read_text())["protocol"]
    with (BASE / "migration.lock").open("a+") as migration_lock:
        fcntl.flock(migration_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if identities["A"] is None or physics.process_identity(expected["A"]["pid"]) != identities["A"]:
                raise RuntimeError("Original A process changed before migration; existing jobs were left alone")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["protocol"] != original_protocol: raise ValueError("A restart protocol changed")
            update = saved["state"]["global_update"]
            status = dict(phase="waiting_for_checkpoint", checkpoint_update=update, minimum_update=minimum,
                training_gpu=args.gpu, evaluation_gpus=extras, sequence="A -> B -> C", updated_utc=control.now())
            if update < minimum:
                write_json(BASE / "resource-migration.json", status); time.sleep(15); continue
            with ExitStack() as reservations:
                available = True
                for device in sorted(extras):
                    handle = reservations.enter_context((band.LOCK_ROOT / f"{device.removeprefix('MIG-')}.lock").open("a+"))
                    try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError: available = False; break
                    info = memory_info(device.removeprefix("MIG-"))
                    if info["used"] > .2 or info["free"] < 8: available = False; break
                if not available:
                    status.update(phase="waiting_for_idle_evaluation_slices")
                    write_json(BASE / "resource-migration.json", status)
                else:
                    for arm in ("B", "C"):
                        out = ARMS[arm][1]
                        if json.loads((out / "status.json").read_text())["phase"] != "queued":
                            raise RuntimeError(f"{arm} is no longer queued; refusing to interrupt it")
                        if physics.process_identity(expected[arm]["pid"]) != identities[arm]:
                            raise RuntimeError(f"{arm} process changed; refusing migration")
                    # Freeze A before snapshotting to avoid a checkpoint/data race.
                    os.kill(expected["A"]["pid"], signal.SIGSTOP)
                    try:
                        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        update = saved["state"]["global_update"]
                        archive = BASE / f"migration-checkpoint-A-update{update}.pt"
                        shutil.copy2(checkpoint, archive)
                        for arm, (_, out) in ARMS.items():
                            shutil.copy2(out / "job.json", out / f"job-before-resource-migration-{time.time_ns()}.json")
                        status.update(phase="switching", checkpoint_update=update, checkpoint_sha256=control.pilot.sha256(archive),
                                      archived_checkpoint=str(archive), old_processes=identities)
                        write_json(BASE / "resource-migration.json", status)
                    except BaseException:
                        os.kill(expected["A"]["pid"], signal.SIGCONT)
                        raise
                    for arm in ("C", "B", "A"): os.kill(expected[arm]["pid"], signal.SIGTERM)
                    os.kill(expected["A"]["pid"], signal.SIGCONT)
                    deadline = time.monotonic()+30
                    while any(physics.process_identity(expected[arm]["pid"]) == identities[arm] for arm in ARMS):
                        if time.monotonic() > deadline: raise TimeoutError("Old launchers have not exited; new pipeline was not started")
                        time.sleep(.2)
                    while memory_info(args.gpu.removeprefix("MIG-"))["used"] > .2:
                        if time.monotonic() > deadline: raise TimeoutError("Training slice has not drained after stopping A")
                        time.sleep(.2)
                    # Release extra-slice locks before their new worker processes start.
                    reservations.close()
                    launch_pipeline(args)
                    status.update(phase="complete", finished_utc=control.now(), resumed_A_update=update,
                                  source_protocol_unchanged=True, original_queued_launchers_replaced=True)
                    write_json(BASE / "resource-migration.json", status)
                    return
            time.sleep(15)


def launch_migration(args):
    selections(args)
    job_file = BASE / "migration-job.json"
    if job_file.exists() and physics.process_identity(json.loads(job_file.read_text())["pid"]) is not None:
        raise RuntimeError("Resource migration is already running")
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", "migrate", "--gpu", args.gpu, "--eval-gpus", args.eval_gpus]
    if args.min_update: command += ["--min-update", str(args.min_update)]
    with (BASE / "migration.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(job_file, dict(pid=child.pid, launched_utc=control.now(), command=command))
    print(json.dumps(dict(pid=child.pid, log=str(BASE / "migration.log"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("worker", "probe", "run", "pipeline", "launch", "migrate", "launch-migration"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--gpu", default="MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25")
    parser.add_argument("--eval-gpus", default="MIG-bef01423-2506-5edb-a289-80c60b1f0297,MIG-412ff209-a29c-5bc8-9aa2-fd1a41474aec")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-update", type=int)
    args = parser.parse_args()
    if args.stage == "worker": worker(args.output_dir, args.parent_pid)
    else: {"probe":probe, "run":run_arm, "pipeline":pipeline, "launch":launch_pipeline,
           "migrate":migrate, "launch-migration":launch_migration}[args.stage](args)


if __name__ == "__main__": main()
