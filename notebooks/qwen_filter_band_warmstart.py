#!/usr/bin/env python3
"""Experiment B: matched Level-1 continued SFT from the original HSC checkpoint."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from transformers import AutoTokenizer

from aion_magnitude.FM_Qwen import resolve_qwen_model_path
from aion_magnitude.filter_photometry import (
    DEFAULT_REGISTRY, SERIALIZER_VERSION, audit_tokens, load_registry, serialize_rows,
)
from aion_magnitude.metrics import summarize_pdf_metrics
from aion_magnitude.qwen_continued_sft import (
    ContinuedSFTProtocol, atomic_torch_save, continue_sft, verify_replay, write_json,
)
from aion_magnitude.qwen_posttraining import TextRedshiftDataset
from aion_magnitude.utils import set_random_seed
import qwen_filter_warmstart as control
from qwen_survey_gpu_guard import memory_info

DEFAULT_OUTPUT = ROOT / "filter-aware-runs/1023/hsc-warmstart/level1"
DEFAULT_CONTROL = control.DEFAULT_OUTPUT
LEGACY_GROUPS = ROOT / "filter-aware-runs/1023/hsc-pilot/legacy-groups.json"
LOCK_ROOT = Path("/scratch/.tmp-gsm/aion_output/gpu_locks") / str(os.getuid())


def check_match(manifest, other):
    """Match scientific inputs, including the original (never continued) weights."""
    for key in ("source_run", "model", "model_source", "source_config", "adapter_config",
                "protocol", "initial_snapshot_sha256", "cohort_sha256", "train_rows", "val_rows",
                "query_manifest_sha256"):
        if manifest[key] != other[key]:
            raise ValueError(f"Experiment A/B mismatch: {key}")
    if manifest["protocol"] != asdict(ContinuedSFTProtocol()):
        raise ValueError("The matched continued-SFT protocol changed")
    if Path(manifest["source_run"]).resolve() != control.SOURCE.resolve():
        raise ValueError("Experiment B must start from the original archived HSC run")


def write_inputs(path, ids, texts):
    with path.open("w") as handle:
        for object_id, text in zip(ids, texts, strict=True):
            handle.write(json.dumps(dict(object_id=str(object_id), text=text)) + "\n")


def read_inputs(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [row["object_id"] for row in rows], [row["text"] for row in rows]


def token_example(tokenizer, text):
    ids = tokenizer(text, truncation=False, padding=False)["input_ids"]
    return dict(text=text, input_ids=ids, tokens=tokenizer.convert_ids_to_tokens(ids), n_tokens=len(ids))


def prepare(args):
    out = args.output_dir
    if out.exists():
        raise FileExistsError("Never overwrite a prepared experiment")
    a_manifest, data, native = control.load_prepared(args.control_dir)
    if a_manifest["experiment"] != "A" or a_manifest["level"] != 0:
        raise ValueError("Expected the matched Level-0 control")
    check_match(a_manifest, a_manifest)
    fast = json.loads((control.FAST / "summary.json").read_text())
    if control.pilot.sha256(control.FAST / "summary.json") != a_manifest["query_manifest_sha256"]:
        raise ValueError("Audited target manifest changed")
    # A's initial_model is only used as a byte-checked copy of the original archive.
    for name, digest in a_manifest["initial_snapshot_sha256"].items():
        original = control.SOURCE / "qlora" / name
        if control.pilot.sha256(original) != digest:
            raise ValueError(f"Initial snapshot is not the archived trained checkpoint: {name}")
        if control.pilot.sha256(args.control_dir / "initial_model" / name) != digest:
            raise ValueError("Experiment A's immutable initial snapshot changed")
    if data["features"].shape != (28127, 11):
        raise ValueError("Expected all original HSC values/errors/EBV")
    metadata = json.loads((control.SOURCE / "prepared.json").read_text())
    if serialize_rows(data, metadata, catalogue="galaxiesml-spec", level=0) != native:
        raise ValueError("Native source inputs differ from A")
    registry = load_registry(verify_curves=True)
    texts = serialize_rows(data, metadata, catalogue="galaxiesml-spec", level=1, registry=registry)
    if any("pivot_nm=" in text or "width_nm=" in text for text in texts):
        raise ValueError("Level 1 must contain no wavelength/width metadata")
    tokenizer = AutoTokenizer.from_pretrained(resolve_qwen_model_path(a_manifest["source_config"]["model_path"]), local_files_only=True)
    audit = audit_tokens(tokenizer, texts, a_manifest["source_config"]["max_length"])
    actual_base = control.prepared_tools.survey.model_source(argparse.Namespace(qwen_model=a_manifest["source_config"]["model_path"]))
    if actual_base != a_manifest["model_source"]:
        raise ValueError("Base model fingerprint differs from A")
    groups = json.loads(LEGACY_GROUPS.read_text())
    if groups["query_manifest_sha256"] != a_manifest["query_manifest_sha256"]:
        raise ValueError("Legacy RELEASE lookup belongs to different validation queries")
    out.mkdir(parents=True)
    for name in ("source_cohort.npz", "reference_validation.pt"):
        shutil.copy2(args.control_dir / name, out / name)
    shutil.copytree(args.control_dir / "initial_model", out / "initial_model")
    write_inputs(out / "source_inputs.jsonl", data["targetid"], texts)
    val_indices = np.flatnonzero(data["split"] == "val")
    # First 32 rows preserve the original eval-batch-8 grouping for strict replay.
    canonical = val_indices[:32]
    write_inputs(out / "canonical_replay_inputs.jsonl", data["targetid"][canonical], [native[i] for i in canonical])
    target_manifest, examples = {}, {"source_train": token_example(tokenizer, texts[np.flatnonzero(data["split"] == "train")[0]]),
                                      "source_val": token_example(tokenizer, texts[val_indices[0]])}
    for kind in ("desi-ls", "clauds"):
        with np.load(control.FAST / f"{kind}_predictions.npz", allow_pickle=False) as archive:
            target = dict(targetid=archive["targetid"], features=archive["features"], z=archive["z_spec"])
        original = json.loads((control.FAST / f"{kind}_inputs.json").read_text())
        if original["object_id"] != [str(x) for x in target["targetid"]]:
            raise ValueError("Target IDs differ from audited queries")
        if serialize_rows(target, fast["targets"][kind], catalogue=kind, level=0) != original["texts"]:
            raise ValueError("Target native values differ from A")
        if kind == "desi-ls":
            target["group"] = np.asarray([groups["groups"][str(x)] for x in target["targetid"]])
        target_texts = serialize_rows(target, fast["targets"][kind], catalogue=kind, level=1, registry=registry)
        target_audit = audit_tokens(tokenizer, target_texts, a_manifest["source_config"]["max_length"])
        np.savez_compressed(out / f"{kind}_cohort.npz", **target)
        write_inputs(out / f"{kind}_inputs.jsonl", target["targetid"], target_texts)
        target_manifest[kind] = dict(n=len(target_texts), label=fast["targets"][kind]["label"],
            data_sha256=control.prepared_tools.sha_arrays(target), inputs_sha256=control.pilot.sha256(out / f"{kind}_inputs.jsonl"),
            original_predictions_sha256=control.pilot.sha256(control.FAST / f"{kind}_predictions.npz"),
            original_inputs_sha256=control.pilot.sha256(control.FAST / f"{kind}_inputs.json"),
            token_audit=target_audit, query_ids=original["object_id"], overlap_audit=fast["targets"][kind]["overlap_audit"])
        examples[kind] = token_example(tokenizer, target_texts[0])
    manifest = dict(a_manifest)
    manifest.update(created_utc=control.now(), experiment="B", level=1, control_dir=str(args.control_dir.resolve()),
        control_manifest_sha256=control.pilot.sha256(args.control_dir / "manifest.json"),
        serializer=f"{SERIALIZER_VERSION} / Level 1 / native values, units, five-decimal precision and NA",
        source_inputs_sha256=control.pilot.sha256(out / "source_inputs.jsonl"), token_audit=audit,
        canonical_inputs_sha256=control.pilot.sha256(out / "canonical_replay_inputs.jsonl"),
        reference_validation_sha256=control.pilot.sha256(out / "reference_validation.pt"),
        registry_sha256=control.pilot.sha256(DEFAULT_REGISTRY), legacy_groups_sha256=control.pilot.sha256(LEGACY_GROUPS),
        targets=target_manifest,
        code_sha256={str(p.relative_to(ROOT)):control.pilot.sha256(p) for p in [Path(__file__),
            ROOT / "notebooks/qwen_filter_warmstart.py", ROOT / "aion_magnitude/qwen_continued_sft.py",
            ROOT / "aion_magnitude/filter_photometry.py", ROOT / "scripts/qwen_survey_gpu_guard.py"]})
    check_match(manifest, a_manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "matched_protocol.json", manifest["protocol"])
    write_json(out / "tokenizer_examples.json", examples)
    write_json(out / "preparation_audit.json", dict(
        source_arrays_identical_to_A=True, source_cohort_file_identical_to_A=control.pilot.sha256(out / "source_cohort.npz") == a_manifest["cohort_file_sha256"],
        labels_values_errors_and_masks_preserved=True, native_source_text_identical_to_A=True,
        target_native_text_identical_to_A=True, original_archive_adapter_and_head_sha256_verified=True,
        source_checkpoint_is_A_continued_checkpoint=False, registry_curves_verified=True,
        target_filter_mapping="named columns; Legacy RELEASE-specific DECam or BASS/MzLS plus WISE; CLAUDS MegaCam/HSC/VIRCAM",
        no_wavelength_or_width_in_source=True, test_evaluated=False, target_labels_fitted=False))
    status = dict(phase="prepared", created_utc=control.now(), test_evaluated=False)
    write_json(out / "status.json", status)
    render_report(out, status)
    print(json.dumps(dict(output=str(out), source_token_audit=audit, targets=target_manifest), indent=2), flush=True)


def load_prepared(out):
    manifest, data, texts = control.load_prepared(out)
    if manifest["experiment"] != "B" or manifest["level"] != 1:
        raise ValueError("Expected an Experiment B Level-1 manifest")
    a_dir = Path(manifest["control_dir"])
    if control.pilot.sha256(a_dir / "manifest.json") != manifest["control_manifest_sha256"]:
        raise ValueError("Matched control manifest changed")
    check_match(manifest, json.loads((a_dir / "manifest.json").read_text()))
    for name, key in [("reference_validation.pt", "reference_validation_sha256"),
                      ("canonical_replay_inputs.jsonl", "canonical_inputs_sha256")]:
        if control.pilot.sha256(out / name) != manifest[key]:
            raise ValueError(f"Prepared input changed: {name}")
    for name, digest in manifest["code_sha256"].items():
        if control.pilot.sha256(ROOT / name) != digest:
            raise ValueError(f"Prepared implementation changed: {name}; review before running")
    return manifest, data, texts


@contextmanager
def reserve_idle_gpu(selection, *, wait, on_wait, poll_seconds=30, lock_root=LOCK_ROOT):
    """Wait without a CUDA context; respect the existing shared advisory locks."""
    device = str(uuid.UUID(selection.removeprefix("MIG-")))
    lock_root.mkdir(parents=True, exist_ok=True)
    while True:
        handle = (lock_root / f"{device}.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            reason = "GPU slice is reserved by another job"
        else:
            try:
                info = memory_info(device)
            except BaseException:
                handle.close()
                raise
            if info["used"] <= .2 and info["free"] >= 8:
                break
            handle.close()
            reason = f"GPU slice is not idle: {info}"
        if not wait:
            raise RuntimeError(reason)
        on_wait(reason)
        time.sleep(poll_seconds)
    try:
        yield dict(uuid=device, memory_before_gib=info)
    finally:
        handle.close()


def metric_row(label, metrics):
    return (f"| {label} | {metrics['cross_entropy']:.6f} | {metrics['nmad']:.6f} | "
            f"{metrics['catastrophic_outlier_fraction']:.3%} | {metrics['bias']:.6f} | "
            f"{metrics['mean_crps']:.6f} | {metrics['p16_p84_coverage']:.3%} |")


def metric_table(rows):
    return ["| Model / evaluation | CE | NMAD | Outliers | Bias | CRPS | 68% coverage |",
            "|---|---:|---:|---:|---:|---:|---:|", *[metric_row(label, m) for label, m in rows], ""]


def render_report(out, status):
    m = json.loads((out / "manifest.json").read_text())
    rows = ["# Experiment B — warm-start Level-1 band-aware QLoRA", "",
        f"Status: **{status['phase']}**. Updated {control.now()}.", "",
        "The run starts from the original completed HSC adapter and trained PDF head, independently of A's continued weights. Restoration of both components is verified tensor by tensor when GPU execution starts. The exact same 22,444 training and 5,683 validation objects, native values/errors/masks, five-decimal formatting and 300-bin 0–4.5 PDF grid are retained. Only the serializer changes to explicit survey and physical-band identities. No wavelengths or widths are added.", "",
        "Training matches A: at most two epochs / 2,806 updates, no head-only warmup, adapter LR 3e-6, head LR 1e-5, effective batch 16, seed 1023, constant learning rates, new AdamW moments, same rank/modules/weight decay/clipping. Source-validation CE selects among update zero and validations every 250 updates plus epoch ends. Test objects are excluded. Target labels enter inference metrics only.", "",
        f"Actual tokenizer audit: {m['token_audit']['n']:,} source texts, maximum {m['token_audit']['max']} / {m['token_audit']['max_seq_length']} tokens, zero truncations. Exact strings and token examples are saved. Target token audits are in the manifest.", ""]
    if status.get("queue_reason"):
        rows += [f"Queued on `{status['gpu']}`: {status['queue_reason']}. Existing GPU jobs retain their reservations; this process waits without a CUDA context.", ""]
    initial_path = out / "serializer_change.json"
    initial = json.loads(initial_path.read_text()) if initial_path.exists() else None
    if initial:
        rows += ["## Serializer-only change and source adaptation", "",
                 "The two zero-update rows use identical original trained weights and the full identical source-validation cohort. Their difference isolates the input-format change.", ""]
        table = [("Original weights / Level 0", initial["original_level0_metrics"]),
                 ("Original weights / Level 1", initial["level1_zero_update_metrics"])]
        if "best_metrics" in status:
            table.append((f"Selected Level 1 / update {status['best_update']}", status["best_metrics"]))
            table.append(("Latest Level-1 validation", status["validation_history"][-1]))
        a_source = Path(m["control_dir"]) / "source_summary.json"
        if a_source.exists():
            a_status = json.loads(a_source.read_text())
            table.append((f"Matched Level 0 A / update {a_status['best_update']}", a_status["best_metrics"]))
        rows += metric_table(table)
        rows += [f"Serializer-only delta CE: {initial['delta_level1_minus_level0']['cross_entropy']:+.6f}; delta NMAD: {initial['delta_level1_minus_level0']['nmad']:+.6f}. Changes are measured rather than presumed to be damage. A separate 32-object canonical Level-0 replay verifies restoration before this diagnostic.", ""]
    else:
        rows += ["Zero-update Level-1 evaluation is pending. No performance conclusion is available yet.", ""]
    if "global_update" in status:
        rows += [f"Completed {status['global_update']:,}/{status['max_updates']:,} optimizer updates. Best candidate: update {status['best_update']}; elapsed training/periodic-validation time {status['elapsed_seconds']/3600:.2f} hours.", ""]
    transfer_path = out / "transfer_summary.json"
    if transfer_path.exists():
        transfer = json.loads(transfer_path.read_text())
        rows += ["## Fixed-weight cross-survey transfer", "",
            "Identical audited 256-object validation queries per target. Legacy uses DESI spectroscopic redshifts and RELEASE-specific DECam or BASS/MzLS grz plus WISE W1/W2. CLAUDS uses ZPHOT and its actual MegaCam u/u_star, HSC grizy and VIRCAM YJHKs bands. Native CLAUDS imputed AB magnitudes and unavailable errors/EBV remain unchanged. There is no target fitting, calibration or target-based selection.", ""]
        table = []
        a_transfer_path = Path(m["control_dir"]) / "transfer_summary.json"
        a_transfer = json.loads(a_transfer_path.read_text()) if a_transfer_path.exists() else None
        for kind, result in transfer["targets"].items():
            table.append((f"{kind} / selected Level 1 B", result["metrics"]))
            if a_transfer:
                other = a_transfer["targets"][kind]
                if other["query_ids"] != result["query_ids"]:
                    raise ValueError("A/B transfer comparison query mismatch")
                table.append((f"{kind} / selected Level 0 A", other["metrics"]))
        rows += metric_table(table)
        if not a_transfer:
            rows += ["Matched A transfer remains pending; no A/B transfer conclusion is available.", ""]
    else:
        rows += ["Fixed-weight target evaluation remains pending until source checkpoint selection finishes.", ""]
    if status["phase"] == "complete":
        best, zero, original = status["best_metrics"], status["zero_update_metrics"], initial["original_level0_metrics"]
        rows += ["## Findings and limits", "",
            f"Continued Level-1 adaptation changes CE by {best['cross_entropy']-zero['cross_entropy']:+.6f} and NMAD by {best['nmad']-zero['nmad']:+.6f} relative to the serializer-only starting point. Relative to the original Level-0 capability, selected Level-1 CE changes by {best['cross_entropy']-original['cross_entropy']:+.6f} and NMAD by {best['nmad']-original['nmad']:+.6f}.", "",
            "Best-CE selection guarantees only non-increasing CE relative to the Level-1 starting point; it does not guarantee retention of original Level-0 performance. Update zero can remain selected. Latest validation and last/selected parameter changes are retained to reveal failed adaptation. Source validation also selects the model, so these are model-selection results, not independent test confirmation. Improved transfer would support the utility of this representation under this protocol, not prove an internal understanding of band identity.", ""]
        if a_source.exists() and a_transfer_path.exists():
            deltas = []
            for kind, result in transfer["targets"].items():
                other = a_transfer["targets"][kind]["metrics"]
                deltas.append(f"{kind}: delta CE {result['metrics']['cross_entropy']-other['cross_entropy']:+.6f}, delta NMAD {result['metrics']['nmad']-other['nmad']:+.6f}, delta outliers {result['metrics']['catastrophic_outlier_fraction']-other['catastrophic_outlier_fraction']:+.3%}")
            rows += ["Relative to matched selected Level-0 A (B minus A): " + "; ".join(deltas) + ". Negative CE/NMAD/outlier differences favor B; assess coverage and bias separately. These fixed query sets do not establish statistical significance or a general advantage across seeds.", ""]
    elif status["phase"] == "failed":
        rows += ["Execution failed; see `failure.json` and `run.log`. No completed scientific result is claimed.", ""]
    for filename in ("learning_curves.png", "serializer_predictions.png", "transfer_predictions.png"):
        if (out / filename).exists():
            rows += [f"![{filename}]({filename})", ""]
    temp = out / "report.md.tmp"
    temp.write_text("\n".join(rows)); temp.replace(out / "report.md")


def subset_evaluation(evaluation, n):
    total = len(evaluation["object_id"])
    return {key:(value[:n] if (key == "object_id" or isinstance(value, torch.Tensor) and value.ndim and len(value) == total) else value)
            for key, value in evaluation.items()}


def transfer(out, manifest, model, tokenizer):
    model.requires_grad_(False); model.eval()
    targets = {}
    for kind, expected in manifest["targets"].items():
        with np.load(out / f"{kind}_cohort.npz", allow_pickle=False) as archive:
            data = {key:archive[key] for key in archive.files}
        if control.prepared_tools.sha_arrays(data) != expected["data_sha256"]:
            raise ValueError("Prepared target values/labels changed")
        if control.pilot.sha256(out / f"{kind}_inputs.jsonl") != expected["inputs_sha256"]:
            raise ValueError("Prepared target text changed")
        ids, texts = read_inputs(out / f"{kind}_inputs.jsonl")
        if ids != expected["query_ids"] or ids != [str(x) for x in data["targetid"]]:
            raise ValueError("Prepared target IDs changed")
        evaluation = control.pilot.evaluate(model, tokenizer, data, texts, manifest["source_config"])
        control.pilot.save_predictions(out / f"{kind}_selected_predictions.npz", data, evaluation)
        targets[kind] = dict(expected, metrics=summarize_pdf_metrics(evaluation))
    write_json(out / "transfer_summary.json", dict(targets=targets, weights_updated=False,
        target_labels_fitted=False, target_calibration=False, source_validation_selection_only=True,
        query_manifest_sha256=manifest["query_manifest_sha256"]))


def plots(out):
    # Reuse the tested source learning-curve plot; no target plot is generated here.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((out / "source_summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    history, training = summary["validation_history"], summary["training_history"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot([x["update"] for x in training], [x["loss"] for x in training], lw=.6)
    axes[0].set(xlabel="Optimizer update", ylabel="Training CE")
    for ax, key in zip(axes[1:], ["cross_entropy", "nmad"]):
        ax.plot([x["update"] for x in history], [x[key] for x in history], marker="o", label="Level 1 B")
        ax.axhline(manifest["reference_metrics"][key], color=".5", ls="--", label="Original Level 0")
        ax.axvline(summary["best_update"], color="red", ls=":")
        a_source = Path(manifest["control_dir"]) / "source_summary.json"
        if a_source.exists():
            a_history = json.loads(a_source.read_text())["validation_history"]
            ax.plot([x["update"] for x in a_history], [x[key] for x in a_history], marker=".", label="Matched Level 0 A")
        ax.set(xlabel="Optimizer update", ylabel=f"Source validation {key}"); ax.legend(fontsize=8)
    for ext in ("png", "pdf"): fig.savefig(out / f"learning_curves.{ext}", dpi=180)
    plt.close(fig)
    original = torch.load(out / "reference_validation.pt", map_location="cpu", weights_only=False)
    initial = torch.load(out / "zero_update_validation.pt", map_location="cpu", weights_only=False)
    selected = torch.load(out / "best_validation.pt", map_location="cpu", weights_only=False)
    # best_validation.pt is the evaluation dictionary saved by the shared trainer.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, evaluation, label in zip(axes, (original, initial, selected), ("Original Level 0", "Zero-update Level 1", "Selected Level 1")):
        ax.scatter(evaluation["z_spec"], evaluation["z_p50"], s=3, alpha=.25)
        ax.plot([0, 4.5], [0, 4.5], color=".4", ls="--")
        ax.set(title=label, xlabel="Source spec-z", ylabel="PDF median", xlim=(0, 4.5), ylim=(0, 4.5))
    for ext in ("png", "pdf"): fig.savefig(out / f"serializer_predictions.{ext}", dpi=180)
    plt.close(fig)
    if not (out / "transfer_summary.json").exists(): return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, kind in zip(axes, ("desi-ls", "clauds")):
        paths = [(out / f"{kind}_selected_predictions.npz", "Selected Level 1 B", "#d06b32"),
                 (Path(manifest["control_dir"]) / f"{kind}_selected_predictions.npz", "Selected Level 0 A", "#3973ac")]
        for path, label, color in paths:
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    ax.scatter(archive["z_spec"], archive["z_p50"], s=9, alpha=.5, label=label, color=color)
        ax.plot([0, 4.5], [0, 4.5], color=".4", ls="--")
        ax.set(title=kind, xlabel="DESI spec-z" if kind == "desi-ls" else "CLAUDS ZPHOT", ylabel="PDF median", xlim=(0, 4.5), ylim=(0, 4.5)); ax.legend(fontsize=8)
    for ext in ("png", "pdf"): fig.savefig(out / f"transfer_predictions.{ext}", dpi=180)
    plt.close(fig)


def run(args):
    out = args.output_dir
    manifest, data, texts = load_prepared(out)
    if (out / "summary.json").exists(): raise FileExistsError("Completed experiment is immutable")
    if (out / "latest.pt").exists() != args.resume:
        raise ValueError("Use --resume exactly when a continuation checkpoint exists")
    def queued(reason):
        status = dict(phase="queued", gpu=args.gpu, queue_reason=reason, updated_utc=control.now(), test_evaluated=False)
        write_json(out / "status.json", status); render_report(out, status)
    with reserve_idle_gpu(args.gpu, wait=args.wait_for_gpu, on_wait=queued) as gpu:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
            raise ValueError("CUDA visibility must match the exclusively reserved MIG slice")
        # Revalidate immutable inputs/code after a potentially long wait.
        manifest, data, texts = load_prepared(out)
        protocol = ContinuedSFTProtocol(**manifest["protocol"])
        write_json(out / "gpu.json", gpu)
        timing_path = out / "timing.json"
        if timing_path.exists(): timing = json.loads(timing_path.read_text())
        else:
            timing = dict(first_run_started_utc=control.now(), excludes_gpu_queue_wait=True)
            write_json(timing_path, timing)
        status = dict(phase="restoring_original_checkpoint", started_utc=control.now(), test_evaluated=False)
        write_json(out / "status.json", status); render_report(out, status)
        set_random_seed(protocol.seed)
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer = control.restore_model(out, manifest)
        def dataset(split):
            rows = np.flatnonzero(data["split"] == split)
            return TextRedshiftDataset([texts[i] for i in rows], data["z"][rows], [str(x) for x in data["targetid"][rows]])
        if args.resume:
            initial = torch.load(out / "zero_update_validation.pt", map_location="cpu", weights_only=False)
        else:
            reference = torch.load(out / "reference_validation.pt", map_location="cpu", weights_only=False)
            ids, canonical_texts = read_inputs(out / "canonical_replay_inputs.jsonl")
            canonical_data = control.pilot.take(data, np.flatnonzero(data["split"] == "val")[:32])
            if ids != [str(x) for x in canonical_data["targetid"]]: raise ValueError("Canonical replay IDs differ")
            canonical = control.pilot.evaluate(model, tokenizer, canonical_data, canonical_texts, manifest["source_config"])
            replay = verify_replay(canonical, subset_evaluation(reference, 32))
            write_json(out / "canonical_replay.json", replay)
            status = dict(phase="zero_update_level1_validation", started_utc=control.now(), test_evaluated=False)
            write_json(out / "status.json", status); render_report(out, status)
            val = control.pilot.take(data, np.flatnonzero(data["split"] == "val"))
            initial = control.pilot.evaluate(model, tokenizer, val, [texts[i] for i in np.flatnonzero(data["split"] == "val")], manifest["source_config"])
            if initial["object_id"] != reference["object_id"]: raise ValueError("Validation IDs changed")
            for key in ("z_spec", "redshift_edges", "redshift_centers"):
                torch.testing.assert_close(initial[key], reference[key], rtol=0, atol=0)
            atomic_torch_save(initial, out / "zero_update_validation.pt")
            old, new = summarize_pdf_metrics(reference), summarize_pdf_metrics(initial)
            write_json(out / "serializer_change.json", dict(n=len(val["z"]), original_level0_metrics=old,
                level1_zero_update_metrics=new, delta_level1_minus_level0={key:new[key]-old[key] for key in control.METRICS},
                same_weights=True, same_objects_and_labels=True, only_serializer_changed=True))
            print("Full Level-1 zero-update diagnostic:", json.dumps(new), flush=True)
        set_random_seed(protocol.seed)
        status = continue_sft(model=model, tokenizer=tokenizer, train_dataset=dataset("train"), val_dataset=dataset("val"),
            initial_evaluation=initial, output_dir=out, protocol=protocol, max_length=manifest["source_config"]["max_length"],
            device="cuda", resume=args.resume, on_progress=lambda state:render_report(out, state))
        status["phase"] = "target_transfer"
        write_json(out / "status.json", status); render_report(out, status)
        transfer(out, manifest, model, tokenizer)
        for name, digest in manifest["initial_snapshot_sha256"].items():
            if control.pilot.sha256(out / "initial_model" / name) != digest:
                raise ValueError("Initial snapshot changed")
        plots(out)
        status.update(phase="complete", finished_utc=control.now(),
            total_run_wall_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(timing["first_run_started_utc"])).total_seconds(),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(out / "summary.json", status); write_json(out / "status.json", status); render_report(out, status)
        print("Experiment B completed:", out / "report.md", flush=True)


def launch(args):
    out = args.output_dir
    load_prepared(out)
    if (out / "summary.json").exists(): raise FileExistsError("Completed experiment is immutable")
    if not args.gpu: raise ValueError("Select one MIG UUID")
    previous = out / "job.json"
    if previous.exists():
        try: os.kill(json.loads(previous.read_text())["pid"], 0)
        except ProcessLookupError: pass
        else: raise RuntimeError("Experiment process is still alive")
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", "run", "--output-dir", str(out.resolve()), "--gpu", args.gpu]
    if args.wait_for_gpu: command.append("--wait-for-gpu")
    if args.resume: command.append("--resume")
    with (out / "run.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(previous, dict(pid=child.pid, launched_utc=control.now(), gpu=args.gpu,
        wait_for_gpu=args.wait_for_gpu, command=command))
    print(json.dumps(dict(pid=child.pid, log=str(out / "run.log"), report=str(out / "report.md"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "launch", "report"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--gpu")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "report":
        if (args.output_dir / "source_summary.json").exists(): plots(args.output_dir)
        render_report(args.output_dir, json.loads((args.output_dir / "status.json").read_text()))
        return
    try:
        {"prepare":prepare, "run":run, "launch":launch}[args.stage](args)
    except Exception as error:
        if args.stage == "run" and (args.output_dir / "manifest.json").exists() and not (args.output_dir / "summary.json").exists():
            write_json(args.output_dir / "failure.json", dict(utc=control.now(), error=repr(error)))
            status = json.loads((args.output_dir / "status.json").read_text())
            status["phase"] = "failed"; write_json(args.output_dir / "status.json", status); render_report(args.output_dir, status)
        raise


if __name__ == "__main__": main()
