#!/usr/bin/env python3
"""Experiment C: independent Level-2 warm start, matched to Level-0/1 controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import torch
from transformers import AutoTokenizer

from aion_magnitude.FM_Qwen import resolve_qwen_model_path
from aion_magnitude.filter_photometry import DEFAULT_REGISTRY, SERIALIZER_VERSION, audit_tokens, load_registry, serialize_rows
from aion_magnitude.metrics import summarize_pdf_metrics
from aion_magnitude.qwen_continued_sft import ContinuedSFTProtocol, atomic_torch_save, continue_sft, verify_replay, write_json
from aion_magnitude.qwen_posttraining import TextRedshiftDataset
from aion_magnitude.utils import set_random_seed
import qwen_filter_band_warmstart as band

control = band.control
DEFAULT_OUTPUT = ROOT / "filter-aware-runs/1023/hsc-warmstart/level2"
PHYSICAL_FIELDS = re.compile(r" pivot_nm=\d+\.\d+ width_nm=\d+\.\d+")


def audit_metadata_delta(level1, level2, registry):
    """Verify every numerical field against its named filter, with no other edits."""
    counts = {}
    for one, two in zip(level1, level2, strict=True):
        if PHYSICAL_FIELDS.sub("", two) != one:
            raise ValueError("Level 2 differs from Level 1 beyond pivot/width metadata")
        for line in two.splitlines()[1:-1]:
            fields = dict(item.split("=", 1) for item in line.split())
            key = fields["filter"]
            entry = registry["filters"][key]
            if fields.get("pivot_nm") != f"{entry['pivot_nm']:.3f}" or fields.get("width_nm") != f"{entry['width_nm']:.3f}":
                raise ValueError(f"Incorrect physical metadata for {key}")
            counts[key] = counts.get(key, 0) + 1
    return dict(n=len(level2), only_pivot_and_width_added=True, filter_observation_counts=counts)


def audit_registry(registry):
    """Reproduce pinned response integrals in nm; hashes are checked by the loader."""
    filters = {}
    for key, entry in registry["filters"].items():
        x, y = np.loadtxt(DEFAULT_REGISTRY.parent / entry["curve_path"], delimiter=",", skiprows=1, unpack=True)
        if entry["wavelength_unit"] != "nm" or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x <= 0) or np.any(np.diff(x) <= 0) or np.any(y < 0) or y.max() <= 0:
            raise ValueError(f"Invalid response/units for {key}")
        pivot = float(np.sqrt(np.trapezoid(y*x, x) / np.trapezoid(y/x, x)))
        width = float(np.trapezoid(y, x) / y.max())
        if not np.isclose(pivot, entry["pivot_nm"], rtol=1e-9) or not np.isclose(width, entry["width_nm"], rtol=1e-9):
            raise ValueError(f"Response integrals do not reproduce registry metadata: {key}")
        filters[key] = dict(pivot_nm=pivot, width_nm=width, wavelength_unit="nm",
            serialized_pivot_nm=f"{entry['pivot_nm']:.3f}", serialized_width_nm=f"{entry['width_nm']:.3f}",
            approximation=entry["approximation"], source_commit=entry["source_commit"],
            original_sha256=entry["original_sha256"], curve_sha256=entry["curve_sha256"])
    return dict(definitions=registry["definitions"], filters=filters, all_profiles_and_original_hashes_verified=True,
                all_integrals_reproduced=True, descriptor_precision="three decimal places in nm")


def prepare(args):
    out = args.output_dir
    if out.exists(): raise FileExistsError("Never overwrite a prepared experiment")
    b_manifest, data, b_texts = band.load_prepared(args.band_control_dir)
    a_dir = Path(b_manifest["control_dir"])
    a_manifest, a_data, _ = control.load_prepared(a_dir)
    band.check_match(b_manifest, a_manifest)
    if control.prepared_tools.sha_arrays(a_data) != control.prepared_tools.sha_arrays(data):
        raise ValueError("Complete A/B source arrays differ")
    if b_manifest["source_config"]["head_hidden_dim"] != 256 or b_manifest["adapter_config"]["r"] != 8:
        raise ValueError("Expected the original trained head and rank-8 adapter")
    for name, digest in b_manifest["initial_snapshot_sha256"].items():
        if control.pilot.sha256(control.SOURCE / "qlora" / name) != digest or control.pilot.sha256(args.band_control_dir / "initial_model" / name) != digest:
            raise ValueError("C must start from the original archived adapter AND trained head")
    registry = load_registry(verify_curves=True)
    registry_audit = audit_registry(registry)
    if control.pilot.sha256(DEFAULT_REGISTRY) != b_manifest["registry_sha256"]:
        raise ValueError("Level 1 and Level 2 must use the same registry")
    metadata = json.loads((control.SOURCE / "prepared.json").read_text())
    texts = serialize_rows(data, metadata, catalogue="galaxiesml-spec", level=2, registry=registry)
    delta = {"source":audit_metadata_delta(b_texts, texts, registry)}
    config = b_manifest["source_config"]
    tokenizer = AutoTokenizer.from_pretrained(resolve_qwen_model_path(config["model_path"]), local_files_only=True)
    token_audit = audit_tokens(tokenizer, texts, config["max_length"])
    actual_base = control.prepared_tools.survey.model_source(argparse.Namespace(qwen_model=config["model_path"]))
    if actual_base != b_manifest["model_source"]: raise ValueError("Base model changed")
    fast = json.loads((control.FAST / "summary.json").read_text())
    if control.pilot.sha256(control.FAST / "summary.json") != b_manifest["query_manifest_sha256"]:
        raise ValueError("Target query manifest changed")
    out.mkdir(parents=True)
    for name in ("source_cohort.npz", "reference_validation.pt", "canonical_replay_inputs.jsonl"):
        shutil.copy2(args.band_control_dir / name, out / name)
    shutil.copytree(args.band_control_dir / "initial_model", out / "initial_model")
    band.write_inputs(out / "source_inputs.jsonl", data["targetid"], texts)
    examples = {"source_train":band.token_example(tokenizer, texts[np.flatnonzero(data["split"] == "train")[0]]),
                "source_val":band.token_example(tokenizer, texts[np.flatnonzero(data["split"] == "val")[0]])}
    targets = {}
    for kind, expected in b_manifest["targets"].items():
        with np.load(args.band_control_dir / f"{kind}_cohort.npz", allow_pickle=False) as archive:
            target = {key:archive[key] for key in archive.files}
        if control.prepared_tools.sha_arrays(target) != expected["data_sha256"]:
            raise ValueError("B target arrays changed")
        input_file = args.band_control_dir / f"{kind}_inputs.jsonl"
        if control.pilot.sha256(input_file) != expected["inputs_sha256"]:
            raise ValueError("B target strings changed")
        ids, one = band.read_inputs(input_file)
        if ids != expected["query_ids"] or ids != [str(x) for x in target["targetid"]]:
            raise ValueError("B target IDs changed")
        with np.load(control.FAST / f"{kind}_predictions.npz", allow_pickle=False) as original:
            for key, old in [("targetid", "targetid"), ("features", "features"), ("z", "z_spec")]:
                np.testing.assert_array_equal(target[key], original[old])
        two = serialize_rows(target, fast["targets"][kind], catalogue=kind, level=2, registry=registry)
        delta[kind] = audit_metadata_delta(one, two, registry)
        audit = audit_tokens(tokenizer, two, config["max_length"])
        shutil.copy2(args.band_control_dir / f"{kind}_cohort.npz", out / f"{kind}_cohort.npz")
        band.write_inputs(out / f"{kind}_inputs.jsonl", target["targetid"], two)
        targets[kind] = dict(expected, inputs_sha256=control.pilot.sha256(out / f"{kind}_inputs.jsonl"), token_audit=audit)
        examples[kind] = band.token_example(tokenizer, two[0])
    manifest = dict(b_manifest)
    manifest.pop("prelaunch_revision", None)
    manifest.update(experiment="C", level=2, created_utc=control.now(),
        band_control_dir=str(args.band_control_dir.resolve()), band_control_manifest_sha256=control.pilot.sha256(args.band_control_dir / "manifest.json"),
        serializer=f"{SERIALIZER_VERSION} / Level 2 / identical Level-1 text plus pivot_nm and width_nm",
        source_inputs_sha256=control.pilot.sha256(out / "source_inputs.jsonl"), token_audit=token_audit, targets=targets,
        code_sha256={str(p.relative_to(ROOT)):control.pilot.sha256(p) for p in [Path(__file__),
            ROOT / "notebooks/qwen_filter_band_warmstart.py", ROOT / "notebooks/qwen_filter_warmstart.py",
            ROOT / "aion_magnitude/qwen_continued_sft.py", ROOT / "aion_magnitude/filter_photometry.py",
            ROOT / "scripts/qwen_survey_gpu_guard.py"]})
    band.check_match(manifest, b_manifest); band.check_match(manifest, a_manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "matched_protocol.json", manifest["protocol"])
    write_json(out / "metadata_delta_audit.json", delta)
    write_json(out / "registry_audit.json", registry_audit)
    shutil.copy2(DEFAULT_REGISTRY, out / "filter_registry_snapshot.json")
    write_json(out / "tokenizer_examples.json", examples)
    write_json(out / "preparation_audit.json", dict(source_arrays_identical_to_A_and_B=True,
        original_archived_adapter_and_trained_head_hashes_verified=True, independent_branch=True,
        source_and_target_texts_differ_from_B_only_in_pivot_and_width=True,
        max_length_common_to_all_arms=config["max_length"], no_truncation=True, test_evaluated=False,
        target_labels_fitted=False, full_transmission_curves_in_prompt=False, learned_filter_embeddings=False))
    status = dict(phase="prepared", created_utc=control.now(), test_evaluated=False)
    write_json(out / "status.json", status); render_report(out, status)
    print(json.dumps(dict(output=str(out), source_token_audit=token_audit,
        target_token_audits={kind:value["token_audit"] for kind,value in targets.items()}, metadata_delta_audit=delta), indent=2), flush=True)


def load_prepared(out):
    manifest, data, texts = control.load_prepared(out)
    if manifest["experiment"] != "C" or manifest["level"] != 2:
        raise ValueError("Expected Experiment C Level 2")
    for key, digest in [("control_dir", "control_manifest_sha256"), ("band_control_dir", "band_control_manifest_sha256")]:
        path = Path(manifest[key]) / "manifest.json"
        if control.pilot.sha256(path) != manifest[digest]: raise ValueError("Matched control manifest changed")
        band.check_match(manifest, json.loads(path.read_text()))
    for name, key in [("reference_validation.pt", "reference_validation_sha256"), ("canonical_replay_inputs.jsonl", "canonical_inputs_sha256"),
                      ("filter_registry_snapshot.json", "registry_sha256")]:
        if control.pilot.sha256(out / name) != manifest[key]: raise ValueError(f"Prepared input changed: {name}")
    for name, digest in manifest["code_sha256"].items():
        if control.pilot.sha256(ROOT / name) != digest: raise ValueError(f"Prepared implementation changed: {name}")
    return manifest, data, texts


def process_identity(pid):
    """Use Linux start ticks to avoid waiting for an unrelated reused PID."""
    try: fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError: return None
    if fields[0] == "Z": return None
    return dict(pid=int(pid), start_ticks=fields[19])


def wait_for_predecessor(identity, on_wait):
    while identity is not None and process_identity(identity["pid"]) == identity:
        on_wait(f"Waiting for Experiment B process {identity['pid']} to finish before competing for its GPU slice")
        time.sleep(30)


def comparison_sources(manifest):
    result = []
    for label, key in [("Level 0 A", "control_dir"), ("Level 1 B", "band_control_dir")]:
        directory = Path(manifest[key])
        if (directory / "source_summary.json").exists():
            path = directory / "summary.json" if (directory / "summary.json").exists() else directory / "source_summary.json"
            summary = json.loads(path.read_text())
            result.append((label, directory, summary))
    return result


def render_report(out, status):
    m = json.loads((out / "manifest.json").read_text())
    rows = ["# Experiment C — warm-start Level-2 filter physics", "", f"Status: **{status['phase']}**. Updated {control.now()}.", "",
        "C is an independent branch from the original archived HSC Qwen3.5-4B-Base rank-8 QLoRA adapter and matching trained 256-hidden-unit PDF head. Both components are verified tensor by tensor when GPU execution begins. It uses the same 22,444 training and 5,683 validation objects, labels, numerical formatting, missingness and 300-bin 0–4.5 grid as A/B. Removing only `pivot_nm` and `width_nm` from every source and target string reproduces B byte for byte.", "",
        "Protocol matches A/B: at most two additional epochs / 2,806 updates; no head-only warmup; adapter LR 3e-6; head LR 1e-5; effective batch 16; seed 1023; unchanged trainable modules, new AdamW moments, constant LRs, clipping and weight decay. Source-validation CE selects among update zero and validations every 250 updates plus epoch ends. No held-out test evaluation or target-label fitting/calibration/selection.", "",
        f"Actual tokenizer audit: source maximum {m['token_audit']['max']} tokens; Legacy {m['targets']['desi-ls']['token_audit']['max']}; CLAUDS {m['targets']['clauds']['token_audit']['max']}. All inputs fit the common {m['source_config']['max_length']}-token limit without truncation.", "",
        "## Filter metadata and approximations", "",
        "Pinned response profiles and original-file hashes are verified. Pivot wavelength is sqrt(∫Rλ dλ / ∫R/λ dλ); rectangular equivalent bandwidth is ∫R dλ / max(R), not FWHM. Both are in nm, serialized to three decimal places. R is the supplied relative photon-counting system response; no extra wavelength factor, SED weights, learned filter embeddings, full transmission curves or additional descriptors enter the prompts. The integrals are reproduced from the vendored profiles in `registry_audit.json`.", "",
        "Source HSC grizy uses nominal HSC 2017 profiles. Exposure mixtures of r/r2 and i/i2 are unavailable in the feature cache, so r and i remain representative approximations; no per-object mixture is invented. Legacy RELEASE 9010/9012 uses representative DECam DR1 grz responses, 9011 uses BASS gr and MzLS z, and all use WISE W1/W2. CLAUDS uses MegaCam u/u_star (MP9302/MP9301), HSC grizy and VIRCAM YJHKs. These nominal curves are not object-specific throughputs. Other registry approximations are preserved in `registry_audit.json` and `filter_registry_snapshot.json`.", ""]
    if status.get("queue_reason"):
        rows += [f"Queued: {status['queue_reason']}. GPU `{status['gpu']}`. Existing jobs retain their reservations; C waits without a CUDA context.", ""]
    initial_path = out / "serializer_change.json"
    initial = json.loads(initial_path.read_text()) if initial_path.exists() else None
    comparisons = comparison_sources(m)
    if initial:
        rows += ["## Source serializer change and recovery", ""]
        table = [("Original archived weights / Level 0", initial["original_level0_metrics"]),
                 ("Original archived weights / Level 2", initial["level2_zero_update_metrics"])]
        b_initial = Path(m["band_control_dir"]) / "serializer_change.json"
        if b_initial.exists(): table.append(("Original archived weights / Level 1", json.loads(b_initial.read_text())["level1_zero_update_metrics"]))
        for label, _, summary in comparisons: table.append((f"Selected {label} / update {summary['best_update']}", summary["best_metrics"]))
        if "best_metrics" in status:
            table += [(f"Selected Level 2 C / update {status['best_update']}", status["best_metrics"]),
                      ("Latest Level-2 validation", status["validation_history"][-1])]
        rows += band.metric_table(table)
        rows += ["The zero-update rows use identical original weights and objects; their differences measure serializer-only changes. Recovery through SFT is measured separately. A 32-object canonical Level-0 replay checks restoration before the full 5,683-object Level-2 diagnostic.", ""]
    else: rows += ["Zero-update source diagnostics and adaptation results are pending; no performance penalty or benefit is presumed.", ""]
    if "global_update" in status:
        rows += [f"Training: {status['global_update']:,}/{status['max_updates']:,} updates; selected update {status['best_update']:,}; training/periodic-validation wall time {status['elapsed_seconds']/3600:.3f} hours; peak allocated GPU memory {status['peak_allocated_gib']:.3f} GiB.", ""]
    transfer_path = out / "transfer_summary.json"
    transfer = json.loads(transfer_path.read_text()) if transfer_path.exists() else None
    target_controls = {}
    if transfer:
        rows += ["## Fixed-weight transfer and matched controls", "",
            "All models use exactly the same audited 256 Legacy and 256 CLAUDS validation objects. Legacy reference: DESI spectroscopic redshift. CLAUDS reference: ZPHOT, not spectroscopy. Existing native CLAUDS imputed AB magnitudes, missing errors/EBV and five-decimal formatting are unchanged.", ""]
        fast = json.loads((control.FAST / "summary.json").read_text())
        table = []
        for label, directory, _ in comparisons:
            path = directory / "transfer_summary.json"
            if path.exists(): target_controls[label] = json.loads(path.read_text())
        for kind, result in transfer["targets"].items():
            table.append((f"{kind} / original archived Level 0", fast["targets"][kind]["metrics"]))
            for label, other in target_controls.items():
                if other["targets"][kind]["query_ids"] != result["query_ids"]: raise ValueError("Control target IDs differ")
                table.append((f"{kind} / selected {label}", other["targets"][kind]["metrics"]))
            table.append((f"{kind} / selected Level 2 C", result["metrics"]))
        rows += band.metric_table(table)
    else: rows += ["Target inference remains pending until source-only checkpoint selection finishes.", ""]
    if status["phase"] == "complete":
        best, zero, original = status["best_metrics"], status["zero_update_metrics"], initial["original_level0_metrics"]
        rows += ["## Findings", "",
            f"Level-2 adaptation changes source CE by {best['cross_entropy']-zero['cross_entropy']:+.6f} and NMAD by {best['nmad']-zero['nmad']:+.6f} relative to its zero-update Level-2 starting point. Relative to the original archived Level-0 capability, selected CE changes by {best['cross_entropy']-original['cross_entropy']:+.6f}, NMAD by {best['nmad']-original['nmad']:+.6f}, and outlier fraction by {best['catastrophic_outlier_fraction']-original['catastrophic_outlier_fraction']:+.3%}.", ""]
        worse = [name for name in ("cross_entropy", "nmad", "catastrophic_outlier_fraction") if best[name] > original[name]]
        rows += [("Selected Level 2 remains numerically worse than the original source model in " + ", ".join(worse) + ". Source quality has not uniformly recovered; target differences cannot provide a clean isolated test of filter-physics transfer." if worse else "Selected Level 2 is no worse than the original source model in the point estimates of CE, NMAD and outlier fraction. This checks recovery on the selection set; it is not independent confirmation of equivalent quality."), ""]
        if "Level 1 B" in target_controls:
            for kind, result in transfer["targets"].items():
                other = target_controls["Level 1 B"]["targets"][kind]["metrics"]
                delta = {key:result["metrics"][key]-other[key] for key in control.METRICS}
                rows += [f"Critical comparison, {kind}, C minus B: CE {delta['cross_entropy']:+.6f}; NMAD {delta['nmad']:+.6f}; outliers {delta['catastrophic_outlier_fraction']:+.3%}; bias {delta['bias']:+.6f}; CRPS {delta['mean_crps']:+.6f}; coverage {delta['p16_p84_coverage']:+.3%}. Negative CE/NMAD/outlier/CRPS differences favor C; bias and coverage require separate interpretation.", ""]
        else: rows += ["The critical selected Level-2 versus Level-1 target comparison is unavailable because B transfer has not completed. No benefit beyond band labels can yet be concluded.", ""]
        rows += [f"Total GPU-run wall time: {status.get('total_run_wall_seconds', 0)/3600:.3f} hours (queue wait excluded; interruptions included if resumed). Peak allocated memory: {status['peak_allocated_gib']:.3f} GiB. GPU UUID and memory at reservation are recorded in `gpu.json`; timing definition is in `timing.json`.", ""]
        costs = [("Level 2 C", status)] + [(label, summary) for label, _, summary in comparisons]
        rows += ["| Arm | Selected / trained updates | Training + validation hours | Total run hours | Peak allocated GiB |", "|---|---:|---:|---:|---:|"]
        for label, summary in costs:
            total_hours = f"{summary['total_run_wall_seconds']/3600:.3f}" if 'total_run_wall_seconds' in summary else "pending"
            rows.append(f"| {label} | {summary['best_update']} / {summary['global_update']} | {summary['elapsed_seconds']/3600:.3f} | {total_hours} | {summary['peak_allocated_gib']:.3f} |")
        rows += [""]
    elif status["phase"] == "failed": rows += ["Execution failed; see `failure.json` and `run.log`. A completed result is unavailable.", ""]
    rows += ["## Interpretation limits", "",
        "The critical comparison is C versus B, since B already supplies explicit band identities. Source validation also selects checkpoints, so these are model-selection results. Retaining the zero-update candidate guarantees non-increasing CE only relative to Level 2, not the original Level 0; latest validation and last/selected parameter changes are saved. Single-seed point estimates on fixed 256-object target samples do not establish statistical significance.", "",
        "Every HSC training galaxy uses essentially the same fixed filter set: wavelength and bandwidth do not vary independently across source objects. Successful source adaptation therefore does not demonstrate learning a physically transferable response dependence. Any transfer improvement is preliminary; multiple training filter systems and shuffled or incorrect metadata controls would be needed. This experiment does not demonstrate universal filter-aware generalisation.", ""]
    for name in ("learning_curves.png", "serializer_predictions.png", "transfer_predictions.png"):
        if (out / name).exists(): rows += [f"![{name}]({name})", ""]
    temp = out / "report.md.tmp"; temp.write_text("\n".join(rows)); temp.replace(out / "report.md")


def plots(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = json.loads((out / "manifest.json").read_text())
    status = json.loads((out / "source_summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    train = status["training_history"]
    axes[0].plot([x["update"] for x in train], [x["loss"] for x in train], lw=.6)
    axes[0].set(xlabel="Optimizer update", ylabel="Level-2 training CE")
    sources = [("Level 2 C", out, status)] + comparison_sources(m)
    for ax, key in zip(axes[1:], ("cross_entropy", "nmad")):
        for label, _, summary in sources:
            history = summary["validation_history"]
            ax.plot([x["update"] for x in history], [x[key] for x in history], marker=".", label=label)
        ax.axhline(m["reference_metrics"][key], ls="--", color=".5", label="Original archived Level 0")
        ax.axvline(status["best_update"], ls=":", color="red")
        ax.set(xlabel="Optimizer update", ylabel=f"Source validation {key}"); ax.legend(fontsize=7)
    for ext in ("png", "pdf"): fig.savefig(out / f"learning_curves.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, name, label in zip(axes, ("reference_validation.pt", "zero_update_validation.pt", "best_validation.pt"),
                              ("Original Level 0", "Zero-update Level 2", "Selected Level 2")):
        evaluation = torch.load(out / name, map_location="cpu", weights_only=False)
        ax.scatter(evaluation["z_spec"], evaluation["z_p50"], s=3, alpha=.25)
        ax.plot([0, 4.5], [0, 4.5], ls="--", color=".4")
        ax.set(title=label, xlabel="Source spec-z", ylabel="PDF median", xlim=(0, 4.5), ylim=(0, 4.5))
    for ext in ("png", "pdf"): fig.savefig(out / f"serializer_predictions.{ext}", dpi=180)
    plt.close(fig)
    if not (out / "transfer_summary.json").exists(): return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, kind in zip(axes, ("desi-ls", "clauds")):
        paths = [(control.FAST / f"{kind}_predictions.npz", "Original Level 0")]
        paths += [(directory / f"{kind}_selected_predictions.npz", label) for label, directory, _ in sources]
        for path, label in paths:
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    ax.scatter(archive["z_spec"], archive["z_p50"], s=8, alpha=.4, label=label)
        ax.plot([0, 4.5], [0, 4.5], ls="--", color=".4")
        ax.set(title=kind, xlabel="DESI spec-z" if kind == "desi-ls" else "CLAUDS ZPHOT", ylabel="PDF median", xlim=(0, 4.5), ylim=(0, 4.5)); ax.legend(fontsize=7)
    for ext in ("png", "pdf"): fig.savefig(out / f"transfer_predictions.{ext}", dpi=180)
    plt.close(fig)


def run(args):
    out = args.output_dir
    manifest, data, texts = load_prepared(out)
    if (out / "summary.json").exists(): raise FileExistsError("Completed experiment is immutable")
    if (out / "latest.pt").exists() != args.resume: raise ValueError("Use --resume exactly when latest.pt exists")
    def queued(reason):
        status = dict(phase="queued", gpu=args.gpu, queue_reason=reason, updated_utc=control.now(), test_evaluated=False)
        write_json(out / "status.json", status); render_report(out, status)
    identity = dict(pid=args.after_pid, start_ticks=args.after_start_ticks) if args.after_pid else None
    wait_for_predecessor(identity, queued)
    with band.reserve_idle_gpu(args.gpu, wait=args.wait_for_gpu, on_wait=queued) as gpu:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu: raise ValueError("CUDA visibility differs from reserved slice")
        manifest, data, texts = load_prepared(out)
        protocol = ContinuedSFTProtocol(**manifest["protocol"])
        write_json(out / "gpu.json", gpu)
        timing_path = out / "timing.json"
        if timing_path.exists(): timing = json.loads(timing_path.read_text())
        else:
            timing = dict(first_run_started_utc=control.now(), excludes_gpu_queue_wait=True, includes_interruptions_if_resumed=True)
            write_json(timing_path, timing)
        status = dict(phase="restoring_original_checkpoint", started_utc=control.now(), test_evaluated=False)
        write_json(out / "status.json", status); render_report(out, status)
        set_random_seed(protocol.seed); torch.cuda.reset_peak_memory_stats()
        model, tokenizer = control.restore_model(out, manifest)
        def dataset(split):
            rows = np.flatnonzero(data["split"] == split)
            return TextRedshiftDataset([texts[i] for i in rows], data["z"][rows], [str(x) for x in data["targetid"][rows]])
        if args.resume: initial = torch.load(out / "zero_update_validation.pt", map_location="cpu", weights_only=False)
        else:
            reference = torch.load(out / "reference_validation.pt", map_location="cpu", weights_only=False)
            ids, canonical_texts = band.read_inputs(out / "canonical_replay_inputs.jsonl")
            indices = np.flatnonzero(data["split"] == "val")
            canonical_data = control.pilot.take(data, indices[:32])
            if ids != [str(x) for x in canonical_data["targetid"]]: raise ValueError("Canonical replay IDs differ")
            canonical = control.pilot.evaluate(model, tokenizer, canonical_data, canonical_texts, manifest["source_config"])
            write_json(out / "canonical_replay.json", verify_replay(canonical, band.subset_evaluation(reference, 32)))
            status = dict(phase="zero_update_level2_validation", started_utc=control.now(), test_evaluated=False)
            write_json(out / "status.json", status); render_report(out, status)
            val = control.pilot.take(data, indices)
            initial = control.pilot.evaluate(model, tokenizer, val, [texts[i] for i in indices], manifest["source_config"])
            if initial["object_id"] != reference["object_id"]: raise ValueError("Validation IDs changed")
            for key in ("z_spec", "redshift_edges", "redshift_centers"):
                torch.testing.assert_close(initial[key], reference[key], rtol=0, atol=0)
            atomic_torch_save(initial, out / "zero_update_validation.pt")
            old, new = summarize_pdf_metrics(reference), summarize_pdf_metrics(initial)
            write_json(out / "serializer_change.json", dict(n=len(val["z"]), original_level0_metrics=old,
                level2_zero_update_metrics=new, delta_level2_minus_level0={key:new[key]-old[key] for key in control.METRICS},
                same_weights=True, same_objects_and_labels=True, only_serializer_changed=True))
            print("Full Level-2 zero-update diagnostic:", json.dumps(new), flush=True)
        set_random_seed(protocol.seed)
        status = continue_sft(model=model, tokenizer=tokenizer, train_dataset=dataset("train"), val_dataset=dataset("val"),
            initial_evaluation=initial, output_dir=out, protocol=protocol, max_length=manifest["source_config"]["max_length"],
            device="cuda", resume=args.resume, on_progress=lambda state:render_report(out, state))
        status["phase"] = "target_transfer"; write_json(out / "status.json", status); render_report(out, status)
        band.transfer(out, manifest, model, tokenizer)
        for name, digest in manifest["initial_snapshot_sha256"].items():
            if control.pilot.sha256(out / "initial_model" / name) != digest: raise ValueError("Original snapshot changed")
        plots(out)
        status.update(phase="complete", finished_utc=control.now(),
            total_run_wall_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(timing["first_run_started_utc"])).total_seconds(),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(out / "summary.json", status); write_json(out / "status.json", status); render_report(out, status)
        print("Experiment C completed:", out / "report.md", flush=True)


def launch(args):
    out = args.output_dir
    manifest, _, _ = load_prepared(out)
    if (out / "summary.json").exists(): raise FileExistsError("Completed experiment is immutable")
    if not args.gpu: raise ValueError("Select one MIG UUID")
    job_file = out / "job.json"
    if job_file.exists() and process_identity(json.loads(job_file.read_text())["pid"]) is not None:
        raise RuntimeError("Experiment C process is still alive")
    b_job = Path(manifest["band_control_dir"]) / "job.json"
    predecessor = process_identity(json.loads(b_job.read_text())["pid"]) if b_job.exists() else None
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", "run", "--output-dir", str(out.resolve()), "--gpu", args.gpu]
    if args.wait_for_gpu: command.append("--wait-for-gpu")
    if args.resume: command.append("--resume")
    if predecessor:
        command += ["--after-pid", str(predecessor["pid"]), "--after-start-ticks", predecessor["start_ticks"]]
    with (out / "run.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(job_file, dict(pid=child.pid, launched_utc=control.now(), gpu=args.gpu, wait_for_gpu=args.wait_for_gpu,
                             predecessor=predecessor, command=command))
    print(json.dumps(dict(pid=child.pid, predecessor=predecessor, report=str(out / "report.md"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "launch", "report"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--band-control-dir", type=Path, default=band.DEFAULT_OUTPUT)
    parser.add_argument("--gpu")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--after-pid", type=int)
    parser.add_argument("--after-start-ticks")
    args = parser.parse_args()
    if bool(args.after_pid) != bool(args.after_start_ticks): parser.error("Predecessor PID and start ticks must be supplied together")
    if args.stage == "report":
        if (args.output_dir / "source_summary.json").exists(): plots(args.output_dir)
        render_report(args.output_dir, json.loads((args.output_dir / "status.json").read_text())); return
    try: {"prepare":prepare, "run":run, "launch":launch}[args.stage](args)
    except Exception as error:
        if args.stage == "run" and (args.output_dir / "manifest.json").exists() and not (args.output_dir / "summary.json").exists():
            write_json(args.output_dir / "failure.json", dict(utc=control.now(), error=repr(error)))
            status = json.loads((args.output_dir / "status.json").read_text()); status["phase"] = "failed"
            write_json(args.output_dir / "status.json", status); render_report(args.output_dir, status)
        raise


if __name__ == "__main__": main()
