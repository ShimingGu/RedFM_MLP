#!/usr/bin/env python3
"""Experiment A: full-cohort Level-0 continued SFT from the trained HSC model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from aion_magnitude.FM_Qwen import load_frozen_qwen, resolve_qwen_model_path
from aion_magnitude.filter_photometry import audit_tokens, serialize_rows
from aion_magnitude.metrics import summarize_pdf_metrics
from aion_magnitude.qwen_continued_sft import (
    ContinuedSFTProtocol, atomic_torch_save, continue_sft, verify_replay, write_json,
)
from aion_magnitude.qwen_posttraining import (
    QwenPhotoZModel, TextRedshiftDataset, evaluate_text_photoz_model,
    make_text_collator, qwen_hidden_size, trainable_parameter_summary,
)
from aion_magnitude.utils import set_random_seed
import qwen_filter_aware as prepared_tools
import qwen_zero_shot_sanity as pilot

SOURCE = Path("/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10")
DEFAULT_OUTPUT = ROOT / "filter-aware-runs/1023/hsc-warmstart/level0"
FAST = ROOT / "fast-check-runs/1023/hsc-qlora-to-legacy-clauds"
METRICS = ("cross_entropy", "nmad", "catastrophic_outlier_fraction", "bias", "mean_crps", "p16_p84_coverage")


def now():
    return datetime.now(timezone.utc).isoformat()


def render_report(out, status):
    manifest = json.loads((out / "manifest.json").read_text())
    p = manifest["protocol"]
    rows = ["# Experiment A — warm-start Level-0 continued SFT", "",
            f"Status: **{status['phase']}**. Updated {now()}.", "",
            "Both the archived trained HSC QLoRA adapter and its matching trained PDF head are restored. Inputs use the original Level-0 serializer, full 22,444/5,683 source train/validation cohorts and the original 300-bin 0–4.5 grid. Test objects do not enter training or checkpoint selection.", "",
            f"Protocol: maximum {p['max_epochs']} additional epochs, no head-only warmup, adapter LR {p['adapter_lr']}, head LR {p['head_lr']}, effective batch {p['microbatch_size']*p['gradient_accumulation_steps']}, seed {p['seed']}, source validation every {p['eval_every_updates']} updates plus epoch ends. Learning rates are constant. AdamW moments are newly initialized; adapter/head weights are warm-started. Epoch-final partial batches use their actual sample count. These choices are recorded in `matched_protocol.json` for identical reuse by Experiments B/C.", ""]
    replay_path = out / "zero_update_replay.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        rows += [f"Zero-update replay: passed on {replay['n']:,} validation objects; max logit difference {replay['max_abs_logit_error']:.8g}, max PDF difference {replay['max_abs_pdf_error']:.8g}.", ""]
    else:
        rows += ["Zero-update validation replay has not completed. No optimisation is permitted until it passes.", ""]
    if "global_update" in status:
        rows += [f"Completed optimizer updates: {status['global_update']:,}/{status['max_updates']:,}. Best source-CE candidate so far: update {status['best_update']:,}. Update zero is included.", "",
                 "| Source validation | CE | NMAD | Outliers | Bias | CRPS | 68% coverage |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        values = [("Zero updates", status["zero_update_metrics"]), ("Best selected so far", status["best_metrics"])]
        if status.get("validation_history"):
            values.append(("Latest validation", status["validation_history"][-1]))
        for label, m in values:
            rows.append(f"| {label} | {m['cross_entropy']:.6f} | {m['nmad']:.6f} | {m['catastrophic_outlier_fraction']:.3%} | {m['bias']:.6f} | {m['mean_crps']:.6f} | {m['p16_p84_coverage']:.3%} |")
        rows += ["", f"Training/periodic-validation wall time so far: {status['elapsed_seconds']/3600:.3f} hours. Peak allocated GPU memory: {status['peak_allocated_gib']:.3f} GiB.", ""]
    transfer_path = out / "transfer_summary.json"
    if transfer_path.exists():
        transfer = json.loads(transfer_path.read_text())
        rows += ["## Fixed-weight transfer after source checkpoint selection", "",
                 "The selected source model and archived baseline use identical audited target validation queries. Legacy labels are DESI spec-z; CLAUDS labels are ZPHOT. No target labels are fitted or used for checkpoint selection.", "",
                 "| Target / model | CE | NMAD | Outliers | Bias | CRPS | 68% coverage |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for kind, result in transfer["targets"].items():
            for name, m in [("Zero-update source", result["zero_update_metrics"]), ("Selected source", result["metrics"])]:
                rows.append(f"| {kind}: {name} | {m['cross_entropy']:.6f} | {m['nmad']:.6f} | {m['catastrophic_outlier_fraction']:.3%} | {m['bias']:.6f} | {m['mean_crps']:.6f} | {m['p16_p84_coverage']:.3%} |")
    else:
        rows += ["Fixed-weight Legacy/CLAUDS transfer remains pending until source checkpoint selection finishes.", ""]
    if status["phase"] == "complete":
        zero, best = status["zero_update_metrics"], status["best_metrics"]
        rows += ["", "## Answer", "",
                 f"The selected checkpoint is update {status['best_update']}. Relative to zero updates, source CE changes by {best['cross_entropy']-zero['cross_entropy']:+.6f}, NMAD by {best['nmad']-zero['nmad']:+.6f}, and CRPS by {best['mean_crps']-zero['mean_crps']:+.6f}. Selection guarantees only non-increasing validation CE because the initial checkpoint is a candidate; the latest-training row above shows whether continued optimisation itself degraded performance.", "",
                 "Changes in other metrics and target performance must be read separately. This unchanged-input control cannot establish any benefit from filter metadata. Source validation also selects the checkpoint, so these are validation/model-selection results rather than independent test confirmation.", ""]
        rows += ["Parameter changes for both selected and last-trained states are recorded in `parameter_changes.json`. The final summary records total run wall time including restoration, replay and transfer; the table above reports the training/periodic-validation portion.", ""]
    elif status["phase"] == "failed":
        rows += ["Execution failed; inspect `run.log` and `failure.json`. A final scientific conclusion is unavailable.", ""]
    else:
        rows += ["The experiment is still in progress. The source-preservation question cannot yet be answered from a final selected checkpoint and transfer evaluation.", ""]
    for name in ("learning_curves.png", "transfer_predictions.png"):
        if (out / name).exists():
            rows += [f"![{name}]({name})", ""]
    temporary = out / "report.md.tmp"
    temporary.write_text("\n".join(rows))
    temporary.replace(out / "report.md")


def prepare(args):
    out = args.output_dir
    if out.exists():
        raise FileExistsError("Use the existing manifest to run/resume; never overwrite a prepared experiment")
    source = args.source_run
    meta = json.loads((source / "prepared.json").read_text())
    config = torch.load(source / "qlora/result.pt", map_location="cpu", weights_only=False, mmap=True)
    qconfig = config["metadata"]["config"]
    adapter = json.loads((source / "qlora/adapter/adapter_config.json").read_text())
    if meta["model"] != "Qwen3.5-4B-Base" or Path(adapter["base_model_name_or_path"]).name != "Qwen3.5-4B-Base":
        raise ValueError("Expected the archived Qwen3.5-4B-Base checkpoint")
    if (adapter["r"] != qconfig["lora_rank"] or adapter["use_dora"]
            or set(adapter["target_modules"]) != set(qconfig["lora_target_modules"].split(","))):
        raise ValueError("Archived adapter configuration mismatch")
    cohort, _ = pilot.read_cohort(source)
    selected = np.flatnonzero(np.isin(cohort["split"], ["train", "val"]))
    data = pilot.take(cohort, selected)
    if [(data["split"] == s).sum() for s in ["train", "val"]] != [22444, 5683]:
        raise ValueError("The complete original HSC train/validation cohorts are required")
    if set(data["group"][data["split"] == "train"]) & set(data["group"][data["split"] == "val"]):
        raise ValueError("Physical source train/validation overlap")
    reference = config["val_evaluation"]
    val = data["split"] == "val"
    if [str(x) for x in data["targetid"][val]] != reference["object_id"]:
        raise ValueError("Archived validation ID order differs")
    torch.testing.assert_close(torch.from_numpy(data["z"][val]), reference["z_spec"], rtol=0, atol=0)
    texts = serialize_rows(data, meta, catalogue="galaxiesml-spec", level=0)
    if texts != pilot.native_texts("galaxiesml-spec", data, meta):
        raise ValueError("Level-0 source text changed")
    tokenizer = AutoTokenizer.from_pretrained(resolve_qwen_model_path(qconfig["model_path"]), local_files_only=True)
    audit = audit_tokens(tokenizer, texts, qconfig["max_length"])
    actual_base = prepared_tools.survey.model_source(argparse.Namespace(qwen_model=qconfig["model_path"]))
    source_contract = json.loads((source / "run_contract.json").read_text())
    if actual_base != source_contract["model_source"]:
        raise ValueError("Base checkpoint fingerprint differs from archived training")
    out.mkdir(parents=True)
    shutil.copytree(source / "qlora/adapter", out / "initial_model/adapter")
    shutil.copy2(source / "qlora/photoz_head.pt", out / "initial_model/photoz_head.pt")
    hashes = {str(p.relative_to(out / "initial_model")):pilot.sha256(p) for p in (out / "initial_model").rglob("*") if p.is_file()}
    fast = json.loads((FAST / "summary.json").read_text())
    for name in ("adapter/adapter_model.safetensors", "photoz_head.pt"):
        if hashes[name] != fast["source_snapshot_sha256"][name]:
            raise ValueError("Audited transfer baseline uses different source weights")
    np.savez_compressed(out / "source_cohort.npz", **data)
    with (out / "source_inputs.jsonl").open("w") as f:
        for i, text in enumerate(texts):
            f.write(json.dumps(dict(object_id=str(data["targetid"][i]), text=text)) + "\n")
    atomic_torch_save(reference, out / "reference_validation.pt")
    protocol = asdict(ContinuedSFTProtocol())
    write_json(out / "matched_protocol.json", protocol)
    manifest = dict(created_utc=now(), experiment="A", level=0, source_run=str(source.resolve()),
                    model=meta["model"], model_source=actual_base, source_config=qconfig,
                    adapter_config=adapter, protocol=protocol, initial_snapshot_sha256=hashes,
                    cohort_sha256=prepared_tools.sha_arrays(data), cohort_file_sha256=pilot.sha256(out / "source_cohort.npz"),
                    source_inputs_sha256=pilot.sha256(out / "source_inputs.jsonl"),
                    train_rows=22444, val_rows=5683, max_updates=2806, token_audit=audit,
                    serializer="exact archived native_photometry_specz_v1 / Level 0",
                    reference_metrics=summarize_pdf_metrics(reference), test_evaluated=False,
                    query_manifest_sha256=pilot.sha256(FAST / "summary.json"),
                    code_sha256={str(p.relative_to(ROOT)):pilot.sha256(p) for p in [Path(__file__), ROOT / "aion_magnitude/qwen_continued_sft.py"]})
    write_json(out / "manifest.json", manifest)
    write_json(out / "status.json", dict(phase="prepared", created_utc=now(), test_evaluated=False))
    render_report(out, dict(phase="prepared"))
    print(json.dumps(dict(output=str(out), train=22444, validation=5683, token_audit=audit, protocol=protocol), indent=2), flush=True)


def load_prepared(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if pilot.sha256(out / "source_inputs.jsonl") != manifest["source_inputs_sha256"]:
        raise ValueError("Source text file changed")
    with np.load(out / "source_cohort.npz", allow_pickle=False) as a:
        data = {k:a[k] for k in a.files}
    if prepared_tools.sha_arrays(data) != manifest["cohort_sha256"]:
        raise ValueError("Source cohort changed")
    payloads = [json.loads(line) for line in (out / "source_inputs.jsonl").read_text().splitlines()]
    if [p["object_id"] for p in payloads] != [str(x) for x in data["targetid"]]:
        raise ValueError("Source text IDs differ")
    if np.any(data["split"] == "test"):
        raise ValueError("Test objects must not enter this experiment")
    return manifest, data, [p["text"] for p in payloads]


def restore_model(out, manifest):
    from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model_state_dict
    from safetensors.torch import load_file

    for name, digest in manifest["initial_snapshot_sha256"].items():
        if pilot.sha256(out / "initial_model" / name) != digest:
            raise ValueError("Initial adapter or PDF head changed")
    config = manifest["source_config"]
    base, tokenizer = load_frozen_qwen(config["model_path"], device="cuda", load_in_4bit=True, torch_dtype="bf16", local_files_only=True)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    qwen = PeftModel.from_pretrained(base, out / "initial_model/adapter", is_trainable=True)
    expected_adapter = load_file(str(out / "initial_model/adapter/adapter_model.safetensors"))
    actual_adapter = get_peft_model_state_dict(qwen)
    if actual_adapter.keys() != expected_adapter.keys():
        raise ValueError("Restored adapter tensor names differ from the trained checkpoint")
    for key, value in actual_adapter.items():
        torch.testing.assert_close(value.cpu(), expected_adapter[key], rtol=0, atol=0, check_dtype=False)
    model = QwenPhotoZModel(qwen, hidden_size=qwen_hidden_size(qwen), n_z_bins=config["n_z_bins"], head_hidden_dim=config["head_hidden_dim"], pooling=config["pooling"]).to("cuda")
    head_state = torch.load(out / "initial_model/photoz_head.pt", map_location="cpu", weights_only=True)
    model.photoz_head.load_state_dict(head_state, strict=True)
    for key, value in model.photoz_head.state_dict().items():
        torch.testing.assert_close(value.cpu(), head_state[key], rtol=0, atol=0)
    model.photoz_head.requires_grad_(True)
    model.freeze_qwen_representation = False
    model.eval()
    summary = trainable_parameter_summary(model)
    if summary["trainable_parameters"] != 2371884:
        raise ValueError("Trainable adapter/head architecture changed")
    write_json(out / "restoration.json", dict(trained_adapter_loaded=True, trained_pdf_head_loaded=True,
               adapter_trainable=True, head_trainable=True, **summary))
    return model, tokenizer


def transfer(out, manifest, model, tokenizer):
    fast = json.loads((FAST / "summary.json").read_text())
    if pilot.sha256(FAST / "summary.json") != manifest["query_manifest_sha256"]:
        raise ValueError("Audited query manifest changed")
    model.requires_grad_(False)
    model.eval()
    targets = {}
    for kind in ("desi-ls", "clauds"):
        with np.load(FAST / f"{kind}_predictions.npz", allow_pickle=False) as a:
            data = dict(targetid=a["targetid"], features=a["features"], z=a["z_spec"])
        original = json.loads((FAST / f"{kind}_inputs.json").read_text())
        texts = serialize_rows(data, fast["targets"][kind], catalogue=kind, level=0)
        if texts != original["texts"] or original["object_id"] != [str(x) for x in data["targetid"]]:
            raise ValueError("Target validation inputs differ from audited queries")
        audit = audit_tokens(tokenizer, texts, manifest["source_config"]["max_length"])
        evaluation = pilot.evaluate(model, tokenizer, data, texts, manifest["source_config"])
        pilot.save_predictions(out / f"{kind}_selected_predictions.npz", data, evaluation)
        targets[kind] = dict(n=len(texts), label=fast["targets"][kind]["label"], metrics=summarize_pdf_metrics(evaluation),
                             zero_update_metrics=fast["targets"][kind]["metrics"], token_audit=audit,
                             query_ids=original["object_id"], overlap_audit=fast["targets"][kind]["overlap_audit"])
    write_json(out / "transfer_summary.json", dict(targets=targets, weights_updated=False, target_labels_fitted=False,
                       query_manifest_sha256=manifest["query_manifest_sha256"]))


def plots(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    status = json.loads((out / "source_summary.json").read_text())
    history, training = status["validation_history"], status["training_history"]
    fig, axes = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    axes[0].plot([x["update"] for x in training],[x["loss"] for x in training],lw=.6)
    axes[0].set(xlabel="Optimizer update",ylabel="Training CE per effective batch")
    for ax, key in zip(axes[1:], ["cross_entropy","nmad"]):
        ax.plot([x["update"] for x in history],[x[key] for x in history],marker="o")
        ax.axhline(history[0][key],color=".5",ls="--")
        ax.axvline(status["best_update"],color="red",ls=":")
        ax.set(xlabel="Optimizer update",ylabel=f"Validation {key}")
    for ext in ("png","pdf"):fig.savefig(out / f"learning_curves.{ext}",dpi=180)
    plt.close(fig)
    if not (out / "transfer_summary.json").exists():
        return
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax, kind in zip(axes,["desi-ls","clauds"]):
        for path,label,color in [(FAST / f"{kind}_predictions.npz","Zero-update source","#3973ac"),
                                 (out / f"{kind}_selected_predictions.npz","Selected source","#d06b32")]:
            with np.load(path,allow_pickle=False) as a:
                ax.scatter(a["z_spec"],a["z_p50"],s=9,alpha=.5,label=label,color=color)
        ax.plot([0,4.5],[0,4.5],color=".4",ls="--")
        ax.set(title=kind,xlabel="DESI spec-z" if kind=="desi-ls" else "CLAUDS ZPHOT",ylabel="PDF median",xlim=(0,4.5),ylim=(0,4.5))
        ax.legend(fontsize=8)
    for ext in ("png","pdf"):fig.savefig(out / f"transfer_predictions.{ext}",dpi=180)
    plt.close(fig)


def run(args):
    out = args.output_dir
    manifest, data, texts = load_prepared(out)
    protocol = ContinuedSFTProtocol(**manifest["protocol"])
    if (out / "summary.json").exists():
        raise FileExistsError("Experiment already completed")
    if (out / "latest.pt").exists() and not args.resume:
        raise FileExistsError("Use --resume for an interrupted continuation")
    if args.resume and not (out / "latest.pt").exists():
        raise FileNotFoundError("No continuation checkpoint to resume")
    timing_path = out / "timing.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
    else:
        timing = dict(first_run_started_utc=now(), wall_time_definition="elapsed since first run start; includes interruptions if resumed")
        write_json(timing_path, timing)
    with pilot.reserve_one_gpu() as gpu:
        if gpu["memory_before_gib"]["used"] > .2:
            raise RuntimeError("Warm-start run requires an idle, unreserved slice")
        write_json(out / "gpu.json", gpu)
        set_random_seed(protocol.seed)
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer = restore_model(out, manifest)
        def dataset(split):
            rows = np.flatnonzero(data["split"] == split)
            return TextRedshiftDataset([texts[i] for i in rows], data["z"][rows], [str(x) for x in data["targetid"][rows]])
        reference = torch.load(out / "reference_validation.pt", map_location="cpu", weights_only=False)
        if args.resume:
            initial = torch.load(out / "zero_update_validation.pt", map_location="cpu", weights_only=False)
        else:
            status = dict(phase="zero_update_replay", started_utc=now(), test_evaluated=False)
            write_json(out / "status.json", status); render_report(out, status)
            loader = DataLoader(dataset("val"), batch_size=protocol.eval_batch_size, shuffle=False,
                        collate_fn=make_text_collator(tokenizer, max_length=manifest["source_config"]["max_length"]))
            initial = evaluate_text_photoz_model(model,loader,device=torch.device("cuda"),
                        redshift_edges=reference["redshift_edges"].to("cuda"),redshift_centers=reference["redshift_centers"])
            replay = verify_replay(initial, reference)
            atomic_torch_save(initial, out / "zero_update_validation.pt")
            write_json(out / "zero_update_replay.json", replay)
            print("Full validation replay passed:", json.dumps(replay), flush=True)
        set_random_seed(protocol.seed)
        status = continue_sft(model=model,tokenizer=tokenizer,train_dataset=dataset("train"),val_dataset=dataset("val"),
                   initial_evaluation=initial,output_dir=out,protocol=protocol,max_length=manifest["source_config"]["max_length"],
                   device="cuda",resume=args.resume,on_progress=lambda s:render_report(out,s))
        transfer(out,manifest,model,tokenizer)
        for name,digest in manifest["initial_snapshot_sha256"].items():
            if pilot.sha256(out / "initial_model" / name) != digest:
                raise ValueError("Initial model snapshot was modified")
        status.update(phase="complete", finished_utc=now(),
                      total_run_wall_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(timing["first_run_started_utc"])).total_seconds(),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(out / "summary.json",status);write_json(out / "status.json",status)
        plots(out);render_report(out,status)
        print("Experiment A completed:",out / "report.md",flush=True)


def launch(args):
    out = args.output_dir
    load_prepared(out)
    if (out / "summary.json").exists():
        raise FileExistsError("Completed experiment is immutable; use --stage report to view/regenerate its report")
    if not args.gpu:
        raise ValueError("--gpu must identify one idle MIG slice")
    previous = out / "job.json"
    if previous.exists():
        pid = json.loads(previous.read_text())["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Experiment process {pid} is still running")
    command = [sys.executable,str(Path(__file__).resolve()),"--stage","run","--output-dir",str(out.resolve())]
    if args.resume:
        command.append("--resume")
    with (out / "run.log").open("a") as log:
        child = subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu),
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(previous,dict(pid=child.pid,launched_utc=now(),gpu=args.gpu,command=command))
    print(json.dumps(dict(pid=child.pid,log=str(out/"run.log"),report=str(out/"report.md"))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("prepare","run","launch","report"),required=True)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--source-run",type=Path,default=SOURCE)
    p.add_argument("--gpu")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.stage=="report":
        status=json.loads((args.output_dir/"status.json").read_text());render_report(args.output_dir,status)
        if (args.output_dir/"source_summary.json").exists():plots(args.output_dir)
        return
    try:
        {"prepare":prepare,"run":run,"launch":launch}[args.stage](args)
    except Exception as error:
        if args.stage=="run" and (args.output_dir/"manifest.json").exists() and not (args.output_dir/"summary.json").exists():
            write_json(args.output_dir/"failure.json",dict(utc=now(),error=repr(error)))
            status=json.loads((args.output_dir/"status.json").read_text())
            status["phase"]="failed";write_json(args.output_dir/"status.json",status);render_report(args.output_dir,status)
        raise


if __name__=="__main__":main()
