#!/usr/bin/env python3
"""Apply the original CLAUDS QLoRA model to native HSC and Legacy validation rows."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import qwen_zero_shot_sanity as pilot
from aion_magnitude.FM_Qwen import load_frozen_qwen
from aion_magnitude.metrics import summarize_pdf_metrics
from aion_magnitude.qwen_posttraining import QwenPhotoZModel, qwen_hidden_size
from aion_magnitude.utils import set_random_seed


def hsc_positions(ids, meta):
    import h5py

    path = next(s["path"] for s in meta["sources"] if s["path"].endswith(".hdf5"))
    with h5py.File(path, "r") as f:
        tids = np.asarray(f["targetid"][:], dtype=np.int64)
        ra = np.asarray(f["DESI_fibermap/TARGET_RA"][:], float)
        dec = np.asarray(f["DESI_fibermap/TARGET_DEC"][:], float)
    index = {int(tid): row for row, tid in enumerate(tids)}
    return np.asarray([(ra[index[int(tid)]], dec[index[int(tid)]]) for tid in ids])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path(
            "/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison-e10"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-queries", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1023)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Use a fresh isolated output directory")
    args.output_dir.mkdir(parents=True)
    start = time.time()
    base_dir = Path("/arc/home/gsm/aion_output/figures")
    prepared = json.loads((args.source_run / "prepared.json").read_text())
    source, source_meta = pilot.clauds_cohort(prepared["catalogue"])
    result = torch.load(
        args.source_run / "qlora/result.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = result["metadata"]["config"]
    val_rows = np.flatnonzero(source["split"] == "val")
    saved = result["val_evaluation"]
    if [str(x) for x in source["targetid"][val_rows]] != saved["object_id"]:
        raise ValueError(
            "Reconstructed CLAUDS validation ID order differs from the training run"
        )
    torch.testing.assert_close(
        torch.from_numpy(source["z"][val_rows]), saved["z_spec"], rtol=0, atol=0
    )
    print(
        f"Reconstructed original CLAUDS source cohort: {len(source['z']):,}; validation IDs and labels match.",
        flush=True,
    )
    output = dict(
        source_run=str(args.source_run),
        source_catalogue=source_meta,
        weights_updated=False,
        calibration_examples=0,
        n_queries=args.n_queries,
        seed=args.seed,
        source_config=config,
        source_exclusion="Entire original CLAUDS source cohort; physical HSC object IDs where available and 1-arcsec coordinates",
        targets={},
    )
    targets = {}
    for kind in ("galaxiesml-spec", "desi-ls"):
        target_run = base_dir / f"qwen-{kind}-qlora-specz-e10"
        data, meta = pilot.read_cohort(target_run)
        excluded = np.array([], dtype=np.int64)
        if kind == "galaxiesml-spec":
            source_groups = np.char.add("hsc:", source["targetid"].astype(str))
            excluded = data["targetid"][np.isin(data["group"], source_groups)]
        # Preserve target validation membership and the original source output grid.
        candidates = pilot.select_queries(
            data,
            excluded,
            args.n_queries * 2,
            args.seed,
            bounds=(config["z_min"], min(4.5, config["z_max"])),
        )
        positions = (
            hsc_positions(data["targetid"][candidates], meta)
            if kind == "galaxiesml-spec"
            else pilot.legacy_positions(data["targetid"][candidates], meta)
        )
        data = dict(
            data,
            ra=np.full(len(data["z"]), np.nan),
            dec=np.full(len(data["z"]), np.nan),
        )
        data["ra"][candidates], data["dec"][candidates] = positions.T
        selected, overlap = pilot.positional_filter(
            data, candidates, source["ra"], source["dec"], args.n_queries
        )
        small = pilot.take(data, selected)
        texts = pilot.native_texts(kind, small, meta)
        targets[kind] = (small, texts)
        output["targets"][kind] = dict(
            label=meta["target"],
            feature_names=meta["feature_names"],
            selected_indices=selected.tolist(),
            overlap_audit=overlap,
            exact_hsc_object_matches_excluded=len(excluded),
            split="validation",
            cohort_cache=(
                meta["cohort_cache"]
                if "cohort_cache" in meta
                else str(target_run / "prepared.json")
            ),
            redshift_range=[float(small["z"].min()), float(small["z"].max())],
            target_preprocessing="unchanged native observed fluxes, one-sigma errors, and EBV",
        )
        (args.output_dir / f"{kind}_inputs.json").write_text(
            json.dumps(
                dict(object_id=[str(x) for x in small["targetid"]], texts=texts),
                indent=2,
            )
        )
        np.savez_compressed(
            args.output_dir / f"{kind}_candidate_positions.npz",
            targetid=data["targetid"][candidates],
            ra=positions[:, 0],
            dec=positions[:, 1],
        )
        print(f"Prepared {kind}: {overlap}", flush=True)
    snapshot = args.output_dir / "source_snapshot"
    shutil.copytree(args.source_run / "qlora/adapter", snapshot / "adapter")
    shutil.copy2(args.source_run / "qlora/photoz_head.pt", snapshot / "photoz_head.pt")
    hashes = {
        str(f.relative_to(snapshot)): pilot.sha256(f)
        for f in snapshot.rglob("*")
        if f.is_file()
    }
    output["source_snapshot_sha256"] = hashes
    with pilot.reserve_one_gpu() as gpu:
        output["gpu"] = gpu
        set_random_seed(args.seed)
        from peft import PeftModel, prepare_model_for_kbit_training

        base, tokenizer = load_frozen_qwen(
            config["model_path"],
            device="cuda",
            load_in_4bit=True,
            torch_dtype="bf16",
            local_files_only=True,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        qwen = PeftModel.from_pretrained(base, snapshot / "adapter", is_trainable=False)
        model = QwenPhotoZModel(
            qwen,
            hidden_size=qwen_hidden_size(qwen),
            n_z_bins=config["n_z_bins"],
            head_hidden_dim=config["head_hidden_dim"],
            pooling=config["pooling"],
        ).to("cuda")
        model.photoz_head.load_state_dict(
            torch.load(
                snapshot / "photoz_head.pt", map_location="cpu", weights_only=True
            )
        )
        model.requires_grad_(False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        small = pilot.take(source, val_rows[:32])
        texts = pilot.native_texts("clauds", small, source_meta)
        pilot.audit_lengths(tokenizer, texts, config["max_length"])
        replay = pilot.evaluate(model, tokenizer, small, texts, config)
        reference = saved["logits"][:32]
        torch.testing.assert_close(replay["logits"], reference, rtol=1e-3, atol=0.02)
        pdf_error = float((replay["pz"] - saved["pz"][:32]).abs().max())
        if pdf_error > 0.005:
            raise ValueError(f"CLAUDS source replay differs: {pdf_error}")
        output["source_replay"] = dict(
            n=32,
            passed=True,
            max_abs_logit_error=float((replay["logits"] - reference).abs().max()),
            max_abs_pdf_error=pdf_error,
        )
        pilot.save_predictions(args.output_dir / "source_replay.npz", small, replay)
        print("Source replay passed:", output["source_replay"], flush=True)
        for kind, (data, texts) in targets.items():
            output["targets"][kind]["token_audit"] = pilot.audit_lengths(
                tokenizer, texts, config["max_length"]
            )
            before = time.time()
            evaluation = pilot.evaluate(model, tokenizer, data, texts, config)
            pilot.save_predictions(
                args.output_dir / f"{kind}_predictions.npz", data, evaluation
            )
            output["targets"][kind]["metrics"] = summarize_pdf_metrics(evaluation)
            output["targets"][kind]["inference_seconds"] = time.time() - before
            print(kind, json.dumps(output["targets"][kind]["metrics"]), flush=True)
            pilot.survey_pipeline.write_json(args.output_dir / "summary.json", output)
    if hashes != {
        str(f.relative_to(snapshot)): pilot.sha256(f)
        for f in snapshot.rglob("*")
        if f.is_file()
    }:
        raise ValueError("Inference artifacts changed during the pilot")
    output["elapsed_seconds"] = time.time() - start
    pilot.survey_pipeline.write_json(args.output_dir / "summary.json", output)


if __name__ == "__main__":
    main()
