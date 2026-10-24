#!/usr/bin/env python3
"""Isolated Level-0/1/2 Qwen experiments using existing cohorts and trainer."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import torch
from transformers import AutoTokenizer

from aion_magnitude import table_models as tm
from aion_magnitude.clauds_bands import ALL_BAND_FLUX_COLUMNS, ALL_BAND_ERROR_COLUMNS
from aion_magnitude.dataset import make_random_split
from aion_magnitude.filter_photometry import (
    DEFAULT_REGISTRY, SERIALIZER_VERSION, CLAUDS_BANDS, audit_tokens,
    clauds_flux_to_nanomaggy, load_registry, serialize_rows,
)
from aion_magnitude.FM_Qwen import resolve_qwen_model_path
from aion_magnitude.qwen_posttraining import QwenPosttrainingConfig, TextRedshiftDataset, train_qlora_photoz
import qwen_posttraining_comparison as clauds_pipeline
import qwen_zero_shot_sanity as pilot
import qwen_survey_posttraining as survey

RUN_ROOT = Path("/arc/home/gsm/aion_output/figures")


def sha_arrays(data):
    """Include values, order, dtypes, physical groups and split membership."""
    h = hashlib.sha256()
    for key in sorted(data):
        a = np.ascontiguousarray(data[key])
        if a.dtype.hasobject:
            raise ValueError(f"Portable cohort hashes require non-object arrays: {key}")
        h.update(key.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def read_clauds(source_run):
    """Reuse the archived catalogue, seed, selection, split and imputation."""
    import fitsio

    prepared = json.loads((source_run / "prepared.json").read_text())
    args = clauds_pipeline.build_parser().parse_args(["--stage", "prepare"])
    catalogue = Path(prepared["catalogue"])
    cat = tm.load_catalogue_data(catalogue, max_rows=args.max_rows, seed=args.seed,
                                include_full121=False, include_morphology=False,
                                z_min=-np.inf, z_max=np.inf)
    splits = make_random_split(len(cat.target), train_fraction=args.train_fraction,
                                val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed)
    features, imputation = tm.impute_from_training(cat.magnitude_features, splits)
    with fitsio.FITS(str(catalogue)) as f:
        columns = [ALL_BAND_FLUX_COLUMNS[b] for b in CLAUDS_BANDS] + [ALL_BAND_ERROR_COLUMNS[b] for b in CLAUDS_BANDS]
        raw = f[1].read(rows=cat.source_row, columns=columns)
    flux = np.column_stack([raw[ALL_BAND_FLUX_COLUMNS[b]] for b in CLAUDS_BANDS])
    error = np.column_stack([raw[ALL_BAND_ERROR_COLUMNS[b]] for b in CLAUDS_BANDS])
    common_f, common_e, valid = clauds_flux_to_nanomaggy(flux, error)
    data = dict(targetid=cat.object_id, group=cat.object_id.astype(str),
                split=np.asarray(splits, dtype="U5"), features=features.to_numpy(dtype=np.float32), z=cat.target,
                missing_mask=~np.isfinite(cat.magnitude_features.to_numpy()),
                common_flux=np.stack([common_f, common_e], axis=-1).astype(np.float32),
                raw_flux_valid=valid, source_row=cat.source_row)
    archived = torch.load(source_run / "qlora/result.pt", map_location="cpu", weights_only=False, mmap=True)
    val = splits == "val"
    if [str(x) for x in cat.object_id[val]] != archived["val_evaluation"]["object_id"]:
        raise ValueError("Reconstructed CLAUDS source validation membership/order changed")
    torch.testing.assert_close(torch.from_numpy(cat.target[val]), archived["val_evaluation"]["z_spec"], rtol=0, atol=0)
    config = archived["metadata"]["config"]
    return data, dict(feature_names=list(features.columns), target="ZPHOT", catalogue=str(catalogue),
                     imputation=imputation, extinction="as_loaded; not changed or inferred", training=config,
                     archived_validation_ids_labels_exact=True)


def read_source(kind):
    if kind == "clauds":
        return read_clauds(RUN_ROOT / "qwen-qwen_posttraining_comparison-e10")
    run = RUN_ROOT / f"qwen-{kind}-qlora-specz-e10"
    data, meta = pilot.read_cohort(run)
    data["missing_mask"] = ~np.isfinite(data["features"])
    return data, meta


def select_pilot(data, n_train, n_val, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for split, n in (("train", n_train), ("val", n_val)):
        eligible = np.flatnonzero(data["split"] == split)
        if len(eligible) < n:
            raise ValueError(f"Only {len(eligible)} {split} rows, requested {n}")
        indices.extend(rng.permutation(eligible)[:n])
    small = pilot.take(data, np.asarray(indices))
    small["split"] = np.asarray(small["split"], dtype="U5")
    if set(small["group"][small["split"] == "train"]) & set(small["group"][small["split"] == "val"]):
        raise ValueError("Physical object leakage between pilot train/validation")
    assert not np.any(small["split"] == "test")
    return small


def prepare(args):
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; existing experiments are immutable")
    registry = load_registry(args.registry, verify_curves=True)
    data, metadata = read_source(args.catalogue_kind)
    small = select_pilot(data, args.train_rows, args.val_rows, args.seed)
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "cohort.npz", **small)
    tokenizer = AutoTokenizer.from_pretrained(resolve_qwen_model_path(args.model), local_files_only=True)
    manifest = dict(catalogue=args.catalogue_kind, source_metadata=metadata,
                    serializer_version=SERIALIZER_VERSION, registry_version=registry["version"],
                    registry_path=str(args.registry.resolve()), registry_sha256=pilot.sha256(args.registry),
                    cohort_sha256=sha_arrays(small), cohort_file_sha256=pilot.sha256(args.output_dir / "cohort.npz"),
                    selection_seed=args.seed, train_rows=args.train_rows, val_rows=args.val_rows,
                    held_out_test_evaluated=False, model=args.model, max_seq_length=args.max_seq_length,
                    representations={})
    # Show one ordinary and one difficult actual row in every representation.
    difficult = np.flatnonzero(small["missing_mask"].any(axis=1))
    if not len(difficult) and args.catalogue_kind != "clauds":
        f = small["features"][:, :-1:2]; e = small["features"][:, 1:-1:2]
        difficult = np.flatnonzero(np.any(f <= e, axis=1))
    example_indices = list(dict.fromkeys([0] + difficult[:1].tolist()))
    for numerics in ("native", "flux"):
        for level in (0, 1, 2):
            if numerics == "flux" and level == 0:
                continue
            # Survey native inputs already are flux; no duplicate harmonized arm.
            if numerics == "flux" and args.catalogue_kind != "clauds":
                continue
            tag = f"level{level}-{numerics}"
            texts = serialize_rows(small, metadata, catalogue=args.catalogue_kind, level=level,
                                   numerics=numerics, registry=registry)
            audit = audit_tokens(tokenizer, texts, args.max_seq_length)
            payload = dict(cohort_sha256=manifest["cohort_sha256"], level=level, numerics=numerics, texts=texts,
                           object_id=[str(x) for x in small["targetid"]])
            survey.write_json(args.output_dir / f"{tag}.json", payload)
            examples = [dict(index=i, object_id=payload["object_id"][i], text=texts[i],
                             input_ids=tokenizer(texts[i], truncation=False)["input_ids"],
                             missing_mask=small["missing_mask"][i].tolist()) for i in example_indices]
            survey.write_json(args.output_dir / f"{tag}-examples.json", examples)
            manifest["representations"][tag] = dict(token_audit=audit, texts_sha256=pilot.sha256(args.output_dir / f"{tag}.json"))
            print(tag, json.dumps(audit), flush=True)
            for example in examples:
                print(f"ACTUAL TOKENIZER INPUT {tag} {example['object_id']}:\n{example['text']}", flush=True)
    # A high-quality small pilot can contain no missing measurements. Find a
    # separate training-only diagnostic row without changing the pilot cohort.
    candidates = np.flatnonzero((data["split"] == "train") & data["missing_mask"].any(axis=1))
    if len(candidates):
        difficult_data = pilot.take(data, candidates[:1])
        diagnostics = []
        for level in (0, 1, 2):
            text = serialize_rows(difficult_data, metadata, catalogue=args.catalogue_kind, level=level, registry=registry)[0]
            diagnostics.append(dict(level=level, source_split="train", object_id=str(difficult_data["targetid"][0]),
                                text=text, input_ids=tokenizer(text, truncation=False)["input_ids"],
                                missing_mask=difficult_data["missing_mask"][0].tolist()))
        survey.write_json(args.output_dir / "missing-record-examples.json", diagnostics)
    survey.write_json(args.output_dir / "manifest.json", manifest)


def load_prepared(folder, level, numerics):
    manifest = json.loads((folder / "manifest.json").read_text())
    with np.load(folder / "cohort.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    tag = f"level{level}-{numerics}"
    payload_path = folder / f"{tag}.json"
    payload = json.loads(payload_path.read_text())
    if (sha_arrays(data) != manifest["cohort_sha256"]
            or pilot.sha256(payload_path) != manifest["representations"][tag]["texts_sha256"]
            or payload["object_id"] != [str(x) for x in data["targetid"]]):
        raise ValueError("Prepared cohort/text/ID provenance mismatch")
    return data, manifest, payload["texts"]


def train(args):
    # Lock and require an actually idle slice, not merely enough remaining VRAM.
    with pilot.reserve_one_gpu() as gpu:
        if gpu["memory_before_gib"]["used"] > .2:
            raise RuntimeError("Pilot requires an idle GPU slice; existing workload detected")
        data, manifest, texts = load_prepared(args.output_dir, args.level, args.numerics)
        if pilot.sha256(manifest["registry_path"]) != manifest["registry_sha256"]:
            raise ValueError("Filter registry changed after preparation")
        tokenizer = AutoTokenizer.from_pretrained(resolve_qwen_model_path(manifest["model"]), local_files_only=True)
        audit_tokens(tokenizer, texts, manifest["max_seq_length"])
        def dataset(split):
            rows = np.flatnonzero(data["split"] == split)
            return TextRedshiftDataset([texts[i] for i in rows], data["z"][rows], [str(x) for x in data["targetid"][rows]])
        source_config = manifest["source_metadata"]["training"]
        config = QwenPosttrainingConfig(**source_config)
        config = replace(config, model_path=manifest["model"], max_length=manifest["max_seq_length"],
                         epochs=args.epochs, head_warmup_epochs=args.head_warmup_epochs,
                         seed=args.seed, eval_batch_size=args.eval_batch_size).normalized()
        output = args.output_dir / f"level{args.level}-{args.numerics}-{args.method}"
        if output.exists():
            raise FileExistsError("Use a fresh pilot output; no implicit overwrite/resume")
        output.mkdir()
        contract = dict(config=asdict(config), method=args.method, gpu=gpu,
                        serializer_version=manifest["serializer_version"], registry_version=manifest["registry_version"],
                        registry_sha256=manifest["registry_sha256"], cohort_sha256=manifest["cohort_sha256"],
                        level=args.level, numerics=args.numerics, test_evaluated=False)
        survey.write_json(output / "contract.json", contract)
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        result = train_qlora_photoz(train_dataset=dataset("train"), val_dataset=dataset("val"), test_dataset=None,
                    output_dir=output, config=config, resume=False, use_dora=args.method == "dora")
        result["metadata"]["filter_aware_contract"] = contract
        torch.save(result, output / "result.pt")
        survey.write_json(output / "summary.json", dict(contract, elapsed_seconds=time.time() - started,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                          history=result["history"], metrics=result["final_metrics"]))


def prepare_legacy_queries(args):
    """Resolve actual imaging releases for the already audited target queries."""
    import fitsio

    output = args.output_dir / "legacy-groups.json"
    if output.exists():
        raise FileExistsError(output)
    fast = ROOT / "fast-check-runs/1023/hsc-qlora-to-legacy-clauds"
    summary = json.loads((fast / "summary.json").read_text())
    with np.load(fast / "desi-ls_predictions.npz", allow_pickle=False) as a:
        data = dict(targetid=a["targetid"], features=a["features"], z=a["z_spec"])
    prepared = json.loads((RUN_ROOT / "qwen-desi-ls-qlora-specz-e10/prepared.json").read_text())
    groups = {}
    used_sources = []
    for source in prepared["sources"]:
        path = Path(source["path"])
        if not path.name.startswith("tractorphot-"):
            continue
        # Revalidate the exact source files used to construct the archived cohort.
        if path.stat().st_size != source["bytes"] or path.stat().st_mtime_ns != source["mtime_ns"]:
            raise ValueError(f"Legacy source fingerprint changed: {path}")
        with fitsio.FITS(str(path)) as f:
            ids = f[1].read(columns=["TARGETID"])["TARGETID"]
            rows = np.flatnonzero(np.isin(ids, data["targetid"]))
            if len(rows):
                matched = f[1].read(rows=rows, columns=["TARGETID", "RELEASE", "BRICKID", "OBJID", "EBV", "FLUX_G", "FLUX_R", "FLUX_Z", "FLUX_W1", "FLUX_W2"])
                for row in matched:
                    index = np.flatnonzero(data["targetid"] == row["TARGETID"])[0]
                    # Match the actual cached photometry, not just a repeated DESI ID.
                    actual = np.array([row[f"FLUX_{b}"] for b in ("G", "R", "Z", "W1", "W2")])
                    expected = data["features"][index, :-1:2]
                    available = np.isfinite(expected)
                    if not np.allclose(actual[available], expected[available], rtol=1e-6, atol=1e-6):
                        continue
                    key = str(int(row["TARGETID"]))
                    group = f"ls:{int(row['RELEASE'])}:{int(row['BRICKID'])}:{int(row['OBJID'])}"
                    if key in groups and groups[key] != group:
                        raise ValueError(f"Ambiguous physical filter mapping for {key}")
                    groups[key] = group
        used_sources.append(source)
        print(f"Resolved Legacy filter identities {len(groups)}/{len(data['targetid'])}", flush=True)
        if len(groups) == len(data["targetid"]):
            break
    if len(groups) != len(data["targetid"]):
        raise ValueError("Incomplete Legacy filter identification")
    data["group"] = np.asarray([groups[str(x)] for x in data["targetid"]])
    registry = load_registry(args.registry, verify_curves=True)
    tokenizer = AutoTokenizer.from_pretrained(resolve_qwen_model_path(args.model), local_files_only=True)
    audits = {}
    for level in (0, 1, 2):
        texts = serialize_rows(data, summary["targets"]["desi-ls"], catalogue="desi-ls", level=level, registry=registry)
        audits[f"level{level}"] = audit_tokens(tokenizer, texts, args.max_seq_length)
        survey.write_json(args.output_dir / f"legacy-query-level{level}.json", dict(texts=texts, object_id=[str(x) for x in data["targetid"]]))
    survey.write_json(output, dict(groups=groups, sources=used_sources, token_audits=audits,
                                  query_manifest_sha256=pilot.sha256(fast / "summary.json")))


def replay(args):
    """Check new Level-0 text against archived source logits on the GPU."""
    from peft import PeftModel, prepare_model_for_kbit_training
    from aion_magnitude.FM_Qwen import load_frozen_qwen
    from aion_magnitude.qwen_posttraining import QwenPhotoZModel, qwen_hidden_size

    fast = ROOT / "fast-check-runs/1023/hsc-qlora-to-legacy-clauds"
    summary = json.loads((fast / "summary.json").read_text())
    meta = json.loads((Path(summary["source_run"]) / "prepared.json").read_text())
    with np.load(fast / "source_replay.npz", allow_pickle=False) as a:
        data = dict(targetid=a["targetid"], features=a["features"], z=a["z_spec"])
        reference = torch.from_numpy(a["logits"])
    texts = serialize_rows(data, meta, catalogue="galaxiesml-spec", level=0)
    assert texts == pilot.native_texts("galaxiesml-spec", data, meta)
    config = summary["source_config"]
    snapshot = fast / "source_snapshot"
    for path, digest in summary["source_snapshot_sha256"].items():
        if pilot.sha256(snapshot / path) != digest:
            raise ValueError("Archived model snapshot checksum mismatch")
    with pilot.reserve_one_gpu() as gpu:
        if gpu["memory_before_gib"]["used"] > .2:
            raise RuntimeError("Replay requires an idle slice")
        base, tokenizer = load_frozen_qwen(config["model_path"], device="cuda", load_in_4bit=True, torch_dtype="bf16", local_files_only=True)
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        qwen = PeftModel.from_pretrained(base, snapshot / "adapter", is_trainable=False)
        model = QwenPhotoZModel(qwen, hidden_size=qwen_hidden_size(qwen), n_z_bins=config["n_z_bins"], head_hidden_dim=config["head_hidden_dim"], pooling="last").to("cuda")
        model.photoz_head.load_state_dict(torch.load(snapshot / "photoz_head.pt", map_location="cpu", weights_only=True))
        model.requires_grad_(False); model.eval()
        audit_tokens(tokenizer, texts, config["max_length"])
        evaluation = pilot.evaluate(model, tokenizer, data, texts, config)
        torch.testing.assert_close(evaluation["logits"], reference, rtol=1e-3, atol=.02)
        error = float((evaluation["logits"] - reference).abs().max())
        survey.write_json(args.output_dir / "level0-archived-replay.json", dict(n=32, max_abs_logit_error=error, passed=True, token_text_exact=True, weights_updated=False))
        print("Level-0 archived replay passed; max logit error", error, flush=True)


def evaluate_transfer(args):
    """Reuse the audited, source-excluded queries from the existing fast-check."""
    from peft import PeftModel, prepare_model_for_kbit_training
    from aion_magnitude.FM_Qwen import load_frozen_qwen
    from aion_magnitude.qwen_posttraining import QwenPhotoZModel, qwen_hidden_size
    from aion_magnitude.metrics import summarize_pdf_metrics

    model_dir = args.output_dir / f"level{args.level}-{args.numerics}-{args.method}"
    contract = json.loads((model_dir / "contract.json").read_text())
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    if manifest["catalogue"] != "galaxiesml-spec":
        raise ValueError("Initial transfer pilot supports HSC source only; other sources need their own overlap-audited query manifest")
    if args.target_numerics == "flux" and args.level == 0:
        raise ValueError("Level 0 preserves native text; harmonized targets require Level 1/2")
    output = model_dir / ("transfer" if args.target_numerics == "native" else "transfer-flux")
    if output.exists():
        raise FileExistsError(output)
    registry = load_registry(manifest["registry_path"], verify_curves=True)
    if pilot.sha256(manifest["registry_path"]) != manifest["registry_sha256"]:
        raise ValueError("Registry changed since training")
    fast = ROOT / "fast-check-runs/1023/hsc-qlora-to-legacy-clauds"
    query_manifest = json.loads((fast / "summary.json").read_text())
    group_path = args.legacy_groups or args.output_dir / "legacy-groups.json"
    if args.level and not group_path.exists():
        raise ValueError("Run --stage legacy-groups before Level-1/2 transfer")
    # Entire archived source cohort was excluded in fast-check; pilot training is
    # strictly a subset of that same source cohort.
    if query_manifest["source_run"] != str(RUN_ROOT / "qwen-galaxiesml-spec-qlora-specz-e10"):
        raise ValueError("Source overlap provenance mismatch")
    clauds_data = None
    if args.target_numerics == "flux":
        clauds_data, _ = read_source("clauds")
    with pilot.reserve_one_gpu() as gpu:
        if gpu["memory_before_gib"]["used"] > .2:
            raise RuntimeError("Transfer requires an idle GPU slice")
        base, tokenizer = load_frozen_qwen(manifest["model"], device="cuda", load_in_4bit=True, torch_dtype="bf16", local_files_only=True)
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        qwen = PeftModel.from_pretrained(base, model_dir / "adapter", is_trainable=False)
        config = contract["config"]
        model = QwenPhotoZModel(qwen, hidden_size=qwen_hidden_size(qwen), n_z_bins=config["n_z_bins"], head_hidden_dim=config["head_hidden_dim"], pooling="last").to("cuda")
        model.photoz_head.load_state_dict(torch.load(model_dir / "photoz_head.pt", weights_only=True, map_location="cpu"))
        model.requires_grad_(False); model.eval()
        output.mkdir()
        summaries = {}
        for kind in ("desi-ls", "clauds"):
            with np.load(fast / f"{kind}_predictions.npz", allow_pickle=False) as a:
                data = dict(targetid=a["targetid"], features=a["features"], z=a["z_spec"])
            meta = query_manifest["targets"][kind]
            if kind == "desi-ls" and args.level:
                # Resolve the real release from Tractor records, never map all LS to DECam.
                group_manifest = json.loads(group_path.read_text())
                if group_manifest["query_manifest_sha256"] != pilot.sha256(fast / "summary.json"):
                    raise ValueError("Legacy groups belong to different queries")
                groups = group_manifest["groups"]
                data["group"] = np.asarray([groups[str(x)] for x in data["targetid"]])
            if kind == "clauds" and args.target_numerics == "flux":
                lookup = {int(x):i for i,x in enumerate(clauds_data["targetid"])}
                rows = [lookup[int(x)] for x in data["targetid"]]
                np.testing.assert_array_equal(clauds_data["features"][rows], data["features"])
                np.testing.assert_array_equal(clauds_data["z"][rows], data["z"])
                data["common_flux"] = clauds_data["common_flux"][rows]
            texts = serialize_rows(data, meta, catalogue=kind, level=args.level, numerics=args.target_numerics, registry=registry)
            audit = audit_tokens(tokenizer, texts, manifest["max_seq_length"])
            survey.write_json(output / f"{kind}_inputs.json", dict(object_id=[str(x) for x in data["targetid"]], texts=texts,
                               numerics=args.target_numerics, serializer_version=SERIALIZER_VERSION))
            if "common_flux" in data:
                np.savez_compressed(output / f"{kind}_common_flux.npz", targetid=data["targetid"], common_flux=data["common_flux"])
            evaluation = pilot.evaluate(model, tokenizer, data, texts, config)
            pilot.save_predictions(output / f"{kind}_predictions.npz", data, evaluation)
            summaries[kind] = dict(n=len(texts), label=meta["label"], numerics=args.target_numerics,
                                   metrics=summarize_pdf_metrics(evaluation), token_audit=audit,
                                   overlap_audit=meta["overlap_audit"], query_ids=[str(x) for x in data["targetid"]])
            print(kind, summaries[kind]["metrics"], flush=True)
        survey.write_json(output / "summary.json", dict(weights_updated=False, target_labels_fitted=False, targets=summaries,
                    query_manifest_sha256=pilot.sha256(fast / "summary.json"), contract=contract))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "train", "transfer", "legacy-groups", "replay"), required=True)
    p.add_argument("--catalogue-kind", choices=("galaxiesml-spec", "desi-ls", "clauds"), default="galaxiesml-spec")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--model", default="Qwen3.5-4B-Base")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--train-rows", type=int, default=128)
    p.add_argument("--val-rows", type=int, default=64)
    p.add_argument("--seed", type=int, default=1023)
    p.add_argument("--level", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--numerics", choices=("native", "flux"), default="native")
    p.add_argument("--target-numerics", choices=("native", "flux"), default="native", help="Independent transfer ablation; does not change source training")
    p.add_argument("--method", choices=("qlora", "dora"), default="qlora")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--head-warmup-epochs", type=int, default=3)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--legacy-groups", type=Path)
    args = p.parse_args()
    if args.train_rows < 1 or args.val_rows < 1:
        p.error("Positive train/validation sizes required")
    {"prepare": prepare, "train": train, "transfer": evaluate_transfer,
     "legacy-groups": prepare_legacy_queries, "replay": replay}[args.stage](args)


if __name__ == "__main__":
    main()
