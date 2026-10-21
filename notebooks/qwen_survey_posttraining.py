#!/usr/bin/env python3
"""Independent native-photometry Qwen QLoRA/DoRA comparisons with spec-z targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion_magnitude import survey_catalogues as surveys
from aion_magnitude.FM_Qwen import (
    QwenEmbeddingConfig,
    QwenSerializationConfig,
    extract_qwen_embeddings_from_texts,
    load_frozen_qwen,
    resolve_qwen_model_path,
    serialize_qwen_feature_row,
)
from aion_magnitude.morphology import save_morphology_comparison_artifacts
from aion_magnitude.qwen_posttraining import (
    QwenPosttrainingConfig,
    TextRedshiftDataset,
    train_qlora_photoz,
)
from aion_magnitude.training import train_single_baseline
from aion_magnitude.utils import make_redshift_grid, set_random_seed


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage", choices=("prepare", "frozen", "adapter", "collect"), required=True
    )
    p.add_argument("--catalogue-kind", choices=tuple(surveys.BANDS), required=True)
    p.add_argument("--method", choices=("qlora", "dora"), required=True)
    p.add_argument("--desi-root", type=Path, default=surveys.DEFAULT_DESI_ROOT)
    p.add_argument("--hsc-file", type=Path, default=surveys.DEFAULT_HSC_FILE)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/scratch/.tmp-gsm/aion_output/cache/qwen-surveys"),
    )
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument(
        "--max-rows",
        type=lambda x: None if x.lower() == "all" else int(x),
        default=300000,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=0.20)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=4.5)
    p.add_argument("--min-deltachi2", type=float, default=40.0)
    p.add_argument("--min-valid-bands", type=int, default=3)
    p.add_argument("--n-z-bins", type=int, default=300)
    p.add_argument("--qwen-model", default="Qwen3.5-4B-Base")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--head-warmup-epochs", type=int, default=3)
    p.add_argument("--head-learning-rate", type=float, default=2e-4)
    p.add_argument("--adapter-learning-rate", type=float, default=1e-5)
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--embedding-batch-size", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--checkpoint-steps", type=int, default=100)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument(
        "--skip-tokenizer-audit",
        action="store_true",
        help="CPU preparation audit only; training still requires a passing token-length audit",
    )
    p.add_argument("--tomographic-samples", type=int, default=100)
    return p


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def cohort_settings(args):
    return dict(
        kind=args.catalogue_kind,
        desi_root=args.desi_root,
        hsc_file=args.hsc_file,
        max_rows=args.max_rows,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        z_min=args.z_min,
        z_max=args.z_max,
        min_deltachi2=args.min_deltachi2,
        min_valid_bands=args.min_valid_bands,
    )


def cache_paths(args):
    settings = cohort_settings(args)
    # Irrelevant source paths do not couple independently trained catalogues.
    if args.catalogue_kind == "desi-ls":
        settings.pop("hsc_file")
    key = digest(
        {
            "settings": settings,
            "schema": surveys.SCHEMA_VERSION,
            "code": hashlib.sha256(Path(surveys.__file__).read_bytes()).hexdigest(),
            "sources": surveys.fingerprints(
                surveys.source_paths(args.catalogue_kind, args.desi_root, args.hsc_file)
            ),
        }
    )
    folder = args.cache_root / args.catalogue_kind / key[:24]
    return folder, key


def load_cohort(args, *, build=False):
    folder, key = cache_paths(args)
    folder.mkdir(parents=True, exist_ok=True)
    path, meta_path = folder / "cohort.npz", folder / "cohort.json"
    with (folder / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists() or not meta_path.exists():
            if not build:
                raise FileNotFoundError("Run --stage prepare first")
            data, metadata = surveys.build_cohort(**cohort_settings(args))
            metadata["cache_key"] = key
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(stream, **data)
            tmp.replace(path)
            write_json(meta_path, metadata)
        metadata = json.loads(meta_path.read_text())
        if metadata["cache_key"] != key:
            raise ValueError("Cohort provenance does not match")
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
    return data, metadata, folder


def serialization(args):
    return QwenSerializationConfig(
        schema_name=f"{args.catalogue_kind}_{surveys.SCHEMA_VERSION}",
        prefix="galaxy native_observed_photometry flux_unit=AB_nanomaggy errors=one_sigma extinction=as_catalogued",
        include_hsc_grizy=False,
        include_object_metadata=False,
    )


def texts_for(args, data, metadata):
    config = serialization(args)
    names = metadata["feature_names"]
    return [
        serialize_qwen_feature_row(dict(zip(names, row)), serialization=config)
        for row in data["features"]
    ]


def training_config(args):
    return QwenPosttrainingConfig(
        model_path=args.qwen_model,
        max_length=args.max_length,
        pooling="last",
        n_z_bins=args.n_z_bins,
        z_min=args.z_min,
        z_max=args.z_max,
        epochs=args.epochs,
        batch_size=args.microbatch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.head_learning_rate,
        lora_learning_rate=args.adapter_learning_rate,
        head_warmup_epochs=args.head_warmup_epochs,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        lora_target_modules=args.target_modules,
        lora_max_grad_norm=0.1,
        seed=args.seed,
        local_files_only=not args.allow_download,
        device="cuda",
    ).normalized()


def model_source(args):
    resolved = Path(str(resolve_qwen_model_path(args.qwen_model))).expanduser()
    if not resolved.is_dir():
        return {"model": str(resolved)}
    files = sorted(
        set(resolved.glob("*.json"))
        | set(resolved.glob("*.safetensors"))
        | set(resolved.glob("*.txt"))
    )
    return {"model": str(resolved.resolve()), "files": surveys.fingerprints(files)}


def run_contract(args, metadata):
    return {
        "catalogue_kind": args.catalogue_kind,
        "method": args.method,
        "cohort_cache_key": metadata["cache_key"],
        "training": asdict(training_config(args)),
        "model_source": model_source(args),
        "serialization": asdict(serialization(args)),
        "frozen_representation": "kbit_prepared_padding_safe_last_v1",
        "frozen_head_batch_size": args.microbatch_size
        * args.gradient_accumulation_steps,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
    }


def assert_contract(args, metadata, *, create=False):
    path = args.output_dir / "run_contract.json"
    contract = json.loads(json.dumps(run_contract(args, metadata), default=str))
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError(
                "Output directory belongs to a different cohort/training configuration; choose a new --output-dir"
            )
    elif create:
        write_json(path, contract)
    else:
        raise FileNotFoundError(
            "Missing run contract; prepare this output directory first"
        )


def token_audit(args, texts, folder):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_qwen_model_path(args.qwen_model),
        local_files_only=not args.allow_download,
        trust_remote_code=True,
    )
    longest = 0
    for start in range(0, len(texts), 512):
        encoded = tokenizer(texts[start : start + 512], truncation=False, padding=False)
        longest = max(longest, max(map(len, encoded["input_ids"])))
    audit = {
        "model": str(resolve_qwen_model_path(args.qwen_model)),
        "max_length": args.max_length,
        "longest_input_tokens": longest,
        "rows_checked": len(texts),
        "passed": longest <= args.max_length,
    }
    write_json(folder / f"token_audit_{digest(audit)[:16]}.json", audit)
    if not audit["passed"]:
        raise ValueError(
            f"Photometry rows need {longest} tokens; increase --max-length above {args.max_length}"
        )
    return audit


def prepare(args):
    data, metadata, folder = load_cohort(args, build=True)
    assert_contract(args, metadata, create=True)
    summary = dict(
        metadata,
        cohort_cache=str(folder),
        method=args.method,
        model=args.qwen_model,
        pooling="last",
        training=asdict(training_config(args)),
    )
    if not args.skip_tokenizer_audit:
        summary["token_audit"] = token_audit(
            args, texts_for(args, data, metadata), folder
        )
    write_json(args.output_dir / "prepared.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


def require_prepared(args, metadata):
    assert_contract(args, metadata)
    summary = json.loads((args.output_dir / "prepared.json").read_text())
    audit = summary.get("token_audit", {})
    if not audit.get("passed") or audit.get("max_length") != args.max_length:
        raise ValueError(
            "Training requires a passing tokenizer audit; rerun prepare without --skip-tokenizer-audit"
        )


def frozen(args, data, metadata, folder):
    texts = texts_for(args, data, metadata)
    identity = {
        "cohort": metadata["cache_key"],
        "model": args.qwen_model,
        "model_source": model_source(args),
        "max_length": args.max_length,
        "serialization": asdict(serialization(args)),
        "state": "kbit_prepared_padding_safe_last_v1",
    }
    cache = folder / f"frozen_{digest(identity)[:24]}.pt"
    # Different methods may prepare the same frozen cache concurrently.
    with cache.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            saved = torch.load(cache, map_location="cpu", weights_only=False)
            if (
                saved["identity"] != identity
                or saved["object_id"] != data["targetid"].tolist()
            ):
                raise ValueError("Frozen cache provenance or ID ordering mismatch")
            embeddings = saved["embedding"]
        else:
            from peft import prepare_model_for_kbit_training

            model, tokenizer = load_frozen_qwen(
                args.qwen_model,
                device="cuda",
                load_in_4bit=True,
                local_files_only=not args.allow_download,
            )
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=False
            )
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            try:
                embeddings = extract_qwen_embeddings_from_texts(
                    texts,
                    model,
                    tokenizer,
                    device=torch.device("cuda"),
                    max_length=args.max_length,
                    batch_size=args.embedding_batch_size,
                    pooling="last",
                    normalize=False,
                )
            finally:
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
            tmp = cache.with_name(cache.name + ".tmp")
            torch.save(
                {
                    "embedding": embeddings,
                    "identity": identity,
                    "object_id": data["targetid"].tolist(),
                },
                tmp,
            )
            tmp.replace(cache)
    product = {
        "object_id": data["targetid"].tolist(),
        "field": [args.catalogue_kind] * len(texts),
        "aion_embedding": embeddings,
        "extra_features": torch.zeros((len(texts), 0)),
        "feature_names": [],
        "z_spec": torch.from_numpy(data["z"]),
        "split_labels": data["split"].tolist(),
        "metadata": metadata,
    }
    edges, centers = make_redshift_grid(args.z_min, args.z_max, args.n_z_bins)
    set_random_seed(args.seed)
    result = train_single_baseline(
        product,
        "aion",
        output_dir=args.output_dir / "frozen",
        n_z_bins=args.n_z_bins,
        redshift_edges=edges,
        redshift_centers=centers,
        epochs=args.epochs,
        learning_rate=args.head_learning_rate,
        weight_decay=0.01,
        train_batch_size=args.microbatch_size * args.gradient_accumulation_steps,
        eval_batch_size=max(args.eval_batch_size, 64),
        device="cuda",
    )
    result.setdefault("metadata", {})["survey_cohort_cache_key"] = metadata["cache_key"]
    torch.save(result, args.output_dir / "frozen/result.pt")


def adapter(args, data, metadata):
    texts = texts_for(args, data, metadata)

    def split(name):
        rows = np.flatnonzero(data["split"] == name)
        return TextRedshiftDataset(
            [texts[i] for i in rows],
            data["z"][rows],
            [str(x) for x in data["targetid"][rows]],
        )

    result = train_qlora_photoz(
        train_dataset=split("train"),
        val_dataset=split("val"),
        test_dataset=split("test"),
        output_dir=args.output_dir / args.method,
        config=training_config(args),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_steps,
        resume=args.resume,
        use_dora=args.method == "dora",
    )
    result.setdefault("metadata", {})["survey_cohort_cache_key"] = metadata["cache_key"]
    torch.save(result, args.output_dir / args.method / "result.pt")


def collect(args, metadata):
    results = {
        kind: torch.load(
            args.output_dir / kind / "result.pt", map_location="cpu", weights_only=False
        )
        for kind in ("frozen", args.method)
    }
    for result in results.values():
        if result["metadata"].get("survey_cohort_cache_key") != metadata["cache_key"]:
            raise ValueError("Result belongs to a different catalogue cohort")
    artifacts = save_morphology_comparison_artifacts(
        results,
        model_kinds=("frozen", args.method),
        output_dir=args.output_dir,
        tomographic_samples=args.tomographic_samples,
        comparison_labels=("frozen-Qwen+head", f"{args.method.upper()}-Qwen+head"),
        comparison_prefix=args.output_dir / f"qwen_{args.catalogue_kind}_{args.method}",
    )
    write_json(
        args.output_dir / "run.json",
        {
            "catalogue": metadata,
            "contract": run_contract(args, metadata),
            "final_metrics": {kind: r["final_metrics"] for kind, r in results.items()},
            "artifacts": artifacts,
        },
    )
    print(
        json.dumps({kind: r["final_metrics"] for kind, r in results.items()}, indent=2),
        flush=True,
    )


def main(argv=None):
    args = parser().parse_args(argv)
    training_config(args)
    if args.stage == "prepare":
        prepare(args)
        return 0
    data, metadata, folder = load_cohort(args)
    require_prepared(args, metadata)
    if args.stage == "frozen":
        frozen(args, data, metadata, folder)
    elif args.stage == "adapter":
        adapter(args, data, metadata)
    else:
        collect(args, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
