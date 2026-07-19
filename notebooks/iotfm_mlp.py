#!/usr/bin/env python3
"""Compare IoTFM embeddings with an MLP using non-photo-z CLAUDS columns."""

from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion_magnitude.clauds_bands import OBJECT_ID_COLUMN, REDSHIFT_COLUMNS
from aion_magnitude.dataset import load_clauds_catalogue_from_fits, make_random_split
from aion_magnitude.Inference_Opt_TFM import (CatalogueSerializationConfig,
    InferenceOptimizedEmbeddingConfig, build_embedding_metadata,
    extract_text_embeddings, load_inference_optimized_transformer,
    serialize_catalogue_row)
from aion_magnitude.training import train_single_baseline

PHOTOZ_PREFIXES = ("ZPHOT", "Z_LOW68", "Z_HIGH68", "Z_CHI", "Z_PEAK", "Posterior-Log", "Likelihood-Log")
ID_COLUMNS = frozenset({"ID"})
LOCATION_COLUMNS = frozenset({"RA", "DEC", "tract", "patch"})


def max_rows_arg(value: str) -> int | None:
    if value.lower() in {"none", "all", "full"}: return None
    value = int(value)
    if value <= 0: raise argparse.ArgumentTypeError("max rows must be positive or 'none'")
    return value


def is_photoz_column(name: str) -> bool:
    return name != "Likelihood-Log_star" and name.startswith(PHOTOZ_PREFIXES)


def select_input_columns(names: Sequence[str], *, include_id=False, include_location=False):
    included, excluded = [], []
    for name in names:
        reject = (is_photoz_column(name) or
                  (name in ID_COLUMNS and not include_id) or
                  (name in LOCATION_COLUMNS and not include_location))
        (excluded if reject else included).append(name)
    return included, excluded


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalogue", type=Path, default=Path("data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits"))
    p.add_argument("--output-dir", type=Path, default=Path("/arc/home/gsm/aion_output/figures/iotfm_mlp"))
    p.add_argument("--cache-root", type=Path, default=Path("/scratch/.tmp-gsm/aion_output/cache/iotfm_mlp"))
    p.add_argument("--max-rows", type=max_rows_arg, default=200000)
    p.add_argument("--model", default="GLM-5.2-0.8B-A0.8B")
    p.add_argument("--embedding-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--pooling", choices=("mean", "last", "mean_last"), default="mean")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--force-recompute-embeddings", action="store_true")
    p.add_argument("--include-id", action="store_true")
    p.add_argument("--include-location", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--n-z-bins", type=int, default=300)
    p.add_argument("--train-fraction", type=float, default=.63)
    p.add_argument("--test-fraction", type=float, default=.32)
    p.add_argument("--val-fraction", type=float, default=.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return p


def serializable(value: Any) -> Any:
    if np.ma.is_masked(value): return None
    if isinstance(value, np.ndarray): return [serializable(v) for v in value.reshape(-1)]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, bytes): return value.decode("utf-8", errors="replace").strip()
    return value


def row_text(table, row, columns, config):
    return serialize_catalogue_row({name: serializable(table[name][row]) for name in columns}, config=config)


def build_mlp_features(table, rows, columns, splits):
    """Train-scaled numeric inputs plus a missing flag for every numeric field."""
    arrays, names, train = [], [], splits == "train"
    for column in columns:
        raw = np.asarray(table[column])[rows]
        if raw.dtype.kind not in "biufc":
            text = np.asarray([serializable(v) for v in raw], dtype=str)
            categories = sorted(set(text[train].tolist()))
            for category in categories:
                arrays.append((text == category).astype(np.float32)); names.append(f"{column}={category}")
            arrays.append((~np.isin(text, categories)).astype(np.float32)); names.append(f"{column}=UNKNOWN")
            continue
        matrix = raw.reshape(len(raw), -1)
        for index in range(matrix.shape[1]):
            values = np.asarray(matrix[:, index], dtype=np.float64)
            finite = np.isfinite(values); fitted = values[train & finite]
            low = float(fitted.min()) if fitted.size else 0.; high = float(fitted.max()) if fitted.size else low
            clean = np.zeros(len(values), dtype=np.float32)
            if high > low: clean[finite] = ((values[finite] - low) / (high - low)).astype(np.float32)
            suffix = "" if matrix.shape[1] == 1 else f"[{index}]"
            arrays += [clean, (~finite).astype(np.float32)]
            names += [column + suffix, column + suffix + "_missing"]
    return torch.from_numpy(np.stack(arrays, axis=1)), names


def get_embeddings(args, table, rows, ids, columns, serialization):
    config = InferenceOptimizedEmbeddingConfig(model_path=args.model, device=args.device,
        max_length=args.max_length, pooling=args.pooling, normalize=args.normalize,
        local_files_only=not args.allow_download, load_in_4bit=args.load_in_4bit)
    metadata = build_embedding_metadata(config, serialization)
    metadata.update(input_columns=columns, include_id=args.include_id, include_location=args.include_location)
    tag = Path(args.model).name.replace("-", "_")
    path = args.cache_root.expanduser() / f"{tag}_id{int(args.include_id)}_location{int(args.include_location)}_n{len(rows)}_{args.pooling}_len{args.max_length}.pt"
    if path.exists() and not args.force_recompute_embeddings:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("object_id") != ids or cached.get("metadata") != metadata:
            raise RuntimeError(f"Embedding cache differs from this run: {path}")
        return torch.as_tensor(cached["embedding"]).float(), path, metadata
    tokenizer, model, device = load_inference_optimized_transformer(config)
    parts = []
    try:
        for start in range(0, len(rows), args.embedding_batch_size):
            batch_rows = rows[start:start + args.embedding_batch_size]
            texts = [row_text(table, int(row), columns, serialization) for row in batch_rows]
            parts.append(extract_text_embeddings(texts, tokenizer, model, device,
                batch_size=args.embedding_batch_size, max_length=args.max_length,
                pooling=args.pooling, normalize=args.normalize))
            stop = min(start + len(batch_rows), len(rows))
            if stop == len(rows) or stop % max(1000, args.embedding_batch_size) == 0:
                print(f"IoTFM whole-catalogue embeddings: {stop:,}/{len(rows):,}")
    finally:
        del tokenizer, model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    embeddings = torch.cat(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"object_id": ids, "embedding": embeddings, "metadata": metadata}, path)
    return embeddings, path, metadata


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.embedding_batch_size, args.train_batch_size, args.eval_batch_size, args.epochs) <= 0:
        raise ValueError("batch sizes and epochs must be positive")
    catalogue, output = args.catalogue.expanduser().resolve(), args.output_dir.expanduser()
    split_dir = args.cache_root.expanduser() / ("all" if args.max_rows is None else f"n{args.max_rows}") / "clauds_split"
    table = load_clauds_catalogue_from_fits(catalogue, split_dir, max_rows=args.max_rows,
        sample_mode="random", sample_seed=args.seed, sample_require_valid_bands=())
    target_all = np.asarray(table[REDSHIFT_COLUMNS["zphot"]], dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(target_all)); target = target_all[rows]
    if not len(rows): raise ValueError("No finite photo-z targets")
    ids = np.asarray(table[OBJECT_ID_COLUMN])[rows].astype(np.int64).tolist()
    columns, excluded = select_input_columns(list(table.keys()), include_id=args.include_id, include_location=args.include_location)
    splits = make_random_split(len(rows), train_fraction=args.train_fraction,
        test_fraction=args.test_fraction, val_fraction=args.val_fraction, seed=args.seed)
    mlp_x, mlp_names = build_mlp_features(table, rows, columns, splits)
    serialization = CatalogueSerializationConfig(schema_name="clauds_whole_catalogue_v1",
        decimals=6, prefix="CLAUDS galaxy catalogue record")
    embeddings, cache_path, metadata = get_embeddings(args, table, rows, ids, columns, serialization)
    common = dict(object_id=ids, field=["catalogue"] * len(rows),
        z_spec=torch.from_numpy(target.astype(np.float32)),
        redshift_reference={"zphot": torch.from_numpy(target.astype(np.float32))},
        split_labels=splits.tolist(), metadata={**metadata, "excluded_columns": excluded,
        "mlp_feature_names": mlp_names, "finite_target_only": True})
    iotfm_product = {**common, "aion_embedding": embeddings, "extra_features": torch.empty((len(rows), 0))}
    mlp_product = {**common, "aion_embedding": torch.empty((len(rows), 0)), "extra_features": mlp_x}
    edges = torch.linspace(float(target.min()), float(target.max()), args.n_z_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    kwargs = dict(epochs=args.epochs, n_z_bins=args.n_z_bins, redshift_edges=edges,
        redshift_centers=centers, train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size, device=args.device)
    results = {"iotfm": train_single_baseline(iotfm_product, "aion", output_dir=output / "iotfm", **kwargs),
               "mlp": train_single_baseline(mlp_product, "tabular", output_dir=output / "mlp", **kwargs)}
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "iotfm_mlp_results.pt"; torch.save(results, summary)
    manifest = dict(catalogue=str(catalogue), model=args.model, n_rows=len(rows), input_columns=columns,
        excluded_columns=excluded, include_id=args.include_id, include_location=args.include_location,
        embedding_cache=str(cache_path), summary=str(summary))
    manifest_path = output / "iotfm_mlp_run.json"; manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"columns included={len(columns)} excluded={len(excluded)}")
    print(f"summary: {summary}\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
