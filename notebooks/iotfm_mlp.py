#!/usr/bin/env python3
"""Compare IoTFM embeddings with an MLP using non-photo-z CLAUDS columns.

IoTFM means inference-optimized transformer feature mapping.
"""

from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion_magnitude.clauds_bands import ALL_BAND_FLUX_COLUMNS, OBJECT_ID_COLUMN, REDSHIFT_COLUMNS
from aion_magnitude.dataset import dataset_for_split, load_clauds_catalogue_from_fits, make_random_split
from aion_magnitude.Inference_Opt_TFM import (CatalogueSerializationConfig,
    InferenceOptimizedEmbeddingConfig, build_embedding_metadata,
    extract_text_embeddings, load_inference_optimized_transformer,
    serialize_catalogue_row)
from aion_magnitude.models import load_baseline_model_from_checkpoint
from aion_magnitude.plotting import (compare_config_loss, compare_nz_lensing_alike,
    compare_pit_histogram, compare_redshift_probability_distribution,
    compare_zpred_vs_zphot)
from aion_magnitude.training import evaluate_model_on_dataset, train_single_baseline
from aion_magnitude.utils import flux_to_ab_mag, resolve_torch_device

PHOTOZ_PREFIXES = ("ZPHOT", "Z_LOW68", "Z_HIGH68", "Z_CHI", "Z_PEAK", "Posterior-Log", "Likelihood-Log")
ID_COLUMNS = frozenset({"ID"})
LOCATION_COLUMNS = frozenset({"RA", "DEC", "tract", "patch"})
MISSINGNESS_COLUMN_PREFIXES = ("isNoData_", "notObserved_")
MAGNITUDE_ONLY_BANDS = ("u", "u_star", "g", "r", "i", "z", "y", "Y", "J", "H", "Ks")
MAG_ZERO_POINT = 23.0


def max_rows_arg(value: str) -> int | None:
    if value.lower() in {"none", "all", "full"}: return None
    value = int(value)
    if value <= 0: raise argparse.ArgumentTypeError("max rows must be positive or 'none'")
    return value


def is_photoz_column(name: str) -> bool:
    return name != "Likelihood-Log_star" and name.startswith(PHOTOZ_PREFIXES)


def select_input_columns(names: Sequence[str], *, include_id=False, include_location=False,
                         ignore_missingness=False):
    included, excluded = [], []
    for name in names:
        reject = (is_photoz_column(name) or
                  (name in ID_COLUMNS and not include_id) or
                  (name in LOCATION_COLUMNS and not include_location) or
                  (ignore_missingness and name.startswith(MISSINGNESS_COLUMN_PREFIXES)))
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
    p.add_argument("--magnitudes-only", action="store_true")
    p.add_argument("--ignore-missingness", action="store_true",
                   help="omit missing fields from transformer text and median-impute the MLP without missing flags")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--tomographic-samples", type=int, default=100)
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


def is_missing(value: Any) -> bool:
    """Return whether a catalogue value contains no usable observation."""
    if np.ma.is_masked(value) or value is None:
        return True
    if isinstance(value, np.ndarray):
        return any(is_missing(item) for item in value.reshape(-1))
    if isinstance(value, np.generic):
        return is_missing(value.item())
    return isinstance(value, (float, complex)) and not np.isfinite(value)


def row_text(table, row, columns, config, *, omit_missing=False):
    values = {name: serializable(table[name][row]) for name in columns}
    if omit_missing:
        values = {name: value for name, value in values.items() if not is_missing(value)}
    return serialize_catalogue_row(values, config=config)


def build_magnitude_input_table(table):
    """Return only the requested eleven CLAUDS AB-magnitude columns."""
    inputs = {}
    for band in MAGNITUDE_ONLY_BANDS:
        magnitude, _ = flux_to_ab_mag(
            table[ALL_BAND_FLUX_COLUMNS[band]], mag_zero_point=MAG_ZERO_POINT
        )
        inputs[f"{band}_mag"] = magnitude
    return inputs


def build_mlp_features(table, rows, columns, splits, *, encode_missingness=True):
    """Build train-scaled inputs, optionally exposing explicit missing flags."""
    arrays, names, train = [], [], splits == "train"
    for column in columns:
        raw = np.ma.asarray(table[column])[rows]
        raw_values = np.asarray(np.ma.getdata(raw))
        if raw_values.dtype.kind not in "biufc":
            text = np.asarray([serializable(v) for v in raw], dtype=str)
            if encode_missingness:
                categories = sorted(set(text[train].tolist()))
                observed = np.ones(len(text), dtype=bool)
            else:
                mask = np.ma.getmaskarray(raw).reshape(len(raw), -1).any(axis=1)
                missing = mask | np.asarray([is_missing(v) for v in raw])
                categories = sorted(set(text[train & ~missing].tolist()))
                observed = ~missing
            for category in categories:
                arrays.append((observed & (text == category)).astype(np.float32))
                names.append(f"{column}={category}")
            arrays.append((observed & ~np.isin(text, categories)).astype(np.float32))
            names.append(f"{column}=UNKNOWN")
            continue
        matrix = raw_values.reshape(len(raw), -1)
        masked = np.ma.getmaskarray(raw).reshape(len(raw), -1)
        for index in range(matrix.shape[1]):
            values = np.asarray(matrix[:, index], dtype=np.float64)
            finite = np.isfinite(values) & ~masked[:, index]
            fitted = values[train & finite]
            low = float(fitted.min()) if fitted.size else 0.; high = float(fitted.max()) if fitted.size else low
            clean = np.zeros(len(values), dtype=np.float32)
            if high > low:
                clean[finite] = ((values[finite] - low) / (high - low)).astype(np.float32)
                if not encode_missingness:
                    median = float(np.median(fitted))
                    clean[~finite] = np.float32((median - low) / (high - low))
            suffix = "" if matrix.shape[1] == 1 else f"[{index}]"
            arrays.append(clean); names.append(column + suffix)
            if encode_missingness:
                arrays.append((~finite).astype(np.float32))
                names.append(column + suffix + "_missing")
    return torch.from_numpy(np.stack(arrays, axis=1)), names


def get_embeddings(args, table, rows, ids, columns, serialization):
    config = InferenceOptimizedEmbeddingConfig(model_path=args.model, device=args.device,
        max_length=args.max_length, pooling=args.pooling, normalize=args.normalize,
        local_files_only=not args.allow_download, load_in_4bit=args.load_in_4bit)
    metadata = build_embedding_metadata(config, serialization)
    metadata.update(input_columns=columns, include_id=args.include_id,
        include_location=args.include_location)
    if args.magnitudes_only:
        metadata["input_mode"] = "magnitudes_only"
    if args.ignore_missingness:
        metadata["missing_value_policy"] = "omit_from_text_train_median_impute_no_indicator"
    tag = Path(args.model).name.replace("-", "_")
    mode_tag = (("_magonly" if args.magnitudes_only else "") +
                ("_missnormal" if args.ignore_missingness else ""))
    path = args.cache_root.expanduser() / f"{tag}{mode_tag}_id{int(args.include_id)}_location{int(args.include_location)}_n{len(rows)}_{args.pooling}_len{args.max_length}.pt"
    if path.exists() and not args.force_recompute_embeddings:
        print(f"Frozen transformer embeddings: loading cache {path}")
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("object_id") != ids or cached.get("metadata") != metadata:
            raise RuntimeError(f"Embedding cache differs from this run: {path}")
        return torch.as_tensor(cached["embedding"]).float(), path, metadata
    print(f"Loading frozen transformer: {metadata['resolved_model_path']}")
    print(f"Requested device: {args.device}; AION model/codec: not used")
    tokenizer, model, device = load_inference_optimized_transformer(config)
    print(f"Frozen transformer device: {device}; extracting catalogue-row embeddings")
    parts = []
    try:
        for start in range(0, len(rows), args.embedding_batch_size):
            batch_rows = rows[start:start + args.embedding_batch_size]
            texts = [row_text(table, int(row), columns, serialization,
                              omit_missing=args.ignore_missingness)
                     for row in batch_rows]
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


def evaluate_trained_branch(product, result, model_kind, split, args, edges, centers):
    """Reload one best checkpoint and return CPU evaluation tensors for plotting."""
    device = resolve_torch_device(args.device)
    model = load_baseline_model_from_checkpoint(result["checkpoint_path"],
        model_kind=model_kind, aion_dim=product["aion_embedding"].shape[1],
        extra_feature_dim=product["extra_features"].shape[1],
        n_z_bins=args.n_z_bins, device=device)
    try:
        return evaluate_model_on_dataset(model, dataset_for_split(product, split), model_kind,
            batch_size=args.eval_batch_size, device=device,
            redshift_edges=edges, redshift_centers=centers)
    finally:
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def save_comparison_artifacts(results, iotfm_product, mlp_product, split, args, edges, centers, output):
    """Save the paired diagnostics used by the repository's other comparisons."""
    labels = (f"frozen transformer ({Path(args.model).name})", "tabular MLP")
    prefix = output / "iotfm_mlp_comparison"
    artifacts = {name: str(Path(f"{prefix}_{suffix}.jpeg")) for name, suffix in {
        "loss": "loss", "scatter": "scatter", "pit": "pit",
        "nz": "nz", "nztomo": "nztomo"}.items()}
    iotfm_eval = evaluate_trained_branch(iotfm_product, results["iotfm"], "iotfm",
        split, args, edges, centers)
    mlp_eval = evaluate_trained_branch(mlp_product, results["mlp"], "tabular",
        split, args, edges, centers)
    fig, _, _ = compare_config_loss(results["iotfm"], results["mlp"],
        output_path=artifacts["loss"], labels=labels)
    plt.close(fig)
    fig, _ = compare_zpred_vs_zphot(iotfm_eval, mlp_eval,
        output_path=artifacts["scatter"], labels=labels, pred_key="z_p50",
        target_label="ZPHOT", pmax=5.0, show_metrics=True)
    plt.close(fig)
    fig, _ = compare_pit_histogram(iotfm_eval, mlp_eval,
        output_path=artifacts["pit"], labels=labels)
    plt.close(fig)
    fig, _, _ = compare_redshift_probability_distribution(iotfm_eval, mlp_eval,
        output_path=artifacts["nz"], labels=labels, gaussian_sigma_bins=1.0)
    plt.close(fig)
    fig, _, _ = compare_nz_lensing_alike(iotfm_eval, mlp_eval,
        output_path=artifacts["nztomo"], labels=labels,
        zphot_bin=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], gaussian_sigma_bins=1.0,
        inferred_bin_key="z_p50", n_samples_per_object=args.tomographic_samples)
    plt.close(fig)
    return artifacts


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.embedding_batch_size, args.train_batch_size, args.eval_batch_size,
           args.tomographic_samples, args.epochs) <= 0:
        raise ValueError("batch sizes and epochs must be positive")
    catalogue, output = args.catalogue.expanduser().resolve(), args.output_dir.expanduser()
    split_dir = args.cache_root.expanduser() / ("all" if args.max_rows is None else f"n{args.max_rows}") / "clauds_split"
    table = load_clauds_catalogue_from_fits(catalogue, split_dir, max_rows=args.max_rows,
        sample_mode="random", sample_seed=args.seed, sample_require_valid_bands=())
    target_all = np.asarray(table[REDSHIFT_COLUMNS["zphot"]], dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(target_all)); target = target_all[rows]
    if not len(rows): raise ValueError("No finite photo-z targets")
    ids = np.asarray(table[OBJECT_ID_COLUMN])[rows].astype(np.int64).tolist()
    splits = make_random_split(len(rows), train_fraction=args.train_fraction,
        test_fraction=args.test_fraction, val_fraction=args.val_fraction, seed=args.seed)
    if args.magnitudes_only:
        input_table = build_magnitude_input_table(table)
        columns = list(input_table)
        source_columns = [ALL_BAND_FLUX_COLUMNS[band] for band in MAGNITUDE_ONLY_BANDS]
        excluded = [name for name in table.keys() if name not in source_columns]
        serialization = CatalogueSerializationConfig(schema_name="clauds_magnitudes_only_v1",
            decimals=6, prefix="CLAUDS galaxy AB magnitudes")
    else:
        input_table = table
        columns, excluded = select_input_columns(list(table.keys()),
            include_id=args.include_id, include_location=args.include_location,
            ignore_missingness=args.ignore_missingness)
        source_columns = columns
        serialization = CatalogueSerializationConfig(schema_name="clauds_whole_catalogue_v1",
            decimals=6, prefix="CLAUDS galaxy catalogue record")
    mlp_x, mlp_names = build_mlp_features(input_table, rows, columns, splits,
        encode_missingness=not args.ignore_missingness)
    embeddings, cache_path, metadata = get_embeddings(
        args, input_table, rows, ids, columns, serialization
    )
    split_counts = {name: int(np.count_nonzero(splits == name)) for name in ("train", "val", "test")}
    print("\nExperiment: frozen inference-optimized transformer vs tabular MLP")
    print(f"AION model/codec: not used; target: ZPHOT; rows: {len(rows):,}; splits: {split_counts}")
    input_mode = "11 AB magnitudes: " + ", ".join(MAGNITUDE_ONLY_BANDS) if args.magnitudes_only else "catalogue columns"
    print(f"Inputs: {input_mode}; {len(columns)} included, {len(excluded)} excluded")
    missing_policy = ("missing fields omitted from transformer text; MLP train-median imputation without indicators"
                      if args.ignore_missingness else "missing values explicitly represented")
    print(f"Missing-value policy: {missing_policy}")
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
    print("\nTraining 1/2: IoTFM frozen-transformer embedding head")
    iotfm_result = train_single_baseline(iotfm_product, "iotfm", output_dir=output / "iotfm", **kwargs)
    print("\nTraining 2/2: matched tabular MLP")
    mlp_result = train_single_baseline(mlp_product, "tabular", output_dir=output / "mlp", **kwargs)
    results = {"iotfm": iotfm_result, "mlp": mlp_result}
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "iotfm_mlp_results.pt"; torch.save(results, summary)
    comparison_split = "test" if split_counts["test"] else "val"
    print(f"\nSaving paired comparison figures from the {comparison_split} split")
    artifacts = save_comparison_artifacts(results, iotfm_product, mlp_product,
        comparison_split, args, edges, centers, output)
    manifest = dict(catalogue=str(catalogue), model=args.model, n_rows=len(rows), input_columns=columns,
        excluded_columns=excluded, include_id=args.include_id, include_location=args.include_location,
        aion_model_used=False, target="ZPHOT", split_counts=split_counts,
        input_mode="magnitudes_only" if args.magnitudes_only else "catalogue_columns",
        missing_value_policy=("omit_from_text_train_median_impute_no_indicator"
                              if args.ignore_missingness else "explicit"),
        magnitude_bands=list(MAGNITUDE_ONLY_BANDS) if args.magnitudes_only else [],
        magnitude_zero_point=MAG_ZERO_POINT if args.magnitudes_only else None,
        source_columns=source_columns,
        embedding_cache=str(cache_path), summary=str(summary),
        comparison_split=comparison_split, comparison_artifacts=artifacts)
    manifest_path = output / "iotfm_mlp_run.json"; manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"summary: {summary}\nmanifest: {manifest_path}")
    for name, path in artifacts.items(): print(f"comparison {name}: {path}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
