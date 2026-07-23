from __future__ import annotations

"""Leakage-safe CLAUDS benchmarks for pretrained tabular regressors.

The public entry point is :func:`main`.  Shell launchers live in
``scripts/table_models``.  Redshift catalogue columns are never model features:
the selected target is visible only as ``y_train`` and is represented as NaN on
validation/test rows in the saved completion table.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from scipy import stats

from .clauds_bands import (
    ALL_BAND_FLUX_COLUMNS,
    HSC_AION_BANDS,
    OBJECT_ID_COLUMN,
    REDSHIFT_COLUMNS,
    select_catalogue_row_indices,
)
from .dataset import dataset_for_split, make_random_split
from .metrics import point_photoz_metrics
from .models import load_baseline_model_from_checkpoint
from .training import evaluate_model_on_dataset, train_single_baseline
from .utils import flux_to_ab_mag, make_redshift_grid, resolve_torch_device


TABLE_MODELS = ("tabpfn", "tabfm", "tabicl")
MAGNITUDE_BANDS = ("u", "u_star", "g", "r", "i", "z", "y", "Y", "J", "H", "Ks")
FLUX_PREFIXES = ("FLUX_APER_2_", "FLUX_APER_3_", "FLUX_PSF_", "FLUX_KRON_", "FLUX_CMODEL_")
FLUX_ERROR_PREFIXES = (
    "FLUXERR_APER_2_", "FLUXERR_APER_3_", "FLUXERR_PSF_",
    "FLUXERR_KRON_", "FLUXERR_CMODEL_",
)
KRON_RADIUS_PREFIX = "RADIUS_KRON_"
MISSING_SENTINEL = -99.0
DEFAULT_CATALOGUE = Path("data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits")
DEFAULT_MORPHOLOGY_DIR = Path("data/clauds/images/tilesv5")
DEFAULT_OUTPUT_ROOT = Path("/arc/home/gsm/aion_output/figures/table_models")
DEFAULT_CACHE_ROOT = Path("/scratch/.tmp-gsm/aion_output/cache")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    feature_mode: str
    image_mode: str = "none"
    runner: str = "table"


COMPARISONS: dict[str, tuple[ArmSpec, ArmSpec]] = {
    "noimage-aion_comparison": (
        ArmSpec("noimage", "magonly"),
        ArmSpec("aion_image", "magonly", "aion"),
    ),
    "aion-timm_comparison": (
        ArmSpec("aion_image", "magonly", "aion"),
        ArmSpec("timm_image", "magonly", "timm"),
    ),
    "mlp_noimage_comparison": (
        ArmSpec("table_noimage", "magonly"),
        ArmSpec("standard_mlp_noimage", "magonly", runner="mlp"),
    ),
    "mlp_aionimage_comparison": (
        ArmSpec("table_aion_image", "magonly", "aion"),
        ArmSpec("standard_mlp_aion_image", "magonly", "aion", "mlp"),
    ),
    "magonly-fulltable": (
        ArmSpec("magnitude_only", "magonly"),
        ArmSpec("full_121", "full121"),
    ),
}


@dataclass
class CatalogueData:
    object_id: np.ndarray
    source_row: np.ndarray
    target: np.ndarray
    magnitude_features: pd.DataFrame
    full_features: pd.DataFrame | None
    detected_redshift_columns: list[str]

    def subset(self, indices: np.ndarray) -> "CatalogueData":
        indices = np.asarray(indices, dtype=np.int64)
        return CatalogueData(
            object_id=self.object_id[indices],
            source_row=self.source_row[indices],
            target=self.target[indices],
            magnitude_features=self.magnitude_features.iloc[indices].reset_index(drop=True),
            full_features=(
                None if self.full_features is None
                else self.full_features.iloc[indices].reset_index(drop=True)
            ),
            detected_redshift_columns=list(self.detected_redshift_columns),
        )


@dataclass
class PreparedArm:
    spec: ArmSpec
    features: pd.DataFrame
    imputation: dict[str, Any]


@dataclass
class ArmResult:
    name: str
    predictions: np.ndarray
    metrics: dict[str, dict[str, float]]
    artifacts: dict[str, str]
    metadata: dict[str, Any]


def max_rows_arg(value: str) -> int | None:
    if str(value).lower() in {"none", "all", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max rows must be positive or 'none'")
    return parsed


def is_redshift_related_column(name: str) -> bool:
    """Conservatively identify any catalogue field that may encode redshift."""
    normalized = str(name).strip().upper().replace("_", "-")
    exact = {value.upper().replace("_", "-") for value in REDSHIFT_COLUMNS.values()}
    return (
        normalized in exact
        or normalized.startswith("ZPHOT")
        or normalized.startswith("Z-LOW")
        or normalized.startswith("Z-HIGH")
        or normalized.startswith("Z-CHI")
        or normalized.startswith("Z-PEAK")
        or "REDSHIFT" in normalized
        or normalized.startswith("POSTERIOR-LOG")
        or normalized.startswith("LIKELIHOOD-LOG")
    )


def assert_no_redshift_features(columns: Sequence[str]) -> None:
    leaked = [name for name in columns if is_redshift_related_column(name)]
    if leaked:
        raise RuntimeError(f"Redshift leakage detected in model features: {leaked}")


def select_full121_columns(names: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    flux = [name for name in names if str(name).startswith(FLUX_PREFIXES)]
    errors = [name for name in names if str(name).startswith(FLUX_ERROR_PREFIXES)]
    radii = [name for name in names if str(name).startswith(KRON_RADIUS_PREFIX)]
    if (len(flux), len(errors), len(radii)) != (55, 55, 11):
        raise ValueError(
            "Expected exactly 55 flux, 55 flux-error, and 11 RADIUS_KRON "
            f"columns; found {(len(flux), len(errors), len(radii))}."
        )
    return flux, errors, radii


def _numeric_column(source: Any, name: str, rows: np.ndarray) -> np.ndarray:
    values = np.ma.asarray(source[name][rows])
    output = np.asarray(np.ma.getdata(values), dtype=np.float64)
    mask = np.ma.getmaskarray(values)
    if np.ndim(mask):
        output = output.copy()
        output[np.asarray(mask, dtype=bool)] = np.nan
    return output


def load_catalogue_data(
    catalogue: str | Path,
    *,
    max_rows: int | None,
    seed: int,
    include_full121: bool,
    z_min: float,
    z_max: float,
    target_column: str = REDSHIFT_COLUMNS["zphot"],
) -> CatalogueData:
    """Read a seeded random catalogue sample into named feature tables."""
    catalogue = Path(catalogue).expanduser().resolve()
    with fits.open(catalogue, memmap=True) as hdul:
        source = hdul[1].data
        names = list(source.names)
        required = [OBJECT_ID_COLUMN, target_column]
        required.extend(ALL_BAND_FLUX_COLUMNS[band] for band in MAGNITUDE_BANDS)
        missing = [name for name in required if name not in names]
        if missing:
            raise KeyError(f"Catalogue is missing required columns: {missing}")
        source_rows = select_catalogue_row_indices(
            len(source), max_rows=max_rows, sample_mode="random", seed=seed,
        )
        target = _numeric_column(source, target_column, source_rows)
        usable = np.isfinite(target) & (target != MISSING_SENTINEL)
        usable &= target >= float(z_min)
        usable &= target <= float(z_max)
        source_rows = source_rows[usable]
        target = target[usable].astype(np.float32)
        object_id = np.asarray(source[OBJECT_ID_COLUMN][source_rows], dtype=np.int64)

        magnitude: dict[str, np.ndarray] = {}
        for band in MAGNITUDE_BANDS:
            flux = _numeric_column(source, ALL_BAND_FLUX_COLUMNS[band], source_rows)
            values, _ = flux_to_ab_mag(flux, mag_zero_point=23.0)
            magnitude[f"{band}_mag"] = values.astype(np.float32)

        full_table: pd.DataFrame | None = None
        if include_full121:
            flux_columns, error_columns, radius_columns = select_full121_columns(names)
            full: dict[str, np.ndarray] = {}
            for name in flux_columns:
                values = _numeric_column(source, name, source_rows)
                valid = np.isfinite(values) & (values != MISSING_SENTINEL)
                full[name] = np.where(valid, values, np.nan).astype(np.float32)
            for name in error_columns:
                values = _numeric_column(source, name, source_rows)
                valid = np.isfinite(values) & (values != MISSING_SENTINEL) & (values > 0)
                full[name] = np.where(valid, values, np.nan).astype(np.float32)
            for name in radius_columns:
                values = _numeric_column(source, name, source_rows)
                valid = np.isfinite(values) & (values != MISSING_SENTINEL) & (values >= 0)
                full[name] = np.where(valid, values, np.nan).astype(np.float32)
            full_table = pd.DataFrame(full, copy=False)

        redshift_columns = [name for name in names if is_redshift_related_column(name)]

    magnitude_table = pd.DataFrame(magnitude, copy=False)
    assert_no_redshift_features(magnitude_table.columns)
    if full_table is not None:
        assert_no_redshift_features(full_table.columns)
    if len(np.unique(object_id)) != len(object_id):
        raise ValueError("Selected catalogue rows contain duplicate object IDs.")
    if len(object_id) < 3:
        raise ValueError("Fewer than three finite target rows remain after selection.")
    return CatalogueData(
        object_id=object_id,
        source_row=source_rows,
        target=target,
        magnitude_features=magnitude_table,
        full_features=full_table,
        detected_redshift_columns=redshift_columns,
    )


def align_catalogue_to_object_ids(data: CatalogueData, object_ids: Sequence[Any]) -> CatalogueData:
    lookup = {int(value): index for index, value in enumerate(data.object_id.tolist())}
    missing = [int(value) for value in object_ids if int(value) not in lookup]
    if missing:
        raise RuntimeError(
            f"Morphology cohort contains {len(missing)} IDs absent from the selected table; "
            f"first missing IDs: {missing[:5]}"
        )
    indices = np.asarray([lookup[int(value)] for value in object_ids], dtype=np.int64)
    return data.subset(indices)


def impute_from_training(
    frame: pd.DataFrame,
    split_labels: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Median-impute non-finite values using training rows only."""
    split_labels = np.asarray(split_labels, dtype=object)
    train = split_labels == "train"
    if not train.any():
        raise ValueError("Cannot impute a table without training rows.")
    output = frame.copy()
    columns: dict[str, Any] = {}
    for name in output.columns:
        values = pd.to_numeric(output[name], errors="coerce").to_numpy(
            dtype=np.float64, copy=True,
        )
        finite = np.isfinite(values)
        fit = train & finite
        if not fit.any():
            raise ValueError(f"Feature {name!r} has no finite training values.")
        fill = float(np.median(values[fit]))
        values[~finite] = fill
        output[name] = values.astype(np.float32)
        columns[str(name)] = {
            "fill": fill,
            "n_missing": int((~finite).sum()),
            "fit_split": "train",
        }
    assert_no_redshift_features(output.columns)
    if not np.isfinite(output.to_numpy(dtype=np.float32)).all():
        raise RuntimeError("Prepared feature table still contains non-finite values.")
    return output, {"method": "median", "fit_split": "train", "columns": columns}


def train_minmax_scale(frame: pd.DataFrame, split_labels: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = frame.to_numpy(dtype=np.float32)
    train = np.asarray(split_labels, dtype=object) == "train"
    low = values[train].min(axis=0)
    span = values[train].max(axis=0) - low
    constant = (~np.isfinite(span)) | (np.abs(span) <= 1e-12)
    span[constant] = 1.0
    scaled = (values - low) / span
    return scaled.astype(np.float32), {
        "mode": "minmax", "fit_split": "train", "offset": low.tolist(),
        "scale": span.tolist(), "constant_feature_mask": constant.tolist(),
    }


def build_masked_target(target: np.ndarray, split_labels: np.ndarray) -> np.ndarray:
    masked = np.full(len(target), np.nan, dtype=np.float32)
    train = np.asarray(split_labels, dtype=object) == "train"
    masked[train] = np.asarray(target, dtype=np.float32)[train]
    return masked


def _image_token_matrix(product: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    token_ids = np.load(product["image_token_ids_path"], mmap_mode="r")
    rows = np.asarray(product["image_token_row_indices"], dtype=np.int64)
    values = np.asarray(token_ids[rows], dtype=np.float32)
    grid = int(round(math.sqrt(values.shape[1])))
    if grid * grid != values.shape[1]:
        raise RuntimeError(f"AION token count {values.shape[1]} is not a square grid.")
    names = [f"aion_token_r{row:02d}_c{column:02d}" for row in range(grid) for column in range(grid)]
    return values, names


def build_morphology_product(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    """Build/reuse the AION-token cohort without using AION redshift features."""
    from .morphology import AIONMorphologyConfig, cache_aion_morphology_tokens

    selection = "all" if args.max_rows is None else f"n{args.max_rows}"
    run_cache = Path(args.cache_root).expanduser() / "table_models" / f"{selection}_seed{args.seed}"
    no_faint_limits = {band: None for band in HSC_AION_BANDS}
    config = AIONMorphologyConfig(
        catalogue_path=Path(args.catalogue),
        morphology_dir=Path(args.morphology_dir),
        cache_root=Path(args.cache_root),
        split_output_dir=run_cache / "clauds_split",
        photometry_cache_path=run_cache / "photometry_no_faint_cut.pt",
        max_rows=args.max_rows,
        sample_mode="random",
        sample_seed=args.seed,
        sample_require_valid_bands=(),
        force_rebuild_photometry=args.force_rebuild_photometry,
        force_rebuild_tokens=args.force_rebuild_tokens,
        preserve_photometry_splits=True,
        split_strategy="random",
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        z_min=args.z_min,
        z_max=args.z_max,
        redshift_include_min=True,
        redshift_include_max=True,
        n_z_bins=args.n_z_bins,
        hsc_mag_faint_limits=no_faint_limits,
        extra_band_invalid_fill="median",
        extra_band_include_valid_flags=False,
        include_grizy_in_mlp=True,
        use_aion_magnitude_embedding=False,
        image_flux_scale=args.image_flux_scale,
        min_cutout_weight_coverage=args.min_cutout_weight_coverage,
        token_batch_size=args.token_batch_size,
        model_kinds=("morphology",),
        feature_scaling="minmax",
        epochs=args.mlp_epochs,
        train_batch_size=args.mlp_train_batch_size,
        eval_batch_size=args.mlp_eval_batch_size,
        seed=args.seed,
        device_choice=args.device,
    ).normalized()
    print("Preparing matched AION image-token cohort", flush=True)
    return cache_aion_morphology_tokens(config), config


def build_timm_features(
    product: Mapping[str, Any], args: argparse.Namespace,
) -> tuple[np.ndarray, list[str], str]:
    from .timm_morphology import (
        TimmMorphologyConfig,
        extract_or_load_timm_embeddings,
    )

    model_tag = args.timm_model.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_")
    metadata = dict(product.get("metadata", {}))
    morph_tag = str(metadata.get("morphology_tag", "cohort"))
    cache_path = (
        Path(args.cache_root).expanduser() / "table_models" / "timm"
        / morph_tag / f"{model_tag}_in{args.timm_input_size}.pt"
    )
    config = TimmMorphologyConfig(
        model_name=args.timm_model,
        pretrained=True,
        input_size=args.timm_input_size,
        batch_size=args.timm_batch_size,
        device=args.timm_device or args.device,
    )
    tensor = extract_or_load_timm_embeddings(
        product,
        morphology_dir=args.morphology_dir,
        cache_path=cache_path,
        config=config,
        force_recompute=args.force_recompute_timm,
    )
    values = tensor.numpy().astype(np.float32, copy=False)
    names = [f"timm_embedding_{index:04d}" for index in range(values.shape[1])]
    return values, names, str(cache_path)


def prepare_arm(
    spec: ArmSpec,
    data: CatalogueData,
    split_labels: np.ndarray,
    *,
    aion_features: tuple[np.ndarray, list[str]] | None,
    timm_features: tuple[np.ndarray, list[str]] | None,
) -> PreparedArm:
    if spec.feature_mode == "magonly":
        base = data.magnitude_features.copy()
    elif spec.feature_mode == "full121":
        if data.full_features is None:
            raise RuntimeError("The full 121-column table was not loaded.")
        base = data.full_features.copy()
    else:
        raise ValueError(f"Unknown feature mode: {spec.feature_mode}")

    if spec.image_mode == "aion":
        if aion_features is None:
            raise RuntimeError("AION image features were not prepared.")
        values, names = aion_features
        base = pd.concat([base, pd.DataFrame(values, columns=names, copy=False)], axis=1)
    elif spec.image_mode == "timm":
        if timm_features is None:
            raise RuntimeError("timm image features were not prepared.")
        values, names = timm_features
        base = pd.concat([base, pd.DataFrame(values, columns=names, copy=False)], axis=1)
    elif spec.image_mode != "none":
        raise ValueError(f"Unknown image mode: {spec.image_mode}")

    prepared, imputation = impute_from_training(base, split_labels)
    return PreparedArm(spec=spec, features=prepared, imputation=imputation)


def resolved_device_name(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(requested)


def package_versions() -> dict[str, str]:
    output = {}
    for package in ("tabpfn", "tabfm", "tabicl", "torch", "scikit-learn"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = "not-installed"
    return output


def create_table_regressor(
    model_name: str,
    args: argparse.Namespace,
    *,
    n_features: int,
) -> Any:
    device = resolved_device_name(args.device)
    model_path = None if args.model_path is None else str(Path(args.model_path).expanduser())
    if not args.allow_model_download and model_path is None:
        raise RuntimeError("--no-allow-model-download requires --model-path.")

    if model_name == "tabicl":
        from tabicl import TabICLRegressor

        return TabICLRegressor(
            n_estimators=args.n_estimators,
            batch_size=args.table_batch_size or 8,
            model_path=model_path,
            allow_auto_download=args.allow_model_download,
            device=device,
            offload_mode=args.offload_mode,
            random_state=args.seed,
            verbose=True,
        )
    if model_name == "tabpfn":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            n_estimators=args.n_estimators,
            model_path=model_path or "auto",
            device=device,
            ignore_pretraining_limits=args.ignore_pretraining_limits,
            memory_saving_mode=args.memory_saving_mode,
            random_state=args.seed,
            show_progress_bar=True,
        )
    if model_name == "tabfm":
        from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch

        frozen = tabfm_v1_0_0_pytorch.load(
            model_type="regression",
            checkpoint_path=model_path,
            device=device,
            dtype=torch.bfloat16 if device.startswith("cuda") else None,
        )
        return TabFMRegressor(
            model=frozen,
            n_estimators=args.n_estimators,
            max_num_features=None,
            max_num_rows=args.tabfm_max_context_rows,
            batch_size=args.table_batch_size or 1,
            random_state=args.seed,
            verbose=True,
        )
    raise ValueError(f"Unknown table model: {model_name}")


def regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    finite = np.isfinite(prediction) & np.isfinite(truth)
    prediction, truth = prediction[finite], truth[finite]
    if not len(truth):
        raise ValueError("No finite prediction/truth pairs are available.")
    base = point_photoz_metrics(
        torch.from_numpy(prediction), torch.from_numpy(truth),
    )
    residual = prediction - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float("nan") if denominator <= 0 else 1.0 - float(np.sum(residual ** 2)) / denominator
    variable_pair = (
        len(truth) > 1
        and float(np.std(prediction)) > 0.0
        and float(np.std(truth)) > 0.0
    )
    pearson = float(np.corrcoef(prediction, truth)[0, 1]) if variable_pair else float("nan")
    spearman = float(stats.spearmanr(prediction, truth).statistic) if variable_pair else float("nan")
    return {
        **{key: float(value) for key, value in base.items()},
        "sigma_nmad": float(base["nmad"]),
        "eta": float(base["catastrophic_outlier_fraction"]),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": r2,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "n_objects": int(len(truth)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _arm_signature(
    arm: PreparedArm,
    data: CatalogueData,
    split_labels: np.ndarray,
    args: argparse.Namespace,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(data.object_id, dtype=np.int64).tobytes())
    digest.update(np.asarray(split_labels, dtype="U5").tobytes())
    record = {
        "model": args.model,
        "spec": asdict(arm.spec),
        "columns": list(arm.features.columns),
        "shape": list(arm.features.shape),
        "n_estimators": args.n_estimators,
        "model_path": args.model_path,
        "seed": args.seed,
    }
    digest.update(json.dumps(record, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def save_completion_artifacts(
    output_dir: Path,
    *,
    data: CatalogueData,
    split_labels: np.ndarray,
    prediction: np.ndarray,
    signature: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    masked = build_masked_target(data.target, split_labels)
    filled = masked.copy()
    held_out = np.asarray(split_labels, dtype=object) != "train"
    filled[held_out] = prediction[held_out]
    npz_path = output_dir / "redshift_completion.npz"
    np.savez_compressed(
        npz_path,
        object_id=data.object_id,
        source_row=data.source_row,
        split=np.asarray(split_labels, dtype="U5"),
        masked_redshift=masked,
        inferred_redshift=prediction,
        filled_redshift=filled,
        true_redshift=data.target,
        signature=np.asarray(signature),
    )
    csv_path = output_dir / "redshift_completion.csv.gz"
    pd.DataFrame({
        "object_id": data.object_id,
        "source_row": data.source_row,
        "split": split_labels,
        "masked_redshift": masked,
        "inferred_redshift": prediction,
        "filled_redshift": filled,
        "true_redshift": data.target,
    }).to_csv(csv_path, index=False, compression="gzip")
    return {"completion_npz": str(npz_path), "completion_csv": str(csv_path)}


def run_table_arm(
    arm: PreparedArm,
    data: CatalogueData,
    split_labels: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    estimator_factory: Callable[[str, argparse.Namespace], Any] | None = None,
) -> ArmResult:
    signature = _arm_signature(arm, data, split_labels, args)
    completion_path = output_dir / "redshift_completion.npz"
    metrics_path = output_dir / "metrics.json"
    if args.resume and completion_path.exists() and metrics_path.exists():
        cached = np.load(completion_path, allow_pickle=False)
        cached_signature = str(np.asarray(cached["signature"]).item())
        if cached_signature == signature:
            prediction = np.asarray(cached["inferred_redshift"], dtype=np.float32)
            metrics = json.loads(metrics_path.read_text())
            print(f"{arm.spec.name}: reusing completed predictions", flush=True)
            return ArmResult(
                name=arm.spec.name,
                predictions=prediction,
                metrics=metrics,
                artifacts={
                    "completion_npz": str(completion_path),
                    "completion_csv": str(output_dir / "redshift_completion.csv.gz"),
                },
                metadata={"signature": signature, "resumed": True},
            )

    train = np.asarray(split_labels, dtype=object) == "train"
    held_out = ~train
    if not train.any() or not held_out.any():
        raise ValueError("Both training and held-out rows are required.")
    factory = estimator_factory
    if factory is None:
        estimator = create_table_regressor(
            args.model, args, n_features=arm.features.shape[1],
        )
    else:
        estimator = factory(args.model, args)
    print(
        f"{arm.spec.name}: fitting {args.model} on {int(train.sum()):,} rows × "
        f"{arm.features.shape[1]:,} features",
        flush=True,
    )
    started = time.monotonic()
    estimator.fit(arm.features.loc[train], data.target[train])
    prediction = np.full(len(data.target), np.nan, dtype=np.float32)
    prediction[held_out] = np.asarray(
        estimator.predict(arm.features.loc[held_out]), dtype=np.float32,
    ).reshape(-1)
    elapsed = time.monotonic() - started
    metrics = {}
    for split in ("val", "test"):
        mask = np.asarray(split_labels, dtype=object) == split
        if mask.any():
            metrics[split] = regression_metrics(prediction[mask], data.target[mask])
    artifacts = save_completion_artifacts(
        output_dir,
        data=data,
        split_labels=split_labels,
        prediction=prediction,
        signature=signature,
    )
    write_json(metrics_path, metrics)
    schema_path = output_dir / "table_schema.json"
    write_json(schema_path, {
        "feature_columns": list(arm.features.columns),
        "n_rows": len(arm.features),
        "n_features": arm.features.shape[1],
        "feature_mode": arm.spec.feature_mode,
        "image_mode": arm.spec.image_mode,
        "redshift_feature_columns": [],
        "masked_target_column": REDSHIFT_COLUMNS["zphot"],
        "imputation": arm.imputation,
    })
    artifacts["metrics"] = str(metrics_path)
    artifacts["schema"] = str(schema_path)
    if args.save_input_table:
        table_path = output_dir / "model_input_table.npz"
        np.savez_compressed(
            table_path,
            features=arm.features.to_numpy(dtype=np.float32),
            columns=np.asarray(arm.features.columns, dtype="U"),
            object_id=data.object_id,
            split=np.asarray(split_labels, dtype="U5"),
            masked_redshift=build_masked_target(data.target, split_labels),
        )
        artifacts["input_table"] = str(table_path)
    del estimator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ArmResult(
        name=arm.spec.name,
        predictions=prediction,
        metrics=metrics,
        artifacts=artifacts,
        metadata={"signature": signature, "elapsed_seconds": elapsed, "resumed": False},
    )


def _base_product(
    data: CatalogueData,
    features: np.ndarray,
    feature_names: Sequence[str],
    split_labels: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "object_id": data.object_id.tolist(),
        "field": ["catalogue"] * len(data.object_id),
        "aion_embedding": torch.empty((len(data.object_id), 0), dtype=torch.float32),
        "extra_features": torch.from_numpy(np.asarray(features, dtype=np.float32)),
        "z_spec": torch.from_numpy(data.target.astype(np.float32)),
        "redshift_reference": {},
        "split_labels": np.asarray(split_labels, dtype=object).tolist(),
        "feature_names": list(feature_names),
        "metadata": dict(metadata),
    }


def run_mlp_arm(
    arm: PreparedArm,
    data: CatalogueData,
    split_labels: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    morphology_product: Mapping[str, Any] | None,
    morphology_config: Any | None,
) -> ArmResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges, centers = make_redshift_grid(args.z_min, args.z_max, args.n_z_bins)
    metadata = {
        "comparison": args.comparison,
        "redshift_feature_columns": [],
        "target_redshift_column": REDSHIFT_COLUMNS["zphot"],
        "target_masking": "train visible; validation/test masked",
        "imputation": arm.imputation,
    }
    if arm.spec.image_mode == "none":
        scaled, scaling = train_minmax_scale(arm.features, split_labels)
        product = _base_product(
            data, scaled, arm.features.columns, split_labels,
            {**metadata, "feature_scaling": scaling},
        )
        result = train_single_baseline(
            product,
            "tabular",
            output_dir=output_dir,
            n_z_bins=args.n_z_bins,
            redshift_edges=edges,
            redshift_centers=centers,
            epochs=args.mlp_epochs,
            learning_rate=args.mlp_learning_rate,
            weight_decay=args.mlp_weight_decay,
            train_batch_size=args.mlp_train_batch_size,
            eval_batch_size=args.mlp_eval_batch_size,
            device=args.device,
        )
        device = resolve_torch_device(args.device)
        model = load_baseline_model_from_checkpoint(
            result["checkpoint_path"],
            model_kind="tabular",
            aion_dim=0,
            extra_feature_dim=scaled.shape[1],
            n_z_bins=args.n_z_bins,
            device=device,
        )
        evaluations = {
            split: evaluate_model_on_dataset(
                model,
                dataset_for_split(product, split),
                "tabular",
                batch_size=args.mlp_eval_batch_size,
                device=device,
                redshift_edges=edges,
                redshift_centers=centers,
            )
            for split in ("val", "test")
        }
    else:
        if morphology_product is None or morphology_config is None:
            raise RuntimeError("The AION-image MLP requires a morphology product.")
        from .morphology import train_single_morphology_model

        product = dict(morphology_product)
        product.update(_base_product(
            data,
            arm.features.iloc[:, : len(MAGNITUDE_BANDS)].to_numpy(dtype=np.float32),
            list(arm.features.columns[: len(MAGNITUDE_BANDS)]),
            split_labels,
            {**dict(morphology_product.get("metadata", {})), **metadata},
        ))
        # _base_product deliberately does not carry image cache pointers.
        product["image_token_ids_path"] = morphology_product["image_token_ids_path"]
        product["image_token_row_indices"] = morphology_product["image_token_row_indices"]
        config = replace(
            morphology_config,
            output_dir=output_dir,
            model_kinds=("morphology",),
            feature_scaling="minmax",
            epochs=args.mlp_epochs,
            learning_rate=args.mlp_learning_rate,
            weight_decay=args.mlp_weight_decay,
            train_batch_size=args.mlp_train_batch_size,
            eval_batch_size=args.mlp_eval_batch_size,
        )
        result = train_single_morphology_model(
            product, "morphology", output_dir=output_dir, config=config, device=args.device,
        )
        evaluations = {
            "val": result["val_evaluation"],
            "test": result["test_evaluation"],
        }
        if evaluations["test"] is None:
            raise RuntimeError("The standard image MLP produced no test evaluation.")

    prediction = np.full(len(data.target), np.nan, dtype=np.float32)
    metrics: dict[str, dict[str, float]] = {}
    for split, evaluation in evaluations.items():
        if evaluation is None:
            continue
        mask = np.asarray(split_labels, dtype=object) == split
        prediction[mask] = torch.as_tensor(
            evaluation["z_p50"],
        ).cpu().numpy().astype(np.float32)
        metrics[split] = regression_metrics(prediction[mask], data.target[mask])
    signature = hashlib.sha256(
        (args.comparison + arm.spec.name + str(args.seed) + str(len(data.target))).encode("utf-8")
    ).hexdigest()
    artifacts = save_completion_artifacts(
        output_dir,
        data=data,
        split_labels=split_labels,
        prediction=prediction,
        signature=signature,
    )
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    artifacts.update(metrics=str(metrics_path), checkpoint=str(result["checkpoint_path"]))
    return ArmResult(
        name=arm.spec.name,
        predictions=prediction,
        metrics=metrics,
        artifacts=artifacts,
        metadata={"standard_mlp": True, "pdf_metrics": result.get("final_metrics", {})},
    )


def save_comparison_plot(
    results: Sequence[ArmResult],
    data: CatalogueData,
    split_labels: np.ndarray,
    output_path: Path,
    *,
    seed: int,
) -> str:
    test = np.flatnonzero(np.asarray(split_labels, dtype=object) == "test")
    if len(test) > 100_000:
        rng = np.random.default_rng(seed)
        test = np.sort(rng.choice(test, size=100_000, replace=False))
    figure, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    low = float(np.min(data.target[test]))
    high = float(np.max(data.target[test]))
    for axis, result in zip(axes[0], results):
        predicted = result.predictions[test]
        axis.hexbin(data.target[test], predicted, gridsize=90, mincnt=1, bins="log", cmap="viridis")
        axis.plot([low, high], [low, high], "r--", linewidth=1)
        axis.set_xlabel("true ZPHOT")
        axis.set_ylabel("inferred redshift")
        metrics = result.metrics["test"]
        axis.set_title(
            f"{result.name}\nR²={metrics['r2']:.4f}, ρ={metrics['spearman_rho']:.4f}\n"
            f"σNMAD={metrics['sigma_nmad']:.4f}, η={metrics['eta']:.2%}"
        )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return str(output_path)


def validate_args(args: argparse.Namespace) -> None:
    if args.comparison not in COMPARISONS:
        raise ValueError(f"Unknown comparison {args.comparison!r}")
    if args.model not in TABLE_MODELS:
        raise ValueError(f"--model must be one of {TABLE_MODELS}")
    if not math.isclose(args.train_fraction + args.test_fraction + args.val_fraction, 1.0):
        raise ValueError("train/test/val fractions must sum to 1")
    positive = (
        args.n_estimators, args.n_z_bins, args.mlp_epochs,
        args.mlp_train_batch_size, args.mlp_eval_batch_size,
        args.token_batch_size, args.timm_batch_size, args.timm_input_size,
    )
    if min(positive) <= 0:
        raise ValueError("batch sizes, estimator count, bins, and epochs must be positive")
    if args.z_max <= args.z_min:
        raise ValueError("--z-max must be greater than --z-min")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", choices=tuple(COMPARISONS), required=True)
    parser.add_argument("--model", choices=TABLE_MODELS, required=True)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--morphology-dir", type=Path, default=DEFAULT_MORPHOLOGY_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--max-rows", type=max_rows_arg, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.63)
    parser.add_argument("--test-fraction", type=float, default=0.32)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--n-z-bins", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--table-batch-size", type=int, default=0,
                        help="0 selects the backend default (TabFM 1; TabICL 8)")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-model-download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-pretraining-limits", action="store_true")
    parser.add_argument("--memory-saving-mode", default="auto")
    parser.add_argument("--offload-mode", default="auto")
    parser.add_argument("--tabfm-max-context-rows", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-input-table", action="store_true")

    parser.add_argument("--force-rebuild-photometry", action="store_true")
    parser.add_argument("--force-rebuild-tokens", action="store_true")
    parser.add_argument("--image-flux-scale", type=float, default=1.0)
    parser.add_argument("--min-cutout-weight-coverage", type=float, default=0.90)
    parser.add_argument("--token-batch-size", type=int, default=64)
    parser.add_argument("--timm-model", default="hf-hub:timm/convnext_tiny.dinov3_lvd1689m")
    parser.add_argument("--timm-input-size", type=int, default=224)
    parser.add_argument("--timm-batch-size", type=int, default=128)
    parser.add_argument("--timm-device")
    parser.add_argument("--force-recompute-timm", action="store_true")

    parser.add_argument("--mlp-epochs", type=int, default=10)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-train-batch-size", type=int, default=256)
    parser.add_argument("--mlp-eval-batch-size", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    catalogue = Path(args.catalogue).expanduser().resolve()
    if not catalogue.exists():
        raise FileNotFoundError(f"Catalogue not found: {catalogue}")
    args.catalogue = catalogue
    args.cache_root = Path(args.cache_root).expanduser()
    output_dir = Path(args.output_root).expanduser() / f"{args.model}_{args.comparison}"
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = COMPARISONS[args.comparison]
    needs_full = any(spec.feature_mode == "full121" for spec in specs)
    needs_images = any(spec.image_mode != "none" for spec in specs)
    needs_timm = any(spec.image_mode == "timm" for spec in specs)
    print(f"comparison: {args.comparison}")
    print(f"table model: {args.model}")
    print(f"output: {output_dir}")
    print("redshift leakage policy: every redshift-related catalogue field excluded")

    data = load_catalogue_data(
        catalogue,
        max_rows=args.max_rows,
        seed=args.seed,
        include_full121=needs_full,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    morphology_product = None
    morphology_config = None
    aion_features = None
    timm_features = None
    timm_cache = None
    if needs_images:
        morphology_dir = Path(args.morphology_dir).expanduser().resolve()
        if not morphology_dir.exists():
            raise FileNotFoundError(f"Morphology directory not found: {morphology_dir}")
        args.morphology_dir = morphology_dir
        morphology_product, morphology_config = build_morphology_product(args)
        data = align_catalogue_to_object_ids(data, morphology_product["object_id"])
        aion_features = _image_token_matrix(morphology_product)
        if needs_timm:
            values, names, timm_cache = build_timm_features(morphology_product, args)
            timm_features = (values, names)

    split_labels = make_random_split(
        len(data.target),
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    split_counts = {
        split: int(np.count_nonzero(split_labels == split))
        for split in ("train", "val", "test")
    }
    prepared = [
        prepare_arm(
            spec, data, split_labels,
            aion_features=aion_features,
            timm_features=timm_features,
        )
        for spec in specs
    ]
    run_manifest = {
        "comparison": args.comparison,
        "model": args.model,
        "output_dir": str(output_dir),
        "catalogue": str(catalogue),
        "n_rows": len(data.target),
        "split_counts": split_counts,
        "seed": args.seed,
        "target_redshift_column": REDSHIFT_COLUMNS["zphot"],
        "redshift_masking": {
            "detected_source_columns": data.detected_redshift_columns,
            "model_feature_columns": [],
            "policy": "target visible only on train; all redshift-derived features excluded; val/test target NaN",
        },
        "arms": [
            {
                **asdict(arm.spec),
                "n_features": arm.features.shape[1],
                "feature_columns": list(arm.features.columns),
                "imputation": arm.imputation,
            }
            for arm in prepared
        ],
        "packages": package_versions(),
        "timm_cache": timm_cache,
        "arguments": vars(args),
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    for arm in prepared:
        arm_dir = output_dir / arm.spec.name
        write_json(arm_dir / "prepared_table_schema.json", {
            "n_rows": len(arm.features),
            "n_features": arm.features.shape[1],
            "columns": list(arm.features.columns),
            "redshift_features": [],
            "masked_target_counts": {
                "visible_train": split_counts["train"],
                "masked_validation_test": split_counts["val"] + split_counts["test"],
            },
        })
        if args.save_input_table:
            arm_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                arm_dir / "prepared_input_table.npz",
                features=arm.features.to_numpy(dtype=np.float32),
                columns=np.asarray(arm.features.columns, dtype="U"),
                object_id=data.object_id,
                split=np.asarray(split_labels, dtype="U5"),
                masked_redshift=build_masked_target(data.target, split_labels),
            )
    if args.prepare_only:
        print("Preparation complete; model fitting skipped by --prepare-only.")
        return 0

    results: list[ArmResult] = []
    for arm in prepared:
        arm_dir = output_dir / arm.spec.name
        if arm.spec.runner == "table":
            result = run_table_arm(arm, data, split_labels, args, arm_dir)
        else:
            result = run_mlp_arm(
                arm, data, split_labels, args, arm_dir,
                morphology_product=morphology_product,
                morphology_config=morphology_config,
            )
        results.append(result)

    plot_path = save_comparison_plot(
        results, data, split_labels, output_dir / "test_redshift_comparison.png", seed=args.seed,
    )
    summary = {
        "comparison": args.comparison,
        "model": args.model,
        "n_rows": len(data.target),
        "split_counts": split_counts,
        "arms": {
            result.name: {
                "metrics": result.metrics,
                "artifacts": result.artifacts,
                "metadata": result.metadata,
            }
            for result in results
        },
        "comparison_plot": plot_path,
    }
    write_json(output_dir / "comparison_results.json", summary)
    print(f"comparison results: {output_dir / 'comparison_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
