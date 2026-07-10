from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import torch

from .clauds_bands import (
    ALL_BAND_FLUX_COLUMNS, ALL_FLAG_COLUMNS, HSC_AION_BANDS, FLAG_COLUMNS,
    BAND_FLUX_COLUMNS, BAND_ERROR_COLUMNS, REDSHIFT_COLUMNS
)
from .utils import flux_to_ab_mag, select_torch_device, load_cached_product

DEFAULT_METRIC_KEYS = (
    "bias",
    "median_bias",
    "nmad",
    "catastrophic_outlier_fraction",
    "cross_entropy",
    "mean_log_score",
    "mean_crps",
    "p16_p84_coverage",
    "pit_mean",
)


def subset_product_rows(
    product: Mapping[str, Any],
    mask: torch.Tensor | np.ndarray | Sequence[bool],
) -> dict[str, Any]:
    """Subset first-axis product rows while preserving cached-product structure."""
    mask_np = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask_np)
    n_rows = len(mask_np)

    output = {}
    for key, value in product.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (n_rows,):
            output[key] = value[torch.as_tensor(indices, dtype=torch.long)]
        elif isinstance(value, np.ndarray) and value.shape[:1] == (n_rows,):
            output[key] = value[indices]
        elif isinstance(value, list) and len(value) == n_rows:
            output[key] = [value[int(idx)] for idx in indices]
        elif isinstance(value, dict):
            output[key] = {
                sub_key: (
                    sub_value[torch.as_tensor(indices, dtype=torch.long)]
                    if isinstance(sub_value, torch.Tensor) and sub_value.shape[:1] == (n_rows,)
                    else sub_value[indices]
                    if isinstance(sub_value, np.ndarray) and sub_value.shape[:1] == (n_rows,)
                    else sub_value
                )
                for sub_key, sub_value in value.items()
            }
        else:
            output[key] = value

    metadata = dict(product.get("metadata", {}))
    metadata["n_usable_rows"] = int(len(indices))
    metadata["row_subset"] = {
        "source_n_rows": int(n_rows),
        "kept_n_rows": int(len(indices)),
    }
    output["metadata"] = metadata
    return output

DEFAULT_EXTRA_BANDS = ("u", "u_star", "Y", "J", "H", "Ks")
EXTRA_BAND_LABELS = {
    "u": "u",
    "u_star": "u*",
    "Y": "Y",
    "J": "J",
    "H": "H",
    "Ks": "Ks",
}
EXTRA_BAND_ALIASES = {
    "u": "u",
    "megacam_u": "u",
    "megacam-u": "u",
    "u*": "u_star",
    "u_star": "u_star",
    "ustar": "u_star",
    "us": "u_star",
    "uS": "u_star",
    "megacam_us": "u_star",
    "megacam-us": "u_star",
    "Y": "Y",
    "y_vircam": "Y",
    "vircam_y": "Y",
    "vircam-y": "Y",
    "J": "J",
    "j": "J",
    "vircam_j": "J",
    "vircam-j": "J",
    "H": "H",
    "h": "H",
    "vircam_h": "H",
    "vircam-h": "H",
    "Ks": "Ks",
    "ks": "Ks",
    "k_s": "Ks",
    "vircam_ks": "Ks",
    "vircam-ks": "Ks",
}


def resolve_extra_band_name(name: str) -> str:
    raw = str(name).strip()
    candidates = (
        raw,
        raw.replace(" ", "_"),
        raw.replace(" ", "_").replace("-", "_"),
        raw.replace(" ", "_").replace("-", "_").lower(),
    )
    for candidate in candidates:
        if candidate in EXTRA_BAND_ALIASES:
            return EXTRA_BAND_ALIASES[candidate]
    raise KeyError(
        f"Unknown extra band {name!r}. Expected one of: "
        f"{', '.join(EXTRA_BAND_LABELS.values())}."
    )

def resolve_extra_band_names(names: Sequence[str] | None = None) -> tuple[str, ...]:
    names = DEFAULT_EXTRA_BANDS if names is None else names
    resolved = []
    for name in names:
        band = resolve_extra_band_name(str(name))
        if band not in resolved:
            resolved.append(band)
    return tuple(resolved)

def extra_band_feature_name(band: str) -> str:
    return f"{band}_mag"

def extra_band_valid_feature_name(band: str) -> str:
    return f"{band}_mag_valid"

def _table_column_names(table) -> set[str]:
    if hasattr(table, "names") and table.names is not None:
        return set(table.names)
    if hasattr(table, "dtype") and table.dtype.names is not None:
        return set(table.dtype.names)
    if isinstance(table, Mapping):
        return set(table.keys())
    raise TypeError("table must be a FITS table, structured array, or mapping of arrays")

def _table_length(table) -> int:
    if isinstance(table, Mapping):
        first_key = next(iter(table))
        return len(table[first_key])
    return len(table)

def _table_column(table, column_name: str, rows=None) -> np.ndarray:
    values = table[column_name]
    if rows is not None:
        values = values[rows]
    return np.asarray(values)

def _warn_missing_band(band: str, reason: str) -> None:
    warnings.warn(
        f"Requested extra band {EXTRA_BAND_LABELS.get(band, band)!r} is unavailable: "
        f"{reason}. The feature will be filled and its valid flag will be false.",
        RuntimeWarning,
        stacklevel=3,
    )

def _valid_from_table_flags(
    table,
    band: str,
    valid: np.ndarray,
    *,
    rows=None,
) -> np.ndarray:
    names = _table_column_names(table)
    output = np.asarray(valid, dtype=bool).copy()
    for prefix in ("has_bad_photometry", "is_no_data", "not_observed"):
        flag_key = f"{prefix}_{band}"
        column_name = ALL_FLAG_COLUMNS.get(flag_key)
        if column_name is not None and column_name in names:
            output &= ~_table_column(table, column_name, rows=rows).astype(bool)
    return output

def _valid_from_split_flags(
    flags: np.ndarray,
    band: str,
    valid: np.ndarray,
) -> np.ndarray:
    output = np.asarray(valid, dtype=bool).copy()
    flag_names = set(flags.dtype.names or ())
    for prefix in ("has_bad_photometry", "is_no_data", "not_observed"):
        flag_name = f"{prefix}_{band}"
        if flag_name in flag_names:
            output &= ~flags[flag_name].astype(bool)
    return output

def extract_extra_band_magnitudes_from_table(
    table,
    *,
    extra_bands: Sequence[str] | None = None,
    rows=None,
    mag_zero_point: float = 23.0,
    warn_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return raw extra-band magnitudes and per-band validity from a table."""
    selected_bands = resolve_extra_band_names(extra_bands)
    sample_column = next(iter(_table_column_names(table)))
    n_rows = _table_length(table) if rows is None else len(_table_column(table, sample_column, rows=rows))
    if not selected_bands:
        return (
            np.empty((n_rows, 0), dtype=np.float32),
            np.empty((n_rows, 0), dtype=bool),
            selected_bands,
        )

    names = _table_column_names(table)
    mags = []
    valid_masks = []
    for band in selected_bands:
        column_name = ALL_BAND_FLUX_COLUMNS.get(band)
        if column_name is None:
            raise KeyError(f"No flux-column definition for extra band {band!r}.")
        if column_name not in names:
            if warn_missing:
                _warn_missing_band(band, f"missing FITS/split column {column_name!r}")
            mag = np.full(n_rows, np.nan, dtype=np.float32)
            valid = np.zeros(n_rows, dtype=bool)
        else:
            flux = _table_column(table, column_name, rows=rows).astype(np.float32, copy=False)
            mag, valid = flux_to_ab_mag(flux, mag_zero_point=mag_zero_point)
            valid = _valid_from_table_flags(table, band, valid, rows=rows)
            if warn_missing and not bool(valid.any()):
                _warn_missing_band(band, "no row has finite positive flux with usable quality flags")
        mags.append(mag.astype(np.float32))
        valid_masks.append(valid.astype(bool))

    return np.stack(mags, axis=1), np.stack(valid_masks, axis=1), selected_bands

def extract_extra_band_magnitudes_from_split_arrays(
    bands: np.ndarray,
    flags: np.ndarray | None = None,
    *,
    extra_bands: Sequence[str] | None = None,
    mag_zero_point: float = 23.0,
    warn_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return raw extra-band magnitudes and per-band validity from split arrays."""
    selected_bands = resolve_extra_band_names(extra_bands)
    n_rows = len(bands)
    if not selected_bands:
        return (
            np.empty((n_rows, 0), dtype=np.float32),
            np.empty((n_rows, 0), dtype=bool),
            selected_bands,
        )

    names = set(bands.dtype.names or ())
    mags = []
    valid_masks = []
    for band in selected_bands:
        field_name = f"flux_cmodel_{band}"
        if field_name not in names:
            if warn_missing:
                _warn_missing_band(band, f"missing split-cache field {field_name!r}")
            mag = np.full(n_rows, np.nan, dtype=np.float32)
            valid = np.zeros(n_rows, dtype=bool)
        else:
            mag, valid = flux_to_ab_mag(bands[field_name], mag_zero_point=mag_zero_point)
            if flags is not None:
                valid = _valid_from_split_flags(flags, band, valid)
            if warn_missing and not bool(valid.any()):
                _warn_missing_band(band, "no row has finite positive flux with usable quality flags")
        mags.append(mag.astype(np.float32))
        valid_masks.append(valid.astype(bool))

    return np.stack(mags, axis=1), np.stack(valid_masks, axis=1), selected_bands

def _fill_magnitude_columns(
    magnitudes: torch.Tensor,
    valid: torch.Tensor,
    *,
    invalid_fill: str | float = "median",
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    mags = torch.as_tensor(magnitudes, dtype=torch.float32)
    if mags.ndim == 1:
        mags = mags[:, None]
    valid_mask = torch.as_tensor(valid, dtype=torch.bool)
    if valid_mask.ndim == 1:
        valid_mask = valid_mask[:, None]
    finite = torch.isfinite(mags)
    valid_mask = valid_mask & finite
    if mags.shape != valid_mask.shape:
        raise ValueError("magnitudes and valid must have the same shape.")
    if mags.shape[1] == 0:
        return mags, valid_mask, []

    filled_columns = []
    fill_values = []
    for col_idx in range(mags.shape[1]):
        column = mags[:, col_idx]
        column_valid = valid_mask[:, col_idx]
        if isinstance(invalid_fill, (int, float)):
            fill_value = float(invalid_fill)
        elif invalid_fill == "median":
            fill_value = float(torch.median(column[column_valid]).item()) if bool(column_valid.any()) else 0.0
        elif invalid_fill == "max_valid":
            fill_value = float(torch.max(column[column_valid]).item()) if bool(column_valid.any()) else 0.0
        else:
            raise ValueError("invalid_fill must be 'median', 'max_valid', or a numeric value.")
        filled_columns.append(torch.where(column_valid, column, torch.full_like(column, fill_value)))
        fill_values.append(fill_value)

    return torch.stack(filled_columns, dim=1), valid_mask, fill_values

def make_extra_band_feature_matrix(
    magnitudes: torch.Tensor | np.ndarray,
    valid: torch.Tensor | np.ndarray,
    *,
    extra_bands: Sequence[str],
    invalid_fill: str | float = "median",
    include_valid_flags: bool = False,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    """Fill invalid magnitudes and build the final MLP feature matrix."""
    selected_bands = resolve_extra_band_names(extra_bands)
    filled, valid_mask, fill_values = _fill_magnitude_columns(
        torch.as_tensor(magnitudes, dtype=torch.float32),
        torch.as_tensor(valid, dtype=torch.bool),
        invalid_fill=invalid_fill,
    )
    if filled.shape[1] != len(selected_bands):
        raise ValueError(
            f"Expected {len(selected_bands)} extra-band columns, got {filled.shape[1]}."
        )

    columns = [filled]
    feature_names = [extra_band_feature_name(band) for band in selected_bands]
    if include_valid_flags:
        columns.append(valid_mask.float())
        feature_names.extend(extra_band_valid_feature_name(band) for band in selected_bands)

    metadata = {
        "extra_bands": list(selected_bands),
        "extra_band_labels": [EXTRA_BAND_LABELS[band] for band in selected_bands],
        "extra_band_feature_names": list(feature_names),
        "extra_band_invalid_fill": invalid_fill,
        "extra_band_fill_values": {
            band: float(value)
            for band, value in zip(selected_bands, fill_values, strict=True)
        },
        "extra_band_valid_counts": {
            band: int(valid_mask[:, idx].sum().item())
            for idx, band in enumerate(selected_bands)
        },
        "extra_band_invalid_counts": {
            band: int((~valid_mask[:, idx]).sum().item())
            for idx, band in enumerate(selected_bands)
        },
        "extra_band_include_valid_flags": bool(include_valid_flags),
    }
    return torch.cat(columns, dim=1), feature_names, metadata

def build_extra_band_feature_matrix_from_table(
    table,
    *,
    rows=None,
    extra_bands: Sequence[str] | None = None,
    mag_zero_point: float = 23.0,
    invalid_fill: str | float = "median",
    include_valid_flags: bool = False,
    warn_missing: bool = True,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    magnitudes, valid, selected_bands = extract_extra_band_magnitudes_from_table(
        table,
        extra_bands=extra_bands,
        rows=rows,
        mag_zero_point=mag_zero_point,
        warn_missing=warn_missing,
    )
    return make_extra_band_feature_matrix(
        magnitudes,
        valid,
        extra_bands=selected_bands,
        invalid_fill=invalid_fill,
        include_valid_flags=include_valid_flags,
    )

def build_grizy_usable_mask_from_split_arrays(
    bands: np.ndarray,
    flags: np.ndarray,
    redshifts: np.ndarray,
    *,
    mag_zero_point: float,
    hsc_mag_faint_limits: Mapping[str, float | None] | None,
    target_redshift_column: str,
    z_min: float = 0.0,
    z_max: float = 6.0,
) -> np.ndarray:
    """Rebuild the cached AION row mask without consulting extra bands."""
    n_rows = len(bands)
    mask = np.ones(n_rows, dtype=bool)
    band_names = set(bands.dtype.names or ())
    flag_names = set(flags.dtype.names or ())

    for band in HSC_AION_BANDS:
        field_name = f"flux_cmodel_{band}"
        if field_name not in band_names:
            raise KeyError(f"Missing required HSC split-cache field: {field_name}")
        mag, valid = flux_to_ab_mag(
            bands[field_name],
            mag_zero_point=mag_zero_point,
        )
        mask &= valid
        if hsc_mag_faint_limits is not None:
            limit = hsc_mag_faint_limits.get(band)
            if limit is not None:
                mask &= np.isfinite(mag) & (mag <= float(limit))

        for flag_prefix in ("is_no_data", "not_observed", "has_bad_photometry"):
            flag_name = f"{flag_prefix}_{band}"
            if flag_name in flag_names:
                mask &= ~flags[flag_name].astype(bool)

    z_name = _split_redshift_name(target_redshift_column)
    z_values = np.asarray(redshifts[z_name], dtype=np.float32)
    mask &= np.isfinite(z_values) & (z_values >= z_min) & (z_values <= z_max)
    return mask

def load_extra_band_magnitudes_from_split_cache(
    product: Mapping[str, Any],
    *,
    extra_bands: Sequence[str] | None = None,
    split_output_dir: str | Path | None = None,
    mag_zero_point: float | None = None,
    hsc_mag_faint_limits: Mapping[str, float | None] | None = None,
    target_redshift_column: str | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    verify_object_ids: bool = True,
    warn_missing: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Load selected extra-band magnitudes aligned to cached product rows."""
    metadata = dict(product.get("metadata", {}))
    split_dir = Path(split_output_dir or metadata["split_output_dir"])
    mag_zero_point = float(mag_zero_point if mag_zero_point is not None else metadata["mag_zero_point"])
    if hsc_mag_faint_limits is None:
        hsc_mag_faint_limits = metadata.get("hsc_mag_faint_limits")
    target_redshift_column = target_redshift_column or metadata.get("target_redshift_column", "ZPHOT")
    z_min = float(z_min if z_min is not None else metadata.get("z_min", 0.0))
    z_max = float(z_max if z_max is not None else metadata.get("z_max", 6.0))

    bands = np.load(split_dir / "clauds_bands.npy", mmap_mode="r")
    flags = np.load(split_dir / "clauds_flags.npy", mmap_mode="r")
    redshifts = np.load(split_dir / "clauds_redshifts.npy", mmap_mode="r")

    usable_mask = build_grizy_usable_mask_from_split_arrays(
        bands,
        flags,
        redshifts,
        mag_zero_point=mag_zero_point,
        hsc_mag_faint_limits=hsc_mag_faint_limits,
        target_redshift_column=target_redshift_column,
        z_min=z_min,
        z_max=z_max,
    )
    magnitudes, valid, selected_bands = extract_extra_band_magnitudes_from_split_arrays(
        bands[usable_mask],
        flags[usable_mask],
        extra_bands=extra_bands,
        mag_zero_point=mag_zero_point,
        warn_missing=warn_missing,
    )

    n_product = int(product["aion_embedding"].shape[0])
    if len(magnitudes) != n_product:
        raise ValueError(
            "Rebuilt usable mask does not match cached product rows: "
            f"{len(magnitudes)} from split cache vs {n_product} in product."
        )

    if verify_object_ids and "object_id" in product:
        cached_ids = np.asarray(product["object_id"], dtype=np.int64)
        split_ids = np.asarray(bands["id"][usable_mask], dtype=np.int64)
        if cached_ids.shape == split_ids.shape and not np.array_equal(cached_ids, split_ids):
            raise ValueError("Cached product object_id order does not match rebuilt split-cache order.")

    return torch.from_numpy(magnitudes), torch.from_numpy(valid), selected_bands

def make_no_extra_feature_product(
    product: Mapping[str, Any],
    *,
    label: str = "no_extra",
) -> dict[str, Any]:
    """Return a cached-product copy with no extra MLP inputs."""
    output = dict(product)
    output["extra_features"] = product["extra_features"][:, :0].clone()
    output["feature_names"] = []

    metadata = dict(product.get("metadata", {}))
    metadata["feature_names"] = []
    metadata["extra_band_training"] = {
        "label": label,
        "include_extra_bands": False,
        "no_extra_features": True,
        "n_extra_features": 0,
    }
    output["metadata"] = metadata
    return output

def make_extra_band_product(
    product: Mapping[str, Any],
    magnitudes: torch.Tensor | np.ndarray,
    valid: torch.Tensor | np.ndarray,
    *,
    extra_bands: Sequence[str],
    label: str = "with_extra",
    invalid_fill: str | float = "median",
    include_valid_flags: bool = False,
) -> dict[str, Any]:
    """Return a cached-product copy whose extra features are selected band mags."""
    extra_features, feature_names, extra_metadata = make_extra_band_feature_matrix(
        magnitudes,
        valid,
        extra_bands=extra_bands,
        invalid_fill=invalid_fill,
        include_valid_flags=include_valid_flags,
    )
    n_rows = int(product["aion_embedding"].shape[0])
    if extra_features.shape[0] != n_rows:
        raise ValueError(f"extra feature matrix has {extra_features.shape[0]} rows, expected {n_rows}.")

    output = dict(product)
    output["extra_features"] = extra_features
    output["feature_names"] = feature_names

    metadata = dict(product.get("metadata", {}))
    metadata["feature_names"] = feature_names
    metadata["extra_band_training"] = {
        "label": label,
        "include_extra_bands": True,
        "n_extra_features": len(feature_names),
        **extra_metadata,
    }
    output["metadata"] = metadata
    return output

def metric_row_from_training_result(
    result: Mapping[str, Any],
    *,
    split: str = "val",
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> dict[str, float]:
    """Extract a compact numeric metric row from train_single_baseline output."""
    metrics = result["final_metrics"][split]
    return {
        key: float(metrics[key])
        for key in metric_keys
        if key in metrics
    }

def summarize_extra_band_ablation(
    results: Mapping[str, Mapping[str, Any]],
    *,
    split: str = "val",
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> dict[str, Any]:
    """Summarize extra-band runs, including with-minus-no-extra deltas."""
    rows = {
        label: metric_row_from_training_result(
            result,
            split=split,
            metric_keys=metric_keys,
        )
        for label, result in results.items()
    }
    summary: dict[str, Any] = {"split": split, "rows": rows}

    if "with_extra" in rows and "no_extra" in rows:
        shared_keys = sorted(set(rows["with_extra"]).intersection(rows["no_extra"]))
        summary["delta_with_minus_no_extra"] = {
            key: rows["with_extra"][key] - rows["no_extra"][key]
            for key in shared_keys
        }

    if "extra_only" in rows and "no_extra" in rows:
        shared_keys = sorted(set(rows["extra_only"]).intersection(rows["no_extra"]))
        summary["delta_extra_only_minus_no_extra"] = {
            key: rows["extra_only"][key] - rows["no_extra"][key]
            for key in shared_keys
        }

    return summary

def format_extra_band_ablation_summary(
    summary: Mapping[str, Any],
    *,
    metric_keys: Sequence[str] = (
        "nmad",
        "catastrophic_outlier_fraction",
        "cross_entropy",
        "mean_crps",
        "p16_p84_coverage",
        "pit_mean",
    ),
    precision: int = 5,
) -> str:
    """Plain-text metric table; avoids pandas so it can be pasted anywhere."""
    rows = summary["rows"]
    labels = list(rows.keys())
    header = ["variant", *metric_keys]
    rendered = [" | ".join(header)]
    rendered.append(" | ".join(["---"] * len(header)))

    for label in labels:
        metric_row = rows[label]
        values = [label]
        for key in metric_keys:
            value = metric_row.get(key)
            values.append("" if value is None else f"{value:.{precision}f}")
        rendered.append(" | ".join(values))

    for delta_label in ("delta_with_minus_no_extra", "delta_extra_only_minus_no_extra"):
        deltas = summary.get(delta_label)
        if deltas:
            values = [delta_label]
            for key in metric_keys:
                value = deltas.get(key)
                values.append("" if value is None else f"{value:.{precision}f}")
            rendered.append(" | ".join(values))

    return "\n".join(rendered)

def run_extra_band_ablation(
    product: Mapping[str, Any] | str | Path,
    *,
    train_single_baseline: Callable[..., Mapping[str, Any]],
    output_dir: str | Path,
    extra_bands: Sequence[str] | None = None,
    magnitudes: torch.Tensor | np.ndarray | None = None,
    valid: torch.Tensor | np.ndarray | None = None,
    split_output_dir: str | Path | None = None,
    require_valid_extra_bands: bool = False,
    invalid_fill: str | float = "median",
    include_valid_flags: bool = False,
    no_extra_model_kind: str = "aion",
    with_extra_model_kind: str = "fusion",
    extra_only_model_kind: str = "tabular",
    variant_order: Sequence[str] = ("no_extra", "with_extra"),
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Train matched no-extra, AION+extra, and optional extra-only baselines."""
    if isinstance(product, (str, Path)):
        product = load_cached_product(product)

    selected_bands = resolve_extra_band_names(extra_bands)
    if magnitudes is None:
        magnitudes, valid, selected_bands = load_extra_band_magnitudes_from_split_cache(
            product,
            extra_bands=selected_bands,
            split_output_dir=split_output_dir,
        )
    else:
        magnitudes = torch.as_tensor(magnitudes, dtype=torch.float32)
        if magnitudes.ndim == 1:
            magnitudes = magnitudes[:, None]
        if valid is None:
            valid = torch.isfinite(magnitudes)
        else:
            valid = torch.as_tensor(valid, dtype=torch.bool)

    magnitudes = torch.as_tensor(magnitudes, dtype=torch.float32)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if magnitudes.ndim == 1:
        magnitudes = magnitudes[:, None]
    if valid.ndim == 1:
        valid = valid[:, None]
    if magnitudes.shape != valid.shape:
        raise ValueError("magnitudes and valid must have the same shape.")
    if magnitudes.shape[0] != int(product["aion_embedding"].shape[0]):
        raise ValueError(
            f"magnitudes have {magnitudes.shape[0]} rows, "
            f"expected {int(product['aion_embedding'].shape[0])}."
        )

    if require_valid_extra_bands:
        keep = valid.all(dim=1) & torch.isfinite(magnitudes).all(dim=1)
        product = subset_product_rows(product, keep)
        magnitudes = magnitudes[keep]
        valid = valid[keep]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    products = {}
    for label in variant_order:
        if label not in {"no_extra", "with_extra", "extra_only"}:
            raise ValueError(f"Unknown extra-band ablation variant: {label!r}")

        if label == "no_extra":
            variant_product = make_no_extra_feature_product(product, label=label)
            model_kind = no_extra_model_kind
        else:
            variant_product = make_extra_band_product(
                product,
                magnitudes,
                valid,
                extra_bands=selected_bands,
                label=label,
                invalid_fill=invalid_fill,
                include_valid_flags=include_valid_flags,
            )
            model_kind = with_extra_model_kind if label == "with_extra" else extra_only_model_kind
        products[label] = variant_product

        variant_output_dir = output_dir / label
        train_kwargs = {
            "output_dir": variant_output_dir,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
        }
        if device is not None:
            train_kwargs["device"] = torch.device(device)

        results[label] = train_single_baseline(
            variant_product,
            model_kind,
            **train_kwargs,
        )

    return {
        "results": results,
        "products": products,
        "summary": summarize_extra_band_ablation(results),
        "output_dir": str(output_dir),
        "model_kinds": {
            "no_extra": no_extra_model_kind,
            "with_extra": with_extra_model_kind,
            "extra_only": extra_only_model_kind,
        },
        "extra_bands": {
            "selected_bands": list(selected_bands),
            "require_valid_extra_bands": bool(require_valid_extra_bands),
            "invalid_fill": invalid_fill,
            "include_valid_flags": bool(include_valid_flags),
            "n_rows": int(magnitudes.shape[0]),
            "valid_counts": {
                band: int(valid[:, idx].sum().item())
                for idx, band in enumerate(selected_bands)
            },
            "invalid_counts": {
                band: int((~valid[:, idx]).sum().item())
                for idx, band in enumerate(selected_bands)
            },
        },
    }


run_extra_bands_ablation = run_extra_band_ablation
summarize_extra_bands_ablation = summarize_extra_band_ablation
format_extra_bands_ablation_summary = format_extra_band_ablation_summary


def load_u_magnitude_from_split_cache(
    product: Mapping[str, Any],
    *,
    split_output_dir: str | Path | None = None,
    mag_zero_point: float | None = None,
    hsc_mag_faint_limits: Mapping[str, float | None] | None = None,
    target_redshift_column: str | None = None,
    verify_object_ids: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load u_mag aligned to rows in a cached AION product.

    The cached product does not store raw grizy/u magnitudes. This function
    reloads the split CLAUDS arrays, rebuilds the same grizy-based usable mask
    used for the AION cache, and returns u_mag for those usable rows.
    """
    metadata = dict(product.get("metadata", {}))
    split_dir = Path(split_output_dir or metadata["split_output_dir"])
    mag_zero_point = float(mag_zero_point if mag_zero_point is not None else metadata["mag_zero_point"])
    if hsc_mag_faint_limits is None:
        hsc_mag_faint_limits = metadata.get("hsc_mag_faint_limits")
    target_redshift_column = target_redshift_column or metadata.get("target_redshift_column", "ZPHOT")

    bands = np.load(split_dir / "clauds_bands.npy", mmap_mode="r")
    flags = np.load(split_dir / "clauds_flags.npy", mmap_mode="r")
    redshifts = np.load(split_dir / "clauds_redshifts.npy", mmap_mode="r")

    usable_mask = build_grizy_usable_mask_from_split_arrays(
        bands,
        flags,
        redshifts,
        mag_zero_point=mag_zero_point,
        hsc_mag_faint_limits=hsc_mag_faint_limits,
        target_redshift_column=target_redshift_column,
    )
    u_mag, valid_u = flux_to_ab_mag(
        bands["flux_cmodel_u"],
        mag_zero_point=mag_zero_point,
    )
    u_mag = u_mag[usable_mask].astype(np.float32)
    valid_u = valid_u[usable_mask].astype(np.bool_)

    n_product = int(product["aion_embedding"].shape[0])
    if len(u_mag) != n_product:
        raise ValueError(
            "Rebuilt usable mask does not match cached product rows: "
            f"{len(u_mag)} from split cache vs {n_product} in product."
        )

    if verify_object_ids and "object_id" in product:
        cached_ids = np.asarray(product["object_id"], dtype=np.int64)
        split_ids = np.asarray(bands["id"][usable_mask], dtype=np.int64)
        if cached_ids.shape == split_ids.shape and not np.array_equal(cached_ids, split_ids):
            raise ValueError("Cached product object_id order does not match rebuilt split-cache order.")

    return torch.from_numpy(u_mag), torch.from_numpy(valid_u)

def make_no_u_feature_product(
    product: Mapping[str, Any],
    *,
    label: str = "no_u",
) -> dict[str, Any]:
    """Return a cached-product copy with no extra MLP inputs for u ablation."""
    output = dict(product)
    output["extra_features"] = product["extra_features"][:, :0].clone()
    output["feature_names"] = []

    metadata = dict(product.get("metadata", {}))
    metadata["feature_names"] = []
    metadata["u_magnitude_ablation"] = {
        "label": label,
        "include_u_magnitude": False,
        "no_extra_features": True,
        "n_extra_features": 0,
    }
    output["metadata"] = metadata
    return output


def _filled_u_magnitude(
    u_magnitude: torch.Tensor,
    valid_u: torch.Tensor | None = None,
    *,
    invalid_fill: str | float = "median",
) -> tuple[torch.Tensor, torch.Tensor, float]:
    u = torch.as_tensor(u_magnitude, dtype=torch.float32).reshape(-1)
    finite = torch.isfinite(u)
    valid = finite if valid_u is None else (torch.as_tensor(valid_u, dtype=torch.bool).reshape(-1) & finite)
    if u.numel() != valid.numel():
        raise ValueError("u_magnitude and valid_u must have the same length.")

    if isinstance(invalid_fill, (int, float)):
        fill_value = float(invalid_fill)
    elif invalid_fill == "median":
        fill_value = float(torch.median(u[valid]).item()) if bool(valid.any()) else 0.0
    elif invalid_fill == "max_valid":
        fill_value = float(torch.max(u[valid]).item()) if bool(valid.any()) else 0.0
    else:
        raise ValueError("invalid_fill must be 'median', 'max_valid', or a numeric value.")

    fill_tensor = torch.full_like(u, fill_value)
    return torch.where(valid, u, fill_tensor), valid, fill_value

def make_u_magnitude_product(
    product: Mapping[str, Any],
    u_magnitude: torch.Tensor | np.ndarray | Sequence[float],
    *,
    valid_u: torch.Tensor | np.ndarray | Sequence[bool] | None = None,
    label: str = "with_u",
    invalid_fill: str | float = "median",
    include_valid_flag: bool = False,
) -> dict[str, Any]:
    """Return a cached-product copy with u_mag as the only extra MLP input."""
    u_filled, valid, fill_value = _filled_u_magnitude(
        torch.as_tensor(u_magnitude, dtype=torch.float32),
        None if valid_u is None else torch.as_tensor(valid_u, dtype=torch.bool),
        invalid_fill=invalid_fill,
    )
    n_rows = int(product["aion_embedding"].shape[0])
    if u_filled.numel() != n_rows:
        raise ValueError(f"u_magnitude has {u_filled.numel()} rows, expected {n_rows}.")

    columns = [u_filled[:, None]]
    feature_names = ["u_mag"]
    if include_valid_flag:
        columns.append(valid.float()[:, None])
        feature_names.append("u_mag_valid")

    output = dict(product)
    output["extra_features"] = torch.cat(columns, dim=1)
    output["feature_names"] = feature_names

    metadata = dict(product.get("metadata", {}))
    metadata["feature_names"] = feature_names
    metadata["u_magnitude_ablation"] = {
        "label": label,
        "include_u_magnitude": True,
        "n_extra_features": len(feature_names),
        "invalid_fill": invalid_fill,
        "invalid_fill_value": fill_value,
        "n_valid_u": int(valid.sum().item()),
        "n_invalid_u": int((~valid).sum().item()),
        "include_valid_flag": bool(include_valid_flag),
    }
    output["metadata"] = metadata
    return output

def summarize_u_magnitude_ablation(
    results: Mapping[str, Mapping[str, Any]],
    *,
    split: str = "val",
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> dict[str, Any]:
    """Summarize with-u and no-u runs, including with-minus-no-u deltas."""
    rows = {
        label: metric_row_from_training_result(
            result,
            split=split,
            metric_keys=metric_keys,
        )
        for label, result in results.items()
    }
    summary: dict[str, Any] = {"split": split, "rows": rows}

    if "with_u" in rows and "no_u" in rows:
        shared_keys = sorted(set(rows["with_u"]).intersection(rows["no_u"]))
        summary["delta_with_minus_no_u"] = {
            key: rows["with_u"][key] - rows["no_u"][key]
            for key in shared_keys
        }

    return summary

def format_u_magnitude_ablation_summary(
    summary: Mapping[str, Any],
    *,
    metric_keys: Sequence[str] = (
        "nmad",
        "catastrophic_outlier_fraction",
        "cross_entropy",
        "mean_crps",
        "p16_p84_coverage",
        "pit_mean",
    ),
    precision: int = 5,
) -> str:
    """Plain-text metric table; avoids pandas so it can be pasted anywhere."""
    rows = summary["rows"]
    labels = list(rows.keys())
    header = ["variant", *metric_keys]
    rendered = [" | ".join(header)]
    rendered.append(" | ".join(["---"] * len(header)))

    for label in labels:
        metric_row = rows[label]
        values = [label]
        for key in metric_keys:
            value = metric_row.get(key)
            values.append("" if value is None else f"{value:.{precision}f}")
        rendered.append(" | ".join(values))

    deltas = summary.get("delta_with_minus_no_u")
    delta_label = "delta_with_minus_no_u"
    if deltas:
        values = [delta_label]
        for key in metric_keys:
            value = deltas.get(key)
            values.append("" if value is None else f"{value:.{precision}f}")
        rendered.append(" | ".join(values))

    return "\n".join(rendered)

def run_u_magnitude_ablation(
    product: Mapping[str, Any] | str | Path,
    *,
    train_single_baseline: Callable[..., Mapping[str, Any]],
    output_dir: str | Path,
    u_magnitude: torch.Tensor | np.ndarray | Sequence[float] | None = None,
    valid_u: torch.Tensor | np.ndarray | Sequence[bool] | None = None,
    split_output_dir: str | Path | None = None,
    require_valid_u: bool = False,
    invalid_fill: str | float = "median",
    include_valid_flag: bool = False,
    no_u_model_kind: str = "aion",
    with_u_model_kind: str = "fusion",
    variant_order: Sequence[str] = ("no_u", "with_u"),
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Train matched AION-only and AION+u_mag baselines.

    no_u uses the notebook's AION-only model, so the MLP sees no u-band
    information. with_u uses a late-fusion model whose extra branch receives
    u_mag as the only extra input.
    """
    if isinstance(product, (str, Path)):
        product = load_cached_product(product)

    if u_magnitude is None:
        u_magnitude, valid_u = load_u_magnitude_from_split_cache(
            product,
            split_output_dir=split_output_dir,
        )
    else:
        u_magnitude = torch.as_tensor(u_magnitude, dtype=torch.float32).reshape(-1)
        if valid_u is not None:
            valid_u = torch.as_tensor(valid_u, dtype=torch.bool).reshape(-1)

    if valid_u is None:
        valid_u = torch.isfinite(torch.as_tensor(u_magnitude, dtype=torch.float32).reshape(-1))
    else:
        valid_u = torch.as_tensor(valid_u, dtype=torch.bool).reshape(-1)

    u_magnitude = torch.as_tensor(u_magnitude, dtype=torch.float32).reshape(-1)
    if u_magnitude.numel() != int(product["aion_embedding"].shape[0]):
        raise ValueError(
            f"u_magnitude has {u_magnitude.numel()} rows, "
            f"expected {int(product['aion_embedding'].shape[0])}."
        )

    if require_valid_u:
        keep = valid_u & torch.isfinite(u_magnitude)
        product = subset_product_rows(product, keep)
        u_magnitude = u_magnitude[keep]
        valid_u = valid_u[keep]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    products = {}
    for label in variant_order:
        if label not in {"no_u", "with_u"}:
            raise ValueError(f"Unknown u-band ablation variant: {label!r}")

        if label == "no_u":
            variant_product = make_no_u_feature_product(product, label=label)
            model_kind = no_u_model_kind
        else:
            variant_product = make_u_magnitude_product(
                product,
                u_magnitude,
                valid_u=valid_u,
                label=label,
                invalid_fill=invalid_fill,
                include_valid_flag=include_valid_flag,
            )
            model_kind = with_u_model_kind
        products[label] = variant_product

        variant_output_dir = output_dir / label
        train_kwargs = {
            "output_dir": variant_output_dir,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
        }
        if device is not None:
            train_kwargs["device"] = torch.device(device)

        results[label] = train_single_baseline(
            variant_product,
            model_kind,
            **train_kwargs,
        )

    return {
        "results": results,
        "products": products,
        "summary": summarize_u_magnitude_ablation(results),
        "output_dir": str(output_dir),
        "model_kinds": {
            "no_u": no_u_model_kind,
            "with_u": with_u_model_kind,
        },
        "u_magnitude": {
            "require_valid_u": bool(require_valid_u),
            "invalid_fill": invalid_fill,
            "include_valid_flag": bool(include_valid_flag),
            "n_rows": int(u_magnitude.numel()),
            "n_valid_u": int(valid_u.sum().item()),
            "n_invalid_u": int((~valid_u).sum().item()),
        },
    }


run_u_band_ablation = run_u_magnitude_ablation
summarize_u_band_ablation = summarize_u_magnitude_ablation
format_u_band_ablation_summary = format_u_magnitude_ablation_summary

