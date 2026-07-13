from __future__ import annotations
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

from .clauds_bands import (
    ALL_BAND_ERROR_COLUMNS, ALL_BAND_FLUX_COLUMNS, ALL_FLAG_COLUMNS,
    BAND_ERROR_COLUMNS, BAND_FLUX_COLUMNS, FLAG_COLUMNS, REDSHIFT_COLUMNS,
    OBJECT_ID_COLUMN, RA_COLUMN, DEC_COLUMN, TRACT_COLUMN, PATCH_COLUMN,
    HSC_AION_BANDS, split_clauds_catalogue, validate_clauds_fits_table,
    default_hsc_mag_faint_limits,
)
from .utils import (
    flux_to_ab_mag, require_columns, table_column_names, table_length,
    table_column, numeric_table_column, string_table_column,
    apply_numpy_mask_to_tensor_dict, select_torch_device, validate_split_fractions,
    finite_scale, asinh_transform,
)
from .extra_bands import (
    build_extra_band_feature_matrix_from_table, resolve_extra_band_names,
    DEFAULT_EXTRA_BANDS
)
from .metrics import normalize_redshift_reference, build_redshift_reference_from_table
from .models import (
    load_aion_mag_adjustment,
    apply_aion_mag_adjustment_to_hsc_features,
    build_aion_mag_adjustment_source_matrix_from_table,
    aion_mag_adjustment_metadata,
)


class CLAUDSSplitCatalogue(Mapping[str, np.ndarray]):
    """Mapping view over arrays produced by clauds_bands.split_clauds_catalogue."""

    def __init__(
        self,
        bands: np.ndarray,
        errors: np.ndarray,
        redshifts: np.ndarray,
        flags: np.ndarray,
    ):
        self.bands = bands
        self.errors = errors
        self.redshifts = redshifts
        self.flags = flags
        self._columns = self._build_columns()

    def _build_columns(self) -> dict[str, np.ndarray]:
        columns: dict[str, np.ndarray] = {
            OBJECT_ID_COLUMN: self.bands["id"],
            RA_COLUMN: self.bands["ra"],
            DEC_COLUMN: self.bands["dec"],
            TRACT_COLUMN: self.bands["tract"],
            PATCH_COLUMN: self.bands["patch"],
        }

        for band, fits_name in ALL_BAND_FLUX_COLUMNS.items():
            columns[fits_name] = self.bands[f"flux_cmodel_{band}"]

        for band, fits_name in ALL_BAND_ERROR_COLUMNS.items():
            columns[fits_name] = self.errors[f"fluxerr_cmodel_{band}"]

        for out_name, fits_name in REDSHIFT_COLUMNS.items():
            columns[fits_name] = self.redshifts[out_name]

        for out_name, fits_name in ALL_FLAG_COLUMNS.items():
            columns[fits_name] = self.flags[out_name]

        return columns

    def __getitem__(self, key: str) -> np.ndarray:
        return self._columns[key]

    def __iter__(self):
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def keys(self):
        return self._columns.keys()


def build_hsc_aion_features_from_table(
    table,
    rows=None,
    mag_zero_point: float = 23.0,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    require_columns(table, [BAND_FLUX_COLUMNS[band] for band in HSC_AION_BANDS])

    hsc_features: dict[str, torch.Tensor] = {}
    valid_mask = None

    for band in HSC_AION_BANDS:
        flux = numeric_table_column(table, BAND_FLUX_COLUMNS[band], rows=rows)
        mag, valid = flux_to_ab_mag(flux, mag_zero_point=mag_zero_point)
        hsc_features[f"{band}_mag"] = torch.from_numpy(mag.astype(np.float32))
        valid_mask = valid if valid_mask is None else (valid_mask & valid)

    return hsc_features, valid_mask


def build_hsc_magnitude_faint_end_mask(
    hsc_features: Mapping[str, torch.Tensor],
    faint_limits: Mapping[str, float | None] | None = None,
) -> np.ndarray:
    """Keep objects brighter than configured grizy faint-end magnitude limits."""
    faint_limits = faint_limits or default_hsc_mag_faint_limits()
    first_band = HSC_AION_BANDS[0]
    n_rows = len(hsc_features[f"{first_band}_mag"])
    mask = np.ones(n_rows, dtype=bool)

    for band in HSC_AION_BANDS:
        limit = faint_limits.get(band)
        if limit is None:
            continue
        mag = hsc_features[f"{band}_mag"].detach().cpu().numpy()
        mask &= np.isfinite(mag) & (mag <= float(limit))

    return mask


def build_field_labels(table, rows=None, field_column: str | None = None) -> np.ndarray:
    names = table_column_names(table)
    if field_column is not None:
        require_columns(table, [field_column])
        return string_table_column(table, field_column, rows=rows)
    if "field" in names:
        return string_table_column(table, "field", rows=rows)
    if "FIELD" in names:
        return string_table_column(table, "FIELD", rows=rows)
    if TRACT_COLUMN in names:
        return np.char.add("tract_", string_table_column(table, TRACT_COLUMN, rows=rows))
    return np.asarray(["unknown"] * (table_length(table) if rows is None else len(table_column(table, OBJECT_ID_COLUMN, rows=rows))))


def add_numeric_feature(
    arrays: list[np.ndarray],
    names: list[str],
    values: np.ndarray,
    feature_name: str,
    *,
    transform_asinh: bool = False,
    add_missing_flag: bool = True,
    scale: float | None = None,
) -> None:
    values = np.asarray(values, dtype=np.float32)
    missing = ~np.isfinite(values)
    clean_values = values.copy()
    clean_values[missing] = 0.0

    if transform_asinh:
        feature_scale = finite_scale(clean_values) if scale is None else scale
        clean_values = asinh_transform(clean_values, feature_scale)
        feature_name = f"asinh_{feature_name}"

    arrays.append(clean_values.astype(np.float32))
    names.append(feature_name)

    if add_missing_flag:
        arrays.append(missing.astype(np.float32))
        names.append(f"{feature_name}_missing")


def build_extra_feature_matrix_from_table(
    table,
    rows=None,
    include_flags: bool = True,
    strict_optional_columns: bool = False,
    extra_bands: Sequence[str] | None = None,
    mag_zero_point: float = 23.0,
    invalid_fill: str | float = "median",
    include_valid_flags: bool = False,
    return_metadata: bool = False,
) -> tuple[torch.Tensor, list[str]] | tuple[torch.Tensor, list[str], dict[str, Any]]:
    del include_flags
    extra_features, feature_names, metadata = build_extra_band_feature_matrix_from_table(
        table,
        rows=rows,
        extra_bands=extra_bands,
        mag_zero_point=mag_zero_point,
        invalid_fill=invalid_fill,
        include_valid_flags=include_valid_flags,
        warn_missing=not strict_optional_columns,
    )
    if strict_optional_columns:
        valid_counts = metadata.get("extra_band_valid_counts", {})
        missing = [band for band, count in valid_counts.items() if int(count) == 0]
        if missing:
            raise KeyError(f"Selected extra bands have no valid rows: {missing}")
    if return_metadata:
        return extra_features, feature_names, metadata
    return extra_features, feature_names


def resolve_include_grizy_in_mlp(
    include_grizy_in_mlp: bool | None,
    *,
    use_aion_embedding: bool,
    use_mlp_features: bool = True,
    extra_bands: Sequence[str] | None = None,
) -> bool:
    """Resolve the grizy-as-tabular default at the point of use."""
    if not use_mlp_features:
        return False
    if include_grizy_in_mlp is None:
        return not use_aion_embedding
    include = bool(include_grizy_in_mlp)
    if not include and not use_aion_embedding and not resolve_extra_band_names(extra_bands):
        warnings.warn(
            "No AION embedding and no extra bands were selected; enabling grizy MLP "
            "features so tabular training has inputs.",
            RuntimeWarning,
            stacklevel=2,
        )
        return True
    return include


def build_grizy_mlp_feature_matrix(
    hsc_features: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, list[str]]:
    """Return grizy magnitudes as ordinary MLP/tabular features."""
    columns = [
        torch.as_tensor(hsc_features[f"{band}_mag"], dtype=torch.float32).reshape(-1)
        for band in HSC_AION_BANDS
    ]
    feature_names = [f"{band}_mag" for band in HSC_AION_BANDS]
    return torch.stack(columns, dim=1), feature_names


def build_hsc_quality_mask_from_table(table, rows=None) -> np.ndarray:
    names = table_column_names(table)
    n_rows = table_length(table) if rows is None else len(table_column(table, OBJECT_ID_COLUMN, rows=rows))
    mask = np.ones(n_rows, dtype=bool)

    for band in HSC_AION_BANDS:
        for flag_prefix in ("is_no_data", "not_observed", "has_bad_photometry"):
            flag_name = f"{flag_prefix}_{band}"
            column_name = FLAG_COLUMNS.get(flag_name)
            if column_name is not None and column_name in names:
                mask &= ~table_column(table, column_name, rows=rows).astype(bool)

    return mask


@dataclass
class CLAUDSPhotoZBatch:
    object_id: list[Any]
    field: list[Any]
    hsc_batch: dict[str, torch.Tensor]
    extra_features: torch.Tensor
    z_spec: torch.Tensor | None = None
    redshift_reference: dict[str, torch.Tensor] | None = None


class CLAUDSPhotoZDataset(Dataset):
    def __init__(
        self,
        object_ids: Sequence[Any],
        fields: Sequence[Any],
        hsc_features: dict[str, torch.Tensor],
        extra_features: torch.Tensor,
        z_spec: torch.Tensor | None = None,
        redshift_reference: Mapping[str, torch.Tensor | np.ndarray] | None = None,
    ):
        self.object_ids = list(object_ids)
        self.fields = list(fields)
        self.hsc_features = hsc_features
        self.extra_features = extra_features.float()
        self.z_spec = None if z_spec is None else z_spec.float()
        self.redshift_reference = normalize_redshift_reference(redshift_reference)

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "object_id": self.object_ids[idx],
            "field": self.fields[idx],
            "hsc_batch": {key: value[idx] for key, value in self.hsc_features.items()},
            "extra_features": self.extra_features[idx],
        }
        if self.z_spec is not None:
            item["z_spec"] = self.z_spec[idx]
        if self.redshift_reference:
            item["redshift_reference"] = {
                key: value[idx]
                for key, value in self.redshift_reference.items()
            }
        return item


def collate_clauds_photoz(items: list[dict[str, Any]]) -> CLAUDSPhotoZBatch:
    hsc_keys = items[0]["hsc_batch"].keys()
    hsc_batch = {
        key: torch.stack([item["hsc_batch"][key] for item in items])
        for key in hsc_keys
    }
    z_spec = None
    if "z_spec" in items[0]:
        z_spec = torch.stack([item["z_spec"] for item in items]).float()
    redshift_reference = None
    if "redshift_reference" in items[0]:
        reference_keys = items[0]["redshift_reference"].keys()
        redshift_reference = {
            key: torch.stack([item["redshift_reference"][key] for item in items]).float()
            for key in reference_keys
        }

    return CLAUDSPhotoZBatch(
        object_id=[item["object_id"] for item in items],
        field=[item["field"] for item in items],
        hsc_batch=hsc_batch,
        extra_features=torch.stack([item["extra_features"] for item in items]).float(),
        z_spec=z_spec,
        redshift_reference=redshift_reference,
    )


@dataclass
class CachedFusionBatch:
    object_id: list[Any]
    field: list[Any]
    aion_embedding: torch.Tensor
    extra_features: torch.Tensor
    z_spec: torch.Tensor | None = None
    redshift_reference: dict[str, torch.Tensor] | None = None


class CachedFusionDataset(Dataset):
    def __init__(
        self,
        object_ids: Sequence[Any],
        fields: Sequence[Any],
        aion_embeddings: torch.Tensor,
        extra_features: torch.Tensor,
        z_spec: torch.Tensor | None = None,
        redshift_reference: Mapping[str, torch.Tensor | np.ndarray] | None = None,
    ):
        self.object_ids = list(object_ids)
        self.fields = list(fields)
        self.aion_embeddings = aion_embeddings.float()
        self.extra_features = extra_features.float()
        self.z_spec = None if z_spec is None else z_spec.float()
        self.redshift_reference = normalize_redshift_reference(redshift_reference)

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "object_id": self.object_ids[idx],
            "field": self.fields[idx],
            "aion_embedding": self.aion_embeddings[idx],
            "extra_features": self.extra_features[idx],
        }
        if self.z_spec is not None:
            item["z_spec"] = self.z_spec[idx]
        if self.redshift_reference:
            item["redshift_reference"] = {
                key: value[idx]
                for key, value in self.redshift_reference.items()
            }
        return item


def collate_cached_fusion(items: list[dict[str, Any]]) -> CachedFusionBatch:
    z_spec = None
    if "z_spec" in items[0]:
        z_spec = torch.stack([item["z_spec"] for item in items]).float()
    redshift_reference = None
    if "redshift_reference" in items[0]:
        reference_keys = items[0]["redshift_reference"].keys()
        redshift_reference = {
            key: torch.stack([item["redshift_reference"][key] for item in items]).float()
            for key in reference_keys
        }

    return CachedFusionBatch(
        object_id=[item["object_id"] for item in items],
        field=[item["field"] for item in items],
        aion_embedding=torch.stack([item["aion_embedding"] for item in items]).float(),
        extra_features=torch.stack([item["extra_features"] for item in items]).float(),
        z_spec=z_spec,
        redshift_reference=redshift_reference,
    )


def load_clauds_catalogue_from_fits(
    catalogue_path: str | Path,
    split_output_dir: str | Path,
    *,
    chunk_size: int = 250_000,
    overwrite: bool = False,
    max_rows: int | None = None,
    sample_mode: str = "head",
    row_start: int | None = None,
    row_stop: int | None = None,
    sample_seed: int = 42,
    sample_require_valid_bands: Sequence[str] = (),
) -> CLAUDSSplitCatalogue:
    split_output_dir = Path(split_output_dir)
    paths = {
        "bands": split_output_dir / "clauds_bands.npy",
        "errors": split_output_dir / "clauds_errors.npy",
        "redshifts": split_output_dir / "clauds_redshifts.npy",
        "flags": split_output_dir / "clauds_flags.npy",
    }
    if overwrite or not all(path.exists() for path in paths.values()) or not split_cache_matches_current_schema(paths):
        paths = split_clauds_catalogue(
            catalogue_path,
            split_output_dir,
            chunk_size=chunk_size,
            overwrite=True,
            max_rows=max_rows,
            sample_mode=sample_mode,
            row_start=row_start,
            row_stop=row_stop,
            sample_seed=sample_seed,
            require_valid_bands=sample_require_valid_bands,
        )
    arrays = {key: np.load(path, mmap_mode="r") for key, path in paths.items()}
    return CLAUDSSplitCatalogue(
        bands=arrays["bands"],
        errors=arrays["errors"],
        redshifts=arrays["redshifts"],
        flags=arrays["flags"],
    )


def clauds_redshift_filter_mask(
    redshift_values: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    z_min: float | None = 0.0,
    z_max: float | None = 6.0,
    include_min: bool = True,
    include_max: bool = True,
) -> np.ndarray:
    """Return a finite-redshift mask for CLAUDS catalogue filtering."""
    if isinstance(redshift_values, torch.Tensor):
        values = redshift_values.detach().cpu().numpy()
    else:
        values = np.asarray(redshift_values)
    values = values.astype(np.float64, copy=False)
    mask = np.isfinite(values)
    if z_min is not None:
        mask &= values >= float(z_min) if include_min else values > float(z_min)
    if z_max is not None:
        mask &= values <= float(z_max) if include_max else values < float(z_max)
    return mask


def build_raw_clauds_photoz_dataset(
    catalogue_path: str | Path,
    split_output_dir: str | Path,
    *,
    split_chunk_size: int = 250_000,
    overwrite_split_cache: bool = False,
    max_rows: int | None = 20_000,
    sample_mode: str = "head",
    sample_row_start: int | None = None,
    sample_row_stop: int | None = None,
    sample_seed: int = 42,
    sample_require_valid_bands: Sequence[str] = (),
    field_column: str | None = None,
    target_redshift_column: str = REDSHIFT_COLUMNS["zphot"],
    z_min: float = 0.0,
    z_max: float = 6.0,
    redshift_include_min: bool = True,
    redshift_include_max: bool = True,
    n_z_bins: int = 300,
    mag_zero_point: float = 23.0,
    hsc_mag_faint_limits: Mapping[str, float | None] | None = None,
    extra_bands: Sequence[str] | None = None,
    extra_band_invalid_fill: str | float = "median",
    extra_band_include_valid_flags: bool = False,
    use_mlp_features: bool = True,
    include_grizy_in_mlp: bool | None = None,
    use_aion_embedding: bool = True,
    aion_mag_adjustment_path: str | Path | None = None,
) -> tuple[CLAUDSPhotoZDataset, list[str], dict[str, Any]]:
    table = load_clauds_catalogue_from_fits(
        catalogue_path,
        split_output_dir,
        chunk_size=split_chunk_size,
        overwrite=overwrite_split_cache,
        max_rows=max_rows,
        sample_mode=sample_mode,
        row_start=sample_row_start,
        row_stop=sample_row_stop,
        sample_seed=sample_seed,
        sample_require_valid_bands=sample_require_valid_bands,
    )
    validate_clauds_fits_table(table, require_redshift=True)

    rows = slice(None)
    object_ids = table_column(table, OBJECT_ID_COLUMN, rows=rows).astype(np.int64)
    fields = build_field_labels(table, rows=rows, field_column=field_column)

    hsc_features, valid_hsc_mask = build_hsc_aion_features_from_table(
        table,
        rows=rows,
        mag_zero_point=mag_zero_point,
    )
    include_grizy_in_mlp = resolve_include_grizy_in_mlp(
        include_grizy_in_mlp,
        use_aion_embedding=use_aion_embedding,
        use_mlp_features=use_mlp_features,
        extra_bands=extra_bands,
    )
    if use_mlp_features:
        extra_features, feature_names, extra_feature_metadata = build_extra_feature_matrix_from_table(
            table,
            rows=rows,
            extra_bands=extra_bands,
            mag_zero_point=mag_zero_point,
            invalid_fill=extra_band_invalid_fill,
            include_valid_flags=extra_band_include_valid_flags,
            return_metadata=True,
        )
        grizy_feature_names: list[str] = []
        if include_grizy_in_mlp:
            grizy_features, grizy_feature_names = build_grizy_mlp_feature_matrix(hsc_features)
            extra_features = torch.cat([grizy_features, extra_features], dim=1)
            feature_names = grizy_feature_names + feature_names
        extra_feature_metadata.update({
            "use_mlp_features": True,
            "include_grizy_in_mlp": bool(include_grizy_in_mlp),
            "grizy_mlp_feature_names": grizy_feature_names,
        })
    else:
        n_rows = len(object_ids)
        extra_features = torch.empty((n_rows, 0), dtype=torch.float32)
        feature_names = []
        extra_feature_metadata = {
            "use_mlp_features": False,
            "include_grizy_in_mlp": False,
            "grizy_mlp_feature_names": [],
            "extra_bands": list(resolve_extra_band_names(extra_bands)),
            "extra_band_feature_names": [],
        }

    hsc_quality_mask = build_hsc_quality_mask_from_table(table, rows=rows)
    hsc_faint_mask = build_hsc_magnitude_faint_end_mask(hsc_features, hsc_mag_faint_limits)
    require_columns(table, [target_redshift_column])
    z_values = numeric_table_column(table, target_redshift_column, rows=rows)
    redshift_reference = build_redshift_reference_from_table(table, rows=rows)
    finite_z = clauds_redshift_filter_mask(
        z_values,
        z_min=z_min,
        z_max=z_max,
        include_min=redshift_include_min,
        include_max=redshift_include_max,
    )
    usable_mask = valid_hsc_mask & hsc_quality_mask & hsc_faint_mask & finite_z

    hsc_features = apply_numpy_mask_to_tensor_dict(hsc_features, usable_mask)
    mask_tensor = torch.as_tensor(usable_mask, dtype=torch.bool)
    extra_features = extra_features[mask_tensor]
    aion_adjustment_metadata: dict[str, Any] = {}
    if aion_mag_adjustment_path is not None:
        adjustment = load_aion_mag_adjustment(aion_mag_adjustment_path)
        source_features, _, source_metadata = build_aion_mag_adjustment_source_matrix_from_table(
            table,
            adjustment,
            rows=rows,
            mag_zero_point=mag_zero_point,
        )
        source_features = source_features[mask_tensor]
        hsc_features = apply_aion_mag_adjustment_to_hsc_features(
            hsc_features,
            source_features,
            adjustment,
        )
        aion_adjustment_metadata.update(
            aion_mag_adjustment_metadata(aion_mag_adjustment_path, adjustment)
        )
        aion_adjustment_metadata["aion_mag_adjustment_source_metadata"] = source_metadata
    z_spec = torch.from_numpy(z_values[usable_mask].astype(np.float32))
    redshift_reference = {
        key: values[mask_tensor]
        for key, values in redshift_reference.items()
    }
    object_ids = object_ids[usable_mask].tolist()
    fields = fields[usable_mask].tolist()

    dataset = CLAUDSPhotoZDataset(
        object_ids=object_ids,
        fields=fields,
        hsc_features=hsc_features,
        extra_features=extra_features,
        z_spec=z_spec,
        redshift_reference=redshift_reference,
    )
    metadata = {
        "catalogue_path": str(catalogue_path),
        "split_output_dir": str(split_output_dir),
        "max_rows": max_rows,
        "sample_mode": sample_mode,
        "sample_row_start": sample_row_start,
        "sample_row_stop": sample_row_stop,
        "sample_seed": sample_seed,
        "sample_require_valid_bands": list(sample_require_valid_bands),
        "n_usable_rows": len(dataset),
        "z_min": z_min,
        "z_max": z_max,
        "redshift_include_min": bool(redshift_include_min),
        "redshift_include_max": bool(redshift_include_max),
        "n_z_bins": n_z_bins,
        "mag_zero_point": mag_zero_point,
        "hsc_mag_faint_limits": dict(hsc_mag_faint_limits or default_hsc_mag_faint_limits()),
        "target_redshift_column": target_redshift_column,
        "redshift_reference_keys": sorted(redshift_reference),
        "feature_names": list(feature_names),
        **aion_adjustment_metadata,
        **extra_feature_metadata,
    }
    return dataset, feature_names, metadata


def make_field_aware_split(
    fields: Sequence[Any],
    test_fields: Sequence[Any],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    fields = np.asarray(fields)
    test_mask = np.isin(fields, np.asarray(test_fields))

    split = np.full(len(fields), "train", dtype=object)
    split[test_mask] = "test"

    train_indices = np.flatnonzero(~test_mask)
    rng = np.random.default_rng(seed)
    rng.shuffle(train_indices)

    n_val = int(round(val_fraction * len(train_indices)))
    if len(train_indices) > 0 and val_fraction > 0:
        n_val = max(1, n_val)
    val_indices = train_indices[:n_val]
    split[val_indices] = "val"

    return split


def split_counts_from_fractions(
    n_items: int,
    train_fraction: float,
    test_fraction: float,
    val_fraction: float,
) -> dict[str, int]:
    validate_split_fractions(train_fraction, test_fraction, val_fraction)
    if n_items < 0:
        raise ValueError("n_items must be non-negative.")

    labels = ("train", "test", "val")
    fractions = np.asarray([train_fraction, test_fraction, val_fraction], dtype=np.float64)
    raw_counts = fractions * n_items
    counts = np.floor(raw_counts).astype(int)
    remainder = int(n_items - counts.sum())
    if remainder > 0:
        order = np.argsort(raw_counts - counts)[::-1]
        counts[order[:remainder]] += 1

    return {label: int(count) for label, count in zip(labels, counts)}


def make_random_split(
    n_items: int,
    *,
    train_fraction: float = 0.20,
    test_fraction: float = 0.75,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Randomly assign rows to train/test/val using exact rounded counts."""
    counts = split_counts_from_fractions(n_items, train_fraction, test_fraction, val_fraction)
    split = np.empty(n_items, dtype=object)
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)

    start = 0
    for label in ("train", "test", "val"):
        stop = start + counts[label]
        split[indices[start:stop]] = label
        start = stop
    return split


def make_split_labels(
    fields: Sequence[Any],
    *,
    split_strategy: str = "random",
    train_fraction: float = 0.20,
    test_fraction: float = 0.75,
    val_fraction: float = 0.05,
    test_fields: Sequence[Any] = (),
    seed: int = 42,
) -> np.ndarray:
    if split_strategy == "random":
        return make_random_split(
            len(fields),
            train_fraction=train_fraction,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            seed=seed,
        )
    if split_strategy == "field":
        return make_field_aware_split(fields, test_fields=test_fields, val_fraction=val_fraction, seed=seed)
    raise ValueError("split_strategy must be 'random' or 'field'.")


def split_metadata(
    split_labels: Sequence[str],
    split_strategy: str,
    train_fraction: float,
    test_fraction: float,
    val_fraction: float,
) -> dict[str, Any]:
    labels, counts = np.unique(np.asarray(split_labels, dtype=object), return_counts=True)
    return {
        "split_strategy": split_strategy,
        "split_fractions": {
            "train": train_fraction,
            "test": test_fraction,
            "val": val_fraction,
        },
        "split_counts": {str(label): int(count) for label, count in zip(labels, counts)},
    }


def leave_one_field_out_splits(fields: Sequence[Any]) -> dict[Any, np.ndarray]:
    unique_fields = list(dict.fromkeys(fields))
    return {field: make_field_aware_split(fields, test_fields=[field]) for field in unique_fields}


def subset_cached_dataset(dataset: CachedFusionDataset, mask: np.ndarray) -> CachedFusionDataset:
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    z_spec = None if dataset.z_spec is None else dataset.z_spec[indices]
    redshift_reference = {
        key: values[indices]
        for key, values in dataset.redshift_reference.items()
    }
    return CachedFusionDataset(
        object_ids=[dataset.object_ids[i] for i in indices],
        fields=[dataset.fields[i] for i in indices],
        aion_embeddings=dataset.aion_embeddings[indices],
        extra_features=dataset.extra_features[indices],
        z_spec=z_spec,
        redshift_reference=redshift_reference,
    )


def dataset_for_split(
    product: Mapping[str, Any],
    split: str = "val",
) -> CachedFusionDataset:
    dataset = CachedFusionDataset(
        object_ids=product["object_id"],
        fields=product["field"],
        aion_embeddings=product["aion_embedding"],
        extra_features=product["extra_features"],
        z_spec=product["z_spec"],
        redshift_reference=product.get("redshift_reference"),
    )
    split_labels = np.asarray(product["split_labels"])
    train_dataset = subset_cached_dataset(dataset, split_labels == "train")
    val_dataset = subset_cached_dataset(dataset, split_labels == "val")
    test_dataset = subset_cached_dataset(dataset, split_labels == "test")
    datasets = {"train": train_dataset, "val": val_dataset, "validation": val_dataset, "test": test_dataset}
    if split not in datasets:
        raise ValueError(f"Unknown split {split!r}. Expected one of: {sorted(datasets)}")
    return datasets[split]


def split_cache_matches_current_schema(paths: Mapping[str, Path]) -> bool:
    try:
        bands = np.load(paths["bands"], mmap_mode="r")
        errors = np.load(paths["errors"], mmap_mode="r")
        flags = np.load(paths["flags"], mmap_mode="r")
    except (OSError, ValueError):
        return False

    band_fields = set(bands.dtype.names or ())
    error_fields = set(errors.dtype.names or ())
    flag_fields = set(flags.dtype.names or ())
    expected_band_fields = {
        "id",
        "ra",
        "dec",
        "tract",
        "patch",
        *(f"flux_cmodel_{band}" for band in ALL_BAND_FLUX_COLUMNS),
    }
    expected_error_fields = {
        "id",
        "ra",
        "dec",
        "tract",
        "patch",
        *(f"fluxerr_cmodel_{band}" for band in ALL_BAND_ERROR_COLUMNS),
    }
    expected_flag_fields = {"id", *ALL_FLAG_COLUMNS}
    return (
        expected_band_fields.issubset(band_fields)
        and expected_error_fields.issubset(error_fields)
        and expected_flag_fields.issubset(flag_fields)
    )
