from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits


# -----------------------------
# Column maps
# -----------------------------

BAND_FLUX_COLUMNS = {
    "u": "FLUX_CMODEL_MegaCam-u",
    "g": "FLUX_CMODEL_HSC-G",
    "r": "FLUX_CMODEL_HSC-R",
    "i": "FLUX_CMODEL_HSC-I",
    "z": "FLUX_CMODEL_HSC-Z",
    "y": "FLUX_CMODEL_HSC-Y",
}

BAND_ERROR_COLUMNS = {
    "u": "FLUXERR_CMODEL_MegaCam-u",
    "g": "FLUXERR_CMODEL_HSC-G",
    "r": "FLUXERR_CMODEL_HSC-R",
    "i": "FLUXERR_CMODEL_HSC-I",
    "z": "FLUXERR_CMODEL_HSC-Z",
    "y": "FLUXERR_CMODEL_HSC-Y",
}

OPTIONAL_EXTRA_BAND_FLUX_COLUMNS = {
    "u_star": "FLUX_CMODEL_MegaCam-uS",
    "Y": "FLUX_CMODEL_VIRCAM-Y",
    "J": "FLUX_CMODEL_VIRCAM-J",
    "H": "FLUX_CMODEL_VIRCAM-H",
    "Ks": "FLUX_CMODEL_VIRCAM-Ks",
}

OPTIONAL_EXTRA_BAND_ERROR_COLUMNS = {
    "u_star": "FLUXERR_CMODEL_MegaCam-uS",
    "Y": "FLUXERR_CMODEL_VIRCAM-Y",
    "J": "FLUXERR_CMODEL_VIRCAM-J",
    "H": "FLUXERR_CMODEL_VIRCAM-H",
    "Ks": "FLUXERR_CMODEL_VIRCAM-Ks",
}

ALL_BAND_FLUX_COLUMNS = {
    **BAND_FLUX_COLUMNS,
    **OPTIONAL_EXTRA_BAND_FLUX_COLUMNS,
}

ALL_BAND_ERROR_COLUMNS = {
    **BAND_ERROR_COLUMNS,
    **OPTIONAL_EXTRA_BAND_ERROR_COLUMNS,
}

REDSHIFT_COLUMNS = {
    "zphot": "ZPHOT",
    "z_low68": "Z_LOW68",
    "z_high68": "Z_HIGH68",
    "z_chi": "Z_CHI",
    "z_peak": "Z_PEAK",
    "posterior_log": "Posterior-Log",
    "likelihood_log": "Likelihood-Log",
}

# Conservative quality / classification flags.
# These are not input features by default; use them for sample cuts.
FLAG_COLUMNS = {
    "is_compact": "isCompact",
    "is_outside_mask": "isOutsideMask",
    "is_star_temp": "isStarTemp",
    "is_star": "isStar",

    "is_clean_u": "isClean_MegaCam-u",
    "is_clean_g": "isClean_HSC-G",
    "is_clean_r": "isClean_HSC-R",
    "is_clean_i": "isClean_HSC-I",
    "is_clean_z": "isClean_HSC-Z",
    "is_clean_y": "isClean_HSC-Y",

    "has_bad_photometry_u": "hasBadPhotometry_MegaCam-u",
    "has_bad_photometry_g": "hasBadPhotometry_HSC-G",
    "has_bad_photometry_r": "hasBadPhotometry_HSC-R",
    "has_bad_photometry_i": "hasBadPhotometry_HSC-I",
    "has_bad_photometry_z": "hasBadPhotometry_HSC-Z",
    "has_bad_photometry_y": "hasBadPhotometry_HSC-Y",

    "is_no_data_u": "isNoData_MegaCam-u",
    "is_no_data_g": "isNoData_HSC-G",
    "is_no_data_r": "isNoData_HSC-R",
    "is_no_data_i": "isNoData_HSC-I",
    "is_no_data_z": "isNoData_HSC-Z",
    "is_no_data_y": "isNoData_HSC-Y",

    "not_observed_u": "notObserved_MegaCam-u",
    "not_observed_g": "notObserved_HSC-G",
    "not_observed_r": "notObserved_HSC-R",
    "not_observed_i": "notObserved_HSC-I",
    "not_observed_z": "notObserved_HSC-Z",
    "not_observed_y": "notObserved_HSC-Y",

    "is_duplicated_u": "isDuplicated_MegaCam-u",
    "is_duplicated_g": "isDuplicated_HSC-G",
    "is_duplicated_r": "isDuplicated_HSC-R",
    "is_duplicated_i": "isDuplicated_HSC-I",
    "is_duplicated_z": "isDuplicated_HSC-Z",
    "is_duplicated_y": "isDuplicated_HSC-Y",
}

OPTIONAL_EXTRA_FLAG_COLUMNS = {
    "is_clean_u_star": "isClean_MegaCam-uS",
    "is_clean_Y": "isClean_VIRCAM-Y",
    "is_clean_J": "isClean_VIRCAM-J",
    "is_clean_H": "isClean_VIRCAM-H",
    "is_clean_Ks": "isClean_VIRCAM-Ks",

    "has_bad_photometry_u_star": "hasBadPhotometry_MegaCam-uS",
    "has_bad_photometry_Y": "hasBadPhotometry_VIRCAM-Y",
    "has_bad_photometry_J": "hasBadPhotometry_VIRCAM-J",
    "has_bad_photometry_H": "hasBadPhotometry_VIRCAM-H",
    "has_bad_photometry_Ks": "hasBadPhotometry_VIRCAM-Ks",

    "is_no_data_u_star": "isNoData_MegaCam-uS",
    "is_no_data_Y": "isNoData_VIRCAM-Y",
    "is_no_data_J": "isNoData_VIRCAM-J",
    "is_no_data_H": "isNoData_VIRCAM-H",
    "is_no_data_Ks": "isNoData_VIRCAM-Ks",

    "not_observed_u_star": "notObserved_MegaCam-uS",
    "not_observed_Y": "notObserved_VIRCAM-Y",
    "not_observed_J": "notObserved_VIRCAM-J",
    "not_observed_H": "notObserved_VIRCAM-H",
    "not_observed_Ks": "notObserved_VIRCAM-Ks",

    "is_duplicated_u_star": "isDuplicated_MegaCam-uS",
    "is_duplicated_Y": "isDuplicated_VIRCAM-Y",
    "is_duplicated_J": "isDuplicated_VIRCAM-J",
    "is_duplicated_H": "isDuplicated_VIRCAM-H",
    "is_duplicated_Ks": "isDuplicated_VIRCAM-Ks",
}

ALL_FLAG_COLUMNS = {
    **FLAG_COLUMNS,
    **OPTIONAL_EXTRA_FLAG_COLUMNS,
}


# -----------------------------
# Dtypes
# -----------------------------

def metadata_dtype() -> list[tuple[str, object]]:
    return [
        ("id", np.int64),
        ("ra", np.float64),
        ("dec", np.float64),
        ("tract", "U4"),
        ("patch", "U3"),
    ]


def bands_dtype() -> np.dtype:
    return np.dtype(
        metadata_dtype()
        + [(f"flux_cmodel_{band}", np.float64) for band in ALL_BAND_FLUX_COLUMNS]
    )


def errors_dtype() -> np.dtype:
    return np.dtype(
        metadata_dtype()
        + [(f"fluxerr_cmodel_{band}", np.float64) for band in ALL_BAND_ERROR_COLUMNS]
    )


def redshifts_dtype() -> np.dtype:
    return np.dtype(
        [("id", np.int64)]
        + [(name, np.float64) for name in REDSHIFT_COLUMNS]
    )


def flags_dtype() -> np.dtype:
    # isOutsideMask is integer in the FITS header; store all others as bool.
    dtype = [("id", np.int64)]
    for out_name in ALL_FLAG_COLUMNS:
        if out_name == "is_outside_mask":
            dtype.append((out_name, np.int16))
        else:
            dtype.append((out_name, np.bool_))
    return np.dtype(dtype)


# -----------------------------
# Utilities
# -----------------------------

def _require_columns(table, required: Iterable[str]) -> None:
    names = set(table.names)
    missing = [col for col in required if col not in names]
    if missing:
        raise KeyError(f"Missing required FITS columns: {missing}")


CATALOGUE_SAMPLE_MODES = ("head", "random")


def normalize_catalogue_row_range(
    n_rows_total: int,
    row_start: int | None = None,
    row_stop: int | None = None,
) -> tuple[int, int]:
    """Return a validated half-open source-row interval."""
    start = 0 if row_start is None else int(row_start)
    stop = n_rows_total if row_stop is None else min(int(row_stop), n_rows_total)
    if start < 0:
        raise ValueError("row_start must be >= 0.")
    if row_stop is not None and int(row_stop) <= start:
        raise ValueError("row_stop must be greater than row_start.")
    if start >= n_rows_total:
        raise ValueError(
            f"row_start={start} is outside a catalogue with {n_rows_total} rows."
        )
    return start, stop


def select_catalogue_row_indices(
    n_rows_total: int,
    *,
    max_rows: int | None = None,
    sample_mode: str = "head",
    row_start: int | None = None,
    row_stop: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Select source rows from an optional interval using head or random sampling."""
    if sample_mode not in CATALOGUE_SAMPLE_MODES:
        raise ValueError(
            f"sample_mode must be one of {CATALOGUE_SAMPLE_MODES}, got {sample_mode!r}."
        )
    start, stop = normalize_catalogue_row_range(n_rows_total, row_start, row_stop)
    available = stop - start
    if max_rows is None:
        sample_size = available
    else:
        if int(max_rows) < 1:
            raise ValueError("max_rows must be >= 1 or None.")
        sample_size = min(int(max_rows), available)

    if sample_mode == "head" or sample_size == available:
        return np.arange(start, start + sample_size, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected = rng.choice(available, size=sample_size, replace=False)
    return np.sort(selected.astype(np.int64, copy=False) + start)


def select_clauds_catalogue_row_indices(
    table,
    *,
    max_rows: int | None = None,
    sample_mode: str = "head",
    row_start: int | None = None,
    row_stop: int | None = None,
    seed: int = 42,
    require_valid_bands: Iterable[str] = (),
) -> np.ndarray:
    """Filter by required-band validity, then sample source rows."""
    if sample_mode not in CATALOGUE_SAMPLE_MODES:
        raise ValueError(
            f"sample_mode must be one of {CATALOGUE_SAMPLE_MODES}, got {sample_mode!r}."
        )
    start, stop = normalize_catalogue_row_range(len(table), row_start, row_stop)
    required_bands = tuple(dict.fromkeys(str(band) for band in require_valid_bands))
    if not required_bands:
        return select_catalogue_row_indices(
            len(table),
            max_rows=max_rows,
            sample_mode=sample_mode,
            row_start=start,
            row_stop=stop,
            seed=seed,
        )

    table_names = set(table.names)
    source_slice = slice(start, stop)
    valid = np.ones(stop - start, dtype=bool)
    for band in required_bands:
        if band not in ALL_BAND_FLUX_COLUMNS:
            raise KeyError(
                f"Unknown required band {band!r}. "
                f"Expected one of: {tuple(ALL_BAND_FLUX_COLUMNS)}."
            )
        flux_name = ALL_BAND_FLUX_COLUMNS[band]
        if flux_name not in table_names:
            valid[:] = False
            continue
        flux = np.asarray(table[flux_name][source_slice])
        valid &= np.isfinite(flux) & (flux > 0)
        for prefix in ("has_bad_photometry", "is_no_data", "not_observed"):
            flag_name = ALL_FLAG_COLUMNS.get(f"{prefix}_{band}")
            if flag_name is not None and flag_name in table_names:
                valid &= ~np.asarray(table[flag_name][source_slice]).astype(bool)

    eligible_rows = np.flatnonzero(valid).astype(np.int64, copy=False) + start
    if len(eligible_rows) == 0:
        raise ValueError(
            f"No rows in [{start}, {stop}) satisfy required bands {required_bands}."
        )
    if max_rows is None:
        sample_size = len(eligible_rows)
    else:
        if int(max_rows) < 1:
            raise ValueError("max_rows must be >= 1 or None.")
        sample_size = min(int(max_rows), len(eligible_rows))

    if sample_mode == "head" or sample_size == len(eligible_rows):
        return eligible_rows[:sample_size]
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(eligible_rows), size=sample_size, replace=False)
    return np.sort(eligible_rows[selected])


def _copy_metadata(dst: np.ndarray, table, source_rows) -> None:
    dst["id"] = table["ID"][source_rows]
    dst["ra"] = table["RA"][source_rows]
    dst["dec"] = table["DEC"][source_rows]
    dst["tract"] = table["tract"][source_rows].astype(str)
    dst["patch"] = table["patch"][source_rows].astype(str)


def _missing_flag_default(out_name: str) -> bool:
    if out_name.startswith("is_clean_"):
        return False
    if out_name.startswith(("has_bad_photometry_", "is_no_data_", "not_observed_")):
        return True
    return False


def split_clauds_catalogue(
    fits_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_size: int = 250_000,
    overwrite: bool = False,
    max_rows: int | None = None,
    sample_mode: str = "head",
    row_start: int | None = None,
    row_stop: int | None = None,
    sample_seed: int = 42,
    require_valid_bands: Iterable[str] = (),
) -> dict[str, Path]:
    """
    Split a CLAUDS-style FITS binary table into small structured arrays.

    Outputs:
      - clauds_bands.npy
      - clauds_errors.npy
      - clauds_redshifts.npy
      - clauds_flags.npy

    Required-band validity is applied first within the half-open
    ``[row_start, row_stop)`` source interval. ``max_rows`` is then applied to
    eligible rows using either deterministic head or seeded random sampling.

    The output files are standard .npy structured arrays and can be loaded with
    np.load(path, mmap_mode="r") to avoid loading everything into memory.
    """
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "bands": output_dir / "clauds_bands.npy",
        "errors": output_dir / "clauds_errors.npy",
        "redshifts": output_dir / "clauds_redshifts.npy",
        "flags": output_dir / "clauds_flags.npy",
    }

    if not overwrite:
        existing = [path for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Pass overwrite=True to replace them:\n"
                + "\n".join(str(path) for path in existing)
            )

    with fits.open(fits_path, memmap=True) as hdul:
        table = hdul[1].data
        table_names = set(table.names)
        selected_rows = select_clauds_catalogue_row_indices(
            table,
            max_rows=max_rows,
            sample_mode=sample_mode,
            row_start=row_start,
            row_stop=row_stop,
            seed=sample_seed,
            require_valid_bands=require_valid_bands,
        )
        n_rows = len(selected_rows)

        required_columns = (
            ["ID", "RA", "DEC", "tract", "patch"]
            + list(BAND_FLUX_COLUMNS.values())
            + list(BAND_ERROR_COLUMNS.values())
            + list(REDSHIFT_COLUMNS.values())
            + list(FLAG_COLUMNS.values())
        )
        _require_columns(table, required_columns)

        bands = np.lib.format.open_memmap(
            output_paths["bands"],
            mode="w+",
            dtype=bands_dtype(),
            shape=(n_rows,),
        )
        errors = np.lib.format.open_memmap(
            output_paths["errors"],
            mode="w+",
            dtype=errors_dtype(),
            shape=(n_rows,),
        )
        redshifts = np.lib.format.open_memmap(
            output_paths["redshifts"],
            mode="w+",
            dtype=redshifts_dtype(),
            shape=(n_rows,),
        )
        flags = np.lib.format.open_memmap(
            output_paths["flags"],
            mode="w+",
            dtype=flags_dtype(),
            shape=(n_rows,),
        )

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            destination_rows = slice(start, stop)
            source_rows = selected_rows[destination_rows]

            # Metadata in bands/errors.
            _copy_metadata(bands[destination_rows], table, source_rows)
            _copy_metadata(errors[destination_rows], table, source_rows)

            # Redshift products.
            redshifts[destination_rows]["id"] = table["ID"][source_rows]
            for out_name, fits_name in REDSHIFT_COLUMNS.items():
                redshifts[destination_rows][out_name] = table[fits_name][source_rows]

            # Band fluxes.
            for band, fits_name in ALL_BAND_FLUX_COLUMNS.items():
                field_name = f"flux_cmodel_{band}"
                if fits_name in table_names:
                    bands[destination_rows][field_name] = table[fits_name][source_rows]
                else:
                    bands[destination_rows][field_name] = np.nan

            # Band errors.
            for band, fits_name in ALL_BAND_ERROR_COLUMNS.items():
                field_name = f"fluxerr_cmodel_{band}"
                if fits_name in table_names:
                    errors[destination_rows][field_name] = table[fits_name][source_rows]
                else:
                    errors[destination_rows][field_name] = np.nan

            # Flags.
            flags[destination_rows]["id"] = table["ID"][source_rows]
            for out_name, fits_name in ALL_FLAG_COLUMNS.items():
                if fits_name in table_names:
                    flags[destination_rows][out_name] = table[fits_name][source_rows]
                else:
                    flags[destination_rows][out_name] = _missing_flag_default(out_name)

            print(f"Copied rows {start:,}–{stop:,} / {n_rows:,}")

        # Make sure memmap buffers are flushed.
        bands.flush()
        errors.flush()
        redshifts.flush()
        flags.flush()

    return output_paths

# Column name constants
OBJECT_ID_COLUMN = "ID"
RA_COLUMN = "RA"
DEC_COLUMN = "DEC"
TRACT_COLUMN = "tract"
PATCH_COLUMN = "patch"
HSC_AION_BANDS = ["g", "r", "i", "z", "y"]
CLAUDS_EXTRA_FLUX_BANDS = ["u", "u_star", "Y", "J", "H", "Ks"]
EXTRA_ERROR_BANDS = ["u", "u_star", "Y", "J", "H", "Ks"]

def default_hsc_mag_faint_limits() -> dict[str, float | None]:
    return {"g": 24.5, "r": 24.5, "i": 24.0, "z": 24.5, "y": 24.5}


def _table_column_names(table) -> set[str]:
    if hasattr(table, "names") and table.names is not None:
        return set(table.names)
    if hasattr(table, "dtype") and table.dtype.names is not None:
        return set(table.dtype.names)
    if hasattr(table, "keys"):
        return set(table.keys())
    raise TypeError("table must be a FITS table, structured array, or mapping of arrays")


def validate_clauds_fits_table(table, require_redshift: bool = False) -> None:
    required = (
        [OBJECT_ID_COLUMN, RA_COLUMN, DEC_COLUMN, TRACT_COLUMN, PATCH_COLUMN]
        + list(BAND_FLUX_COLUMNS.values())
        + list(BAND_ERROR_COLUMNS.values())
        + list(FLAG_COLUMNS.values())
    )
    if require_redshift:
        required += list(REDSHIFT_COLUMNS.values())
    names = _table_column_names(table)
    missing = [column for column in required if column not in names]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
