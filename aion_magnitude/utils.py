from __future__ import annotations
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import torch


def load_cached_product(path: str | Path) -> dict[str, Any]:
    """Load a torch cached product produced by the AION-CLAUDS module/notebook."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def set_random_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def available_torch_devices() -> list[str]:
    devices = []
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


def select_torch_device(choice: str = "auto") -> torch.device:
    available = available_torch_devices()
    if choice == "auto":
        return torch.device(available[0])
    if choice not in {"mps", "cuda", "cpu"}:
        raise ValueError(f"Unknown device choice: {choice}")
    if choice not in available:
        raise RuntimeError(f"Requested device {choice!r} is not available. Available: {available}")
    return torch.device(choice)


def resolve_torch_device(device_or_choice: torch.device | str | None = None) -> torch.device:
    if device_or_choice is None:
        return select_torch_device()
    if isinstance(device_or_choice, torch.device):
        return device_or_choice
    return select_torch_device(device_or_choice)


def make_redshift_grid(
    z_min: float = 0.0,
    z_max: float = 6.0,
    n_z_bins: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    edges = torch.linspace(z_min, z_max, n_z_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def configure_redshift_grid(
    z_min: float = 0.0,
    z_max: float = 6.0,
    n_z_bins: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a redshift grid for compatibility with older notebooks."""
    return make_redshift_grid(z_min, z_max, n_z_bins)


def flux_to_ab_mag(
    flux: np.ndarray,
    mag_zero_point: float = 23.0,
) -> tuple[np.ndarray, np.ndarray]:
    flux = np.asarray(flux, dtype=np.float32)
    valid = np.isfinite(flux) & (flux > 0)
    mag = np.full(flux.shape, np.nan, dtype=np.float32)
    mag[valid] = -2.5 * np.log10(flux[valid]) + mag_zero_point
    return mag, valid


def finite_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return fallback
    scale = np.nanmedian(np.abs(values[finite]))
    if not np.isfinite(scale) or scale <= 0:
        return fallback
    return float(scale)


def asinh_transform(values: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return np.arcsinh(values / scale).astype(np.float32)


def tensor_to_numpy_1d(value: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def gaussian_kernel_1d(sigma_bins: float, truncate: float = 4.0) -> np.ndarray:
    """Build a normalized 1D Gaussian kernel where sigma is measured in bins."""
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(truncate * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_smooth_1d(values: np.ndarray, sigma_bins: float, truncate: float = 4.0) -> np.ndarray:
    """Smooth a 1D array with a Gaussian filter measured in bin widths."""
    values = np.asarray(values, dtype=np.float64)
    kernel = gaussian_kernel_1d(sigma_bins, truncate=truncate)
    if kernel.size == 1:
        return values.copy()
    radius = (kernel.size - 1) // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def apply_numpy_mask_to_tensor_dict(
    tensors: dict[str, torch.Tensor],
    mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
    return {key: value[mask_tensor] for key, value in tensors.items()}


def table_column_names(table) -> set[str]:
    if hasattr(table, "names") and table.names is not None:
        return set(table.names)
    if hasattr(table, "dtype") and table.dtype.names is not None:
        return set(table.dtype.names)
    if isinstance(table, Mapping):
        return set(table.keys())
    raise TypeError("table must be a FITS table, structured array, or mapping of arrays")


def require_columns(table, required: Iterable[str]) -> None:
    names = table_column_names(table)
    missing = [column for column in required if column not in names]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def table_length(table) -> int:
    if isinstance(table, Mapping):
        first_key = next(iter(table))
        return len(table[first_key])
    return len(table)


def table_column(table, column_name: str, rows=None) -> np.ndarray:
    values = table[column_name]
    if rows is not None:
        values = values[rows]
    return np.asarray(values)


def numeric_table_column(table, column_name: str, rows=None, dtype=np.float32) -> np.ndarray:
    return table_column(table, column_name, rows=rows).astype(dtype, copy=False)


def string_table_column(table, column_name: str, rows=None) -> np.ndarray:
    return table_column(table, column_name, rows=rows).astype(str)


def validate_split_fractions(
    train_fraction: float,
    test_fraction: float,
    val_fraction: float,
) -> None:
    fractions = {
        "train_fraction": train_fraction,
        "test_fraction": test_fraction,
        "val_fraction": val_fraction,
    }
    for name, value in fractions.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    total = train_fraction + test_fraction + val_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_fraction + test_fraction + val_fraction must sum to 1.0, "
            f"got {total:.6f}."
        )


def _path_tag(value: str) -> str:
    return str(value).replace("*", "star").replace("/", "_").replace(" ", "_")
