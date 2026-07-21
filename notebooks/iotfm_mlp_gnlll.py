#!/usr/bin/env python3
"""Train matched IoTFM and MLP photo-z regressors with GNLLL.

GNLLL means Gaussian negative log-likelihood loss.  Each branch predicts a
Gaussian photo-z distribution.  Its mean head receives 55 CLAUDS flux
measurements plus 11 Kron-radius measurements.  Its variance head receives
the 55 matching flux uncertainties plus the inferred redshift mean.

IoTFM means inference-optimized transformer feature mapping.  In the IoTFM
branch, the 66 mean-signal columns are first mapped through a frozen
transformer.  The matched MLP branch receives the same 66 columns directly.
Both branches use the same tabular 55-column uncertainty input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import iotfm_mlp as base
from aion_magnitude.Inference_Opt_TFM import CatalogueSerializationConfig
from aion_magnitude.clauds_bands import select_catalogue_row_indices
from aion_magnitude.metrics import point_photoz_metrics
from aion_magnitude.plotting import (
    compare_config_loss,
    compare_nz_lensing_alike,
    compare_pit_histogram,
    compare_redshift_probability_distribution,
    compare_zpred_vs_zphot,
)


GNLLL_EXPANSION = "Gaussian negative log-likelihood loss"
FLUX_PREFIXES = ("FLUX_APER_2_", "FLUX_APER_3_", "FLUX_PSF_", "FLUX_KRON_", "FLUX_CMODEL_")
FLUX_ERROR_PREFIXES = (
    "FLUXERR_APER_2_", "FLUXERR_APER_3_", "FLUXERR_PSF_",
    "FLUXERR_KRON_", "FLUXERR_CMODEL_",
)
KRON_RADIUS_PREFIX = "RADIUS_KRON_"
FEATURE_SCALING_MODES = ("none", "physical_robust")
MISSING_SENTINEL = -99.0
ROBUST_NORMAL_IQR = 1.349
ROBUST_CLIP = 8.0
FULL_COLUMN_CACHE_VERSION = 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalogue", type=Path,
                   default=Path("data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/arc/home/gsm/aion_output/figures/iotfm_mlp_gnlll"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("/scratch/.tmp-gsm/aion_output/cache/iotfm_mlp_gnlll"))
    p.add_argument("--input-cache-root", type=Path,
                   default=Path("/scratch/.tmp-gsm/aion_output/cache/iotfm_mlp_gnlll_input"))
    p.add_argument("--max-rows", type=base.max_rows_arg, default=200000)
    p.add_argument("--model", default="GLM-5.2-0.8B-A0.8B")
    p.add_argument("--embedding-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--pooling", choices=("mean", "last", "mean_last"), default="mean")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--force-recompute-embeddings", action="store_true")
    p.add_argument("--force-recompute-input-cache", action="store_true")
    p.add_argument("--feature-scaling", choices=FEATURE_SCALING_MODES, default="none",
                   help="physical_robust applies the transform in logs/cosmos_scaling.md")
    p.add_argument("--mean-warmup-epochs", type=int, default=10,
                   help="mean-only SmoothL1 epochs before joint GNLLL training")
    p.add_argument("--gnlll-epochs", type=int, default=20,
                   help="joint Gaussian negative log-likelihood loss epochs")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mean-hidden-dim", type=int, default=256)
    p.add_argument("--variance-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--variance-floor", type=float, default=1e-6)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--n-z-bins", type=int, default=300)
    p.add_argument("--tomographic-samples", type=int, default=100)
    p.add_argument("--train-fraction", type=float, default=.63)
    p.add_argument("--test-fraction", type=float, default=.32)
    p.add_argument("--val-fraction", type=float, default=.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return p


def select_gnlll_columns(names: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    """Return the 55 flux, 11 size, and 55 flux-error columns in catalogue order."""
    flux = [name for name in names if name.startswith(FLUX_PREFIXES)]
    radii = [name for name in names if name.startswith(KRON_RADIUS_PREFIX)]
    errors = [name for name in names if name.startswith(FLUX_ERROR_PREFIXES)]
    expected = (55, 11, 55)
    actual = (len(flux), len(radii), len(errors))
    if actual != expected:
        raise ValueError(
            "GNLLL requires exactly 55 flux, 11 RADIUS_KRON, and 55 FLUXERR "
            f"columns; found {actual}."
        )
    return flux, radii, errors

def load_full_gnlll_catalogue(catalogue: Path, cache_root: Path, *,
                              max_rows: int | None, seed: int,
                              force_recompute: bool = False):
    """Load a random-row cache preserving all 121 GNLLL FITS inputs.

    The general CLAUDS split cache intentionally keeps only one cModel flux
    and error per band. This dedicated cache preserves the five estimators.
    """
    catalogue = catalogue.expanduser().resolve()
    cache_root = cache_root.expanduser()
    row_tag = "all" if max_rows is None else f"n{max_rows}"
    stem = f"cosmos_gnlll_full121_v{FULL_COLUMN_CACHE_VERSION}_{row_tag}_seed{seed}"
    cache_path = cache_root / "full_input" / f"{stem}.npy"
    metadata_path = cache_path.with_suffix(".json")
    source_stat = catalogue.stat()
    expected_source = {
        "cache_version": FULL_COLUMN_CACHE_VERSION,
        "catalogue": str(catalogue),
        "catalogue_size": source_stat.st_size,
        "catalogue_mtime_ns": source_stat.st_mtime_ns,
        "max_rows": max_rows,
        "sample_mode": "random",
        "sample_seed": seed,
    }
    if cache_path.exists() and metadata_path.exists() and not force_recompute:
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in expected_source.items()):
            raise RuntimeError(
                f"Full-column input cache does not match this run: {cache_path}. "
                "Use --force-recompute-input-cache or a different cache root."
            )
        table = np.load(cache_path, mmap_mode="r")
        select_gnlll_columns(table.dtype.names or ())
        return table, cache_path, metadata

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = cache_path.with_suffix(".partial.npy")
    with fits.open(catalogue, memmap=True) as hdul:
        source = hdul[1].data
        flux_columns, radius_columns, error_columns = select_gnlll_columns(source.names)
        input_columns = flux_columns + radius_columns + error_columns
        required = [base.OBJECT_ID_COLUMN, base.REDSHIFT_COLUMNS["zphot"], *input_columns]
        missing = [name for name in required if name not in source.names]
        if missing:
            raise KeyError(f"COSMOS FITS catalogue is missing GNLLL columns: {missing}")
        selected_rows = select_catalogue_row_indices(
            len(source), max_rows=max_rows, sample_mode="random", seed=seed,
        )
        dtype = np.dtype(
            [("SOURCE_ROW", np.int64), (base.OBJECT_ID_COLUMN, np.int64),
             (base.REDSHIFT_COLUMNS["zphot"], np.float64)]
            + [(name, np.float64) for name in input_columns]
        )
        cached = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=dtype, shape=(len(selected_rows),),
        )
        for start in range(0, len(selected_rows), 50_000):
            stop = min(start + 50_000, len(selected_rows))
            source_rows = selected_rows[start:stop]
            cached["SOURCE_ROW"][start:stop] = source_rows
            for name in required:
                cached[name][start:stop] = source[name][source_rows]
            print(f"GNLLL full-column cache: {stop:,}/{len(selected_rows):,}")
        cached.flush()
        del cached
    os.replace(partial_path, cache_path)
    metadata = {
        **expected_source,
        "n_rows": len(selected_rows),
        "input_columns": input_columns,
        "column_groups": {"flux": flux_columns, "radius": radius_columns,
                          "flux_error": error_columns},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return np.load(cache_path, mmap_mode="r"), cache_path, metadata


def clean_column(table, rows: np.ndarray, column: str, kind: str):
    """Return numeric values and the valid mask under the COSMOS sentinel policy."""
    raw = np.ma.asarray(table[column])[rows]
    values = np.asarray(np.ma.getdata(raw), dtype=np.float64)
    valid = ~np.ma.getmaskarray(raw) & np.isfinite(values) & (values != MISSING_SENTINEL)
    if kind == "error":
        valid &= values > 0
    elif kind == "radius":
        valid &= values >= 0
    elif kind != "flux":
        raise ValueError(f"Unknown GNLLL feature kind: {kind}")
    return values, valid


def robust_scale_parameters(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    scale = float((q75 - q25) / ROBUST_NORMAL_IQR)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def build_gnlll_features(table, rows: np.ndarray, columns: Sequence[str],
                         kinds: Sequence[str], splits: np.ndarray, *,
                         mode: str) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Sanitize/impute inputs and optionally apply physical robust scaling."""
    if mode not in FEATURE_SCALING_MODES:
        raise ValueError(f"mode must be one of {FEATURE_SCALING_MODES}, got {mode!r}")
    if len(columns) != len(kinds):
        raise ValueError("columns and kinds must have the same length")
    train = splits == "train"
    arrays, metadata = [], []
    for column, kind in zip(columns, kinds):
        values, valid = clean_column(table, rows, column, kind)
        fit = train & valid
        if not fit.any():
            raise ValueError(f"No valid training values for {column}")
        transformed = np.full(len(values), np.nan, dtype=np.float64)
        if mode == "none":
            transformed[valid] = values[valid]
            fill = float(np.median(values[fit]))
            output = np.where(valid, transformed, fill)
            transform_metadata = {"transform": "identity", "imputation_value": fill}
        else:
            if kind == "flux":
                error_column = column.replace("FLUX_", "FLUXERR_", 1)
                error_values, error_valid = clean_column(table, rows, error_column, "error")
                error_fit = train & error_valid
                if not error_fit.any():
                    raise ValueError(f"No valid training values for matching {error_column}")
                softening = float(np.median(error_values[error_fit]))
                transformed[valid] = np.arcsinh(values[valid] / softening)
                transform_metadata = {
                    "transform": "asinh", "softening_scale": softening,
                    "matching_error_column": error_column,
                }
            elif kind == "error":
                transformed[valid] = np.log(values[valid])
                transform_metadata = {"transform": "log"}
            else:
                positive_fit = fit & (values > 0)
                radius_scale = float(np.median(values[positive_fit])) if positive_fit.any() else 1.0
                transformed[valid] = np.log1p(values[valid] / radius_scale)
                transform_metadata = {"transform": "log1p", "radius_scale": radius_scale}
            center, scale = robust_scale_parameters(transformed[fit])
            output = np.zeros(len(values), dtype=np.float64)
            output[valid] = np.clip(
                (transformed[valid] - center) / scale, -ROBUST_CLIP, ROBUST_CLIP,
            )
            transform_metadata.update(
                center=center, scale=scale, robust_normal_iqr=ROBUST_NORMAL_IQR,
                clip=[-ROBUST_CLIP, ROBUST_CLIP], imputation_value=0.0,
            )
        arrays.append(output.astype(np.float32))
        metadata.append({
            "column": column, "kind": kind, "mode": mode,
            "n_missing": int((~valid).sum()), "fit_split": "train",
            **transform_metadata,
        })
    return torch.from_numpy(np.stack(arrays, axis=1)), metadata


def scale_target_from_train(target: torch.Tensor, splits: np.ndarray, *, mode: str):
    """Affinely standardize ZPHOT only for the physical-robust experiment."""
    if mode == "none":
        metadata = {"mode": "none", "offset": 0.0, "scale": 1.0, "fit_split": "train"}
        return target.clone(), metadata
    train = torch.from_numpy(splits == "train")
    offset = float(target[train].mean())
    scale = float(target[train].std(unbiased=False))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Training ZPHOT has no usable standard deviation")
    metadata = {"mode": "standard", "offset": offset, "scale": scale,
                "fit_split": "train"}
    return (target - offset) / scale, metadata



def standardize_from_train(features: torch.Tensor, split_labels: np.ndarray):
    """Standardize features using training rows only and return fit statistics."""
    train = torch.from_numpy(split_labels == "train")
    mean = features[train].mean(dim=0)
    scale = features[train].std(dim=0, unbiased=False).clamp_min(1e-6)
    return (features - mean) / scale, mean, scale


class HeteroscedasticPhotoZRegressor(nn.Module):
    """Gaussian photo-z regressor with separate mean and variance information."""

    def __init__(self, mean_input_dim: int, error_input_dim: int, *,
                 mean_hidden_dim: int = 256, variance_hidden_dim: int = 128,
                 dropout: float = 0.1, variance_floor: float = 1e-6):
        super().__init__()
        self.variance_floor = float(variance_floor)
        self.mean_net = nn.Sequential(
            nn.Linear(mean_input_dim, mean_hidden_dim), nn.GELU(),
            nn.LayerNorm(mean_hidden_dim), nn.Dropout(dropout),
            nn.Linear(mean_hidden_dim, mean_hidden_dim // 2), nn.GELU(),
            nn.Linear(mean_hidden_dim // 2, 1),
        )
        self.variance_features = nn.Sequential(
            nn.Linear(error_input_dim + 1, variance_hidden_dim), nn.GELU(),
            nn.LayerNorm(variance_hidden_dim), nn.Dropout(dropout),
        )
        self.variance_output = nn.Linear(variance_hidden_dim, 1)

    def predict_mean(self, mean_features: torch.Tensor) -> torch.Tensor:
        return self.mean_net(mean_features).squeeze(-1)

    def forward(self, mean_features: torch.Tensor, error_features: torch.Tensor):
        mean = self.predict_mean(mean_features)
        # Detaching keeps the uncertainty objective from changing the mean merely
        # to make variance prediction easier; GNLLL still trains mean via residuals.
        variance_input = torch.cat((error_features, mean.detach().unsqueeze(-1)), dim=-1)
        raw_variance = self.variance_output(self.variance_features(variance_input)).squeeze(-1)
        variance = F.softplus(raw_variance) + self.variance_floor
        return mean, variance


def make_loader(mean_features: torch.Tensor, error_features: torch.Tensor,
                target: torch.Tensor, split_labels: np.ndarray, split: str,
                *, batch_size: int, shuffle: bool) -> DataLoader:
    mask = torch.from_numpy(split_labels == split)
    dataset = TensorDataset(mean_features[mask], error_features[mask], target[mask])
    if not len(dataset):
        raise ValueError(f"The {split!r} split is empty")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate(model: HeteroscedasticPhotoZRegressor, loader: DataLoader,
             device: torch.device, target_transform: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    means, variances, targets = [], [], []
    for mean_x, error_x, target in loader:
        mean, variance = model(mean_x.to(device), error_x.to(device))
        means.append(mean.cpu()); variances.append(variance.cpu()); targets.append(target.cpu())
    model_mean = torch.cat(means)
    model_variance = torch.cat(variances)
    model_target = torch.cat(targets)
    model_residual = model_target - model_mean
    nll = .5 * (model_variance.log() + model_residual.square() / model_variance)

    offset = float(target_transform["offset"])
    target_scale = float(target_transform["scale"])
    mean = offset + target_scale * model_mean
    variance = target_scale ** 2 * model_variance
    target = offset + target_scale * model_target
    sigma = variance.sqrt()
    residual = target - mean
    physical_nll = .5 * (variance.log() + residual.square() / variance)
    point = point_photoz_metrics(mean, target)
    metrics = {
        **point,
        "gaussian_nll": float(physical_nll.mean()),
        "gaussian_nll_model_space": float(nll.mean()),
        "mae": float(residual.abs().mean()),
        "rmse": float(residual.square().mean().sqrt()),
        "coverage_68": float((residual.abs() <= sigma).float().mean()),
        "coverage_95": float((residual.abs() <= 1.96 * sigma).float().mean()),
        "standardized_residual_mean": float((residual / sigma).mean()),
        "standardized_residual_std": float((residual / sigma).std(unbiased=False)),
        "mean_predicted_sigma": float(sigma.mean()),
    }
    return {"mean": mean, "variance": variance, "target": target, "metrics": metrics}


@torch.no_grad()
def initialize_variance_head(model: HeteroscedasticPhotoZRegressor,
                             train_loader: DataLoader, device: torch.device) -> float:
    """Start the variance head at the current mean-head residual variance."""
    model.eval()
    squared_residuals = []
    for mean_x, _, target in train_loader:
        mean = model.predict_mean(mean_x.to(device)).cpu()
        squared_residuals.append((target - mean).square())
    initial_variance = max(float(torch.cat(squared_residuals).mean()), model.variance_floor * 10)
    softplus_target = max(initial_variance - model.variance_floor, 1e-12)
    inverse_softplus = softplus_target if softplus_target > 20 else math.log(math.expm1(softplus_target))
    model.variance_output.weight.zero_()
    model.variance_output.bias.fill_(inverse_softplus)
    return initial_variance


def train_branch(name: str, mean_features: torch.Tensor, error_features: torch.Tensor,
                 target: torch.Tensor, split_labels: np.ndarray, args,
                 output_dir: Path, target_transform: dict[str, Any],
                 preprocessing_metadata: dict[str, Any],
                 ) -> tuple[HeteroscedasticPhotoZRegressor, dict[str, Any]]:
    """Warm up the mean, then jointly optimize mean and variance with GNLLL."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = base.resolve_torch_device(args.device)
    model = HeteroscedasticPhotoZRegressor(
        mean_features.shape[1], error_features.shape[1],
        mean_hidden_dim=args.mean_hidden_dim,
        variance_hidden_dim=args.variance_hidden_dim,
        dropout=args.dropout, variance_floor=args.variance_floor,
    ).to(device)
    train_loader = make_loader(mean_features, error_features, target, split_labels, "train",
                               batch_size=args.train_batch_size, shuffle=True)
    val_loader = make_loader(mean_features, error_features, target, split_labels, "val",
                             batch_size=args.eval_batch_size, shuffle=False)
    history: list[dict[str, Any]] = []

    warm_optimizer = torch.optim.AdamW(model.mean_net.parameters(), lr=args.learning_rate,
                                       weight_decay=args.weight_decay)
    for epoch in range(args.mean_warmup_epochs):
        model.train(); total = 0.; count = 0
        for mean_x, _, batch_target in train_loader:
            mean_x, batch_target = mean_x.to(device), batch_target.to(device)
            warm_optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model.predict_mean(mean_x), batch_target)
            loss.backward(); warm_optimizer.step()
            total += float(loss.detach()) * len(batch_target); count += len(batch_target)
        val = evaluate(model, val_loader, device, target_transform)
        row = {"epoch": epoch + 1, "stage": "mean_warmup", "train_loss": total / count,
               "val_gaussian_nll": val["metrics"]["gaussian_nll_model_space"],
               "val_nmad": val["metrics"]["nmad"]}
        history.append(row)
        print(f"{name} mean warm-up {epoch + 1:02d}/{args.mean_warmup_epochs}: "
              f"SmoothL1={row['train_loss']:.6f}, val NMAD={row['val_nmad']:.6f}")

    initial_variance = initialize_variance_head(model, train_loader, device)
    criterion = nn.GaussianNLLLoss(eps=args.variance_floor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for joint_epoch in range(args.gnlll_epochs):
        model.train(); total = 0.; count = 0
        for mean_x, error_x, batch_target in train_loader:
            mean_x = mean_x.to(device); error_x = error_x.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, variance = model(mean_x, error_x)
            loss = criterion(mean, batch_target, variance)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(batch_target); count += len(batch_target)
        val = evaluate(model, val_loader, device, target_transform)
        val_loss = val["metrics"]["gaussian_nll_model_space"]
        row = {"epoch": args.mean_warmup_epochs + joint_epoch + 1, "stage": "joint_gnlll",
               "train_loss": total / count, "val_gaussian_nll": val_loss,
               "val_nmad": val["metrics"]["nmad"],
               "val_coverage_68": val["metrics"]["coverage_68"]}
        history.append(row)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"{name} GNLLL {joint_epoch + 1:02d}/{args.gnlll_epochs}: "
              f"train={row['train_loss']:.6f}, val={val_loss:.6f}, "
              f"NMAD={row['val_nmad']:.6f}, cov68={row['val_coverage_68']:.3f}")

    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_gnlll.pt"
    result = {
        "target_transform": target_transform,
        "preprocessing": preprocessing_metadata,
        "model_kind": name,
        "gnlll_expansion": GNLLL_EXPANSION,
        "history": history,
        "best_val_gaussian_nll": best_loss,
        "initial_variance": initial_variance,
        "checkpoint_path": str(checkpoint),
        "mean_input_dim": int(mean_features.shape[1]),
        "error_input_dim": int(error_features.shape[1]),
    }
    torch.save({**result, "state_dict": {k: v.detach().cpu() for k, v in best_state.items()}}, checkpoint)
    return model, result


def gaussian_evaluation(raw: dict[str, Any], edges: torch.Tensor,
                        centers: torch.Tensor) -> dict[str, Any]:
    """Convert continuous Gaussian predictions to repository-compatible binned PDFs."""
    mean, variance, target = raw["mean"], raw["variance"], raw["target"]
    log_p = -.5 * ((centers[None, :] - mean[:, None]).square() / variance[:, None]
                   + variance[:, None].log())
    pz = torch.softmax(log_p, dim=-1)
    sigma = variance.sqrt()
    return {
        "pz": pz, "z_spec": target, "z_p50": mean,
        "z_p16": mean - sigma, "z_p84": mean + sigma, "z_mean": mean,
        "predicted_variance": variance, "predicted_sigma": sigma,
        "redshift_edges": edges, "redshift_centers": centers,
        "redshift_reference": {"zphot": target}, "metrics": raw["metrics"],
    }


def save_figures(iotfm_result, mlp_result, iotfm_eval, mlp_eval, args, output: Path):
    labels = (f"IoTFM GNLLL ({Path(args.model).name})", "tabular MLP GNLLL")
    prefix = output / "iotfm_mlp_gnlll_comparison"
    artifacts = {key: str(Path(f"{prefix}_{suffix}.jpeg")) for key, suffix in {
        "loss": "loss", "scatter": "scatter", "pit": "pit",
        "nz": "nz", "nztomo": "nztomo", "uncertainty": "uncertainty",
    }.items()}
    fig, _, _ = compare_config_loss(
        iotfm_result, mlp_result, output_path=artifacts["loss"], labels=labels,
        train_key="train_loss", val_key="val_gaussian_nll",
        title="Mean warm-up followed by Gaussian NLL training",
    ); plt.close(fig)
    fig, _ = compare_zpred_vs_zphot(
        iotfm_eval, mlp_eval, output_path=artifacts["scatter"], labels=labels,
        pred_key="z_p50", target_label="ZPHOT", pmax=5.0, show_metrics=True,
    ); plt.close(fig)
    fig, _ = compare_pit_histogram(
        iotfm_eval, mlp_eval, output_path=artifacts["pit"], labels=labels,
    ); plt.close(fig)
    fig, _, _ = compare_redshift_probability_distribution(
        iotfm_eval, mlp_eval, output_path=artifacts["nz"], labels=labels,
        gaussian_sigma_bins=1.0,
    ); plt.close(fig)
    fig, _, _ = compare_nz_lensing_alike(
        iotfm_eval, mlp_eval, output_path=artifacts["nztomo"], labels=labels,
        zphot_bin=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        inferred_bin_key="z_p50", n_samples_per_object=args.tomographic_samples,
    ); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, evaluation, label in zip(axes, (iotfm_eval, mlp_eval), labels):
        sigma = evaluation["predicted_sigma"].numpy()
        error = (evaluation["z_spec"] - evaluation["z_p50"]).abs().numpy()
        quantiles = np.quantile(sigma, np.linspace(0, 1, 11))
        x, y = [], []
        for low, high in zip(quantiles[:-1], quantiles[1:]):
            keep = (sigma >= low) & (sigma <= high if high == quantiles[-1] else sigma < high)
            if keep.any():
                x.append(float(np.mean(sigma[keep])))
                y.append(float(np.sqrt(np.mean(error[keep] ** 2))))
        limit = max(x + y) if x and y else 1.0
        ax.plot([0, limit], [0, limit], color="black", linewidth=1)
        ax.plot(x, y, marker="o"); ax.set_title(label); ax.grid(alpha=.2)
        ax.set_xlabel("mean predicted sigma")
    axes[0].set_ylabel("empirical RMSE")
    fig.suptitle("Photo-z uncertainty calibration"); fig.tight_layout()
    fig.savefig(artifacts["uncertainty"], dpi=160); plt.close(fig)
    return artifacts


def validate_args(args) -> None:
    positive = (args.embedding_batch_size, args.train_batch_size, args.eval_batch_size,
                args.gnlll_epochs, args.n_z_bins, args.tomographic_samples,
                args.learning_rate, args.mean_hidden_dim, args.variance_hidden_dim,
                args.variance_floor)
    if min(positive) <= 0 or args.mean_warmup_epochs < 0:
        raise ValueError("batch sizes, dimensions, GNLLL epochs, rates, and variance floor must be positive")
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("dropout must be in [0, 1)")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)
    # Fixed attributes let the existing frozen-transformer extractor serialize
    # the already sanitized/scaled 66-column signal table.
    args.include_id = False; args.include_location = False
    args.magnitudes_only = False; args.ignore_missingness = True
    args.exclude_flags = False; args.no_classification = False

    catalogue = args.catalogue.expanduser().resolve()
    output = args.output_dir.expanduser()
    table, input_cache, input_cache_metadata = load_full_gnlll_catalogue(
        catalogue, args.input_cache_root, max_rows=args.max_rows, seed=args.seed,
        force_recompute=args.force_recompute_input_cache,
    )
    target_all = np.asarray(table[base.REDSHIFT_COLUMNS["zphot"]], dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(target_all) & (target_all != MISSING_SENTINEL))
    if not len(rows):
        raise ValueError("No finite, non-sentinel ZPHOT targets")
    target_original = torch.from_numpy(target_all[rows].astype(np.float32))

    flux_columns, radius_columns, error_columns = select_gnlll_columns(
        table.dtype.names or (),
    )
    signal_columns = flux_columns + radius_columns
    ids = np.asarray(table[base.OBJECT_ID_COLUMN])[rows].astype(np.int64).tolist()
    splits = base.make_random_split(
        len(rows), train_fraction=args.train_fraction, test_fraction=args.test_fraction,
        val_fraction=args.val_fraction, seed=args.seed,
    )
    split_counts = {name: int(np.count_nonzero(splits == name))
                    for name in ("train", "val", "test")}

    signal_x, signal_scaling = build_gnlll_features(
        table, rows, signal_columns,
        ["flux"] * len(flux_columns) + ["radius"] * len(radius_columns),
        splits, mode=args.feature_scaling,
    )
    error_x, error_scaling = build_gnlll_features(
        table, rows, error_columns, ["error"] * len(error_columns),
        splits, mode=args.feature_scaling,
    )
    target, target_transform = scale_target_from_train(
        target_original, splits, mode=args.feature_scaling,
    )
    if signal_x.shape[1] != 66 or error_x.shape[1] != 55:
        raise RuntimeError(
            f"Expected 66 signal and 55 error dimensions; got "
            f"{signal_x.shape[1]} and {error_x.shape[1]}"
        )
    signal_preprocessing_signature = hashlib.sha256(
        json.dumps(signal_scaling, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    args.cache_root = args.cache_root.expanduser() / (
        f"signal_preprocessing_{signal_preprocessing_signature}"
    )

    # IoTFM and MLP see exactly the same sanitized/scaled 66 signal values.
    embedding_table = {
        column: signal_x[:, index].numpy()
        for index, column in enumerate(signal_columns)
    }
    embedding_rows = np.arange(len(rows), dtype=np.int64)
    serialization = CatalogueSerializationConfig(
        schema_name=f"clauds_gnlll_signal_{args.feature_scaling}_v2", decimals=6,
        prefix=f"CLAUDS galaxy redshift-signal record ({args.feature_scaling})",
    )
    embeddings, embedding_cache, embedding_metadata = base.get_embeddings(
        args, embedding_table, embedding_rows, ids, signal_columns, serialization,
    )
    embeddings, embedding_mean, embedding_scale = standardize_from_train(embeddings, splits)
    embedding_standardization = {
        "mean": embedding_mean.tolist(), "scale": embedding_scale.tolist(),
        "fit_split": "train",
    }
    shared_preprocessing = {
        "feature_scaling": args.feature_scaling,
        "error_scaling": error_scaling,
        "target_transform": target_transform,
        "missing_value_policy": "sentinel_and_nonfinite_train_median_no_indicator",
    }
    iotfm_preprocessing = {
        **shared_preprocessing, "signal_scaling": signal_scaling,
        "embedding_standardization": embedding_standardization,
    }
    mlp_preprocessing = {**shared_preprocessing, "signal_scaling": signal_scaling}

    print("\nExperiment: matched IoTFM vs MLP heteroscedastic photo-z regression")
    print(f"GNLLL: {GNLLL_EXPANSION}; target: ZPHOT; rows: {len(rows):,}; splits: {split_counts}")
    print("Mean signal: 55 flux estimators + 11 Kron radii")
    print("Variance signal: 55 flux errors + detached inferred redshift mean")
    print(f"Feature/target scaling: {args.feature_scaling}")
    print("Missing policy: -99/non-finite sanitized, train-median imputation, no indicators")

    print("\nTraining 1/2: frozen IoTFM signal embedding + Gaussian NLL head")
    iotfm_model, iotfm_result = train_branch(
        "iotfm_gnlll", embeddings, error_x, target, splits, args, output / "iotfm",
        target_transform,
        iotfm_preprocessing,
    )
    print("\nTraining 2/2: matched tabular MLP + Gaussian NLL head")
    mlp_model, mlp_result = train_branch(
        "mlp_gnlll", signal_x, error_x, target, splits, args, output / "mlp",
        target_transform,
        mlp_preprocessing,
    )

    device = base.resolve_torch_device(args.device)
    comparison_split = "test" if split_counts["test"] else "val"
    test_iotfm_loader = make_loader(
        embeddings, error_x, target, splits, comparison_split,
        batch_size=args.eval_batch_size, shuffle=False,
    )
    test_mlp_loader = make_loader(
        signal_x, error_x, target, splits, comparison_split,
        batch_size=args.eval_batch_size, shuffle=False,
    )
    iotfm_raw = evaluate(iotfm_model.to(device), test_iotfm_loader, device, target_transform)
    mlp_raw = evaluate(mlp_model.to(device), test_mlp_loader, device, target_transform)
    z_min, z_max = float(target_original.min()), float(target_original.max())
    edges = torch.linspace(z_min, z_max, args.n_z_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    iotfm_eval = gaussian_evaluation(iotfm_raw, edges, centers)
    mlp_eval = gaussian_evaluation(mlp_raw, edges, centers)
    iotfm_result["test_metrics"] = iotfm_raw["metrics"]
    mlp_result["test_metrics"] = mlp_raw["metrics"]

    output.mkdir(parents=True, exist_ok=True)
    artifacts = save_figures(
        iotfm_result, mlp_result, iotfm_eval, mlp_eval, args, output,
    )
    summary = output / "iotfm_mlp_gnlll_results.pt"
    torch.save({"iotfm": iotfm_result, "mlp": mlp_result}, summary)
    manifest = {
        "experiment": f"iotfm_mlp_gnlll_{args.feature_scaling}",
        "gnlll_expansion": GNLLL_EXPANSION,
        "gnlll_formula": "0.5 * (log(variance) + (target - mean)^2 / variance)",
        "catalogue": str(catalogue), "target": "ZPHOT", "model": args.model,
        "n_rows": len(rows), "split_counts": split_counts,
        "mean_signal": "55 flux estimators + 11 Kron radii",
        "variance_signal": "55 flux errors + detached inferred redshift mean",
        "flux_columns": flux_columns, "radius_columns": radius_columns,
        "flux_error_columns": error_columns,
        "feature_scaling": args.feature_scaling,
        "signal_preprocessing_signature": signal_preprocessing_signature,
        "signal_scaling": signal_scaling, "error_scaling": error_scaling,
        "target_transform": target_transform,
        "missing_value_policy": "sentinel_and_nonfinite_train_median_no_indicator",
        "mean_warmup_epochs": args.mean_warmup_epochs,
        "gnlll_epochs": args.gnlll_epochs, "variance_floor": args.variance_floor,
        "input_cache": str(input_cache), "input_cache_metadata": input_cache_metadata,
        "embedding_cache": str(embedding_cache),
        "embedding_metadata": embedding_metadata,
        "embedding_standardization": {
            "mean": embedding_mean.tolist(), "scale": embedding_scale.tolist(),
            "fit_split": "train",
        },
        "comparison_split": comparison_split,
        "metrics": {"iotfm": iotfm_raw["metrics"], "mlp": mlp_raw["metrics"]},
        "summary": str(summary), "artifacts": artifacts,
    }
    manifest_path = output / "iotfm_mlp_gnlll_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nsummary: {summary}\nmanifest: {manifest_path}")
    for key, path in artifacts.items():
        print(f"comparison {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
