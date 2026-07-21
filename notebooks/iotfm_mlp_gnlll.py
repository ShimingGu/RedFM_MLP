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
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalogue", type=Path,
                   default=Path("data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/arc/home/gsm/aion_output/figures/iotfm_mlp_gnlll"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("/scratch/.tmp-gsm/aion_output/cache/iotfm_mlp_gnlll"))
    p.add_argument("--max-rows", type=base.max_rows_arg, default=200000)
    p.add_argument("--model", default="GLM-5.2-0.8B-A0.8B")
    p.add_argument("--embedding-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--pooling", choices=("mean", "last", "mean_last"), default="mean")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--force-recompute-embeddings", action="store_true")
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
             device: torch.device) -> dict[str, Any]:
    model.eval()
    means, variances, targets = [], [], []
    for mean_x, error_x, target in loader:
        mean, variance = model(mean_x.to(device), error_x.to(device))
        means.append(mean.cpu()); variances.append(variance.cpu()); targets.append(target.cpu())
    mean = torch.cat(means)
    variance = torch.cat(variances)
    target = torch.cat(targets)
    sigma = variance.sqrt()
    residual = target - mean
    nll = .5 * (variance.log() + residual.square() / variance)
    point = point_photoz_metrics(mean, target)
    metrics = {
        **point,
        "gaussian_nll": float(nll.mean()),
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
                 output_dir: Path) -> tuple[HeteroscedasticPhotoZRegressor, dict[str, Any]]:
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
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "stage": "mean_warmup", "train_loss": total / count,
               "val_gaussian_nll": val["metrics"]["gaussian_nll"],
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
        val = evaluate(model, val_loader, device)
        val_loss = val["metrics"]["gaussian_nll"]
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
    # These fixed attributes let us reuse the well-tested IoTFM cache/extraction
    # routine while keeping identity, location, flags, and classifications out.
    args.include_id = False; args.include_location = False
    args.magnitudes_only = False; args.ignore_missingness = True
    args.exclude_flags = False; args.no_classification = False

    catalogue = args.catalogue.expanduser().resolve()
    output = args.output_dir.expanduser()
    split_dir = args.cache_root.expanduser() / (
        "all" if args.max_rows is None else f"n{args.max_rows}"
    ) / "clauds_split"
    table = base.load_clauds_catalogue_from_fits(
        catalogue, split_dir, max_rows=args.max_rows, sample_mode="random",
        sample_seed=args.seed, sample_require_valid_bands=(),
    )
    rows, target_np, _ = base.select_training_rows(table, no_classification=False)
    if not len(rows):
        raise ValueError("No finite ZPHOT targets")
    flux_columns, radius_columns, error_columns = select_gnlll_columns(list(table.keys()))
    signal_columns = flux_columns + radius_columns
    ids = np.asarray(table[base.OBJECT_ID_COLUMN])[rows].astype(np.int64).tolist()
    splits = base.make_random_split(
        len(rows), train_fraction=args.train_fraction, test_fraction=args.test_fraction,
        val_fraction=args.val_fraction, seed=args.seed,
    )
    split_counts = {name: int(np.count_nonzero(splits == name))
                    for name in ("train", "val", "test")}
    target = torch.from_numpy(target_np.astype(np.float32))

    signal_x, signal_feature_names = base.build_mlp_features(
        table, rows, signal_columns, splits, encode_missingness=False,
    )
    error_x, error_feature_names = base.build_mlp_features(
        table, rows, error_columns, splits, encode_missingness=False,
    )
    if signal_x.shape[1] != 66 or error_x.shape[1] != 55:
        raise RuntimeError(f"Expected 66 signal and 55 error dimensions; got "
                           f"{signal_x.shape[1]} and {error_x.shape[1]}")
    serialization = CatalogueSerializationConfig(
        schema_name="clauds_gnlll_signal_v1", decimals=6,
        prefix="CLAUDS galaxy redshift-signal record",
    )
    embeddings, embedding_cache, embedding_metadata = base.get_embeddings(
        args, table, rows, ids, signal_columns, serialization,
    )
    embeddings, embedding_mean, embedding_scale = standardize_from_train(embeddings, splits)

    print("\nExperiment: matched IoTFM vs MLP heteroscedastic photo-z regression")
    print(f"GNLLL: {GNLLL_EXPANSION}; target: ZPHOT; rows: {len(rows):,}; splits: {split_counts}")
    print("Mean signal: 55 flux estimators + 11 Kron radii")
    print("Variance signal: 55 flux errors + detached inferred redshift mean")
    print("Missing numeric values: training-median imputation; no missingness indicators")

    print("\nTraining 1/2: frozen IoTFM signal embedding + Gaussian NLL head")
    iotfm_model, iotfm_result = train_branch(
        "iotfm_gnlll", embeddings, error_x, target, splits, args, output / "iotfm",
    )
    print("\nTraining 2/2: matched tabular MLP + Gaussian NLL head")
    mlp_model, mlp_result = train_branch(
        "mlp_gnlll", signal_x, error_x, target, splits, args, output / "mlp",
    )

    device = base.resolve_torch_device(args.device)
    comparison_split = "test" if split_counts["test"] else "val"
    test_iotfm_loader = make_loader(embeddings, error_x, target, splits, comparison_split,
                                    batch_size=args.eval_batch_size, shuffle=False)
    test_mlp_loader = make_loader(signal_x, error_x, target, splits, comparison_split,
                                  batch_size=args.eval_batch_size, shuffle=False)
    iotfm_raw = evaluate(iotfm_model.to(device), test_iotfm_loader, device)
    mlp_raw = evaluate(mlp_model.to(device), test_mlp_loader, device)
    z_min, z_max = float(target.min()), float(target.max())
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
        "experiment": "iotfm_mlp_gnlll",
        "gnlll_expansion": GNLLL_EXPANSION,
        "gnlll_formula": "0.5 * (log(variance) + (target - mean)^2 / variance)",
        "catalogue": str(catalogue), "target": "ZPHOT", "model": args.model,
        "n_rows": len(rows), "split_counts": split_counts,
        "mean_signal": "55 flux estimators + 11 Kron radii",
        "variance_signal": "55 flux errors + detached inferred redshift mean",
        "flux_columns": flux_columns, "radius_columns": radius_columns,
        "flux_error_columns": error_columns,
        "signal_feature_names": signal_feature_names,
        "error_feature_names": error_feature_names,
        "missing_value_policy": "train_median_impute_no_indicator",
        "mean_warmup_epochs": args.mean_warmup_epochs,
        "gnlll_epochs": args.gnlll_epochs, "variance_floor": args.variance_floor,
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
