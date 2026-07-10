from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import AIONMagnitudeConfig, resolve_training_paths
from .clauds_bands import HSC_AION_BANDS, REDSHIFT_COLUMNS
from .utils import (
    select_torch_device, set_random_seed, make_redshift_grid
)
from .dataset import (
    dataset_for_split, CachedFusionDataset, collate_cached_fusion
)
from .models import (
    build_baseline_model, load_baseline_model_from_checkpoint
)
from .metrics import (
    redshift_cross_entropy_loss, point_photoz_metrics, discrete_crps,
    pit_values, conformal_hpd_threshold, conformal_hpd_set_mask,
    evaluate_conformal_hpd, summarize_pdf_metrics
)


def logits_from_cached_batch(
    model: nn.Module,
    batch: CachedFusionBatch,
    model_kind: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    device = resolve_torch_device(device)
    aion_embedding = batch.aion_embedding.to(device)
    extra_features = batch.extra_features.to(device)

    if model_kind == "fusion":
        return model(aion_embedding, extra_features)
    if model_kind == "aion":
        return model(aion_embedding)
    if model_kind == "tabular":
        return model(extra_features)

    raise ValueError(f"Unknown model_kind: {model_kind}")


def train_pdf_model_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_kind: str = "fusion",
    device: torch.device | str | None = None,
    redshift_edges: torch.Tensor | None = None,
) -> float:
    device = resolve_torch_device(device)
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        if batch.z_spec is None:
            raise ValueError("Training requires z_spec labels.")

        optimizer.zero_grad(set_to_none=True)
        logits = logits_from_cached_batch(model, batch, model_kind=model_kind, device=device)
        z_spec = batch.z_spec.to(device)
        loss = redshift_cross_entropy_loss(logits, z_spec, edges=redshift_edges)
        loss.backward()
        optimizer.step()

        batch_size = z_spec.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate_pdf_model(
    model: nn.Module,
    loader: DataLoader,
    model_kind: str = "fusion",
    device: torch.device | str | None = None,
    redshift_edges: torch.Tensor | None = None,
    redshift_centers: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float]:
    device = resolve_torch_device(device)
    model.eval()
    logits_parts = []
    z_parts = []
    redshift_reference_parts: dict[str, list[torch.Tensor]] = {}

    for batch in loader:
        logits = logits_from_cached_batch(model, batch, model_kind=model_kind, device=device)
        logits_parts.append(logits.cpu())
        if batch.z_spec is not None:
            z_parts.append(batch.z_spec.cpu())
        if batch.redshift_reference:
            for key, values in batch.redshift_reference.items():
                redshift_reference_parts.setdefault(key, []).append(values.cpu())

    logits = torch.cat(logits_parts, dim=0)
    if redshift_edges is None:
        redshift_edges, inferred_centers = make_redshift_grid(n_z_bins=logits.shape[-1])
        if redshift_centers is None:
            redshift_centers = inferred_centers
    if redshift_centers is None:
        redshift_centers = 0.5 * (redshift_edges[:-1] + redshift_edges[1:])
    output: dict[str, torch.Tensor | float] = {
        "logits": logits,
        "redshift_edges": redshift_edges.detach().cpu(),
        "redshift_centers": redshift_centers.detach().cpu(),
    }
    output.update(predict_photoz_from_logits(logits, centers=redshift_centers))

    if z_parts:
        z_spec = torch.cat(z_parts, dim=0)
        output["z_spec"] = z_spec
        output["loss"] = redshift_cross_entropy_loss(logits, z_spec, edges=redshift_edges).item()
    if redshift_reference_parts:
        output["redshift_reference"] = {
            key: torch.cat(parts, dim=0)
            for key, parts in redshift_reference_parts.items()
        }

    return output


def cached_product_to_dataset(product: Mapping[str, Any]) -> CachedFusionDataset:
    return CachedFusionDataset(
        object_ids=product["object_id"],
        fields=product["field"],
        aion_embeddings=product["aion_embedding"],
        extra_features=product["extra_features"],
        z_spec=product["z_spec"],
        redshift_reference=product.get("redshift_reference"),
    )


def split_cached_product(product: Mapping[str, Any]) -> tuple[CachedFusionDataset, CachedFusionDataset, CachedFusionDataset]:
    dataset = cached_product_to_dataset(product)
    split_labels = np.asarray(product["split_labels"])
    return (
        subset_cached_dataset(dataset, split_labels == "train"),
        subset_cached_dataset(dataset, split_labels == "val"),
        subset_cached_dataset(dataset, split_labels == "test"),
    )


def make_cached_loader(dataset: CachedFusionDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_cached_fusion,
    )


def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: CachedFusionDataset,
    model_kind: str,
    *,
    batch_size: int = 512,
    device: torch.device | str | None = None,
    redshift_edges: torch.Tensor | None = None,
    redshift_centers: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float]:
    loader = make_cached_loader(dataset, batch_size, shuffle=False)
    return evaluate_pdf_model(
        model,
        loader,
        model_kind=model_kind,
        device=device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )


def evaluate_model_by_field(
    model: nn.Module,
    dataset: CachedFusionDataset,
    model_kind: str,
    *,
    batch_size: int = 512,
    min_objects: int = 5,
    device: torch.device | str | None = None,
    redshift_edges: torch.Tensor | None = None,
    redshift_centers: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    fields = np.asarray(dataset.fields)
    metrics_by_field = {}
    for field in sorted(dict.fromkeys(fields.tolist())):
        mask = fields == field
        if int(mask.sum()) < min_objects:
            continue
        field_dataset = subset_cached_dataset(dataset, mask)
        evaluation = evaluate_model_on_dataset(
            model,
            field_dataset,
            model_kind,
            batch_size=batch_size,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        )
        field_metrics = summarize_pdf_metrics(evaluation)
        field_metrics["n_objects"] = int(mask.sum())
        metrics_by_field[str(field)] = field_metrics
    return metrics_by_field


def save_calibration_artifacts(
    evaluation: dict[str, torch.Tensor | float],
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = output_dir / f"{prefix}_zpred_vs_zphot.png"
    pit_path = output_dir / f"{prefix}_pit_histogram.png"
    fig, _ = plot_zpred_vs_zphot(
        evaluation,
        scatter_path,
        title=f"{prefix}: z_pred vs z_phot",
    )
    plt.close(fig)
    fig, _ = plot_pit_histogram(
        evaluation,
        pit_path,
        title=f"{prefix}: PIT histogram",
    )
    plt.close(fig)
    return {
        "diagnostics": calibration_diagnostics(evaluation),
        "zpred_vs_zphot_plot": str(scatter_path),
        "pit_histogram_plot": str(pit_path),
    }


def train_single_baseline(
    product: Mapping[str, Any],
    model_kind: str,
    *,
    output_dir: str | Path,
    n_z_bins: int = 300,
    redshift_edges: torch.Tensor | None = None,
    redshift_centers: torch.Tensor | None = None,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    device = resolve_torch_device(device)
    if redshift_edges is None or redshift_centers is None:
        redshift_edges, redshift_centers = make_redshift_grid(n_z_bins=n_z_bins)
    train_dataset, val_dataset, test_dataset = split_cached_product(product)
    if len(train_dataset) == 0:
        raise ValueError("No training rows are available. Check TEST_FIELDS and split labels.")
    if len(val_dataset) == 0:
        raise ValueError("No validation rows are available. Increase data size or val_fraction.")

    train_loader = make_cached_loader(train_dataset, train_batch_size, shuffle=True)
    val_loader = make_cached_loader(val_dataset, eval_batch_size, shuffle=False)

    aion_dim = product["aion_embedding"].shape[1]
    extra_feature_dim = product["extra_features"].shape[1]
    if model_kind in {"tabular", "fusion"} and extra_feature_dim == 0:
        raise ValueError(
            f"model_kind={model_kind!r} requires at least one MLP feature. "
            "Set use_mlp_features=True with grizy and/or extra bands, or use model_kind='aion'."
        )
    if model_kind in {"aion", "fusion"} and aion_dim == 0:
        raise ValueError(
            f"model_kind={model_kind!r} requires AION embeddings. "
            "Set use_aion_embedding=True, or use model_kind='tabular' for MLP-only training."
        )
    model = build_baseline_model(model_kind, aion_dim, extra_feature_dim, n_z_bins=n_z_bins).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = train_pdf_model_one_epoch(
            model,
            train_loader,
            optimizer,
            model_kind=model_kind,
            device=device,
            redshift_edges=redshift_edges,
        )
        val_eval = evaluate_pdf_model(
            model,
            val_loader,
            model_kind=model_kind,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        )
        val_metrics = summarize_pdf_metrics(val_eval)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"{model_kind:7s} epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['cross_entropy']:.4f} "
            f"val_nmad={val_metrics['nmad']:.4f}"
        )
        if val_metrics["cross_entropy"] < best_val_loss:
            best_val_loss = val_metrics["cross_entropy"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_eval = evaluate_model_on_dataset(
        model,
        val_dataset,
        model_kind,
        batch_size=eval_batch_size,
        device=device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )
    final_metrics: dict[str, Any] = {
        "val": summarize_pdf_metrics(val_eval),
        "val_by_field": evaluate_model_by_field(
            model,
            val_dataset,
            model_kind,
            batch_size=eval_batch_size,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        ),
    }
    calibration: dict[str, Any] = {
        "val": save_calibration_artifacts(val_eval, output_dir, f"{model_kind}_val")
    }

    if len(test_dataset) > 0:
        test_eval = evaluate_model_on_dataset(
            model,
            test_dataset,
            model_kind,
            batch_size=eval_batch_size,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        )
        final_metrics["test"] = summarize_pdf_metrics(test_eval)
        final_metrics["test_by_field"] = evaluate_model_by_field(
            model,
            test_dataset,
            model_kind,
            batch_size=eval_batch_size,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        )
        calibration["test"] = save_calibration_artifacts(test_eval, output_dir, f"{model_kind}_test")
        calibration["conformal_hpd"] = evaluate_conformal_hpd(val_eval, test_eval)
    else:
        calibration["conformal_hpd"] = {
            "status": "skipped_no_test_split",
            "message": "Set TEST_FIELDS to evaluate conformal HPD coverage on held-out fields.",
        }

    checkpoint_path = output_dir / f"{model_kind}_baseline.pt"
    torch.save(
        {
            "model_kind": model_kind,
            "state_dict": model.state_dict(),
            "history": history,
            "final_metrics": final_metrics,
            "calibration": calibration,
            "metadata": {
                **dict(product.get("metadata", {})),
                "n_z_bins": n_z_bins,
                "redshift_edges": redshift_edges.detach().cpu(),
                "redshift_centers": redshift_centers.detach().cpu(),
            },
        },
        checkpoint_path,
    )

    return {
        "model_kind": model_kind,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "final_metrics": final_metrics,
        "calibration": calibration,
    }


def train_all_baselines(
    product: Mapping[str, Any],
    *,
    output_dir: str | Path,
    model_kinds: Sequence[str] = ("tabular", "aion", "fusion"),
    n_z_bins: int = 300,
    redshift_edges: torch.Tensor | None = None,
    redshift_centers: torch.Tensor | None = None,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    device = resolve_torch_device(device)
    if redshift_edges is None or redshift_centers is None:
        redshift_edges, redshift_centers = make_redshift_grid(n_z_bins=n_z_bins)
    results = {}
    for model_kind in model_kinds:
        results[model_kind] = train_single_baseline(
            product,
            model_kind,
            output_dir=output_dir,
            n_z_bins=n_z_bins,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            device=device,
        )
    summary_path = Path(output_dir) / "baseline_results.pt"
    torch.save(results, summary_path)
    return results


def run_baseline_training(
    config: AIONMagnitudeConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Execute the notebook's embedding-cache and baseline-training workflow."""
    config = make_magnitude_config(config, **overrides)
    set_random_seed(config.seed)
    redshift_edges, redshift_centers = make_redshift_grid(config.z_min, config.z_max, config.n_z_bins)
    run_device = select_torch_device(config.device_choice)
    paths = resolve_training_paths(config)

    print("Starting embedding extraction and caching...")
    cached_product = build_and_cache_aion_embeddings_from_config(config)
    print(f"Cached product loaded from: {paths['cache_path']}")
    print(f"Metadata: {cached_product.get('metadata', {})}")

    print("\nStarting baseline training...")
    baseline_results = train_all_baselines(
        cached_product,
        output_dir=paths["baseline_output_dir"],
        model_kinds=config.model_kinds,
        n_z_bins=config.n_z_bins,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
        epochs=config.baseline_epochs,
        learning_rate=config.baseline_learning_rate,
        weight_decay=config.baseline_weight_decay,
        train_batch_size=config.baseline_train_batch_size,
        eval_batch_size=config.baseline_eval_batch_size,
        device=run_device,
    )
    print("Baseline training completed successfully!")

    train_dataset, val_dataset, test_dataset = split_cached_product(cached_product)
    return {
        "config": config,
        "device": run_device,
        "paths": paths,
        "cached_product": cached_product,
        "baseline_results": baseline_results,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


def load_and_evaluate_baseline(
    product: Mapping[str, Any],
    *,
    model_kind: str = "fusion",
    split: str = "val",
    checkpoint_path: str | Path | None = None,
    config: AIONMagnitudeConfig | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Load a trained checkpoint and evaluate it on a cached product split."""
    config = make_magnitude_config(config)
    redshift_edges, redshift_centers = make_redshift_grid(config.z_min, config.z_max, config.n_z_bins)
    run_device = resolve_torch_device(device if device is not None else config.device_choice)
    paths = resolve_training_paths(config)
    if checkpoint_path is None:
        checkpoint_path = Path(paths["baseline_output_dir"]) / f"{model_kind}_baseline.pt"

    model = load_baseline_model_from_checkpoint(
        checkpoint_path,
        model_kind=model_kind,
        aion_dim=product["aion_embedding"].shape[1],
        extra_feature_dim=product["extra_features"].shape[1],
        n_z_bins=config.n_z_bins,
        device=run_device,
    )
    dataset = dataset_for_split(product, split)
    evaluation = evaluate_model_on_dataset(
        model,
        dataset,
        model_kind,
        batch_size=batch_size or config.baseline_eval_batch_size,
        device=run_device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )
    return {
        "model": model,
        "dataset": dataset,
        "evaluation": evaluation,
        "checkpoint_path": Path(checkpoint_path),
        "model_kind": model_kind,
        "split": split,
    }


def run_training_and_evaluation(
    config: AIONMagnitudeConfig | None = None,
    *,
    model_kind: str = "fusion",
    split: str = "val",
    **overrides: Any,
) -> dict[str, Any]:
    """Run training, then load/evaluate one trained model for plotting."""
    config = make_magnitude_config(config, **overrides)
    if not config.use_mlp_features and model_kind != "aion":
        warnings.warn(
            "use_mlp_features=False requested AION-only training; evaluating model_kind='aion'.",
            RuntimeWarning,
            stacklevel=2,
        )
        model_kind = "aion"
    elif not config.use_aion_embedding and model_kind != "tabular":
        warnings.warn(
            "use_aion_embedding=False requested MLP-only training; evaluating model_kind='tabular'.",
            RuntimeWarning,
            stacklevel=2,
        )
        model_kind = "tabular"
    training = run_baseline_training(config)
    evaluated = load_and_evaluate_baseline(
        training["cached_product"],
        model_kind=model_kind,
        split=split,
        config=training["config"],
        device=training["device"],
    )
    return {**training, **evaluated}
