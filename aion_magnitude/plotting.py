from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import matplotlib.pyplot as plt

from .clauds_bands import REDSHIFT_COLUMNS, HSC_AION_BANDS
from .utils import (
    gaussian_smooth_1d, make_redshift_grid
)
from .metrics import (
    redshift_probability_distribution, pit_values, point_photoz_metrics
)


def plot_zpred_vs_zphot(
    evaluation: dict[str, torch.Tensor | float],
    output_path: str | Path | None = None,
    *,
    pred_key: str = "z_p50",
    target_label: str = "z_phot",
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = "z_pred vs z_phot",
    max_points: int = 50_000,
    pmin = None,
    pmax = None,
    show_metrics: bool = False,
    seed: int = 42,
    ax=None,
):
    if "z_spec" not in evaluation:
        raise ValueError("Evaluation output does not include z_spec.")

    z_true = evaluation["z_spec"].detach().cpu().numpy()
    z_pred = evaluation[pred_key].detach().cpu().numpy()
    if len(z_true) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z_true), size=max_points, replace=False)
        z_true = z_true[idx]
        z_pred = z_pred[idx]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
    else:
        fig = ax.figure
    ax.scatter(z_true, z_pred, s=4, alpha=0.25, linewidths=0)
    redshift_edges = evaluation.get("redshift_edges")
    if redshift_edges is not None:
        edge_values = redshift_edges.detach().cpu().numpy()
        default_min = float(edge_values[0])
        default_max = float(edge_values[-1])
    else:
        default_min = 0.0
        default_max = 6.0
    if pmin is None:
        lim_min = float(np.nanmin([np.nanmin(z_true), np.nanmin(z_pred), default_min]))
    else:
        lim_min = pmin
    if pmax is None:
        lim_max = float(np.nanmax([np.nanmax(z_true), np.nanmax(z_pred), default_max]))
    else:
        lim_max = pmax
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=1.0)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(target_label if x_label is None else x_label)
    ax.set_ylabel(pred_key if y_label is None else y_label)

    if show_metrics:
        metrics = point_photoz_metrics(evaluation[pred_key], evaluation["z_spec"])
        nmad = metrics["nmad"]
        olf = metrics["catastrophic_outlier_fraction"]

        # Generic regression diagnostics computed on the plotted sample.
        r2 = 1.0 - (np.sum((z_true - z_pred) ** 2) / np.sum((z_true - np.mean(z_true)) ** 2))
        pearson_r = np.corrcoef(z_true, z_pred)[0, 1]

        title = (
            f"{title}\n"
            f"$\\sigma_{{NMAD}}$={nmad:.4f}, $\\eta$={olf:.2%}\n"
            f"$R^2$={r2:.4f}, Pearson $\\rho$={pearson_r:.4f}"
        )
        
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if output_path is not None or created_fig:
        fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    return fig, ax


def plot_pit_histogram(
    evaluation: dict[str, torch.Tensor | float],
    output_path: str | Path | None = None,
    *,
    bins: int = 20,
    title: str = "PIT histogram",
    ax=None,
):
    pit = pit_values(
        evaluation["pz"],
        evaluation["z_spec"],
        edges=evaluation.get("redshift_edges"),
    ).detach().cpu().numpy()
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
    else:
        fig = ax.figure
    ax.hist(pit, bins=bins, range=(0.0, 1.0), histtype="stepfilled", alpha=0.7)
    ax.set_xlabel("PIT")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if output_path is not None or created_fig:
        fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    return fig, ax


def apply_baseline_to_catalogue(
    *,
    catalogue_path: str | Path,
    checkpoint_path: str | Path,
    model_kind: str = "fusion",
    split_output_dir: str | Path | None = None,
    cache_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_rows: int | None = 20_000,
    field_column: str | None = None,
    target_redshift_column: str = REDSHIFT_COLUMNS["zphot"],
    z_min: float = 0.0,
    z_max: float = 6.0,
    n_z_bins: int = 300,
    mag_zero_point: float = 23.0,
    hsc_mag_faint_limits: Mapping[str, float | None] | None = None,
    extra_bands: Sequence[str] | None = None,
    extra_band_invalid_fill: str | float = "median",
    extra_band_include_valid_flags: bool = False,
    use_mlp_features: bool = True,
    include_grizy_in_mlp: bool | None = None,
    batch_size: int = 512,
    force_recompute_embeddings: bool = False,
    use_aion_embedding: bool = True,
    aion_input_bands: Sequence[str] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Apply a trained baseline checkpoint to a second CLAUDS-style catalogue."""
    device = resolve_torch_device(device)
    include_grizy_in_mlp = resolve_include_grizy_in_mlp(
        include_grizy_in_mlp,
        use_aion_embedding=use_aion_embedding,
        use_mlp_features=use_mlp_features,
        extra_bands=extra_bands,
    )
    redshift_edges, redshift_centers = make_redshift_grid(z_min, z_max, n_z_bins)
    run_tag = make_cache_run_tag(catalogue_path, max_rows, mag_zero_point)
    if split_output_dir is None:
        split_output_dir = Path("cache") / f"apply_split_{run_tag}"
    if cache_path is None:
        cache_path = Path("cache") / f"apply_aion_embeddings_{run_tag}.pt"
    if output_dir is None:
        output_dir = Path("cache") / f"apply_{model_kind}_{run_tag}"

    product = build_and_cache_aion_embeddings(
        catalogue_path=catalogue_path,
        split_output_dir=split_output_dir,
        cache_path=cache_path,
        max_rows=max_rows,
        field_column=field_column,
        target_redshift_column=target_redshift_column,
        z_min=z_min,
        z_max=z_max,
        n_z_bins=n_z_bins,
        mag_zero_point=mag_zero_point,
        hsc_mag_faint_limits=hsc_mag_faint_limits or default_hsc_mag_faint_limits(),
        extra_bands=extra_bands,
        extra_band_invalid_fill=extra_band_invalid_fill,
        extra_band_include_valid_flags=extra_band_include_valid_flags,
        use_mlp_features=use_mlp_features,
        include_grizy_in_mlp=include_grizy_in_mlp,
        test_fields=[],
        batch_size=batch_size,
        force_recompute_embeddings=force_recompute_embeddings,
        use_aion_embedding=use_aion_embedding,
        aion_input_bands=aion_input_bands,
        device=device,
    )

    print(product.get("metadata", {}))

    dataset = cached_product_to_dataset(product)
    model = load_baseline_model_from_checkpoint(
        checkpoint_path,
        model_kind=model_kind,
        aion_dim=product["aion_embedding"].shape[1],
        extra_feature_dim=product["extra_features"].shape[1],
        n_z_bins=n_z_bins,
        device=device,
    )
    evaluation = evaluate_model_on_dataset(
        model,
        dataset,
        model_kind,
        batch_size=batch_size,
        device=device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )
    metrics = summarize_pdf_metrics(evaluation) if dataset.z_spec is not None else None
    by_field = evaluate_model_by_field(
        model,
        dataset,
        model_kind,
        batch_size=batch_size,
        device=device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = output_dir / f"{model_kind}_zpred_vs_zphot.png"
    fig, _ = plot_zpred_vs_zphot(
        evaluation,
        output_path=scatter_path,
        pred_key="z_p50",
        x_label=target_redshift_column,
        y_label="new z_p50",
        title=f"{model_kind}: applied catalogue z_pred vs z_phot",
        show_metrics=True,
    )
    plt.close(fig)

    return {
        "product": product,
        "dataset": dataset,
        "evaluation": evaluation,
        "metrics": metrics,
        "by_field": by_field,
        "scatter_path": str(scatter_path),
        "cache_path": str(cache_path),
        "output_dir": str(output_dir),
    }


def plot_redshift_probability_distribution(
    evaluation: Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    gaussian_sigma_bins: float = 0.0,
    gaussian_truncate: float = 4.0,
    include_true: bool = True,
    include_catalogue_hist: bool | None = None,
    title: str = "Redshift probability distribution",
    ax=None,
):
    """Plot true and recovered redshift PDFs after applying the same smoothing kernel."""
    if include_catalogue_hist is not None:
        include_true = include_catalogue_hist
    pdf_data = redshift_probability_distribution(
        evaluation,
        gaussian_sigma_bins=gaussian_sigma_bins,
        gaussian_truncate=gaussian_truncate,
    )
    centers = pdf_data["centers"]
    recovered_density = pdf_data["recovered_density"]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
    else:
        fig = ax.figure
    smoothing_label = f"Gaussian sigma={gaussian_sigma_bins:g} bins"
    if include_true and "true_density" in pdf_data:
        ax.step(
            centers,
            pdf_data["true_density"],
            where="mid",
            linewidth=2.0,
            label=f"true ({smoothing_label})",
        )
    ax.step(
        centers,
        recovered_density,
        where="mid",
        linewidth=2.0,
        label=f"recovered ({smoothing_label})",
    )

    ax.set_xlabel("redshift")
    ax.set_ylabel("probability density")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    if output_path is not None or created_fig:
        fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    return fig, ax, pdf_data


def plot_nz_lensing_alike(
    evaluation: Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    zphot_bin: Sequence[float],
    inferred_bin_key: str = "z_p50",
    n_samples_per_object: int = 10,
    hist_bins: int | Sequence[float] | None = None,
    z_range: tuple[float, float] | None = None,
    gaussian_sigma_bins: float = 0.0,
    gaussian_truncate: float = 4.0,
    seed: int | None = 42,
    use_pz_if_available: bool = True,
    plot_empty_bins: bool = False,
    title: str = "Tomographic n(z)",
    ax=None,
):
    """Plot input and inferred tomographic n(z) curves with shared bin colors."""
    zphot_bins = validate_zphot_bins(zphot_bin)
    labels = tomographic_bin_labels(zphot_bins)
    n_tomo_bins = len(labels)
    if inferred_bin_key not in evaluation:
        raise ValueError(f"evaluation does not include inferred_bin_key={inferred_bin_key!r}.")

    input_bin_values, input_samples = sample_catalogue_redshift_per_object(
        evaluation,
        n_samples_per_object=n_samples_per_object,
        seed=seed,
    )
    inferred_bin_values = tensor_to_numpy_1d(evaluation[inferred_bin_key])
    inferred_samples = sample_inferred_redshift_per_object(
        evaluation,
        n_samples_per_object=n_samples_per_object,
        seed=None if seed is None else seed + 1,
        use_pz_if_available=use_pz_if_available,
    )

    if input_samples.shape[0] != inferred_samples.shape[0]:
        raise ValueError("Input and inferred samples must have the same number of objects.")
    if inferred_bin_values.size != input_samples.shape[0]:
        raise ValueError("inferred_bin_key must have one value per evaluated object.")

    input_bin_index = assign_tomographic_bins(input_bin_values, zphot_bins)
    inferred_bin_index = assign_tomographic_bins(inferred_bin_values, zphot_bins)
    hist_edges = resolve_redshift_hist_edges(
        evaluation,
        hist_bins=hist_bins,
        z_range=z_range,
        zphot_bin=zphot_bins,
    )
    hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = [f"C{i}" for i in range(10)]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
    else:
        fig = ax.figure
    nz_data: dict[str, Any] = {
        "hist_edges": hist_edges,
        "hist_centers": hist_centers,
        "zphot_bin": zphot_bins,
        "tomographic_labels": labels,
        "input_bin_index": input_bin_index,
        "inferred_bin_index": inferred_bin_index,
        "input": {},
        "inferred": {},
        "n_samples_per_object": n_samples_per_object,
        "inferred_bin_key": inferred_bin_key,
        "gaussian_sigma_bins": float(gaussian_sigma_bins),
    }

    for tomo_idx, label in enumerate(labels):
        color = colors[tomo_idx % len(colors)]
        input_mask = input_bin_index == tomo_idx
        inferred_mask = inferred_bin_index == tomo_idx

        input_density, input_probability, input_n_samples = probability_density_from_samples(
            input_samples[input_mask],
            hist_edges,
            gaussian_sigma_bins=gaussian_sigma_bins,
            gaussian_truncate=gaussian_truncate,
        )
        inferred_density, inferred_probability, inferred_n_samples = probability_density_from_samples(
            inferred_samples[inferred_mask],
            hist_edges,
            gaussian_sigma_bins=gaussian_sigma_bins,
            gaussian_truncate=gaussian_truncate,
        )

        nz_data["input"][label] = {
            "density": input_density,
            "probability": input_probability,
            "n_galaxies": int(input_mask.sum()),
            "n_samples": input_n_samples,
        }
        nz_data["inferred"][label] = {
            "density": inferred_density,
            "probability": inferred_probability,
            "n_galaxies": int(inferred_mask.sum()),
            "n_samples": inferred_n_samples,
        }

        if not plot_empty_bins and input_n_samples == 0 and inferred_n_samples == 0:
            continue
        if plot_empty_bins or input_n_samples > 0:
            ax.step(
                hist_centers,
                input_density,
                where="mid",
                color=color,
                linestyle=":",
                linewidth=1.6,
                label=f"{label} input",
            )
        if plot_empty_bins or inferred_n_samples > 0:
            ax.step(
                hist_centers,
                inferred_density,
                where="mid",
                color=color,
                linestyle="--",
                linewidth=1.6,
                label=f"{label} inferred",
            )

    ax.set_xlabel("redshift")
    ax.set_ylabel("probability density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    if output_path is not None or created_fig:
        fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    return fig, ax, nz_data


def _save_comparison_figure(fig, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)


def compare_zpred_vs_zphot(
    evaluation_1: Mapping[str, Any],
    evaluation_2: Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    labels: Sequence[str] = ("config_1", "config_2"),
    pred_key: str = "z_p50",
    target_label: str = "z_phot",
    title: str = "z_pred vs z_phot comparison",
    max_points: int = 50_000,
    pmin=None,
    pmax=None,
    show_metrics: bool = True,
    seed: int = 42,
):
    """Compare two z_pred-vs-z_phot plots as left/right subplots."""
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two strings.")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharex=True, sharey=True)
    plot_zpred_vs_zphot(
        evaluation_1,
        pred_key=pred_key,
        target_label=target_label,
        title=str(labels[0]),
        max_points=max_points,
        pmin=pmin,
        pmax=pmax,
        show_metrics=show_metrics,
        seed=seed,
        ax=axes[0],
    )
    plot_zpred_vs_zphot(
        evaluation_2,
        pred_key=pred_key,
        target_label=target_label,
        title=str(labels[1]),
        max_points=max_points,
        pmin=pmin,
        pmax=pmax,
        show_metrics=show_metrics,
        seed=seed,
        ax=axes[1],
    )
    fig.suptitle(title)
    fig.tight_layout()
    _save_comparison_figure(fig, output_path)
    return fig, axes


def compare_pit_histogram(
    evaluation_1: Mapping[str, Any],
    evaluation_2: Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    labels: Sequence[str] = ("config_1", "config_2"),
    bins: int = 20,
    title: str = "PIT histogram comparison",
):
    """Compare two PIT histograms as left/right subplots."""
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two strings.")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), sharex=True, sharey=True)
    plot_pit_histogram(evaluation_1, bins=bins, title=str(labels[0]), ax=axes[0])
    plot_pit_histogram(evaluation_2, bins=bins, title=str(labels[1]), ax=axes[1])
    fig.suptitle(title)
    fig.tight_layout()
    _save_comparison_figure(fig, output_path)
    return fig, axes


def compare_redshift_probability_distribution(
    evaluation_1: Mapping[str, Any],
    evaluation_2: Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    labels: Sequence[str] = ("config_1", "config_2"),
    gaussian_sigma_bins: float = 0.0,
    gaussian_truncate: float = 4.0,
    include_true: bool = True,
    title: str = "Redshift probability distribution comparison",
):
    """Compare two redshift probability distributions as top/bottom subplots."""
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two strings.")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0), sharex=True)
    _, _, pdf_data_1 = plot_redshift_probability_distribution(
        evaluation_1,
        gaussian_sigma_bins=gaussian_sigma_bins,
        gaussian_truncate=gaussian_truncate,
        include_true=include_true,
        title=str(labels[0]),
        ax=axes[0],
    )
    _, _, pdf_data_2 = plot_redshift_probability_distribution(
        evaluation_2,
        gaussian_sigma_bins=gaussian_sigma_bins,
        gaussian_truncate=gaussian_truncate,
        include_true=include_true,
        title=str(labels[1]),
        ax=axes[1],
    )
    fig.suptitle(title)
    fig.tight_layout()
    _save_comparison_figure(fig, output_path)
    return fig, axes, (pdf_data_1, pdf_data_2)


def compare_nz_lensing_alike(
    evaluation_1: Mapping[str, Any],
    evaluation_2: Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    labels: Sequence[str] = ("config_1", "config_2"),
    zphot_bin: Sequence[float],
    inferred_bin_key: str = "z_p50",
    n_samples_per_object: int = 10,
    hist_bins: int | Sequence[float] | None = None,
    z_range: tuple[float, float] | None = None,
    gaussian_sigma_bins: float = 0.0,
    gaussian_truncate: float = 4.0,
    seed: int | None = 42,
    use_pz_if_available: bool = True,
    plot_empty_bins: bool = False,
    title: str = "Tomographic n(z) comparison",
):
    """Compare two tomographic n(z) plots as top/bottom subplots."""
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two strings.")
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 9.0), sharex=True)
    _, _, nz_data_1 = plot_nz_lensing_alike(
        evaluation_1,
        zphot_bin=zphot_bin,
        inferred_bin_key=inferred_bin_key,
        n_samples_per_object=n_samples_per_object,
        hist_bins=hist_bins,
        z_range=z_range,
        gaussian_sigma_bins=gaussian_sigma_bins,
        gaussian_truncate=gaussian_truncate,
        seed=seed,
        use_pz_if_available=use_pz_if_available,
        plot_empty_bins=plot_empty_bins,
        title=str(labels[0]),
        ax=axes[0],
    )
    _, _, nz_data_2 = plot_nz_lensing_alike(
        evaluation_2,
        zphot_bin=zphot_bin,
        inferred_bin_key=inferred_bin_key,
        n_samples_per_object=n_samples_per_object,
        hist_bins=hist_bins,
        z_range=z_range,
        gaussian_sigma_bins=gaussian_sigma_bins,
        gaussian_truncate=gaussian_truncate,
        seed=None if seed is None else seed + 10_000,
        use_pz_if_available=use_pz_if_available,
        plot_empty_bins=plot_empty_bins,
        title=str(labels[1]),
        ax=axes[1],
    )
    fig.suptitle(title)
    fig.tight_layout()
    _save_comparison_figure(fig, output_path)
    return fig, axes, (nz_data_1, nz_data_2)


def _history_from_run_or_result(
    source: Mapping[str, Any],
    *,
    model_kind: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    if "history" in source:
        return list(source["history"]), source.get("model_kind", model_kind)

    if "baseline_results" in source:
        baseline_results = source["baseline_results"]
        selected_kind = model_kind or source.get("model_kind")
        if selected_kind is None:
            if len(baseline_results) != 1:
                raise ValueError(
                    "model_kind is required when source['baseline_results'] contains multiple models."
                )
            selected_kind = next(iter(baseline_results))
        if selected_kind not in baseline_results:
            raise KeyError(f"model_kind={selected_kind!r} not found in baseline_results.")
        result = baseline_results[selected_kind]
        return list(result["history"]), selected_kind

    if "results" in source:
        results = source["results"]
        selected_kind = model_kind
        if selected_kind is None:
            if len(results) != 1:
                raise ValueError("model_kind is required when source['results'] contains multiple models.")
            selected_kind = next(iter(results))
        if selected_kind not in results:
            raise KeyError(f"model_kind={selected_kind!r} not found in results.")
        result = results[selected_kind]
        return list(result["history"]), selected_kind

    raise KeyError("source must include 'history', 'baseline_results', or 'results'.")


def _history_arrays(
    history: Sequence[Mapping[str, Any]],
    *,
    train_key: str = "train_loss",
    val_key: str = "val_cross_entropy",
) -> dict[str, np.ndarray]:
    if not history:
        raise ValueError("history is empty.")
    epochs = np.asarray([row.get("epoch", idx) for idx, row in enumerate(history)], dtype=np.float64)
    output = {"epoch": epochs}
    for key in (train_key, val_key):
        if all(key in row for row in history):
            output[key] = np.asarray([float(row[key]) for row in history], dtype=np.float64)
    return output


def compare_config_loss(
    run_or_pair_1: Mapping[str, Any],
    run_or_result_2: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    *,
    model_kind_1: str | None = None,
    model_kind_2: str | None = None,
    labels: Sequence[str] = ("config_1", "config_2"),
    train_key: str = "train_loss",
    val_key: str = "val_cross_entropy",
    title: str = "Training loss comparison",
    ax=None,
):
    """Plot train and validation loss curves for two config runs.

    Accepts either:
      - the dict returned by run_config_pair(...), or
      - two run_training_and_evaluation(...) / train_single_baseline(...) results.
    """
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two strings.")

    if run_or_result_2 is None:
        if "run_1" not in run_or_pair_1 or "run_2" not in run_or_pair_1:
            raise ValueError("Pass run_or_result_2, or pass the dict returned by run_config_pair(...).")
        source_1 = run_or_pair_1["run_1"]
        source_2 = run_or_pair_1["run_2"]
        model_kind_1 = model_kind_1 or run_or_pair_1.get("model_kind_1")
        model_kind_2 = model_kind_2 or run_or_pair_1.get("model_kind_2")
    else:
        source_1 = run_or_pair_1
        source_2 = run_or_result_2

    history_1, resolved_kind_1 = _history_from_run_or_result(source_1, model_kind=model_kind_1)
    history_2, resolved_kind_2 = _history_from_run_or_result(source_2, model_kind=model_kind_2)
    arrays_1 = _history_arrays(history_1, train_key=train_key, val_key=val_key)
    arrays_2 = _history_arrays(history_2, train_key=train_key, val_key=val_key)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
    else:
        fig = ax.figure

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    entries = (
        (arrays_1, str(labels[0]), resolved_kind_1, colors[0 % len(colors)]),
        (arrays_2, str(labels[1]), resolved_kind_2, colors[1 % len(colors)]),
    )
    for arrays, label, resolved_kind, color in entries:
        kind_suffix = "" if resolved_kind is None else f" [{resolved_kind}]"
        if train_key in arrays:
            ax.plot(
                arrays["epoch"],
                arrays[train_key],
                color=color,
                linestyle="-",
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=f"{label}{kind_suffix} train",
            )
        if val_key in arrays:
            ax.plot(
                arrays["epoch"],
                arrays[val_key],
                color=color,
                linestyle="--",
                linewidth=1.8,
                marker="s",
                markersize=3,
                label=f"{label}{kind_suffix} val",
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    if output_path is not None or created_fig:
        fig.tight_layout()
    _save_comparison_figure(fig, output_path)
    return fig, ax, (arrays_1, arrays_2)


def _default_evaluation_model_kind(config) -> str:
    if not config.use_mlp_features:
        return "aion"
    return "fusion" if config.use_aion_embedding else "tabular"


def run_config_pair(
    config_1,
    config_2,
    *,
    model_kind_1: str | None = None,
    model_kind_2: str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Run two configs and return evaluated runs for comparison plotting."""
    config_1 = make_magnitude_config(config_1)
    config_2 = make_magnitude_config(config_2)
    run_1 = run_training_and_evaluation(
        config_1,
        model_kind=model_kind_1 or _default_evaluation_model_kind(config_1),
        split=split,
    )
    run_2 = run_training_and_evaluation(
        config_2,
        model_kind=model_kind_2 or _default_evaluation_model_kind(config_2),
        split=split,
    )
    return {
        "config_1": config_1,
        "config_2": config_2,
        "run_1": run_1,
        "run_2": run_2,
        "evaluation_1": run_1["evaluation"],
        "evaluation_2": run_2["evaluation"],
        "model_kind_1": run_1["model_kind"],
        "model_kind_2": run_2["model_kind"],
        "split": split,
    }


def plot_redshift_pdf_comparison(
    evaluation: Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    bins: int | Sequence[float] = 80,
    z_range: tuple[float, float] | None = None,
    include_keys: Sequence[str] = ("z_p50", "z_mean", "z_mode"),
    old_label: str = "catalogue z_phot",
    labels: Mapping[str, str] | None = None,
    title: str = "Redshift PDF comparison",
):
    """Compare the old catalogue z_phot distribution with inferred z summaries."""
    labels = dict(labels or {})
    z_old = tensor_to_numpy_1d(evaluation["z_spec"])
    if z_range is None:
        redshift_edges = evaluation.get("redshift_edges")
        if redshift_edges is None:
            z_range = (0.0, 6.0)
        else:
            edges_np = redshift_edges.detach().cpu().numpy()
            z_range = (float(edges_np[0]), float(edges_np[-1]))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    hist_data = {}
    density, edges = np.histogram(z_old, bins=bins, range=z_range, density=True)
    centers_np = 0.5 * (edges[:-1] + edges[1:])
    ax.step(centers_np, density, where="mid", linewidth=1.8, label=old_label)
    hist_data["z_phot"] = {"centers": centers_np, "density": density, "edges": edges}

    for key in include_keys:
        if key not in evaluation:
            continue
        values = tensor_to_numpy_1d(evaluation[key])
        density, edges = np.histogram(values, bins=bins, range=z_range, density=True)
        centers_np = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centers_np, density, where="mid", linewidth=1.5, label=labels.get(key, key))
        hist_data[key] = {"centers": centers_np, "density": density, "edges": edges}

    ax.set_xlabel("redshift")
    ax.set_ylabel("probability density")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    return fig, ax, hist_data


def plot_sampled_z_inferred_distribution(
    evaluation: Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    n_samples_per_object: int = 1,
    bins: int | Sequence[float] = 80,
    z_range: tuple[float, float] | None = None,
    seed: int | None = 42,
    title: str = "Sampled inferred redshift distribution",
):
    z_sampled = sample_z_inferred_distribution(
        evaluation,
        n_samples_per_object=n_samples_per_object,
        seed=seed,
        flatten=True,
    )
    z_phot = tensor_to_numpy_1d(evaluation["z_spec"])
    if z_range is None:
        redshift_edges = evaluation.get("redshift_edges")
        if redshift_edges is None:
            z_range = (0.0, 6.0)
        else:
            edges_np = redshift_edges.detach().cpu().numpy()
            z_range = (float(edges_np[0]), float(edges_np[-1]))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(z_phot, bins=bins, range=z_range, density=True, histtype="step", linewidth=1.8, label="catalogue z_phot")
    ax.hist(z_sampled, bins=bins, range=z_range, density=True, histtype="stepfilled", alpha=0.35, label="sampled z_inferred")
    ax.set_xlabel("redshift")
    ax.set_ylabel("probability density")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    return fig, ax, z_sampled
