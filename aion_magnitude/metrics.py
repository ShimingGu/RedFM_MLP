from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import torch.nn.functional as F

from .clauds_bands import REDSHIFT_COLUMNS, OBJECT_ID_COLUMN
from .utils import (
    gaussian_kernel_1d, gaussian_smooth_1d, make_redshift_grid,
    table_column_names, numeric_table_column, tensor_to_numpy_1d,
)


def redshift_cross_entropy_loss(
    logits: torch.Tensor,
    z_spec: torch.Tensor,
    edges: torch.Tensor | None = None,
    *,
    z_min: float = 0.0,
    z_max: float = 6.0,
) -> torch.Tensor:
    if edges is None:
        edges, _ = make_redshift_grid(z_min, z_max, logits.shape[-1])
    edges = edges.to(z_spec.device)
    target_bin = torch.bucketize(z_spec, edges) - 1
    target_bin = target_bin.clamp(0, logits.shape[-1] - 1)
    return F.cross_entropy(logits, target_bin.long())


def predict_photoz_from_logits(
    logits: torch.Tensor,
    centers: torch.Tensor | None = None,
    *,
    z_min: float = 0.0,
    z_max: float = 6.0,
) -> dict[str, torch.Tensor]:
    pz = torch.softmax(logits, dim=-1)
    if centers is None:
        _, centers = make_redshift_grid(z_min, z_max, logits.shape[-1])
    centers = centers.to(logits.device)

    z_mean = torch.sum(pz * centers[None, :], dim=-1)
    z_mode = centers[torch.argmax(pz, dim=-1)]
    cdf = torch.cumsum(pz, dim=-1)

    def quantile(q: float) -> torch.Tensor:
        idx = torch.argmax((cdf >= q).to(torch.int64), dim=-1)
        return centers[idx]

    return {
        "pz": pz,
        "z_mean": z_mean,
        "z_mode": z_mode,
        "z_p16": quantile(0.16),
        "z_p50": quantile(0.50),
        "z_p84": quantile(0.84),
    }


def normalize_redshift_reference(
    redshift_reference: Mapping[str, torch.Tensor | np.ndarray] | None = None,
) -> dict[str, torch.Tensor]:
    if redshift_reference is None:
        return {}
    normalized = {}
    for key, values in redshift_reference.items():
        tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        normalized[str(key)] = tensor.float()
    return normalized


def build_redshift_reference_from_table(table, rows=None) -> dict[str, torch.Tensor]:
    names = table_column_names(table)
    redshift_reference = {}
    for key in ("zphot", "z_low68", "z_high68", "z_peak"):
        column_name = REDSHIFT_COLUMNS.get(key)
        if column_name is not None and column_name in names:
            values = numeric_table_column(table, column_name, rows=rows)
            redshift_reference[key] = torch.from_numpy(values.astype(np.float32))
    return redshift_reference


def point_photoz_metrics(
    z_pred: torch.Tensor,
    z_spec: torch.Tensor,
    outlier_threshold: float = 0.15,
) -> dict[str, float]:
    z_pred = z_pred.detach().cpu()
    z_spec = z_spec.detach().cpu()
    delta = (z_pred - z_spec) / (1.0 + z_spec)
    median_delta = torch.median(delta)

    return {
        "bias": delta.mean().item(),
        "median_bias": median_delta.item(),
        "nmad": (1.48 * torch.median(torch.abs(delta - median_delta))).item(),
        "catastrophic_outlier_fraction": (torch.abs(delta) > outlier_threshold).float().mean().item(),
    }


def binned_log_score(
    logits: torch.Tensor,
    z_spec: torch.Tensor,
    edges: torch.Tensor | None = None,
) -> torch.Tensor:
    if edges is None:
        edges, _ = make_redshift_grid(n_z_bins=logits.shape[-1])
    edges = edges.to(z_spec.device)
    target_bin = torch.bucketize(z_spec, edges) - 1
    target_bin = target_bin.clamp(0, logits.shape[-1] - 1)
    log_pz = torch.log_softmax(logits, dim=-1)
    return -log_pz.gather(1, target_bin[:, None].long()).squeeze(1)


def discrete_crps(
    pz: torch.Tensor,
    z_spec: torch.Tensor,
    centers: torch.Tensor | None = None,
) -> torch.Tensor:
    if centers is None:
        _, centers = make_redshift_grid(n_z_bins=pz.shape[-1])
    centers = centers.to(pz.device)
    z_spec = z_spec.to(pz.device)
    cdf = torch.cumsum(pz, dim=-1)
    observed_cdf = (centers[None, :] >= z_spec[:, None]).float()
    dz = torch.mean(torch.diff(centers)) if centers.numel() > 1 else torch.tensor(1.0, device=pz.device)
    return torch.sum((cdf - observed_cdf) ** 2, dim=-1) * dz


def pit_values(
    pz: torch.Tensor,
    z_spec: torch.Tensor,
    edges: torch.Tensor | None = None,
) -> torch.Tensor:
    if edges is None:
        edges, _ = make_redshift_grid(n_z_bins=pz.shape[-1])
    edges = edges.to(z_spec.device)
    target_bin = torch.bucketize(z_spec, edges) - 1
    target_bin = target_bin.clamp(0, pz.shape[-1] - 1)
    cdf = torch.cumsum(pz.to(z_spec.device), dim=-1)
    return cdf.gather(1, target_bin[:, None].long()).squeeze(1)


def photoz_quantiles(
    pz: torch.Tensor,
    quantiles: Sequence[float],
    centers: torch.Tensor | None = None,
) -> dict[float, torch.Tensor]:
    if centers is None:
        _, centers = make_redshift_grid(n_z_bins=pz.shape[-1])
    centers = centers.to(pz.device)
    cdf = torch.cumsum(pz, dim=-1)
    values = {}
    for q in quantiles:
        idx = torch.argmax((cdf >= q).to(torch.int64), dim=-1)
        values[float(q)] = centers[idx]
    return values


def credible_interval_coverage(
    pz: torch.Tensor,
    z_spec: torch.Tensor,
    levels: Sequence[float] = (0.50, 0.68, 0.90, 0.95),
    centers: torch.Tensor | None = None,
) -> dict[str, float]:
    coverages = {}
    quantile_requests = []
    bounds = []
    for level in levels:
        low_q = 0.5 * (1.0 - level)
        high_q = 1.0 - low_q
        quantile_requests.extend([low_q, high_q])
        bounds.append((level, low_q, high_q))

    quantile_values = photoz_quantiles(pz, quantile_requests, centers=centers)
    for level, low_q, high_q in bounds:
        low = quantile_values[low_q].cpu()
        high = quantile_values[high_q].cpu()
        z_cpu = z_spec.cpu()
        coverages[f"coverage_{int(round(level * 100)):02d}"] = ((z_cpu >= low) & (z_cpu <= high)).float().mean().item()
    return coverages


def summarize_pdf_metrics(evaluation: dict[str, torch.Tensor | float]) -> dict[str, float]:
    if "z_spec" not in evaluation:
        raise ValueError("Evaluation output does not include z_spec.")

    z_spec = evaluation["z_spec"]
    logits = evaluation["logits"]
    pz = evaluation["pz"]
    edges = evaluation.get("redshift_edges")
    centers = evaluation.get("redshift_centers")

    metrics = point_photoz_metrics(evaluation["z_p50"], z_spec)
    metrics.update({
        "cross_entropy": redshift_cross_entropy_loss(logits, z_spec, edges=edges).item(),
        "mean_log_score": binned_log_score(logits, z_spec, edges=edges).mean().item(),
        "mean_crps": discrete_crps(pz, z_spec, centers=centers).mean().item(),
        "p16_p84_coverage": ((z_spec >= evaluation["z_p16"]) & (z_spec <= evaluation["z_p84"])).float().mean().item(),
        "pit_mean": pit_values(pz, z_spec, edges=edges).mean().item(),
    })
    return metrics


def calibration_diagnostics(
    evaluation: dict[str, torch.Tensor | float],
    pit_bins: int = 20,
) -> dict[str, Any]:
    if "z_spec" not in evaluation:
        raise ValueError("Evaluation output does not include z_spec.")

    pz = evaluation["pz"]
    z_spec = evaluation["z_spec"]
    edges = evaluation.get("redshift_edges")
    centers = evaluation.get("redshift_centers")
    pit = pit_values(pz, z_spec, edges=edges).detach().cpu().numpy()
    pit_hist, pit_edges = np.histogram(pit, bins=pit_bins, range=(0.0, 1.0), density=False)
    interval_coverage = credible_interval_coverage(pz, z_spec, centers=centers)

    return {
        "pit_mean": float(np.mean(pit)),
        "pit_std": float(np.std(pit)),
        "pit_hist": pit_hist.tolist(),
        "pit_edges": pit_edges.tolist(),
        **interval_coverage,
    }


def hpd_mass_at_true_bin(
    pz: torch.Tensor,
    z_spec: torch.Tensor,
    edges: torch.Tensor | None = None,
) -> torch.Tensor:
    pz = pz.detach().cpu()
    z_spec = z_spec.detach().cpu()
    if edges is None:
        edges, _ = make_redshift_grid(n_z_bins=pz.shape[-1])
    edges = edges.cpu()
    target_bin = torch.bucketize(z_spec, edges) - 1
    target_bin = target_bin.clamp(0, pz.shape[-1] - 1)
    true_prob = pz.gather(1, target_bin[:, None]).squeeze(1)
    return torch.sum(torch.where(pz >= true_prob[:, None], pz, torch.zeros_like(pz)), dim=-1)


def conformal_hpd_threshold(
    calibration_scores: torch.Tensor,
    alpha: float,
) -> float:
    scores = torch.sort(calibration_scores.detach().cpu()).values
    n = scores.numel()
    if n == 0:
        raise ValueError("Need at least one calibration score.")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return scores[rank - 1].item()


def conformal_hpd_set_mask(pz: torch.Tensor, threshold: float) -> torch.Tensor:
    sorted_pz, sorted_idx = torch.sort(pz.detach().cpu(), dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_pz, dim=-1)
    sorted_keep = cumulative <= threshold
    sorted_keep[:, 0] = True
    first_over = torch.argmax((cumulative >= threshold).to(torch.int64), dim=-1)
    sorted_keep[torch.arange(sorted_keep.shape[0]), first_over] = True

    keep = torch.zeros_like(sorted_keep, dtype=torch.bool)
    keep.scatter_(1, sorted_idx, sorted_keep)
    return keep


def evaluate_conformal_hpd(
    calibration_evaluation: dict[str, torch.Tensor | float],
    test_evaluation: dict[str, torch.Tensor | float],
    alphas: Sequence[float] = (0.32, 0.10, 0.05),
) -> dict[str, dict[str, float]]:
    calibration_scores = hpd_mass_at_true_bin(
        calibration_evaluation["pz"],
        calibration_evaluation["z_spec"],
        edges=calibration_evaluation.get("redshift_edges"),
    )
    test_scores = hpd_mass_at_true_bin(
        test_evaluation["pz"],
        test_evaluation["z_spec"],
        edges=test_evaluation.get("redshift_edges"),
    )
    pz_test = test_evaluation["pz"].detach().cpu()
    centers = test_evaluation.get("redshift_centers")
    if centers is None:
        _, centers = make_redshift_grid(n_z_bins=pz_test.shape[-1])
    dz = torch.mean(torch.diff(centers.detach().cpu())).item()

    output = {}
    for alpha in alphas:
        threshold = conformal_hpd_threshold(calibration_scores, alpha)
        keep = conformal_hpd_set_mask(pz_test, threshold)
        coverage = (test_scores <= threshold).float().mean().item()
        output[f"coverage_{int(round((1.0 - alpha) * 100)):02d}"] = {
            "alpha": float(alpha),
            "threshold": float(threshold),
            "empirical_coverage": float(coverage),
            "mean_set_bins": keep.float().sum(dim=-1).mean().item(),
            "mean_set_width": keep.float().sum(dim=-1).mean().item() * dz,
        }

    return output


def redshift_probability_distribution(
    evaluation: Mapping[str, Any],
    *,
    gaussian_sigma_bins: float = 0.0,
    gaussian_truncate: float = 4.0,
) -> dict[str, np.ndarray | float]:
    """Aggregate true and recovered redshift distributions on the same grid."""
    pz = evaluation["pz"].detach().cpu().double().numpy()
    if pz.ndim != 2:
        raise ValueError("evaluation['pz'] must have shape [n_objects, n_z_bins].")

    centers = evaluation.get("redshift_centers")
    edges = evaluation.get("redshift_edges")
    if centers is None or edges is None:
        edges_t, centers_t = make_redshift_grid(n_z_bins=pz.shape[-1])
        centers = centers if centers is not None else centers_t
        edges = edges if edges is not None else edges_t

    centers_np = tensor_to_numpy_1d(centers)
    edges_np = tensor_to_numpy_1d(edges)
    if edges_np.size != centers_np.size + 1:
        raise ValueError("redshift_edges must have one more element than redshift_centers.")

    recovered_raw_probability = pz.mean(axis=0)
    recovered_raw_probability = recovered_raw_probability / max(
        recovered_raw_probability.sum(),
        np.finfo(np.float64).tiny,
    )
    recovered_probability = gaussian_smooth_1d(
        recovered_raw_probability,
        gaussian_sigma_bins,
        truncate=gaussian_truncate,
    )
    recovered_probability = np.clip(recovered_probability, 0.0, None)
    recovered_probability = recovered_probability / max(
        recovered_probability.sum(),
        np.finfo(np.float64).tiny,
    )

    true_raw_probability = None
    true_probability = None
    if "z_spec" in evaluation:
        z_true = tensor_to_numpy_1d(evaluation["z_spec"])
        true_counts, _ = np.histogram(z_true, bins=edges_np)
        true_raw_probability = true_counts.astype(np.float64)
        true_raw_probability = true_raw_probability / max(
            true_raw_probability.sum(),
            np.finfo(np.float64).tiny,
        )
        true_probability = gaussian_smooth_1d(
            true_raw_probability,
            gaussian_sigma_bins,
            truncate=gaussian_truncate,
        )
        true_probability = np.clip(true_probability, 0.0, None)
        true_probability = true_probability / max(
            true_probability.sum(),
            np.finfo(np.float64).tiny,
        )

    bin_widths = np.diff(edges_np)
    output = {
        "centers": centers_np,
        "edges": edges_np,
        "bin_widths": bin_widths,
        "recovered_raw_probability": recovered_raw_probability,
        "recovered_probability": recovered_probability,
        "recovered_raw_density": recovered_raw_probability / bin_widths,
        "recovered_density": recovered_probability / bin_widths,
        # Backward-compatible aliases for the recovered distribution.
        "raw_probability": recovered_raw_probability,
        "probability": recovered_probability,
        "raw_density": recovered_raw_probability / bin_widths,
        "density": recovered_probability / bin_widths,
        "gaussian_sigma_bins": float(gaussian_sigma_bins),
    }
    if true_raw_probability is not None and true_probability is not None:
        output.update({
            "true_raw_probability": true_raw_probability,
            "true_probability": true_probability,
            "true_raw_density": true_raw_probability / bin_widths,
            "true_density": true_probability / bin_widths,
        })
    return output


def validate_zphot_bins(zphot_bin: Sequence[float]) -> np.ndarray:
    bins = np.asarray(zphot_bin, dtype=np.float64)
    if bins.ndim != 1 or bins.size == 0:
        raise ValueError("zphot_bin must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(bins)):
        raise ValueError("zphot_bin must contain only finite values.")
    if not np.all(bins > 0):
        raise ValueError("zphot_bin edges must be positive; z <= 0 objects are dropped.")
    if not np.all(np.diff(bins) > 0):
        raise ValueError("zphot_bin must be strictly increasing.")
    return bins


def tomographic_bin_labels(zphot_bin: Sequence[float]) -> list[str]:
    bins = validate_zphot_bins(zphot_bin)
    labels = [f"(0, {bins[0]:g}]"]
    labels.extend(
        f"({bins[i - 1]:g}, {bins[i]:g}]"
        for i in range(1, len(bins))
    )
    labels.append(f"({bins[-1]:g}, infinity]")
    return labels


def assign_tomographic_bins(z_values: Sequence[float] | np.ndarray, zphot_bin: Sequence[float]) -> np.ndarray:
    bins = validate_zphot_bins(zphot_bin)
    values = tensor_to_numpy_1d(z_values)
    assignment = np.full(values.shape, -1, dtype=np.int64)
    valid = np.isfinite(values) & (values > 0)
    assignment[valid] = np.searchsorted(bins, values[valid], side="left")
    return assignment


def catalogue_redshift_reference(evaluation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    reference: dict[str, np.ndarray] = {}
    nested = evaluation.get("redshift_reference")
    if isinstance(nested, Mapping):
        for key, values in nested.items():
            reference[str(key)] = tensor_to_numpy_1d(values)

    for key in ("zphot", "z_low68", "z_high68", "z_peak"):
        if key not in reference and key in evaluation:
            reference[key] = tensor_to_numpy_1d(evaluation[key])

    if "zphot" not in reference and "z_spec" in evaluation:
        reference["zphot"] = tensor_to_numpy_1d(evaluation["z_spec"])
    return reference


def sample_lognormal_from_percentiles(
    median: Sequence[float] | np.ndarray,
    low68: Sequence[float] | np.ndarray | None = None,
    high68: Sequence[float] | np.ndarray | None = None,
    mode: Sequence[float] | np.ndarray | None = None,
    *,
    n_samples_per_object: int = 10,
    seed: int | None = 42,
) -> np.ndarray:
    if n_samples_per_object < 1:
        raise ValueError("n_samples_per_object must be >= 1.")

    median_np = tensor_to_numpy_1d(median)
    n_objects = median_np.size
    samples = np.full((n_objects, n_samples_per_object), np.nan, dtype=np.float64)
    valid_median = np.isfinite(median_np) & (median_np > 0)
    if not valid_median.any():
        return samples

    sigma = np.zeros(n_objects, dtype=np.float64)
    has_sigma = np.zeros(n_objects, dtype=bool)
    if low68 is not None and high68 is not None:
        low_np = tensor_to_numpy_1d(low68)
        high_np = tensor_to_numpy_1d(high68)
        if low_np.size != n_objects or high_np.size != n_objects:
            raise ValueError("low68/high68 must have the same length as median.")
        valid_bounds = (
            valid_median
            & np.isfinite(low_np)
            & np.isfinite(high_np)
            & (low_np > 0)
            & (high_np > low_np)
        )
        sigma[valid_bounds] = 0.5 * (np.log(high_np[valid_bounds]) - np.log(low_np[valid_bounds]))
        has_sigma = valid_bounds & np.isfinite(sigma) & (sigma > 0)

    if mode is not None:
        mode_np = tensor_to_numpy_1d(mode)
        if mode_np.size != n_objects:
            raise ValueError("mode must have the same length as median.")
        valid_mode = (
            valid_median
            & ~has_sigma
            & np.isfinite(mode_np)
            & (mode_np > 0)
            & (mode_np < median_np)
        )
        sigma[valid_mode] = np.sqrt(np.maximum(np.log(median_np[valid_mode] / mode_np[valid_mode]), 0.0))
        has_sigma = has_sigma | (valid_mode & np.isfinite(sigma) & (sigma > 0))

    rng = np.random.default_rng(seed)
    stochastic = valid_median & has_sigma
    if stochastic.any():
        draws = rng.normal(
            loc=np.log(median_np[stochastic])[:, None],
            scale=sigma[stochastic][:, None],
            size=(int(stochastic.sum()), n_samples_per_object),
        )
        samples[stochastic] = np.exp(draws)

    point_like = valid_median & ~has_sigma
    if point_like.any():
        samples[point_like] = median_np[point_like, None]

    return samples


def sample_catalogue_redshift_per_object(
    evaluation: Mapping[str, Any],
    *,
    n_samples_per_object: int = 10,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    reference = catalogue_redshift_reference(evaluation)
    if "zphot" not in reference:
        raise ValueError("evaluation must include redshift_reference['zphot'] or z_spec.")
    samples = sample_lognormal_from_percentiles(
        reference["zphot"],
        low68=reference.get("z_low68"),
        high68=reference.get("z_high68"),
        mode=reference.get("z_peak"),
        n_samples_per_object=n_samples_per_object,
        seed=seed,
    )
    return reference["zphot"], samples


def sample_inferred_redshift_per_object(
    evaluation: Mapping[str, Any],
    *,
    n_samples_per_object: int = 10,
    seed: int | None = 43,
    use_pz_if_available: bool = True,
) -> np.ndarray:
    if use_pz_if_available and "pz" in evaluation:
        return sample_z_inferred_distribution(
            evaluation,
            n_samples_per_object=n_samples_per_object,
            seed=seed,
            flatten=False,
        )
    if "z_p50" not in evaluation:
        raise ValueError("evaluation must include pz or z_p50 for inferred sampling.")
    return sample_lognormal_from_percentiles(
        tensor_to_numpy_1d(evaluation["z_p50"]),
        low68=tensor_to_numpy_1d(evaluation["z_p16"]) if "z_p16" in evaluation else None,
        high68=tensor_to_numpy_1d(evaluation["z_p84"]) if "z_p84" in evaluation else None,
        mode=tensor_to_numpy_1d(evaluation["z_mode"]) if "z_mode" in evaluation else None,
        n_samples_per_object=n_samples_per_object,
        seed=seed,
    )


def resolve_redshift_hist_edges(
    evaluation: Mapping[str, Any],
    *,
    hist_bins: int | Sequence[float] | None = None,
    z_range: tuple[float, float] | None = None,
    zphot_bin: Sequence[float] = (),
) -> np.ndarray:
    if hist_bins is None:
        redshift_edges = evaluation.get("redshift_edges")
        if redshift_edges is not None:
            return tensor_to_numpy_1d(redshift_edges)
        upper = max(6.0, float(validate_zphot_bins(zphot_bin)[-1]))
        return np.linspace(0.0, upper, 121)

    if np.isscalar(hist_bins):
        n_bins = int(hist_bins)
        if n_bins < 1:
            raise ValueError("hist_bins must be >= 1.")
        if z_range is None:
            redshift_edges = evaluation.get("redshift_edges")
            if redshift_edges is not None:
                edge_values = tensor_to_numpy_1d(redshift_edges)
                z_range = (float(edge_values[0]), float(edge_values[-1]))
            else:
                z_range = (0.0, max(6.0, float(validate_zphot_bins(zphot_bin)[-1])))
        return np.linspace(float(z_range[0]), float(z_range[1]), n_bins + 1)

    edges = np.asarray(hist_bins, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("hist_bins edges must be a strictly increasing 1D sequence.")
    return edges


def probability_density_from_samples(
    samples: np.ndarray,
    edges: np.ndarray,
    *,
    gaussian_sigma_bins: float = 0.0,
    gaussian_truncate: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    flat_samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    valid = np.isfinite(flat_samples) & (flat_samples > 0)
    flat_samples = flat_samples[valid]
    counts, _ = np.histogram(flat_samples, bins=edges)
    n_used = int(counts.sum())
    probability = counts.astype(np.float64)
    if n_used > 0:
        probability /= probability.sum()
        probability = gaussian_smooth_1d(
            probability,
            gaussian_sigma_bins,
            truncate=gaussian_truncate,
        )
        probability = np.clip(probability, 0.0, None)
        probability /= max(probability.sum(), np.finfo(np.float64).tiny)
    bin_widths = np.diff(edges)
    return probability / bin_widths, probability, n_used


def sample_z_inferred_distribution(
    evaluation: Mapping[str, Any],
    *,
    n_samples_per_object: int = 1,
    centers: torch.Tensor | None = None,
    seed: int | None = 42,
    flatten: bool = True,
) -> np.ndarray:
    """Sample an inferred redshift distribution from each object's full p(z)."""
    pz = evaluation["pz"].detach().cpu().float()
    if centers is None:
        centers = evaluation.get("redshift_centers")
    if centers is None:
        _, centers = make_redshift_grid(n_z_bins=pz.shape[-1])
    centers = centers.detach().cpu().float()
    if pz.ndim != 2:
        raise ValueError("evaluation['pz'] must have shape [n_objects, n_z_bins].")
    if n_samples_per_object < 1:
        raise ValueError("n_samples_per_object must be >= 1.")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

    sample_idx = torch.multinomial(
        pz,
        num_samples=n_samples_per_object,
        replacement=True,
        generator=generator,
    )
    samples = centers[sample_idx].numpy()
    if flatten:
        return samples.reshape(-1)
    return samples


def sample_population_z_distribution(
    evaluation: Mapping[str, Any],
    *,
    n_samples: int = 100_000,
    centers: torch.Tensor | None = None,
    seed: int | None = 42,
) -> np.ndarray:
    """Sample from the population-level inferred p(z), averaged over catalogue objects."""
    pz = evaluation["pz"].detach().cpu().float()
    population_pz = pz.mean(dim=0)
    if centers is None:
        centers = evaluation.get("redshift_centers")
    if centers is None:
        _, centers = make_redshift_grid(n_z_bins=pz.shape[-1])
    centers = centers.detach().cpu().float()
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    sample_idx = torch.multinomial(population_pz, num_samples=n_samples, replacement=True, generator=generator)
    return centers[sample_idx].numpy()
