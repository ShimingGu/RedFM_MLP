from __future__ import annotations

"""Experimental/code-heritage search helpers for the optional AION M adapter."""

import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import aion_magnitude as am


DEFAULT_ILC_EXTRA_SOURCE_BANDS = ("u", "u_star", "Y", "J", "Ks")
DEFAULT_ILC_SOURCE_BANDS = tuple(am.HSC_AION_BANDS) + DEFAULT_ILC_EXTRA_SOURCE_BANDS
DEFAULT_ILC_TARGET_BANDS = tuple(am.HSC_AION_BANDS)

ILC_MODE_ACTIVE_EXTRA_BANDS = {
    "none": (),
    "no_effect": (),
    "grizy_only": (),
    "u_u_star_Y": ("u", "u_star", "Y"),
    "uuy": ("u", "u_star", "Y"),
    "all": DEFAULT_ILC_EXTRA_SOURCE_BANDS,
    "all_extra": DEFAULT_ILC_EXTRA_SOURCE_BANDS,
}


def resolve_ilc_mode(mode: str) -> str:
    key = str(mode).replace("*", "_star").replace(",", "_").replace(" ", "_")
    if key not in ILC_MODE_ACTIVE_EXTRA_BANDS:
        raise ValueError(
            f"Unknown ILC mode {mode!r}. Expected one of: "
            f"{sorted(ILC_MODE_ACTIVE_EXTRA_BANDS)}."
        )
    if key in {"no_effect", "none"}:
        return "grizy_only"
    if key in {"uuy"}:
        return "u_u_star_Y"
    if key in {"all_extra"}:
        return "all"
    return key


def active_source_bands_for_mode(mode: str) -> tuple[str, ...]:
    mode = resolve_ilc_mode(mode)
    active_extra = tuple(ILC_MODE_ACTIVE_EXTRA_BANDS[mode])
    if mode == "all":
        active_extra = DEFAULT_ILC_EXTRA_SOURCE_BANDS
    return tuple(am.HSC_AION_BANDS) + active_extra


def resolve_ilc_source_bands(names: Sequence[str] | None = None) -> tuple[str, ...]:
    names = DEFAULT_ILC_SOURCE_BANDS if names is None else names
    resolved = []
    for name in names:
        raw = str(name).strip()
        lower = raw.lower()
        if raw != "Y" and lower in am.HSC_AION_BANDS:
            band = lower
        else:
            band = am.resolve_extra_band_names([raw])[0]
        if band not in resolved:
            resolved.append(band)

    # The source matrix built from a CLAUDS table is grizy first, then selected extra bands.
    ordered = [
        band for band in am.HSC_AION_BANDS
        if band in resolved
    ]
    ordered.extend(
        band for band in resolved
        if band not in am.HSC_AION_BANDS
    )
    return tuple(ordered)


def extra_source_bands(source_bands: Sequence[str]) -> tuple[str, ...]:
    return tuple(band for band in source_bands if band not in am.HSC_AION_BANDS)


def includes_grizy_source(source_bands: Sequence[str]) -> bool:
    return any(band in am.HSC_AION_BANDS for band in source_bands)


def make_active_mask(
    *,
    target_bands: Sequence[str],
    source_bands: Sequence[str],
    active_source_bands: Sequence[str],
) -> torch.Tensor:
    active = set(active_source_bands)
    mask = torch.zeros((len(target_bands), len(source_bands)), dtype=torch.float32)
    for column, band in enumerate(source_bands):
        if band in active:
            mask[:, column] = 1.0
    return mask


@dataclass
class AIONILCConfig:
    """Configuration for training a linear correction before frozen AION."""

    catalogue_path: str | Path = Path("data/clauds/COSMOS-HSCpipe-Phosphoros.fits")
    max_rows: int | None = None
    cache_root: str | Path = Path("cache")
    split_output_dir: str | Path | None = None
    output_dir: str | Path = Path("cache/aion_ilc")
    solution_path: str | Path | None = None

    mode: str = "u_u_star_Y"
    source_bands: Sequence[str] = field(default_factory=lambda: list(DEFAULT_ILC_SOURCE_BANDS))
    target_bands: Sequence[str] = field(default_factory=lambda: list(DEFAULT_ILC_TARGET_BANDS))
    source_invalid_fill: str | float = "median"
    delta_clip_mag: float | None = None

    aion_head_checkpoint_path: str | Path | None = None

    split_chunk_size: int = 250_000
    overwrite_split_cache: bool = False
    field_column: str | None = None
    split_strategy: str = "random"
    train_fraction: float = 0.20
    test_fraction: float = 0.75
    val_fraction: float = 0.05
    test_fields: Sequence[str] = field(default_factory=list)
    target_redshift_column: str = am.REDSHIFT_COLUMNS["zphot"]

    z_min: float = 0.0
    z_max: float = 6.0
    n_z_bins: int = 300
    mag_zero_point: float = 23.0
    hsc_mag_faint_limits: Mapping[str, float | None] = field(
        default_factory=am.default_hsc_mag_faint_limits
    )

    epochs: int = 10
    train_batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 1e-2
    optimizer_kind: str = "spsa"
    finite_diff_step: float = 0.03
    spsa_beta1: float = 0.9
    spsa_beta2: float = 0.99
    spsa_eps: float = 1e-8

    prior_kind: str = "gaussian"
    prior_weight: float = 1.0
    gaussian_sigma: float = 0.3
    uniform_bound: float = 1.0

    max_train_batches: int | None = None
    max_val_batches: int | None = None
    seed: int = 42
    device_choice: str = "auto"

    def normalized(self) -> "AIONILCConfig":
        source_bands = resolve_ilc_source_bands(self.source_bands)
        target_bands = tuple(str(band) for band in self.target_bands)
        mode = resolve_ilc_mode(self.mode)
        active = active_source_bands_for_mode(mode)
        missing_active = [band for band in active if band not in source_bands]
        if missing_active:
            raise ValueError(
                f"Mode {mode!r} requires source bands absent from source_bands: {missing_active}."
            )
        prior_kind = str(self.prior_kind).lower()
        if prior_kind not in {"gaussian", "uniform", "none"}:
            raise ValueError("prior_kind must be 'gaussian', 'uniform', or 'none'.")
        optimizer_kind = str(self.optimizer_kind).lower()
        if optimizer_kind not in {"spsa", "autograd"}:
            raise ValueError("optimizer_kind must be 'spsa' or 'autograd'.")
        if float(self.finite_diff_step) <= 0.0:
            raise ValueError("finite_diff_step must be positive.")
        if not (0.0 <= float(self.spsa_beta1) < 1.0):
            raise ValueError("spsa_beta1 must be in [0, 1).")
        if not (0.0 <= float(self.spsa_beta2) < 1.0):
            raise ValueError("spsa_beta2 must be in [0, 1).")
        am.validate_split_fractions(self.train_fraction, self.test_fraction, self.val_fraction)
        if self.split_strategy not in {"random", "field"}:
            raise ValueError("split_strategy must be 'random' or 'field'.")
        return replace(
            self,
            catalogue_path=Path(self.catalogue_path),
            cache_root=Path(self.cache_root),
            split_output_dir=None if self.split_output_dir is None else Path(self.split_output_dir),
            output_dir=Path(self.output_dir),
            solution_path=None if self.solution_path is None else Path(self.solution_path),
            mode=mode,
            source_bands=source_bands,
            target_bands=target_bands,
            aion_head_checkpoint_path=None
            if self.aion_head_checkpoint_path is None
            else Path(self.aion_head_checkpoint_path),
            test_fields=list(self.test_fields),
            hsc_mag_faint_limits=dict(self.hsc_mag_faint_limits),
            optimizer_kind=optimizer_kind,
            prior_kind=prior_kind,
        )


class AIONLinearMagnitudeAdapter(nn.Module):
    """Learned linear delta-magnitude adapter for the frozen AION grizy input."""

    def __init__(
        self,
        *,
        source_mean: torch.Tensor,
        source_std: torch.Tensor,
        target_bands: Sequence[str] = DEFAULT_ILC_TARGET_BANDS,
        source_bands: Sequence[str] = DEFAULT_ILC_SOURCE_BANDS,
        active_source_bands: Sequence[str] = (),
        delta_clip_mag: float | None = None,
        init_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        self.target_bands = tuple(target_bands)
        self.source_bands = tuple(source_bands)
        self.delta_clip_mag = delta_clip_mag
        matrix_shape = (len(self.target_bands), len(self.source_bands))
        if init_matrix is None:
            init_matrix = torch.zeros(matrix_shape, dtype=torch.float32)
        init_matrix = torch.as_tensor(init_matrix, dtype=torch.float32)
        if tuple(init_matrix.shape) != matrix_shape:
            raise ValueError(f"init_matrix must have shape {matrix_shape}, got {tuple(init_matrix.shape)}.")
        self.matrix = nn.Parameter(init_matrix.clone())
        self.register_buffer("source_mean", torch.as_tensor(source_mean, dtype=torch.float32).reshape(1, -1))
        self.register_buffer("source_std", torch.as_tensor(source_std, dtype=torch.float32).reshape(1, -1).clamp_min(1e-6))
        self.register_buffer(
            "active_mask",
            make_active_mask(
                target_bands=self.target_bands,
                source_bands=self.source_bands,
                active_source_bands=active_source_bands,
            ),
        )

    def effective_matrix(self) -> torch.Tensor:
        return self.matrix * self.active_mask

    def adjusted_hsc_batch(
        self,
        hsc_batch: Mapping[str, torch.Tensor],
        source_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        source_features = source_features.float()
        standardized = (source_features - self.source_mean) / self.source_std
        delta = standardized @ self.effective_matrix().T
        if self.delta_clip_mag is not None:
            delta = delta.clamp(min=-float(self.delta_clip_mag), max=float(self.delta_clip_mag))
        adjusted = {key: value.float() for key, value in hsc_batch.items()}
        for column, band in enumerate(self.target_bands):
            key = f"{band}_mag"
            adjusted[key] = adjusted[key] + delta[:, column].to(adjusted[key].device)
        return adjusted

    def prior_loss(self, *, kind: str, gaussian_sigma: float = 0.3) -> torch.Tensor:
        if kind == "none" or self.active_mask.sum() == 0:
            return self.matrix.new_tensor(0.0)
        if kind == "gaussian":
            sigma = max(float(gaussian_sigma), 1e-6)
            active_values = self.effective_matrix()[self.active_mask.bool()]
            return 0.5 * torch.mean((active_values / sigma) ** 2)
        if kind == "uniform":
            return self.matrix.new_tensor(0.0)
        raise ValueError(f"Unknown prior kind: {kind}")

    def clamp_uniform_(self, bound: float | None) -> None:
        with torch.no_grad():
            self.matrix.mul_(self.active_mask)
            if bound is not None:
                self.matrix.clamp_(min=-float(bound), max=float(bound))
            self.matrix.mul_(self.active_mask)


def infer_aion_dim_from_checkpoint(checkpoint_path: str | Path) -> int:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    for key, value in checkpoint["state_dict"].items():
        if key.endswith("photoz_head.net.0.weight"):
            return int(value.shape[1])
    raise RuntimeError(f"Could not infer AION input dimension from {checkpoint_path}.")


def default_aion_head_checkpoint_path(config: AIONILCConfig) -> Path:
    baseline_config = am.AIONMagnitudeConfig(
        catalogue_path=config.catalogue_path,
        max_rows=config.max_rows,
        cache_root=config.cache_root,
        split_output_dir=config.split_output_dir,
        z_min=config.z_min,
        z_max=config.z_max,
        n_z_bins=config.n_z_bins,
        mag_zero_point=config.mag_zero_point,
        hsc_mag_faint_limits=config.hsc_mag_faint_limits,
        split_strategy=config.split_strategy,
        train_fraction=config.train_fraction,
        test_fraction=config.test_fraction,
        val_fraction=config.val_fraction,
        test_fields=config.test_fields,
        target_redshift_column=config.target_redshift_column,
        extra_bands=(),
        use_aion_embedding=True,
        use_mlp_features=False,
        include_grizy_in_mlp=False,
        model_kinds=("aion",),
    )
    paths = am.resolve_training_paths(baseline_config)
    return Path(paths["baseline_output_dir"]) / "aion_baseline.pt"


def default_solution_path(config: AIONILCConfig) -> Path:
    run_tag = am.make_cache_run_tag(config.catalogue_path, config.max_rows, config.mag_zero_point)
    return Path(config.output_dir) / f"aion_ilc_{run_tag}_{config.mode}.pt"


def split_indices(dataset, config: AIONILCConfig) -> dict[str, np.ndarray]:
    labels = am.make_split_labels(
        dataset.fields,
        split_strategy=config.split_strategy,
        train_fraction=config.train_fraction,
        test_fraction=config.test_fraction,
        val_fraction=config.val_fraction,
        test_fields=config.test_fields,
        seed=config.seed,
    )
    return {
        "train": np.flatnonzero(labels == "train"),
        "val": np.flatnonzero(labels == "val"),
        "test": np.flatnonzero(labels == "test"),
    }


def source_standardization(source_features: torch.Tensor, train_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    train_source = source_features[torch.as_tensor(train_indices, dtype=torch.long)].float()
    mean = train_source.mean(dim=0)
    std = train_source.std(dim=0, unbiased=False)
    std = torch.where(torch.isfinite(std) & (std > 1e-6), std, torch.ones_like(std))
    mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
    return mean, std


def make_loader(dataset, indices: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=am.collate_clauds_photoz,
    )


def _limited_batches(loader, limit: int | None):
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def _forward_ilc(
    adapter: AIONLinearMagnitudeAdapter,
    aion,
    codec_manager,
    photoz_head: nn.Module,
    batch,
    *,
    device: torch.device,
    track_input_grad: bool,
) -> torch.Tensor:
    hsc_batch = {key: value.to(device) for key, value in batch.hsc_batch.items()}
    source_features = batch.extra_features.to(device)
    adjusted_hsc = adapter.adjusted_hsc_batch(hsc_batch, source_features)
    embedding = am.extract_hsc_aion_embedding(
        adjusted_hsc,
        aion,
        codec_manager,
        device=device,
        track_input_grad=track_input_grad,
    )
    return photoz_head(embedding)


def train_one_epoch(
    adapter: AIONLinearMagnitudeAdapter,
    aion,
    codec_manager,
    photoz_head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    config: AIONILCConfig,
    redshift_edges: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    adapter.train(optimizer is not None)
    total_loss = 0.0
    total_ce = 0.0
    total_prior = 0.0
    n_batches = 0
    for batch in _limited_batches(loader, config.max_train_batches):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        z_spec = batch.z_spec.to(device)
        logits = _forward_ilc(
            adapter,
            aion,
            codec_manager,
            photoz_head,
            batch,
            device=device,
            track_input_grad=optimizer is not None,
        )
        ce_loss = am.redshift_cross_entropy_loss(logits, z_spec, edges=redshift_edges)
        prior = adapter.prior_loss(kind=config.prior_kind, gaussian_sigma=config.gaussian_sigma)
        loss = ce_loss + float(config.prior_weight) * prior
        if optimizer is not None:
            if not ce_loss.requires_grad:
                raise RuntimeError(
                    "ILC cross-entropy has no autograd path back to the magnitude-adjustment matrix M. "
                    "The frozen AION input codec/encoder appears non-differentiable with respect "
                    "to magnitude values, so gradient training of M cannot update the matrix. "
                    "A prior term can make the total loss require gradients, but it cannot optimize "
                    "M toward better photo-z performance by itself. "
                    "Use a derivative-free adapter search or train a differentiable surrogate instead."
                )
            loss.backward()
            if adapter.matrix.grad is None:
                raise RuntimeError(
                    "ILC matrix M did not receive gradients. "
                    "The AION magnitude-input path is not differentiable in this setup."
                )
            optimizer.step()
            if config.prior_kind == "uniform":
                adapter.clamp_uniform_(config.uniform_bound)
            else:
                adapter.clamp_uniform_(None)
        total_loss += float(loss.detach().cpu())
        total_ce += float(ce_loss.detach().cpu())
        total_prior += float(prior.detach().cpu())
        n_batches += 1
    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "cross_entropy": total_ce / denom,
        "prior": total_prior / denom,
        "n_batches": n_batches,
    }


@torch.no_grad()
def _ilc_loss_terms(
    adapter: AIONLinearMagnitudeAdapter,
    aion,
    codec_manager,
    photoz_head: nn.Module,
    batch,
    *,
    config: AIONILCConfig,
    redshift_edges: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float]:
    z_spec = batch.z_spec.to(device)
    logits = _forward_ilc(
        adapter,
        aion,
        codec_manager,
        photoz_head,
        batch,
        device=device,
        track_input_grad=False,
    )
    ce_loss = am.redshift_cross_entropy_loss(logits, z_spec, edges=redshift_edges)
    prior = adapter.prior_loss(kind=config.prior_kind, gaussian_sigma=config.gaussian_sigma)
    loss = ce_loss + float(config.prior_weight) * prior
    return (
        float(loss.detach().cpu()),
        float(ce_loss.detach().cpu()),
        float(prior.detach().cpu()),
    )


def train_one_epoch_spsa(
    adapter: AIONLinearMagnitudeAdapter,
    aion,
    codec_manager,
    photoz_head: nn.Module,
    loader: DataLoader,
    *,
    config: AIONILCConfig,
    redshift_edges: torch.Tensor,
    device: torch.device,
    state: dict[str, torch.Tensor | int] | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor | int]]:
    """Black-box SPSA update for M when AION tokenization blocks autograd."""
    adapter.train(False)
    if state is None:
        state = {
            "step": 0,
            "m": torch.zeros_like(adapter.matrix),
            "v": torch.zeros_like(adapter.matrix),
        }

    theta = adapter.matrix.detach().clone()
    active_mask = adapter.active_mask.to(device)
    total_loss = 0.0
    total_ce = 0.0
    total_prior = 0.0
    total_probe_gap = 0.0
    n_batches = 0

    for batch in _limited_batches(loader, config.max_train_batches):
        perturb = torch.randint(
            low=0,
            high=2,
            size=theta.shape,
            device=device,
            dtype=torch.int8,
        ).float()
        perturb = (perturb * 2.0 - 1.0) * active_mask
        if not torch.any(perturb):
            break

        finite_step = float(config.finite_diff_step)
        with torch.no_grad():
            adapter.matrix.copy_(theta + finite_step * perturb)
            if config.prior_kind == "uniform":
                adapter.clamp_uniform_(config.uniform_bound)
            else:
                adapter.clamp_uniform_(None)
        plus_loss, plus_ce, plus_prior = _ilc_loss_terms(
            adapter,
            aion,
            codec_manager,
            photoz_head,
            batch,
            config=config,
            redshift_edges=redshift_edges,
            device=device,
        )

        with torch.no_grad():
            adapter.matrix.copy_(theta - finite_step * perturb)
            if config.prior_kind == "uniform":
                adapter.clamp_uniform_(config.uniform_bound)
            else:
                adapter.clamp_uniform_(None)
        minus_loss, minus_ce, minus_prior = _ilc_loss_terms(
            adapter,
            aion,
            codec_manager,
            photoz_head,
            batch,
            config=config,
            redshift_edges=redshift_edges,
            device=device,
        )

        grad = ((plus_loss - minus_loss) / (2.0 * finite_step)) * perturb
        grad = grad * active_mask
        state["step"] = int(state["step"]) + 1
        step_index = int(state["step"])
        beta1 = float(config.spsa_beta1)
        beta2 = float(config.spsa_beta2)
        state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1.0 - beta2) * grad.square()
        m_hat = state["m"] / (1.0 - beta1 ** step_index)
        v_hat = state["v"] / (1.0 - beta2 ** step_index)
        theta = theta - float(config.learning_rate) * m_hat / (v_hat.sqrt() + float(config.spsa_eps))
        theta = theta * active_mask
        if config.prior_kind == "uniform":
            theta = theta.clamp(min=-float(config.uniform_bound), max=float(config.uniform_bound))

        with torch.no_grad():
            adapter.matrix.copy_(theta)

        total_loss += 0.5 * (plus_loss + minus_loss)
        total_ce += 0.5 * (plus_ce + minus_ce)
        total_prior += 0.5 * (plus_prior + minus_prior)
        total_probe_gap += abs(plus_loss - minus_loss)
        n_batches += 1

    with torch.no_grad():
        adapter.matrix.copy_(theta)
        if config.prior_kind == "uniform":
            adapter.clamp_uniform_(config.uniform_bound)
        else:
            adapter.clamp_uniform_(None)

    denom = max(n_batches, 1)
    return (
        {
            "loss": total_loss / denom,
            "cross_entropy": total_ce / denom,
            "prior": total_prior / denom,
            "probe_loss_gap": total_probe_gap / denom,
            "n_batches": n_batches,
        },
        state,
    )


@torch.no_grad()
def evaluate_ilc(
    adapter: AIONLinearMagnitudeAdapter,
    aion,
    codec_manager,
    photoz_head: nn.Module,
    loader: DataLoader,
    *,
    config: AIONILCConfig,
    redshift_edges: torch.Tensor,
    redshift_centers: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    adapter.eval()
    z_pred = []
    z_true = []
    total_ce = 0.0
    n_batches = 0
    for batch in _limited_batches(loader, config.max_val_batches):
        z_spec = batch.z_spec.to(device)
        logits = _forward_ilc(
            adapter,
            aion,
            codec_manager,
            photoz_head,
            batch,
            device=device,
            track_input_grad=False,
        )
        ce_loss = am.redshift_cross_entropy_loss(logits, z_spec, edges=redshift_edges)
        predictions = am.predict_photoz_from_logits(logits, centers=redshift_centers)
        z_pred.append(predictions["z_p50"].detach().cpu())
        z_true.append(z_spec.detach().cpu())
        total_ce += float(ce_loss.detach().cpu())
        n_batches += 1
    if not z_pred:
        return {
            "cross_entropy": float("nan"),
            "nmad": float("nan"),
            "catastrophic_outlier_fraction": float("nan"),
            "bias": float("nan"),
            "n_batches": 0,
        }
    metrics = am.point_photoz_metrics(torch.cat(z_pred), torch.cat(z_true))
    metrics["cross_entropy"] = total_ce / max(n_batches, 1)
    metrics["n_batches"] = n_batches
    return metrics


def train_aion_ilc_matrix(config: AIONILCConfig | None = None, **overrides: Any) -> dict[str, Any]:
    """Train only the linear M adapter against a frozen AION-only photo-z head."""
    config = AIONILCConfig() if config is None else config
    if overrides:
        config = replace(config, **overrides)
    config = config.normalized()
    am.set_random_seed(config.seed)
    device = am.select_torch_device(config.device_choice)
    split_output_dir = (
        Path(config.split_output_dir)
        if config.split_output_dir is not None
        else Path(config.cache_root) / f"clauds_split_{am.make_cache_run_tag(config.catalogue_path, config.max_rows, config.mag_zero_point)}"
    )
    dataset, source_feature_names, dataset_metadata = am.build_raw_clauds_photoz_dataset(
        config.catalogue_path,
        split_output_dir,
        split_chunk_size=config.split_chunk_size,
        overwrite_split_cache=config.overwrite_split_cache,
        max_rows=config.max_rows,
        field_column=config.field_column,
        target_redshift_column=config.target_redshift_column,
        z_min=config.z_min,
        z_max=config.z_max,
        n_z_bins=config.n_z_bins,
        mag_zero_point=config.mag_zero_point,
        hsc_mag_faint_limits=config.hsc_mag_faint_limits,
        extra_bands=extra_source_bands(config.source_bands),
        extra_band_invalid_fill=config.source_invalid_fill,
        extra_band_include_valid_flags=False,
        use_mlp_features=True,
        include_grizy_in_mlp=includes_grizy_source(config.source_bands),
        use_aion_embedding=True,
    )
    indices = split_indices(dataset, config)
    if len(indices["train"]) == 0 or len(indices["val"]) == 0:
        raise ValueError("ILC training requires non-empty train and validation splits.")

    source_mean, source_std = source_standardization(dataset.extra_features, indices["train"])
    active_source_bands = active_source_bands_for_mode(config.mode)
    adapter = AIONLinearMagnitudeAdapter(
        source_mean=source_mean,
        source_std=source_std,
        target_bands=config.target_bands,
        source_bands=config.source_bands,
        active_source_bands=active_source_bands,
        delta_clip_mag=config.delta_clip_mag,
    ).to(device)

    checkpoint_path = Path(config.aion_head_checkpoint_path or default_aion_head_checkpoint_path(config))
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"AION-only photo-z head checkpoint not found: {checkpoint_path}. "
            "Run the AION-only baseline first, then train the ILC matrix."
        )
    aion_dim = infer_aion_dim_from_checkpoint(checkpoint_path)
    photoz_head = am.load_baseline_model_from_checkpoint(
        checkpoint_path,
        model_kind="aion",
        aion_dim=aion_dim,
        extra_feature_dim=0,
        n_z_bins=config.n_z_bins,
        device=device,
    )
    photoz_head.eval()
    for parameter in photoz_head.parameters():
        parameter.requires_grad = False

    aion, codec_manager = am.load_frozen_aion(device=device)
    aion.eval()
    for parameter in aion.parameters():
        parameter.requires_grad = False

    train_loader = make_loader(
        dataset,
        indices["train"],
        batch_size=config.train_batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = make_loader(
        dataset,
        indices["val"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=config.seed,
    )
    redshift_edges, redshift_centers = am.make_redshift_grid(config.z_min, config.z_max, config.n_z_bins)
    redshift_edges = redshift_edges.to(device)
    redshift_centers = redshift_centers.to(device)

    has_active_parameters = bool(adapter.active_mask.sum().item() > 0)
    optimizer = None
    if has_active_parameters:
        if config.optimizer_kind == "autograd":
            optimizer = torch.optim.AdamW([adapter.matrix], lr=config.learning_rate, weight_decay=0.0)
        elif config.optimizer_kind != "spsa":
            raise ValueError(f"Unknown optimizer_kind: {config.optimizer_kind!r}.")
    else:
        warnings.warn(
            "ILC mode has no active source bands; saving the identity/no-effect matrix.",
            RuntimeWarning,
            stacklevel=2,
        )

    history = []
    best_matrix = adapter.effective_matrix().detach().cpu().clone()
    best_val_loss = float("inf")
    spsa_state = None
    for epoch in range(config.epochs):
        if has_active_parameters and config.optimizer_kind == "spsa":
            train_metrics, spsa_state = train_one_epoch_spsa(
                adapter,
                aion,
                codec_manager,
                photoz_head,
                train_loader,
                config=config,
                redshift_edges=redshift_edges,
                device=device,
                state=spsa_state,
            )
        else:
            train_metrics = train_one_epoch(
                adapter,
                aion,
                codec_manager,
                photoz_head,
                train_loader,
                optimizer,
                config=config,
                redshift_edges=redshift_edges,
                device=device,
            )
        val_metrics = evaluate_ilc(
            adapter,
            aion,
            codec_manager,
            photoz_head,
            val_loader,
            config=config,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
            device=device,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"ilc mode={config.mode} epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_cross_entropy']:.4f} "
            f"val_nmad={row['val_nmad']:.4f}"
        )
        if row["val_cross_entropy"] < best_val_loss:
            best_val_loss = row["val_cross_entropy"]
            best_matrix = adapter.effective_matrix().detach().cpu().clone()
        if not has_active_parameters:
            break

    with torch.no_grad():
        adapter.matrix.copy_(best_matrix.to(device))
    final_val_metrics = evaluate_ilc(
        adapter,
        aion,
        codec_manager,
        photoz_head,
        val_loader,
        config=config,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
        device=device,
    )
    solution = {
        "matrix": adapter.effective_matrix().detach().cpu(),
        "raw_matrix": adapter.matrix.detach().cpu(),
        "active_mask": adapter.active_mask.detach().cpu(),
        "source_mean": source_mean.detach().cpu(),
        "source_std": source_std.detach().cpu(),
        "source_bands": list(config.source_bands),
        "source_feature_names": list(source_feature_names),
        "source_invalid_fill": config.source_invalid_fill,
        "target_bands": list(config.target_bands),
        "active_source_bands": list(active_source_bands),
        "mode": config.mode,
        "prior_kind": config.prior_kind,
        "prior_weight": config.prior_weight,
        "gaussian_sigma": config.gaussian_sigma,
        "uniform_bound": config.uniform_bound,
        "delta_clip_mag": config.delta_clip_mag,
        "aion_head_checkpoint_path": str(checkpoint_path),
        "history": history,
        "final_val_metrics": final_val_metrics,
        "dataset_metadata": dataset_metadata,
        "config": asdict(config),
    }
    solution_path = Path(config.solution_path or default_solution_path(config))
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(solution, solution_path)
    return {
        "config": config,
        "solution": solution,
        "solution_path": solution_path,
        "history": history,
        "final_val_metrics": final_val_metrics,
        "dataset_metadata": dataset_metadata,
    }


def train_aion_ilc_modes(
    modes: Sequence[str] = ("none", "u_u_star_Y", "all"),
    config: AIONILCConfig | None = None,
    **overrides: Any,
) -> dict[str, dict[str, Any]]:
    """Train/save ILC matrices for multiple active-source modes."""
    results = {}
    base_config = AIONILCConfig() if config is None else config
    for mode in modes:
        mode_config = replace(base_config, mode=mode)
        results[resolve_ilc_mode(mode)] = train_aion_ilc_matrix(mode_config, **overrides)
    return results


def check_ilc_gradient_flow(config: AIONILCConfig | None = None, **overrides: Any) -> dict[str, Any]:
    """Run one tiny batch and report whether CE loss can update M through AION."""
    config = AIONILCConfig() if config is None else config
    if overrides:
        config = replace(config, **overrides)
    config = config.normalized()
    device = am.select_torch_device(config.device_choice)
    split_output_dir = (
        Path(config.split_output_dir)
        if config.split_output_dir is not None
        else Path(config.cache_root) / f"clauds_split_{am.make_cache_run_tag(config.catalogue_path, config.max_rows, config.mag_zero_point)}"
    )
    dataset, _, _ = am.build_raw_clauds_photoz_dataset(
        config.catalogue_path,
        split_output_dir,
        split_chunk_size=config.split_chunk_size,
        overwrite_split_cache=config.overwrite_split_cache,
        max_rows=config.max_rows,
        field_column=config.field_column,
        target_redshift_column=config.target_redshift_column,
        z_min=config.z_min,
        z_max=config.z_max,
        n_z_bins=config.n_z_bins,
        mag_zero_point=config.mag_zero_point,
        hsc_mag_faint_limits=config.hsc_mag_faint_limits,
        extra_bands=extra_source_bands(config.source_bands),
        extra_band_invalid_fill=config.source_invalid_fill,
        extra_band_include_valid_flags=False,
        use_mlp_features=True,
        include_grizy_in_mlp=includes_grizy_source(config.source_bands),
        use_aion_embedding=True,
    )
    indices = split_indices(dataset, config)
    source_mean, source_std = source_standardization(dataset.extra_features, indices["train"])
    adapter = AIONLinearMagnitudeAdapter(
        source_mean=source_mean,
        source_std=source_std,
        target_bands=config.target_bands,
        source_bands=config.source_bands,
        active_source_bands=active_source_bands_for_mode(config.mode),
        delta_clip_mag=config.delta_clip_mag,
    ).to(device)
    loader = make_loader(
        dataset,
        indices["train"],
        batch_size=min(config.train_batch_size, 8),
        shuffle=False,
        seed=config.seed,
    )
    batch = next(iter(loader))
    checkpoint_path = Path(config.aion_head_checkpoint_path or default_aion_head_checkpoint_path(config))
    aion_dim = infer_aion_dim_from_checkpoint(checkpoint_path)
    photoz_head = am.load_baseline_model_from_checkpoint(
        checkpoint_path,
        model_kind="aion",
        aion_dim=aion_dim,
        extra_feature_dim=0,
        n_z_bins=config.n_z_bins,
        device=device,
    )
    photoz_head.eval()
    for parameter in photoz_head.parameters():
        parameter.requires_grad = False
    aion, codec_manager = am.load_frozen_aion(device=device)
    redshift_edges, _ = am.make_redshift_grid(config.z_min, config.z_max, config.n_z_bins)
    logits = _forward_ilc(
        adapter,
        aion,
        codec_manager,
        photoz_head,
        batch,
        device=device,
        track_input_grad=True,
    )
    loss = am.redshift_cross_entropy_loss(
        logits,
        batch.z_spec.to(device),
        edges=redshift_edges.to(device),
    )
    matrix_grad_abs_max = None
    matrix_grad_abs_sum = None
    backward_error = None
    if loss.requires_grad:
        try:
            loss.backward()
            if adapter.matrix.grad is not None:
                matrix_grad_abs_max = float(adapter.matrix.grad.abs().max().detach().cpu())
                matrix_grad_abs_sum = float(adapter.matrix.grad.abs().sum().detach().cpu())
        except RuntimeError as exc:
            backward_error = str(exc)
    return {
        "mode": config.mode,
        "logits_requires_grad": bool(logits.requires_grad),
        "loss_requires_grad": bool(loss.requires_grad),
        "loss": float(loss.detach().cpu()),
        "matrix_grad_is_none": adapter.matrix.grad is None,
        "matrix_grad_abs_max": matrix_grad_abs_max,
        "matrix_grad_abs_sum": matrix_grad_abs_sum,
        "backward_error": backward_error,
    }


@torch.no_grad()
def check_ilc_finite_step_sensitivity(
    config: AIONILCConfig | None = None,
    *,
    step_values: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0),
    pattern: str = "identity",
    **overrides: Any,
) -> dict[str, Any]:
    """Measure black-box AION response to finite M perturbations."""
    config = AIONILCConfig() if config is None else config
    if overrides:
        config = replace(config, **overrides)
    config = config.normalized()
    device = am.select_torch_device(config.device_choice)
    split_output_dir = (
        Path(config.split_output_dir)
        if config.split_output_dir is not None
        else Path(config.cache_root) / f"clauds_split_{am.make_cache_run_tag(config.catalogue_path, config.max_rows, config.mag_zero_point)}"
    )
    dataset, _, _ = am.build_raw_clauds_photoz_dataset(
        config.catalogue_path,
        split_output_dir,
        split_chunk_size=config.split_chunk_size,
        overwrite_split_cache=config.overwrite_split_cache,
        max_rows=config.max_rows,
        field_column=config.field_column,
        target_redshift_column=config.target_redshift_column,
        z_min=config.z_min,
        z_max=config.z_max,
        n_z_bins=config.n_z_bins,
        mag_zero_point=config.mag_zero_point,
        hsc_mag_faint_limits=config.hsc_mag_faint_limits,
        extra_bands=extra_source_bands(config.source_bands),
        extra_band_invalid_fill=config.source_invalid_fill,
        extra_band_include_valid_flags=False,
        use_mlp_features=True,
        include_grizy_in_mlp=includes_grizy_source(config.source_bands),
        use_aion_embedding=True,
    )
    indices = split_indices(dataset, config)
    source_mean, source_std = source_standardization(dataset.extra_features, indices["train"])
    adapter = AIONLinearMagnitudeAdapter(
        source_mean=source_mean,
        source_std=source_std,
        target_bands=config.target_bands,
        source_bands=config.source_bands,
        active_source_bands=active_source_bands_for_mode(config.mode),
        delta_clip_mag=config.delta_clip_mag,
    ).to(device)
    loader = make_loader(
        dataset,
        indices["train"],
        batch_size=min(config.train_batch_size, 32),
        shuffle=False,
        seed=config.seed,
    )
    batch = next(iter(loader))
    checkpoint_path = Path(config.aion_head_checkpoint_path or default_aion_head_checkpoint_path(config))
    aion_dim = infer_aion_dim_from_checkpoint(checkpoint_path)
    photoz_head = am.load_baseline_model_from_checkpoint(
        checkpoint_path,
        model_kind="aion",
        aion_dim=aion_dim,
        extra_feature_dim=0,
        n_z_bins=config.n_z_bins,
        device=device,
    )
    photoz_head.eval()
    aion, codec_manager = am.load_frozen_aion(device=device)
    redshift_edges, _ = am.make_redshift_grid(config.z_min, config.z_max, config.n_z_bins)
    redshift_edges = redshift_edges.to(device)

    def fill_pattern(step: float) -> None:
        adapter.matrix.zero_()
        if pattern == "identity":
            source_to_col = {band: col for col, band in enumerate(adapter.source_bands)}
            for row, band in enumerate(adapter.target_bands):
                col = source_to_col.get(band)
                if col is not None and adapter.active_mask[row, col] > 0:
                    adapter.matrix[row, col] = float(step)
        elif pattern == "active_constant":
            adapter.matrix.copy_(adapter.active_mask * float(step))
        elif pattern == "active_random":
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed)
            random_matrix = torch.randn(
                adapter.matrix.shape,
                generator=generator,
                device=device,
                dtype=adapter.matrix.dtype,
            )
            adapter.matrix.copy_(random_matrix * adapter.active_mask * float(step))
        else:
            raise ValueError("pattern must be 'identity', 'active_constant', or 'active_random'.")

    adapter.matrix.zero_()
    baseline_logits = _forward_ilc(
        adapter,
        aion,
        codec_manager,
        photoz_head,
        batch,
        device=device,
        track_input_grad=False,
    )
    baseline_ce = am.redshift_cross_entropy_loss(
        baseline_logits,
        batch.z_spec.to(device),
        edges=redshift_edges,
    )

    rows = []
    for step in step_values:
        fill_pattern(float(step))
        logits = _forward_ilc(
            adapter,
            aion,
            codec_manager,
            photoz_head,
            batch,
            device=device,
            track_input_grad=False,
        )
        ce_loss = am.redshift_cross_entropy_loss(
            logits,
            batch.z_spec.to(device),
            edges=redshift_edges,
        )
        delta = logits - baseline_logits
        rows.append(
            {
                "step": float(step),
                "cross_entropy": float(ce_loss.detach().cpu()),
                "delta_cross_entropy": float((ce_loss - baseline_ce).detach().cpu()),
                "mean_abs_logit_delta": float(delta.abs().mean().detach().cpu()),
                "max_abs_logit_delta": float(delta.abs().max().detach().cpu()),
            }
        )

    return {
        "mode": config.mode,
        "pattern": pattern,
        "baseline_cross_entropy": float(baseline_ce.detach().cpu()),
        "steps": rows,
    }


def print_ilc_matrix(solution_or_result: Mapping[str, Any]) -> None:
    """Print a compact target x source view of a learned ILC matrix."""
    solution = solution_or_result.get("solution", solution_or_result)
    matrix = torch.as_tensor(solution["matrix"], dtype=torch.float32)
    target_bands = list(solution.get("target_bands", DEFAULT_ILC_TARGET_BANDS))
    source_bands = list(solution.get("source_bands", DEFAULT_ILC_SOURCE_BANDS))
    header = "target/source " + " ".join(f"{band:>10s}" for band in source_bands)
    print(header)
    for row, band in enumerate(target_bands):
        values = " ".join(f"{float(matrix[row, col]):10.4f}" for col in range(matrix.shape[1]))
        print(f"{band:>13s} {values}")
