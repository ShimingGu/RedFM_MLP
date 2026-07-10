from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from .clauds_bands import HSC_AION_BANDS, REDSHIFT_COLUMNS, default_hsc_mag_faint_limits
from .utils import validate_split_fractions, _path_tag


class AIONMagnitudeConfig:
    """Defaulted configuration for the notebook training workflow.

    Override values here or pass keyword overrides to run_baseline_training() /
    run_training_and_evaluation() instead of editing global notebook cells.
    """

    catalogue_path: str | Path = Path("data/clauds/DEEP23-HSCpipe-Phosphoros.fits")
    max_rows: int | None = None
    cache_root: str | Path = Path("cache")
    split_output_dir: str | Path | None = None
    cache_path: str | Path | None = None
    baseline_output_dir: str | Path | None = None

    split_chunk_size: int = 250_000
    overwrite_split_cache: bool = False
    force_recompute_embeddings: bool = False
    field_column: str | None = None
    split_strategy: str = "random"
    train_fraction: float = 0.20
    test_fraction: float = 0.75
    val_fraction: float = 0.05
    test_fields: Sequence[str] = field(default_factory=list)
    target_redshift_column: str = REDSHIFT_COLUMNS["zphot"]

    z_min: float = 0.0
    z_max: float = 6.0
    n_z_bins: int = 300

    mag_zero_point: float = 23.0
    hsc_mag_faint_limits: Mapping[str, float | None] = field(
        default_factory=default_hsc_mag_faint_limits
    )
    extra_bands: Sequence[str] = field(default_factory=lambda: list(DEFAULT_EXTRA_BANDS))
    extra_band_invalid_fill: str | float = "median"
    extra_band_include_valid_flags: bool = False
    use_aion_embedding: bool = True
    use_mlp_features: bool = True
    include_grizy_in_mlp: bool | None = None
    aion_input_bands: Sequence[str] = field(default_factory=lambda: list(HSC_AION_BANDS))
    aion_mag_adjustment_path: str | Path | None = None
    aion_mag_adjustment_tag: str | None = None

    aion_embedding_batch_size: int = 512
    baseline_epochs: int = 20
    baseline_train_batch_size: int = 256
    baseline_eval_batch_size: int = 512
    baseline_learning_rate: float = 1e-3
    baseline_weight_decay: float = 1e-4
    model_kinds: Sequence[str] = ("tabular", "aion", "fusion")

    seed: int = 42
    device_choice: str = "auto"

    def normalized(self) -> "AIONMagnitudeConfig":
        config = replace(
            self,
            catalogue_path=Path(self.catalogue_path),
            cache_root=Path(self.cache_root),
            split_output_dir=None if self.split_output_dir is None else Path(self.split_output_dir),
            cache_path=None if self.cache_path is None else Path(self.cache_path),
            baseline_output_dir=None if self.baseline_output_dir is None else Path(self.baseline_output_dir),
            aion_mag_adjustment_path=None if self.aion_mag_adjustment_path is None else Path(self.aion_mag_adjustment_path),
            aion_mag_adjustment_tag=None if self.aion_mag_adjustment_tag is None else str(self.aion_mag_adjustment_tag),
            hsc_mag_faint_limits=dict(self.hsc_mag_faint_limits),
            test_fields=list(self.test_fields),
            extra_bands=resolve_extra_band_names(self.extra_bands),
            aion_input_bands=tuple(self.aion_input_bands),
            model_kinds=tuple(self.model_kinds),
        )
        include_grizy_in_mlp = config.include_grizy_in_mlp
        if include_grizy_in_mlp is not None:
            config = replace(config, include_grizy_in_mlp=bool(include_grizy_in_mlp))
        validate_split_fractions(config.train_fraction, config.test_fraction, config.val_fraction)
        if config.split_strategy not in {"random", "field"}:
            raise ValueError("split_strategy must be 'random' or 'field'.")
        if config.use_aion_embedding and tuple(config.aion_input_bands) != tuple(HSC_AION_BANDS):
            warnings.warn(
                "Disabling individual HSC grizy bands is not currently supported because "
                "the frozen AION embedding expects the full grizy input; the requested "
                "AION band selection will be ignored for now.",
                RuntimeWarning,
                stacklevel=2,
            )
            config = replace(config, aion_input_bands=tuple(HSC_AION_BANDS))
        if not config.use_mlp_features:
            if not config.use_aion_embedding:
                raise ValueError("use_mlp_features=False requires use_aion_embedding=True for AION-only mode.")
            non_aion = [kind for kind in config.model_kinds if kind != "aion"]
            if non_aion:
                warnings.warn(
                    "use_mlp_features=False selects AION-only mode; "
                    "training will run model_kinds=('aion',).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            config = replace(
                config,
                model_kinds=("aion",),
                include_grizy_in_mlp=False,
            )
        elif (
            config.use_aion_embedding
            and not resolve_include_grizy_in_mlp(
                config.include_grizy_in_mlp,
                use_aion_embedding=config.use_aion_embedding,
                use_mlp_features=config.use_mlp_features,
                extra_bands=config.extra_bands,
            )
            and not config.extra_bands
        ):
            non_aion = [kind for kind in config.model_kinds if kind != "aion"]
            if non_aion:
                warnings.warn(
                    "No MLP features were selected: AION is enabled, extra_bands is empty, "
                    "and grizy is not duplicated into the MLP. Training will run "
                    "model_kinds=('aion',). Set include_grizy_in_mlp=True or select "
                    "extra_bands to train tabular/fusion models.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            config = replace(
                config,
                use_mlp_features=False,
                model_kinds=("aion",),
                include_grizy_in_mlp=False,
            )
        if not config.use_aion_embedding:
            if config.aion_mag_adjustment_path is not None:
                warnings.warn(
                    "aion_mag_adjustment_path was provided but use_aion_embedding=False; "
                    "the AION magnitude adjustment is ignored for tabular-only training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                config = replace(config, aion_mag_adjustment_path=None, aion_mag_adjustment_tag=None)
            non_tabular = [kind for kind in config.model_kinds if kind != "tabular"]
            if non_tabular:
                warnings.warn(
                    "use_aion_embedding=False disables AION and fusion model kinds; "
                    "training will run model_kinds=('tabular',) for MLP-only mode.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            config = replace(config, model_kinds=("tabular",), aion_input_bands=tuple())
        return config


def make_magnitude_config(
    config: AIONMagnitudeConfig | None = None,
    **overrides: Any,
) -> AIONMagnitudeConfig:
    """Return a normalized config with optional keyword overrides."""
    config = AIONMagnitudeConfig() if config is None else config
    if overrides:
        config = replace(config, **overrides)
    return config.normalized()


def resolve_training_paths(config: AIONMagnitudeConfig) -> dict[str, Path | str]:
    config = config.normalized()
    include_grizy_in_mlp = resolve_include_grizy_in_mlp(
        config.include_grizy_in_mlp,
        use_aion_embedding=config.use_aion_embedding,
        use_mlp_features=config.use_mlp_features,
        extra_bands=config.extra_bands,
    )
    run_tag = make_cache_run_tag(config.catalogue_path, config.max_rows, config.mag_zero_point)
    if (
        float(config.z_min) != 0.0
        or float(config.z_max) != 6.0
        or int(config.n_z_bins) != 300
    ):
        z_min_tag = f"{config.z_min:g}".replace(".", "p")
        z_max_tag = f"{config.z_max:g}".replace(".", "p")
        run_tag = f"{run_tag}_z{z_min_tag}_{z_max_tag}_bins{int(config.n_z_bins)}"
    if not config.use_mlp_features:
        extra_tag = "aiononly"
    else:
        extra_tag = "noextra" if not config.extra_bands else "extra_" + "_".join(_path_tag(band) for band in config.extra_bands)
        if include_grizy_in_mlp:
            extra_tag = f"grizy_{extra_tag}"
        if config.extra_band_include_valid_flags:
            extra_tag = f"{extra_tag}_validflag"
    ilc_tag = aion_mag_adjustment_tag(config.aion_mag_adjustment_path, config.aion_mag_adjustment_tag)
    if ilc_tag is not None:
        extra_tag = f"{extra_tag}_ilc_{ilc_tag}"
    experiment_tag = f"{run_tag}_{extra_tag}"
    if not config.use_aion_embedding:
        experiment_tag = f"{experiment_tag}_noaion"
    cache_root = Path(config.cache_root)
    split_output_dir = Path(config.split_output_dir) if config.split_output_dir is not None else cache_root / f"clauds_split_{run_tag}"
    cache_file_prefix = "clauds_aion_embeddings" if config.use_aion_embedding else "clauds_noaion_catalogue"
    cache_run_tag = run_tag if ilc_tag is None or not config.use_aion_embedding else f"{run_tag}_ilc_{ilc_tag}"
    cache_path = Path(config.cache_path) if config.cache_path is not None else cache_root / f"{cache_file_prefix}_{cache_run_tag}.pt"
    baseline_output_dir = (
        Path(config.baseline_output_dir)
        if config.baseline_output_dir is not None
        else cache_root / f"baselines_{experiment_tag}"
    )
    return {
        "run_tag": run_tag,
        "experiment_tag": experiment_tag,
        "split_output_dir": split_output_dir,
        "cache_path": cache_path,
        "baseline_output_dir": baseline_output_dir,
    }
