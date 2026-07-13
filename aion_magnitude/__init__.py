from __future__ import annotations

from importlib import import_module

# Export clauds_bands constants and functions
from .clauds_bands import (
    BAND_FLUX_COLUMNS,
    BAND_ERROR_COLUMNS,
    OPTIONAL_EXTRA_BAND_FLUX_COLUMNS,
    OPTIONAL_EXTRA_BAND_ERROR_COLUMNS,
    ALL_BAND_FLUX_COLUMNS,
    ALL_BAND_ERROR_COLUMNS,
    REDSHIFT_COLUMNS,
    FLAG_COLUMNS,
    OPTIONAL_EXTRA_FLAG_COLUMNS,
    ALL_FLAG_COLUMNS,
    metadata_dtype,
    bands_dtype,
    errors_dtype,
    redshifts_dtype,
    flags_dtype,
    split_clauds_catalogue,
    validate_clauds_fits_table,
    OBJECT_ID_COLUMN,
    RA_COLUMN,
    DEC_COLUMN,
    TRACT_COLUMN,
    PATCH_COLUMN,
    HSC_AION_BANDS,
    CLAUDS_EXTRA_FLUX_BANDS,
    EXTRA_ERROR_BANDS,
    default_hsc_mag_faint_limits,
    CATALOGUE_SAMPLE_MODES,
    normalize_catalogue_row_range,
    select_catalogue_row_indices,
    select_clauds_catalogue_row_indices,
)

# Export utils
from .utils import (
    load_cached_product,
    set_random_seed,
    available_torch_devices,
    select_torch_device,
    resolve_torch_device,
    make_redshift_grid,
    configure_redshift_grid,
    flux_to_ab_mag,
    finite_scale,
    asinh_transform,
    tensor_to_numpy_1d,
    gaussian_kernel_1d,
    gaussian_smooth_1d,
    apply_numpy_mask_to_tensor_dict,
    table_column_names,
    require_columns,
    table_length,
    table_column,
    numeric_table_column,
    string_table_column,
    validate_split_fractions,
    _path_tag,
)

# Export config
from .config import (
    AIONMagnitudeConfig,
    make_magnitude_config,
    resolve_training_paths,
)

# Export dataset
from .dataset import (
    CLAUDSSplitCatalogue,
    build_hsc_aion_features_from_table,
    build_hsc_magnitude_faint_end_mask,
    build_field_labels,
    add_numeric_feature,
    build_extra_feature_matrix_from_table,
    resolve_include_grizy_in_mlp,
    build_hsc_quality_mask_from_table,
    CLAUDSPhotoZBatch,
    CLAUDSPhotoZDataset,
    collate_clauds_photoz,
    CachedFusionBatch,
    CachedFusionDataset,
    collate_cached_fusion,
    build_raw_clauds_photoz_dataset,
    build_grizy_mlp_feature_matrix,
    clauds_redshift_filter_mask,
    make_field_aware_split,
    split_counts_from_fractions,
    make_random_split,
    make_split_labels,
    split_metadata,
    leave_one_field_out_splits,
    subset_cached_dataset,
    dataset_for_split,
    split_cache_matches_current_schema,
)

# Export models
from .models import (
    ExtraPhotometryEncoder,
    PhotoZHead,
    TabularPhotoZModel,
    AIONOnlyPhotoZModel,
    CLAUDSPhotoZModel,
    load_frozen_aion,
    extract_hsc_aion_embedding,
    build_baseline_model,
    load_baseline_model_from_checkpoint,
    load_aion_mag_adjustment,
    aion_mag_adjustment_tag,
    aion_mag_adjustment_metadata,
    validate_cached_aion_mag_adjustment,
    apply_aion_mag_adjustment_to_hsc_features,
    build_aion_mag_adjustment_source_matrix_from_table,
    AION_AVAILABLE,
    AION_FINETUNE_MODES,
    AION_IMPORT_ERROR,
    HSC_MAG_TOKEN_EMBEDDING_NAMES,
)

# Export caching
from .caching import (
    extract_aion_embeddings_to_memory,
    save_cached_product,
    ensure_cached_product_redshift_reference,
    refresh_cached_product_catalogue_features,
    make_cache_run_tag,
    build_grizy_mlp_feature_matrix,
    build_and_cache_aion_embeddings,
    build_and_cache_aion_embeddings_from_config,
)

# Export metrics
from .metrics import (
    redshift_cross_entropy_loss,
    predict_photoz_from_logits,
    normalize_redshift_reference,
    build_redshift_reference_from_table,
    point_photoz_metrics,
    binned_log_score,
    discrete_crps,
    pit_values,
    photoz_quantiles,
    credible_interval_coverage,
    summarize_pdf_metrics,
    calibration_diagnostics,
    hpd_mass_at_true_bin,
    conformal_hpd_threshold,
    conformal_hpd_set_mask,
    evaluate_conformal_hpd,
    redshift_probability_distribution,
    validate_zphot_bins,
    tomographic_bin_labels,
    assign_tomographic_bins,
    catalogue_redshift_reference,
    sample_lognormal_from_percentiles,
    sample_catalogue_redshift_per_object,
    sample_inferred_redshift_per_object,
    resolve_redshift_hist_edges,
    probability_density_from_samples,
    sample_z_inferred_distribution,
    sample_population_z_distribution,
)

# Export training
from .training import (
    logits_from_cached_batch,
    train_pdf_model_one_epoch,
    evaluate_pdf_model,
    cached_product_to_dataset,
    split_cached_product,
    make_cached_loader,
    evaluate_model_on_dataset,
    evaluate_model_by_field,
    save_calibration_artifacts,
    train_single_baseline,
    train_all_baselines,
    run_baseline_training,
    load_and_evaluate_baseline,
    run_training_and_evaluation,
)

# Export plotting
from .plotting import (
    plot_zpred_vs_zphot,
    plot_pit_histogram,
    apply_baseline_to_catalogue,
    plot_redshift_probability_distribution,
    plot_nz_lensing_alike,
    compare_zpred_vs_zphot,
    compare_pit_histogram,
    compare_redshift_probability_distribution,
    compare_nz_lensing_alike,
    compare_config_loss,
    run_config_pair,
    plot_redshift_pdf_comparison,
    plot_sampled_z_inferred_distribution,
)

# Export extra_bands/u_band functionality
from .extra_bands import (
    DEFAULT_EXTRA_BANDS,
    EXTRA_BAND_LABELS,
    resolve_extra_band_name,
    resolve_extra_band_names,
    extra_band_feature_name,
    extra_band_valid_feature_name,
    extract_extra_band_magnitudes_from_table,
    extract_extra_band_magnitudes_from_split_arrays,
    make_extra_band_feature_matrix,
    build_extra_band_feature_matrix_from_table,
    build_grizy_usable_mask_from_split_arrays,
    load_extra_band_magnitudes_from_split_cache,
    make_no_extra_feature_product,
    make_extra_band_product,
    summarize_extra_band_ablation,
    format_extra_band_ablation_summary,
    run_extra_band_ablation,
    run_extra_bands_ablation,
    summarize_extra_bands_ablation,
    format_extra_bands_ablation_summary,
    
    # u-band specific
    load_u_magnitude_from_split_cache,
    make_no_u_feature_product,
    make_u_magnitude_product,
    summarize_u_magnitude_ablation,
    format_u_magnitude_ablation_summary,
    run_u_magnitude_ablation,
    run_u_band_ablation,
)

_MORPHOLOGY_EXPORTS = {
    "AION_IMAGE_TOKEN_KEY",
    "AION_IMAGE_BAND_ALIAS",
    "AION_IMAGE_INPUT_SIZE",
    "AION_IMAGE_GRID_SIZE",
    "DEFAULT_AION_IMAGE_QUANTIZER_LEVELS",
    "AIONMorphologyConfig",
    "resolve_morphology_paths",
    "cache_aion_morphology_tokens",
    "FSQTokenDecoder",
    "ImageTokenFactorEncoder",
    "PhotometryOnlyPhotoZModel",
    "MorphologyResidualPhotoZModel",
    "MorphologyTokenBatch",
    "MorphologyTokenDataset",
    "collate_morphology_token_batch",
    "make_morphology_loader",
    "morphology_product_to_dataset",
    "split_morphology_product",
    "train_morphology_one_epoch",
    "evaluate_morphology_model",
    "train_single_morphology_model",
    "run_morphology_experiment",
}


def __getattr__(name: str):
    if name not in _MORPHOLOGY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".morphology", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _MORPHOLOGY_EXPORTS)
