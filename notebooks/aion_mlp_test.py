#!/usr/bin/env python3
"""Executable version of notebooks/aion_mlp_test.ipynb.

The default remains a small AION-only package smoke test. The
``standard-comparison`` mode reproduces the notebook's active comparison:
grizy MLP-only versus grizy AION-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REDFM_MLP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REDFM_MLP_ROOT.parent
if str(REDFM_MLP_ROOT) not in sys.path:
    sys.path.insert(0, str(REDFM_MLP_ROOT))

import aion_magnitude as am

DEFAULT_CATALOGUE = Path("data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits")
DEFAULT_OUTPUT_DIR = Path("/arc/home/gsm/aion_output/figures")
DEFAULT_CACHE_ROOT = Path("/scratch/.tmp-gsm/aion_output/cache")
FULL_EXTRA_BANDS = ("u", "u_star", "Y", "J", "H", "Ks")


def parse_max_rows(value: str) -> int | None:
    if value.lower() in {"none", "all", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-rows must be positive, or 'none'.")
    return parsed


def resolve_existing_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        REDFM_MLP_ROOT / path,
        WORKSPACE_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_figure(fig, path: Path, *, dpi: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RedFM_MLP notebook tests.")
    parser.add_argument(
        "--mode",
        choices=("aion-only", "standard-comparison", "standard-full-comparison"),
        default="aion-only",
        help="Run the small AION smoke test or the notebook's paired comparison.",
    )
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--max-rows", type=parse_max_rows, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--n-z-bins", type=int, default=300)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--tomographic-samples", type=int, default=100)
    return parser


def magnitude_config(
    args: argparse.Namespace,
    catalogue_path: Path,
    cache_root: Path,
    *,
    use_aion_embedding: bool,
    use_mlp_features: bool,
    include_grizy_in_mlp: bool,
    model_kinds: tuple[str, ...],
    extra_bands: tuple[str, ...] = (),
) -> am.AIONMagnitudeConfig:
    """Build one side of the notebook experiment with shared settings."""
    return am.AIONMagnitudeConfig(
        catalogue_path=catalogue_path,
        max_rows=args.max_rows,
        cache_root=cache_root,
        force_recompute_embeddings=args.force_recompute_embeddings,
        z_min=0.0,
        z_max=args.z_max,
        n_z_bins=args.n_z_bins,
        split_strategy="random",
        train_fraction=0.20,
        test_fraction=0.75,
        val_fraction=0.05,
        baseline_epochs=args.epochs,
        baseline_train_batch_size=args.train_batch_size,
        baseline_eval_batch_size=args.eval_batch_size,
        aion_embedding_batch_size=args.embedding_batch_size,
        hsc_mag_faint_limits={"g": 24.5, "r": 24.5, "i": 24.0, "z": 24.5, "y": 24.5},
        extra_bands=extra_bands,
        extra_band_invalid_fill="median",
        extra_band_include_valid_flags=False,
        use_aion_embedding=use_aion_embedding,
        use_mlp_features=use_mlp_features,
        include_grizy_in_mlp=include_grizy_in_mlp,
        aion_input_bands=("g", "r", "i", "z", "y"),
        aion_mag_adjustment_path=None,
        model_kinds=model_kinds,
        device_choice=args.device,
    )


def print_resolved_paths(label: str, config: am.AIONMagnitudeConfig) -> None:
    print(f"{label} resolved paths:")
    for key, value in am.resolve_training_paths(config).items():
        print(f"  {key}: {value}")


def run_standard_comparison(
    args: argparse.Namespace,
    catalogue_path: Path,
    output_dir: Path,
    cache_root: Path,
) -> int:
    """Reproduce the active paired-comparison cells in the notebook."""
    tabular_config = magnitude_config(
        args,
        catalogue_path,
        cache_root,
        use_aion_embedding=False,
        use_mlp_features=True,
        include_grizy_in_mlp=True,
        model_kinds=("tabular",),
    )
    aion_config = magnitude_config(
        args,
        catalogue_path,
        cache_root,
        use_aion_embedding=True,
        use_mlp_features=False,
        include_grizy_in_mlp=False,
        model_kinds=("aion",),
    )
    print_resolved_paths("grizy MLP-only", tabular_config)
    print_resolved_paths("grizy AION-only", aion_config)

    # Keep the same order and labels as run_config_pair(config_2, config_1)
    # in the notebook.
    labels = ("grizy-MLP-only", "grizy-aion-only")
    pair = am.run_config_pair(tabular_config, aion_config, split="test")
    eval_1 = pair["evaluation_1"]
    eval_2 = pair["evaluation_2"]
    prefix = output_dir / "grizy_aion_mlp_comparison"

    fig, _, _ = am.compare_config_loss(pair, labels=labels)
    save_figure(fig, Path(f"{prefix}_loss.jpeg"), dpi=100)

    fig, _ = am.compare_zpred_vs_zphot(
        eval_1,
        eval_2,
        labels=labels,
        pred_key="z_p50",
        pmax=5.0,
        show_metrics=True,
    )
    save_figure(fig, Path(f"{prefix}_scatter.jpeg"), dpi=100)

    fig, _ = am.compare_pit_histogram(eval_1, eval_2, labels=labels)
    save_figure(fig, Path(f"{prefix}_pit.jpeg"), dpi=100)

    fig, _, _ = am.compare_redshift_probability_distribution(
        eval_1,
        eval_2,
        labels=labels,
        gaussian_sigma_bins=1.0,
    )
    save_figure(fig, Path(f"{prefix}_nz.jpeg"), dpi=100)

    fig, _, _ = am.compare_nz_lensing_alike(
        eval_1,
        eval_2,
        labels=labels,
        zphot_bin=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        gaussian_sigma_bins=1.0,
        inferred_bin_key="z_p50",
        n_samples_per_object=args.tomographic_samples,
    )
    save_figure(fig, Path(f"{prefix}_nztomo.jpeg"), dpi=100)
    return 0


def run_standard_full_comparison(
    args: argparse.Namespace,
    catalogue_path: Path,
    output_dir: Path,
    cache_root: Path,
) -> int:
    """Compare AION-routed grizy against MLP-routed grizy with the same extras."""
    aion_fusion_config = magnitude_config(
        args,
        catalogue_path,
        cache_root,
        use_aion_embedding=True,
        use_mlp_features=True,
        include_grizy_in_mlp=False,
        model_kinds=("fusion",),
        extra_bands=FULL_EXTRA_BANDS,
    )
    tabular_config = magnitude_config(
        args,
        catalogue_path,
        cache_root,
        use_aion_embedding=False,
        use_mlp_features=True,
        include_grizy_in_mlp=True,
        model_kinds=("tabular",),
        extra_bands=FULL_EXTRA_BANDS,
    )
    labels = (
        "grizy-aion+uu*YJHKs-mlp",
        "grizy-mlp+uu*YJHKs-mlp",
    )
    print_resolved_paths(labels[0], aion_fusion_config)
    print_resolved_paths(labels[1], tabular_config)

    pair = am.run_config_pair(
        aion_fusion_config,
        tabular_config,
        model_kind_1="fusion",
        model_kind_2="tabular",
        split="test",
    )
    eval_1 = pair["evaluation_1"]
    eval_2 = pair["evaluation_2"]
    prefix = output_dir / "standard_full_comparison"

    fig, _, _ = am.compare_config_loss(pair, labels=labels)
    save_figure(fig, Path(f"{prefix}_loss.jpeg"), dpi=100)

    fig, _ = am.compare_zpred_vs_zphot(
        eval_1,
        eval_2,
        labels=labels,
        pred_key="z_p50",
        pmax=5.0,
        show_metrics=True,
    )
    save_figure(fig, Path(f"{prefix}_scatter.jpeg"), dpi=100)

    fig, _ = am.compare_pit_histogram(eval_1, eval_2, labels=labels)
    save_figure(fig, Path(f"{prefix}_pit.jpeg"), dpi=100)

    fig, _, _ = am.compare_redshift_probability_distribution(
        eval_1,
        eval_2,
        labels=labels,
        gaussian_sigma_bins=1.0,
    )
    save_figure(fig, Path(f"{prefix}_nz.jpeg"), dpi=100)

    fig, _, _ = am.compare_nz_lensing_alike(
        eval_1,
        eval_2,
        labels=labels,
        zphot_bin=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        gaussian_sigma_bins=1.0,
        inferred_bin_key="z_p50",
        n_samples_per_object=args.tomographic_samples,
    )
    save_figure(fig, Path(f"{prefix}_nztomo.jpeg"), dpi=100)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalogue_path = resolve_existing_path(args.catalogue)
    output_dir = Path(args.output_dir).expanduser()
    cache_root = Path(args.cache_root).expanduser()

    if not catalogue_path.exists():
        raise FileNotFoundError(f"Catalogue not found: {catalogue_path}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    print(f"aion_magnitude: {am.__file__}")
    print(f"available devices: {am.available_torch_devices()}")
    print(f"catalogue: {catalogue_path}")
    print(f"output figures: {output_dir}")
    print(f"cache root: {cache_root}")

    if args.mode == "standard-comparison":
        return run_standard_comparison(
            args,
            catalogue_path,
            output_dir,
            cache_root,
        )
    if args.mode == "standard-full-comparison":
        return run_standard_full_comparison(
            args,
            catalogue_path,
            output_dir,
            cache_root,
        )

    config = magnitude_config(
        args,
        catalogue_path,
        cache_root,
        use_aion_embedding=True,
        use_mlp_features=False,
        include_grizy_in_mlp=False,
        model_kinds=("aion",),
    )

    print_resolved_paths("AION-only", config)

    run = am.run_training_and_evaluation(config, model_kind="aion", split="test")
    test_eval = run["evaluation"]
    model_label = run["model_kind"].replace("_", " ").title()

    metrics = am.summarize_pdf_metrics(test_eval)
    metrics_path = output_dir / "aion_test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(jsonable(metrics), indent=2, sort_keys=True) + "\n")
    print(f"saved {metrics_path}")
    print("test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6g}")

    fig, _ = am.plot_zpred_vs_zphot(
        test_eval,
        pred_key="z_p50",
        target_label="z_phot",
        title=f"{model_label} Test: z_pred vs z_phot",
        pmax=5.0,
        show_metrics=True,
    )
    save_figure(fig, output_dir / "aion_test_zp50_vs_zphot.jpeg")

    fig, _ = am.plot_pit_histogram(
        test_eval,
        title=f"{model_label} Test: PIT Histogram",
    )
    save_figure(fig, output_dir / "aion_test_pit_histogram.jpeg")

    for pred_key in ("z_mean", "z_mode"):
        fig, _ = am.plot_zpred_vs_zphot(
            test_eval,
            pred_key=pred_key,
            target_label="z_phot",
            title=f"{model_label} Test: {pred_key} vs z_phot",
            pmax=5.0,
            show_metrics=True,
        )
        save_figure(fig, output_dir / f"aion_test_{pred_key}_vs_zphot.jpeg")

    gaussian_sigma_bins = 2.0
    fig, _, _ = am.plot_redshift_probability_distribution(
        test_eval,
        gaussian_sigma_bins=gaussian_sigma_bins,
        include_true=True,
        title=f"{model_label} Test: redshift probability distribution",
    )
    save_figure(fig, output_dir / "aion_test_redshift_probability_distribution.jpeg")

    fig, _, _ = am.plot_nz_lensing_alike(
        test_eval,
        zphot_bin=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        gaussian_sigma_bins=gaussian_sigma_bins,
        inferred_bin_key="z_p50",
        n_samples_per_object=args.tomographic_samples,
        title=f"{model_label} Test: tomographic n(z)",
    )
    save_figure(fig, output_dir / "aion_test_tomographic_nz.jpeg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
