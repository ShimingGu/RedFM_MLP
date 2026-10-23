#!/usr/bin/env python3
"""Plot the saved zero-shot pilot; no model inference or parameter fitting."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aion_magnitude.metrics import summarize_pdf_metrics


def continuous_pit(pz, redshift, edges):
    """CDF of a piecewise-uniform PDF, interpolated inside the occupied bin."""
    index = np.clip(
        np.searchsorted(edges, redshift, side="left") - 1, 0, pz.shape[1] - 1
    )
    rows = np.arange(len(redshift))
    cdf = np.cumsum(pz, axis=1)
    before = np.where(index > 0, cdf[rows, np.maximum(index - 1, 0)], 0)
    fraction = np.clip(
        (redshift - edges[index]) / (edges[index + 1] - edges[index]), 0, 1
    )
    return before + fraction * pz[rows, index]


def save(fig, folder, stem):
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{stem}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run_dir / "summary.json").read_text())
    source = torch.load(
        Path(summary["source_run"]) / "qlora/result.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    source_data = {
        key: value.numpy()
        for key, value in source["val_evaluation"].items()
        if isinstance(value, torch.Tensor)
    }
    datasets = [
        (
            "HSC: source validation",
            "DESI spec-z",
            source_data,
            source["final_metrics"]["val"],
        )
    ]
    for kind, title, label in (
        ("desi-ls", "DESI–Legacy: zero shot", "DESI spec-z"),
        ("clauds", "CLAUDS: zero shot", "ZPHOT (photometric reference)"),
    ):
        with np.load(
            args.run_dir / f"{kind}_predictions.npz", allow_pickle=False
        ) as archive:
            data = {key: archive[key] for key in archive.files}
        datasets.append((title, label, data, summary["targets"][kind]["metrics"]))
    colors = ("#3973ac", "#d06b32", "#7c589b")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.8), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, (title, label, data, metrics), color in zip(axes, datasets, colors):
        ax.scatter(
            data["z_spec"],
            data["z_p50"],
            s=8,
            alpha=0.35 if len(data["z_spec"]) > 1000 else 0.65,
            color=color,
            rasterized=True,
        )
        ax.plot([0, 4.5], [0, 4.5], color="0.35", ls="--", lw=1)
        ax.set(
            xlim=(0, 4.5),
            ylim=(0, 4.5),
            xlabel=label,
            title=f"{title}\nN = {len(data['z_spec']):,}",
            aspect="equal",
        )
        text = (
            f"NMAD  {metrics['nmad']:.4f}\nOutliers  {metrics['catastrophic_outlier_fraction']:.1%}\n"
            f"Bias  {metrics['bias']:.4f}\n"
            f"CE  {metrics['cross_entropy']:.3f}\nCRPS  {metrics['mean_crps']:.3f}\n68% coverage  {metrics['p16_p84_coverage']:.1%}"
        )
        ax.text(
            0.04,
            0.96,
            text,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.85"),
        )
    axes[0].set_ylabel("Predicted redshift (PDF median)")
    fig.suptitle(
        "Fixed HSC-trained Qwen3.5-4B QLoRA + head: raw survey transfer\nNo calibration examples or weight updates; target validation subsets, source-overlap excluded",
        fontsize=12,
    )
    save(fig, args.plot_dir, "hsc_qlora_raw_transfer_predictions")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7), constrained_layout=True)
    for column, ((title, label, data, metrics), color) in enumerate(
        zip(datasets, colors)
    ):
        pit = continuous_pit(data["pz"], data["z_spec"], data["redshift_edges"])
        ax = axes[0, column]
        ax.hist(pit, bins=np.linspace(0, 1, 16), density=True, color=color, alpha=0.8)
        ax.axhline(1, ls="--", color="0.3", lw=1)
        ax.set(
            title=title,
            xlabel="Within-bin interpolated PIT",
            ylabel="Density",
            xlim=(0, 1),
        )
        axes[1, column].hist(
            data["z_spec"],
            bins=np.linspace(0, 4.5, 46),
            density=True,
            histtype="step",
            color="0.2",
            label=label,
            linewidth=1.5,
        )
        density = data["pz"].mean(axis=0) / np.diff(data["redshift_edges"])
        axes[1, column].plot(
            data["redshift_centers"],
            density,
            color=color,
            label="Mean predicted PDF",
            lw=1.5,
        )
        axes[1, column].set(
            xlabel="Redshift", ylabel="Normalized density", xlim=(0, 4.5)
        )
        axes[1, column].legend(fontsize=8)
        if column:
            summary["targets"]["desi-ls" if column == 1 else "clauds"][
                "interpolated_pit_mean"
            ] = float(pit.mean())
    fig.suptitle(
        "Raw transfer: probability calibration and redshift distributions\nPIT uses piecewise-uniform interpolation; reported historical metrics retain the original definitions",
        fontsize=12,
    )
    save(fig, args.plot_dir, "hsc_qlora_raw_transfer_calibration")

    # Reuse the already saved target-trained frozen control on the exact Legacy queries.
    legacy_run = Path("/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10")
    baseline = torch.load(
        legacy_run / "frozen/result.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    saved = baseline["val_evaluation"]
    legacy_meta = json.loads((legacy_run / "prepared.json").read_text())
    if baseline["metadata"]["survey_cohort_cache_key"] != legacy_meta["cache_key"]:
        raise ValueError("Legacy frozen control and query cohort differ")
    with np.load(
        Path(legacy_meta["cohort_cache"]) / "cohort.npz", allow_pickle=False
    ) as cohort:
        val_mask = cohort["split"] == "val"
        val_ids = cohort["targetid"][val_mask]
        # Cached baseline evaluation preserves this order but does not save IDs.
        torch.testing.assert_close(
            saved["z_spec"], torch.from_numpy(cohort["z"][val_mask]), rtol=0, atol=0
        )
    indices = {str(tid): i for i, tid in enumerate(val_ids)}
    query_ids = datasets[1][2]["targetid"]
    selected = [indices[str(tid)] for tid in query_ids]
    evaluation = {
        key: (
            value[selected]
            if isinstance(value, torch.Tensor)
            and value.ndim
            and value.shape[0] == len(val_ids)
            else value
        )
        for key, value in saved.items()
    }
    matched_baseline = summarize_pdf_metrics(evaluation)
    summary["desi_legacy_target_trained_frozen_head_reference_same_queries"] = (
        matched_baseline
    )
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    rows = [
        "# Raw zero-shot transfer sanity check — 2026-10-23",
        "",
        "Source: completed GalaxiesML-Spec HSC grizy Qwen3.5-4B-Base QLoRA adapter and its matching PDF head.",
        "",
        "All weights fixed. No support examples, target-label fitting, calibration, or band remapping. Source redshift grid remains 300 bins over 0–4.5.",
        "",
        "| Evaluation | N | NMAD | Outliers | Bias | CE | CRPS | 68% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for title, label, data, m in datasets:
        rows.append(
            f"| {title} | {len(data['z_spec']):,} | {m['nmad']:.4f} | {m['catastrophic_outlier_fraction']:.2%} | {m['bias']:.4f} | {m['cross_entropy']:.4f} | {m['mean_crps']:.4f} | {m['p16_p84_coverage']:.2%} |"
        )
    m = matched_baseline
    rows.append(
        f"| Legacy-trained frozen-Qwen head, same Legacy queries (reference) | {len(query_ids)} | {m['nmad']:.4f} | {m['catastrophic_outlier_fraction']:.2%} | {m['bias']:.4f} | {m['cross_entropy']:.4f} | {m['mean_crps']:.4f} | {m['p16_p84_coverage']:.2%} |"
    )
    rows += [
        "",
        "Raw transfer is substantially degraded on Legacy relative to its own target-trained frozen-head reference. CLAUDS predictions collapse to approximately z=0.13–0.14, with a large negative bias and 96.5% catastrophic disagreement. Its median-centered NMAD alone does not describe this failure.",
        "",
        "Source validation is shown for context and was used for checkpoint selection; it is not a matched target population.",
        "",
        "Both target samples use validation objects with reference redshifts in 0–4.5, fixed seed 1023, exact-ID exclusions, and >1 arcsec separation from every galaxy in the entire source cohort. CLAUDS is scored against ZPHOT, not spec-z; its original AB-magnitude serializer and training-only imputation are preserved. This also changes numerical representation relative to source fluxes. These small pilot samples do not establish a scientific transfer benchmark.",
        "",
        f"Source replay: {json.dumps(summary['source_replay'])}",
        "",
        "Prediction figure: [PNG](hsc_qlora_raw_transfer_predictions.png), [PDF](hsc_qlora_raw_transfer_predictions.pdf).",
        "",
        "Calibration figure: [PNG](hsc_qlora_raw_transfer_calibration.png), [PDF](hsc_qlora_raw_transfer_calibration.pdf).",
        "",
        f"Full run artifacts: `{args.run_dir.resolve()}`.",
        "",
    ]
    (args.plot_dir / "README.md").write_text("\n".join(rows))
    shutil_source = args.run_dir / "summary.json"
    import shutil

    shutil.copy2(shutil_source, args.plot_dir / "summary.json")
    print(
        json.dumps(
            {
                "plot_dir": str(args.plot_dir),
                "matched_legacy_frozen_control": matched_baseline,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
