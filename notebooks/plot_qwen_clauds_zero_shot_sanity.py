#!/usr/bin/env python3
"""Plot the reverse CLAUDS-to-HSC/Legacy fixed-weight sanity check."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_qwen_zero_shot_sanity import continuous_pit, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run_dir / "summary.json").read_text())
    saved = torch.load(
        Path(summary["source_run"]) / "qlora/result.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    original = {
        key: value.numpy()
        for key, value in saved["val_evaluation"].items()
        if isinstance(value, torch.Tensor)
    }
    datasets = [
        (
            "CLAUDS: source validation",
            "ZPHOT (photometric reference)",
            original,
            saved["final_metrics"]["val"],
        )
    ]
    for kind, title in (
        ("galaxiesml-spec", "GalaxiesML-Spec: zero shot"),
        ("desi-ls", "DESI–Legacy: zero shot"),
    ):
        with np.load(
            args.run_dir / f"{kind}_predictions.npz", allow_pickle=False
        ) as archive:
            data = {key: archive[key] for key in archive.files}
        datasets.append(
            (title, "DESI spec-z", data, summary["targets"][kind]["metrics"])
        )
        summary["targets"][kind]["prediction_quantiles"] = np.quantile(
            data["z_p50"], [0, 0.25, 0.5, 0.75, 1]
        ).tolist()
        summary["targets"][kind]["reference_quantiles"] = np.quantile(
            data["z_spec"], [0, 0.25, 0.5, 0.75, 1]
        ).tolist()
    upper = float(original["redshift_edges"][-1])
    colors = ("#7c589b", "#3973ac", "#d06b32")
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
            s=7,
            color=color,
            alpha=0.3 if len(data["z_spec"]) > 1000 else 0.65,
            rasterized=True,
        )
        ax.plot([0, upper], [0, upper], ls="--", color="0.35", lw=1)
        ax.set(
            xlim=(0, upper),
            ylim=(0, upper),
            xlabel=label,
            title=f"{title}\nN = {len(data['z_spec']):,}",
            aspect="equal",
        )
        text = (
            f"NMAD  {metrics['nmad']:.4f}\nOutliers  {metrics['catastrophic_outlier_fraction']:.1%}\n"
            f"Bias  {metrics['bias']:.4f}\nCE  {metrics['cross_entropy']:.3f}\nCRPS  {metrics['mean_crps']:.3f}\n68% coverage  {metrics['p16_p84_coverage']:.1%}"
        )
        ax.text(
            0.04,
            0.96,
            text,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.9),
        )
    axes[0].set_ylabel("Predicted redshift (PDF median)")
    fig.suptitle(
        "Fixed CLAUDS-trained Qwen3.5-4B QLoRA + head: raw survey transfer\nNative input formats; no calibration or weight updates; source-overlap excluded",
        fontsize=12,
    )
    save(fig, args.plot_dir, "clauds_qlora_raw_transfer_predictions")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7), constrained_layout=True)
    for column, ((title, label, data, metrics), color) in enumerate(
        zip(datasets, colors)
    ):
        pit = continuous_pit(data["pz"], data["z_spec"], data["redshift_edges"])
        axes[0, column].hist(
            pit, bins=np.linspace(0, 1, 16), density=True, color=color, alpha=0.8
        )
        axes[0, column].axhline(1, color="0.3", ls="--", lw=1)
        axes[0, column].set(
            title=title,
            xlabel="Within-bin interpolated PIT",
            ylabel="Density",
            xlim=(0, 1),
        )
        axes[1, column].hist(
            data["z_spec"],
            bins=np.linspace(0, upper, 61),
            density=True,
            histtype="step",
            color="0.2",
            lw=1.5,
            label=label,
        )
        density = data["pz"].mean(axis=0) / np.diff(data["redshift_edges"])
        axes[1, column].plot(
            data["redshift_centers"],
            density,
            color=color,
            lw=1.5,
            label="Mean predicted PDF",
        )
        axes[1, column].set(
            xlabel="Redshift", ylabel="Normalized density", xlim=(0, upper)
        )
        axes[1, column].legend(fontsize=8)
        if column:
            summary["targets"]["galaxiesml-spec" if column == 1 else "desi-ls"][
                "interpolated_pit_mean"
            ] = float(pit.mean())
    fig.suptitle(
        "CLAUDS-source raw transfer: calibration and redshift distributions\nOriginal source grid retained; PIT uses piecewise-uniform within-bin interpolation",
        fontsize=12,
    )
    save(fig, args.plot_dir, "clauds_qlora_raw_transfer_calibration")
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.plot_dir / "clauds_source_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# CLAUDS-source raw zero-shot transfer — 2026-10-23",
        "",
        "Source: original 10-epoch CLAUDS Qwen3.5-4B-Base QLoRA adapter and matching PDF head. Weights remain fixed; no target calibration examples or target-label fitting.",
        "",
        "| Evaluation | N | NMAD | Outliers | Bias | CE | CRPS | 68% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for title, label, data, m in datasets:
        lines.append(
            f"| {title} | {len(data['z_spec']):,} | {m['nmad']:.4f} | {m['catastrophic_outlier_fraction']:.2%} | {m['bias']:.4f} | {m['cross_entropy']:.4f} | {m['mean_crps']:.4f} | {m['p16_p84_coverage']:.2%} |"
        )
    lines += [
        "",
        "Both target samples fail under this raw interface: predicted medians concentrate near z=3.8–3.9, all 256 queries in each sample are catastrophic outliers, and nominal 68% interval coverage is zero. The source replay exactly reproduces the archived model, so this is not a checkpoint-restoration mismatch.",
        "",
        "Source reference labels are CLAUDS ZPHOT; both target labels are DESI spec-z. Target inputs retain their native observed flux/error/EBV serializers; the source was trained on CLAUDS AB magnitudes. This deliberately tests raw transfer with both input-system/format and label-definition shifts, without harmonizing bands or units through remapping.",
        "",
        f"The original 300-bin source grid is retained: {summary['source_config']['z_min']}–{summary['source_config']['z_max']}. Cross-entropy values should not be compared directly against the earlier HSC-source pilot's different grid.",
        "",
        "Queries are 256 validation objects per target, within the source grid and the target catalogue's 0–4.5 range. Exact HSC object matches are excluded where available, and queries are farther than 1 arcsec from every object in the entire original CLAUDS source cohort. Source validation is shown only as context; it was used for checkpoint selection and is a different population.",
        "",
        f"Source replay: {json.dumps(summary['source_replay'])}",
        "",
        "Predictions: [PNG](clauds_qlora_raw_transfer_predictions.png), [PDF](clauds_qlora_raw_transfer_predictions.pdf).",
        "",
        "Calibration: [PNG](clauds_qlora_raw_transfer_calibration.png), [PDF](clauds_qlora_raw_transfer_calibration.pdf).",
        "",
        f"Run artifacts: `{args.run_dir.resolve()}`.",
        "",
    ]
    (args.plot_dir / "CLAUDS_source_README.md").write_text("\n".join(lines))
    index = args.plot_dir / "README.md"
    link = "[CLAUDS-source transfer to HSC and Legacy](CLAUDS_source_README.md)"
    if index.exists() and link not in index.read_text():
        index.write_text(
            index.read_text() + "\nAdditional sanity check: " + link + ".\n"
        )
    print(
        json.dumps(
            {key: value["metrics"] for key, value in summary["targets"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
