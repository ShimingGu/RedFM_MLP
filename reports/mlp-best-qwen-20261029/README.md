# Existing MLP vs. best compatible Qwen (2026-10-29)

`mlp_best_qwen_predictions.pdf/png/svg` and `mlp_best_qwen_nztomo.pdf/png/svg` compare the same validation objects in HSC–DESI (5,683) and CLAUDS (15,000). They are checkpoint-selection diagnostics, not held-out test confirmation. Existing MLP and Qwen training budgets differ.

Best Qwen is defined by lowest saved validation CE among completed evaluations with exactly matching object IDs, order, reference redshifts and PDF grid. HSC selects the Qwen3.5-4B Level-0 continued-QLoRA checkpoint at update 2,500. CLAUDS selects Qwen3.5-4B QLoRA. All completed eligible candidates and exclusions are recorded in `candidate_audit.json`. Recent matched-transfer training uses different train/validation cohorts and cannot be ranked on the MLP's saved validation cohort; inference-time corrupted-metadata conditions are not additional trained models. Unfinished 0.8B CLAUDS adapter jobs are not completed candidates. No complete Legacy-specific numeric MLP evaluation was located, so there is no Legacy panel.

HSC uses native photometric fluxes/errors and EBV with DESI spectroscopy. CLAUDS uses 11 catalogue magnitudes and catalogue ZPHOT references (not spec-z). Saved predictions and targets are unchanged. Both MLPs are the existing matched-numeric-MLP baselines from the capacity studies; these are ordinary numeric MLP models, despite the parent GLM experiment name. Matching IDs, labels and grids are asserted. Input representation/training provenance is in the source run metadata; this is a descriptive comparison, not a matched-budget architecture intervention.

Tomography selects objects by each model's PDF median. For each model/bin, solid stacked PDF density and dashed reference histogram contain exactly the same selected objects. Both densities integrate to one on the full model grid. Thus MLP/Qwen bin memberships can differ; the panel reports both counts. This avoids comparing different reference and predicted selections under the same model. Exact sums of saved PDFs avoid Monte Carlo noise. Every third PDF edge defines the coarser display histogram. For CLAUDS, dashed lines show the empirical catalogue ZPHOT distribution; they are not spectroscopic truth or reconstructed reference-PDF stacks.

HSC axes zoom to 0–2; all metrics and normalization use the full 0–4.5 grid. CLAUDS shows 0–6. Point metrics use the saved PDF median, the repository's lower-median NMAD convention, and outlier threshold |(zpred−zref)/(1+zref)| > 0.15.

`metrics.csv` contains plot metrics. `tomographic_curves.npz` preserves all binned curves and counts. `provenance.json` contains exact selected source paths and SHA256 hashes. Rebuild using the existing project environment:

```bash
/scratch/RedFM_MLP/.pixi/envs/default/bin/python reports/plot_mlp_best_qwen.py
```

The script uses CPU-only saved-artifact analysis and plotting; it never runs inference or training.
