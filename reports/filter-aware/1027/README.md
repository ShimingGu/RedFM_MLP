# Matched HSC/Legacy frozen evaluation

[Completed first-pass report](hsc-legacy-paired/report.md): five completed HSC-trained models, each evaluated with both observations of the same 1,024 independent galaxies and 256 validation-exposed galaxies. Includes paired confidence intervals, redshift stratification, source-only optical support, per-object differences and figures. Seven scientific checks passed. No completed Legacy-trained adapter/head is available for reverse transfer.

The full independent-pool evaluation was launched at approximately **2026-10-27 00:39 UTC**, using the three free MIG slices; the existing CLAUDS jobs retain their four slices. It evaluates all **61,696 independent pairs**, with the same 256-object exploratory validation subset, under unchanged cuts, models, serializers and support rules. Verified first-pass prediction chunks are reused.

Persistent output directory:

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full
```

Current status is recorded in `filter-aware-runs/1027/hsc-legacy-paired-full/supervisor_status.json`; model progress is in `<model>/status.json`, and logs are in `execution_logs/`. The supervisor queues all five models on three advisory-locked slices, then generates the full report automatically at `reports/filter-aware/1027/hsc-legacy-paired-full/report.md`. Completed chunks are retained if execution is interrupted.

The supervisor is `notebooks/supervise_hsc_legacy_paired.py`. Exact process commands and GPU assignments are stored in `*_launch.json`. Raw audit products, exact inputs, model identities, checkpoints' hashes, predictions and code snapshots are in project storage through the ignored `filter-aware-runs` symlink. The compact first-pass report occupies about 1.3 MB; no checkpoints were added to Git.
