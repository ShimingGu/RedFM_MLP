# Filter-metadata counterfactual experiment

The persistent GPU queue was launched at **2026-10-27T15:19:49.155709+00:00**. All conditions, mappings, galaxy IDs, redshift bins, metrics and 1,000 shared paired-bootstrap draws were frozen before new counterfactual scoring. The experiment uses the exact **61,696 P galaxies**, in original order, from the completed matched HSC/Legacy benchmark; no V objects enter this experiment.

The primary C matrix contains **13 conditions × two observational systems**: correct metadata; forward/reverse/seeded paired permutations; cyclic/seeded wavelength-only permutations; cyclic/seeded bandwidth-only permutations; ±2% and ±5% pivot shifts; and a numerical placebo. Frozen B is an external reference with different trained weights. Only `pivot_nm` and `width_nm` change; all other input bytes are preserved.

Archived full-cohort B/C metrics reproduce exactly. Twelve integrity tests and a synthetic end-to-end report test passed. Every complete input fits 512 tokens (maximum observed: 278). The placebo has exactly equal token lengths for every record in both domains, with changed token IDs; this does not establish equal attention or isolate physical reasoning. Other interventions have measured length differences, including +1 token for the HSC +5% pivot shift and shorter Legacy inputs for several permutations.

Live replay passed **exactly** for B and C on 256 objects in each observational system. All **24 counterfactual smoke cells** passed finite/normalized-PDF checks on the fixed 32-object hash subset, without scoring against real redshifts. The supervisor has advanced to full-cohort baseline replay on all three slices. These checks establish implementation compatibility, not a scientific metadata effect.

The autonomous stages are:

1. Exact live first-256 B/C replay in each domain, followed by 32-object label-blind counterfactual feasibility tests using dummy labels and no scientific scoring.
2. Exact live full-P replay of B and correct C in both domains. Any mismatch stops the queue.
3. All 24 perturbed C domain cells on the full P cohort, on three advisory-locked GPU slices.
4. Automatic paired-bootstrap analysis, H1–H4 interpretation, per-object outputs, redshift stratification and figures.

The expected wall time is approximately **20–24 hours**, subject to measured throughput. Checkpoints, physical filter registry and the original benchmark are protected by SHA256 checks. No training, target fitting or calibration occurs; the existing CLAUDS jobs keep their four slices.

Persistent directory:

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/metadata-counterfactual
```

Current status: `filter-aware-runs/1027/metadata-counterfactual/supervisor_status.json`. Logs: `execution_logs/`. Full inference progress: `jobs/<condition>-<domain>/status.json`. Policy SHA256: `6ae49322e71e7b83e72763c1e7f129f76ee9c7f4e1250be8dbb7dffe6ac891f0`.

[Predeclared policy](evaluation_policy_v1.json), [fixed metadata mappings](metadata_mappings_v1.json), [full token audit](tokenization_audit.json), and [reproduced archived baselines](archived_baseline_reproduction.json) are available now. The completed scientific report will be written here as `report.md` automatically. An unfinished experiment must not be interpreted as a completed counterfactual result.

To reproduce in a new output directory:

```bash
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/filter_metadata_counterfactual.py prepare --output-dir <new-output-directory>
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/supervise_filter_metadata_counterfactual.py --output-dir <new-output-directory>
```

The supervisor holds an exclusive process lock; completed inference chunks are resumable. Exact launch commands/PIDs, hashes and code snapshots are preserved in project storage. Raw predictions, input strings, token IDs and bootstrap arrays stay outside Git through the ignored storage link.
