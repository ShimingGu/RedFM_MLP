# Table-model comparisons

Every launcher requires one of `--model=tabpfn`, `--model=tabfm`, or
`--model=tabicl`. The launcher retains the generic `tabxx` name, while the
result directory begins with the selected backend, for example:

```bash
./scripts/table_models/tabxx_noimage-aion_comparison.sh --model=tabicl
# /arc/home/gsm/aion_output/figures/table_models/tabicl_noimage-aion_comparison/
```

The five launchers are:

- `tabxx_noimage-aion_comparison.sh`: magnitude-only table without images
  versus the same table plus 2,880 normalized AION FSQ-factor columns
  (five decoded factors at each position of the 24x24 token grid).
- `tabxx_aion-timm_comparison.sh`: magnitude-only table plus decoded AION FSQ
  factors versus the same magnitudes plus a frozen timm image embedding.
- `tabxx-mlp_noimage_comparison.sh`: magnitude-only table model versus the
  repository's standard magnitude-only PDF MLP.
- `tabxx-mlp_aionimage_comparison.sh`: magnitude-only table model with decoded
  AION FSQ-factor columns versus the standard magnitude MLP with its trainable
  decoded AION-token image path.
- `tabxx_magonly-fulltable.sh`: 11 AB magnitudes versus exactly 55 fluxes,
  55 flux errors, and 11 Kron radii, without images.

Use a small preparation run before downloading a model or launching a large
experiment:

```bash
./scripts/table_models/tabxx_magonly-fulltable.sh \
  --model=tabicl --max-rows=2000 --prepare-only --save-input-table
```

All runs use seeded random sampling (default 50,000 rows, seed 42) and a
63%/32%/5% train/test/validation split. `ZPHOT` is visible only on training
rows. It is NaN on validation/test rows in the model-facing completion table.
Every other redshift-derived catalogue column is excluded from the features.
Imputation statistics are fitted on training rows only.

Each table arm writes `redshift_completion.npz`,
`redshift_completion.csv.gz`, `metrics.json`, and `table_schema.json`. The
compressed completion table contains the original masked target, inferred
held-out redshift, filled target, and evaluation-only truth. Add
`--save-input-table` to persist the potentially large feature matrix.

After both arms finish, every comparison—including
`tabxx_aion-timm_comparison.sh`—writes the side-by-side redshift image
`test_redshift_comparison.png` and its metrics/artifact index
`comparison_results.json` in the comparison output directory.

Model checkpoints download on first use unless `--no-allow-model-download` and
`--model-path=...` are supplied. TabICL code and weights are BSD-3-Clause.
TabFM and TabPFN weights are limited to non-commercial/non-production use.
TabPFN also requires the user to accept its model license and provide the
normal Prior Labs authentication token in a headless session.
