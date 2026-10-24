# Level-2 filter-aware input implementation

This work implements `SFT-label_1023.md` in isolated outputs. Archived runs and fast-check artifacts are read-only inputs. The final measured pilot results are collected in the project-root `SFT-label_results_1023.md`; the commands below use fresh directories unless explicitly continuing completed stages.

## Pipeline and representations

`aion_magnitude/filter_photometry.py` provides three serializers:

- Level 0 calls the original row serializer with its original prefix, schema, feature order, five-decimal values and missing token. It is native-only by design.
- Level 1 emits structured rows with survey, measurement type, units, exact band bindings, numerical values/errors, and EBV when available. Unknown Legacy optical releases fail rather than receiving guessed filters.
- Level 2 adds only `pivot_nm` and `width_nm` to each Level-1 band row. Removing these fields gives the exact Level-1 text. No labels, IDs or source-row coordinates are inserted into any prompt.

The original Level 0 already identifies bands and surveys. Level 0 versus Level 1 therefore tests structure and spelling as well as identity presentation. Only Level 1 versus Level 2 isolates the additional physical summaries under this design. A matched-length or shuffled-metadata control remains a useful follow-up.

All levels use the existing Qwen3.5-4B-Base text backbone, last-nonpadding-token pooling, 256-hidden-unit PDF head, categorical CE loss and original redshift grid. QLoRA and DoRA both use the existing trainer. The only shared-trainer extension is `test_dataset=None`, which omits creation/evaluation/reporting of the held-out test split. Existing callers supplying test datasets retain their evaluation path.

Training configuration, serializer version, registry version/hash, text hashes, cohort hash, IDs and split membership are recorded. Source epochs, seeds and subset budgets are explicit in each pilot contract. No existing adapters, ranks, target-module choices or learning rates were replaced with a new tuning strategy.

## Physical mapping and response definitions

`configs/filter_registry_v1.json` contains 21 actual response profiles: HSC grizy and r2/i2 alternatives; MegaCam MP9302 u and MP9301 u-star; DECam grz; BASS gr; MzLS z; WISE W1/W2; VIRCAM YJHKs. Original public curve files, normalized nm/response CSVs and upstream licenses are vendored in `configs/filter_responses_v1/`.

Every profile records a pinned upstream commit, URL, SHA256 hashes, units and approximation note. Pivot wavelength is `sqrt(integral R*lambda / integral R/lambda)` and equivalent rectangular width is `integral R / max(R)`, using the same wavelength grid and nm units. These quantities are independent of a galaxy SED. They are not arbitrary central wavelengths or FWHMs. Negative scan-noise response samples are clipped at zero and counted. Original response data are retained unchanged.

Identification and component provenance are documented in the response-directory README. SVO was unavailable; measured public responses distributed by DESI speclite and sedpy supplied a reproducible alternative. The nominal HSC r/i choices approximate unknown r/r2 and i/i2 coadd mixtures. DECam DR1 system curves are representative approximations for southern Legacy DR9, not exact object-specific responses.

Legacy RELEASE 9010/9012 uses DECam grz; 9011 uses BASS gr and MzLS z. All use WISE W1/W2. For the 256 existing fast-check queries, the resolver recovered 192 rows from RELEASE 9010, 63 from 9011 and one from 9012, matching the actual saved flux values as well as TARGETID. It checks source-file fingerprints. Lowercase HSC y and uppercase VIRCAM Y remain distinct.

## Numerical conventions and missingness

HSC retains the existing valid cmodel magnitudes/errors converted to AB nanomaggies: `f=10**((22.5-m)/2.5)` and `sigma_f=f*ln(10)/2.5*sigma_m`. The uncertainty expression is first-order. Legacy retains signed native AB nanomaggies, sigma from inverse variance, and the existing validity cuts. No logarithm is applied to signed fluxes. EBV is supplied as in the original pipeline; no new extinction correction is applied.

Native CLAUDS retains the existing cmodel-flux-to-AB-magnitude inputs at catalogue zero point 23, followed by the exact training-only median imputation. The historical model has no magnitude-error or EBV input. Levels 1/2 retain those values, with sigma/EBV explicitly unavailable; original missing masks are preserved in the cohort sidecar, not silently inferred from imputed values.

The optional CLAUDS common-flux path instead reads the actual `FLUX_CMODEL_*` and `FLUXERR_CMODEL_*` columns. Both are multiplied by `10**((22.5-23)/2.5)` to obtain AB nanomaggies. Sentinel -99 is missing; valid negative/zero fluxes remain valid, and invalid uncertainties remain unavailable. It does not exponentiate imputed magnitudes. Flux validity and historical magnitude missingness are separate masks. This path is an independent numerical ablation; it does not claim to harmonize extinction treatment, apertures or filter responses.

CLAUDS source validation IDs and labels can be checked exactly against the archive. All pilots subsample only the original train/validation partitions. Held-out test objects are never selected for training, method selection or evaluation here.

The usable CLAUDS preparation is `clauds-validation-portable/`. An initial validation attempt exposed object-typed split strings from the legacy splitter: values were identical but raw byte hashes were process-dependent. The runner now stores fixed-width Unicode split arrays and rejects object arrays in cohort hashing. `clauds-validation/INVALID.md` records the superseded, never-trained bundle.

## Measured tokenizer inputs

The manifests record full actual token-length audits; `*-examples.json` contains the exact text and tokenizer IDs. `missing-record-examples.json` supplies a genuine missing-band source-training example even when the high-quality small pilot happens to contain none. It does not alter pilot membership.

| Input cohort | L0 max | L1 max | L2 max |
|---|---:|---:|---:|
| HSC native, 192 pilot records | 237 | 172 | 270 |
| Legacy native, 256 audited queries | 221 | 173 | 276 |
| CLAUDS native, 192 validation-preparation records | 140 | 243 | 462 |
| CLAUDS common flux, same 192 records | — | 302 | 521 |

The shared pilot limit is 1024. Every complete constructed record must pass a no-truncation audit before training or inference. The Legacy/CLAUDS target transfer paths also audit every actual query. GPU memory and elapsed time are measured for each completed training arm in its `summary.json`; token counts are not presented as memory measurements.

## Validation

`tests/test_filter_photometry.py` checks every physical mapping and response lookup, nm units, flux/error units, independent uncertainty validity, negative/zero fluxes, missing values, source-label and object-order preservation, split isolation, no redshift fields in serialized inputs, Level-1/2 ablation equivalence, hash verification and overflow rejection. It recomputes registry summaries from vendored curves. A real training-loop unit test verifies that `test_dataset=None` neither evaluates nor reports test results.

All 512 archived Legacy/CLAUDS target input strings are byte-identical through Level 0. The GPU Level-0 source replay reproduces all 32 archived HSC logit vectors with maximum absolute error 0.0. See `hsc-pilot/level0-archived-replay.json`. This is distinct from training a fresh Level-0 pilot head.

Run the relevant checks from the repository root:

```bash
.pixi/envs/default/bin/python -m pytest -q tests/test_filter_photometry.py tests/test_qwen_posttraining.py tests/test_qwen_dora_rlvr.py tests/test_qwen_posttraining_comparison.py tests/test_survey_posttraining.py tests/test_qwen_zero_shot_sanity.py
```

## Reproducible commands

Prepare a fresh shared HSC cohort and all three native input levels:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_aware.py --stage prepare --catalogue-kind galaxiesml-spec --output-dir filter-aware-runs/new-hsc-pilot --train-rows 128 --val-rows 64 --max-seq-length 1024
.pixi/envs/default/bin/python notebooks/qwen_filter_aware.py --stage legacy-groups --output-dir filter-aware-runs/new-hsc-pilot
```

Select an unused MIG UUID in `CUDA_VISIBLE_DEVICES`. The runner acquires the shared advisory lock and also requires less than 0.2 GiB already used. Low reported memory does not override another launcher's reservation. The launcher runs the three arms and native zero-shot transfer sequentially:

```bash
CUDA_VISIBLE_DEVICES=YOUR_IDLE_MIG_UUID .pixi/envs/default/bin/python scripts/run_filter_aware_pilot.py --output-dir filter-aware-runs/new-hsc-pilot --epochs 5 --head-warmup-epochs 3 --transfer
.pixi/envs/default/bin/python notebooks/plot_filter_aware.py --run-dir filter-aware-runs/new-hsc-pilot --report filter-aware-runs/new-hsc-pilot/report.md
```

To train one DoRA arm with the same prepared data, use a fresh method-specific output:

```bash
CUDA_VISIBLE_DEVICES=YOUR_IDLE_MIG_UUID .pixi/envs/default/bin/python notebooks/qwen_filter_aware.py --stage train --output-dir filter-aware-runs/new-hsc-pilot --level 2 --method dora --epochs 5 --head-warmup-epochs 3
```

Prepare CLAUDS native/common-flux inputs, then optionally train a distinct flux arm:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_aware.py --stage prepare --catalogue-kind clauds --output-dir filter-aware-runs/new-clauds-pilot --train-rows 128 --val-rows 64 --max-seq-length 1024
CUDA_VISIBLE_DEVICES=YOUR_IDLE_MIG_UUID .pixi/envs/default/bin/python notebooks/qwen_filter_aware.py --stage train --output-dir filter-aware-runs/new-clauds-pilot --level 2 --numerics flux --epochs 5 --head-warmup-epochs 3
```

An independent numerical-transfer ablation can reuse an already trained HSC Level-1/2 native source model while converting the same CLAUDS queries using their raw flux/error columns. Its outputs go to `transfer-flux/`, separate from native `transfer/`:

```bash
CUDA_VISIBLE_DEVICES=YOUR_IDLE_MIG_UUID .pixi/envs/default/bin/python notebooks/qwen_filter_aware.py --stage transfer --output-dir filter-aware-runs/new-hsc-pilot --level 2 --target-numerics flux
```

Legacy source preparation is implemented through the same `--catalogue-kind desi-ls` entrypoint, reading its exact archived cohort cache and parsing its physical groups. That full cache is absent from this session. Restore the cache referenced by the original `prepared.json` before launching; this implementation does not silently rebuild a different selection. The 256 queried Legacy rows are independently available and were fully filter-resolved.

No CLAUDS-source or Legacy-source matched training, DoRA training, common-flux training/transfer, mixed-survey training or final test evaluation was launched as part of the initial HSC pilot. These remain follow-ups, not completed results.

## Larger source-learning follow-up — not executed

The 128-object, 40-update pilot remains severely underfit in all three arms. Its small metric differences do not answer the scientific transfer question. The next useful comparison should give each source model enough optimisation to learn its own validation population, then repeat zero-shot transfer and add repeated seeds and metadata controls. For example, the following matched 1024-training/256-validation, 10-epoch configuration is reproducible, but was **not launched**; it is substantially more expensive than the completed implementation check:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_aware.py --stage prepare --catalogue-kind galaxiesml-spec --output-dir filter-aware-runs/hsc-source-learning --train-rows 1024 --val-rows 256 --max-seq-length 1024
.pixi/envs/default/bin/python notebooks/qwen_filter_aware.py --stage legacy-groups --output-dir filter-aware-runs/hsc-source-learning
CUDA_VISIBLE_DEVICES=YOUR_IDLE_MIG_UUID .pixi/envs/default/bin/python scripts/run_filter_aware_pilot.py --output-dir filter-aware-runs/hsc-source-learning --epochs 10 --head-warmup-epochs 3 --transfer
```

Do not interpret the larger budget as a guarantee of convergence. Check its source learning curves before treating its cross-survey results as an adequate scientific ablation. The completed three training arms used about 24 minutes of GPU time in total (including each arm's model loading and validation); the measured per-arm peaks were 5.45 GiB allocated. Those are measurements of this pilot, not a memory or runtime guarantee for larger configurations.
