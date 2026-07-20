# Science status and targets — 2026-07-19

## Central scientific question

The project is testing whether frozen transformer representations can improve galaxy photo-z estimation relative to ordinary MLP inputs, and which source of information produces any improvement:

- numerical catalogue measurements;
- explicit physical descriptions of catalogue columns;
- tokenized galaxy images;
- transformer architecture/model family;
- task-specific post-training rather than frozen representations;
- larger training samples and richer catalogue inputs.

The comparisons must keep catalogue rows, randomized splits, redshift target, downstream training, and evaluation definitions aligned. Otherwise an apparent model improvement may instead come from different galaxies, cuts, or target leakage.

## Completed experiments found in the figure archive

The following completed products exist under `/arc/home/gsm/aion_output/figures`:

- Standard grizy AION versus MLP comparisons.
- `qwen_aion_comparison`: frozen Qwen versus frozen AION on matched grizy inputs and the same photo-z head.
- `image-on_comparison_mlp`: grizy MLP without versus with tokenized CLAUDS u-band images.
- `image-on_comparison_all_mlp`: all-magnitude MLP without versus with tokenized u-band images.
- `image-on_comparison_aion`: frozen grizy-AION representation without versus with tokenized u-band images.
- `qwen-mlp_full_image_comparison`: physical all-magnitude descriptions plus the tokenized galaxy image through Qwen, compared with the corresponding all-magnitude MLP/image-token branch.

The completed physical-magnitude-plus-image run used:

- a random request of 200,000 catalogue rows;
- 90,789 selected galaxies after the finite-target/photometry selection represented in the report;
- 11,451 morphology-matched training galaxies;
- 2,911 morphology-matched validation galaxies;
- Qwen3-8B-Base;
- mean pooling;
- maximum Qwen sequence length 2048;
- 4-bit loading;
- no magnitude faint-end cut;
- no imposed redshift-range cut beyond requiring a finite target;
- AION image tokenization, but not AION's image-to-redshift embedding.

Its manifest is:

```text
/arc/home/gsm/aion_output/figures/qwen-mlp_full_image_comparison/qwen_mlp_full_run.json
```

## Active experiment

The currently running experiment is magnitude-only Qwen3-8B-Instruct with versus without physical meanings for the input columns.

This experiment isolates whether explicit semantic/physical descriptions help when the underlying numerical magnitude information is unchanged. It should not be interpreted as an image-token test or as a comparison against an MLP.

Do not perform another CUDA model-load test in this session until this run has finished.

## Physical-meaning interpretation

Physical column descriptions do not materially increase transformer compute compared with model size and sequence processing, provided the descriptions do not cause severe sequence-length growth. Therefore they should remain in the intended physical-meaning branch.

The tokenized galaxy image is described to Qwen neutrally as the tokenized galaxy image. The prompt does not impose a fixed morphological interpretation. The aim is to allow the model or later task-specific adaptation to learn useful morphological associations rather than declaring them in advance.

The magnitude-only meaning-versus-no-meaning result should be examined before attributing gains in the completed physical-magnitude-plus-image run to column semantics.

## Image-token interpretation

Image tokenization and Qwen embedding extraction are distinct stages:

- AION supplies the image tokenizer/codec.
- The experiments do not use AION's internal image-to-redshift prediction as the image feature.
- Image tokens can be cached independently and reused for later training.
- Timing must separate catalogue loading, AION tokenization, text serialization/tokenization, transformer forward passes, and downstream photo-z training.

The completed MLP image-on comparisons provide controls for whether image tokens help even without Qwen. These should be consulted before claiming that a Qwen-plus-image improvement arises from transformer reasoning over morphology.

## Whole-catalogue experiment

The new files are:

```text
aion_magnitude/Inference_Opt_TFM.py
notebooks/iotfm_mlp.py
scripts/iotfm_mlp.sh
tests/test_iotfm_mlp.py
```

The intended whole-catalogue experiment compares:

- a frozen transformer embedding of each ordered catalogue record followed by the existing photo-z head;
- a conventional MLP representation of the same allowed catalogue information.

The CLAUDS FITS catalogue has 235 columns. The current default selection includes 209 and excludes 26:

- `ID` is excluded by default behind its own switch;
- `RA`, `DEC`, `tract`, and `patch` are excluded by default behind a separate location switch;
- 21 photo-z output fields are always excluded.

The excluded photo-z families are the primary, NIR, and 6B variants of:

- `ZPHOT`;
- `Z_LOW68`;
- `Z_HIGH68`;
- `Z_CHI`;
- `Z_PEAK`;
- `Posterior-Log`;
- `Likelihood-Log`.

`Likelihood-Log_star` remains included because it represents stellar classification information rather than a photo-z solution.

Only galaxies without a finite photo-z training target are removed by design. Missing catalogue measurements are retained as information:

- the transformer text representation uses an explicit missing token;
- the MLP representation fills the numerical value and adds a missingness indicator;
- categorical fields are encoded with train-derived categories and an unknown category;
- array-valued catalogue fields such as `FLAG_FIELD_BINARY` are retained component by component.

The identifier and location switches are independent:

```bash
IOTFM_INCLUDE_ID=1 ./scripts/iotfm_mlp.sh
IOTFM_INCLUDE_LOCATION=1 ./scripts/iotfm_mlp.sh
```

Both are off for the planned first run. This avoids an initial result being dominated by object numbering or spatial/sample-selection correlations. Later ablations can intentionally restore them to measure how informative those correlations are.

The default randomized split in `iotfm_mlp.py` is currently:

- train: 63%;
- test: 32%;
- validation: 5%.

The transformer and MLP branches receive exactly the same split labels.

## Inference-optimization interpretation

`inference-optimization/GLM-5.2-0.8B-A0.8B` is a small testing/development checkpoint that retains selected GLM architectural patterns with substantially reduced depth, dimensions, heads, and experts. It is not the full GLM-5.2 architecture with factual knowledge merely removed, and it should not be treated as a proxy for the full model's capability.

Such a checkpoint remains scientifically useful as a cheap structured transformer feature map or as a candidate for later task-specific adaptation. A frozen result must be reported as an empirical result for that checkpoint, not as evidence about full GLM-5.2.

## Planned Qwen3.5 change

The intended model for the first whole-catalogue experiment has changed to:

```text
Qwen3.5-4B-Base
```

This change is not complete yet. The present problems are:

- `scripts/iotfm_mlp.sh` still defaults to `GLM-5.2-0.8B-A0.8B`;
- `Inference_Opt_TFM.py` does not yet resolve the short name `Qwen3.5-4B-Base` through the existing local Qwen registry;
- the generic base-transformer loading and whole-catalogue forward path have not yet received their CUDA smoke test with Qwen3.5-4B-Base.

The checkpoint should be loaded through the base transformer (`AutoModel`) rather than a causal-language-model head so that vocabulary-sized logits are never materialized.

## Required next sequence

After the active magnitude-only Qwen run finishes:

1. Confirm the GPU allocation is idle and healthy.
2. Resolve the persistent local `Qwen3.5-4B-Base` checkpoint.
3. CUDA-smoke-test tokenizer and base-model loading with a few short whole-catalogue records.
4. Record actual model class, dtype, hidden width, output shape, finiteness, deterministic behavior, and peak VRAM.
5. Fix `Inference_Opt_TFM.py` to support the Qwen short name/local registry cleanly.
6. Change the `iotfm_mlp.sh` default from GLM to `Qwen3.5-4B-Base`.
7. Run CPU-only unit and catalogue-preprocessing tests again.
8. Run an end-to-end CUDA test with approximately 1,000 randomly selected catalogue rows and one training epoch.
9. Inspect tokenizer-length and truncation statistics for the 209-column records. A nominal maximum length of 2048 must not be accepted without measuring truncation.
10. If successful, test approximately 10,000 rows to estimate stable throughput, VRAM, host RAM, cache size, and full-run duration.
11. Only then launch the planned 200,000-row experiment.

Suggested first end-to-end scale after the Qwen fix:

```bash
AION_MAX_ROWS=1000 \
AION_EPOCHS=1 \
IOTFM_EMBEDDING_BATCH_SIZE=1 \
./scripts/iotfm_mlp.sh
```

## Caching requirements

Large transformer extraction should become resumable before a long catalogue run:

- write embeddings in verified chunks rather than accumulating the entire catalogue in GPU or host memory;
- save object IDs/source indices with every chunk;
- store the exact ordered included and excluded column lists;
- store serialization settings, tokenizer maximum length, pooling, normalization, dtype, quantization, checkpoint identity/revision, and split seed;
- verify restart does not duplicate or skip rows;
- ensure cached embeddings can train the downstream model without reloading Qwen;
- use experiment-specific filenames so simultaneous sessions cannot overwrite each other's caches.

The present `iotfm_mlp.py` cache is not yet chunk-resumable and should not be trusted for a multi-day full extraction until that is improved and tested.

## Comparisons still scientifically useful

After the current and whole-catalogue runs, the clean comparison matrix should include:

1. MLP on the chosen numerical catalogue representation.
2. Frozen Qwen on magnitude values without physical descriptions.
3. Frozen Qwen on the same magnitudes with physical descriptions.
4. MLP with versus without tokenized galaxy images.
5. Qwen magnitude representation plus image tokens versus a matched MLP/image-token branch.
6. Frozen Qwen3.5-4B-Base on the 209-column catalogue versus the matched whole-catalogue MLP.
7. Identifier/location ablations for the whole-catalogue experiment.
8. Frozen versus task-adapted transformer, only after the frozen comparison is established.
9. Sample-size scaling so a gain is not confused with a lucky or unrepresentative random draw.

## Interpretation safeguards

- Do not compare models trained on different galaxy populations as if only architecture changed.
- Do not allow any photo-z estimate, interval, likelihood, posterior, or spectroscopic-redshift target into the model inputs.
- Report both the requested catalogue sample size and the final usable/morphology-matched sample size.
- Report uncertainty across random seeds where practical.
- Report photo-z metrics overall and in magnitude/redshift subsets, including outlier rate and calibration/PIT behavior.
- Distinguish gains from column semantics, image tokens, richer measurements, spatial correlations, and task-specific adaptation.
- Treat unexpectedly strong results with `ID` or sky-position fields as possible survey-selection learning, not automatically as improved physical inference.
- Treat poor frozen-transformer performance as evidence about the chosen representation/checkpoint, not proof that transformers cannot help after task-specific adaptation.

## Infrastructure note

Troubleshooting in the other cluster session was abandoned. Its older Pixi could not parse the current manifest/lock format and lacked permission to self-update `/usr/local/bin/pixi`. No further work should be based on that session for now.

This checkout's `pixi.toml` was restored after an accidental attempted edit; there is no remaining `pixi.toml` diff from that incident.
