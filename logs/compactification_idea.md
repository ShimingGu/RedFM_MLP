# Compactifying AION image tokens for Qwen

Date: 2026-07-20

## Purpose

The completed physical-magnitude-plus-image Qwen experiment used prompts of approximately 3,623 Qwen tokens but configured `max_length=2048`. Consequently, every prompt was truncated and only about 47% of the serialized 24×24 AION image-token grid survived.

Mean pooling then averaged the remaining galaxy-specific information with a large amount of shared prompt text. The resulting embeddings varied very little between galaxies and produced severely underfit/collapsed photo-z predictions.

The next experiment will change compactification and pooling together:

- compact the tokenized image to a centered 16×16 grid;
- append a dedicated final representation marker;
- use last-token pooling instead of mean pooling.

This is a deliberately simple first correction, not the final image representation.

## First implementation: centered 16×16 crop

The original AION token grid has shape 24×24. The new default Qwen image input will take the centered rows and columns:

```python
compact_grid = token_grid[4:20, 4:20]
```

This retains 256 central image tokens instead of all 576 tokens. For centered galaxy cutouts, it prioritizes the galaxy core and nearby structure while removing the outer four-token border.

Measured with the local Qwen tokenizer:

```text
raw 24×24 comma-separated grid: approximately 2,879 image tokens
centered 16×16 grid:             approximately 1,279 image tokens
centered 12×12 grid:               approximately 719 image tokens
```

The 16×16 representation is expected to fit within a 2,048-token context when combined with a compact magnitude section. The exact total must still be measured across all rows before a long extraction.

## Serialization sketch

The image remains described neutrally and retains its two-dimensional row order:

```text
tokenized galaxy image; source_grid=24x24; center_crop=16x16; row_order=top_to_bottom:
3527,3534,...;
3527,3527,...;
...
Combined galaxy representation:
```

The final marker is intentional. With last-token pooling, the pooled state should correspond to a final contextual position after the complete magnitude and image record, rather than to the bottom-right image code itself.

No fixed morphological meaning will be assigned to individual AION codes.

## Configuration requirements

The 16×16 crop must be a default option, not a hard-coded irreversible replacement. The serializer should retain explicit controls such as:

```text
image_input_mode = center_crop
image_crop_size = 16
source_image_grid_size = 24
```

Supported/planned modes should include:

- `full_grid`: original 24×24 grid with all 576 tokens;
- `center_crop`: centered crop, initially defaulting to 16×16;
- a future lossless global-palette representation of the complete 24×24 grid.

The original token cache must remain unchanged. Cropping should happen only while serializing a batch for Qwen. This ensures that later experiments can return to the complete 24×24 grid without rerunning AION image tokenization.

Cache filenames and metadata must distinguish at least:

- source grid size;
- image-input mode;
- crop size;
- serialization schema/version;
- pooling mode;
- final-marker text.

## Why use the 16×16 crop first?

Reasons for using it first:

- implementation and validation are straightforward;
- no new code vocabulary or palette mechanism is required;
- the galaxy is expected to be centered in the cutout;
- the physical magnitude description plus approximately 1,279 image tokens should remain below 2,048;
- it directly fixes systematic sequence truncation;
- it gives a quick empirical result before implementing a more elaborate lossless format.

This choice is lossy. It can remove companions, extended low-surface-brightness structure, edge asymmetry, background context, and centering failures. Results must therefore be described as using the centered 16×16 AION-token crop, not the complete tokenized image.

## Preserve the 24×24 option

The serializer should ultimately expose an explicit image-grid mode rather than hard-code cropping. Suggested modes:

```text
center16      # new near-term default
full24_raw    # original representation, retained for reproducibility
full24_palette # later lossless compact representation
```

Equivalent configuration could use a crop-size argument plus an encoding argument, but cache metadata and filenames must distinguish every combination.

The original 24×24 token store must not be overwritten. Cropping should occur while constructing Qwen input text, leaving the cached AION tokens unchanged. This allows all future representations to be generated from the same source product.

## Pooling change in the same step

For the next experiment, compactification and pooling will be changed together intentionally:

- image input: centered 16×16 crop;
- pooling: last non-padding token rather than mean pooling.

Append a stable final phrase after the image grid, for example:

```text
Combined galaxy representation:
```

The pooled state should correspond to the last token of this marker, not to the bottom-right image code. In a causal transformer, that state can attend to the preceding magnitude descriptions and image sequence.

Because compactification and pooling change together, this run measures their combined correction. It will not separately identify how much improvement comes from cropping versus pooling. A later ablation can compare mean and last pooling using the same compact input if that distinction becomes scientifically important.

## Required implementation properties

- Default the new corrected image experiment to `center16`.
- Keep `full24_raw` selectable.
- Do not alter or replace the underlying 24×24 AION token cache.
- Store the selected grid mode, crop bounds, serialized grid shape, pooling mode, and final-marker text in cache metadata.
- Include those settings in the embedding-cache filename/tag.
- Reject cached embeddings whose grid or pooling metadata differs.
- Preserve magnitude-first and image-second ordering.
- Continue describing the image neutrally as the tokenized galaxy image.
- Do not inject fixed morphological labels or interpretations.
- Print prompt token-length statistics before a large extraction.
- Fail before extraction if any prompt will be truncated unless truncation was explicitly authorized for a diagnostic run.
- Print flushed progress during extraction.
- Prefer chunked/resumable cache output before another multi-hour or multi-day run.

## Validation before a large run

On a representative sample, record:

- minimum, median, 95th-percentile, and maximum prompt token lengths;
- number and fraction exceeding `max_length`;
- exact serialized image shape;
- embedding shape and finiteness;
- per-dimension embedding standard deviation;
- galaxy-to-galaxy RMS variation;
- pairwise cosine-similarity distribution;
- deterministic repeat agreement;
- peak GPU memory and throughput.

The large run should proceed only if the truncation count is zero and the embeddings show materially greater galaxy-to-galaxy variation than the collapsed previous result.

## Later return to the complete image

The preferred later 24×24 solution is a stable global palette:

1. Determine the sorted set of AION codes used by the selected token product.
2. Assign each code a stable compact symbol.
3. Encode all 24 rows with explicit row boundaries.
4. Save the exact code-symbol mapping and a mapping hash in metadata.
5. Reuse the identical mapping for every split and any resumed extraction.
6. Reject unknown codes rather than silently changing their meaning.

Measured global-palette image length was approximately 599 Qwen tokens while preserving all 576 positions. This is substantially smaller than the 16×16 raw-ID crop and is lossless, but it requires more careful implementation and interpretation. We will return to it after establishing the simple 16×16 corrected baseline.

## Scientific comparison sequence

1. Existing 24×24 raw image, mean pooling, truncated — completed failure baseline.
2. Centered 16×16 image, last pooling, zero truncation — next corrected experiment.
3. If useful, centered 16×16 with mean versus last pooling — pooling ablation.
4. Full 24×24 palette image with the selected pooling method — lossless full-image return.
5. Magnitudes without versus with the compact image on identical galaxies — image-information ablation.
6. If frozen Qwen still cannot use AION codes, test a trainable token adapter/LoRA path as a separate experiment.

The centered 16×16 run is deliberately pragmatic: obtain a clean, fast result now while preserving a clear route back to the full tokenized galaxy image.
