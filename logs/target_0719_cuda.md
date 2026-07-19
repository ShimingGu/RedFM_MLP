# CUDA validation targets — 2026-07-19

These checks are intentionally deferred so the currently running experiment is not disturbed. Run them on a fresh GPU allocation or after the active process has finished.

## Before loading a model

- Record the GPU model, assigned GPU/slice, CUDA version, driver version, PyTorch version, and free VRAM.
- Confirm `CUDA_VISIBLE_DEVICES` identifies only the intended allocation.
- Check for unrelated processes with `nvidia-smi`; do not terminate processes merely because they are visible.
- Confirm the persistent checkpoint paths and available disk space.
- Use a tiny synthetic/catalogue sample first. Do not begin a full catalogue extraction until the smoke tests pass.
- Save the exact model name, dtype, quantization mode, batch size, maximum sequence length, pooling method, peak VRAM, elapsed time, and output shape for every run.

## Inference_Opt_TFM.py: GLM smoke test

Target checkpoint:

```text
/arc/home/gsm/hf_models/GLM-5.2-0.8B-A0.8B
```

Registered name:

```text
GLM-5.2-0.8B-A0.8B
```

- Load the tokenizer and base transformer through `load_inference_optimized_transformer()` on CUDA.
- Verify that `AutoModel`, rather than a causal-LM head, is loaded.
- Verify the actual model class, device placement, dtype, hidden size, and parameter count.
- Confirm `model.training` is false and parameters have `requires_grad=False` in the frozen configuration.
- Embed two or three short serialized catalogue rows.
- Verify `use_cache=False`, finite embeddings, expected row count, and expected embedding width.
- Verify mean, last-token, and mean+last pooling shapes.
- Verify normalized embeddings have approximately unit L2 norm when `normalize=True`.
- Verify an empty input batch returns the correct empty tensor shape.
- Repeat the same input twice and check deterministic output within the selected dtype's tolerance.
- Compare CPU and CUDA output for a very small batch using a tolerance appropriate for FP32/BF16.

## Precision and memory tests

- Test the GLM checkpoint in FP32 if it fits, recording peak allocated and reserved CUDA memory.
- Test BF16 on H100-compatible hardware and compare embeddings against FP32.
- Test `torch_dtype="auto"` and verify that CUDA selects BF16 as currently intended.
- Test increasing batch sizes starting at 1, then 2, 4, 8, and higher only while memory remains safe.
- Test representative sequence lengths, including short rows and the longest expected all-property row.
- Record examples/second and tokens/second after a warm-up batch.
- If 4-bit loading is needed, separately verify bitsandbytes compatibility with `glm_moe_dsa`, device placement, output finiteness, peak memory, speed, and embedding drift. Do not assume this custom MoE architecture supports 4-bit loading correctly.
- After deleting the test objects, run garbage collection and `torch.cuda.empty_cache()` only within the test process, then verify that memory is released when the process exits.

## Catalogue serialization and batching

- Test real rows containing normal floats, missing values, flags, very large/small values, and all intended column names.
- Confirm feature order is identical between training, validation, test, cached catalogues, and later inference.
- Confirm no column is silently dropped or reordered when converting from the catalogue/DataFrame to serialized text.
- Confirm rows longer than `max_length` are detected and quantify the truncation rate before the main experiment.
- Inspect tokenizer lengths for a representative sample and choose `max_length` from the observed distribution rather than assuming 2048 is required.
- Verify that tokenization on CPU and model execution on GPU do not cause avoidable device transfers or memory growth across batches.
- Run at least several hundred batches and check that allocated/reserved memory stabilizes rather than increasing continuously.

## Cached embedding production

- Add or exercise a resumable catalogue-embedding path before a large run.
- Store embeddings in chunks rather than accumulating the entire catalogue in GPU or system memory.
- Save row identifiers and/or source indices beside every embedding chunk.
- Save model and serialization metadata from `build_embedding_metadata()` with the cache.
- Record the checkpoint revision or local-file hashes, feature-name order, serialization configuration, dtype, pooling, normalization, and maximum sequence length.
- Verify that interruption and restart neither duplicate nor skip rows.
- Verify the final concatenated ordering against the original randomized train/validation/test indices.
- Verify cached embeddings can be loaded by downstream photo-z training without loading the transformer again.

## Existing Qwen model CUDA matrix

Test the existing Qwen backend with these registered models:

```text
Qwen3-8B-Base
Qwen3-4B-Base
Qwen3.5-0.8B-Base
Qwen3.5-4B-Base
Qwen3.5-4B
```

For each model that fits the allocation:

- Confirm the full registered name resolves to the intended persistent checkpoint.
- Verify `AutoModel` is used and no vocabulary-sized logits are materialized.
- Verify tokenizer compatibility, padding behavior, attention masks, hidden width, output shape, finite values, and deterministic inference.
- Measure cold-load time, first-batch time, steady-state throughput, peak VRAM, and host RAM.
- Start with batch size 1 and a tiny sample before testing production batch sizes.
- Compare BF16 and any supported quantized mode for speed, memory, and embedding drift.
- Exercise both magnitude-only prompts and the physical-magnitude-plus-tokenized-image prompts used by the comparison scripts.
- Confirm the physical column meaning is present only in the intended branch of `qwen-qwen_comparison.sh`.
- Confirm `qwen-mlp_full_image_comparison.sh` feeds column meaning plus the tokenized galaxy image to Qwen and leaves the MLP input unchanged.

## Aion image-token path

- Confirm the `aion` package and `CodecManager` import in the actual run environment.
- Load the image codec once on the intended device and verify its model/cache path.
- Tokenize a handful of real galaxy images and inspect token tensor shape, dtype, device, and valid-value range.
- Confirm the tokens are described to Qwen only as the tokenized galaxy image, without injecting a fixed morphological interpretation.
- Verify image-token and catalogue-row alignment after random subsampling and train/validation/test splitting.
- Measure tokenization time separately from Qwen forward time.
- Check whether precomputing and caching image tokens materially improves subsequent experiment time.
- Check that two concurrent sessions do not write partial files to the same cache target; use experiment-specific output names and atomic chunk completion where possible.

## Failure and recovery checks

- Reproduce only on an otherwise idle allocation any prior CUDA allocator/NVML assertion; capture the complete traceback, PyTorch/CUDA versions, GPU state, and memory use.
- Test graceful handling of CUDA OOM on a deliberately small, disposable smoke job. Ensure partial output is either valid and resumable or clearly marked incomplete.
- Confirm that terminating an embedding run does not corrupt completed chunks.
- Confirm restart resumes from the last verified chunk rather than recomputing the full catalogue.
- Confirm output filenames differ across model, prompt/input condition, sample split, dtype, and pooling configuration.

## Scientific comparison targets

- Establish an MLP baseline using the identical randomized galaxies and identical train/validation/test split.
- Compare frozen GLM embeddings plus the same downstream predictor against the MLP.
- Compare Qwen embeddings, GLM embeddings, and raw-feature MLP inputs without changing the downstream split or target definition.
- Separate the effects of model family, physical column descriptions, image tokens, added catalogue properties, and sample size.
- Report photo-z metrics overall and in scientifically relevant magnitude/redshift subsets, including outlier rate and uncertainty across random seeds.
- Treat the GLM result as an empirical architecture-test result, not as evidence about full GLM-5.2 capability.
- If frozen GLM embeddings are weak, test task-specific adaptation only as a distinct experiment; do not conflate it with the frozen-feature comparison.

## Minimum gate before a long run

A long extraction should start only after all of the following are true:

- The correct checkpoint and model class are loaded.
- A real mini-batch produces finite embeddings of the expected width.
- Catalogue/image rows align with saved identifiers.
- Truncation rate is measured and acceptable.
- Peak VRAM and host RAM are stable over repeated batches.
- Throughput gives an acceptable end-to-end duration estimate.
- Chunked output and restart recovery have been tested.
- The exact experimental configuration is saved with the embeddings.
