# Targeted GLM-5.2-0.8B-A0.8B Failure Investigation

Date: 2026-08-28

## Objective

Determine why the current GLM-5.2-0.8B-A0.8B post-training runs underperform the frozen-embedding baseline, while separating four possible causes:

1. A mismatched prediction-head training protocol.
2. A difference between cached and live/prepared GLM representations.
3. An unsuitable prompt, layer, or pooling interface.
4. Adapter learning rate or GLM-specific target-module coverage.

This checkpoint is not the official GLM 5.2 release, but it remains a meaningful test of this architecture and training path.

## Current evidence

The completed full-photometry results are:

| Method | Test cross-entropy | NMAD | Catastrophic fraction |
|---|---:|---:|---:|
| Frozen GLM head | 4.9128 | 0.2454 | 0.5374 |
| QLoRA | 5.1947 | 0.3692 | 0.6670 |
| DoRA | 5.1938 | 0.3692 | 0.6672 |
| IA3 | 5.1934 | 0.3704 | 0.6689 |
| Residual adapter | 4.8977 | 0.2443 | 0.5349 |

The most important finding is that QLoRA, DoRA, and IA3 all selected epoch 0 as their best validation checkpoint. During the first three epochs, the representation is detached and the adapter learning rate is zero. Adapter learning begins only at epoch 3, after which validation performance worsens even though training loss continues to fall.

Consequently, the saved best PEFT checkpoints effectively contain an undertrained head and an untrained adapter. The present result does not yet show that GLM cannot be adapted.

There is also a substantial training-control mismatch:

| Setting | Cached frozen head | Direct PEFT trainer |
|---|---:|---:|
| Effective batch size | 256 | 16 |
| Head learning rate | 2e-4 constant | 2e-4 linearly scheduled |
| Head weight decay | 1e-4 | 1e-2 |
| Approximate optimizer updates | 2,350 | 37,500 |

The working hypothesis is therefore that the apparent failure begins in head training, before adapter optimization. Adapter placement and the model's intrinsic capacity should be tested only after this control is resolved.

## Experimental principles

- Diagnose with the full photometry task. Do not use the 429-row morphology cohort for causal conclusions.
- Preserve the existing train, validation, and test split.
- Select configurations using validation cross-entropy. Treat NMAD, catastrophic fraction, bias, and CRPS as secondary diagnostics.
- Compare runs at matched examples seen and optimizer-update counts, not merely matched epochs.
- Use one fixed seed for screening and three seeds only for configurations that survive screening.
- Do not inspect the untouched test set while choosing configurations.
- Save optimizer settings, target modules, trainable parameter names, and checkpoint-selection epoch with every run.

## Stage 0: representation invariants

This is a no-training or very-low-cost test on a fixed set of approximately 2,048 validation objects.

Evaluate identical serialized inputs through four paths:

1. Existing cached frozen embeddings.
2. Live 4-bit GLM in evaluation mode.
3. Live GLM after `prepare_model_for_kbit_training` but before PEFT insertion.
4. PEFT-wrapped GLM with adapters at their exact identity/zero initialization.

For each path, save:

- Token IDs and attention masks.
- Token-length and truncation statistics.
- Last non-padding token IDs.
- Pooled embeddings.
- Logits from the already trained frozen-baseline head.
- Cross-entropy, NMAD, bias, catastrophic fraction, and CRPS.

Compare embeddings using cosine similarity, normalized RMSE, and maximum absolute difference. Verify that the live and cached pooling implementations choose the same token. Also test a checkpoint save-and-reload round trip.

Pass condition:

- Mean embedding cosine similarity should be approximately 0.999 or higher.
- The trained frozen head should reproduce the cached validation metrics within ordinary numerical noise.

Interpretation:

- Failure before `prepare_model_for_kbit_training` indicates tokenization, model mode, caching, quantization, or pooling differences.
- Failure only after preparation indicates a preparation/dtype/model-state issue.
- Failure only after PEFT wrapping indicates that the supposedly inactive adapter is not an identity transformation.
- Matching logits and metrics clears the live representation path and points to the head optimizer protocol.

## Stage 1: matched frozen-head controls

Keep every GLM parameter and every adapter frozen. Run the following 2-by-2 comparison:

| Representation path | Baseline head protocol | Current direct-training head protocol |
|---|---|---|
| Cached embeddings | A: existing control | B |
| Live frozen GLM | C | D: current behavior |

For the baseline protocol, reproduce the successful cached-head settings exactly. If a physical batch size of 256 is not possible in the live path, use gradient accumulation while matching the effective batch, examples seen, optimizer steps, learning-rate curve, weight decay, evaluation cadence, and random seed as closely as possible. Keep the base GLM in evaluation mode while the prediction head remains in training mode.

Decision rules:

- If B and D fail similarly while A and C succeed, the current head optimizer protocol is the cause.
- If A and B succeed but C and D fail, the live representation/model-mode path is the cause.
- If C reproduces A, online GLM training is a valid base for the adapter experiment.
- If all four succeed, investigate adapter activation, checkpoint loading, and the transition to joint training.

The required gate for continuing is that C reaches within roughly 1% of A on validation cross-entropy and NMAD.

## Stage 2: warm-start causal adapter test

Do not initialize PEFT runs with a fresh prediction head. Load the fully trained frozen-baseline head into the live 4-bit GLM and verify its metrics before the first update.

Screen these runs:

| Run | Head state | Adapter state | Head LR | Adapter LR |
|---|---|---|---:|---:|
| Zero-adapter control | Loaded and frozen | Frozen | 0 | 0 |
| QLoRA-low | Loaded and frozen | Trainable | 0 | 1e-6 |
| QLoRA-mid | Loaded and frozen | Trainable | 0 | 3e-6 |
| QLoRA-joint | Loaded and trainable | Trainable | 1e-5 | 1e-6 |
| QLoRA-calibrated | Loaded and initially frozen | Trainable, then frozen | 0, then low LR | Best screening LR |

For `QLoRA-calibrated`, train the adapter alone first, then freeze it and perform one short, low-learning-rate head-calibration phase.

Evaluate every 250 optimizer updates. Record per-module gradient norms, adapter parameter norms, adapter-output delta norms, and the distance from the initialization checkpoint. Include the zero-adapter control for the same number of forward passes.

Early-stop a configuration if validation cross-entropy deteriorates consistently by more than 2% without a corresponding improvement in the secondary metrics.

Interpretation:

- Improvement with a frozen, pretrained head means the original failure was caused mainly by head initialization or scheduling.
- Immediate degradation at both adapter learning rates indicates either unsuitable targets or a weak final-token representation.
- Improvement during adapter-only training followed by degradation during joint training indicates destructive head movement.

## Stage 3: layer, pooling, and prompt probes

If a warm-started adapter still cannot improve the result, test whether the current interface exposes the relevant GLM information.

Cache hidden states from the embedding output and all six transformer blocks for a screening split such as 20,000 training and 5,000 validation objects. At each layer, compare:

- Last non-padding token pooling.
- Attention-mask-aware mean pooling.
- Concatenated last-token and mean pooling.
- A small learned attention-pooling probe.

Use an identical lightweight prediction head and identical optimizer settings for every probe. Include Qwen on the same sample as a positive architecture/control comparison.

Also compare three prompt forms:

1. The current prose prompt with named magnitudes.
2. A compact, labelled numeric representation.
3. A representation ending in a dedicated query or prediction token.

Interpretation:

- A better intermediate layer means the final layer is overly specialized for next-token generation.
- Better mean or learned pooling means last-token pooling is the wrong interface.
- Better compact/query-token prompting means the representation problem is formatting rather than capacity.
- No useful layer/pooling combination motivates broader adapter targets or a limited full-finetuning ceiling test.

## Stage 4: GLM-specific adapter placement

The current QLoRA and DoRA targets cover attention and indexer projections, while the dense MLPs and packed routed-expert tensors remain frozen. Once Stages 0-3 establish a valid training control, compare parameter-budget-matched target groups:

1. Current attention/indexer targets.
2. Dense MLP projections in the non-expert blocks.
3. Attention plus dense MLP targets.
4. Router and shared-expert components.
5. Routed-expert parameters in only the last two sparse blocks.
6. All eligible projections with rank reduced to match the trainable-parameter budget.

Use ordinary QLoRA first; DoRA and IA3 are unnecessary until one target set shows a reproducible gain. Log gradient and update norms by module so a run cannot be labelled unsuccessful when its selected parameters were effectively inactive.

The near-identical existing QLoRA, DoRA, and IA3 results are not evidence that the methods behave identically: all three selected a checkpoint from before adapter training.

## Stage 5: capacity ceiling

Use this only if no layer, prompt, pooling rule, or adapter placement works.

- Fully fine-tune the last transformer block on a 20,000-object screening subset using a small learning rate.
- Optionally unfreeze only its layer norms and output projections as an intermediate control.
- Compare against the frozen GLM head, a simple tabular MLP, and the matched Qwen probe.

If limited full fine-tuning improves while PEFT does not, the problem is adapter expressivity or placement. If it does not improve, the checkpoint probably lacks a useful mapping for this input representation and objective.

## Recommended execution order

Run only one stage at a time and use its result to choose the next:

1. Replay the trained frozen head through cached, live, prepared, and zero-adapter paths.
2. Run the 2-by-2 matched frozen-head control.
3. Run warm-started, head-frozen QLoRA at `1e-6` and `3e-6`.
4. Run the layer/pooling/prompt probe if adapter-only training fails.
5. Expand adapter targets only after identifying a representation with predictive signal.
6. Run the limited full-finetuning ceiling only if targeted PEFT still fails.

```text
Does the trained frozen head replay online?
├── No → representation, preparation, tokenization, or pooling mismatch
└── Yes
    └── Does matched online head training reproduce the cached baseline?
        ├── No → optimizer, batch, scheduling, or train/eval-mode problem
        └── Yes
            └── Does warm-start adapter-only training improve?
                ├── Yes → original failure was head initialization/scheduling
                └── No
                    └── Does another layer, pool, or prompt expose signal?
                        ├── Yes → change the model-to-head interface
                        └── No → expand GLM targets, then test the capacity ceiling
```

## Output layout

Use one expandable root such as:

```text
/arc/home/gsm/aion_output/figures/glm52-failure-analysis-0828/
├── stage0_invariants/
├── stage1_head_controls/
├── stage2_warmstart_qlora/
├── stage3_representation_probes/
├── stage4_target_sweep/
├── stage5_capacity_ceiling/
└── manifest.json
```

Each stage should contain its configuration, seed, split identity, metric history, selected checkpoint, and compact comparison plots. The root manifest should make it possible to add later RLVR or alternative-GLM results without changing the earlier conclusions.

## Expected first conclusion

The strongest current prediction is that the trained-head replay will succeed and the matched control will identify the direct head-training protocol as the primary cause. The adapter-target hypothesis becomes credible only if a warm-started, head-frozen adapter fails after the live representation and optimizer controls have passed.
