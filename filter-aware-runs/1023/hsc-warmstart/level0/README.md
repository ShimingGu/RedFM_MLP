# Experiment A: warm-start Level-0 QLoRA control

Execution specification: project-root `filter_aware_1023a.md`.

The durable job restores **both trained components** from `/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10/qlora/`. It verifies every restored adapter/head tensor, preserves the exact original source inputs and full 22,444/5,683 train/validation cohorts, and requires a full zero-update validation replay before any optimisation. This is separate from the earlier fresh-head 128-object pilot.

The source model is Qwen3.5-4B-Base; rank 8, alpha 16, dropout 0.05, q/k/v/o projection adapters, original PDF-head architecture and 300-bin 0–4.5 grid. Both adapter and head train from update one. There is no head-only warmup.

`matched_protocol.json` is the shared continued-SFT protocol for later Experiments B/C: maximum two epochs, adapter LR 3e-6, head LR 1e-5, effective batch 16, seed 1023 and validation every 250 optimizer updates, with additional epoch-end validation. The complete cohort produces 1,403 updates per epoch and at most 2,806 updates. Selection uses source-validation CE and includes the zero-update checkpoint; ties retain the earlier checkpoint.

The specification does not prescribe optimizer-moment restoration or an LR schedule. This implementation explicitly uses new AdamW moments and constant prescribed learning rates, with no LR warmup. Existing head/adapter weight decay and clipping settings are retained. The final effective batch in each epoch contains 12 objects; its loss is normalized by those 12 objects. These details must be reused identically in B/C.

The run uses an idle, exclusively locked MIG slice. It writes a restart checkpoint every 50 optimizer updates and at validation/epoch boundaries. No test objects enter the optimizer, source validation, or target evaluation. Target validation queries are the exact audited 256 Legacy and 256 CLAUDS objects used by the earlier fast-check. Target data enter inference only after source checkpoint selection.

## Status and final results

- `report.md`: automatically updated concise report; explicitly distinguishes pending stages from completed results.
- `status.json`: current phase, optimizer progress, best candidate, metrics and histories.
- `run.log`: durable process output.
- `job.json`: process ID and selected MIG slice.
- `zero_update_replay.json`: full 5,683-object replay checks and source metrics, when completed.
- `source_summary.json`: selected source checkpoint and source histories, when completed.
- `parameter_changes.json`: adapter/head changes for both last-trained and selected states relative to initialization.
- `selected/`: selected adapter, PDF head and trainable parameter state; zero-update initialization can remain selected.
- `transfer_summary.json`: zero-update versus selected-source metrics on identical audited Legacy/CLAUDS queries, when completed.
- `summary.json`: final completed-run summary; absent while work remains.
- `learning_curves.png/.pdf`, `transfer_predictions.png/.pdf`: automatically generated diagnostics after source selection/final transfer.

Source validation is used for selection, so improvements on it are model-selection results rather than independent test evidence. Best-CE selection can preserve CE by retaining update zero even if training worsens; the report also shows the latest validation metrics and both last/selected parameter changes. NMAD, coverage and cross-survey metrics must be assessed separately.

## Code and verification

- `notebooks/qwen_filter_warmstart.py`: preparation, strict trained-state restoration, full replay, durable launch, final transfer and report.
- `aion_magnitude/qwen_continued_sft.py`: matched protocol, accumulation, zero-update selection, resumable training and parameter-change diagnostics.
- `tests/test_qwen_continued_sft.py`: protocol, replay identity, strict selection, partial-batch normalization, actual parameter updates and bitwise-equivalent restart with dropout and shuffled batches.

Preparation records hashes of both implementation files. The source checkpoint, old fast-check and prior pilot are read-only inputs. All scientific outputs from this experiment remain in this directory. Existing full-cohort test labels are not copied into the prepared run.

## Commands

The initial job was launched with:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_warmstart.py --stage launch --gpu MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25
```

The process runs independently of the chat and performs final source selection, fixed-weight transfer and report generation automatically. Do not launch a second copy while its process is active.

After an interruption, the following resumes the saved optimizer, parameter, batch-position and RNG state, provided the slice is idle and unreserved:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_warmstart.py --stage launch --resume --gpu MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25
```

Regenerate the current/final report from saved results without training:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_warmstart.py --stage report
```

The implementation additionally saves `initial_model/`, `reference_validation.pt`, `zero_update_validation.pt`, source-only `source_cohort.npz`, and the exact `source_inputs.jsonl` so the run is reviewable and reproducible. `timing.json` defines total wall time since the first run start, including interruptions if resumed; `elapsed_seconds` in training status measures the accumulated training/periodic-validation portion.

## Resource update: sequential multi-GPU evaluation

See [MULTI_GPU.md](../MULTI_GPU.md) for the checkpointed migration and current launcher. The order remains A → B → C; each active arm uses one training slice and two validation/transfer replicas. Training and the scientific protocol are unchanged. The new sequential supervisor replaces the original individual launchers after migration.
