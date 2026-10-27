# Experiment B: warm-start Level-1 band-aware QLoRA

Specification: project-root `filter_aware_1023b.md`.

This experiment uses the **original archived HSC Qwen3.5-4B-Base QLoRA adapter and trained PDF head**, never A's continued weights. Preparation verifies the original archive hashes and copies A's immutable initial snapshot, source train/validation arrays and archived validation reference. The complete 22,444/5,683 cohorts, values, errors, missingness, labels and 300-bin 0–4.5 grid match A. Level 1 changes only the serializer to named physical bands and survey identity; it adds no wavelength or bandwidth information.

`matched_protocol.json` is identical to A: two additional epochs maximum, zero head-only warmup, adapter LR 3e-6, head LR 1e-5, effective batch 16, seed 1023, constant learning rates and newly initialized AdamW moments. Adapter coverage, PDF head, weight decay, clipping, partial-batch normalization, validation schedule and source-CE selection are unchanged. Update zero is a candidate. No held-out test objects are copied into the prepared source data or evaluated.

## Execution and GPU queue

All slices were reserved when B was prepared. B is launched as a durable process waiting for the slice used by A:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_band_warmstart.py --stage launch --wait-for-gpu --gpu MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25
```

`job.json` records its PID and exact command. While queued, the process creates no CUDA context and respects the same per-user advisory GPU lock used by existing jobs. It starts only after acquiring that lock and observing at most 0.2 GiB used and at least 8 GiB free. The lock is retained throughout the experiment. No existing process is interrupted. Waiting for this slice does not make A's continued checkpoint an input to B.

After the slice is released, B automatically:

1. Rechecks prepared data, code and original checkpoint fingerprints; restores and verifies every adapter/head tensor.
2. Replays the first 32 canonical Level-0 validation records using the original batch grouping, checking predictions against the archived reference. A already independently verified the entire 5,683-object canonical replay.
3. Evaluates all 5,683 source-validation objects using Level 1 before any updates. `serializer_change.json` compares these predictions with the archived full Level-0 reference using identical weights and labels. A serializer change is not presumed harmful.
4. Runs matched continued SFT, saving restart checkpoints every 50 updates and at validation/epoch boundaries. Source-validation CE alone selects the checkpoint, including update zero.
5. Evaluates the selected model on the identical audited 256 Legacy and 256 CLAUDS validation queries. No target fitting, calibration or selection occurs. Legacy bands are RELEASE-specific DECam or BASS/MzLS grz plus WISE W1/W2; CLAUDS uses MegaCam u/u_star, HSC grizy and VIRCAM YJHKs. Legacy uses DESI spec-z; CLAUDS uses ZPHOT. Existing native CLAUDS imputed magnitudes and unavailable errors/EBV remain unchanged.
6. Writes source/transfer metrics, learning curves, prediction plots and a report comparing with matched selected Level-0 A when available.

The job runs independently of this chat. A queued status is not a completed zero-update diagnostic or a training result. `report.md` explicitly shows pending stages. `timing.json` excludes GPU queue wait from run wall time. If A fails and releases the slice, B can still run; its report identifies an unavailable A comparison instead of inventing one.

To resume after an interruption once `latest.pt` exists, first ensure the old B process is no longer alive, then use:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_band_warmstart.py --stage launch --resume --wait-for-gpu --gpu MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25
```

If interrupted before `latest.pt` is first saved, launch without `--resume`; the original weights and zero-update diagnostics are replayed. Completed runs are protected from relaunch. Reports can be regenerated without training:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_band_warmstart.py --stage report
```

## Artifacts and verification

- `manifest.json`, `matched_protocol.json`, `initial_model/`: reproducible configuration, fingerprints and the original trained state.
- `source_cohort.npz`, `source_inputs.jsonl`, `reference_validation.pt`: full source train/validation data, exact Level-1 strings and archived validation-only predictions.
- `preparation_audit.json`, `independent_input_audit.json`: matched source/target identities and values, 309,397 independent source value/error/EBV formatting checks, physical band assignments and no wavelength/width fields.
- `tokenizer_examples.json`: actual strings, token IDs and token strings. All 28,127 source records and both target sets are audited without truncation: maximum lengths 174 source, 173 Legacy, 243 CLAUDS, against a limit of 512.
- `canonical_replay.json`, `zero_update_validation.pt`, `serializer_change.json`: generated when GPU diagnostics run.
- `status.json`, `run.log`, `job.json`, `gpu.json`: live execution state, logs and reservation provenance.
- `source_summary.json`, `parameter_changes.json`, `selected/`: selected and last-trained source results and adapter/head changes.
- `transfer_summary.json`, `*_selected_predictions.npz`: fixed-weight target predictions and metrics.
- `report.md`, `learning_curves.png/.pdf`, `serializer_predictions.png/.pdf`, `transfer_predictions.png/.pdf`: automatically generated report and plots.

Implementation: `notebooks/qwen_filter_band_warmstart.py`, reusing strict restoration and preparation checks from `notebooks/qwen_filter_warmstart.py`, training/resume from `aion_magnitude/qwen_continued_sft.py`, and the existing registry/Level-1 serializer. A's execution code and active job are unchanged.

Validation: **42 tests passed** across the new band-aware runner tests, continued-SFT tests, serializer tests and original Qwen training tests. The new tests cover rejection of A's continued weights or changed cohorts/configurations, lock-aware waiting even when reserved VRAM is empty, refusal of occupied slices, reservation lifetime, propagation of training errors, canonical replay slicing, and the completed matched-control report/plot path. Existing tests check deterministic training resume and zero-update checkpoint selection. `tests.log` and `validation-tests.xml` contain results. No B GPU inference or learning result is claimed until its queued process reaches those stages.

Source validation is also the selection set. CE retention relative to Level 1 does not guarantee retention relative to the original Level 0, and changes in NMAD, outliers, bias, CRPS, coverage and transfer are reported separately. A single seed and two fixed 256-object target samples cannot establish statistical significance or general internal band understanding.

## Resource update: sequential multi-GPU evaluation

See [MULTI_GPU.md](../MULTI_GPU.md) for the checkpointed migration and current launcher. The order remains A → B → C; each active arm uses one training slice and two validation/transfer replicas. Training and the scientific protocol are unchanged. The new sequential supervisor replaces the original individual launchers after migration.
