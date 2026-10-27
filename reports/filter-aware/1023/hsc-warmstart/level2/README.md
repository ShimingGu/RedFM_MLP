# Experiment C: independent Level-2 filter-physics warm start

Specification: project-root `filter_aware_1023c.md`.

C starts from the **same original archived HSC Qwen3.5-4B-Base rank-8 QLoRA adapter and trained 256-hidden-unit PDF head as A/B**, independently of their continued weights. Both components are checked against the archive at preparation and verified tensor by tensor at GPU restoration. The full 22,444 training / 5,683 validation cohort, all values/errors/masks/labels, original feature definitions and 300-bin 0–4.5 grid match A/B.

The exact existing Level-2 serializer adds only `pivot_nm` and `width_nm` to B's Level-1 strings. Every source and target record was checked: deleting those fields reproduces B byte for byte. No flux conversion, new units, numerical reformatting, learned filter embeddings, full response curves or other descriptors enter the prompts.

The common maximum sequence length remains 512. Actual tokenization audits cover all 28,127 source objects and both 256-object target sets, with zero truncations. Maximum lengths: HSC 272, Legacy 276, CLAUDS 462. Exact strings, token IDs and token strings are preserved.

## Filter-response verification

`configs/filter_registry_v1.json` and all 21 pinned original and normalized response files pass their hashes. The numerical summaries are independently reproduced from the vendored response grids in nm:

- Pivot wavelength: sqrt(∫Rλ dλ / ∫R/λ dλ).
- Rectangular equivalent bandwidth: ∫R dλ / max(R), not FWHM.

The existing relative photon-counting system response is used without another wavelength factor or SED weighting. Values are serialized to three decimals in nm. `registry_audit.json` records the definitions, reproduced integrals, original/normalized hashes, source commits, serialized values and approximations. `filter_registry_snapshot.json` preserves the complete registry used to generate inputs.

HSC r/r2 and i/i2 exposure mixtures are unavailable in the original feature cache. The existing nominal HSC r/i representative approximations remain explicit; no mixture is invented. Legacy RELEASE 9010/9012 uses representative DECam DR1 grz; RELEASE 9011 uses BASS gr and MzLS z; all use WISE W1/W2. CLAUDS uses MegaCam u/u_star (MP9302/MP9301), HSC grizy and VIRCAM YJHKs. Metadata are attached by named feature and actual RELEASE, never feature-array position. Registry entries preserve the limitations of nominal system responses versus object-specific throughputs.

## Matched training and evaluation

`matched_protocol.json` is identical to A/B: maximum two additional epochs / 2,806 updates, no head-only warmup, adapter LR 3e-6, head LR 1e-5, effective batch 16, seed 1023. Original adapter coverage, architecture, head, trainable budget, optimizer setup, constant LRs, clipping, weight decay and partial-batch normalization are unchanged. Source-validation CE selects among update zero and validations every 250 optimizer updates plus epoch ends.

Before updates, the job restores the original checkpoint, checks a 32-object canonical Level-0 replay with the original batch grouping, and evaluates the full 5,683-object source validation set with Level 2. `serializer_change.json` isolates the resulting input-format effect from subsequent adaptation. A separately checked the complete canonical Level-0 replay against the archive.

After source-only selection, C evaluates exactly the same audited 256 Legacy and 256 CLAUDS validation objects as A/B, with frozen selected weights. Legacy uses DESI spectroscopic redshifts; CLAUDS uses ZPHOT. Native CLAUDS imputed magnitudes and unavailable errors/EBV are retained. There is no target-label fitting, recalibration or target-based selection, and no held-out test evaluation.

The report separates serializer-only effects, source recovery and fixed-weight transfer. It compares source/target CE, NMAD, outlier fraction, bias, CRPS and 68% coverage against the original archived Level-0 model and selected A/B controls, with **C versus B as the critical comparison**. Learning curves, prediction plots, selected/trained update counts, wall time and peak allocated GPU memory are generated automatically.

## Durable execution and queue order

All slices were reserved at launch. C waits for B's process to finish, then acquires the same shared advisory GPU lock and requires the slice to be idle (at most 0.2 GiB used, at least 8 GiB free). C never competes with B for A's released slice. Predecessor identity includes Linux process start ticks to avoid confusing an unrelated reused PID. Waiting creates no CUDA context; the GPU lock is retained throughout C execution. Existing A/B and other jobs remain unchanged.

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_physics_warmstart.py --stage launch --wait-for-gpu --gpu MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25
```

`job.json` records C's PID, predecessor identity and exact command. The durable process runs independently of the chat and automatically performs diagnostics, SFT, source selection, target inference, plotting and reporting. `status.json` and `report.md` distinguish queued, pending, training and completed results. Waiting for B is scheduling only: B's continued weights are never loaded into C. If B exits unsuccessfully, C can proceed once the slice is safely available and will identify missing B comparisons.

After an interruption, ensure the previous C process is no longer active. Once `latest.pt` exists, restore optimizer/model/RNG/batch position with:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_physics_warmstart.py --stage launch --resume --wait-for-gpu --gpu MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25
```

Before the first restart checkpoint exists, relaunch without `--resume`; the original trained model and initial diagnostics are replayed. Completed experiments are protected from relaunch. Regenerate reports/plots without training:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_physics_warmstart.py --stage report
```

## Reproducible artifacts

- `manifest.json`, `matched_protocol.json`, `initial_model/`: matched settings, provenance, input/code fingerprints and original trained adapter/head.
- `source_cohort.npz`, `source_inputs.jsonl`, `reference_validation.pt`, `canonical_replay_inputs.jsonl`: source train/validation-only arrays, exact Level-2 inputs and canonical validation-only replay reference.
- `*_cohort.npz`, `*_inputs.jsonl`: exact audited target-validation data and Level-2 strings.
- `preparation_audit.json`, `metadata_delta_audit.json`, `registry_audit.json`, `filter_registry_snapshot.json`, `tokenizer_examples.json`: cohort identity, per-filter metadata correspondence, registry definitions/approximations and actual tokenizer inputs.
- `canonical_replay.json`, `zero_update_validation.pt`, `serializer_change.json`: generated once GPU diagnostics execute.
- `source_summary.json`, `parameter_changes.json`, `selected/`, `best_validation.pt`: selected source checkpoint, latest/selected diagnostics and adapter/head parameter changes.
- `transfer_summary.json`, `*_selected_predictions.npz`: selected fixed-weight target metrics and PDFs.
- `report.md`, `learning_curves.png/.pdf`, `serializer_predictions.png/.pdf`, `transfer_predictions.png/.pdf`: automatic scientific report and diagnostic plots.
- `job.json`, `status.json`, `gpu.json`, `timing.json`, `run.log`: durable execution, queue, reservation and computational-cost records. Queue time is excluded from GPU-run wall time; interruptions are included if resumed.
- `prepare.log`, `tests.log`, `validation-tests.xml`: preparation and verification evidence.

Implementation: `notebooks/qwen_filter_physics_warmstart.py`, reusing the existing Level-2 serializer, strict A checkpoint restoration, B's tested lock/target inference helpers, and the shared continued-SFT training/resume implementation. A/B code and manifests are unchanged.

Validation: **51 tests passed** across C/B runner, continued-SFT, serializer and original trainer tests. C-specific coverage checks that metadata are the only changes for all survey/instrument mappings, rejects incorrect numerical metadata or changed values, recomputes registry integrals, enforces predecessor order while handling PID reuse, executes the zero-update → training/selection → target inference path with a CPU simulation, and renders the completed original/A/B/C report and plots including costs. GPU restoration, zero-update metrics and training results remain pending while C is queued.

Source recovery must be assessed before claiming a clean transfer comparison. HSC-only training uses essentially one fixed filter set, so successful adaptation cannot establish a physically transferable response dependence. Single-seed target point estimates are preliminary; multiple training filter systems and shuffled/incorrect metadata controls are needed before broader claims. The report preserves these limits and does not assume metadata improve performance.

## Resource update: sequential multi-GPU evaluation

See [MULTI_GPU.md](../MULTI_GPU.md) for the checkpointed migration and current launcher. The order remains A → B → C; each active arm uses one training slice and two validation/transfer replicas. Training and the scientific protocol are unchanged. The new sequential supervisor replaces the original individual launchers after migration.
