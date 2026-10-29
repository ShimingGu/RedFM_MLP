# Bidirectional_1029 execution recovery

Authorisation: `Bidirectional_1029.md`. Execution began 2026-10-28 23:19 UTC.

Persistent run root:
`filter-aware-runs/1028/original-bidirectional-transfer`
(symlink resolves to `/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/original-bidirectional-transfer`).

Detached supervisor was launched as PID 105808; verify the live PID/lock/status before resuming. Read `status.json`, `supervisor.log`, `logs/*`, `cells/*/*/progress.json` and `consolidated_report.md` there. Compact report: `reports/filter-aware/1029/bidirectional-seed42/report.md`.

All four original source-validation replays passed exactly. Both HSC original models additionally reproduced 256 archived predictions exactly in both observation systems. Three original Legacy cells were running on the three designated MIG slices; fourth Legacy cell and HSC archive slicing queued. Historical cohort is the exact prepared 61,310-object P_mutual in prepared order. Cross-domain Level-0 cells are explicitly interface-stress diagnostics with actual named destination-band measurements, never positional substitutions.

The supervisor completes all eight historical cells and their report before launching exactly four fresh seed-42 models. New model outputs use the original approved package's `future_runs/{hsc,legacy}-{qlora,dora}-seed42` paths. Existing package manifest and original code remain unchanged. Prepared entrypoint loaded full P-containing files before slicing, so execution uses a versioned source-only worker; controller staged T/V-only files. Worker has an audit hook denying known P-containing artifacts and uses the same prepared configurations, fresh initialisations, saved permutations and optimiser/resume semantics. Selected and final toy weights/validation match the original loop exactly, including interrupted resume and partial windows. Ten original/prepared-plus-isolation tests passed, plus an eight-cell/1,000-shared-draw historical analysis integration test.

Do not change the frozen execution files without auditing and versioning their hashes. `manifest.json`, `policy.json`, `audit.json`, `verification.json` and `code_snapshot/` preserve provenance. Original preparation manifest must remain SHA256 `bc7ac99fda7629f1235fb1a4c547e699fa9039984e82a474ebe654d418183661`.

Resume the same gated supervisor only after confirming its lock is free:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache \
pixi run --locked python notebooks/bidirectional_1029.py supervise
```

Use a detached process with persistent stdout/stderr for long execution. It resumes verified existing inference chunks and each training run's own optimizer/scheduler/RNG checkpoint. Do not restart with a different initialisation. Do not invoke the old prepared launcher, which reads full P-containing data. First actual head-only and joint backward measurements are recorded within the authorized training budget when those phases begin.

**Hard boundaries:** no new-model locked-P inference, including subsets or selected/final comparisons; no seeds 43/44; no historical-result-based training changes. Training worker receives only 16,600 T and 4,257 V from its own observational system; exactly 10,380 updates and 166,000 presentations. Keep unrelated CLAUDS jobs untouched. GPU availability and established advisory locks are checked on every launch. Failure stops subsequent launches and retains resumable state. `complete.json` at the run root means both stages and the final source-only report finished; `stage1_complete.json` alone does not mean training finished.
