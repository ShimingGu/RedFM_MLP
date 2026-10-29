# CLAUDS Qwen3.5-0.8B evaluation timeout recovery

Both QLoRA and DoRA completed all 37,500 optimizer updates. Their final periodic checkpoints retain epoch 9, next batch 60,000, all optimizer/scheduler/RNG state, and nine completed source-validation epochs. The tenth source-validation pass is repeated after restoring the final training state, preserving the original checkpoint-selection rule. No additional optimizer updates are needed.

The failure was a fixed 3,600-second deadline on an entire evaluation request. The test cohort has 225,000 objects. Previous 15,000-object validation requests took approximately 392 seconds for QLoRA and 420 seconds for DoRA, implying roughly 98 and 105 minutes for the test request before overhead. Both runs therefore reached the same predictable timeout despite healthy inference workers.

The transport now records actual completed batches without changing batch contents, grouping, ordering or RNG consumption. Each incomplete worker/request gets a one-hour inactivity deadline. Only an increase in its completed-row count for the matching request renews that deadline; stale request files, repeated heartbeats, and progress by a different worker cannot hide a stalled shard. Worker exit still fails immediately. Progress is published in the worker session's `progress.json`, with completed rows, total rows, elapsed time and the request identity.

Verification: **16 tests passed**, covering long progressing requests, stalled workers, stale request identifiers, independently stalled shards, dead workers, batch preservation, reassembly and RNG equivalence. Both actual checkpoints also passed a **32-object local-versus-remote replay with exactly zero logit and PDF error**.

Both runs were restarted as detached processes at 2026-10-29 12:31 UTC from their own update-37,500 checkpoint. QLoRA parent PID: 128026; DoRA parent PID: 128044. Original checkpoints are preserved. The four established CLAUDS training/evaluation slices are reused; the three bidirectional seed-42 training slices remain assigned to their existing jobs. The DoRA training UUID typo in `notebooks/recent_runs.txt` was corrected.

Expected remaining work is source-validation replay and the full test inference, about two hours at the previous measured rates. Results are not claimed complete at relaunch. Successful completion saves each model's `result.pt`, adapter, prediction head and `{method}_run.json` summary in its original output directory.

Runtime audit and backups:
`/arc/projects/ots/Cosmic_Imprint_of_Time/clauds_evaluation_recovery/20261029T123116Z`

Checkpoint hashes, exact launch commands, replay proofs, original evaluation timing records and source hashes are recorded there and in [recovery.json](/scratch/RedFM_MLP/reports/clauds-timeout-20261029/recovery.json).

Logs:

- QLoRA: `/arc/home/gsm/aion_output/figures/qwen3.5-0.8b-clauds-qlora-e10/multigpu_logs/qlora.log`
- DoRA: `/arc/home/gsm/aion_output/figures/qwen3.5-0.8b-clauds-dora-e10/dora.log`
