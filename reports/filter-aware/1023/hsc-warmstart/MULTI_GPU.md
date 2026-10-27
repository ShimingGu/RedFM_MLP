# Sequential A → B → C with additional evaluation GPUs

Resource change requested on 2026-10-23 after the two-slice Qwen3.5-0.8B QLoRA run finished. Both released slices were checked to be idle and available under the existing shared advisory locks; other jobs' reservations remain untouched.

Each experiment uses the same three slices:

| Purpose | MIG slice |
|---|---|
| Training and optimizer | `MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25` |
| Validation/transfer replica 1 | `MIG-bef01423-2506-5edb-a289-80c60b1f0297` |
| Validation/transfer replica 2 | `MIG-412ff209-a29c-5bc8-9aa2-fd1a41474aec` |

**The experiment order is strictly A → B → C.** The supervisor waits for the current arm to exit successfully and write its completed summary before starting the next arm. An arm failure stops the pipeline. The two evaluation replicas belong to the active arm; B/C never run alongside A, and C never runs alongside B.

Training stays on its original slice. This accelerates validation and transfer inference; optimizer updates remain on one GPU. No data-parallel optimizer, changed microbatching, altered accumulation, learning-rate change, architecture change or MIG peer access is introduced. The effective training batch remains 16, with the original optimizer/RNG/batch-position resume state. B/C still restore the original archived trained adapter and PDF head independently.

The inference replicas retain the original batch-of-eight grouping, including its padding and last partial batch, and reassemble original query order before computing global metrics. Every evaluation synchronizes the current adapter/head tensors from the training model, including after selected-model weights are frozen for target inference. The replica models have no optimizers and never train. CPU checkpoint files carry the small trainable state between slices; this avoids requiring CUDA peer access or NCCL support between MIG devices.

## Safe migration of the existing run

A reached update 250 and began its serial source-validation pass. The durable migration process waits for the completed update-250 checkpoint, including that validation and any best-model update. It then:

1. Checks that the original B/C processes are still queued and that the two extra slices can be exclusively reserved while idle.
2. Pauses A briefly and archives its complete checkpoint and original job metadata.
3. Replaces the old queued launchers, stops the old A process, waits for the training slice to drain, and starts the new sequential supervisor.
4. Resumes A from that checkpoint with optimizer, RNG and next-batch position restored. Any microbatches computed after the saved checkpoint are replayed from its deterministic state; completed checkpoint progress is retained.

`resource-migration.json` records whether this is still pending or completed, plus the archived checkpoint hash and resumed update. `migration-job.json` and `migration.log` identify the durable migration process. No user interaction is needed at the switch.

`pipeline-job.json`, `pipeline.json` and `pipeline.log` identify the supervisor and active arm after migration. Each arm's `job.json` then points to its active worker and supervisor. Original job records are archived alongside it.

## Verification and outputs

Before migration, the two real GPU replicas replayed 32 canonical validation objects. Both the maximum logit difference and maximum PDF difference against the archived serial reference were **zero**. See `parallel-validation-probe/replay.json` and its per-worker restoration records/logs.

A second real-GPU check synchronized A's trained update-200 adapter/head state. Two-replica inference and serial batch-of-eight inference again matched with **zero logit/PDF differences** on the same 32 source-validation objects. These trained predictions differed from the original checkpoint, confirming the workers used the updated state. See `parallel-checkpoint-probe/replay.json`; these diagnostic scores were not used for checkpoint selection.

**61 relevant tests passed.** They include exact global metric reassembly, preservation of original batch grouping (including 5,683-object partial batches), rejection of changed IDs/labels/duplicate queries, synchronization of selected frozen weights, the main process's original CPU RNG draw at validation, checkpoint/RNG resume, zero-update selection, physical metadata checks, and successful/failing sequential pipeline order.

The original A/B/C scientific implementations and manifests are unchanged. The new execution layer records its own fingerprints in each arm's `execution_backend.json`. Each arm also saves:

- `evaluation_workers/`: GPU/model restoration provenance and per-replica logs.
- `evaluation_acceleration.json`: evaluation sizes, original batch grouping, elapsed time and replica peak allocated memory.
- The existing source/target prediction, selection, training and report artifacts.

Reports explicitly state that training uses one slice and evaluation uses two. Their original peak-memory field covers the training slice; replica peaks are recorded separately. A's total wall time includes the earlier single-GPU portion and its migration; it is not a uniform three-GPU timing benchmark against B/C.

## Commands

Migration was launched using:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_parallel.py --stage launch-migration --min-update 250
```

After migration, use the **sequential pipeline**, rather than the old per-arm launchers, for any restart. Once its previous supervisor and children are no longer active:

```bash
.pixi/envs/default/bin/python notebooks/qwen_filter_parallel.py --stage launch
```

This skips completed arms and automatically uses `--resume` where `latest.pt` exists. It refuses to launch while an old per-arm process is still alive. Device assignments above are the command defaults; the exact launched commands are preserved in job JSON files.
