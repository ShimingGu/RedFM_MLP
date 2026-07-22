# Multi-GPU physical-context evaluation

This is the cluster handoff for `aion-magnitude` v0.5.0. The implementation
supports one GPU and several GPUs through the same launcher. Today, several
GPUs run independent physical-context cases in parallel. The execution API
also reserves a case-sharding contract for a future implementation that gives
several GPUs to one case.

## What the experiment tests

The manifest `configs/evals/qwen_physical_context.json` fixes Qwen pooling to
`last` and normalization to `false`. Its four independent cases vary only the
physical prompt:

1. `physical_global`: one global explanation of magnitudes and colours;
2. `physical_compact`: global context plus concise wavelength/region per band;
3. `physical_full`: the existing full instrument/passband/note descriptions;
4. `physical_full_summary`: full context plus a final representation marker.

Every case also trains an independently cached terse-Qwen baseline on the same
data. This is deliberate: it avoids cross-process cache writes and makes every
case self-contained. It costs extra Qwen extraction time.

Each successful case records three kinds of evidence:

- representation health: finite fraction, row norms, feature variance,
  effective/numerical rank, and consecutive-row cosine distance;
- post-training health: finite history, initial/final loss, loss reduction,
  and whether training loss decreased;
- photo-z comparison: physical, terse, and physical-minus-terse deltas for
  NMAD, outlier fraction, cross entropy, CRPS, bias, and calibration metrics.

Pydantic Evals organizes and reports those measurements. It cannot by itself
fix a collapsed representation; the report tells us whether the break first
appears in the frozen Qwen embedding, downstream optimization, or only the
photo-z metrics.

## Cluster setup

After pulling this directory on the cluster, install v0.5.0 in the environment
that already contains the project CUDA/PyTorch stack. For an existing virtual
environment, point the launcher at that exact Python:

```bash
cd /path/to/aion_tutorial/version_control/v0.5.0
/path/to/environment/bin/python -m pip install -e '.[qwen-cuda,evals]'
export PYTHON_BIN=/path/to/environment/bin/python
```

If the cluster uses this repository's Pixi environment, its CUDA/Qwen packages
are already declared. Install the project and the optional evaluator into that
environment; the launcher detects Pixi automatically:

```bash
pixi install
pixi run python -m pip install -e '.[evals]'
```

Set paths for the cluster. The defaults match the original ARC layout, so only
export values that differ:

```bash
export AION_CATALOGUE=/path/to/COSMOS-HSCpipe-Phosphoros.fits
export AION_MORPHOLOGY_DIR=/path/to/tilesv5
export AION_CACHE_ROOT=/scratch/$USER/aion_output/cache
export AION_EVAL_OUTPUT_DIR=/scratch/$USER/aion_output/evals/qwen_physical_context
export QWEN_MODEL=/path/to/Qwen3-8B-Base
```

The shared photometry/morphology cache must exist before several workers start.
If it does not, prebuild it with the existing single-process pipeline or run a
small one-worker smoke experiment first. The per-case Qwen caches are already
isolated and are safe to build concurrently.

## One GPU

The same launcher falls back to one worker and runs all four cases sequentially:

```bash
export AION_EVAL_WORKERS=1
export AION_MAX_ROWS=20000
export AION_EPOCHS=3
bash scripts/run-qwen-evals-multigpu.sh
```

For a short plumbing test, reduce `AION_MAX_ROWS` and `AION_EPOCHS`. Such a run
is not a scientifically meaningful comparison.

## Several GPUs

For an interactive allocation with four visible GPUs:

```bash
export AION_EVAL_WORKERS=4
export AION_MAX_ROWS=20000
export AION_EPOCHS=3
bash scripts/run-qwen-evals-multigpu.sh
```

Inside Slurm the launcher uses `srun` with one task and one visible GPU per
worker. Outside Slurm it uses `torchrun`. Worker 0 through worker 3 each receive
one physical-context case. More workers than cases are left idle; fewer workers
process their assigned cases sequentially.

For a batch submission, adjust account, partition, memory, and time for the
target cluster, then run:

```bash
sbatch --account=ACCOUNT --partition=PARTITION scripts/slurm-qwen-evals.sbatch
```

Inspect the assignment before spending GPU time:

```bash
python -m aion_magnitude.evaluation_cli plan \
  --manifest configs/evals/qwen_physical_context.json \
  --worker-count 4 --strategy auto
```

## Outputs and resume

The output directory contains:

```text
qwen_physical_context/
├── 000_physical_global.json
├── 000_physical_global/worker.log
├── 001_physical_compact.json
├── 002_physical_full.json
├── 003_physical_full_summary.json
├── summary.json
└── pydantic_report.json
```

Each case subdirectory also contains the original plots, checkpoints, and
`qwen_mlp_full_results.pt`. The launcher resumes valid successful artifacts by
default. Set `AION_EVAL_RESUME=0` to rerun cases, or
`AION_FORCE_RECOMPUTE_EMBEDDINGS=1` to rebuild their embeddings. Resume checks
the manifest metadata and case inputs, so changing a physical mode invalidates
the old JSON artifact.

Set `AION_EVAL_PYDANTIC_REPORT=0` to skip the final Pydantic report. If the
optional dependency is absent, GPU cases and `summary.json` still complete and
the launcher prints a skip message.

## Future: several GPUs for one case

The generic execution layer already provides `case_rank`, `case_world_size`,
and `context.shard_bounds(n_rows)`, and accepts
`--supports-case-sharding --gpus-per-case N`. The current Qwen task explicitly
rejects that mode because it does not yet merge embedding shards.

Implementing it later requires the case task to:

1. shard catalogue rows deterministically by stable `object_id` order;
2. write one physical and terse embedding cache per `case_rank`;
3. synchronize after extraction;
4. merge shards on rank 0 and verify every `object_id` occurs exactly once;
5. either train only on rank 0 or add a separately specified distributed
   training backend;
6. let only rank 0 emit the merged scientific result.

Do not simply enable the flag for `run_qwen_qwen_case`: until that merge exists,
it raises `NotImplementedError` instead of silently training on partial data.

This local machine cannot validate the real scheduler, CUDA visibility, model
checkpoint, or cluster filesystem. The first cluster job should therefore use
a small smoke configuration and verify all four `worker.log` files before the
full run.
