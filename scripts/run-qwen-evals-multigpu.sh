#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then PYTHON_CMD=("$PYTHON_BIN")
elif command -v pixi >/dev/null 2>&1; then
    PYTHON_CMD=(pixi run --manifest-path "$REPO_ROOT/pixi.toml" python)
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PYTHON_CMD=("$REPO_ROOT/.venv/bin/python")
else echo "No project Python environment found." >&2; exit 1; fi

MANIFEST="${AION_EVAL_MANIFEST:-$REPO_ROOT/configs/evals/qwen_physical_context.json}"
OUTPUT_DIR="${AION_EVAL_OUTPUT_DIR:-/arc/home/gsm/aion_output/evals/qwen_physical_context}"
if [[ -n "${AION_EVAL_WORKERS:-}" ]]; then
    WORKERS="$AION_EVAL_WORKERS"
elif [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[1-9][0-9]*$ ]]; then
    WORKERS="$SLURM_GPUS_ON_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "-1" ]]; then
    IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    WORKERS="${#VISIBLE_GPU_IDS[@]}"
else
    WORKERS=1
fi
TASK="aion_magnitude.evaluation.qwen_tasks:run_qwen_qwen_case"
RESUME_FLAGS=()
[[ "${AION_EVAL_RESUME:-1}" == 1 ]] && RESUME_FLAGS+=(--resume)
export MPLCONFIGDIR="${MPLCONFIGDIR:-${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}/xdg}"
mkdir -p -- "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "AION_EVAL_WORKERS must be a positive integer, got: $WORKERS" >&2
    exit 2
fi

cd -- "$REPO_ROOT"
"${PYTHON_CMD[@]}" -m aion_magnitude.evaluation_cli plan \
    --manifest "$MANIFEST" --worker-count "$WORKERS" --strategy auto

worker_status=0
if [[ "$WORKERS" == 1 ]]; then
    "${PYTHON_CMD[@]}" -m aion_magnitude.evaluation_cli worker \
        --manifest "$MANIFEST" --task "$TASK" --output-dir "$OUTPUT_DIR" \
        --strategy auto "${RESUME_FLAGS[@]}" || worker_status=$?
else
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
        srun --exclusive --ntasks="$WORKERS" --gpus-per-task=1 --gpu-bind=single:1 \
            "${PYTHON_CMD[@]}" -m aion_magnitude.evaluation_cli worker \
            --manifest "$MANIFEST" --task "$TASK" --output-dir "$OUTPUT_DIR" \
            --strategy auto "${RESUME_FLAGS[@]}" || worker_status=$?
    else
        "${PYTHON_CMD[@]}" -m torch.distributed.run --standalone --nproc-per-node="$WORKERS" \
            -m aion_magnitude.evaluation_cli worker \
            --manifest "$MANIFEST" --task "$TASK" --output-dir "$OUTPUT_DIR" \
            --strategy auto "${RESUME_FLAGS[@]}" || worker_status=$?
    fi
fi

collect_status=0
"${PYTHON_CMD[@]}" -m aion_magnitude.evaluation_cli collect \
    --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" --worker-count "$WORKERS" \
    --strategy auto --summary-path "$OUTPUT_DIR/summary.json" || collect_status=$?

if [[ "$worker_status" != 0 ]]; then exit "$worker_status"; fi
if [[ "$collect_status" != 0 ]]; then exit "$collect_status"; fi

if [[ "${AION_EVAL_PYDANTIC_REPORT:-1}" == 1 ]]; then
    if "${PYTHON_CMD[@]}" -c "import pydantic_evals" >/dev/null 2>&1; then
        "${PYTHON_CMD[@]}" -m aion_magnitude.evaluation.qwen_report \
            --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" \
            --worker-count "$WORKERS" --report-path "$OUTPUT_DIR/pydantic_report.json"
    else
        echo "Pydantic Evals is not installed; skipped the scientific report." >&2
    fi
fi
