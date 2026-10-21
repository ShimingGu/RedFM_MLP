#!/usr/bin/env bash
# One catalogue/method comparison: frozen control and adapter on separate GPUs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CATALOGUE=""
METHOD=""
PREPARE_ONLY=0
EXTRA_ARGS=()
ORIGINAL_ARGS=("$@")
while (( $# )); do
    case "$1" in
        --catalogue-kind) CATALOGUE="${2:?Missing catalogue kind}"; shift 2 ;;
        --method) METHOD="${2:?Missing method}"; shift 2 ;;
        --prepare-only) PREPARE_ONLY=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
case "$CATALOGUE" in desi-ls|galaxiesml-spec) ;; *) echo 'Use --catalogue-kind desi-ls|galaxiesml-spec' >&2; exit 2 ;; esac
case "$METHOD" in qlora|dora) ;; *) echo 'Use --method qlora|dora' >&2; exit 2 ;; esac
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("$PYTHON_BIN")
elif command -v pixi >/dev/null 2>&1; then
    PYTHON_CMD=(pixi run --manifest-path "${PIXI_MANIFEST:-$REPO_ROOT/pixi.toml}" python)
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_CMD=("$REPO_ROOT/.venv/bin/python")
else
    echo 'Use the project Pixi environment or set PYTHON_BIN.' >&2; exit 1
fi
OUTPUT_DIR="${SURVEY_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-$CATALOGUE-$METHOD-specz-e10}"
CHECKPOINT_DIR="${SURVEY_CHECKPOINT_DIR:-/arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/$CATALOGUE-$METHOD-specz-e10}"
COMMON_ARGS=(
    --catalogue-kind "$CATALOGUE" --method "$METHOD"
    --desi-root "${DESI_LS_ROOT:-/arc/projects/ots/Cosmic_Imprint_of_Time/DESI-LS}"
    --hsc-file "${GALAXIESMLSPEC_FILE:-/arc/projects/ots/Cosmic_Imprint_of_Time/GalaxiesMLSpec/DESI_HSC_64x64_v2.hdf5}"
    --output-dir "$OUTPUT_DIR" --checkpoint-dir "$CHECKPOINT_DIR"
    --cache-root "${SURVEY_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache/qwen-surveys}"
    --qwen-model "${QWEN_MODEL:-Qwen3.5-4B-Base}"
    --max-rows "${SURVEY_MAX_ROWS:-300000}" --seed "${SURVEY_SEED:-42}"
    --train-fraction 0.20 --val-fraction 0.05
    --z-min 0 --z-max 4.5 --n-z-bins 300
    --min-deltachi2 40 --min-valid-bands 3
    --epochs 10 --head-warmup-epochs 3
    --head-learning-rate 2e-4 --adapter-learning-rate 1e-5
    --microbatch-size 1 --gradient-accumulation-steps 16
    --rank 8 --alpha 16 --dropout 0.05
    --target-modules q_proj,k_proj,v_proj,o_proj
    --max-length 512 --embedding-batch-size 8 --eval-batch-size 8
    --checkpoint-steps 100
)
# Hold UUID-based reservations across preparation and both worker processes.
# NVML is queried on individual MIG handles, without loading torch or a model.
if (( ! PREPARE_ONLY )) && [[ "${SURVEY_GPU_GUARDED:-0}" != 1 ]]; then
    GPU_LIST="${SURVEY_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
    exec "${PYTHON_CMD[@]}" "$REPO_ROOT/scripts/qwen_survey_gpu_guard.py" \
        --devices "$GPU_LIST" \
        --lock-root "${SURVEY_GPU_LOCK_ROOT:-/scratch/.tmp-gsm/aion_output/gpu_locks/$UID}" \
        --min-free-gib "${SURVEY_MIN_FREE_GIB:-8}" \
        -- bash "$SCRIPT_DIR/qwen-survey_posttraining.sh" "${ORIGINAL_ARGS[@]}"
fi
cd -- "$REPO_ROOT"
"${PYTHON_CMD[@]}" -c 'import torch, numpy, fitsio, h5py'
if (( ! PREPARE_ONLY )); then
    GPU_LIST="${SURVEY_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
    IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
    if (( ${#GPU_IDS[@]} != 2 )) || [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
        echo 'Set SURVEY_GPU_DEVICES to two distinct CUDA IDs or MIG UUIDs (frozen,adapter).' >&2; exit 2
    fi
    env CUDA_VISIBLE_DEVICES="$GPU_LIST" "${PYTHON_CMD[@]}" -c \
        'import torch, bitsandbytes, peft, transformers; from peft import LoraConfig; assert torch.cuda.is_available() and torch.cuda.device_count() == 2, "Two visible CUDA devices are required"; assert "use_dora" in LoraConfig.__dataclass_fields__'
fi
# Preparation is CPU-only, shared by both arms, and locks the per-catalogue cache.
"${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_survey_posttraining.py" \
    --stage prepare "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
if (( PREPARE_ONLY )); then exit 0; fi
mkdir -p -- "$OUTPUT_DIR/multigpu_logs"
echo "$CATALOGUE $METHOD: frozen GPU=${GPU_IDS[0]}; adapter GPU=${GPU_IDS[1]}"
echo "Logs: $OUTPUT_DIR/multigpu_logs/{frozen,adapter}.log"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
env CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_survey_posttraining.py" \
    --stage frozen "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" > "$OUTPUT_DIR/multigpu_logs/frozen.log" 2>&1 &
pids+=("$!")
env CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_survey_posttraining.py" \
    --stage adapter "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" > "$OUTPUT_DIR/multigpu_logs/adapter.log" 2>&1 &
pids+=("$!")
active=("${pids[@]}")
while (( ${#active[@]} )); do
    next=()
    for pid in "${active[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            next+=("$pid")
        elif ! wait "$pid"; then
            echo "Worker failed; inspect $OUTPUT_DIR/multigpu_logs" >&2; exit 1
        fi
    done
    active=("${next[@]}")
    if (( ${#active[@]} )); then
        sleep "${POSTTRAIN_PROGRESS_SECONDS:-30}"
        tail -n 1 "$OUTPUT_DIR/multigpu_logs/frozen.log" "$OUTPUT_DIR/multigpu_logs/adapter.log" || true
    fi
done
pids=()
trap - EXIT INT TERM
"${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_survey_posttraining.py" \
    --stage collect "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
