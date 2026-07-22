#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the requested historical spelling (mutigpu) in this launcher's name.
# Override this list to avoid GPUs already occupied by another run, for example:
# AION_QWEN_GPU_DEVICES=1,2 ./scripts/qwen-qwen_comparison_short_mutigpu.sh
DEVICE_LIST="${AION_QWEN_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
IFS=',' read -r -a GPU_IDS <<< "$DEVICE_LIST"
if (( ${#GPU_IDS[@]} < 2 )); then
    echo "Need two GPUs; set AION_QWEN_GPU_DEVICES to two comma-separated device IDs." >&2
    exit 2
fi
PHYSICAL_GPU="${GPU_IDS[0]}"
TERSE_GPU="${GPU_IDS[1]}"

OUTPUT_DIR="${AION_SHORT_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-qwen-comparison-short-n300000-random-s42}"
LOG_DIR="$OUTPUT_DIR/multigpu_logs"
mkdir -p -- "$LOG_DIR"

COMMON_ENV=(
    AION_MAX_ROWS=300000
    AION_SHORT_OUTPUT_DIR="$OUTPUT_DIR"
)

FINAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --force-recompute-qwen|--force-rebuild-photometry|--force-rebuild-tokens) ;;
        *) FINAL_ARGS+=("$arg") ;;
    esac
done

echo "Preparing the shared seeded-random 300,000-row cohort on GPU $PHYSICAL_GPU."
env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" AION_DEVICE=cuda \
    AION_OUTPUT_DIR="$OUTPUT_DIR/prepare" \
    "$SCRIPT_DIR/qwen-qwen_comparison.sh" \
    --precompute-qwen-branch prepare "$@" \
    >"$LOG_DIR/prepare.log" 2>&1

echo "Extracting physical embeddings on GPU $PHYSICAL_GPU and terse embeddings on GPU $TERSE_GPU."
echo "Worker logs: $LOG_DIR/physical.log and $LOG_DIR/terse.log"
env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" AION_DEVICE=cuda \
    AION_OUTPUT_DIR="$OUTPUT_DIR/precompute_physical" \
    "$SCRIPT_DIR/qwen-qwen_comparison.sh" \
    --precompute-qwen-branch physical "$@" \
    >"$LOG_DIR/physical.log" 2>&1 &
physical_pid=$!

env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="$TERSE_GPU" AION_DEVICE=cuda \
    AION_OUTPUT_DIR="$OUTPUT_DIR/precompute_terse" \
    "$SCRIPT_DIR/qwen-qwen_comparison.sh" \
    --precompute-qwen-branch terse "$@" \
    >"$LOG_DIR/terse.log" 2>&1 &
terse_pid=$!

cleanup_workers() {
    kill "$physical_pid" "$terse_pid" 2>/dev/null || true
}
trap cleanup_workers INT TERM

physical_status=0
terse_status=0
wait "$physical_pid" || physical_status=$?
wait "$terse_pid" || terse_status=$?
trap - INT TERM
if (( physical_status != 0 || terse_status != 0 )); then
    echo "Parallel extraction failed (physical=$physical_status, terse=$terse_status)." >&2
    echo "Inspect $LOG_DIR/physical.log and $LOG_DIR/terse.log." >&2
    exit 1
fi

echo "Both Qwen caches are ready; training paired photo-z heads on GPU $PHYSICAL_GPU."
exec env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" AION_DEVICE=cuda \
    AION_FORCE_RECOMPUTE_EMBEDDINGS=0 \
    AION_OUTPUT_DIR="$OUTPUT_DIR" \
    "$SCRIPT_DIR/qwen-qwen_comparison.sh" "${FINAL_ARGS[@]}"
