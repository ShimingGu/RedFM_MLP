#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point. AION RLVR now always continues the
# completed encoder-level QLoRA policy; it never trains a residual vector arm.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/aion-qlora_rlvr_posttraining.sh" "$@"
