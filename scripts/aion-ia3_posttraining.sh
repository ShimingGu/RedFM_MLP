#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
AION_EMBEDDING_METHOD=ia3 exec "$SCRIPT_DIR/aion-embedding_method_posttraining.sh" "$@"
