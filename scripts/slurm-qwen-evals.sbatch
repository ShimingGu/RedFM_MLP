#!/usr/bin/env bash
#SBATCH --job-name=qwen-photoz-evals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Submit with cluster-specific account/partition options, for example:
# sbatch --account=... --partition=... scripts/slurm-qwen-evals.sbatch
export AION_EVAL_WORKERS="${AION_EVAL_WORKERS:-4}"
export AION_MAX_ROWS="${AION_MAX_ROWS:-20000}"
export AION_EPOCHS="${AION_EPOCHS:-3}"

bash "$SCRIPT_DIR/run-qwen-evals-multigpu.sh"
