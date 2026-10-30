#!/bin/bash
#SBATCH --account=project_465003656
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --output=/scratch/project_465003656/sal-torch/logs/%j-%x.out
#SBATCH --error=/scratch/project_465003656/sal-torch/logs/%j-%x.err

# Generic single-GCD runner. The script and its arguments come from the
# command line, so submit as:
#
#   sbatch --job-name=exp02 --time=01:00:00 experiments/run_exp.sh \
#          scripts/run_jepa_sal.py --smoke --control --model facebook/dinov2-base
#
# SAL_VENV selects the environment: ".venv" (transformers 4.46.0, the tested
# env) or ".venv-tf47" (transformers 4.47.1, the only one that loads I-JEPA).
# Pass it through with: sbatch --export=ALL,SAL_VENV=.venv-tf47 ...

set -euo pipefail

ROOT=/scratch/project_465003656/sal-torch
source "$ROOT/${SAL_VENV:-.venv}/bin/activate"

export PYTHONUNBUFFERED=1
# Compute nodes have no internet: everything must come from the pre-filled cache.
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# LUMI-G gives ~7 cores per GCD; torch otherwise sees all 128 and oversubscribes
# (cost a 7.5x slowdown on the RunPod A100 run before it was pinned).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}

# MIOpen keeps a SQLite kernel database, by default under $HOME/.config/miopen,
# which on LUMI is Lustre. SQLite on Lustre is unreliable, and many jobs writing
# it at once corrupts it: a batch of 28 parallel jobs was followed by
# "RuntimeError: miopenStatusInternalError" on the first conv2d of a run whose
# code had already worked 28 times. Give every job its own database on the
# node-local disk instead. The cost is that kernel compilation (~60s) is paid
# per job rather than cached across them, which is the right trade for jobs
# measured in tens of minutes.
export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID:-local}-$$"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT

cd "$ROOT"
echo "=== $(date -Is) | node $(hostname) | job ${SLURM_JOB_ID:-local} ==="
echo "venv: ${SAL_VENV:-.venv}"
echo "cmd:  python $*"
python -c "import torch, transformers, sal; print('torch', torch.__version__, '| transformers', transformers.__version__, '| sal', sal.__version__, '|', torch.cuda.get_device_name(0))"

python "$@"

echo "=== done $(date -Is) ==="
