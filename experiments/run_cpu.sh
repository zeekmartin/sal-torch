#!/bin/bash
#SBATCH --account=project_465003656
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --output=/scratch/project_465003656/sal-torch/logs/%j-%x.out
#SBATCH --error=/scratch/project_465003656/sal-torch/logs/%j-%x.err

# CPU-only runner, for work that would otherwise bill GPU-hours for nothing:
# JPEG decoding, resizing, cache building. The project has 1000 CPU core-hours
# and 5000 GPU-hours, and decoding 1.28M ImageNet images does not need a GCD.
#
#   sbatch --job-name=encode-in1k --time=04:00:00 experiments/run_cpu.sh \
#          experiments/encode_dataset.py --data-cache .../data_cache
#
# Same environment contract as run_exp.sh: SAL_VENV picks the venv, the HF cache
# is offline, thread counts are pinned to the allocation.

set -euo pipefail

ROOT=/scratch/project_465003656/sal-torch
source "$ROOT/${SAL_VENV:-.venv}/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$ROOT"
echo "=== $(date -Is) | node $(hostname) | job ${SLURM_JOB_ID:-local} | cpus ${SLURM_CPUS_PER_TASK:-?} ==="
echo "venv: ${SAL_VENV:-.venv}"
echo "cmd:  python $*"

python "$@"
