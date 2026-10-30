#!/bin/bash
#SBATCH --job-name=exp01-fi-scan
#SBATCH --account=project_465003656
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465003656/sal-torch/logs/%j-%x.out
#SBATCH --error=/scratch/project_465003656/sal-torch/logs/%j-%x.err

# exp01 - FI + plasticity scan, ROCm reproduction check.
# Budget: 1 GCD x 0.5h = 0.5 GPU-hours (walltime capped at 30 min).

set -euo pipefail

ROOT=/scratch/project_465003656/sal-torch
# SAL_VENV picks the environment: ".venv" (transformers 4.46.0, the tested env)
# or ".venv-tf47" (transformers 4.47.1, the only one that can load I-JEPA).
source "$ROOT/${SAL_VENV:-.venv}/bin/activate"
echo "venv: ${SAL_VENV:-.venv}"

export PYTHONUNBUFFERED=1
# Compute nodes have no internet: everything must come from the pre-filled cache.
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# LUMI-G gives ~7 cores per GCD; torch otherwise sees all 128 and oversubscribes.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-7}

cd "$ROOT/experiments"
echo "=== $(date -Is) | node $(hostname) | job ${SLURM_JOB_ID:-local} ==="
rocm-smi --showproductname 2>/dev/null | head -5 || true

python exp01_fi_scan.py "$@"

echo "=== done $(date -Is) ==="
