#!/bin/bash
#SBATCH -p rigelgpuQ
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH -J TMmix_eval
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# --- Environment setup ---
module load gnu12/12.3.0
module load nvhpc-hpcx-cuda12/24.7

if [ -f "/home/ychoi35/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/ychoi35/miniconda3/etc/profile.d/conda.sh"
    conda activate pymatgen || true
fi

echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- Paths (edit if needed) ---
PROJECT_ROOT="/home/ychoi35/project_NaLayered_MLIP"
SCRIPT_PATH="${PROJECT_ROOT}/src/mace_eval_tmmix_tmstaging.py"
ORDERING_ROOT="${PROJECT_ROOT}/ordering_doubleTM"
MODEL_PATH="${PROJECT_ROOT}/data/mace-omat-0-medium.model"

cd "${PROJECT_ROOT}"

# --- Run ---
# Optional quick check only (uncomment):
# srun python "${SCRIPT_PATH}" \
#   --root "${ORDERING_ROOT}" \
#   --dry-run

srun python "${SCRIPT_PATH}" \
  --root "${ORDERING_ROOT}" \
  --model "${MODEL_PATH}" \
  --device auto \
  --overwrite
