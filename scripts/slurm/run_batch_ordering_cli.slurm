#!/bin/bash
#SBATCH -p rigelgpuQ
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH -J staging_NaLayered
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# --- Environment setup ---

module load gnu12/12.3.0
module load nvhpc-hpcx-cuda12/24.7

# Try to activate Conda (adjust path/env as needed)
if [ -f "/home/ychoi35/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/ychoi35/miniconda3/etc/profile.d/conda.sh"
    conda activate pymatgen || true
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# --- Run ---
srun python /home/ychoi35/project_NaLayered_MLIP/src/ordering_calculation_mace_cli.py \
	--staging-root /home/ychoi35/project_NaLayered_MLIP/ordering_staging \
    --output-root /home/ychoi35/project_NaLayered_MLIP/ordering_staging_output \
	--model /home/ychoi35/project_NaLayered_MLIP/data/mace-omat-0-medium.model \
	--relax-cell
