#!/bin/bash
#SBATCH -J tm_ordering
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err


# --- Environment setup ---
module load gnu12/12.3.0
module load nvhpc-hpcx-cuda12/24.7
module load intel-oneapi/2022.3.1

# Try to activate Conda (adjust path/env as needed)
if [ -f "/home/ychoi35/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/ychoi35/miniconda3/etc/profile.d/conda.sh"
    conda activate pymatgen || true
fi

# --- Run ---
# Use srun so SLURM starts one Python rank per task across nodes.
# Parallel model:
#   - rank sharding across SLURM tasks (SLURM_PROCID / SLURM_NTASKS)
#   - local multiprocessing via --workers (set to cpus-per-task)
srun --ntasks="${SLURM_NTASKS}" \
  python /home/ychoi35/project_NaLayered_MLIP/src/util_tmmix_tmstaging.py \
  --tm-a Mn \
  --tm-b Co \
  --steps all \
  --max-ranked 100 \
  --top-elec 5 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --large-tm-threshold 42 \
  --timeout-sec 3600 \
  --timeout-start-elec 2 \
  --input-root /home/ychoi35/project_NaLayered_MLIP/data \
  --output-root /home/ychoi35/project_NaLayered_MLIP/ordering_doubleTM
