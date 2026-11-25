#!/bin/bash -l
# sbatch -p gpu_4090 --gpus=1 ./submit_pt.sh
#SBATCH --job-name=strain_pt
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

ROOT=/data/run01/scw6gd9/openDIS
cd "$ROOT/examples/10_strain_hardening"

module load gcc/12.2
module load cuda/12.1
module load cmake/3.30.0
module load miniforge/24.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/run01/scw6gd9/conda-envs/opendis-gpu

export PYTHONPATH="$ROOT/core/exadis/python:$ROOT/core/pydis/python:$ROOT/lib:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ROOT/build/core/exadis/kokkos/core/src:$ROOT/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Verify inputs and GPU allocation before starting the simulation.
test -r 180chains_16.10e.data
test -r test_strain_hardening_exadis.py
nvidia-smi

TIMEFORMAT='simulation_wall_seconds=%3R'
time python -u test_strain_hardening_exadis.py
