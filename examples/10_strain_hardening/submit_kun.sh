#!/bin/bash
#SBATCH --partition=ksagnormal01
#SBATCH --job-name=strain_kun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the case directory}"
root=$(cd ../.. && pwd -P)
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0
export PYTHONPATH="$root/core/exadis/python:$root/core/pydis/python:$root/lib:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$root/build/core/exadis/kokkos/core/src:$root/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_PROC_BIND=spread OMP_PLACES=threads
test -r 180chains_16.10e.data
TIMEFORMAT='simulation_wall_seconds=%3R'
time python test_strain_hardening_exadis.py
