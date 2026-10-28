#!/bin/bash
#SBATCH --job-name=CuNiSi_A_vtk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=120:00:00
#SBATCH --partition=ksagnormal01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -eo pipefail
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0

export OPENDIS_ROOT="${OPENDIS_ROOT:-$HOME/openDIS}"
export PYTHONPATH="$OPENDIS_ROOT/core/exadis/python:$OPENDIS_ROOT/core/pydis/python:$OPENDIS_ROOT/lib:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$OPENDIS_ROOT/build/core/exadis/kokkos/core/src:$OPENDIS_ROOT/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
cd "${SLURM_SUBMIT_DIR:?Submit from A or B directory}"

for folder in "${CASE_OUTPUT:-output}_relax" "${CASE_OUTPUT:-output}"; do
    python vtk.py --input "$folder" --stride "${VTK_STRIDE:-1}"
done
