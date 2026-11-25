#!/bin/bash
# sbatch submit_verify_pk_xh_cpu.sh
#SBATCH --job-name=cr_verify_pk
#SBATCH --partition=xhacnormalb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module unload compiler/devtoolset/7.3.1 2>/dev/null || true
module load compiler/gcc/12.2.0
cd "${SLURM_SUBMIT_DIR:?Submit from cr_coherency_test}"
project_root=$(cd .. && pwd)
export PYTHONPATH="$project_root/core/exadis/python:$project_root/core/pydis/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$project_root/build/core/exadis/kokkos/core/src:$project_root/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -B -u verify_cosine_pk.py
