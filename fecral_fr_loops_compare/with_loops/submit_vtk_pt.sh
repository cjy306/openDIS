#!/bin/bash -l
# sbatch -p gpu_4090 --gpus=1 ./submit_vtk_pt.sh
#SBATCH --job-name=vtk_with_loops_pt
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

export OPENDIS_ROOT=/data/run01/scw6gd9/openDIS
cd "$OPENDIS_ROOT/fecral_fr_loops_compare/with_loops"

module load gcc/12.2
module load cuda/12.1
module load cmake/3.30.0
module load miniforge/24.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/run01/scw6gd9/conda-envs/opendis-gpu

export PYTHONPATH="$OPENDIS_ROOT/core/exadis/python:$OPENDIS_ROOT/core/pydis/python:$OPENDIS_ROOT/lib:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$OPENDIS_ROOT/build/core/exadis/kokkos/core/src:$OPENDIS_ROOT/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export PYTHONDONTWRITEBYTECODE=1

converted=0
for folder in output_pm2000_seg40_10_relax output_pm2000_seg40_10; do
    if [ -d "$folder" ]; then
        python -X faulthandler -u vtk.py --input "$folder"
        converted=1
    fi
done
if [ "$converted" -eq 0 ]; then
    echo "No simulation output in $PWD" >&2
    exit 1
fi
