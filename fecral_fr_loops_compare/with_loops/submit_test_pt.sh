#!/bin/bash -l
# sbatch -p gpu_4090 --gpus=1 ./submit_test_pt.sh
#SBATCH --job-name=fr_with_loops_pt
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

export OPENDIS_ROOT=/data/run01/scw6gd9/openDIS
CASE_DIR="$OPENDIS_ROOT/fecral_fr_loops_compare/with_loops"
OUT=output_pm2000_seg40_10
RESTART=11100

cd "$CASE_DIR"
# Fail rather than start from scratch if the checkpoint is missing.
test -r test.py || { echo "ERROR: test.py not found" >&2; exit 1; }
test -s "$OUT/restart.$RESTART.exadis" || { echo "ERROR: missing or empty checkpoint $OUT/restart.$RESTART.exadis" >&2; exit 1; }

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
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "[restart] case=$CASE_DIR step=$RESTART job=${SLURM_JOB_ID:-unknown} node=$(hostname)"
nvidia-smi
TIMEFORMAT='simulation_wall_seconds=%3R'
time python -X faulthandler -u test.py --out "$OUT" --restart "$RESTART"
