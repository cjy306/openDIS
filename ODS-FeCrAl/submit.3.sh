#!/bin/bash
#SBATCH --job-name=opendis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=xhacnormalb        # CPU: xhhctdnormal / xhacnormala(64) / xhacnormalb(128)
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00

# ============ 只改这两行 ============
PYFILE=/work/home/cjy306/openDIS/ODS-FeCrAl/test_oxide_verify.py
PYARGS="--tau 10  --steps 5000 --out output_ox_tau10"      
# ====================================

# --- 雄衡环境(验证过,勿动) ---
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module unload compiler/devtoolset/7.3.1 2>/dev/null
module load compiler/gcc/12.2.0
export PYTHONPATH=$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=spread OMP_PLACES=threads

cd "$(dirname "$PYFILE")"
python "$PYFILE" $PYARGS