#!/bin/bash
#SBATCH --job-name=opendis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=ksagnormal01       # 昆山 GPU 队列
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00                # 旧昆山脚本用过 48:00:00, 队列允许

# ============ 只改这两行 ============
PYFILE=/public/home/cjy306/openDIS/ODS-FeCrAl/test_caseA_baseline.py
PYARGS="--init output_relax_seed12345/config.9800.data --out output_caseA_high"
# ====================================

# --- 昆山环境(submit_kun.sh 验证过,勿动) ---
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0
export PYTHONPATH=$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=spread OMP_PLACES=threads

cd "$(dirname "$PYFILE")"
python "$PYFILE" $PYARGS