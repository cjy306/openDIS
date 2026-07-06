#!/bin/bash
#SBATCH --job-name=opendis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# ===== 每次改这里：队列/核数(不超单节点)/时长 =====
#SBATCH --partition=xhacnormalb      # CPU:xhhctdnormal/xhacnormala(64) xhacnormalb(128); GPU:xhhgnormal01
##SBATCH --gres=gpu:1                 # 用GPU解开此行；纯CPU注释掉
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00

PYFILE=/work/home/cjy306/openDIS/breakaway/test_breakaway_prismatic.py
# ===================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module unload compiler/devtoolset/7.3.1 2>/dev/null
module load compiler/gcc/12.2.0
export PYTHONPATH=$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:$LD_LIBRARY_PATH

cd $(dirname "$PYFILE")
python "$PYFILE"