#!/bin/bash
#SBATCH --job-name=opendis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# ===== 每次改这里：队列/核数(不超单节点)/时长 =====
#SBATCH --partition=ksagnormal01     # GPU:ksagnormal01(昆山默认GPU版); 纯CPU:kshcnormal(32)
#SBATCH --gres=gpu:1                 # GPU版保留;纯CPU注释掉此行
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

PYFILE=/public/home/cjy306/openDIS/HomeWork/你的脚本.py   # 注意 /public/home
# ===================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0
export PYTHONPATH=$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:$LD_LIBRARY_PATH

cd $(dirname "$PYFILE")
python "$PYFILE"