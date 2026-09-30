#!/bin/bash
# Convert saved configurations only.
#SBATCH --job-name=vtk_run_02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-vtk-run_02-%j.out
#SBATCH --error=slurm-vtk-run_02-%j.err
#SBATCH --time=120:00:00
#SBATCH --partition=ksagnormal01     # 昆山 GPU 队列
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# ============================================================================
#  环境加载
# ============================================================================
set -e

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0
export PYTHONPATH=$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

cd /public/home/cjy306/openDIS/stress_overshoot_5runs/run_02
python vtk.py
