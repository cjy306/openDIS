#!/bin/bash
# ============================================================================
#  OpenDiS 续跑脚本 —— 昆山 GPU 版(从指定步数接着跑,不重新生成)
#  用法:  sbatch submit_continue.sh
# ============================================================================
#SBATCH --job-name=continue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=ksagnormal01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# --- 环境 ---
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0
export PYTHONPATH=$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

set -e
echo "=== Job $SLURM_JOB_ID 开始 $(date) 节点 $SLURM_NODELIST ==="
nvidia-smi | head -15

# --- 直接续跑,不生成初始构型 ---
echo ">>> 从第 5000 步续跑..."
cd /public/home/cjy306/openDIS/ODS-FeCrAl
python test_caseA_baseline.py --restart 300700

echo "=== Job 结束 $(date) ==="