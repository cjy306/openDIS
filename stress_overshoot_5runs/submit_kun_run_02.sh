#!/bin/bash
# ============================================================================
#  OpenDiS 生成+模拟 串联提交脚本 —— 昆山 GPU 版
#  一次 GPU 申请内: 先生成初始位错构型, 再直接跑模拟, 省去两次抢 GPU
#  用法:  sbatch submit_kun_run_02.sh
# ============================================================================

#SBATCH --job-name=overshoot_run_02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-run_02-%j.out
#SBATCH --error=slurm-run_02-%j.err
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

# 任一步出错就停，生成成功后才运行模拟

echo "=== Job $SLURM_JOB_ID 开始 $(date) 节点 $SLURM_NODELIST ==="
nvidia-smi | head -15

# ============================================================================
#  ★★★ 第一步:生成初始位错构型 ★★★
# ============================================================================
echo ">>> [1/2] 生成初始构型..."
cd /public/home/cjy306/openDIS/stress_overshoot_5runs/run_02
python generate.py
echo ">>> 生成完成"

# ============================================================================
#  ★★★ 第二步:跑模拟 ★★★
# ============================================================================
echo ">>> [2/2] 开始模拟..."
cd /public/home/cjy306/openDIS/stress_overshoot_5runs/run_02
python test.py
echo ">>> 模拟完成"

echo "=== Job 结束 $(date) ==="