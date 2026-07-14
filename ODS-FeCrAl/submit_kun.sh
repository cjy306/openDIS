#!/bin/bash
# ============================================================================
#  OpenDiS 生成+模拟 串联提交脚本 —— 昆山 GPU 版
#  一次 GPU 申请内: 先生成初始位错构型, 再直接跑模拟, 省去两次抢 GPU
#  用法:  sbatch submit_gen_run.sh
# ============================================================================

#SBATCH --job-name=gen_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=ksagnormal01     # 昆山 GPU 队列
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# ============================================================================
#  环境加载
# ============================================================================
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0
export PYTHONPATH=$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

set -e   # 任一步出错就停,避免生成失败还硬跑模拟

echo "=== Job $SLURM_JOB_ID 开始 $(date) 节点 $SLURM_NODELIST ==="
nvidia-smi | head -15

# ============================================================================
#  ★★★ 第一步:生成初始位错构型(填你的生成脚本路径和参数) ★★★
# ============================================================================
echo ">>> [1/2] 生成初始构型..."
cd /public/home/cjy306/openDIS/ODS-FeCrAl
python generate_caseD_smoke.py
echo ">>> 生成完成"

# ============================================================================
#  ★★★ 第二步:跑模拟(填你的运行脚本路径和参数) ★★★
# ============================================================================
echo ">>> [2/2] 开始模拟..."
cd /public/home/cjy306/openDIS/ODS-FeCrAl
python test_caseD_smoke.py
echo ">>> 模拟完成"

echo "=== Job 结束 $(date) ==="