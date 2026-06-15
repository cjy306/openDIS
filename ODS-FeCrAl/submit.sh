#!/bin/bash
#SBATCH --job-name=opendis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ========== 每次只改这里 ==========
#SBATCH --partition=ksagnormal01      # GPU用ksagnormal01；纯CPU用kshcnormal
##SBATCH --gres=gpu:1                  # 用GPU填gpu:1(几张改数字)；纯CPU把这行删掉或注释
#SBATCH --cpus-per-task=48             # 用几核
#SBATCH --time=2:00:00                # 跑多久 时:分:秒

PYFILE=/public/home/cjy306/openDIS/HomeWork/test_Cu_twin.py   # 要跑的脚本完整路径
# ==================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0

cd $(dirname "$PYFILE")
python "$PYFILE"