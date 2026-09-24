#!/bin/bash
#SBATCH --job-name=fr1um_post
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-fr1um-post-%j.out
#SBATCH --error=slurm-fr1um-post-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=ksagnormal01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0

export PYTHONPATH="$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

work_dir=/public/home/cjy306/openDIS/breakaway
cd "$work_dir"

echo "=== FR 1 um 后处理开始：$(date)；作业 $SLURM_JOB_ID ==="
test -s init_fr1um_no_obstacles/init_config.data
test -s init_fr1um_point_obstacles/init_config.data
test -s output_fr1um_no_obstacles/stress_strain_dens.dat
test -s output_fr1um_point_obstacles/stress_strain_dens.dat

cmp init_fr1um_no_obstacles/init_config.data \
    init_fr1um_point_obstacles/init_config.data
echo "两组初始位错网络完全一致"

echo ">>> 绘制应力/位错密度对比图"
python plot_fr1um_compare.py
test -s post_fr1um_compare/fr1um_compare.png
test -s post_fr1um_compare/fr1um_compare_summary.txt

echo "=== FR 1 um 后处理成功结束：$(date) ==="
