#!/bin/bash
#SBATCH --job-name=fr1um_cmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=ksagnormal01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0
export PYTHONPATH=$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=spread OMP_PLACES=threads

set -e
BASE_DIR=/public/home/cjy306/openDIS/breakaway
cd $BASE_DIR

echo '>>> [1/7] 生成无障碍初始网络'
python generate_fr1um_no_obstacles.py

echo '>>> [2/7] 生成点障碍初始网络'
python generate_fr1um_point_obstacles.py

cmp init_fr1um_no_obstacles/init_config.data \
    init_fr1um_point_obstacles/init_config.data
echo '两组初始位错网络完全一致'

echo '>>> [3/7] 运行无障碍组'
python test_fr1um_no_obstacles.py

echo '>>> [4/7] 运行点障碍组'
python test_fr1um_point_obstacles.py

echo '>>> [5/7] 绘制对比曲线'
python plot_fr1um_compare.py

echo '>>> [6/7] 转换两组VTK'
python visualize_fr1um_compare.py

echo '>>> [7/7] 检查结果文件'
test -s post_fr1um_compare/fr1um_compare.png
test -s post_fr1um_compare/fr1um_compare_summary.txt
test -s vtk_fr1um_compare/point_obstacles/obstacles.vtk

echo '全部流程完成'
