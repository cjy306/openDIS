#!/bin/bash
#SBATCH --job-name=orowan_geo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=xhacnormalb
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00

# 雄衡环境：与 breakaway/submit.sh 保持一致。
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module unload compiler/devtoolset/7.3.1 2>/dev/null
module load compiler/gcc/12.2.0
export PYTHONPATH=$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=spread OMP_PLACES=threads

set -e
cd $HOME/openDIS/orowan_test

echo '>>> [1/4] 生成FR源与硬球初始构型'
python generate_orowan.py

echo '>>> [2/4] 运行Orowan几何模型'
python test_orowan.py

echo '>>> [3/4] 绘制应力与位错密度曲线'
python plot_orowan.py

echo '>>> [4/4] 转换网络与障碍物VTK'
python vtk_orowan.py

test -s post_orowan/orowan_response.png
test -s post_orowan/orowan_summary.txt
test -s vtk_orowan/config.0.vtk
test -s vtk_orowan/obstacles.vtk

echo 'Orowan四步流程全部完成'
