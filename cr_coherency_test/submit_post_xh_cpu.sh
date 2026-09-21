#!/bin/bash
#SBATCH --job-name=cr_coh_post
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-coherency-post-%j.out
#SBATCH --error=slurm-coherency-post-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=xhacnormalb
#SBATCH --cpus-per-task=128

set -eo pipefail

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module unload compiler/devtoolset/7.3.1 2>/dev/null || true
module load compiler/gcc/12.2.0

set -u

project_root=/work/home/cjy306/openDIS
case_dir=$project_root/cr_coherency_test

export PYTHONPATH="$project_root/core/exadis/python:$project_root/core/pydis/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$project_root/build/core/exadis/kokkos/core/src:$project_root/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

cd "$case_dir"

echo "=== Cr相干应力后处理开始：$(date)；作业 $SLURM_JOB_ID ==="
test -s output_no_coherency/stress_strain_dens.dat
test -s output_with_coherency/stress_strain_dens.dat

echo ">>> [1/2] 绘制两组对比图"
python plot_coherency_compare.py
test -s post_coherency/coherency_compare.png
test -s post_coherency/coherency_compare_summary.txt

echo ">>> [2/2] 转换两组 VTK"
python vtk_coherency_compare.py
for vtk_dir in vtk_no_coherency vtk_with_coherency; do
    if ! compgen -G "$vtk_dir/config.*.vtk" >/dev/null; then
        echo "未找到 $vtk_dir 的 VTK 输出" >&2
        exit 1
    fi
done

echo "=== Cr相干应力后处理成功结束：$(date) ==="
