#!/bin/bash
#SBATCH --job-name=factorA_post
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-factorA-post-%j.out
#SBATCH --error=slurm-factorA-post-%j.err
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

work_dir=/public/home/cjy306/openDIS/fcc_cu_factorA_slip_system_fraction
cd "$work_dir"

seed=12345
cases=(A25 A50 A75 A100)

echo "=== 因素A后处理开始：$(date)；作业 $SLURM_JOB_ID ==="
for case_name in "${cases[@]}"; do
    test -s "output_${case_name}_seed${seed}/stress_strain_dens.dat"
done

echo ">>> [1/2] 绘制四组对比图"
python plot_factor_a.py
test -s "post_factor_a_seed${seed}/factor_a_compare_seed${seed}.png"

echo ">>> [2/2] 转换四组VTK"
python vtk_factor_a.py
for case_name in "${cases[@]}"; do
    if ! compgen -G "vtk_${case_name}_seed${seed}/config.*.vtk" > /dev/null; then
        echo "未找到 ${case_name} 的VTK输出" >&2
        exit 1
    fi
done

echo "=== 因素A后处理成功结束：$(date) ==="
