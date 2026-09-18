#!/bin/bash
#SBATCH --job-name=fr1um_cases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --output=slurm-fr1um-%A_%a.out
#SBATCH --error=slurm-fr1um-%A_%a.err
#SBATCH --time=96:00:00
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

case_names=(no_obstacles point_obstacles)
generate_scripts=(generate_fr1um_no_obstacles.py generate_fr1um_point_obstacles.py)
test_scripts=(test_fr1um_no_obstacles.py test_fr1um_point_obstacles.py)
init_dirs=(init_fr1um_no_obstacles init_fr1um_point_obstacles)
output_dirs=(output_fr1um_no_obstacles output_fr1um_point_obstacles)

case_name=${case_names[$SLURM_ARRAY_TASK_ID]}
generate_script=${generate_scripts[$SLURM_ARRAY_TASK_ID]}
test_script=${test_scripts[$SLURM_ARRAY_TASK_ID]}
init_dir=${init_dirs[$SLURM_ARRAY_TASK_ID]}
output_dir=${output_dirs[$SLURM_ARRAY_TASK_ID]}

echo "=== ${case_name} 开始：$(date)；作业 ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
nvidia-smi

echo ">>> [1/2] $generate_script"
python "$generate_script"
test -s "$init_dir/init_config.data"
if [[ "$case_name" == point_obstacles ]]; then
    test -s "$init_dir/obstacles.data"
fi

echo ">>> [2/2] $test_script"
python "$test_script"
test -s "$output_dir/stress_strain_dens.dat"

echo "=== ${case_name} 成功结束：$(date) ==="
