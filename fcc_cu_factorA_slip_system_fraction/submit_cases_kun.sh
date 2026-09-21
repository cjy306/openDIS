#!/bin/bash
#SBATCH --job-name=factorA_cases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --output=slurm-factorA-%A_%a.out
#SBATCH --error=slurm-factorA-%A_%a.err
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

work_dir=/public/home/cjy306/openDIS/fcc_cu_factorA_slip_system_fraction
cd "$work_dir"

cases=(A25 A50 A75 A100)
case_name=${cases[$SLURM_ARRAY_TASK_ID]}
seed=12345
init_dir="init_${case_name}_seed${seed}"
output_dir="output_${case_name}_seed${seed}"

echo "=== ${case_name} 开始：$(date)；作业 ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
nvidia-smi

echo ">>> [1/2] generate_${case_name}.py"
python "generate_${case_name}.py" --seed "$seed"
test -s "$init_dir/init_config.data"

echo ">>> [2/2] test_${case_name}.py"
python "test_${case_name}.py" --init "$init_dir" --out "$output_dir"
test -s "$output_dir/stress_strain_dens.dat"

echo "=== ${case_name} 成功结束：$(date) ==="
