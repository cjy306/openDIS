#!/bin/bash
#SBATCH --job-name=fr3um_init_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-init-check-%j.out
#SBATCH --error=slurm-init-check-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=ksagnormal01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

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
cd /public/home/cjy306/openDIS/stress_overshoot_3um_4runs

date -Is
hostname
printf 'Job=%s Nodes=%s CUDA_VISIBLE_DEVICES=%s OMP_NUM_THREADS=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "${CUDA_VISIBLE_DEVICES-unset}" "$OMP_NUM_THREADS"
module list 2>&1
nvidia-smi || true
# Preserve the exit status of the initialization check, even after diagnostics.
set +e
python -u -X faulthandler diagnose_initialize.py
init_status=$?
printf 'Initialization process exit code: %s\n' "$init_status"
# Compare assigned nodes for the two failed jobs, when accounting is available.
sacct -j 123962611,124031768 --format=JobID,State,ExitCode,NodeList,MaxRSS
exit "$init_status"
