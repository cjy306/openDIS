#!/bin/bash
# Convert existing 3 um FR snapshots only; no generation or simulation.
#SBATCH --job-name=vtk_fr3um_12345
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-vtk-fr3um-%j.out
#SBATCH --error=slurm-vtk-fr3um-%j.err
#SBATCH --time=120:00:00
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

cd /public/home/cjy306/openDIS/stress_overshoot_3um_seed12345
python vtk.py
