#!/bin/bash
#SBATCH --partition=comp
#SBATCH --job-name=load
#SBATCH --nodes=1           # Request 1 node
#SBATCH --ntasks=1          # Total number of tasks
#SBATCH --cpus-per-task=48  # Number of CPU cores per task
#SBATCH --time 48:00:00
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
module purge
module load miniforge/25.3.1
source activate opendis_cpu
python /data/home/dg000246b/openDIS/HomeWork/test_Cu.py --restart 1303