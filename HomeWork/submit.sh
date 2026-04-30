#!/bin/bash
#SBATCH --partition=comp
#SBATCH --job-name=load
#SBATCH --nodes=1           # Request 1 node
#SBATCH --ntasks=1          # Total number of tasks
#SBATCH --cpus-per-task=48  # Number of CPU cores per task
#SBATCH --time 24:00:00
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
# 取消注释下面这行以开启 MAX_CONN debug 输出，前缀即为文件名前缀
#export EXADIS_DEBUG_MAX_CONN=job_twin
module purge
module load miniforge/25.3.1
source activate opendis_cpu
python /data/home/dg000246b/openDIS/HomeWork/test_Cu_twin.py