#!/bin/bash
# sbatch submit_verify_stress_xh_cpu.sh
#SBATCH --job-name=cr_verify_stress
#SBATCH --partition=xhacnormalb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
cd "${SLURM_SUBMIT_DIR:?Submit from cr_coherency_test}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -B -u verify_cosine_stress.py
