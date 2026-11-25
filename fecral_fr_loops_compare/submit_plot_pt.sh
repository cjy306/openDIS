#!/bin/bash -l
# sbatch -p cpusx ./submit_plot_pt.sh
#SBATCH --job-name=plot_loops_pt
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

export OPENDIS_ROOT=/data/run01/scw6gd9/openDIS
cd "$OPENDIS_ROOT/fecral_fr_loops_compare"

module load miniforge/24.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/run01/scw6gd9/conda-envs/opendis-gpu

export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

python -u plot.py --input output_pm2000_seg40_10
