#!/bin/bash
#SBATCH --job-name=cr_coherency
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=xhacnormalb
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00

set -eo pipefail

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
module load nvidia/cuda/12.1
module unload compiler/devtoolset/7.3.1 2>/dev/null || true
module load compiler/gcc/12.2.0

set -u

PROJECT_ROOT=/work/home/cjy306/openDIS
CASE_DIR=$PROJECT_ROOT/cr_coherency_test

export PYTHONPATH=$PROJECT_ROOT/core/exadis/python:$PROJECT_ROOT/core/pydis/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$PROJECT_ROOT/build/core/exadis/kokkos/core/src:$PROJECT_ROOT/build/core/exadis/src:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo ">>> [1/8] Build pyexadis"
cmake --build "$PROJECT_ROOT/build" --target pyexadis -j "$SLURM_CPUS_PER_TASK"

echo ">>> [2/8] Run pure-Python field regressions"
cd "$PROJECT_ROOT"
python -m unittest cr_coherency_test.test_coherency_field_unit -v

echo ">>> [3/8] Run focused coherency-force regressions"
cd "$PROJECT_ROOT/core/exadis/tests/unit_tests"
python test_force.py coherency_zero
python test_force.py coherency_uniform
python test_force.py coherency_periodic
python test_force.py coherency_sign

echo ">>> [4/8] Generate the shared loop and stress field"
cd "$CASE_DIR"
python generate_coherency_case.py
test -s "$CASE_DIR/init_coherency/init_config.data"
test -s "$CASE_DIR/init_coherency/coherency_stress.npz"

echo ">>> [5/8] Run the no-coherency reference"
python test_no_coherency.py
test -s "$CASE_DIR/output_no_coherency/stress_strain_dens.dat"

echo ">>> [6/8] Run the coherency-stress case"
python test_with_coherency.py
test -s "$CASE_DIR/output_with_coherency/stress_strain_dens.dat"

echo ">>> [7/8] Plot the paired comparison"
python plot_coherency_compare.py
test -s "$CASE_DIR/post_coherency/coherency_compare.png"

echo ">>> [8/8] Convert both trajectories to VTK"
python vtk_coherency_compare.py
compgen -G "$CASE_DIR/vtk_no_coherency/*.vtk" >/dev/null
compgen -G "$CASE_DIR/vtk_with_coherency/*.vtk" >/dev/null

echo ">>> Cr coherency validation workflow completed"
