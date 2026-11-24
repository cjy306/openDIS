#!/bin/bash -l
#SBATCH --job-name=OpenDiS_build_pt
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# N40: sbatch -p gpu_4090 --gpus=1 ./submit_rebuild_paratera.sh [clean]
set -eo pipefail

# 1. Explicit environment (do not rely on interactive .bashrc)
module load gcc/12.2
module load cuda/12.1
module load cmake/3.30.0
module load miniforge/24.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/run01/scw6gd9/conda-envs/opendis-gpu

export OPENDIS_ROOT=/data/run01/scw6gd9/openDIS
ROOT="$OPENDIS_ROOT"
cd "$ROOT"
export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ROOT/build/core/exadis/kokkos/core/src:$ROOT/build/core/exadis/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/core/exadis/python:$ROOT/core/pydis/python:$ROOT/lib${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# 2. Check the requested GPU and local dependencies; no network access
hostname
module list
gcc --version
nvcc --version
cmake --version
python --version
nvidia-smi
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 4090'
for f in CMakeLists.txt core/exadis/CMakeLists.txt core/exadis/kokkos/CMakeLists.txt core/exadis/python/pybind11/CMakeLists.txt core/pydis/python/ctypesgen/run.py core/pydis/python/pydis/graph/graph.py; do
    test -f "$f" || { echo "Missing: $f" >&2; exit 1; }
done
test -x core/exadis/kokkos/bin/nvcc_wrapper
test -x core/pydis/python/ctypesgen/run.py

# 3. Mixed obstacle binding, same patch as the Shandong build
PATCH="$ROOT/CuNiSi_AB/core_patch/mixed_obstacles.patch"
test -f "$PATCH"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo 'Mixed-obstacle patch already applied'
else
    git apply --check "$PATCH"
    git apply "$PATCH"
fi

# 4. First / incremental / explicitly clean build
case "${1:-}" in
    '') ;;
    clean) rm -rf -- "$ROOT/build" ;;
    *) echo 'Usage: sbatch -p gpu_4090 --gpus=1 ./submit_rebuild_paratera.sh [clean]' >&2; exit 2 ;;
esac
if [ -f build/CMakeCache.txt ]; then
    if ! grep -Eiq '^Kokkos_ENABLE_CUDA:BOOL=(ON|TRUE|1)$' build/CMakeCache.txt || ! grep -Eiq '^Kokkos_ARCH_ADA89:BOOL=(ON|TRUE|1)$' build/CMakeCache.txt || grep -Eiq '^Kokkos_ARCH_(AMPERE80|AMPERE86):BOOL=(ON|TRUE|1)$' build/CMakeCache.txt; then
        echo 'Incompatible cache. Resubmit with clean.' >&2
        exit 1
    fi
fi
cmake -S "$ROOT" -B "$ROOT/build" \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ARCH_ADA89=ON \
    -DEXADIS_PYTHON_BINDING=ON \
    -DPYTHON_EXECUTABLE="$(command -v python)" \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" -j6

# 5. Verify the actual extension and initialize Kokkos on the GPU node
python - <<'PY'
import os
from pathlib import Path
import pyexadis
root = Path(os.environ['OPENDIS_ROOT']).resolve()
module = Path(pyexadis.__file__).resolve()
print('pyexadis:', module, flush=True)
assert module.is_relative_to(root / 'core/exadis/python'), 'Unexpected pyexadis path'
assert hasattr(pyexadis.ExaDisNet, 'load_obstacles_mixed'), 'Mixed-obstacle interface missing'
pyexadis.initialize()
pyexadis.finalize()
print('GPU runtime and mixed-obstacle interface: OK', flush=True)
PY

echo 'OpenDiS build successful'
