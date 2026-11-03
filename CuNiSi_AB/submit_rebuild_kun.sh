#!/bin/bash
#SBATCH --job-name=CuNiSi_build
#SBATCH --partition=ksagnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Submit from the cluster: sbatch submit_rebuild_kun.sh
# Incremental GPU build; apply mixed-obstacle patch and verify the runtime.
set -eo pipefail
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate opendis-gpu
ROOT="${OPENDIS_ROOT:-$HOME/openDIS}"
export OPENDIS_ROOT="$ROOT"
export PYTHONPATH="$ROOT/core/exadis/python:$ROOT/core/pydis/python:$ROOT/lib:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ROOT/build/core/exadis/kokkos/core/src:$ROOT/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"

# ---------- 自动判断在哪个区 (靠家目录路径) ----------
#   昆山: /public/home/...  -> GPU 版, 架构 ADA89 (RTX4090)
#   雄安: /work/home/...    -> CPU 版
if [[ "$HOME" == /public/home/* ]]; then
  SITE="昆山"; DEFAULT_MODE="gpu"; GPU_ARCH="ADA89"
elif [[ "$HOME" == /work/home/* ]]; then
  SITE="雄安"; DEFAULT_MODE="cpu"; GPU_ARCH="AMPERE86"
else
  SITE="未知"; DEFAULT_MODE="cpu"; GPU_ARCH="AMPERE86"
fi

# ---------- 解析参数: clean / cpu / gpu ----------
DO_CLEAN=0
MODE="gpu"
for arg in "$@"; do
  case "$arg" in
    clean) DO_CLEAN=1 ;;
    cpu)   MODE="cpu" ;;
    gpu)   MODE="gpu" ;;
  esac
done
echo "[rebuild] 区: $SITE | 模式: $MODE${GPU_ARCH:+ (arch $GPU_ARCH)} | HOME=$HOME"

# ---------- 1. 编译器 gcc 12.2 (避免 ABI 混用) ----------
module unload compiler/devtoolset/7.3.1 2>/dev/null || true
module load nvidia/cuda/12.1
module load compiler/gcc/12.2.0
echo "[rebuild] g++ = $(which g++)"

# ---------- 2. 补执行权限 ----------
chmod +x configure.sh core/exadis/kokkos/bin/* 2>/dev/null || true

# ---------- 3. 自动打 kokkos git 检测兜底补丁 ----------
PATCH_FILE="$ROOT/core/exadis/kokkos/cmake/build_env_info.cmake"
if [ -f "$PATCH_FILE" ] && ! grep -q 'GIT_COMMIT_HASH "unknown"' "$PATCH_FILE"; then
  echo "[rebuild] 打 kokkos 补丁..."
  sed -i 's/  check_git_read(GIT_HASH_CACHE)/  check_git_read(GIT_HASH_CACHE)\n  if(NOT GIT_COMMIT_HASH)\n    set(GIT_COMMIT_HASH "unknown")\n  endif()\n  if(NOT GIT_CLEAN_STATUS)\n    set(GIT_CLEAN_STATUS "clean")\n  endif()/' "$PATCH_FILE"
fi

# Apply the mixed-obstacle binding once; stop on an incompatible patch.
PATCH="$ROOT/CuNiSi_AB/core_patch/mixed_obstacles.patch"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "[rebuild] Mixed-obstacle patch already applied"
else
  git apply --check "$PATCH"
  git apply "$PATCH"
fi
nvidia-smi

# ---------- 4. 全量还是增量 ----------
if [ "$DO_CLEAN" = "1" ] || [ ! -d "$ROOT/build" ]; then
  echo "[rebuild] 全量编译: rm -rf build + configure + make ($MODE 版)"
  rm -rf build
  if [ "$MODE" = "gpu" ]; then
    ./configure.sh -DKokkos_ENABLE_CUDA=On -DKokkos_ARCH_${GPU_ARCH}=ON \
      -DBUILD_SHARED_LIBS=ON
  else
    ./configure.sh -DKokkos_ENABLE_OPENMP=On -DKokkos_ENABLE_SERIAL=On \
      -DBUILD_SHARED_LIBS=ON \
      -DFFTW_INC_DIR=$CONDA_PREFIX/include -DFFTW_LIB_DIR=$CONDA_PREFIX/lib
  fi
else
  echo "[rebuild] 增量编译: make"
fi

# ---------- 5. 编译 ----------
cmake --build "$ROOT/build" -j"${SLURM_CPUS_PER_TASK:-8}"

# Verify the same import path used by the A/B jobs. Failure fails this job.
python - <<'PY'
import pyexadis
print("pyexadis:", pyexadis.__file__, flush=True)
if not hasattr(pyexadis.ExaDisNet, "load_obstacles_mixed"):
    raise RuntimeError("Built/imported module still lacks load_obstacles_mixed; check the module path above")
pyexadis.initialize()
pyexadis.finalize()
print("GPU runtime and mixed-obstacle interface: OK", flush=True)
PY
