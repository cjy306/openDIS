#!/bin/bash
#SBATCH --job-name=CuNiSi_build_sd
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# 山东 A800 (80 GB): sbatch submit_rebuild_shandong.sh [clean]
# 默认 ROOT=$HOME/openDIS；其他位置通过 OPENDIS_ROOT 指定。
# Conda 路径和软件模块与山东 .bashrc 保持一致。
# Incremental GPU build; apply mixed-obstacle patch and verify the runtime.
set -eo pipefail
# ---------- 山东 GPU 环境（按已提供的 .bashrc） ----------
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
module unload compiler/devtoolset/7.3.1 2>/dev/null || true
module load compiler/gcc/12.2.0
module load nvidia/cuda/12.1
module load compiler/cmake/3.23.3
module load git/2.30.2-gcc-7.3.1
module load anaconda3/2023.09
conda activate opendis-gpu
export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
ROOT="${OPENDIS_ROOT:-$HOME/openDIS}"
export OPENDIS_ROOT="$ROOT"
export PYTHONPATH="$ROOT/core/exadis/python:$ROOT/core/pydis/python:$ROOT/lib:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ROOT/build/core/exadis/kokkos/core/src:$ROOT/build/core/exadis/src:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"

# 山东 A800 使用 SM80；不按 HOME 猜测地区。
GPU_ARCH="AMPERE80"
DO_CLEAN=0
for arg in "$@"; do
  case "$arg" in
    clean) DO_CLEAN=1 ;;
    gpu) ;;
    *) echo "Usage: sbatch submit_rebuild_shandong.sh [clean] [gpu]" >&2; exit 2 ;;
  esac
done
echo "[rebuild] 山东 | GPU $GPU_ARCH | ROOT=$ROOT"
echo "[rebuild] g++ = $(command -v g++)"

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
if [ "$DO_CLEAN" = "1" ] || [ ! -f "$ROOT/build/CMakeCache.txt" ]; then
  echo "[rebuild] 全量编译: 山东 A800 GPU 版"
  # configure.sh 会清理 ROOT/build，先确认当前目录是 OpenDiS 根目录。
  test -f "$ROOT/CMakeLists.txt"
  test -f "$ROOT/core/exadis/CMakeLists.txt"
  bash ./configure.sh -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ARCH_AMPERE80=ON -DEXADIS_PYTHON_BINDING=ON \
    -DPYTHON_EXECUTABLE="$(command -v python)" -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release
else
  # 防止将 CPU / 昆山 ADA89 缓存当成山东 A800 增量构建。
  if ! grep -Eq '^Kokkos_ENABLE_CUDA:BOOL=(ON|On|on|TRUE|1)$' build/CMakeCache.txt ||
     ! grep -Eq '^Kokkos_ARCH_AMPERE80:BOOL=(ON|On|on|TRUE|1)$' build/CMakeCache.txt ||
     grep -Eq '^Kokkos_ARCH_(AMPERE86|ADA89):BOOL=(ON|On|on|TRUE|1)$' build/CMakeCache.txt; then
    echo "[rebuild] 缓存不是山东 A800 GPU 配置，请重新提交: sbatch submit_rebuild_shandong.sh clean" >&2
    exit 1
  fi
  echo "[rebuild] 增量编译"
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
