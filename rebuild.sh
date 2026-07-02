#!/bin/bash
# ============================================================================
#  OpenDiS / ExaDiS 通用一键编译脚本 (昆山 & 雄安 自动适配)
#  用法:
#     bash rebuild.sh            默认: 增量编译(build 不存在则自动全量)
#     bash rebuild.sh clean      强制全量重编
#     bash rebuild.sh cpu        强制编 CPU 版 (OpenMP)
#     bash rebuild.sh gpu        强制编 GPU 版 (CUDA)
#     可组合: bash rebuild.sh clean gpu
#  注意: 先 srun 进计算节点再跑, 不要在登录节点编译。
#        GPU 版编译在 CPU 节点也可(有 nvcc 即可), 但运行必须 GPU 节点。
# ============================================================================
set -e
ROOT=$HOME/openDIS
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
MODE="$DEFAULT_MODE"
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
module load nvidia/cuda/12.1 2>/dev/null || true
module load compiler/gcc/12.2.0 2>/dev/null || true
echo "[rebuild] g++ = $(which g++)"

# ---------- 2. 补执行权限 ----------
chmod +x configure.sh core/exadis/kokkos/bin/* 2>/dev/null || true

# ---------- 3. 自动打 kokkos git 检测兜底补丁 ----------
PATCH_FILE="$ROOT/core/exadis/kokkos/cmake/build_env_info.cmake"
if [ -f "$PATCH_FILE" ] && ! grep -q 'GIT_COMMIT_HASH "unknown"' "$PATCH_FILE"; then
  echo "[rebuild] 打 kokkos 补丁..."
  sed -i 's/  check_git_read(GIT_HASH_CACHE)/  check_git_read(GIT_HASH_CACHE)\n  if(NOT GIT_COMMIT_HASH)\n    set(GIT_COMMIT_HASH "unknown")\n  endif()\n  if(NOT GIT_CLEAN_STATUS)\n    set(GIT_CLEAN_STATUS "clean")\n  endif()/' "$PATCH_FILE"
fi

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
cmake --build "$ROOT/build" -j$(nproc)

# ---------- 6. 验证 import ----------
echo "✅ 编译完成！验证 pyexadis:"
if [ "$MODE" = "gpu" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "⚠ 当前节点无 GPU 驱动(如在 CPU 节点编 GPU 版), 跳过 import 验证。"
  echo "   到 GPU 节点后运行: python -c \"import pyexadis; print('OK')\""
else
  python -c "import pyexadis; print('pyexadis OK:', pyexadis.__file__)" \
    || echo "⚠ import 失败: 若报 libcuda not found, 说明当前不在 GPU 节点(GPU版需 GPU 节点运行)"
fi