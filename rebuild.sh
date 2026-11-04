#!/bin/bash
# ============================================================================
#  OpenDiS / ExaDiS 通用一键编译脚本 (昆山 / 山东 / 雄安 自动适配)
#
#  用法:
#     bash rebuild.sh            默认: 增量编译(build 不存在则自动全量)
#     bash rebuild.sh clean      强制全量重编
#     bash rebuild.sh cpu        强制编 CPU 版
#     bash rebuild.sh gpu        强制编 GPU 版
#     可组合: bash rebuild.sh clean gpu
#
#  注意:
#     1. 建议先进入计算节点后再编译，不要长期在登录节点编译
#     2. GPU 版运行必须在 GPU 节点
# ============================================================================

set -e

ROOT="$HOME/openDIS"
cd "$ROOT"

# ---------- 自动判断在哪个区 ----------
# 昆山: /public/home/...      -> 默认 GPU, ADA89
# 山东: slurm 分区含 qdagnormal -> 默认 GPU, AMPERE80
# 雄安: /work/home/... 但不是山东 -> 默认 CPU, AMPERE86

SITE="未知"
DEFAULT_MODE="cpu"
GPU_ARCH="AMPERE86"

if [[ "$HOME" == /public/home/* ]]; then
    SITE="昆山"
    DEFAULT_MODE="gpu"
    GPU_ARCH="ADA89"
elif sinfo -h -o "%P" 2>/dev/null | tr ' ' '\n' | grep -q '^qdagnormal'; then
    SITE="山东"
    DEFAULT_MODE="gpu"
    GPU_ARCH="AMPERE80"
elif [[ "$HOME" == /work/home/* ]]; then
    SITE="雄安"
    DEFAULT_MODE="cpu"
    GPU_ARCH="AMPERE86"
fi

# ---------- 解析参数 ----------
DO_CLEAN=0
MODE="$DEFAULT_MODE"

for arg in "$@"; do
    case "$arg" in
        clean) DO_CLEAN=1 ;;
        cpu)   MODE="cpu" ;;
        gpu)   MODE="gpu" ;;
    esac
done

echo "[rebuild] 区: $SITE | 模式: $MODE | GPU_ARCH: $GPU_ARCH | HOME=$HOME"

# ---------- 模块环境 ----------
module unload compiler/devtoolset/7.3.1 2>/dev/null || true
module load compiler/gcc/12.2.0 2>/dev/null || true
module load nvidia/cuda/12.1 2>/dev/null || true
module load compiler/cmake/3.23.3 2>/dev/null || true
module load git/2.30.2-gcc-7.3.1 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || true

# ---------- Conda ----------
if [ -f /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh ]; then
    source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
fi

if command -v conda >/dev/null 2>&1; then
    conda activate opendis-gpu 2>/dev/null || true
fi

echo "[rebuild] python = $(which python || true)"
echo "[rebuild] g++    = $(which g++ || true)"
echo "[rebuild] nvcc   = $(which nvcc || true)"
echo "[rebuild] cmake  = $(which cmake || true)"

# 给 nvcc_wrapper 指定宿主编译器
export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"

# ---------- 补执行权限 ----------
chmod +x configure.sh 2>/dev/null || true
chmod +x core/exadis/kokkos/bin/* 2>/dev/null || true

# ---------- Kokkos 补丁 ----------
PATCH_FILE="$ROOT/core/exadis/kokkos/cmake/build_env_info.cmake"
if [ -f "$PATCH_FILE" ] && ! grep -q 'GIT_COMMIT_HASH "unknown"' "$PATCH_FILE"; then
    echo "[rebuild] 打 kokkos 补丁..."
    sed -i 's/  check_git_read(GIT_HASH_CACHE)/  check_git_read(GIT_HASH_CACHE)\
  if(NOT GIT_COMMIT_HASH)\
    set(GIT_COMMIT_HASH "unknown")\
  endif()\
  if(NOT GIT_CLEAN_STATUS)\
    set(GIT_CLEAN_STATUS "clean")\
  endif()/' "$PATCH_FILE"
fi

# ---------- 配置 ----------
if [ "$DO_CLEAN" = "1" ] || [ ! -d "$ROOT/build" ]; then
    echo "[rebuild] 全量编译: rm -rf build + configure ($MODE 版)"
    rm -rf build

    if [ "$MODE" = "gpu" ]; then
        ./configure.sh \
            -DKokkos_ENABLE_CUDA=On \
            -DKokkos_ENABLE_CUDA_LAMBDA=On \
            -DKokkos_ARCH_${GPU_ARCH}=ON \
            -DEXADIS_PYTHON_BINDING=ON \
            -DPYTHON_EXECUTABLE="$(command -v python)" \
            -DBUILD_SHARED_LIBS=ON \
            -DCMAKE_BUILD_TYPE=Release
    else
        ./configure.sh \
            -DKokkos_ENABLE_OPENMP=On \
            -DKokkos_ENABLE_SERIAL=On \
            -DEXADIS_PYTHON_BINDING=ON \
            -DPYTHON_EXECUTABLE="$(command -v python)" \
            -DBUILD_SHARED_LIBS=ON \
            -DFFTW_INC_DIR="$CONDA_PREFIX/include" \
            -DFFTW_LIB_DIR="$CONDA_PREFIX/lib" \
            -DCMAKE_BUILD_TYPE=Release
    fi
else
    echo "[rebuild] 增量编译: 直接 make"
fi

# ---------- 编译 ----------
BUILD_JOBS=${BUILD_JOBS:-8}
echo "[rebuild] 编译并行数: $BUILD_JOBS"
cmake --build "$ROOT/build" -j"$BUILD_JOBS"

# ---------- 运行时环境 ----------
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$HOME/openDIS/build/core/exadis/kokkos/core/src:$HOME/openDIS/build/core/exadis/src:${LD_LIBRARY_PATH}"
export PYTHONPATH="$HOME/openDIS/core/exadis/python:$HOME/openDIS/core/pydis/python:${PYTHONPATH}"

# ---------- 验证 ----------
echo "✅ 编译完成，验证 pyexadis ..."
if [ "$MODE" = "gpu" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "⚠ 当前节点没有 GPU 驱动，跳过 GPU import 验证"
    echo "   到 GPU 节点后执行: python -c \"import pyexadis; print('OK', pyexadis.__file__)\""
else
    python -c "import pyexadis; print('pyexadis OK:', pyexadis.__file__)" \
        || echo "⚠ import 失败：如果是 GPU 版且报 libcuda / 运行时错误，通常说明当前不在 GPU 节点"
fi