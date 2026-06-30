#!/bin/bash
# ============================================================================
#  OpenDiS / ExaDiS 一键编译脚本 —— SCNet 雄安区
#  用法:
#     bash rebuild.sh          改完 C++ 后增量编译(快; build 不存在则自动全量)
#     bash rebuild.sh clean    强制全量重编(换编译器 / import 报 undefined symbol 时用)
#  注意: 必须先 srun 进计算节点再跑,不要在登录节点编译。
# ============================================================================
set -e

ROOT=$HOME/openDIS
cd "$ROOT"

# --- 1. 确认编译器是 gcc 12.2(避免 ABI 混用) ---
module unload compiler/devtoolset/7.3.1 2>/dev/null || true
module load compiler/gcc/12.2.0 2>/dev/null || true
echo "[rebuild] g++ = $(which g++)"

# --- 2. 补执行权限(权限经常掉) ---
chmod +x configure.sh core/exadis/kokkos/bin/* 2>/dev/null || true

# --- 3. 自动打 kokkos git 检测兜底补丁(防 git pull 覆盖后编译失败) ---
PATCH_FILE="$ROOT/core/exadis/kokkos/cmake/build_env_info.cmake"
if [ -f "$PATCH_FILE" ] && ! grep -q 'GIT_COMMIT_HASH "unknown"' "$PATCH_FILE"; then
  echo "[rebuild] 打 kokkos 补丁..."
  sed -i 's/  check_git_read(GIT_HASH_CACHE)/  check_git_read(GIT_HASH_CACHE)\n  if(NOT GIT_COMMIT_HASH)\n    set(GIT_COMMIT_HASH "unknown")\n  endif()\n  if(NOT GIT_CLEAN_STATUS)\n    set(GIT_CLEAN_STATUS "clean")\n  endif()/' "$PATCH_FILE"
fi

# --- 4. 决定增量还是全量 ---
#   带 clean 参数, 或 build 目录不存在 -> 全量(configure + make)
#   否则 -> 增量(只 make)
if [ "$1" = "clean" ] || [ ! -d "$ROOT/build" ]; then
  echo "[rebuild] 全量编译: rm -rf build + configure + make"
  rm -rf build
  ./configure.sh -DKokkos_ENABLE_OPENMP=On -DKokkos_ENABLE_SERIAL=On \
    -DBUILD_SHARED_LIBS=ON \
    -DFFTW_INC_DIR=$CONDA_PREFIX/include -DFFTW_LIB_DIR=$CONDA_PREFIX/lib
else
  echo "[rebuild] 增量编译: make"
fi

# --- 5. 编译 ---
cmake --build "$ROOT/build" -j$(nproc)

# --- 6. 验证 import(靠 .bashrc 里已配好的 PYTHONPATH / LD_LIBRARY_PATH) ---
echo "✅ 编译完成！验证 pyexadis:"
python -c "import pyexadis; print('pyexadis OK:', pyexadis.__file__)"