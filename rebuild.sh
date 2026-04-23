#!/bin/bash
set -e
cd /data/home/dg000246b/openDIS/build
make clean
make -j$(nproc)

echo "✅ 编译完成！"
echo "pyexadis 路径验证："
cd /data/home/dg000246b/openDIS/HomeWork
python3 -c "
import os, sys
pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths]
import pyexadis
print(pyexadis.__file__)
"