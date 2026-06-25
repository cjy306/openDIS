"""
ParaDiS data → VTK 转换脚本(仿照 HomeWork/paraview.py,精简到本课题:只有棱柱环网络,无杂质/孪晶面)

直接在下面 "配置" 区改 INPUT / OUTPUT 再运行:
  python paraview.py

INPUT 既可以是目录(转里面所有 *.data),也可以是单个 .data 文件。
OUTPUT 是 VTK 输出目录(不存在会自动创建)。
相对路径都相对本脚本所在目录解析,绝对路径原样使用。
"""
import os, sys, glob, re
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_utils import read_paradis, write_vtk

# ========== 配置 ==========
INPUT  = "/data/home/dg000246b/openDIS/inital_dislocation_structures/output_caseC_0.28%_seed12345/load"   # 单个 .data 文件 或 含 *.data 的目录
OUTPUT = "vtk_C3"                 # VTK 输出目录
START  = 0                       # 起始步号(含),None 表示不限
END    = None                       # 结束步号(含),None 表示不限
# =========================


def convert(in_path, out_dir, start=None, end=None):
    os.makedirs(out_dir, exist_ok=True)

    # 收集要转换的 .data 文件
    if os.path.isfile(in_path):
        data_files = [in_path]
    else:
        data_files = sorted(glob.glob(os.path.join(in_path, '*.data')))

    # 按步号过滤
    if start is not None or end is not None:
        filtered = []
        for f in data_files:
            m = re.search(r'(\d+)', os.path.basename(f))
            if m:
                step = int(m.group(1))
                if start is not None and step < start:
                    continue
                if end is not None and step > end:
                    continue
            filtered.append(f)
        data_files = filtered

    if not data_files:
        print(f"No .data files found in {in_path}")
        return

    print(f"Converting {len(data_files)} files...")
    pyexadis.initialize()

    for idx, data_file in enumerate(data_files):
        basename = os.path.basename(data_file)
        name = basename.replace('.data', '')
        print(f"  [{idx+1}/{len(data_files)}] {basename}")
        try:
            net = read_paradis(data_file)
            vtk_file = os.path.join(out_dir, f'{name}.vtk')
            write_vtk(net, vtk_file, verbose=False)
        except Exception as e:
            print(f"    Failed: {e}")

    pyexadis.finalize()
    print(f"Done. Output: {out_dir}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    in_path = INPUT  if os.path.isabs(INPUT)  else os.path.join(base, INPUT)
    out_dir = OUTPUT if os.path.isabs(OUTPUT) else os.path.join(base, OUTPUT)

    convert(in_path, out_dir, start=START, end=END)
