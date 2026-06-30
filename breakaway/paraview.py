"""
ParaDiS data → VTK 转换(breakaway 课题版,仿 inital_dislocation_structures/paraview.py)

比基础版多一步:把障碍 obstacles.data 也导成 obstacles.vtk(点 + radius 标量),
便于在 ParaView 里把障碍画成球,与位错线叠加观察钉扎/脱钉。

直接在下面 "配置" 区改 INPUT / OUTPUT / OBSTACLES 再运行:
  python paraview.py

INPUT     : 单个 .data 或含 *.data 的目录(位错构型快照)
OUTPUT    : VTK 输出目录(自动创建)
OBSTACLES : 障碍文件 obstacles.data(cx cy cz radius,单位 b);None 则跳过
START/END : 步号过滤(含端点),None 不限
相对路径相对本脚本目录解析,绝对路径原样使用。坐标统一为 b,障碍与位错自动对齐。

ParaView 里看障碍:加载 obstacles.vtk → Glyph 过滤器 → Glyph Type=Sphere,
  Scale Array=radius, Scale Factor=2(球半径=radius,Sphere 默认半径 0.5 → ×2)。
"""
import os, sys, glob, re
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_utils import read_paradis, write_vtk

# ========== 配置 ==========
INPUT     = "output_breakaway_single"               # 位错快照: 单 .data 或含 *.data 的目录
OUTPUT    = "vtk_breakaway"                          # VTK 输出目录
OBSTACLES = "init_breakaway/obstacles.data"          # 障碍文件(预生成); None 跳过
START     =5000                                       # 起始步号(含), None 不限
END       = None                                     # 结束步号(含), None 不限
# =========================


def write_obstacles_vtk(obs_path, out_dir):
    """obstacles.data (cx cy cz radius, 单位 b) → obstacles.vtk(点 + radius 标量)。"""
    data = np.loadtxt(obs_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    centers, radii = data[:, :3], data[:, 3]
    n = len(centers)

    vtk_file = os.path.join(out_dir, 'obstacles.vtk')
    with open(vtk_file, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('breakaway obstacles\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write(f'POINTS {n} float\n')
        for c in centers:
            f.write(f'{c[0]:.6e} {c[1]:.6e} {c[2]:.6e}\n')
        # 每个点配一个 VERTICES 单元,否则 ParaView 视作 0 单元、渲染不出来
        f.write(f'VERTICES {n} {2*n}\n')
        for i in range(n):
            f.write(f'1 {i}\n')
        f.write(f'POINT_DATA {n}\n')
        f.write('SCALARS radius float 1\n')
        f.write('LOOKUP_TABLE default\n')
        for r in radii:
            f.write(f'{r:.6e}\n')
    print(f"Obstacles: {n} spheres -> {vtk_file}")


def convert(in_path, out_dir, start=None, end=None):
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(in_path):
        data_files = [in_path]
    else:
        data_files = sorted(glob.glob(os.path.join(in_path, '*.data')))

    # 过滤掉 obstacles.data 自身(它不是位错构型)
    data_files = [f for f in data_files
                  if os.path.basename(f) != 'obstacles.data']

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
        print(f"No dislocation .data files found in {in_path}")
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

    if OBSTACLES is not None:
        obs_path = OBSTACLES if os.path.isabs(OBSTACLES) else os.path.join(base, OBSTACLES)
        if os.path.isfile(obs_path):
            os.makedirs(out_dir, exist_ok=True)
            write_obstacles_vtk(obs_path, out_dir)
        else:
            print(f"Obstacles file not found: {obs_path}")
