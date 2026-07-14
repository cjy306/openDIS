"""
ParaDiS data → VTK 转换(ODS-FeCrAl 课题版,仿 breakaway/paraview.py)

在 breakaway 版基础上保留本课题特有两样:
  - PBC 折叠:write_vtk 为保线段连续会把端点放到盒外镜像,这里 % Lbox 折回主盒
    (盒边自动从每帧数据读取,不写死——课题里有 2μm / 500nm 多种盒)
  - LoopType 染色:给出 INIT_DIR(含 loop_type.txt)时,最早帧叠加段类别标量
    (仅初始帧;段数对齐保护,演化帧自动跳过)
氧化物:OXIDES 指向 oxides.data(cx cy cz Rp,单位 b)时导出 oxides.vtk(点+Rp 标量)。

直接在下面 "配置" 区改 INPUT / OUTPUT / OXIDES 再运行:
  python paraview.py

ParaView 里看氧化物:加载 oxides.vtk → Glyph 过滤器 → Glyph Type=Sphere,
  Scale Array=Rp, Scale Factor=2(球半径=Rp,Sphere 默认半径 0.5 → ×2)。
"""
import os, sys, glob, re
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_utils import read_paradis, write_vtk

# ========== 配置 ==========
INPUT    = "output_oxide_verify"                 # 位错快照: 单 .data 或含 *.data 的目录
OUTPUT   = "vtk_oxide"                    # VTK 输出目录
OXIDES   = "init_data_oxide_verify/oxides.data"  # 氧化物文件(cx cy cz Rp, 单位 b); None 跳过
INIT_DIR = None                                  # init_data 目录(含 loop_type.txt); None 跳过染色
START    = None                                  # 起始步号(含), None 不限
END      = None                                  # 结束步号(含), None 不限
STRIDE   = 100                                   # 抽帧间隔(步号整除才转); None=全转.
                                                 # write_freq=1 的炉子必配,否则上万帧
WRAP     = False                                  # PBC 折叠(要看穿盒连续线改 False)
# =========================


def write_oxides_vtk(ox_path, out_dir):
    """oxides.data (cx cy cz Rp, 单位 b) → oxides.vtk(点 + Rp 标量)。"""
    data = np.loadtxt(ox_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    centers, Rp = data[:, :3], data[:, 3]
    n = len(centers)

    vtk_file = os.path.join(out_dir, 'oxides.vtk')
    with open(vtk_file, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('oxide particles (Gaussian potential)\n')
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
        f.write('SCALARS Rp float 1\n')
        f.write('LOOKUP_TABLE default\n')
        for r in Rp:
            f.write(f'{r:.6e}\n')
    print(f"Oxides: {n} spheres -> {vtk_file}")


def wrap_vtk_pbc(vtk_file, Lbox):
    """把 VTK 段端点坐标折回 [0, Lbox)(无量纲 b)。跳过前 8 个晶胞顶点。"""
    with open(vtk_file, 'r') as f:
        lines = f.readlines()

    in_points = False
    total_points = 0
    points_written = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith('POINTS'):
            in_points = True
            total_points = int(line.strip().split()[1])
            points_written = 0
            continue
        if not in_points:
            continue
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        if points_written >= 8:   # 前 8 个是晶胞顶点,不折
            x = float(parts[0]) % Lbox
            y = float(parts[1]) % Lbox
            z = float(parts[2]) % Lbox
            lines[idx] = f"{x:.8e} {y:.8e} {z:.8e}\n"
        points_written += 1
        if points_written >= total_points:
            break

    with open(vtk_file, 'w') as f:
        f.writelines(lines)


def _step_of(fname):
    m = re.search(r'(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1


def convert(in_path, out_dir, init_dir=None, start=None, end=None, stride=None, wrap=True):
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(in_path):
        data_files = [in_path]
    else:
        data_files = sorted(glob.glob(os.path.join(in_path, '*.data')))

    # 过滤非位错构型文件
    data_files = [f for f in data_files
                  if os.path.basename(f) not in ('obstacles.data', 'oxides.data')
                  and 'restart' not in os.path.basename(f)]

    if start is not None or end is not None or stride is not None:
        filtered = []
        last_step = max((_step_of(f) for f in data_files), default=-1)
        for f in data_files:
            s = _step_of(f)
            if s >= 0:
                if start is not None and s < start:
                    continue
                if end is not None and s > end:
                    continue
                if stride is not None and s % stride != 0 and s != last_step:
                    continue          # 抽稀,但末帧永远保留(判定要看它)
            filtered.append(f)
        data_files = filtered

    if not data_files:
        print(f"No dislocation .data files found in {in_path}")
        return

    # 初始帧 LoopType(可选):仅最早一帧、且段数对齐时叠加
    loop_type = None
    first_step = min((_step_of(f) for f in data_files), default=-1)
    if init_dir:
        lt_file = os.path.join(init_dir, 'loop_type.txt')
        if os.path.isfile(lt_file):
            loop_type = np.loadtxt(lt_file).astype(float)
            print(f"LoopType: {len(loop_type)} segs, apply to earliest frame step={first_step}")
        else:
            print(f"[skip] {lt_file} not found")

    print(f"Converting {len(data_files)} files...")
    pyexadis.initialize()

    for idx, data_file in enumerate(data_files):
        basename = os.path.basename(data_file)
        name = basename.replace('.data', '')
        print(f"  [{idx+1}/{len(data_files)}] {basename}")
        try:
            net = read_paradis(data_file)
            Lbox = float(np.array(net.cell.h)[0][0])   # 盒边自动读取(立方盒)

            segprops = {}
            if loop_type is not None and _step_of(data_file) == first_step:
                nseg = net.num_segments()
                if nseg == len(loop_type):
                    segprops = {'LoopType': loop_type}
                else:
                    print(f"    [skip LoopType] segs({nseg}) != loop_type({len(loop_type)})")

            vtk_file = os.path.join(out_dir, f'{name}.vtk')
            write_vtk(net, vtk_file, segprops=segprops, crystal='BCC', verbose=False)
            if wrap:
                wrap_vtk_pbc(vtk_file, Lbox)
        except Exception as e:
            print(f"    Failed: {e}")

    pyexadis.finalize()
    print(f"Done. Output: {out_dir}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    in_path  = INPUT  if os.path.isabs(INPUT)  else os.path.join(base, INPUT)
    out_dir  = OUTPUT if os.path.isabs(OUTPUT) else os.path.join(base, OUTPUT)
    init_dir = (INIT_DIR if os.path.isabs(INIT_DIR) else os.path.join(base, INIT_DIR)) if INIT_DIR else None

    convert(in_path, out_dir, init_dir=init_dir, start=START, end=END, stride=STRIDE, wrap=WRAP)

    if OXIDES is not None:
        ox_path = OXIDES if os.path.isabs(OXIDES) else os.path.join(base, OXIDES)
        if os.path.isfile(ox_path):
            os.makedirs(out_dir, exist_ok=True)
            write_oxides_vtk(ox_path, out_dir)
        else:
            print(f"Oxides file not found: {ox_path}")
