"""
ODS-FeCrAl 项目:位错网络 data → VTK 转换(供 ParaView 可视化)

参考 HomeWork/paraview.py,但针对本项目精简:
  - 本项目障碍物(a/2<111> 环、a<100> PINNED 环)本身就是位错段,已在网络里
    → 不像 HomeWork 那样需单独转球形析出物/孪晶面
  - 复用 ExaDiS 自带 write_vtk:每段自动带 SlipSystemID / DislocationType /
    CharacterAngle / SegmentLength 字段,ParaView 里可按这些染色
  - 初始帧可叠加 loop_type 染色(LoopType 标量:0=可动直线 1=<111>环 2=<100>环 3=FR源)
    ⚠️ 仅初始帧:loop_type 按生成时段顺序对齐,而仿真输出帧经运动/remesh/反应后
       段顺序与段数已变,无法对齐 → 演化帧不加 LoopType,只看位错形貌+滑移系。

用法:
  # 转某工况所有帧
  python paraview.py --sim output_caseA_seed12345 --out vtk_caseA_seed12345
  # 指定步数范围
  python paraview.py --sim output_caseB_seed12345 --out vtk_caseB --start 0 --end 5000
  # 叠加初始帧 loop_type 染色(给出 init_data 目录)
  python paraview.py --sim output_caseB_seed12345 --out vtk_caseB --init init_data_caseB_seed12345
"""
import os, sys, glob, re
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_utils import read_paradis, write_vtk

BURGMAG = 0.248e-9   # b [m](与生成/模拟脚本一致)
LBOX_M  = 300e-9     # bulk 周期盒边长 [m](与生成/模拟脚本一致)


def _step_of(fname):
    """从 config.<step>.data 文件名提取步数;取不到返回 -1。"""
    m = re.search(r'(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else -1


def wrap_vtk_pbc(vtk_file, Lbox):
    """把 VTK 段端点坐标折回 [0, Lbox)(无量纲,以 b 为单位)。
    write_vtk 用 closest_image() 保证线段连续,会把端点放到盒外镜像位置;
    本函数把它们 % Lbox 折回主盒,使 ParaView 显示全部落在盒内。
    跳过前 8 个点(晶胞顶点),只折段端点。"""
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


def convert(sim_dir, out_dir, init_dir=None, start=None, end=None, wrap=True):
    os.makedirs(out_dir, exist_ok=True)
    Lbox_b = LBOX_M / BURGMAG   # 无量纲盒边(折叠用)

    # 收集 config.*.data(排除 restart 等非构型文件)
    data_files = sorted(glob.glob(os.path.join(sim_dir, '*.data')))
    data_files = [f for f in data_files if 'restart' not in os.path.basename(f)]

    # 步数范围过滤
    if start is not None or end is not None:
        kept = []
        for f in data_files:
            s = _step_of(f)
            if start is not None and s >= 0 and s < start:
                continue
            if end is not None and s >= 0 and s > end:
                continue
            kept.append(f)
        data_files = kept

    if not data_files:
        print(f"未在 {sim_dir} 找到 config.*.data")
        return

    # 初始帧 loop_type(可选):仅用于最早一帧染色
    loop_type = None
    first_step = min((_step_of(f) for f in data_files), default=-1)
    if init_dir:
        lt_file = os.path.join(init_dir, 'loop_type.txt')
        if os.path.exists(lt_file):
            loop_type = np.loadtxt(lt_file).astype(float)
            print(f"已载入 loop_type ({len(loop_type)} 段),仅叠加到最早帧 step={first_step}")
        else:
            print(f"[提示] {lt_file} 不存在,跳过 loop_type 染色")

    print(f"转换 {len(data_files)} 帧 ...")
    pyexadis.initialize()
    for i, f in enumerate(data_files):
        name = os.path.basename(f).replace('.data', '')
        step = _step_of(f)
        vtk_file = os.path.join(out_dir, f'{name}.vtk')

        net = read_paradis(f)
        segprops = {}
        # 仅当帧的段数与 loop_type 长度一致时才叠加(对齐保护:仿真后段数会变)
        if loop_type is not None and step == first_step:
            nseg = net.num_segments()
            if nseg == len(loop_type):
                segprops = {'LoopType': loop_type}
            else:
                print(f"  [跳过LoopType] 段数({nseg}) != loop_type({len(loop_type)}),该帧已非初始构型")

        write_vtk(net, vtk_file, segprops=segprops, crystal='BCC', verbose=False)
        if wrap:
            wrap_vtk_pbc(vtk_file, Lbox_b)
        print(f"  [{i+1}/{len(data_files)}] {name}.vtk"
              + ("  +LoopType" if segprops else ""))

    pyexadis.finalize()
    print(f"完成。输出目录: {out_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ODS-FeCrAl 位错网络 data → VTK')
    parser.add_argument('--sim', required=True, help='模拟输出目录(如 output_caseA_seed12345)')
    parser.add_argument('--out', required=True, help='VTK 输出目录')
    parser.add_argument('--init', default=None, help='init_data 目录(给出则初始帧叠加 LoopType 染色)')
    parser.add_argument('--start', type=int, default=None, help='起始步(含)')
    parser.add_argument('--end', type=int, default=None, help='结束步(含)')
    parser.add_argument('--no-wrap', action='store_true',
                        help='不折叠 PBC(保留穿盒连续线;默认折回盒内)')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    sim_dir  = args.sim  if os.path.isabs(args.sim)  else os.path.join(base, args.sim)
    out_dir  = args.out  if os.path.isabs(args.out)  else os.path.join(base, args.out)
    init_dir = (args.init if os.path.isabs(args.init) else os.path.join(base, args.init)) if args.init else None

    convert(sim_dir, out_dir, init_dir=init_dir, start=args.start, end=args.end,
            wrap=not args.no_wrap)
