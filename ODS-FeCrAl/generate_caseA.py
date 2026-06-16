"""
生成 Case A 初始配置:混合基体位错网络(无障碍物)—— ODS-FeCrAl 课题

Case A = 纯基体(75% 可动直线 + 25% FR 源),无环 / 无 α′ / 无氧化物。
是 B~F 的公共底座。基体生成逻辑见 generate_microstructure.build_matrix。

用法(仿 HomeWork/generate_config.py):
  python generate_caseA.py --seed 12345
  → 写 init_data_caseA_seed12345/init_config.data + loop_type.txt + init_config_labeled.vtk
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import write_vtk
from generate_microstructure import build_matrix

# 参数(材料 = Yan 2023 一套,见 CLAUDE_FeCrAl.md §3.5)
BURGMAG    = 0.248e-9    # b [m]
LBOX_M     = 5.0e-6      # bulk 周期盒 [m]
RHO_TARGET = 1.0e12      # 基体总位错密度 [1/m^2]
FRS_FRACTION = 0.25      # FR 源占比(25%);其余 75% 为可动直线
FRS_LENGTH_M = 1.0e-6    # 单个 FR 源长度 [m]


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成 Case A 混合基体构型')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_data_caseA_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    cell = pyexadis.Cell(LBOX_M / BURGMAG)
    nodes, segs, loop_type = build_matrix(
        cell, LBOX_M, BURGMAG, RHO_TARGET, rng,
        frs_fraction=FRS_FRACTION, frs_length_m=FRS_LENGTH_M)

    G = ExaDisNet(cell, nodes, segs)
    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)

    lt_file = os.path.join(out_dir, 'loop_type.txt')
    np.savetxt(lt_file, loop_type, fmt='%d')

    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file,
              segprops={'LoopType': loop_type.astype(float)}, verbose=False)

    print(f'Case A 初始构型已写出: {out_file}')
    print(f'段类别标记: {lt_file}  (0=直线 3=FR源, 共 {len(loop_type)} 段)')
    print(f'带标记 VTK: {vtk_file}  (ParaView 按 LoopType 染色)')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
