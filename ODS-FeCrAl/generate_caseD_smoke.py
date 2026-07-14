"""
generate_caseD_smoke.py —— Case D 冒烟测试的障碍场生成(几何投影版)

在基体盒内随机撒硬球障碍(Orowan 几何投影,type=0),写 obstacles.data 到 init 目录。
两段分离:本脚本只产几何;test_caseD_smoke.py 读取并运行。
⚠️ 冒烟定位:无限硬障碍 = Orowan 上限强化,只回答"屈服升不升",数字不进论文。

用法(昆山):
  python generate_caseD_smoke.py                    # 默认 rhop=2.5e21, R=10nm
  python generate_caseD_smoke.py --rhop 1e21 --R 10
之后:
  python test_caseD_smoke.py
可视化:paraview.py 的 OXIDES 指向 init_data_caseD_smoke/obstacles.data(同格式,直接画球)
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet

BURGMAG = 0.248e-9


def main():
    import argparse
    pyexadis.initialize()
    p = argparse.ArgumentParser(description='生成冒烟测试障碍场(硬球,Orowan 投影)')
    p.add_argument('--matrix', type=str, default='output_relax_seed12345/config.9800.data',
                   help='基体构型(只用来读盒尺寸)')
    p.add_argument('--rhop', type=float, default=2.5e21, help='障碍数密度 [1/m^3]')
    p.add_argument('--R',    type=float, default=10.0,   help='障碍半径 [nm]')
    p.add_argument('--seed', type=int,   default=1)
    p.add_argument('--out',  type=str,   default='init_data_caseD_smoke')
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mat_file = args.matrix if os.path.isabs(args.matrix) else os.path.join(base_dir, args.matrix)
    out_dir  = args.out    if os.path.isabs(args.out)    else os.path.join(base_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # 盒尺寸从基体文件读(立方盒)
    G = ExaDisNet()
    G.read_paradis(mat_file)
    Lbox = float(np.array(G.cell.h)[0][0])          # [b]
    Lbox_m = Lbox * BURGMAG
    V_m3 = Lbox_m ** 3

    N = int(round(args.rhop * V_m3))
    R_b = args.R * 1e-9 / BURGMAG
    rng = np.random.RandomState(args.seed)
    centers = rng.uniform(0.0, Lbox, size=(N, 3))   # PBC 盒,均匀随机,不避让

    obs = np.hstack([centers, np.full((N, 1), R_b)])
    np.savetxt(os.path.join(out_dir, 'obstacles.data'), obs, fmt='%.6e')

    # 量级预告:L=(2*D*rhop)^-1/2, Orowan RSS ~ (1/2pi)*mu*b/(L-D)
    D = 2 * args.R * 1e-9
    L = (2 * D * args.rhop) ** -0.5
    tau_oro = 0.159 * 81e9 * BURGMAG / (L - D)
    print(f'基体盒: {Lbox_m*1e6:.2f} um, 障碍: {N} 个, R={args.R}nm, rhop={args.rhop:.2e}/m^3')
    print(f'滑移面等效间距 L≈{L*1e9:.0f}nm, Orowan 分切应力量级 ~{tau_oro/1e6:.0f} MPa '
          f'(单轴 ~{tau_oro/1e6/0.41:.0f} MPa) —— 预期屈服抬升的量级参考')
    print(f'已写出: {out_dir}/obstacles.data')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
