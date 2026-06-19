"""
生成初始位错配置:随机棱柱环网络 (BCC Fe) —— 预变形历史课题公共底座

五个工况 (A/B/C1/C2/C3) 共用同一个随机棱柱环网络作为初始构型。
本脚本只负责生成该网络并写出 init_config.data;后续的预弛豫 / 预变形 /
二次弛豫 / 正式加载由各工况的模拟脚本从这个 .data 出发串联。

棱柱环 = a/2<111> 柏氏矢量(面外分量),沿滑移柱面保守运动:
  既不像无钉扎闭合滑移环那样线张力收缩自湮灭,
  也不像穿盒无限直线那样整体无阻平移。
4 个 a/2<111> 柏氏矢量轮替,随机位置。

参数(2026-06,工况 A 起步,均可调):
  - 盒子 5μm 立方 (bulk 周期 RVE)
  - ρ = 1e12 /m^2  → 5μm 盒下总线长 ~125μm
  - 环半径 100nm (直径 200nm) → 每环周长 ~628nm → ~200 个环
  Fe BCC:b = 0.248 nm (a=0.2866nm, b=a·√3/2)

用法:
  python generate_loops.py --seed 12345
  → 写 init_loops_seed12345/{init_config.data, init_config_labeled.vtk}
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import generate_prismatic_config, write_vtk

# ============================================================
# 参数(均可调,待用户最终敲定)
# ============================================================
BURGMAG    = 0.248e-9    # b [m] —— Fe BCC a/2<111>
LBOX_M     = 5.0e-6      # bulk 周期盒 5μm 立方
RHO_TARGET = 1.0e12      # 初始位错密度 [1/m^2]
RADIUS_M   = 100e-9      # 棱柱环半径 [m](直径 200nm)
MAXSEG_B   = 200         # 离散段长上限 [b](≈50nm)


def _num_loops(rho, Lbox_m, radius_m):
    """由目标密度反算棱柱环数量:总线长 ρ·V = num · 2πR。"""
    L_target = rho * Lbox_m**3          # 总线长 [m]
    per_loop = 2.0 * np.pi * radius_m   # 单环周长 [m]
    return max(1, int(round(L_target / per_loop)))


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成随机棱柱环网络 (BCC Fe)')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_loops_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    # 无量纲化(以 b 为单位)
    Lbox_b    = LBOX_M / BURGMAG
    radius_b  = RADIUS_M / BURGMAG
    num_loops = _num_loops(RHO_TARGET, LBOX_M, RADIUS_M)
    print(f'盒 {LBOX_M*1e6:.1f}μm, ρ={RHO_TARGET:.1e}/m^2 → {num_loops} 个棱柱环 '
          f'(半径 {RADIUS_M*1e9:.0f}nm = {radius_b:.0f}b)')

    G = generate_prismatic_config('bcc', Lbox_b, num_loops, radius_b,
                                  maxseg=MAXSEG_B, seed=args.seed)
    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)

    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file, crystal='BCC', verbose=False)

    print(f'初始棱柱环网络已写出: {out_file}')
    print(f'VTK: {vtk_file}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
