"""
生成初始位错配置:随机剪切环(滑移环)网络 (BCC Fe)

与 generate_loops.py(棱柱环)对照:本脚本生成的是**剪切环 / 滑移环**——
柏氏矢量躺在环平面内 (b·n=0),环位于 {110} 滑移面上,可在面内膨胀/收缩
承载塑性剪切,即 Frank-Read 源激活后甩出的那种闭合环。

物理注意:剪切环在自身线张力作用下会收缩,除非分解剪应力超过 ~μb/(2R);
  半径越大阈值越低,越扛得住线张力收缩。R=200nm、Fe(μ≈82GPa, b=0.248nm)
  对应阈值约 50 MPa(R=100nm 时约 100 MPa)。
  因此剪切环适合**直接加载**(受载即膨胀,见 caseShear_load.py),
  不适合无载预弛豫(小环低应力下会自湮灭)。

参数(5μm bulk RVE,加大半径+加密,均可调):
  - 盒子 5μm 立方 (bulk 周期 RVE)
  - ρ = 5e12 /m^2  → 5μm 盒下总线长 ~625μm
  - 环半径 200nm (直径 400nm) → 每环周长 ~1257nm → ~500 个环
  Fe BCC:b = 0.248 nm (a=0.2866nm, b=a·√3/2)
  12 个 <111>{110} 滑移系轮替,随机位置。

用法:
  python generate_shear_loops.py --seed 12345
  → 写 init_shear_seed12345/{init_config.data, init_config_labeled.vtk}
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import generate_glide_config, write_vtk

# ============================================================
# 参数(沿用棱柱环底座,均可调)
# ============================================================
BURGMAG    = 0.248e-9    # b [m] —— Fe BCC a/2<111>
LBOX_M     = 5.0e-6      # bulk 周期盒 5μm 立方
RHO_TARGET = 5.0e12      # 初始位错密度 [1/m^2]
RADIUS_M   = 200e-9      # 剪切环半径 [m](直径 400nm)
MAXSEG_B   = 600         # 离散段长上限 [b](≈149nm,与 caseShear_load.py maxseg 一致)
NSIDES     = 12          # 圆环离散多边形边数


def _num_loops(rho, Lbox_m, radius_m):
    """由目标密度反算环数量:总线长 ρ·V = num · 2πR。"""
    L_target = rho * Lbox_m**3          # 总线长 [m]
    per_loop = 2.0 * np.pi * radius_m   # 单环周长 [m]
    return max(1, int(round(L_target / per_loop)))


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成随机剪切环(滑移环)网络 (BCC Fe)')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_shear_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    # 无量纲化(以 b 为单位)
    Lbox_b    = LBOX_M / BURGMAG
    radius_b  = RADIUS_M / BURGMAG
    num_loops = _num_loops(RHO_TARGET, LBOX_M, RADIUS_M)
    print(f'盒 {LBOX_M*1e6:.1f}μm, ρ={RHO_TARGET:.1e}/m^2 → {num_loops} 个剪切环 '
          f'(半径 {RADIUS_M*1e9:.0f}nm = {radius_b:.0f}b)')

    G = generate_glide_config('bcc', Lbox_b, num_loops, radius_b,
                              nsides=NSIDES, maxseg=MAXSEG_B, seed=args.seed)
    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)

    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file, crystal='BCC', verbose=False)

    print(f'初始剪切环网络已写出: {out_file}')
    print(f'VTK: {vtk_file}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
