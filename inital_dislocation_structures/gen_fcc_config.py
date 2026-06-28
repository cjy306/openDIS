"""
生成 FCC Cu 初始位错构型 —— "类型 × 半径 → 力学性能" 研究的公共生成器

三种类型(均 FCC Cu, 5um bulk RVE, rho 锁定为同一值):
  glide      滑移环(闭合, b.n=0, {111}<110>), 限高 Schmid 滑移系
  prismatic  棱柱环(b 面外, 沿柱面保守运动)
  fr         Frank-Read 源(钉扎有限段, 臂长 L = 2R), 限高 Schmid 滑移系

类型是不同塑性机制, 不是单一参数:
  glide      自由闭合环, 直接承载剪切。本密度(rho=1e12)下邻居内应力 ~3MPa 远
             小于线张力坍缩应力 ~35MPa, 自由环会自湮灭 -> 加载时用 sigma0 预应力
             从激活阈值起步(见 fcc_load.py), 这是明确的建模假设。
  prismatic  近 sessile 障碍, interaction-limited。
  fr         可增殖源, source-limited。臂长 L=2R 使其激活应力 ~mu*b/L 与
             环的 mu*b/(2R) 同量级, 三类型在同一半径横坐标上可比。

按目标密度反算数量: 总线长 rho*V = num * (单元线长)。
  环   单元线长 = 2*pi*R(周长)
  FR   单元线长 = L = 2R(初始直段)

用法:
  python gen_fcc_config.py --type glide     --seed 12345
  python gen_fcc_config.py --type prismatic --seed 12345
  python gen_fcc_config.py --type fr        --seed 12345
  -> 写 init_<type>_seed<seed>/{init_config.data, init_config_labeled.vtk}
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import (generate_glide_config, generate_prismatic_config,
                            insert_frank_read_src, write_vtk)

# ============================================================
# 参数(FCC Cu, 事实源 = core/exadis/examples/22_fcc_Cu_15um_1e3)
# ============================================================
CRYSTAL    = 'fcc'
BURGMAG    = 2.55e-10    # b [m] —— Cu a/2<110>
LBOX_M     = 5.0e-6      # bulk 周期盒 5um 立方
RHO_TARGET = 1.0e12      # 初始位错密度 [1/m^2](三类型统一锁定)
RADIUS_M   = 200e-9      # 环半径 [m]; FR 臂长 L = 2R = 400nm
MAXSEG_B   = 200         # 离散段长上限 [b](约 51nm)
NSIDES     = 12          # 环离散多边形边数

# [001] 拉伸下 12 个 {111}<110> 系的 Schmid 因子: 8 个为 0.408, 4 个为 0。
# 剔除 m=0 的 4 个(0-based 索引 2,5,8,11), 留 8 个高 Schmid 系给 glide/fr,
# 使所有源都能在 [001] 下开动(否则零 Schmid 系上的源永远不动 / glide 必塌)。
HIGH_SCHMID_001 = [0, 1, 3, 4, 6, 7, 9, 10]

# FCC 12 个 <110>{111} 滑移系(b.n=0), 与 pyexadis_utils 内置一致; FR 用
_FCC_B = np.array([
    [0.,1.,-1.], [1.,0.,-1.], [1.,-1.,0.],
    [0.,1.,-1.], [1.,0.,1.],  [1.,1.,0.],
    [0.,1.,1.],  [1.,0.,-1.], [1.,1.,0.],
    [0.,1.,1.],  [1.,0.,1.],  [1.,-1.,0.],
])
_FCC_N = np.array([
    [1.,1.,1.],  [1.,1.,1.],  [1.,1.,1.],
    [-1.,1.,1.], [-1.,1.,1.], [-1.,1.,1.],
    [1.,-1.,1.], [1.,-1.,1.], [1.,-1.,1.],
    [1.,1.,-1.], [1.,1.,-1.], [1.,1.,-1.],
])


def _num_units(rho, Lbox_m, per_unit_m):
    """由目标密度反算单元数量: 总线长 rho*V = num * per_unit。"""
    L_target = rho * Lbox_m**3
    return max(1, int(round(L_target / per_unit_m)))


def generate_fr_config(Lbox_b, num_src, length_b, seed, sysids):
    """生成 FR 源网络(钉扎有限段), 仿 generate_glide_config 的撒点逻辑。

    每个源是 {111}<110> 系上一条长 length_b 的直段, 两端 PINNED(由
    insert_frank_read_src 实现), theta=0 即沿 burgers(螺型起始)。
    """
    b = _FCC_B / np.linalg.norm(_FCC_B, axis=1)[:, None]
    n = _FCC_N / np.linalg.norm(_FCC_N, axis=1)[:, None]
    sel = np.array(sysids)
    nsel = len(sel)
    numnodes = max(3, int(np.ceil(length_b / MAXSEG_B)) + 1)   # 初始离散与 maxseg 对齐

    cell = pyexadis.Cell(Lbox_b)
    np.random.seed(seed)
    pos = np.random.rand(num_src, 3)
    pos = np.array(cell.origin) + np.matmul(pos, np.array(cell.h).T)

    nodes, segs = [], []
    for i in range(num_src):
        isys = sel[i % nsel]
        nodes, segs = insert_frank_read_src(cell, nodes, segs, b[isys], n[isys],
                                            length_b, pos[i], theta=0.0, numnodes=numnodes)
    return ExaDisNet(cell, nodes, segs)


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成 FCC Cu 初始位错构型 (glide/prismatic/fr)')
    parser.add_argument('--type', required=True, choices=['glide', 'prismatic', 'fr'])
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_{args.type}_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    Lbox_b   = LBOX_M / BURGMAG
    radius_b = RADIUS_M / BURGMAG

    if args.type == 'glide':
        num = _num_units(RHO_TARGET, LBOX_M, 2.0 * np.pi * RADIUS_M)
        print(f'[glide] {num} 环, R={RADIUS_M*1e9:.0f}nm, 限 {len(HIGH_SCHMID_001)} 高Schmid系')
        G = generate_glide_config(CRYSTAL, Lbox_b, num, radius_b, nsides=NSIDES,
                                  maxseg=MAXSEG_B, seed=args.seed, sysids=HIGH_SCHMID_001)
    elif args.type == 'prismatic':
        num = _num_units(RHO_TARGET, LBOX_M, 2.0 * np.pi * RADIUS_M)
        print(f'[prismatic] {num} 环, R={RADIUS_M*1e9:.0f}nm')
        G = generate_prismatic_config(CRYSTAL, Lbox_b, num, radius_b,
                                      maxseg=MAXSEG_B, seed=args.seed)
    else:  # fr
        length_b = 2.0 * radius_b
        num = _num_units(RHO_TARGET, LBOX_M, 2.0 * RADIUS_M)
        print(f'[fr] {num} 源, L=2R={2*RADIUS_M*1e9:.0f}nm, 限 {len(HIGH_SCHMID_001)} 高Schmid系')
        G = generate_fr_config(Lbox_b, num, length_b, args.seed, HIGH_SCHMID_001)

    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)
    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file, crystal='FCC', verbose=False)
    print(f'写出: {out_file}')
    print(f'VTK:  {vtk_file}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
