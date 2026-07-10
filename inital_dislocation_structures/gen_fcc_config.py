"""
生成 FCC Cu 初始位错构型 —— "类型 × 半径 → 力学性能" 研究的公共生成器

两种类型(均 FCC Cu, 5um bulk RVE, rho 统一), 都是不同塑性机制:
  prismatic  棱柱环(b 面外, 沿柱面保守运动), interaction-limited
  fr         Frank-Read 源(钉扎有限段, 臂长 L=2R; 限高 Schmid + 掺零 Schmid 林源), source-limited

(滑移环 glide 已移除: 闭合环在 bulk PBC 无自由表面会自湮灭, 直接加载/弛豫都不行,
 见记忆 fcc-initial-structure-study。)

按目标密度反算数量: 总线长 rho*V = num * (单元线长)。
  棱柱环 单元线长 = 4 * PRISM_R (FCC Nsides=4 菱形, 边长=R, 周长 4R, 不是圆的 2*pi*R!)
  FR     单元线长 = L = 2 * RADIUS_M(初始直段)
  当前 FR 臂长 = 棱柱边长 = 400nm 对齐; 但两者半径参数不同(RADIUS_M vs PRISM_R_M), 别混。

用法:
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
from pyexadis_utils import (generate_prismatic_config,
                            insert_frank_read_src, write_vtk)

# ============================================================
# 参数(FCC Cu, 事实源 = core/exadis/examples/22_fcc_Cu_15um_1e3)
# ============================================================
CRYSTAL    = 'fcc'
BURGMAG    = 2.55e-10    # b [m] —— Cu a/2<110>
LBOX_M     = 5.0e-6      # bulk 周期盒 5um 立方
RADIUS_M   = 200e-9      # FR 定义半径; FR 臂长 L = 2R = 400nm
PRISM_R_M  = 400e-9      # 棱柱环半径 = 边长(FCC 菱形边长 = R); 取 400nm 与 FR 臂长齐平
MAXSEG_B   = 200         # 离散段长上限 [b](约 51nm)

# 密度 [1/m^2]: FR/棱柱 直接加载, 统一用适中密度 RHO_LOAD。
RHO_LOAD    = 2.0e12
DEFAULT_RHO = {'prismatic': RHO_LOAD, 'fr': RHO_LOAD}

# [001] 拉伸下 12 个 {111}<110> 系的 Schmid 因子: 8 个为 0.408, 4 个为 0。
# 高 Schmid 8 个(0-based)作 FR 主群, 保证主群在 [001] 下都能开动。
HIGH_SCHMID_001 = [0, 1, 3, 4, 6, 7, 9, 10]
# 零 Schmid 4 个; FR 额外掺 N_ZERO 个当"林源"(不被 [001] 直接驱动, 但可结点 /
# 被内应力激活, 更接近真实源分布)。随机挑系 + 随机位置, 加在主群之上。
ZERO_SCHMID_001 = [2, 5, 8, 11]
N_ZERO          = 2

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


def generate_fr_config(Lbox_b, num_src, length_b, seed, sysids, zero_sysids=None, n_zero=0):
    """生成 FR 源网络(钉扎有限段), 仿 generate_glide_config 的撒点逻辑。

    每个源是 {111}<110> 系上一条长 length_b 的直段, 两端 PINNED(由
    insert_frank_read_src 实现), theta=0 即沿 burgers(螺型起始)。

    主群 num_src 个落在 sysids(高 Schmid)系上; 另在 zero_sysids 上追加 n_zero 个
    "林源"(随机挑系 + 随机位置), 它们不被 [001] 直接驱动, 但可结点 / 被内应力激活。
    """
    b = _FCC_B / np.linalg.norm(_FCC_B, axis=1)[:, None]
    n = _FCC_N / np.linalg.norm(_FCC_N, axis=1)[:, None]
    sel = np.array(sysids)
    nsel = len(sel)
    numnodes = max(3, int(np.ceil(length_b / MAXSEG_B)) + 1)   # 初始离散与 maxseg 对齐

    cell = pyexadis.Cell(Lbox_b)
    np.random.seed(seed)
    nodes, segs = [], []

    # 主群: 高 Schmid 系
    pos = np.random.rand(num_src, 3)
    pos = np.array(cell.origin) + np.matmul(pos, np.array(cell.h).T)
    for i in range(num_src):
        isys = sel[i % nsel]
        nodes, segs = insert_frank_read_src(cell, nodes, segs, b[isys], n[isys],
                                            length_b, pos[i], theta=0.0, numnodes=numnodes)

    # 追加: n_zero 个零 Schmid 林源(随机挑系 + 随机位置)
    if zero_sysids is not None and n_zero > 0:
        zsel = np.array(zero_sysids)
        zsys = zsel[np.random.randint(0, len(zsel), n_zero)]
        zpos = np.random.rand(n_zero, 3)
        zpos = np.array(cell.origin) + np.matmul(zpos, np.array(cell.h).T)
        for i in range(n_zero):
            isys = zsys[i]
            nodes, segs = insert_frank_read_src(cell, nodes, segs, b[isys], n[isys],
                                                length_b, zpos[i], theta=0.0, numnodes=numnodes)
    return ExaDisNet(cell, nodes, segs)


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成 FCC Cu 初始位错构型 (prismatic/fr)')
    parser.add_argument('--type', required=True, choices=['prismatic', 'fr'])
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--rho', type=float, default=None,
                        help=f'密度 [1/m^2]; 默认按类型 {DEFAULT_RHO}')
    parser.add_argument('--n-zero', type=int, default=N_ZERO,
                        help=f'FR 掺入的零 Schmid 林源数(默认 {N_ZERO}); 仅 --type fr 生效')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    rho = args.rho if args.rho is not None else DEFAULT_RHO[args.type]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_{args.type}_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    Lbox_b   = LBOX_M / BURGMAG
    radius_b = RADIUS_M / BURGMAG

    if args.type == 'prismatic':
        # FCC 棱柱环是 Nsides=4 菱形, 边长=R, 周长=4R(不是 2*pi*R)。用 PRISM_R_M
        prism_radius_b = PRISM_R_M / BURGMAG
        num = _num_units(rho, LBOX_M, 4.0 * PRISM_R_M)
        print(f'[prismatic] rho={rho:.1e}, {num} 环, 边长R={PRISM_R_M*1e9:.0f}nm (周长4R)')
        G = generate_prismatic_config(CRYSTAL, Lbox_b, num, prism_radius_b,
                                      maxseg=MAXSEG_B, seed=args.seed)
    else:  # fr
        length_b = 2.0 * radius_b
        num = _num_units(rho, LBOX_M, 2.0 * RADIUS_M)
        print(f'[fr] rho={rho:.1e}, {num} 主群源(高Schmid) + {args.n_zero} 零Schmid林源, '
              f'L=2R={2*RADIUS_M*1e9:.0f}nm')
        G = generate_fr_config(Lbox_b, num, length_b, args.seed, HIGH_SCHMID_001,
                               zero_sysids=ZERO_SCHMID_001, n_zero=args.n_zero)

    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)
    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file, crystal='FCC', verbose=False)
    print(f'写出: {out_file}')
    print(f'VTK:  {vtk_file}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
