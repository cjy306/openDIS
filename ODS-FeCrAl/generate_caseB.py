"""
生成 Case B 初始配置:混合基体 + 辐照位错环(无 α′、无氧化物)—— ODS-FeCrAl 课题

Case B = Case A 基体 + 两类辐照环(见 CLAUDE_FeCrAl.md §4 矩阵 B 行):
  - 基体:75% 可动直线 + 25% FR 源(同 Case A 配方)
  - a/2<111> 六边形棱柱环:可动(可被相干应力转动 → 相消干涉主角),
    用 ExaDiS 自带 insert_prismatic_loop 生成,UNCONSTRAINED
  - a<100>  方形棱柱环:不可动 sessile,全节点 PINNED
    (已核实:PINNED 段弹性场仍进 N²/FFT/自力三路径→障碍有效;迁移率令其速度=0→冻结;
     顺带是 §2.2 免疫对照 proxy。代价:PINNED 环不被滑移位错反应吸收,屈服阶段可接受)
  目标:单纯辐照环硬化。

辐照环参数 = Zhang et al. 2020, J. Nucl. Mater. 533, 152094(MA956 中子辐照 4.36 dpa):
  a/2<111>: 平均直径 16.6 nm, 数密度 3.73e21 /m^3  (≈52%)
  a<100> : 平均直径 18.6 nm, 数密度 3.44e21 /m^3  (≈48%)

用法:
  python generate_caseB.py --seed 12345
  → 写 init_data_caseB_seed12345/{init_config.data, loop_type.txt, init_config_labeled.vtk}
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager, NodeConstraints
from pyexadis_utils import (insert_infinite_line, insert_frank_read_src,
                            insert_prismatic_loop, write_vtk)

# ============================================================
# 参数(材料 = Yan 2023 一套;辐照环 = Zhang 2020)
# ============================================================
BURGMAG    = 0.248e-9     # b [m]
LBOX_M     = 5.0e-6       # bulk 周期盒 [m]

# --- 基体混合网络(同 Case A:75% 可动直线 + 25% FR 源) ---
RHO_TARGET   = 1.0e12     # 基体总位错密度 [1/m^2]
FRS_FRACTION = 0.25
FRS_LENGTH_M = 1.0e-6

# --- a/2<111> 可动辐照环(Zhang 2020) ---
N111_DENS = 3.73e21       # 数密度 [1/m^3]
N111_DIAM = 16.6e-9       # 平均直径 [m]

# --- a<100> 不可动辐照环(Zhang 2020) ---
N100_DENS = 3.44e21       # 数密度 [1/m^3]
N100_DIAM = 18.6e-9       # 平均直径 [m]

# BCC 12 个 <111>{110} 滑移系(基体用)
_BCC_SLIP_B = np.array([
    [-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.], [1., -1., 1.],
    [-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.], [1., -1., 1.],
    [-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.], [1., -1., 1.],
])
_BCC_SLIP_N = np.array([
    [0., -1., 1.], [0., -1., 1.], [0., 1., 1.], [0., 1., 1.],
    [1., 0., 1.], [-1., 0., 1.], [1., 0., 1.], [-1., 0., 1.],
    [1., 1., 0.], [-1., 1., 0.], [-1., 1., 0.], [1., 1., 0.],
])
_B111 = np.array([[1., 1., 1.], [-1., 1., 1.], [1., -1., 1.], [1., 1., -1.]])  # a/2<111>
_B100 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])                   # a<100>

# loop_type:0=直线基体, 3=FR源, 1=a/2<111>环, 2=a<100>环
LT_LINE, LT_FRS, LT_111, LT_100 = 0, 3, 1, 2


def insert_sessile_loop_100(cell, nodes, segs, burg, radius, center, numnodes=20):
    """插入 a<100> 不可动 (sessile) 棱柱环 —— 全节点 PINNED 的刚性弹性场障碍。
    burg 与环面垂直(b·n!=0,故 sessile);习惯面法向 n 取 = b。
    """
    b = np.asarray(burg, dtype=float); b = b / np.linalg.norm(b)
    n = b.copy()
    ref = np.array([1.0, 0.0, 0.0]) if abs(b[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref); u = u / np.linalg.norm(u)
    v = np.cross(n, u);   v = v / np.linalg.norm(v)

    center = np.asarray(center, dtype=float)
    istart = len(nodes)
    for i in range(numnodes):
        theta = 2.0 * np.pi * i / numnodes
        p = center + radius * (np.cos(theta) * u + np.sin(theta) * v)
        nodes.append(np.concatenate((p, [NodeConstraints.PINNED_NODE])))
    for i in range(numnodes):
        n1 = istart + i
        n2 = istart + (i + 1) % numnodes
        segs.append(np.concatenate(([n1, n2], b, n)))
    return nodes, segs


def build_matrix(cell, rng, segs, nodes, loop_type, verbose=True):
    """追加混合基体(75% 直线 + 25% FR)到给定 nodes/segs/loop_type。"""
    Lbox = LBOX_M / BURGMAG
    vol_m3 = LBOX_M ** 3
    L_total_m = RHO_TARGET * vol_m3
    L_frs_m, L_line_m = FRS_FRACTION * L_total_m, (1.0 - FRS_FRACTION) * L_total_m
    b_sys = _BCC_SLIP_B / np.linalg.norm(_BCC_SLIP_B, axis=1)[:, None]
    n_sys = _BCC_SLIP_N / np.linalg.norm(_BCC_SLIP_N, axis=1)[:, None]
    nsys = b_sys.shape[0]
    origin, h = np.array(cell.origin), np.array(cell.h)

    acc, placed, attempt = 0.0, 0, 0
    while acc < 0.99 * L_line_m and attempt < 100000:
        isys = placed % nsys
        pos = origin + np.matmul(rng.rand(3), h.T)
        Lb = insert_infinite_line(cell, nodes, segs, b_sys[isys], n_sys[isys], pos, trial=True)
        if Lb is None or Lb < 0:
            attempt += 1; continue
        nseg0 = len(segs)
        nodes, segs = insert_infinite_line(cell, nodes, segs, b_sys[isys], n_sys[isys], pos)
        loop_type += [LT_LINE] * (len(segs) - nseg0)
        acc += Lb * BURGMAG; placed += 1; attempt = 0
    if verbose: print('可动直线位错: %d 条, 线长 %.3e m' % (placed, acc))

    frs_len_b = FRS_LENGTH_M / BURGMAG
    acc, placed, attempt = 0.0, 0, 0
    while acc < 0.99 * L_frs_m and attempt < 100000:
        isys = placed % nsys
        center = rng.uniform(frs_len_b, Lbox - frs_len_b, size=3)
        nseg0 = len(segs)
        nodes, segs = insert_frank_read_src(cell, nodes, segs, b_sys[isys], n_sys[isys], frs_len_b, center)
        loop_type += [LT_FRS] * (len(segs) - nseg0)
        acc += FRS_LENGTH_M; placed += 1; attempt += 1
    if verbose: print('Frank-Read 源: %d 个, 线长 %.3e m' % (placed, acc))
    return nodes, segs, loop_type


def _num(density):
    return max(1, int(round(density * LBOX_M**3)))


def build_caseB(seed, verbose=True):
    rng = np.random.RandomState(seed)
    Lbox = LBOX_M / BURGMAG
    cell = pyexadis.Cell(Lbox)
    nodes, segs, loop_type = [], [], []

    # 1) 基体混合网络
    nodes, segs, loop_type = build_matrix(cell, rng, segs, nodes, loop_type, verbose)

    # 2) a/2<111> 可动辐照环
    n111 = _num(N111_DENS); R111 = (0.5 * N111_DIAM) / BURGMAG
    pos111 = rng.uniform(R111, Lbox - R111, size=(n111, 3))
    for i in range(n111):
        burg = _B111[i % len(_B111)] / np.linalg.norm(_B111[i % len(_B111)])
        nseg0 = len(segs)
        nodes, segs = insert_prismatic_loop('bcc', cell, nodes, segs, burg, R111, pos111[i], maxseg=-1)
        loop_type += [LT_111] * (len(segs) - nseg0)
    if verbose: print('a/2<111> 可动环: %d 个, R=%.1f b (D=%.1f nm)' % (n111, R111, N111_DIAM*1e9))

    # 3) a<100> 不可动辐照环(全节点 PINNED)
    n100 = _num(N100_DENS); R100 = (0.5 * N100_DIAM) / BURGMAG
    pos100 = rng.uniform(R100, Lbox - R100, size=(n100, 3))
    for i in range(n100):
        nseg0 = len(segs)
        nodes, segs = insert_sessile_loop_100(cell, nodes, segs, _B100[i % len(_B100)], R100, pos100[i])
        loop_type += [LT_100] * (len(segs) - nseg0)
    if verbose: print('a<100> 不可动环: %d 个, R=%.1f b (D=%.1f nm), 全节点 PINNED' % (n100, R100, N100_DIAM*1e9))

    G = ExaDisNet(cell, nodes, segs)
    return G, np.array(loop_type, dtype=int)


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成 Case B 配置(基体+辐照环)')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_data_caseB_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    G, loop_type = build_caseB(args.seed)
    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)

    lt_file = os.path.join(out_dir, 'loop_type.txt')
    np.savetxt(lt_file, loop_type, fmt='%d')

    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file,
              segprops={'LoopType': loop_type.astype(float)}, verbose=False)

    print(f'Case B 初始构型已写出: {out_file}')
    print(f'段类别标记: {lt_file}  (0=直线 3=FR 1=<111>环 2=<100>环, 共 {len(loop_type)} 段)')
    print(f'带标记 VTK: {vtk_file}  (ParaView 按 LoopType 染色)')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
