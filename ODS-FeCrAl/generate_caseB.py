"""
生成 Case B 初始配置:基体可动滑移网络 + 辐照位错环(无 α′、无氧化物)

Case B = Case A 基体 + 两类辐照环(见 CLAUDE_FeCrAl.md §4 矩阵 B 行):
  - a/2<111> 六边形棱柱环:可动(可被相干应力转动 → 相消干涉主角),
    用 ExaDiS 自带 insert_prismatic_loop 生成,UNCONSTRAINED
  - a<100>  方形棱柱环:不可动 sessile,用 insert_sessile_loop_100 生成,全节点 PINNED
  目标:单纯辐照环硬化。

辐照环参数 = Zhang et al. 2020, J. Nucl. Mater. 533, 152094(MA956 中子辐照 4.36 dpa):
  总环密度 7.17e21 /m^3
    a/2<111>: 平均直径 16.6 nm, 数密度 3.73e21 /m^3  (≈52%)
    a<100> : 平均直径 18.6 nm, 数密度 3.44e21 /m^3  (≈48%)

用法(仿 HomeWork/generate_config.py):
  python generate_caseB.py --seed 12345
  → 写 init_data_caseB_seed12345/init_config.data
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import insert_prismatic_loop, write_vtk   # a/2<111> 可动环(ExaDiS 自带)
from generate_glide_loop import (                          # 本课题自写
    insert_glide_loop, insert_sessile_loop_100,
    _BCC_SLIP_B, _BCC_SLIP_N,
)

# ============================================================
# 参数(材料 = Yan 2023 一套;辐照环 = Zhang 2020)
# ============================================================
BURGMAG    = 0.248e-9     # b [m]
LBOX_M     = 5.0e-6       # bulk 周期盒 [m]

# --- 基体可动滑移网络(同 Case A) ---
RHO_TARGET   = 1.0e12     # 基体可动位错密度 [1/m^2]
GLIDE_RMIN_M = 0.10e-6
GLIDE_RMAX_M = 0.20e-6

# --- a/2<111> 可动辐照环(Zhang 2020) ---
N111_DENS = 3.73e21       # 数密度 [1/m^3]
N111_DIAM = 16.6e-9       # 平均直径 [m]

# --- a<100> 不可动辐照环(Zhang 2020) ---
N100_DENS = 3.44e21       # 数密度 [1/m^3]
N100_DIAM = 18.6e-9       # 平均直径 [m]

# a<100> 的三种柏氏矢量
_B100 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
# a/2<111> 的四种柏氏矢量
_B111 = np.array([[1., 1., 1.], [-1., 1., 1.], [1., -1., 1.], [1., 1., -1.]])


def _num_from_density(density, Lbox_m):
    """由数密度和盒体积算环数量(四舍五入,至少 1)。"""
    return max(1, int(round(density * Lbox_m**3)))


def _random_positions(num, Lbox, margin, rng):
    """在 [margin, Lbox-margin]^3 内生成 num 个随机位置(无量纲)。"""
    lo, hi = margin, Lbox - margin
    return rng.uniform(lo, hi, size=(num, 3))


def build_caseB(seed, verbose=True):
    rng = np.random.RandomState(seed)

    Lbox = LBOX_M / BURGMAG
    cell = pyexadis.Cell(Lbox)
    nodes, segs = [], []

    b_glide = _BCC_SLIP_B / np.linalg.norm(_BCC_SLIP_B, axis=1)[:, None]
    n_glide = _BCC_SLIP_N / np.linalg.norm(_BCC_SLIP_N, axis=1)[:, None]
    nsys = b_glide.shape[0]

    # loop_type 标记(按段顺序;0=基体滑移, 1=a/2<111>可动环, 2=a<100>不可动环)
    # 段在 nodes/segs 中顺序追加,故各类对应连续区间,用 len(segs) 差值切片。
    loop_type = []

    # ---------- 1) 基体可动滑移网络(按密度填充)----------
    vol_m3 = LBOX_M**3
    L_target_m = RHO_TARGET * vol_m3
    acc_m = 0.0
    placed_glide = 0
    attempt = 0
    while acc_m < 0.99 * L_target_m and attempt < 100000:
        R_m = rng.uniform(GLIDE_RMIN_M, GLIDE_RMAX_M)
        R_b = R_m / BURGMAG
        margin = R_b
        pos = rng.uniform(margin, Lbox - margin, size=3)
        isys = placed_glide % nsys
        nseg0 = len(segs)
        nodes, segs = insert_glide_loop(cell, nodes, segs, b_glide[isys], n_glide[isys],
                                        R_b, pos, numnodes=20)
        loop_type += [0] * (len(segs) - nseg0)
        acc_m += 2.0 * np.pi * R_m
        placed_glide += 1
        attempt += 1
    if verbose:
        print('基体可动滑移环: %d 个, 密度 %.2e /m^2' % (placed_glide, acc_m/vol_m3))

    # ---------- 2) a/2<111> 可动辐照环 ----------
    n111 = _num_from_density(N111_DENS, LBOX_M)
    R111_b = (0.5 * N111_DIAM) / BURGMAG
    pos111 = _random_positions(n111, Lbox, R111_b, rng)
    for i in range(n111):
        burg = _B111[i % len(_B111)] / np.linalg.norm(_B111[i % len(_B111)])
        nseg0 = len(segs)
        nodes, segs = insert_prismatic_loop('bcc', cell, nodes, segs, burg,
                                            R111_b, pos111[i], maxseg=-1)
        loop_type += [1] * (len(segs) - nseg0)
    if verbose:
        print('a/2<111> 可动辐照环: %d 个, R=%.1f b (D=%.1f nm)' % (n111, R111_b, N111_DIAM*1e9))

    # ---------- 3) a<100> 不可动辐照环(全节点 PINNED)----------
    n100 = _num_from_density(N100_DENS, LBOX_M)
    R100_b = (0.5 * N100_DIAM) / BURGMAG
    pos100 = _random_positions(n100, Lbox, R100_b, rng)
    for i in range(n100):
        burg = _B100[i % len(_B100)]
        nseg0 = len(segs)
        nodes, segs = insert_sessile_loop_100(cell, nodes, segs, burg,
                                              R100_b, pos100[i], numnodes=20)
        loop_type += [2] * (len(segs) - nseg0)
    if verbose:
        print('a<100> 不可动辐照环: %d 个, R=%.1f b (D=%.1f nm), 全节点 PINNED'
              % (n100, R100_b, N100_DIAM*1e9))

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

    # loop_type 伴随文件(按段顺序;0=基体, 1=a/2<111>环, 2=a<100>环)
    lt_file = os.path.join(out_dir, 'loop_type.txt')
    np.savetxt(lt_file, loop_type, fmt='%d')

    # 直接出一份带 LoopType 标记的初始构型 VTK(段顺序=生成顺序,与 loop_type 严格对齐,
    # 不经 .data 往返,零重排风险)。ParaView 里按 LoopType 染色即可区分三类。
    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file,
              segprops={'LoopType': loop_type.astype(float)}, verbose=False)

    print(f'Case B 初始构型已写出: {out_file}')
    print(f'段类别标记已写出: {lt_file}  (0=基体 1=<111>环 2=<100>环, 共 {len(loop_type)} 段)')
    print(f'带标记 VTK 已写出: {vtk_file}  (ParaView 按 LoopType 染色)')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
