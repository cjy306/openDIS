"""
生成 Case A 初始配置:纯 Frank-Read 源基体(无障碍物)—— ODS-FeCrAl 课题

Case A = 纯基体,无环 / 无 α′ / 无氧化物,是 B~F 的公共底座。
基体 = 多个 Frank-Read 闭合回路(照搬 ExaDiS 官方 02_frank_read_src 构型):
  每个回路 5 节点矩形:顶边中点 UNCONSTRAINED(可动臂),其余 4 角 PINNED 兜成闭合。
  - 初始即有可动段(顶边中点)→ 加载即弓出产微塑性
  - 可动臂被两角钉住、回路闭合 → 弓出到 τ_c≈μb/L 后吐环增殖(标准 FR 增殖)
  - 不用穿盒无限直线:无限直线整体无阻平移会把应力摁到 ~1MPa(实测),已弃用。

演化历史(教训,见 CLAUDE_FeCrAl.md §4):
  ① 无钉扎闭合滑移环 → 线张力收缩两步自湮灭(2600→312→0)
  ② 穿盒无限直线 → 整体无阻平移,应力摁死在 1MPa、密度死平、FR 不开动
  ③ pyexadis 自带 insert_frank_read_src(开放线段)→ 改用官方闭合回路构型(本文件)

用法(仿 HomeWork/generate_config.py):
  python generate_caseA.py --seed 12345
  → 写 init_data_caseA_seed12345/{init_config.data, loop_type.txt, init_config_labeled.vtk}
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager, NodeConstraints
from pyexadis_utils import write_vtk

# ============================================================
# 参数(材料 = Yan 2023 一套;尺度/密度 = Pachaury 2023)
# ============================================================
BURGMAG    = 0.248e-9    # b [m]
LBOX_M     = 300e-9      # bulk 周期盒 300nm 立方
RHO_TARGET = 2.0e14      # 基体总位错密度 [1/m^2]
ARM_LEN_M  = 60e-9       # FR 回路臂长 [m](τ_c≈μb/L;盒子的 1/5,留弓出空间)

# 单一取向滑移系(先验证用):b=1/2[111], n=(0 -1 1)
# [001] 加载下 Schmid 因子 = (b·F)(n·F) = (1/√3)(1/√2) = 0.41(最大,确保可动臂受力)
_B_SINGLE = np.array([1., 1., 1.])
_N_SINGLE = np.array([0., -1., 1.])

LT_FR = 3   # loop_type 标记:FR 回路基体(Case B 另用 1=<111>环, 2=<100>环)


def insert_fr_loop(nodes, segs, burg, plane, arm_len_b, center):
    """插入一个 Frank-Read 闭合回路(照搬官方 02_frank_read_src 构型,旋转到给定滑移系)。

    官方构型(标准取向):b‖x,可动臂‖y(刃型),虚拟臂‖-z;三者两两正交。
    本函数把官方的 (x,y,z) 轴整体替换为局部正交基 (b̂, e, f):
      b̂ = burg 方向
      e  = (n × b̂) 归一  → 可动臂方向(在滑移面内、垂直 b,纯刃型,对应官方 y)
      f  = b̂ × e         → 虚拟臂方向(对应官方 -z 的非共面兜底段)
    因只是刚体旋转,官方各段 b·plane=0 的合法性自动保持。

    参数(均无量纲,以 b 为单位):burg/plane 会归一化;arm_len_b 臂长;center 回路中心。
    """
    b = np.asarray(burg, dtype=float); b = b / np.linalg.norm(b)
    n = np.asarray(plane, dtype=float); n = n / np.linalg.norm(n)
    e = np.cross(n, b); e = e / np.linalg.norm(e)    # 可动臂方向(刃型,面内垂直 b)
    f = np.cross(b, e); f = f / np.linalg.norm(f)    # 虚拟臂方向(对应官方 -z)

    L = arm_len_b
    c = np.asarray(center, dtype=float)
    istart = len(nodes)
    # 节点对应官方 rn(把 y→e, z→f):
    #  0 顶左角(钉) -e/2; 1 顶中(可动) 0; 2 顶右角(钉) +e/2; 3 底右角(钉) +e/2 - f; 4 底左角(钉) -e/2 - f
    pts = [
        (c - 0.5 * L * e,            NodeConstraints.PINNED_NODE),
        (c,                          NodeConstraints.UNCONSTRAINED),
        (c + 0.5 * L * e,            NodeConstraints.PINNED_NODE),
        (c + 0.5 * L * e - L * f,    NodeConstraints.PINNED_NODE),
        (c - 0.5 * L * e - L * f,    NodeConstraints.PINNED_NODE),
    ]
    for p, cst in pts:
        nodes.append(np.concatenate((p, [cst])))

    N = len(pts)
    for i in range(N):
        n1 = istart + i
        n2 = istart + (i + 1) % N   # 末节点接回首节点 → 闭合回路
        ldir = pts[(i + 1) % N][0] - pts[i][0]
        pn = np.cross(b, ldir)      # 滑移面法向 = b × 线方向(官方做法)
        nn = np.linalg.norm(pn)
        pn = pn / nn if nn > 1e-10 else n
        segs.append(np.concatenate(([n1, n2], b, pn)))
    return nodes, segs


def build_matrix(cell, rng, verbose=True):
    """按目标密度铺多个 FR 闭合回路(单一取向 b=1/2[111],n=(0-11),Schmid=0.41)。

    每个矩形回路周长 ≈ 4*臂长。返回 (nodes, segs, loop_type)。
    """
    Lbox = LBOX_M / BURGMAG
    vol_m3 = LBOX_M ** 3
    L_target_m = RHO_TARGET * vol_m3
    arm_b = ARM_LEN_M / BURGMAG
    per_loop_m = 4.0 * ARM_LEN_M

    b1 = _B_SINGLE / np.linalg.norm(_B_SINGLE)
    n1 = _N_SINGLE / np.linalg.norm(_N_SINGLE)

    origin, h = np.array(cell.origin), np.array(cell.h)
    margin = arm_b   # 留边距,使回路初始不跨周期边界

    nodes, segs, loop_type = [], [], []
    acc_m, placed = 0.0, 0
    while acc_m < 0.99 * L_target_m and placed < 100000:
        center = rng.uniform(margin, Lbox - margin, size=3)
        nseg0 = len(segs)
        nodes, segs = insert_fr_loop(nodes, segs, b1, n1, arm_b, center)
        loop_type += [LT_FR] * (len(segs) - nseg0)
        acc_m += per_loop_m
        placed += 1
    if verbose:
        print('FR 闭合回路(单一取向 b=1/2[111] n=(0-11) Schmid=0.41): %d 个 (臂长 %.0f nm), '
              '线长 %.3e m (目标 %.3e, 密度 %.2e /m^2)'
              % (placed, ARM_LEN_M * 1e9, acc_m, L_target_m, acc_m / vol_m3))
    return nodes, segs, np.array(loop_type, dtype=int)


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成 Case A 纯 FR 回路基体构型')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_data_caseA_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    cell = pyexadis.Cell(LBOX_M / BURGMAG)
    nodes, segs, loop_type = build_matrix(cell, rng)

    G = ExaDisNet(cell, nodes, segs)
    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)

    lt_file = os.path.join(out_dir, 'loop_type.txt')
    np.savetxt(lt_file, loop_type, fmt='%d')

    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file,
              segprops={'LoopType': loop_type.astype(float)}, verbose=False)

    print(f'Case A 初始构型已写出: {out_file}')
    print(f'段类别标记: {lt_file}  (3=FR回路, 共 {len(loop_type)} 段)')
    print(f'带标记 VTK: {vtk_file}  (ParaView 按 LoopType 染色)')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
