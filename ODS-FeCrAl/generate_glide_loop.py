"""
生成 Case A 初始位错配置:可动滑移环 (glide / shear loop) —— ODS-FeCrAl 课题

用法 (仿 HomeWork/generate_config.py,先生成、再由 test 脚本读取):
  python generate_glide_loop.py            → 写 init_data_caseA/init_config.data
  python generate_glide_loop.py --seed 23451 --out init_data_caseA_seed23451

与 ExaDiS 自带的 insert_prismatic_loop() 区别(关键物理):
  - 棱柱环 (prismatic): 柏氏矢量有面外分量 (b·n != 0),环只能沿滑移柱面平移,
    不在面内扩张产塑性 → 这是"辐照环"(Case B 障碍物)。
  - 滑移环 (glide):     柏氏矢量在环所在 {110} 滑移面内 (b·n = 0),环可在该面内
    扩张/收缩、扫过面积承载塑性剪切 → 完全可动、闭合无端点 = 无钉扎。
    这是 bulk DD 测流变/屈服时自洽的可动位错源 (Case A 基准的基体结构)。

数据契约 (取自 core/exadis/python/pyexadis_utils.py):
  - 节点: [x, y, z, constraint],constraint 用 NodeConstraints.UNCONSTRAINED (无钉扎)
  - 段:   [n1, n2, bx, by, bz, px, py, pz]  (节点对 + 单位柏氏矢量 + 单位滑移面法向)
  - 闭合环靠末节点接回首节点 (n2 = istart),无端点 → 无钉扎
  - 柏氏矢量按单位矢量存储,物理幅值由 state["burgmag"] 承载 (与 ExaDiS 约定一致)
"""

import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import NodeConstraints, ExaDisNet


# ============================================================
# 参数 (材料 = Yan 2023 一套,见 CLAUDE_FeCrAl.md §3.5)
# ============================================================
BURGMAG    = 0.248e-9    # 柏氏矢量模 b [m]
LBOX_M     = 5.0e-6      # bulk 周期盒边长 [m]
RHO_TARGET = 1.0e12      # 初始可动位错密度 [1/m^2]
RADIUS_MIN_M = 0.10e-6   # 环半径下限 [m]
RADIUS_MAX_M = 0.20e-6   # 环半径上限 [m]


# BCC 的 12 个 <111>{110} 滑移系 (b 在 plane 内,b·n=0;取自 generate_line_config)
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


def insert_glide_loop(cell, nodes, segs, burg, plane, radius, center,
                      numnodes=20, Rorient=None):
    """向 nodes/segs 列表中插入一个平面滑移环 (可动、无钉扎)。

    参数:
      cell:     网络盒子 (pyexadis.Cell)
      nodes:    节点列表 (会被追加)
      segs:     段列表 (会被追加)
      burg:     柏氏矢量 (会被归一化);必须在 plane 内 (b·n=0)
      plane:    滑移面法向 (会被归一化)
      radius:   环半径 (无量纲,以 burgmag 为单位)
      center:   环心位置 (无量纲)
      numnodes: 环离散节点数
      Rorient:  晶体取向矩阵 (可选)

    返回: 更新后的 (nodes, segs)
    """
    b = np.asarray(burg, dtype=float)
    b = b / np.linalg.norm(b)
    n = np.asarray(plane, dtype=float)
    n = n / np.linalg.norm(n)

    # 滑移环的硬性判据: 柏氏矢量必须在滑移面内
    if np.abs(np.dot(b, n)) >= 1e-5:
        raise ValueError('滑移环要求 b·n=0 (柏氏矢量在滑移面内);'
                         '若 b 有面外分量请用 insert_prismatic_loop (棱柱环)')

    # 面内正交基: u 取 b 方向 (b 在面内),v = n × u
    u = b.copy()
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    if Rorient is not None:
        Rorient = np.asarray(Rorient, dtype=float)
        Rorient = Rorient / np.linalg.norm(Rorient, axis=1)[:, None]
        b = np.matmul(b, Rorient.T)
        n = np.matmul(n, Rorient.T)
        u = np.matmul(u, Rorient.T)
        v = np.matmul(v, Rorient.T)

    center = np.asarray(center, dtype=float)
    istart = len(nodes)
    for i in range(numnodes):
        theta = 2.0 * np.pi * i / numnodes
        p = center + radius * (np.cos(theta) * u + np.sin(theta) * v)
        nodes.append(np.concatenate((p, [NodeConstraints.UNCONSTRAINED])))

    for i in range(numnodes):
        n1 = istart + i
        n2 = istart + (i + 1) % numnodes   # 末节点接回首节点 → 闭合、无端点、无钉扎
        segs.append(np.concatenate(([n1, n2], b, n)))

    return nodes, segs


def generate_glide_loop_config(crystal, Lbox_m, burgmag, target_density,
                               radius_min_m=0.05e-6, radius_max_m=0.15e-6,
                               numnodes=20, Rorient=None, seed=-1,
                               min_sep=1.5, max_attempts=100000, verbose=True):
    """按目标位错密度生成可动滑移环初始构型 (Case A 基准的基体位错结构)。

    环随机分布在 12 个 <111>{110} 滑移系上,带防重叠检查;持续插入直到累计
    位错线长达到 target_density * V。所有几何量内部转成无量纲 (以 burgmag 为单位)。

    参数:
      crystal:        'BCC' (本课题只用 BCC)
      Lbox_m:         盒子边长 [m] (立方盒)
      burgmag:        柏氏矢量模 b [m] (无量纲化基准)
      target_density: 目标位错密度 [1/m^2]
      radius_min_m:   环半径下限 [m]
      radius_max_m:   环半径上限 [m] (半径偏小可减缓滑移环自发坍缩)
      numnodes:       每个环的离散节点数
      Rorient:        晶体取向矩阵 (可选)
      seed:           随机种子 (>0 时固定,用于多随机实现)
      min_sep:        防重叠间距因子 (两环心距 > min_sep*(r1+r2) 才接受)

    返回: ExaDisNet

    注意 (物理): 闭合滑移环受自身线张力驱动倾向收缩湮灭,加载初期密度可能回落到
    动态平衡。这是滑移环做基体结构的固有特性,已知并接受 (见 CLAUDE_FeCrAl.md §4)。
    """
    if crystal not in ['BCC', 'bcc']:
        raise ValueError('generate_glide_loop_config 目前只支持 BCC,收到 %s' % crystal)

    b_sys = _BCC_SLIP_B / np.linalg.norm(_BCC_SLIP_B, axis=1)[:, None]
    n_sys = _BCC_SLIP_N / np.linalg.norm(_BCC_SLIP_N, axis=1)[:, None]
    nsys = b_sys.shape[0]

    # --- 无量纲化 (以 burgmag 为单位,与 ExaDiS 约定一致) ---
    Lbox = Lbox_m / burgmag
    volume_m3 = Lbox_m ** 3
    L_target_m = target_density * volume_m3   # 需累计的总位错线长 [m]

    if verbose:
        print('generate_glide_loop_config(): 目标密度 %.2e /m^2, 盒 %.2f um, '
              '需总线长 %.3e m' % (target_density, Lbox_m * 1e6, L_target_m))

    cell = pyexadis.Cell(Lbox)
    if seed > 0:
        np.random.seed(seed)

    origin = np.array(cell.origin)
    h = np.array(cell.h)

    centers = []      # 环心 (无量纲)
    radii_b = []      # 半径 (无量纲)
    nodes, segs = [], []
    placed = 0
    accumulated_m = 0.0
    attempt = 0

    while accumulated_m < 0.99 * L_target_m and attempt < max_attempts:
        R_m = np.random.uniform(radius_min_m, radius_max_m)
        R_b = R_m / burgmag

        # 带防重叠的随机放置;留边界余量,使环初始时不跨周期边界
        margin = R_b
        frac = np.random.rand(3)
        pos = origin + np.matmul(frac, h.T)
        # 夹到 [margin, Lbox-margin] 内,避免初始跨边界
        pos = np.clip(pos, origin + margin, origin + Lbox - margin)

        ok = True
        for c, rc in zip(centers, radii_b):
            if np.linalg.norm(pos - c) < min_sep * (R_b + rc):
                ok = False
                break
        if not ok:
            attempt += 1
            continue

        isys = placed % nsys
        burg, plane = b_sys[isys], n_sys[isys]
        nodes, segs = insert_glide_loop(cell, nodes, segs, burg, plane,
                                        R_b, pos, numnodes=numnodes,
                                        Rorient=Rorient)
        centers.append(pos)
        radii_b.append(R_b)
        accumulated_m += 2.0 * np.pi * R_m   # 环周长 [m]
        placed += 1
        attempt = 0
        if verbose and placed % 20 == 0:
            print('  已放置 %d 环, 累计线长 %.3e m (%.0f%%)'
                  % (placed, accumulated_m, 100.0 * accumulated_m / L_target_m))

    achieved = accumulated_m / volume_m3
    if verbose:
        print('共放置 %d 个可动滑移环, 实际密度 %.2e /m^2 (目标 %.2e)'
              % (placed, achieved, target_density))
    if placed == 0:
        raise RuntimeError('未能放置任何环;请检查盒子/半径/密度设置')

    G = ExaDisNet(cell, nodes, segs)
    return G


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成 Case A 可动滑移环初始构型')
    parser.add_argument('--seed', type=int, default=12345, help='随机种子(多随机实现各不同)')
    parser.add_argument('--out', type=str, default=None,
                        help='输出目录(缺省 init_data_caseA_seed<seed>)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_data_caseA_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    G = generate_glide_loop_config(
        crystal='bcc',
        Lbox_m=LBOX_M,
        burgmag=BURGMAG,
        target_density=RHO_TARGET,
        radius_min_m=RADIUS_MIN_M,
        radius_max_m=RADIUS_MAX_M,
        seed=args.seed,
    )
    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)
    print(f'初始构型已写出: {out_file}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
