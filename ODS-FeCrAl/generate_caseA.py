"""
生成 Case A 初始配置:混合基体位错网络(无障碍物)—— ODS-FeCrAl 课题

Case A = 纯基体,无环 / 无 α′ / 无氧化物,是 B~F 的公共底座。
基体配方(少量可动直线 + FR 源主体):
  - 少量穿盒无限直线(~20% 密度,UNCONSTRAINED 全可动):初始即可动,加载第一步就
    弓曲滑移、扫面积 → 提供初始微塑性,**绕过 FR 源的"初始可动位错=0"饥饿骤降**。
  - FR 源主体(~80% 密度,两端 PINNED):弓出增殖、提供密度主体。
    其中含长段(120nm)与短段(80nm)以分散激活应力。

为何这样配(关键教训,见 CLAUDE_FeCrAl.md §4):
  - **FR 饥饿骤降**:纯 FR 初始可动位错≈0,零外力预弛豫也无法让 FR 放出位错(需 τ>μb/L
    的外力);一加载则应力涨到一批源同时失稳 → 雪崩 → 应力崩塌(实测 362MPa 峰后跌到 20MPa)。
    解法:让初始构型本身含少量可动位错(穿盒直线初始即可动)→ 第一步即载微塑性,无饥饿。
  - 穿盒直线"长度不可控、一条就超密度"在此不再是问题:它只占少量,密度主体由 FR 配。
  - 无钉扎闭合滑移环已弃用(线张力向心收缩,加载两步内自湮灭 2600→312→0)。

用法(仿 HomeWork/generate_config.py):
  python generate_caseA.py --seed 12345
  → 写 init_data_caseA_seed12345/{init_config.data, loop_type.txt, init_config_labeled.vtk}
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import insert_infinite_line, insert_frank_read_src, write_vtk

# ============================================================
# 参数(材料 = Yan 2023 一套;尺度/密度 = Pachaury 2023)
# ============================================================
BURGMAG      = 0.248e-9    # b [m]
LBOX_M       = 300e-9      # bulk 周期盒 300nm 立方(Pachaury 纳米尺度)
RHO_TARGET   = 2.0e14      # 基体总位错密度 [1/m^2](Pachaury ~1.8-2.2e14)
LINE_FRAC    = 0.20        # 可动穿盒直线占总线长比例(绕过 FR 饥饿,载初始微塑性)
LONG_LEN_M   = 60e-9       # FR 长段长度 [m](盒子的 1/5,留弓出空间、避免镜像自交)
SHORT_LEN_M  = 40e-9       # FR 短段长度 [m](分散激活应力)
MAXSEG_B     = 80          # 离散段长上限 [b](≈20nm,与 test 脚本 maxseg 一致)

# BCC 12 个 <111>{110} 滑移系(b 在 plane 内,b·n=0)
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

LT_LINE  = 0   # 可动穿盒直线
LT_FR    = 3   # FR 源(长段+短段统一标记)


def _fill_lines(cell, rng, nodes, segs, loop_type, L_target_m, b_sys, n_sys, verbose):
    """填少量穿盒无限直线(UNCONSTRAINED 全可动),按目标线长。"""
    origin, h = np.array(cell.origin), np.array(cell.h)
    nsys = b_sys.shape[0]
    acc, placed, attempt = 0.0, 0, 0
    while acc < 0.99 * L_target_m and attempt < 100000:
        isys = placed % nsys
        pos = origin + np.matmul(rng.rand(3), h.T)
        Lb = insert_infinite_line(cell, nodes, segs, b_sys[isys], n_sys[isys], pos,
                                  maxseg=MAXSEG_B, trial=True)
        if Lb is None or Lb < 0:
            attempt += 1; continue
        nseg0 = len(segs)
        nodes, segs = insert_infinite_line(cell, nodes, segs, b_sys[isys], n_sys[isys], pos,
                                           maxseg=MAXSEG_B)
        loop_type += [LT_LINE] * (len(segs) - nseg0)
        acc += Lb * BURGMAG; placed += 1; attempt = 0
    if verbose:
        print('可动穿盒直线: %d 条, 线长 %.3e m (目标 %.3e)' % (placed, acc, L_target_m))
    return nodes, segs, loop_type


def _fill_fr(cell, rng, nodes, segs, loop_type, L_target_m, seg_len_m, lt_tag,
             b_sys, n_sys, label, verbose):
    """按目标线长填一群指定长度的 FR 段(两端 PINNED),追加到 nodes/segs/loop_type。"""
    Lbox = LBOX_M / BURGMAG
    nsys = b_sys.shape[0]
    seg_len_b = seg_len_m / BURGMAG
    margin = 0.5 * seg_len_b
    acc, placed, attempt = 0.0, 0, 0
    while acc < 0.99 * L_target_m and attempt < 100000:
        isys = placed % nsys
        center = rng.uniform(margin, Lbox - margin, size=3)
        nseg0 = len(segs)
        nodes, segs = insert_frank_read_src(cell, nodes, segs, b_sys[isys], n_sys[isys],
                                            seg_len_b, center)
        loop_type += [lt_tag] * (len(segs) - nseg0)
        acc += seg_len_m
        placed += 1
        attempt += 1
    if verbose:
        print('%s: %d 段 (长 %.0f nm), 线长 %.3e m (目标 %.3e)'
              % (label, placed, seg_len_m*1e9, acc, L_target_m))
    return nodes, segs, loop_type


def build_matrix(cell, rng, verbose=True):
    """生成基体(少量可动穿盒直线 + FR 源主体),返回 (nodes, segs, loop_type)。

    loop_type 按段顺序:0=可动直线, 3=FR 源(Case B 另用 1=<111>环, 2=<100>环,不冲突)。
    """
    vol_m3 = LBOX_M ** 3
    L_total_m = RHO_TARGET * vol_m3
    L_line_m = LINE_FRAC * L_total_m
    L_fr_m   = (1.0 - LINE_FRAC) * L_total_m
    # FR 主体的长段/短段各半(分散激活应力)
    L_fr_long_m  = 0.5 * L_fr_m
    L_fr_short_m = 0.5 * L_fr_m

    b_sys = _BCC_SLIP_B / np.linalg.norm(_BCC_SLIP_B, axis=1)[:, None]
    n_sys = _BCC_SLIP_N / np.linalg.norm(_BCC_SLIP_N, axis=1)[:, None]

    nodes, segs, loop_type = [], [], []
    # 1) 少量可动直线(绕过 FR 饥饿、载初始微塑性)
    nodes, segs, loop_type = _fill_lines(cell, rng, nodes, segs, loop_type, L_line_m,
                                         b_sys, n_sys, verbose)
    # 2) FR 源主体(增殖 + 密度主体),长段+短段
    nodes, segs, loop_type = _fill_fr(cell, rng, nodes, segs, loop_type, L_fr_long_m,
                                      LONG_LEN_M, LT_FR, b_sys, n_sys, 'FR长段', verbose)
    nodes, segs, loop_type = _fill_fr(cell, rng, nodes, segs, loop_type, L_fr_short_m,
                                      SHORT_LEN_M, LT_FR, b_sys, n_sys, 'FR短段', verbose)
    return nodes, segs, np.array(loop_type, dtype=int)


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
    print(f'段类别标记: {lt_file}  (0=可动直线 3=FR源, 共 {len(loop_type)} 段)')
    print(f'带标记 VTK: {vtk_file}  (ParaView 按 LoopType 染色)')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
