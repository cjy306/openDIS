"""
生成 Case A 初始配置:混合基体位错网络(无障碍物)—— ODS-FeCrAl 课题

Case A = 纯基体,无环 / 无 α′ / 无氧化物,是 B~F 的公共底座。
基体配方(沿用 Pachaury 2023):
  - 75% 可动直线位错(穿周期盒,首尾经 PBC 相接;UNCONSTRAINED 全可动)
    → 从第 0 步即全长可动,加载即产生渐进微塑性,应力-应变曲线平滑
  - 25% Frank-Read 源(两端 PINNED)
    → 加载下弓出增殖,保证大应变下位错密度供给

为何不用闭合滑移环(历史教训):无钉扎闭合滑移环受线张力向心收缩,零外力下稳态=
缩成零 → 加载两步内全部自湮灭(实测 nodes 2600→312→0)。直线穿周期盒去掉"闭合"
即可三者(可动+无钉扎+稳定)兼得。

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
# 参数(材料 = Yan 2023 一套,见 CLAUDE_FeCrAl.md §3.5)
# ============================================================
BURGMAG      = 0.248e-9    # b [m]
LBOX_M       = 5.0e-6      # bulk 周期盒 [m]
RHO_TARGET   = 1.0e12      # 基体总位错密度 [1/m^2]
FRS_FRACTION = 0.25        # FR 源占比(25%);其余 75% 为可动直线
FRS_LENGTH_M = 1.0e-6      # 单个 FR 源长度 [m]

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

LOOP_TYPE_LINE = 0   # 直线可动位错
LOOP_TYPE_FRS  = 3   # Frank-Read 源


def build_matrix(cell, rng, verbose=True):
    """生成混合基体(75% 可动直线 + 25% FR 源),返回 (nodes, segs, loop_type)。

    loop_type 按段顺序:0=直线, 3=FR 源(Case B 另用 1=<111>环, 2=<100>环,不冲突)。
    """
    Lbox = LBOX_M / BURGMAG
    vol_m3 = LBOX_M ** 3
    L_total_m = RHO_TARGET * vol_m3
    L_frs_m  = FRS_FRACTION * L_total_m
    L_line_m = (1.0 - FRS_FRACTION) * L_total_m

    b_sys = _BCC_SLIP_B / np.linalg.norm(_BCC_SLIP_B, axis=1)[:, None]
    n_sys = _BCC_SLIP_N / np.linalg.norm(_BCC_SLIP_N, axis=1)[:, None]
    nsys = b_sys.shape[0]

    origin = np.array(cell.origin)
    h = np.array(cell.h)
    nodes, segs, loop_type = [], [], []

    # ---------- 1) 75% 可动直线位错(穿周期盒)----------
    acc_line_m = 0.0
    placed_line = 0
    attempt = 0
    while acc_line_m < 0.99 * L_line_m and attempt < 100000:
        isys = placed_line % nsys
        burg, plane = b_sys[isys], n_sys[isys]
        pos = origin + np.matmul(rng.rand(3), h.T)
        Lline_b = insert_infinite_line(cell, nodes, segs, burg, plane, pos, trial=True)
        if Lline_b is None or Lline_b < 0:
            attempt += 1
            continue
        nseg0 = len(segs)
        nodes, segs = insert_infinite_line(cell, nodes, segs, burg, plane, pos)
        loop_type += [LOOP_TYPE_LINE] * (len(segs) - nseg0)
        acc_line_m += Lline_b * BURGMAG
        placed_line += 1
        attempt = 0
    if verbose:
        print('可动直线位错: %d 条, 线长 %.3e m (目标 %.3e, 75%%)' % (placed_line, acc_line_m, L_line_m))

    # ---------- 2) 25% Frank-Read 源(两端 PINNED)----------
    frs_len_b = FRS_LENGTH_M / BURGMAG
    margin = frs_len_b
    acc_frs_m = 0.0
    placed_frs = 0
    attempt = 0
    while acc_frs_m < 0.99 * L_frs_m and attempt < 100000:
        isys = placed_frs % nsys
        burg, plane = b_sys[isys], n_sys[isys]
        center = rng.uniform(margin, Lbox - margin, size=3)
        nseg0 = len(segs)
        nodes, segs = insert_frank_read_src(cell, nodes, segs, burg, plane, frs_len_b, center)
        loop_type += [LOOP_TYPE_FRS] * (len(segs) - nseg0)
        acc_frs_m += FRS_LENGTH_M
        placed_frs += 1
        attempt += 1
    if verbose:
        print('Frank-Read 源: %d 个, 线长 %.3e m (目标 %.3e, 25%%)' % (placed_frs, acc_frs_m, L_frs_m))

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
    print(f'段类别标记: {lt_file}  (0=直线 3=FR源, 共 {len(loop_type)} 段)')
    print(f'带标记 VTK: {vtk_file}  (ParaView 按 LoopType 染色)')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
