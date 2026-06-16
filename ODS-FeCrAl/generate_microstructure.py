"""
混合基体位错网络生成器 —— ODS-FeCrAl 课题 (Case A / B 基体共用)

配方 (沿用 Pachaury 2023):
  - 75% 可动直线位错 (穿周期盒,首尾经 PBC 相接;UNCONSTRAINED 全可动)
    → 从第 0 步即全长可动,加载即产生渐进微塑性,应力-应变曲线平滑
  - 25% Frank-Read 源 (两端 PINNED)
    → 加载下弓出增殖,保证大应变下位错密度持续供给

为何不用闭合滑移环 (历史教训,见 CLAUDE_FeCrAl.md §4):
  无钉扎闭合滑移环受线张力向心收缩,零外力下稳态=缩成零 → 加载两步内全部自湮灭
  (实测 nodes 2600→312→0)。"可动+无钉扎+稳定" 三者中闭合环必丢稳定;
  直线穿周期盒去掉"闭合"即可三者兼得 → 故基体改用 直线(主)+FR(增殖) 混合。

数据契约同 ExaDiS (节点 [x,y,z,constraint];段 [n1,n2, b, plane])。
两个底层插入函数直接复用 ExaDiS 自带:
  insert_infinite_line (直线,UNCONSTRAINED) / insert_frank_read_src (FR,两端 PINNED)

loop_type 标记 (供可视化/分析区分基体子类):
  0 = 直线可动位错;  3 = Frank-Read 源
  (Case B 另用 1=a/2<111>环, 2=a<100>环;数字不冲突)
"""
import numpy as np
from pyexadis_utils import insert_infinite_line, insert_frank_read_src

# BCC 12 个 <111>{110} 滑移系 (b 在 plane 内,b·n=0)
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


def build_matrix(cell, Lbox_m, burgmag, target_density, rng,
                 frs_fraction=0.25, frs_length_m=1.0e-6, maxseg=-1, verbose=True):
    """生成混合基体位错网络,追加到新的 nodes/segs 并返回 (nodes, segs, loop_type)。

    参数:
      cell:           pyexadis.Cell
      Lbox_m:         盒边长 [m]
      burgmag:        b [m]
      target_density: 基体总位错密度 [1/m^2]
      rng:            np.random.RandomState (由调用方按 seed 建,保证多实现可复现)
      frs_fraction:   FR 源占总线长比例 (0.25 = 25%)
      frs_length_m:   单个 FR 源长度 [m]
      maxseg:         离散段长上限 (无量纲;-1 用 insert 函数默认)

    返回: (nodes, segs, loop_type)  loop_type 按段顺序,值 0(直线)/3(FR)
    """
    Lbox = Lbox_m / burgmag
    vol_m3 = Lbox_m ** 3
    L_total_m = target_density * vol_m3
    L_frs_m  = frs_fraction * L_total_m
    L_line_m = (1.0 - frs_fraction) * L_total_m

    b_sys = _BCC_SLIP_B / np.linalg.norm(_BCC_SLIP_B, axis=1)[:, None]
    n_sys = _BCC_SLIP_N / np.linalg.norm(_BCC_SLIP_N, axis=1)[:, None]
    nsys = b_sys.shape[0]

    origin = np.array(cell.origin)
    h = np.array(cell.h)

    nodes, segs, loop_type = [], [], []

    # ---------- 1) 75% 可动直线位错 (穿周期盒) ----------
    acc_line_m = 0.0
    placed_line = 0
    attempt = 0
    while acc_line_m < 0.99 * L_line_m and attempt < 100000:
        isys = placed_line % nsys
        burg, plane = b_sys[isys], n_sys[isys]
        # 随机起点 (盒内任意),线沿 PBC 自动闭合
        pos = origin + np.matmul(rng.rand(3), h.T)
        nseg0 = len(segs)
        # 先 trial 求该线的 PBC 周期长度 (无量纲) → 折算线长 [m]
        Lline_b = insert_infinite_line(cell, nodes, segs, burg, plane, pos,
                                       maxseg=maxseg, trial=True)
        if Lline_b is None or Lline_b < 0:
            attempt += 1
            continue
        nodes, segs = insert_infinite_line(cell, nodes, segs, burg, plane, pos, maxseg=maxseg)
        loop_type += [LOOP_TYPE_LINE] * (len(segs) - nseg0)
        acc_line_m += Lline_b * burgmag
        placed_line += 1
        attempt = 0
    if verbose:
        print('可动直线位错: %d 条, 线长 %.3e m (目标 %.3e, 75%%)' % (placed_line, acc_line_m, L_line_m))

    # ---------- 2) 25% Frank-Read 源 (两端 PINNED) ----------
    frs_len_b = frs_length_m / burgmag
    margin = frs_len_b
    acc_frs_m = 0.0
    placed_frs = 0
    attempt = 0
    while acc_frs_m < 0.99 * L_frs_m and attempt < 100000:
        isys = placed_frs % nsys
        burg, plane = b_sys[isys], n_sys[isys]
        center = rng.uniform(margin, Lbox - margin, size=3)
        nseg0 = len(segs)
        nodes, segs = insert_frank_read_src(cell, nodes, segs, burg, plane,
                                            frs_len_b, center)
        loop_type += [LOOP_TYPE_FRS] * (len(segs) - nseg0)
        acc_frs_m += frs_length_m
        placed_frs += 1
        attempt += 1
    if verbose:
        print('Frank-Read 源: %d 个, 线长 %.3e m (目标 %.3e, 25%%)' % (placed_frs, acc_frs_m, L_frs_m))

    return nodes, segs, np.array(loop_type, dtype=int)
