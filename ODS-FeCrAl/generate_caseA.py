"""
生成 Case A 初始配置:混合基体位错网络(无障碍物)—— ODS-FeCrAl 课题

Case A = 纯基体,无环 / 无 α′ / 无氧化物,是 B~F 的公共底座。
基体配方(沿用 Pachaury 2023 的 75%直线+25%单臂源,均以两端钉扎 FR 段实现):
  - 75% 长 FR 段(~200nm,接近盒边):加载即大段弓出滑移 → 提供渐进微塑性,曲线平滑
  - 25% 短 FR 段(~80nm):弓出到临界应力增殖放环 → 保证位错密度供给
  两者都是 insert_frank_read_src(两端 PINNED),只是长度不同;长度可控 → 精确配密度。

为何不用其他形式(历史教训,见 CLAUDE_FeCrAl.md §4):
  - 无钉扎闭合滑移环:线张力向心收缩,加载两步内全自湮灭(实测 2600→312→0)。
  - 穿盒无限直线:无钉扎可动,但长度锁死,纳米盒里一条就 ~5e15/m² 远超目标密度。
  结论:纳米盒里"可动+长度可控+稳定"必须有端点 → 用两端钉扎 FR 段(钉扎不影响屈服物理)。

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
from pyexadis_utils import insert_frank_read_src, write_vtk

# ============================================================
# 参数(材料 = Yan 2023 一套;尺度/密度 = Pachaury 2023)
# ============================================================
BURGMAG      = 0.248e-9    # b [m]
LBOX_M       = 300e-9      # bulk 周期盒 300nm 立方(Pachaury 纳米尺度)
RHO_TARGET   = 2.0e14      # 基体总位错密度 [1/m^2](Pachaury ~1.8-2.2e14)
LONG_FRAC    = 0.75        # 长 FR 段占总线长比例(微塑性)
LONG_LEN_M   = 120e-9      # 长 FR 段长度 [m](盒子的 0.4,留弓出空间)
SHORT_LEN_M  = 80e-9       # 短 FR 段长度 [m](增殖)

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

LT_LONG  = 0   # 长 FR 段(微塑性)
LT_SHORT = 3   # 短 FR 段(增殖)


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
    """生成基体(75% 长 FR 段 + 25% 短 FR 段),返回 (nodes, segs, loop_type)。

    loop_type 按段顺序:0=长FR段, 3=短FR段(Case B 另用 1=<111>环, 2=<100>环,不冲突)。
    """
    vol_m3 = LBOX_M ** 3
    L_total_m = RHO_TARGET * vol_m3
    L_long_m  = LONG_FRAC * L_total_m
    L_short_m = (1.0 - LONG_FRAC) * L_total_m

    b_sys = _BCC_SLIP_B / np.linalg.norm(_BCC_SLIP_B, axis=1)[:, None]
    n_sys = _BCC_SLIP_N / np.linalg.norm(_BCC_SLIP_N, axis=1)[:, None]

    nodes, segs, loop_type = [], [], []
    nodes, segs, loop_type = _fill_fr(cell, rng, nodes, segs, loop_type, L_long_m,
                                      LONG_LEN_M, LT_LONG, b_sys, n_sys, '长FR段(微塑性)', verbose)
    nodes, segs, loop_type = _fill_fr(cell, rng, nodes, segs, loop_type, L_short_m,
                                      SHORT_LEN_M, LT_SHORT, b_sys, n_sys, '短FR段(增殖)', verbose)
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
    print(f'段类别标记: {lt_file}  (0=长FR段 3=短FR段, 共 {len(loop_type)} 段)')
    print(f'带标记 VTK: {vtk_file}  (ParaView 按 LoopType 染色)')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
