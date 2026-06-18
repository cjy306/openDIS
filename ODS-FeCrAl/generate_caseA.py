"""
生成 Case A 初始配置:纯棱柱环基体(无障碍物)—— ODS-FeCrAl 课题

Case A = 纯基体,无 α′ / 无氧化物,是 B~F 的公共底座。
基体 = 多个 a/2<111> 棱柱位错环(用 pyexadis 自带 generate_prismatic_config):
  棱柱环柏氏矢量有面外分量,沿滑移柱面保守运动(不向心坍缩、不像无限直线无阻平移)。
  4 个 a/2<111> 柏氏矢量轮替,随机位置。

尺度说明(2026-06 重定):
  - 盒子 2μm 立方(bulk RVE,不绑实验柱尺寸;由"ρ + 合理环尺寸 + 足够环数"反推)。
    300nm 是早期误抄 Pachaury 纳米柱尺寸,bulk RVE 不需要等于实验试样。
  - ρ=5e12 /m^2。2μm 盒下总线长 ~4e4 nm → 配直径 40nm 环 → ~320 个环(统计够)。

演化历史(教训,见 CLAUDE_FeCrAl.md §4):
  ① 无钉扎闭合滑移环 → 线张力收缩自湮灭
  ② 穿盒无限直线 → 整体无阻平移,应力摁死 1MPa、密度死平
  ③ FR 源(开放/官方回路)→ 饥饿或激活应力问题
  ④ 棱柱环(本文件)→ 沿柱面保守运动,既不坍缩也不无阻平移

用法:
  python generate_caseA.py --seed 12345
  → 写 init_data_caseA_seed12345/{init_config.data, init_config_labeled.vtk}
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import generate_prismatic_config, write_vtk

# ============================================================
# 参数(材料 = Yan 2023 一套)
# ============================================================
BURGMAG    = 0.248e-9    # b [m]
LBOX_M     = 2.0e-6      # bulk 周期盒 2μm 立方
RHO_TARGET = 5.0e12      # 基体位错密度 [1/m^2]
RADIUS_M   = 38e-9       # 棱柱环半径 [m](直径 76nm;每环周长~240nm,配 maxseg=160b 约6段)
MAXSEG_B   = 160         # 离散段长上限 [b](≈40nm,与 test 脚本 maxseg 一致)


def _num_loops(rho, Lbox_m, radius_m):
    """由目标密度反算棱柱环数量:总线长 ρ·V = num * 2πR。"""
    L_target = rho * Lbox_m**3          # 总线长 [m]
    per_loop = 2.0 * np.pi * radius_m   # 单环周长 [m]
    return max(1, int(round(L_target / per_loop)))


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成 Case A 棱柱环基体构型')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_data_caseA_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    # 无量纲化(以 b 为单位)
    Lbox_b   = LBOX_M / BURGMAG
    radius_b = RADIUS_M / BURGMAG
    num_loops = _num_loops(RHO_TARGET, LBOX_M, RADIUS_M)
    print(f'盒 {LBOX_M*1e6:.1f}μm, ρ={RHO_TARGET:.1e}/m^2 → {num_loops} 个棱柱环 '
          f'(半径 {RADIUS_M*1e9:.0f}nm = {radius_b:.0f}b)')

    G = generate_prismatic_config('bcc', Lbox_b, num_loops, radius_b,
                                  maxseg=MAXSEG_B, seed=args.seed)
    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)

    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file, crystal='BCC', verbose=False)

    print(f'Case A 初始构型已写出: {out_file}')
    print(f'VTK: {vtk_file}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
