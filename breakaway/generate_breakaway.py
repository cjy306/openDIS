"""
breakaway 预生成:单根棱柱环 + 滑移面上的球形障碍 (BCC Fe),一起生成。

输出 init_breakaway/{init_config.data, obstacles.data, init_config.vtk},
供 test_breakaway_prismatic.py 读取。障碍摆在选定段的滑移面上(中点沿 ±b 偏移),
滑移面含 b 故偏移后仍在面内,环沿 b 平移必扫过 → 保证接触。

用法:  python generate_breakaway.py
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import insert_prismatic_loop, write_vtk

# ===== 参数(占位值,自行调整) =====
BURGMAG    = 0.248e-9    # b [m] —— Fe BCC a/2<111>
LBOX_M     = 1.5e-6      # 盒 1.5μm 立方
RADIUS_M   = 200e-9      # 棱柱环半径(τ_act≈μb/2R≈51MPa,每边~4段)
MAXSEG_B   = 200         # 离散段长上限 [b]
BURG_DIR   = np.array([1., 1., 1.])   # 柏氏矢量方向 <111>

N_SEG_OBS    = 3         # 放障碍的段数(每段 ±b 各一 → 共 2*N 个)
H_OFFSET_B   = 300       # 障碍沿 b 偏离环平面 [b]
OBS_RADIUS_B = 80        # 障碍半径/捕获横截面 [b]


def main():
    pyexadis.initialize()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(base_dir, 'init_breakaway')
    os.makedirs(out_dir, exist_ok=True)

    Lbox_b   = LBOX_M / BURGMAG
    radius_b = RADIUS_M / BURGMAG

    # 单根棱柱环(盒中心)
    cell   = pyexadis.Cell(Lbox_b)
    center = np.array(cell.origin) + np.matmul([0.5, 0.5, 0.5], np.array(cell.h).T)
    b_hat  = BURG_DIR / np.linalg.norm(BURG_DIR)
    nodes, segs = insert_prismatic_loop('bcc', cell, [], [], b_hat,
                                        radius_b, center, maxseg=MAXSEG_B)
    G = ExaDisNet(cell, nodes, segs)
    nodes_arr = np.array(nodes)
    print(f'盒 {LBOX_M*1e6:.1f}μm, 单根棱柱环 b={BURG_DIR.tolist()}, '
          f'半径 {RADIUS_M*1e9:.0f}nm = {radius_b:.0f}b')

    # 障碍:选定段中点沿 ±b 偏移,落在该段滑移面上
    pick = np.unique(np.linspace(0, len(segs) - 1, N_SEG_OBS, dtype=int))
    centers, radii = [], []
    for idx in pick:
        n1, n2 = int(segs[idx][0]), int(segs[idx][1])
        M = 0.5 * (nodes_arr[n1, :3] + nodes_arr[n2, :3])
        for sign in (+1.0, -1.0):
            centers.append(M + sign * H_OFFSET_B * b_hat)
            radii.append(float(OBS_RADIUS_B))
    centers, radii = np.array(centers), np.array(radii)
    print(f'放了 {len(centers)} 个障碍(滑移面上, ±{H_OFFSET_B}b, 半径 {OBS_RADIUS_B}b)')

    # 写出
    G.write_data(os.path.join(out_dir, 'init_config.data'))
    np.savetxt(os.path.join(out_dir, 'obstacles.data'),
               np.hstack([centers, radii[:, None]]),
               header='cx cy cz radius  (units of b)')
    write_vtk(DisNetManager(G), os.path.join(out_dir, 'init_config.vtk'),
              crystal='BCC', verbose=False)
    print(f'已写出: {out_dir}/{{init_config.data, obstacles.data, init_config.vtk}}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
