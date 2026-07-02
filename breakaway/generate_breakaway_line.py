"""
breakaway 验证构型(最干净):单根刃型直无限位错线 + 滑移面上的点障碍 (BCC Fe)

为什么用直线:整条均匀滑移、待在单一滑移面、不变形(不像棱柱环胀成 stadium),
障碍摆在面上必被扫过 → 钉住/弓出/脱钉一目了然。就是文献的验证 setup。

输出 init_breakaway/{init_config.data, obstacles.data, init_config.vtk},
与棱柱环版 generate_breakaway.py **同目录、互斥**:跑哪个生成器,
test_breakaway_prismatic.py 就读哪个(不用改测试脚本)。

障碍:沿线取 N_OBS 个点,各沿 ±g(滑移方向 b̂) 偏 H_OFFSET_B,落在滑移面内。
phi_crit 用 C++ 默认 90°。单位一律 b。

用法:  python generate_breakaway_line.py
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_utils import insert_infinite_line, write_vtk

# ===== 参数(占位值,自行调整) =====
BURGMAG    = 0.248e-9    # b [m] —— Fe BCC a/2<111>
LBOX_M     = 1.0e-6      # 盒 1μm 立方
MAXSEG_B   = 200         # 离散段长上限 [b]
BURG_DIR   = np.array([1.,  1., 1.])    # 柏氏矢量 <111>
PLANE_DIR  = np.array([0., -1., 1.])    # 滑移面 {110},须 ⊥ b ([0,-1,1]·[1,1,1]=0 ✓)
THETA      = 90.0        # 位向角:90=刃型, 0=螺型

N_OBS        = 3         # 沿线放障碍的点数(每点 ±g 各一 → 共 2*N 个)
H_OFFSET_B   = 300       # 障碍沿滑移方向 g 偏离线的距离 [b](线滑这么远后接触)
OBS_RADIUS_B = 40        # 障碍捕获半径 [b](点障碍,取小;别小于每步位移)


def main():
    pyexadis.initialize()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(base_dir, 'init_breakaway')
    os.makedirs(out_dir, exist_ok=True)

    Lbox_b = LBOX_M / BURGMAG
    cell   = pyexadis.Cell(Lbox_b)
    origin = np.array(cell.center())

    b_hat     = BURG_DIR  / np.linalg.norm(BURG_DIR)
    plane_hat = PLANE_DIR / np.linalg.norm(PLANE_DIR)

    # 单根刃型直无限线(PBC 周期)
    nodes, segs = insert_infinite_line(cell, [], [], b_hat, plane_hat, origin,
                                       theta=THETA, maxseg=MAXSEG_B, trial=False)
    G = ExaDisNet(cell, nodes, segs)
    nodes_arr = np.array(nodes)
    print(f'盒 {LBOX_M*1e6:.1f}μm, 单根刃型直线 b={BURG_DIR.tolist()}, '
          f'面 {PLANE_DIR.tolist()}, {len(nodes_arr)} 节点')

    # 线方向 xi(刃型) 与滑移方向 g(=b̂);障碍沿 ±g 偏移,落在滑移面内
    g = b_hat
    # insert_infinite_line 的节点从盒中心一路铺到盒外的 originpbc(未折回),
    # 直接取锚点会把障碍摆到盒外。先筛出坐标在 [0,Lbox] 的盒内节点做锚点;
    # 沿面内 b̂ 偏移不离滑移面,偏移后再校验仍在盒内才收 -> 障碍全在盒内、且与
    # 旁边那段线共仿射滑移面(h≈0),不触发离面的 y/z 周期绕。
    inbox = np.all((nodes_arr[:, :3] >= 0.0) & (nodes_arr[:, :3] <= Lbox_b), axis=1)
    inbox_idx = np.where(inbox)[0]
    pick = inbox_idx[np.unique(np.linspace(0, len(inbox_idx) - 1, N_OBS, dtype=int))]
    centers, radii = [], []
    for idx in pick:
        M = nodes_arr[idx, :3]
        for sign in (+1.0, -1.0):
            C = M + sign * H_OFFSET_B * g
            if np.all((C >= 0.0) & (C <= Lbox_b)):
                centers.append(C)
                radii.append(float(OBS_RADIUS_B))
    centers, radii = np.array(centers), np.array(radii)
    print(f'放了 {len(centers)} 个障碍(盒内滑移面上, ±{H_OFFSET_B}b, 捕获半径 {OBS_RADIUS_B}b)')

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
