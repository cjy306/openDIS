"""
breakaway 验证构型:单根刃型直线 + 滑移面上随机撒的一片点障碍 (BCC Fe)

用途:看一根位错线**同时**与多个点障碍作用 —— 在障碍林里弓出、逐个突破。
这是从"单个能钉"走向"一片给出集体突破应力 τ_c"的第一步(本脚本只造构型,先看现象)。
障碍全撒在线的滑移面内(h=0)、全在盒内、等强度(phi_crit 用 C++ 默认 90°)。

与 generate_breakaway_line.py / generate_breakaway.py 同目录、互斥:跑哪个生成器,
test_breakaway_prismatic.py 就读哪个(不用改测试脚本)。
输出 init_breakaway/{init_config.data, obstacles.data, init_config.vtk}。单位一律 b。

用法:  python generate_breakaway_field.py
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

N_OBS        = 40        # 滑移面上随机点障碍数(整片;调大 -> 更密 -> 间距 L 更小)
OBS_RADIUS_B = 40        # 障碍捕获半径 [b](点障碍,取小)
RNG_SEED     = 0         # 随机种子(固定 -> 可复现)


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

    # 单根刃型直无限线(PBC 周期),过盒中心 origin
    nodes, segs = insert_infinite_line(cell, [], [], b_hat, plane_hat, origin,
                                       theta=THETA, maxseg=MAXSEG_B, trial=False)
    G = ExaDisNet(cell, nodes, segs)
    print(f'盒 {LBOX_M*1e6:.1f}μm, 单根刃型直线 b={BURG_DIR.tolist()}, 面 {PLANE_DIR.tolist()}')

    # 滑移面内正交基:e1=线方向(刃型 theta=90 时 = cross(n, b̂)), e2=滑移方向 b̂
    # 二者都 ⊥ 面法向(在面内)且互相 ⊥,故 (s,t)->origin+s*e1+t*e2 是面上的等面积参数化
    e1 = np.cross(plane_hat, b_hat); e1 /= np.linalg.norm(e1)   # 线方向 [-2,1,1]/√6
    e2 = b_hat                                                  # 滑移方向

    # 在过 origin 的滑移面上随机撒点,拒绝盒外;因基在面内 -> 所有点 h=0(严格在面上)
    rng = np.random.default_rng(RNG_SEED)
    centers, tried = [], 0
    while len(centers) < N_OBS and tried < 100 * N_OBS:
        s, t = rng.uniform(-Lbox_b, Lbox_b, size=2)
        p = origin + s * e1 + t * e2
        tried += 1
        if np.all((p >= 0.0) & (p <= Lbox_b)):
            centers.append(p)
    centers = np.array(centers)
    radii   = np.full(len(centers), float(OBS_RADIUS_B))

    # 估计面内数密度与平均间距 L(FM 统计要用):接受率 × 采样方面积 = 盒内面面积
    A_plane = (len(centers) / tried) * (2 * Lbox_b) ** 2
    L_mean  = np.sqrt(A_plane / len(centers)) if len(centers) else 0.0
    print(f'撒了 {len(centers)} 个点障碍(滑移面内 h=0, 捕获半径 {OBS_RADIUS_B}b, seed={RNG_SEED})')
    print(f'  估计面内平均间距 L ≈ {L_mean:.0f} b = {L_mean*BURGMAG*1e9:.1f} nm')

    # 写出(与其他生成器同格式)
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
