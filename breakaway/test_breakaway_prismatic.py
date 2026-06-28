"""
breakaway(切过)机制测试 —— 单根棱柱环 + 滑移面上的球形障碍 (BCC Fe)

验证 CollisionOrowan::handle_breakaway:
  位错段扫过障碍 → 障碍处插钉节点(v=0) → 两臂弓出、夹角减小 →
  |Σ单位臂向量| > 2cos(phi_crit/2) 时脱钉。phi_crit 用 C++ 默认 90°。

设计(单根、可控):
  - 在盒中心建 1 根 a/2[111] 棱柱环(法向=b,沿 ±b 保守滑移)。
  - 障碍摆在选定段的"滑移面上":取段中点 M,沿 ±b 偏移 H_OFFSET_B。
    滑移面含 b,故 M±h·b̂ 仍在该段滑移面内(垂直门 h²<r² 恒开),
    环沿 b 平移即扫过障碍 → 保证接触。±两侧都放,不管往哪滑都命中。

接线: collision_mode='Orowan' 才进 handle_breakaway。单位一律 b。
参数为占位值,自行调整。
用法:  python test_breakaway_prismatic.py
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
from pyexadis_utils import insert_prismatic_loop, write_vtk

# ============================================================
# 参数(占位值,自行调整)
# ============================================================
BURGMAG    = 0.248e-9    # b [m] —— Fe BCC a/2<111>
LBOX_M     = 1.5e-6      # 测试盒 1.5μm 立方(环/盒~7.5:1,减小PBC自作用)
RADIUS_M   = 200e-9      # 棱柱环半径 [m](τ_act≈μb/2R≈51MPa,每边~4段)
MAXSEG_B   = 200         # 离散段长上限 [b] (≈50nm)
BURG_DIR   = np.array([1., 1., 1.])   # 棱柱环柏氏矢量方向(<111>)

# 障碍(小夹杂,走 breakaway;phi_crit 用 C++ 默认 90°)
N_SEG_OBS    = 3         # 在几段上放障碍(每段沿 ±b 各一个 → 共 2*N 个)
H_OFFSET_B   = 300       # 障碍沿 b 偏离环平面的距离 [b](环滑移这么远后接触)
OBS_RADIUS_B = 80        # 障碍半径/捕获横截面 [b]

# 加载
ERATE       = 1e3        # 应变率 [1/s]
EDIR        = np.array([0., 0., 1.])   # 加载轴 [001]
MAX_STRAIN  = 0.01

# Fe BCC 材料/迁移率(占位)
state = {
    "crystal":   'bcc',
    "burgmag":   BURGMAG,
    "mu":        82e9,
    "nu":        0.29,
    "a":         4.0,
    "maxseg":    MAXSEG_B,
    "minseg":    50,
    "rtol":      1.0,
    "rann":      2.0,
    "nextdt":    1e-9,
    "maxdt":     1e-8,
    "use_glide_planes":       1,
    "num_bcc_plane_families": 1,
}


def build_single_loop(Lbox_b, radius_b):
    """在盒中心建单根 a/2[111] 棱柱环,返回 (G, nodes, segs, b_hat, center)。"""
    cell = pyexadis.Cell(Lbox_b)
    origin = np.array(cell.origin)
    h      = np.array(cell.h)
    center = origin + np.matmul(np.array([0.5, 0.5, 0.5]), h.T)

    burg = BURG_DIR / np.linalg.norm(BURG_DIR)
    nodes, segs = [], []
    nodes, segs = insert_prismatic_loop('bcc', cell, nodes, segs, burg,
                                        radius_b, center, maxseg=MAXSEG_B)
    G = ExaDisNet(cell, nodes, segs)
    return G, np.array(nodes), segs, burg, center


def place_obstacles_on_glide(nodes_arr, segs, b_hat, n_seg, h_off, radius_b):
    """在 n_seg 个均匀选取的段的滑移面上(中点 ±h·b̂)放等半径障碍。"""
    nseg = len(segs)
    pick = np.unique(np.linspace(0, nseg - 1, n_seg, dtype=int))
    centers, radii = [], []
    for idx in pick:
        s = segs[idx]
        n1, n2 = int(s[0]), int(s[1])
        M = 0.5 * (nodes_arr[n1, :3] + nodes_arr[n2, :3])
        for sign in (+1.0, -1.0):
            centers.append(M + sign * h_off * b_hat)
            radii.append(float(radius_b))
    return np.array(centers), np.array(radii)


def main():
    pyexadis.initialize()

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output_breakaway_single')
    os.makedirs(output_dir, exist_ok=True)

    Lbox_b   = LBOX_M / BURGMAG
    radius_b = RADIUS_M / BURGMAG
    print(f'盒 {LBOX_M*1e6:.1f}μm, 单根棱柱环 b={BURG_DIR.tolist()}, '
          f'半径 {RADIUS_M*1e9:.0f}nm = {radius_b:.0f}b')

    # --- 单根棱柱环 ---
    G, nodes_arr, segs, b_hat, center = build_single_loop(Lbox_b, radius_b)

    # --- 障碍摆在滑移面上 ---
    centers_b, radii_b = place_obstacles_on_glide(nodes_arr, segs, b_hat,
                                                  N_SEG_OBS, H_OFFSET_B, OBS_RADIUS_B)
    G.load_obstacles([list(c) for c in centers_b], list(radii_b))
    print(f'放了 {len(centers_b)} 个障碍(滑移面上, ±{H_OFFSET_B}b, 半径 {OBS_RADIUS_B}b)')

    # 存初始构型 + 障碍, 便于 ParaView 对照
    write_vtk(DisNetManager(G), os.path.join(output_dir, 'init_config.vtk'),
              crystal='BCC', verbose=False)
    np.savetxt(os.path.join(output_dir, 'obstacles.data'),
               np.hstack([centers_b, radii_b[:, None]]),
               header='cx cy cz radius  (units of b)')

    net  = DisNetManager(G)
    cell = net.get_disnet(ExaDisNet).cell

    # --- 模块: collision_mode='Orowan' 触发 handle_breakaway ---
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state,
                            Medge=15000.0, Mscrew=3000.0, Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[], state=state,
                                force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Orowan', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state,
                         force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        erate=ERATE,
        edir=EDIR,
        max_strain=MAX_STRAIN,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=1,
        write_dir=output_dir,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
