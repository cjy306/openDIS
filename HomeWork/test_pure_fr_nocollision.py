"""判定实验：5µm 单根 FR 源（循环子 + 旋转）+ 关闭碰撞
   —— 判断 PBC 折线是不是"自镜像伪碰撞(机制A)"造成的

设计：与 test_pure_fr_subcycling_rorient.py（5µm，会穿 PBC、出折线）完全一致，
唯一区别是 collision = None（关闭碰撞处理）。
单根 FR 源唯一的碰撞就是环自身/自镜像，所以关掉碰撞 = 排除伪碰撞。

判读（与 5µm 开碰撞版对照）：
  - 折线/扭结消失 → 确诊是【自镜像伪碰撞 机制A】，根治方向是 PBC 下的碰撞检测
  - 折线还在     → 排除碰撞，根因在力 / remesh 的跨边界处理

注意：关碰撞后环无法正常掐断放环，这是预期的；本实验只看"环鼓出穿 PBC 的
阶段还有没有折线"，不关心放环。
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Topology, Remesh
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e

from pyexadis_utils import insert_frank_read_src


# 坐标系旋转矩阵（与正式模拟 test_Cu_twin.py 完全一致）
Rorient = [
    [ 1/np.sqrt(2), -1/np.sqrt(2),  0           ],
    [ 1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
    [ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],
]

state = {
    "crystal": 'fcc',
    "Rorient": Rorient,
    "burgmag": 0.2556e-9,
    "mu":      48e9,
    "nu":      0.324,
    "a":       4.0,
    "maxseg":  200,
    "minseg":  50,
    "rtol":    1.0,
    "rann":    2.0,
    "nextdt":  1e-9,
    "maxdt":   1e-8,
}


def main():
    pyexadis.initialize()

    burgmag = state["burgmag"]
    R = np.array(Rorient)

    Lbox_m = 5e-6          # 与开碰撞版相同的 5µm 盒子（会穿 PBC）
    Lbox = int(round(Lbox_m / burgmag))
    cell = pyexadis.Cell(Lbox)

    # 可开动滑移系：晶体 (111)[0,-1,1]，[001]加载 Schmid≈0.408
    b_crystal = np.array([0, -1, 1]) / np.sqrt(2)
    n_crystal = np.array([1,  1, 1]) / np.sqrt(3)
    burg  = R @ b_crystal
    plane = R @ n_crystal
    plane = plane / np.linalg.norm(plane)

    # FR 源放盒子中心，长度 1µm（与开碰撞版相同）
    length_m = 1.0e-6
    length_b = length_m / burgmag
    center_b = np.array([Lbox/2, Lbox/2, Lbox/2])
    numnodes = max(3, int(round(length_b / 100.0)))

    nodes, segs = [], []
    nodes, segs = insert_frank_read_src(
        cell, nodes, segs, burg, plane, length_b, center_b, numnodes=numnodes)

    G = ExaDisNet(cell, nodes, segs)
    net = DisNetManager(G)
    exadis_net = net.get_disnet(ExaDisNet)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_pure_fr_nocollision')
    os.makedirs(output_dir, exist_ok=True)

    # 力/积分器：与开碰撞版完全一致
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=exadis_net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=50.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 10.0, 60.0, 200.0], state=state, force=calforce, mobility=mobility)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    # ★ 唯一区别：关闭碰撞（SimulateNetworkPerf 会自动用 collision_mode='None' 空操作）
    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=None,
        topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        erate=1e4,
        edir=np.array([0.0, -2.0/np.sqrt(6), 1.0/np.sqrt(3)]),  # [001] crystal direction in rotated frame
        max_strain=0.01,
        burgmag=burgmag,
        state=state,
        print_freq=1,
        write_freq=10,
        write_dir=output_dir,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
