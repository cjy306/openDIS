"""测试：循环子应力 + 循环子积分 + 旋转(Rorient)，单根可开动 FR 源

目的：在与正式模拟 test_Cu_twin.py 完全一致的力/积分器/旋转配置下，
隔离测试单根 Frank-Read 源（无孪晶面、无杂质），验证碰撞 swap 修复后
子循环积分下的折线问题是否消失。

要点：
  - 力模块  = SUBCYCLING_MODEL（循环子应力），与正式一致
  - 积分器  = Subcycling + rgroups=[0,10,60,200]（循环子积分），与正式一致
  - Rorient = 与正式一致（(111) 旋转为 z=const）
  - 碰撞    = Orowan（继承自 Retroactive，折线修复就在 Retroactive::handle 内；
              无杂质/孪晶时 Orowan/孪晶部分为空操作）
  - FR 源放在盒子中心，用 (111)[0,-1,1] 滑移系（[001]加载 Schmid≈0.408），
    可正常开动并穿过 PBC 边界（即原折线出现的场景）
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e

from pyexadis_utils import insert_frank_read_src


# 坐标系旋转矩阵（与正式模拟 test_Cu_twin.py 完全一致）
# [100]/[010]/[001] → [1-10]/[11-2]/[111]，使 (111) 在新坐标系中为 z=const
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

    Lbox_m = 5e-6
    Lbox = int(round(Lbox_m / burgmag))
    cell = pyexadis.Cell(Lbox)

    # 可开动滑移系：晶体 (111)[0,-1,1]，[001]加载 Schmid≈0.408
    # 转到旋转系（与 generate_config.py 一致：burg = R @ b, plane = R @ n）
    b_crystal = np.array([0, -1, 1]) / np.sqrt(2)
    n_crystal = np.array([1,  1, 1]) / np.sqrt(3)
    burg  = R @ b_crystal
    plane = R @ n_crystal
    plane = plane / np.linalg.norm(plane)
    # 旋转保内积，b·n = 0，几何自洽（不会触发 not-orthogonal 警告）

    # FR 源放在盒子中心，长度 1µm（足够开动并会穿过 PBC 边界）
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

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_pure_fr_subcycling_rorient')
    os.makedirs(output_dir, exist_ok=True)

    # 力/积分器：与正式模拟一致（循环子应力 + 循环子积分）
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=exadis_net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=50.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 10.0, 60.0, 200.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Orowan', state=state)  # 继承 Retroactive，折线修复在其内
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
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
