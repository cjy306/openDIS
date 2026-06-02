"""判定实验：大盒子单根 FR 源（循环子 + 旋转）—— 判断折线是不是 PBC 穿越造成的

设计：与 test_pure_fr_subcycling_rorient.py 完全一致，唯一区别是
  - 盒子放大 4 倍：5µm → 20µm
  - Ngrid 同比例放大：64 → 256（保持 FFT 网格分辨率 ~78nm/cell 不变）
源长度仍 1µm，放盒子中心。环膨胀到 ~2-4µm 时，距离 20µm 边界还很远，
**整个过程不会穿过 PBC 边界**。

判读：
  - 折线消失 → 确诊是 PBC 穿越（机制A 自镜像 / 边界处力·碰撞不一致）
  - 折线还在 → 排除 PBC，根因在循环子 / remesh 本身

与 5µm 版（test_pure_fr_subcycling_rorient.py，会穿 PBC、出折线）构成 A/B 对照。
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

    Lbox_m = 20e-6          # 大盒子：5µm → 20µm（4倍），让环不穿 PBC
    Lbox = int(round(Lbox_m / burgmag))
    cell = pyexadis.Cell(Lbox)

    # 可开动滑移系：晶体 (111)[0,-1,1]，[001]加载 Schmid≈0.408
    b_crystal = np.array([0, -1, 1]) / np.sqrt(2)
    n_crystal = np.array([1,  1, 1]) / np.sqrt(3)
    burg  = R @ b_crystal
    plane = R @ n_crystal
    plane = plane / np.linalg.norm(plane)

    # FR 源放盒子中心，长度仍 1µm（与 5µm 版相同），但环膨胀远不到 20µm 边界
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

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_pure_fr_bigbox')
    os.makedirs(output_dir, exist_ok=True)

    # 力/积分器：与 5µm 版完全一致，仅 Ngrid 同比例 64→256 保持网格分辨率
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=32, cell=exadis_net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=50.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 10.0, 60.0, 200.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Orowan', state=state)
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
