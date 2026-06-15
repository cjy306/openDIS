"""验证实验：纯弛豫(零外力)能否消除 FR 源的屈服骤降
   —— 第 2 段：对弛豫后的构型加载

读第 1 段(test_relax_step1.py)弛豫后的末态构型，从零应力加载到 1%，
画应力-应变看骤降在不在。

判读：
  - 仍有大骤降(类似 357→低 或 129→低) → 证实【纯弛豫不能消骤降】
    (零外力弛豫不放可动位错，加载时 FR 仍饥饿) —— 我预期是这个结果
  - 骤降消失 → 推翻我的判断，纯弛豫也能消骤降(那说明弛豫期 FR 在内应力下放了位错)

用法：
  1. 先跑 test_relax_step1.py 得到 output_relax/ 里的末态 config
  2. 把【最后一个】 config.*.data 路径填到下面 RELAXED_CONFIG
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


# ★ 填入第1段弛豫输出的【最后一个】 config 文件名（在 output_relax/ 里）
RELAXED_CONFIG = 'config.2000.data'   # 按实际末态步数改
ERATE = 1e4

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

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = os.path.join(base_dir, 'init_data_twin_rorient')   # 障碍物/孪晶面来源(不变)
    relax_dir  = os.path.join(base_dir, 'output_relax')
    output_dir = os.path.join(base_dir, 'output_relax_load')
    os.makedirs(output_dir, exist_ok=True)

    obs_data = np.loadtxt(os.path.join(init_dir, 'obstacles.data'))
    if obs_data.ndim == 1:
        obs_data = obs_data.reshape(1, -1)
    centers_b, radii_b = obs_data[:, :3], obs_data[:, 3]

    tp_data = np.loadtxt(os.path.join(init_dir, 'twin_planes.data'))
    if tp_data.ndim == 1:
        tp_data = tp_data.reshape(1, -1)
    twin_points_b  = tp_data[:, :3].tolist()
    twin_normals_b = tp_data[:, 3:].tolist()

    # 读弛豫后的构型
    G = ExaDisNet()
    G.read_paradis(os.path.join(relax_dir, RELAXED_CONFIG))
    net = DisNetManager(G)
    exadis_net = net.get_disnet(ExaDisNet)
    if len(centers_b) > 0:
        exadis_net.load_obstacles([list(c) for c in centers_b], list(radii_b))
    if len(twin_points_b) > 0:
        exadis_net.load_twin_planes(twin_points_b, twin_normals_b)

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=exadis_net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=50.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 10.0, 60.0, 200.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Orowan', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        erate=ERATE,
        edir=np.array([0.0, -2.0/np.sqrt(6), 1.0/np.sqrt(3)]),
        max_strain=0.01,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=100,
        write_dir=output_dir,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
