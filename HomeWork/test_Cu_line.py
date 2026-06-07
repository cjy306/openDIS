"""模拟脚本（无限直线初始构型）：读 init_data_line/，输出 output_Cu_line/

与 test_Cu_twin.py 的力/积分器/参数完全一致，唯一区别是初始位错构型用
无限直线（可动、Burgers守恒）替代 FR 源，用于检验"屈服骤降"是否消失。

erate 设为变量：默认 1e3，便于和 FR 源的 1e3 曲线直接对比。
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


ERATE = 1e3   # 应变率：默认 1e3（与 FR 1e3 曲线对比）；想让位错动得更欢可改 1e4

# 坐标系旋转矩阵（与 test_Cu_twin.py 一致）
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
    init_dir   = os.path.join(base_dir, 'init_data_line')
    output_dir = os.path.join(base_dir, 'output_Cu_line')
    os.makedirs(output_dir, exist_ok=True)

    # 加载杂质
    obs_data = np.loadtxt(os.path.join(init_dir, 'obstacles.data'))
    if obs_data.ndim == 1:
        obs_data = obs_data.reshape(1, -1)
    centers_b = obs_data[:, :3]
    radii_b   = obs_data[:, 3]
    print(f"Loaded {len(centers_b)} precipitates")

    # 加载孪晶面
    tp_data = np.loadtxt(os.path.join(init_dir, 'twin_planes.data'))
    if tp_data.ndim == 1:
        tp_data = tp_data.reshape(1, -1)
    twin_points_b  = tp_data[:, :3].tolist()
    twin_normals_b = tp_data[:, 3:].tolist()
    print(f"Loaded {len(twin_points_b)} twin boundary planes")

    # 读初始构型（无限直线）
    G = ExaDisNet()
    G.read_paradis(os.path.join(init_dir, 'init_config.data'))
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
        edir=np.array([0.0, -2.0/np.sqrt(6), 1.0/np.sqrt(3)]),  # [001] crystal direction in rotated frame
        max_strain=0.01,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=10,
        write_dir=output_dir,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
