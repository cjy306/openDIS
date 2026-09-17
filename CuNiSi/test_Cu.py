"""
Cu FCC 位错动力学模拟：球形杂质（Orowan bypass）
初始构型从 init_data/init_config.data 读取
杂质数据从 init_data/obstacles.npz 读取
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e


state = {
    "crystal": 'fcc',
    "burgmag": 0.2556e-9,
    "mu":      48e9,
    "nu":      0.324,
    "a":       4.0,
    "maxseg":  200,
    "minseg":  50,
    "rtol":    1.0,
    "rann":    2.0,
    "nextdt":  1e-9,
    "maxdt":   1e-6,
}


def main():
    pyexadis.initialize()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', type=int, help='从指定步骤重启')
    args = parser.parse_args()

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = os.path.join(base_dir, 'init_data_precip')
    output_dir = os.path.join(base_dir, 'output_Cu_fcc')
    os.makedirs(output_dir, exist_ok=True)

    # 加载杂质数据
    obs_data = np.loadtxt(os.path.join(init_dir, 'obstacles.data'))
    if obs_data.ndim == 1:
        obs_data = obs_data.reshape(1, -1)
    centers_b = obs_data[:, :3]
    radii_b   = obs_data[:, 3]
    print(f"Loaded {len(centers_b)} precipitates")

    if args.restart is not None:
        net, restart = read_restart(
            state=state,
            restart_file=os.path.join(output_dir, f'restart.{args.restart}.exadis'))
    else:
        G = ExaDisNet()
        G.read_paradis(os.path.join(init_dir, 'init_config.data'))
        net = DisNetManager(G)
        restart = None

    # 加载球形杂质到 C++ System
    exadis_net = net.get_disnet(ExaDisNet)
    if len(centers_b) > 0:
        exadis_net.load_obstacles([list(c) for c in centers_b], list(radii_b))

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=32, cell=exadis_net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=20000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 50.0, 300.0, 800.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Orowan', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        erate=1e3,
        edir=np.array([0., 0., 1.]),
        max_strain=0.01,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=100,
        write_dir=output_dir,
        restart=restart,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
