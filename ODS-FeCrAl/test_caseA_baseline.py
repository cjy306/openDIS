"""
Case A —— ODS-FeCrAl 基准工况(无障碍物,纯可动滑移位错网络屈服)
初始构型从 init_data_caseA_seed<seed>/init_config.data 读取
(先运行 generate_glide_loop.py 生成该文件)

材料参数 = Yan 2023 一套(arXiv:2309.03703,见 CLAUDE_FeCrAl.md §3.5):
  b=0.248 nm, mu=81 GPa, nu=0.3
加载:单轴应变率控制,[001] 方向;屈服由 extract_yield.py 后处理(0.2% offset)
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')


state = {
    "crystal":   'bcc',
    "burgmag":   0.248e-9,
    "mu":        81e9,
    "nu":        0.3,
    "a":         2.0,
    "maxseg":    80,
    "minseg":    20,
    "rtol":      0.5,
    "rann":      1.0,
    "nextdt":    1e-9,
    "maxdt":     1e-7,
    "use_glide_planes":       1,
    "num_bcc_plane_families": 1,
}


def main():
    pyexadis.initialize()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345, help='随机实现种子(对应 init_data 目录)')
    parser.add_argument('--restart', type=int, help='从指定步骤重启')
    args = parser.parse_args()

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = os.path.join(base_dir, f'init_data_caseA_seed{args.seed}')
    output_dir = os.path.join(base_dir, f'output_caseA_seed{args.seed}')
    os.makedirs(output_dir, exist_ok=True)

    if args.restart is not None:
        net, restart = read_restart(
            state=state,
            restart_file=os.path.join(output_dir, f'restart.{args.restart}.exadis'))
    else:
        G = ExaDisNet()
        G.read_paradis(os.path.join(init_dir, 'init_config.data'))
        net = DisNetManager(G)
        restart = None

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=32, cell=net.get_disnet(ExaDisNet).cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=15000.0, Mscrew=3000.0, Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 10.0, 25.0, 50.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        erate=1e3,
        edir=np.array([0., 0., 1.]),
        max_strain=0.005,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=10,
        write_dir=output_dir,
        restart=restart,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
