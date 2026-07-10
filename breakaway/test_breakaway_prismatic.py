"""
breakaway(切过)机制测试 —— 读预生成的环+障碍,加载运行 (BCC Fe)

先跑 generate_breakaway.py 生成 init_breakaway/{init_config.data, obstacles.data}。
本脚本只负责:读构型 → 读障碍并 load_obstacles → 加载运行。
collision_mode='Orowan' 才进含 handle_breakaway 的 CollisionOrowan。
phi_crit 用 C++ 默认 90°(load_obstacles 不显式传)。单位一律 b。

用法:  python test_breakaway_prismatic.py
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh

# ===== 加载参数(占位值,自行调整) =====
ERATE      = 1e3                       # 应变率 [1/s]
EDIR       = np.array([0., 0., 1.])    # 加载轴 [001](Schmid≈0.408,环沿[111]滑移)
MAX_STEP   = 10000                      # 跑满 5000 步就停(停机判据用步数,不用应变)

state = {
    "crystal":   'bcc',
    "burgmag":   0.248e-9,
    "mu":        82e9,
    "nu":        0.29,
    "a":         4.0,
    "maxseg":    200,
    "minseg":    50,
    "rtol":      1.0,
    "rann":      2.0,
    "nextdt":    1e-9,
    "maxdt":     1e-8,
    "use_glide_planes":       1,
    "num_bcc_plane_families": 1,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', type=int, help='从指定步骤重启(读 output 里的 restart.<步号>.exadis)')
    args = parser.parse_args()

    pyexadis.initialize()
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = os.path.join(base_dir, 'init_breakaway')
    output_dir = os.path.join(base_dir, 'output_breakaway_single')
    os.makedirs(output_dir, exist_ok=True)

    # 读构型: 重启 或 从预生成 init 起步
    if args.restart is not None:
        net, restart = read_restart(
            state=state,
            restart_file=os.path.join(output_dir, f'restart.{args.restart}.exadis'))
    else:
        G = ExaDisNet()
        G.read_paradis(os.path.join(init_dir, 'init_config.data'))
        net = DisNetManager(G)
        restart = None

    # 读障碍并加载(重启/起步都要;障碍存在 System 上,不随 restart 恢复;默认 phi_crit=90°)
    # type=1(breakaway 切过型点障碍):只走 handle_breakaway 钉/脱,不受 handle_orowan 硬球投影
    obs = np.loadtxt(os.path.join(init_dir, 'obstacles.data'))
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    centers_b, radii_b = obs[:, :3], obs[:, 3]
    net.get_disnet(ExaDisNet).load_obstacles([list(c) for c in centers_b], list(radii_b), type=1)
    print(f'读入 {len(centers_b)} 个障碍')

    cell = net.get_disnet(ExaDisNet).cell

    # collision_mode='Orowan' 触发 handle_breakaway
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=32, cell=cell)
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
        max_step=MAX_STEP,

        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=1,
        write_dir=output_dir,
        restart=restart,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
