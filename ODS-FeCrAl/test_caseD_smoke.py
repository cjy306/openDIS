"""
test_caseD_smoke.py —— Case D 冒烟测试:基体 + 硬球障碍(几何投影) vs 无障碍

读封版基体 + init_data_caseD_smoke/obstacles.data(type=0 Orowan 硬球),
应变率加载,与 test_caseA_baseline 同一套 state/模块/加载,唯二差别:
  1. load_obstacles(type=0) 注入硬球
  2. collision_mode='Orowan'(激活几何投影;空表时行为同基线)
对照组 = output_caseA_high(同基体、无障碍、Retroactive)。
判据:σ-ε 曲线叠加看屈服抬升;量级参考见 generate 脚本打印的 Orowan 估计。
⚠️ 冒烟定位:无限硬障碍上限,回答"升不升",数字不进论文。

用法(昆山 GPU):
  python generate_caseD_smoke.py     # 先生成障碍场(一次)
  python test_caseD_smoke.py
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
except ImportError:
    raise ImportError('Cannot import pyexadis')


state = {
    "crystal":   'bcc',
    "burgmag":   0.248e-9,
    "mu":        81e9,
    "nu":        0.3,
    "a":         3.0,
    "maxseg":    160,
    "minseg":    40,
    "rtol":      0.75,
    "rann":      1.5,
    "nextdt":    1e-9,
    "maxdt":     1e-8,
    "use_glide_planes":       1,
    "num_bcc_plane_families": 2,
}


def main():
    pyexadis.initialize()
    import argparse
    p = argparse.ArgumentParser(description='Case D 冒烟:基体+硬球障碍,应变率加载')
    p.add_argument('--init', type=str, default='output_relax_seed12345/config.9800.data',
                   help='基体构型(封版帧)')
    p.add_argument('--obs',  type=str, default='init_data_caseD_smoke/obstacles.data')
    p.add_argument('--maxstrain', type=float, default=0.005, help='总应变(0.5%% 够读 0.2%% offset)')
    p.add_argument('--out',  type=str, default='output_caseD_smoke')
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    init_file = args.init if os.path.isabs(args.init) else os.path.join(base_dir, args.init)
    obs_file  = args.obs  if os.path.isabs(args.obs)  else os.path.join(base_dir, args.obs)
    out_dir   = args.out  if os.path.isabs(args.out)  else os.path.join(base_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    G = ExaDisNet()
    G.read_paradis(init_file)
    net = DisNetManager(G)

    # 硬球障碍:type=0 = OBSTACLE_OROWAN(几何投影),走 handle_orowan,
    # 与 breakaway 点障碍(type=1)分型互斥,不碰 oxides 表(三表独立)
    obs = np.loadtxt(obs_file)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    G.load_obstacles([list(c) for c in obs[:, :3]], list(obs[:, 3]), type=0)
    print(f'[smoke] {len(obs)} 个硬球障碍已载入 (R={obs[0,3]:.1f}b)')

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=G.cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=15000.0, Mscrew=3000.0,
                            Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[], state=state,
                                force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Orowan', state=state)   # 激活几何投影
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    cross_slip = CrossSlip(cross_slip_mode='ForceBasedParallel', state=state, force=calforce)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        cross_slip=cross_slip,
        loading_mode='strain_rate',
        erate=1e3,
        edir=np.array([0., 0., 1.]),
        max_strain=args.maxstrain,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=100,
        write_dir=out_dir,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
