"""
工况 B —— 预弛豫后的网络 → 正式加载
(预弛豫 是 / 预变形 否 / 二次弛豫 否 / 正式加载 是)

弛豫已由 diagnose_relax.py 统一做过并存为 relaxed_config.data(B/C 共用同一份),
本脚本不再自己弛豫,直接读 relaxed 网络做单轴加载。

与 caseA_baseline.py 的唯一区别:读的初始构型不同
  - A: init_config.data      (原始未弛豫 N0)
  - B: relaxed_config.data   (弛豫后)
对比 A vs B 即隔离出"预弛豫"对屈服的影响。

材料/迁移率/加载参数与 caseA_baseline.py 完全一致(占位值待最终敲定)。

前置:先跑 generate_loops.py → diagnose_relax.py(产出 relaxed_config.data)。
用法:python caseB_load.py --seed 12345
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


# ⚠️ Fe BCC 参数 —— 与 caseA_baseline.py 一致,占位值待最终敲定
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
    pyexadis.initialize()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345, help='随机实现种子(对应 init_loops 目录)')
    parser.add_argument('--restart', type=int, help='从指定步骤重启')
    args = parser.parse_args()

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = os.path.join(base_dir, f'init_loops_seed{args.seed}')
    output_dir = os.path.join(base_dir, f'output_caseB_seed{args.seed}')
    os.makedirs(output_dir, exist_ok=True)

    if args.restart is not None:
        net, restart = read_restart(
            state=state,
            restart_file=os.path.join(output_dir, f'restart.{args.restart}.exadis'))
    else:
        G = ExaDisNet()
        G.read_paradis(os.path.join(init_dir, 'relaxed_config.data'))  # 弛豫后的共用网络
        net = DisNetManager(G)
        restart = None

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.get_disnet(ExaDisNet).cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=15000.0, Mscrew=3000.0, Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        erate=1e4,                       # ⚠️ 与 caseA 一致,待定
        edir=np.array([0., 0., 1.]),
        max_strain=0.01,                 # ⚠️ 与 caseA 一致,待定
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=50,
        write_dir=output_dir,
        restart=restart,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
