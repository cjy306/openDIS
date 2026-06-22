"""
工况 C 加载段 (load) —— 读"预变形+二次弛豫后"的网络 → 正式加载

前置:先跑 caseC_prep.py(同 seed、同 predeform)产出 predeformed_relaxed_config.data,
并用 plot_relax.py 确认 relax2/ 已收敛。本脚本只做正式加载(与 A/B 加载条件相同)。

--predeform 用于定位对应 prep 的产出目录(0.1 / 0.3 / 0.5)。跑 C2/C3 只改这个参数。
用法:python caseC_load.py --seed 12345 --predeform 0.1
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


# ⚠️ Fe BCC 参数 —— 与 caseA/caseB/caseC_prep 一致,占位值待最终敲定
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
    parser.add_argument('--seed', type=int, default=12345, help='随机实现种子')
    parser.add_argument('--predeform', type=float, default=0.18, help='预变形总应变 [%%],定位 prep 产出目录')
    parser.add_argument('--restart', type=int, help='从指定步骤重启')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(base_dir, f'output_caseC_{args.predeform}%_seed{args.seed}')
    load_dir = os.path.join(out_dir, 'load')
    os.makedirs(load_dir, exist_ok=True)

    if args.restart is not None:
        net, restart = read_restart(
            state=state,
            restart_file=os.path.join(load_dir, f'restart.{args.restart}.exadis'))
    else:
        # 读 prep 产出的"预变形+二次弛豫后"构型
        cfg = os.path.join(out_dir, 'predeformed_relaxedC2_config.data')
        G = ExaDisNet()
        G.read_paradis(cfg)
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
        erate=1e4,                       # ⚠️ 与 caseA/B 一致,待定
        edir=np.array([0., 0., 1.]),
        max_strain=0.01,                 # ⚠️ 与 caseA/B 一致,待定
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=100,
        write_dir=load_dir,
        restart=restart,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
