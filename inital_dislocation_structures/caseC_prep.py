"""
工况 C 预处理段 (prep) —— 读弛豫网络 → 预变形 → 二次弛豫 → 存档(不加载)

拆成 prep / load 两步:确保二次弛豫"收敛验证过"再加载,与第一次弛豫的处理一致
(第一次:diagnose_relax 单独跑 → 存 relaxed_config → 看曲线确认 → 再加载)。

流程(两段,各自输出子目录):
  读 relaxed_config.data (B/C 共用的弛豫后网络,diagnose_relax.py 产出)
  ├─ Phase 1 预变形:strain_rate 加到总应变目标 → predeform/
  └─ Phase 2 二次弛豫:卸载到零外载,弛豫到平衡(仿第一次)→ relax2/
  → 存 predeformed_relaxed_config.data,停。

跑完务必:plot_relax.py 看 relax2/ 曲线确认 ρ/Nnodes/dt 走平;
没走平就调大 --relax2-steps 重跑。确认后再跑 caseC_load.py 做正式加载。

--predeform 区分 C1/C2/C3:0.1 / 0.3 / 0.5 (单位 %)。跑 C2/C3 只改这个参数。
用法:python caseC_prep.py --seed 12345 --predeform 0.1
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')


# ⚠️ Fe BCC 参数 —— 与 caseA/caseB 一致,占位值待最终敲定
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

ERATE = 1e4                          # ⚠️ 预变形应变率 [1/s],与正式加载一致,待定
EDIR  = np.array([0., 0., 1.])       # 加载轴 [001](预变形与加载同轴)


def build_modules(cell):
    """两段共用的物理模块,只建一次(SUBCYCLING_MODEL 需要 cell)。"""
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=15000.0, Mscrew=3000.0, Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    return dict(calforce=calforce, mobility=mobility, timeint=timeint,
                collision=collision, topology=topology, remesh=remesh)


def main():
    pyexadis.initialize()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345, help='随机实现种子(对应 init_loops 目录)')
    parser.add_argument('--predeform', type=float, default=0.28,help='预变形总应变 [%%],如 0.1 / 0.3 / 0.5')
    parser.add_argument('--relax2-steps', type=int, default=10000, help='二次弛豫步数(偏大,跑完看 relax2/ 曲线验证收敛)')
    args = parser.parse_args()

    predeform_strain = args.predeform / 100.0   # % → 应变

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = os.path.join(base_dir, f'init_loops_seed{args.seed}')
    out_dir    = os.path.join(base_dir, f'output_caseC_{args.predeform}%_seed{args.seed}')
    predef_dir = os.path.join(out_dir, 'predeform')
    relax2_dir = os.path.join(out_dir, 'relax2')
    os.makedirs(predef_dir, exist_ok=True)
    os.makedirs(relax2_dir, exist_ok=True)

    # 读 B/C 共用的弛豫后网络
    G = ExaDisNet()
    G.read_paradis(os.path.join(init_dir, 'relaxed_config.data'))
    net = DisNetManager(G)

    cell = net.get_disnet(ExaDisNet).cell
    mods = build_modules(cell)

    # ===== Phase 1:预变形(strain_rate 加到总应变目标)=====
    pre_sim = SimulateNetworkPerf(
        **mods,
        loading_mode='strain_rate',
        erate=ERATE, edir=EDIR,
        max_strain=predeform_strain,
        burgmag=state["burgmag"], state=state,
        print_freq=1, write_freq=100, write_dir=predef_dir,
    )
    pre_sim.run(net, state)
    state.pop("istep", None)

    # ===== Phase 2:二次弛豫(卸载到零外载,仿第一次)=====
    relax2_sim = SimulateNetworkPerf(
        **mods,
        loading_mode='stress',
        applied_stress=np.zeros(6),
        num_steps=args.relax2_steps,
        burgmag=state["burgmag"], state=state,
        print_freq=1, write_freq=100, write_dir=relax2_dir,
    )
    relax2_sim.run(net, state)

    # 存预变形+二次弛豫后的构型(正式加载的起点),停。先验证 relax2 收敛再跑 load。
    out_cfg = os.path.join(out_dir, 'predeformed_relaxedC3_config.data')
    net.get_disnet(ExaDisNet).write_data(out_cfg)
    print(f'预变形+二次弛豫后构型已存: {out_cfg}')
    print(f'下一步:plot_relax.py 看 {relax2_dir} 确认收敛,再跑 caseC_load.py --predeform {args.predeform}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
