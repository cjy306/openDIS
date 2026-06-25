"""
工况 C 二次弛豫 —— 续跑 (continue) —— 从已存构型接着零应力弛豫

用途:caseC_prep.py 跑完 relax2 后若 ρ/Nnodes 仍未走平(预变形后的稠密结构
静态回复慢,10000 步常不够),用本脚本从 predeformed_relaxed_config.data
接着弛豫,不必从 relaxed_config 重做预变形。

物理依据:过阻尼一阶 EOM (dr/dt = M·F),网络演化只依赖当前构型;.data 已完整
保存构型,故"从 .data 续跑"与"一口气多跑"等价(dt 自适应会重新爬,无缝)。

流程:
  读 output_caseC_p{predeform}_seed{seed}/predeformed_relaxed_config.data
  → 零应力弛豫 --steps 步 → relax2_cont/
  → 覆盖回 predeformed_relaxed_config.data(续跑可重复,每次读最新、写最新)

跑完:plot_relax.py 看 relax2_cont/ 是否走平;没平就再跑一次本脚本(它会读
刚覆盖的 .data 接着续)。平了再跑 caseC_load.py。

用法:python caseC_relax2_cont.py --seed 12345 --predeform 0.18 --steps 20000
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


# ⚠️ Fe BCC 参数 —— 与 caseC_prep 一致
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


def build_modules(cell):
    """与 caseC_prep 同款物理模块(SUBCYCLING_MODEL 需要 cell)。"""
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
    parser.add_argument('--seed', type=int, default=12345, help='随机实现种子')
    parser.add_argument('--predeform', type=float, default=0.18, help='预变形总应变 [%%],定位 output_caseC 目录')
    parser.add_argument('--steps', type=int, default=20000, help='续跑的二次弛豫步数')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(base_dir, f'output_caseC_{args.predeform}%_seed{args.seed}')
    cont_dir = os.path.join(out_dir, 'relax2_cont')
    os.makedirs(cont_dir, exist_ok=True)

    cfg = os.path.join(out_dir, 'predeformed_relaxedC2_config.data')
    if not os.path.exists(cfg):
        raise FileNotFoundError(f'找不到 {cfg};请先跑 caseC_prep.py --predeform {args.predeform}')

    # 读 relax2 终态构型
    G = ExaDisNet()
    G.read_paradis(cfg)
    net = DisNetManager(G)

    cell = net.get_disnet(ExaDisNet).cell
    mods = build_modules(cell)

    # 续跑:零应力弛豫(与 caseC_prep Phase 2 相同条件)
    relax2_sim = SimulateNetworkPerf(
        **mods,
        loading_mode='stress',
        applied_stress=np.zeros(6),
        num_steps=args.steps,
        burgmag=state["burgmag"], state=state,
        print_freq=1, write_freq=100, write_dir=cont_dir,
    )
    relax2_sim.run(net, state)

    # 覆盖回 .data(续跑可重复:每次读最新、写最新)
    net.get_disnet(ExaDisNet).write_data(cfg)
    print(f'续跑 {args.steps} 步完成,已覆盖: {cfg}')
    print(f'下一步:plot_relax.py 看 {cont_dir} 是否走平;没平就再跑本脚本续;平了再跑 caseC_load.py --predeform {args.predeform}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
