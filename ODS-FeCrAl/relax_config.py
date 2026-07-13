"""
relax_config.py —— Case A 基体制备:选定帧的零外力弛豫

流程:读预变形选定帧(默认 config.1094200.data)→ 外加应力=0 弛豫 --steps 步
→ 打印弛豫前后密度(判据:掉一半就重议)→ 存 relaxed_config.data。

判敛看密度-时间曲线(物理时间,非步数):python plot_relax.py
  数据源 = 输出目录 stress_strain_dens.dat,out_props 已配 time 列。
曲线未趋平 → 从 relaxed_config.data 再跑一轮加大 --steps。

state/模块与 test_caseA_baseline.py 完全一致(仅应力为零):
  基体制备与生产工况同数值环境;FR 钉扎节点保留(2026-07-13 拍板,共模+防塌缩)。

用法(昆山):
  python relax_config.py
  python relax_config.py --config output_caseA_seed12345/config.1094200.data --steps 5000
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
    from pyexadis_utils import dislocation_density
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
    p = argparse.ArgumentParser(description='Case A 选定帧零外力弛豫')
    p.add_argument('--config', type=str, default='output_caseA_seed12345/config.1094200.data',
                   help='预变形选定帧')
    p.add_argument('--steps',  type=int, default=10000)
    p.add_argument('--out',    type=str, default='output_relax_seed12345')
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
    out_dir  = args.out    if os.path.isabs(args.out)    else os.path.join(base_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    G = ExaDisNet()
    G.read_paradis(cfg_file)
    net = DisNetManager(G)

    rho0 = dislocation_density(net, state["burgmag"])
    print(f'[relax] 弛豫前密度: {rho0:.4e} /m^2  ({cfg_file})')

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=G.cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=15000.0, Mscrew=3000.0,
                            Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[], state=state,
                                force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    cross_slip = CrossSlip(cross_slip_mode='ForceBasedParallel', state=state, force=calforce)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        cross_slip=cross_slip,
        loading_mode='stress',
        applied_stress=np.zeros(6),          # 零外力弛豫
        max_step=args.steps,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,                        # .dat 每 10 步一行(propfreq=print_freq)
        write_freq=100,                       # 弛豫过程快照,可 paraview 检查
        write_dir=out_dir,
        out_props=['step', 'time', 'dt', 'density', 'Nnodes'],   # time=物理总时间[s];Nnodes 必须大写N(driver.h 只认这个写法)
    )
    sim.run(net, state)

    rho1 = dislocation_density(net, state["burgmag"])
    keep = rho1 / rho0 * 100
    print(f'[relax] 弛豫后密度: {rho1:.4e} /m^2  (保持率 {keep:.1f}%)')
    if keep < 50:
        print('[relax] ⚠️ 密度掉超一半 —— 按判据重议要不要弛豫/换帧')

    out_cfg = os.path.join(out_dir, 'relaxed_config.data')
    net.get_disnet(ExaDisNet).write_data(out_cfg)
    print(f'[relax] 弛豫后构型: {out_cfg}')
    print('[relax] 判敛: python plot_relax.py 看密度-时间曲线,未趋平则从 relaxed_config.data 续跑')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
