"""
Phase-1 诊断:对 N0 做一次长时间零外载弛豫,用来标定标准弛豫步数。

目的不是出物理结果,而是看构型从"几何构造态"松弛到"力学平衡态"需要多少步:
  - 跑足够长(DIAG_STEPS,故意偏大)
  - print_freq=1 → stress_strain_dens.dat 记下每一步的 ρ / Nnodes / Nsegs / dt
  - 用 plot 看三条曲线在第几步走平:
      ρ(Density)走平 + Nnodes/Nsegs 不再漂移 + dt 爬升并稳定在 maxdt 附近
    那个步数(加余量)就是 caseB/C 该用的 RELAX_STEPS。

⚠️ 健康性判断:棱柱环面外柏氏矢量,弛豫应是温和的(尖角磨圆、ρ 基本守恒)。
   若 ρ 持续暴跌,说明环在坍缩/被 remesh 吃掉 —— 不是"弛豫到位",
   要回头查环半径 / maxseg / minseg(见 generate_loops.py)。

材料/迁移率参数与 caseA/caseB 一致(占位值)。

用法:
  python diagnose_relax.py --seed 12345 [--steps 5000]
  → 输出 output_diag_relax_seed<seed>/stress_strain_dens.dat
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


# ⚠️ 与 caseA/caseB 一致,占位值待最终敲定
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
    parser.add_argument('--steps', type=int, default=5000, help='诊断弛豫步数(故意偏大,看曲线走平)')
    args = parser.parse_args()

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = os.path.join(base_dir, f'init_loops_seed{args.seed}')
    output_dir = os.path.join(base_dir, f'output_diag_relax_seed{args.seed}')
    os.makedirs(output_dir, exist_ok=True)

    G = ExaDisNet()
    G.read_paradis(os.path.join(init_dir, 'init_config.data'))
    net = DisNetManager(G)

    cell = net.get_disnet(ExaDisNet).cell
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=15000.0, Mscrew=3000.0, Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='stress',
        applied_stress=np.zeros(6),     # 零外载,只在内应力下弛豫
        num_steps=args.steps,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,                   # 每步都记 → 细粒度收敛曲线
        write_freq=100,
        write_dir=output_dir,
    )
    sim.run(net, state)

    # 弛豫后构型存档:作为 B/C 系列的共用初始网络(它们不必各自再弛豫)
    # 确认收敛后(plot_relax.py 看曲线走平),这个末态即生产用的 relaxed 网络。
    net.get_disnet(ExaDisNet).write_data(os.path.join(init_dir, 'relaxed_config.data'))
    print(f'弛豫后构型已存: {os.path.join(init_dir, "relaxed_config.data")}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
