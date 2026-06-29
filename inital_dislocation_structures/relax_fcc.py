"""
FCC Cu 滑移环网络弛豫 (Track B) —— 零应力弛豫 + cross-slip

把高密度滑移环播种构型(gen_fcc_config.py --type glide, rho~1.5e14)弛豫成
结点钉扎的真实网络, 弛豫后存 relaxed_config.data, 再交给 fcc_load.py 加载。

为什么必须弛豫:
  rho=1e12 下自由滑移环线张力坍缩自湮灭(已实测, 0.2ns 全死)。
  Motz et al., Acta Mater 57 (2009) 1744 的做法: 高密度播种 -> 弛豫成网。
  本课题 bulk PBC(无自由表面, 位错跑不掉), 播种 ~1.5e14 弛豫后落到 ~1.5e13。

怎么判断收敛(用时间, 不是步数):
  画 stress_strain_dens.dat 的 density vs Time 走平(= Motz Fig.1a 横轴是时间);
  辅以 Nnodes/Nsegs 不漂、dt 爬升稳。
  -> 默认 .dat 只有 Step/Strain/Stress/Density, 故用 out_props 显式索要 time 等列。
  注意: 旧做法改 pyexadis_base.step_print_info 对 Perf 驱动无效(Perf 走 C++
  driver.cpp 输出, 不经 Python step_print_info), 这就是之前列出不来、只能抠
  slurm 日志的根因。out_props 才是 Perf 的正道。若部署版 .dat 仍无 Time 列,
  slurm 控制台每步也打印 tottime(driver.cpp:454), 可兜底。

零应力弛豫 = loading_mode='stress' + applied_stress 默认 0, 跑固定步数。
材料/迁移率 = FCC Cu(同 fcc_load.py, 事实源算例22)。
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


# FCC Cu(同 fcc_load.py); 弛豫 maxdt 放大些(settle 后可走大步)
state = {
    "crystal":   'fcc',
    "burgmag":   2.55e-10,   # b [m], Cu a/2<110>
    "mu":        54.6e9,     # 剪切模量 [Pa]
    "nu":        0.324,      # 泊松比
    "a":         6.0,        # 核心参数
    "maxseg":    200.0,      # 约 51nm
    "minseg":    40.0,       # 约 10nm
    "rtol":      10.0,
    "rann":      10.0,
    "nextdt":    1e-10,
    "maxdt":     1e-8,       # 弛豫比加载放大一档
}

INIT  = 'init_glide_seed12345'   # 待弛豫的滑移环播种构型(换 seed 改这里或 --init)
STEPS = 20000                    # 弛豫步数(看 density-vs-time 是否走平, 不够用 --steps 加)

OUT_PROPS = ['step', 'strain', 'stress', 'density', 'Nnodes', 'Nsegs', 'dt', 'time']


def main():
    pyexadis.initialize()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', default=INIT, help=f'播种构型目录(默认 {INIT})')
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--out', default=None, help='输出目录名; 默认 init 前缀换成 relax')
    args = parser.parse_args()

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = args.init if os.path.isabs(args.init) else os.path.join(base_dir, args.init)
    name       = os.path.basename(os.path.normpath(init_dir))
    out_name   = args.out or (('relax' + name[4:]) if name.startswith('init') else ('relax_' + name))
    output_dir = os.path.join(base_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)

    G = ExaDisNet()
    G.read_paradis(os.path.join(init_dir, 'init_config.data'))
    net = DisNetManager(G)

    calforce   = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64,
                          cell=net.get_disnet(ExaDisNet).cell)
    mobility   = MobilityLaw(mobility_law='FCC_0', state=state,
                             Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint    = TimeIntegration(integrator='Subcycling', rgroups=[], state=state,
                                 force=calforce, mobility=mobility)
    collision  = Collision(collision_mode='Retroactive', state=state)
    topology   = Topology(topology_mode='TopologyParallel', state=state,
                          force=calforce, mobility=mobility)
    remesh     = Remesh(remesh_rule='LengthBased', state=state)
    cross_slip = CrossSlip(cross_slip_mode='ForceBasedParallel', state=state, force=calforce)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh, cross_slip=cross_slip,
        loading_mode='stress',            # 零应力(applied_stress 默认 0) = 纯弛豫
        num_steps=args.steps,
        burgmag=state["burgmag"], state=state,
        print_freq=10, write_freq=1000,
        out_props=OUT_PROPS,
        write_dir=output_dir,
    )
    sim.run(net, state)

    relaxed = os.path.join(output_dir, 'relaxed_config.data')
    net.get_disnet(ExaDisNet).write_data(relaxed)
    print(f'弛豫完成, 网络写出: {relaxed}')
    print(f'收敛判断: 画 {output_dir}/stress_strain_dens.dat 的 density vs Time 看是否走平')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
