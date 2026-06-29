"""
FCC Cu 加载脚本 —— 应变率加载([001] 单轴)

sigma0 已关闭(=0): 不再用预应力救自由滑移环(实测救不上, 见对话纪要)。
  Track A(FR/棱柱)本就稳定, 直接零应力加载即可;
  Track B(滑移环)先经 relax_fcc.py 弛豫成网, 加载的是 relaxed_config.data。
  下方 sigma0 注入机制保留(SIGMA0=0 即不生效), 以备需要时再启用。

读 --init 指定的初始构型目录里的 init_config.data, [001] 单轴应变率加载。
加载逻辑与位错类型无关(三类型物理完全一致), 换工况 = 换 --init 指向的网络。

sigma0 预应力的原理与作用:
  - 通过构造函数的 applied_stress 注入。SimulateNetworkPerf 是 C++ driver,
    driver.cpp:601 在非 restart 时执行 system->extstress = ctrl.appstress, 即把
    sigma0 作为初始应力; 之后 strain_rate 控制(driver.cpp:412-419)从它累加。
  - 作用: 跳过 0->sigma0 的纯弹性段。该段对 FR/棱柱环本就只是弹性、什么都不发生;
    却是自由 glide 环被线张力坍缩的死亡窗口(rho=1e12 下邻居撑不住)。跳过它 =
    对 FR/棱柱无损, 对 glide 救活。这是明确的建模假设, 非物理平衡态的工程处理。

  - 仍是应变率加载(不是应力控制), 屈服/硬化由位错动力学在 erate 下自己决定。

曲线偏置: C++ 端 strain 从 0 记、stress 从 sigma0 记 -> 曲线起点 (0, sigma0)。
  取屈服时弹性线为 stress = sigma0 + E*strain(截距 sigma0); 或后处理令
  strain += sigma0/E 还原成过原点的正常拉伸。E = 2*mu*(1+nu)。

材料/迁移率 = FCC Cu, 事实源 core/exadis/examples/22_fcc_Cu_15um_1e3
(Bertin et al., MSMSE 27(7) 075014, 2019)。maxseg/minseg 已按 200nm 环改小
(算例22 是 15um 大盒用 2000/300, 对本课题 200nm 环过大)。
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


# FCC Cu(事实源: 算例22)
state = {
    "crystal":   'fcc',
    "burgmag":   2.55e-10,   # b [m], Cu a/2<110>
    "mu":        54.6e9,     # 剪切模量 [Pa]
    "nu":        0.324,      # 泊松比
    "a":         4.0,        # 核心参数
    "maxseg":    200.0,      # 约 51nm(算例22 为 2000, 对 200nm 环过大)
    "minseg":    40.0,       # 约 10nm
    "rtol":      1.0,
    "rann":      2.0,
    "nextdt":    1e-10,
    "maxdt":     1e-9,
}

INIT      = 'init_fr_seed12345'        # 初始构型目录(换工况改这里; 命令行 --init 可覆盖)
SIGMA0    = 0.0                        # 初始预应力 [Pa], 已关闭; 从零应力正常加载
EDIR      = np.array([0., 0., 1.])     # 加载轴 [001]
ERATE     = 1e3                        # 应变率 [1/s]
MAXSTRAIN = 0.01                       # 最大应变 1%


def main():
    pyexadis.initialize()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', default=INIT,
                        help=f'初始构型目录(默认顶部 INIT={INIT}); 换工况改顶部常量或用此覆盖')
    parser.add_argument('--out', default=None,
                        help='输出目录名; 默认把 init 目录名前缀 init 换成 output')
    parser.add_argument('--restart', type=int, help='从指定步骤重启')
    args = parser.parse_args()

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = args.init if os.path.isabs(args.init) else os.path.join(base_dir, args.init)
    name       = os.path.basename(os.path.normpath(init_dir))
    out_name   = args.out or (('output' + name[4:]) if name.startswith('init') else ('output_' + name))
    output_dir = os.path.join(base_dir, out_name)
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

    # 初始预应力 sigma0 (Voigt: xx,yy,zz,yz,xz,xy)
    applied_stress = np.array([0., 0., SIGMA0, 0., 0., 0.])

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64,
                         cell=net.get_disnet(ExaDisNet).cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state,
                            Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[], state=state,
                                force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state,
                         force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        applied_stress=applied_stress,
        erate=ERATE, edir=EDIR, max_strain=MAXSTRAIN,
        burgmag=state["burgmag"], state=state,
        print_freq=1, write_freq=100,
        write_dir=output_dir, restart=restart,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
