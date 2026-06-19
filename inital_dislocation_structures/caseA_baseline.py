"""
工况 A —— 基线:同一随机棱柱环网络,无预处理,直接正式加载
(预弛豫 否 / 预变形 否 / 二次弛豫 否 / 正式加载 是)

初始构型从 init_loops_seed<seed>/init_config.data 读取
(先运行 generate_loops.py 生成该文件)。

写法参考 HomeWork/test_Cu_pure.py,材料换成 Fe BCC。
材料/迁移率参数为占位值,待用户最终敲定(见下方 ⚠️ 标注)。
单轴应变率控制,[001] 方向;屈服由后处理(0.2% offset)从
stress_strain_dens.dat 提取。
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


# ⚠️ Fe BCC 参数 —— 占位值,待用户最终敲定
state = {
    "crystal":   'bcc',
    "burgmag":   0.248e-9,   # b [m], Fe a/2<111>
    "mu":        82e9,       # 剪切模量 [Pa] ⚠️待定
    "nu":        0.29,       # 泊松比 ⚠️待定
    "a":         4.0,        # 核心参数
    "maxseg":    200,        # ≈50nm,与生成脚本 MAXSEG_B 一致
    "minseg":    50,         # ≈12nm
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
    output_dir = os.path.join(base_dir, f'output_caseA_seed{args.seed}')
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

    # ⚠️ 迁移率参数 (BCC_0B) —— 占位值,待用户最终敲定
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
        erate=1e4,                       # ⚠️应变率 [1/s] 待定
        edir=np.array([0., 0., 1.]),     # 加载轴 [001]
        max_strain=0.01,                 # ⚠️最大应变 待定
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=100,
        write_dir=output_dir,
        restart=restart,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
