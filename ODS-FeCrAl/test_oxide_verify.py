"""
test_oxide_verify.py —— 氧化物高斯势力最小验证(第3+4级,纯模拟脚本)

初始构型从 init_data_oxide_verify/ 读取(先跑 generate_oxide_verify.py 生成):
  init_config.data —— 一条穿盒无限刃位错
  oxides.data      —— 氧化物几何(cx cy cz Rp,单位 b)
本脚本只负责:读构型 → 注入氧化物(A 由命令行给)→ 恒应力加载。

A 的单位换算(推导,待 A 扫描验证):
  ExaDiS 内部长度以 b 计、应力 Pa;假设内部力单位=Pa*b^2 →
  A_int = A_SI / burgmag^3;Lehtinen A_SI=1.56e-18 Pa*m^3 → A_int ≈ 1.02e11(默认)。
  ⚠️ 若 A 扫描显示 1.02e11 不在"强障碍"档,按行为修正换算。

用法:
  python generate_oxide_verify.py            # 先生成(一次即可)
  python test_oxide_verify.py                # 默认 A=1.02e11, tau=50MPa
  python test_oxide_verify.py --A 0          # 对照:不注入,应自由通过
  python test_oxide_verify.py --A 1.02e10 --out output_ox_A01x   # A 扫描(换输出目录!)
之后:paraview.py 配置 INPUT=输出目录, OXIDES=init_data_oxide_verify/oxides.data
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


BURGMAG = 0.248e-9
A_INT_DEFAULT = 1.56e-18 / BURGMAG**3   # ≈1.02e11(Lehtinen 换算,见文件头)

# 滑移系(与 generate_oxide_verify.py 一致,用于构造分解切应力)
B_SLIP = np.array([1., 1., 1.]) / np.sqrt(3.0)
N_SLIP = np.array([0., -1., 1.]) / np.sqrt(2.0)

state = {
    "crystal":   'bcc',
    "burgmag":   BURGMAG,
    "mu":        81e9,
    "nu":        0.3,
    "a":         2.0,
    "maxseg":    40,
    "minseg":    10,
    "rtol":      0.5,
    "rann":      1.0,
    "nextdt":    1e-10,
    "maxdt":     1e-8,
    "use_glide_planes":       1,
    "num_bcc_plane_families": 1,
}


def applied_stress_voigt(tau_Pa, b, n):
    """纯剪应力张量 σ=τ(b̂⊗n̂+n̂⊗b̂),该滑移系分解切应力恰为 τ。Voigt [xx,yy,zz,yz,xz,xy]。"""
    sig = tau_Pa * (np.outer(b, n) + np.outer(n, b))
    return np.array([sig[0,0], sig[1,1], sig[2,2], sig[1,2], sig[0,2], sig[0,1]])


def main():
    pyexadis.initialize()
    import argparse
    p = argparse.ArgumentParser(description='氧化物高斯势力最小验证(纯模拟)')
    p.add_argument('--A',     type=float, default=A_INT_DEFAULT,
                   help=f'高斯势强度(内部单位),默认 {A_INT_DEFAULT:.3e};0=不注入对照')
    p.add_argument('--tau',   type=float, default=50.0, help='分解切应力 [MPa]')
    p.add_argument('--steps', type=int,   default=2000, help='最大步数')
    p.add_argument('--init',  type=str,   default='init_data_oxide_verify')
    p.add_argument('--out',   type=str,   default='output_oxide_verify')
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    init_dir = args.init if os.path.isabs(args.init) else os.path.join(base_dir, args.init)
    out_dir  = args.out  if os.path.isabs(args.out)  else os.path.join(base_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # 读初始构型
    G = ExaDisNet()
    G.read_paradis(os.path.join(init_dir, 'init_config.data'))
    net = DisNetManager(G)

    # 读氧化物几何 + 注入(A 由命令行;A=0 为对照组)
    if args.A > 0:
        ox = np.loadtxt(os.path.join(init_dir, 'oxides.data'))
        if ox.ndim == 1:
            ox = ox.reshape(1, -1)
        centers = [list(c) for c in ox[:, :3]]
        Rp_list = list(ox[:, 3])
        A_list  = [args.A] * len(Rp_list)
        G.load_oxides(centers, Rp_list, A_list)
        print(f'[verify] {len(Rp_list)} oxide(s), A={args.A:.3e}, Rp={Rp_list[0]:.1f}b, tau={args.tau}MPa')
    else:
        print(f'[verify] A=0 对照组:不注入氧化物,位错应自由通过, tau={args.tau}MPa')

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=32, cell=G.cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=15000.0, Mscrew=3000.0,
                            Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[],
                                state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)   # 不用 Orowan 碰撞
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='stress',
        applied_stress=applied_stress_voigt(args.tau * 1e6, B_SLIP, N_SLIP),
        max_step=args.steps,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=1,
        write_dir=out_dir,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
