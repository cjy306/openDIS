"""
test_oxide_verify.py —— 氧化物高斯势力最小验证(第3+4级)

场景:一条穿盒无限刃位错 + 玻移路径正前方一个氧化物,恒定切应力驱动。
目的:眼见为实地验证 (3) 力存在且方向正确(位错被挡/弯曲/绕过),
     (4) A 旋钮有效(临界应力随 A 单调),顺带校验 A 的单位换算推导。

A 的单位换算(推导,待A扫描验证):
  ExaDiS 内部长度以 b 计、应力以 Pa 计;由迁移率 v=M*f/L 反推内部力单位 = Pa*b^2。
  匹配 F_phys = 2*A_SI*r/Rp^2 与 F_int = 2*A_int*r_int/Rp_int^2 得:
      A_int = A_SI / burgmag^3
  Lehtinen 2018: A_SI = 1.56e-18 Pa*m^3, b=0.248nm → A_int ≈ 1.02e11(默认值)。
  ⚠️ 基于"内部力=Pa*b^2"假设;若 A 扫描显示 1.02e11 不是强障碍档,按扫描结果修正。

用法:
  python test_oxide_verify.py                         # 默认:A=1.02e11, tau=50MPa, Rp=10nm
  python test_oxide_verify.py --tau 100               # 加大驱动应力(看 Orowan 绕过)
  python test_oxide_verify.py --A 1.02e10             # A 缩 10 倍(第4级扫描:0.1x/1x/10x)
  python test_oxide_verify.py --A 0                   # A=0 对照(位错应自由通过)
之后:python paraview.py --sim output_oxide_verify --out vtk_oxide_verify
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
    from pyexadis_utils import insert_infinite_line
except ImportError:
    raise ImportError('Cannot import pyexadis')


BURGMAG = 0.248e-9      # b [m]
LBOX_M  = 0.5e-6        # 500nm 盒(小,验证跑得快)
A_INT_DEFAULT = 1.56e-18 / BURGMAG**3   # ≈1.02e11,Lehtinen A_SI 换算(见文件头推导)

state = {
    "crystal":   'bcc',
    "burgmag":   BURGMAG,
    "mu":        81e9,
    "nu":        0.3,
    "a":         2.0,
    "maxseg":    40,     # ≈10nm:比 Rp(10nm)细,保证位错"看得见"氧化物
    "minseg":    10,
    "rtol":      0.5,
    "rann":      1.0,
    "nextdt":    1e-10,
    "maxdt":     1e-8,
    "use_glide_planes":       1,
    "num_bcc_plane_families": 1,
}


def build_config(Rp_b, A_int):
    """一条穿盒无限刃位错 + 滑移路径正前方一个氧化物。返回 (net, oxide_args)。"""
    Lbox = LBOX_M / BURGMAG
    cell = pyexadis.Cell(Lbox)
    center = np.array(cell.center())

    # 滑移系 b=1/2[111], n=(0,-1,1);刃位错线方向 e = n×b(垂直 b,在滑移面内)
    b = np.array([1., 1., 1.]) / np.sqrt(3.0)
    n = np.array([0., -1., 1.]) / np.sqrt(2.0)
    e = np.cross(n, b); e /= np.linalg.norm(e)

    # 位错起始位置:从盒心沿 -b 方向退开 1/4 盒(受切应力后沿 +b 向氧化物滑去)。
    # 起点与盒心同在该滑移面内(位移沿 b,b·n=0),故氧化物中心恰在位错滑移面上。
    origin = center - 0.25 * Lbox * b
    nodes, segs = [], []
    nodes, segs = insert_infinite_line(cell, nodes, segs, b, n, origin,
                                       linedir=e, maxseg=state["maxseg"])
    net = DisNetManager(ExaDisNet(cell, nodes, segs))

    # 氧化物:盒心一个
    oxide_centers = [list(center)]
    oxide_Rp      = [Rp_b]
    oxide_A       = [A_int]
    return net, (oxide_centers, oxide_Rp, oxide_A), b, n


def applied_stress_voigt(tau_Pa, b, n):
    """构造纯剪应力张量 σ = τ(b̂⊗n̂+n̂⊗b̂),使该滑移系分解切应力恰为 τ。
    返回 Voigt [xx,yy,zz,yz,xz,xy]。"""
    S = np.outer(b, n) + np.outer(n, b)
    sig = tau_Pa * S
    return np.array([sig[0,0], sig[1,1], sig[2,2], sig[1,2], sig[0,2], sig[0,1]])


def main():
    import argparse
    pyexadis.initialize()
    p = argparse.ArgumentParser(description='氧化物高斯势力最小验证')
    p.add_argument('--A',    type=float, default=A_INT_DEFAULT,
                   help=f'高斯势强度(内部单位),默认 {A_INT_DEFAULT:.3e}(=Lehtinen 换算)')
    p.add_argument('--Rp',   type=float, default=10.0, help='高斯势宽度 [nm],默认 10')
    p.add_argument('--tau',  type=float, default=50.0, help='分解切应力 [MPa],默认 50')
    p.add_argument('--steps', type=int,  default=2000, help='最大步数')
    p.add_argument('--out',  type=str,   default=None)
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, 'output_oxide_verify')
    os.makedirs(out_dir, exist_ok=True)

    Rp_b = args.Rp * 1e-9 / BURGMAG
    net, (ox_c, ox_rp, ox_a), b, n = build_config(Rp_b, args.A)
    print(f'[verify] A={args.A:.3e} (internal), Rp={args.Rp}nm={Rp_b:.1f}b, tau={args.tau}MPa')

    # 注入氧化物(第2级验证:应打印 "[System] 1 oxide particles loaded")
    exadis_net = net.get_disnet(ExaDisNet)
    if args.A > 0:
        exadis_net.load_oxides(ox_c, ox_rp, ox_a)
        # 落盘供 paraview.py 画球(cx cy cz Rp,单位 b)
        ox_arr = np.hstack((np.array(ox_c), np.array(ox_rp).reshape(-1, 1)))
        np.savetxt(os.path.join(out_dir, 'oxides.data'), ox_arr, fmt='%.6e')
    else:
        print('[verify] A=0 对照组:不注入氧化物,位错应自由滑过盒心')

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=32, cell=exadis_net.cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=15000.0, Mscrew=3000.0,
                            Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[],
                                state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)   # 不用 Orowan 碰撞!
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='stress',
        applied_stress=applied_stress_voigt(args.tau * 1e6, b, n),
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
