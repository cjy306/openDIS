"""
run_orowan_sim.py
==================
奥罗万机制验证模拟脚本
体系：FCC 铜，5μm 盒子，580nm 立方夹杂，体积分数 10%

参数设计依据：
  - maxseg = 500b：保证 夹杂边长(2275b)/maxseg = 4.5 > 3，
                   insert_surface_nodes 能稳定捕捉交点
  - minseg = 100b：足够小，保证表面节点密度
  - rann   = 2b：  奥罗万环形成需要两臂在夹杂背面能被 collision 检测到，
                   2b 远小于通道宽度(2627b)，安全且有效
  - nextdt = 1e-12：让积分器从极小时间步开始自适应增长，
                    避免首步因步长过大跨越 rann 触发异常碰撞
  - Subcycling rgroups = [0, 50, 250, 500]：与 minseg/maxseg 匹配
"""

import os
import sys
import traceback
import numpy as np

# ── 夹杂参数（必须在 import pyexadis 之前设置环境变量）─────────────────────
os.environ['INCLUSION_A']        = '1e-6'    # 1000 nm
os.environ['INCLUSION_VOL_FRAC'] = '0.064'

# ── 路径设置 ──────────────────────────────────────────────────────────────────
pydis_paths = [
    '../../python',
    '../../lib',
    '../../core/pydis/python',
    '../../core/exadis/python/',
    '/data/home/dg000246a/OpenDis/python/framework',
]
for p in pydis_paths:
    ap = os.path.abspath(p)
    if ap not in sys.path:
        sys.path.append(ap)

np.set_printoptions(threshold=20, edgeitems=5)

try:
    import pyexadis
    from pyexadis_base import (
        ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart,
        CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
    )
except ImportError as e:
    raise ImportError(f'Cannot import pyexadis: {e}')


# ══════════════════════════════════════════════════════════════════════════════
# 网络初始化工具
# ══════════════════════════════════════════════════════════════════════════════

def load_network(datafile):
    """
    从 ExaDiS 格式数据文件读取初始网络。
    依次尝试两种读取方法，均失败则抛出异常。
    """
    print(f"读取初始配置: {datafile}")
    for method_name, loader in [
        ("read_data",    lambda G: G.read_data(datafile)),
        ("read_paradis", lambda G: G.read_paradis(datafile)),
    ]:
        try:
            G = ExaDisNet()
            loader(G)
            net = DisNetManager(G)
            print(f"  ✓ {method_name} 成功")
            print(f"  cell.h      = {G.cell.h}")
            print(f"  节点数      = {net.num_nodes()}")
            print(f"  线段数      = {net.num_segments()}")
            return net, None
        except Exception as e:
            print(f"  ✗ {method_name} 失败: {e}")

    raise RuntimeError(f"所有方法均无法读取文件: {datafile}")


def print_param_check(state):
    """启动前打印关键参数，方便核查。"""
    burgmag       = state['burgmag']
    a_phys        = float(os.environ.get('INCLUSION_A', 0))
    vf            = float(os.environ.get('INCLUSION_VOL_FRAC', 0))
    a_dim         = a_phys / burgmag
    Lbox_m        = 5e-6
    Lbox_dim      = Lbox_m / burgmag
    n             = max(1, int(round((vf * Lbox_dim**3 / a_dim**3) ** (1/3))))
    spacing       = Lbox_dim / n
    channel       = spacing - a_dim

    print("\n" + "=" * 55)
    print("参数核查")
    print("=" * 55)
    print(f"  夹杂边长      : {a_phys*1e9:.0f} nm  =  {a_dim:.0f} b")
    print(f"  体积分数      : {vf:.2f}  →  {n}×{n}×{n} = {n**3} 个夹杂")
    print(f"  通道宽度      : {channel:.0f} b  =  {channel*burgmag*1e9:.0f} nm")
    print(f"  maxseg        : {state['maxseg']} b  (a/maxseg = {a_dim/state['maxseg']:.1f}，建议 > 3)")
    print(f"  minseg        : {state['minseg']} b")
    print(f"  rann          : {state['rann']} b  (rann/channel = {state['rann']/channel:.4f})")
    print(f"  nextdt        : {state['nextdt']:.2e} s")
    print(f"  maxdt         : {state['maxdt']:.2e} s")
    print("=" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 主模拟函数
# ══════════════════════════════════════════════════════════════════════════════

def run_orowan_verification():

    pyexadis.initialize()

    # ── 材料和数值参数 ────────────────────────────────────────────────────────
    burgmag = 2.55e-10   # m，FCC 铜

    state = {
        "crystal":  'fcc',
        "burgmag":  burgmag,

        # 弹性常数（FCC Cu）
        "mu":       54.6e9,    # Pa
        "nu":       0.324,

        # 线弹性核心半径（无量纲，单位 b）
        "a":        1.0,

        # 段长限制
        # maxseg = 500b：夹杂边长(2275b)/500 = 4.5 > 3，保证每个夹杂面
        #   至少有 4 段覆盖，insert_surface_nodes 能稳定找到边界交点
        "maxseg":   800.0,
        # minseg = 100b：保证夹杂表面节点足够密集，同时不使网络过于庞大
        "minseg":   200.0,

        # 碰撞检测容差
        # rtol = 0.25b：节点位置误差容差，建议为 0.25*a
        "rtol":     0.25,
        # rann = 2b：collision 检测距离。
        #   物理意义：两位错臂在夹杂背面靠近到 2b 以内时，判定为相遇，
        #   触发 merge，形成奥罗万环。
        #   设为 2b 远小于通道宽度(2627b)，不会误触发；
        #   同时足够大，保证绕过的两臂能被可靠检测到。
        "rann":     2.0,

        # 时间步长
        # nextdt = 1e-12s：初始步长极小，让积分器（Subcycling）
        #   自适应增长。若初始步长太大（如原来的 4.71e-10s），
        #   首步位移就可能超过 rann，触发虚假 collision。
        "nextdt":   1e-12,
        # maxdt = 1e-9s：比原来的 5e-10s 略大，给积分器更多空间，
        #   在远离夹杂时用大步长加速。
        "maxdt":    1e-9,
    }

    # ── 路径 ──────────────────────────────────────────────────────────────────
    data_file  = '/data/home/dg000246a/OpenDis/examples/work/fr_network.data'
    output_dir = '/data/home/dg000246a/OpenDis/examples/work/test_orowan8_10pct'
    os.makedirs(output_dir, exist_ok=True)

    # ── 网络初始化 ────────────────────────────────────────────────────────────
    restart_id = sys.argv[1] if len(sys.argv) > 1 else None

    if restart_id is None:
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"找不到初始网络文件: {data_file}\n"
                f"请先运行 generate_fr_network.py 生成初始网络"
            )
        net, restart = load_network(data_file)
    else:
        restart_file = os.path.join(output_dir, f'restart.{restart_id}.exadis')
        print(f"从重启文件加载: {restart_file}")
        net, restart = read_restart(
            state=state,
            restart_file=restart_file,
        )

    print_param_check(state)

    # ── 力计算模块 ────────────────────────────────────────────────────────────
    # SUBCYCLING_MODEL：对远场用 FFT，对近场精确计算，效率高
    calforce = CalForce(
        force_mode='SUBCYCLING_MODEL',
        state=state,
        Ngrid=64,          # FFT 网格数，64³ 对 5μm 盒子足够
        cell=net.cell,
    )

    # ── 迁移率模块 ────────────────────────────────────────────────────────────
    # FCC_0：各向同性迁移率，刃型和螺型分量相同
    mobility = MobilityLaw(
        mobility_law='FCC_0',
        state=state,
        Medge=64103.0,     # 刃型位错迁移率 (Pa·s)^{-1}，FCC Cu 标准值
        Mscrew=64103.0,    # 螺型位错迁移率
        vmax=50.0,        # 速度上限（无量纲），防止数值不稳定
    )

    # ── 时间积分模块 ──────────────────────────────────────────────────────────
    # Subcycling：对不同长度的段使用不同步长，兼顾精度和效率
    # rgroups 划分：[0, 50] 短段（精细），[50, 250] 中段，[250, 500] 长段
    # 与 minseg(100) 和 maxseg(500) 匹配
    timeint = TimeIntegration(
        integrator='Subcycling',
        rgroups=[0.0, 100.0, 400.0, 800.0],
        state=state,
        force=calforce,
        mobility=mobility,
    )

    # ── 碰撞模块 ──────────────────────────────────────────────────────────────
    # Retroactive：事后碰撞检测，在每步结束后检查是否有段在时间步内相交
    # 这是奥罗万环形成的关键模块：当绕过夹杂的两臂在背面靠近到 rann 内，
    # 此模块触发 merge，形成封闭环
    collision = Collision(collision_mode='Retroactive', state=state)

    # ── 拓扑模块 ──────────────────────────────────────────────────────────────
    topology = Topology(
        topology_mode='TopologyParallel',
        state=state,
        force=calforce,
        mobility=mobility,
    )

    # ── 重网格模块 ────────────────────────────────────────────────────────────
    remesh = Remesh(remesh_rule='LengthBased', state=state)
    #cross_slip = CrossSlip(state=state, cross_slip_mode='ForceBasedParallel', force=calforce)
    # ── 加载条件 ──────────────────────────────────────────────────────────────
    loading_mode = 'strain_rate'
    erate        = 1000.0            # 应变率 /s
    max_strain   = 0.005             # 目标应变 0.5%
    edir         = np.array([0., 0., 1.])   # 加载方向 [001]

    # ── 模拟控制 ──────────────────────────────────────────────────────────────
    sim = SimulateNetworkPerf(
        calforce=calforce,
        mobility=mobility,
        timeint=timeint,
        collision=collision,
        topology=topology,
        remesh=remesh,
        #cross_slip=cross_slip,
        vis=None,
        loading_mode=loading_mode,
        erate=erate,
        edir=edir,
        max_strain=max_strain,
        burgmag=state['burgmag'],
        state=state,
        # 输出频率设置
        print_freq=10,           # 每步打印一行状态
        plot_freq=50,            # 关闭实时可视化（服务器上运行时设为 0）
        plot_pause_seconds=0.0001,
        write_freq=1,          # 每 50 步写一次配置文件（减少 I/O 开销）
        write_dir=output_dir,
        restart=restart,
    )

    print("=" * 55)
    print("开始模拟")
    print("=" * 55)
    print(f"  目标应变    : {max_strain*100:.1f}%")
    print(f"  应变率      : {erate} /s")
    print(f"  加载方向    : {edir}")
    print(f"  输出目录    : {output_dir}")
    print("=" * 55 + "\n")

    try:
        sim.run(net, state)
        print("\n✓ 模拟正常完成")
    except Exception as e:
        print(f"\n✗ 模拟出错: {e}")
        traceback.print_exc()

    pyexadis.finalize()


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.environ['OMP_PROC_BIND'] = 'spread'
    os.environ['OMP_PLACES']    = 'threads'

    try:
        run_orowan_verification()
    except Exception as e:
        print(f"运行失败: {e}")
        traceback.print_exc()