"""
生成 Case A 种子构型:Frank-Read 源闭合回路 —— ODS-FeCrAl 课题

用途:作为"预变形"的种子。流程:
  本种子 → 加载到 1% → 看应力-应变曲线取流变平台帧 → 弛豫 → 得到 Case A 基体。
  (预变形是数值手段:让 DDD 演化出稳定可动的位错网络;平台态由密度+外力决定,
   与初始种子形态无关,故种子用 FR 即可。)

FR 回路构型(照搬 ExaDiS 官方 02_frank_read_src,旋转到各滑移系):
  5 节点矩形:顶边中点 UNCONSTRAINED(可动臂),其余 4 角 PINNED 兜成闭合回路。
  覆盖全 12 个 <111>{110} 滑移系([001] 加载分类):
    - 8 活动系(Schmid=0.41)均分主体密度 → 加载开动、增殖(主力)
    - 4 林位错系(Schmid=0)每系象征性放几个 → 不开动,提供交割硬化
  (交滑移在 test 脚本里开启,让活动位错还能跨面 → 三维流变网络)

用法:
  python generate_caseA.py --seed 12345
  → 写 init_data_caseA_seed12345/{init_config.data, init_config_labeled.vtk}
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager, NodeConstraints
from pyexadis_utils import write_vtk

# ============================================================
# 参数(材料 = Yan 2023 一套)
# ============================================================
BURGMAG    = 0.248e-9    # b [m]
LBOX_M     = 2.0e-6      # bulk 周期盒 2μm 立方
RHO_TARGET = 5.0e12      # 种子位错密度 [1/m^2]
ARM_LEN_M  = 200e-9      # FR 回路臂长 [m](τ_c≈μb/L≈100MPa)
FOREST_PER_SYS = 2       # 每个 Schmid=0 林位错系放几个 FR(象征性,提供交割硬化)

# 12 个 BCC <111>{110} 滑移系(b 在 plane 内,b·n=0);[001] 加载 Schmid 分类:
#   0-7  = 活动系(n 含 z 分量,Schmid=0.41)→ 加载开动、增殖(主力)
#   8-11 = 林位错系(n=(110)型无 z,Schmid=0)→ 不开动,象征性放,提供交割硬化
_SLIP_B = np.array([
    [-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.], [1., -1., 1.],   # 0-3 活动
    [-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.], [1., -1., 1.],   # 4-7 活动
    [-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.], [1., -1., 1.],   # 8-11 林
])
_SLIP_N = np.array([
    [0., -1., 1.], [0., -1., 1.], [0., 1., 1.], [0., 1., 1.],     # 0-3 活动(n含z)
    [1., 0., 1.], [-1., 0., 1.], [1., 0., 1.], [-1., 0., 1.],     # 4-7 活动(n含z)
    [1., 1., 0.], [-1., 1., 0.], [-1., 1., 0.], [1., 1., 0.],     # 8-11 林(n无z,Schmid=0)
])
_ACTIVE_IDS = list(range(8))    # Schmid=0.41
_FOREST_IDS = list(range(8, 12))  # Schmid=0


def insert_fr_loop(nodes, segs, burg, plane, arm_len_b, center):
    """插入一个 Frank-Read 闭合回路(官方 02_frank_read_src 构型,旋转到给定滑移系)。

    官方(标准取向):b‖x,可动臂‖y(刃型),虚拟臂‖-z,三者两两正交。
    本函数把 (x,y,z) 轴换成局部正交基 (b̂, e, f):
      b̂=burg;  e=(n×b̂) 归一(可动臂,面内垂直 b,刃型);  f=b̂×e(虚拟臂,对应官方 -z)。
    刚体旋转保持官方各段 b·plane=0 的合法性。
    """
    b = np.asarray(burg, dtype=float); b = b / np.linalg.norm(b)
    n = np.asarray(plane, dtype=float); n = n / np.linalg.norm(n)
    e = np.cross(n, b); e = e / np.linalg.norm(e)
    f = np.cross(b, e); f = f / np.linalg.norm(f)

    L = arm_len_b
    c = np.asarray(center, dtype=float)
    istart = len(nodes)
    pts = [
        (c - 0.5 * L * e,          NodeConstraints.PINNED_NODE),    # 0 顶左角
        (c,                        NodeConstraints.UNCONSTRAINED),  # 1 顶中(可动)
        (c + 0.5 * L * e,          NodeConstraints.PINNED_NODE),    # 2 顶右角
        (c + 0.5 * L * e - L * f,  NodeConstraints.PINNED_NODE),    # 3 底右角
        (c - 0.5 * L * e - L * f,  NodeConstraints.PINNED_NODE),    # 4 底左角
    ]
    for p, cst in pts:
        nodes.append(np.concatenate((p, [cst])))

    N = len(pts)
    for i in range(N):
        n1 = istart + i
        n2 = istart + (i + 1) % N   # 末节点接回首节点 → 闭合回路
        ldir = pts[(i + 1) % N][0] - pts[i][0]
        pn = np.cross(b, ldir)
        nn = np.linalg.norm(pn)
        pn = pn / nn if nn > 1e-10 else n
        segs.append(np.concatenate(([n1, n2], b, pn)))
    return nodes, segs


def build_seed(cell, rng, verbose=True):
    """铺 FR 回路覆盖全 12 滑移系:8 活动系均分主体密度 + 4 林系每系 FOREST_PER_SYS 个。

    每个矩形回路周长 ≈ 4*臂长。总密度 = RHO_TARGET。
    """
    Lbox = LBOX_M / BURGMAG
    vol_m3 = LBOX_M ** 3
    L_target_m = RHO_TARGET * vol_m3
    arm_b = ARM_LEN_M / BURGMAG
    per_loop_m = 4.0 * ARM_LEN_M

    bsys = _SLIP_B / np.linalg.norm(_SLIP_B, axis=1)[:, None]
    nsys = _SLIP_N / np.linalg.norm(_SLIP_N, axis=1)[:, None]

    origin, h = np.array(cell.origin), np.array(cell.h)
    margin = arm_b
    nodes, segs = [], []

    def place(isys, ntimes):
        nonlocal nodes, segs
        for _ in range(ntimes):
            center = rng.uniform(margin, Lbox - margin, size=3)
            nodes, segs = insert_fr_loop(nodes, segs, bsys[isys], nsys[isys], arm_b, center)

    # 1) 林位错系:每系固定 FOREST_PER_SYS 个(象征性,提供交割硬化)
    n_forest = 0
    for isys in _FOREST_IDS:
        place(isys, FOREST_PER_SYS)
        n_forest += FOREST_PER_SYS
    forest_len_m = n_forest * per_loop_m

    # 2) 活动系:8 系均分剩余(主体)密度
    L_active_m = max(0.0, L_target_m - forest_len_m)
    n_active_total = int(round(L_active_m / per_loop_m))
    per_active = max(1, n_active_total // len(_ACTIVE_IDS))
    n_active = 0
    for isys in _ACTIVE_IDS:
        place(isys, per_active)
        n_active += per_active

    total = n_active + n_forest
    achieved = total * per_loop_m / vol_m3
    if verbose:
        print('FR 回路: 活动系 %d 个(8系×%d) + 林系 %d 个(4系×%d), 共 %d, 臂长 %.0fnm, '
              '密度 %.2e /m^2 (目标 %.2e)'
              % (n_active, per_active, n_forest, FOREST_PER_SYS, total,
                 ARM_LEN_M * 1e9, achieved, RHO_TARGET))
    return nodes, segs


def main():
    import argparse
    pyexadis.initialize()
    parser = argparse.ArgumentParser(description='生成 Case A 预变形种子(FR 回路)')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, f'init_data_caseA_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    cell = pyexadis.Cell(LBOX_M / BURGMAG)
    nodes, segs = build_seed(cell, rng)

    G = ExaDisNet(cell, nodes, segs)
    out_file = os.path.join(out_dir, 'init_config.data')
    G.write_data(out_file)

    vtk_file = os.path.join(out_dir, 'init_config_labeled.vtk')
    write_vtk(DisNetManager(G), vtk_file, crystal='BCC', verbose=False)

    print(f'Case A 种子已写出: {out_file}')
    print(f'VTK: {vtk_file}')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
