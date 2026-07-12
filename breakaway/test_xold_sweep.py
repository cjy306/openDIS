"""
test_xold_sweep.py —— 验证扫掠重建 xold 升级(CHANGELOG 改动[5])的可行性

改动[5]把 Phase A 的步初位置从 "pos - dt*v 反推" 换成 "system->xold 记录"。
本脚本验证升级后 捕获→钉扎→释放 全链行为正确。

用法:
  1) 跑仿真(超算,需已编译含改动[5]的 .so):
       python test_xold_sweep.py run [--max_step 10000] [--erate 1e3]
     构型/参数与 test_breakaway_prismatic.py 完全一致,输出到 output_xold_sweep/
  2) 验证(本地即可,只读 .data 文件):
       python test_xold_sweep.py verify output_xold_sweep
     基线对比:同一命令跑在旧版输出上 → python test_xold_sweep.py verify output_breakaway_single

验证判据(自动 PASS/FAIL):
  V1 捕获发生   : 存在帧含 constraint=7 且贴近某障碍中心的节点(排除远离障碍的边界锚点)
  V2 钉点位置   : 钉扎期间 |pin - 障碍中心| < tol —— xold 版检测/τ回退/Cproj 没跑偏
  V3 钉扎零漂移 : pin 节点逐帧位移 < drift tol —— 无瞬移、无抖动
  V4 释放完成   : 曾被钉的障碍在末帧不再有 pin —— 机制循环完整
日志侧(人工): grep "Breakaway" slurm-*.out 应照常出现 captured/depin 行。
"""
import os, sys, re, glob
import numpy as np

# ===== 与 test_breakaway_prismatic.py 保持一致的参数 =====
EDIR  = np.array([0., 0., 1.])
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

PINNED = 7          # constraint 值
TOL_PIN   = 5.0     # V2: 钉点离障碍中心容差 [b]
TOL_DRIFT = 0.5     # V3: 钉扎期间逐帧漂移容差 [b]


# ---------------------------------------------------------------- run 模式
def run(max_step, erate):
    pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
    [sys.path.append(os.path.abspath(p)) for p in pyexadis_paths if p not in sys.path]
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh

    pyexadis.initialize()
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = os.path.join(base_dir, 'init_breakaway')
    output_dir = os.path.join(base_dir, 'output_xold_sweep')
    os.makedirs(output_dir, exist_ok=True)

    G = ExaDisNet()
    G.read_paradis(os.path.join(init_dir, 'init_config.data'))
    net = DisNetManager(G)

    obs = np.loadtxt(os.path.join(init_dir, 'obstacles.data'))
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    centers_b, radii_b = obs[:, :3], obs[:, 3]
    net.get_disnet(ExaDisNet).load_obstacles([list(c) for c in centers_b], list(radii_b), type=1)
    print(f'读入 {len(centers_b)} 个障碍 (type=1 breakaway)')

    cell = net.get_disnet(ExaDisNet).cell
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=32, cell=cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state,
                            Medge=15000.0, Mscrew=3000.0, Mclimb=100.0, vmax=30000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[], state=state,
                                force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Orowan', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state,
                         force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate', erate=erate, edir=EDIR, max_step=max_step,
        burgmag=state["burgmag"], state=state,
        print_freq=1, write_freq=1, write_dir=output_dir)
    sim.run(net, state)
    pyexadis.finalize()
    print(f'\n跑完。验证: python test_xold_sweep.py verify {os.path.basename(output_dir)}')


# ---------------------------------------------------------------- verify 模式
_PRIMARY = re.compile(
    r'^\s*(\d+),\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+(\d+)\s+(\d+)\s*$')

def read_frame(path):
    """解析 ParaDiS legacy .data,返回 (nodes[x,y,z,narms,constraint], box_lengths)"""
    nodes, box = [], None
    with open(path) as f:
        in_nodal = False
        for line in f:
            if not in_nodal:
                if box is None:
                    t = line.split()
                    if len(t) == 7 and ',' not in t[0]:  # dom minx miny minz maxx maxy maxz
                        try:
                            v = [float(x) for x in t]
                            box = np.array(v[4:7]) - np.array(v[1:4])
                        except ValueError:
                            pass
                if 'nodalData' in line:
                    in_nodal = True
                continue
            m = _PRIMARY.match(line)
            if m:
                nodes.append([float(m.group(3)), float(m.group(4)), float(m.group(5)),
                              int(m.group(6)), int(m.group(7))])
    return np.array(nodes), box

def min_image(d, box):
    return d - box * np.round(d / box)

def verify(output_dir):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out = output_dir if os.path.isdir(output_dir) else os.path.join(base_dir, output_dir)
    obs = np.loadtxt(os.path.join(base_dir, 'init_breakaway', 'obstacles.data'))
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    centers, radii = obs[:, :3], obs[:, 3]
    Nobs = len(centers)

    files = glob.glob(os.path.join(out, 'config.*.data'))
    steps = sorted(int(re.search(r'config\.(\d+)\.data', f).group(1)) for f in files)
    if not steps:
        print(f'FAIL: {out} 里没有 config.*.data'); return
    print(f'{len(steps)} 帧 (step {steps[0]} ~ {steps[-1]}), {Nobs} 个障碍, '
          f'tol_pin={TOL_PIN}b tol_drift={TOL_DRIFT}b\n')

    # 逐帧: pin_of_obs[j] = 该帧钉在障碍 j 上的节点位置(取最近的一个), None=无
    history = []          # 每帧一个 dict {j: pos}
    box = None
    for s in steps:
        nodes, b = read_frame(os.path.join(out, f'config.{s}.data'))
        if box is None: box = b
        frame = {}
        if len(nodes):
            pins = nodes[nodes[:, 4] == PINNED]
            for p in pins:
                d = np.linalg.norm(min_image(centers - p[:3], box), axis=1)
                j = int(np.argmin(d))
                if d[j] < max(3 * TOL_PIN, radii[j]):        # 排除远处的边界锚点
                    if j not in frame or d[j] < np.linalg.norm(
                            min_image(centers[j] - frame[j], box)):
                        frame[j] = p[:3]
        history.append(frame)

    ever = sorted({j for f in history for j in f})
    # V1
    print(f'V1 捕获发生   : {"PASS" if ever else "FAIL"}  '
          f'(被钉过的障碍 {len(ever)}/{Nobs}: {ever})')
    # V2
    dmax = 0.0
    for f in history:
        for j, p in f.items():
            dmax = max(dmax, np.linalg.norm(min_image(centers[j] - p, box)))
    print(f'V2 钉点位置   : {"PASS" if ever and dmax < TOL_PIN else "FAIL"}  '
          f'(max|pin-障碍| = {dmax:.3f} b)')
    # V3
    drift_max, recapture = 0.0, 0
    for j in ever:
        prev, seen = None, False
        for f in history:
            cur = f.get(j)
            if cur is not None and prev is not None:
                drift_max = max(drift_max, np.linalg.norm(min_image(cur - prev, box)))
            if cur is not None and prev is None and seen:
                recapture += 1
            if cur is not None:
                seen = True
            prev = cur
    print(f'V3 钉扎零漂移 : {"PASS" if drift_max < TOL_DRIFT else "FAIL"}  '
          f'(max 逐帧漂移 = {drift_max:.3f} b, 重捕事件 = {recapture})')
    # V4
    still = sorted(j for j in ever if history[-1].get(j) is not None)
    print(f'V4 释放完成   : {"PASS" if not still else "FAIL"}  '
          f'(末帧仍被钉: {still if still else "无"})')
    # 附:每颗障碍的钉扎时长
    for j in ever:
        on = [i for i, f in enumerate(history) if j in f]
        print(f'   障碍 {j}: 钉扎帧 {steps[on[0]]} ~ {steps[on[-1]]} (共 {len(on)} 帧)')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('run');    r.add_argument('--max_step', type=int, default=10000)
    r.add_argument('--erate', type=float, default=1e3)
    v = sub.add_parser('verify'); v.add_argument('output_dir')
    a = ap.parse_args()
    if a.mode == 'run':
        run(a.max_step, a.erate)
    else:
        verify(a.output_dir)
