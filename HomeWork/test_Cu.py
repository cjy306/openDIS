import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart, NodeConstraints
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e


# 模拟参数
state = {
    "crystal": 'fcc',
    "burgmag": 0.2556e-9,
    "mu":      48e9,
    "nu":      0.324,
    "a":       2.0,
    "maxseg":  400,
    "minseg":  100,
    "rtol":    0.5,
    "rann":    1.0,
    "nextdt":  1e-9,
    "maxdt":   1e-6,
}


# ============================================================
# 生成球形杂质（返回 Burgers 单位和米制坐标）
# ============================================================
def generate_precipitates(Lbox_m, burgmag, count, diameter_m, seed=12345):
    rng = np.random.RandomState(seed)
    radius_m = diameter_m / 2.0
    radius_b = radius_m / burgmag
    margin_m = radius_m * 1.5

    centers_b, radii_b = [], []
    centers_m, radii_m = [], []

    for _ in range(count):
        for _ in range(800):
            c_m = np.array([rng.uniform(margin_m, Lbox_m - margin_m) for _ in range(3)])
            overlap = any(np.linalg.norm(c_m - centers_m[i]) < (radius_m + radii_m[i]) * 1.4
                          for i in range(len(centers_m)))
            if not overlap:
                centers_b.append(c_m / burgmag)
                radii_b.append(radius_b)
                centers_m.append(c_m)
                radii_m.append(radius_m)
                break

    return (np.array(centers_b) if centers_b else np.empty((0, 3)),
            np.array(radii_b)  if radii_b  else np.empty(0),
            np.array(centers_m) if centers_m else np.empty((0, 3)),
            np.array(radii_m)  if radii_m  else np.empty(0))


# ============================================================
# 插入棱柱位错环
# ============================================================
def insert_prismatic_loop(cell, nodes, segs, burg, radius, center, maxseg=-1):
    b = -1.0 * burg
    Nsides = 4
    b0  = 1.0/np.sqrt(2.0)*np.array([[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,0],[-1,1,0]])
    n01 = np.array([[-1,-1,1],[-1,1,1],[-1,1,1],[1,1,1],[-1,1,1],[1,1,1]])
    n02 = np.array([[1,-1,1],[1,1,1],[-1,-1,1],[1,-1,1],[-1,1,-1],[1,1,-1]])
    bcol = np.abs(np.abs(np.dot(b0, b))-1.0)
    ib = bcol.argmin()
    if bcol[ib] > 1e-5:
        raise ValueError('FCC Burgers vector must be of the 1/2<110> type')

    p1 = n01[ib] / np.linalg.norm(n01[ib])
    p2 = n02[ib] / np.linalg.norm(n02[ib])
    l1 = np.cross(p1, b); l1 = l1 / np.linalg.norm(l1)
    l2 = np.cross(p2, b); l2 = l2 / np.linalg.norm(l2)
    e = np.array([-0.5*l1-0.5*l2, +0.5*l1-0.5*l2, +0.5*l1+0.5*l2, -0.5*l1+0.5*l2])
    n = np.array([p1, p2, p1, p2])
    n = n / np.linalg.norm(n, axis=1)[:,None]

    istart = len(nodes)
    Nnodes = 0
    for i in range(Nsides):
        l = radius*(e[(i+1)%Nsides]-e[i])
        Nseg = int(np.ceil(np.linalg.norm(l)/maxseg)) if maxseg > 0 else 1
        for j in range(Nseg):
            p = radius*e[i]+1.0*j/Nseg*l+center
            nodes.append(np.concatenate((p, [NodeConstraints.UNCONSTRAINED])))
            n1 = istart+Nnodes
            n2 = istart if (i == Nsides-1 and j == Nseg-1) else n1+1
            segs.append(np.concatenate(([n1, n2], b, n[i])))
            Nnodes += 1
    return nodes, segs


# ============================================================
# 按密度生成位错网络
# ============================================================
def generate_dislocation_network(Lbox_m, burgmag, target_density, seed=12345,
                                  precip_centers_m=None, precip_radii_m=None):
    Lbox = int(round(Lbox_m / burgmag))
    total_length = target_density * Lbox_m**3

    rng  = np.random.RandomState(seed)
    cell = pyexadis.Cell(Lbox)
    nodes, segs = [], []
    accumulated = 0.0
    loop_radii_m, loop_centers_m = [], []

    bset = np.array([[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,0],[-1,1,0]]) / np.sqrt(2.0)
    bset = bset / np.linalg.norm(bset, axis=1)[:, None]

    attempt, loop_count = 0, 0
    while accumulated < total_length * 0.85 and attempt < 15000:
        r_m = rng.uniform(0.15e-6, 0.5e-6)
        r_b = r_m / burgmag
        margin = r_m * 1.3
        low, high = margin, Lbox_m - margin
        if high <= low:
            attempt += 1; continue

        c_m = np.array([rng.uniform(low, high) for _ in range(3)])

        # 检查与已有位错环重叠
        overlap = any(np.linalg.norm(c_m - loop_centers_m[i]) < (r_m + loop_radii_m[i]) * 1.3
                      for i in range(len(loop_centers_m)))

        # 检查与杂质重叠
        if not overlap and precip_centers_m is not None and len(precip_centers_m) > 0:
            overlap = any(np.linalg.norm(c_m - precip_centers_m[i]) < (r_m + precip_radii_m[i]) * 1.5
                          for i in range(len(precip_centers_m)))

        if overlap:
            attempt += 1; continue

        burg = bset[loop_count % len(bset)]
        circ = 2 * np.pi * r_m
        nseg = max(8, min(150, int(np.ceil(circ / (burgmag * 20)))))

        try:
            nodes, segs = insert_prismatic_loop(
                cell, nodes.copy(), segs.copy(), burg, r_b, c_m / burgmag, nseg)
            loop_radii_m.append(r_m)
            loop_centers_m.append(c_m)
            accumulated += circ
            loop_count += 1
            attempt = 0
        except Exception:
            attempt += 1

    return ExaDisNet(cell, nodes, segs)


# ============================================================
# 模拟主函数
# ============================================================
def run_simulation(net, output_dir, restart_id=None, centers_b=None, radii_b=None):
    restart = None
    if restart_id is None:
        net_manager = DisNetManager(net)
    else:
        net_manager, restart = read_restart(
            state=state,
            restart_file=os.path.join(output_dir, f'restart.{restart_id}.exadis')
        )

    # 加载球形杂质到 C++ System
    if centers_b is not None and len(centers_b) > 0:
        exadis_net = net_manager.get_disnet(ExaDisNet)
        exadis_net.load_obstacles([list(c) for c in centers_b], list(radii_b))

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=10000.0, Mscrew=1000.0, vmax=20000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Orowan', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state,
                         force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        erate=1e3,
        edir=np.array([0., 0., 1.]),
        max_strain=0.01,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=100,
        write_dir=output_dir,
        restart=restart,
    )
    sim.run(net_manager, state)


# ============================================================
# 主程序
# ============================================================
def main():
    pyexadis.initialize()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', type=int, help='从指定步骤重启')
    args = parser.parse_args()

    burgmag        = state["burgmag"]
    Lbox_m         = 5e-6
    target_density = float(os.environ.get('RHO_TARGET', '1e12'))
    sphere_count   = int(os.environ.get('SPHERE_COUNT', '100'))
    output_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_Cu_fcc')
    os.makedirs(output_dir, exist_ok=True)

    # 生成球形杂质
    centers_b, radii_b, centers_m, radii_m = generate_precipitates(
        Lbox_m, burgmag, count=sphere_count, diameter_m=200e-9)
    print(f"Generated {len(centers_b)} precipitates")

    if args.restart is not None:
        run_simulation(ExaDisNet(), output_dir=output_dir,
                       restart_id=args.restart,
                       centers_b=centers_b, radii_b=radii_b)
    else:
        G = generate_dislocation_network(
            Lbox_m, burgmag, target_density, seed=12345,
            precip_centers_m=centers_m, precip_radii_m=radii_m)
        run_simulation(G, output_dir=output_dir,
                       centers_b=centers_b, radii_b=radii_b)

    pyexadis.finalize()


if __name__ == "__main__":
    main()
