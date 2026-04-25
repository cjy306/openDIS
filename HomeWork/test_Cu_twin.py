"""
Cu FCC 位错动力学模拟：球形杂质（Orowan bypass）+ 孪晶面（twin boundary blocking）
"""
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


# 模拟参数 (与 test_Cu.py 一致)
state = {
    "crystal": 'fcc',
    "burgmag": 0.2556e-9,
    "mu":      48e9,
    "nu":      0.324,
    "a":       3.0,
    "maxseg":  700,
    "minseg":  100,
    "rtol":    0.75,
    "rann":    1.5,
    "nextdt":  1e-10,
    "maxdt":   1e-9,
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
# 生成孪晶面（返回平面上的点和法向量，Burgers 单位）
# ============================================================
def generate_twin_planes(Lbox_b, twin_z_fractions):
    """
    在盒子的指定 z 分数位置放置孪晶面。
    例如 twin_z_fractions=[0.5] 表示在盒子中间放一个孪晶面。
    twin_z_fractions=[0.33, 0.67] 表示在 1/3 和 2/3 处各放一个。
    所有孪晶面法向量为 [0,0,1]（垂直于 z 轴）。
    """
    points_b  = []
    normals_b = []
    for frac in twin_z_fractions:
        z = Lbox_b * frac
        points_b.append([Lbox_b/2.0, Lbox_b/2.0, z])
        normals_b.append([0.0, 0.0, 1.0])
        print(f"  Twin plane at z = {z:.1f} b (fraction = {frac})")
    return points_b, normals_b


def save_twin_planes_data(filename, points_b, normals_b, Lbox_b, burgmag):
    """保存孪晶面信息为 data 文件"""
    with open(filename, 'w') as f:
        f.write("# Twin Boundary Planes Data File\n")
        f.write("#" + "="*68 + "\n")
        f.write(f"# Burgers vector magnitude: {burgmag:.6e} m\n")
        f.write(f"# Box size: {Lbox_b:.2f} b\n")
        f.write(f"# Total planes: {len(points_b)}\n")
        f.write("#" + "="*68 + "\n")
        f.write("# All coordinates in Burgers vector units (b)\n")
        f.write("#\n")
        f.write("# Column format:\n")
        f.write("#   1. ID\n")
        f.write("#   2. Point_X(b)  - X coordinate of a point on the plane\n")
        f.write("#   3. Point_Y(b)  - Y coordinate\n")
        f.write("#   4. Point_Z(b)  - Z coordinate\n")
        f.write("#   5. Normal_X    - X component of unit normal\n")
        f.write("#   6. Normal_Y    - Y component\n")
        f.write("#   7. Normal_Z    - Z component\n")
        f.write("#" + "="*68 + "\n")
        for i in range(len(points_b)):
            p = points_b[i]
            n = normals_b[i]
            f.write(f"{i+1:6d} {p[0]:16.8e} {p[1]:16.8e} {p[2]:16.8e} "
                    f"{n[0]:10.6f} {n[1]:10.6f} {n[2]:10.6f}\n")
        f.write("#" + "="*68 + "\n")
        f.write(f"# END OF DATA ({len(points_b)} planes)\n")
    print(f"Saved twin planes data: {filename}")


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
# 按密度生成位错网络（避开杂质和孪晶面）
# ============================================================
def generate_dislocation_network(Lbox_m, burgmag, target_density, seed=12345,
                                  precip_centers_m=None, precip_radii_m=None,
                                  twin_z_positions_m=None):
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
        r_m = rng.uniform(0.3e-6, 1.0e-6)
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

        # 检查与孪晶面重叠（位错环中心不能太靠近孪晶面）
        if not overlap and twin_z_positions_m is not None:
            twin_margin = r_m * 1.5  # 环半径的1.5倍安全距离
            for tz in twin_z_positions_m:
                if abs(c_m[2] - tz) < twin_margin:
                    overlap = True
                    break

        if overlap:
            attempt += 1; continue

        burg = bset[loop_count % len(bset)]
        circ = 2 * np.pi * r_m
        nseg = max(4, min(30, int(np.ceil(circ / (burgmag * state["maxseg"])))))

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
def run_simulation(net, output_dir, restart_id=None,
                   centers_b=None, radii_b=None,
                   twin_points_b=None, twin_normals_b=None):
    restart = None
    if restart_id is None:
        net_manager = DisNetManager(net)
    else:
        net_manager, restart = read_restart(
            state=state,
            restart_file=os.path.join(output_dir, f'restart.{restart_id}.exadis')
        )

    exadis_net = net_manager.get_disnet(ExaDisNet)

    # 加载球形杂质
    if centers_b is not None and len(centers_b) > 0:
        exadis_net.load_obstacles([list(c) for c in centers_b], list(radii_b))

    # 加载孪晶面
    if twin_points_b is not None and len(twin_points_b) > 0:
        exadis_net.load_twin_planes(twin_points_b, twin_normals_b)

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
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
    Lbox_b         = Lbox_m / burgmag
    target_density = float(os.environ.get('RHO_TARGET', '1e12'))
    sphere_count   = int(os.environ.get('SPHERE_COUNT', '100'))
    output_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_Cu_twin')
    os.makedirs(output_dir, exist_ok=True)

    # ---- 球形杂质 ----
    centers_b, radii_b, centers_m, radii_m = generate_precipitates(
        Lbox_m, burgmag, count=sphere_count, diameter_m=200e-9)
    print(f"Generated {len(centers_b)} spherical precipitates")

    # ---- 孪晶面 ----
    # 在盒子 z 方向 1/2 处放一个孪晶面（可修改为多个）
    twin_z_fractions = [0.5]
    twin_points_b, twin_normals_b = generate_twin_planes(Lbox_b, twin_z_fractions)
    print(f"Generated {len(twin_points_b)} twin boundary planes")

    # 保存孪晶面 data 文件
    save_twin_planes_data(
        os.path.join(output_dir, 'twin_planes.data'),
        twin_points_b, twin_normals_b, Lbox_b, burgmag)

    # 孪晶面 z 坐标（米制，用于位错网络生成时避开）
    twin_z_positions_m = [Lbox_m * f for f in twin_z_fractions]

    if args.restart is not None:
        run_simulation(ExaDisNet(), output_dir=output_dir,
                       restart_id=args.restart,
                       centers_b=centers_b, radii_b=radii_b,
                       twin_points_b=twin_points_b, twin_normals_b=twin_normals_b)
    else:
        G = generate_dislocation_network(
            Lbox_m, burgmag, target_density, seed=12345,
            precip_centers_m=centers_m, precip_radii_m=radii_m,
            twin_z_positions_m=twin_z_positions_m)
        run_simulation(G, output_dir=output_dir,
                       centers_b=centers_b, radii_b=radii_b,
                       twin_points_b=twin_points_b, twin_normals_b=twin_normals_b)

    pyexadis.finalize()


if __name__ == "__main__":
    main()
