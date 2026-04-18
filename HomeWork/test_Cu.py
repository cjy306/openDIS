import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart, NodeConstraints
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
    from pyexadis_base import get_exadis_params
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e


# ============================================================
# 球形杂质类
# ============================================================
class SphericalPrecipitates:

    def __init__(self, Lbox_m, burgmag, seed=42):
        self.Lbox_m    = Lbox_m
        self.burgmag   = burgmag
        self.Lbox_b    = Lbox_m / burgmag
        self.centers   = []
        self.radii     = []
        self.types     = []
        self.centers_m = []
        self.radii_m   = []
        self.rng = np.random.RandomState(seed)

    def generate(self, large_count, large_diameter_m, small_count=0, small_diameter_m=0):
        large_radius_m = large_diameter_m / 2.0
        small_radius_m = small_diameter_m / 2.0
        large_radius_b = large_radius_m / self.burgmag
        small_radius_b = small_radius_m / self.burgmag

        all_specs = []
        for _ in range(large_count):
            all_specs.append(('large', large_radius_m, large_radius_b))
        for _ in range(small_count):
            all_specs.append(('small', small_radius_m, small_radius_b))

        for stype, r_m, r_b in all_specs:
            margin_m = r_m * 1.5
            for attempt in range(800):
                cx = self.rng.uniform(margin_m, self.Lbox_m - margin_m)
                cy = self.rng.uniform(margin_m, self.Lbox_m - margin_m)
                cz = self.rng.uniform(margin_m, self.Lbox_m - margin_m)
                center_m = np.array([cx, cy, cz])

                overlap = False
                for i in range(len(self.centers_m)):
                    if np.linalg.norm(center_m - self.centers_m[i]) < (r_m + self.radii_m[i]) * 1.4:
                        overlap = True
                        break

                if not overlap:
                    self.centers.append(center_m / self.burgmag)
                    self.radii.append(r_b)
                    self.types.append(stype)
                    self.centers_m.append(center_m)
                    self.radii_m.append(r_m)
                    break

        self.centers   = np.array(self.centers)   if self.centers   else np.empty((0, 3))
        self.radii     = np.array(self.radii)     if self.radii     else np.empty(0)
        self.centers_m = np.array(self.centers_m) if self.centers_m else np.empty((0, 3))
        self.radii_m   = np.array(self.radii_m)   if self.radii_m   else np.empty(0)
        return self

    def save_data_file(self, filename):
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(filename, 'w') as f:
            f.write("# Spherical Precipitates Data File\n")
            f.write(f"# Total precipitates: {len(self.centers)}\n")
            f.write(f"# Burgers vector magnitude: {self.burgmag:.6e} m\n")
            f.write("# ID  X(b)  Y(b)  Z(b)  Radius(b)  Diameter(nm)\n")
            f.write("#" + "-"*60 + "\n")
            for i in range(len(self.centers)):
                c = self.centers[i]
                r = self.radii[i]
                d = self.radii_m[i] * 2 * 1e9
                f.write(f"{i+1:8d} {c[0]:16.8e} {c[1]:16.8e} "
                        f"{c[2]:16.8e} {r:16.8e} {d:12.2f}\n")


# ============================================================
# 插入棱柱位错环
# ============================================================
def insert_prismatic_loop(crystal, cell, nodes, segs, burg, radius, center, maxseg=-1, Rorient=None):
    b = -1.0 * burg
    if crystal.lower() != 'fcc':
        raise ValueError(f'Unsupported crystal type: {crystal}')

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

    if Rorient is not None:
        Rorient = np.array(Rorient)
        Rorient = Rorient / np.linalg.norm(Rorient, axis=1)[:,None]
        b = np.matmul(b, Rorient.T)
        e = np.matmul(e, Rorient.T)
        n = np.matmul(n, Rorient.T)

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
# 生成位错网络
# ============================================================
def generate_prismatic_config_by_density(crystal, Lbox_m, burgmag, target_density,
                                         radius_min_m=0.15e-6, radius_max_m=0.5e-6,
                                         seed=12345, precipitates=None):
    Lbox         = int(round(Lbox_m / burgmag))
    volume_m3    = Lbox_m**3
    total_length = target_density * volume_m3

    rng  = np.random.RandomState(seed)
    cell = pyexadis.Cell(Lbox)
    nodes, segs = [], []
    accumulated_length_m = 0.0
    radii_m, centers     = [], []

    bset = np.array([[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,0],[-1,1,0]]) / np.sqrt(2.0)
    bset = bset / np.linalg.norm(bset, axis=1)[:, None]
    nburg = bset.shape[0]

    max_attempts = 15000
    attempt    = 0
    loop_count = 0

    while accumulated_length_m < total_length * 0.85 and attempt < max_attempts:
        radius_m = rng.uniform(radius_min_m, radius_max_m)
        radius   = radius_m / burgmag
        margin   = radius_m * 1.3
        low, high = margin, Lbox_m - margin
        if high <= low:
            attempt += 1
            continue

        center_m = np.array([rng.uniform(low, high) for _ in range(3)])
        center   = center_m / burgmag

        overlap = False
        for i, ec in enumerate(centers):
            if np.linalg.norm(center_m - ec) < (radius_m + radii_m[i]) * 1.3:
                overlap = True
                break

        if not overlap and precipitates is not None and len(precipitates.centers) > 0:
            for i in range(len(precipitates.centers_m)):
                if np.linalg.norm(center_m - precipitates.centers_m[i]) < (radius_m + precipitates.radii_m[i]) * 1.5:
                    overlap = True
                    break

        if overlap:
            attempt += 1
            continue

        burg = bset[loop_count % nburg]
        circumference_m = 2 * np.pi * radius_m
        segs_per_loop = max(8, min(150, int(np.ceil(circumference_m / (burgmag * 20)))))

        try:
            new_nodes, new_segs = insert_prismatic_loop(
                crystal.lower(), cell, nodes.copy(), segs.copy(),
                burg, radius, center, segs_per_loop)
            nodes, segs = new_nodes, new_segs
            radii_m.append(radius_m)
            centers.append(center_m)
            accumulated_length_m += circumference_m
            loop_count += 1
            attempt = 0
        except Exception:
            attempt += 1

    return ExaDisNet(cell, nodes, segs)


# ============================================================
# 模拟主函数
# ============================================================
def run_simulation(net, output_dir, restart_id=None, precipitates=None):

    state = {
        "crystal": 'fcc',
        "burgmag": 0.2556e-9,
        "mu":      48e9,
        "nu":      0.324,
        "a":       2.0,
        "maxseg":  200,
        "minseg":  50,
        "rtol":    0.5,
        "rann":    1.0,
        "nextdt":  1e-9,
        "maxdt":   1e-6,
    }

    restart = None
    if restart_id is None:
        net_manager = DisNetManager(net)
    else:
        restart_filename = f'restart.{restart_id}.exadis'
        net_manager, restart = read_restart(
            state=state,
            restart_file=os.path.join(output_dir, restart_filename)
        )

    # 加载球形杂质到 C++ System，由 CollisionOrowan 在每步自动执行 Orowan 约束
    # 直接传 Burgers 单位的值，无需依赖 system 内部的 burgmag
    if precipitates is not None and len(precipitates.centers) > 0:
        exadis_net = net_manager.get_disnet(ExaDisNet)
        centers_b = [list(c) for c in precipitates.centers]
        radii_b   = list(precipitates.radii)
        exadis_net.load_obstacles(centers_b, radii_b)

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net_manager.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=10000.0, Mscrew=1000.0,
                            Mclimb=1.0, vmax=20000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 50.0, 300.0, 800.0],
                                state=state, force=calforce, mobility=mobility)
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

    burgmag           = 0.2556e-9
    Lbox_m            = 5e-6
    target_density    = float(os.environ.get('RHO_TARGET', '1e12'))
    sphere_diameter_m = 200e-9
    sphere_count      = int(os.environ.get('SPHERE_COUNT', '100'))
    output_dir        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_Cu_fcc')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    precipitates = SphericalPrecipitates(Lbox_m, burgmag, seed=12345)
    precipitates.generate(large_count=sphere_count, large_diameter_m=sphere_diameter_m)
    precipitates.save_data_file(os.path.join(output_dir, 'precipitates.data'))

    if args.restart is not None:
        run_simulation(ExaDisNet(), output_dir=output_dir,
                       restart_id=args.restart,
                       precipitates=precipitates)
    else:
        G = generate_prismatic_config_by_density(
            crystal='fcc', Lbox_m=Lbox_m, burgmag=burgmag,
            target_density=target_density,
            radius_min_m=0.15e-6, radius_max_m=0.5e-6,
            seed=12345, precipitates=precipitates
        )
        run_simulation(G, output_dir=output_dir, precipitates=precipitates)

    pyexadis.finalize()


if __name__ == "__main__":
    main()