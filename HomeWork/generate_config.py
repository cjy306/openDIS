"""
生成初始位错配置（Frank-Read 源）+ 球形杂质 + 孪晶面
按 mode 生成 3 份独立配置：
  python generate_config.py --mode pure    → init_data_pure/
  python generate_config.py --mode precip  → init_data_precip/
  python generate_config.py --mode twin    → init_data_twin/
  python generate_config.py --mode all     → 依次生成全部三份
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, NodeConstraints
    from pyexadis_utils import insert_frank_read_src
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e


# ============================================================
# 参数
# ============================================================
BURGMAG  = 0.2556e-9
LBOX_M   = 5e-6


# ============================================================
# FCC 滑移系（12个：4个{111}面 × 3个<110>方向）
# ============================================================
FCC_SLIP_SYSTEMS = []
_fcc_planes = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]], dtype=float)
_fcc_burgers = np.array([[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,0],[-1,1,0]], dtype=float) / np.sqrt(2.0)
for _p in _fcc_planes:
    _pn = _p / np.linalg.norm(_p)
    for _b in _fcc_burgers:
        if abs(np.dot(_b, _pn)) < 1e-5:
            FCC_SLIP_SYSTEMS.append((_b.copy(), _pn.copy()))


# ============================================================
# 生成球形杂质
# ============================================================
def generate_precipitates(Lbox_m, burgmag, count, diameter_m, seed=12345,
                          z_range_m=None):
    """z_range_m: (z_min, z_max) 米制，限制杂质 z 坐标范围。None 则不限制。"""
    rng = np.random.RandomState(seed)
    radius_m = diameter_m / 2.0
    radius_b = radius_m / burgmag
    margin_m = radius_m * 1.5

    centers_b, radii_b = [], []
    centers_m, radii_m = [], []

    for _ in range(count):
        for _ in range(800):
            cx = rng.uniform(margin_m, Lbox_m - margin_m)
            cy = rng.uniform(margin_m, Lbox_m - margin_m)
            if z_range_m is not None:
                cz = rng.uniform(z_range_m[0] + margin_m, z_range_m[1] - margin_m)
            else:
                cz = rng.uniform(margin_m, Lbox_m - margin_m)
            c_m = np.array([cx, cy, cz])
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
# 生成孪晶面
# ============================================================
def generate_twin_planes(Lbox_b, twin_z_fractions):
    points_b, normals_b = [], []
    for frac in twin_z_fractions:
        z = Lbox_b * frac
        points_b.append([Lbox_b/2.0, Lbox_b/2.0, z])
        normals_b.append([0.0, 0.0, 1.0])
        print(f"  Twin plane at z = {z:.1f} b (fraction = {frac})")
    return points_b, normals_b


# ============================================================
# 按密度生成 Frank-Read 源网络（避开杂质和孪晶面）
# ============================================================
def generate_dislocation_network(Lbox_m, burgmag, target_density, seed=12345,
                                  precip_centers_m=None, precip_radii_m=None,
                                  z_range_m=None):
    """z_range_m: (z_min, z_max) 米制，FR 源严格限制在此区间内。None 则全盒子。"""
    Lbox = int(round(Lbox_m / burgmag))
    total_length = target_density * Lbox_m**3

    rng  = np.random.RandomState(seed)
    cell = pyexadis.Cell(Lbox)
    nodes, segs = [], []
    accumulated = 0.0
    src_lengths_m, src_centers_m = [], []

    nsys = len(FCC_SLIP_SYSTEMS)

    # x,y 方向范围
    xy_margin_frac = 0.05  # 距盒子边缘的安全距离

    attempt, src_count = 0, 0
    while accumulated < total_length * 0.85 and attempt < 15000:
        length_m = rng.uniform(0.3e-6, 1.0e-6)
        length_b = length_m / burgmag
        margin = length_m * 0.8

        # x, y: 全盒子范围
        xy_low  = margin
        xy_high = Lbox_m - margin

        # z: 严格限制在两孪晶面之间
        if z_range_m is not None:
            z_low  = z_range_m[0] + margin
            z_high = z_range_m[1] - margin
        else:
            z_low  = margin
            z_high = Lbox_m - margin

        if xy_high <= xy_low or z_high <= z_low:
            attempt += 1; continue

        cx = rng.uniform(xy_low, xy_high)
        cy = rng.uniform(xy_low, xy_high)
        cz = rng.uniform(z_low, z_high)
        c_m = np.array([cx, cy, cz])

        # 检查与已有 FR 源重叠
        overlap = any(np.linalg.norm(c_m - src_centers_m[i]) < (length_m + src_lengths_m[i]) * 0.6
                      for i in range(len(src_centers_m)))

        # 检查与杂质重叠
        if not overlap and precip_centers_m is not None and len(precip_centers_m) > 0:
            overlap = any(np.linalg.norm(c_m - precip_centers_m[i]) < (length_m * 0.5 + precip_radii_m[i]) * 1.5
                          for i in range(len(precip_centers_m)))

        if overlap:
            attempt += 1; continue

        burg, plane = FCC_SLIP_SYSTEMS[src_count % nsys]
        numnodes = max(3, int(round(length_b / 100.0)))

        try:
            nodes, segs = insert_frank_read_src(
                cell, nodes, segs, burg, plane, length_b, c_m / burgmag,
                numnodes=numnodes)
            src_lengths_m.append(length_m)
            src_centers_m.append(c_m)
            accumulated += length_m
            src_count += 1
            attempt = 0
        except Exception:
            attempt += 1

    print(f"Generated {src_count} Frank-Read sources, total length = {accumulated:.3e} m")
    return ExaDisNet(cell, nodes, segs)


# ============================================================
# 生成单份配置并保存
# ============================================================
def generate_one(mode, target_density, sphere_count, twin_z_fracs):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, f'init_data_{mode}')
    os.makedirs(output_dir, exist_ok=True)

    Lbox_b = LBOX_M / BURGMAG
    use_precip = mode in ('precip', 'twin')
    use_twin   = mode == 'twin'

    # ---- 计算 z 范围（twin 模式下限制在两孪晶面之间）----
    z_range_m = None
    if use_twin and len(twin_z_fracs) >= 2:
        z_min_m = LBOX_M * min(twin_z_fracs)
        z_max_m = LBOX_M * max(twin_z_fracs)
        z_range_m = (z_min_m, z_max_m)
        print(f"  [{mode}] Active zone: z = [{z_min_m*1e6:.1f}, {z_max_m*1e6:.1f}] um")
        print(f"  [{mode}] Vacuum buffer: {z_min_m*1e6:.1f} um (bottom) + {(LBOX_M-z_max_m)*1e6:.1f} um (top)")

    # ---- 球形杂质 ----
    centers_m, radii_m = None, None
    if use_precip:
        centers_b, radii_b, centers_m, radii_m = generate_precipitates(
            LBOX_M, BURGMAG, count=sphere_count, diameter_m=200e-9,
            z_range_m=z_range_m)
        print(f"  [{mode}] Generated {len(centers_b)} precipitates")
        obs_file = os.path.join(output_dir, 'obstacles.data')
        with open(obs_file, 'w') as f:
            f.write(f"# Spherical obstacles ({len(centers_b)} total)\n")
            f.write("# center_x(b)  center_y(b)  center_z(b)  radius(b)\n")
            for i in range(len(centers_b)):
                c = centers_b[i]
                f.write(f"{c[0]:.8e}  {c[1]:.8e}  {c[2]:.8e}  {radii_b[i]:.8e}\n")

    # ---- 孪晶面 ----
    if use_twin:
        twin_points_b, twin_normals_b = generate_twin_planes(Lbox_b, twin_z_fracs)
        print(f"  [{mode}] Generated {len(twin_points_b)} twin planes")
        tp_file = os.path.join(output_dir, 'twin_planes.data')
        with open(tp_file, 'w') as f:
            f.write(f"# Twin boundary planes ({len(twin_points_b)} total)\n")
            f.write("# point_x(b)  point_y(b)  point_z(b)  normal_x  normal_y  normal_z\n")
            for i in range(len(twin_points_b)):
                p = twin_points_b[i]
                n = twin_normals_b[i]
                f.write(f"{p[0]:.8e}  {p[1]:.8e}  {p[2]:.8e}  {n[0]:.8f}  {n[1]:.8f}  {n[2]:.8f}\n")

    # ---- FR 源位错网络 ----
    G = generate_dislocation_network(
        LBOX_M, BURGMAG, target_density, seed=12345,
        precip_centers_m=centers_m, precip_radii_m=radii_m,
        z_range_m=z_range_m)

    data_file = os.path.join(output_dir, 'init_config.data')
    G.write_data(data_file)
    print(f"  [{mode}] Saved: {output_dir}/")


# ============================================================
# 主程序
# ============================================================
def main():
    pyexadis.initialize()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pure', 'precip', 'twin', 'all'], default='all',
                        help='pure=无障碍, precip=杂质, twin=杂质+孪晶面, all=全部生成')
    args = parser.parse_args()

    target_density = float(os.environ.get('RHO_TARGET', '1e12'))
    sphere_count   = int(os.environ.get('SPHERE_COUNT', '10'))
    twin_z_fracs   = [0.2, 0.8]  # 上下两个孪晶面，中间为位错活动区，外侧为真空缓冲

    modes = ['pure', 'precip', 'twin'] if args.mode == 'all' else [args.mode]

    for mode in modes:
        print(f"\n===== Generating config: {mode} =====")
        generate_one(mode, target_density, sphere_count, twin_z_fracs)

    pyexadis.finalize()
    print("\nDone.")


if __name__ == "__main__":
    main()
