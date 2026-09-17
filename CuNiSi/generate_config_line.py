"""生成初始位错配置（无限直线版）+ 球形杂质 + 孪晶面 → init_data_line/

用 insert_infinite_line 替代 FR 源：
  - 全部节点 UNCONSTRAINED（无钉扎端点）→ 初始即可动 → 无"位错饥饿→屈服骤降"
  - 经 PBC 闭合的周期线 → Burgers 处处守恒（不再有悬空端点）

注意/局限：
  - 无限直线沿 PBC 贯穿盒子；origin 放在活动区 z∈[0.2,0.8]，但倾斜面上的线
    会沿 z 延展，可能触及孪晶面——交给模拟里的 twin-wall 处理（与 FR 版同理）
  - 密度按整盒子体积算（与之前一致）
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet
    from pyexadis_utils import insert_infinite_line
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e


# ============================================================
# 参数
# ============================================================
BURGMAG  = 0.2556e-9
LBOX_M   = 5e-6
RHO_TARGET   = float(os.environ.get('RHO_TARGET', '1e12'))
SPHERE_COUNT = int(os.environ.get('SPHERE_COUNT', '50'))
TWIN_Z_FRACS = [0.2, 0.8]
MAXSEG_B = 200          # 初始离散段长（b），与模拟 maxseg 一致


# ============================================================
# 坐标系旋转矩阵：[100]/[010]/[001] → [1-10]/[11-2]/[111]
# ============================================================
_R = np.array([
    [ 1/np.sqrt(2), -1/np.sqrt(2),  0           ],
    [ 1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
    [ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],
])

# ============================================================
# FCC 滑移系：活动系（高Schmid）+ 森林系（零Schmid）
# ============================================================
FCC_SLIP_SYSTEMS = []
FCC_FOREST_SYSTEMS = []
_fcc_planes = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]], dtype=float)
_fcc_burgers = np.array([[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,0],[-1,1,0]], dtype=float) / np.sqrt(2.0)
_loading_dir = np.array([0,0,1], dtype=float)
for _p in _fcc_planes:
    _pn = _p / np.linalg.norm(_p)
    for _b in _fcc_burgers:
        if abs(np.dot(_b, _pn)) < 1e-5:
            _b_hat = _b / np.linalg.norm(_b)
            _schmid = abs(np.dot(_loading_dir, _b_hat) * np.dot(_loading_dir, _pn))
            _b_rot = _R @ _b
            _pn_rot = _R @ _pn
            _pn_rot /= np.linalg.norm(_pn_rot)
            if _schmid < 1e-3:
                FCC_FOREST_SYSTEMS.append((_b_rot.copy(), _pn_rot.copy()))
            else:
                FCC_SLIP_SYSTEMS.append((_b_rot.copy(), _pn_rot.copy()))


# ============================================================
# 球形杂质
# ============================================================
def generate_precipitates(Lbox_m, burgmag, count, diameter_m, seed=12345, z_range_m=None):
    rng = np.random.RandomState(seed)
    radius_m = diameter_m / 2.0
    radius_b = radius_m / burgmag
    margin_m = radius_m * 1.5
    centers_b, radii_b, centers_m, radii_m = [], [], [], []
    for _ in range(count):
        for _ in range(800):
            cx = rng.uniform(margin_m, Lbox_m - margin_m)
            cy = rng.uniform(margin_m, Lbox_m - margin_m)
            if z_range_m is not None:
                cz = rng.uniform(z_range_m[0] + margin_m, z_range_m[1] - margin_m)
            else:
                cz = rng.uniform(margin_m, Lbox_m - margin_m)
            c_m = np.array([cx, cy, cz])
            if not any(np.linalg.norm(c_m - centers_m[i]) < (radius_m + radii_m[i]) * 1.4
                       for i in range(len(centers_m))):
                centers_b.append(c_m / burgmag); radii_b.append(radius_b)
                centers_m.append(c_m); radii_m.append(radius_m)
                break
    return (np.array(centers_b) if centers_b else np.empty((0, 3)),
            np.array(radii_b)  if radii_b  else np.empty(0),
            np.array(centers_m) if centers_m else np.empty((0, 3)),
            np.array(radii_m)  if radii_m  else np.empty(0))


# ============================================================
# 孪晶面
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
# 无限直线位错网络
# ============================================================
def generate_line_network(Lbox_m, burgmag, target_density, seed=12345,
                          precip_centers_m=None, precip_radii_m=None, z_range_m=None):
    Lbox = int(round(Lbox_m / burgmag))
    total_length = target_density * Lbox_m**3   # 按整盒子体积
    rng  = np.random.RandomState(seed)
    cell = pyexadis.Cell(Lbox)
    nodes, segs = [], []
    accumulated = 0.0
    origins_m, slip_ids = [], []
    nsys = len(FCC_SLIP_SYSTEMS)

    def try_place_line(burg, plane, slip_id):
        """随机放一条无限直线（origin 在活动区、避杂质、避同系近重合）。
        成功返回该线长度(m)，失败返回 0。"""
        nonlocal nodes, segs
        margin = 0.1e-6
        for _ in range(300):
            ox = rng.uniform(margin, Lbox_m - margin)
            oy = rng.uniform(margin, Lbox_m - margin)
            if z_range_m is not None:
                oz = rng.uniform(z_range_m[0] + margin, z_range_m[1] - margin)
            else:
                oz = rng.uniform(margin, Lbox_m - margin)
            o_m = np.array([ox, oy, oz])

            # 避杂质
            if precip_centers_m is not None and len(precip_centers_m) > 0:
                if any(np.linalg.norm(o_m - precip_centers_m[i]) < precip_radii_m[i] * 1.5
                       for i in range(len(precip_centers_m))):
                    continue
            # 避同滑移系近重合（防止偶极立即湮灭）
            if any(slip_ids[j] == slip_id and np.linalg.norm(o_m - origins_m[j]) < 0.3e-6
                   for j in range(len(origins_m))):
                continue

            theta = rng.uniform(0.0, 360.0)
            o_b = o_m / burgmag
            L = insert_infinite_line(cell, nodes, segs, burg, plane, o_b,
                                     theta=theta, maxseg=MAXSEG_B, trial=True)
            if not (isinstance(L, (int, float)) and L > 0):
                continue   # 太长/无效，换位置
            insert_infinite_line(cell, nodes, segs, burg, plane, o_b,
                                 theta=theta, maxseg=MAXSEG_B)
            origins_m.append(o_m); slip_ids.append(slip_id)
            return L * burgmag   # b → m
        return 0.0

    # ---- 活动系：填充密度 ----
    count, attempt = 0, 0
    while accumulated < total_length * 0.9 and attempt < 5000:
        slip_id = count % nsys
        burg, plane = FCC_SLIP_SYSTEMS[slip_id]
        Lm = try_place_line(burg, plane, slip_id)
        if Lm > 0:
            accumulated += Lm; count += 1; attempt = 0
        else:
            attempt += 1
    print(f"  Active lines: {count}, length = {accumulated:.3e} m (target {total_length:.3e})")

    # ---- 森林系：每系 1-2 条 ----
    nf = 0
    for fi, (burg, plane) in enumerate(FCC_FOREST_SYSTEMS):
        for _ in range(rng.randint(1, 3)):
            if try_place_line(burg, plane, nsys + fi) > 0:
                nf += 1
    print(f"  Forest lines: {nf} (on {len(FCC_FOREST_SYSTEMS)} zero-Schmid systems)")
    print(f"Generated {count + nf} infinite lines total")
    return ExaDisNet(cell, nodes, segs)


# ============================================================
# 主程序：生成 twin+precip 配置到 init_data_line/
# ============================================================
def main():
    pyexadis.initialize()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'init_data_line')
    os.makedirs(output_dir, exist_ok=True)

    Lbox_b = LBOX_M / BURGMAG
    z_min_m = LBOX_M * min(TWIN_Z_FRACS)
    z_max_m = LBOX_M * max(TWIN_Z_FRACS)
    z_range_m = (z_min_m, z_max_m)
    print(f"Active zone: z = [{z_min_m*1e6:.1f}, {z_max_m*1e6:.1f}] um")

    # 杂质
    centers_b, radii_b, centers_m, radii_m = generate_precipitates(
        LBOX_M, BURGMAG, count=SPHERE_COUNT, diameter_m=200e-9, z_range_m=z_range_m)
    print(f"Generated {len(centers_b)} precipitates")
    with open(os.path.join(output_dir, 'obstacles.data'), 'w') as f:
        f.write(f"# Spherical obstacles ({len(centers_b)} total)\n")
        f.write("# center_x(b)  center_y(b)  center_z(b)  radius(b)\n")
        for i in range(len(centers_b)):
            c = centers_b[i]
            f.write(f"{c[0]:.8e}  {c[1]:.8e}  {c[2]:.8e}  {radii_b[i]:.8e}\n")

    # 孪晶面
    twin_points_b, twin_normals_b = generate_twin_planes(Lbox_b, TWIN_Z_FRACS)
    print(f"Generated {len(twin_points_b)} twin planes")
    with open(os.path.join(output_dir, 'twin_planes.data'), 'w') as f:
        f.write(f"# Twin boundary planes ({len(twin_points_b)} total)\n")
        f.write("# point_x(b)  point_y(b)  point_z(b)  normal_x  normal_y  normal_z\n")
        for i in range(len(twin_points_b)):
            p, n = twin_points_b[i], twin_normals_b[i]
            f.write(f"{p[0]:.8e}  {p[1]:.8e}  {p[2]:.8e}  {n[0]:.8f}  {n[1]:.8f}  {n[2]:.8f}\n")

    # 无限直线位错网络
    G = generate_line_network(
        LBOX_M, BURGMAG, RHO_TARGET, seed=12345,
        precip_centers_m=centers_m, precip_radii_m=radii_m, z_range_m=z_range_m)
    G.write_data(os.path.join(output_dir, 'init_config.data'))
    print(f"Saved: {output_dir}/")

    pyexadis.finalize()


if __name__ == "__main__":
    main()
