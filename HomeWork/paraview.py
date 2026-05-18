"""
ParaDiS data → VTK 转换脚本
用法:
  python paraview.py --sim output_Cu_twin_no_rorient --init init_data_twin_no_rorient --out vtk_twin
  python paraview.py --sim output_Cu_fcc  --init init_data_precip --out vtk_precip
  python paraview.py --sim output_Cu_fcc_pure --out vtk_pure
# 只转换第 3700 到 5000 步
python paraview.py --sim output_Cu_twin_no_rorient --init init_data_twin_no_rorient --out vtk_twin --start 3700 --end 5000

# 只指定起始步
python paraview.py --sim output_Cu_twin_no_rorient --init init_data_twin_no_rorient --out vtk_twin --start 3700

# 只指定结束步
python paraview.py --sim output_Cu_twin_no_rorient --init init_data_twin_no_rorient --out vtk_twin --end 5000
"""
import os, sys, glob, time
import numpy as np

sys.path.append('/data/home/dg000246b/openDIS/core/exadis/python/')
pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_utils import read_paradis, write_vtk
from pyexadis_base import ExaDisNet

BURGMAG = 0.2556e-9
LBOX_M  = 5e-6


# ============================================================
# 球形杂质
# ============================================================
class SphericalPrecipitates:
    def __init__(self, Lbox_b, burgmag):
        self.Lbox_b = Lbox_b
        self.burgmag = burgmag
        self.centers = np.empty((0, 3))
        self.radii = np.empty(0)

    def load(self, filename):
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.centers = data[:, :3]
        self.radii = data[:, 3]
        print(f"  Loaded {len(self.centers)} precipitates from {os.path.basename(filename)}")
        return self

    def is_inside_any_sphere(self, points):
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        inside = np.zeros(points.shape[0], dtype=bool)
        for i in range(len(self.centers)):
            dist = np.linalg.norm(points - self.centers[i], axis=1)
            inside |= (dist <= self.radii[i])
        return inside

    def export_vtk(self, filename, resolution=20):
        if len(self.centers) == 0:
            return
        all_points = []
        all_cells = []
        point_offset = 0

        for i in range(len(self.centers)):
            center = self.centers[i]
            radius = self.radii[i]
            theta = np.linspace(0, np.pi, resolution)
            phi = np.linspace(0, 2 * np.pi, resolution)

            pts = []
            for t in theta:
                for p in phi:
                    pts.append([
                        center[0] + radius * np.sin(t) * np.cos(p),
                        center[1] + radius * np.sin(t) * np.sin(p),
                        center[2] + radius * np.cos(t),
                    ])
            all_points.extend(pts)

            for j in range(resolution - 1):
                for k in range(resolution - 1):
                    p1 = point_offset + j * resolution + k
                    p2 = point_offset + j * resolution + (k + 1)
                    p3 = point_offset + (j + 1) * resolution + (k + 1)
                    p4 = point_offset + (j + 1) * resolution + k
                    all_cells.append([4, p1, p2, p3, p4])
            point_offset += len(pts)

        all_points = np.array(all_points)
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\nSpherical precipitates\nASCII\nDATASET POLYDATA\n")
            f.write(f"\nPOINTS {len(all_points)} float\n")
            for pt in all_points:
                f.write(f"{pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")
            total_size = sum(len(c) for c in all_cells)
            f.write(f"\nPOLYGONS {len(all_cells)} {total_size}\n")
            for c in all_cells:
                f.write(" ".join(map(str, c)) + "\n")
        print(f"  Precipitates VTK saved: {os.path.basename(filename)}")


# ============================================================
# 孪晶面
# ============================================================
def export_twin_planes_vtk(filename, Lbox_b, points_b, normals_b):
    nplanes = len(points_b)
    if nplanes == 0:
        return
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\nTwin planes\nASCII\nDATASET POLYDATA\n")
        f.write(f"\nPOINTS {nplanes * 4} float\n")
        for i in range(nplanes):
            p = np.array(points_b[i])
            n = np.array(normals_b[i])
            n = n / np.linalg.norm(n)
            t1 = np.cross(n, [1, 0, 0]) if abs(n[2]) > 0.9 else np.cross(n, [0, 0, 1])
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            half = Lbox_b * 0.75
            for s1, s2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                c = p + s1 * half * t1 + s2 * half * t2
                f.write(f"{c[0]:.6e} {c[1]:.6e} {c[2]:.6e}\n")
        f.write(f"\nPOLYGONS {nplanes} {nplanes * 5}\n")
        for i in range(nplanes):
            b = i * 4
            f.write(f"4 {b} {b+1} {b+2} {b+3}\n")
        f.write(f"\nCELL_DATA {nplanes}\nSCALARS PlaneID int\nLOOKUP_TABLE default\n")
        for i in range(nplanes):
            f.write(f"{i}\n")
    print(f"  Twin planes VTK saved: {os.path.basename(filename)} ({nplanes} planes)")


# ============================================================
# 主转换
# ============================================================
def wrap_vtk_pbc(vtk_file, Lbox):
    """Fold VTK segment endpoint coordinates back into [0, Lbox).
    write_vtk uses closest_image() for segment continuity, which can place
    endpoints outside the primary cell.  This function wraps those coordinates
    so ParaView shows everything inside the simulation box."""
    with open(vtk_file, 'r') as f:
        lines = f.readlines()

    in_points = False
    total_points = 0
    points_written = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith('POINTS'):
            in_points = True
            total_points = int(line.strip().split()[1])
            points_written = 0
            continue
        if not in_points:
            continue
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        if points_written >= 8:
            x = float(parts[0]) % Lbox
            y = float(parts[1]) % Lbox
            z = float(parts[2]) % Lbox
            lines[idx] = f"{x:.8e} {y:.8e} {z:.8e}\n"
        points_written += 1
        if points_written >= total_points:
            break

    with open(vtk_file, 'w') as f:
        f.writelines(lines)


def convert(sim_dir, out_dir, init_dir=None, start=None, end=None):
    os.makedirs(out_dir, exist_ok=True)
    Lbox_b = LBOX_M / BURGMAG

    # 加载杂质
    precipitates = None
    if init_dir:
        obs_file = os.path.join(init_dir, 'obstacles.data')
        if os.path.exists(obs_file):
            precipitates = SphericalPrecipitates(Lbox_b, BURGMAG).load(obs_file)
            precipitates.export_vtk(os.path.join(out_dir, 'precipitates.vtk'))

    # 加载孪晶面
    if init_dir:
        tp_file = os.path.join(init_dir, 'twin_planes.data')
        if os.path.exists(tp_file):
            tp_data = np.loadtxt(tp_file)
            if tp_data.ndim == 1:
                tp_data = tp_data.reshape(1, -1)
            export_twin_planes_vtk(
                os.path.join(out_dir, 'twin_planes.vtk'),
                Lbox_b, tp_data[:, :3].tolist(), tp_data[:, 3:].tolist())

    # 转换 config.*.data 文件
    all_data = sorted(glob.glob(os.path.join(sim_dir, '*.data')))
    data_files = [f for f in all_data
                  if os.path.basename(f) not in ('obstacles.data', 'twin_planes.data')]

    # Filter by step range
    if start is not None or end is not None:
        import re
        filtered = []
        for f in data_files:
            m = re.search(r'(\d+)', os.path.basename(f))
            if m:
                step = int(m.group(1))
                if start is not None and step < start:
                    continue
                if end is not None and step > end:
                    continue
            filtered.append(f)
        data_files = filtered

    if not data_files:
        print(f"No .data files found in {sim_dir}")
        return

    print(f"Converting {len(data_files)} files...")
    pyexadis.initialize()

    for idx, data_file in enumerate(data_files):
        basename = os.path.basename(data_file)
        name = basename.replace('.data', '')
        print(f"  [{idx+1}/{len(data_files)}] {basename}")
        try:
            net = read_paradis(data_file)
            vtk_file = os.path.join(out_dir, f'{name}.vtk')
            write_vtk(net, vtk_file, precipitates=precipitates, verbose=False)
        except Exception as e:
            print(f"    Failed: {e}")

    pyexadis.finalize()
    print(f"Done. Output: {out_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ParaDiS data to VTK converter')
    parser.add_argument('--sim', required=True, help='Simulation output directory (e.g. output_Cu_twin)')
    parser.add_argument('--init', default=None, help='Init data directory (e.g. init_data_twin)')
    parser.add_argument('--out', required=True, help='VTK output directory (e.g. vtk_twin)')
    parser.add_argument('--start', type=int, default=None, help='Start step number (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End step number (inclusive)')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    sim_dir  = os.path.join(base, args.sim)  if not os.path.isabs(args.sim)  else args.sim
    out_dir  = os.path.join(base, args.out)  if not os.path.isabs(args.out)  else args.out
    init_dir = os.path.join(base, args.init) if args.init and not os.path.isabs(args.init) else args.init

    convert(sim_dir, out_dir, init_dir, start=args.start, end=args.end)
