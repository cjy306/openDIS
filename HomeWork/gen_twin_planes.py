#!/usr/bin/env python3
"""
独立的孪晶面 data 文件生成脚本
用法: python gen_twin_planes.py --output-dir output_Cu_twin --fractions 0.5
      python gen_twin_planes.py --output-dir output_Cu_twin --fractions 0.33 0.67
"""
import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='生成孪晶面 data 文件')
    parser.add_argument('--output-dir', type=str, default='output_Cu_twin',
                        help='输出目录 (默认: output_Cu_twin)')
    parser.add_argument('--filename', type=str, default='twin_planes.data',
                        help='输出文件名 (默认: twin_planes.data)')
    parser.add_argument('--fractions', type=float, nargs='+', default=[0.5],
                        help='孪晶面在 z 方向的分数位置 (默认: 0.5)')
    parser.add_argument('--normal', type=float, nargs=3, default=[0, 0, 1],
                        help='孪晶面法向量 (默认: 0 0 1)')
    parser.add_argument('--box-size', type=float, default=5.0,
                        help='盒子尺寸 (μm) (默认: 5.0)')
    parser.add_argument('--burgmag', type=float, default=0.2556,
                        help='柏氏矢量大小 (nm) (默认: 0.2556 for Cu)')
    args = parser.parse_args()

    burgmag = args.burgmag * 1e-9
    Lbox_m  = args.box_size * 1e-6
    Lbox_b  = Lbox_m / burgmag
    normal  = np.array(args.normal)
    normal  = normal / np.linalg.norm(normal)

    os.makedirs(args.output_dir, exist_ok=True)
    filepath = os.path.join(args.output_dir, args.filename)

    points_b  = []
    normals_b = []
    for frac in args.fractions:
        z = Lbox_b * frac
        points_b.append([Lbox_b / 2.0, Lbox_b / 2.0, z])
        normals_b.append(normal.tolist())

    with open(filepath, 'w') as f:
        f.write("# Twin Boundary Planes Data File\n")
        f.write("#" + "=" * 68 + "\n")
        f.write(f"# Burgers vector magnitude: {burgmag:.6e} m\n")
        f.write(f"# Box size: {Lbox_b:.2f} b ({args.box_size} um)\n")
        f.write(f"# Total planes: {len(points_b)}\n")
        f.write("#" + "=" * 68 + "\n")
        f.write("# All coordinates in Burgers vector units (b)\n")
        f.write("#\n")
        f.write("# Column format:\n")
        f.write("#   1. ID\n")
        f.write("#   2. Point_X(b)\n")
        f.write("#   3. Point_Y(b)\n")
        f.write("#   4. Point_Z(b)\n")
        f.write("#   5. Normal_X\n")
        f.write("#   6. Normal_Y\n")
        f.write("#   7. Normal_Z\n")
        f.write("#" + "=" * 68 + "\n")
        for i in range(len(points_b)):
            p = points_b[i]
            n = normals_b[i]
            f.write(f"{i+1:6d} {p[0]:16.8e} {p[1]:16.8e} {p[2]:16.8e} "
                    f"{n[0]:10.6f} {n[1]:10.6f} {n[2]:10.6f}\n")
        f.write("#" + "=" * 68 + "\n")
        f.write(f"# END OF DATA ({len(points_b)} planes)\n")

    print(f"Twin planes data saved: {filepath}")
    print(f"  Planes: {len(points_b)}")
    for i, frac in enumerate(args.fractions):
        print(f"  #{i+1}: z = {Lbox_b * frac:.1f} b (fraction = {frac}), "
              f"normal = [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")


if __name__ == "__main__":
    main()
