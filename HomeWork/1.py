#!/usr/bin/env python3
"""
独立的球形杂质生成脚本
用法: python 1.py --output-dir output_Cu_fcc --count 100
"""

import os
import sys
import numpy as np
import argparse


class SphericalPrecipitates:
    """球形杂质管理类"""
    
    def __init__(self, Lbox_m, burgmag, seed=42):
        self.Lbox_m = Lbox_m
        self.burgmag = burgmag
        self.Lbox_b = Lbox_m / burgmag
        self.centers = []
        self.radii = []
        self.types = []
        self.centers_m = []
        self.radii_m = []
        self.rng = np.random.RandomState(seed)
        print(f"初始化: Lbox={Lbox_m*1e6:.2f}μm, burgmag={burgmag*1e9:.4f}nm")

    def generate(self, count, diameter_m):
        """生成球形杂质分布（与test_Cu.py保持一致）"""
        print("\n" + "="*70)
        print("开始生成球形杂质分布")
        print("="*70)
        print(f"球形杂质: {count}个, 直径={diameter_m*1e9:.1f}nm")

        radius_m = diameter_m / 2.0
        radius_b = radius_m / self.burgmag

        max_attempts = 800
        placed = 0

        for idx in range(count):
            margin_m = radius_m * 1.5
            for attempt in range(max_attempts):
                cx = self.rng.uniform(margin_m, self.Lbox_m - margin_m)
                cy = self.rng.uniform(margin_m, self.Lbox_m - margin_m)
                cz = self.rng.uniform(margin_m, self.Lbox_m - margin_m)
                center_m = np.array([cx, cy, cz])

                overlap = False
                for i in range(len(self.centers_m)):
                    dist = np.linalg.norm(center_m - self.centers_m[i])
                    min_dist = (radius_m + self.radii_m[i]) * 1.4
                    if dist < min_dist:
                        overlap = True
                        break

                if not overlap:
                    center_b = center_m / self.burgmag
                    self.centers.append(center_b)
                    self.radii.append(radius_b)
                    self.types.append('large')
                    self.centers_m.append(center_m)
                    self.radii_m.append(radius_m)
                    placed += 1
                    if placed % 20 == 0:
                        print(f"  进度: {placed}/{count} ({100*placed/count:.1f}%)")
                    break

        self.centers = np.array(self.centers) if self.centers else np.empty((0, 3))
        self.radii = np.array(self.radii) if self.radii else np.empty(0)
        self.centers_m = np.array(self.centers_m) if len(self.centers_m) > 0 else np.empty((0, 3))
        self.radii_m = np.array(self.radii_m) if len(self.radii_m) > 0 else np.empty(0)

        print(f"\n✅ 成功放置: {placed}/{count} 个球形杂质")
        print(f"   centers数组形状: {self.centers.shape}")
        print(f"   radii数组形状: {self.radii.shape}")
        print("="*70)
        return self

    def _calculate_volume_fraction(self):
        """计算杂质体积分数"""
        if len(self.radii_m) == 0:
            return 0.0
        total_volume = self.Lbox_m ** 3
        precipitate_volume = np.sum((4/3) * np.pi * self.radii_m**3)
        return precipitate_volume / total_volume

    def save_data_file(self, filename):
        """保存为data格式文件"""
        print(f"\n正在保存球形杂质data文件...")
        print(f"   目标文件: {filename}")

        try:
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"   创建目录: {output_dir}")

            with open(filename, 'w') as f:
                f.write("# Spherical Precipitates Data File\n")
                f.write("# Generated for Cu FCC Dislocation Dynamics Simulation\n")
                f.write("#" + "="*78 + "\n")
                f.write("\n")
                f.write("# SIMULATION PARAMETERS\n")
                f.write(f"# Burgers vector magnitude: {self.burgmag:.6e} m ({self.burgmag*1e9:.4f} nm)\n")
                f.write(f"# Box size: {self.Lbox_m:.6e} m ({self.Lbox_m*1e6:.2f} μm)\n")
                f.write(f"# Box size in Burgers: {self.Lbox_b:.2f} b\n")
                f.write(f"# Total precipitates: {len(self.centers)}\n")
                f.write(f"# Volume fraction: {self._calculate_volume_fraction():.6f}\n")
                f.write("\n")

                if len(self.radii_m) > 0:
                    f.write("# STATISTICS\n")
                    f.write(f"# Diameter: {self.radii_m[0]*2*1e9:.2f} nm (all spheres)\n")
                    f.write(f"# Total volume: {np.sum((4/3)*np.pi*self.radii_m**3):.6e} m^3\n")
                    f.write("\n")

                f.write("#" + "="*78 + "\n")
                f.write("# DATA FORMAT\n")
                f.write("#" + "="*78 + "\n")
                f.write("# All coordinates and radii are in Burgers vector units (b)\n")
                f.write("#\n")
                f.write("# Column format:\n")
                f.write("#   1. ID          - Precipitate ID (integer)\n")
                f.write("#   2. X(b)        - X coordinate in Burgers units\n")
                f.write("#   3. Y(b)        - Y coordinate in Burgers units\n")
                f.write("#   4. Z(b)        - Z coordinate in Burgers units\n")
                f.write("#   5. Radius(b)   - Radius in Burgers units\n")
                f.write("#   6. Diameter(nm)- Diameter in nanometers\n")
                f.write("#" + "="*78 + "\n")
                f.write("\n")
                f.write(f"# {'ID':>6} {'X(b)':>16} {'Y(b)':>16} {'Z(b)':>16} "
                        f"{'Radius(b)':>16} {'Diameter(nm)':>12}\n")
                f.write("#" + "-"*78 + "\n")

                for i in range(len(self.centers)):
                    center = self.centers[i]
                    radius = self.radii[i]
                    diameter_nm = self.radii_m[i] * 2 * 1e9
                    f.write(f"{i+1:8d} {center[0]:16.8e} {center[1]:16.8e} {center[2]:16.8e} "
                            f"{radius:16.8e} {diameter_nm:12.2f}\n")

                f.write("#" + "="*78 + "\n")
                f.write(f"# END OF DATA ({len(self.centers)} precipitates)\n")
                f.write("#" + "="*78 + "\n")

            if not os.path.exists(filename):
                print(f"❌ 文件写入后不存在: {filename}")
                return False

            file_size = os.path.getsize(filename)
            print(f"\n✅ 球形杂质data文件已成功保存！")
            print(f"   文件路径: {filename}")
            print(f"   文件大小: {file_size} bytes ({file_size/1024:.2f} KB)")
            print(f"   包含杂质: {len(self.centers)} 个")

            print(f"\n文件内容预览:")
            with open(filename, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    print(f"   {line.rstrip()}")

            return True

        except Exception as e:
            print(f"\n❌ 保存失败！")
            print(f"   错误信息: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_vtk_file(self, filename):
        """保存为VTK格式（用于ParaView可视化）"""
        print(f"\n正在保存VTK文件...")

        try:
            with open(filename, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("Spherical Precipitates\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")
                f.write(f"POINTS {len(self.centers)} float\n")

                for center in self.centers:
                    f.write(f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n")

                f.write(f"\nPOINT_DATA {len(self.centers)}\n")
                f.write("SCALARS Radius float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for radius in self.radii:
                    f.write(f"{radius:.6f}\n")

                f.write("\nSCALARS Diameter_nm float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for radius_m in self.radii_m:
                    f.write(f"{radius_m*2*1e9:.2f}\n")

            print(f"✅ VTK文件已保存: {filename}")
            return True

        except Exception as e:
            print(f"❌ VTK保存失败: {e}")
            return False


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description='生成球形杂质data文件（独立脚本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python 1.py --output-dir output_Cu_fcc --count 100 --diameter 200 --seed 12345
        """
    )

    parser.add_argument('--output-dir', type=str, default='output_Cu_fcc',
                        help='输出目录 (默认: output_Cu_fcc)')
    parser.add_argument('--count', type=int, default=100,
                        help='球形杂质数量 (默认: 100)')
    parser.add_argument('--diameter', type=float, default=200,
                        help='球形杂质直径 (nm) (默认: 200)')
    parser.add_argument('--box-size', type=float, default=5.0,
                        help='盒子尺寸 (μm) (默认: 5.0)')
    parser.add_argument('--burgmag', type=float, default=0.2556,
                        help='柏氏矢量大小 (nm) (默认: 0.2556 for Cu)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='随机数种子 (默认: 12345)')
    parser.add_argument('--filename', type=str, default='precipitates.data',
                        help='输出文件名 (默认: precipitates.data)')
    parser.add_argument('--vtk', action='store_true',
                        help='同时生成VTK格式文件')

    args = parser.parse_args()

    # 单位转换
    Lbox_m = args.box_size * 1e-6   # μm -> m
    burgmag = args.burgmag * 1e-9   # nm -> m
    diameter_m = args.diameter * 1e-9  # nm -> m

    print("\n" + "="*70)
    print("球形杂质生成脚本")
    print("="*70)
    print(f"输出目录: {args.output_dir}")
    print(f"输出文件: {args.filename}")
    print(f"盒子尺寸: {args.box_size} μm ({Lbox_m/burgmag:.0f} b)")
    print(f"柏氏矢量: {args.burgmag} nm (Cu FCC)")
    print(f"球形杂质: {args.count}个, 直径={args.diameter}nm")
    print(f"随机种子: {args.seed}")
    print(f"生成VTK: {'是' if args.vtk else '否'}")
    print("="*70 + "\n")

    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print(f"✅ 创建输出目录: {args.output_dir}\n")

        precipitates = SphericalPrecipitates(Lbox_m, burgmag, seed=args.seed)
        precipitates.generate(count=args.count, diameter_m=diameter_m)

        output_file = os.path.join(args.output_dir, args.filename)
        success = precipitates.save_data_file(output_file)

        if not success:
            print("\n❌ data文件生成失败！")
            sys.exit(1)

        if args.vtk:
            vtk_file = os.path.join(args.output_dir, args.filename.replace('.data', '.vtk'))
            precipitates.save_vtk_file(vtk_file)

        print("\n" + "="*70)
        print("✅ 任务完成！")
        print("="*70)
        print(f"\n生成的文件:")
        print(f"  1. Data文件: {output_file}")
        if args.vtk:
            print(f"  2. VTK文件: {vtk_file}")
        print(f"\n杂质信息:")
        print(f"  数量: {len(precipitates.centers)} 个")
        print(f"  直径: {args.diameter} nm")
        print(f"  体积分数: {precipitates._calculate_volume_fraction():.6f}")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ 程序失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()