import os, glob
import sys
import numpy as np
import time

# 设置OpenMP环境变量
os.environ['OMP_PROC_BIND'] = 'spread'
os.environ['OMP_PLACES'] = 'threads'

# 添加模块所在目录到查找路径
sys.path.append('/data/home/dg000246b/openDIS/core/exadis/python/')
pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_utils import read_paradis, write_vtk
from pyexadis_base import ExaDisNet
# ========== 用户可修改的配置 ==========
INPUT_PATH = '/data/home/dg000246b/openDIS/HomeWork/output_Cu_fcc'
OUTPUT_PATH = '/data/home/dg000246b/openDIS/HomeWork/output_CU_vtk'
# 孪晶面 z 方向分数位置（空列表 [] 表示无孪晶面，仅在没有 twin_planes.data 时使用）
TWIN_Z_FRACTIONS = []

class SphericalPrecipitates:
    """球形杂质管理类"""
    
    def __init__(self, Lbox_m, burgmag):
        self.Lbox_m = Lbox_m
        self.burgmag = burgmag
        self.Lbox_b = Lbox_m / burgmag
        self.centers = np.empty((0, 3))
        self.radii = np.empty(0)
        self.types = []
        
    def load_from_file(self, filename):
        """从文件加载杂质信息（支持6列和7列格式）"""
        if not os.path.exists(filename):
            print(f"⚠️  杂质信息文件不存在: {filename}")
            return self
            
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            centers = []
            radii = []
            types = []
            
            print(f"\n🔍 解析 {os.path.basename(filename)}")
            
            data_line_count = 0
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                
                if data_line_count < 3:
                    print(f"   数据行{data_line_count+1}: 列数={len(parts)}")
                
                try:
                    if len(parts) >= 7:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        r = float(parts[4])
                        ptype = parts[5]
                        
                    elif len(parts) >= 6:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        r = float(parts[4])
                        ptype = 'large'
                        
                    else:
                        continue
                    
                    centers.append([x, y, z])
                    radii.append(r)
                    types.append(ptype)
                    data_line_count += 1
                    
                    if data_line_count <= 3:
                        print(f"      ✅ X={x:.1f}, Y={y:.1f}, Z={z:.1f}, R={r:.1f}, Type={ptype}")
                    
                except (ValueError, IndexError):
                    continue
            
            self.centers = np.array(centers) if centers else np.empty((0, 3))
            self.radii = np.array(radii) if radii else np.empty(0)
            self.types = types
            
            if len(self.centers) > 0:
                print(f"\n✅ 成功加载 {len(self.centers)} 个球形杂质")
                print(f"   坐标范围: X=[{self.centers[:,0].min():.1f}, {self.centers[:,0].max():.1f}] b")
                print(f"   坐标范围: Y=[{self.centers[:,1].min():.1f}, {self.centers[:,1].max():.1f}] b")
                print(f"   坐标范围: Z=[{self.centers[:,2].min():.1f}, {self.centers[:,2].max():.1f}] b")
                print(f"   半径范围: [{self.radii.min():.1f}, {self.radii.max():.1f}] b")
                print(f"   半径范围: [{self.radii.min()*self.burgmag*1e9:.1f}, {self.radii.max()*self.burgmag*1e9:.1f}] nm")
                print(f"   盒子尺寸: {self.Lbox_b:.1f} b ({self.Lbox_m*1e6:.1f} μm)")
            
            return self
            
        except Exception as e:
            print(f"❌ 加载杂质信息失败: {e}")
            import traceback
            traceback.print_exc()
            return self
    
    def is_inside_any_sphere(self, points):
        """判断点是否在任何球形杂质内部"""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        N = points.shape[0]
        inside = np.zeros(N, dtype=bool)
        
        if len(self.centers) == 0:
            return inside
        
        for i in range(len(self.centers)):
            dist = np.linalg.norm(points - self.centers[i], axis=1)
            inside |= (dist <= self.radii[i])
        
        return inside
    
    def export_vtk_geometry(self, filename, resolution=20):
        """导出球形杂质的几何体为VTK文件"""
        if len(self.centers) == 0:
            print(f"⚠️  没有球形杂质可导出")
            return
        
        print(f"\n生成球形杂质几何体...")
        print(f"   球体数量: {len(self.centers)}")
        print(f"   分辨率: {resolution}x{resolution}")
        
        all_points = []
        all_cells = []
        all_types = []
        all_radii = []
        point_offset = 0
        
        # 为每个球生成顶点和面
        for i in range(len(self.centers)):
            center = self.centers[i]
            radius = self.radii[i]
            ptype = self.types[i] if i < len(self.types) else 'sphere'
            
            # 生成球面上的点（使用球坐标）
            theta = np.linspace(0, np.pi, resolution)
            phi = np.linspace(0, 2*np.pi, resolution)
            
            # 生成网格点
            points_this_sphere = []
            for t in theta:
                for p in phi:
                    x = center[0] + radius * np.sin(t) * np.cos(p)
                    y = center[1] + radius * np.sin(t) * np.sin(p)
                    z = center[2] + radius * np.cos(t)
                    points_this_sphere.append([x, y, z])
            
            all_points.extend(points_this_sphere)
            
            # 生成四边形面片
            for j in range(resolution - 1):
                for k in range(resolution - 1):
                    p1 = point_offset + j * resolution + k
                    p2 = point_offset + j * resolution + (k + 1)
                    p3 = point_offset + (j + 1) * resolution + (k + 1)
                    p4 = point_offset + (j + 1) * resolution + k
                    all_cells.append([4, p1, p2, p3, p4])
            
            # 记录每个球的类型和半径
            num_cells_this_sphere = (resolution - 1) * (resolution - 1)
            type_val = 1 if ptype == 'large' else 0
            all_types.extend([type_val] * num_cells_this_sphere)
            all_radii.extend([radius] * num_cells_this_sphere)
            
            point_offset += len(points_this_sphere)
        
        all_points = np.array(all_points)
        
        # 调试：检查生成的几何体范围
        print(f"\n🔍 几何体统计:")
        print(f"   总点数: {len(all_points)}")
        print(f"   总面片数: {len(all_cells)}")
        print(f"   X范围: [{all_points[:,0].min():.1f}, {all_points[:,0].max():.1f}] b")
        print(f"   Y范围: [{all_points[:,1].min():.1f}, {all_points[:,1].max():.1f}] b")
        print(f"   Z范围: [{all_points[:,2].min():.1f}, {all_points[:,2].max():.1f}] b")
        print(f"   盒子尺寸: {self.Lbox_b:.1f} b")
        
        # 写入VTK文件
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Spherical precipitates geometry\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            
            # 写入点
            f.write(f"\nPOINTS {len(all_points)} float\n")
            for pt in all_points:
                f.write(f"{pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")
            
            # 写入面片
            total_size = sum(len(cell) for cell in all_cells)
            f.write(f"\nPOLYGONS {len(all_cells)} {total_size}\n")
            for cell in all_cells:
                f.write(" ".join(map(str, cell)) + "\n")
            
            # 写入单元数据
            f.write(f"\nCELL_DATA {len(all_cells)}\n")
            
            # 球类型（0=小球，1=大球）
            f.write("\nSCALARS PrecipitateType int\n")
            f.write("LOOKUP_TABLE default\n")
            for val in all_types:
                f.write(f"{val}\n")
            
            # 球半径（Burgers单位）
            f.write("\nSCALARS Radius_b float\n")
            f.write("LOOKUP_TABLE default\n")
            for val in all_radii:
                f.write(f"{val:.6e}\n")
            
            # 球半径（纳米）
            f.write("\nSCALARS Radius_nm float\n")
            f.write("LOOKUP_TABLE default\n")
            for val in all_radii:
                f.write(f"{val*self.burgmag*1e9:.6e}\n")
        
        print(f"✅ 球形杂质几何体已保存: {os.path.basename(filename)}")

def load_twin_planes_data(filename):
    """从 twin_planes.data 文件加载孪晶面信息，返回 (points_b, normals_b) 列表"""
    if not os.path.exists(filename):
        print(f"  Twin planes data file not found: {filename}")
        return [], []
    points_b, normals_b = [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                points_b.append([float(parts[1]), float(parts[2]), float(parts[3])])
                normals_b.append([float(parts[4]), float(parts[5]), float(parts[6])])
    print(f"  Loaded {len(points_b)} twin planes from {os.path.basename(filename)}")
    return points_b, normals_b


def export_twin_planes_vtk_from_data(filename, Lbox_b, points_b, normals_b):
    """
    根据平面点+法向量导出孪晶面VTK文件。
    每个平面生成一个覆盖整个盒子截面的矩形。
    """
    if not points_b:
        return

    nplanes = len(points_b)
    npoints = nplanes * 4
    ncells  = nplanes

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Twin boundary planes\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"\nPOINTS {npoints} float\n")
        for i in range(nplanes):
            p = np.array(points_b[i])
            n = np.array(normals_b[i])
            n = n / np.linalg.norm(n)

            # 构造平面上的两个正交切向量
            if abs(n[2]) > 0.9:
                t1 = np.cross(n, [1, 0, 0])
            else:
                t1 = np.cross(n, [0, 0, 1])
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            t2 = t2 / np.linalg.norm(t2)

            # 生成覆盖整个盒子的矩形（半对角线长度 = Lbox * sqrt(2)/2）
            half = Lbox_b * 0.75
            corners = [
                p - half * t1 - half * t2,
                p + half * t1 - half * t2,
                p + half * t1 + half * t2,
                p - half * t1 + half * t2,
            ]
            for c in corners:
                f.write(f"{c[0]:.6e} {c[1]:.6e} {c[2]:.6e}\n")

        f.write(f"\nPOLYGONS {ncells} {ncells * 5}\n")
        for i in range(nplanes):
            base = i * 4
            f.write(f"4 {base} {base+1} {base+2} {base+3}\n")

        f.write(f"\nCELL_DATA {ncells}\n")
        f.write("SCALARS PlaneID int\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(nplanes):
            f.write(f"{i}\n")

    print(f"  Twin planes VTK saved: {os.path.basename(filename)} ({nplanes} planes)")


def export_twin_planes_vtk(filename, Lbox_b, twin_z_fractions):
    """
    导出孪晶面为VTK文件，每个孪晶面是一个覆盖整个盒子的矩形。
    Lbox_b: 盒子尺寸（Burgers 单位）
    twin_z_fractions: z 方向分数位置列表，如 [0.5] 表示 z=Lbox_b/2
    """
    if not twin_z_fractions:
        return

    nplanes = len(twin_z_fractions)
    npoints = nplanes * 4  # 每个平面4个角点
    ncells  = nplanes      # 每个平面1个四边形

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Twin boundary planes\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # 写入点：每个平面4个角
        f.write(f"\nPOINTS {npoints} float\n")
        for frac in twin_z_fractions:
            z = Lbox_b * frac
            f.write(f"0.0 0.0 {z:.6e}\n")
            f.write(f"{Lbox_b:.6e} 0.0 {z:.6e}\n")
            f.write(f"{Lbox_b:.6e} {Lbox_b:.6e} {z:.6e}\n")
            f.write(f"0.0 {Lbox_b:.6e} {z:.6e}\n")

        # 写入面片
        f.write(f"\nPOLYGONS {ncells} {ncells * 5}\n")
        for i in range(nplanes):
            base = i * 4
            f.write(f"4 {base} {base+1} {base+2} {base+3}\n")

        # 写入单元数据
        f.write(f"\nCELL_DATA {ncells}\n")
        f.write("SCALARS TwinPlaneZ float\n")
        f.write("LOOKUP_TABLE default\n")
        for frac in twin_z_fractions:
            f.write(f"{Lbox_b * frac:.6e}\n")

    print(f"Twin planes VTK saved: {filename} ({nplanes} planes)")


def compute_orowan_flags(net_manager, precipitates, dist_factor=1.5):

    G = net_manager.get_disnet(ExaDisNet)
    data = G.export_data()

    segsnid   = data['segs']['nodeids']
    positions = data['nodes']['positions']
    nsegs     = segsnid.shape[0]

    orowan_flag = np.zeros(nsegs, dtype=int)

    if precipitates is None or len(precipitates.centers) == 0:
        return orowan_flag

    links = G.net.physical_links()
    orowan_count = 0

    for link in links:
        if len(link) < 3:
            continue

        seg_ids = np.array(link)

        # 正确的闭合判断：每个节点恰好出现2次
        all_node_ids = segsnid[seg_ids].ravel()
        unique_nodes, counts = np.unique(all_node_ids, return_counts=True)
        if not np.all(counts == 2):
            continue  # 不闭合，跳过

        # 计算环的几何中心
        center = positions[unique_nodes].mean(axis=0)

        # 判断是否靠近某个球形杂质
        near = False
        for j in range(len(precipitates.centers)):
            dist = np.linalg.norm(center - precipitates.centers[j])
            if dist < precipitates.radii[j] * dist_factor:
                near = True
                break

        if near:
            orowan_flag[seg_ids] = 1
            orowan_count += 1

    print(f"   识别到 Orowan 环: {orowan_count} 个 ({int(np.sum(orowan_flag))} 条线段)")
    return orowan_flag
def convert_paradis_to_vtk_with_precipitates(input_dir, output_dir):
    """
    将ParaDiS数据文件转换为VTK，并生成球形杂质几何体
    节点级别的OutsideSphere字段已集成到write_vtk()中
    """
    print("="*60)
    print("ParaDiS到VTK转换 + 球形杂质几何体")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    
    # 加载球形杂质信息
    precipitates_file = os.path.join(input_dir, 'precipitates.data')
    precipitates = None
    
    if os.path.exists(precipitates_file):
        print(f"\n加载球形杂质信息...")
        precipitates = SphericalPrecipitates(Lbox_m=5e-6, burgmag=0.2556e-9)
        precipitates.load_from_file(precipitates_file)
        
        # 生成球形杂质几何体VTK文件
        if len(precipitates.centers) > 0:
            precipitates_vtk = os.path.join(output_dir, 'precipitates.vtk')
            precipitates.export_vtk_geometry(precipitates_vtk, resolution=20)
    else:
        print(f"\n⚠️  未找到杂质信息文件: {precipitates_file}")

    # 生成孪晶面VTK文件
    Lbox_b = 5e-6 / 0.2556e-9
    twin_data_file = os.path.join(input_dir, 'twin_planes.data')
    if os.path.exists(twin_data_file):
        # 优先从 data 文件读取（支持任意方向的孪晶面）
        tp, tn = load_twin_planes_data(twin_data_file)
        if tp:
            twin_vtk = os.path.join(output_dir, 'twin_planes.vtk')
            export_twin_planes_vtk_from_data(twin_vtk, Lbox_b, tp, tn)
    elif TWIN_Z_FRACTIONS:
        # 回退：用配置里的 z 分数
        twin_vtk = os.path.join(output_dir, 'twin_planes.vtk')
        export_twin_planes_vtk(twin_vtk, Lbox_b, TWIN_Z_FRACTIONS)
    
    # 查找所有 .data 文件（排除 precipitates.data 和 twin_planes.data）
    all_data = sorted(glob.glob(os.path.join(input_dir, '*.data')))
    data_files = [f for f in all_data
                  if os.path.basename(f) not in ('precipitates.data', 'twin_planes.data')]

    if len(data_files) == 0:
        print(f"\n❌ 未找到.data文件")
        return

    print(f"\n找到 {len(data_files)} 个data文件")
    print("="*60)
    
    # 初始化pyexadis
    pyexadis.initialize()
    
    success_count = 0
    
    # 处理所有data文件
    for idx, data_file in enumerate(data_files):
        try:
            basename = os.path.basename(data_file)
            name = basename.replace('.data', '')

            print(f"\n[{idx+1}/{len(data_files)}] 处理: {basename}")

            # 读取ParaDiS数据
            net_manager = read_paradis(data_file)
            orowan_seg_flag = compute_orowan_flags(net_manager, precipitates)
            vtk_file = os.path.join(output_dir, f'{name}.vtk')
            write_vtk(
                net_manager, 
                vtk_file,
                segprops={'OrowanLoop': orowan_seg_flag},
                precipitates=precipitates,  # 传入precipitates对象
                verbose=True
            )
            print(f"   ✅ VTK文件已生成（包含OutsideSphere字段）")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理失败: {basename}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 清理pyexadis
    pyexadis.finalize()
    
    print("\n" + "="*60)
    print(f"✅ 转换完成: 成功 {success_count}/{len(data_files)} 个文件")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    # 验证生成的文件
    print("\n生成的文件:")
    all_files = sorted(glob.glob(os.path.join(output_dir, '*.vtk')))
    for vf in all_files[:10]:  # 只显示前10个
        print(f"  - {os.path.basename(vf)}")
    if len(all_files) > 10:
        print(f"  ... 还有 {len(all_files)-10} 个文件")
    
    print("\n" + "="*60)
    print("📊 ParaView使用说明:")
    print("="*60)
    print("1. 打开位错网络:")
    print("   - File → Open → config.0.vtk")
    print("   - Coloring → OutsideSphere")
    print("   - 0 = 球内节点（红色），1 = 球外节点（蓝色）")
    print("")
    print("2. 打开球形杂质:")
    print("   - File → Open → precipitates.vtk")
    print("   - Coloring → Radius_nm (显示半径)")
    print("   - Representation → Surface")
    print("   - Opacity → 0.3 (半透明)")
    print("")
    print("3. 同时显示位错和杂质:")
    print("   - 在Pipeline Browser中同时激活两个文件")
    print("   - 观察节点与杂质的相互作用")
    print("="*60)

# 在 paraview.py 末尾加上这个函数
def convert_debug_files(debug_dir, output_dir):
    """转换 debug_max_conn_*.data 文件用于分析"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_files = sorted(glob.glob(os.path.join(debug_dir, 'debug_max_conn_*.data')))
    if len(data_files) == 0:
        print(f"未找到 debug 文件")
        return

    # 加载杂质信息（从有夹杂版的输出目录）
    precipitates_file = '/data/home/dg000246b/openDIS/HomeWork/output_Cu_fcc/precipitates.data'
    precipitates = None
    if os.path.exists(precipitates_file):
        precipitates = SphericalPrecipitates(Lbox_m=5e-6, burgmag=0.2556e-9)
        precipitates.load_from_file(precipitates_file)

    pyexadis.initialize()

    for data_file in data_files:
        basename = os.path.basename(data_file)
        name     = basename.replace('.data', '')
        print(f"\n处理: {basename}")

        net_manager     = read_paradis(data_file)
        orowan_seg_flag = compute_orowan_flags(net_manager, precipitates)

        vtk_file = os.path.join(output_dir, f'{name}.vtk')
        write_vtk(
            net_manager,
            vtk_file,
            segprops={'OrowanLoop': orowan_seg_flag},
            precipitates=precipitates,
            verbose=True
        )
        print(f"   ✅ {name}.vtk")
    pyexadis.finalize()
    print(f"\n转换完成，输出目录: {output_dir}")
# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'debug':
        # python paraview.py debug
        convert_debug_files(
            debug_dir='/data/home/dg000246b/openDIS/HomeWork/debug',
            output_dir='/data/home/dg000246b/openDIS/HomeWork/debug_vtk'
        )
    else:
        convert_paradis_to_vtk_with_precipitates(INPUT_PATH, OUTPUT_PATH)