# 在服务器上单独测试
import os
import sys
import numpy as np
pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
for path in pyexadis_paths:
    abspath = os.path.abspath(path)
    if abspath not in sys.path:
        sys.path.append(abspath)

np.set_printoptions(threshold=20, edgeitems=5)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, SimulateNetwork, read_restart, NodeConstraints
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
    from framework.disnet_manager import DisNetManager
    from pyexadis_utils import read_paradis
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e


burgmag = 0.2556e-9
Lbox_m  = 5e-6
Lbox_b  = Lbox_m / burgmag  # 盒子尺寸（Burgers单位）

pyexadis.initialize()
net = read_paradis('output_Cu_fcc/config.200.data')
G = net.get_disnet(ExaDisNet)
data = G.export_data()
segsnid = data['segs']['nodeids']
links = G.net.physical_links()

# 加载杂质
import sys
sys.path.append('/data/home/dg000246b/openDIS/HomeWork')
from paraview import SphericalPrecipitates
prec = SphericalPrecipitates(Lbox_m=Lbox_m, burgmag=burgmag)
prec.load_from_file('output_Cu_fcc/precipitates.data')
print(f'加载杂质: {len(prec.centers)} 个')
print(f'杂质半径: {prec.radii[0]:.1f} b')
print(f'盒子尺寸: {Lbox_b:.1f} b')

# 找闭合环并判断是否靠近杂质
orowan_count = 0
for link in links:
    seg_ids = np.array(link)
    all_nodes = segsnid[seg_ids].ravel()
    unique, counts = np.unique(all_nodes, return_counts=True)
    if not np.all(counts == 2):
        continue
    pos = data['nodes']['positions'][unique]
    center = pos.mean(axis=0)
    # 检查是否靠近任意杂质
    for j in range(len(prec.centers)):
        dist = np.linalg.norm(center - prec.centers[j])
        if dist < prec.radii[j] * 3.0:
            orowan_count += 1
            print(f'  Orowan环: {len(link)}段, 中心=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}), 距杂质{j}距离={dist:.1f} b')
            break

print(f'识别到 Orowan 环: {orowan_count} 个')
pyexadis.finalize()