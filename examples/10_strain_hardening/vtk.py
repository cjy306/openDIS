import os, glob
import sys
os.environ['OMP_PROC_BIND'] = 'spread'
os.environ['OMP_PLACES'] = 'threads'
# 添加模块所在目录到查找路径（就像告诉C编译器.h文件在哪里）
sys.path.append('/data/home/dg000246b/openDIS/core/exadis/python/')
pyexadis_paths = ['../../python', '../../lib', '../../core/pydis/python', '../../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_utils import read_paradis, write_vtk

pyexadis.initialize()

# 分别设置输入和输出路径
input_path = '/data/home/dg000246b/openDIS/examples/10_strain_hardening/output_fcc_Cu_15um_1e3'
output_path = '/data/home/dg000246b/openDIS/examples/10_strain_hardening/output_Cu_vtk'  

os.makedirs(output_path, exist_ok=True)

# 遍历输入目录中的所有.data文件
for input_file in glob.glob(os.path.join(input_path, '*.data')):
    # 读取PARADIS文件
    N = read_paradis(input_file)
    
    # 获取输入文件名（不含路径）
    filename = os.path.basename(input_file)
    
    # 构建输出文件路径（保持相同文件名，只改扩展名）
    output_file = os.path.join(output_path, os.path.splitext(filename)[0] + '.vtk')
    
    # 写入VTK文件到输出路径
    write_vtk(N, output_file)
    
    print(f"转换完成: {input_file} -> {output_file}")
