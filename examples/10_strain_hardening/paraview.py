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
input_path = '/data/home/dg000246b/openDIS/examples/10_strain_hardening/output_Cu_vtk'
output_path = '/data/home/dg000246b/openDIS/examples/10_strain_hardening/output_fcc_Cu_15um_1e3'
for f in glob.glob(output_path+'/*.data'):
    N = read_paradis(f)
    write_vtk(N, os.path.splitext(f)[0]+'.vtk')
