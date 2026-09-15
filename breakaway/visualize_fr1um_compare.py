"""使用 pyexadis_utils.write_vtk 转换两组构型，并导出点障碍标记。"""
import glob
import os
import re
import sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if path not in sys.path]

import pyexadis
from pyexadis_utils import read_paradis, write_vtk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES = [
    ('no_obstacles', os.path.join(BASE_DIR, 'output_fr1um_no_obstacles')),
    ('point_obstacles', os.path.join(BASE_DIR, 'output_fr1um_point_obstacles')),
]
OBSTACLES = os.path.join(BASE_DIR, 'init_fr1um_point_obstacles', 'obstacles.data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'vtk_fr1um_compare')


def step_number(path):
    match = re.findall(r'\d+', os.path.basename(path))
    return int(match[-1]) if match else -1


def convert_case(label, input_dir):
    out_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    data_files = sorted(glob.glob(os.path.join(input_dir, '*.data')), key=step_number)
    if not data_files:
        raise FileNotFoundError('没有找到构型文件: %s' % input_dir)
    for i, data_file in enumerate(data_files):
        vtk_file = os.path.join(out_dir,
                                os.path.basename(data_file).replace('.data', '.vtk'))
        # 位错网络VTK直接使用pyexadis自带write_vtk，不自行重写转换格式。
        write_vtk(read_paradis(data_file), vtk_file, crystal='BCC', verbose=False)
        print('[%s %d/%d] %s' % (label, i + 1, len(data_files), vtk_file))


def write_obstacles_vtk():
    obstacles = np.loadtxt(OBSTACLES)
    if obstacles.ndim == 1:
        obstacles = obstacles.reshape(1, -1)
    out_dir = os.path.join(OUTPUT_DIR, 'point_obstacles')
    os.makedirs(out_dir, exist_ok=True)
    vtk_file = os.path.join(out_dir, 'obstacles.vtk')
    # PHYS-APPROX: 球形glyph只表示点障碍捕获半径，不代表真实氧化物表面；
    # 上限是不能据此解释颗粒几何，升级路径是专门的点/捕获体可视化。
    with open(vtk_file, 'w') as stream:
        stream.write('# vtk DataFile Version 3.0\n')
        stream.write('type-1 point obstacles; capture-radius markers\n')
        stream.write('ASCII\nDATASET POLYDATA\n')
        stream.write('POINTS %d float\n' % len(obstacles))
        for x, y, z, _ in obstacles:
            stream.write('%.8e %.8e %.8e\n' % (x, y, z))
        stream.write('VERTICES %d %d\n' % (len(obstacles), 2 * len(obstacles)))
        for i in range(len(obstacles)):
            stream.write('1 %d\n' % i)
        stream.write('POINT_DATA %d\n' % len(obstacles))
        stream.write('SCALARS capture_radius float 1\nLOOKUP_TABLE default\n')
        for radius in obstacles[:, 3]:
            stream.write('%.8e\n' % radius)
    print('点障碍VTK:', vtk_file)


def main():
    pyexadis.initialize()
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for label, input_dir in CASES:
            convert_case(label, input_dir)
        write_obstacles_vtk()
    finally:
        pyexadis.finalize()
    print('VTK转换完成:', OUTPUT_DIR)


if __name__ == '__main__':
    main()
