"""
生成氧化物高斯势验证的初始构型(两段分离:本脚本生成,test_oxide_verify.py 读取)

产物(写入 init_data_oxide_verify/):
  init_config.data  —— 一条穿盒无限刃位错(b=1/2[111], n=(0-11), 线向=n×b)
  oxides.data       —— 氧化物几何: cx cy cz Rp(单位 b);盒心一个
A 是扫描参数,不写进文件,由 test_oxide_verify.py --A 传入。

用法:
  python generate_oxide_verify.py            # 默认 Rp=10nm
  python generate_oxide_verify.py --Rp 5
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet
from pyexadis_utils import insert_infinite_line

BURGMAG  = 0.248e-9      # b [m]
LBOX_M   = 0.5e-6        # 500nm 盒(小,验证跑得快)
MAXSEG_B = 40            # 离散段长 ≈10nm(比 Rp 细,位错"看得见"氧化物)


def main():
    import argparse
    pyexadis.initialize()
    p = argparse.ArgumentParser(description='生成氧化物验证初始构型')
    p.add_argument('--Rp',  type=float, default=10.0, help='高斯势宽度 [nm],默认 10')
    p.add_argument('--out', type=str,   default=None)
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, 'init_data_oxide_verify')
    os.makedirs(out_dir, exist_ok=True)

    Lbox = LBOX_M / BURGMAG
    cell = pyexadis.Cell(Lbox)
    center = np.array(cell.center())

    # 滑移系 b=1/2[111], n=(0,-1,1);刃位错线方向 e=n×b(在滑移面内、垂直 b)
    b = np.array([1., 1., 1.]) / np.sqrt(3.0)
    n = np.array([0., -1., 1.]) / np.sqrt(2.0)
    e = np.cross(n, b); e /= np.linalg.norm(e)

    # 起点从盒心沿 -b 退 1/4 盒(位移⊥n,与氧化物同滑移面;受切应力后沿 +b 撞向氧化物)
    origin = center - 0.25 * Lbox * b
    nodes, segs = insert_infinite_line(cell, [], [], b, n, origin,
                                       linedir=e, maxseg=MAXSEG_B)
    G = ExaDisNet(cell, nodes, segs)
    G.write_data(os.path.join(out_dir, 'init_config.data'))

    # 氧化物几何:盒心一个(cx cy cz Rp,单位 b)
    Rp_b = args.Rp * 1e-9 / BURGMAG
    ox = np.array([[center[0], center[1], center[2], Rp_b]])
    np.savetxt(os.path.join(out_dir, 'oxides.data'), ox, fmt='%.6e')

    print(f'初始构型: {out_dir}/init_config.data')
    print(f'氧化物:   {out_dir}/oxides.data  (盒心 1 个, Rp={args.Rp}nm={Rp_b:.1f}b)')
    pyexadis.finalize()


if __name__ == "__main__":
    main()
