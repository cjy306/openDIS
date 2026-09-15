"""
生成与无障碍组相同的 5 um / 1 um FR 网络，并加入 type=1 点障碍。

点障碍布置在23个活动滑移系FR源的活动臂前后，共92个。
输出: init_fr1um_point_obstacles/{init_config.data, init_config.vtk, obstacles.data}
"""
import os
import sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if path not in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager, NodeConstraints
from pyexadis_utils import write_vtk

BURGMAG = 0.248e-9
LBOX_M = 5.0e-6
RHO_TARGET = 1.0e12
ARM_LEN_M = 1.0e-6
MAXSEG_B = 160.0
FOREST_PER_SYS = 2
OBS_OFFSET_B = 300.0
OBS_OFFPLANE_B = 20.0
OBS_RADIUS_B = 40.0

SLIP_B = np.array([
    [-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.], [1., -1., 1.],
    [-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.], [1., -1., 1.],
    [-1., 1., 1.], [1., 1., 1.], [-1., -1., 1.], [1., -1., 1.],
])
SLIP_N = np.array([
    [0., -1., 1.], [0., -1., 1.], [0., 1., 1.], [0., 1., 1.],
    [1., 0., 1.], [-1., 0., 1.], [1., 0., 1.], [-1., 0., 1.],
    [1., 1., 0.], [-1., 1., 0.], [-1., 1., 0.], [1., 1., 0.],
])


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def insert_fr_loop(nodes, segs, burg, plane, arm_len_b, center):
    """插入闭合FR回路，并返回活动臂上的两个障碍锚点。"""
    # PHYS-APPROX: 三条钉扎边是维持有限FR源闭合拓扑的虚拟臂；上限是它们不参与
    # 真实网络演化，升级路径是使用预变形/弛豫得到的拓扑自洽位错网络。
    burg, plane = unit(burg), unit(plane)
    edge = unit(np.cross(plane, burg))
    closure = unit(np.cross(burg, edge))
    center = np.asarray(center)
    corners = (
        center - 0.5 * arm_len_b * edge,
        center + 0.5 * arm_len_b * edge,
        center + 0.5 * arm_len_b * edge - arm_len_b * closure,
        center - 0.5 * arm_len_b * edge - arm_len_b * closure,
    )

    first = len(nodes)
    nodes.append(np.concatenate((corners[0], [NodeConstraints.PINNED_NODE])))
    previous = first
    for iedge in range(4):
        start, end = corners[iedge], corners[(iedge + 1) % 4]
        nseg = int(np.ceil(np.linalg.norm(end - start) / MAXSEG_B))
        for ipart in range(1, nseg + 1):
            if iedge == 3 and ipart == nseg:
                current = first
            else:
                point = start + (end - start) * ipart / nseg
                movable = iedge == 0 and ipart < nseg
                constraint = (NodeConstraints.UNCONSTRAINED if movable
                              else NodeConstraints.PINNED_NODE)
                nodes.append(np.concatenate((point, [constraint])))
                current = len(nodes) - 1
            seg_plane = unit(np.cross(burg, end - start))
            segs.append(np.concatenate(([previous, current], burg, seg_plane)))
            previous = current

    anchors = (corners[0] + 0.25 * (corners[1] - corners[0]),
               corners[0] + 0.75 * (corners[1] - corners[0]))
    return nodes, segs, anchors, burg, plane


def build_network_and_obstacles(rng):
    Lbox_b = LBOX_M / BURGMAG
    arm_b = ARM_LEN_M / BURGMAG
    total_loops = int(round(RHO_TARGET * LBOX_M ** 3 / (4.0 * ARM_LEN_M)))
    nforest = 4 * FOREST_PER_SYS
    nactive = total_loops - nforest
    base, remainder = divmod(nactive, 8)
    counts = [base + (i < remainder) for i in range(8)] + [FOREST_PER_SYS] * 4
    bsys = SLIP_B / np.linalg.norm(SLIP_B, axis=1)[:, None]
    nsys = SLIP_N / np.linalg.norm(SLIP_N, axis=1)[:, None]
    margin = arm_b + OBS_OFFSET_B + OBS_OFFPLANE_B
    nodes, segs, obstacles = [], [], []
    for isys, count in enumerate(counts):
        for _ in range(count):
            center = rng.uniform(margin, Lbox_b - margin, size=3)
            _, _, anchors, burg, plane = insert_fr_loop(
                nodes, segs, bsys[isys], nsys[isys], arm_b, center)
            if isys < 8:
                # PHYS-APPROX: 与活动源关联布点只用于定性验证切过强化；
                # 上限是不代表真实颗粒统计，升级路径是后续标定随机障碍场。
                for anchor in anchors:
                    for sign in (+1.0, -1.0):
                        point = anchor + sign * OBS_OFFSET_B * burg + OBS_OFFPLANE_B * plane
                        obstacles.append(np.concatenate((point, [OBS_RADIUS_B])))

    achieved = total_loops * 4.0 * ARM_LEN_M / LBOX_M ** 3
    print('FR回路: 活动系%d个 + 林位错系%d个, 共%d个' % (nactive, nforest, total_loops))
    print('节点%d, 线段%d, 实际密度 %.3e m^-2' % (len(nodes), len(segs), achieved))
    print('type=1点障碍: %d个, R=%.0fb' % (len(obstacles), OBS_RADIUS_B))
    return Lbox_b, nodes, segs, np.asarray(obstacles)


def main():
    pyexadis.initialize()
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, 'init_fr1um_point_obstacles')
        os.makedirs(out_dir, exist_ok=True)
        Lbox_b, nodes, segs, obstacles = build_network_and_obstacles(
            np.random.RandomState(12345))
        network = ExaDisNet(pyexadis.Cell(Lbox_b), nodes, segs)
        network.write_data(os.path.join(out_dir, 'init_config.data'))
        write_vtk(DisNetManager(network), os.path.join(out_dir, 'init_config.vtk'),
                  crystal='BCC', verbose=False)
        np.savetxt(os.path.join(out_dir, 'obstacles.data'), obstacles, fmt='%.10e',
                   header='cx cy cz radius (units of b; type=1 point obstacles)')
        print('点障碍初始网络已写出:', out_dir)
    finally:
        pyexadis.finalize()


if __name__ == '__main__':
    main()
