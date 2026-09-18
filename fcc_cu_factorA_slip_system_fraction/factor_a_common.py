"""FCC Cu [001] 因素A：固定空间母版，仅改变初始滑移系占比。"""

import argparse
import csv
import math
import os
from pathlib import Path
import random
import sys


CRYSTAL = "fcc"
BURGMAG_M = 2.55e-10
LBOX_M = 5.0e-6
FR_LENGTH_M = 1.0e-6
NUM_SOURCES = 128
MAXSEG_B = 200
TARGET_RHO = 1.0e12
ACTUAL_RHO = NUM_SOURCES * FR_LENGTH_M / LBOX_M**3

HIGH_SCHMID_001 = (0, 1, 3, 4, 6, 7, 9, 10)
ZERO_SCHMID_001 = (2, 5, 8, 11)
CASES = {"A25": 32, "A50": 64, "A75": 96, "A100": 128}

_FCC_B = (
    (0., 1., -1.), (1., 0., -1.), (1., -1., 0.),
    (0., 1., -1.), (1., 0., 1.), (1., 1., 0.),
    (0., 1., 1.), (1., 0., -1.), (1., 1., 0.),
    (0., 1., 1.), (1., 0., 1.), (1., -1., 0.),
)
_FCC_N = (
    (1., 1., 1.), (1., 1., 1.), (1., 1., 1.),
    (-1., 1., 1.), (-1., 1., 1.), (-1., 1., 1.),
    (1., -1., 1.), (1., -1., 1.), (1., -1., 1.),
    (1., 1., -1.), (1., 1., -1.), (1., 1., -1.),
)


def make_layout(seed, case):
    """返回固定的128个分数坐标及指定因素A组的滑移系编号。"""
    if case not in CASES:
        raise ValueError("unknown factor-A case: %s" % case)
    position_rng = random.Random(seed)
    positions = [tuple(position_rng.random() for _ in range(3))
                 for _ in range(NUM_SOURCES)]
    assignment_rng = random.Random(seed ^ 0x5A17A5EED)
    order = list(range(NUM_SOURCES))
    assignment_rng.shuffle(order)
    active_count = CASES[case]
    system_ids = [-1] * NUM_SOURCES
    for rank, source_index in enumerate(order):
        if rank < active_count:
            system_ids[source_index] = HIGH_SCHMID_001[rank % len(HIGH_SCHMID_001)]
        else:
            system_ids[source_index] = ZERO_SCHMID_001[rank % len(ZERO_SCHMID_001)]
    return positions, system_ids


def _load_pyexadis_modules():
    repo_root = Path(__file__).resolve().parent.parent
    paths = (
        repo_root / "python", repo_root / "lib",
        repo_root / "core" / "pydis" / "python",
        repo_root / "core" / "exadis" / "python",
    )
    for path in paths:
        if str(path) not in sys.path:
            sys.path.append(str(path))
    import numpy as np
    import pyexadis
    from pyexadis_base import DisNetManager, ExaDisNet
    from pyexadis_utils import insert_frank_read_src, write_vtk
    return np, pyexadis, DisNetManager, ExaDisNet, insert_frank_read_src, write_vtk


def generate_case(case, seed, out_dir):
    """生成一个因素A初始构型，并写出data、VTK和位置/滑移系清单。"""
    (np, pyexadis, DisNetManager, ExaDisNet,
     insert_frank_read_src, write_vtk) = _load_pyexadis_modules()
    positions_fractional, system_ids = make_layout(seed, case)
    burgers = np.asarray(_FCC_B, dtype=float)
    burgers /= np.linalg.norm(burgers, axis=1)[:, None]
    normals = np.asarray(_FCC_N, dtype=float)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    box_b = LBOX_M / BURGMAG_M
    length_b = FR_LENGTH_M / BURGMAG_M
    numnodes = max(3, int(math.ceil(length_b / MAXSEG_B)) + 1)
    cell = pyexadis.Cell(box_b)
    positions = (np.asarray(cell.origin)
                 + np.matmul(np.asarray(positions_fractional), np.asarray(cell.h).T))
    nodes, segments = [], []
    for position, system_id in zip(positions, system_ids):
        nodes, segments = insert_frank_read_src(
            cell, nodes, segments, burgers[system_id], normals[system_id],
            length_b, position, theta=0.0, numnodes=numnodes)
    network = ExaDisNet(cell, nodes, segments)
    os.makedirs(out_dir, exist_ok=True)
    data_file = os.path.join(out_dir, "init_config.data")
    vtk_file = os.path.join(out_dir, "init_config_labeled.vtk")
    layout_file = os.path.join(out_dir, "source_layout.csv")
    network.write_data(data_file)
    write_vtk(DisNetManager(network), vtk_file, crystal="FCC", verbose=False)
    high_ids = set(HIGH_SCHMID_001)
    with open(layout_file, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_id", "fx", "fy", "fz", "slip_system_id", "group"))
        for source_id, (position, system_id) in enumerate(
                zip(positions_fractional, system_ids)):
            group = "high_schmid" if system_id in high_ids else "zero_schmid"
            writer.writerow((source_id, *position, system_id, group))
    return data_file, vtk_file, layout_file


def run_cli(case):
    parser = argparse.ArgumentParser(
        description="生成FCC Cu [001]因素A的%s初始FR构型" % case)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(base_dir, "init_%s_seed%d" % (case, args.seed))
    _, pyexadis, _, _, _, _ = _load_pyexadis_modules()
    pyexadis.initialize()
    try:
        files = generate_case(case, args.seed, out_dir)
    finally:
        pyexadis.finalize()
    active_count = CASES[case]
    print("[%s] high-Schmid=%d, zero-Schmid=%d, seed=%d"
          % (case, active_count, NUM_SOURCES - active_count, args.seed))
    print("FR sources=%d, length=%.3f um, rho=%.3e 1/m^2 (target %.3e)"
          % (NUM_SOURCES, FR_LENGTH_M * 1.0e6, ACTUAL_RHO, TARGET_RHO))
    for path in files:
        print("wrote: %s" % path)

