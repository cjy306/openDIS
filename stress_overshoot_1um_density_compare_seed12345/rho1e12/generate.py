"""Resolve the existing OpenDiS installation without modifying it."""
import os
import sys
from pathlib import Path

def setup():
    candidates = ([Path(os.environ['OPENDIS_ROOT'])] if 'OPENDIS_ROOT' in os.environ else [])
    candidates += list(Path(__file__).resolve().parents)
    candidates += [Path.home() / 'openDIS', Path('E:/openDIS')]
    for root in candidates:
        if (root / 'core/exadis/python/pyexadis_base.py').is_file():
            for part in ('python', 'lib', 'core/pydis/python', 'core/exadis/python'):
                sys.path.insert(0, str(root / part))
            return root
    raise FileNotFoundError('Set OPENDIS_ROOT to your OpenDiS repository')


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
NUM_SOURCES = 125
MAXSEG_B = 200
TARGET_RHO = 1.0e12
ACTUAL_RHO = NUM_SOURCES * FR_LENGTH_M / LBOX_M**3

HIGH_SCHMID_001 = (0, 1, 3, 4, 6, 7, 9, 10)
ZERO_SCHMID_001 = (2, 5, 8, 11)
# PHYS-APPROX: 125 sources cannot split exactly over 12 systems; ceiling = 10 or 11 per system; upgrade = jointly adjust source count and box volume to retain density.
CASES = {"uniform12_1um_rho1e12": None}

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
    """Return seeded positions and balanced slip-system assignments."""
    if case not in CASES:
        raise ValueError("unknown factor-A case: %s" % case)
    position_rng = random.Random(seed)
    positions = [tuple(position_rng.random() for _ in range(3))
                 for _ in range(NUM_SOURCES)]
    assignment_rng = random.Random(seed ^ 0x5A17A5EED)
    order = list(range(NUM_SOURCES))
    assignment_rng.shuffle(order)
    systems = list(range(12))
    assignment_rng.shuffle(systems)
    system_ids = [-1] * NUM_SOURCES
    for rank, source_index in enumerate(order):
        system_ids[source_index] = systems[rank % 12]
    return positions, system_ids


def _load_pyexadis_modules():
    setup()
    import numpy as np
    import pyexadis
    from pyexadis_base import DisNetManager, ExaDisNet
    from pyexadis_utils import insert_frank_read_src, write_vtk
    return np, pyexadis, DisNetManager, ExaDisNet, insert_frank_read_src, write_vtk


def generate_case(case, seed, out_dir):
    """Write the initial network, VTK and source layout."""
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
    cell = pyexadis.Cell(h=box_b * np.eye(3), is_periodic=[True, True, True])
    positions = (np.asarray(cell.origin)
                 + np.matmul(np.asarray(positions_fractional), np.asarray(cell.h).T))
    angle_rng = random.Random(seed ^ 0x6C8E9CF5)
    angles = [angle_rng.uniform(0.0, 180.0) for _ in range(NUM_SOURCES)]
    nodes, segments = [], []
    for position, system_id, theta in zip(positions, system_ids, angles):
        nodes, segments = insert_frank_read_src(
            cell, nodes, segments, burgers[system_id], normals[system_id],
            length_b, position, theta=theta, numnodes=numnodes)
    network = ExaDisNet(cell, nodes, segments)
    if Path(out_dir, 'init_config.data').exists():
        raise FileExistsError('Initial network exists; reuse it or select a new --out')
    os.makedirs(out_dir, exist_ok=True)
    data_file = os.path.join(out_dir, "init_config.data")
    vtk_file = os.path.join(out_dir, "init_config_labeled.vtk")
    layout_file = os.path.join(out_dir, "source_layout.csv")
    network.write_data(data_file)
    write_vtk(DisNetManager(network), vtk_file, crystal="FCC", verbose=False)
    high_ids = set(HIGH_SCHMID_001)
    with open(layout_file, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_id", "fx", "fy", "fz", "slip_system_id", "group", "theta_deg"))
        for source_id, (position, system_id) in enumerate(
                zip(positions_fractional, system_ids)):
            group = "high_schmid" if system_id in high_ids else "zero_schmid"
            writer.writerow((source_id, *position, system_id, group, angles[source_id]))
    return data_file, vtk_file, layout_file


def run_cli(case, default_seed, base_dir):
    parser = argparse.ArgumentParser(
        description="Generate FCC Cu %s FR network" % case)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    base_dir = os.path.abspath(base_dir)
    out_dir = args.out or os.path.join(base_dir, "init")
    _, pyexadis, _, _, _, _ = _load_pyexadis_modules()
    pyexadis.initialize()
    try:
        files = generate_case(case, args.seed, out_dir)
    finally:
        pyexadis.finalize()
    from collections import Counter
    print('slip system counts:', dict(sorted(Counter(make_layout(args.seed, case)[1]).items())))
    print("FR sources=%d, length=%.3f um, rho=%.3e 1/m^2 (target %.3e)"
          % (NUM_SOURCES, FR_LENGTH_M * 1.0e6, ACTUAL_RHO, TARGET_RHO))
    for path in files:
        print("wrote: %s" % path)



if __name__ == "__main__":
    run_cli("uniform12_1um_rho1e12", 12345, Path(__file__).resolve().parent)
