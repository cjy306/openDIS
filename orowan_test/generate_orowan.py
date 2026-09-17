"""Generate one refined Frank-Read source and three hard-sphere obstacles.

Outputs in ``init_orowan/``:
  init_config.data  -- 5 um periodic cell with one 1 um FR source
  obstacles.data    -- sphere centers and radii in Burgers-vector units
"""

import argparse
import os
import sys

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYEXADIS_PATHS = [
    os.path.join(BASE_DIR, "..", "python"),
    os.path.join(BASE_DIR, "..", "lib"),
    os.path.join(BASE_DIR, "..", "core", "pydis", "python"),
    os.path.join(BASE_DIR, "..", "core", "exadis", "python"),
]
for path in PYEXADIS_PATHS:
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.append(path)

import pyexadis
from pyexadis_base import ExaDisNet, NodeConstraints


BURGMAG = 0.248e-9
LBOX_M = 5.0e-6
FR_LENGTH_M = 1.0e-6
OBSTACLE_RADIUS_M = 50.0e-9
MAXSEG_B = 40.0

BURG = np.array([1.0, 1.0, 1.0])
PLANE = np.array([0.0, -1.0, 1.0])

# (offset along the FR line, distance ahead along +b), in metres.
OBSTACLE_OFFSETS_M = np.array([
    [0.0, 250.0e-9],
    [-250.0e-9, 450.0e-9],
    [250.0e-9, 650.0e-9],
])


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def insert_refined_fr_loop(nodes, segs, burg, plane, length_b, center):
    """Insert a refined rectangular FR loop with one mobile arm."""
    # PHYS-APPROX: the three pinned sides are a virtual closure for a finite FR
    # source in a periodic bulk cell; ceiling = they do not represent an
    # evolving junction network; upgrade = use a relaxed, topology-consistent
    # dislocation network containing a naturally pinned source.
    burg = unit(burg)
    plane = unit(plane)
    line_dir = unit(np.cross(plane, burg))
    closure_dir = unit(np.cross(burg, line_dir))
    center = np.asarray(center, dtype=float)

    corners = (
        center - 0.5 * length_b * line_dir,
        center + 0.5 * length_b * line_dir,
        center + 0.5 * length_b * line_dir - length_b * closure_dir,
        center - 0.5 * length_b * line_dir - length_b * closure_dir,
    )

    first = len(nodes)
    nodes.append(np.concatenate((corners[0], [NodeConstraints.PINNED_NODE])))
    previous = first

    for edge_id in range(4):
        start = corners[edge_id]
        end = corners[(edge_id + 1) % 4]
        nseg = int(np.ceil(np.linalg.norm(end - start) / MAXSEG_B))

        for part in range(1, nseg + 1):
            if edge_id == 3 and part == nseg:
                current = first
            else:
                position = start + (end - start) * part / nseg
                is_mobile = edge_id == 0 and part < nseg
                constraint = (NodeConstraints.UNCONSTRAINED if is_mobile
                              else NodeConstraints.PINNED_NODE)
                nodes.append(np.concatenate((position, [constraint])))
                current = len(nodes) - 1

            segment_plane = unit(np.cross(burg, end - start))
            segs.append(np.concatenate(
                ([previous, current], burg, segment_plane)))
            previous = current

    return line_dir


def build_configuration():
    lbox_b = LBOX_M / BURGMAG
    fr_length_b = FR_LENGTH_M / BURGMAG
    cell = pyexadis.Cell(lbox_b)
    fr_center = np.asarray(cell.center(), dtype=float)

    nodes = []
    segs = []
    burg = unit(BURG)
    plane = unit(PLANE)
    line_dir = insert_refined_fr_loop(
        nodes, segs, burg, plane, fr_length_b, fr_center)

    # PHYS-APPROX: R=50 nm is deliberately enlarged relative to typical ODS
    # particles; ceiling = this case validates geometry rather than a physical
    # particle-size distribution; upgrade = repeat with measured radii after
    # the contact algorithm and remeshing have passed convergence checks.
    offsets_b = OBSTACLE_OFFSETS_M / BURGMAG
    centers = np.array([
        fr_center + line_offset * line_dir + glide_offset * burg
        for line_offset, glide_offset in offsets_b
    ])
    radii = np.full(len(centers), OBSTACLE_RADIUS_M / BURGMAG)

    return cell, nodes, segs, centers, radii


def main():
    parser = argparse.ArgumentParser(
        description="Generate the single-FR, staggered-obstacle Orowan case")
    parser.add_argument(
        "--out", default=os.path.join(BASE_DIR, "init_orowan"),
        help="output directory (default: orowan_test/init_orowan)")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    pyexadis.initialize()
    try:
        cell, nodes, segs, centers, radii = build_configuration()
        network = ExaDisNet(cell, nodes, segs)
        network.write_data(os.path.join(out_dir, "init_config.data"))

        obstacle_data = np.column_stack((centers, radii))
        np.savetxt(
            os.path.join(out_dir, "obstacles.data"),
            obstacle_data,
            fmt="%.10e",
            header="cx cy cz radius (units of b)",
        )

        print("Orowan initial configuration written to:", out_dir)
        print("cell: %.1f um; FR mobile arm: %.1f um" %
              (LBOX_M * 1.0e6, FR_LENGTH_M * 1.0e6))
        print("nodes: %d; segments: %d; max initial segment: %.2f nm" %
              (len(nodes), len(segs), MAXSEG_B * BURGMAG * 1.0e9))
        print("obstacles: %d; radius: %.1f nm" %
              (len(radii), OBSTACLE_RADIUS_M * 1.0e9))
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
