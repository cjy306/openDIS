"""Convert Orowan snapshots and hard-sphere obstacle markers to VTK."""

import glob
import os
import re
import sys

import numpy as np


def add_pyexadis_paths(base_dir):
    for path in (
        os.path.join(base_dir, "..", "python"),
        os.path.join(base_dir, "..", "lib"),
        os.path.join(base_dir, "..", "core", "pydis", "python"),
        os.path.join(base_dir, "..", "core", "exadis", "python"),
    ):
        path = os.path.abspath(path)
        if path not in sys.path:
            sys.path.append(path)


def step_number(path):
    match = re.search(r"config\.(\d+)\.data$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def write_obstacles_vtk(filepath, output_dir):
    obstacles = np.loadtxt(filepath)
    if obstacles.ndim == 1:
        obstacles = obstacles.reshape(1, -1)
    if obstacles.shape[1] != 4:
        raise ValueError("obstacles.data must contain cx cy cz radius")

    vtk_file = os.path.join(output_dir, "obstacles.vtk")
    with open(vtk_file, "w", encoding="ascii") as stream:
        stream.write("# vtk DataFile Version 3.0\n")
        stream.write("Orowan hard-sphere obstacles\n")
        stream.write("ASCII\nDATASET POLYDATA\n")
        stream.write("POINTS %d float\n" % len(obstacles))
        for x, y, z, _ in obstacles:
            stream.write("%.8e %.8e %.8e\n" % (x, y, z))
        stream.write("VERTICES %d %d\n" % (len(obstacles), 2 * len(obstacles)))
        for index in range(len(obstacles)):
            stream.write("1 %d\n" % index)
        stream.write("POINT_DATA %d\n" % len(obstacles))
        stream.write("SCALARS radius float 1\nLOOKUP_TABLE default\n")
        for radius in obstacles[:, 3]:
            stream.write("%.8e\n" % radius)
    print("Obstacle markers written:", vtk_file)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    add_pyexadis_paths(base_dir)

    import pyexadis
    from pyexadis_utils import read_paradis, write_vtk

    input_dir = os.path.join(base_dir, "output_orowan")
    output_dir = os.path.join(base_dir, "vtk_orowan")
    data_files = sorted(
        glob.glob(os.path.join(input_dir, "config.*.data")),
        key=step_number,
    )
    if not data_files:
        raise FileNotFoundError("No config.*.data snapshots found in %s" % input_dir)

    os.makedirs(output_dir, exist_ok=True)
    pyexadis.initialize()
    try:
        for index, data_file in enumerate(data_files, start=1):
            vtk_file = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(data_file))[0] + ".vtk",
            )
            write_vtk(
                read_paradis(data_file),
                vtk_file,
                crystal="BCC",
                verbose=False,
            )
            print("[%d/%d] %s" % (index, len(data_files), vtk_file))
    finally:
        pyexadis.finalize()

    write_obstacles_vtk(
        os.path.join(base_dir, "init_orowan", "obstacles.data"),
        output_dir,
    )
    print("VTK conversion complete:", output_dir)
    print("ParaView obstacles: Glyph -> Sphere -> Scale Array=radius -> Scale Factor=2")


if __name__ == "__main__":
    main()
