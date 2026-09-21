"""Zero-stress single-loop run with the Cr coherency-stress field."""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for relative_path in ("core/exadis/python", "core/pydis/python", "python", "lib"):
    path = os.path.join(REPO_ROOT, relative_path)
    if path not in sys.path:
        sys.path.append(path)

import pyexadis
from pyexadis_base import (
    CalForce,
    Collision,
    DisNetManager,
    ExaDisNet,
    MobilityLaw,
    Remesh,
    SimulateNetworkPerf,
    TimeIntegration,
    Topology,
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_stress(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"run generate_coherency_case.py first: {path}")
    with np.load(path, allow_pickle=False) as data:
        stress = np.asarray(data["stress"], dtype=np.float64, order="C")
        component_order = tuple(str(value) for value in data["component_order"])
        grid_shape = tuple(int(value) for value in data["grid_shape"])
        box_m = np.asarray(data["box_m"], dtype=float)
        correlation_length_m = float(data["correlation_length_m"])
    if stress.shape != (64, 64, 64, 6) or grid_shape != (64, 64, 64):
        raise ValueError(f"unexpected coherency grid shape: {stress.shape}")
    if component_order != ("xx", "yy", "zz", "yz", "xz", "xy"):
        raise ValueError(f"unexpected stress component order: {component_order}")
    if not np.allclose(box_m, [300e-9, 300e-9, 300e-9]):
        raise ValueError(f"stress field box does not match the DDD cell: {box_m}")
    if not np.isfinite(stress).all():
        raise ValueError("coherency stress contains non-finite values")
    if 10.0 * 0.248e-9 >= correlation_length_m:
        raise ValueError("maxseg must remain smaller than the field correlation length")
    return stress


def main() -> None:
    init_file = os.path.join(SCRIPT_DIR, "init_coherency", "init_config.data")
    if not os.path.isfile(init_file):
        raise FileNotFoundError(f"run generate_coherency_case.py first: {init_file}")
    stress = _load_stress(
        os.path.join(SCRIPT_DIR, "init_coherency", "coherency_stress.npz")
    )
    output_dir = os.path.join(SCRIPT_DIR, "output_with_coherency")
    os.makedirs(output_dir, exist_ok=True)

    state = {
        "crystal": "bcc", "burgmag": 0.248e-9, "mu": 81e9, "nu": 0.3,
        "a": 3.0, "maxseg": 10.0, "minseg": 3.0,
        "rtol": 0.75, "rann": 1.5,
        "nextdt": 1e-12, "maxdt": 1e-9,
        "use_glide_planes": 1, "num_bcc_plane_families": 2,
    }

    pyexadis.initialize()
    try:
        network = ExaDisNet()
        network.read_paradis(init_file)
        net = DisNetManager(network)
        calforce = CalForce(
            force_mode="SUBCYCLING_COHERENCY_MODEL", state=state,
            Ngrid=64, cell=network.cell, coherency_stress=stress,
        )
        mobility = MobilityLaw(
            mobility_law="BCC_0B", state=state,
            Medge=15000.0, Mscrew=3000.0, Mclimb=100.0, vmax=30000.0,
        )
        timeint = TimeIntegration(
            integrator="Subcycling", rgroups=[], state=state,
            force=calforce, mobility=mobility,
        )
        collision = Collision(collision_mode="Retroactive", state=state)
        topology = Topology(
            topology_mode="TopologyParallel", state=state,
            force=calforce, mobility=mobility,
        )
        remesh = Remesh(remesh_rule="LengthBased", state=state)

        print(f"[with coherency] initial SHA-256: {_sha256(init_file)}")
        simulation = SimulateNetworkPerf(
            calforce=calforce,
            mobility=mobility,
            timeint=timeint,
            collision=collision,
            topology=topology,
            remesh=remesh,
            cross_slip=None,
            loading_mode="stress",
            applied_stress=np.zeros(6),
            max_step=10000,
            burgmag=state["burgmag"],
            state=state,
            print_freq=1,
            write_freq=100,
            write_dir=output_dir,
            out_props=["step", "time", "dt", "density", "Nnodes"],
        )
        simulation.run(net, state)
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
