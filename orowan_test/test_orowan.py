"""Run the single-FR, staggered hard-sphere Orowan geometry case.

Run ``generate_orowan.py`` first to create ``init_orowan/``.
"""

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


STATE = {
    "crystal": "bcc",
    "burgmag": 0.248e-9,
    "mu": 81.0e9,
    "nu": 0.3,
    "a": 2.0,
    "maxseg": 40.0,
    "minseg": 10.0,
    "rtol": 0.5,
    "rann": 1.0,
    "nextdt": 1.0e-10,
    "maxdt": 1.0e-8,
    "use_glide_planes": 1,
    "num_bcc_plane_families": 1,
}


def load_case():
    init_dir = os.path.join(BASE_DIR, "init_orowan")
    network = ExaDisNet()
    network.read_paradis(os.path.join(init_dir, "init_config.data"))

    obstacles = np.loadtxt(os.path.join(init_dir, "obstacles.data"))
    if obstacles.ndim == 1:
        obstacles = obstacles.reshape(1, -1)
    if obstacles.shape[1] != 4:
        raise ValueError("obstacles.data must contain cx cy cz radius")

    network.load_obstacles(
        [list(center) for center in obstacles[:, :3]],
        list(obstacles[:, 3]),
        type=0,
    )
    return network, len(obstacles)


def main():
    pyexadis.initialize()
    try:
        output_dir = os.path.join(BASE_DIR, "output_orowan")
        os.makedirs(output_dir, exist_ok=True)
        network, obstacle_count = load_case()
        net = DisNetManager(network)

        calforce = CalForce(
            force_mode="SUBCYCLING_MODEL",
            state=STATE,
            Ngrid=64,
            cell=network.cell,
        )
        mobility = MobilityLaw(
            mobility_law="BCC_0B_OrowanGeometry",
            state=STATE,
            Medge=15000.0,
            Mscrew=3000.0,
            Mclimb=100.0,
            vmax=30000.0,
        )
        timeint = TimeIntegration(
            integrator="SubcyclingOrowanGeometry",
            rgroups=[],
            state=STATE,
            force=calforce,
            mobility=mobility,
        )
        collision = Collision(collision_mode="OrowanGeometry", state=STATE)
        topology = Topology(
            topology_mode="TopologyParallel",
            state=STATE,
            force=calforce,
            mobility=mobility,
        )
        remesh = Remesh(remesh_rule="OrowanGeometry", state=STATE)

        # PHYS-APPROX: cross-slip is disabled to isolate single-plane Orowan
        # contact; ceiling = this run cannot represent cross-slip-assisted
        # escape; upgrade = add ForceBasedParallel only after the geometric
        # contact and release lifecycle has passed the single-plane test.
        simulation = SimulateNetworkPerf(
            calforce=calforce,
            mobility=mobility,
            timeint=timeint,
            collision=collision,
            topology=topology,
            remesh=remesh,
            cross_slip=None,
            loading_mode="strain_rate",
            erate=1.0e3,
            edir=np.array([0.0, 1.0, 0.0]),  # [010] tension drives the FR arm toward +b
            max_strain=0.003,
            burgmag=STATE["burgmag"],
            state=STATE,
            print_freq=1,
            write_freq=1,
            write_dir=output_dir,
        )

        print("[orowan] obstacles: %d (type=0 hard spheres)" % obstacle_count)
        print("[orowan] strain rate: 1.000e+03 1/s; "
              "axis: [0.0, 1.0, 0.0]; max strain: 0.0050")
        print("[orowan] modules: BCC_0B_OrowanGeometry / "
              "SubcyclingOrowanGeometry / OrowanGeometry collision+remesh")
        simulation.run(net, STATE)
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
