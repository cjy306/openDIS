"""FCC Cu [001] 正式加载：A100 初始滑移系占比构型。"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
for path in (REPO_ROOT / "python", REPO_ROOT / "lib",
             REPO_ROOT / "core" / "pydis" / "python",
             REPO_ROOT / "core" / "exadis" / "python"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import pyexadis
from pyexadis_base import (CalForce, Collision, DisNetManager, ExaDisNet,
                           MobilityLaw, Remesh, SimulateNetworkPerf,
                           TimeIntegration, Topology, read_restart)


CASE = "A100"
DEFAULT_INIT = "init_A100_seed12345"

state = {
    "crystal": "fcc",
    "burgmag": 2.55e-10,
    "mu": 54.6e9,
    "nu": 0.324,
    "a": 4.0,
    "maxseg": 200.0,
    "minseg": 40.0,
    "rtol": 1.0,
    "rann": 2.0,
    "nextdt": 1e-10,
    "maxdt": 1e-9,
}


def main():
    parser = argparse.ArgumentParser(description="加载FCC Cu因素A的A100初始构型")
    parser.add_argument("--init", default=DEFAULT_INIT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--restart", type=int)
    args = parser.parse_args()

    init_dir = Path(args.init) if os.path.isabs(args.init) else BASE_DIR / args.init
    if args.out:
        output_dir = Path(args.out) if os.path.isabs(args.out) else BASE_DIR / args.out
    else:
        init_name = init_dir.name
        output_name = ("output" + init_name[4:] if init_name.startswith("init")
                       else "output_" + init_name)
        output_dir = BASE_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pyexadis.initialize()
    try:
        if args.restart is not None:
            net, restart = read_restart(
                state=state,
                restart_file=str(output_dir / f"restart.{args.restart}.exadis"))
        else:
            network = ExaDisNet()
            network.read_paradis(str(init_dir / "init_config.data"))
            net = DisNetManager(network)
            restart = None

        applied_stress = np.zeros(6)
        calforce = CalForce(force_mode="SUBCYCLING_MODEL", state=state, Ngrid=64,
                            cell=net.get_disnet(ExaDisNet).cell)
        mobility = MobilityLaw(mobility_law="FCC_0", state=state,
                               Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
        timeint = TimeIntegration(integrator="Subcycling", rgroups=[], state=state,
                                  force=calforce, mobility=mobility)
        collision = Collision(collision_mode="Retroactive", state=state)
        topology = Topology(topology_mode="TopologyParallel", state=state,
                            force=calforce, mobility=mobility)
        remesh = Remesh(remesh_rule="LengthBased", state=state)
        simulation = SimulateNetworkPerf(
            calforce=calforce, mobility=mobility, timeint=timeint,
            collision=collision, topology=topology, remesh=remesh,
            loading_mode="strain_rate", applied_stress=applied_stress,
            erate=1e3, edir=np.array([0.0, 0.0, 1.0]), max_strain=0.01,
            burgmag=state["burgmag"], state=state,
            print_freq=1, write_freq=500,
            write_dir=str(output_dir), restart=restart)
        simulation.run(net, state)
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
