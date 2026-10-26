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
import os
from pathlib import Path
import sys

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
setup()

import pyexadis
from pyexadis_base import (CalForce, Collision, DisNetManager, ExaDisNet,
                           MobilityLaw, Remesh, SimulateNetworkPerf,
                           TimeIntegration, Topology, read_restart)


CASE = "uniform12_1um_rho1e13"
SEED = 12345
DEFAULT_INIT = "init"

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


def make_modules(net, state):
    calforce = CalForce(force_mode="SUBCYCLING_MODEL", state=state, Ngrid=128,
                        cell=net.get_disnet(ExaDisNet).cell)
    mobility = MobilityLaw(mobility_law="FCC_0", state=state,
                           Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint = TimeIntegration(integrator="Subcycling", rgroups=[], state=state,
                              force=calforce, mobility=mobility)
    collision = Collision(collision_mode="Retroactive", state=state)
    topology = Topology(topology_mode="TopologyParallel", state=state,
                        force=calforce, mobility=mobility)
    remesh = Remesh(remesh_rule="LengthBased", state=state)
    return dict(calforce=calforce, mobility=mobility, timeint=timeint,
                collision=collision, topology=topology, remesh=remesh, cross_slip=None)


def main():
    parser = argparse.ArgumentParser(description="Load the uniform-12-slip-system FCC Cu network")
    parser.add_argument("--init", default=DEFAULT_INIT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--restart", type=int)
    args = parser.parse_args()

    init_dir = Path(args.init) if os.path.isabs(args.init) else BASE_DIR / args.init
    if args.out:
        output_dir = Path(args.out) if os.path.isabs(args.out) else BASE_DIR / args.out
    else:
        output_dir = BASE_DIR / "output"
    if args.restart is None and (output_dir / 'stress_strain_dens.dat').exists():
        raise FileExistsError('Loading output exists; use --restart or a new --out')
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
        simulation = SimulateNetworkPerf(
            **make_modules(net, state),
            loading_mode="strain_rate", applied_stress=applied_stress,
            erate=1e3, edir=np.array([0.0, 0.0, 1.0]), max_strain=0.005,
            burgmag=state["burgmag"], state=state,
            print_freq=1, write_freq=1000,
            write_dir=str(output_dir), restart=restart)
        simulation.run(net, state)
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
