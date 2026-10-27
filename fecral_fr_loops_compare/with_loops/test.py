"""Relax the complete case, then load to 0.5% total strain."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from runtime import setup
from relaxation import run_relaxation


DEFAULT_INIT = "init"

state = {
    "crystal": "bcc",
    "burgmag": 0.248e-9,
    "mu": 81e9,
    "nu": 0.3,
    "a": 2.0,
    # PHYS-APPROX: trial mesh for 20 nm loops; ceiling = unverified convergence;
    # upgrade = repeat with finer mesh and compare loop survival and stress.
    "maxseg": 20.0,
    "minseg": 5.0,
    "rtol": 0.5,
    "rann": 1.0,
    "nextdt": 1e-12,
    "maxdt": 1e-9,
    "use_glide_planes": 1,
    "enforce_glide_planes": 1,
    "num_bcc_plane_families": 1,
}

def make_modules(net, state):
    from pyexadis_base import (CalForce, MobilityLaw, TimeIntegration,
                               Collision, Topology, Remesh, ExaDisNet)
    calforce = CalForce(force_mode="SUBCYCLING_MODEL", state=state, Ngrid=128,
                     cell=net.get_disnet(ExaDisNet).cell)
    # PHYS-APPROX: default BCC_NL fits pure Fe, not FeCrAl;
    # ceiling = mechanism comparison; upgrade = alloy-calibrated mobility.
    mobility = MobilityLaw(mobility_law="BCC_NL", state=state, tempK=300.0)
    timeint = TimeIntegration(integrator="Subcycling", rgroups=[], state=state,
                              force=calforce, mobility=mobility)
    collision = Collision(collision_mode="Retroactive", state=state)
    topology = Topology(topology_mode="TopologyParallel", state=state,
                        force=calforce, mobility=mobility)
    remesh = Remesh(remesh_rule="LengthBased", state=state, remove_small_loops=False)
    return dict(calforce=calforce, mobility=mobility, timeint=timeint,
                collision=collision, topology=topology, remesh=remesh, cross_slip=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default=DEFAULT_INIT)
    parser.add_argument("--out", default="output")
    parser.add_argument("--restart", type=int, help="Restart an existing loading stage")
    parser.add_argument("--relax-max-time", type=float, default=1e-6)
    parser.add_argument("--relax-max-steps", type=int, default=1000000)
    args = parser.parse_args()
    setup()
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart

    out = BASE_DIR / args.out
    init_dir = BASE_DIR / args.init
    init = init_dir / "init_config.data"
    if args.restart is None and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; use --restart or --out")
    pyexadis.initialize()
    try:
        if args.restart is None:
            metadata = json.loads((init_dir/"metadata.json").read_text(encoding="utf-8"))
            if not np.isclose(metadata["burgmag_m"], state["burgmag"], rtol=1e-12, atol=0):
                raise ValueError("Generator and loading Burgers magnitudes differ")
            network = ExaDisNet()
            network.read_paradis(str(init))
            net = DisNetManager(network)
            relaxed = run_relaxation(net, state, make_modules,
                out.parent/(out.name+"_relax"), init,
                max_time=args.relax_max_time, max_steps=args.relax_max_steps)
            # Fresh driver and state reset loading reference, NOT relaxed geometry.
            network = ExaDisNet()
            network.read_paradis(str(relaxed))
            net = DisNetManager(network)
            restart = None
        else:
            net, restart = read_restart(state=state,
                restart_file=str(out/f"restart.{args.restart}.exadis"))
        out.mkdir(parents=True, exist_ok=True)
        if args.restart is None:
            net.get_disnet(ExaDisNet).write_data(str(out/"loading_start.data"))
        sim = SimulateNetworkPerf(
            **make_modules(net,state), state=state, burgmag=state["burgmag"],
            loading_mode="strain_rate", applied_stress=np.zeros(6),
            erate=1e3, edir=np.array([0.0,0.0,1.0]), max_strain=0.005,
            print_freq=1, write_freq=500, write_dir=str(out), restart=restart,
            out_props=["step","strain","stress","density","time","dt","Nnodes","Nsegs"])
        final_state = sim.run(net,state)
        (out/"loading_summary.json").write_text(json.dumps(
            {"final_total_strain": float(final_state["strain"]),
             "target_total_strain": 0.005}, indent=2), encoding="utf-8")
        net.get_disnet(ExaDisNet).write_data(str(out/"loading_final.data"))
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
