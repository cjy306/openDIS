"""Unrelaxed initial network under [112] tension at 1e3/s to 0.1% (speed screening)."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("OPENDIS_ROOT", BASE_DIR.parent.parent))
for part in ("python", "lib", "core/pydis/python", "core/exadis/python"):
    sys.path.append(str(REPO_ROOT / part))
Rorient = np.array([[1., -1., 0.], [1., 1., -2.], [1., 1., 1.]])
Rorient /= np.linalg.norm(Rorient, axis=1)[:, None]
state = {
    "crystal": "fcc", "Rorient": Rorient.tolist(),
    "burgmag": 0.2556e-9, "mu": 48e9, "nu": .324, "a": 4.,
    # PHYS-APPROX: trial mesh/reaction distance; ceiling = unverified convergence;
    # upgrade = mesh and reaction-distance sensitivity checks.
    "maxseg": 20., "minseg": 5., "rtol": .25, "rann": .5,
    "nextdt": 1e-12, "maxdt": 1e-9,
    "use_glide_planes": 1, "enforce_glide_planes": 1,
}


def make_modules(net, state):
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, ExaDisNet
    force = CalForce(force_mode="SUBCYCLING_MODEL", state=state, Ngrid=64, cell=net.get_disnet(ExaDisNet).cell)
    # PHYS-APPROX: pure-Cu drag and velocity cap; ceiling = no CuNiSi/293K fit;
    # upgrade = alloy mobility and velocity-law calibration.
    mobility = MobilityLaw(mobility_law="FCC_0", state=state, Medge=62500., Mscrew=62500., vmax=1557.9)
    timeint = TimeIntegration(integrator="Subcycling", rgroups=[], state=state, force=force, mobility=mobility)
    collision = Collision(collision_mode="Orowan", state=state)
    topology = Topology(topology_mode="TopologyParallel", state=state, force=force, mobility=mobility)
    remesh = Remesh(remesh_rule="LengthBased", state=state, remove_small_loops=False)
    return dict(calforce=force, mobility=mobility, timeint=timeint, collision=collision,
                topology=topology, remesh=remesh, cross_slip=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="init")
    parser.add_argument("--out", default="output")
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args()
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    if not hasattr(pyexadis.ExaDisNet,"load_obstacles_mixed"):
        raise RuntimeError("Apply core_patch/mixed_obstacles.patch and rebuild pyexadis")
    if args.check_runtime:
        print(pyexadis.__file__)
        return
    init, out = BASE_DIR/args.init, BASE_DIR/args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output exists; select a new --out")
    if (init/"particles_small.npy").exists():
        particles = np.vstack([np.load(init/"particles_small.npy"), np.load(init/"particles_large.npy")])
        twins = np.load(init/"twin_planes.npy")
    else:  # Existing inputs from earlier runs.
        with np.load(init/"obstacles.npz") as data:
            particles, twins = data["particles"].copy(), data["twins"].copy()
    meta = json.loads((init/"metadata.json").read_text(encoding="utf-8"))
    if not np.isclose(meta["burgmag_m"],state["burgmag"],rtol=1e-12,atol=0):
        raise ValueError("Generator and state Burgers magnitudes differ")
    pyexadis.initialize()
    try:
        network = ExaDisNet()
        network.read_paradis(str(init/"init_config.data"))
        network.net.load_obstacles_mixed(particles[:,:3].tolist(),particles[:,3].tolist(),
                                        particles[:,4].astype(int).tolist(),particles[:,5].tolist())
        if len(twins):
            network.net.load_twin_planes(twins[:,:3].tolist(),twins[:,3:].tolist())
        net = DisNetManager(network)
        # PHYS-APPROX: load unrelaxed seeds; ceiling = initial transients can affect yield;
        # upgrade = compare with equilibrated initial networks.
        out.mkdir(parents=True, exist_ok=True)
        network.write_data(str(out/"loading_start.data"))
        loading_state = dict(state)
        edir = Rorient @ (np.array([1.,1.,2.])/np.sqrt(6))
        sim = SimulateNetworkPerf(**make_modules(net,loading_state),state=loading_state,
            burgmag=state["burgmag"],loading_mode="strain_rate",applied_stress=np.zeros(6),
            erate=1e3,edir=edir,max_strain=0.001,rotation=False,
            print_freq=1,write_freq=1000,write_dir=str(out),
            out_props=["step","strain","stress","density","time","dt","Nnodes","Nsegs"])
        (out/"run_settings.json").write_text(json.dumps(dict(state=state,erate=1e3,
            edir=edir.tolist(),target_strain=0.001,relaxation=False),indent=2),encoding="utf-8")
        final = sim.run(net,loading_state)
        print("Final strain:",final["strain"])
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
