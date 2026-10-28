"""Zero-stress relaxation followed by [112] tension at 1e3/s to 1%."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
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
    force = CalForce(force_mode="SUBCYCLING_MODEL", state=state, Ngrid=128, cell=net.get_disnet(ExaDisNet).cell)
    # PHYS-APPROX: pure-Cu drag and velocity cap; ceiling = no CuNiSi/293K fit;
    # upgrade = alloy mobility and velocity-law calibration.
    mobility = MobilityLaw(mobility_law="FCC_0", state=state, Medge=62500., Mscrew=62500., vmax=1557.9)
    timeint = TimeIntegration(integrator="Subcycling", rgroups=[], state=state, force=force, mobility=mobility)
    collision = Collision(collision_mode="Orowan", state=state)
    topology = Topology(topology_mode="TopologyParallel", state=state, force=force, mobility=mobility)
    remesh = Remesh(remesh_rule="LengthBased", state=state, remove_small_loops=False)
    return dict(calforce=force, mobility=mobility, timeint=timeint, collision=collision,
                topology=topology, remesh=remesh, cross_slip=None)


def tensor_norm(v):
    return float(np.sqrt(np.dot(v[:3],v[:3])+2*np.dot(v[3:],v[3:])))


def relax(net, folder, max_time, max_steps):
    from pyexadis_base import ExaDisNet, SimulateNetwork
    from pyexadis_utils import dislocation_density
    if max_time < 1e-8 or max_steps < 1:
        raise ValueError("Relaxation limits must allow a full 10 ns window")
    folder.mkdir(parents=True, exist_ok=True)
    current = dict(state)
    sim = SimulateNetwork(state=current, **make_modules(net,current), loading_mode="stress",
        applied_stress=np.zeros(6), burgmag=state["burgmag"], print_freq=None, write_freq=None)
    sim.t0, current["time"] = time.perf_counter(), 0.
    elapsed, activity, increment, converged = 0., 0., np.zeros(6), False
    net.get_disnet(ExaDisNet).write_data(str(folder / "config.0.data"))
    with (folder / "relax.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step","time_s","dt_s","density_m-2","window_net_norm","window_activity"])
        writer.writerow([0,0,0,dislocation_density(net,state["burgmag"]),"",""])
        for step in range(1,max_steps+1):
            current["istep"] = step
            sim.step(net,current)
            elapsed += current["dt"]
            increment += current["dEp"]
            activity += tensor_norm(current["dEp"])
            # PHYS-APPROX: whole-step windows may exceed 10ns by one step;
            # ceiling = longer window; upgrade = exact integration window boundaries.
            complete = elapsed >= 1e-8*(1-1e-12)
            norm = tensor_norm(increment)
            converged = complete and norm < 1e-7
            writer.writerow([step,current["time"],current["dt"],dislocation_density(net,state["burgmag"]),
                             norm if complete else "",activity if complete else ""])
            if complete:
                print("relax time:",current["time"],"window norm:",norm,flush=True)
                net.get_disnet(ExaDisNet).write_data(str(folder / f"config.{step}.data"))
                stream.flush()
                elapsed, activity, increment = 0., 0., np.zeros(6)
            if converged or current["time"] >= max_time:
                break
    net.get_disnet(ExaDisNet).write_data(str(folder / "relaxed_config.data"))
    (folder / "summary.json").write_text(json.dumps(dict(converged=bool(converged),
        time_s=current["time"],steps=step,window_s=1e-8,tolerance=1e-7),indent=2),encoding="utf-8")
    if not converged:
        raise RuntimeError("Relaxation did not converge; loading was not started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="init")
    parser.add_argument("--out", default="output")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--relax-max-time", type=float, default=1e-6)
    parser.add_argument("--relax-max-steps", type=int, default=1000000)
    args = parser.parse_args()
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    if not hasattr(pyexadis.ExaDisNet,"load_obstacles_mixed"):
        raise RuntimeError("Apply core_patch/mixed_obstacles.patch and rebuild pyexadis")
    if args.check_runtime:
        print(pyexadis.__file__)
        return
    init, out = BASE_DIR/args.init, BASE_DIR/args.out
    relax_out = out.with_name(out.name+"_relax")
    if any(p.exists() and any(p.iterdir()) for p in (out,relax_out)):
        raise FileExistsError("Output exists; select a new --out")
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
        relax(net,relax_out,args.relax_max_time,args.relax_max_steps)
        # Keep this network: .data reload loses breakaway sphere_id.
        out.mkdir(parents=True, exist_ok=True)
        network.write_data(str(out/"loading_start.data"))
        loading_state = dict(state)
        edir = Rorient @ (np.array([1.,1.,2.])/np.sqrt(6))
        sim = SimulateNetworkPerf(**make_modules(net,loading_state),state=loading_state,
            burgmag=state["burgmag"],loading_mode="strain_rate",applied_stress=np.zeros(6),
            erate=1e3,edir=edir,max_strain=0.01,rotation=False,
            print_freq=1,write_freq=500,write_dir=str(out),
            out_props=["step","strain","stress","density","time","dt","Nnodes","Nsegs"])
        (out/"run_settings.json").write_text(json.dumps(dict(state=state,erate=1e3,
            edir=edir.tolist(),target_strain=0.01),indent=2),encoding="utf-8")
        final = sim.run(net,loading_state)
        print("Final strain:",final["strain"])
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
