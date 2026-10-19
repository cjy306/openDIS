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


CASE = "uniform12_3um"
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
    return dict(calforce=calforce, mobility=mobility, timeint=timeint,
                collision=collision, topology=topology, remesh=remesh, cross_slip=None)


"""Full non-overlapping physical-time windows; Voigt order xx,yy,zz,yz,xz,xy."""
import numpy as np

class WindowMonitor:
    def __init__(self, window=1e-8, tolerance=1e-7):
        if not np.isfinite([window, tolerance]).all() or min(window, tolerance) <= 0:
            raise ValueError('window and tolerance must be finite and positive')
        self.window, self.tolerance = window, tolerance
        self.elapsed = 0.0
        self.net = np.zeros(6)
        self.activity = 0.0

    @staticmethod
    def norm(v):
        return float(np.sqrt(np.dot(v[:3], v[:3]) + 2*np.dot(v[3:], v[3:])))

    def add(self, dt, increment):
        v = np.asarray(increment, dtype=float)
        if not np.isfinite(dt) or dt <= 0 or v.shape != (6,) or not np.isfinite(v).all():
            raise ValueError('invalid time step or plastic strain increment')
        self.elapsed += dt
        self.net += v
        self.activity += self.norm(v)
        # PHYS-APPROX: whole-step windows can exceed 10 ns by one step; ceiling = conservative longer window; upgrade = integrator-controlled exact window boundary.
        if self.elapsed < self.window*(1-1e-12):
            return None
        # Activity is diagnostic only; convergence uses the net tensor increment.
        result = dict(duration=self.elapsed, net_norm=self.norm(self.net),
                      activity=self.activity, converged=self.norm(self.net) < self.tolerance)
        self.elapsed = 0.0
        self.net[:] = 0.0
        self.activity = 0.0
        return result

"""Zero-applied-stress relaxation using the verified Python ExaDiS driver."""
import csv
import hashlib
import json
import time
import numpy as np

def run_relaxation(net, initial_state, modules_factory, folder, input_file,
                   max_time=1e-6, max_steps=1000000):
    from pyexadis_base import ExaDisNet, SimulateNetwork
    if max_time < 1e-8 or not np.isfinite(max_time) or max_steps < 1:
        raise ValueError('relaxation limits must permit a complete 10 ns window')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'relax.csv').exists():
        raise FileExistsError(f'{folder}/relax.csv exists; select a new --out directory')
    state = dict(initial_state)
    simulation = SimulateNetwork(state=state, **modules_factory(net, state),
        loading_mode='stress', applied_stress=np.zeros(6),
        burgmag=state['burgmag'], print_freq=None, write_freq=None)
    simulation.t0 = time.perf_counter()
    state['time'] = 0.0
    monitor = WindowMonitor()
    cumulative = np.zeros(6)
    activity = 0.0
    result = None
    last_window = None
    converged = False
    net.get_disnet(ExaDisNet).write_data(str(folder / 'config.0.data'))
    with (folder / 'relax.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step','time_s','dt_s','density_m-2','ep_xx','ep_yy','ep_zz',
                         'ep_yz','ep_xz','ep_xy','cumulative_activity','window_duration_s',
                         'window_net_norm','window_activity','converged'])
        for step in range(1, max_steps+1):
            state['istep'] = step
            simulation.step(net, state)
            dt = state['dt']
            result = monitor.add(dt, state['dEp'])
            cumulative += state['dEp']
            activity += monitor.norm(state['dEp'])
            converged = result is not None and result['converged']
            writer.writerow([step,state['time'],dt,state['density'],*cumulative,activity,
                result['duration'] if result else '',result['net_norm'] if result else '',
                result['activity'] if result else '',int(converged)])
            if result is not None or step % 100 == 0:
                stream.flush()
            if result is not None:
                last_window = result
                print(f"relax: t={state['time']*1e9:.4f} ns, "
                      f"net={result['net_norm']:.4e}, activity={result['activity']:.4e}", flush=True)
                net.get_disnet(ExaDisNet).write_data(str(folder / f'config.{step}.data'))
            if converged or state['time'] >= max_time:
                break
    final_file = folder / 'relaxed_config.data'
    net.get_disnet(ExaDisNet).write_data(str(final_file))
    summary = dict(converged=converged,time_s=state['time'],steps=step,
        window_s=monitor.window,tolerance=monitor.tolerance,
        criterion='Frobenius norm of net plastic strain tensor increment in full window',
        last_completed_window=last_window,
        cumulative_ep=cumulative.tolist(),cumulative_activity=activity,
        initial_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
        relaxed_sha256=hashlib.sha256(final_file.read_bytes()).hexdigest(),
        max_time_s=max_time,max_steps=max_steps)
    (folder / 'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if not converged:
        raise RuntimeError('Relaxation did not converge; loading was NOT started. Run plot_relax.py.')
    return final_file


def main():
    parser = argparse.ArgumentParser(description="Load the uniform-12-slip-system FCC Cu network")
    parser.add_argument("--init", default=DEFAULT_INIT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--restart", type=int)
    parser.add_argument('--relax-max-time', type=float, default=1e-6, help='seconds')
    parser.add_argument('--relax-max-steps', type=int, default=1000000)
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
            relaxed_file = run_relaxation(net, state, make_modules, output_dir.parent / (output_dir.name + '_relax'),
                init_dir / 'init_config.data', args.relax_max_time, args.relax_max_steps)
            # Fresh network and driver reset loading strain/time while preserving relaxed geometry.
            network = ExaDisNet()
            network.read_paradis(str(relaxed_file))
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
