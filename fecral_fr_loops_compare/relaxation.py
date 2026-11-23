"""Reused window criterion and relaxation runner from the existing FR comparison."""
from pathlib import Path
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
                   max_time=1e-6, max_steps=1000000, resume=False):
    from pyexadis_base import ExaDisNet, SimulateNetwork, DisNetManager
    if max_time < 1e-8 or not np.isfinite(max_time) or max_steps < 1:
        raise ValueError('relaxation limits must permit a complete 10 ns window')
    folder.mkdir(parents=True, exist_ok=True)
    history = folder / 'relax.csv'
    checkpoint = folder / 'relax_restart.json'
    if not resume and history.exists():
        raise FileExistsError(f'{folder}/relax.csv exists; select a new --out directory')
    state = dict(initial_state)
    initial_hash = hashlib.sha256(input_file.read_bytes()).hexdigest()
    saved = None
    step = 0
    if resume:
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved['initial_sha256'] != initial_hash or saved['parameters'] != initial_state:
            raise ValueError('Relaxation checkpoint belongs to a different input or state')
        step = saved['step']
        if not saved['window']['converged'] and (step >= max_steps or saved['time'] >= max_time):
            raise ValueError('Increase the relaxation limit to continue this checkpoint')
        if history.stat().st_size < saved['csv_bytes']:
            raise ValueError('Relaxation history is shorter than the checkpoint')
        network = ExaDisNet()
        network.read_paradis(str(folder / f'config.{step}.data'))
        net = DisNetManager(network)
        state['nextdt'] = saved['dt']
    simulation = SimulateNetwork(state=state, **modules_factory(net, state),
        loading_mode='stress', applied_stress=np.zeros(6),
        burgmag=state['burgmag'], print_freq=None, write_freq=None)
    simulation.t0 = time.perf_counter()
    state['time'] = saved['time'] if saved else 0.0
    monitor = WindowMonitor()
    cumulative = np.array(saved['cumulative_ep']) if saved else np.zeros(6)
    activity = saved['activity'] if saved else 0.0
    result = None
    last_window = saved['window'] if saved else None
    converged = bool(last_window and last_window['converged'])
    if saved:
        # Discard only rows after the last committed geometry.
        with history.open('r+b') as raw:
            raw.truncate(saved['csv_bytes'])
    else:
        net.get_disnet(ExaDisNet).write_data(str(folder / 'config.0.data'))
    with history.open('a' if saved else 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if not saved:
            writer.writerow(['step','time_s','dt_s','density_m-2','ep_xx','ep_yy','ep_zz',
                         'ep_yz','ep_xz','ep_xy','cumulative_activity','window_duration_s',
                         'window_net_norm','window_activity','converged'])
        for step in range(step+1, (step if converged else max_steps)+1):
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
                snapshot = folder / f'config.{step}.data'
                partial = snapshot.with_suffix('.tmp')
                net.get_disnet(ExaDisNet).write_data(str(partial))
                partial.replace(snapshot)
                record = dict(step=step, time=state['time'], dt=dt, window=result,
                              cumulative_ep=cumulative.tolist(), activity=activity,
                              parameters=initial_state, initial_sha256=initial_hash,
                              csv_bytes=stream.tell())
                pending = checkpoint.with_suffix('.tmp')
                pending.write_text(json.dumps(record), encoding='utf-8')
                pending.replace(checkpoint)
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
        raise RuntimeError('Relaxation did not converge; loading was NOT started. Inspect relax.csv and summary.json.')
    return final_file

