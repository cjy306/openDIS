"""FCC Cu 3 um FR repeats; shared generation and simulation settings."""
from pathlib import Path
import csv
import math
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
B = 2.55e-10
BOX = 5e-6
LENGTH = 3e-6
COUNT = 43
HIGH = (0, 1, 3, 4, 6, 7, 9, 10)
ZERO = (2, 5, 8, 11)
BURGERS = ((0,1,-1),(1,0,-1),(1,-1,0),
           (0,1,-1),(1,0,1),(1,1,0),
           (0,1,1),(1,0,-1),(1,1,0),
           (0,1,1),(1,0,1),(1,-1,0))
NORMALS = ((1,1,1),)*3 + ((-1,1,1),)*3 + ((1,-1,1),)*3 + ((1,1,-1),)*3

def setup():
    for path in (ROOT/'python', ROOT/'lib', ROOT/'core/pydis/python',
                 ROOT/'core/exadis/python'):
        if str(path) not in sys.path:
            sys.path.append(str(path))

def make_layout(seed):
    rng = random.Random(seed)
    positions = [tuple(rng.random() for _ in range(3)) for _ in range(COUNT)]
    order = list(range(COUNT))
    random.Random(seed ^ 0x5A17A5EED).shuffle(order)
    systems = [-1]*COUNT
    # PHYS-APPROX: integer count gives 32/43 high-Schmid and rho=1.032e12;
    # ceiling = not exact A75/1.024e12; upgrade = jointly redesign box/count.
    for rank, index in enumerate(order):
        systems[index] = HIGH[rank % 8] if rank < 32 else ZERO[rank % 4]
    rng = random.Random(seed ^ 0x6C8E9CF5)
    angles = [rng.uniform(0, 180) for _ in range(COUNT)]
    return positions, systems, angles

def generate(seed, directory):
    setup()
    import numpy as np
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager
    from pyexadis_utils import insert_frank_read_src, write_vtk
    output = Path(directory)/f'init_A75_3um_seed{seed}'
    pyexadis.initialize()
    try:
        positions, systems, angles = make_layout(seed)
        cell = pyexadis.Cell(h=BOX/B*np.eye(3), is_periodic=[True]*3)
        centers = np.asarray(cell.origin)+np.asarray(positions)@np.asarray(cell.h).T
        burgers = np.asarray(BURGERS, dtype=float)
        burgers /= np.linalg.norm(burgers, axis=1)[:,None]
        normals = np.asarray(NORMALS, dtype=float)
        normals /= np.linalg.norm(normals, axis=1)[:,None]
        nodes, segs = [], []
        for center, system, angle in zip(centers, systems, angles):
            nodes, segs = insert_frank_read_src(
                cell, nodes, segs, burgers[system], normals[system], LENGTH/B,
                center, theta=angle, numnodes=max(3, math.ceil(LENGTH/B/200)+1))
        net = ExaDisNet(cell, nodes, segs)
        output.mkdir(parents=True, exist_ok=True)
        net.write_data(str(output/'init_config.data'))
        write_vtk(DisNetManager(net), str(output/'init_config_labeled.vtk'),
                  crystal='FCC', verbose=False)
        with (output/'source_layout.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('source_id','fx','fy','fz','slip_system_id','group','theta_deg'))
            for i, (p, s, a) in enumerate(zip(positions, systems, angles)):
                writer.writerow((i,*p,s,'high_schmid' if s in HIGH else 'zero_schmid',a))
        print(f'seed={seed}, sources={COUNT}, rho={COUNT*LENGTH/BOX**3:.6e}, output={output}')
    finally:
        pyexadis.finalize()

def simulate(seed, directory):
    import argparse
    parser = argparse.ArgumentParser(description='Run FCC Cu 3 um FR simulation')
    parser.add_argument('--restart', type=int)
    args = parser.parse_args()
    setup()
    import numpy as np
    import pyexadis
    from pyexadis_base import (ExaDisNet, DisNetManager, CalForce, MobilityLaw,
        TimeIntegration, Collision, Topology, Remesh, SimulateNetworkPerf, read_restart)
    state = dict(crystal='fcc', burgmag=B, mu=54.6e9, nu=0.324, a=4.0,
                 maxseg=200.0, minseg=40.0, rtol=1.0, rann=2.0,
                 nextdt=1e-10, maxdt=1e-9)
    directory = Path(directory)
    output = directory/f'output_A75_3um_seed{seed}'
    output.mkdir(exist_ok=True)
    pyexadis.initialize()
    try:
        if args.restart is not None:
            net, restart = read_restart(state=state,
                restart_file=str(output/f'restart.{args.restart}.exadis'))
        else:
            network = ExaDisNet()
            network.read_paradis(str(directory/f'init_A75_3um_seed{seed}'/'init_config.data'))
            net, restart = DisNetManager(network), None
        force = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64,
                         cell=net.get_disnet(ExaDisNet).cell)
        mobility = MobilityLaw(mobility_law='FCC_0', state=state,
                               Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
        timeint = TimeIntegration(integrator='Subcycling', rgroups=[], state=state,
                                  force=force, mobility=mobility)
        collision = Collision(collision_mode='Retroactive', state=state)
        topology = Topology(topology_mode='TopologyParallel', state=state,
                            force=force, mobility=mobility)
        remesh = Remesh(remesh_rule='LengthBased', state=state)
        sim = SimulateNetworkPerf(calforce=force, mobility=mobility, timeint=timeint,
            collision=collision, topology=topology, remesh=remesh, cross_slip=None,
            loading_mode='strain_rate', applied_stress=np.zeros(6), erate=1e3,
            edir=np.array([0.,0.,1.]), max_strain=0.005, burgmag=B, state=state,
            print_freq=1, write_freq=1000, write_dir=str(output), restart=restart)
        sim.run(net,state)
    finally:
        pyexadis.finalize()
