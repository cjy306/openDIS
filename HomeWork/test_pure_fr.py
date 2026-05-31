"""对照组：纯 FR 源弓出，无孪晶面、无杂质"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e

from pyexadis_utils import insert_frank_read_src

Rorient = [
    [ 1/np.sqrt(2), -1/np.sqrt(2),  0           ],
    [ 1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
    [ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],
]

state = {
    "crystal": 'fcc',
    "Rorient": Rorient,
    "burgmag": 0.2556e-9,
    "mu":      48e9,
    "nu":      0.324,
    "a":       4.0,
    "maxseg":  200,
    "minseg":  50,
    "rtol":    1.0,
    "rann":    2.0,
    "nextdt":  1e-9,
    "maxdt":   1e-8,
}


def main():
    pyexadis.initialize()

    burgmag = state["burgmag"]
    Lbox_m = 5e-6
    Lbox = int(round(Lbox_m / burgmag))
    cell = pyexadis.Cell(Lbox)

    R = np.array(Rorient)

    # 单个 FR 源：(111)[1-10] 滑移系，旋转到新坐标系
    burg  = R @ (np.array([1, -1, 0]) / np.sqrt(2))
    plane = R @ (np.array([1,  1, 1]) / np.sqrt(3))

    length_m = 1.0e-6  # 1 μm
    length_b = length_m / burgmag
    center_b = np.array([Lbox/2, Lbox/2, Lbox/2])  # 盒子中心
    numnodes = max(3, int(round(length_b / 100.0)))

    nodes, segs = [], []
    nodes, segs = insert_frank_read_src(
        cell, nodes, segs, burg, plane, length_b, center_b, numnodes=numnodes)

    G = ExaDisNet(cell, nodes, segs)
    net = DisNetManager(G)
    exadis_net = net.get_disnet(ExaDisNet)

    # 不加载杂质、不加载孪晶面

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_pure_fr')
    os.makedirs(output_dir, exist_ok=True)

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=exadis_net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=50.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 5.0, 30.0, 100.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        erate=1e4,
        edir=np.array([0.0, -2.0/np.sqrt(6), 1.0/np.sqrt(3)]),
        max_strain=0.01,
        burgmag=burgmag,
        state=state,
        print_freq=1,
        write_freq=1,
        write_dir=output_dir,
    )
    sim.run(net, state)
    pyexadis.finalize()


if __name__ == "__main__":
    main()
