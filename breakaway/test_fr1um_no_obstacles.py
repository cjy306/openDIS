"""5 um / 1 um FR 无障碍基准工况，默认加载到0.5%总应变。"""
import os
import sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if path not in sys.path]

import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip

state = {
    'crystal': 'bcc', 'burgmag': 0.248e-9, 'mu': 81e9, 'nu': 0.3, 'a': 3.0,
    'maxseg': 160, 'minseg': 40, 'rtol': 0.75, 'rann': 1.5,
    'nextdt': 1e-9, 'maxdt': 1e-8,
    'use_glide_planes': 1, 'num_bcc_plane_families': 2,
}


def main():
    pyexadis.initialize()
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        init_file = os.path.join(base_dir, 'init_fr1um_no_obstacles', 'init_config.data')
        output_dir = os.path.join(base_dir, 'output_fr1um_no_obstacles')
        os.makedirs(output_dir, exist_ok=True)

        network = ExaDisNet()
        network.read_paradis(init_file)
        net = DisNetManager(network)
        calforce = CalForce(force_mode='SUBCYCLING_MODEL', state=state,
                            Ngrid=64, cell=network.cell)
        mobility = MobilityLaw(mobility_law='BCC_0B', state=state,
                               Medge=15000.0, Mscrew=3000.0,
                               Mclimb=100.0, vmax=30000.0)
        timeint = TimeIntegration(integrator='Subcycling', rgroups=[], state=state,
                                  force=calforce, mobility=mobility)
        # 与点障碍组保持相同碰撞模块；本组障碍表为空。
        collision = Collision(collision_mode='Orowan', state=state)
        topology = Topology(topology_mode='TopologyParallel', state=state,
                            force=calforce, mobility=mobility)
        remesh = Remesh(remesh_rule='LengthBased', state=state)
        cross_slip = CrossSlip(cross_slip_mode='ForceBasedParallel', state=state,
                               force=calforce)

        sim = SimulateNetworkPerf(
            calforce=calforce, mobility=mobility, timeint=timeint,
            collision=collision, topology=topology, remesh=remesh,
            cross_slip=cross_slip,
            loading_mode='strain_rate', erate=1e3,
            edir=np.array([0., 0., 1.]), max_strain=0.005,
            burgmag=state['burgmag'], state=state,
            print_freq=1, write_freq=100, write_dir=output_dir,
        )
        print('[baseline] 无障碍，加载到0.5%总应变')
        sim.run(net, state)
    finally:
        pyexadis.finalize()


if __name__ == '__main__':
    main()
