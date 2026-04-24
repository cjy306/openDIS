import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]
np.set_printoptions(threshold=20, edgeitems=5)

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')


def run_simulation(net, state, output_dir, restart=None,):
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state,force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='strain_rate',
        erate=1e3,
        edir=np.array([0., 0., 1.]),
        max_strain=0.01,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=100,
        write_dir=output_dir,
        restart=restart)
    sim.run(net, state)


def main():
    pyexadis.initialize()

    state = {
        "crystal":   'fcc',
        "burgmag":   0.2556e-9,
        "mu":        48e9,
        "nu":        0.324,
        "a":         2.0,
        "maxseg":    200,
        "minseg":    50,
        "rtol":      0.5,
        "rann":      1.0,
        "nextdt":    1e-9,
        "maxdt":     1e-6,
    }

    output_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_Cu_fcc_pure')
    restart_freq = 100
    burgmag      = state["burgmag"]
    Lbox_m       = 5e-6
    Lbox         = int(round(Lbox_m / burgmag))

    restart_id = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if restart_id is None:
            G = ExaDisNet()
            G.read_paradis(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'output_Cu_fcc', 'config.0.data'   # ← 直接用有夹杂版生成的初始构型
            ))
            net     = DisNetManager(G)
            restart = None
        else:
            # 从 restart 文件继续
            restart_file = os.path.join(output_dir, f'restart.{restart_id}.exadis')
            net, restart = read_restart(state=state, restart_file=restart_file)

        run_simulation(net, state, output_dir,restart=restart)

    except Exception as e:
        import traceback
        print(f"模拟失败: {e}")
        traceback.print_exc()
    finally:
        try:
            pyexadis.finalize()
        except:
            pass


if __name__ == "__main__":
    main()