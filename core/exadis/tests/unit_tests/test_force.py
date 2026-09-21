import os, sys, io
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
    from pyexadis_utils import insert_prismatic_loop
except ImportError:
    raise ImportError('Cannot import pyexadis')


class CalForceFFT:
    def __init__(self, state: dict, **kwargs) -> None:
        from pyexadis_base import get_module_arg, get_exadis_params
        params = get_exadis_params(state)
        self.Ngrid = get_module_arg('CalForceFFT', kwargs, 'Ngrid')
        if isinstance(self.Ngrid, int): self.Ngrid = 3*[self.Ngrid]
        cell = get_module_arg('CalForceFFT', kwargs, 'cell')
        fftparams = pyexadis.Force.ForceFFT.Params(Ngrid=self.Ngrid)
        self.forcefft = pyexadis.Force.ForceFFT.make(params=params, fparams=fftparams, cell=cell)
        
    def PreCompute(self, N: DisNetManager, state: dict) -> dict:
        G = N.get_disnet(ExaDisNet)
        self.forcefft.pre_compute_force(G.net)
        return state
    
    def NodeForce(self, N: DisNetManager, state: dict, pre_compute=True) -> dict:
        if pre_compute:
            self.PreCompute(N, state)

        G = N.get_disnet(ExaDisNet)
        f = self.forcefft.compute_force(G.net, applied_stress=np.zeros(6), pre_compute=pre_compute)
        
        state["nodeforces"] = np.array(f)
        state["nodeforcetags"] = N.export_data()["nodes"]["tags"]
        return state
    
    def OneNodeForce(self, N: DisNetManager, state: dict, tag, update_state=True) -> np.array:
        f = np.zeros(3)
        raise TypeError("OneNodeForce not implemented for CalForceFFT")
        return f


def _expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _coherency_network(shift_x=0.0):
    cell = pyexadis.Cell(200.0)
    burgers = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    nodes, segments = insert_prismatic_loop(
        'bcc', cell, [], [], burgers, 20.0,
        np.array([100.0 + shift_x, 100.0, 100.0]), maxseg=10.0,
    )
    return ExaDisNet(cell, nodes, segments)


def _coherency_state(applied_stress=None):
    return {
        "crystal": 'bcc', "burgmag": 2.48e-10, "mu": 81e9, "nu": 0.3,
        "a": 3.0, "maxseg": 10.0, "minseg": 3.0, "rtol": 1.0,
        "rann": 2.0, "nextdt": 1e-12, "maxdt": 1e-9,
        "applied_stress": np.zeros(6) if applied_stress is None else np.asarray(applied_stress),
    }


def _all_node_forces(force, network, state):
    manager = DisNetManager(network)
    force.PreCompute(manager, state)
    return np.asarray([
        force.OneNodeForce(manager, state, tuple(tag), update_state=False)
        for tag in network.get_tags()
    ])


def test_coherency_force(name):
    pyexadis.initialize(verbose=False)
    try:
        network = _coherency_network()
        state = _coherency_state()
        zero = np.zeros((4, 4, 4, 6), dtype=np.float64)

        if name == 'coherency_zero':
            _expect_value_error(lambda: CalForce(
                force_mode='SUBCYCLING_COHERENCY_MODEL', state=state,
                Ngrid=4, cell=network.cell, coherency_stress=np.zeros((4, 4, 4)),
            ))
            bad = zero.copy()
            bad[0, 0, 0, 0] = np.nan
            _expect_value_error(lambda: CalForce(
                force_mode='SUBCYCLING_COHERENCY_MODEL', state=state,
                Ngrid=4, cell=network.cell, coherency_stress=bad,
            ))
            legacy = CalForce(
                force_mode='SUBCYCLING_MODEL', state=state, Ngrid=4, cell=network.cell
            )
            coherency = CalForce(
                force_mode='SUBCYCLING_COHERENCY_MODEL', state=state,
                Ngrid=4, cell=network.cell, coherency_stress=zero,
            )
            reference = _all_node_forces(legacy, network, state)
            actual = _all_node_forces(coherency, network, state)
            error = np.max(np.abs(actual - reference)) / max(1.0, np.max(np.abs(reference)))

        elif name == 'coherency_uniform':
            sigma = np.array([12e6, -7e6, 5e6, 3e6, -2e6, 4e6])
            field = np.broadcast_to(sigma, (4, 4, 4, 6)).copy()
            baseline = CalForce(
                force_mode='SUBCYCLING_MODEL', state=state, Ngrid=4, cell=network.cell
            )
            field_force = CalForce(
                force_mode='SUBCYCLING_COHERENCY_MODEL', state=state,
                Ngrid=4, cell=network.cell, coherency_stress=field,
            )
            zero_force = _all_node_forces(baseline, network, state)
            added_field = _all_node_forces(field_force, network, state) - zero_force
            loaded_state = _coherency_state(sigma)
            loaded = CalForce(
                force_mode='SUBCYCLING_MODEL', state=loaded_state,
                Ngrid=4, cell=network.cell,
            )
            added_uniform = _all_node_forces(loaded, network, loaded_state) - zero_force
            error = np.max(np.abs(added_field - added_uniform)) / max(
                1.0, np.max(np.abs(added_uniform))
            )

        elif name == 'coherency_periodic':
            field = np.zeros((4, 4, 4, 6))
            field[..., 5] = np.arange(4)[:, None, None] * 1e6
            original = CalForce(
                force_mode='SUBCYCLING_COHERENCY_MODEL', state=state,
                Ngrid=4, cell=network.cell, coherency_stress=field,
            )
            shifted_network = _coherency_network(shift_x=200.0)
            shifted = CalForce(
                force_mode='SUBCYCLING_COHERENCY_MODEL', state=state,
                Ngrid=4, cell=shifted_network.cell, coherency_stress=field,
            )
            first = _all_node_forces(original, network, state)
            second = _all_node_forces(shifted, shifted_network, state)
            error = np.max(np.abs(first - second)) / max(1.0, np.max(np.abs(first)))

        elif name == 'coherency_sign':
            field = np.zeros((4, 4, 4, 6))
            field[..., 0] = 8e6
            baseline = CalForce(
                force_mode='SUBCYCLING_MODEL', state=state, Ngrid=4, cell=network.cell
            )
            positive = CalForce(
                force_mode='SUBCYCLING_COHERENCY_MODEL', state=state,
                Ngrid=4, cell=network.cell, coherency_stress=field,
            )
            negative = CalForce(
                force_mode='SUBCYCLING_COHERENCY_MODEL', state=state,
                Ngrid=4, cell=network.cell, coherency_stress=-field,
            )
            common = _all_node_forces(baseline, network, state)
            signed_sum = (
                _all_node_forces(positive, network, state) - common
                + _all_node_forces(negative, network, state) - common
            )
            error = np.max(np.abs(signed_sum)) / max(1.0, np.max(np.abs(common)))

        else:
            raise ValueError(f"Invalid coherency force test = '{name}'")

        print(f"{name} normalized_error {error:.16e}")
        if error >= 1e-8:
            raise AssertionError(f"{name} normalized error {error} exceeds 1e-8")
    finally:
        pyexadis.finalize()
        

def test_force(name='lt'):
    if name is not None and name.startswith('coherency_'):
        return test_coherency_force(name)
    
    pyexadis.initialize(verbose=False)
    
    state = {
        "crystal": 'fcc',
        "burgmag": 2.55e-10,
        "mu": 54.6e9,
        "nu": 0.324,
        "a": 6.0,
        "maxseg": 2000.0,
        "minseg": 300.0,
        "rtol": 10.0,
        "rann": 10.0,
        "nextdt": 1e-10,
        "maxdt": 1e-9,
    }
    
    G = ExaDisNet().read_paradis('../../examples/22_fcc_Cu_15um_1e3/180chains_16.10e.data', verbose=False)
    #G = ExaDisNet().generate_prismatic_config(state["crystal"], 50000.0, 1, 2000.0, maxseg=state["maxseg"], seed=1234)
    N = DisNetManager(G)
    
    state["applied_stress"] = np.array([10e6, 5e6, 20e6, 3e6, 7e6, 1e6])
    
    if name == 'lt':
        calforce = CalForce(force_mode='LINE_TENSION_MODEL', state=state)
    elif name == 'cutoff':
        calforce = CalForce(force_mode='CUTOFF_MODEL', state=state, cutoff=7500.0)
    elif name == 'ddd_fft':
        calforce = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=64, cell=N.cell)
    elif name == 'fft':
        calforce = CalForceFFT(state=state, Ngrid=64, cell=N.cell)
    else:
        raise ValueError(f"Invalid force type = '{name}'")
    
    calforce.PreCompute(N, state)
    calforce.NodeForce(N, state, pre_compute=False)
    
    results = ''
    for f in state["nodeforces"]:
        results += '%e %e %e\n' % tuple(f)
    
    if 0:
        # write reference results in a file
        with open(f"expected_output/test_force_{name}.dat", 'w') as f:
            f.write(results)
    else:
        # print current results in the console
        print(results)
    
    pyexadis.finalize()
    

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    test_force(name)
