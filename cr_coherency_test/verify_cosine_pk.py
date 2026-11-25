"""Verify incremental ExaDiS endpoint forces against a straight-segment PK solution."""
import numpy as np

from coherency_field import isotropic_coherency_stress


def endpoint_forces(model, network, manager, state):
    model.PreCompute(manager, state)
    return np.array([
        model.OneNodeForce(manager, state, tuple(tag), update_state=False)
        for tag in network.get_tags()
    ])


def verify():
    import pyexadis
    from pyexadis_base import CalForce, DisNetManager, ExaDisNet, NodeConstraints

    n = 64
    box_m, length_m, b_m = 300e-9, 100e-9, 0.25e-9
    mean_cr, amplitude, beta = 0.13, 0.02, 0.02
    mu, nu = 81e9, 0.3
    cosine = np.cos(2 * np.pi * np.arange(n) / n)
    concentration = np.broadcast_to((mean_cr + amplitude * cosine)[:, None, None], (n, n, n))
    stress = isotropic_coherency_stress(concentration, mean_cr, beta, mu, nu, box_m)
    peak_pa = 2 * mu * (1 + nu) * beta * amplitude / (1 - nu)
    peak_force = peak_pa * length_m * b_m / 2
    state = dict(crystal="bcc", burgmag=b_m, mu=mu, nu=nu, a=3.0,
                 maxseg=length_m / b_m, minseg=3.0, rtol=1.0, rann=2.0,
                 nextdt=1e-12, maxdt=1e-9, applied_stress=np.zeros(6))
    box = box_m / b_m
    cell = pyexadis.Cell(box)
    options = dict(state=state, cell=cell, Ngrid=n)
    baseline = CalForce(force_mode="SUBCYCLING_MODEL", **options)
    loaded = CalForce(force_mode="SUBCYCLING_COHERENCY_MODEL", coherency_stress=stress, **options)
    # The off-grid midpoint samples the mean of the two adjacent grid-plane stresses.
    cases = [(0, 1.0), (n // 4, 0.0), (n // 2, -1.0),
             (8.5, (cosine[8] + cosine[9]) / 2)]
    errors = []
    print(f"pyexadis: {pyexadis.__file__}")
    print(f"b={b_m:g} m, segment={length_m:g} m, peak endpoint force={peak_force*1e9:.9f} nN")
    print("x/L       node   ref_Fx(nN)    actual_Fx(nN)   actual_Fy(nN)   actual_Fz(nN)   normalized_error")
    for plane, factor in cases:
        x = plane / n * box
        nodes = [[x, box / 2, box / 4 + dz, NodeConstraints.PINNED_NODE]
                 for dz in (0, length_m / b_m)]
        network = ExaDisNet(cell, nodes, [[0, 1, 0., 1., 0., 1., 0., 0.]])
        manager = DisNetManager(network)
        off = endpoint_forces(baseline, network, manager, state)
        on = endpoint_forces(loaded, network, manager, state)
        # ExaDiS uses normalized Burgers vectors and positions in units of b.
        actual = (on - off) * b_m**2
        if actual.shape != (2, 3) or not np.isfinite(actual).all():
            raise ValueError("Invalid endpoint force output")
        expected = np.array([-peak_force * factor, 0., 0.])
        for label, vector in zip("AB", actual):
            error = np.linalg.norm(vector - expected) / peak_force
            errors.append(error)
            print(f"{plane/n:.7f}  {label}  {expected[0]*1e9:13.9f}  "
                  + "  ".join(f"{v*1e9:14.9f}" for v in vector) + f"  {error:.4e}")
    limit = 1e-12
    print(f"Maximum normalized endpoint error: {max(errors):.6e} (limit {limit:g})")
    if max(errors) > limit:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    import pyexadis

    pyexadis.initialize(verbose=False)
    try:
        status = verify()
    finally:
        pyexadis.finalize()
    raise SystemExit(status)
