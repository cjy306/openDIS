"""Check the two endpoint PK forces of an oblique segment in the (1,2,3) wave."""
from itertools import product

import numpy as np

from coherency_field import isotropic_coherency_stress
from verify_cosine_pk import endpoint_forces


def reference_forces(grid_position, n, peak_vector):
    """Continuum and eight-corner references, independent of the FFT stress array."""
    mode = np.array([1, 2, 3])
    base = np.floor(grid_position)
    fraction = grid_position - base
    exact_wave = np.cos(2 * np.pi * (mode @ grid_position) / n)
    interpolated_wave = 0.0
    for corner in product((0, 1), repeat=3):
        weight = np.prod(np.where(corner, fraction, 1 - fraction))
        phase = 2 * np.pi * (mode @ (base + corner)) / n
        interpolated_wave += weight * np.cos(phase)
    return exact_wave * peak_vector, interpolated_wave * peak_vector


def verify():
    import pyexadis
    from pyexadis_base import CalForce, DisNetManager, ExaDisNet, NodeConstraints

    n = 64
    box_m, length_m, b_m = 300e-9, 100e-9, 0.25e-9
    mean_cr, amplitude, beta = 0.13, 0.02, 0.02
    mu, nu = 81e9, 0.3
    i, j, k = np.ogrid[:n, :n, :n]
    concentration = mean_cr + amplitude * np.cos(2 * np.pi * (i + 2*j + 3*k) / n)
    stress = isotropic_coherency_stress(concentration, mean_cr, beta, mu, nu, box_m)

    tangent = np.array([1, 1, -1]) / np.sqrt(3)
    burgers = np.ones(3) / np.sqrt(3)
    plane = np.array([-1, 1, 0]) / np.sqrt(2)
    scale_pa = 2 * mu * (1 + nu) * beta * amplitude / (1 - nu)
    # At unit cosine: (q q^T - I) @ burgers = (-8,-2,4)/(14 sqrt(3)).
    # Crossing with tangent gives (-2,-4,-6)/42.
    peak_vector = scale_pa * b_m * length_m / 2 * np.array([-2, -4, -6]) / 42
    state = dict(crystal="bcc", burgmag=b_m, mu=mu, nu=nu, a=3.0,
                 maxseg=length_m / b_m, minseg=3.0, rtol=1.0, rann=2.0,
                 nextdt=1e-12, maxdt=1e-9, applied_stress=np.zeros(6))
    box = box_m / b_m
    cell = pyexadis.Cell(box)
    options = dict(state=state, cell=cell, Ngrid=n)
    off = CalForce(force_mode="SUBCYCLING_MODEL", **options)
    on = CalForce(force_mode="SUBCYCLING_COHERENCY_MODEL", coherency_stress=stress, **options)

    print(f"pyexadis: {pyexadis.__file__}")
    print("Grid=64^3, L=300 nm, segment=100 nm, b=0.25 nm; force units: nN")
    print("line=[1,1,-1], Burgers=[1,1,1] (both normalized)")
    errors = []
    for grid_position in (np.array([24.2, 24.4, 20.7]), np.array([56.2, 24.4, 20.7])):
        exact, reference = reference_forces(grid_position, n, peak_vector)
        midpoint = grid_position * box / n
        half_segment = 0.5 * length_m / b_m * tangent
        nodes = [[*pos, NodeConstraints.PINNED_NODE]
                 for pos in (midpoint - half_segment, midpoint + half_segment)]
        network = ExaDisNet(cell, nodes, [[0, 1, *burgers, *plane]])
        manager = DisNetManager(network)
        baseline = endpoint_forces(off, network, manager, state)
        actual = (endpoint_forces(on, network, manager, state) - baseline) * b_m**2
        if actual.shape != (2, 3) or not np.isfinite(actual).all():
            raise ValueError("Invalid endpoint forces")
        print(f"Midpoint (grid units): {grid_position}")
        print("continuum reference (Fx,Fy,Fz):", exact * 1e9)
        print("8-corner reference  (Fx,Fy,Fz):", reference * 1e9)
        interp_error = np.linalg.norm(reference - exact) / np.linalg.norm(exact)
        print(f"Relative interpolation error vs continuum: {interp_error:.6%}")
        for label, vector in zip("AB", actual):
            error = np.linalg.norm(vector - reference) / np.linalg.norm(reference)
            errors.append(error)
            print(f"actual node {label} (Fx,Fy,Fz): {vector*1e9}; relative error vs 8 corners: {error:.6e}")
    tolerance = 1e-12
    print(f"Maximum relative endpoint error vs 8 corners: {max(errors):.6e} (limit {tolerance:g})")
    if max(errors) > tolerance:
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
