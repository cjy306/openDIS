"""Compare the FFT stress of an oblique periodic composition wave to its exact solution."""
import numpy as np

from coherency_field import isotropic_coherency_stress


def main():
    n, box_m = 64, 300e-9
    mean_cr, amplitude, beta = 0.13, 0.02, 0.02
    mu_pa, nu = 81e9, 0.3
    i, j, k = np.ogrid[:n, :n, :n]
    wave = np.cos(2 * np.pi * (i + 2*j + 3*k) / n)
    concentration = mean_cr + amplitude * wave
    stress = isotropic_coherency_stress(concentration, mean_cr, beta, mu_pa, nu, box_m)
    if stress.shape != (n, n, n, 6) or not np.isfinite(stress).all():
        raise ValueError("Invalid FFT stress output")

    # Rotate the uniaxial-wave solution into the grid frame:
    # sigma = E beta amplitude cos(phase) (q q^T - I) / (1 - nu),
    # where q = (1,2,3)/sqrt(14). Write out the coefficients independently.
    scale_pa = 2 * mu_pa * (1 + nu) * beta * amplitude / (1 - nu)
    components = ("xx", "yy", "zz", "yz", "xz", "xy")
    coefficients = np.array([-13, -10, -5, 6, 3, 2]) / 14
    print(f"Grid={n}^3, L={box_m:g} m, mode=(1,2,3), mean Cr={mean_cr}, amplitude={amplitude}")
    print(f"beta={beta}, mu={mu_pa:g} Pa, nu={nu}")
    print("Origin values in MPa; errors over all grid points.")
    print("component    ref_origin_MPa    actual_origin_MPa    max_error_Pa    error/ref_peak")
    errors = []
    for index, (name, coefficient) in enumerate(zip(components, coefficients)):
        origin_pa = scale_pa * coefficient
        error = np.max(np.abs(stress[..., index] - origin_pa * wave))
        relative = error / abs(origin_pa)
        print(f"{name:>6} {origin_pa/1e6:17.8f} {stress[0,0,0,index]/1e6:20.8f}"
              f" {error:15.6e} {relative:17.6e}")
        errors.append(relative)
    tolerance = 1e-12
    print(f"Maximum component-normalized error: {max(errors):.6e} (limit {tolerance:g})")
    if max(errors) > tolerance:
        print("FAIL")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
