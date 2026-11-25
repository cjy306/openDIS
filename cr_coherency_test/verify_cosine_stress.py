"""Check all six FFT stress components against the periodic cosine solution."""
import numpy as np

from coherency_field import COMPONENT_ORDER, isotropic_coherency_stress


def main():
    n = 64
    box_m = 300e-9
    mean_cr, amplitude, beta = 0.13, 0.02, 0.02
    mu_pa, nu = 81e9, 0.3
    tolerance = 1e-12
    cosine = np.cos(2 * np.pi * np.arange(n) / n)
    concentration = np.broadcast_to(
        (mean_cr + amplitude * cosine)[:, None, None], (n, n, n)
    )
    stress = isotropic_coherency_stress(concentration, mean_cr, beta, mu_pa, nu, box_m)
    if stress.shape != (n, n, n, 6) or not np.isfinite(stress).all():
        raise ValueError("Invalid FFT stress output")

    # Under zero mean stress, sigma_xx = 0 and sigma_yy = sigma_zz = -E e*/(1-nu).
    young_pa = 2 * mu_pa * (1 + nu)
    peak_pa = young_pa * beta * amplitude / (1 - nu)
    transverse = -peak_pa * cosine[:, None, None]
    print(f"Grid: {n}^3, L={box_m:g} m, mean Cr={mean_cr}, amplitude={amplitude}")
    print(f"Benchmark beta={beta}, mu={mu_pa:g} Pa, nu={nu}")
    print(f"Analytical peak magnitude: {peak_pa/1e6:.6f} MPa")
    print("x/L      analytical_yy_MPa     FFT_yy_MPa")
    for i in range(0, n, n // 4):
        print(f"{i/n:.2f}     {transverse[i,0,0]/1e6:17.8f}   {stress[i,0,0,1]/1e6:13.8f}")

    errors = []
    for index, component in enumerate(COMPONENT_ORDER):
        reference = transverse if component in ("yy", "zz") else 0.0
        error = np.max(np.abs(stress[..., index] - reference))
        errors.append(error)
        print(f"{component}: max absolute error = {error:.8e} Pa")
    normalized_error = max(errors) / peak_pa
    print(f"Overall max absolute error: {max(errors):.8e} Pa")
    print(f"Normalized max error: {normalized_error:.8e} (limit {tolerance:g})")
    if normalized_error > tolerance:
        raise SystemExit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()
