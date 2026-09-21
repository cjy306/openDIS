"""Periodic Cr concentration and isotropic coherency-stress fields."""

from __future__ import annotations

import numpy as np


COMPONENT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")


def _shape3(shape) -> tuple[int, int, int]:
    result = tuple(int(value) for value in shape)
    if len(result) != 3 or any(value < 2 for value in result):
        raise ValueError("shape must contain three grid dimensions >= 2")
    return result


def _box3(box_m) -> np.ndarray:
    box = np.asarray(box_m, dtype=float)
    if box.ndim == 0:
        box = np.repeat(box, 3)
    if box.shape != (3,) or not np.isfinite(box).all() or np.any(box <= 0.0):
        raise ValueError("box_m must be a positive scalar or three positive lengths")
    return box


def _wavevectors(shape, box_m):
    box = _box3(box_m)
    return np.meshgrid(
        *(2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
          for n, length in zip(shape, box)),
        indexing="ij",
    )


def generate_periodic_concentration(
    shape,
    box_m,
    mean_cr: float,
    std_cr: float,
    correlation_length_m: float,
    seed: int,
) -> np.ndarray:
    """Generate a reproducible periodic Gaussian Cr atomic-fraction field."""
    shape = _shape3(shape)
    if not np.isfinite(mean_cr) or not 0.0 <= mean_cr <= 1.0:
        raise ValueError("mean_cr must be a finite atomic fraction in [0, 1]")
    if not np.isfinite(std_cr) or std_cr <= 0.0:
        raise ValueError("std_cr must be positive")
    if not np.isfinite(correlation_length_m) or correlation_length_m <= 0.0:
        raise ValueError("correlation_length_m must be positive")

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(shape)
    wavevectors = _wavevectors(shape, box_m)
    k2 = sum(component * component for component in wavevectors)

    # PHYS-APPROX: stationary Gaussian composition statistics with one isotropic
    # correlation length; ceiling = no measured C35M spectrum or bounds-aware
    # chemistry; upgrade = fit the spectral density to APT/SANS data.
    spectral_filter = np.exp(-0.25 * correlation_length_m**2 * k2)
    fluctuation = np.fft.ifftn(np.fft.fftn(white) * spectral_filter).real
    fluctuation -= fluctuation.mean()
    numerical_std = fluctuation.std()
    if numerical_std == 0.0:
        raise ValueError("filtered concentration field has zero variance")
    fluctuation *= std_cr / numerical_std
    return np.ascontiguousarray(mean_cr + fluctuation, dtype=np.float64)


def isotropic_coherency_stress(
    concentration: np.ndarray,
    mean_cr: float,
    misfit_coefficient: float,
    mu_pa: float,
    nu: float,
    box_m,
) -> np.ndarray:
    """Solve periodic homogeneous-isotropic elasticity for hydrostatic eigenstrain."""
    concentration = np.asarray(concentration, dtype=np.float64)
    shape = _shape3(concentration.shape)
    if not np.isfinite(concentration).all():
        raise ValueError("concentration must contain only finite values")
    if not np.isfinite(mean_cr) or not np.isfinite(misfit_coefficient):
        raise ValueError("mean_cr and misfit_coefficient must be finite")
    if not np.isfinite(mu_pa) or mu_pa <= 0.0:
        raise ValueError("mu_pa must be positive")
    if not np.isfinite(nu) or not -1.0 < nu < 0.5:
        raise ValueError("nu must lie between -1 and 0.5")

    wavevectors = _wavevectors(shape, box_m)
    k2 = sum(component * component for component in wavevectors)
    nonzero = k2 > 0.0
    eigenstrain_hat = np.fft.fftn(
        misfit_coefficient * (concentration - mean_cr)
    )
    eigenstrain_hat[(0, 0, 0)] = 0.0
    # PHYS-APPROX: remove even-grid Nyquist planes so mixed Fourier derivatives
    # retain Hermitian symmetry; ceiling = discards the shortest represented
    # wavelength; upgrade = a staggered-grid discrete Green operator.
    for axis, size in enumerate(shape):
        if size % 2 == 0:
            index = [slice(None)] * 3
            index[axis] = size // 2
            eigenstrain_hat[tuple(index)] = 0.0

    lame = 2.0 * mu_pa * nu / (1.0 - 2.0 * nu)
    strain_factor = (3.0 * lame + 2.0 * mu_pa) / (lame + 2.0 * mu_pa)
    inv_k2 = np.zeros_like(k2)
    inv_k2[nonzero] = 1.0 / k2[nonzero]

    # PHYS-APPROX: homogeneous isotropic elasticity; ceiling = no BCC cubic
    # anisotropy or composition-dependent moduli; upgrade = anisotropic Fourier
    # Green operator using C11/C12/C44(c).
    strain_hat = np.empty((3, 3) + shape, dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            strain_hat[i, j] = (
                strain_factor
                * eigenstrain_hat
                * wavevectors[i]
                * wavevectors[j]
                * inv_k2
            )

    trace_strain_hat = sum(strain_hat[i, i] for i in range(3))
    stress_hat = np.empty_like(strain_hat)
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            stress_hat[i, j] = (
                lame * (trace_strain_hat - 3.0 * eigenstrain_hat) * delta
                + 2.0
                * mu_pa
                * (strain_hat[i, j] - eigenstrain_hat * delta)
            )
            stress_hat[i, j][(0, 0, 0)] = 0.0

    components = (
        stress_hat[0, 0],
        stress_hat[1, 1],
        stress_hat[2, 2],
        stress_hat[1, 2],
        stress_hat[0, 2],
        stress_hat[0, 1],
    )
    stress = np.stack([np.fft.ifftn(value).real for value in components], axis=-1)
    return np.ascontiguousarray(stress, dtype=np.float64)


def equilibrium_residual(stress: np.ndarray, box_m) -> float:
    """Return ||div(sigma)|| / ||k sigma|| for a periodic six-component field."""
    stress = np.asarray(stress, dtype=np.float64)
    if stress.ndim != 4 or stress.shape[-1] != 6:
        raise ValueError("stress must have shape (Nx, Ny, Nz, 6)")
    shape = _shape3(stress.shape[:3])
    if not np.isfinite(stress).all():
        raise ValueError("stress must contain only finite values")

    wavevectors = _wavevectors(shape, box_m)
    tensor = np.empty((3, 3) + shape, dtype=np.complex128)
    hats = [np.fft.fftn(stress[..., index]) for index in range(6)]
    tensor[0, 0], tensor[1, 1], tensor[2, 2] = hats[0], hats[1], hats[2]
    tensor[1, 2] = tensor[2, 1] = hats[3]
    tensor[0, 2] = tensor[2, 0] = hats[4]
    tensor[0, 1] = tensor[1, 0] = hats[5]

    divergence = [
        sum(wavevectors[j] * tensor[i, j] for j in range(3))
        for i in range(3)
    ]
    numerator = np.sqrt(sum(np.vdot(value, value).real for value in divergence))
    k2 = sum(component * component for component in wavevectors)
    denominator = np.sqrt(
        sum(np.vdot(np.sqrt(k2) * value, np.sqrt(k2) * value).real
            for value in tensor.reshape(9, *shape))
    )
    return float(numerator / denominator) if denominator > 0.0 else 0.0
