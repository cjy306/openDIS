"""Generate the shared single-loop configuration and Cr coherency field."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for relative_path in ("core/exadis/python", "core/pydis/python", "python", "lib"):
    path = os.path.join(REPO_ROOT, relative_path)
    if path not in sys.path:
        sys.path.append(path)

import pyexadis
from pyexadis_base import ExaDisNet
from pyexadis_utils import insert_prismatic_loop

from coherency_field import (
    COMPONENT_ORDER,
    equilibrium_residual,
    generate_periodic_concentration,
    isotropic_coherency_stress,
)


def main() -> None:
    output_dir = os.path.join(SCRIPT_DIR, "init_coherency")
    os.makedirs(output_dir, exist_ok=True)

    concentration = generate_periodic_concentration(
        (64, 64, 64), 300e-9, 0.13, 0.02, 50e-9, 12345
    )
    stress = isotropic_coherency_stress(
        concentration, 0.13, 0.02, 81e9, 0.3, 300e-9
    )
    residual = equilibrium_residual(stress, 300e-9)

    np.savez_compressed(
        os.path.join(output_dir, "coherency_stress.npz"),
        stress=stress,
        concentration=concentration,
        component_order=np.asarray(COMPONENT_ORDER),
        box_m=np.asarray([300e-9, 300e-9, 300e-9]),
        grid_shape=np.asarray(stress.shape[:3], dtype=np.int64),
        correlation_length_m=np.asarray(50e-9),
        seed=np.asarray(12345, dtype=np.int64),
    )

    metadata = {
        "purpose": "module validation; parameters are not calibrated C35M data",
        "component_order": list(COMPONENT_ORDER),
        "box_m": [300e-9, 300e-9, 300e-9],
        "grid_shape": [64, 64, 64],
        "mean_cr": 0.13,
        "std_cr": 0.02,
        "correlation_length_m": 50e-9,
        "misfit_coefficient_per_atomic_fraction": 0.02,
        "mu_pa": 81e9,
        "nu": 0.3,
        "seed": 12345,
        "equilibrium_residual": residual,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    pyexadis.initialize()
    try:
        cell = pyexadis.Cell(300e-9 / 0.248e-9)
        center = np.asarray(cell.origin) + np.matmul(
            np.asarray([0.5, 0.5, 0.5]), np.asarray(cell.h).T
        )
        burgers = np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        nodes, segments = insert_prismatic_loop(
            "bcc",
            cell,
            [],
            [],
            burgers,
            0.5 * 16.6e-9 / 0.248e-9,
            center,
            maxseg=10.0,
        )
        network = ExaDisNet(cell, nodes, segments)
        network.write_data(os.path.join(output_dir, "init_config.data"))
    finally:
        pyexadis.finalize()

    stress_norm = np.linalg.norm(stress, axis=-1)
    print(f"Initial configuration: {os.path.join(output_dir, 'init_config.data')}")
    print(f"Stress field: {os.path.join(output_dir, 'coherency_stress.npz')}")
    print(f"Stress RMS norm: {np.sqrt(np.mean(stress_norm**2)):.6e} Pa")
    print(f"Stress component range: [{stress.min():.6e}, {stress.max():.6e}] Pa")
    print(f"Fourier equilibrium residual: {residual:.6e}")
    print(f"maxseg/correlation_length: {10.0 * 0.248e-9 / 50e-9:.6f}")


if __name__ == "__main__":
    main()
