import unittest
from pathlib import Path

import numpy as np

from cr_coherency_test.coherency_field import (
    equilibrium_residual,
    generate_periodic_concentration,
    isotropic_coherency_stress,
)
from cr_coherency_test.plot_coherency_compare import (
    require_output_files,
    unsigned_normal_angle,
)


class CoherencyFieldTests(unittest.TestCase):
    def test_seed_is_deterministic(self):
        first = generate_periodic_concentration(
            (8, 8, 8), 300e-9, 0.13, 0.02, 50e-9, 12345
        )
        second = generate_periodic_concentration(
            (8, 8, 8), 300e-9, 0.13, 0.02, 50e-9, 12345
        )
        np.testing.assert_array_equal(first, second)

    def test_uniform_composition_has_zero_fluctuation_stress(self):
        concentration = np.full((8, 8, 8), 0.13)
        stress = isotropic_coherency_stress(
            concentration, 0.13, 0.02, 81e9, 0.3, 300e-9
        )
        np.testing.assert_allclose(stress, 0.0, atol=1e-6)

    def test_stress_scales_linearly_with_misfit(self):
        concentration = generate_periodic_concentration(
            (8, 8, 8), 300e-9, 0.13, 0.02, 50e-9, 7
        )
        stress_one = isotropic_coherency_stress(
            concentration, 0.13, 0.01, 81e9, 0.3, 300e-9
        )
        stress_two = isotropic_coherency_stress(
            concentration, 0.13, 0.02, 81e9, 0.3, 300e-9
        )
        np.testing.assert_allclose(
            stress_two, 2.0 * stress_one, rtol=1e-12, atol=1e-5
        )

    def test_generated_stress_is_equilibrated_and_finite(self):
        concentration = generate_periodic_concentration(
            (16, 16, 16), 300e-9, 0.13, 0.02, 50e-9, 9
        )
        stress = isotropic_coherency_stress(
            concentration, 0.13, 0.02, 81e9, 0.3, 300e-9
        )
        self.assertEqual(stress.shape, (16, 16, 16, 6))
        self.assertTrue(np.isfinite(stress).all())
        self.assertLess(equilibrium_residual(stress, 300e-9), 1e-10)

    def test_plane_normal_is_sign_invariant(self):
        angle = unsigned_normal_angle(
            np.asarray([1.0, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0])
        )
        self.assertAlmostEqual(angle, 0.0)

    def test_missing_output_raises(self):
        with self.assertRaises(FileNotFoundError):
            require_output_files(Path("missing-directory"))


if __name__ == "__main__":
    unittest.main()
