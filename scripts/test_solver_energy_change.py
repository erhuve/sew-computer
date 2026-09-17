import unittest

import numpy as np

from solver_energy_change import membrane_energy_change


class MembraneEnergyChangeTests(unittest.TestCase):
    def reference(self, deformation, areas, materials):
        deformation, areas, materials = [np.asarray(value, dtype=np.longdouble)
                                        for value in (deformation, areas, materials)]
        area = np.sqrt(np.sum(np.cross(deformation[:, 0], deformation[:, 1]) ** 2, axis=1))
        shear = materials[:, 0]
        bulk = shear + materials[:, 1]
        alpha = 1 + shear / np.maximum(bulk, np.longdouble(1e-6))
        return np.sum(areas * (shear * np.sum(deformation ** 2, axis=(1, 2)) / 2
                              + bulk * (area - alpha) ** 2 / 2))

    def test_heterogeneous_changes_match_longdouble_reference(self):
        random = np.random.default_rng(4201)
        deformation = random.normal(size=(30, 2, 3))
        delta = random.normal(size=(30, 2, 3)) * .03
        areas = random.uniform(.001, .01, 30)
        materials = np.column_stack((random.uniform(100, 10000, 30),
                                     random.uniform(100, 20000, 30), np.zeros(30)))
        expected = self.reference(deformation.astype(np.longdouble) + delta, areas, materials) - self.reference(deformation, areas, materials)
        actual = membrane_energy_change(deformation, delta, areas, materials)
        np.testing.assert_allclose(actual, float(expected), rtol=1e-13, atol=1e-12)

    def test_tiny_change_survives_large_absolute_energy(self):
        deformation = np.array([[[2., 0, 0], [0, 1., 0]]])
        delta = np.array([[[1e-17, 0, 0], [0, 0, 0]]])
        areas = np.ones(1)
        materials = np.array([[10000., 20000., 0]])
        actual = membrane_energy_change(deformation, delta, areas, materials)
        expected = 40000 * 1e-17 + 20000 * 1e-34
        self.assertGreater(actual, 0)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
        self.assertEqual(float(self.reference(deformation + delta, areas, materials)
                               - self.reference(deformation, areas, materials)), 0.)

    def test_step_reversal_and_composition(self):
        deformation = np.array([[[1.2, .1, .2], [.2, .9, .1]]])
        first_delta = np.array([[[.01, -.02, .03], [.04, -.01, .02]]])
        second_delta = first_delta * -.3
        areas, materials = np.array([.03]), np.array([[9000., 12000., 0.]])
        forward = membrane_energy_change(deformation, first_delta, areas, materials)
        reverse = membrane_energy_change(deformation + first_delta, -first_delta, areas, materials)
        self.assertAlmostEqual(forward, -reverse, places=12)
        sequential = forward + membrane_energy_change(deformation + first_delta, second_delta, areas, materials)
        combined = membrane_energy_change(deformation, first_delta + second_delta, areas, materials)
        self.assertAlmostEqual(sequential, combined, places=12)

    def test_directional_derivative_and_zero_change(self):
        deformation = np.array([[[1.2, .1, .2], [.2, .9, .1]]])
        direction = np.array([[[.1, -.2, .3], [.4, -.1, .2]]])
        areas, materials = np.array([.03]), np.array([[9000., 12000., 0.]])
        step = 1e-5
        reference = (self.reference(deformation.astype(np.longdouble) + step * direction, areas, materials)
                     - self.reference(deformation.astype(np.longdouble) - step * direction, areas, materials)) / (2 * step)
        actual = (membrane_energy_change(deformation, step * direction, areas, materials)
                  - membrane_energy_change(deformation, -step * direction, areas, materials)) / (2 * step)
        np.testing.assert_allclose(actual, float(reference), rtol=1e-10, atol=1e-9)
        self.assertEqual(membrane_energy_change(deformation, np.zeros_like(deformation), areas, materials), 0.)

    def test_invalid_and_degenerate_endpoints_rejected(self):
        deformation = np.array([[[1., 0, 0], [0, 1., 0]]])
        areas, materials = np.ones(1), np.array([[10000., 20000., 0.]])
        for delta in (-deformation, np.full_like(deformation, np.nan)):
            with self.assertRaises(ValueError):
                membrane_energy_change(deformation, delta, areas, materials)
        with self.assertRaises(ValueError):
            membrane_energy_change(np.zeros_like(deformation), deformation, areas, materials)
        with self.assertRaises(ValueError):
            membrane_energy_change(deformation, np.zeros_like(deformation), -areas, materials)
        self.assertEqual(membrane_energy_change(np.empty((0, 2, 3)), np.empty((0, 2, 3)),
                                                np.empty(0), np.empty((0, 3))), 0.)


if __name__ == "__main__":
    unittest.main()
