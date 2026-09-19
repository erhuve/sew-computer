import unittest

import ipctk
import numpy as np

from solver_ipc_contact import IpcSurfaceContact


def square_grid(subdivisions, width=.001):
    positions = np.array([[width * column / subdivisions, width * row / subdivisions, 0.]
                          for row in range(subdivisions + 1) for column in range(subdivisions + 1)])
    faces = []
    for row in range(subdivisions):
        for column in range(subdivisions):
            first = row * (subdivisions + 1) + column
            second, third, fourth = first + 1, first + subdivisions + 1, first + subdivisions + 2
            faces.extend([[first, second, fourth], [first, fourth, third]])
    return positions, np.array(faces)


def narrow_contact(rest, faces, activation=.00002, pressure=10000.):
    return IpcSurfaceContact(rest, faces, activation_distance_m=activation,
                             minimum_distance_m=.0001, stiffness=pressure,
                             energy_profile="area-improved-max")


class ContactRangeAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def test_narrow_range_preserves_stationary_flat_source(self):
        positions, faces = square_grid(4)
        original = positions.copy()
        broad = narrow_contact(positions, faces, activation=.001)
        narrow = narrow_contact(positions, faces)
        self.assertGreater(broad.energy(positions), 0)
        self.assertGreater(np.linalg.norm(broad.gradient(positions)), 1e-6)
        narrow.validate_state(positions)
        self.assertEqual(narrow.energy(positions), 0)
        np.testing.assert_array_equal(narrow.gradient(positions), np.zeros_like(positions))
        np.testing.assert_array_equal(positions, original)

    def test_refinement_can_reactivate_intrinsic_contact(self):
        positions, faces = square_grid(6)
        contact = narrow_contact(positions, faces)
        contact.validate_state(positions)
        self.assertGreater(contact.energy(positions), 0)
        self.assertGreater(np.linalg.norm(contact.gradient(positions)), 1e-6)

    def test_refinement_below_minimum_separation_rejects(self):
        positions, faces = square_grid(8)
        contact = narrow_contact(positions, faces)
        with self.assertRaisesRegex(ValueError, "minimum surface separation"):
            contact.validate_state(positions)

    def test_local_fan_fold_retains_force_and_continuous_guard(self):
        rest = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0], [.002, .002, 0]])
        faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
        contact = narrow_contact(rest, faces)
        start = rest.copy()
        start[3] = [.008, -.002, .00013]
        contact.validate_state(start)
        self.assertGreater(contact.energy(start), 0)
        self.assertGreater(np.linalg.norm(contact.gradient(start)), .01)
        end = start.copy()
        end[3, 2] *= -1
        limit = contact.step_limit(start, end)
        self.assertGreater(limit, 0)
        self.assertLess(limit, .5)
        self.assertFalse(contact.path_safe(start, end))

    def test_crossing_layers_detected_outside_activation_range(self):
        layer = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0]])
        rest = np.vstack([layer, layer])
        contact = narrow_contact(rest, np.array([[0, 1, 2], [3, 4, 5]]))
        start = rest.copy()
        start[3:, 2] = .002
        end = start.copy()
        end[3:, 2] = -.002
        self.assertEqual(contact.energy(start), 0)
        limit = contact.step_limit(start, end)
        self.assertGreater(limit, 0)
        self.assertLess(limit, .475)
        self.assertFalse(contact.path_safe(start, end))
        contact.validate_state(start + limit * (end - start))

    def test_aligned_triangle_pressure_and_range_follow_analytic_barrier(self):
        layer = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0]])
        rest = np.vstack([layer, layer])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        area_weight = 1.5 * .5 * .01 ** 2
        minimum = .0001
        for activation in (.00001, .00002, .00004):
            for pressure in (1000., 10000., 100000.):
                with self.subTest(activation=activation, pressure=pressure):
                    contact = narrow_contact(rest, faces, activation, pressure)
                    distance = minimum + activation / 2
                    positions = rest.copy()
                    positions[3:, 2] = distance
                    squared_gap = activation * (2 * minimum + activation)
                    ratio = (distance ** 2 - minimum ** 2) / squared_gap
                    density = -(ratio - 1) ** 2 * np.log(ratio)
                    derivative = -2 * (ratio - 1) * np.log(ratio) - (ratio - 1) ** 2 / ratio
                    expected_energy = area_weight * pressure * activation * density
                    expected_gradient = area_weight * pressure * activation * derivative * 2 * distance / squared_gap
                    np.testing.assert_allclose(contact.energy(positions), expected_energy, rtol=1e-12)
                    gradient = contact.gradient(positions)
                    np.testing.assert_allclose(gradient[3:, 2].sum(), expected_gradient, rtol=1e-12)
                    np.testing.assert_allclose(gradient.sum(axis=0), np.zeros(3), atol=1e-12)

    def test_active_narrow_contact_repels_layer_with_pinned_surface(self):
        from solver_global_sewing import GlobalSewingSolver
        from test_solver_ipc_integration import IpcIntegrationTests

        model, rest, _, _, _ = IpcIntegrationTests().fixture(fixed=True, gap=.00011)
        contact = narrow_contact(rest, model.tri_indices.numpy())
        solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}], 1e-6, contact=contact)
        initial_energy = contact.energy(rest)
        final, velocities, report = solver.step(rest, np.zeros_like(rest), solver.sewing @ rest,
                                                .0001, max_evaluations=30)
        self.assertTrue(report["converged"], report)
        self.assertLessEqual(report["gradientInfinityNorm"], 1e-6)
        self.assertFalse(report["accepted"])
        self.assertTrue(contact.path_safe(rest, final))
        np.testing.assert_array_equal(final[:3], rest[:3])
        np.testing.assert_array_equal(velocities[:3], np.zeros((3, 3)))
        self.assertGreater(float(final[3:, 2].min()), float(rest[3:, 2].min()))
        self.assertLess(contact.energy(final), initial_energy)
        np.testing.assert_array_equal(contact.rest_positions, rest)


if __name__ == "__main__":
    unittest.main()
