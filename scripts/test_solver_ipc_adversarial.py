import unittest

import numpy as np

from solver_ipc_contact import IpcSurfaceContact


class IpcContactAdversarialTests(unittest.TestCase):
    def test_unrepresentable_contact_scales_reject_before_native_construction(self):
        positions, faces = self.fixture()
        for activation, minimum, stiffness in ((1e200, 1e-4, 1.), (1e-200, 1e-210, 1.),
                                                (.001, 1e200, 1.), (1e-20, 1., 1.),
                                                (1e100, 1e90, 1e100)):
            with self.subTest(activation=activation, minimum=minimum), self.assertRaises(ValueError):
                IpcSurfaceContact(positions, faces, activation_distance_m=activation,
                                  minimum_distance_m=minimum, stiffness=stiffness)

    def fixture(self, gap=0.0005):
        positions = np.array([[0., 0., 0.], [0.01, 0., 0.], [0., 0.01, 0.],
                              [0.001, 0.001, gap], [0.008, 0.001, gap],
                              [0.001, 0.008, gap]])
        return positions, np.array([[0, 1, 2], [3, 4, 5]])

    def contact(self, positions, faces):
        return IpcSurfaceContact(positions, faces, activation_distance_m=0.001,
                                 minimum_distance_m=0.0001, stiffness=2.)

    def test_linear_separation_uses_squared_toolkit_distance(self):
        positions, faces = self.fixture()
        contact = self.contact(positions, faces)
        contact.validate_state(positions)
        self.assertGreater(contact.energy(positions), 0.)
        for gap in [0., 0.00005, 0.0001]:
            invalid = positions.copy()
            invalid[3:, 2] = gap
            with self.assertRaises(ValueError):
                contact.validate_state(invalid)

    def test_coplanar_overlapping_layers_reject(self):
        positions, faces = self.fixture(0.)
        contact = self.contact(positions, faces)
        with self.assertRaises(ValueError):
            contact.validate_state(positions)

    def test_transverse_intersection_rejects_despite_positive_primitive_distance(self):
        positions = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.],
                              [0., 0., -1.], [0., 0., 1.], [0.1, 0., 1.]])
        contact = self.contact(positions, [[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            contact.validate_state(positions)

    def test_endpoint_safe_tunneling_rejects(self):
        positions, faces = self.fixture(0.002)
        contact = self.contact(positions, faces)
        endpoint = positions.copy()
        endpoint[3:, 2] = -0.002
        contact.validate_state(endpoint)
        limit = contact.step_limit(positions, endpoint)
        self.assertGreater(limit, 0.)
        self.assertLess(limit, 0.475)
        self.assertFalse(contact.path_safe(positions, endpoint))
        contact.validate_state(positions + limit * (endpoint - positions))

    def test_aligned_multitriangle_layers_have_conservative_ccd_bound(self):
        layer = np.array([[0., 0., 0.], [.01, 0., 0.], [.01, .01, 0.], [0., .01, 0.],
                          [.005, .005, 0.]])
        layer_faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        positions = np.vstack([layer, layer + [0., 0., .002]])
        faces = np.vstack([layer_faces, layer_faces + len(layer)])
        contact = self.contact(positions, faces)
        endpoint = positions.copy()
        endpoint[:len(layer), 2] += .25 / 240
        endpoint[len(layer):, 2] -= .25 / 240
        limit = contact.step_limit(positions, endpoint)
        analytic_first_contact = (.002 - .0001) / (.5 / 240)
        self.assertTrue(np.isfinite(limit))
        self.assertGreater(limit, 0.)
        self.assertLessEqual(limit, analytic_first_contact)
        stopped = positions + limit * (endpoint - positions)
        contact.validate_state(stopped)
        self.assertGreater(stopped[len(layer):, 2].min() - stopped[:len(layer), 2].max(), .0001)

    def test_gradient_finite_differences_use_vertex_major_order(self):
        positions, faces = self.fixture()
        contact = self.contact(positions, faces)
        gradient = contact.gradient(positions)
        epsilon = 1e-8
        numerical = np.zeros_like(positions)
        for vertex in range(len(positions)):
            for axis in range(3):
                plus, minus = positions.copy(), positions.copy()
                plus[vertex, axis] += epsilon
                minus[vertex, axis] -= epsilon
                numerical[vertex, axis] = (contact.energy(plus) - contact.energy(minus)) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=1e-10)
        np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-12)

    def test_nonmanifold_vertex_rejects(self):
        positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                              [-1., 0., 0.], [0., -1., 0.]])
        with self.assertRaises(ValueError):
            self.contact(positions, [[0, 1, 2], [0, 3, 4]])

    def test_inconsistent_shared_edge_orientation_rejects(self):
        positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
        with self.assertRaises(ValueError):
            self.contact(positions, [[0, 1, 2], [0, 1, 3]])

    def test_duplicate_and_repeated_vertex_faces_reject(self):
        positions, faces = self.fixture()
        for invalid in [np.vstack([faces, faces[0, ::-1]]), [[0, 0, 2], [3, 4, 5]],
                        [[0, 1, 6], [3, 4, 5]], faces.astype(float)]:
            with self.assertRaises(ValueError):
                self.contact(positions, invalid)

    def test_collision_cache_does_not_alias_caller(self):
        positions, faces = self.fixture()
        contact = self.contact(positions, faces)
        initial_energy = contact.energy(positions)
        positions[3:, 2] *= 0.75
        self.assertGreater(contact.energy(positions), initial_energy)
        positions[3:, 2] = 0.00005
        with self.assertRaises(ValueError):
            contact.energy(positions)

    def test_search_metric_is_finite_symmetric_positive_semidefinite(self):
        positions, faces = self.fixture()
        matrix = self.contact(positions, faces).hessian(positions).toarray()
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-10)


if __name__ == "__main__":
    unittest.main()
