import unittest
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from solver_bending import ElasticDihedralBending


class BendingTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0.2, 0.8, 0.3], [0.6, -0.7, -0.1], [0., 0., 0.], [1., 0., 0.]])
        self.indices = np.array([[0, 1, 2, 3]])
        self.bending = ElasticDihedralBending(4, self.indices, [0.1], [1.], [3.])

    def test_closed_form_angle_and_energy(self):
        angle = np.arctan2(0.3, 0.8) + np.arctan2(-0.1, 0.7)
        assert_allclose(self.bending.angles(self.positions), [-angle], atol=1e-15)
        self.assertAlmostEqual(self.bending.energy(self.positions), 1.5 * (-angle - 0.1) ** 2)

    def test_analytic_gradient_random_hinges(self):
        generator = np.random.default_rng(773)
        for _ in range(12):
            positions = self.positions + generator.normal(0, 0.06, (4, 3))
            numeric = np.zeros((4, 3))
            for vertex in range(4):
                for axis in range(3):
                    plus, minus = positions.copy(), positions.copy()
                    plus[vertex, axis] += 1e-6
                    minus[vertex, axis] -= 1e-6
                    numeric[vertex, axis] = (self.bending.energy(plus) - self.bending.energy(minus)) / 2e-6
            assert_allclose(self.bending.gradient(positions), numeric, atol=2e-9, rtol=2e-7)

    def test_force_torque_and_rigid_covariance(self):
        rotation, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
        transformed = self.positions @ rotation.T + [0.2, -0.3, 0.7]
        gradient = self.bending.gradient(self.positions)
        assert_allclose(gradient.sum(axis=0), 0, atol=1e-14)
        assert_allclose(np.cross(self.positions, gradient).sum(axis=0), 0, atol=1e-14)
        assert_allclose(self.bending.gradient(transformed), gradient @ rotation.T, atol=1e-14)
        self.assertAlmostEqual(self.bending.energy(transformed), self.bending.energy(self.positions))

    def test_mirror_orientation_with_rest_sign(self):
        mirror = np.diag([-1., 1., 1.])
        reflected = ElasticDihedralBending(4, self.indices, [-0.1], [1.], [3.])
        assert_allclose(reflected.angles(self.positions @ mirror), -self.bending.angles(self.positions))
        self.assertAlmostEqual(reflected.energy(self.positions @ mirror), self.bending.energy(self.positions))
        assert_allclose(reflected.gradient(self.positions @ mirror), self.bending.gradient(self.positions) @ mirror)

    def test_edge_orientation_reversal(self):
        reversed_edge = ElasticDihedralBending(4, [[0, 1, 3, 2]], [-0.1], [1.], [3.])
        assert_allclose(reversed_edge.angles(self.positions), -self.bending.angles(self.positions))
        assert_allclose(reversed_edge.gradient(self.positions), self.bending.gradient(self.positions), atol=1e-14)
        self.assertAlmostEqual(reversed_edge.energy(self.positions), self.bending.energy(self.positions))

    def test_coupled_gauss_newton_metric(self):
        matrix = self.bending.hessian(self.positions).toarray()
        assert_allclose(matrix, matrix.T, atol=1e-15)
        self.assertGreater(np.linalg.norm(matrix[:3, 3:6]), 0.1)
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-13)
        for direction in np.eye(3):
            assert_allclose(matrix @ np.tile(direction, 4), 0, atol=1e-14)
        at_rest = ElasticDihedralBending(4, self.indices, self.bending.angles(self.positions), [1.], [3.])
        numeric = np.empty((12, 12))
        for dof in range(12):
            plus, minus = self.positions.copy(), self.positions.copy()
            plus.flat[dof] += 1e-6
            minus.flat[dof] -= 1e-6
            numeric[:, dof] = ((at_rest.gradient(plus) - at_rest.gradient(minus)) / 2e-6).ravel()
        assert_allclose(matrix, numeric, atol=2e-9)

    def test_energy_change_and_directional_precision(self):
        direction = -self.bending.gradient(self.positions)
        for scale in (1e-3, 1e-7, 1e-12):
            candidate = self.positions + scale * direction
            change = self.bending.energy_change(self.positions, candidate)
            self.assertLess(change, 0)
            if scale > 1e-10:
                self.assertAlmostEqual(change, self.bending.energy(candidate) - self.bending.energy(self.positions), delta=1e-15)
            else:
                predicted = np.sum(self.bending.gradient(self.positions) * (candidate - self.positions))
                assert_allclose(change, predicted, rtol=2e-6)
        self.assertEqual(self.bending.energy_change(self.positions, self.positions), 0.)

    def test_branch_and_degenerate_guards(self):
        folded = np.array([[0., 1., 0.], [0., 1., 1e-5], [0., 0., 0.], [1., 0., 0.]])
        crossed = folded.copy()
        crossed[1, 2] *= -1
        with self.assertRaisesRegex(ValueError, "branch"):
            self.bending.energy_change(folded, crossed)
        folded[1, 2] = 0
        with self.assertRaisesRegex(ValueError, "branch"):
            self.bending.energy(folded)
        collapsed = self.positions.copy()
        collapsed[0] = collapsed[2]
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            self.bending.energy(collapsed)

    def test_uniform_geometric_scaling(self):
        for scale in (1e-6, 1e6):
            assert_allclose(self.bending.angles(self.positions * scale), self.bending.angles(self.positions))
            assert_allclose(self.bending.gradient(self.positions * scale) * scale, self.bending.gradient(self.positions), atol=1e-14)

    def test_multiple_hinges_and_unused_vertex(self):
        bending = ElasticDihedralBending(5, np.repeat(self.indices, 2, axis=0), [0.1, 0.1], [1., 1.], 3.)
        positions = np.vstack((self.positions, [7., 8., 9.]))
        assert_allclose(bending.gradient(positions)[:4], 2 * self.bending.gradient(self.positions))
        assert_allclose(bending.gradient(positions)[4], 0.)
        assert_allclose(bending.hessian(positions).toarray()[:12, :12], 2 * self.bending.hessian(self.positions).toarray())

    def test_invalid_inputs(self):
        for stiffness in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                ElasticDihedralBending(4, self.indices, [0.], [1.], stiffness)
        for indices in ([[0, 0, 2, 3]], [[0, 1, 2, 4]], [[0, 1, 2, 2.5]]):
            with self.assertRaises(ValueError):
                ElasticDihedralBending(4, indices, [0.], [1.], 1.)
        with self.assertRaises(ValueError):
            self.bending.energy(np.full((4, 3), np.nan))

    def test_empty_model_and_boundary_edges(self):
        def array(values):
            return SimpleNamespace(numpy=lambda: np.asarray(values))
        model = SimpleNamespace(particle_mass=array([1.] * 4), edge_bending_properties=None)
        empty = ElasticDihedralBending.from_model(model)
        self.assertEqual(empty.energy(self.positions), 0.)
        assert_allclose(empty.gradient(self.positions), 0.)
        self.assertEqual(empty.hessian(self.positions).nnz, 0)
        self.assertEqual(empty.energy_change(self.positions, self.positions), 0.)
        model.edge_indices = array([[-1, 1, 2, 3], [0, 1, 2, 3]])
        model.edge_bending_properties = array([[3., 0.], [3., 0.]])
        model.edge_rest_angle = array([0.1, 0.1])
        model.edge_rest_length = array([1., 1.])
        imported = ElasticDihedralBending.from_model(model)
        assert_allclose(imported.gradient(self.positions), self.bending.gradient(self.positions))


if __name__ == "__main__":
    unittest.main()
