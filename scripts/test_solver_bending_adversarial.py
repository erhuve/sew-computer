import unittest

import numpy as np

from solver_bending import ElasticDihedralBending


class BendingAdversarialTests(unittest.TestCase):
    def points(self, angle=.7):
        return np.array([[.2, 1., 0.], [.7, -np.cos(angle), -np.sin(angle)],
                         [0., 0., 0.], [1., 0., 0.]])

    def bending(self, rest=.2, scale=1.):
        return ElasticDihedralBending(4, [[0, 1, 2, 3]], [rest], [scale], [2.3])

    def test_signed_angle_and_scalar_energy(self):
        for angle in (-2.9, -.7, 0., .4, 2.9):
            with self.subTest(angle=angle):
                bending = self.bending()
                np.testing.assert_allclose(bending.angles(self.points(angle)), [angle], atol=1e-14)
                self.assertAlmostEqual(bending.energy(self.points(angle)), 1.15 * (angle - .2) ** 2, places=13)

    def test_independent_random_finite_difference_gradient(self):
        rng = np.random.default_rng(917)
        bending = self.bending()
        for trial in range(8):
            points = self.points() + rng.normal(size=(4, 3)) * .12
            exact = bending.gradient(points)
            numerical = np.empty_like(points)
            for vertex in range(4):
                for axis in range(3):
                    perturbation = np.zeros_like(points)
                    perturbation[vertex, axis] = 1e-6
                    numerical[vertex, axis] = (bending.energy(points + perturbation)
                                               - bending.energy(points - perturbation)) / 2e-6
            with self.subTest(trial=trial):
                np.testing.assert_allclose(exact, numerical, rtol=2e-7, atol=2e-9)

    def test_rigid_and_mirror_covariance(self):
        points = self.points()
        bending = self.bending()
        rotation = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))[0]
        for transform in (rotation, rotation @ np.diag([-1., 1., 1.])):
            parity = np.linalg.det(transform)
            transformed = points @ transform + [5., -3., 2.]
            counterpart = self.bending(rest=.2 * parity)
            self.assertAlmostEqual(bending.energy(points), counterpart.energy(transformed), places=12)
            np.testing.assert_allclose(counterpart.gradient(transformed), bending.gradient(points) @ transform,
                                       rtol=2e-12, atol=2e-12)
            block_transform = np.kron(np.eye(4), transform)
            np.testing.assert_allclose(counterpart.hessian(transformed).toarray(),
                block_transform.T @ bending.hessian(points).toarray() @ block_transform,
                rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(bending.gradient(points).sum(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(np.cross(points, bending.gradient(points)).sum(axis=0), 0, atol=1e-14)

    def test_geometric_scaling_with_rest_length(self):
        points = self.points()
        reference = self.bending()
        for scale in (1e-6, 1e-3, 1e3, 1e6):
            bending = self.bending(scale=scale)
            with self.subTest(scale=scale):
                self.assertAlmostEqual(bending.energy(points * scale) / scale, reference.energy(points), places=12)
                np.testing.assert_allclose(bending.gradient(points * scale), reference.gradient(points), atol=2e-12)
                np.testing.assert_allclose(bending.hessian(points * scale).toarray() * scale,
                                           reference.hessian(points).toarray(), atol=2e-12)

    def test_gauss_newton_is_coupled_psd_but_not_exact_away_from_rest(self):
        points = self.points()
        bending = self.bending()
        metric = bending.hessian(points).toarray()
        self.assertGreater(np.linalg.norm(metric[:3, 3:6]), .1)
        self.assertGreaterEqual(np.linalg.eigvalsh(metric).min(), -1e-12)
        direction = np.random.default_rng(8).normal(size=(4, 3))
        derivative = (bending.gradient(points + 1e-6 * direction)
                      - bending.gradient(points - 1e-6 * direction)).ravel() / 2e-6
        self.assertGreater(np.linalg.norm(metric @ direction.ravel() - derivative), .1)
        at_rest = self.bending(rest=.7)
        derivative = (at_rest.gradient(points + 1e-6 * direction)
                      - at_rest.gradient(points - 1e-6 * direction)).ravel() / 2e-6
        np.testing.assert_allclose(at_rest.hessian(points) @ direction.ravel(), derivative, rtol=1e-7, atol=1e-8)

    def test_branch_and_collapsed_hinges_reject(self):
        bending = self.bending()
        for angle in (np.pi, -np.pi, np.pi - 1e-10):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                bending.energy(self.points(angle))
        with self.assertRaises(ValueError):
            bending.energy_change(self.points(3.1), self.points(-3.1))
        for vertex, target in ((3, 2), (0, 2), (1, 3)):
            points = self.points()
            points[vertex] = points[target]
            for scale in (1e-6, 1., 1e6):
                with self.subTest(vertex=vertex, scale=scale), self.assertRaises(ValueError):
                    bending.gradient(points * scale)

    def test_tiny_energy_change_and_antisymmetry(self):
        bending = self.bending()
        start, end = self.points(.7), self.points(.7 + 1e-10)
        actual_angle_change = np.longdouble(np.arctan2(-end[1, 2], -end[1, 1])) - np.longdouble(.7)
        expected = float(np.longdouble(2.3) * actual_angle_change * (.5 + actual_angle_change / 2))
        change = bending.energy_change(start, end)
        np.testing.assert_allclose(change, expected, rtol=1e-6, atol=1e-17)
        self.assertAlmostEqual(change, -bending.energy_change(end, start), places=24)


if __name__ == '__main__':
    unittest.main()
