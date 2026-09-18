import unittest

import numpy as np
from numpy.testing import assert_allclose

from solver_fold_barrier import LocalAngularFoldBarrier


class LocalFoldBarrierTests(unittest.TestCase):
    def points(self, angle):
        return np.array([[.2, 1., 0.], [.7, -np.cos(angle), -np.sin(angle)],
                         [0., 0., 0.], [1., 0., 0.]])

    def barrier(self, **kwargs):
        return LocalAngularFoldBarrier(4, [[0, 1, 2, 3]], **kwargs)

    def test_scalar_energy_and_near_fold_growth(self):
        barrier = self.barrier(stiffness_joules=.3)
        for angle in (-3., -2., -.2, 0., .2, 2., 3.):
            gap = (np.pi - abs(angle)) / (np.pi / 2)
            expected = -.3 * (gap - 1) ** 2 * np.log(gap) if gap < 1 else 0.
            self.assertAlmostEqual(barrier.energy(self.points(angle)), expected, places=13)
        self.assertGreater(barrier.energy(self.points(np.pi - 1e-6)),
                           barrier.energy(self.points(np.pi - 1e-3)))

    def test_force_by_independent_finite_difference(self):
        barrier = self.barrier(stiffness_joules=.7)
        generator = np.random.default_rng(1487)
        for angle in (-2.5, 2.1, 2.7):
            points = self.points(angle) + generator.normal(size=(4, 3)) * .02
            numerical = np.empty_like(points)
            for vertex in range(4):
                for axis in range(3):
                    delta = np.zeros_like(points)
                    delta[vertex, axis] = 1e-6
                    numerical[vertex, axis] = (barrier.energy(points + delta)
                                               - barrier.energy(points - delta)) / 2e-6
            assert_allclose(barrier.gradient(points), numerical, atol=3e-8, rtol=2e-7)

    def test_residual_jacobian_and_energy_identity(self):
        barrier = self.barrier()
        for angle in (-2.5, .2, 2.5):
            points = self.points(angle)
            residual = barrier.residual(points)
            self.assertAlmostEqual(.5 * float(residual @ residual), barrier.energy(points), places=14)
            assert_allclose((barrier.jacobian(points).T @ residual).reshape(4, 3),
                            barrier.gradient(points), atol=1e-14)
            direction = np.random.default_rng(91).normal(size=(4, 3))
            numerical = (barrier.residual(points + 1e-6 * direction)
                         - barrier.residual(points - 1e-6 * direction)) / 2e-6
            assert_allclose(barrier.jacobian(points) @ direction.ravel(), numerical, rtol=1e-7, atol=1e-9)

    def test_activation_is_twice_continuous(self):
        barrier = self.barrier(activation_angle=1.4)
        for sign in (-1, 1):
            points = self.points(sign * (1.4 - 1e-8))
            self.assertEqual(barrier.energy(points), 0.)
            self.assertEqual(barrier.jacobian(points).shape, (len(barrier.indices), 12))
            assert_allclose(barrier.gradient(points), 0., atol=0.)
            self.assertEqual(barrier.hessian(points).nnz, 0)
            points = self.points(sign * (1.4 + 1e-8))
            self.assertLess(barrier.energy(points), 1e-23)
            self.assertLess(np.max(np.abs(barrier.gradient(points))), 1e-14)
            self.assertLess(np.max(np.abs(barrier.hessian(points).data)), 1e-6)

    def test_rigid_motion_mirror_force_and_torque(self):
        barrier = self.barrier()
        points = self.points(2.3)
        rotation = np.linalg.qr(np.random.default_rng(99).normal(size=(3, 3)))[0]
        for transform in (rotation, rotation @ np.diag([-1., 1., 1.])):
            moved = points @ transform + [4., -2., .7]
            self.assertAlmostEqual(barrier.energy(moved), barrier.energy(points), places=13)
            assert_allclose(barrier.gradient(moved), barrier.gradient(points) @ transform, atol=1e-12)
            blocks = np.kron(np.eye(4), transform)
            assert_allclose(barrier.hessian(moved).toarray(),
                            blocks.T @ barrier.hessian(points).toarray() @ blocks, atol=1e-11)
        assert_allclose(barrier.gradient(points).sum(axis=0), 0., atol=1e-14)
        assert_allclose(np.cross(points, barrier.gradient(points)).sum(axis=0), 0., atol=1e-14)

    def test_metric_psd_coupled_and_not_exact_hessian(self):
        barrier = self.barrier()
        points = self.points(2.4)
        metric = barrier.hessian(points).toarray()
        self.assertGreater(np.linalg.norm(metric[:3, 3:6]), .1)
        self.assertGreaterEqual(np.linalg.eigvalsh(metric).min(), -1e-12)
        direction = np.random.default_rng(31).normal(size=(4, 3))
        exact_action = (barrier.gradient(points + 1e-6 * direction)
                        - barrier.gradient(points - 1e-6 * direction)).ravel() / 2e-6
        self.assertGreater(np.linalg.norm(metric @ direction.ravel() - exact_action), .1)

    def test_stable_energy_change_and_activation_crossing(self):
        barrier = self.barrier()
        for start_angle, end_angle in ((2.3, 2.3 + 1e-10), (-2.3, -2.3 - 1e-10),
                                       (1.5, 1.7), (1.7, 1.5), (-1.7, -1.5)):
            start, end = self.points(start_angle), self.points(end_angle)
            change = barrier.energy_change(start, end)
            self.assertAlmostEqual(change, -barrier.energy_change(end, start), places=15)
            assert_allclose(change, barrier.energy(end) - barrier.energy(start), atol=1e-15)
            if abs(end_angle - start_angle) < 1e-8:
                linear_change = np.sum(barrier.gradient(start) * (end - start))
                assert_allclose(change, linear_change, rtol=1e-8, atol=1e-18)
        self.assertEqual(barrier.energy_change(self.points(2.3), self.points(2.3)), 0.)

    def test_invalid_inputs_and_geometry_reject(self):
        for kwargs in ({"activation_angle": 0.}, {"activation_angle": np.pi},
                       {"activation_angle": np.nan}, {"stiffness_joules": -1.},
                       {"stiffness_joules": np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.barrier(**kwargs)
        barrier = self.barrier()
        with self.assertRaises(ValueError):
            barrier.energy(self.points(np.pi))
        with self.assertRaises(ValueError):
            barrier.energy_change(self.points(3.), self.points(-3.))
        points = self.points(2.3)
        points[0] = points[2]
        with self.assertRaises(ValueError):
            barrier.energy(points)
        with self.assertRaises(ValueError):
            barrier.stiffness_joules[0] = 7.

    def test_empty_and_zero_stiffness(self):
        empty = LocalAngularFoldBarrier(4, np.empty((0, 4), dtype=int))
        zero = self.barrier(stiffness_joules=0.)
        for barrier in (empty, zero):
            points = self.points(2.3)
            self.assertEqual(barrier.energy(points), 0.)
            assert_allclose(barrier.gradient(points), 0.)
            self.assertEqual(barrier.hessian(points).nnz, 0)
            self.assertEqual(barrier.energy_change(points, self.points(2.4)), 0.)


if __name__ == '__main__':
    unittest.main()
