from math import comb
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult

from solver_hinge_sweep import _dot, _product, _split, hinge_sweep_safe


def polynomial_value(coefficients, fraction):
    degree = coefficients.shape[1] - 1
    return sum(comb(degree, index) * fraction ** index * (1 - fraction) ** (degree - index)
               * coefficients[:, index] for index in range(degree + 1))


class HingeSweepAdversarialTests(unittest.TestCase):
    def test_bernstein_geometry_matches_direct_affine_geometry(self):
        generator = np.random.default_rng(91823)
        points = generator.normal(size=(17, 2, 4, 3))
        first = _product(points[:, :, 2] - points[:, :, 0], points[:, :, 3] - points[:, :, 0], np.cross)
        second = _product(points[:, :, 3] - points[:, :, 1], points[:, :, 2] - points[:, :, 1], np.cross)
        cosine = _product(first, second, _dot)
        sine = _product(_product(first, second, np.cross), points[:, :, 3] - points[:, :, 2], _dot)
        for fraction in (0., .013, .271, .5, .899, 1.):
            current = points[:, 0] * (1 - fraction) + points[:, 1] * fraction
            first_direct = np.cross(current[:, 2] - current[:, 0], current[:, 3] - current[:, 0])
            second_direct = np.cross(current[:, 3] - current[:, 1], current[:, 2] - current[:, 1])
            assert_allclose(polynomial_value(cosine, fraction), _dot(first_direct, second_direct), atol=2e-13)
            assert_allclose(polynomial_value(sine, fraction),
                            _dot(np.cross(first_direct, second_direct), current[:, 3] - current[:, 2]), atol=2e-13)
            left, right = _split(sine)
            assert_allclose(polynomial_value(left, fraction), polynomial_value(sine, fraction / 2), atol=2e-13)
            assert_allclose(polynomial_value(right, fraction), polynomial_value(sine, .5 + fraction / 2), atol=2e-13)

    def test_off_grid_tangential_triangle_collapse_without_normal_sign_change(self):
        for root in (.137391, .371283, .823717):
            def points(fraction):
                offset = fraction - root
                return np.array([[1 + offset, offset, 0.], [0., 1., 0.],
                                 [0., 0., 0.], [1., offset, 0.]])
            for fraction in (0., 1.):
                current = points(fraction)
                normal = np.cross(current[2] - current[0], current[3] - current[0])
                self.assertLess(normal[2], 0.)
            current = points(root)
            assert_allclose(np.cross(current[2] - current[0], current[3] - current[0]), 0., atol=0.)
            self.assertFalse(hinge_sweep_safe(points(0), points(1), [[0, 1, 2, 3]]))

    def test_random_certified_paths_have_no_sampled_branch_or_degeneracy(self):
        generator = np.random.default_rng(512971)
        certified = 0
        for _ in range(80):
            start, end = generator.normal(size=(2, 4, 3))
            if not hinge_sweep_safe(start, end, [[0, 1, 2, 3]], max_intervals=4096):
                continue
            certified += 1
            fractions = np.linspace(0, 1, 1001)[:, None, None]
            points = start + fractions * (end - start)
            first = np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0])
            second = np.cross(points[:, 3] - points[:, 1], points[:, 2] - points[:, 1])
            edge = points[:, 3] - points[:, 2]
            angles = np.arctan2(_dot(np.cross(first, second), edge) / np.linalg.norm(edge, axis=1),
                                _dot(first, second))
            self.assertLess(np.max(np.abs(np.diff(angles))), np.pi)
            self.assertGreater(np.linalg.norm(first, axis=1).min(), 1e-8)
            self.assertGreater(np.linalg.norm(second, axis=1).min(), 1e-8)
        self.assertGreater(certified, 5)

    def test_global_solver_checks_optimizer_and_physical_segments(self):
        from solver_fold_barrier import LocalAngularFoldBarrier
        from test_solver_bending_integration import BendingIntegrationTests

        for failed_segment in (0, 1):
            solver, previous = BendingIntegrationTests().fixture()
            solver.fold_barrier = LocalAngularFoldBarrier(len(previous), solver.bending.indices)
            checked = []

            def inspect_sweep(start, end, indices):
                checked.append((start.copy(), end.copy()))
                return len(checked) - 1 != failed_segment

            def inspect_descent(evaluate, start, maximum, hessian, objective, **kwargs):
                intermediate = start.copy()
                intermediate[2] += .001
                candidate = intermediate.copy()
                candidate[5] += .001
                self.assertTrue(np.isinf(kwargs['energy_change_function'](intermediate, candidate)))
                assert_allclose(checked[0][0].ravel(), intermediate)
                if failed_segment == 1:
                    assert_allclose(checked[1][0], previous)
                    assert_allclose(checked[1][1], checked[0][1])
                return OptimizeResult(x=start, success=False, status=0, message='adversarial probe', nfev=1)

            with patch('solver_hinge_sweep.hinge_sweep_safe', inspect_sweep), \
                    patch('solver_global_sewing._direct_descent', inspect_descent):
                solver.step(previous, np.zeros_like(previous), np.zeros((1, 3)), .01)
            self.assertEqual(len(checked), failed_segment + 1)


if __name__ == '__main__':
    unittest.main()
