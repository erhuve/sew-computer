import unittest

import numpy as np
from scipy.sparse import csr_matrix, eye

from solver_global_sewing import _direct_descent


class BendingSearchTests(unittest.TestCase):
    def quadratic(self, positions, jacobian=False):
        return eye(len(positions), format='csr') if jacobian else positions.copy()

    def solve(self, metric, start=None, budget=50, energy_change=None):
        if start is None:
            start = np.array([0., 1.])
        return _direct_descent(self.quadratic, start, budget,
            lambda positions: eye(len(positions), format='csr'),
            lambda positions: float(positions @ positions / 2),
            coupled_hessian=lambda positions: csr_matrix(metric),
            energy_change_function=energy_change)

    def test_spd_coupled_direction_named_without_exact_claim(self):
        result = self.solve(np.eye(2))
        self.assertTrue(result.success)
        self.assertEqual(result.coupled_steps, 1)
        self.assertEqual(result.exact_steps, 0)
        self.assertEqual(result.projected_steps, 0)
        self.assertEqual(result.direction_history[0]['method'], 'coupled')
        np.testing.assert_allclose(result.x, 0, atol=1e-14)

    def test_invalid_primary_metric_falls_back_including_indefinite_descent(self):
        for matrix in (np.diag([-1., 1.]), np.zeros((2, 2)), np.array([[1., .1], [0., 1.]]),
                       np.full((2, 2), np.nan), np.diag([1., np.inf])):
            with self.subTest(matrix=matrix):
                result = self.solve(matrix)
                self.assertTrue(result.success)
                self.assertEqual(result.coupled_steps, 0)
                self.assertEqual(result.exact_steps, 0)
                self.assertEqual(result.projected_steps, 1)

    def test_coupled_line_search_failure_retries_projected_metric(self):
        result = self.solve(np.eye(2) * 2. ** -30)
        self.assertTrue(result.success)
        self.assertEqual(result.coupled_steps, 0)
        self.assertEqual(result.projected_steps, 1)
        self.assertGreater(result.nfev, 24)
        self.assertLessEqual(result.nfev, 50)

    def test_shared_budget_and_candidate_domain_guard(self):
        calls = []

        def forbidden(start, end):
            calls.append((start.copy(), end.copy()))
            return float('inf')

        for budget in (2, 9, 40):
            result = self.solve(np.eye(2), budget=budget, energy_change=forbidden)
            self.assertFalse(result.success)
            self.assertEqual(result.nfev, budget)
            self.assertEqual(result.coupled_steps, 0)
            self.assertEqual(result.projected_steps, 0)
            np.testing.assert_array_equal(result.x, [0., 1.])
        self.assertTrue(calls)

    def test_mutually_exclusive_primary_metrics(self):
        with self.assertRaises(ValueError):
            _direct_descent(self.quadratic, np.ones(2), 10, lambda positions: eye(2),
                            lambda positions: float(positions @ positions / 2),
                            exact_hessian=lambda positions: eye(2), coupled_hessian=lambda positions: eye(2))


if __name__ == '__main__':
    unittest.main()
