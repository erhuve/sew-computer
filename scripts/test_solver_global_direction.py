import unittest

import numpy as np
from scipy.sparse import csr_matrix, eye

from solver_global_sewing import _direct_descent, _positive_definite_direction


class GlobalDirectionTests(unittest.TestCase):
    def test_energy_change_rejects_uphill_step_hidden_by_constant(self):
        evaluate, quadratic = self.quadratic(np.eye(1), np.zeros(1))

        def energy_change(start, end):
            delta = end - start
            return float((start + .5 * delta) @ delta)

        result = _direct_descent(evaluate, np.ones(1), 2, lambda positions: eye(1) * .1,
                                 lambda positions: 1e20 + quadratic(positions),
                                 energy_change_function=energy_change)
        self.assertFalse(result.success)
        np.testing.assert_array_equal(result.x, np.ones(1))
        self.assertEqual(result.projected_steps, 0)
        self.assertEqual(result.nfev, 2)

    def test_tiny_represented_descent_converges_despite_constant_energy(self):
        target = np.array([np.nextafter(1., 0.)])
        evaluate, quadratic = self.quadratic(np.eye(1) * 1e12, target)
        changes = []

        def energy_change(start, end):
            delta = end - start
            change = float(1e12 * (start - target + .5 * delta) @ delta)
            changes.append(change)
            return change

        result = _direct_descent(evaluate, np.ones(1), 3, lambda positions: eye(1) * 1e12,
                                 lambda positions: 1e20 + quadratic(positions),
                                 energy_change_function=energy_change)
        self.assertTrue(result.success)
        np.testing.assert_array_equal(result.x, target)
        self.assertEqual(result.energy_history, [1e20, 1e20])
        self.assertLess(changes[0], 0)

    def test_unrepresented_step_never_counts_as_descent(self):
        evaluate, objective = self.quadratic(np.eye(1), np.zeros(1))
        result = _direct_descent(evaluate, np.ones(1), 3, lambda positions: eye(1) * 1e100,
                                 objective, energy_change_function=lambda start, end: -1.)
        self.assertFalse(result.success)
        self.assertEqual(result.projected_steps, 0)
        np.testing.assert_array_equal(result.x, np.ones(1))

    def quadratic(self, matrix, target):
        factor = np.linalg.cholesky(matrix).T

        def evaluate(positions, jacobian=False):
            return csr_matrix(factor) if jacobian else factor @ (positions - target)

        def objective(positions):
            residual = factor @ (positions - target)
            return float(residual @ residual / 2)

        return evaluate, objective

    def test_exact_quadratic_direction_reaches_analytical_solution(self):
        matrix = np.array([[4., 1.], [1., 2.]])
        target = np.array([.3, -.7])
        evaluate, objective = self.quadratic(matrix, target)
        result = _direct_descent(evaluate, np.array([4., -3.]), 5,
                                 lambda positions: eye(2) * 20, objective,
                                 exact_hessian=lambda positions: csr_matrix(matrix))
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, target, atol=1e-12)
        self.assertEqual(result.exact_steps, 1)
        self.assertEqual(result.projected_steps, 0)
        self.assertEqual(result.nfev, 2)

    def test_ascent_and_singular_exact_directions_use_projected_fallback(self):
        evaluate, objective = self.quadratic(np.eye(2), np.zeros(2))
        for matrix in (np.diag([-1., 1.]), np.zeros((2, 2)), np.full((2, 2), np.nan)):
            with self.subTest(matrix=matrix.tolist()):
                result = _direct_descent(evaluate, np.array([1., 0.]), 6,
                                         lambda positions: eye(2), objective,
                                         exact_hessian=lambda positions: csr_matrix(matrix))
                self.assertTrue(result.success)
                np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-12)
                self.assertEqual(result.exact_steps, 0)
                self.assertEqual(result.projected_steps, 1)
                self.assertLessEqual(result.nfev, 6)

    def test_failed_exact_line_search_retries_projected_direction(self):
        evaluate, objective = self.quadratic(np.eye(1), np.zeros(1))
        result = _direct_descent(evaluate, np.ones(1), 40,
                                 lambda positions: eye(1), objective,
                                 exact_hessian=lambda positions: eye(1) * 2. ** -30)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, np.zeros(1), atol=1e-12)
        self.assertEqual(result.exact_steps, 0)
        self.assertEqual(result.projected_steps, 1)
        self.assertGreater(result.nfev, 2)
        self.assertLessEqual(result.nfev, 40)
        self.assertTrue(np.all(np.diff(result.energy_history) < 0))

    def test_failed_line_search_respects_evaluation_budget(self):
        evaluate, objective = self.quadratic(np.eye(1), np.zeros(1))
        for budget in (1, 2, 7):
            with self.subTest(budget=budget):
                result = _direct_descent(evaluate, np.ones(1), budget,
                                         lambda positions: eye(1), objective,
                                         exact_hessian=lambda positions: eye(1) * 2. ** -30)
                self.assertFalse(result.success)
                np.testing.assert_array_equal(result.x, np.ones(1))
                self.assertEqual(result.nfev, budget)
                self.assertEqual(result.energy_history, [.5])
                self.assertEqual(result.exact_steps, 0)
                self.assertEqual(result.projected_steps, 0)

    def test_no_descent_direction_does_not_report_success(self):
        evaluate, objective = self.quadratic(np.eye(1), np.zeros(1))
        result = _direct_descent(evaluate, np.ones(1), 8,
                                 lambda positions: -eye(1), objective,
                                 exact_hessian=lambda positions: -eye(1))
        self.assertFalse(result.success)
        np.testing.assert_array_equal(result.x, np.ones(1))
        self.assertEqual(result.energy_history, [.5])

    def test_stationary_initial_state_needs_no_direction(self):
        evaluate, objective = self.quadratic(np.eye(2), np.zeros(2))

        def unavailable(positions):
            self.fail("Stationary state must not request a search direction")

        result = _direct_descent(evaluate, np.zeros(2), 5, unavailable,
                                 objective, exact_hessian=unavailable)
        self.assertTrue(result.success)
        self.assertEqual(result.nfev, 1)
        self.assertEqual(result.exact_steps, 0)
        self.assertEqual(result.projected_steps, 0)

    def test_sparse_positive_definiteness_against_dense_spectrum(self):
        random = np.random.default_rng(409)
        for dimension in (3, 8, 20):
            for index in range(60):
                with self.subTest(dimension=dimension, sample=index):
                    basis, _ = np.linalg.qr(random.normal(size=(dimension, dimension)))
                    eigenvalues = np.geomspace(1e-4, 1e3, dimension)
                    if index % 2:
                        eigenvalues[0] *= -1
                    matrix = (basis * eigenvalues) @ basis.T
                    gradient = np.ones(dimension)
                    direction = _positive_definite_direction(csr_matrix(matrix), gradient)
                    if index % 2:
                        self.assertLess(np.linalg.eigvalsh(matrix)[0], 0)
                        self.assertIsNone(direction)
                    else:
                        self.assertIsNotNone(direction)
                        np.testing.assert_allclose(matrix @ direction, -gradient, atol=1e-8)
                        self.assertLess(float(gradient @ direction), 0)

    def test_indefinite_descent_is_rejected(self):
        matrix = np.diag([-1., 1.])
        gradient = np.array([0., 1.])
        self.assertLess(float(gradient @ np.linalg.solve(matrix, -gradient)), 0)
        self.assertIsNone(_positive_definite_direction(csr_matrix(matrix), gradient))
        evaluate, objective = self.quadratic(np.eye(2), np.zeros(2))
        result = _direct_descent(evaluate, gradient, 5, lambda positions: eye(2), objective,
                                 exact_hessian=lambda positions: csr_matrix(matrix))
        self.assertTrue(result.success)
        self.assertEqual(result.exact_steps, 0)
        self.assertEqual(result.projected_steps, 1)

    def test_asymmetric_and_nonfinite_metrics_are_rejected(self):
        for matrix in (np.array([[2., .5], [.1, 2.]]),
                       np.array([[1., np.inf], [np.inf, 1.]]),
                       np.array([[1., np.nan], [np.nan, 1.]])):
            with self.subTest(matrix=matrix.tolist()):
                self.assertIsNone(_positive_definite_direction(csr_matrix(matrix), np.ones(2)))

    def test_tiny_and_ill_conditioned_positive_pivots_fall_back(self):
        for matrix in (np.eye(2) * 1e-14, np.diag([1e-14, 1.]), np.diag([1., 1e14])):
            with self.subTest(matrix=matrix.tolist()):
                self.assertGreater(np.linalg.eigvalsh(matrix)[0], 0)
                self.assertIsNone(_positive_definite_direction(csr_matrix(matrix), np.ones(2)))
                evaluate, objective = self.quadratic(np.eye(2), np.zeros(2))
                result = _direct_descent(evaluate, np.ones(2), 5, lambda positions: eye(2), objective,
                                         exact_hessian=lambda positions: csr_matrix(matrix))
                self.assertTrue(result.success)
                self.assertEqual(result.exact_steps, 0)
                self.assertEqual(result.projected_steps, 1)


if __name__ == "__main__":
    unittest.main()
