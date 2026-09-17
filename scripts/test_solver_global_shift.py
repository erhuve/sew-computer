import unittest

import numpy as np
from scipy.sparse import csr_matrix, diags

from solver_global_shift import shifted_positive_definite_direction


class GlobalShiftTests(unittest.TestCase):
    def test_exact_spd_matches_dense_solution_without_shift(self):
        matrix = np.array([[4., 1.], [1., 2.]])
        gradient = np.array([3., -2.])
        direction, report = shifted_positive_definite_direction(
            csr_matrix(matrix), gradient, np.array([2., 3.]))
        np.testing.assert_allclose(direction, np.linalg.solve(matrix, -gradient), atol=1e-12)
        self.assertEqual(report["lambda"], 0.)
        self.assertEqual(len(report["attempts"]), 1)
        self.assertFalse(report["fallbackRequired"])

    def test_indefinite_and_singular_metrics_match_shifted_dense_solve(self):
        gradient = np.array([1., -2.])
        inertia = np.array([2., 3.])
        for matrix in (np.diag([-10., 1.]), np.array([[1., 1.], [1., 1.]])):
            with self.subTest(matrix=matrix.tolist()):
                direction, report = shifted_positive_definite_direction(csr_matrix(matrix), gradient, inertia)
                self.assertGreater(report["lambda"], 0)
                shifted = matrix + report["lambda"] * np.diag(inertia)
                self.assertGreater(np.linalg.eigvalsh(shifted)[0], 0)
                np.testing.assert_allclose(direction, np.linalg.solve(shifted, -gradient), atol=1e-12)
                self.assertLess(float(gradient @ direction), 0)
                self.assertLessEqual(len(report["attempts"]), 12)

    def test_mass_scaled_shift_preserves_zero_momentum_direction(self):
        inertia = np.array([2., 3.])
        internal = -10 * np.array([[1., -1.], [-1., 1.]])
        matrix = np.diag(inertia) + internal
        direction, report = shifted_positive_definite_direction(
            csr_matrix(matrix), np.array([1., -1.]), inertia)
        self.assertGreater(report["lambda"], 0)
        self.assertAlmostEqual(float(inertia @ direction), 0, places=13)
        shifted = matrix + report["lambda"] * np.diag(inertia)
        np.testing.assert_allclose(shifted @ np.ones(2), (1 + report["lambda"]) * inertia)

    def test_trial_cap_requests_caller_fallback(self):
        direction, report = shifted_positive_definite_direction(
            diags([-1e20, 1.]), np.ones(2), np.ones(2))
        self.assertIsNone(direction)
        self.assertIsNone(report["lambda"])
        self.assertTrue(report["fallbackRequired"])
        self.assertEqual(len(report["attempts"]), 12)
        self.assertFalse(any(attempt["accepted"] for attempt in report["attempts"]))

    def test_malformed_inputs_rejected(self):
        cases = [
            (np.ones((2, 3)), np.ones(2), np.ones(2)),
            (np.eye(2), np.ones(3), np.ones(2)),
            (np.eye(2), np.ones(2), np.ones(3)),
            (np.eye(2), np.array([np.nan, 1.]), np.ones(2)),
            (np.eye(2), np.ones(2), np.array([0., 1.])),
            (np.eye(2), np.ones(2), np.array([-1., 1.])),
            (np.eye(2), np.ones(2), np.array([np.inf, 1.])),
            (np.diag([np.inf, 1.]), np.ones(2), np.ones(2)),
            (np.array([[2., 1.], [0., 2.]]), np.ones(2), np.ones(2)),
            (np.empty((0, 0)), np.empty(0), np.empty(0)),
        ]
        for matrix, gradient, inertia in cases:
            with self.subTest(shape=matrix.shape, gradient=gradient.tolist(), inertia=inertia.tolist()):
                with self.assertRaises(ValueError):
                    shifted_positive_definite_direction(csr_matrix(matrix), gradient, inertia)

    def test_inputs_are_unchanged(self):
        matrix = csr_matrix(np.diag([-10., 1.]))
        gradient, inertia = np.array([1., -2.]), np.array([2., 3.])
        original_matrix, original_gradient, original_inertia = matrix.copy(), gradient.copy(), inertia.copy()
        shifted_positive_definite_direction(matrix, gradient, inertia)
        np.testing.assert_array_equal(matrix.toarray(), original_matrix.toarray())
        np.testing.assert_array_equal(gradient, original_gradient)
        np.testing.assert_array_equal(inertia, original_inertia)


if __name__ == "__main__":
    unittest.main()
