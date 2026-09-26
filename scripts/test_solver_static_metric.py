"""Independent algebraic checks of a numerical search-metric congruence.

These tests establish neither physical equilibrium nor garment acceptance.
Rational elimination supplies the solution oracle; no native garment fixture
or time integrator participates in these small matrix checks.
"""
from fractions import Fraction as F
import unittest

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags

from solver_global_sewing import _positive_definite_direction
from solver_static_equilibrium import _scale_metric


def rational_direction(matrix, gradient):
    """Solve A p = -g by exact elimination, independently of sparse LU."""
    rows = [[F(float(value)) for value in row] + [-F(float(rhs))]
            for row, rhs in zip(matrix, gradient)]
    size = len(rows)
    for column in range(size):
        pivot = next(row for row in range(column, size) if rows[row][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for row in range(size):
            if row != column:
                factor = rows[row][column]
                rows[row] = [left - factor * right
                             for left, right in zip(rows[row], rows[column])]
    return [row[-1] for row in rows]


def buffer_snapshot(matrix, gradient):
    arrays = [gradient, matrix.data]
    arrays.extend([matrix.row, matrix.col] if matrix.format == 'coo'
                  else [matrix.indices, matrix.indptr])
    return [(array.dtype.str, array.shape, array.strides, array.tobytes())
            for array in arrays]


class StaticMetricTests(unittest.TestCase):
    def mapped_direction(self, matrix, gradient, shift=0.):
        scaled, rhs, inverse, stiffness = _scale_metric(matrix, gradient)
        shifted = scaled + diags(np.full(len(gradient), shift)) if shift else scaled
        direction = _positive_definite_direction(shifted, rhs)
        self.assertIsNotNone(direction)
        self.assertEqual(direction.dtype, np.dtype(np.float64))
        mapped = inverse * direction
        self.assertTrue(np.isfinite(mapped).all())
        return mapped, scaled, rhs, inverse, stiffness

    def assert_rational_solution(self, matrix, gradient, actual):
        expected = rational_direction(matrix, gradient)
        np.testing.assert_allclose(actual, [float(value) for value in expected],
                                   rtol=2e-13, atol=2e-14)
        # Check the original physical system, not only the transformed solve.
        for row, rhs in zip(matrix, gradient):
            residual = F(float(rhs)) + sum(F(float(a)) * F(float(p))
                                          for a, p in zip(row, actual))
            magnitude = max(F(1), abs(F(float(rhs))),
                            sum(abs(F(float(a)) * F(float(p)))
                                for a, p in zip(row, actual)))
            self.assertLessEqual(abs(residual), F(1, 10**12) * magnitude)

    def test_heterogeneous_diagonal_has_exact_unshifted_physical_solution(self):
        stiffness = np.array([2.**1000, 2.**-1000, 4.], dtype=np.float64)
        gradient = np.array([2.**500, 2.**-500, 3.], dtype=np.float64)
        matrix = diags(stiffness, format='csc')
        result, scaled, rhs, inverse, row_scale = self.mapped_direction(matrix, gradient)
        expected = np.array([-2.**-500, -2.**500, -.75])
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(scaled.toarray(), np.eye(3))
        np.testing.assert_array_equal(row_scale, stiffness)
        for diagonal, force, direction in zip(stiffness, gradient, result):
            self.assertEqual(F(float(diagonal)) * F(float(direction)) + F(float(force)), 0)
        self.assertTrue(np.isfinite(rhs).all())
        self.assertTrue(np.all(inverse > 0.))

    def test_coupled_solution_and_coordinate_permutation_use_rational_oracle(self):
        physical = np.array([[16., 2., -1.], [2., 4., 1.], [-1., 1., 1.]])
        gradient = np.array([1.25, -.75, 2.])
        oracle = rational_direction(physical, gradient)
        for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
            with self.subTest(order=order):
                indices = list(order)
                matrix = physical[np.ix_(indices, indices)]
                force = gradient[indices]
                result, *_ = self.mapped_direction(csc_matrix(matrix), force)
                self.assert_rational_solution(matrix, force, result)
                np.testing.assert_allclose(result, [float(oracle[i]) for i in indices],
                                           rtol=2e-13, atol=2e-14)
                self.assertLess(sum(F(float(a)) * F(float(b))
                                    for a, b in zip(force, result)), 0)

    def test_dimensionless_shift_maps_to_physical_diagonal_not_scalar_shift(self):
        physical = np.array([[16., 2.], [2., 4.]])
        gradient = np.array([3., -1.])
        # These row scales have exactly representable square roots/inverses.
        # The identity is exact for this witness, not a general rounding claim.
        for shift in (.5, 2.):
            with self.subTest(shift=shift):
                result, scaled, _, inverse, stiffness = self.mapped_direction(
                    csc_matrix(physical), gradient, shift)
                np.testing.assert_array_equal(inverse, [.25, .5])
                np.testing.assert_array_equal(stiffness, [16., 4.])
                shifted_physical = physical + np.diag([16. * shift, 4. * shift])
                self.assert_rational_solution(shifted_physical, gradient, result)
                wrong = rational_direction(physical + np.eye(2) * shift, gradient)
                self.assertGreater(max(abs(F(float(p)) - q)
                                       for p, q in zip(result, wrong)), F(1, 100))
                reconstructed = ((scaled.toarray() + np.eye(2) * shift)
                                 / inverse[:, None] / inverse[None, :])
                np.testing.assert_array_equal(reconstructed, shifted_physical)

    def test_nullspace_zero_rows_and_negative_curvature_are_not_removed(self):
        witnesses = [
            ([[4., 0.], [0., 0.]], [2., 1.], 1., [4., 1.]),
            ([[1., -1.], [-1., 1.]], [1., -1.], 1., [1., 1.]),
            ([[0., 1.], [1., 0.]], [1., 0.], 2., [1., 1.]),
            ([[0., 0.], [0., 0.]], [1., -2.], 1., [1., 1.]),
        ]
        for values, forces, shift, reference in witnesses:
            with self.subTest(matrix=values):
                physical, gradient = np.array(values), np.array(forces)
                scaled, rhs, _, stiffness = _scale_metric(csc_matrix(physical), gradient)
                self.assertEqual(scaled.shape, physical.shape)
                np.testing.assert_array_equal(stiffness, reference)
                self.assertIsNone(_positive_definite_direction(scaled, rhs))
                result, *_ = self.mapped_direction(csc_matrix(physical), gradient, shift)
                self.assert_rational_solution(physical + np.diag(reference) * shift,
                                              gradient, result)
                self.assertLess(sum(F(float(a)) * F(float(b))
                                    for a, b in zip(gradient, result)), 0)

    def test_malformed_nonfinite_and_asymmetric_systems_are_rejected(self):
        good = csc_matrix(np.eye(2))
        gradient = np.array([1., -1.])
        invalid = [
            (np.eye(2), gradient),
            (csc_matrix(np.eye(2, dtype=np.float32)), gradient),
            (csc_matrix(np.eye(2, dtype=np.int64)), gradient),
            (csc_matrix(np.ones((2, 3))), gradient),
            (csc_matrix((0, 0), dtype=np.float64), np.empty(0)),
            (good, [1., -1.]),
            (good, gradient.astype(np.float32)),
            (good, gradient.reshape(2, 1)),
            (good, np.array([1.])),
            (good, np.array([np.inf, 1.])),
            (good, np.array([np.nan, 1.])),
            (csc_matrix([[np.inf, 0.], [0., 1.]]), gradient),
            (csc_matrix([[1., np.nan], [np.nan, 1.]]), gradient),
            (csc_matrix([[2., 1.], [0., 2.]]), gradient),
        ]
        for index, (matrix, rhs) in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                _scale_metric(matrix, rhs)

    def test_extreme_finite_inputs_cannot_silently_lose_nonzero_values(self):
        smallest = float.fromhex('0x0.0000000000001p-1022')
        invalid = [
            (csc_matrix([[2.**-1000]]), np.array([2.**1000])),
            (csc_matrix([[2.**1000]]), np.array([smallest])),
            (csc_matrix([[2.**1000, 2.**-1000], [2.**-1000, 2.**1000]]),
             np.array([1., 1.])),
        ]
        for index, (matrix, gradient) in enumerate(invalid):
            before = buffer_snapshot(matrix, gradient)
            with self.subTest(case=index), self.assertRaises(ValueError):
                _scale_metric(matrix, gradient)
            self.assertEqual(buffer_snapshot(matrix, gradient), before)

    def test_sparse_inputs_stay_unchanged_and_outputs_are_detached(self):
        physical = [[4., 1.], [1., 1.]]
        matrices = [csc_matrix(physical), csr_matrix(physical), coo_matrix(physical),
                    # Unsorted duplicate first-row entries still mean H[0,0]=4.
                    csr_matrix((np.array([1., 3., 1., 1., 1.]),
                                np.array([1, 0, 0, 0, 1]), np.array([0, 3, 5])),
                               shape=(2, 2))]
        for matrix in matrices:
            with self.subTest(format=matrix.format, stored_entries=matrix.nnz):
                gradient = np.array([2., 3.])
                # Immutable input storage catches hidden sparse canonicalization.
                arrays = [matrix.data, gradient]
                arrays += ([matrix.row, matrix.col] if matrix.format == 'coo'
                           else [matrix.indices, matrix.indptr])
                for array in arrays:
                    array.flags.writeable = False
                before = buffer_snapshot(matrix, gradient)
                scaled, rhs, inverse, stiffness = _scale_metric(matrix, gradient)
                self.assertEqual(buffer_snapshot(matrix, gradient), before)
                np.testing.assert_array_equal(scaled.toarray(), [[1., .5], [.5, 1.]])
                np.testing.assert_array_equal(rhs, [1., 3.])
                np.testing.assert_array_equal(inverse, [.5, 1.])
                np.testing.assert_array_equal(stiffness, [4., 1.])
                scaled.data[:] = 7.
                rhs[:] = 8.
                inverse[:] = 9.
                stiffness[:] = 10.
                self.assertEqual(buffer_snapshot(matrix, gradient), before)


if __name__ == '__main__':
    unittest.main()
