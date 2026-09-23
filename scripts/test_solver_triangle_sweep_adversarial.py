"""Independent attack controls for affine material-frame nondegeneracy."""

import unittest
from unittest.mock import patch

import numpy as np

from solver_triangle_sweep import triangle_sweep_safe, verify_triangle_sweep_exact


class TriangleSweepAdversarialTests(unittest.TestCase):
    def test_exact_dyadic_tangent_collapses_at_extreme_binary_scales(self):
        for root in (.125, .375, .875):
            for exponent in (-500, 0, 500):
                with self.subTest(root=root, exponent=exponent):
                    # All endpoint coordinates and the root are dyadic, so the
                    # triangle normal really is -(t-root)^2, not a rounded
                    # decimal approximation with a potentially nonzero minimum.
                    def points(time):
                        delta = time - root
                        return np.ldexp(np.array([[0., 0., 0.], [1., delta, 0.],
                                                  [1. + delta, delta, 0.]]), exponent)
                    start, end = points(0.), points(1.)
                    self.assertFalse(triangle_sweep_safe(start, end, [[0, 1, 2]]))
                    with self.assertRaises(ValueError):
                        verify_triangle_sweep_exact(start, end, [[0, 1, 2]])

    def test_one_collapsing_face_cannot_hide_behind_safe_faces(self):
        base = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
        start = np.concatenate([base + [index, 0., 0.] for index in range(9)])
        faces = np.arange(len(start)).reshape((-1, 3))
        end = start.copy()
        end[faces[4]] = -base + [4., 0., 0.]
        for order in (np.arange(9), np.roll(np.arange(9), 4), np.roll(np.arange(9), -4)):
            with self.subTest(order=order.tolist()):
                self.assertFalse(triangle_sweep_safe(start, end, faces[order]))
                with self.assertRaises(ValueError):
                    verify_triangle_sweep_exact(start, end, faces[order])

    def test_exact_verifier_does_not_trust_the_numerical_direction_predicate(self):
        start = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        end = -start
        with patch("solver_hinge_sweep._nonzero", return_value=np.ones(1, dtype=bool)):
            with self.assertRaises(ValueError):
                verify_triangle_sweep_exact(start, end, [[0, 1, 2]])

    def test_face_budget_rejects_before_materializing_per_face_geometry(self):
        start = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 1, 2]])
        with patch("solver_triangle_sweep.np.stack", side_effect=AssertionError("Unbounded per-face allocation")):
            self.assertFalse(triangle_sweep_safe(start, start, faces, max_intervals=1))


if __name__ == "__main__":
    unittest.main()
