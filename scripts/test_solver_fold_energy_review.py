"""Independent finite-update and mixed-batch review of fold energy changes."""

from decimal import localcontext
import unittest

import numpy as np

from solver_fold_barrier import LocalAngularFoldBarrier
from test_solver_fold_adversarial import decimal_hinge_energy


def points(angle):
    return np.array([[.25, 1., 0.], [.75, -np.cos(angle), np.sin(angle)],
                     [0., 0., 0.], [1., 0., 0.]])


def difference(start, end, activation, stiffness):
    with localcontext() as context:
        context.prec = 65
        return float(decimal_hinge_energy(end, activation, stiffness)
                     - decimal_hinge_energy(start, activation, stiffness))


class FoldEnergyReviewTests(unittest.TestCase):
    def test_finite_uniform_scaling_preserves_angular_energy(self):
        barrier = LocalAngularFoldBarrier(4, [[0, 1, 2, 3]], .3, .7)
        for angle in (-2.7, -2.3, 2.3, 2.7):
            start = points(angle)
            for exponent in (-36, -28, -20, -12, -4, 4, 12, 20, 28, 36):
                with self.subTest(angle=angle, exponent=exponent):
                    # Binary scaling is exact for these finite float inputs;
                    # all triangle shapes and signed dihedral angles agree.
                    end = np.ldexp(start, exponent)
                    expected = difference(start, end, .3, .7)
                    self.assertLess(abs(expected), 1e-60)
                    np.testing.assert_allclose(barrier.energy_change(start, end), expected, rtol=2e-6, atol=1e-14)
                    np.testing.assert_allclose(barrier.energy_change(end, start), -expected, rtol=2e-6, atol=1e-14)

    def test_tiny_rotation_after_rescaling_keeps_original_precision(self):
        barrier = LocalAngularFoldBarrier(4, [[0, 1, 2, 3]], 1.4, .7)
        for exponent in (-36, 0, 36):
            for angle in (-2.7, -1.6, 1.6, 2.7):
                with self.subTest(exponent=exponent, angle=angle):
                    start, end = (np.ldexp(points(value), exponent) for value in (angle, angle + 1e-12))
                    expected = difference(start, end, 1.4, .7)
                    np.testing.assert_allclose(barrier.energy_change(start, end), expected, rtol=2e-6, atol=1e-25)
                    np.testing.assert_allclose(barrier.energy_change(end, start), -expected, rtol=2e-6, atol=1e-25)

    def test_sign_and_activation_crossings_match_endpoint_oracle(self):
        for activation, first, last in ((.3, -.8, .9), (.3, .8, -.9),
                                        (.3, -1.2, 1.2), (1.4, 1.4 - 1e-5, 1.4 + 1e-5),
                                        (1.4, -1.4 - 1e-5, -1.4 + 1e-5), (1.4, 1.5, 2.9)):
            with self.subTest(activation=activation, first=first, last=last):
                barrier = LocalAngularFoldBarrier(4, [[0, 1, 2, 3]], activation, .7)
                start, end = points(first), points(last)
                expected = difference(start, end, activation, .7)
                np.testing.assert_allclose(barrier.energy_change(start, end), expected, rtol=2e-6, atol=1e-25)
                np.testing.assert_allclose(barrier.energy_change(end, start), -expected, rtol=2e-6, atol=1e-25)

    def test_mixed_batch_subsets_use_each_hinges_own_parameters(self):
        starts = [points(angle) for angle in (2.7, 2.3, 1.3, -1.5, -.8, 2.3)]
        ends = [points(2.7 + 1e-12), np.ldexp(points(2.3), -28), points(1.5),
                points(-1.3), points(.9), points(2.4)]
        activations = [1.4, .3, 1.4, 1.4, .3, 1.4]
        stiffnesses = [.7, .7, .7, .7, .7, 0.]
        indices = np.arange(24).reshape((-1, 4))
        barrier = LocalAngularFoldBarrier(24, indices, activations, stiffnesses)
        expected = sum(difference(start, end, activation, stiffness)
                       for start, end, activation, stiffness in zip(starts, ends, activations, stiffnesses))
        start, end = np.concatenate(starts), np.concatenate(ends)
        np.testing.assert_allclose(barrier.energy_change(start, end), expected, rtol=2e-6, atol=1e-25)
        np.testing.assert_allclose(barrier.energy_change(end, start), -expected, rtol=2e-6, atol=1e-25)

    def test_branch_crossings_and_degenerate_endpoints_still_reject(self):
        barrier = LocalAngularFoldBarrier(4, [[0, 1, 2, 3]], .3, .7)
        for start, end in ((points(3.), points(-3.)),
                           (points(2.3), np.zeros((4, 3))),
                           (np.zeros((4, 3)), points(2.3))):
            with self.assertRaises(ValueError):
                barrier.energy_change(start, end)


if __name__ == "__main__":
    unittest.main()
