from decimal import Decimal, localcontext
import unittest

import numpy as np
from numpy.testing import assert_allclose

from solver_fold_barrier import LocalAngularFoldBarrier
from solver_hinge_sweep import hinge_sweep_safe


def decimal_atan(value):
    """Independent high-precision arctangent using half-angle reduction."""
    reduced = value
    for _ in range(8):
        reduced /= 1 + (1 + reduced * reduced).sqrt()
    squared, term, total = -reduced * reduced, reduced, reduced
    for index in range(1, 100):
        term *= squared
        following = total + term / (2 * index + 1)
        if following == total:
            return total * 256
        total = following
    raise AssertionError("Decimal arctangent did not converge")


def decimal_atan2(y, x):
    # Mathematical pi is required by atan2. The barrier separately uses its
    # declared binary64 np.pi constant, preserving the production energy law.
    pi = 16 * decimal_atan(Decimal(1) / 5) - 4 * decimal_atan(Decimal(1) / 239)
    if x == 0:
        return pi / 2 if y > 0 else -pi / 2
    angle = decimal_atan(y / x)
    if x < 0:
        angle += pi if y >= 0 else -pi
    return angle


def decimal_hinge_energy(points, activation, stiffness):
    points = [[Decimal.from_float(float(value)) for value in point] for point in points]
    subtract = lambda first, second: [a - b for a, b in zip(first, second)]
    dot = lambda first, second: sum(a * b for a, b in zip(first, second))

    def cross(first, second):
        return [first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0]]

    first = cross(subtract(points[2], points[0]), subtract(points[3], points[0]))
    second = cross(subtract(points[3], points[1]), subtract(points[2], points[1]))
    edge = subtract(points[3], points[2])
    angle = abs(decimal_atan2(dot(cross(first, second), edge) / dot(edge, edge).sqrt(), dot(first, second)))
    activation = Decimal.from_float(float(activation))
    if angle <= activation:
        return Decimal(0)
    pi = Decimal.from_float(float(np.pi))
    gap = (pi - angle) / (pi - activation)
    return -Decimal.from_float(float(stiffness)) * (gap - 1) ** 2 * gap.ln()


class FoldBarrierAdversarialTests(unittest.TestCase):
    def points(self, angle):
        return np.array([[0.25, 1., 0.], [0.75, -np.cos(angle), np.sin(angle)],
                         [0., 0., 0.], [1., 0., 0.]])

    def test_scalar_curvature_against_independent_second_difference(self):
        for activation in (0.3, 1.4, 2.8):
            barrier = LocalAngularFoldBarrier(4, [[0, 1, 2, 3]], activation, 0.7)
            for angle in (activation + 0.05, (activation + np.pi) / 2, np.pi - 0.01):
                for sign in (-1., 1.):
                    signed_angle = sign * angle
                    points = self.points(signed_angle)
                    tangent = np.zeros((4, 3))
                    tangent[1] = [0., np.sin(signed_angle), np.cos(signed_angle)]
                    metric_curvature = tangent.ravel() @ barrier.hessian(points) @ tangent.ravel()
                    delta = 1e-6
                    numerical = (barrier.energy(self.points(signed_angle + delta))
                                 - 2 * barrier.energy(points)
                                 + barrier.energy(self.points(signed_angle - delta))) / delta ** 2
                    assert_allclose(metric_curvature, numerical, rtol=0.002, atol=1e-6)

    def test_energy_change_matches_decimal_scalar_oracle(self):
        barrier = LocalAngularFoldBarrier(4, [[0, 1, 2, 3]], 1.4, 0.7)
        for angle, delta in ((1.4 + 1e-6, 1e-11), (2.7, 1e-12),
                             (np.pi - 1e-5, 1e-10), (np.pi - 2e-8, 1e-12)):
            start, end = self.points(angle), self.points(angle + delta)
            with localcontext() as context:
                context.prec = 65
                pi = Decimal.from_float(float(np.pi))
                activation = Decimal.from_float(1.4)
                stiffness = Decimal.from_float(0.7)
                energies = []
                for points in (start, end):
                    scalar_angle = decimal_atan2(Decimal.from_float(float(points[1, 2])),
                                                -Decimal.from_float(float(points[1, 1])))
                    gap = (pi - scalar_angle) / (pi - activation)
                    energies.append(-stiffness * (gap - 1) ** 2 * gap.ln())
                expected = float(energies[1] - energies[0])
            assert_allclose(barrier.energy_change(start, end), expected, rtol=2e-6, atol=1e-25)

    def test_tiny_general_hinge_changes_match_decimal_geometry(self):
        barrier = LocalAngularFoldBarrier(4, [[0, 1, 2, 3]], 1.4, 0.7)
        generator = np.random.default_rng(3201)
        for angle in (-2.7, -1.6, 1.6, 2.7):
            start = self.points(angle) + generator.normal(size=(4, 3)) * .03
            for magnitude in (1e-8, 1e-10, 1e-12):
                end = start + magnitude * generator.normal(size=(4, 3))
                with localcontext() as context:
                    context.prec = 65
                    expected = float(decimal_hinge_energy(end, 1.4, .7)
                                     - decimal_hinge_energy(start, 1.4, .7))
                observed = barrier.energy_change(start, end)
                assert_allclose(observed, expected, rtol=2e-6, atol=1e-25)
                assert_allclose(barrier.energy_change(end, start), -expected, rtol=2e-6, atol=1e-25)

    def test_batch_energy_and_shared_vertex_scatter(self):
        points = np.concatenate((self.points(2.4), self.points(-2.1)[:2]))
        indices = [[0, 1, 2, 3], [4, 5, 2, 3]]
        batch = LocalAngularFoldBarrier(6, indices, [1.5, 1.7], [0.3, 0.8])
        individual = [LocalAngularFoldBarrier(6, [row], activation, stiffness)
                      for row, activation, stiffness in zip(indices, [1.5, 1.7], [0.3, 0.8])]
        assert_allclose(batch.energy(points), sum(barrier.energy(points) for barrier in individual))
        assert_allclose(batch.gradient(points), sum(barrier.gradient(points) for barrier in individual))
        assert_allclose(batch.hessian(points).toarray(),
                        sum(barrier.hessian(points).toarray() for barrier in individual))

    def test_scale_behavior_energy_force_metric(self):
        points = self.points(2.4)
        barrier = LocalAngularFoldBarrier(4, [[0, 1, 2, 3]], 1.4, 0.7)
        for scale in (1e-6, 1e-2, 1e2, 1e6):
            assert_allclose(barrier.energy(points * scale), barrier.energy(points), rtol=1e-12)
            assert_allclose(barrier.gradient(points * scale) * scale, barrier.gradient(points), rtol=1e-12, atol=1e-14)
            assert_allclose(barrier.hessian(points * scale).toarray() * scale ** 2,
                            barrier.hessian(points).toarray(), rtol=1e-12, atol=1e-13)

    def test_endpoint_energy_accounting_does_not_certify_swept_geometry(self):
        start = self.points(0.)
        end = start.copy()
        end[:2, 1] *= -1
        indices = [[0, 1, 2, 3]]
        barrier = LocalAngularFoldBarrier(4, indices)
        self.assertEqual(barrier.energy_change(start, end), 0.)
        self.assertFalse(hinge_sweep_safe(start, end, indices))


if __name__ == "__main__":
    unittest.main()
