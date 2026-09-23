"""Independent Decimal review of cancellation fallback, from original inputs."""
from decimal import Decimal, localcontext
import unittest

import numpy as np

from solver_material_grippers import MaterialGrippers


def decimal_oracle(positions, weights, stiffness, targets, activation):
    """No solver-derived residual, coefficient, contribution or Fraction used."""
    with localcontext() as context:
        context.prec = 150
        decimal = lambda value: Decimal.from_float(float(value))
        q = [[decimal(value) for value in point] for point in positions]
        targets = [[decimal(value) for value in point] for point in targets]
        nodal = [[Decimal(0) for _ in range(3)] for _ in q]
        energy, forces = Decimal(0), []
        for row, k, target, active in zip(weights, stiffness, targets, activation):
            row = [decimal(value) for value in row]
            anchor = [sum(row[i] * q[i][axis] for i in range(3)) for axis in range(3)]
            error = [point - goal for point, goal in zip(anchor, target)]
            coefficient = decimal(k) * decimal(active)
            energy += coefficient * sum(value * value for value in error) / 2
            force = [-coefficient * value for value in error]
            forces.append(force)
            for vertex in range(3):
                for axis in range(3):
                    nodal[vertex][axis] += row[vertex] * force[axis]
        def cross(first, second):
            return [first[1] * second[2] - first[2] * second[1],
                    first[2] * second[0] - first[0] * second[2],
                    first[0] * second[1] - first[1] * second[0]]
        cloth_torque = [sum(cross(point, force)[axis] for point, force in zip(q, nodal)) for axis in range(3)]
        tool_torque = [sum(cross(target, [-value for value in force])[axis]
                           for target, force in zip(targets, forces)) for axis in range(3)]
        return energy, np.array([[-float(value) for value in force] for force in nodal]), {
            "clothTorqueNewtonMeters": [float(value) for value in cloth_torque],
            "toolTorqueNewtonMeters": [float(value) for value in tool_torque],
            "netTorqueResidualNewtonMeters": [float(first + second) for first, second in zip(cloth_torque, tool_torque)]}


def potential(weights, stiffness, targets, activation):
    recipe = MaterialGrippers(["cloth"] * 3, [[0, 1, 2]], [
        {"id": f"g{index}", "instanceId": "cloth", "triangleIndex": 0,
         "weights": row, "stiffnessNPerM": k} for index, (row, k) in enumerate(zip(weights, stiffness))])
    return recipe.potential(targets, activation)


class GripperPrecisionReviewTests(unittest.TestCase):
    def test_original_input_force_cancellation_and_both_directional_energy_signs(self):
        q = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        weights, stiffness = [[1., 0., 0.]] * 3, [1e12] * 3
        cases = [([[100., 0., 0.], [float(np.nextafter(-100., 0.)), 0., 0.], [-1.5625e-14, 0., 0.]], [1.] * 3),
                 ([[100., 0., 0.], [-100., 0., 0.], [100., 0., 0.]], [.1, .3, .2])]
        for targets, activation in cases:
            with self.subTest(activation=activation):
                grip = potential(weights, stiffness, targets, activation)
                before, gradient, _ = decimal_oracle(q, weights, stiffness, targets, activation)
                self.assertGreater(abs(gradient[0, 0]), .001)
                np.testing.assert_allclose(grip.gradient(q), gradient, rtol=2e-15, atol=0)
                for shift in (2. ** -60, -2. ** -60, -gradient[0, 0] / sum(k * a for k, a in zip(stiffness, activation))):
                    end = q.copy()
                    end[0, 0] = shift
                    after, _, _ = decimal_oracle(end, weights, stiffness, targets, activation)
                    with localcontext() as context:
                        context.prec = 150
                        expected = float(after - before)
                    actual = grip.energy_change(q, end)
                    self.assertNotEqual(expected, 0.)
                    self.assertEqual(np.signbit(actual), np.signbit(expected))
                    self.assertAlmostEqual(actual, expected, delta=abs(expected) * 3e-15)
                    self.assertAlmostEqual(grip.energy_change(end, q), -expected, delta=abs(expected) * 3e-15)

    def test_cancellation_inside_cross_products_preserves_tool_torque_balance(self):
        q = np.array([[100., 100., 0.], [99., 100., 0.], [100., 99., 0.]])
        weights, stiffness, activation = [[1., 0., 0.]], [1e12], [1.]
        targets = [[-100., float(np.nextafter(-100., 0.)), 0.]]
        grip = potential(weights, stiffness, targets, activation)
        _, _, expected = decimal_oracle(q, weights, stiffness, targets, activation)
        self.assertGreater(expected["clothTorqueNewtonMeters"][2], 1.)
        diagnostics = grip.diagnostics(q)
        for key, value in expected.items():
            with self.subTest(field=key):
                np.testing.assert_allclose(diagnostics[key], value, rtol=2e-15, atol=0)

    def test_weighted_source_anchor_cancellation_uses_original_weights_and_coordinates(self):
        q = np.array([[100., 0., 0.], [-100., 1., 0.], [20., 0., 1.]])
        weights, stiffness, targets, activation = [[.2, .3, .5]], [1e12], [[0., .3, .5]], [1.]
        grip = potential(weights, stiffness, targets, activation)
        before, expected, _ = decimal_oracle(q, weights, stiffness, targets, activation)
        self.assertGreater(np.max(np.abs(expected)), .001)
        np.testing.assert_allclose(grip.gradient(q), expected, rtol=2e-15, atol=0)
        for sign in (-1, 1):
            end = q.copy()
            end[2, 0] = np.nextafter(end[2, 0], sign * np.inf)
            after, _, _ = decimal_oracle(end, weights, stiffness, targets, activation)
            with localcontext() as context:
                context.prec = 150
                change = float(after - before)
            self.assertAlmostEqual(grip.energy_change(q, end), change, delta=abs(change) * 3e-15)


if __name__ == "__main__":
    unittest.main()
