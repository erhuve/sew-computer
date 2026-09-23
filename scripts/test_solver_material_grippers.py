"""Portable independent mathematical checks for external material grippers."""

import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import unittest

import numpy as np

from solver_material_grippers import (MaterialGrippers, MaterialGripperSchedule, PARAMETER_ORDER,
                                     SCHEDULE_PROFILE, WEIGHT_SUM_TOLERANCE)


def decimal_energy(positions, anchors, faces, targets, activation):
    """Independent endpoint oracle on the exact supplied binary float values."""
    decimal = lambda value: Decimal.from_float(float(value))
    with localcontext() as context:
        context.prec = 150
        result = Decimal(0)
        for index, anchor in enumerate(anchors):
            square = Decimal(0)
            for axis in range(3):
                coordinate = sum(decimal(weight) * decimal(positions[vertex, axis])
                                 for weight, vertex in zip(anchor["weights"], faces[anchor["triangleIndex"]]))
                square += (coordinate - decimal(targets[index][axis])) ** 2
            result += decimal(anchor["stiffnessNPerM"]) * decimal(activation[index]) * square / 2
        return +result


def decimal_gradient(positions, anchors, faces, targets, activation):
    decimal = lambda value: Decimal.from_float(float(value))
    with localcontext() as context:
        context.prec = 150
        result = [[Decimal(0) for _ in range(3)] for _ in positions]
        for index, anchor in enumerate(anchors):
            face = faces[anchor["triangleIndex"]]
            coefficient = decimal(anchor["stiffnessNPerM"]) * decimal(activation[index])
            for axis in range(3):
                coordinate = sum(decimal(weight) * decimal(positions[vertex, axis])
                                 for weight, vertex in zip(anchor["weights"], face))
                force = coefficient * (coordinate - decimal(targets[index][axis]))
                for weight, vertex in zip(anchor["weights"], face):
                    result[vertex][axis] += decimal(weight) * force
        return np.array([[float(value) for value in point] for point in result])


class MaterialGripperTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[.01, .02, .003], [.03, -.01, .004], [-.01, .04, -.003],
                                   [.02, .05, .008], [-.01, .06, .015], [.04, .08, -.002]])
        self.instances = ["shell"] * 3 + ["facing"] * 3
        self.faces = [[0, 1, 2], [3, 4, 5]]
        self.anchors = [
            {"id": "shell-grip", "instanceId": "shell", "triangleIndex": 0, "weights": [.2, .3, .5], "stiffnessNPerM": 37.},
            {"id": "shell-edge", "instanceId": "shell", "triangleIndex": 0, "weights": [.25, .75, 0.], "stiffnessNPerM": 23.},
            {"id": "facing-grip", "instanceId": "facing", "triangleIndex": 1, "weights": [.5, .25, .25], "stiffnessNPerM": 51.}]
        self.targets = np.array([[.012, .025, -.003], [.015, .0, .007], [.02, .05, .018]])
        self.activation = np.array([.75, 0., .4])
        self.recipe = MaterialGrippers(self.instances, self.faces, self.anchors)
        self.potential = self.recipe.potential(self.targets, self.activation)

    def oracle_matrix(self):
        result = np.zeros((len(self.anchors), len(self.positions)))
        for index, anchor in enumerate(self.anchors):
            result[index, self.faces[anchor["triangleIndex"]]] = anchor["weights"]
        return result

    def test_analytic_energy_derivatives_sparse_structure_and_shared_anchors(self):
        matrix = self.oracle_matrix()
        coefficient = np.array([item["stiffnessNPerM"] for item in self.anchors]) * self.activation
        residual = matrix @ self.positions - self.targets
        expected = np.kron(matrix.T @ np.diag(coefficient) @ matrix, np.eye(3))
        np.testing.assert_allclose(self.potential.hessian().toarray(), expected, rtol=2e-15, atol=1e-14)
        np.testing.assert_allclose(self.potential.gradient(self.positions), matrix.T @ (coefficient[:, None] * residual), rtol=3e-15, atol=1e-15)
        np.testing.assert_allclose(self.potential.jacobian().toarray(), np.kron(np.diag(np.sqrt(coefficient)) @ matrix, np.eye(3)), atol=1e-15)
        np.testing.assert_allclose(self.potential.residual(self.positions), (np.sqrt(coefficient)[:, None] * residual).ravel(), atol=1e-15)
        self.assertAlmostEqual(self.potential.energy(self.positions), .5 * np.sum(coefficient[:, None] * residual ** 2), places=16)
        self.assertGreaterEqual(np.linalg.eigvalsh(expected).min(), -1e-14)
        np.testing.assert_allclose(self.potential.jacobian().T @ self.potential.residual(self.positions),
                                   self.potential.gradient(self.positions).ravel(), atol=1e-15)
        self.assertLessEqual(self.potential.hessian().nnz, 27 * len(self.anchors))
        active = self.recipe.potential(self.targets, [1., 1., 1.])
        separate = sum(MaterialGrippers(self.instances, self.faces, [anchor]).potential([target], [1.]).hessian()
                       for anchor, target in zip(self.anchors, self.targets))
        np.testing.assert_allclose(active.hessian().toarray(), separate.toarray(), atol=1e-14)

    def test_finite_differences_including_exact_constant_hessian(self):
        epsilon = 1e-6
        gradient = self.potential.gradient(self.positions).ravel()
        hessian = self.potential.hessian(self.positions).toarray()
        jacobian = self.potential.jacobian(self.positions).toarray()
        for index, direction in enumerate(np.eye(self.positions.size).reshape((-1, *self.positions.shape))):
            upper, lower = self.positions + epsilon * direction, self.positions - epsilon * direction
            self.assertAlmostEqual((self.potential.energy(upper) - self.potential.energy(lower)) / (2 * epsilon), gradient[index], delta=2e-12)
            np.testing.assert_allclose((self.potential.gradient(upper) - self.potential.gradient(lower)).ravel() / (2 * epsilon), hessian[:, index], atol=2e-11, rtol=2e-11)
            np.testing.assert_allclose((self.potential.residual(upper) - self.potential.residual(lower)) / (2 * epsilon), jacobian[:, index], atol=2e-11, rtol=2e-11)
        np.testing.assert_array_equal(self.potential.hessian(self.positions * 3).toarray(), hessian)

    def test_rigid_covariance_and_cloth_tool_force_torque_balance(self):
        rotation = np.linalg.qr(np.random.default_rng(391).normal(size=(3, 3)))[0]
        shift = np.array([.2, -.3, .1])
        moved = self.recipe.potential(self.targets @ rotation + shift, self.activation)
        positions = self.positions @ rotation + shift
        self.assertAlmostEqual(moved.energy(positions), self.potential.energy(self.positions), delta=2e-16)
        np.testing.assert_allclose(moved.gradient(positions), self.potential.gradient(self.positions) @ rotation, atol=2e-15)
        transform = np.kron(np.eye(len(self.positions)), rotation.T)
        np.testing.assert_allclose(moved.hessian().toarray(), transform @ self.potential.hessian().toarray() @ transform.T, atol=1e-14)
        report = self.potential.diagnostics(self.positions)
        np.testing.assert_allclose(report["anchorPositionsMeters"], self.oracle_matrix() @ self.positions, atol=1e-17)
        np.testing.assert_array_equal(report["anchorForcesNewtons"], -report["toolReactionsNewtons"])
        np.testing.assert_array_equal(report["nodalForcesNewtons"], -self.potential.gradient(self.positions))
        np.testing.assert_allclose(report["netForceResidualNewtons"], 0, atol=2e-16)
        np.testing.assert_allclose(report["netTorqueResidualNewtonMeters"], 0, atol=2e-17)
        self.assertGreater(np.linalg.norm(report["totalClothForceNewtons"]), .1)
        np.testing.assert_allclose(report["clothTorqueNewtonMeters"], np.cross(report["anchorPositionsMeters"], report["anchorForcesNewtons"]).sum(axis=0), atol=2e-17)

    def test_zero_activation_is_exactly_absent_and_target_motion_does_no_work(self):
        potential = self.recipe.potential(self.targets + 40., np.zeros(3))
        self.assertEqual(potential.energy(self.positions), 0.)
        self.assertEqual(potential.energy_change(self.positions, self.positions + 30.), 0.)
        np.testing.assert_array_equal(potential.gradient(self.positions), np.zeros_like(self.positions))
        np.testing.assert_array_equal(potential.residual(self.positions), np.zeros(9))
        self.assertEqual(potential.hessian().nnz, 0)
        self.assertEqual(potential.jacobian().nnz, 0)
        report = self.recipe.parameter_energy_change(self.positions, self.targets, [0.] * 3, self.targets + 40., [0.] * 3)
        for key in ("totalParameterWorkJoules", "targetParameterWorkJoules", "activationParameterWorkJoules", "releaseEnergyRemovedJoules"):
            self.assertEqual(report[key], 0.)

    def test_position_change_matches_high_precision_endpoint_oracle(self):
        cases = []
        tiny = self.positions.copy()
        tiny[0, 0] = np.nextafter(tiny[0, 0], np.inf)
        cases.append((self.positions, tiny, self.targets))
        translated = self.positions + 60.
        moved = translated.copy()
        moved[0, 2] = np.nextafter(moved[0, 2], np.inf)
        cases.append((translated, moved, self.targets + 60.))
        cases.append((self.positions, -self.positions, np.zeros_like(self.targets)))
        cases.append((self.positions, self.positions * -700., self.targets))
        for start, end, targets in cases:
            with self.subTest(start=start[0].tolist()):
                potential = self.recipe.potential(targets, self.activation)
                with localcontext() as context:
                    context.prec = 150
                    expected = float(decimal_energy(end, self.anchors, self.faces, targets, self.activation)
                                     - decimal_energy(start, self.anchors, self.faces, targets, self.activation))
                actual = potential.energy_change(start, end)
                self.assertAlmostEqual(actual, expected, delta=max(1e-30, abs(expected) * 2e-14))
                self.assertAlmostEqual(potential.energy_change(end, start), -actual, delta=max(1e-30, abs(actual) * 2e-15))

    def test_discrete_parameter_work_target_first_decimal_and_release_cancellation(self):
        cases = [([0., 0., 0.], [1., .5, .75], self.targets + 50.),
                 ([1., 1., 1.], [0., 0., 0.], self.targets + 70.),
                 ([.3, .9, .1], [.7, .2, .4], self.targets + .01),
                 ([.3, .9, .1], [.3, .9, .1], np.nextafter(self.targets, np.inf))]
        for before_activation, after_activation, after_targets in cases:
            report = self.recipe.parameter_energy_change(self.positions, self.targets, before_activation, after_targets, after_activation)
            with localcontext() as context:
                context.prec = 150
                before = decimal_energy(self.positions, self.anchors, self.faces, self.targets, before_activation)
                middle = decimal_energy(self.positions, self.anchors, self.faces, after_targets, before_activation)
                after = decimal_energy(self.positions, self.anchors, self.faces, after_targets, after_activation)
                expected = [float(after - before), float(middle - before), float(after - middle)]
            actual = [report[key] for key in ("totalParameterWorkJoules", "targetParameterWorkJoules", "activationParameterWorkJoules")]
            self.assertEqual(actual, expected)
            self.assertEqual(report["parameterOrder"], PARAMETER_ORDER)
            self.assertLessEqual(abs(actual[0] - math.fsum(actual[1:])), report["roundedComponentSumErrorBoundJoules"])
            self.assertGreaterEqual(report["activationIncreaseWorkJoules"], 0.)
            self.assertGreaterEqual(report["releaseEnergyRemovedJoules"], 0.)
            self.assertAlmostEqual(report["activationIncreaseWorkJoules"] - report["releaseEnergyRemovedJoules"], report["activationParameterWorkJoules"], delta=report["roundedComponentSumErrorBoundJoules"])
        release = self.recipe.parameter_energy_change(self.positions, self.targets, [1.] * 3, self.targets + 70., [0.] * 3)
        self.assertLess(release["totalParameterWorkJoules"], 0.)
        self.assertGreater(abs(release["targetParameterWorkJoules"]), 1e7 * abs(release["totalParameterWorkJoules"]))

    def test_captured_inputs_and_every_exposed_result_are_isolated(self):
        faces, anchors, instances = copy.deepcopy((self.faces, self.anchors, self.instances))
        targets, activation = self.targets.copy(), self.activation.copy()
        recipe = MaterialGrippers(instances, faces, anchors)
        potential = recipe.potential(targets, activation)
        before = potential.energy(self.positions)
        faces[0][0] = 4
        anchors[0]["weights"][0] = 0.
        instances[0] = "changed"
        targets[:] = 99.
        activation[:] = 0.
        recipe.faces[:] = 0
        recipe.anchors[0]["weights"][0] = 0.
        potential.targets[:] = 99.
        potential.activation[:] = 0.
        potential.hessian().data[:] = 0.
        potential.jacobian().data[:] = 0.
        potential.gradient(self.positions)[:] = 0.
        potential.residual(self.positions)[:] = 0.
        for value in potential.diagnostics(self.positions).values():
            if isinstance(value, np.ndarray):
                value[:] = 99.
        self.assertEqual(potential.energy(self.positions), before)
        for owner, name in ((recipe, "_faces"), (recipe, "_weights"), (potential, "_targets"), (potential, "_activation")):
            with self.assertRaises(AttributeError):
                setattr(owner, name, None)
            with self.assertRaises(AttributeError):
                delattr(owner, name)
            with self.assertRaises(ValueError):
                getattr(owner, name).flags.writeable = True

    def test_source_identity_normalization_and_budgets_fail_closed(self):
        for case in ("duplicate id", "unknown field", "bad id", "other instance", "negative triangle", "float triangle", "bool triangle",
                     "negative weight", "zero sum", "nonunit sum", "nan weight", "bool weight", "zero stiffness", "large stiffness", "bool stiffness"):
            anchors = copy.deepcopy(self.anchors)
            first = anchors[0]
            if case == "duplicate id": anchors[1]["id"] = first["id"]
            elif case == "unknown field": first["accepted"] = True
            elif case == "bad id": first["id"] = "bad id"
            elif case == "other instance": first["instanceId"] = "facing"
            elif case == "negative triangle": first["triangleIndex"] = -1
            elif case == "float triangle": first["triangleIndex"] = 0.
            elif case == "bool triangle": first["triangleIndex"] = False
            elif case == "negative weight": first["weights"] = [-.1, .5, .6]
            elif case == "zero sum": first["weights"] = [0., 0., 0.]
            elif case == "nonunit sum": first["weights"] = [.2, .3, .50000001]
            elif case == "nan weight": first["weights"][0] = np.nan
            elif case == "bool weight": first["weights"] = [False, 0., 1.]
            elif case == "zero stiffness": first["stiffnessNPerM"] = 0.
            elif case == "large stiffness": first["stiffnessNPerM"] = 1e13
            else: first["stiffnessNPerM"] = True
            with self.subTest(case=case), self.assertRaises(ValueError):
                MaterialGrippers(self.instances, self.faces, anchors)
        for faces in ([[0, 1, 3]], [[0, 0, 2]], [[0., 1., 2.]], [[False, 1, 2]], [[0, 1, 99]], [[0, 1, 2], [2, 0, 1]], []):
            with self.subTest(faces=faces), self.assertRaises(ValueError):
                MaterialGrippers(self.instances, faces, self.anchors)
        for instances, faces, anchors in ((["s"] * 25001, self.faces, self.anchors),
                                         (self.instances, [[0, 1, 2]] * 50001, self.anchors),
                                         (self.instances, self.faces, self.anchors * 86),
                                         (self.instances, self.faces, [])):
            with self.assertRaises(ValueError):
                MaterialGrippers(instances, faces, anchors)

    def test_weights_are_not_silently_renormalized_and_errors_are_reported(self):
        anchors = copy.deepcopy(self.anchors[:1])
        anchors[0]["weights"] = [.2, .3, .5 + WEIGHT_SUM_TOLERANCE / 2]
        recipe = MaterialGrippers(self.instances, self.faces, anchors)
        self.assertEqual(recipe.anchors, anchors)
        potential = recipe.potential([[.02, .01, .03]], [1.])
        report = potential.diagnostics(self.positions)
        self.assertNotEqual(report["anchorWeightSums"][0], 1.)
        self.assertGreater(np.linalg.norm(report["netForceResidualNewtons"]), 0.)
        np.testing.assert_allclose(report["netForceResidualNewtons"],
            (report["anchorWeightSums"][0] - 1.) * report["anchorForcesNewtons"][0], atol=1e-16)

    def test_malformed_parameters_positions_and_optional_positions_reject(self):
        for targets, activation in (([[0., 0., 0.]], self.activation), (self.targets, [.1]),
                                     (self.targets, [True, 0., 1.]), (self.targets, [-.1, 0., 1.]),
                                     (self.targets, [np.nan, 0., 1.]), (self.targets + 101., self.activation)):
            with self.assertRaises(ValueError):
                self.recipe.potential(targets, activation)
        bad_positions = [np.zeros((5, 3)), np.full((6, 3), np.nan), np.full((6, 3), 101.), [[False, 0., 0.]] * 6]
        for positions in bad_positions:
            for method in (self.potential.energy, self.potential.gradient, self.potential.residual,
                           self.potential.hessian, self.potential.jacobian, self.potential.diagnostics):
                with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                    method(positions)
            with self.assertRaises(ValueError):
                self.potential.energy_change(self.positions, positions)

    def cancellation_fixture(self):
        positions = np.array([[0., 0., 0.], [.03125, 0., 0.], [0., .03125, 0.]])
        anchors = [{"id": f"grip-{index}", "instanceId": "cloth", "triangleIndex": 0,
                    "weights": [1., 0., 0.], "stiffnessNPerM": 1e12} for index in range(3)]
        targets = np.array([[100., 0., 0.], [np.nextafter(-100., 0.), 0., 0.], [-1.5625e-14, 0., 0.]])
        return positions, anchors, targets

    def test_large_cancelling_grips_cannot_create_false_zero_force_or_reaction(self):
        positions, anchors, targets = self.cancellation_fixture()
        expected = decimal_gradient(positions, anchors, [[0, 1, 2]], targets, [1.] * 3)
        self.assertGreater(abs(expected[0, 0]), 1e-6)
        for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
            selected_anchors, selected_targets = [anchors[index] for index in order], targets[order]
            potential = MaterialGrippers(["cloth"] * 3, [[0, 1, 2]], selected_anchors).potential(selected_targets, [1.] * 3)
            np.testing.assert_array_equal(potential.gradient(positions), expected)
            report = potential.diagnostics(positions)
            np.testing.assert_array_equal(report["nodalForcesNewtons"], -expected)
            np.testing.assert_array_equal(report["totalClothForceNewtons"], -expected.sum(axis=0))
            np.testing.assert_array_equal(report["totalToolReactionNewtons"], expected.sum(axis=0))
            np.testing.assert_array_equal(report["netForceResidualNewtons"], [0., 0., 0.])
            # Rounded residual/Jacobian products cannot retain every cancelled
            # bit. Their discrepancy must fit their own product-rounding scale;
            # the physical gradient is checked independently above without it.
            jacobian, residual = potential.jacobian(), potential.residual(positions)
            scale = np.asarray(abs(jacobian).T @ np.abs(residual))
            difference = np.abs(jacobian.T @ residual - expected.ravel())
            self.assertTrue(np.all(difference <= 64 * np.finfo(float).eps * scale))

    def test_cancelled_directional_work_has_correct_sign_and_binary_rounding(self):
        positions, anchors, targets = self.cancellation_fixture()
        potential = MaterialGrippers(["cloth"] * 3, [[0, 1, 2]], anchors).potential(targets, [1.] * 3)
        gradient = decimal_gradient(positions, anchors, [[0, 1, 2]], targets, [1.] * 3)[0, 0]
        for displacement in (-gradient / 3e12, -1e-17, -1e-16, 1e-16):
            end = positions.copy()
            end[0, 0] = displacement
            with localcontext() as context:
                context.prec = 150
                expected = float(decimal_energy(end, anchors, [[0, 1, 2]], targets, [1.] * 3)
                                 - decimal_energy(positions, anchors, [[0, 1, 2]], targets, [1.] * 3))
            self.assertNotEqual(expected, 0.)
            self.assertEqual(potential.energy_change(positions, end), expected)
            self.assertEqual(potential.energy_change(end, positions), -expected)
            self.assertEqual(math.copysign(1., expected), math.copysign(1., displacement))
            self.assertEqual(potential.energy(end), potential.energy(positions))

    def test_weighted_anchor_subtraction_and_parameter_products_use_original_inputs(self):
        positions = np.array([[100., 100., 0.], [99., 100., 0.], [100., 99., 0.]])
        anchors = [{"id": "weighted", "instanceId": "cloth", "triangleIndex": 0,
                    "weights": [.2, .3, .5], "stiffnessNPerM": 100000000000.1}]
        targets = [[np.nextafter(99.7, 0.), np.nextafter(99.5, 0.), 0.]]
        activation = [.3]
        potential = MaterialGrippers(["cloth"] * 3, [[0, 1, 2]], anchors).potential(targets, activation)
        expected = decimal_gradient(positions, anchors, [[0, 1, 2]], targets, activation)
        np.testing.assert_array_equal(potential.gradient(positions), expected)
        np.testing.assert_array_equal(potential.diagnostics(positions)["nodalForcesNewtons"], -expected)
        end = positions.copy()
        end[1, 1] = np.nextafter(end[1, 1], 0.)
        with localcontext() as context:
            context.prec = 150
            change = float(decimal_energy(end, anchors, [[0, 1, 2]], targets, activation)
                           - decimal_energy(positions, anchors, [[0, 1, 2]], targets, activation))
        self.assertEqual(potential.energy_change(positions, end), change)

    def test_torque_cancellation_inside_cross_products_is_retained(self):
        positions = np.array([[100., 100., 0.], [99., 100., 0.], [100., 99., 0.]])
        anchors = [{"id": "vertex", "instanceId": "cloth", "triangleIndex": 0,
                    "weights": [1., 0., 0.], "stiffnessNPerM": 1e12}]
        target = [[-100., np.nextafter(-100., 0.), 0.]]
        potential = MaterialGrippers(["cloth"] * 3, [[0, 1, 2]], anchors).potential(target, [1.])
        report = potential.diagnostics(positions)
        with localcontext() as context:
            context.prec = 150
            decimal = lambda value: Decimal.from_float(float(value))
            forces = [-decimal(1e12) * (decimal(positions[0, axis]) - decimal(target[0][axis])) for axis in range(3)]
            torque = float(decimal(positions[0, 0]) * forces[1] - decimal(positions[0, 1]) * forces[0])
        self.assertGreater(abs(torque), 1.)
        np.testing.assert_array_equal(report["clothTorqueNewtonMeters"], [0., 0., torque])
        np.testing.assert_array_equal(report["toolTorqueNewtonMeters"], [0., 0., -torque])
        np.testing.assert_array_equal(report["netTorqueResidualNewtonMeters"], [0., 0., 0.])
        np.testing.assert_array_equal(report["nodalForcesNewtons"], -potential.gradient(positions))


class MaterialGripperScheduleTests(unittest.TestCase):
    def setUp(self):
        self.recipe = {"profile": SCHEDULE_PROFILE, "gripperIds": ["one", "two"], "knots": [
            {"fraction": 0., "targetsMeters": [[0., 0., 0.], [1., 0., 0.]], "activation": [0., 1.]},
            {"fraction": .25, "targetsMeters": [[1., 2., 3.], [2., 3., 4.]], "activation": [1., .5]},
            {"fraction": 1., "targetsMeters": [[-1., -2., -3.], [5., 6., 7.]], "activation": [0., 0.]}]}
        self.schedule = MaterialGripperSchedule(self.recipe, 4, gripper_ids=["one", "two"])

    def test_stateless_exact_knots_interpolation_release_and_isolation(self):
        for knot in self.recipe["knots"]:
            targets, activation = self.schedule.parameters(knot["fraction"])
            np.testing.assert_array_equal(targets, knot["targetsMeters"])
            np.testing.assert_array_equal(activation, knot["activation"])
        targets, activation = self.schedule.parameters(Fraction(5, 8))
        np.testing.assert_array_equal(targets, [[0., 0., 0.], [3.5, 4.5, 5.5]])
        np.testing.assert_array_equal(activation, [.5, .25])
        targets[:], activation[:] = 99., 99.
        self.recipe["knots"][0]["targetsMeters"][0][0] = 99.
        self.recipe["gripperIds"][0] = "changed"
        np.testing.assert_array_equal(self.schedule.parameters(0)[0], [[0., 0., 0.], [1., 0., 0.]])
        np.testing.assert_array_equal(self.schedule.parameters(1)[1], [0., 0.])
        self.assertEqual(self.schedule.gripper_ids, ("one", "two"))
        with self.assertRaises(AttributeError):
            self.schedule._activation = None
        with self.assertRaises(ValueError):
            self.schedule._targets.flags.writeable = True

    def test_malformed_or_unbounded_schedule_and_time_reject(self):
        for case in ("unknown", "identity order", "duplicate time", "off grid", "bad target", "bad activation", "missing endpoint", "too many"):
            recipe = copy.deepcopy(self.recipe)
            if case == "unknown": recipe["accepted"] = True
            elif case == "identity order": recipe["gripperIds"].reverse()
            elif case == "duplicate time": recipe["knots"][1]["fraction"] = 0.
            elif case == "off grid": recipe["knots"][1]["fraction"] = .125
            elif case == "bad target": recipe["knots"][0]["targetsMeters"][0][0] = np.nan
            elif case == "bad activation": recipe["knots"][0]["activation"][0] = True
            elif case == "missing endpoint": recipe["knots"][-1]["fraction"] = .75
            else: recipe["knots"] *= 22
            with self.subTest(case=case), self.assertRaises(ValueError):
                MaterialGripperSchedule(recipe, 4, gripper_ids=["one", "two"])
        for count in (0, 3, 8192, True):
            with self.assertRaises(ValueError):
                MaterialGripperSchedule(self.recipe, count, gripper_ids=["one", "two"])
        for fraction in (True, -.1, 1.1, np.nan, np.inf, Fraction(1, 3), Fraction(1, 2 ** 41)):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                self.schedule.parameters(fraction)

    def test_constant_admission_bound_targets_remain_exact_on_uneven_intervals(self):
        recipe = {"profile": SCHEDULE_PROFILE, "gripperIds": ["one"], "knots": [
            {"fraction": fraction, "targetsMeters": [[100., -100., 100.]], "activation": [1.]}
            for fraction in (0., 1. / 16, 1.)]}
        schedule = MaterialGripperSchedule(recipe, 16, gripper_ids=["one"])
        for tick in range(257):
            target, activation = schedule.parameters(Fraction(tick, 256))
            np.testing.assert_array_equal(target, [[100., -100., 100.]])
            np.testing.assert_array_equal(activation, [1.])


if __name__ == "__main__":
    unittest.main()
