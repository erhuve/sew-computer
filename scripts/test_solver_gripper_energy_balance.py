"""Independent quadratic and Decimal gripper-work accounting checks."""

import copy
from decimal import Decimal, localcontext
import math
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from solver_bending import ElasticDihedralBending
from solver_energy_balance import global_energy_transition
from solver_material_grippers import MaterialGrippers


class GripperEnergyBalanceTests(unittest.TestCase):
    def fixture(self, *, pinned=False, sewing=True, weights=None, stiffness=None):
        weights = np.array([[.2, .3, .5], [1., 0., 0.]] if weights is None else weights)
        stiffness = np.array([25., 11.] if stiffness is None else stiffness)
        recipe = MaterialGrippers(["cloth:shell"] * 3, [[0, 1, 2]], [
            {"id": f"grip-{index}", "instanceId": "cloth:shell", "triangleIndex": 0,
             "weights": row.tolist(), "stiffnessNPerM": float(k)}
            for index, (row, k) in enumerate(zip(weights, stiffness))])
        solver = SimpleNamespace(
            mass=np.array([0. if pinned else 2., 3., 4.]), active=np.array([not pinned, True, True]),
            sewing=csr_matrix([[1., -1., 0.], [0., 1., -1.]] if sewing else np.empty((0, 3))), compliance=.04,
            bending=ElasticDihedralBending(3, np.empty((0, 4), dtype=int), [], [], []),
            poses=np.empty((0, 2, 2)), faces=np.empty((0, 3), dtype=int),
            areas=np.empty(0), materials=np.empty((0, 3)), material_grippers=recipe)
        return solver, weights, stiffness

    @staticmethod
    def mechanical(solver, rows, stiffness, q, v, sewing_targets, gripper_targets, activation):
        sewing = solver.sewing @ q - sewing_targets
        grips = rows @ q - gripper_targets
        return (np.sum(solver.mass[:, None] * v ** 2) / 2
                + np.sum(sewing ** 2) / (2 * solver.compliance)
                + np.sum((stiffness * activation)[:, None] * grips ** 2) / 2)

    @staticmethod
    def backward_euler(solver, rows, stiffness, q, v, sewing_targets, gripper_targets, activation, dt):
        spring_matrix = solver.sewing.toarray().T @ solver.sewing.toarray() / solver.compliance
        spring_matrix += rows.T @ np.diag(stiffness * activation) @ rows
        matrix = np.diag(solver.mass / dt ** 2) + spring_matrix
        rhs = solver.mass[:, None] / dt ** 2 * (q + dt * v)
        rhs += solver.sewing.T @ sewing_targets / solver.compliance
        rhs += rows.T @ ((stiffness * activation)[:, None] * gripper_targets)
        active, fixed = np.flatnonzero(solver.active), np.flatnonzero(~solver.active)
        final = q.copy()
        final[active] = np.linalg.solve(matrix[np.ix_(active, active)],
            rhs[active] - matrix[np.ix_(active, fixed)] @ q[fixed])
        return final, (final - q) / dt, spring_matrix

    def check_step(self, solver, rows, stiffness, q, v, old_sewing, sewing, old_targets, targets,
                   old_activation, activation, dt=.05):
        final, velocity, spring_matrix = self.backward_euler(solver, rows, stiffness, q, v,
            sewing, targets, activation, dt)
        result = global_energy_transition(solver, q, final, v, velocity, old_sewing, sewing, dt,
            previous_gripper_targets=old_targets, gripper_targets=targets,
            previous_gripper_activation=old_activation, gripper_activation=activation)
        old_error, target_error = rows @ q - old_targets, rows @ q - targets
        grip_before = np.sum((stiffness * old_activation)[:, None] * old_error ** 2) / 2
        grip_fixed = np.sum((stiffness * activation)[:, None] * target_error ** 2) / 2
        grip_after = np.sum((stiffness * activation)[:, None] * (rows @ final - targets) ** 2) / 2
        grip_target_work = np.sum((stiffness * old_activation)[:, None] * (target_error ** 2 - old_error ** 2)) / 2
        grip_activation_work = np.sum((stiffness * (activation - old_activation))[:, None] * target_error ** 2) / 2
        increase = np.sum((stiffness * np.maximum(activation - old_activation, 0))[:, None] * target_error ** 2) / 2
        release = np.sum((stiffness * np.maximum(old_activation - activation, 0))[:, None] * target_error ** 2) / 2
        sewing_work = (np.sum((solver.sewing @ q - sewing) ** 2)
                       - np.sum((solver.sewing @ q - old_sewing) ** 2)) / (2 * solver.compliance)
        mechanical = self.mechanical(solver, rows, stiffness, final, velocity, sewing, targets, activation) - self.mechanical(
            solver, rows, stiffness, q, v, old_sewing, old_targets, old_activation)
        displacement = final - q
        dissipation = (np.sum(solver.mass[:, None] * (velocity - v) ** 2) / 2
                       + np.sum(displacement * (spring_matrix @ displacement)) / 2)
        expected = {"gripperBeforeJoules": grip_before, "gripperAfterJoules": grip_after,
            "gripperFixedPositionAfterJoules": grip_fixed,
            "gripperFixedParameterChangeJoules": grip_after - grip_fixed,
            "gripperChangeJoules": grip_after - grip_before,
            "gripperTargetParameterWorkJoules": grip_target_work,
            "gripperActivationParameterWorkJoules": grip_activation_work,
            "gripperActivationIncreaseWorkJoules": increase, "gripperReleaseEnergyRemovedJoules": release,
            "gripperParameterWorkJoules": grip_fixed - grip_before,
            "targetParameterWorkJoules": sewing_work + grip_target_work,
            "externalParameterWorkJoules": sewing_work + grip_fixed - grip_before,
            "mechanicalChangeJoules": mechanical,
            "mechanicalChangeMinusParameterWorkJoules": -dissipation,
            "mechanicalChangeMinusTargetWorkJoules": -dissipation + grip_activation_work}
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertAlmostEqual(result[key], value, delta=3e-12 * max(1., abs(value)))
        self.assertIs(result["accepted"], False)
        return final, velocity, result

    def test_fixed_targets_exact_quadratic_backward_euler(self):
        solver, rows, stiffness = self.fixture()
        q = np.array([[.2, -.1, .3], [1.1, .4, -.2], [-.3, .2, .1]])
        v = np.array([[.3, -.2, .1], [-.2, .5, .1], [.1, -.1, .2]])
        sewing, targets = np.array([[.1, -.2, .05], [-.2, .1, .3]]), np.array([[.4, .2, .1], [0., .1, .2]])
        _, _, result = self.check_step(solver, rows, stiffness, q, v, sewing, sewing, targets, targets,
                                       np.array([.4, .8]), np.array([.4, .8]))
        self.assertEqual(result["externalParameterWorkJoules"], 0.)
        self.assertLess(result["mechanicalChangeMinusParameterWorkJoules"], 0.)

    def test_mixed_activation_release_and_target_motion_with_fixed_vertex(self):
        solver, rows, stiffness = self.fixture(pinned=True)
        q = np.array([[.2, -.1, .3], [1.1, .4, -.2], [-.3, .2, .1]])
        v = np.array([[0., 0., 0.], [-.2, .5, .1], [.1, -.1, .2]])
        _, _, result = self.check_step(solver, rows, stiffness, q, v,
            np.zeros((2, 3)), np.array([[.1, -.2, .05], [-.2, .1, .3]]),
            np.array([[.4, .2, .1], [0., .1, .2]]), np.array([[.3, -.2, .5], [.6, .2, -.1]]),
            np.array([.2, .9]), np.array([.8, .4]))
        self.assertGreater(result["gripperActivationIncreaseWorkJoules"], 0.)
        self.assertGreater(result["gripperReleaseEnergyRemovedJoules"], 0.)
        self.assertNotEqual(result["externalParameterWorkJoules"], result["targetParameterWorkJoules"])

    def test_multiple_backward_euler_steps_telescope_with_final_release(self):
        solver, rows, stiffness = self.fixture()
        q = np.array([[.2, -.1, .3], [1.1, .4, -.2], [-.3, .2, .1]])
        v = np.zeros_like(q)
        sewing, targets, activation = np.zeros((2, 3)), np.zeros((2, 3)), np.array([0., .3])
        before = self.mechanical(solver, rows, stiffness, q, v, sewing, targets, activation)
        reports = []
        for index, next_activation in enumerate(([.2, .6], [1., .4], [.5, .9], [0., 0.])):
            next_sewing = sewing + [.03, -.01, .02]
            next_targets = targets + [[.08, -.04, .03], [-.02, .06, .01]]
            next_activation = np.asarray(next_activation)
            q, v, result = self.check_step(solver, rows, stiffness, q, v, sewing, next_sewing,
                targets, next_targets, activation, next_activation, dt=.01 * (index + 1))
            reports.append(result)
            sewing, targets, activation = next_sewing, next_targets, next_activation
        change = self.mechanical(solver, rows, stiffness, q, v, sewing, targets, activation) - before
        self.assertAlmostEqual(math.fsum(row["mechanicalChangeJoules"] for row in reports), change, places=11)
        self.assertAlmostEqual(math.fsum(row["externalParameterWorkJoules"] + row["mechanicalChangeMinusParameterWorkJoules"]
                                       for row in reports), change, places=11)
        self.assertEqual(reports[-1]["gripperAfterJoules"], 0.)

    def static_report(self, old_targets, targets, old_activation, activation, *, q=None, stiffness=25.):
        solver, _, _ = self.fixture(sewing=False, weights=[[1., 0., 0.]], stiffness=[stiffness])
        q = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]) if q is None else q
        return global_energy_transition(solver, q, q, np.zeros_like(q), np.zeros_like(q),
            np.empty((0, 3)), np.empty((0, 3)), 1., previous_gripper_targets=old_targets, gripper_targets=targets,
            previous_gripper_activation=old_activation, gripper_activation=activation)

    def test_target_first_release_removes_potential_without_claiming_damping(self):
        result = self.static_report([[1., 0., 0.]], [[3., 0., 0.]], [1.], [0.])
        self.assertEqual(result["gripperBeforeJoules"], 0.)
        self.assertEqual(result["gripperAfterJoules"], 0.)
        self.assertEqual(result["gripperTargetParameterWorkJoules"], 50.)
        self.assertEqual(result["gripperReleaseEnergyRemovedJoules"], 50.)
        self.assertEqual(result["gripperActivationParameterWorkJoules"], -50.)
        self.assertEqual(result["externalParameterWorkJoules"], 0.)
        self.assertEqual(result["mechanicalChangeMinusTargetWorkJoules"], -50.)
        self.assertEqual(result["mechanicalChangeMinusParameterWorkJoules"], 0.)
        self.assertIn("not claimed physical dissipation", result["scope"])

    def test_inactive_target_motion_is_zero_work(self):
        result = self.static_report([[1., 0., 0.]], [[3., 2., -1.]], [0.], [0.])
        self.assertTrue(all(value == 0. for key, value in result.items()
                            if key.endswith("Joules") and "ErrorBound" not in key))

    def test_decimal_tiny_target_change_survives_equal_rounded_energies(self):
        q = np.array([[100., 0., 0.], [99., 0., 0.], [100., 1., 0.]])
        shift = 2. ** -50
        result = self.static_report([[0., 0., 0.]], [[shift, 0., 0.]], [1.], [1.], q=q, stiffness=1e12)
        with localcontext() as context:
            context.prec = 100
            expected = float(Decimal.from_float(1e12) / 2 *
                ((Decimal(100) - Decimal.from_float(shift)) ** 2 - Decimal(100) ** 2))
        self.assertEqual(result["gripperBeforeJoules"], result["gripperAfterJoules"])
        self.assertNotEqual(expected, 0.)
        self.assertEqual(result["gripperParameterWorkJoules"], expected)
        self.assertEqual(result["mechanicalChangeJoules"], expected)
        self.assertEqual(result["mechanicalChangeMinusParameterWorkJoules"], 0.)

    def test_decimal_total_survives_target_activation_component_cancellation(self):
        activation = np.nextafter(.25, np.inf)
        result = self.static_report([[0., 0., 0.]], [[-1., 0., 0.]], [1.], [activation], stiffness=1e12)
        with localcontext() as context:
            context.prec = 100
            k, new_activation = Decimal.from_float(1e12), Decimal.from_float(float(activation))
            expected_total = float(k / 2 * (4 * new_activation - 1))
            expected_target, expected_activation = float(k * Decimal('1.5')), float(k * 2 * (new_activation - 1))
        self.assertEqual(result["gripperParameterWorkJoules"], expected_total)
        self.assertEqual(result["gripperTargetParameterWorkJoules"], expected_target)
        self.assertEqual(result["gripperActivationParameterWorkJoules"], expected_activation)
        component_sum = math.fsum((expected_target, expected_activation))
        self.assertNotEqual(component_sum, expected_total)
        self.assertLessEqual(abs(component_sum - expected_total), result["gripperParameterWorkComponentSumErrorBoundJoules"])
        self.assertEqual(result["externalParameterWorkJoules"], expected_total)
        self.assertEqual(result["mechanicalChangeJoules"], expected_total)
        reverse = self.static_report([[-1., 0., 0.]], [[0., 0., 0.]], [activation], [1.], stiffness=1e12)
        self.assertEqual(math.fsum((result["mechanicalChangeJoules"], reverse["mechanicalChangeJoules"])), 0.)

    def test_decimal_fixed_parameter_motion_avoids_endpoint_subtraction(self):
        solver, _, _ = self.fixture(sewing=False, weights=[[1., 0., 0.]], stiffness=[1e12])
        q = np.array([[99., 0., 0.], [98., 1., 0.], [98., 0., 0.]])
        final = q.copy()
        final[0, 0] = np.nextafter(q[0, 0], np.inf)
        velocity = final - q
        result = global_energy_transition(solver, q, final, np.zeros_like(q), velocity,
            np.empty((0, 3)), np.empty((0, 3)), 1., previous_gripper_targets=[[0., 0., 0.]],
            gripper_targets=[[0., 0., 0.]], previous_gripper_activation=[1.], gripper_activation=[1.])
        with localcontext() as context:
            context.prec = 100
            before, after = Decimal.from_float(q[0, 0]), Decimal.from_float(final[0, 0])
            expected = float(Decimal.from_float(1e12) / 2 * (after ** 2 - before ** 2))
        self.assertAlmostEqual(result["gripperFixedParameterChangeJoules"], expected, delta=1e-14)
        self.assertNotEqual(result["gripperAfterJoules"] - result["gripperBeforeJoules"], expected)

    def test_all_four_gripper_arguments_required_iff_recipe_exists(self):
        solver, _, _ = self.fixture(sewing=False)
        q, velocity, targets = np.zeros((3, 3)), np.zeros((3, 3)), np.empty((0, 3))
        parameters = {"previous_gripper_targets": [[0., 0., 0.]] * 2, "gripper_targets": [[0., 0., 0.]] * 2,
                      "previous_gripper_activation": [0., 0.], "gripper_activation": [0., 0.]}
        names = list(parameters)
        for mask in range(16):
            selected = {name: parameters[name] for index, name in enumerate(names) if mask & (1 << index)}
            solver.material_grippers = None
            if selected:
                with self.subTest(recipe=False, mask=mask), self.assertRaisesRegex(ValueError, "recipe"):
                    global_energy_transition(solver, q, q, velocity, velocity, targets, targets, 1., **selected)
            else:
                result = global_energy_transition(solver, q, q, velocity, velocity, targets, targets, 1.)
                self.assertEqual(result["targetParameterWorkJoules"], result["externalParameterWorkJoules"])
                self.assertEqual(result["mechanicalChangeMinusTargetWorkJoules"], result["mechanicalChangeMinusParameterWorkJoules"])
            solver.material_grippers = self.fixture(sewing=False)[0].material_grippers
            if mask < 15:
                with self.subTest(recipe=True, mask=mask), self.assertRaisesRegex(ValueError, "Both endpoint"):
                    global_energy_transition(solver, q, q, velocity, velocity, targets, targets, 1., **selected)
            else:
                global_energy_transition(solver, q, q, velocity, velocity, targets, targets, 1., **selected)

    def test_recipe_validation_and_nonfinite_work_cannot_be_hidden(self):
        solver, _, _ = self.fixture(sewing=False, weights=[[1., 0., 0.]], stiffness=[25.])
        recipe = solver.material_grippers
        q, velocity, targets = np.zeros((3, 3)), np.zeros((3, 3)), np.empty((0, 3))
        parameters = {"previous_gripper_targets": [[0., 0., 0.]], "gripper_targets": [[0., 0., 0.]],
                      "previous_gripper_activation": [0.], "gripper_activation": [0.]}
        for key, invalid in (("gripper_targets", [[np.nan, 0., 0.]]), ("gripper_targets", [0.]),
                             ("previous_gripper_activation", [True]), ("gripper_activation", [1.1])):
            changed = copy.deepcopy(parameters)
            changed[key] = invalid
            with self.subTest(parameter=key, value=invalid), self.assertRaises(ValueError):
                global_energy_transition(solver, q, q, velocity, velocity, targets, targets, 1., **changed)
        for field, bad in (("totalParameterWorkJoules", np.inf), ("releaseEnergyRemovedJoules", -1.),
                           ("parameterOrder", "activation-first")):
            def corrupt(*args, field=field, bad=bad):
                values = recipe.parameter_energy_change(*args)
                values[field] = bad
                return values
            solver.material_grippers = SimpleNamespace(potential=recipe.potential, parameter_energy_change=corrupt)
            with self.subTest(field=field), self.assertRaises(ValueError):
                global_energy_transition(solver, q, q, velocity, velocity, targets, targets, 1., **parameters)


if __name__ == "__main__":
    unittest.main()
