"""Independent discrete sewing work and quadratic dynamics checks."""

import copy
from decimal import Decimal, localcontext
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy.sparse import block_diag, csr_matrix

from solver_bending import ElasticDihedralBending
from solver_distance_sewing import DistanceSewing
from solver_energy_balance import global_energy_transition
from solver_material_grippers import MaterialGrippers
from solver_normal_sewing import NormalOffsetSewing


def fixture(mode="vector", *, compliance=.04, rows=None):
    rows = [[-1., 0., 0., 1.]] if rows is None else rows
    matrix = csr_matrix(rows)
    count = matrix.shape[1]
    solver = SimpleNamespace(mass=np.arange(1, count + 1, dtype=float), active=np.ones(count, dtype=bool),
        sewing=matrix, compliance=compliance, sewing_mode=mode,
        bending=ElasticDihedralBending(count, np.empty((0, 4), dtype=int), [], [], []),
        poses=np.empty((0, 2, 2)), faces=np.empty((0, 3), dtype=int), areas=np.empty(0), materials=np.empty((0, 3)),
        sewing_frame_faces=np.array([[0, 1, 2]] * matrix.shape[0], dtype=int),
        sewing_sides=np.ones(matrix.shape[0], dtype=int))
    def potential(targets, *, activation=None):
        if mode == "normal-offset":
            return NormalOffsetSewing(solver.sewing, targets, solver.compliance,
                solver.sewing_frame_faces, solver.sewing_sides, activation=activation)
        return DistanceSewing(solver.sewing, targets, solver.compliance, activation=activation)
    solver.sewing_potential = potential
    return solver


def positions(height=.8):
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [.2, .3, height]])


def targets(mode, value):
    return np.array([[0., 0., value]]) if mode == "vector" else np.array([value])


def transition(solver, before, after, old_targets, new_targets, old_activation, activation, *, velocity=None, dt=1., **extra):
    return global_energy_transition(solver, before, after, np.zeros_like(before) if velocity is None else velocity,
        (after - before) / dt, old_targets, new_targets, dt,
        previous_sewing_activation=old_activation, sewing_activation=activation, **extra)


def oracle(solver, before, after, old_targets, new_targets, old_activation, activation):
    """180-digit endpoint energies; geometry samples independently reconstructed.

    Irrational lengths/normals deliberately use the declared binary64 sampling
    convention. No implementation residual or work helper is called.
    """
    with localcontext() as context:
        context.prec = 180
        d = lambda value: Decimal.from_float(float(value))
        matrix = solver.sewing.toarray()
        mode = solver.sewing_mode
        def errors(q, target):
            result = []
            for index, row in enumerate(matrix):
                if old_activation[index] == 0 and activation[index] == 0:
                    result.append([Decimal(0)])
                    continue
                anchor = [sum((d(weight) * d(point[axis]) for weight, point in zip(row, q)), Decimal(0))
                          for axis in range(3)]
                if mode == "distance":
                    length = float(np.linalg.norm(row @ q))
                    result.append([d(length) - d(target[index])])
                elif mode == "normal-offset":
                    triangle = q[solver.sewing_frame_faces[index]]
                    cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                    normal = cross / np.linalg.norm(cross)
                    result.append([value - d(target[index]) * int(solver.sewing_sides[index]) * d(axis)
                                   for value, axis in zip(anchor, normal)])
                else:
                    result.append([value - d(goal) for value, goal in zip(anchor, target[index])])
            return [sum((value ** 2 for value in row), Decimal(0)) / (2 * d(solver.compliance)) for row in result]
        before_errors, fixed_errors, after_errors = errors(before, old_targets), errors(before, new_targets), errors(after, new_targets)
        before_energy = sum((d(a) * e for a, e in zip(old_activation, before_errors)), Decimal(0))
        fixed_energy = sum((d(a) * e for a, e in zip(activation, fixed_errors)), Decimal(0))
        after_energy = sum((d(a) * e for a, e in zip(activation, after_errors)), Decimal(0))
        target_work = sum((d(a) * (new - old) for a, old, new in zip(old_activation, before_errors, fixed_errors)), Decimal(0))
        activation_terms = [(d(new) - d(old)) * error for old, new, error in zip(old_activation, activation, fixed_errors)]
        return {"sewingBeforeJoules": float(before_energy), "sewingAfterJoules": float(after_energy),
            "sewingFixedPositionAfterJoules": float(fixed_energy),
            "sewingFixedParameterChangeJoules": float(after_energy - fixed_energy),
            "sewingChangeJoules": float(after_energy - before_energy),
            "sewingTargetParameterWorkJoules": float(target_work),
            "sewingActivationParameterWorkJoules": float(sum(activation_terms, Decimal(0))),
            "sewingParameterWorkJoules": float(fixed_energy - before_energy),
            "sewingActivationIncreaseWorkJoules": float(sum((max(value, 0) for value in activation_terms), Decimal(0))),
            "sewingReleaseEnergyRemovedJoules": float(sum((max(-value, 0) for value in activation_terms), Decimal(0)))}


class SewingActivationEnergyTests(unittest.TestCase):
    def assert_accounting(self, result, expected):
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertAlmostEqual(result[key], value, delta=2e-12 * max(1., abs(value)))
        self.assertFalse(result["accepted"])

    def test_vector_distance_normal_match_independent_endpoint_accounting(self):
        before = positions()
        after = before + [[.02, -.01, .01], [-.01, .01, .03], [.03, -.02, -.01], [.04, .01, .06]]
        for mode in ("vector", "distance", "normal-offset"):
            solver = fixture(mode, rows=[[-1., 0., 0., 1.], [0., -.5, -.5, 1.]])
            old = np.repeat(targets(mode, .2), 2, axis=0)
            new = np.repeat(targets(mode, .4), 2, axis=0)
            with self.subTest(mode=mode):
                result = transition(solver, before, after, old, new, [.2, .8], [.7, .3])
                expected = oracle(solver, before, after, old, new, [.2, .8], [.7, .3])
                self.assert_accounting(result, expected)
                self.assertAlmostEqual(result["mechanicalChangeJoules"], expected["sewingChangeJoules"] + result["kineticChangeJoules"])
                self.assertEqual(result["targetParameterWorkJoules"], result["sewingTargetParameterWorkJoules"])
                self.assertEqual(result["externalParameterWorkJoules"], result["sewingParameterWorkJoules"])
                self.assertAlmostEqual(result["mechanicalChangeMinusParameterWorkJoules"],
                    expected["sewingFixedParameterChangeJoules"] + result["kineticChangeJoules"])
                self.assertAlmostEqual(result["mechanicalChangeMinusTargetWorkJoules"],
                    expected["sewingFixedParameterChangeJoules"] + expected["sewingActivationParameterWorkJoules"]
                    + result["kineticChangeJoules"])

    def test_engagement_release_and_inactive_target_motion_have_explicit_work(self):
        q = positions()
        for mode in ("vector", "distance", "normal-offset"):
            solver = fixture(mode)
            old, new = targets(mode, .2), targets(mode, .4)
            for a0, a1 in (([0.], [.7]), ([.7], [0.]), ([0.], [0.])):
                with self.subTest(mode=mode, before=a0, after=a1):
                    result = transition(solver, q, q, old, new, a0, a1)
                    self.assert_accounting(result, oracle(solver, q, q, old, new, a0, a1))
                    self.assertEqual(result["sewingFixedParameterChangeJoules"], 0.)
                    self.assertEqual(result["mechanicalChangeMinusParameterWorkJoules"], 0.)
                    if a0 == [0.]:
                        self.assertEqual(result["sewingTargetParameterWorkJoules"], 0.)
                    if a1 == [0.]:
                        self.assertEqual(result["sewingAfterJoules"], 0.)
                    if a0 == [0.] and a1 != [0.]:
                        self.assertGreater(result["sewingActivationIncreaseWorkJoules"], 0.)
                    if a0 != [0.] and a1 == [0.]:
                        self.assertGreater(result["sewingReleaseEnergyRemovedJoules"], 0.)

    def test_all_pending_rows_skip_huge_residual_and_degenerate_geometry(self):
        q = np.array([[1e308, 0., 0.], [1e308, 0., 0.], [1e308, 0., 0.], [-1e308, 0., 0.]])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for mode in ("vector", "distance", "normal-offset"):
                solver = fixture(mode)
                old, new = targets(mode, 1e308), targets(mode, -1e308 if mode == "vector" else 1e300)
                with self.subTest(mode=mode):
                    result = transition(solver, q, q, old, new, [0.], [0.])
                    self.assertTrue(all(value == 0 for key, value in result.items()
                                        if key.endswith("Joules") and "ErrorBound" not in key))

    def test_partial_activation_never_evaluates_huge_pending_rows(self):
        for mode in ("vector", "distance", "normal-offset"):
            solver = fixture(mode, rows=block_diag((csr_matrix([[-1., 0., 0., 1.]]),) * 2).toarray())
            solver.sewing_frame_faces = np.array([[0, 1, 2], [4, 5, 6]])
            q = np.concatenate((np.array([[1e308, 0., 0.]] * 3 + [[-1e308, 0., 0.]]), positions()))
            old = np.concatenate((targets(mode, 1e308), targets(mode, .2)))
            new = np.concatenate((targets(mode, -1e308 if mode == "vector" else 1e300), targets(mode, .3)))
            with self.subTest(mode=mode), np.errstate(over="raise", invalid="raise", divide="raise"):
                result = transition(solver, q, q, old, new, [0., .2], [0., .8])
                reference = transition(fixture(mode), positions(), positions(), targets(mode, .2), targets(mode, .3), [.2], [.8])
                for key, value in reference.items():
                    if key.endswith("Joules"):
                        self.assertEqual(result[key], value)

    def test_newly_active_coincident_distance_or_degenerate_frame_rejects(self):
        for mode in ("distance", "normal-offset"):
            solver = fixture(mode)
            q = np.zeros((4, 3))
            target = targets(mode, .1)
            with self.subTest(mode=mode):
                transition(solver, q, q, target, target, [0.], [0.])
                with self.assertRaises(ValueError):
                    transition(solver, q, q, target, target, [0.], [1.])
                # A released row need not define sewing geometry at the new q.
                # The enclosing solver independently guards all cloth triangles.
                result = transition(solver, positions(), q, target, target, [1.], [0.])
                self.assertEqual(result["sewingAfterJoules"], 0.)

    def test_decimal_target_activation_cancellation_retains_small_total(self):
        q = positions(3.)
        q[3, :2] = 0.
        a1 = float(np.nextafter(.25, 1.))
        for mode in ("vector", "distance", "normal-offset"):
            solver = fixture(mode, compliance=1e-12)
            old, new = targets(mode, 2.), targets(mode, 1.)
            with self.subTest(mode=mode):
                result = transition(solver, q, q, old, new, [1.], [a1])
                expected = oracle(solver, q, q, old, new, [1.], [a1])
                for key in expected:
                    self.assertEqual(result[key], expected[key])
                rounded_sum = math.fsum((result["sewingTargetParameterWorkJoules"], result["sewingActivationParameterWorkJoules"]))
                self.assertNotEqual(rounded_sum, result["sewingParameterWorkJoules"])
                self.assertLessEqual(abs(rounded_sum - result["sewingParameterWorkJoules"]),
                    result["sewingParameterWorkComponentSumErrorBoundJoules"])
                self.assertNotEqual(result["sewingParameterWorkJoules"], 0.)
                self.assertEqual(result["mechanicalChangeJoules"], result["sewingParameterWorkJoules"])
                self.assertEqual(result["mechanicalChangeMinusParameterWorkJoules"], 0.)

    def test_decimal_tiny_target_motion_survives_equal_rounded_endpoint_energies(self):
        q = positions(100.)
        q[3, :2] = 0.
        for mode in ("vector", "distance", "normal-offset"):
            solver = fixture(mode, compliance=1e-12)
            old, new = targets(mode, 1.), targets(mode, math.nextafter(1., math.inf))
            with self.subTest(mode=mode):
                result = transition(solver, q, q, old, new, [1.], [1.])
                expected = oracle(solver, q, q, old, new, [1.], [1.])
                self.assertEqual(result["sewingBeforeJoules"], result["sewingAfterJoules"])
                self.assertEqual(result["sewingParameterWorkJoules"], expected["sewingParameterWorkJoules"])
                self.assertNotEqual(result["sewingParameterWorkJoules"], 0.)

    def test_decimal_fixed_parameter_motion_uses_stable_source_displacement(self):
        q = positions(100.)
        q[3, :2] = 0.
        after = q.copy()
        after[3, 2] = math.nextafter(100., math.inf)
        for mode in ("vector", "distance", "normal-offset"):
            solver = fixture(mode, compliance=1e-12)
            target = targets(mode, 1.)
            with self.subTest(mode=mode):
                result = transition(solver, q, after, target, target, [.7], [.7])
                expected = oracle(solver, q, after, target, target, [.7], [.7])
                self.assertEqual(result["sewingFixedParameterChangeJoules"], expected["sewingFixedParameterChangeJoules"])
                self.assertNotEqual(result["sewingAfterJoules"] - result["sewingBeforeJoules"],
                                    result["sewingFixedParameterChangeJoules"])

    def test_vector_backward_euler_and_multistep_telescope(self):
        solver = fixture(rows=[[-1., 0., 0., 1.], [0., -.5, -.5, 1.]])
        matrix, q = solver.sewing.toarray(), positions()
        velocity = np.array([[.1, -.1, .2], [.2, 0., -.1], [0., .2, .1], [-.1, .1, 0.]])
        target, activation = np.array([[0., 0., .1], [.1, -.1, .2]]), np.array([0., .3])
        def energy(q, v, t, a):
            return np.sum(solver.mass[:, None] * v ** 2) / 2 + np.sum(a[:, None] * (matrix @ q - t) ** 2) / (2 * solver.compliance)
        before_energy, reports = energy(q, velocity, target, activation), []
        for index, next_activation in enumerate(([.2, .6], [1., .4], [.5, .9], [0., 0.])):
            dt = .02 * (index + 1)
            next_target = target + [[.03, -.04, .02], [-.02, .01, .04]]
            next_activation = np.array(next_activation)
            spring = matrix.T @ np.diag(next_activation) @ matrix / solver.compliance
            diagonal = np.diag(solver.mass / dt ** 2)
            after = np.linalg.solve(diagonal + spring,
                diagonal @ (q + dt * velocity) + matrix.T @ (next_activation[:, None] * next_target) / solver.compliance)
            next_velocity = (after - q) / dt
            result = transition(solver, q, after, target, next_target, activation, next_activation, velocity=velocity, dt=dt)
            loss = np.sum(solver.mass[:, None] * (next_velocity - velocity) ** 2) / 2
            loss += np.sum((after - q) * (spring @ (after - q))) / 2
            self.assertAlmostEqual(result["mechanicalChangeMinusParameterWorkJoules"], -loss, places=12)
            reports.append(result)
            q, velocity, target, activation = after, next_velocity, next_target, next_activation
        expected = energy(q, velocity, target, activation) - before_energy
        self.assertAlmostEqual(math.fsum(item["mechanicalChangeJoules"] for item in reports), expected, places=12)
        self.assertAlmostEqual(math.fsum(item["externalParameterWorkJoules"] + item["mechanicalChangeMinusParameterWorkJoules"]
                                        for item in reports), expected, places=12)

    def test_optional_pair_strict_activation_and_legacy_path(self):
        solver, q = fixture(), positions()
        target = targets("vector", .1)
        arguments = (solver, q, q, np.zeros_like(q), np.zeros_like(q), target, target, 1.)
        with mock.patch("solver_energy_balance._weighted_sewing_transition", side_effect=AssertionError("Legacy path changed")):
            legacy = global_energy_transition(*arguments)
            self.assertEqual(legacy, global_energy_transition(*arguments, previous_sewing_activation=None, sewing_activation=None))
            self.assertNotIn("sewingParameterWorkJoules", legacy)
        for key in ("previous_sewing_activation", "sewing_activation"):
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "Both endpoint"):
                global_energy_transition(*arguments, **{key: [1.]})
        for bad in ([True], [np.bool_(False)], [np.nan], [np.inf], [-.1], [1.1], [], [[1.]], [1., 0.], ["1"]):
            for key in ("previous_sewing_activation", "sewing_activation"):
                parameters = {"previous_sewing_activation": [1.], "sewing_activation": [1.], key: bad}
                with self.subTest(key=key, invalid=bad), self.assertRaises(ValueError):
                    global_energy_transition(*arguments, **parameters)

    def test_coupled_fold_and_gripper_accounting_remains_separate(self):
        solver, q = fixture(), positions()
        solver.fold_actuation = SimpleNamespace(potential=lambda target:
            ElasticDihedralBending(4, np.array([[0, 1, 2, 3]]), target, [1.], [.03]))
        solver.material_grippers = MaterialGrippers(["panel"] * 4, [[0, 1, 2]], [{"id": "grip", "instanceId": "panel",
            "triangleIndex": 0, "weights": [.25, .25, .5], "stiffnessNPerM": 7.}])
        after = q + [[.01, .02, -.01], [-.02, .01, .01], [.01, -.01, .02], [.02, .01, .03]]
        old, new = targets("vector", .2), targets("vector", .3)
        extras = {"previous_fold_targets": [.1], "fold_targets": [.3],
            "previous_gripper_targets": [[.2, .3, .1]], "gripper_targets": [[.3, .1, .2]],
            "previous_gripper_activation": [.2], "gripper_activation": [.8]}
        result = transition(solver, q, after, old, new, [.4], [.7], **extras)
        baseline = copy.copy(solver)
        baseline.sewing = csr_matrix((0, 4))
        other = global_energy_transition(baseline, q, after, np.zeros_like(q), after - q,
            np.empty((0, 3)), np.empty((0, 3)), 1., **extras)
        for key, value in other.items():
            if key.startswith(("fold", "gripper")):
                self.assertEqual(result[key], value)
        for total, sewing in (("mechanicalChangeJoules", "sewingChangeJoules"),
                ("externalParameterWorkJoules", "sewingParameterWorkJoules"),
                ("targetParameterWorkJoules", "sewingTargetParameterWorkJoules"),
                ("mechanicalChangeMinusParameterWorkJoules", "sewingFixedParameterChangeJoules")):
            self.assertAlmostEqual(result[total], other[total] + result[sewing], places=12)
        self.assertAlmostEqual(result["mechanicalChangeMinusTargetWorkJoules"],
            other["mechanicalChangeMinusTargetWorkJoules"] + result["sewingFixedParameterChangeJoules"]
            + result["sewingActivationParameterWorkJoules"], places=12)


if __name__ == "__main__":
    unittest.main()
