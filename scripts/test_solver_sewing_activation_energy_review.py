"""Independent high-precision review of explicit sewing activation accounting.

The Decimal oracle evaluates endpoint potentials directly from original binary
inputs. Axis-aligned distance/frame fixtures make real geometry exact, so tiny
motion tests do not reuse the implementation's rationalized-delta formula.
"""

from decimal import Decimal, localcontext
import math
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from solver_bending import ElasticDihedralBending
from solver_distance_sewing import DistanceSewing
from solver_energy_balance import global_energy_transition
from solver_material_grippers import MaterialGrippers
from solver_normal_sewing import NormalOffsetSewing


def decimal(value):
    return Decimal.from_float(float(value))


def fixture(rows, mode="vector", compliance=.003):
    rows = csr_matrix(rows)
    count = rows.shape[1]
    model = SimpleNamespace(
        mass=np.ones(count), active=np.ones(count, dtype=bool), sewing=rows,
        compliance=compliance, sewing_mode=mode,
        bending=ElasticDihedralBending(count, np.empty((0, 4), dtype=int), [], [], []),
        poses=np.empty((0, 2, 2)), faces=np.empty((0, 3), dtype=int),
        areas=np.empty(0), materials=np.empty((0, 3)))
    if mode == "distance":
        model.sewing_potential = lambda targets, *, activation=None: DistanceSewing(
            rows, targets, compliance, activation=activation)
    elif mode == "normal-offset":
        model.sewing_frames = np.array([[0, 1, 2]])
        model.sewing_sides = np.array([1])
        model.sewing_potential = lambda targets, *, activation=None: NormalOffsetSewing(
            rows, targets, compliance, model.sewing_frames, model.sewing_sides, activation=activation)
    return model


def report(model, start, end, old_targets, targets, old_activation, activation, **parameters):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return global_energy_transition(model, start, end, np.zeros_like(start), end - start,
        old_targets, targets, 1., previous_sewing_activation=old_activation,
        sewing_activation=activation, **parameters)


class SewingActivationEnergyReviewTests(unittest.TestCase):
    def test_pending_geometry_and_huge_target_changes_are_never_evaluated(self):
        rows = [[-1., 0., 0., 1.]]
        positions = np.zeros((4, 3))
        for mode in ("vector", "distance", "normal-offset"):
            with self.subTest(mode=mode):
                model = fixture(rows, mode)
                old_targets = [[1e308, -1e308, 1e308]] if mode == "vector" else [1e308]
                targets = [[-1e308, 1e308, -1e308]] if mode == "vector" else [5e307]
                with np.errstate(all="raise"):
                    result = report(model, positions, positions, old_targets, targets, [0.], [0.])
                for key, value in result.items():
                    if key.endswith("Joules") and "ErrorBound" not in key:
                        self.assertEqual(value, 0., key)
                # Turning a pending row on requires the old geometry as well.
                if mode != "vector":
                    with self.assertRaises(ValueError):
                        report(model, positions, positions, [1.], [1.], [0.], [1.])

    def test_monotone_activation_total_survives_cancelling_target_work(self):
        model = fixture([[1., -1.]], compliance=2. ** -40)
        positions = np.array([[100., 0., 0.], [0., 0., 0.]])
        old_target, target = [[0., 0., 0.]], [[50., 0., 0.]]
        new_activation = np.nextafter(1., 0.)
        result = report(model, positions, positions, old_target, target, [.25], [new_activation])
        with localcontext() as context:
            context.prec = 180
            scale = 1 / (2 * decimal(model.compliance))
            before = scale * Decimal('.25') * Decimal(100) ** 2
            fixed = scale * decimal(new_activation) * Decimal(50) ** 2
            expected = float(fixed - before)
            target_work = float(scale * Decimal('.25') * (Decimal(50) ** 2 - Decimal(100) ** 2))
            activation_work = float(scale * (decimal(new_activation) - Decimal('.25')) * Decimal(50) ** 2)
        self.assertLess(expected, 0.)
        self.assertEqual(result["sewingParameterWorkJoules"], expected)
        self.assertEqual(result["sewingTargetParameterWorkJoules"], target_work)
        self.assertEqual(result["sewingActivationParameterWorkJoules"], activation_work)
        self.assertEqual(result["externalParameterWorkJoules"], expected)
        self.assertEqual(result["mechanicalChangeJoules"], expected)
        self.assertNotEqual(math.fsum((target_work, activation_work)), expected)
        self.assertLessEqual(abs(math.fsum((target_work, activation_work)) - expected),
                            result["sewingParameterWorkComponentSumErrorBoundJoules"])
        reverse = report(model, positions, positions, target, old_target, [new_activation], [.25])
        self.assertEqual(math.fsum((result["sewingChangeJoules"], reverse["sewingChangeJoules"])), 0.)

    def test_tiny_normal_motion_matches_decimal_endpoint_potentials(self):
        model = fixture([[-1., 0., 0., 1.]], "normal-offset")
        start = np.array([[100., 0., 0.], [100., 1., 0.], [100., 0., 1.], [0., 0., 0.]])
        for direction in (-1., 1.):
            with self.subTest(direction=direction):
                end = start.copy()
                end[3, 0] = direction * 2. ** -60
                result = report(model, start, end, [100.], [100.], [.7], [.7])
                with localcontext() as context:
                    context.prec = 180
                    old = decimal(start[3, 0]) - decimal(start[0, 0]) - Decimal(100)
                    new = decimal(end[3, 0]) - decimal(end[0, 0]) - Decimal(100)
                    expected = float(decimal(.7) * (new ** 2 - old ** 2) / (2 * decimal(model.compliance)))
                self.assertNotEqual(expected, 0.)
                self.assertEqual(result["sewingBeforeJoules"], result["sewingAfterJoules"])
                self.assertEqual(result["sewingFixedParameterChangeJoules"], expected)
                self.assertEqual(result["sewingChangeJoules"], expected)

    def test_tiny_distance_motion_matches_decimal_endpoint_potentials(self):
        model = fixture([[1., -1.]], "distance")
        start = np.array([[100., 0., 0.], [0., 0., 0.]])
        for direction in (-1., 1.):
            with self.subTest(direction=direction):
                end = start.copy()
                end[1, 0] = direction * 2. ** -60
                result = report(model, start, end, [50.], [50.], [.7], [.7])
                with localcontext() as context:
                    context.prec = 180
                    old = abs(decimal(start[0, 0]) - decimal(start[1, 0])) - Decimal(50)
                    new = abs(decimal(end[0, 0]) - decimal(end[1, 0])) - Decimal(50)
                    expected = float(decimal(.7) * (new ** 2 - old ** 2) / (2 * decimal(model.compliance)))
                self.assertNotEqual(expected, 0.)
                self.assertEqual(result["sewingBeforeJoules"], result["sewingAfterJoules"])
                self.assertEqual(result["sewingFixedParameterChangeJoules"], expected)
                self.assertEqual(result["sewingChangeJoules"], expected)

    def test_release_requires_old_frame_but_skips_final_inactive_frame(self):
        model = fixture([[-1., 0., 0., 1.]], "normal-offset", compliance=.5)
        start = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 2.]])
        end = np.zeros_like(start)
        result = report(model, start, end, [1.], [3.], [1.], [0.])
        self.assertEqual(result["sewingBeforeJoules"], 1.)
        self.assertEqual(result["sewingAfterJoules"], 0.)
        self.assertEqual(result["sewingTargetParameterWorkJoules"], 0.)
        self.assertEqual(result["sewingReleaseEnergyRemovedJoules"], 1.)
        self.assertEqual(result["sewingChangeJoules"], -1.)
        self.assertEqual(result["sewingFixedParameterChangeJoules"], 0.)
        self.assertFalse(result["accepted"])
        # This energy-only fixture is not a geometry acceptance certificate.
        with self.assertRaises(ValueError):
            report(model, end, start, [3.], [1.], [0.], [1.])

    def test_sewing_fold_gripper_target_first_cycle_has_exact_known_work(self):
        model = fixture([[1., -1., 0., 0.]], compliance=.5)
        positions = np.array([[0., 1., 0.], [0., -1., 0.], [0., 0., 0.], [1., 0., 0.]])
        model.fold_actuation = SimpleNamespace(potential=lambda targets: ElasticDihedralBending(
            4, [[0, 1, 2, 3]], targets, [1.], [4.]))
        model.material_grippers = MaterialGrippers(["panel"] * 4, [[2, 3, 0]], [{
            "id": "source-grip", "instanceId": "panel", "triangleIndex": 0,
            "weights": [1., 0., 0.], "stiffnessNPerM": 2.}])
        parameters = dict(previous_fold_targets=[.25], fold_targets=[.5],
            previous_gripper_targets=[[0., 0., 0.]], gripper_targets=[[1., 0., 0.]],
            previous_gripper_activation=[1.], gripper_activation=[0.])
        result = report(model, positions, positions, [[0., .5, 0.]], [[0., 1., 0.]], [.25], [.75], **parameters)
        expected = {"sewingTargetParameterWorkJoules": -.3125,
            "sewingActivationParameterWorkJoules": .5, "sewingParameterWorkJoules": .1875,
            "foldTargetParameterWorkJoules": .375, "gripperTargetParameterWorkJoules": 1.,
            "gripperActivationParameterWorkJoules": -1., "targetParameterWorkJoules": 1.0625,
            "externalParameterWorkJoules": .5625, "mechanicalChangeJoules": .5625,
            "mechanicalChangeMinusTargetWorkJoules": -.5, "mechanicalChangeMinusParameterWorkJoules": 0.}
        for key, value in expected.items():
            self.assertEqual(result[key], value, key)
        reverse_parameters = dict(previous_fold_targets=[.5], fold_targets=[.25],
            previous_gripper_targets=[[1., 0., 0.]], gripper_targets=[[0., 0., 0.]],
            previous_gripper_activation=[0.], gripper_activation=[1.])
        reverse = report(model, positions, positions, [[0., 1., 0.]], [[0., .5, 0.]], [.75], [.25], **reverse_parameters)
        self.assertEqual(math.fsum((result["mechanicalChangeJoules"], reverse["mechanicalChangeJoules"])), 0.)
        self.assertEqual(math.fsum((result["externalParameterWorkJoules"], reverse["externalParameterWorkJoules"])), 0.)


if __name__ == "__main__":
    unittest.main()
