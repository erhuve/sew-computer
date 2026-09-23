import copy
from types import SimpleNamespace
import unittest

import numpy as np

from solver_adaptive_contact import adaptive_contact_step
from solver_assembly_schedule import AssemblySchedule


def staged_recipe():
    return {"profile": "sewing-fold-progress-v1", "knots": [
        {"fraction": 0., "sewingProgress": 0., "foldProgress": 0.},
        {"fraction": .5, "sewingProgress": 1., "foldProgress": 0.},
        {"fraction": 1., "sewingProgress": 1., "foldProgress": 1.}]}


class AssemblyScheduleTests(unittest.TestCase):
    def test_bad_schedules_reject(self):
        original = staged_recipe()
        cases = [None, {}, {**original, "profile": "unknown"}, {**original, "extra": 1}]
        for key, value in (("fraction", .3), ("fraction", 0.), ("fraction", True),
                           ("foldProgress", float("nan")), ("sewingProgress", -1.),
                           ("foldProgress", 1.1)):
            changed = copy.deepcopy(original)
            changed["knots"][1][key] = value
            cases.append(changed)
        for index, key, value in ((0, "foldProgress", .1), (2, "sewingProgress", .9)):
            changed = copy.deepcopy(original)
            changed["knots"][index][key] = value
            cases.append(changed)
        for recipe in cases:
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                AssemblySchedule(recipe, 4)

    def test_sew_then_fold_preserves_state_velocity_and_rejected_interval(self):
        calls = []

        def step(positions, velocities, targets, duration, fold_targets):
            calls.append((positions.copy(), velocities.copy(), targets.copy(), fold_targets.copy(), duration))
            if len(calls) == 3:
                positions[:] = 999
                velocities[:] = 999
                return positions, velocities, {"converged": False, "gradientInfinityNorm": 1.}
            return positions + duration, velocities + duration, {"converged": True, "gradientInfinityNorm": 0.}

        actuator = SimpleNamespace(potential=lambda angles: SimpleNamespace(rest_angles=np.asarray(angles)))
        solver = SimpleNamespace(step=step, fold_actuation=actuator)
        positions, velocities, report = adaptive_contact_step(solver, np.zeros((1, 3)), np.ones((1, 3)),
            np.full((1, 3), 2.), np.zeros((1, 3)), 1., initial_subdivisions=4,
            initial_fold_targets=[0.], fold_targets=[2.], assembly_schedule=staged_recipe())
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["rejectedSteps"]), 1)
        for record, (start, velocity, target, fold, duration) in zip(report["attempts"], calls):
            fraction = record["endFraction"]
            np.testing.assert_array_equal(start, record["startFraction"])
            np.testing.assert_array_equal(velocity, 1 + record["startFraction"])
            np.testing.assert_array_equal(target, max(0., 2 - 4 * fraction))
            np.testing.assert_array_equal(fold, max(0., 4 * fraction - 2))
            self.assertEqual(duration, record["endFraction"] - record["startFraction"])
            self.assertFalse(record["startFraction"] < .5 < record["endFraction"])
        np.testing.assert_array_equal(positions, 1.)
        np.testing.assert_array_equal(velocities, 2.)


if __name__ == "__main__":
    unittest.main()
