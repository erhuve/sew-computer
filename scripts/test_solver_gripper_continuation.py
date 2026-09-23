"""Portable adaptive accounting tests using an analytic quadratic solver.

These test numerical state/control/journal behavior, not cloth construction.
"""

import copy
import unittest
from unittest import mock

import numpy as np
from scipy.sparse import csr_matrix

from solver_adaptive_contact import adaptive_contact_step
from solver_bending import ElasticDihedralBending
import solver_energy_balance
from solver_material_grippers import MaterialGrippers, SCHEDULE_PROFILE


def control_schedule():
    return {"profile": SCHEDULE_PROFILE, "gripperIds": ["grip"], "knots": [
        {"fraction": fraction, "targetsMeters": [target], "activation": [activation]}
        for fraction, target, activation in (
            (0., [.03, .05, .02], .2), (.25, [.03, .05, .02], 1.),
            (.5, [.06, .05, .03], 1.), (.75, [.06, .05, .03], .5), (1., [.06, .05, .03], 0.))]}


def independent_parameters(fraction):
    # Direct equations, independent of the schedule interpolator.
    if fraction <= .25:
        return np.array([[.03, .05, .02]]), np.array([.2 + 3.2 * fraction])
    if fraction <= .5:
        amount = 4 * fraction - 1
        return np.array([[.03 + .03 * amount, .05, .02 + .01 * amount]]), np.array([1.])
    return np.array([[.06, .05, .03]]), np.array([2 - 2 * fraction])


class QuadraticSolver:
    def __init__(self):
        self.mass, self.active = np.array([.2, .3, .4]), np.ones(3, dtype=bool)
        self.rows, self.stiffness = np.array([[.2, .3, .5]]), 10.
        self.material_grippers = MaterialGrippers(["cloth:shell"] * 3, [[0, 1, 2]], [{
            "id": "grip", "instanceId": "cloth:shell", "triangleIndex": 0,
            "weights": [.2, .3, .5], "stiffnessNPerM": self.stiffness}])
        self.sewing, self.compliance = csr_matrix((0, 3)), .04
        self.poses, self.faces = np.empty((0, 2, 2)), np.empty((0, 3), dtype=int)
        self.areas, self.materials = np.empty(0), np.empty((0, 3))
        self.bending = ElasticDihedralBending(3, np.empty((0, 4), dtype=int), [], [], [])
        self.calls, self.reject_duration_above, self.timeout_call = [], None, None

    def step(self, positions, velocities, targets, duration, *, gripper_targets, gripper_activation, **options):
        self.calls.append((positions.copy(), velocities.copy(), duration,
                           gripper_targets.copy(), gripper_activation.copy()))
        if self.timeout_call == len(self.calls):
            positions[:] = 999
            raise TimeoutError("synthetic bounded solver interruption")
        if self.reject_duration_above is not None and duration > self.reject_duration_above:
            positions[:], velocities[:], gripper_targets[:], gripper_activation[:] = 999, 777, 555, 333
            return positions, velocities, {"converged": False, "gradientInfinityNorm": 1.}
        k = self.stiffness * gripper_activation[0]
        matrix = np.diag(self.mass / duration ** 2) + k * self.rows.T @ self.rows
        rhs = self.mass[:, None] / duration ** 2 * (positions + duration * velocities)
        rhs += k * self.rows.T @ gripper_targets
        following = np.linalg.solve(matrix, rhs)
        residual = matrix @ following - rhs
        return following, (following - positions) / duration, {
            "converged": True, "gradientInfinityNorm": float(np.max(np.abs(residual))),
            "gripperTargetsMeters": gripper_targets.tolist(), "gripperActivation": gripper_activation.tolist()}


class ObservedJournal:
    def __init__(self, testcase, events):
        self.testcase, self.events, self.outcomes = testcase, events, []

    def start(self, record):
        self.events.append(("start", record["attemptId"]))

    def outcome(self, record, positions=None, velocities=None):
        if record["outcome"] == "accepted":
            self.testcase.assertIn("energyBalance", record["step"])
            self.testcase.assertIn("gripperMomentum", record["step"])
            self.testcase.assertIsNotNone(positions)
            self.testcase.assertIsNotNone(velocities)
            self.testcase.assertEqual(self.events[-1][0], "energy")
        else:
            self.testcase.assertIsNone(positions)
            self.testcase.assertIsNone(velocities)
        self.outcomes.append(copy.deepcopy(record))
        self.events.append(("outcome", record["attemptId"]))

    def finish(self, reason, complete):
        self.events.append(("finish", complete))


class GripperContinuationTests(unittest.TestCase):
    def fixture(self):
        return QuadraticSolver(), np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]]), np.zeros((3, 3))

    def run_control(self, solver, positions, velocities, **options):
        return adaptive_contact_step(solver, positions, velocities, np.empty((0, 3)), np.empty((0, 3)), .4,
            initial_subdivisions=4, gripper_schedule=control_schedule(), **options)

    def test_retries_use_original_fractions_and_commit_work_before_journal(self):
        solver, initial, velocity = self.fixture()
        solver.reject_duration_above = .05
        events, captures = [], {}
        journal = ObservedJournal(self, events)
        original_energy = solver_energy_balance.global_energy_transition

        def energy(*args, **kwargs):
            events.append(("energy", None))
            return original_energy(*args, **kwargs)

        def on_accept(positions, velocities, record):
            self.assertEqual(events[-1], ("outcome", record["attemptId"]))
            captures[record["endFraction"]] = (positions.copy(), velocities.copy())
            positions[:], velocities[:] = 666, 444
            record["step"]["energyBalance"]["externalParameterWorkJoules"] = 999
            events.append(("callback", record["attemptId"]))

        unchanged = initial.copy()
        with mock.patch.object(solver_energy_balance, "global_energy_transition", side_effect=energy):
            final, final_velocity, report = self.run_control(solver, initial, velocity,
                attempt_journal=journal, on_accept=on_accept)
        self.assertTrue(report["complete"])
        self.assertFalse(report["accepted"])
        self.assertEqual(len(report["acceptedSteps"]), 8)
        self.assertEqual(len(report["rejectedSteps"]), 4)
        self.assertEqual(sum(event[0] == "energy" for event in events), 8)
        for record, call in zip(report["attempts"], solver.calls):
            before = (initial, velocity) if record["startFraction"] == 0 else captures[record["startFraction"]]
            np.testing.assert_array_equal(call[0], before[0])
            np.testing.assert_array_equal(call[1], before[1])
            target, active = independent_parameters(record["endFraction"])
            np.testing.assert_allclose(call[3], target, rtol=0, atol=1e-16)
            np.testing.assert_allclose(call[4], active, rtol=0, atol=1e-16)
            if record["converged"]:
                self.assertLessEqual(max(abs(value) for value in record["step"]["gripperMomentum"]["residualNs"]),
                                     record["step"]["gripperMomentum"]["toleranceNs"])
                self.assertNotEqual(record["step"]["energyBalance"]["externalParameterWorkJoules"], 999)
            else:
                self.assertNotIn("energyBalance", record["step"])
        np.testing.assert_array_equal(initial, unchanged)
        np.testing.assert_array_equal(velocity, 0.)
        np.testing.assert_array_equal(final, captures[1.][0])
        np.testing.assert_array_equal(final_velocity, captures[1.][1])
        self.assertEqual(report["acceptedSteps"][-1]["step"]["gripperActivation"], [0.])

    def test_energy_failure_or_nonfinite_diagnostic_rejects_before_commit(self):
        original_energy = solver_energy_balance.global_energy_transition
        for mode in ("exception", "nonfinite"):
            solver, initial, velocity = self.fixture()
            events = []
            journal = ObservedJournal(self, events)

            def fail(*args, **kwargs):
                events.append(("energy", None))
                if mode == "exception":
                    raise ValueError("synthetic invalid work")
                result = original_energy(*args, **kwargs)
                result["gripperParameterWorkJoules"] = float("nan")
                return result

            with self.subTest(mode=mode), mock.patch.object(solver_energy_balance, "global_energy_transition", side_effect=fail):
                final, final_velocity, report = self.run_control(solver, initial, velocity, max_depth=0,
                    attempt_journal=journal)
            self.assertFalse(report["complete"])
            self.assertEqual(report["acceptedSteps"], [])
            self.assertEqual(journal.outcomes[0]["outcome"], "rejected")
            self.assertIn("error", journal.outcomes[0])
            np.testing.assert_array_equal(final, initial)
            np.testing.assert_array_equal(final_velocity, velocity)

    def test_false_stationarity_cannot_bypass_momentum_accounting(self):
        solver, initial, velocity = self.fixture()

        def inconsistent(positions, velocities, targets, duration, **kwargs):
            updated = np.full_like(positions, .1)
            return positions + duration * updated, updated, {"converged": True, "gradientInfinityNorm": 0.}

        solver.step = inconsistent
        final, final_velocity, report = self.run_control(solver, initial, velocity, max_depth=0)
        self.assertEqual(report["acceptedSteps"], [])
        self.assertIn("momentum accounting", report["rejectedSteps"][0]["error"]["message"])
        np.testing.assert_array_equal(final, initial)
        np.testing.assert_array_equal(final_velocity, velocity)

    def test_interruption_preserves_last_accepted_state_and_work_only(self):
        solver, initial, velocity = self.fixture()
        solver.timeout_call = 2
        accepted = []
        final, final_velocity, report = self.run_control(solver, initial, velocity,
            on_accept=lambda q, v, record: accepted.append((q, v, record)))
        self.assertFalse(report["complete"])
        self.assertEqual(report["completedFraction"], .25)
        self.assertEqual(report["reason"], "solver-resource-or-runtime-failure")
        self.assertEqual(len(accepted), 1)
        np.testing.assert_array_equal(final, accepted[0][0])
        np.testing.assert_array_equal(final_velocity, accepted[0][1])
        self.assertIn("energyBalance", accepted[0][2]["step"])
        self.assertNotIn("step", report["rejectedSteps"][0])

    def test_recipe_schedule_pair_and_schedule_only_parameters_are_mandatory(self):
        solver, initial, velocity = self.fixture()
        arguments = (solver, initial, velocity, np.empty((0, 3)), np.empty((0, 3)), .4)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            adaptive_contact_step(*arguments, initial_subdivisions=4)
        for name in ("gripper_targets", "gripper_activation"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "captured schedule"):
                self.run_control(solver, initial, velocity, **{name: None})
        with self.assertRaisesRegex(ValueError, "initial interval boundaries"):
            adaptive_contact_step(*arguments, initial_subdivisions=2, gripper_schedule=control_schedule())
        solver.material_grippers = None
        with self.assertRaisesRegex(ValueError, "supplied together"):
            self.run_control(solver, initial, velocity)
        self.assertEqual(solver.calls, [])

    def test_unsupported_schedule_refinement_bound_rejects_before_journal(self):
        solver, initial, velocity = self.fixture()
        events = []
        with self.assertRaises(ValueError):
            adaptive_contact_step(solver, initial, velocity, np.empty((0, 3)), np.empty((0, 3)), .4,
                initial_subdivisions=4096, max_attempts=4096, max_depth=30,
                gripper_schedule=control_schedule(), attempt_journal=ObservedJournal(self, events))
        self.assertEqual(solver.calls, [])
        self.assertEqual(events, [])
        # Exactly 40 dyadic levels remains admitted; stop the first trial with
        # a bounded synthetic timeout rather than traverse the whole grid.
        solver.timeout_call = 1
        _, _, report = adaptive_contact_step(solver, initial, velocity, np.empty((0, 3)), np.empty((0, 3)), .4,
            initial_subdivisions=4096, max_attempts=4096, max_depth=28, gripper_schedule=control_schedule())
        self.assertEqual(len(solver.calls), 1)
        self.assertEqual(report["attempts"][0]["endFraction"], 1 / 4096)
        self.assertEqual(report["reason"], "solver-resource-or-runtime-failure")


if __name__ == "__main__":
    unittest.main()
