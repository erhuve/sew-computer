import unittest

import numpy as np

from solver_adaptive_contact import adaptive_contact_step


class AdaptiveContactTests(unittest.TestCase):
    def run_fixture(self, step, **options):
        class Solver:
            pass

        solver = Solver()
        solver.step = step
        positions = np.zeros((1, 3))
        velocities = np.ones((1, 3))
        initial_targets = np.zeros((1, 3))
        targets = np.full((1, 3), 8.)
        result = adaptive_contact_step(solver, positions, velocities, initial_targets, targets, 1., **options)
        np.testing.assert_array_equal(positions, 0.)
        np.testing.assert_array_equal(velocities, 1.)
        np.testing.assert_array_equal(initial_targets, 0.)
        np.testing.assert_array_equal(targets, 8.)
        return result

    def test_rejected_state_rolls_back_and_linear_ramp_preserves_time(self):
        calls = []

        def step(positions, velocities, targets, duration, **options):
            calls.append((positions.copy(), velocities.copy(), targets.copy(), duration, options))
            if duration > .25:
                positions[:] = 999.
                velocities[:] = 777.
                return positions, velocities, {"converged": False, "gradientInfinityNorm": 3.}
            return positions + duration, velocities + duration, {"converged": True, "gradientInfinityNorm": 1e-7}

        positions, velocities, report = self.run_fixture(step, max_evaluations=123)
        self.assertTrue(report["complete"])
        self.assertFalse(report["accepted"])
        self.assertEqual(report["completedDurationSeconds"], 1.)
        self.assertEqual(len(report["rejectedSteps"]), 3)
        self.assertEqual(sum(entry["durationSeconds"] for entry in report["acceptedSteps"]), 1.)
        np.testing.assert_array_equal(positions, 1.)
        np.testing.assert_array_equal(velocities, 2.)
        for entry, call in zip(report["attempts"], calls):
            np.testing.assert_array_equal(call[0], entry["startFraction"])
            np.testing.assert_array_equal(call[1], 1 + entry["startFraction"])
            np.testing.assert_array_equal(call[2], 8 * entry["endFraction"])
            self.assertEqual(call[4], {"max_evaluations": 123})

    def test_attempt_budget_returns_last_valid_state_and_duration(self):
        def step(positions, velocities, targets, duration):
            valid = float(targets[0, 0]) <= 4.
            return positions + duration, velocities, {"converged": valid, "gradientInfinityNorm": 0. if valid else 1.}

        positions, _, report = self.run_fixture(step, initial_subdivisions=2, max_attempts=2)
        self.assertFalse(report["complete"])
        self.assertEqual(report["reason"], "attempt-budget-exhausted")
        self.assertEqual(report["completedDurationSeconds"], .5)
        np.testing.assert_array_equal(positions, .5)

    def test_depth_limit_and_false_convergence_fail_closed(self):
        for residual in (float("nan"), 1e-4, None, True):
            with self.subTest(residual=residual):
                def step(positions, velocities, targets, duration):
                    return positions + 999, velocities, {"converged": True, "gradientInfinityNorm": residual}

                positions, _, report = self.run_fixture(step, max_depth=1)
                self.assertFalse(report["complete"])
                self.assertEqual(report["reason"], "subdivision-depth-exhausted")
                self.assertEqual(len(report["attempts"]), 2)
                np.testing.assert_array_equal(positions, 0.)

    def test_resource_failure_stops_without_committing_mutated_state(self):
        def step(positions, velocities, targets, duration):
            positions[:] = 9
            raise TimeoutError("CPU budget exhausted")

        positions, _, report = self.run_fixture(step)
        self.assertFalse(report["complete"])
        self.assertEqual(report["reason"], "solver-resource-or-runtime-failure")
        self.assertEqual(len(report["attempts"]), 1)
        self.assertEqual(report["rejectedSteps"][0]["error"]["type"], "TimeoutError")
        np.testing.assert_array_equal(positions, 0.)

    def test_numerical_rejection_is_retried_with_smaller_duration(self):
        def step(positions, velocities, targets, duration):
            if duration > .5:
                raise ValueError("Physical path rejected")
            return positions + duration, velocities, {"converged": True, "gradientInfinityNorm": 0.}

        _, _, report = self.run_fixture(step)
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["attempts"]), 3)
        self.assertEqual(report["rejectedSteps"][0]["error"]["type"], "ValueError")

    def test_invalid_subdivision_controls_rejected(self):
        for options in ({"initial_subdivisions": 3}, {"initial_subdivisions": 0},
                        {"initial_subdivisions": 4, "max_attempts": 2}, {"max_depth": -1},
                        {"max_depth": True}, {"max_attempts": 0}, {"max_attempts": 4097}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_fixture(lambda *args: None, **options)

    def test_accepted_callback_receives_isolated_states_and_records(self):
        times = []

        def step(positions, velocities, targets, duration):
            return positions + duration, velocities, {"converged": True, "gradientInfinityNorm": 0.}

        def capture(positions, velocities, record):
            times.append(record["completedDurationSeconds"])
            positions[:] = 100.
            velocities[:] = 200.
            record["step"]["converged"] = False

        positions, velocities, report = self.run_fixture(step, initial_subdivisions=2, on_accept=capture)
        self.assertEqual(times, [.5, 1.])
        np.testing.assert_array_equal(positions, 1.)
        np.testing.assert_array_equal(velocities, 1.)
        self.assertTrue(all(record["step"]["converged"] for record in report["acceptedSteps"]))

    def test_callback_failure_propagates_and_noncallable_rejected(self):
        def step(positions, velocities, targets, duration):
            return positions, velocities, {"converged": True, "gradientInfinityNorm": 0.}

        def capture(*args):
            raise ValueError("Artifact capture failed")

        with self.assertRaisesRegex(ValueError, "Artifact capture failed"):
            self.run_fixture(step, on_accept=capture)
        with self.assertRaisesRegex(ValueError, "callable"):
            self.run_fixture(step, on_accept=7)


if __name__ == "__main__":
    unittest.main()
