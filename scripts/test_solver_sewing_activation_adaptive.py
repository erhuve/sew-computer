"""Independent adaptive control/work checks with an analytic quadratic model.

The synthetic linear fold channel tests schedule and accounting composition;
it is not a geometric hinge or a source-pattern/construction certificate.
"""

import copy
from decimal import Decimal, localcontext
import math
import unittest
from unittest import mock

import numpy as np
from scipy.sparse import csr_matrix

from solver_adaptive_contact import adaptive_contact_step
from solver_bending import ElasticDihedralBending
import solver_energy_balance
from solver_material_grippers import MaterialGrippers, SCHEDULE_PROFILE


ROW_IDS = ["synthetic:row:first", "synthetic:row:second"]


def sewing_schedule():
    return {"profile": "sewing-row-activation-v1", "rowIds": ROW_IDS.copy(), "knots": [
        {"fraction": fraction, "activation": activation}
        for fraction, activation in ((0., [0., 0.]), (.25, [.5, 0.]), (.5, [1., 0.]),
                                     (.75, [1., .5]), (1., [1., 1.]))]}


def sewing_weights(fraction):
    return np.array([min(2 * fraction, 1.), max(2 * fraction - 1., 0.)])


def gripper_schedule():
    return {"profile": SCHEDULE_PROFILE, "gripperIds": ["grip"], "knots": [
        {"fraction": 0., "targetsMeters": [[.02, .03, 0.]], "activation": [1.]},
        {"fraction": 1., "targetsMeters": [[.06, .04, .01]], "activation": [0.]}]}


class LinearFoldPotential:
    """Exactly quadratic internal actuator used only to test control plumbing."""
    def __init__(self, targets):
        self.rest_angles = np.asarray(targets).copy()
        self.weights = np.array([2.])

    def angles(self, positions):
        return np.array([positions[1, 0] - positions[2, 0]])

    def energy(self, positions):
        return float(np.sum(self.weights * (self.angles(positions) - self.rest_angles) ** 2) / 2)

    def energy_change(self, start, end):
        error = self.angles(start) - self.rest_angles
        delta = self.angles(end - start)
        return float(np.sum(self.weights * (error + .5 * delta) * delta))


class QuadraticSewingSolver:
    def __init__(self, *, coupled=False):
        self.mass = np.array([.2, .3, .4, .5])
        self.active = np.ones(4, dtype=bool)
        self.sewing = csr_matrix([[1., -1., 0., 0.], [0., 0., 1., -1.]])
        self.compliance, self.sewing_mode = .04, "vector"
        self.poses, self.faces = np.empty((0, 2, 2)), np.empty((0, 3), dtype=int)
        self.areas, self.materials = np.empty(0), np.empty((0, 3))
        self.bending = ElasticDihedralBending(4, np.empty((0, 4), dtype=int), [], [], [])
        self.calls, self.reject_duration_above, self.interrupt_call = [], None, None
        self.interruption = TimeoutError
        self.coupled = coupled
        if coupled:
            self.material_grippers = MaterialGrippers(["synthetic"] * 4, [[0, 1, 2]], [{
                "id": "grip", "instanceId": "synthetic", "triangleIndex": 0,
                "weights": [.25, .25, .5], "stiffnessNPerM": 4.}])
            self.fold_actuation = type("LinearFoldRecipe", (), {"potential": staticmethod(LinearFoldPotential)})()

    def step(self, positions, velocities, targets, duration, *, sewing_activation, **options):
        self.calls.append({"q": positions.copy(), "v": velocities.copy(), "targets": targets.copy(),
            "duration": duration, "activation": sewing_activation.copy(), "options": copy.deepcopy(options)})
        if self.interrupt_call == len(self.calls):
            positions[:], velocities[:] = 999., 777.
            raise self.interruption("synthetic interrupted trial")
        if self.reject_duration_above is not None and duration > self.reject_duration_above:
            positions[:], velocities[:], targets[:], sewing_activation[:] = 999., 777., 555., 333.
            return positions, velocities, {"converged": False, "gradientInfinityNorm": 1.}
        c = self.sewing.toarray()
        stiffness = np.kron(c.T @ np.diag(sewing_activation) @ c / self.compliance, np.eye(3))
        rhs = (c.T @ (sewing_activation[:, None] * targets) / self.compliance).ravel()
        if self.coupled:
            grip = np.kron([[.25, .25, .5, 0.]], np.eye(3))
            grip_stiffness = 4 * options["gripper_activation"][0]
            stiffness += grip_stiffness * grip.T @ grip
            rhs += grip_stiffness * grip.T @ options["gripper_targets"].ravel()
            fold = np.zeros(12)
            fold[3], fold[6] = 1., -1.
            stiffness += 2 * np.outer(fold, fold)
            rhs += 2 * options["fold_targets"][0] * fold
        inertia = np.repeat(self.mass / duration ** 2, 3)
        matrix = np.diag(inertia) + stiffness
        rhs += inertia * (positions + duration * velocities).ravel()
        following = np.linalg.solve(matrix, rhs).reshape((-1, 3))
        error = np.max(np.abs(c @ following - targets), axis=1)
        positive = sewing_activation > 0
        self.calls[-1]["following"] = following.copy()
        diagnostic = {"converged": True, "gradientInfinityNorm": float(np.max(np.abs(matrix @ following.ravel() - rhs))),
            "sewingActivationExplicit": True, "sewingActivation": sewing_activation.tolist(),
            "activeSewingRows": np.flatnonzero(positive).tolist(),
            "pendingSewingRows": np.flatnonzero(~positive).tolist(),
            "sewingRowTargetErrorsM": [float(value) if active else None for value, active in zip(error, positive)],
            "sewingTargetErrorM": float(np.max(error[positive], initial=0)),
            "sewingTargetErrorMetric": "Unweighted maximum absolute Cartesian component per active vector/normal row; absolute scalar-distance error in distance mode; pending rows excluded"}
        return following, (following - positions) / duration, diagnostic


class Journal:
    def __init__(self, testcase, events):
        self.testcase, self.events, self.outcomes = testcase, events, []

    def start(self, record):
        self.events.append(("start", record["attemptId"]))

    def outcome(self, record, positions=None, velocities=None):
        if record["outcome"] == "accepted":
            self.testcase.assertIn("sewingParameterWorkJoules", record["step"]["energyBalance"])
            self.testcase.assertEqual(self.events[-1][0], "energy")
            self.testcase.assertIsNotNone(positions)
            self.testcase.assertIsNotNone(velocities)
        else:
            self.testcase.assertIsNone(positions)
            self.testcase.assertIsNone(velocities)
        self.outcomes.append(copy.deepcopy(record))
        self.events.append(("outcome", record["attemptId"]))

    def finish(self, reason, complete):
        self.events.append(("finish", complete))


class SewingActivationAdaptiveTests(unittest.TestCase):
    def fixture(self, coupled=False):
        solver = QuadraticSewingSolver(coupled=coupled)
        q = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [.1, .1, 0.]])
        return solver, q, np.zeros_like(q)

    def run_schedule(self, solver, q, v, **options):
        return adaptive_contact_step(solver, q, v, [[.02, 0., 0.], [.03, .04, 0.]],
            [[.01, .02, 0.], [.02, .01, .01]], .4, initial_subdivisions=4,
            sewing_schedule=sewing_schedule(), sewing_row_ids=ROW_IDS, **options)

    @staticmethod
    def decimal_work(q, before_targets, targets, old_activation, activation, compliance):
        with localcontext() as context:
            context.prec = 160
            d = lambda value: Decimal.from_float(float(value))
            old_energy, new_energy, target_work = Decimal(0), Decimal(0), Decimal(0)
            for row, (first, second) in enumerate(((0, 1), (2, 3))):
                before = sum(((d(q[first, axis]) - d(q[second, axis]) - d(before_targets[row, axis])) ** 2
                              for axis in range(3)), Decimal(0)) / (2 * d(compliance))
                fixed = sum(((d(q[first, axis]) - d(q[second, axis]) - d(targets[row, axis])) ** 2
                             for axis in range(3)), Decimal(0)) / (2 * d(compliance))
                old_energy += d(old_activation[row]) * before
                new_energy += d(activation[row]) * fixed
                target_work += d(old_activation[row]) * (fixed - before)
            return float(new_energy - old_energy), float(target_work)

    def test_retries_keep_original_fraction_controls_and_accepted_only_work(self):
        solver, q, v = self.fixture()
        solver.reject_duration_above = .05
        events, states = [], {0.: (q.copy(), v.copy())}
        journal = Journal(self, events)
        original = solver_energy_balance.global_energy_transition

        def energy(*args, **kwargs):
            events.append(("energy", None))
            return original(*args, **kwargs)

        def accepted(positions, velocities, record):
            self.assertEqual(events[-1], ("outcome", record["attemptId"]))
            states[record["endFraction"]] = (positions.copy(), velocities.copy())
            positions[:], velocities[:] = 111., 222.
            record["step"]["energyBalance"]["sewingParameterWorkJoules"] = 333.

        with mock.patch.object(solver_energy_balance, "global_energy_transition", side_effect=energy):
            final, final_velocity, result = self.run_schedule(solver, q, v,
                attempt_journal=journal, on_accept=accepted)
        self.assertTrue(result["complete"])
        self.assertIn("sewingActivationInterpolation", result)
        self.assertEqual((len(result["acceptedSteps"]), len(result["rejectedSteps"])), (8, 4))
        self.assertEqual(sum(event[0] == "energy" for event in events), 8)
        initial_targets = np.array([[.02, 0., 0.], [.03, .04, 0.]])
        final_targets = np.array([[.01, .02, 0.], [.02, .01, .01]])
        for record, call in zip(result["attempts"], solver.calls):
            start, end = record["startFraction"], record["endFraction"]
            np.testing.assert_array_equal(call["q"], states[start][0])
            np.testing.assert_array_equal(call["v"], states[start][1])
            np.testing.assert_array_equal(call["activation"], sewing_weights(end))
            expected_targets = final_targets if end == 1 else initial_targets + end * (final_targets - initial_targets)
            np.testing.assert_array_equal(call["targets"], expected_targets)
            if record["outcome"] != "accepted":
                self.assertNotIn("energyBalance", record["step"])
                continue
            diagnostic, a = record["step"], sewing_weights(end)
            self.assertNotIn("gripperMomentum", diagnostic)
            self.assertEqual(diagnostic["activeSewingRows"], np.flatnonzero(a > 0).tolist())
            self.assertEqual(diagnostic["pendingSewingRows"], np.flatnonzero(a == 0).tolist())
            errors = np.max(np.abs(solver.sewing @ states[end][0] - expected_targets), axis=1)
            self.assertEqual(diagnostic["sewingRowTargetErrorsM"], [float(e) if w else None for e, w in zip(errors, a)])
            self.assertEqual(diagnostic["sewingTargetErrorM"], float(np.max(errors[a > 0], initial=0)))
            old_targets = initial_targets + start * (final_targets - initial_targets)
            total, target = self.decimal_work(states[start][0], old_targets, expected_targets,
                                            sewing_weights(start), a, solver.compliance)
            self.assertEqual(diagnostic["energyBalance"]["sewingParameterWorkJoules"], total)
            self.assertEqual(diagnostic["energyBalance"]["sewingTargetParameterWorkJoules"], target)
        np.testing.assert_array_equal(final, states[1.][0])
        np.testing.assert_array_equal(final_velocity, states[1.][1])
        final_energy = np.sum(solver.mass[:, None] * final_velocity ** 2) / 2
        final_energy += np.sum((solver.sewing @ final - final_targets) ** 2) / (2 * solver.compliance)
        summed = math.fsum(row["step"]["energyBalance"]["mechanicalChangeJoules"] for row in result["acceptedSteps"])
        self.assertAlmostEqual(summed, final_energy, delta=1e-13)

    def test_energy_error_or_nonfinite_work_rejects_before_journal_acceptance(self):
        original = solver_energy_balance.global_energy_transition
        for failure in ("raise", "nan"):
            solver, q, v = self.fixture()
            events = []
            journal = Journal(self, events)

            def energy(*args, **kwargs):
                events.append(("energy", None))
                if failure == "raise":
                    raise ValueError("synthetic sewing work failure")
                result = original(*args, **kwargs)
                result["sewingActivationParameterWorkJoules"] = float("nan")
                return result

            with self.subTest(failure=failure), mock.patch.object(solver_energy_balance, "global_energy_transition", side_effect=energy):
                final, velocity, result = self.run_schedule(solver, q, v, max_depth=0, attempt_journal=journal)
            self.assertEqual(result["acceptedSteps"], [])
            self.assertEqual(journal.outcomes[0]["outcome"], "rejected")
            np.testing.assert_array_equal(final, q)
            np.testing.assert_array_equal(velocity, v)

    def test_interruption_retains_only_accepted_prefix(self):
        for exception in (TimeoutError, KeyboardInterrupt):
            solver, q, v = self.fixture()
            solver.interrupt_call, solver.interruption = 2, exception
            events, accepted = [], []
            journal = Journal(self, events)
            original = solver_energy_balance.global_energy_transition

            def energy(*args, **kwargs):
                events.append(("energy", None))
                return original(*args, **kwargs)

            with self.subTest(exception=exception.__name__), mock.patch.object(solver_energy_balance, "global_energy_transition", side_effect=energy):
                options = dict(attempt_journal=journal, on_accept=lambda q, v, r: accepted.append((q, v, r)))
                if exception is KeyboardInterrupt:
                    with self.assertRaises(KeyboardInterrupt):
                        self.run_schedule(solver, q, v, **options)
                else:
                    final, velocity, result = self.run_schedule(solver, q, v, **options)
                    self.assertEqual(result["completedFraction"], .25)
                    self.assertEqual(result["reason"], "solver-resource-or-runtime-failure")
                    np.testing.assert_array_equal(final, accepted[0][0])
                    np.testing.assert_array_equal(velocity, accepted[0][1])
            self.assertEqual(len(accepted), 1)
            self.assertEqual(sum(event[0] == "energy" for event in events), 1)
            self.assertNotIn("step", journal.outcomes[-1])
            self.assertEqual(journal.outcomes[-1]["outcome"], "interrupted" if exception is KeyboardInterrupt else "rejected")

    def test_combined_grippers_fold_and_sewing_controls_are_composed_once(self):
        solver, q, v = self.fixture(coupled=True)
        assembly = {"profile": "sewing-fold-progress-v1", "knots": [
            {"fraction": 0., "sewingProgress": 0., "foldProgress": 0.},
            {"fraction": .5, "sewingProgress": 1., "foldProgress": 1.},
            {"fraction": 1., "sewingProgress": 1., "foldProgress": 1.}]}
        original = solver_energy_balance.global_energy_transition
        with mock.patch.object(solver_energy_balance, "global_energy_transition", wraps=original) as energy:
            _, _, result = self.run_schedule(solver, q, v, gripper_schedule=gripper_schedule(),
                initial_fold_targets=[.1], fold_targets=[-.2], assembly_schedule=assembly)
        self.assertTrue(result["complete"])
        self.assertEqual(energy.call_count, 4)
        for record, call, energy_call in zip(result["acceptedSteps"], solver.calls, energy.call_args_list):
            end, start = record["endFraction"], record["startFraction"]
            progress, old_progress = min(2 * end, 1.), min(2 * start, 1.)
            np.testing.assert_array_equal(call["activation"], sewing_weights(end))
            np.testing.assert_allclose(call["options"]["fold_targets"], [.1 + progress * (-.2 - .1)], atol=1e-16)
            np.testing.assert_allclose(call["options"]["gripper_activation"], [1 - end], atol=1e-16)
            kwargs = energy_call.kwargs
            np.testing.assert_array_equal(kwargs["previous_sewing_activation"], sewing_weights(start))
            np.testing.assert_array_equal(kwargs["sewing_activation"], sewing_weights(end))
            np.testing.assert_allclose(kwargs["previous_fold_targets"], [.1 + old_progress * (-.2 - .1)], atol=1e-16)
            self.assertIn("gripperMomentum", record["step"])
            balance = record["step"]["energyBalance"]
            self.assertEqual(balance["externalParameterWorkJoules"], math.fsum(balance[key] for key in
                ("sewingParameterWorkJoules", "foldTargetParameterWorkJoules", "gripperParameterWorkJoules")))

    def test_pair_count_identity_raw_override_and_refinement_admission(self):
        solver, q, v = self.fixture()
        arguments = (solver, q, v, [[.01, 0., 0.]] * 2, [[.02, 0., 0.]] * 2, .4)
        base = dict(initial_subdivisions=4, sewing_schedule=sewing_schedule(), sewing_row_ids=ROW_IDS)
        variants = [dict(base, sewing_schedule=None), dict(base, sewing_row_ids=None),
            dict(base, sewing_row_ids=ROW_IDS[::-1]), dict(base, sewing_row_ids=ROW_IDS[:1]),
            dict(base, sewing_activation=[1., 1.]),
            dict(base, initial_subdivisions=4096, max_attempts=4096, max_depth=30)]
        one = {"profile": "sewing-row-activation-v1", "rowIds": ROW_IDS[:1], "knots": [
            {"fraction": 0., "activation": [0.]}, {"fraction": 1., "activation": [1.]}]}
        variants.append(dict(base, sewing_schedule=one, sewing_row_ids=ROW_IDS[:1]))
        for options in variants:
            events = []
            with self.subTest(options=options), self.assertRaises(ValueError):
                adaptive_contact_step(*arguments, attempt_journal=Journal(self, events), **options)
            self.assertEqual(events, [])
        self.assertEqual(solver.calls, [])
        solver.interrupt_call = 1
        _, _, result = adaptive_contact_step(*arguments, sewing_schedule=sewing_schedule(), sewing_row_ids=ROW_IDS,
            initial_subdivisions=4096, max_attempts=4096, max_depth=28)
        self.assertEqual(len(solver.calls), 1)
        self.assertEqual(result["attempts"][0]["endFraction"], 1 / 4096)

    def test_step_cannot_mutate_controls_or_forge_activation_diagnostics(self):
        for attack in ("passed-activation", "passed-targets", "reported-activation", "boolean-activation",
                       "explicit-flag", "missing-pending", "boolean-index", "duplicate-active"):
            solver, q, v = self.fixture()
            original = solver.step

            def corrupt(positions, velocities, targets, duration, **options):
                following, velocity, diagnostic = original(positions, velocities, targets, duration, **options)
                if attack == "passed-activation":
                    options["sewing_activation"][:] = 1.
                elif attack == "passed-targets":
                    targets[:] = 0.
                elif attack == "reported-activation":
                    diagnostic["sewingActivation"][0] = np.nextafter(diagnostic["sewingActivation"][0], 1.)
                elif attack == "boolean-activation":
                    diagnostic["sewingActivation"][1] = False
                elif attack == "explicit-flag":
                    diagnostic["sewingActivationExplicit"] = 1
                elif attack == "missing-pending":
                    del diagnostic["pendingSewingRows"]
                elif attack == "boolean-index":
                    diagnostic["activeSewingRows"] = [False]
                else:
                    diagnostic["activeSewingRows"] = [0, 0]
                return following, velocity, diagnostic

            solver.step = corrupt
            with self.subTest(attack=attack), mock.patch.object(solver_energy_balance, "global_energy_transition",
                    wraps=solver_energy_balance.global_energy_transition) as energy:
                final, velocity, result = self.run_schedule(solver, q, v, max_depth=0)
                self.assertEqual(len(result["acceptedSteps"]), 0, attack)
                self.assertIn("error", result["rejectedSteps"][0], attack)
                energy.assert_not_called()
                np.testing.assert_array_equal(final, q)
                np.testing.assert_array_equal(velocity, v)


if __name__ == "__main__":
    unittest.main()
