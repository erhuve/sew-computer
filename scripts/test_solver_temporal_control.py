"""Exact controller witnesses and orchestration attacks; no garment fixtures."""
import copy
from fractions import Fraction as F
import unittest
from unittest.mock import patch

import numpy as np

from solver_temporal_control import (PROFILE, MOTION_FIELDS, discrepancy, energy_defect, policy,
                                     rational, run_trials, strict_array)


def declaration(**changes):
    return {"profile": PROFILE, "positionToleranceM": .3, "velocityToleranceMPerS": .1,
            "numericalEnergyBudgetJ": 1., "maxEvaluationBudget": 4096, **changes}


def energy(motion=0., radius=F()):
    from solver_energy_balance import _VARYING_BASE_SCALARS
    terms = {key: float(motion) if index == 0 else 0. for index, key in enumerate(MOTION_FIELDS)}
    report = {**dict.fromkeys(_VARYING_BASE_SCALARS, 0.), "accepted": False, **terms, "temporalMotion": {"termsJoules": terms.copy(),
        "knownErrorBoundJoules": rational(radius)}}
    if radius:
        report["continuousCableEnergy"] = {"work": {"certificate": {"changeErrorBoundJoules": rational(radius)}},
            "aggregationTermsJoules": {"mechanicalChangeMinusParameterWorkJoules": terms.copy()}}
    return report


class Trials:
    def __init__(self):
        self.calls = []
        self.reject = set()
        self.error = {}
        self.alias = False
        self.buffer = np.zeros((1, 3))

    def __call__(self, q, v, a, b, h, record):
        number = len(self.calls)+1
        self.calls.append((q.copy(), v.copy(), a, b, h))
        if number in self.error:
            raise self.error[number]
        nv = v.copy()
        nv[0, 0] += h
        nq = q+h*nv
        record["step"] = {"evaluations": 2, "energyBalance": energy(-(nq[0, 0]-q[0, 0]))}
        valid = number not in self.reject
        if self.alias:
            self.buffer[:] = nq
            nq = self.buffer
        return nq, nv, valid, False, None


def execute(evaluator=None, *, declarations=None, **limits):
    return run_trials(Trials() if evaluator is None else evaluator, np.zeros((1, 3)), np.zeros((1, 3)),
                      np.ones(1), 1., declaration=declaration(**(declarations or {})),
                      **{"max_depth": 5, "max_attempts": 128, "initial_subdivisions": 1,
                         "evaluation_limit": 8, **limits})


class TemporalIndicatorTests(unittest.TestCase):
    def test_exact_squared_boundary_prevents_rounded_norm_false_pass(self):
        q = np.array([[1., 2**-28, 0.]])
        result = discrepancy(q, np.zeros_like(q), 1.)
        self.assertEqual(result["maximumApproximate"], 1.)
        self.assertFalse(result["withinThreshold"])
        self.assertEqual(result["maximumSquared"], rational(1+F(1, 2**56)))

    def test_overflow_and_subnormal_do_not_change_exact_decision(self):
        for number, bound, expected in ((1e308, 1e307, False), (2**-1074, 2**-1074, True)):
            q = np.array([[number, 0., 0.]])
            self.assertEqual(discrepancy(q, np.zeros_like(q), bound)["withinThreshold"], expected)
        self.assertFalse(discrepancy(np.array([[2**-1074, 2**-1074, 0.]]), np.zeros((1, 3)), 2**-1074)["withinThreshold"])

    def test_maximum_vertex_not_rms_and_no_alignment(self):
        q = np.zeros((100, 3)); q[99, 2] = 1.
        result = discrepancy(q, np.zeros_like(q), .1)
        self.assertEqual(result["maximumVertex"], 99)
        self.assertFalse(result["withinThreshold"])

    def test_exact_energy_and_uncertainty_boundary(self):
        v, w = np.zeros((1, 3)), np.array([[1., 0., 0.]])
        result = energy_defect(np.ones(1), v, w, energy(-.25, F(1, 8)), F(3, 8))
        self.assertEqual(result["nominalDefectJoules"], rational(F(1, 4)))
        self.assertEqual(result["outcome"], "within-budget")
        for allocation, expected in ((F(1, 4), "uncertainty-overlap"), (F(1, 16), "exceeds-budget")):
            self.assertEqual(energy_defect(np.ones(1), v, w, energy(-.25, F(1, 8)), allocation)["outcome"], expected)

    def test_energy_reduction_retains_tiny_kinetic_change_under_cancellation(self):
        old = np.array([[1., 0., 0.]])
        new = np.array([[1., 2**-500, 0.]])
        result = energy_defect(np.ones(1), old, new, energy(), F(1))
        self.assertEqual(result["nominalDefectJoules"], rational(F(1, 2**1001)))

    def test_policy_rejects_bool_lossy_values_missing_fields_and_unbounded_budget(self):
        cases = [declaration(positionToleranceM=True), declaration(positionToleranceM=2**53+1),
                 declaration(velocityToleranceMPerS=0.), declaration(numericalEnergyBudgetJ=float("inf")),
                 declaration(maxEvaluationBudget=True), declaration(maxEvaluationBudget=40960001)]
        missing = declaration(); del missing["numericalEnergyBudgetJ"]; cases.append(missing)
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ValueError): policy(item)
        with self.assertRaises(ValueError): strict_array([[2**53+1, 1., 0.]])
        with self.assertRaises(ValueError): strict_array([[True, 1., 0.]])

    def test_inconsistent_motion_payload_and_removed_radius_reject(self):
        for attack in ("term", "radius"):
            report = energy(-.125, F(1, 8))
            if attack == "term": report["temporalMotion"]["termsJoules"]["membraneChangeJoules"] = 0.
            else: report["temporalMotion"]["knownErrorBoundJoules"] = rational(F())
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                energy_defect(np.ones(1), np.zeros((1, 3)), np.ones((1, 3)), report, F(10))

    def test_stored_kinetic_cancellation_and_tiny_allocation(self):
        old = np.array([[2.**27, 0., 0.]])
        new = old.copy(); new[0, 0] = np.nextafter(old[0, 0], np.inf)
        result = energy_defect(np.ones(1), old, new, energy(-4.), F(1, 2**52))
        self.assertEqual(result["nominalDefectJoules"], rational(F(1, 2**51)))
        self.assertEqual(result["outcome"], "exceeds-budget")
        result = energy_defect(np.ones(1), old, old, energy(0., F(1, 2**1076)), F(1, 2**1075))
        self.assertEqual(result["outcome"], "within-budget")

    def test_missing_and_nonfinite_public_work_fields_reject(self):
        from solver_energy_balance import _VARYING_BASE_SCALARS
        for key in _VARYING_BASE_SCALARS:
            for invalid in (None, True, float("nan"), float("inf")):
                report = energy()
                if invalid is None: del report[key]
                else: report[key] = invalid
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    energy_defect(np.ones(1), np.zeros((1, 3)), np.zeros((1, 3)), report, F(1))


class TemporalOrchestrationTests(unittest.TestCase):
    def test_constant_acceleration_commits_fine_native_velocity_and_work(self):
        fixture = Trials(); q, v, row = execute(fixture)
        self.assertTrue(row["complete"])
        self.assertFalse(row["accepted"])
        np.testing.assert_array_equal(q, [[.75, 0., 0.]])
        np.testing.assert_array_equal(v, [[1., 0., 0.]])
        self.assertEqual([step["attemptId"] for step in row["acceptedSteps"]], [2, 3])
        self.assertEqual([step["outcome"] for step in row["attempts"]], ["discarded", "committed", "committed"])
        self.assertEqual(row["conditionalAbsoluteEnergyBoundJoules"], rational(F(1, 4)))
        self.assertEqual(row["resources"]["chargedEvaluationAllowance"], 24)

    def test_temporal_rejection_refines_and_recomputes_right_from_new_midpoint(self):
        fixture = Trials(); q, v, row = execute(fixture, declarations={"positionToleranceM": .1})
        self.assertTrue(row["complete"])
        self.assertEqual(len(fixture.calls), 9)
        self.assertEqual(row["temporalAssessments"][0]["outcome"], "temporally-rejected")
        self.assertEqual([step["endFraction"] for step in row["acceptedSteps"]], [.25, .5, .75, 1.])
        self.assertEqual(fixture.calls[6][0][0, 0], .1875)
        self.assertNotEqual(fixture.calls[6][0][0, 0], fixture.calls[2][0][0, 0])
        self.assertEqual(q[0, 0], .625)
        self.assertEqual(v[0, 0], 1.)

    def test_each_numerical_failure_stage_discards_provisional_state(self):
        for stage in (1, 2, 3):
            with self.subTest(stage=stage):
                fixture = Trials(); fixture.reject.add(stage)
                q, _, row = execute(fixture, max_depth=1)
                self.assertEqual(len(fixture.calls), stage)
                self.assertEqual(row["reason"], "temporal-depth-exhausted")
                self.assertFalse(row["acceptedSteps"])
                np.testing.assert_array_equal(q, 0.)
                self.assertEqual(row["temporalAssessments"][0]["outcome"], "numerically-rejected")

    def test_failed_coarse_uses_numerical_subdivision_without_error_estimate(self):
        fixture = Trials(); fixture.reject.add(1)
        _, _, row = execute(fixture)
        self.assertTrue(row["complete"])
        self.assertEqual(len(fixture.calls), 7)
        self.assertNotIn("position", row["temporalAssessments"][0])

    def test_budget_exhaustion_at_every_stage_never_commits_a_half(self):
        for amount in (0, 1, 2):
            for dimension in ("attempt", "evaluation"):
                with self.subTest(amount=amount, dimension=dimension):
                    fixture = Trials()
                    kwargs = {"max_attempts": amount} if dimension == "attempt" else {"declarations": {"maxEvaluationBudget": max(1, amount*8)}}
                    q, _, row = execute(fixture, **kwargs)
                    self.assertEqual(len(fixture.calls), amount)
                    self.assertEqual(row["reason"], dimension+"-budget-exhausted")
                    self.assertFalse(row["acceptedSteps"])
                    np.testing.assert_array_equal(q, 0.)

    def test_exhaustion_retains_prior_complete_pair(self):
        q, _, row = execute(initial_subdivisions=2, max_attempts=5)
        self.assertEqual(row["completedFraction"], .5)
        self.assertEqual(len(row["acceptedSteps"]), 2)
        self.assertEqual(q[0, 0], .1875)
        self.assertEqual(row["resources"]["mechanicalTrials"], 5)

    def test_reused_output_buffer_cannot_replace_retained_first_half(self):
        fixture = Trials(); fixture.alias = True
        q, _, row = execute(fixture)
        self.assertTrue(row["complete"])
        self.assertEqual(q[0, 0], .75)
        self.assertEqual(row["conditionalAbsoluteEnergyBoundJoules"], rational(F(1, 4)))

    def test_per_half_energy_defects_cannot_cancel(self):
        base = Trials()
        def evaluate(*args):
            result = base(*args)
            record = args[-1]
            if record["role"] == "fine-first": record["step"]["energyBalance"] = energy(.875)
            if record["role"] == "fine-second": record["step"]["energyBalance"] = energy(-1.375)
            return result
        _, _, row = execute(evaluate, max_depth=1, declarations={"numericalEnergyBudgetJ": .5})
        self.assertEqual([item["nominalDefectJoules"] for item in row["temporalAssessments"][0]["fineEnergy"]],
                         [rational(F(1)), rational(F(-1))])
        self.assertFalse(row["complete"])

    def test_uncertainty_floor_rejects_without_loosening_policy(self):
        base = Trials()
        def evaluate(*args):
            result = base(*args)
            old = args[-1]["step"]["energyBalance"]
            args[-1]["step"]["energyBalance"] = energy(old["membraneChangeJoules"], F(1))
            return result
        _, _, row = execute(evaluate, max_depth=2)
        self.assertFalse(row["complete"])
        self.assertFalse(row["acceptedSteps"])
        self.assertEqual(row["policy"]["numericalEnergyBudgetJ"], 1.)

    def test_keyboard_interrupt_exposes_only_prior_committed_prefix(self):
        fixture = Trials(); fixture.error[5] = KeyboardInterrupt("synthetic interrupt")
        with self.assertRaises(KeyboardInterrupt) as caught:
            execute(fixture, initial_subdivisions=2)
        q, _, row = caught.exception.temporal_result
        self.assertEqual(row["completedFraction"], .5)
        self.assertEqual(q[0, 0], .1875)
        self.assertEqual(len(row["acceptedSteps"]), 2)
        self.assertEqual(row["resources"]["chargedEvaluationAllowance"], 40)

    def test_fatal_failures_at_every_trial_preserve_prefix_and_charge(self):
        for stage in (4, 5, 6):
            for kind in (TimeoutError, RuntimeError, MemoryError):
                with self.subTest(stage=stage, kind=kind):
                    fixture = Trials(); fixture.error[stage] = kind("synthetic failure")
                    q, _, row = execute(fixture, initial_subdivisions=2)
                    self.assertEqual(row["reason"], "solver-resource-or-runtime-failure")
                    self.assertEqual(row["completedFraction"], .5)
                    self.assertEqual(len(row["acceptedSteps"]), 2)
                    self.assertEqual(q[0, 0], .1875)
                    self.assertEqual(row["resources"]["chargedEvaluationAllowance"], stage*8)

    def test_exception_during_assessment_exposes_previous_pair(self):
        from solver_temporal_control import discrepancy as real
        calls = 0
        def interrupt(*args):
            nonlocal calls
            calls += 1
            if calls == 3: raise KeyboardInterrupt("between trials and commit")
            return real(*args)
        with patch("solver_temporal_control.discrepancy", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt) as caught:
                execute(initial_subdivisions=2)
        q, _, report = caught.exception.temporal_result
        self.assertEqual(report["completedFraction"], .5)
        self.assertEqual(q[0, 0], .1875)
        self.assertEqual(len(report["transactions"]), 1)

    def test_unrepresentable_half_duration_rejects_instead_of_adding_time(self):
        def stationary(q, v, a, b, h, record):
            record["step"] = {"energyBalance": energy()}
            return q, v, True, False, None
        _, _, report = run_trials(stationary, np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1), 3*2**-1074,
            declaration=declaration(), max_depth=1, max_attempts=3, initial_subdivisions=1, evaluation_limit=8)
        self.assertFalse(report["complete"])
        self.assertEqual(report["reason"], "substep-duration-unrepresentable")
        self.assertEqual(report["resources"]["mechanicalTrials"], 1)
        self.assertFalse(report["acceptedSteps"])

    def test_trial_retained_report_alias_cannot_rewrite_first_half(self):
        base = Trials(); old = None
        def evaluate(*args):
            nonlocal old
            result = base(*args)
            if old is not None: old["step"]["energyBalance"]["temporalMotion"]["termsJoules"].clear()
            old = args[-1]
            return result
        _, _, report = execute(evaluate)
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["acceptedSteps"][0]["step"]["energyBalance"]["temporalMotion"]["termsJoules"]), 8)
        self.assertEqual(report["acceptedSteps"][0]["state"]["positionsMeters"], [[.25, 0., 0.]])


if __name__ == "__main__": unittest.main()
