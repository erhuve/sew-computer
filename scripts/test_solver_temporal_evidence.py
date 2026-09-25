"""Real controller observations and mutations for independent raw admission."""
import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from solver_temporal_evidence import AuditLimits, EVALUATOR, audit_snapshot, metric, energy_reduction
from solver_temporal_execution import TemporalExecution, TemporalStopRequested
from test_solver_temporal_control import Trials, execute, energy


LIMITS = AuditLimits(32*1024*1024, 64, 2000000, 8192, 10000, 4096, 4096)
RUN = hashlib.sha256(b"independent-admission-controller-test-v1").hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def audit(snapshot, *, expected_context=None):
    # Expectations originate outside the serialized envelope. Mutation tests
    # normally rebind bytes to test semantics rather than just hash rejection.
    expected = (hashlib.sha256(encode(snapshot["context"])).hexdigest() if snapshot["available"] else None)
    envelope = {"profile": "conditional-temporal-process-transport-v1", "accepted": False,
                "runIdentity": RUN, "snapshot": snapshot}
    raw = encode(envelope)+b"\n"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"snapshot.json"
        path.write_bytes(raw)
        return audit_snapshot(path, expected_bytes=len(raw), expected_sha256=hashlib.sha256(raw).hexdigest(),
            expected_run_identity=RUN, expected_context_sha256=expected if expected_context is None else expected_context,
            evaluator_profile=EVALUATOR, limits=LIMITS)


def captured(evaluate=None, **options):
    handle = TemporalExecution()
    execute(evaluate, execution=handle, **options)
    return handle.snapshot()


class RetainedControllerEvidenceTests(unittest.TestCase):
    def test_full_pair_and_nonuniform_refinement_match_original_prefix(self):
        for options, pairs in (({}, 1), ({"initial_subdivisions": 2}, 2),
                               ({"declarations": {"positionToleranceM": .1}}, 2)):
            with self.subTest(options=options):
                snapshot = captured(**options)
                result = audit(snapshot)
                self.assertTrue(result["numericalPrefixComplete"])
                self.assertFalse(result["accepted"])
                self.assertEqual(result["committedTransactions"], pairs)
                self.assertEqual(result["stateSha256"], snapshot["state"]["sha256"])
                self.assertEqual(result["conditionalAbsoluteEnergyBoundJoules"], snapshot["conditionalAbsoluteEnergyBoundJoules"])

    def test_nonfatal_numeric_rejection_replays_failed_parentage(self):
        trials = Trials(); trials.reject = {1, 5}
        snapshot = captured(trials)
        result = audit(snapshot)
        self.assertTrue(result["numericalPrefixComplete"])
        self.assertGreater(len(snapshot["temporalAssessments"]), result["committedTransactions"])

    def test_attempt_and_evaluation_budget_at_each_missing_role(self):
        for maximum in (1, 2, 3, 4, 5):
            for kind in ("attempt", "evaluation"):
                options = {"max_attempts": maximum} if kind == "attempt" else {"declarations": {"maxEvaluationBudget": maximum*8}}
                # A single initial interval supports limits smaller than two.
                with self.subTest(maximum=maximum, kind=kind):
                    snapshot = captured(**options)
                    result = audit(snapshot)
                    self.assertEqual(result["reservedTrials"], min(maximum, 3))
                    self.assertEqual(result["numericalPrefixComplete"], maximum >= 3)

    def test_depth_exhaustion_preserves_zero_prefix(self):
        snapshot = captured(max_depth=1, declarations={"positionToleranceM": .01})
        result = audit(snapshot)
        self.assertFalse(result["numericalPrefixComplete"])
        self.assertEqual(result["committedFineTrials"], 0)

    def test_precancel_and_unavailable_admission_have_distinct_results(self):
        unused = audit(TemporalExecution().snapshot())
        self.assertFalse(unused["available"])
        self.assertNotIn("completedFraction", unused)
        handle = TemporalExecution(); handle.request_stop()
        with self.assertRaises(TemporalStopRequested): execute(execution=handle)
        result = audit(handle.snapshot())
        self.assertTrue(result["available"])
        self.assertEqual(result["completedFraction"], 0.)
        self.assertEqual(result["reservedTrials"], 0)

    def test_before_after_every_retained_boundary(self):
        boundaries = (("_reserve", 4), ("_observe", 7), ("_observe", 8), ("_record_trial", 4),
                      ("_record_assessment", 2), ("_commit", 2), ("_finished", 1), ("_report_ready", 1))
        for name, at in boundaries:
            for after in (False, True):
                with self.subTest(name=name, after=after):
                    handle, calls = TemporalExecution(), 0
                    original = getattr(TemporalExecution, name)
                    def boundary(self, *args):
                        nonlocal calls
                        calls += 1
                        if calls == at and not after:
                            self.request_stop(); raise TemporalStopRequested("before publication")
                        value = original(self, *args)
                        if calls == at:
                            self.request_stop(); raise TemporalStopRequested("after publication")
                        return value
                    with patch.object(TemporalExecution, name, boundary), self.assertRaises(TemporalStopRequested):
                        execute(execution=handle, initial_subdivisions=2)
                    snapshot = handle.snapshot()
                    result = audit(snapshot)
                    self.assertEqual(result["completedFraction"], snapshot["completedFraction"])
                    self.assertEqual(result["reservedTrials"], len(snapshot["reservations"]))
                    self.assertEqual(result["recordedTrials"], len(snapshot["attempts"]))

    def test_invalid_trial_can_retain_non_authoritative_state_and_converged(self):
        handle, trials = TemporalExecution(), Trials()
        def interrupted(*args):
            output = trials(*args)
            if len(trials.calls) == 4:
                args[-1].update(converged=True, state={"arbitrary": "post-return diagnostic"})
                raise KeyboardInterrupt("after diagnostic fields")
            return output
        with self.assertRaises(KeyboardInterrupt): execute(interrupted, execution=handle, initial_subdivisions=2)
        snapshot = handle.snapshot()
        self.assertTrue(snapshot["attempts"][-1]["converged"])
        result = audit(snapshot)
        self.assertEqual(result["completedFraction"], .5)
        self.assertEqual(result["stateSha256"], snapshot["acceptedSteps"][-1]["state"]["sha256"])

    def test_fatal_callback_return_is_not_forced_to_exception_outcome(self):
        trials = Trials()
        def fatal(*args):
            q, v, _, _, _ = trials(*args)
            return q, v, False, True, None
        snapshot = captured(fatal)
        self.assertEqual(snapshot["attempts"][0]["outcome"], "numerically-rejected")
        self.assertEqual(audit(snapshot)["committedFineTrials"], 0)

    def test_reservation_identity_and_charge_mutations_reject(self):
        original = captured(declarations={"positionToleranceM": .1})
        attacks = {
            "parent": lambda s: s["reservations"][3]["trial"].update(parentAttemptId=None),
            "role": lambda s: s["reservations"][1]["trial"].update(role="coarse"),
            "start": lambda s: s["reservations"][2]["trial"].update(startStateSha256=s["state"]["sha256"]),
            "depth": lambda s: s["reservations"][3]["trial"].update(depth=0),
            "interval": lambda s: s["reservations"][0]["trial"].update(initialInterval=1),
            "float": lambda s: s["reservations"][0]["trial"].update(startFraction=0),
            "charge": lambda s: s["resources"].update(chargedEvaluationAllowance=0),
            "marker": lambda s: s["reservations"][0].update(recordPublished=False),
            "observation": lambda s: s["reservations"][0].update(observation="solver-entered"),
        }
        for name, change in attacks.items():
            value = copy.deepcopy(original); change(value)
            with self.subTest(name=name), self.assertRaises(ValueError): audit(value)

    def test_accepted_pairs_states_and_metrics_cannot_be_substituted(self):
        original = captured()
        attacks = {
            "half-pair": lambda s: s["acceptedSteps"].pop(),
            "coarse-state": lambda s: s.update(state=s["attempts"][0]["state"]),
            "coarse-commit": lambda s: s["transactions"][0].update(fineTrialIds=[1, 3]),
            "raw-outcome": lambda s: s["attempts"][1].update(outcome="committed"),
            "false-metric": lambda s: s["temporalAssessments"][0]["position"].update(maximumVertex=True),
            "energy": lambda s: s["temporalAssessments"][0]["fineEnergy"][0]["absoluteUpperBoundJoules"].update(numerator="0"),
            "total": lambda s: s["conditionalAbsoluteEnergyBoundJoules"].update(numerator="0"),
            "bits": lambda s: s["attempts"][0]["state"]["positionsMeters"][0].__setitem__(1, -.0),
            "false-fraction": lambda s: s.update(completedFraction=.5),
        }
        for name, change in attacks.items():
            value = copy.deepcopy(original); change(value)
            with self.subTest(name=name), self.assertRaises(ValueError): audit(value)

    def test_policy_and_mass_changes_need_separate_expected_context(self):
        original = captured()
        expected = hashlib.sha256(encode(original["context"])).hexdigest()
        for change in (lambda s: s["context"]["massKg"].__setitem__(0, 2.),
                       lambda s: s["context"]["policy"].update(positionToleranceM=1.)):
            value = copy.deepcopy(original); change(value)
            with self.assertRaises(ValueError): audit(value, expected_context=expected)

    def test_metric_rounding_never_changes_exact_decision(self):
        self.assertFalse(metric([[1., 2.**-28, 0.]], [[0., 0., 0.]], 1.)["withinThreshold"])
        self.assertTrue(metric([[2.**-1074, 0., 0.]], [[0., 0., 0.]], 2.**-1074)["withinThreshold"])
        self.assertIsNone(metric([[1e308, 0., 0.]], [[-1e308, 0., 0.]], 1.)["maximumApproximate"])

    def test_exact_energy_retains_cancellation_and_public_terms(self):
        old, new = [[1., 0., 0.]], [[1., 2.**-500, 0.]]
        result = energy_reduction([1.], old, new, energy(), F(1), None)
        self.assertEqual(result["nominalDefectJoules"], {"numerator": "1", "denominator": str(2**1001)})
        for key in ("kineticChangeJoules", "externalParameterWorkJoules", "bendingAfterJoules"):
            value = energy(); del value[key]
            with self.subTest(key=key), self.assertRaises(ValueError): energy_reduction([1.], old, new, value, F(1), None)


class RetainedNumericalEvidenceTests(unittest.TestCase):
    def test_real_bounded_contact_with_fixed_and_varying_cable_sums(self):
        from dataclasses import asdict
        from solver_adaptive_contact import adaptive_contact_step
        from solver_contact_work import WorkPolicy
        from solver_contact_work_control import ContactWorkControl
        from test_solver_cable_global import fixture
        from test_solver_cable_varying_global import cloth_fixture
        from test_solver_temporal_control import declaration
        for varying in (False, True):
            with self.subTest(varying=varying):
                _, solver, q, recipe = cloth_fixture() if varying else fixture()
                solver.contact_work_control = ContactWorkControl(solver.contact, asdict(WorkPolicy()))
                options = {}
                if varying:
                    targets, activation = recipe.initial_parameters
                    options["cable_parameter_schedule"] = {
                        "profile": "cable-target-activation-v1", "geometrySha256": recipe.geometry_sha256,
                        "cellIds": list(recipe.cell_ids), "knots": [
                            {"fraction": f, "targetsMeters": targets.tolist(), "activation": activation.tolist()}
                            for f in (0., 1.)]}
                handle = TemporalExecution(); empty = np.empty((0, 3))
                adaptive_contact_step(solver, q, np.zeros_like(q), empty, empty, .0008,
                    temporal_policy=declaration(), temporal_execution=handle, max_depth=2,
                    max_attempts=32, max_evaluations=32, **options)
                snapshot = handle.snapshot()
                result = audit(snapshot)
                self.assertTrue(result["numericalPrefixComplete"], snapshot["controllerReason"])
                changed = copy.deepcopy(snapshot)
                # Forge the same contact radius removal in both public copies:
                # the context still requires this control and rejects it.
                for row in changed["attempts"]+changed["acceptedSteps"]:
                    report = row["step"]["energyBalance"]
                    del report["boundedContactWork"]
                    del report["boundedContactMechanical"]
                with self.assertRaises(ValueError): audit(changed)
                for attack in ("endpoint", "rounding", "sum"):
                    changed = copy.deepcopy(snapshot)
                    report = changed["attempts"][1]["step"]["energyBalance"]
                    if attack == "endpoint": report["boundedContactWork"]["start"]["positionsSha256"] = "0"*64
                    elif attack == "rounding":
                        report["boundedContactMechanical"]["assemblyRoundingBoundsJoules"]["mechanicalChangeJoules"] = {"numerator": "1", "denominator": "1"}
                    else: report["boundedContactMechanical"]["aggregationTermsJoules"]["mechanicalChangeJoules"]["contactChangeJoules"] += 1.
                    with self.subTest(attack=attack), self.assertRaises(ValueError): audit(changed)

    def test_real_oscillator_refined_raw_history(self):
        from test_solver_temporal_global import oscillator, solve
        solver, q = oscillator()
        handle = TemporalExecution()
        solve(solver, q, temporal_execution=handle, thresholds={"positionToleranceM": 1e-5,
              "velocityToleranceMPerS": 1e-5, "numericalEnergyBudgetJ": .001})
        result = audit(handle.snapshot())
        self.assertTrue(result["numericalPrefixComplete"])
        self.assertGreater(result["committedTransactions"], 1)
        self.assertIn("adaptive-response-and-control-interpolation", result["notEvaluated"])

    def test_real_varying_cable_work_and_enrichment_failure(self):
        from solver_adaptive_contact import adaptive_contact_step
        from test_solver_cable_varying_global import particle_fixture
        from test_solver_temporal_control import declaration
        _, solver, q, recipe = particle_fixture(activation=0.)
        raw = {"profile": "cable-target-activation-v1", "geometrySha256": recipe.geometry_sha256,
               "cellIds": list(recipe.cell_ids), "knots": [
            {"fraction": 0., "targetsMeters": [[1., 1.]], "activation": [0.]},
            {"fraction": .5, "targetsMeters": [[1., 1.]], "activation": [1.]},
            {"fraction": 1., "targetsMeters": [[1., 1.]], "activation": [0.]}]}
        handle = TemporalExecution(); empty = np.empty((0, 3))
        with patch("solver_adaptive_contact._varying_cable_totals", side_effect=ValueError("enrichment failed")), self.assertRaises(ValueError):
            adaptive_contact_step(solver, q, np.zeros_like(q), empty, empty, .125,
                temporal_policy=declaration(), temporal_execution=handle, initial_subdivisions=2,
                max_depth=2, max_evaluations=32, cable_parameter_schedule=raw)
        snapshot = handle.snapshot()
        self.assertEqual(snapshot["enrichment"], "failed")
        self.assertTrue(audit(snapshot)["numericalPrefixComplete"])
        # Deleting a certificate together with its radius cannot erase the
        # externally captured work control. Rebind file bytes, retain context.
        changed = copy.deepcopy(snapshot)
        for row in changed["attempts"]+changed["acceptedSteps"]:
            report = row["step"]["energyBalance"]
            del report["varyingCableEnergy"]
            report["temporalMotion"]["knownErrorBoundJoules"] = {"numerator": "0", "denominator": "1"}
        with self.assertRaises(ValueError): audit(changed)


if __name__ == "__main__": unittest.main()
