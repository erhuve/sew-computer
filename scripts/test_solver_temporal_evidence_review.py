"""Independent serialized admission attacks; stdlib fixtures, no producer imports.

Expected records are hand-authored below. In particular the whole-pair example
does not obey a backward-Euler position/velocity law: that is outside the
declared controller-callback profile, while its exact kinetic accounting is
1/32 - 1/32 = 0 on H1 and 0 on H2. No mechanical/path accuracy is asserted.
"""
import copy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import solver_temporal_evidence as evidence


RUN = "a" * 64
EVALUATOR = "controller-callback-binary64-le-v1"
TRANSPORT = "conditional-temporal-process-transport-v1"
RETAINED = "conditional-temporal-retained-run-v1"
AUDIT = "retained-temporal-prefix-audit-v1"

# Explicit independent public field inventory, not imported from the producer.
PUBLIC_SCALARS = (
    "membraneChangeJoules", "bendingChangeJoules", "bendingBeforeJoules", "bendingAfterJoules",
    "foldBarrierChangeJoules", "foldBarrierBeforeJoules", "foldBarrierAfterJoules",
    "contactChangeJoules", "contactBeforeJoules", "contactAfterJoules", "kineticChangeJoules",
    "sewingChangeJoules", "sewingBeforeJoules", "sewingAfterJoules",
    "foldActuationChangeJoules", "foldActuationBeforeJoules", "foldActuationAfterJoules",
    "foldTargetParameterWorkJoules", "gripperBeforeJoules", "gripperAfterJoules",
    "gripperFixedPositionAfterJoules", "gripperFixedParameterChangeJoules", "gripperChangeJoules",
    "gripperParameterWorkJoules", "gripperTargetParameterWorkJoules", "gripperActivationParameterWorkJoules",
    "gripperActivationIncreaseWorkJoules", "gripperReleaseEnergyRemovedJoules",
    "gripperParameterWorkComponentSumErrorBoundJoules", "targetParameterWorkJoules", "externalParameterWorkJoules",
    "mechanicalChangeJoules", "mechanicalChangeMinusTargetWorkJoules", "mechanicalChangeMinusParameterWorkJoules",
)
MOTION = (
    "membraneChangeJoules", "bendingChangeJoules", "foldBarrierChangeJoules",
    "contactChangeJoules", "sewingFixedParameterChangeJoules",
    "foldFixedParameterChangeJoules", "gripperFixedParameterChangeJoules",
    "cableFixedParameterChangeJoules",
)


def serialized(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def context_sha(value):
    return sha(serialized(value)[:-1])


def ratio(numerator=0, denominator=1):
    return {"numerator": str(numerator), "denominator": str(denominator)}


def state(q, v):
    raw = str((len(q), 3)).encode("ascii") + b"\0"
    raw += b"".join(struct.pack("<d", x) for rows in (q, v) for row in rows for x in row)
    return {"sha256": sha(raw), "positionsMeters": copy.deepcopy(q), "velocitiesMPerS": copy.deepcopy(v)}


def unavailable(**changes):
    result = {"profile": RETAINED, "accepted": False, "bound": False,
              "stopRequested": False, "stopReason": None, "available": False,
              "admissionFailure": None}
    result.update(changes)
    return {"profile": TRANSPORT, "accepted": False, "runIdentity": RUN, "snapshot": result}


def empty_available():
    initial = state([[0., 0., 0.]], [[0., 0., 0.]])
    context = {
        "initialState": initial, "massKg": [1.], "requestedDurationSeconds": 1.,
        "policy": {"profile": "conditional-step-doubling-v1", "positionToleranceM": .25,
                   "velocityToleranceMPerS": .25, "numericalEnergyBudgetJ": 1., "maxEvaluationBudget": 3},
        "maxDepth": 1, "maxAttempts": 3, "initialSubdivisions": 1,
        "evaluationLimit": 1, "adaptiveContext": None,
    }
    raw = {"profile": RETAINED, "accepted": False, "bound": True,
           "stopRequested": True, "stopReason": "requested", "available": True,
           "context": context, "phase": "controller-finished", "controllerReason": "interrupted",
           "reportReady": False, "enrichment": "not-requested", "failures": [],
           "complete": False, "completedFraction": 0., "state": copy.deepcopy(initial),
           "acceptedSteps": [], "transactions": [], "conditionalAbsoluteEnergyBoundJoules": ratio(),
           "reservations": [], "attempts": [], "temporalAssessments": [],
           "resources": {"reservedTrials": 0, "recordedTrials": 0, "chargedEvaluationAllowance": 0}}
    return {"profile": TRANSPORT, "accepted": False, "runIdentity": RUN, "snapshot": raw}


def energy(first):
    terms = dict.fromkeys(MOTION, 0.)
    terms["membraneChangeJoules"] = -.03125 if first else 0.
    return {**dict.fromkeys(PUBLIC_SCALARS, 0.), **terms,
            "kineticChangeJoules": .03125 if first else 0., "accepted": False,
            "temporalMotion": {"termsJoules": terms.copy(), "knownErrorBoundJoules": ratio()}}


def complete_pair():
    envelope = empty_available()
    raw = envelope["snapshot"]
    initial = raw["state"]
    half = state([[.125, 0., 0.]], [[.25, 0., 0.]])
    final = state([[.5, 0., 0.]], [[.25, 0., 0.]])
    for ident, parent, left, right, depth, role, start, end in (
        (1, None, 0., 1., 0, "coarse", initial, final),
        (2, 1, 0., .5, 1, "fine-first", initial, half),
        (3, 1, .5, 1., 1, "fine-second", half, final),
    ):
        base = {"attemptId": ident, "parentAttemptId": parent, "initialInterval": 0,
                "startFraction": left, "endFraction": right, "durationSeconds": right-left,
                "depth": depth, "role": role, "converged": False,
                "evaluationAllowanceCharged": 1, "startStateSha256": start["sha256"]}
        raw["reservations"].append({"trial": copy.deepcopy(base), "observation": "return-observed",
                                    "recordPublished": True})
        trial = {**base, "numericallyValid": True, "fatal": False, "outcome": "provisional",
                 "state": copy.deepcopy(end), "step": {"evaluations": 1, "energyBalance": energy(ident != 3)}}
        raw["attempts"].append(trial)
        if ident != 1:
            accepted = copy.deepcopy(trial)
            accepted.update(outcome="committed", transactionId=1, completedDurationSeconds=right)
            raw["acceptedSteps"].append(accepted)
    metric = {"maximumSquared": ratio(), "toleranceSquared": ratio(1, 16),
              "maximumApproximate": 0., "maximumVertex": 0, "withinThreshold": True}
    fine_energy = [{"exactStoredKineticChangeJoules": ratio(1, 32), "nominalDefectJoules": ratio(),
                    "knownErrorBoundJoules": ratio(), "absoluteUpperBoundJoules": ratio(),
                    "allocationJoules": ratio(1, 2), "outcome": "within-budget"},
                   {"exactStoredKineticChangeJoules": ratio(), "nominalDefectJoules": ratio(),
                    "knownErrorBoundJoules": ratio(), "absoluteUpperBoundJoules": ratio(),
                    "allocationJoules": ratio(1, 2), "outcome": "within-budget"}]
    raw["temporalAssessments"] = [{"assessmentId": 1, "startFraction": 0., "endFraction": 1., "depth": 0,
                                   "trialIds": [1, 2, 3], "outcome": "indicators-passed-awaiting-commit",
                                   "position": metric, "velocity": copy.deepcopy(metric), "fineEnergy": fine_energy}]
    raw["transactions"] = [{"transactionId": 1, "fineTrialIds": [2, 3], "startFraction": 0., "endFraction": 1.}]
    raw.update(stopRequested=False, stopReason=None, phase="report-ready", controllerReason="complete",
               reportReady=True, complete=True, completedFraction=1., state=final,
               resources={"reservedTrials": 3, "recordedTrials": 3, "chargedEvaluationAllowance": 3})
    return envelope


class EvidenceReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "snapshot.json"
        self.limits = evidence.AuditLimits(input_bytes=200_000, nesting_depth=32,
            json_values=20_000, string_characters=100_000, vertices=8, trials=16, assessments=16)

    def write(self, value=None, *, data=None):
        raw = serialized(unavailable() if value is None else value) if data is None else data
        self.path.write_bytes(raw)
        return raw

    def audit(self, value=None, *, data=None, **options):
        if value is None and data is None:
            value = unavailable()
        raw = self.write(value, data=data)
        snapshot = value.get("snapshot", {}) if type(value) is dict else {}
        expected_context = context_sha(snapshot["context"]) if "context" in snapshot else None
        args = {"expected_bytes": len(raw), "expected_sha256": sha(raw), "expected_run_identity": RUN,
                "expected_context_sha256": expected_context, "evaluator_profile": EVALUATOR, "limits": self.limits}
        args.update(options)
        return evidence.audit_snapshot(self.path, **args)

    def assert_rejected(self, value=None, *, data=None, **options):
        with self.assertRaises((ValueError, OSError)):
            self.audit(value, data=data, **options)

    def test_unavailable_envelopes_remain_unavailable_and_unaccepted(self):
        variants = (unavailable(), unavailable(bound=True, admissionFailure={"type": "ValueError",
                    "message": "not admitted", "stage": "controller-or-report"}),
                    unavailable(stopRequested=True, stopReason="cpu-soft-limit"))
        for value in variants:
            with self.subTest(snapshot=value["snapshot"]):
                result = self.audit(value)
                self.assertIs(result["accepted"], False)
                self.assertEqual(result["profile"], AUDIT)
                self.assertIs(result["available"], False)
                for invented in ("state", "acceptedSteps", "transactions"):
                    self.assertNotIn(invented, result)

    def test_unavailable_cannot_fabricate_numerical_state_or_context(self):
        for field, value in (("state", {}), ("complete", True), ("completedFraction", 1.),
                             ("resources", {}), ("context", {}), ("acceptedSteps", [])):
            attack = unavailable(); attack["snapshot"][field] = value
            with self.subTest(field=field): self.assert_rejected(attack)
        self.assert_rejected(unavailable(), expected_context_sha256="b" * 64)

    def test_envelope_profiles_fields_and_literal_booleans_are_strict(self):
        for location, field, replacement in (("outer", "accepted", 0), ("outer", "profile", RETAINED),
                ("raw", "accepted", 0), ("raw", "bound", 0), ("raw", "available", 0),
                ("raw", "stopRequested", 0), ("raw", "profile", TRANSPORT),
                ("raw", "available", True), ("raw", "stopReason", "not-requested")):
            value = unavailable()
            target = value if location == "outer" else value["snapshot"]
            target[field] = replacement
            with self.subTest(location=location, field=field): self.assert_rejected(value)
        for location in ("outer", "raw"):
            value = unavailable(); target = value if location == "outer" else value["snapshot"]
            target["unexpected"] = None
            with self.subTest(extra=location): self.assert_rejected(value)

    def test_external_run_artifact_and_evaluator_identities_are_not_self_attested(self):
        for key, invalid in (("expected_sha256", "b"*64), ("expected_sha256", "A"*64),
                ("expected_sha256", 1), ("expected_run_identity", "b"*64),
                ("expected_run_identity", "A"*64), ("expected_run_identity", True),
                ("evaluator_profile", "adaptive-backward-euler"), ("evaluator_profile", None)):
            with self.subTest(key=key, value=invalid): self.assert_rejected(**{key: invalid})
        changed = unavailable(); changed["runIdentity"] = "b"*64
        self.assert_rejected(changed)
        size = len(serialized(unavailable()))
        for invalid in (True, float(size), size-1, size+1):
            with self.subTest(bytes=invalid): self.assert_rejected(expected_bytes=invalid)

    def test_canonical_bytes_include_key_order_spacing_and_exact_one_newline(self):
        value = unavailable(); good = serialized(value)
        variants = (good[:-1], good+b"\n", b" "+good,
                    (json.dumps(value, sort_keys=True)+"\n").encode("ascii"),
                    (json.dumps(value, sort_keys=False, separators=(",", ":"))+"\n").encode("ascii"),
                    good.replace(b'"accepted"', b'"\\u0061ccepted"', 1))
        for raw in variants:
            with self.subTest(bytes=len(raw)): self.assert_rejected(data=raw)

    def test_duplicate_keys_nonfinite_overflow_and_trailing_objects_reject(self):
        raw = serialized(unavailable())
        variants = (raw.replace(b'"accepted":false', b'"accepted":false,"accepted":false', 1),
                    raw.replace(b'"stopReason":null', b'"stopReason":NaN'),
                    raw.replace(b'"stopReason":null', b'"stopReason":Infinity'),
                    raw.replace(b'"stopReason":null', b'"stopReason":-Infinity'),
                    raw.replace(b'"stopReason":null', b'"stopReason":1e9999'),
                    raw+b"{}\n", b"\xef\xbb\xbf"+raw,
                    raw.replace(b'"runIdentity"', '"runIdentit\u00e9"'.encode("utf-8")))
        for value in variants:
            with self.subTest(prefix=value[:40]): self.assert_rejected(data=value)
        for value in (b"null\n", b"[]\n", b"true\n", b"1\n"):
            with self.subTest(root=value): self.assert_rejected(data=value)

    def test_lexical_depth_ignores_escaped_brackets_quotes_and_chunk_boundaries(self):
        message = "["*35000 + '\\"' + "]"*35000
        value = unavailable(bound=True, admissionFailure={"type": "ValueError", "message": message,
                                                          "stage": "controller-or-report"})
        self.assertIs(self.audit(value)["available"], False)
        shorter = unavailable(bound=True, admissionFailure={"type": "ValueError", "message": '[\\"[]\\]é',
                                                            "stage": "controller-or-report"})
        real_read = os.read
        with patch.object(evidence.os, "read", side_effect=lambda fd, count: real_read(fd, min(count, 7))):
            self.assertIs(self.audit(shorter)["available"], False)

    def test_explicit_depth_value_string_and_file_budgets(self):
        self.assert_rejected(limits=replace(self.limits, nesting_depth=1))
        self.assert_rejected(limits=replace(self.limits, json_values=1))
        self.assert_rejected(limits=replace(self.limits, string_characters=63))
        self.assert_rejected(limits=replace(self.limits, input_bytes=len(serialized(unavailable()))-1))
        deep = b'{"x":'+b"["*33+b"0"+b"]"*33+b"}\n"
        self.assert_rejected(data=deep)
        tree = {"a": [1]}
        self.assertEqual(evidence.bounded_tree(tree, replace(self.limits, json_values=4, nesting_depth=2)), 4)
        with self.assertRaises(ValueError):
            evidence.bounded_tree(tree, replace(self.limits, json_values=3))
        with self.assertRaises(ValueError):
            evidence.bounded_tree(1 << 13607, self.limits)

    def test_audit_limits_require_bounded_raw_integers_and_exact_policy_type(self):
        maxima = {"input_bytes": 2**31, "nesting_depth": 128, "json_values": 100_000_000,
                  "string_characters": 2**20, "vertices": 1_000_000, "trials": 4096, "assessments": 4096}
        for field, maximum in maxima.items():
            for bad in (True, 1., 0, -1, maximum+1):
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    replace(self.limits, **{field: bad})
        self.assert_rejected(limits={})
        class Derived(evidence.AuditLimits):
            pass
        self.assert_rejected(limits=Derived(**self.limits.__dict__))

    def test_pure_typed_helpers_preserve_signed_zero_and_canonical_rationals(self):
        for a, b in ((True, 1), (1, 1.), (0., -0.), ({"v": 0.}, {"v": -0.}), ([1], [True])):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError): evidence.same(a, b)
        evidence.same({"v": [-0., 1.]}, {"v": [-0., 1.]})
        self.assertEqual(evidence.fraction(ratio(-1, 2)), Fraction(-1, 2))
        for bad in (ratio(-1, 2), ratio(0, 2), ratio(2, 4), ratio(1, -2), ratio(1, 0),
                    {"numerator": "+1", "denominator": "2"},
                    {"numerator": "-0", "denominator": "1"},
                    {"numerator": "01", "denominator": "2"},
                    {"numerator": 1, "denominator": "2"},
                    {"numerator": "1"*4097, "denominator": "1"}):
            with self.subTest(bad=str(bad)[:70]), self.assertRaises(ValueError):
                evidence.fraction(bad, nonnegative=True)
        for bad in (True, 1, float("nan"), float("inf")):
            with self.subTest(number=bad), self.assertRaises(ValueError): evidence.number(bad)

    def test_read_only_final_regular_file_and_symlink_rejections(self):
        raw = self.write()
        args = {"expected_bytes": len(raw), "expected_sha256": sha(raw), "expected_run_identity": RUN,
                "expected_context_sha256": None, "evaluator_profile": EVALUATOR, "limits": self.limits}
        other = self.path.with_name("snapshot.tmp"); other.write_bytes(raw)
        with self.assertRaises(ValueError): evidence.audit_snapshot(other, **args)
        self.path.unlink(); self.path.symlink_to(other)
        with self.assertRaises((OSError, ValueError)): evidence.audit_snapshot(self.path, **args)
        self.path.unlink(); self.path.mkdir()
        with self.assertRaises((OSError, ValueError)): evidence.audit_snapshot(self.path, **args)
        self.path.rmdir(); self.path.write_bytes(raw)
        linked = self.path.parent / "linked"; linked.symlink_to(self.path.parent, target_is_directory=True)
        with self.assertRaises((OSError, ValueError)): evidence.audit_snapshot(linked/"snapshot.json", **args)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFO")
    def test_fifo_is_rejected_without_blocking_read(self):
        os.mkfifo(self.path)
        with self.assertRaises((OSError, ValueError)):
            evidence.audit_snapshot(self.path, expected_bytes=1, expected_sha256=sha(b"\n"),
                expected_run_identity=RUN, expected_context_sha256=None, evaluator_profile=EVALUATOR, limits=self.limits)

    def test_file_replacement_during_read_is_rejected_even_when_bytes_match(self):
        raw = self.write(); real_read = os.read; replaced = False
        def reading(fd, count):
            nonlocal replaced
            chunk = real_read(fd, count)
            if chunk and not replaced:
                replaced = True
                self.path.unlink(); self.path.write_bytes(raw)
            return chunk
        with patch.object(evidence.os, "read", side_effect=reading), self.assertRaises(ValueError):
            evidence.audit_snapshot(self.path, expected_bytes=len(raw), expected_sha256=sha(raw),
                expected_run_identity=RUN, expected_context_sha256=None, evaluator_profile=EVALUATOR, limits=self.limits)

    def test_independent_complete_pair_admits_only_callback_accounting_scope(self):
        value = complete_pair()
        result = self.audit(value)
        self.assertEqual(result["profile"], AUDIT)
        self.assertIs(result["accepted"], False)
        self.assertIs(result["available"], True)
        self.assertIs(result["numericalPrefixComplete"], True)
        self.assertEqual(result["completedFraction"], 1.)
        self.assertEqual(result["committedTransactions"], 1)
        self.assertEqual(result["committedFineTrials"], 2)
        self.assertEqual(result["stateSha256"], value["snapshot"]["state"]["sha256"])
        self.assertEqual(result["conditionalAbsoluteEnergyBoundJoules"], ratio())
        for deferred in ("primitive-work-enclosures", "adaptive-response-and-control-interpolation",
                         "actual-process-outcome", "contact-and-triangle-paths"):
            self.assertIn(deferred, result["notEvaluated"])
        # These explicit fixture facts ensure no producer-generated oracle slips in.
        self.assertEqual(Fraction(.25)**2/2, Fraction(1, 32))
        self.assertNotEqual((.5-.125)/.5, .25)
        self.assertTrue(all(not record["converged"] for record in value["snapshot"]["attempts"]))

    def test_empty_prefix_and_charged_unrecorded_reservation_are_distinct(self):
        self.assertIs(self.audit(empty_available())["available"], True)
        for marker in ("reserved", "dispatch-authorized", "return-observed"):
            value = empty_available(); raw = value["snapshot"]
            reservation = copy.deepcopy(complete_pair()["snapshot"]["reservations"][0])
            reservation.pop("recordPublished"); reservation["observation"] = marker
            raw["reservations"] = [reservation]
            raw["resources"].update(reservedTrials=1, chargedEvaluationAllowance=1)
            with self.subTest(marker=marker):
                result = self.audit(value)
                self.assertIs(result["accepted"], False)
                self.assertEqual((result["reservedTrials"], result["recordedTrials"],
                                  result["chargedEvaluationAllowance"]), (1, 0, 1))
                self.assertEqual((result["committedTransactions"], result["committedFineTrials"]), (0, 0))
            wrong = copy.deepcopy(value); wrong["snapshot"]["resources"]["chargedEvaluationAllowance"] = 0
            self.assert_rejected(wrong)

    def test_passed_uncommitted_assessment_preserves_empty_prefix(self):
        value = complete_pair(); raw = value["snapshot"]
        raw.update(acceptedSteps=[], transactions=[], state=copy.deepcopy(raw["context"]["initialState"]),
                   complete=False, completedFraction=0., controllerReason="interrupted",
                   stopRequested=True, stopReason="requested")
        result = self.audit(value)
        self.assertIs(result["accepted"], False)
        self.assertIs(result["numericalPrefixComplete"], False)
        self.assertEqual(result["completedFraction"], 0.)
        self.assertEqual(result["committedFineTrials"], 0)
        missing_second = complete_pair(); missing_second["snapshot"]["acceptedSteps"].pop()
        self.assert_rejected(missing_second)

    def test_full_committed_prefix_survives_failed_finish_observation(self):
        value = complete_pair(); raw = value["snapshot"]
        raw.update(controllerReason="interrupted", stopRequested=True, stopReason="requested",
                   enrichment="failed", failures=[{"type": "MemoryError", "message": "diagnostic allocation",
                                                   "stage": "enrichment"}])
        result = self.audit(value)
        self.assertIs(result["accepted"], False)
        self.assertIs(result["numericalPrefixComplete"], True)
        altered = copy.deepcopy(value); altered["snapshot"]["complete"] = False
        self.assert_rejected(altered)

    def test_external_context_required_and_not_recovered_from_snapshot(self):
        value = complete_pair()
        for invalid in (None, "b"*64, "A"*64, True):
            with self.subTest(context=invalid): self.assert_rejected(value, expected_context_sha256=invalid)
        before = context_sha(value["snapshot"]["context"])
        value["snapshot"]["context"]["massKg"] = [2.]
        self.assert_rejected(value, expected_context_sha256=before)

    def test_raw_state_shape_type_hash_and_signed_zero_attacks(self):
        for attack in ("signed-zero", "integer", "shape", "digest"):
            value = empty_available(); raw = value["snapshot"]
            if attack == "signed-zero": raw["state"]["positionsMeters"][0][1] = -0.
            elif attack == "integer": raw["state"]["positionsMeters"][0][1] = 0
            elif attack == "shape": raw["state"]["positionsMeters"][0].pop()
            else: raw["state"]["sha256"] = "0"*64
            with self.subTest(attack=attack): self.assert_rejected(value)
        two = empty_available(); doubled = state([[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]])
        two["snapshot"]["context"].update(initialState=doubled, massKg=[1., 1.])
        two["snapshot"]["state"] = copy.deepcopy(doubled)
        self.assert_rejected(two, limits=replace(self.limits, vertices=1))

    def test_reservation_identity_types_charges_and_roles_are_bound(self):
        for key, bad in (("attemptId", True), ("parentAttemptId", None), ("initialInterval", 1),
                         ("startFraction", .25), ("durationSeconds", 1.), ("depth", 0),
                         ("role", "coarse"), ("evaluationAllowanceCharged", 0), ("startStateSha256", "b"*64)):
            value = complete_pair()
            value["snapshot"]["reservations"][1]["trial"][key] = bad
            with self.subTest(key=key): self.assert_rejected(value)
        self.assert_rejected(complete_pair(), limits=replace(self.limits, trials=2))
        for bad in (False, 1, None):
            value = complete_pair(); value["snapshot"]["reservations"][1]["recordPublished"] = bad
            with self.subTest(published=bad): self.assert_rejected(value)

    def test_valid_records_require_a_successful_return_observation(self):
        for index in range(3):
            for marker in ("reserved", "dispatch-authorized"):
                value = complete_pair()
                value["snapshot"]["reservations"][index]["observation"] = marker
                with self.subTest(trial=index+1, marker=marker): self.assert_rejected(value)

    def test_complete_prefix_cannot_claim_unreached_terminal_resource_gate(self):
        for reason in ("attempt-budget-exhausted", "evaluation-budget-exhausted",
                       "temporal-depth-exhausted", "substep-duration-underflow",
                       "substep-duration-unrepresentable", "solver-resource-or-runtime-failure"):
            value = complete_pair(); value["snapshot"]["controllerReason"] = reason
            with self.subTest(reason=reason): self.assert_rejected(value)
        value = empty_available(); value["snapshot"]["controllerReason"] = "solver-resource-or-runtime-failure"
        self.assert_rejected(value)
        for reason in ("attempt-budget-exhausted", "evaluation-budget-exhausted"):
            value = complete_pair(); raw = value["snapshot"]
            raw.update(acceptedSteps=[], transactions=[], state=copy.deepcopy(raw["context"]["initialState"]),
                       complete=False, completedFraction=0., controllerReason=reason)
            # A passing assessment awaiting commit never requests another trial.
            # Reaching the numerical limit is not evidence of taking its gate.
            with self.subTest(uncommitted_reason=reason): self.assert_rejected(value)

    def test_half_pair_coarse_substitution_and_transaction_reordering_reject(self):
        cases = []
        value = complete_pair(); value["snapshot"]["transactions"][0]["fineTrialIds"] = [2]; cases.append(value)
        value = complete_pair(); value["snapshot"]["transactions"][0]["fineTrialIds"] = [3, 2]; cases.append(value)
        value = complete_pair(); value["snapshot"]["acceptedSteps"][0]["attemptId"] = 1; cases.append(value)
        value = complete_pair(); value["snapshot"]["acceptedSteps"].reverse(); cases.append(value)
        value = complete_pair(); value["snapshot"]["acceptedSteps"][0]["state"]["positionsMeters"][0][0] = .5; cases.append(value)
        for index, item in enumerate(cases):
            with self.subTest(attack=index): self.assert_rejected(item)

    def test_exact_metric_kinetic_allocation_and_total_mutations_reject(self):
        for name, field, replacement in (("position", "maximumSquared", ratio(1, 16)),
                ("velocity", "withinThreshold", False), ("position", "maximumVertex", True),
                ("position", "toleranceSquared", ratio(1, 8)), ("velocity", "maximumApproximate", 1.)):
            value = complete_pair(); value["snapshot"]["temporalAssessments"][0][name][field] = replacement
            with self.subTest(name=name, field=field): self.assert_rejected(value)
        for key, replacement in (("exactStoredKineticChangeJoules", ratio()),
                ("allocationJoules", ratio(1)), ("nominalDefectJoules", ratio(1, 32)),
                ("knownErrorBoundJoules", ratio(1, 32)), ("absoluteUpperBoundJoules", ratio(1, 32))):
            value = complete_pair(); value["snapshot"]["temporalAssessments"][0]["fineEnergy"][0][key] = replacement
            with self.subTest(energy=key): self.assert_rejected(value)
        value = complete_pair(); value["snapshot"]["conditionalAbsoluteEnergyBoundJoules"] = ratio(1, 32)
        self.assert_rejected(value)

    def test_fine_energy_completeness_payload_and_signed_zero_copies_reject(self):
        for field in PUBLIC_SCALARS:
            value = complete_pair()
            for record in (value["snapshot"]["attempts"][1], value["snapshot"]["acceptedSteps"][0]):
                del record["step"]["energyBalance"][field]
            with self.subTest(missing=field): self.assert_rejected(value)
        for field in ("membraneChangeJoules", "cableFixedParameterChangeJoules"):
            value = complete_pair()
            for record in (value["snapshot"]["attempts"][1], value["snapshot"]["acceptedSteps"][0]):
                record["step"]["energyBalance"]["temporalMotion"]["termsJoules"][field] = 0.
                if field == "cableFixedParameterChangeJoules":
                    record["step"]["energyBalance"][field] = -0.
            with self.subTest(mismatch=field): self.assert_rejected(value)


if __name__ == "__main__":
    unittest.main()
