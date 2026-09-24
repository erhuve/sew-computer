"""Interruption witnesses for opt-in temporal retention; synthetic mechanics."""
import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import solver_temporal_control as controller
from solver_temporal_execution import (PROFILE, TemporalExecution, TemporalStopRequested,
                                       context_value, enrich_result)
from test_solver_temporal_control import Trials, execute


class TemporalExecutionTests(unittest.TestCase):
    def assert_prefix(self, handle, *, fraction=.5, reservations=4, attempts=None):
        snapshot = handle.snapshot()
        self.assertEqual(snapshot["profile"], PROFILE)
        self.assertEqual(snapshot["completedFraction"], fraction)
        self.assertEqual(snapshot["complete"], fraction == 1.)
        self.assertEqual(snapshot["resources"]["reservedTrials"], reservations)
        self.assertEqual(snapshot["resources"]["chargedEvaluationAllowance"], reservations*8)
        self.assertEqual([row["trial"]["attemptId"] for row in snapshot["reservations"]],
                         list(range(1, reservations+1)))
        if attempts is not None:
            self.assertEqual(len(snapshot["attempts"]), attempts)
        self.assertEqual(len(snapshot["acceptedSteps"]), 0 if fraction == 0. else 2 if fraction == .5 else 4)
        ids = {row["attemptId"] for row in snapshot["attempts"]}
        for transaction in snapshot["transactions"]:
            self.assertEqual(len(transaction["fineTrialIds"]), 2)
            self.assertTrue(set(transaction["fineTrialIds"]) <= ids)
        return snapshot

    def test_exact_type_single_use_and_reinitialization(self):
        class Subclass(TemporalExecution): pass
        for invalid in (False, object(), Subclass()):
            with self.subTest(type=type(invalid)), self.assertRaises(ValueError):
                execute(execution=invalid)
        handle = TemporalExecution("cpu-soft-limit")
        with self.assertRaises(ValueError): handle.__init__("other")
        execute(execution=handle)
        with self.assertRaises(ValueError): execute(execution=handle)
        handle.request_stop(); handle.request_stop()
        self.assertEqual(handle.stop_reason, "cpu-soft-limit")

    def test_pre_requested_handle_retains_admission_but_authorizes_no_trials(self):
        handle, trials = TemporalExecution(), Trials()
        handle.request_stop()
        with self.assertRaises(TemporalStopRequested): execute(trials, execution=handle)
        self.assertFalse(trials.calls)
        self.assert_prefix(handle, fraction=0., reservations=0, attempts=0)

    def test_failed_admission_is_unavailable_and_handle_cannot_be_reused(self):
        handle = TemporalExecution()
        with self.assertRaises(ValueError): execute(execution=handle, declarations={"positionToleranceM": True})
        self.assertFalse(handle.snapshot()["available"])
        self.assertEqual(handle.snapshot()["admissionFailure"]["type"], "ValueError")
        with self.assertRaises(ValueError): execute(execution=handle)

    def test_direct_context_is_encoded_or_rejected_before_dispatch(self):
        handle, trials = TemporalExecution(), Trials()
        execute(trials, execution=handle, retained_context={"fraction": Fraction(1, 8)})
        self.assertEqual(handle.snapshot()["context"]["adaptiveContext"],
                         {"fraction": {"numerator": "1", "denominator": "8"}})
        for value in (object(), float("inf")):
            handle, trials = TemporalExecution(), Trials()
            with self.subTest(type=type(value)), self.assertRaises(ValueError):
                execute(trials, execution=handle, retained_context={"unsupported": value})
            self.assertFalse(trials.calls)
            self.assertFalse(handle.snapshot()["available"])

    def test_unused_handle_preserves_mechanics_and_legacy_none_report(self):
        baseline, explicit = execute(initial_subdivisions=2), execute(initial_subdivisions=2, execution=None)
        self.assertEqual(json.dumps(baseline[2]), json.dumps(explicit[2]))
        handle = TemporalExecution()
        retained = execute(initial_subdivisions=2, execution=handle)
        for original, actual in zip(baseline[:2], retained[:2]):
            self.assertEqual(original.tobytes(), actual.tobytes())
        report = copy.deepcopy(retained[2])
        report["profile"] = report.pop("controllerProfile")
        del report["execution"]
        report["resources"]["mechanicalTrials"] = report["resources"].pop("recordedTrials")
        del report["resources"]["reservedTrials"]
        self.assertEqual(report, baseline[2])
        self.assert_prefix(handle, fraction=1., reservations=6, attempts=6)

    def test_latched_stop_survives_normal_return_and_cleanup_replacements_at_each_stage(self):
        for stage in (4, 5, 6):
            for replacement in (None, ValueError, RuntimeError, MemoryError, KeyboardInterrupt):
                with self.subTest(stage=stage, replacement=replacement):
                    handle, trials = TemporalExecution(), Trials()
                    def evaluate(*args):
                        result = trials(*args)
                        if len(trials.calls) == stage:
                            handle.request_stop()
                            if replacement is not None:
                                try: raise TemporalStopRequested("initial stop")
                                finally: raise replacement("cleanup replacement")
                        return result
                    with self.assertRaises(TemporalStopRequested):
                        execute(evaluate, initial_subdivisions=2, execution=handle)
                    self.assertEqual(len(trials.calls), stage)
                    snapshot = self.assert_prefix(handle, reservations=stage, attempts=stage)
                    if replacement:
                        self.assertTrue(any(row["type"] == replacement.__name__ for row in snapshot["failures"]))

    def test_reservation_dispatch_return_and_publication_boundaries_are_distinct(self):
        # A boundary marker is not an exact solver-entry counter. Check real
        # entries independently and retain any charged but unreturned record.
        cases = (("_reserve", False, 3, 3, 3, "return-observed"),
                 ("_reserve", True, 4, 3, 3, "reserved"),
                 ("_observe", False, 4, 3, 4, "reserved"),
                 ("_observe", True, 4, 3, 4, "dispatch-authorized"),
                 ("_record_trial", False, 4, 4, 3, "return-observed"),
                 ("_record_trial", True, 4, 4, 4, "return-observed"))
        for name, after, reservations, entries, attempts, observation in cases:
            with self.subTest(name=name, after=after):
                handle, trials = TemporalExecution(), Trials()
                original = getattr(TemporalExecution, name)
                def boundary(self, *args):
                    target = (len(trials.calls) == (4 if name == "_record_trial" else 3)
                              and (name != "_observe" or args[0] == "dispatch-authorized"))
                    if target and not after:
                        self.request_stop(); raise TemporalStopRequested("before boundary")
                    value = original(self, *args)
                    if target:
                        self.request_stop(); raise TemporalStopRequested("after boundary")
                    return value
                with patch.object(TemporalExecution, name, boundary), self.assertRaises(TemporalStopRequested) as caught:
                    execute(trials, initial_subdivisions=2, execution=handle)
                self.assertEqual(len(trials.calls), entries)
                snapshot = self.assert_prefix(handle, reservations=reservations, attempts=attempts)
                self.assertEqual(snapshot["reservations"][-1]["observation"], observation)
                self.assertEqual(len(caught.exception.temporal_result[2]["attempts"]), attempts)
                self.assertEqual(caught.exception.temporal_result[2]["resources"]["recordedTrials"], attempts)

    def test_returned_callback_with_interrupted_return_marker_remains_truthful(self):
        for after in (False, True):
            handle, trials = TemporalExecution(), Trials()
            original = TemporalExecution._observe
            def boundary(self, observation):
                target = observation == "return-observed" and len(trials.calls) == 4
                if target and not after:
                    self.request_stop(); raise TemporalStopRequested("before return marker")
                original(self, observation)
                if target:
                    self.request_stop(); raise TemporalStopRequested("after return marker")
            with self.subTest(after=after), patch.object(TemporalExecution, "_observe", boundary), \
                 self.assertRaises(TemporalStopRequested):
                execute(trials, initial_subdivisions=2, execution=handle)
            self.assertEqual(len(trials.calls), 4)
            snapshot = self.assert_prefix(handle, reservations=4, attempts=4)
            self.assertEqual(snapshot["reservations"][-1]["observation"],
                             "return-observed" if after else "dispatch-authorized")

    def test_trial_cannot_rewrite_retained_reservation_identity(self):
        for key, bad in (("attemptId", True), ("evaluationAllowanceCharged", 0), ("startFraction", .25)):
            handle, trials = TemporalExecution(), Trials()
            def evaluate(*args):
                result = trials(*args)
                args[-1][key] = bad
                return result
            with self.subTest(key=key), self.assertRaises(ValueError):
                execute(evaluate, execution=handle)
            self.assertEqual(len(trials.calls), 1)
            self.assert_prefix(handle, fraction=0., reservations=1, attempts=0)

    def test_primary_keyboard_interrupt_survives_secondary_trial_publication_failure(self):
        handle, trials = TemporalExecution(), Trials()
        primary = KeyboardInterrupt("primary trial interruption")
        trials.error[4] = primary
        original = TemporalExecution._record_trial
        def publication(self, record):
            if record["attemptId"] == 4: raise MemoryError("secondary publication")
            return original(self, record)
        with patch.object(TemporalExecution, "_record_trial", publication), self.assertRaises(KeyboardInterrupt) as caught:
            execute(trials, initial_subdivisions=2, execution=handle)
        self.assertIs(caught.exception, primary)
        snapshot = self.assert_prefix(handle, reservations=4, attempts=3)
        self.assertTrue(any(row["stage"] == "controller-secondary" and row["type"] == "MemoryError"
                            for row in snapshot["failures"]))

    def test_first_stop_during_discrepancy_or_energy_cannot_commit_or_retry(self):
        for name in ("discrepancy", "energy_defect"):
            handle, trials = TemporalExecution(), Trials()
            original = getattr(controller, name)
            def boundary(*args):
                result = original(*args)
                if len(trials.calls) == 6: handle.request_stop()
                return result
            with self.subTest(name=name), patch.object(controller, name, boundary), self.assertRaises(TemporalStopRequested):
                execute(trials, initial_subdivisions=2, execution=handle)
            self.assertEqual(len(trials.calls), 6)
            self.assert_prefix(handle, reservations=6, attempts=6)

    def test_atomic_commit_retains_whole_old_or_new_pair_including_full_window(self):
        for after in (False, True):
            handle, trials = TemporalExecution(), Trials()
            original = TemporalExecution._commit
            def boundary(self, committed):
                if committed[2] == 1. and not after:
                    self.request_stop(); raise TemporalStopRequested("before commit")
                original(self, committed)
                if committed[2] == 1.:
                    self.request_stop(); raise TemporalStopRequested("after commit")
            with self.subTest(after=after), patch.object(TemporalExecution, "_commit", boundary):
                with self.assertRaises(TemporalStopRequested) as caught:
                    execute(trials, initial_subdivisions=2, execution=handle)
            self.assert_prefix(handle, fraction=1. if after else .5, reservations=6, attempts=6)
            self.assertEqual(caught.exception.temporal_result[2]["complete"], after)
            self.assertEqual(caught.exception.temporal_result[2]["reason"], "stop-requested")

    def test_report_failure_preserves_full_prefix_and_primary_interruption(self):
        for stopped in (False, True):
            handle, trials = TemporalExecution(), Trials()
            original_commit = TemporalExecution._commit
            def commit(self, value):
                original_commit(self, value)
                if value[2] == 1. and stopped:
                    self.request_stop(); raise TemporalStopRequested("after final pair")
            with self.subTest(stopped=stopped), patch.object(TemporalExecution, "_commit", commit), \
                 patch.object(TemporalExecution, "_finished", side_effect=MemoryError("report allocation")):
                with self.assertRaises(TemporalStopRequested if stopped else MemoryError):
                    execute(trials, initial_subdivisions=2, execution=handle)
            snapshot = self.assert_prefix(handle, fraction=1., reservations=6, attempts=6)
            self.assertFalse(snapshot["reportReady"])
            self.assertTrue(any(row["stage"] == "report" and row["type"] == "MemoryError" for row in snapshot["failures"]))

    def test_first_stop_during_report_conversion_keeps_full_raw_state(self):
        handle, trials = TemporalExecution(), Trials()
        original = controller.state_record
        def conversion(q, v):
            if len(trials.calls) == 6 and not np.any(q):
                handle.request_stop(); raise TemporalStopRequested("final JSON conversion")
            return original(q, v)
        with patch.object(controller, "state_record", conversion), self.assertRaises(TemporalStopRequested):
            execute(trials, initial_subdivisions=2, execution=handle)
        self.assert_prefix(handle, fraction=1., reservations=6, attempts=6)

    def test_public_report_state_metadata_and_snapshots_cannot_mutate_raw_prefix(self):
        handle = TemporalExecution()
        q, v, report = execute(initial_subdivisions=2, execution=handle)
        expected = handle.snapshot()
        q.shape = (3, 1); q[:] = -123.
        v.dtype = np.uint8; v[:] = 255
        report["acceptedSteps"][0]["state"]["positionsMeters"][0][0] = -123.
        report["transactions"].clear()
        report["attempts"][0]["step"].clear()
        exposed = handle.snapshot()
        exposed["context"]["initialState"].clear()
        exposed["reservations"][0]["trial"].clear()
        exposed["acceptedSteps"].clear()
        self.assertEqual(handle.snapshot(), expected)

    def test_enrichment_mutates_only_detached_output_and_preserves_primary(self):
        for stopped in (False, True):
            handle = TemporalExecution()
            result = execute(initial_subdivisions=2, execution=handle)
            before = handle.snapshot()
            primary = TemporalStopRequested("primary") if stopped else None
            if stopped: handle.request_stop()
            def enrich(report):
                report["acceptedSteps"][0]["step"].clear()
                report["partiallyEnriched"] = True
                raise ValueError("accepted-work totals")
            with self.subTest(stopped=stopped), self.assertRaises(TemporalStopRequested if stopped else ValueError) as caught:
                enrich_result(handle, result, enrich, primary=primary)
            if stopped: self.assertIs(caught.exception, primary)
            self.assertNotIn("partiallyEnriched", result[2])
            self.assertEqual(handle.snapshot()["acceptedSteps"], before["acceptedSteps"])
            self.assertEqual(handle.snapshot()["enrichment"], "failed")

    def test_context_rejects_unknown_or_lossy_values_and_preserves_signed_zero(self):
        invalids = [object(), float("nan")]
        if np.dtype(np.longdouble).itemsize > 8:
            invalids.append(np.array([1.], dtype=np.longdouble))
        for invalid in invalids:
            with self.subTest(type=type(invalid)), self.assertRaises(ValueError): context_value(invalid)
        original = np.array([[0., -0.]])
        encoded = context_value(original)
        self.assertEqual([value.hex() for value in encoded["values"][0]], ["0x0.0p+0", "-0x0.0p+0"])
        original[:] = 99.
        self.assertEqual(encoded["shape"], [1, 2])
        self.assertEqual(encoded["values"], [[0., -0.]])


@unittest.skipUnless(hasattr(signal, "SIGXCPU") and os.name == "posix", "requires POSIX CPU signals")
class TemporalSignalTests(unittest.TestCase):
    def run_child(self, action):
        source = r'''
import json, os, resource, signal, sys
from solver_temporal_execution import TemporalExecution, TemporalStopRequested
from test_solver_temporal_control import execute, Trials
handle, trials = TemporalExecution("cpu-soft-limit"), Trials()
signals = 0
def stopped(signum, frame):
    global signals
    signals += 1
    already = handle.stop_requested
    handle.request_stop()
    if not already: raise TemporalStopRequested("SIGXCPU")
signal.signal(signal.SIGXCPU, stopped)
def evaluate(*args):
    result = trials(*args)
    if len(trials.calls) == 4:
        if sys.argv[1] == "hard": os.kill(os.getpid(), signal.SIGKILL)
        if sys.argv[1] == "keyboard": raise KeyboardInterrupt("actual child interruption")
        used = resource.getrusage(resource.RUSAGE_SELF)
        soft = int(used.ru_utime+used.ru_stime)+1
        resource.setrlimit(resource.RLIMIT_CPU, (soft, soft+3))
        while True: pass
    return result
try:
    execute(evaluate, initial_subdivisions=2, execution=handle)
except BaseException as error:
    if handle.stop_requested:
        os.kill(os.getpid(), signal.SIGXCPU)
        os.kill(os.getpid(), signal.SIGXCPU)
    print(json.dumps({"type": type(error).__name__, "entries": len(trials.calls),
                      "signals": signals, "snapshot": handle.snapshot()}), flush=True)
    sys.exit(86)
sys.exit(99)
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent), OPENBLAS_NUM_THREADS="1")
        return subprocess.run([sys.executable, "-c", source, action], env=env,
                              text=True, capture_output=True, timeout=15)

    def test_actual_soft_cpu_signal_and_repeated_requests_retain_prefix(self):
        child = self.run_child("cpu")
        self.assertEqual(child.returncode, 86, child.stderr)
        row = json.loads(child.stdout)
        self.assertEqual(row["type"], "TemporalStopRequested")
        self.assertGreaterEqual(row["signals"], 3)
        self.assertEqual(row["entries"], 4)
        self.assertEqual(row["snapshot"]["completedFraction"], .5)
        self.assertEqual(row["snapshot"]["resources"]["chargedEvaluationAllowance"], 32)

    def test_actual_keyboard_interruption_keeps_prefix_without_fabricated_stop_request(self):
        child = self.run_child("keyboard")
        self.assertEqual(child.returncode, 86, child.stderr)
        row = json.loads(child.stdout)
        self.assertEqual(row["type"], "KeyboardInterrupt")
        self.assertFalse(row["snapshot"]["stopRequested"])
        self.assertEqual(row["snapshot"]["completedFraction"], .5)

    def test_hard_kill_has_no_recovery_claim(self):
        child = self.run_child("hard")
        self.assertEqual(child.returncode, -signal.SIGKILL, child.stderr)
        self.assertEqual(child.stdout, "")


if __name__ == "__main__": unittest.main()
