import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from solver_adaptive_contact import adaptive_contact_step
from solver_attempt_journal import AttemptJournal, recover_attempt_journal, strict_loads
from solver_process_budget import (ProgressStore, atomic_bytes, atomic_json, group_members,
                                  json_bytes, prepare_worker, sha256, supervise)


HERE = Path(__file__).resolve()


def capture(directory):
    report = {"accepted": False, "terminal": False, "completed": False,
              "classification": "uncompleted-research-diagnostic", "acceptedStateArtifacts": [],
              "sourceDigests": {}, "arguments": {"step_seconds": .01, "subdivisions": 2,
                                                    "max_depth": 3, "max_attempts": 8}}
    for name, key in (("canonical.json", "canonicalSha256"), ("placement.json", "placementSha256")):
        report[key] = atomic_json(directory / name, {"synthetic": True})["sha256"]
    snapshot = directory / "source-snapshot"
    snapshot.mkdir()
    for path in [*HERE.parent.glob("solver_*.py"), HERE]:
        report["sourceDigests"][path.name] = atomic_bytes(snapshot / path.name, path.read_bytes())["sha256"]
    ProgressStore(directory).save(report, "captured")
    return report


def run_numerical_worker(directory):
    from solver_process_budget import recover_progress
    import ipctk
    import newton
    import warp as wp
    from solver_global_sewing import GlobalSewingSolver
    from solver_ipc_contact import IpcSurfaceContact

    prepare_worker(20, os.getppid())
    report = recover_progress(directory)[0]["report"]
    ipctk.set_num_threads(1)
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for height in (0., .002):
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, height), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0]],
            indices=[0, 1, 2], density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0)
    builder.set_coloring([[vertex] for vertex in range(6)])
    model = builder.finalize(device="cpu")
    rest = model.particle_q.numpy().astype(float)
    velocity = np.zeros_like(rest)
    contact = IpcSurfaceContact(rest, model.tri_indices.numpy(), activation_distance_m=.005,
                               minimum_distance_m=.0001, stiffness=1e5)
    numerical = GlobalSewingSolver(model, [{0: 1., 1: -1.}], 1e-6, contact=contact)
    initial_targets = numerical.sewing @ rest
    journal = AttemptJournal(directory, report, rest, velocity, dt=.01,
                             initial_subdivisions=2, max_depth=3, max_attempts=8)
    report["attemptJournalRequired"] = True
    ProgressStore(directory).save(report, "adaptive-journal-ready")

    class InterruptedRetry:
        count = 0

        def step(self, positions, velocities, targets, dt, **options):
            self.count += 1
            expected_end = (.5, 1., .75)[self.count - 1]
            np.testing.assert_array_equal(targets, initial_targets + expected_end * (.99 * initial_targets - initial_targets))
            if self.count == 3:
                signal.signal(signal.SIGTERM, lambda *args: None)
                atomic_json(directory / "retry-ready.json", {"pid": os.getpid()})
                os.kill(os.getppid(), signal.SIGTERM)
                hashlib.pbkdf2_hmac("sha256", b"retry", b"native", 2_000_000_000)
                raise AssertionError("Native retry unexpectedly returned")
            result = numerical.step(positions, velocities, targets, dt,
                                    max_evaluations=1000 if self.count == 1 else 2)
            if result[2]["converged"] is not (self.count == 1):
                raise AssertionError(f"Unexpected real solver convergence: {result[2]}")
            return result

    adaptive_contact_step(InterruptedRetry(), rest, velocity, initial_targets, .99 * initial_targets,
                          .01, initial_subdivisions=2, max_depth=3, max_attempts=8, attempt_journal=journal)
    raise AssertionError("Interrupted continuation returned")


class JournalAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.report = capture(self.directory)
        self.positions = np.zeros((3, 3))
        self.velocities = np.zeros_like(self.positions)
        self.targets = np.zeros((1, 3))

    def journal(self):
        return AttemptJournal(self.directory, self.report, self.positions, self.velocities,
                              dt=.01, initial_subdivisions=2, max_depth=3, max_attempts=8)

    def run_adaptive(self, solver):
        return adaptive_contact_step(solver, self.positions, self.velocities, self.targets, self.targets,
                                     .01, initial_subdivisions=2, max_depth=3, max_attempts=8,
                                     attempt_journal=self.journal())

    def complete_fixture(self):
        class Stationary:
            def step(self, positions, velocities, targets, dt):
                return positions, velocities, {"converged": True, "gradientInfinityNorm": 0.}
        self.run_adaptive(Stationary())

    def rewrite_chain(self, sequence, mutate):
        previous = None
        for index, path in enumerate(sorted(self.directory.glob("attempt-*.json"))):
            envelope = strict_loads(path.read_bytes())
            if index == sequence:
                mutate(envelope["payload"])
            envelope["payload"]["previousSha256"] = previous
            envelope["sha256"] = sha256(json_bytes(envelope["payload"]))
            content = json_bytes(envelope)
            path.chmod(0o600)
            path.write_bytes(content)
            previous = sha256(content)

    def test_real_accepted_rejected_native_interrupted_retry(self):
        report = supervise([sys.executable, str(self.directory / "source-snapshot" / HERE.name),
                            "--worker", str(self.directory)], self.directory, self.report,
                           cpu_limit_seconds=20, wall_limit_seconds=25., terminate_grace_seconds=.1)
        self.assertTrue((self.directory / "retry-ready.json").exists(), report)
        self.assertFalse(report["completed"])
        self.assertTrue(report["terminal"])
        self.assertFalse(report["accepted"])
        self.assertTrue(report["captureIntegrityVerified"])
        self.assertTrue(report["supervision"]["cleanupComplete"])
        self.assertTrue(report["supervision"]["hardKillSent"])
        self.assertEqual(group_members(report["supervision"]["processGroupId"]), [])
        adaptive = report["adaptive"]
        self.assertEqual([entry["outcome"] for entry in adaptive["attempts"]],
                         ["accepted", "rejected", "interrupted"])
        self.assertEqual(adaptive["completedFraction"], .5)
        self.assertEqual(adaptive["completedDurationSeconds"], .005)
        self.assertEqual(adaptive["stationarityToleranceN"], 1e-6)
        first, rejected, retry = adaptive["attempts"]
        self.assertEqual(first["lastAcceptedState"]["path"], "initial-state.json")
        self.assertEqual(rejected["lastAcceptedState"], first["acceptedState"])
        self.assertEqual(retry["lastAcceptedState"], first["acceptedState"])
        self.assertEqual((retry["attemptId"], retry["parentAttemptId"], retry["initialInterval"],
                          retry["startFraction"], retry["endFraction"], retry["depth"], retry["durationSeconds"]),
                         (3, 2, 1, .5, .75, 1, .0025))
        self.assertFalse(rejected["step"]["converged"])
        self.assertLessEqual(first["step"]["gradientInfinityNorm"], 1e-6)
        self.assertEqual(report["acceptedStateArtifacts"], [first["acceptedState"]])
        self.assertEqual(report["attemptJournal"]["lastAcceptedState"], first["acceptedState"])
        self.assertEqual(report["recoveryErrors"], [])
        for entry in (first["lastAcceptedState"], first["acceptedState"]):
            self.assertEqual(sha256((self.directory / entry["path"]).read_bytes()), entry["sha256"])
        recovered = recover_attempt_journal(self.directory, self.report)
        errors = recovered["attemptJournal"]["errors"]
        self.assertEqual(errors, [])
        self.assertEqual(recovered["adaptive"]["attempts"], adaptive["attempts"])
        self.assertEqual(strict_loads((self.directory / "report.json").read_bytes()), report)
        self.assertEqual(report["sourceDigests"], self.report["sourceDigests"])
        self.assertEqual(report["arguments"], self.report["arguments"])

    def test_nonfinite_diagnostics_in_converged_candidate_never_admitted(self):
        class Nonfinite:
            def step(self, positions, velocities, targets, dt):
                positions[1, 2] = float("-inf")
                return positions, velocities, {"converged": True, "gradientInfinityNorm": 0.,
                    "nested/~": [np.float64("nan"), {"energy": float("inf")}]}
        final, _, report = self.run_adaptive(Nonfinite())
        np.testing.assert_array_equal(final, self.positions)
        self.assertEqual(report["acceptedSteps"], [])
        self.assertEqual(report["completedDurationSeconds"], 0.)
        recovered = recover_attempt_journal(self.directory, self.report)
        errors = recovered["attemptJournal"]["errors"]
        self.assertEqual(errors, [])
        for attempt in recovered["adaptive"]["attempts"]:
            self.assertEqual(attempt["nonfiniteDiagnostics"], [
                {"path": "/step/nested~1~0/0", "$nonfinite": "NaN"},
                {"path": "/step/nested~1~0/1/energy", "$nonfinite": "+Infinity"},
                {"path": "/candidatePositions/1/2", "$nonfinite": "-Infinity"}])
            self.assertEqual(attempt["step"]["nested/~"][0],
                             {"$nonfinite": "NaN", "path": "/step/nested~1~0/0"})
        for path in self.directory.glob("attempt-*.json"):
            strict_loads(path.read_bytes())

    def test_unknown_exception_is_durable_before_propagation(self):
        class Broken:
            def step(self, *args):
                raise LookupError("unexpected solver diagnostic")
        with self.assertRaisesRegex(LookupError, "unexpected"):
            self.run_adaptive(Broken())
        recovered = recover_attempt_journal(self.directory, self.report)
        errors = recovered["attemptJournal"]["errors"]
        self.assertEqual(errors, [])
        self.assertEqual(len(recovered["adaptive"]["attempts"]), 1)
        self.assertEqual(recovered["adaptive"]["attempts"][0]["error"]["type"], "LookupError")
        self.assertEqual(recovered["adaptive"]["completedFraction"], 0.)

    def test_malformed_start_relationship_rejects_valid_hash_chain(self):
        self.complete_fixture()
        self.rewrite_chain(3, lambda event: event["data"].update(parentAttemptId=1))
        recovered = recover_attempt_journal(self.directory, self.report, interruption="test")
        errors = recovered["attemptJournal"]["errors"]
        self.assertTrue(errors)
        self.assertEqual(len(recovered["acceptedStateArtifacts"]), 1)
        self.assertEqual(recovered["adaptive"]["completedFraction"], .5)
        self.assertFalse(recovered["adaptive"]["complete"])

    def test_malformed_accepted_state_never_advances_prefix(self):
        self.complete_fixture()
        artifact = self.directory / "accepted-0002.json"
        state = strict_loads(artifact.read_bytes())
        state["positionsMeters"][0][0] = {"$nonfinite": "NaN"}
        artifact.chmod(0o600)
        artifact.write_bytes(json_bytes(state))
        digest = sha256(artifact.read_bytes())
        self.rewrite_chain(4, lambda event: event["data"]["acceptedState"].update(sha256=digest))
        recovered = recover_attempt_journal(self.directory, self.report, interruption="invalid-state")
        errors = recovered["attemptJournal"]["errors"]
        self.assertTrue(errors)
        self.assertEqual(recovered["adaptive"]["completedFraction"], .5)
        self.assertEqual(len(recovered["acceptedStateArtifacts"]), 1)

    def test_missing_middle_record_does_not_skip_to_later_acceptance(self):
        self.complete_fixture()
        (self.directory / "attempt-000003.json").unlink()
        recovered = recover_attempt_journal(self.directory, self.report)
        errors = recovered["attemptJournal"]["errors"]
        self.assertTrue(errors)
        self.assertEqual(recovered["adaptive"]["completedFraction"], .5)
        self.assertFalse(recovered["adaptive"]["complete"])

    def test_nonfinite_strict_json_literals_and_duplicate_keys_rejected(self):
        for content in (b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e999}',
                        b'{"x":0,"x":1}'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                strict_loads(content)

    def test_malformed_candidate_retains_nonfinite_step_diagnostics(self):
        class Malformed:
            def step(self, positions, velocities, targets, dt):
                return [["not-a-number"]], velocities, {
                    "converged": False, "gradientInfinityNorm": float("nan"), "energy": float("inf")}

        self.run_adaptive(Malformed())
        recovered = recover_attempt_journal(self.directory, self.report)
        self.assertEqual(recovered["attemptJournal"]["errors"], [])
        self.assertEqual(recovered["acceptedStateArtifacts"], [])
        for attempt in recovered["adaptive"]["attempts"]:
            self.assertEqual(attempt["nonfiniteDiagnostics"], [
                {"$nonfinite": "NaN", "path": "/step/gradientInfinityNorm"},
                {"$nonfinite": "+Infinity", "path": "/step/energy"}])
            self.assertEqual(attempt["error"]["type"], "ValueError")

    def test_boolean_accepted_duration_rejected_even_with_consistent_hashes(self):
        self.report["arguments"]["step_seconds"] = 2.
        journal = AttemptJournal(self.directory, self.report, self.positions, self.velocities,
                                 dt=2., initial_subdivisions=2, max_depth=3, max_attempts=8)

        class Stationary:
            def step(self, positions, velocities, targets, dt):
                return positions, velocities, {"converged": True, "gradientInfinityNorm": 0.}

        adaptive_contact_step(Stationary(), self.positions, self.velocities, self.targets, self.targets,
                              2., initial_subdivisions=2, max_depth=3, max_attempts=8, attempt_journal=journal)
        artifact = self.directory / "accepted-0001.json"
        state = strict_loads(artifact.read_bytes())
        state["record"]["completedDurationSeconds"] = True
        artifact.chmod(0o600)
        artifact.write_bytes(json_bytes(state))

        def malformed(event):
            event["data"]["completedDurationSeconds"] = True
            event["data"]["acceptedState"]["sha256"] = sha256(artifact.read_bytes())

        self.rewrite_chain(2, malformed)
        recovered = recover_attempt_journal(self.directory, self.report)
        self.assertEqual(recovered["acceptedStateArtifacts"], [])
        self.assertEqual(recovered["adaptive"]["completedFraction"], 0.)
        self.assertTrue(recovered["attemptJournal"]["errors"])

    def test_malformed_outcome_identity_and_flags_fail_closed(self):
        self.complete_fixture()
        paths = sorted(self.directory.glob("attempt-*.json"))
        originals = {path: path.read_bytes() for path in paths}
        changes = ({"attemptId": True}, {"startFraction": .25}, {"endFraction": .75},
                   {"durationSeconds": .003}, {"depth": 1}, {"parentAttemptId": 0},
                   {"lastAcceptedState": {"path": "accepted-0001.json", "sha256": "invalid"}},
                   {"fatal": 0}, {"converged": 1}, {"outcome": "complete"},
                   {"completedDurationSeconds": .01})
        for change in changes:
            with self.subTest(change=change):
                for path, content in originals.items():
                    path.chmod(0o600)
                    path.write_bytes(content)
                self.rewrite_chain(2, lambda event: event["data"].update(change))
                recovered = recover_attempt_journal(self.directory, self.report)
                self.assertTrue(recovered["attemptJournal"]["errors"])
                self.assertEqual(recovered["acceptedStateArtifacts"], [])
                self.assertEqual(recovered["adaptive"]["completedDurationSeconds"], 0.)

    def test_malformed_nonfinite_tag_and_paths_fail_closed(self):
        class Nonfinite:
            def step(self, positions, velocities, targets, dt):
                return positions, velocities, {"converged": False, "gradientInfinityNorm": float("nan")}
        self.run_adaptive(Nonfinite())
        paths = sorted(self.directory.glob("attempt-*.json"))
        originals = {path: path.read_bytes() for path in paths}
        mutations = (
            lambda data: data["step"]["gradientInfinityNorm"].update(path="/elsewhere"),
            lambda data: data["step"]["gradientInfinityNorm"].update(extra=True),
            lambda data: data["step"]["gradientInfinityNorm"].update({"$nonfinite": "infinite"}),
            lambda data: data.update(nonfiniteDiagnostics=[]),
            lambda data: data["nonfiniteDiagnostics"].append(data["nonfiniteDiagnostics"][0]),
            lambda data: data.update(error={"type": 1, "message": "bad"}),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                for path, content in originals.items():
                    path.chmod(0o600)
                    path.write_bytes(content)
                self.rewrite_chain(2, lambda event: mutate(event["data"]))
                recovered = recover_attempt_journal(self.directory, self.report)
                self.assertTrue(recovered["attemptJournal"]["errors"])
                self.assertFalse(recovered["adaptive"]["complete"])
                self.assertEqual(recovered["acceptedStateArtifacts"], [])

    def test_false_finish_reason_does_not_complete_or_erase_prefix(self):
        self.complete_fixture()
        self.rewrite_chain(5, lambda event: event["data"].update(complete=False, reason="invented"))
        recovered = recover_attempt_journal(self.directory, self.report)
        self.assertTrue(recovered["attemptJournal"]["errors"])
        self.assertFalse(recovered["adaptive"]["complete"])
        self.assertEqual(len(recovered["acceptedStateArtifacts"]), 2)
        self.assertEqual(recovered["adaptive"]["completedDurationSeconds"], .01)

    def test_partial_record_and_stale_pointer_do_not_advance(self):
        self.complete_fixture()
        path = self.directory / "attempt-000004.json"
        path.chmod(0o600)
        path.write_bytes(b'{"payload":')
        atomic_json(self.directory / "progress.json", {"invalid": "stale pointer"}, replace=True)
        recovered = recover_attempt_journal(self.directory, self.report, interruption="truncated")
        self.assertTrue(recovered["attemptJournal"]["errors"])
        self.assertEqual(recovered["adaptive"]["completedFraction"], .5)
        self.assertEqual(recovered["adaptive"]["attempts"][-1]["outcome"], "interrupted")
        self.assertTrue(recovered["adaptive"]["attempts"][-1]["recoveredWithoutDurableOutcome"])
        self.assertEqual(path.read_bytes(), b'{"payload":')

    def test_accepted_state_symlink_rejected_even_with_matching_bytes(self):
        self.complete_fixture()
        path = self.directory / "accepted-0002.json"
        target = self.directory / "copied-state.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        recovered = recover_attempt_journal(self.directory, self.report)
        self.assertTrue(recovered["attemptJournal"]["errors"])
        self.assertEqual(len(recovered["acceptedStateArtifacts"]), 1)
        self.assertEqual(recovered["adaptive"]["completedFraction"], .5)

    def test_accepted_before_callback_failure_is_preserved(self):
        class Stationary:
            def step(self, positions, velocities, targets, dt):
                return positions, velocities, {"converged": True, "gradientInfinityNorm": 0.}
        def callback(*args):
            raise ValueError("post-commit callback fault")
        with self.assertRaisesRegex(ValueError, "post-commit"):
            adaptive_contact_step(Stationary(), self.positions, self.velocities, self.targets, self.targets,
                                  .01, initial_subdivisions=2, max_depth=3, max_attempts=8,
                                  attempt_journal=self.journal(), on_accept=callback)
        recovered = recover_attempt_journal(self.directory, self.report, interruption="callback failed")
        self.assertEqual(recovered["attemptJournal"]["errors"], [])
        self.assertEqual(recovered["adaptive"]["completedFraction"], .5)
        self.assertEqual(len(recovered["adaptive"]["attempts"]), 1)
        self.assertFalse(recovered["adaptive"]["complete"])

    def test_underflow_finishes_without_start_or_solver_execution(self):
        tiny = float(np.nextafter(0., 1.))
        self.report["arguments"]["step_seconds"] = tiny
        journal = AttemptJournal(self.directory, self.report, self.positions, self.velocities,
                                 dt=tiny, initial_subdivisions=2, max_depth=3, max_attempts=8)
        solver = mock.Mock(sewing_mode="vector", fold_actuation=None, controlled_fold_actuation=None, material_grippers=None)
        adaptive_contact_step(solver, self.positions, self.velocities, self.targets, self.targets,
                              tiny, initial_subdivisions=2, max_depth=3, max_attempts=8,
                              attempt_journal=journal)
        solver.step.assert_not_called()
        recovered = recover_attempt_journal(self.directory, self.report)
        self.assertEqual(recovered["attemptJournal"]["errors"], [])
        self.assertEqual(recovered["adaptive"]["reason"], "substep-duration-underflow")
        self.assertEqual(recovered["adaptive"]["attempts"], [])

    def test_failed_outcome_publication_never_admits_orphan_state(self):
        from solver_attempt_journal import atomic_json as publish
        class Stationary:
            def step(self, positions, velocities, targets, dt):
                return positions, velocities, {"converged": True, "gradientInfinityNorm": 0.}
        def fail_outcome(path, value, **options):
            if Path(path).name == "attempt-000002.json":
                raise OSError("injected publication fault")
            return publish(path, value, **options)
        with mock.patch("solver_attempt_journal.atomic_json", side_effect=fail_outcome):
            with self.assertRaisesRegex(OSError, "publication fault"):
                self.run_adaptive(Stationary())
        self.assertTrue((self.directory / "accepted-0001.json").exists())
        recovered = recover_attempt_journal(self.directory, self.report, interruption="publication fault")
        self.assertEqual(recovered["acceptedStateArtifacts"], [])
        self.assertEqual(recovered["adaptive"]["completedFraction"], 0.)
        self.assertEqual(recovered["adaptive"]["attempts"][0]["outcome"], "interrupted")
        self.assertEqual(recovered["attemptJournal"]["errors"], [])

    def test_evidence_budget_rejects_before_overwrite_and_keeps_prefix(self):
        self.complete_fixture()
        with mock.patch("solver_attempt_journal.MAX_RECORD_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "No intact"):
                recover_attempt_journal(self.directory, self.report)
        journal_before = {path: path.read_bytes() for path in self.directory.glob("attempt-*.json")}
        with self.assertRaises(FileExistsError):
            self.journal()
        self.assertEqual({path: path.read_bytes() for path in journal_before}, journal_before)
        with mock.patch("solver_attempt_journal.MAX_JOURNAL_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "No intact"):
                recover_attempt_journal(self.directory, self.report)

    def test_supervisor_rejects_boolean_final_duration(self):
        self.report["arguments"]["step_seconds"] = 1.
        journal = AttemptJournal(self.directory, self.report, self.positions, self.velocities,
                                 dt=1., initial_subdivisions=2, max_depth=3, max_attempts=8)

        class Stationary:
            def step(self, positions, velocities, targets, dt):
                return positions, velocities, {"converged": True, "gradientInfinityNorm": 0.}

        adaptive_contact_step(Stationary(), self.positions, self.velocities, self.targets, self.targets,
                              1., initial_subdivisions=2, max_depth=3, max_attempts=8, attempt_journal=journal)
        report = dict(self.report, **recover_attempt_journal(self.directory, self.report))
        report.update(attemptJournalRequired=True, classification="completed-research-interval")
        report["stateArtifact"] = atomic_json(self.directory / "state.json", {
            "positionsMeters": self.positions.tolist(),
            "velocitiesMetersPerSecond": self.velocities.tolist(),
            "completedDurationSeconds": True, "accepted": False})
        ProgressStore(self.directory).save(report, "worker-finished", exit_code=0)
        terminal = supervise([sys.executable, "-c", "pass"], self.directory, self.report,
                             cpu_limit_seconds=5, wall_limit_seconds=5.)
        self.assertFalse(terminal["completed"])
        self.assertNotIn("stateArtifact", terminal)
        self.assertTrue(terminal["recoveryErrors"])
        self.assertEqual(terminal["adaptive"]["completedDurationSeconds"], 1.)
        self.assertEqual(len(terminal["acceptedStateArtifacts"]), 2)

    def test_terminal_recovery_preserves_nonfinite_rejection_before_interruption(self):
        class InterruptedAfterNonfinite:
            count = 0

            def step(self, positions, velocities, targets, dt):
                self.count += 1
                if self.count == 1:
                    return positions + dt, velocities, {"converged": True, "gradientInfinityNorm": 0.}
                if self.count == 2:
                    return positions, velocities, {"converged": False, "gradientInfinityNorm": float("nan"),
                                                   "nested/~": [float("inf"), float("-inf")]}
                raise KeyboardInterrupt("injected interruption")

        with self.assertRaises(KeyboardInterrupt):
            self.run_adaptive(InterruptedAfterNonfinite())
        terminal = supervise([sys.executable, "-c", "pass"], self.directory, self.report,
                             cpu_limit_seconds=5, wall_limit_seconds=5.)
        self.assertTrue(terminal["terminal"])
        self.assertFalse(terminal["completed"])
        self.assertFalse(terminal["accepted"])
        self.assertTrue(terminal["captureIntegrityVerified"])
        self.assertEqual(terminal["recoveryErrors"], [])
        first, rejected, retry = terminal["adaptive"]["attempts"]
        self.assertEqual([first["outcome"], rejected["outcome"], retry["outcome"]],
                         ["accepted", "rejected", "interrupted"])
        self.assertEqual(terminal["adaptive"]["completedDurationSeconds"], .005)
        self.assertEqual(terminal["adaptive"]["completedFraction"], .5)
        self.assertEqual(retry["lastAcceptedState"], first["acceptedState"])
        self.assertEqual(retry["parentAttemptId"], rejected["attemptId"])
        self.assertEqual(rejected["nonfiniteDiagnostics"], [
            {"$nonfinite": "NaN", "path": "/step/gradientInfinityNorm"},
            {"$nonfinite": "+Infinity", "path": "/step/nested~1~0/0"},
            {"$nonfinite": "-Infinity", "path": "/step/nested~1~0/1"}])
        self.assertEqual(strict_loads((self.directory / "report.json").read_bytes()), terminal)
        self.assertEqual(terminal["sourceDigests"], self.report["sourceDigests"])
        self.assertEqual(group_members(terminal["supervision"]["processGroupId"]), [])

    def test_exhausted_writer_budget_reserves_durable_interruption(self):
        import solver_attempt_journal as journal_module

        journal = self.journal()
        record = {"attemptId": 1, "parentAttemptId": None, "initialInterval": 0,
                  "startFraction": 0., "endFraction": .5, "durationSeconds": .005,
                  "depth": 0, "converged": False}
        journal.start(record)
        outcome = dict(record, outcome="rejected", fatal=False,
                       step={"converged": False, "gradientInfinityNorm": 1.})
        cap = journal.byte_count + journal_module.RECOVERY_RESERVE_BYTES + 1
        with mock.patch.object(journal_module, "MAX_JOURNAL_BYTES", cap):
            with self.assertRaisesRegex(ValueError, "byte budget"):
                journal.outcome(outcome)
            recovered = recover_attempt_journal(self.directory, self.report, interruption="Evidence budget exhausted")
            self.assertEqual(recovered["attemptJournal"]["errors"], [])
            attempt = recovered["adaptive"]["attempts"][0]
            self.assertEqual(attempt["outcome"], "interrupted")
            self.assertNotIn("recoveredWithoutDurableOutcome", attempt)
            self.assertEqual(recovered["adaptive"]["completedFraction"], 0.)
        replay = recover_attempt_journal(self.directory, self.report)
        self.assertEqual(replay["adaptive"]["attempts"], recovered["adaptive"]["attempts"])

    def test_journal_reserves_outcome_bytes_before_executing_solver(self):
        from solver_attempt_journal import MAX_RECORD_BYTES, RECOVERY_RESERVE_BYTES
        journal = self.journal()
        solver = mock.Mock(sewing_mode="vector", fold_actuation=None, controlled_fold_actuation=None, material_grippers=None)
        limit = journal.byte_count + MAX_RECORD_BYTES + RECOVERY_RESERVE_BYTES
        with mock.patch("solver_attempt_journal.MAX_JOURNAL_BYTES", limit):
            with self.assertRaisesRegex(ValueError, "byte budget"):
                adaptive_contact_step(solver, self.positions, self.velocities, self.targets, self.targets,
                                      .01, initial_subdivisions=2, max_depth=3, max_attempts=8,
                                      attempt_journal=journal)
            recovered = recover_attempt_journal(self.directory, self.report)
        solver.step.assert_not_called()
        self.assertIsNone(journal.active)
        self.assertEqual(recovered["adaptive"]["attempts"], [])
        self.assertEqual(recovered["attemptJournal"]["errors"], [])
        self.assertEqual(len(recovered["attemptJournal"]["events"]), 1)
        self.assertFalse(recovered["adaptive"]["complete"])

    def test_nonfinite_tag_index_counts_toward_diagnostic_byte_budget(self):
        from solver_attempt_journal import diagnostic_json
        sample = {"residual": float("nan")}
        tagged, tags = diagnostic_json(sample)
        result_size = len(json_bytes(tagged))
        combined_size = len(json_bytes([tagged, tags]))
        self.assertGreater(combined_size, result_size)
        with mock.patch("solver_attempt_journal.MAX_RECORD_BYTES", 2 * result_size):
            with self.assertRaisesRegex(ValueError, "Diagnostic byte budget"):
                diagnostic_json(sample)

    def test_header_provenance_and_controls_checked(self):
        self.complete_fixture()
        changed = copy.deepcopy(self.report)
        changed["arguments"]["max_depth"] = 4
        with self.assertRaises(ValueError):
            recover_attempt_journal(self.directory, changed)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        run_numerical_worker(Path(sys.argv[2]))
    else:
        unittest.main()
