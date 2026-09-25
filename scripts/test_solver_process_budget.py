import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from solver_process_budget import (ProgressStore, atomic_bytes, atomic_json, group_members,
                                   json_bytes, prepare_worker, read_regular, recover_progress,
                                   sha256, supervise)


HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
PYTHON = sys.executable


def initial_capture(directory):
    (directory / "source-snapshot").mkdir()
    canonical = atomic_json(directory / "canonical.json", {"synthetic": True})
    placement = atomic_json(directory / "placement.json", {"synthetic": True})
    source = atomic_bytes(directory / "source-snapshot" / "fixture.py", b"pass\n")
    report = {"accepted": False, "completed": False, "terminal": False,
              "classification": "uncompleted-research-diagnostic",
              "arguments": {"synthetic": True}, "acceptedStateArtifacts": [],
              "canonicalSha256": canonical["sha256"], "placementSha256": placement["sha256"],
              "sourceDigests": {"fixture.py": source["sha256"]}}
    ProgressStore(directory).save(report, "captured")
    return report


def state(directory, name):
    return atomic_json(directory / name, {"accepted": False, "positionsMeters": [[0, 0, 0]],
                                          "velocitiesMetersPerSecond": [[0, 0, 0]]})


def native_stuck():
    def deferred_handler(signum, frame):
        raise TimeoutError("Python handler ran")
    signal.signal(signal.SIGTERM, deferred_handler)
    signal.signal(signal.SIGXCPU, deferred_handler)
    hashlib.pbkdf2_hmac("sha256", b"public-synthetic-test", b"salt", 2_000_000_000)
    raise AssertionError("Native fixture unexpectedly finished")


def fixture_worker(directory, mode):
    prepare_worker(30, os.getppid())
    payload, _, _ = recover_progress(directory)
    report = payload["report"]
    progress = ProgressStore(directory)
    report["acceptedStateArtifacts"].append(state(directory, "accepted-0001.json"))
    progress.save(report, "accepted-one")
    atomic_json(directory / "ready.json", {"pid": os.getpid(), "pgid": os.getpgrp()})
    if mode == "native":
        native_stuck()
    if mode == "rlimit-hard-kill":
        resource.setrlimit(resource.RLIMIT_CPU, (1, 2))
        native_stuck()
    if mode == "sleep":
        time.sleep(60)
    if mode == "signal":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "nonzero":
        os._exit(7)
    if mode == "unclean-zero":
        os._exit(0)
    if mode == "orphan-artifact":
        state(directory, "accepted-0002.json")
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "partial-write":
        (directory / ".accepted-0002.json.crashed.tmp").write_bytes(b'{"positionsMeters":[')
        os.kill(os.getpid(), signal.SIGKILL)
    if mode in ("descendants", "descendant-exit", "descendant-cpu"):
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if mode != "descendant-cpu":
                os.fork()
                while True:
                    time.sleep(60)
            native_stuck()
        atomic_json(directory / "descendant.json", {"pid": child})
        if mode == "descendant-exit":
            report["adaptive"] = {"complete": True}
            report["classification"] = "completed-research-interval"
            progress.save(report, "worker-finished", exit_code=0)
            os._exit(0)
        native_stuck() if mode == "descendants" else time.sleep(60)
    report["stateArtifact"] = state(directory, "state.json")
    report["adaptive"] = {"complete": mode not in ("partial", "failure")}
    report["classification"] = ("completed-research-interval" if report["adaptive"]["complete"]
                                 else "partial-research-interval")
    exit_code = 1 if mode in ("partial", "failure", "inconsistent") else 0
    if mode == "failure":
        report["failure"] = {"type": "ValueError", "message": "synthetic failure"}
    if mode == "provenance":
        report["canonicalSha256"] = "wrong"
    progress.save(report, "worker-finished", exit_code=exit_code)
    if mode == "after-finished":
        native_stuck()
    if mode == "corrupt-state":
        os.chmod(directory / "state.json", 0o600)
        (directory / "state.json").write_bytes(b"not json")
    if mode == "corrupt-capture":
        os.chmod(directory / "canonical.json", 0o600)
        (directory / "canonical.json").write_bytes(b"{}")
    if mode == "corrupt-source":
        path = directory / "source-snapshot" / "fixture.py"
        os.chmod(path, 0o600)
        path.write_bytes(b"changed\n")
    return exit_code


def fixture_supervisor(directory, mode):
    report = recover_progress(directory)[0]["report"]
    result = supervise([PYTHON, str(HERE), "--worker", str(directory), mode], directory, report,
                       cpu_limit_seconds=30, wall_limit_seconds=30, terminate_grace_seconds=.1)
    return 0 if result["completed"] else 1


class ProcessBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="solver-budget-public-test-")
        self.directory = Path(self.temporary.name)
        self.initial = initial_capture(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def run_worker(self, mode, *, wall=8, cpu=30, command=None):
        report = supervise(command or [PYTHON, str(HERE), "--worker", str(self.directory), mode],
                           self.directory, self.initial, cpu_limit_seconds=cpu,
                           wall_limit_seconds=wall, terminate_grace_seconds=.1)
        self.assertEqual(report, json.loads((self.directory / "report.json").read_bytes()))
        self.assertIs(report["accepted"], False)
        self.assertTrue(report["terminal"])
        self.assertTrue(report["supervision"]["cleanupComplete"], report)
        pgid = report["supervision"]["processGroupId"]
        if pgid is not None:
            self.assertEqual(group_members(pgid), [])
            with self.assertRaises(ProcessLookupError):
                os.killpg(pgid, 0)
        self.assertGreaterEqual(report["wallSeconds"], report["workerBudgetWallSeconds"])
        self.assertAlmostEqual(report["cpuSeconds"], report["supervision"]["cpuUserSeconds"]
                               + report["supervision"]["cpuSystemSeconds"])
        return report

    def assert_uncompleted(self, report):
        self.assertFalse(report["completed"])
        self.assertNotEqual(report["classification"], "completed-research-interval")
        self.assertIsNot(report.get("adaptive", {}).get("complete"), True)

    def test_normal_completed_exit(self):
        report = self.run_worker("normal")
        self.assertTrue(report["completed"])
        self.assertTrue(report["captureIntegrityVerified"])
        self.assertEqual(report["supervision"]["reason"], "worker-exited")
        self.assertEqual(report["supervision"]["workerReturnCode"], 0)
        self.assertEqual(len(report["acceptedStateArtifacts"]), 1)
        self.assertFalse(report["supervision"]["hardKillSent"])

    def test_normal_partial_exit(self):
        report = self.run_worker("partial")
        self.assert_uncompleted(report)
        self.assertEqual(report["classification"], "partial-research-interval")
        self.assertNotIn("failure", report)
        self.assertEqual(report["supervision"]["workerReturnCode"], 1)

    def test_interrupted_capture_drops_cached_gripper_work_summary(self):
        self.initial["gripperWorkSummary"] = {"acceptedSteps": 2, "gripperParameterWorkJoules": .125}
        self.initial["sewingWorkSummary"] = {"acceptedSteps": 2, "sewingParameterWorkJoules": .25}
        ProgressStore(self.directory).save(self.initial, "synthetic-worker-aggregate")
        report = self.run_worker("signal")
        self.assert_uncompleted(report)
        self.assertEqual(len(report["acceptedStateArtifacts"]), 1)
        self.assertNotIn("gripperWorkSummary", report)
        self.assertNotIn("sewingWorkSummary", report)

    def test_normal_failure_exit(self):
        report = self.run_worker("failure")
        self.assert_uncompleted(report)
        self.assertEqual(report["failure"]["type"], "ValueError")

    def test_inconsistent_completed_exit_fails_closed(self):
        report = self.run_worker("inconsistent")
        self.assert_uncompleted(report)
        self.assertEqual(report["failure"]["message"], "inconsistent-worker-completion")

    def test_native_wall_timeout_requires_hard_kill(self):
        report = self.run_worker("native", wall=.6)
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "wall-budget-exhausted")
        self.assertEqual(report["supervision"]["workerSignal"], signal.SIGKILL)
        self.assertTrue(report["supervision"]["hardKillSent"])
        self.assertEqual(len(report["acceptedStateArtifacts"]), 1)
        self.assertLess(report["wallSeconds"], 5)

    def test_parent_cpu_budget_stops_native_call(self):
        report = self.run_worker("native", cpu=1)
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "cpu-budget-exhausted")
        self.assertTrue(report["supervision"]["hardKillSent"])
        self.assertGreaterEqual(report["cpuSeconds"], .9)
        self.assertLess(report["supervisorCpuSeconds"], report["cpuSeconds"])

    def test_kernel_hard_kill_still_recovers_ledger(self):
        report = self.run_worker("rlimit-hard-kill")
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "worker-signal")
        self.assertEqual(report["supervision"]["workerSignal"], signal.SIGKILL)
        self.assertFalse(report["supervision"]["hardKillSent"])
        self.assertEqual(len(report["acceptedStateArtifacts"]), 1)
        self.assertGreater(report["cpuSeconds"], 1.5)

    def test_sleep_uses_wall_not_supervisor_cpu(self):
        report = self.run_worker("sleep", wall=.6)
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "wall-budget-exhausted")
        self.assertLess(report["cpuSeconds"], .4)
        self.assertGreaterEqual(report["wallSeconds"], .6)
        # The fixed grace permits escalation; scheduling does not guarantee
        # that the worker has exited before that grace expires.
        self.assertIn(report["supervision"]["workerReturnCode"], (-signal.SIGTERM, -signal.SIGKILL))

    def test_signal_failure(self):
        report = self.run_worker("signal")
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["workerSignal"], signal.SIGKILL)

    def test_nonzero_unclean_failure(self):
        report = self.run_worker("nonzero")
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["workerReturnCode"], 7)

    def test_zero_without_final_checkpoint_is_not_complete(self):
        report = self.run_worker("unclean-zero")
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "worker-incomplete")

    def test_finished_checkpoint_before_stuck_exit_is_not_complete(self):
        report = self.run_worker("after-finished", wall=.6)
        self.assert_uncompleted(report)
        self.assertEqual(report["lastWorkerPhase"], "worker-finished")

    def test_orphan_state_is_not_admitted(self):
        report = self.run_worker("orphan-artifact")
        self.assert_uncompleted(report)
        self.assertEqual(len(report["acceptedStateArtifacts"]), 1)
        self.assertEqual(report["uncommittedStateArtifacts"], ["accepted-0002.json"])

    def test_crash_mid_temporary_write_keeps_prior_checkpoint(self):
        report = self.run_worker("partial-write")
        self.assert_uncompleted(report)
        self.assertEqual(len(report["acceptedStateArtifacts"]), 1)
        self.assertEqual(report["recoveryErrors"], [])
        self.assertEqual(report["uncommittedStateArtifacts"], [])

    def test_descendants_killed_and_reaped(self):
        report = self.run_worker("descendants", wall=.6)
        self.assert_uncompleted(report)
        self.assertTrue(report["supervision"]["hardKillSent"])

    def test_completed_leader_cannot_abandon_descendants(self):
        report = self.run_worker("descendant-exit")
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "worker-left-descendants")
        self.assertEqual(report["supervision"]["workerReturnCode"], 0)

    def test_descendant_cpu_included_in_budget_and_accounting(self):
        report = self.run_worker("descendant-cpu", cpu=1)
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "cpu-budget-exhausted")
        self.assertGreaterEqual(report["cpuSeconds"], .9)

    def test_corrupt_state_recovers_prior_accepted_state(self):
        report = self.run_worker("corrupt-state")
        self.assert_uncompleted(report)
        self.assertEqual(report["lastWorkerPhase"], "accepted-one")
        self.assertEqual(len(report["acceptedStateArtifacts"]), 1)
        self.assertTrue(report["recoveryErrors"])

    def test_provenance_mutation_rejected(self):
        report = self.run_worker("provenance")
        self.assert_uncompleted(report)
        self.assertEqual(report["canonicalSha256"], self.initial["canonicalSha256"])
        self.assertEqual(report["lastWorkerPhase"], "accepted-one")
        self.assertEqual(len(report["acceptedStateArtifacts"]), 1)
        self.assertTrue(report["recoveryErrors"])

    def test_provenance_boolean_numeric_alias_rejected(self):
        changed = dict(self.initial, arguments={"synthetic": 1})
        ProgressStore(self.directory).save(changed, "changed-provenance")
        payload, _, errors = recover_progress(self.directory, expected_report=self.initial)
        self.assertEqual(payload["phase"], "captured")
        self.assertIs(payload["report"]["arguments"]["synthetic"], True)
        self.assertTrue(errors)

    def test_modified_input_fails_parent_integrity_check(self):
        report = self.run_worker("corrupt-capture")
        self.assert_uncompleted(report)
        self.assertFalse(report["captureIntegrityVerified"])
        self.assertEqual(report["supervision"]["reason"], "capture-integrity-failure")

    def test_modified_source_fails_parent_integrity_check(self):
        report = self.run_worker("corrupt-source")
        self.assert_uncompleted(report)
        self.assertFalse(report["captureIntegrityVerified"])

    def test_spawn_failure_produces_terminal_report(self):
        report = self.run_worker("unused", command=[str(self.directory / "nonexistent")])
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "supervisor-failure")
        self.assertIsNone(report["supervision"]["processGroupId"])

    def test_supervisor_signals_produce_failure_and_cleanup(self):
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signum), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                initial_capture(directory)
                process = subprocess.Popen([PYTHON, str(HERE), "--supervisor", str(directory), "native"],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                worker_pid = None
                try:
                    deadline = time.monotonic() + 5
                    while not (directory / "ready.json").exists() and process.poll() is None:
                        if time.monotonic() >= deadline:
                            self.fail("Fixture did not reach native call")
                        time.sleep(.01)
                    worker_pid = json.loads((directory / "ready.json").read_bytes())["pid"]
                    time.sleep(.05)
                    process.send_signal(signum)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 1, (stdout, stderr))
                    report = json.loads((directory / "report.json").read_bytes())
                    self.assert_uncompleted(report)
                    self.assertEqual(report["supervision"]["supervisorSignal"], signum)
                    self.assertEqual(report["supervision"]["reason"], "supervisor-interrupted")
                    self.assertTrue(report["supervision"]["cleanupComplete"])
                    self.assertEqual(group_members(worker_pid), [])
                finally:
                    if worker_pid is not None:
                        try:
                            os.killpg(worker_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)

    def test_supervisor_sigkill_kills_direct_worker_without_false_completion(self):
        from solver_process_budget import _prctl
        old_subreaper = ctypes.c_int()
        _prctl(37, ctypes.byref(old_subreaper))
        _prctl(36, 1)
        process = None
        worker_pid = None
        reaped = False
        try:
            process = subprocess.Popen([PYTHON, str(HERE), "--supervisor", str(self.directory), "native"],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            deadline = time.monotonic() + 5
            while not (self.directory / "ready.json").exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    self.fail("Fixture did not reach native call")
                time.sleep(.01)
            worker_pid = json.loads((self.directory / "ready.json").read_bytes())["pid"]
            process.kill()
            process.communicate(timeout=5)
            deadline = time.monotonic() + 5
            while True:
                pid, status = os.waitpid(worker_pid, os.WNOHANG)
                if pid:
                    reaped = True
                    self.assertEqual(os.waitstatus_to_exitcode(status), -signal.SIGKILL)
                    break
                if time.monotonic() >= deadline:
                    self.fail("Orphan worker did not honor parent-death signal")
                time.sleep(.01)
            self.assertEqual(group_members(worker_pid), [])
            self.assertFalse((self.directory / "report.json").exists())
            payload, _, errors = recover_progress(self.directory)
            self.assertEqual(errors, [])
            self.assertFalse(payload["report"]["completed"])
            self.assertFalse(payload["report"]["terminal"])
        finally:
            if worker_pid is not None and not reaped:
                try:
                    os.killpg(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)
            if worker_pid is not None and not reaped:
                os.waitpid(worker_pid, 0)
            _prctl(36, old_subreaper.value)

    def test_interrupt_during_terminal_publication_cannot_complete(self):
        self.initial["gripperWorkSummary"] = {"acceptedSteps": 1, "gripperParameterWorkJoules": .125}
        self.initial["sewingWorkSummary"] = {"acceptedSteps": 1, "sewingParameterWorkJoules": .25}
        ProgressStore(self.directory).save(self.initial, "synthetic-worker-aggregate")
        original = atomic_json
        injected = False

        def interrupted_publication(path, value, **kwargs):
            nonlocal injected
            if Path(path).name == "report.json" and value.get("completed") and not injected:
                injected = True
                os.kill(os.getpid(), signal.SIGTERM)
            return original(path, value, **kwargs)

        with mock.patch("solver_process_budget.atomic_json", side_effect=interrupted_publication):
            report = self.run_worker("normal")
        self.assertTrue(injected)
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "supervisor-interrupted")
        self.assertNotIn("gripperWorkSummary", report)
        self.assertNotIn("sewingWorkSummary", report)

    def test_proc_failure_still_kills_and_reaps_worker(self):
        with mock.patch("solver_process_budget.group_members", side_effect=OSError("synthetic proc failure")):
            report = supervise([PYTHON, str(HERE), "--worker", str(self.directory), "sleep"],
                               self.directory, self.initial, cpu_limit_seconds=30, wall_limit_seconds=8)
        self.assert_uncompleted(report)
        self.assertFalse(report["supervision"]["cleanupComplete"])
        self.assertIn(report["supervision"]["workerSignal"], (signal.SIGTERM, signal.SIGKILL))
        self.assertEqual(group_members(report["supervision"]["processGroupId"]), [])

    def test_subreaper_restore_failure_still_publishes_failure(self):
        import solver_process_budget as budget
        original = budget._prctl
        calls = 0

        def failed_restore(option, value):
            nonlocal calls
            if option == 36:
                calls += 1
                if calls == 2:
                    original(option, value)
                    raise OSError("synthetic restore failure")
            return original(option, value)

        with mock.patch("solver_process_budget._prctl", side_effect=failed_restore):
            report = self.run_worker("normal")
        self.assert_uncompleted(report)
        self.assertEqual(report["supervision"]["reason"], "supervisor-cleanup-failure")
        self.assertEqual(report["supervision"]["error"]["message"], "synthetic restore failure")

    def test_progress_never_grants_terminal_completion(self):
        progress = ProgressStore(self.directory)
        progress.save(dict(self.initial, terminal=True, completed=True), "worker-finished", exit_code=0)
        payload, _, errors = recover_progress(self.directory)
        self.assertEqual(errors, [])
        self.assertTrue(payload["workerFinished"])
        self.assertFalse(payload["report"]["terminal"])
        self.assertFalse(payload["report"]["completed"])
        self.assertFalse(payload["report"]["accepted"])
        with self.assertRaises(ValueError):
            progress.save(self.initial, "invalid-exit", exit_code=True)
        self.assertEqual(progress.sequence, 2)

    def test_handlers_restored_after_normal_run(self):
        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        before = [signal.getsignal(signum) for signum in signals]
        self.run_worker("normal")
        self.assertEqual(before, [signal.getsignal(signum) for signum in signals])

    def test_unrelated_child_is_neither_killed_nor_reaped(self):
        unrelated = subprocess.Popen([PYTHON, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            self.run_worker("normal")
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_invalid_budgets_rejected_without_spawn(self):
        for wall in (0, -1, float("inf"), float("nan"), 7201):
            with self.subTest(wall=wall), mock.patch("solver_process_budget.subprocess.Popen") as launch:
                with self.assertRaises(ValueError):
                    supervise([], self.directory, self.initial, cpu_limit_seconds=1, wall_limit_seconds=wall)
                launch.assert_not_called()

    def test_atomic_publication_failure_preserves_old_pointer(self):
        before = (self.directory / "progress.json").read_bytes()
        report = dict(self.initial, acceptedStateArtifacts=[state(self.directory, "accepted-0001.json")])
        with mock.patch("solver_process_budget.os.replace", side_effect=OSError("synthetic interruption")):
            with self.assertRaises(OSError):
                ProgressStore(self.directory).save(report, "accepted-one")
        self.assertEqual((self.directory / "progress.json").read_bytes(), before)
        recovered, descriptor, errors = recover_progress(self.directory)
        self.assertEqual(recovered["phase"], "accepted-one")
        self.assertEqual(errors, [])
        self.assertEqual(descriptor["sha256"], sha256(read_regular(self.directory / descriptor["path"])))
        self.assertEqual(list(self.directory.glob(".*.tmp")), [])

    def test_partial_latest_checkpoint_falls_back_with_error(self):
        (self.directory / "progress-000002.json").write_bytes(b'{"payload":')
        payload, _, errors = recover_progress(self.directory)
        self.assertEqual(payload["sequence"], 1)
        self.assertEqual(len(errors), 1)

    def test_checkpoint_hash_mismatch_falls_back(self):
        ProgressStore(self.directory).save(dict(self.initial, note="before"), "second")
        path = self.directory / "progress-000002.json"
        envelope = json.loads(path.read_bytes())
        envelope["payload"]["report"]["note"] = "after"
        os.chmod(path, 0o600)
        path.write_bytes(json_bytes(envelope))
        payload, _, errors = recover_progress(self.directory)
        self.assertEqual(payload["sequence"], 1)
        self.assertTrue(errors)

    def test_atomic_evidence_never_overwrites_or_follows_symlink(self):
        path = self.directory / "immutable.json"
        atomic_bytes(path, b"original")
        with self.assertRaises(FileExistsError):
            atomic_bytes(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"original")
        link = self.directory / "link.json"
        link.symlink_to(path)
        with self.assertRaises(OSError):
            read_regular(link)
        with self.assertRaises(FileExistsError):
            atomic_bytes(link, b"replacement")
        self.assertEqual(path.read_bytes(), b"original")
        self.assertEqual(path.stat().st_mode & 0o777, 0o400)

    def test_true_acceptance_and_invalid_ledger_rejected(self):
        with self.assertRaises(ValueError):
            ProgressStore(self.directory).save(dict(self.initial, accepted=True), "bad")
        descriptor = state(self.directory, "accepted-0002.json")
        ProgressStore(self.directory).save(dict(self.initial, acceptedStateArtifacts=[descriptor]), "bad-ledger")
        payload, _, errors = recover_progress(self.directory)
        self.assertEqual(payload["sequence"], 1)
        self.assertTrue(errors)

    def test_empty_or_all_corrupt_checkpoints_fail(self):
        path = self.directory / "progress-000001.json"
        path.unlink()
        with self.assertRaises(ValueError):
            recover_progress(self.directory)
        path.write_bytes(b"{}")
        with self.assertRaises(ValueError):
            recover_progress(self.directory)

    def test_strict_json_recovery_rejects_duplicate_keys_and_nonfinite_numbers(self):
        ProgressStore(self.directory).save(self.initial, "newer")
        newest = self.directory / "progress-000002.json"
        original = newest.read_bytes()
        for malformed in (original.replace(b'"sequence":2', b'"sequence":2,"sequence":2'),
                          b'{"payload":NaN}', b'{"payload":Infinity}', b'{"payload":1e999}'):
            with self.subTest(malformed=malformed[:40]):
                newest.chmod(0o600)
                newest.write_bytes(malformed)
                payload, descriptor, errors = recover_progress(self.directory, expected_report=self.initial)
                self.assertEqual(descriptor["path"], "progress-000001.json")
                self.assertEqual(len(errors), 1)
                self.assertFalse(payload["report"]["completed"])

    def test_cli_wall_bounds_default_and_existing_output(self):
        command = [PYTHON, str(ROOT / "scripts/spike-contact-continuation.py"),
                   "--canonical", str(self.directory / "canonical.json"),
                   "--placement", str(self.directory / "placement.json"),
                   "--output", str(self.directory / "new-output"),
                   "--activation-distance-m", ".002", "--minimum-distance-m", ".00025",
                   "--pressure-pa", "100", "--target-fraction", ".98"]
        for wall in ("0", "7201", "nan"):
            result = subprocess.run(command + ["--wall-limit-seconds", wall], capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((self.directory / "new-output").exists())
        command[command.index("--output") + 1] = str(self.directory)
        before = (self.directory / "canonical.json").read_bytes()
        result = subprocess.run(command, capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"FileExistsError", result.stderr)
        self.assertEqual((self.directory / "canonical.json").read_bytes(), before)
        result = subprocess.run([PYTHON, command[1], "--help"], capture_output=True, timeout=5)
        self.assertIn(b"default: 480", result.stdout)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(fixture_worker(Path(sys.argv[2]), sys.argv[3]))
    if len(sys.argv) > 1 and sys.argv[1] == "--supervisor":
        raise SystemExit(fixture_supervisor(Path(sys.argv[2]), sys.argv[3]))
    unittest.main()
