"""Real Linux process/signal witnesses; scripted mechanics, no garment motion."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import solver_temporal_supervision as supervisor
from solver_temporal_transport import ExportPolicy, PrivateDirectory


RUN_ID = "b"*64
EXPORT = ExportPolicy(2**20, 2**16, 2**16)
POLICY = supervisor.ProcessPolicy(8, 10, 12., .3, 2**30, 1., .01, EXPORT)


def fixture(mode, directory):
    """Trusted child fixtures are deliberately hostile to orderly shutdown."""
    if mode == "limits":
        print(json.dumps({"cpu": resource.getrlimit(resource.RLIMIT_CPU),
                          "memory": resource.getrlimit(resource.RLIMIT_AS),
                          "file": resource.getrlimit(resource.RLIMIT_FSIZE),
                          "core": resource.getrlimit(resource.RLIMIT_CORE)}), flush=True)
        return 0
    if mode == "descendant":
        reader, writer = os.pipe()
        pid = os.fork()
        if not pid:
            os.close(reader)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print("DESCENDANT_READY", flush=True)
            os.write(writer, b"1"); os.close(writer)
            time.sleep(20.)
            os._exit(0)
        os.close(writer)
        if os.read(reader, 1) != b"1": raise RuntimeError("Descendant did not become ready")
        os.close(reader)
        return 0
    if mode == "flood":
        while True: os.write(1, b"x"*65536)
    if mode == "parent-signal":
        os.kill(os.getppid(), signal.SIGINT)
        time.sleep(20.)
        return 0
    if mode in ("forged", "malformed"):
        child = Path(directory)/"worker"
        child.mkdir(mode=0o700)
        (child/"snapshot.json").write_bytes(
            b'{"accepted":true,"complete":true,"runIdentity":"stale"}\n' if mode == "forged" else b"{broken")
        return 7 if mode == "forged" else 0

    # Imports count against the already installed child CPU/memory limits.
    import solver_temporal_process as worker
    from solver_temporal_execution import TemporalExecution
    from test_solver_temporal_control import Trials, execute
    handle, trials = TemporalExecution("process-stop"), Trials()
    before_mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
    before_handlers = {number: signal.getsignal(number) for number in worker.MANAGED}

    def operation(execution):
        def evaluate(*args):
            value = trials(*args)
            if len(trials.calls) == 4:
                if mode in ("term", "int", "repeat-export"):
                    parent = os.getpid()
                    pid = os.fork()
                    if not pid:
                        os.kill(parent, signal.SIGTERM if mode == "term" else signal.SIGINT)
                        os._exit(0)
                    os.waitpid(pid, 0)
                elif mode == "cpu":
                    print("FOURTH_TRIAL_READY", flush=True)
                    while True: pass
                elif mode in ("native", "native-cpu"):
                    print("NATIVE_FOURTH_TRIAL_READY", flush=True)
                    hashlib.pbkdf2_hmac("sha256", b"fixture", b"salt", 2_000_000_000)
                elif mode == "kill":
                    os.kill(os.getpid(), signal.SIGKILL)
            return value
        result = execute(evaluate, initial_subdivisions=2, execution=execution,
                         **({"max_attempts": 3} if mode == "partial" else {}))
        if mode == "full-error": raise RuntimeError("optional enrichment failed after full commit")
        return result

    publish = PrivateDirectory.publish
    def with_late_signals(store, name, value):
        if name == "snapshot.json" and mode in ("repeat-export", "late-first"):
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGXCPU)
        return publish(store, name, value)
    with patch.object(PrivateDirectory, "publish", with_late_signals):
        result = worker.run_temporal_worker(operation, directory, handle,
            run_identity=RUN_ID, export_policy=EXPORT)
    restored = (before_mask == signal.pthread_sigmask(signal.SIG_BLOCK, ()) and
                all(signal.getsignal(number) == value for number, value in before_handlers.items()))
    print(json.dumps({"trialsEntered": len(trials.calls), "restored": restored,
                      "proposedExitCode": result["proposedExitCode"]}), flush=True)
    return result["proposedExitCode"]


@unittest.skipUnless(sys.platform == "linux", "Linux process controls required")
class TemporalSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.count = 0

    def launch(self, mode, *, policy=POLICY, command=None):
        self.count += 1
        directory = self.root/str(self.count)
        argv = command or [sys.executable, str(Path(__file__).resolve()), "--fixture", mode, str(directory)]
        before_mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
        handlers = {number: signal.getsignal(number) for number in supervisor._PARENT_SIGNALS}
        result = supervisor.supervise_temporal(argv, directory, run_identity=RUN_ID, policy=policy)
        self.assertFalse(supervisor._PARENT_OWNER)
        self.assertEqual(before_mask, signal.pthread_sigmask(signal.SIG_BLOCK, ()))
        self.assertTrue(all(signal.getsignal(number) == old for number, old in handlers.items()))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["workerReturnCode"], os.waitstatus_to_exitcode(result["workerRawWaitStatus"]))
        self.assertTrue(result["cleanupComplete"], result)
        self.assertFalse(result["remainingGroupMembers"], result)
        self.assertEqual(result["snapshotValidation"],
            "deferred; hashes alone do not admit structure, provenance, numerical completion or path correctness")
        for artifact in result["artifacts"]:
            raw = (directory/artifact["path"]).read_bytes()
            self.assertEqual(artifact["bytes"], len(raw))
            self.assertEqual(artifact["sha256"], hashlib.sha256(raw).hexdigest())
        return result, directory

    def raw(self, directory):
        return json.loads((directory/"worker/snapshot.json").read_text())["snapshot"]

    def metadata(self, directory):
        return json.loads((directory/"worker/outcome.json").read_text())

    def stdout_summary(self, directory):
        return json.loads((directory/"worker.log").read_text().strip().splitlines()[-1])

    def test_normal_mechanics_and_snapshot_match_unwrapped_control(self):
        from solver_temporal_execution import TemporalExecution
        from test_solver_temporal_control import execute
        baseline = TemporalExecution("process-stop")
        execute(initial_subdivisions=2, execution=baseline)
        result, directory = self.launch("normal")
        self.assertEqual(result["workerReturnCode"], 0)
        self.assertTrue(result["processExitSucceeded"], result)
        self.assertEqual(self.raw(directory), baseline.snapshot())
        self.assertEqual(self.stdout_summary(directory),
            {"trialsEntered": 6, "restored": True, "proposedExitCode": 0})
        self.assertTrue(result["byteObservationWithoutErrors"])

    def test_actual_external_term_and_int_preserve_whole_pair_and_charge(self):
        for mode, number in (("term", signal.SIGTERM), ("int", signal.SIGINT),
                             ("repeat-export", signal.SIGINT)):
            with self.subTest(mode=mode):
                result, directory = self.launch(mode)
                raw, metadata = self.raw(directory), self.metadata(directory)
                self.assertEqual(result["workerReturnCode"], 86)
                self.assertFalse(result["processExitSucceeded"])
                self.assertEqual(metadata["firstSignal"]["number"], number)
                self.assertEqual(metadata["firstSignal"]["observedPhase"], "numerical")
                self.assertEqual(raw["completedFraction"], .5)
                self.assertFalse(raw["complete"])
                self.assertEqual(raw["resources"]["reservedTrials"], 4)
                self.assertEqual(raw["resources"]["chargedEvaluationAllowance"], 32)
                self.assertEqual(len(raw["acceptedSteps"]), 2)
                self.assertEqual(self.stdout_summary(directory),
                    {"trialsEntered": 4, "restored": True, "proposedExitCode": 86})

    def test_real_cpu_soft_limit_exports_a_committed_prefix(self):
        result, directory = self.launch("cpu", policy=replace(POLICY, cpu_soft_seconds=2, cpu_hard_seconds=4))
        self.assertEqual(result["workerReturnCode"], 86, (result, (directory/"worker.log").read_text()))
        self.assertEqual(self.metadata(directory)["firstSignal"]["number"], signal.SIGXCPU)
        self.assertEqual(self.raw(directory)["completedFraction"], .5)
        self.assertEqual(self.stdout_summary(directory)["trialsEntered"], 4)
        self.assertLess(result["elapsedSeconds"], 10.)

    def test_incomplete_return_and_failed_full_prefix_have_distinct_numerical_truth(self):
        for mode, fraction, complete in (("partial", .5, False), ("full-error", 1., True),
                                         ("late-first", 1., True)):
            with self.subTest(mode=mode):
                result, directory = self.launch(mode)
                self.assertEqual(result["workerReturnCode"], 86 if mode == "late-first" else 1)
                raw = self.raw(directory)
                self.assertEqual(raw["completedFraction"], fraction)
                self.assertIs(raw["complete"], complete)
                if mode == "late-first":
                    self.assertEqual(self.metadata(directory)["firstSignal"]["observedPhase"], "exporting")
                self.assertFalse(result["processExitSucceeded"])

    def test_native_blocker_wall_hard_kill_has_actual_signal_and_no_snapshot(self):
        result, directory = self.launch("native", policy=replace(POLICY, wall_soft_seconds=2., wall_grace_seconds=.2))
        self.assertIn("NATIVE_FOURTH_TRIAL_READY", (directory/"worker.log").read_text())
        self.assertEqual(result["workerReturnCode"], -signal.SIGKILL)
        self.assertEqual(result["workerSignal"], signal.SIGKILL)
        self.assertEqual(result["reason"], "wall-soft-limit")
        self.assertEqual([row["signal"] for row in result["signalsSent"]][:2], [signal.SIGTERM, signal.SIGKILL])
        self.assertFalse((directory/"worker/snapshot.json").exists())
        self.assertLess(result["elapsedSeconds"], 5.)

    def test_native_blocker_cpu_hard_kill_cannot_promise_export(self):
        result, directory = self.launch("native-cpu", policy=replace(POLICY, cpu_soft_seconds=1, cpu_hard_seconds=2))
        self.assertIn("NATIVE_FOURTH_TRIAL_READY", (directory/"worker.log").read_text())
        self.assertEqual(result["workerReturnCode"], -signal.SIGKILL)
        self.assertFalse((directory/"worker/snapshot.json").exists())
        self.assertLess(result["elapsedSeconds"], 6.)

    def test_direct_sigkill_is_not_reported_as_cpu_or_numerical_success(self):
        result, directory = self.launch("kill")
        self.assertEqual(result["workerReturnCode"], -signal.SIGKILL)
        self.assertFalse(result["signalsSent"])
        self.assertFalse(result["processExitSucceeded"])
        self.assertFalse((directory/"worker/snapshot.json").exists())

    def test_descendant_is_killed_and_reaped_after_zero_leader_exit(self):
        result, directory = self.launch("descendant")
        self.assertIn("DESCENDANT_READY", (directory/"worker.log").read_text())
        self.assertEqual(result["workerReturnCode"], 0)
        self.assertEqual(result["reason"], "worker-left-live-descendants")
        self.assertFalse(result["processExitSucceeded"])
        self.assertEqual(len(result["reapedProcessUsage"]), 2)
        self.assertIn(signal.SIGKILL, [row["signal"] for row in result["signalsSent"]])

    def test_log_flood_is_bounded_and_stops_the_worker(self):
        policy = replace(POLICY, export=replace(EXPORT, log_bytes=4096))
        result, directory = self.launch("flood", policy=policy)
        self.assertEqual((directory/"worker.log").stat().st_size, 4096)
        self.assertEqual(result["logBytesRetained"], 4096)
        self.assertGreater(result["logBytesObservedDiscarded"], 0)
        self.assertEqual(result["reason"], "log-byte-budget-exhausted")
        self.assertFalse(result["processExitSucceeded"])

    def test_child_cannot_certify_actual_exit_or_validate_forged_and_malformed_bytes(self):
        for mode, expected in (("forged", 7), ("malformed", 0)):
            with self.subTest(mode=mode):
                result, directory = self.launch(mode)
                self.assertEqual(result["workerReturnCode"], expected)
                self.assertFalse(result["accepted"])
                self.assertTrue(result["byteObservationWithoutErrors"])
                self.assertEqual(len(result["artifacts"]), 2)
                self.assertTrue(all("deferred" in row["validation"] for row in result["artifacts"]))

    def test_limits_precede_exec_and_one_argument_commands_are_supported(self):
        result, directory = self.launch("limits")
        observed = self.stdout_summary(directory)
        self.assertEqual(observed["cpu"], [8, 10])
        self.assertEqual(observed["memory"], [2**30, 2**30])
        self.assertEqual(observed["file"], [EXPORT.file_size_limit]*2)
        self.assertEqual(observed["core"], [0, 0])
        result, _ = self.launch("unused", command=["/bin/true"])
        self.assertTrue(result["processExitSucceeded"], result)

    def test_actual_parent_signal_and_late_signal_do_not_forge_worker_failure(self):
        result, _ = self.launch("parent-signal")
        self.assertEqual(result["supervisorFirstSignal"], signal.SIGINT)
        self.assertEqual(result["reason"], "supervisor-signal")
        original = os.fsync
        fired = False
        def late(fd):
            nonlocal fired
            if not fired:
                fired = True
                os.kill(os.getpid(), signal.SIGTERM)
            return original(fd)
        with patch.object(supervisor.os, "fsync", late):
            result, _ = self.launch("unused", command=["/bin/true"])
        self.assertEqual(result["workerReturnCode"], 0)
        self.assertEqual(result["supervisorFirstSignal"], signal.SIGTERM)
        self.assertEqual(result["reason"], "supervisor-signal")
        self.assertFalse(result["processExitSucceeded"])

    def test_log_close_failure_preserves_exit_and_restores_parent_ownership(self):
        original = os.close
        fired = False
        def close(fd):
            nonlocal fired
            target = os.readlink(f"/proc/self/fd/{fd}")
            original(fd)
            if target.endswith("/worker.log") and not fired:
                fired = True
                raise OSError("injected post-close failure")
        with patch.object(supervisor.os, "close", close):
            result, _ = self.launch("unused", command=["/bin/true"])
        self.assertTrue(fired)
        self.assertEqual(result["workerReturnCode"], 0)
        self.assertIn("log-close", [row["stage"] for row in result["supervisionErrors"]])
        self.assertFalse(result["processExitSucceeded"])

    def test_empty_proc_sample_still_terminates_a_live_owned_leader(self):
        with patch.object(supervisor, "group_members", return_value=[]):
            result, _ = self.launch("unused", command=["/bin/sleep", "20"],
                policy=replace(POLICY, wall_soft_seconds=.15, wall_grace_seconds=.1))
        self.assertEqual(result["workerReturnCode"], -signal.SIGTERM)
        self.assertLess(result["elapsedSeconds"], 2.)

    def test_reap_drain_consults_absolute_deadline_between_children(self):
        original_wait, original_members = os.wait4, supervisor.group_members
        count = 0
        leader_reaped = False
        def wait4(pid, flags):
            nonlocal count, leader_reaped
            if not leader_reaped:
                result = original_wait(pid, flags)
                if result[0]: leader_reaped = True
                return result
            # Simulate a continuously available descendant stream without
            # spawning thousands of processes; the real leader was reaped.
            count += 1
            time.sleep(.012)
            return 10_000_000+count, 0, resource.getrusage(resource.RUSAGE_CHILDREN)
        def members(pid):
            return [{"pid": 10_000_000, "state": "Z", "cpuSeconds": 0.}] if leader_reaped else original_members(pid)
        with patch.object(supervisor.os, "wait4", wait4), patch.object(supervisor, "group_members", members):
            result = supervisor.supervise_temporal(["/bin/true"], self.root/"reap",
                run_identity=RUN_ID, policy=replace(POLICY, cleanup_seconds=.05))
        self.assertTrue(leader_reaped)
        self.assertGreater(count, 0)
        self.assertLess(count, 12)
        self.assertFalse(result["cleanupComplete"])
        self.assertEqual(result["workerReturnCode"], 0)

    def test_empty_proc_sample_waits_for_owned_leader_after_hard_kill(self):
        original = os.wait4
        calls = 0
        def delayed(pid, flags):
            nonlocal calls
            calls += 1
            if calls <= 3: return 0, 0, None
            return original(pid, flags)
        command = [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)"]
        with patch.object(supervisor, "group_members", return_value=[]), patch.object(supervisor.os, "wait4", delayed):
            result, _ = self.launch("unused", command=command,
                policy=replace(POLICY, wall_soft_seconds=.4, wall_grace_seconds=.1))
        self.assertEqual(result["workerReturnCode"], -signal.SIGKILL)
        self.assertGreaterEqual(calls, 4)
        self.assertLess(result["elapsedSeconds"], 2.)

    def test_parent_initial_mask_failure_does_not_block_or_spawn(self):
        original = signal.pthread_sigmask
        before = original(signal.SIG_BLOCK, ())
        with patch.object(supervisor.signal, "pthread_sigmask", side_effect=OSError("mask query")) as mask, \
             patch.object(supervisor.subprocess, "Popen") as spawn:
            result = supervisor.supervise_temporal(["/bin/true"], self.root/"mask",
                                                  run_identity=RUN_ID, policy=POLICY)
        self.assertEqual(mask.call_count, 1)
        spawn.assert_not_called()
        self.assertFalse(supervisor._PARENT_OWNER)
        self.assertEqual(original(signal.SIG_BLOCK, ()), before)
        self.assertIsNone(result["workerRawWaitStatus"])
        self.assertFalse(result["processExitSucceeded"])

    def test_parent_partial_handler_installation_restores_actual_handlers(self):
        original = signal.signal
        before = {number: signal.getsignal(number) for number in supervisor._PARENT_SIGNALS}
        fired = False
        def install(number, handler):
            nonlocal fired
            old = original(number, handler)
            if number == signal.SIGINT and not fired:
                fired = True
                raise OSError("installation failed after change")
            return old
        with patch.object(supervisor.signal, "signal", install), patch.object(supervisor.subprocess, "Popen") as spawn:
            result = supervisor.supervise_temporal(["/bin/true"], self.root/"install",
                                                  run_identity=RUN_ID, policy=POLICY)
        spawn.assert_not_called()
        self.assertTrue(fired)
        self.assertTrue(all(signal.getsignal(number) == old for number, old in before.items()))
        self.assertFalse(supervisor._PARENT_OWNER)
        self.assertFalse(result["processExitSucceeded"])

    def test_parent_subreaper_restoration_failure_retains_actual_exit(self):
        original = supervisor._prctl
        changes = 0
        def prctl(option, value):
            nonlocal changes
            result = original(option, value)
            if option == 36:
                changes += 1
                if changes == 2: raise OSError("injected after actual subreaper restoration")
            return result
        with patch.object(supervisor, "_prctl", prctl):
            result, _ = self.launch("unused", command=["/bin/true"])
        self.assertEqual(result["workerReturnCode"], 0)
        self.assertFalse(result["processExitSucceeded"])
        self.assertIn("subreaper-restoration", [row["stage"] for row in result["supervisionErrors"]])

    def test_existing_job_directory_rejects_before_spawn(self):
        destination = self.root/"existing"; destination.mkdir()
        with patch.object(supervisor.subprocess, "Popen") as spawn, self.assertRaises(FileExistsError):
            supervisor.supervise_temporal(["/bin/true"], destination, run_identity=RUN_ID, policy=POLICY)
        spawn.assert_not_called()
        self.assertFalse(supervisor._PARENT_OWNER)


class TemporalJobObservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/"job"
        self.store = PrivateDirectory.create(self.root, EXPORT)
        self.addCleanup(self.store.close)
        (self.root/"worker").mkdir(mode=0o700)
        (self.root/"worker/snapshot.json").write_bytes(b"{}\n")

    def observe(self, policy=EXPORT):
        return supervisor.observe_job(self.store, policy)

    def test_temporary_bytes_are_bounded_and_never_a_complete_snapshot(self):
        name = ".snapshot.json."+"a"*16+".tmp"
        (self.root/"worker"/name).write_bytes(b"{partial")
        artifacts, errors, total = self.observe()
        self.assertFalse(errors)
        self.assertEqual(total, 11)
        self.assertEqual(sum(row["temporary"] for row in artifacts), 1)
        self.assertTrue(all("deferred" in row["validation"] for row in artifacts))
        artifacts, errors, _ = self.observe(replace(EXPORT, snapshot_bytes=3))
        self.assertTrue(errors)
        self.assertEqual([row["path"] for row in artifacts], ["worker/snapshot.json"])

    def test_unknown_names_excess_temporaries_and_root_files_reject_without_hiding_final(self):
        for names in (("arbitrary",), (".snapshot.json.bad.tmp",),
                      tuple(".snapshot.json."+letter*16+".tmp" for letter in "ab"),
                      tuple("extra"+str(i) for i in range(5))):
            with self.subTest(names=names):
                for name in names: (self.root/"worker"/name).write_bytes(b"x")
                artifacts, errors, _ = self.observe()
                self.assertTrue(errors)
                self.assertIn("worker/snapshot.json", [row["path"] for row in artifacts])
                for name in names: (self.root/"worker"/name).unlink()
        (self.root/"unexpected").write_bytes(b"x")
        self.assertTrue(self.observe()[1])

    def test_insecure_child_directory_and_symlink_temporary_are_rejected(self):
        os.chmod(self.root/"worker", 0o755)
        self.assertTrue(self.observe()[1])
        os.chmod(self.root/"worker", 0o700)
        name = ".snapshot.json."+"a"*16+".tmp"
        (self.root/"worker"/name).symlink_to("snapshot.json")
        artifacts, errors, _ = self.observe()
        self.assertTrue(errors)
        self.assertEqual([row["path"] for row in artifacts], ["worker/snapshot.json"])

    def test_per_artifact_limits_account_for_hardlink_names_conservatively(self):
        for name in ("snapshot.json", "outcome.json"):
            path = self.root/"worker"/name
            if name == "outcome.json": path.write_bytes(b"{}\n")
            os.link(path, path.parent/("."+name+"."+"c"*16+".tmp"))
        (self.root/"worker.log").write_bytes(b"log")
        policy = ExportPolicy(3, 3, 3)
        artifacts, errors, total = self.observe(policy)
        self.assertFalse(errors)
        self.assertEqual(len(artifacts), 5)
        self.assertEqual(total, policy.maximum_named_bytes)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--fixture":
        raise SystemExit(fixture(sys.argv[2], sys.argv[3]))
    unittest.main()
