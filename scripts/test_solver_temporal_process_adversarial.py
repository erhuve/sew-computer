"""Adversarial signal/publication boundaries with scripted temporal mechanics.

Signal calls are an explicit deterministic fake: this suite does not send OS
signals. The separate subprocess suite owns actual Linux delivery/supervision.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import solver_temporal_process as worker
import solver_temporal_transport as transport
from solver_temporal_execution import TemporalExecution, TemporalStopRequested
from test_solver_temporal_control import Trials, execute


RUN_ID = "a"*64
POLICY = transport.ExportPolicy(2**20, 2**16, 4096)


class SignalBackend:
    """Track exact mask/handler changes without altering process signal state."""
    def __init__(self):
        self.mask = {signal.SIGUSR1}
        self.initial_mask = self.mask.copy()
        self.handlers = {number: object() for number in worker.MANAGED}
        self.initial_handlers = self.handlers.copy()
        self.fail_first_query = False
        self.fail_install = None
        self.fail_restore = None
        self.installation_error_seen = False
        self.restoration_error_seen = False
        self.on_unmask = None

    def pthread_sigmask(self, how, values):
        values = set(values)
        if self.fail_first_query and how == signal.SIG_BLOCK and not values:
            self.fail_first_query = False
            raise OSError("initial mask query failed")
        old = self.mask.copy()
        if how == signal.SIG_BLOCK:
            self.mask |= values
        elif how == signal.SIG_SETMASK:
            self.mask = values
        else:
            raise AssertionError("Unexpected mask operation")
        hook = self.on_unmask
        if hook is not None and how == signal.SIG_SETMASK:
            self.on_unmask = None
            hook()
        return old

    def getsignal(self, number):
        return self.handlers[number]

    def signal(self, number, value):
        old = self.handlers[number]
        if (number == self.fail_install and value is not self.initial_handlers[number]
                and not self.installation_error_seen):
            self.installation_error_seen = True
            # Exercise an API failing after it changed the handler.
            self.handlers[number] = value
            raise OSError("partial handler installation failed")
        if (number == self.fail_restore and value is self.initial_handlers[number]
                and not self.restoration_error_seen):
            self.restoration_error_seen = True
            raise OSError("handler restoration failed")
        self.handlers[number] = value
        return old

    def emit(self, number):
        return self.handlers[number](number, None)

    @contextmanager
    def installed(self):
        with patch.object(worker.sys, "platform", "linux"), \
             patch.object(worker.signal, "pthread_sigmask", self.pthread_sigmask), \
             patch.object(worker.signal, "getsignal", self.getsignal), \
             patch.object(worker.signal, "signal", self.signal):
            yield self


class TemporalWorkerAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.signals = SignalBackend()
        self.execution = TemporalExecution("managed-process-stop")

    def run_worker(self, operation, **kwargs):
        with self.signals.installed():
            result = worker.run_temporal_worker(operation, self.directory, self.execution,
                run_identity=RUN_ID, export_policy=kwargs.get("policy", POLICY))
        self.assertIsNone(worker._OWNER)
        return result

    def raw(self):
        return json.loads((self.directory/"worker/snapshot.json").read_text())["snapshot"]

    def metadata(self):
        return json.loads((self.directory/"worker/outcome.json").read_text())

    def assert_restored(self):
        self.assertEqual(self.signals.mask, self.signals.initial_mask)
        for number in worker.MANAGED:
            self.assertIs(self.signals.handlers[number], self.signals.initial_handlers[number])

    def test_full_scripted_return_exports_raw_before_metadata_and_restores(self):
        trials = Trials()
        result = self.run_worker(lambda h: execute(trials, initial_subdivisions=2, execution=h))
        self.assertEqual(result["proposedExitCode"], 0)
        self.assertFalse(result["actualExitObserved"])
        self.assertTrue(self.raw()["complete"])
        self.assertEqual(self.raw()["resources"]["reservedTrials"], 6)
        self.assertEqual(len(trials.calls), 6)
        self.assertEqual(self.metadata()["rawSnapshot"], result["snapshotArtifact"])
        self.assert_restored()

    def test_success_without_handle_admission_is_not_completion(self):
        calls = []
        result = self.run_worker(lambda h: calls.append(h))
        self.assertEqual(calls, [self.execution])
        self.assertEqual(result["proposedExitCode"], 1)
        self.assertFalse(self.raw()["available"])
        self.assert_restored()

    def test_pre_requested_handle_prevents_operation_and_does_not_invent_prefix(self):
        self.execution.request_stop()
        calls = []
        result = self.run_worker(lambda h: calls.append(h))
        self.assertFalse(calls)
        self.assertNotEqual(result["proposedExitCode"], 0)
        self.assertFalse(self.raw()["available"])
        self.assertTrue(self.raw()["stopRequested"])
        self.assert_restored()

    def test_initial_mask_query_failure_does_not_introduce_a_new_blocked_mask(self):
        self.signals.fail_first_query = True
        calls = []
        result = self.run_worker(lambda h: calls.append(h))
        self.assertFalse(calls)
        self.assertEqual(result["proposedExitCode"], 1)
        self.assertIn("initial mask query failed", result["escapingOperationOrInstallationError"]["message"])
        self.assert_restored()

    def test_partial_installation_restores_every_attempted_handler_and_original_mask(self):
        self.signals.fail_install = signal.SIGINT
        calls = []
        result = self.run_worker(lambda h: calls.append(h))
        self.assertFalse(calls)
        self.assertTrue(self.signals.installation_error_seen)
        self.assertEqual(result["proposedExitCode"], 1)
        self.assert_restored()

    def test_signal_on_installation_unmask_prevents_operation(self):
        self.signals.on_unmask = lambda: self.signals.emit(signal.SIGTERM)
        calls = []
        result = self.run_worker(lambda h: calls.append(h))
        self.assertFalse(calls)
        self.assertEqual(result["firstSignal"]["observedPhase"], "installing")
        self.assertNotEqual(result["proposedExitCode"], 0)
        self.assertFalse(self.raw()["available"])
        self.assert_restored()

    def test_first_active_exception_and_escaping_cleanup_error_are_distinct(self):
        trials = Trials()
        def operation(handle):
            execute(trials, initial_subdivisions=2, execution=handle)
            try:
                raise LookupError("original active failure")
            except LookupError:
                try:
                    self.signals.emit(signal.SIGTERM)
                finally:
                    self.signals.emit(signal.SIGINT)
                    raise ValueError("escaping cleanup replacement")
        result = self.run_worker(operation)
        first = result["firstSignal"]
        self.assertEqual(first["number"], signal.SIGTERM)
        self.assertEqual(first["activeExceptionAtFirstSignal"]["type"], "LookupError")
        self.assertEqual(first["activeExceptionAtFirstSignal"]["message"], "original active failure")
        self.assertEqual(result["escapingOperationOrInstallationError"]["type"], "ValueError")
        self.assertEqual(result["proposedExitCode"], 1)
        self.assertTrue(self.raw()["complete"])
        self.assertEqual(len(trials.calls), 6)
        self.assert_restored()

    def test_signal_during_fourth_trial_preserves_pair_and_stops_further_dispatch(self):
        for replacement in (None, ValueError):
            with self.subTest(replacement=replacement):
                # Each invocation needs its own unused handle and directory.
                parent = self.directory/("plain" if replacement is None else "cleanup")
                parent.mkdir(mode=0o700)
                handle, trials = TemporalExecution("managed-process-stop"), Trials()
                def evaluate(*args):
                    result = trials(*args)
                    if len(trials.calls) == 4:
                        try:
                            self.signals.emit(signal.SIGTERM)
                        finally:
                            self.signals.emit(signal.SIGINT)
                            if replacement is not None:
                                raise replacement("trial cleanup replacement")
                    return result
                with self.signals.installed():
                    result = worker.run_temporal_worker(
                        lambda h: execute(evaluate, initial_subdivisions=2, execution=h),
                        parent, handle, run_identity=RUN_ID, export_policy=POLICY)
                raw = json.loads((parent/"worker/snapshot.json").read_text())["snapshot"]
                self.assertEqual(len(trials.calls), 4)
                self.assertEqual(raw["completedFraction"], .5)
                self.assertFalse(raw["complete"])
                self.assertEqual(raw["resources"]["reservedTrials"], 4)
                self.assertEqual(raw["resources"]["chargedEvaluationAllowance"], 32)
                self.assertEqual(len(raw["acceptedSteps"]), 2)
                self.assertEqual(result["firstSignal"]["number"], signal.SIGTERM)
                self.assertNotEqual(result["proposedExitCode"], 0)
                self.assertIsNone(worker._OWNER)
                self.assert_restored()

    def test_full_committed_prefix_survives_dedicated_signal_stop(self):
        trials = Trials()
        def operation(handle):
            execute(trials, execution=handle)
            self.signals.emit(signal.SIGXCPU)
        result = self.run_worker(operation)
        self.assertEqual(result["proposedExitCode"], 86)
        self.assertTrue(self.raw()["complete"])
        self.assertEqual(self.raw()["completedFraction"], 1.)
        self.assertTrue(self.raw()["stopRequested"])
        self.assertEqual(len(trials.calls), 3)
        self.assert_restored()

    def test_late_repeated_signals_latch_without_aborting_snapshot_export(self):
        publish = transport.PrivateDirectory.publish
        def late(store, name, value):
            if name == "snapshot.json":
                self.signals.emit(signal.SIGTERM)
                self.signals.emit(signal.SIGINT)
            return publish(store, name, value)
        with patch.object(transport.PrivateDirectory, "publish", late):
            result = self.run_worker(lambda h: execute(execution=h))
        self.assertEqual(result["proposedExitCode"], 86)
        self.assertEqual(result["firstSignal"]["number"], signal.SIGTERM)
        self.assertEqual(result["firstSignal"]["observedPhase"], "exporting")
        self.assertTrue(self.raw()["complete"])
        self.assertTrue(result["stopRequested"])
        self.assertTrue(self.metadata()["stopRequested"])
        # Snapshot acquisition precedes this injected signal; no mutation of its
        # already detached as-of state is needed to preserve the process stop.
        self.assert_restored()

    def test_signal_after_metadata_capture_does_not_rewrite_published_history(self):
        publish = transport.PrivateDirectory.publish
        def late(store, name, value):
            result = publish(store, name, value)
            if name == "outcome.json":
                self.signals.emit(signal.SIGTERM)
            return result
        with patch.object(transport.PrivateDirectory, "publish", late):
            result = self.run_worker(lambda h: execute(execution=h))
        self.assertEqual(result["proposedExitCode"], 86)
        self.assertEqual(self.metadata()["proposedExitCode"], 0)
        self.assertIsNone(self.metadata()["firstSignal"])
        self.assertTrue(self.raw()["complete"])
        self.assert_restored()

    def test_restoration_failure_keeps_valid_full_snapshot_and_asof_metadata(self):
        self.signals.fail_restore = signal.SIGINT
        result = self.run_worker(lambda h: execute(execution=h))
        self.assertEqual(result["proposedExitCode"], 1)
        self.assertTrue(self.raw()["complete"])
        self.assertEqual(self.metadata()["proposedExitCode"], 0)
        self.assertTrue(any(row["stage"] == "signal-restoration" for row in result["secondaryFailures"]))
        self.assertEqual(self.signals.mask, self.signals.initial_mask)
        for number in set(worker.MANAGED)-{signal.SIGINT}:
            self.assertIs(self.signals.handlers[number], self.signals.initial_handlers[number])

    def test_snapshot_acquisition_failure_still_retains_error_metadata(self):
        with patch.object(TemporalExecution, "snapshot", side_effect=MemoryError("snapshot allocation")):
            result = self.run_worker(lambda h: execute(execution=h))
        self.assertEqual(result["proposedExitCode"], 1)
        self.assertFalse((self.directory/"worker/snapshot.json").exists())
        self.assertIsNone(self.metadata()["rawSnapshot"])
        self.assertTrue(any(row["type"] == "MemoryError" for row in result["secondaryFailures"]))
        self.assert_restored()

    def test_metadata_cap_failure_preserves_complete_snapshot(self):
        result = self.run_worker(lambda h: execute(execution=h),
                                 policy=transport.ExportPolicy(2**20, 1, 4096))
        self.assertEqual(result["proposedExitCode"], 1)
        self.assertTrue(self.raw()["complete"])
        self.assertFalse((self.directory/"worker/outcome.json").exists())
        self.assertTrue(any(row["stage"] == "outcome-publication" for row in result["secondaryFailures"]))
        self.assert_restored()

    def test_broken_exception_formatter_is_bounded_and_does_not_prevent_export(self):
        class BrokenError(Exception):
            def __str__(self): raise MemoryError("formatting itself failed")
        def operation(handle):
            execute(execution=handle)
            raise BrokenError()
        result = self.run_worker(operation)
        self.assertEqual(result["proposedExitCode"], 1)
        self.assertTrue(result["escapingOperationOrInstallationError"]["messageIncomplete"])
        self.assertTrue(self.raw()["complete"])
        self.assert_restored()

    def test_existing_worker_or_symlink_rejects_before_operation(self):
        target = self.directory/"worker"
        for kind in ("directory", "symlink"):
            with self.subTest(kind=kind):
                if kind == "directory": target.mkdir()
                else: target.symlink_to(self.directory, target_is_directory=True)
                calls = []
                with self.signals.installed(), self.assertRaises((OSError, ValueError)):
                    worker.run_temporal_worker(lambda h: calls.append(h), self.directory,
                        self.execution, run_identity=RUN_ID, export_policy=POLICY)
                self.assertFalse(calls)
                self.assertIsNone(worker._OWNER)
                if kind == "directory": target.rmdir()
                else: target.unlink()
                self.assert_restored()


class TemporalTransportAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)

    def store(self, name="evidence", policy=POLICY):
        return transport.PrivateDirectory.create(self.parent/name, policy)

    def test_cap_includes_newline_and_exact_boundary_publishes(self):
        expected = b'{"a":1}\n'
        with self.store(policy=transport.ExportPolicy(len(expected), 1024, 1024)) as store:
            row = store.publish("snapshot.json", {"a": 1})
            self.assertEqual(row["bytes"], len(expected))
            self.assertEqual(row["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual((self.parent/"evidence/snapshot.json").read_bytes(), expected)
        with self.store("short", transport.ExportPolicy(len(expected)-1, 1024, 1024)) as store:
            with self.assertRaises(ValueError): store.publish("snapshot.json", {"a": 1})
            self.assertFalse(store.observations["snapshot.json"]["contentComplete"])
        self.assertFalse((self.parent/"short/snapshot.json").exists())

    def test_short_writes_hash_only_the_bytes_actually_written(self):
        write = os.write
        def short(fd, value): return write(fd, value[:2])
        value = {"text": "hello Ω", "values": list(range(25))}
        expected = (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                    separators=(",", ":"))+"\n").encode()
        with self.store() as store, patch.object(transport.os, "write", short):
            row = store.publish("snapshot.json", value)
        self.assertEqual((self.parent/"evidence/snapshot.json").read_bytes(), expected)
        self.assertEqual(row["bytes"], len(expected))
        self.assertEqual(row["sha256"], hashlib.sha256(expected).hexdigest())

    def test_zero_progress_and_nonfinite_encodings_never_publish_final(self):
        cases = [("zero", {"x": 1}), ("nan", {"x": float("nan")}),
                 ("inf", {"x": float("inf")})]
        for name, value in cases:
            with self.subTest(case=name), self.store(name) as store:
                if name == "zero":
                    with patch.object(transport.os, "write", return_value=0), self.assertRaises(OSError):
                        store.publish("snapshot.json", value)
                else:
                    with self.assertRaises(ValueError): store.publish("snapshot.json", value)
                self.assertFalse(store.observations["snapshot.json"]["contentComplete"])
                self.assertFalse((self.parent/name/"snapshot.json").exists())
                self.assertEqual(list((self.parent/name).iterdir()), [])

    def test_existing_file_and_symlink_are_never_replaced(self):
        for kind in ("file", "symlink"):
            with self.subTest(kind=kind), self.store(kind) as store:
                original = self.parent/"target"
                original.write_bytes(b"existing evidence")
                final = self.parent/kind/"snapshot.json"
                if kind == "file": final.write_bytes(b"existing evidence")
                else: final.symlink_to(original)
                with self.assertRaises(FileExistsError): store.publish("snapshot.json", {"bad": True})
                self.assertEqual(final.read_bytes(), b"existing evidence")
                self.assertEqual(original.read_bytes(), b"existing evidence")
                with self.assertRaises(ValueError): store.publish("snapshot.json", {})

    def test_descriptor_anchors_publication_after_directory_rename(self):
        with self.store() as store:
            (self.parent/"evidence").rename(self.parent/"moved")
            (self.parent/"evidence").mkdir(mode=0o700)
            store.publish("snapshot.json", {"anchored": True})
        self.assertTrue((self.parent/"moved/snapshot.json").is_file())
        self.assertFalse((self.parent/"evidence/snapshot.json").exists())

    def test_file_fsync_failure_prevents_final_publication(self):
        sync = os.fsync
        with self.store() as store:
            def fail_file(fd):
                if fd != store.fd: raise OSError("file sync failure")
                return sync(fd)
            with patch.object(transport.os, "fsync", fail_file), self.assertRaises(OSError):
                store.publish("snapshot.json", {"x": 1})
            self.assertFalse(store.observations["snapshot.json"]["finalLinked"])
        self.assertFalse((self.parent/"evidence/snapshot.json").exists())

    def test_directory_fsync_failure_preserves_final_and_truthful_stage(self):
        sync = os.fsync
        with self.store() as store:
            def fail_directory(fd):
                if fd == store.fd: raise OSError("directory sync failure")
                return sync(fd)
            with patch.object(transport.os, "fsync", fail_directory), self.assertRaises(OSError):
                store.publish("snapshot.json", {"x": 1})
            observation = store.observations["snapshot.json"]
            self.assertTrue(observation["fileSynced"])
            self.assertTrue(observation["finalLinked"])
            self.assertFalse(observation["directorySynced"])
            self.assertEqual(json.loads((self.parent/"evidence/snapshot.json").read_text()), {"x": 1})
            self.assertEqual(store.observe("snapshot.json", 1024)["validation"],
                             "bytes-only-structural-and-numerical-audit-deferred")

    def test_post_link_unlink_failure_does_not_delete_published_file(self):
        unlink = os.unlink
        failures = []
        with self.store() as store:
            def fail_once(path, *args, **kwargs):
                if path.startswith(".snapshot.json.") and not failures:
                    failures.append(path)
                    raise OSError("post-link unlink failure")
                return unlink(path, *args, **kwargs)
            with patch.object(transport.os, "unlink", fail_once), self.assertRaises(OSError):
                store.publish("snapshot.json", {"x": 1})
            self.assertTrue(store.observations["snapshot.json"]["finalLinked"])
            self.assertEqual(json.loads((self.parent/"evidence/snapshot.json").read_text()), {"x": 1})

    def test_observer_rejects_symlink_directory_fifo_and_oversize(self):
        with self.store() as store:
            final = self.parent/"evidence/snapshot.json"
            external = self.parent/"external"
            external.write_bytes(b"{}\n")
            final.symlink_to(external)
            with self.assertRaises((OSError, ValueError)): store.observe("snapshot.json", 16)
            final.unlink(); final.mkdir()
            with self.assertRaises((OSError, ValueError)): store.observe("snapshot.json", 16)
            final.rmdir(); os.mkfifo(final)
            with self.assertRaises((OSError, ValueError)): store.observe("snapshot.json", 16)
            final.unlink(); final.write_bytes(b"12345")
            with self.assertRaises(ValueError): store.observe("snapshot.json", 4)

    def test_observer_detects_same_inode_mutation_during_streamed_hash(self):
        with self.store() as store:
            path = self.parent/"evidence/snapshot.json"
            path.write_bytes(b'{"value":1}\n')
            read = os.read
            changed = []
            def mutate(fd, size):
                value = read(fd, size)
                if value and not changed:
                    changed.append(True)
                    path.write_bytes(b'{"value":1234}\n')
                return value
            with patch.object(transport.os, "read", mutate), self.assertRaises(ValueError):
                store.observe("snapshot.json", 1024)


if __name__ == "__main__":
    unittest.main()
