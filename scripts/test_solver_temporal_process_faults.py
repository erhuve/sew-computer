"""Supplemental publication/restoration faults; separately recorded coverage."""
from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import solver_temporal_process as worker
import solver_temporal_supervision as supervisor
import solver_temporal_transport as transport


RUN_ID = "d"*64
EXPORT = transport.ExportPolicy(2**20, 2**16, 2**16)
POLICY = supervisor.ProcessPolicy(8, 10, 12., .3, 2**30, 1., .01, EXPORT)


def fixture(mode, directory):
    from solver_temporal_execution import TemporalExecution
    from test_solver_temporal_control import execute
    handle = TemporalExecution("supplemental-process-fault")
    prior_calls = []
    if mode == "restored-custom":
        def prior(number, frame):
            prior_calls.append(number)
            raise RuntimeError("restored prior handler raised")
        signal.signal(signal.SIGTERM, prior)
    elif mode == "restored-default":
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    before_mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
    before_handlers = {number: signal.getsignal(number) for number in worker.MANAGED}

    def operation(execution):
        result = execute(initial_subdivisions=2, execution=execution)
        if mode == "kernel-fsize":
            # Deliberately injected stricter fixture limit, prospectively fixed
            # here. This is a fault witness, not a policy change in a study.
            resource.setrlimit(resource.RLIMIT_FSIZE, (128, 128))
        return result

    original_restore = worker._Signals.restore
    def pending_on_restore(scope):
        if mode in ("restored-custom", "restored-default"):
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            os.kill(os.getpid(), signal.SIGTERM)
            if signal.SIGTERM not in signal.sigpending():
                raise AssertionError("Expected blocked pending TERM")
        return original_restore(scope)

    original_publish = transport.PrivateDirectory.publish
    def publish(store, name, value):
        if name == "snapshot.json" and mode == "late-xfsz":
            os.kill(os.getpid(), signal.SIGXFSZ)
        return original_publish(store, name, value)

    original_write = os.write
    def no_space(fd, value):
        if mode == "enospc-outcome" and ".outcome.json." in os.readlink(f"/proc/self/fd/{fd}"):
            raise OSError(errno.ENOSPC, "injected metadata disk exhaustion")
        return original_write(fd, value)

    original_copy = worker.copy.deepcopy
    def allocation(value, memo=None, **kwargs):
        if mode == "outcome-allocation" and isinstance(value, dict) and "finalLinked" in value:
            raise MemoryError("injected outcome diagnostic allocation")
        return original_copy(value, memo, **kwargs)

    original_encode = transport.json.JSONEncoder.iterencode
    def encoding(encoder, value, _one_shot=False):
        if mode == "outcome-encoding" and isinstance(value, dict) and "asOf" in value:
            def fail_after_bytes():
                yield '{"incomplete":'
                raise ValueError("injected metadata encoding failure")
            return fail_after_bytes()
        return original_encode(encoder, value, _one_shot=_one_shot)

    try:
        with patch.object(worker._Signals, "restore", pending_on_restore), \
             patch.object(transport.PrivateDirectory, "publish", publish), \
             patch.object(transport.os, "write", no_space), \
             patch.object(worker.copy, "deepcopy", allocation), \
             patch.object(transport.json.JSONEncoder, "iterencode", encoding):
            result = worker.run_temporal_worker(operation, directory, handle,
                run_identity=RUN_ID, export_policy=EXPORT)
        print("WORKER_RESULT "+json.dumps({
            "proposedExitCode": result["proposedExitCode"],
            "firstSignal": result["firstSignal"],
            "secondaryFailures": result["secondaryFailures"],
            "priorHandlerCalls": prior_calls,
            "actualExitObserved": result["actualExitObserved"]}), flush=True)
        return result["proposedExitCode"]
    finally:
        # A default-restored pending TERM kills before this runs; its absence is
        # intentional. Every orderly escaping error still checks restoration.
        restored = (signal.pthread_sigmask(signal.SIG_BLOCK, ()) == before_mask and
                    all(signal.getsignal(n) == h for n, h in before_handlers.items()))
        print("RESTORATION "+json.dumps({"restored": restored, "ownerReleased": worker._OWNER is None}), flush=True)


@unittest.skipUnless(sys.platform == "linux", "Prospective pinned Linux supplemental suite")
class TemporalPublicationFaults(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def store(self):
        return transport.PrivateDirectory.create(self.root/"worker", EXPORT)

    def test_pre_link_syscall_failure_never_publishes_own_content(self):
        with self.store() as store, \
             patch.object(transport.os, "link", side_effect=OSError(errno.EIO, "pre-link failure")):
            with self.assertRaises(OSError) as caught:
                store.publish("snapshot.json", {"our": "content"})
            self.assertEqual(caught.exception.errno, errno.EIO)
            row = store.observations["snapshot.json"]
            self.assertTrue(row["contentComplete"])
            self.assertTrue(row["fileSynced"])
            self.assertFalse(row["finalLinked"])
            self.assertEqual(row["publicationFailure"]["type"], "OSError")
            self.assertEqual(list((self.root/"worker").iterdir()), [])

    def test_eexist_race_preserves_rival_final_and_own_failure_stage(self):
        original_link = os.link
        rival = b'{"rival":true}\n'
        def race(source, destination, *, src_dir_fd, dst_dir_fd, follow_symlinks):
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                         0o400, dir_fd=dst_dir_fd)
            try:
                count = os.write(fd, rival)
                if count != len(rival): raise AssertionError("Fixture rival short write")
            finally:
                os.close(fd)
            return original_link(source, destination, src_dir_fd=src_dir_fd,
                                 dst_dir_fd=dst_dir_fd, follow_symlinks=follow_symlinks)
        with self.store() as store, patch.object(transport.os, "link", race):
            with self.assertRaises(FileExistsError):
                store.publish("snapshot.json", {"our": "different content"})
            self.assertEqual((self.root/"worker/snapshot.json").read_bytes(), rival)
            row = store.observations["snapshot.json"]
            self.assertFalse(row["finalLinked"])
            self.assertTrue(row["temporaryRemoved"])
            observed = store.observe("snapshot.json", EXPORT.snapshot_bytes)
            self.assertEqual(observed["sha256"], hashlib.sha256(rival).hexdigest())
            self.assertIn("deferred", observed["validation"])
            self.assertEqual([p.name for p in (self.root/"worker").iterdir()], ["snapshot.json"])

    def test_enospc_before_raw_publication_keeps_snapshot_unavailable(self):
        with self.store() as store, \
             patch.object(transport.os, "write", side_effect=OSError(errno.ENOSPC, "no space")):
            with self.assertRaises(OSError) as caught:
                store.publish("snapshot.json", {"complete": True})
            self.assertEqual(caught.exception.errno, errno.ENOSPC)
            row = store.observations["snapshot.json"]
            self.assertFalse(row["contentComplete"])
            self.assertFalse(row["finalLinked"])
            self.assertEqual(list((self.root/"worker").iterdir()), [])


@unittest.skipUnless(sys.platform == "linux", "Prospective pinned Linux supplemental suite")
class TemporalRealFaults(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.counter = 0

    def launch(self, mode):
        self.counter += 1
        directory = self.root/str(self.counter)
        command = [sys.executable, str(Path(__file__).resolve()), "--fixture", mode, str(directory)]
        result = supervisor.supervise_temporal(command, directory, run_identity=RUN_ID, policy=POLICY)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["processExitSucceeded"])
        self.assertEqual(os.waitstatus_to_exitcode(result["workerRawWaitStatus"]), result["workerReturnCode"])
        self.assertTrue(result["cleanupComplete"], result)
        self.assertFalse(result["remainingGroupMembers"])
        self.assertFalse(result["signalsSent"], result)
        self.assertTrue(result["byteObservationWithoutErrors"], result)
        for artifact in result["artifacts"]:
            raw = (directory/artifact["path"]).read_bytes()
            self.assertEqual(artifact["bytes"], len(raw))
            self.assertEqual(artifact["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertIn("deferred", artifact["validation"])
        return result, directory

    def raw(self, directory):
        return json.loads((directory/"worker/snapshot.json").read_text())["snapshot"]

    def prefixed(self, directory, prefix):
        lines=[line[len(prefix):] for line in (directory/"worker.log").read_text().splitlines() if line.startswith(prefix)]
        self.assertEqual(len(lines), 1)
        return json.loads(lines[0])

    def assert_orderly_restoration(self, directory):
        self.assertEqual(self.prefixed(directory,"RESTORATION "), {"restored":True,"ownerReleased":True})

    def test_pending_term_invokes_restored_custom_handler_and_preserves_full_raw(self):
        result, directory = self.launch("restored-custom")
        self.assertEqual(result["workerReturnCode"], 1)
        self.assertTrue(self.raw(directory)["complete"])
        metadata=json.loads((directory/"worker/outcome.json").read_text())
        self.assertEqual(metadata["proposedExitCode"], 0)
        self.assertIsNone(metadata["firstSignal"])
        observed=self.prefixed(directory,"WORKER_RESULT ")
        self.assertEqual(observed["priorHandlerCalls"], [signal.SIGTERM])
        self.assertIsNone(observed["firstSignal"])
        self.assertTrue(any(row["stage"]=="signal-restoration" and row["type"]=="RuntimeError"
                            for row in observed["secondaryFailures"]))
        self.assert_orderly_restoration(directory)

    def test_pending_term_under_restored_default_has_actual_signal_exit(self):
        result, directory = self.launch("restored-default")
        self.assertEqual(result["workerReturnCode"], -signal.SIGTERM)
        self.assertEqual(result["workerSignal"], signal.SIGTERM)
        self.assertTrue(self.raw(directory)["complete"])
        metadata=json.loads((directory/"worker/outcome.json").read_text())
        self.assertEqual(metadata["proposedExitCode"], 0)
        self.assertIsNone(metadata["firstSignal"])
        self.assertNotIn("WORKER_RESULT ", (directory/"worker.log").read_text())

    def test_sigxfsz_during_export_latches_and_preserves_full_published_raw(self):
        result, directory = self.launch("late-xfsz")
        self.assertEqual(result["workerReturnCode"], 86)
        self.assertTrue(self.raw(directory)["complete"])
        observed=self.prefixed(directory,"WORKER_RESULT ")
        self.assertEqual(observed["firstSignal"]["number"],signal.SIGXFSZ)
        self.assertEqual(observed["firstSignal"]["observedPhase"],"exporting")
        self.assert_orderly_restoration(directory)

    def test_kernel_file_limit_failure_is_nonzero_with_no_invented_snapshot(self):
        result, directory = self.launch("kernel-fsize")
        self.assertEqual(result["workerReturnCode"], 1)
        self.assertFalse((directory/"worker/snapshot.json").exists())
        self.assertFalse((directory/"worker/outcome.json").exists())
        observed=self.prefixed(directory,"WORKER_RESULT ")
        self.assertEqual(observed["firstSignal"]["number"],signal.SIGXFSZ)
        self.assertEqual(observed["firstSignal"]["observedPhase"],"exporting")
        self.assertTrue(any(row["stage"]=="snapshot-acquisition-or-publication"
                            for row in observed["secondaryFailures"]))
        self.assert_orderly_restoration(directory)

    def test_enospc_after_complete_raw_keeps_snapshot_and_failed_process(self):
        result, directory = self.launch("enospc-outcome")
        self.assertEqual(result["workerReturnCode"], 1)
        self.assertTrue(self.raw(directory)["complete"])
        self.assertFalse((directory/"worker/outcome.json").exists())
        observed=self.prefixed(directory,"WORKER_RESULT ")
        self.assertTrue(any(row["stage"]=="outcome-publication" and row["type"]=="OSError"
                            for row in observed["secondaryFailures"]))
        self.assert_orderly_restoration(directory)

    def test_diagnostic_allocation_failure_after_raw_keeps_actual_failure(self):
        result, directory = self.launch("outcome-allocation")
        self.assertEqual(result["workerReturnCode"], 1)
        self.assertTrue(self.raw(directory)["complete"])
        self.assertFalse((directory/"worker/outcome.json").exists())
        self.assertIn("injected outcome diagnostic allocation", (directory/"worker.log").read_text())
        self.assertNotIn("WORKER_RESULT ", (directory/"worker.log").read_text())
        self.assert_orderly_restoration(directory)

    def test_metadata_encoding_failure_after_raw_is_never_published(self):
        result, directory = self.launch("outcome-encoding")
        self.assertEqual(result["workerReturnCode"], 1)
        self.assertTrue(self.raw(directory)["complete"])
        self.assertFalse((directory/"worker/outcome.json").exists())
        observed=self.prefixed(directory,"WORKER_RESULT ")
        self.assertTrue(any(row["stage"]=="outcome-publication" and row["type"]=="ValueError"
                            for row in observed["secondaryFailures"]))
        self.assert_orderly_restoration(directory)


if __name__ == "__main__":
    if len(sys.argv)==4 and sys.argv[1]=="--fixture":
        raise SystemExit(fixture(sys.argv[2],sys.argv[3]))
    unittest.main()
