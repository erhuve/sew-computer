"""Behavioral directory inventory regressions without live-anchor seek queries."""
import errno
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import solver_temporal_supervision as supervisor
from solver_temporal_transport import ExportPolicy, PrivateDirectory


POLICY = ExportPolicy(4096, 4096, 4096)


class TemporalInventoryIndependentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.root = self.parent / "job"
        self.store = PrivateDirectory.create(self.root, POLICY)
        self.addCleanup(self.store.close)
        (self.root / "worker").mkdir(mode=0o700)
        (self.root / "worker/snapshot.json").write_bytes(b"{}\n")

    def observed(self):
        artifacts, errors, size = supervisor.observe_job(self.store, POLICY)
        self.assertEqual([row["path"] for row in artifacts], ["worker/snapshot.json"])
        self.assertEqual(size, 3)
        return errors

    def test_first_observation_rejects_root_name_created_after_anchor_open(self):
        (self.root / "unexpected").write_bytes(b"x")
        errors = self.observed()
        self.assertEqual([row["stage"] for row in errors], ["job-inventory"])

    def test_inventory_follows_anchor_after_path_is_replaced(self):
        (self.root / "unexpected").write_bytes(b"x")
        moved = self.parent / "moved"
        self.root.rename(moved)
        self.root.mkdir(mode=0o700)
        errors = self.observed()
        self.assertEqual([row["stage"] for row in errors], ["job-inventory"])
        self.assertTrue((moved / "worker/snapshot.json").is_file())
        self.assertEqual(list(self.root.iterdir()), [])

    @unittest.skipUnless(sys.platform == "linux", "Passive Linux fdinfo position witness")
    def test_inventory_leaves_an_existing_directory_stream_untouched(self):
        (self.root / "unexpected").write_bytes(b"x")
        # Reopen only to establish a stream containing the newly created names.
        # The production observer must not consume or rewind that shared stream.
        self.store.close()
        self.store = PrivateDirectory.open(self.root, POLICY)
        self.addCleanup(self.store.close)
        def position():
            rows = Path(f"/proc/self/fdinfo/{self.store.fd}").read_text().splitlines()
            return next(row for row in rows if row.startswith("pos:"))
        with os.scandir(self.store.fd) as entries:
            first = next(entries).name
            before = position()
            errors = self.observed()
            self.assertEqual(position(), before)
            remaining = [entry.name for entry in entries]
        self.assertEqual([row["stage"] for row in errors], ["job-inventory"])
        self.assertEqual(sorted([first, *remaining]), ["unexpected", "worker"])

    def test_inventory_open_failure_preserves_final_file_observation(self):
        original = os.open
        def deny_inventory(path, flags, *args, **kwargs):
            if path == ".":
                raise OSError(errno.EIO, "injected inventory open failure")
            return original(path, flags, *args, **kwargs)
        with patch.object(supervisor.os, "open", deny_inventory):
            errors = self.observed()
        self.assertEqual([row["stage"] for row in errors], ["job-inventory", "worker-inventory"])
        self.assertTrue(all(row["type"] == "OSError" for row in errors))

    @unittest.skipUnless(sys.platform == "linux", "Linux descriptor lifetime witness")
    def test_failed_scans_close_owned_descriptors_and_keep_final_file(self):
        before = set(os.listdir("/proc/self/fd"))
        with patch.object(supervisor.os, "scandir", side_effect=OSError(errno.EIO, "injected scan failure")):
            for _ in range(8):
                errors = self.observed()
                self.assertEqual([row["stage"] for row in errors], ["job-inventory", "worker-inventory"])
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)

    @unittest.skipUnless(sys.platform == "linux", "Linux descriptor lifetime witness")
    def test_successful_scans_close_owned_descriptors(self):
        before = set(os.listdir("/proc/self/fd"))
        for _ in range(8):
            self.assertFalse(self.observed())
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)


if __name__ == "__main__":
    unittest.main()
