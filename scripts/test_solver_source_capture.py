"""Captured nested engine dependencies must never resolve through live paths."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from solver_process_budget import captured_source_path, verify_capture


class SourceCaptureTests(unittest.TestCase):
    def fixture(self, root):
        snapshot = root / "source-snapshot"
        (snapshot / "services" / "engine").mkdir(parents=True)
        report = {"sourceDigests": {}}
        for name, key in (("canonical.json", "canonicalSha256"), ("placement.json", "placementSha256")):
            content = b"{}\n"
            (root / name).write_bytes(content)
            report[key] = hashlib.sha256(content).hexdigest()
        for name in ("solver_example.py", "solver-contact.requirements.txt", "services/engine/assembly.py"):
            content = ("# " + name + "\n").encode()
            (snapshot / name).write_bytes(content)
            report["sourceDigests"][name] = hashlib.sha256(content).hexdigest()
        return report

    def test_nested_captured_engine_bytes_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.fixture(root)
            verify_capture(root, report)
            (root / "source-snapshot/services/engine/assembly.py").write_text("# changed\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_capture(root, report)

    def test_manifest_cannot_escape_or_invent_another_dependency_subtree(self):
        invalid = ("../engine.py", "/engine.py", "services/../engine.py", "services/engine/../../engine.py",
                   "services/engine//engine.py", "services/engine/./engine.py", "services/engine/sub/engine.py",
                   "services/other/engine.py", "engine/assembly.py", "services/engine/requirements.txt", "", True)
        for name in invalid:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Invalid captured source path"):
                captured_source_path(Path("/unused"), name)

    def test_nested_parent_symbolic_links_cannot_redirect_to_live_sources(self):
        for level in ("source-snapshot", "source-snapshot/services", "source-snapshot/services/engine"):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                redirected = root / "live"
                redirected.mkdir()
                link = root / level
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(redirected, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "symbolic links"):
                    captured_source_path(root, "services/engine/assembly.py")


if __name__ == "__main__":
    unittest.main()
