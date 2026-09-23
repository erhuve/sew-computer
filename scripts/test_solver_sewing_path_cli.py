"""Optional exact seam-path gate through the actual captured replay CLI."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import test_solver_sewing_continuation_cli as continuation


@unittest.skipUnless(sys.platform.startswith("linux"), "Captured worker supervision requires Linux")
class SewingPathCliTests(unittest.TestCase):
    def replay(self, output, *arguments):
        (output / "verified-replay.json").unlink(missing_ok=True)
        return subprocess.run([sys.executable, str(Path(__file__).with_name("replay-rest-filtered-continuation.py")),
            str(output), "--cpu-limit-seconds", "30", *arguments],
            capture_output=True, text=True, timeout=40)

    def control(self, root, source, *, enabled=True):
        helper = continuation.SewingContinuationCliTests()
        process, output, report = helper.run_control(root, source, enabled=enabled)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr + str(report))
        self.assertTrue(report["completed"])
        return output, report

    def test_requested_bound_proves_complete_paths_or_rejects_without_new_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output, report = self.control(Path(directory), continuation.SewingContinuationCliTests.source("distance"))
            original_report = (output / "report.json").read_bytes()
            result = self.replay(output, "--sewing-path-tolerance-m", ".001")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            evidence = json.loads((output / "verified-replay.json").read_text())
            self.assertIs(evidence["accepted"], False)
            self.assertEqual(evidence["sewingPathToleranceM"], .001)
            self.assertEqual(evidence["sewingPathVerifierSha256"], hashlib.sha256(
                Path(__file__).with_name("solver_sewing_sweep.py").read_bytes()).hexdigest())
            self.assertEqual(len(evidence["states"]), len(report["acceptedStateArtifacts"]))
            digests = set()
            for item in evidence["states"]:
                proof = item["sewingPathVerification"]
                self.assertIs(proof["accepted"], False)
                self.assertIs(proof["verified"], True)
                self.assertEqual(proof["sourceSha256"], report["sewingBinding"]["sourceSha256"])
                self.assertEqual(proof["endFraction"], item["endFraction"])
                self.assertEqual([row["rowId"] for row in proof["rows"]], report["sewingBinding"]["rowIds"])
                active_count = 0 if item["endFraction"] <= .5 else 2
                self.assertEqual(proof["checkedRows"], active_count)
                self.assertEqual(proof["pendingRows"], 2 - active_count)
                self.assertIn("planned-ramp geometric reference", proof["targetReference"])
                self.assertIn("No captured activation or assembly knot", proof["knotPolicy"])
                digests.add(proof["inputSha256"])
            self.assertEqual(len(digests), len(evidence["states"]))
            result = self.replay(output, "--sewing-path-tolerance-m", "1e-20")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sewing", result.stderr.lower())
            self.assertNotIn("FileExistsError", result.stderr)
            self.assertFalse((output / "verified-replay.json").exists())
            self.assertEqual((output / "report.json").read_bytes(), original_report)
            # A failed requested geometric gate does not rewrite numerical
            # evidence or make the narrower ordinary replay fail.
            result = self.replay(output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            evidence = json.loads((output / "verified-replay.json").read_text())
            self.assertIsNone(evidence["sewingPathToleranceM"])
            self.assertIsNone(evidence["sewingPathVerifierSha256"])
            self.assertTrue(all("sewingPathVerification" not in item for item in evidence["states"]))

    def test_mode_or_uncaptured_controls_cannot_acquire_distance_proof(self):
        for captured in (True, False):
            with self.subTest(captured=captured), tempfile.TemporaryDirectory() as directory:
                source = continuation.SewingContinuationCliTests.source("vector")
                if not captured:
                    source.pop("sewingActuation")
                output, _ = self.control(Path(directory), source, enabled=captured)
                result = self.replay(output, "--sewing-path-tolerance-m", ".001")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires captured scalar-distance controls", result.stderr)
                self.assertFalse((output / "verified-replay.json").exists())

    def test_invalid_tolerance_rejects_before_reading_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("0", "-1", "nan", "inf", "1e-400"):
                with self.subTest(value=value):
                    result = self.replay(root, "--sewing-path-tolerance-m", value)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("finite and strictly positive", result.stderr)
                    self.assertNotIn("FileNotFoundError", result.stderr)
                    self.assertFalse((root / "verified-replay.json").exists())


if __name__ == "__main__":
    unittest.main()
