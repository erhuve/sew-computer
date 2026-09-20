import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


class ContactContinuationCliTests(unittest.TestCase):
    def command(self, canonical, placement, output):
        return [sys.executable, str(Path(__file__).with_name("spike-contact-continuation.py")),
                "--canonical", str(canonical), "--placement", str(placement), "--output", str(output),
                "--activation-distance-m", ".00002", "--minimum-distance-m", ".0001",
                "--pressure-pa", "10000", "--target-fraction", "1", "--subdivisions", "1"]

    def test_invalid_parameters_rejected_before_output_creation(self):
        for extra in (["--pressure-pa", "nan"], ["--target-fraction", "0"],
                      ["--subdivisions", "3"], ["--max-depth", "31"],
                      ["--max-attempts", "0"], ["--cpu-limit-seconds", "0"],
                      ["--max-attempts", "4097"], ["--max-evaluations", "10001"],
                      ["--ccd-profile", "unknown"], ["--contact-model", "unknown"],
                      ["--sewing-mode", "unknown"],
                      ["--contact-model", "rest-filtered", "--ccd-profile", "swept-plane-tight-inclusion"]):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                result = subprocess.run(self.command("missing", "missing", output) + extra,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(output.exists())

    def test_ccd_profile_is_opt_in_and_default_is_unchanged(self):
        script = Path(__file__).with_name("spike-contact-continuation.py")
        spec = importlib.util.spec_from_file_location("contact_continuation_cli", script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        arguments = self.command("canonical", "placement", "output")[2:]
        for extra, expected in (([], "tight-inclusion"),
                                (["--ccd-profile", "temporal-separation-tight-inclusion"],
                                 "temporal-separation-tight-inclusion")):
            with self.subTest(profile=expected), mock.patch.object(sys, "argv", [str(script), *arguments, *extra]):
                self.assertEqual(cli.parse_arguments().ccd_profile, expected)
                self.assertEqual(cli.parse_arguments().contact_model, "area-improved-max")
                self.assertEqual(cli.parse_arguments().sewing_mode, "vector")

    def test_distance_profile_snapshot_executes_with_journal(self):
        command = self.command
        with mock.patch.object(self, "command", side_effect=lambda *args:
                               command(*args) + ["--sewing-mode", "distance", "--contact-model", "rest-filtered",
                                                "--ccd-profile", "temporal-separation-tight-inclusion"]):
            self.test_synthetic_stationary_interval_is_complete_but_not_accepted()

    def test_rest_filtered_profile_snapshot_executes_with_journal(self):
        command = self.command
        with mock.patch.object(self, "command", side_effect=lambda *args:
                               command(*args) + ["--contact-model", "rest-filtered"]):
            self.test_synthetic_stationary_interval_is_complete_but_not_accepted()

    def test_temporal_profile_snapshot_executes_with_journal(self):
        command = self.command
        with mock.patch.object(self, "command", side_effect=lambda *args:
                               command(*args) + ["--ccd-profile", "temporal-separation-tight-inclusion"]):
            self.test_synthetic_stationary_interval_is_complete_but_not_accepted()

    def test_rest_filtered_temporal_snapshot_executes_with_journal(self):
        command = self.command
        with mock.patch.object(self, "command", side_effect=lambda *args:
                               command(*args) + ["--contact-model", "rest-filtered",
                                                "--ccd-profile", "temporal-separation-tight-inclusion"]):
            self.test_synthetic_stationary_interval_is_complete_but_not_accepted()

    def test_digest_mismatch_retains_captured_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical, placement, output = root / "canonical.json", root / "placement.json", root / "output"
            canonical.write_text("{}")
            placement.write_text(json.dumps({"canonicalDigest": "wrong"}))
            result = subprocess.run(self.command(canonical, placement, output),
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["accepted"])
            self.assertIn("canonicalDigest", report["failure"]["message"])
            self.assertEqual(report["canonicalSha256"], hashlib.sha256(canonical.read_bytes()).hexdigest())
            self.assertEqual((output / "canonical.json").read_bytes(), canonical.read_bytes())
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)

    def test_synthetic_stationary_interval_is_complete_but_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical, placement, output = root / "canonical.json", root / "placement.json", root / "output"
            rest = [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.],
                    [0., 0., .01], [.1, 0., .01], [0., .1, .01]]
            source = {"restMeters": rest, "placedMeters": rest, "triangles": [0, 1, 2, 3, 4, 5],
                      "instanceOffsets": {"first": 0, "second": 3},
                      "embeddedConstraints": {"constraints": [{"terms": [
                          {"instanceId": "first", "vertex": 0, "coefficient": 1.},
                          {"instanceId": "second", "vertex": 0, "coefficient": -1.}]}]}}
            canonical.write_text(json.dumps(source))
            placement.write_text(json.dumps({"placedMeters": rest,
                "canonicalDigest": hashlib.sha256(canonical.read_bytes()).hexdigest()}))
            result = subprocess.run(self.command(canonical, placement, output),
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["adaptive"]["complete"])
            self.assertTrue(report["completed"])
            self.assertTrue(report["terminal"])
            self.assertTrue(report["attemptJournal"]["finished"])
            self.assertEqual(report["attemptJournal"]["errors"], [])
            self.assertEqual(report["recoveryErrors"], [])
            self.assertIn("solver_attempt_journal.py", report["sourceDigests"])
            self.assertEqual([attempt["outcome"] for attempt in report["adaptive"]["attempts"]], ["accepted"])
            self.assertEqual(report["attemptJournal"]["lastAcceptedState"], report["acceptedStateArtifacts"][-1])
            self.assertTrue(report["inactiveReferenceGatePassed"])
            self.assertFalse(report["accepted"])
            self.assertFalse(report["finalToolkitHasIntersections"])
            self.assertEqual(report["finalIndependentSurfaceOracle"]["intersectingPairCount"], 0)
            self.assertTrue((output / "state.json").exists())
            self.assertEqual(len(report["acceptedStateArtifacts"]), 1)
            artifact = report["acceptedStateArtifacts"][0]
            content = (output / artifact["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), artifact["sha256"])
            self.assertEqual(json.loads(content)["record"]["completedDurationSeconds"], 1 / 240)
            if report["arguments"].get("sewing_mode") == "distance":
                replay_command = [sys.executable, str(Path(__file__).with_name("replay-rest-filtered-continuation.py")),
                                  str(output)]
                replay = subprocess.run(replay_command, capture_output=True, text=True, timeout=30)
                self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
                verification = json.loads((output / "verified-replay.json").read_text())
                self.assertTrue(verification["finalStateVerified"])
                self.assertEqual(len(verification["states"]), 1)
                corrupted = json.loads(content)
                corrupted["positionsMeters"][0][0] += .001
                (output / artifact["path"]).write_text(json.dumps(corrupted))
                replay = subprocess.run(replay_command, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(replay.returncode, 0)
                self.assertIn("AssertionError", replay.stderr)
            rejected_output = root / "active-reference"
            rejected = subprocess.run(self.command(canonical, placement, rejected_output)
                                      + ["--activation-distance-m", ".02"],
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(rejected.returncode, 1, rejected.stdout + rejected.stderr)
            rejection = json.loads((rejected_output / "report.json").read_text())
            self.assertFalse(rejection["inactiveReferenceGatePassed"])
            self.assertIn("nonzero initial contact", rejection["failure"]["message"])
            self.assertNotIn("adaptive", rejection)


if __name__ == "__main__":
    unittest.main()
