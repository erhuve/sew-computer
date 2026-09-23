import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_solver_fold_actuation import fold_fixture
from test_solver_assembly_schedule import staged_recipe


class FoldContinuationCliTests(unittest.TestCase):
    def fixture(self):
        _, solver, positions = fold_fixture()
        return {"restMeters": positions.tolist(), "placedMeters": positions.tolist(),
                "triangles": solver.faces.ravel().tolist(), "instanceOffsets": {"panel": 0},
                "embeddedConstraints": {"constraints": []},
                "foldActuation": {"hinges": solver.fold_actuation.hinges.tolist(),
                    "stiffnessJoules": [.02], "initialAnglesRadians": [0.], "targetAnglesRadians": [.4]}}

    def run_control(self, root, source, enabled=True, scheduled=False):
        canonical, placement, output = root / "canonical.json", root / "placement.json", root / "run"
        canonical.write_text(json.dumps(source))
        placement.write_text(json.dumps({"placedMeters": source["placedMeters"],
            "canonicalDigest": hashlib.sha256(canonical.read_bytes()).hexdigest()}))
        command = [sys.executable, str(Path(__file__).with_name("spike-contact-continuation.py")),
            "--canonical", str(canonical), "--placement", str(placement), "--output", str(output),
            "--contact-model", "rest-filtered", "--ccd-profile", "temporal-separation-tight-inclusion",
            "--activation-distance-m", ".001", "--minimum-distance-m", ".0001", "--pressure-pa", "10000",
            "--target-fraction", "1", "--subdivisions", "2", "--step-seconds", ".02"]
        if enabled:
            command.append("--fold-actuation")
        if scheduled:
            command.append("--assembly-schedule")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return result, output, json.loads((output / "report.json").read_text())

    def test_captured_moving_fold_replays_and_tampering_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, report = self.run_control(Path(directory), self.fixture())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + str(report.get("failure")))
            self.assertTrue(report["completed"])
            self.assertFalse(report["accepted"])
            self.assertEqual(report["attemptJournal"]["errors"], [])
            self.assertIsNone(report["finalMaximumAnchorGapM"])
            self.assertGreater(report["finalFoldAnglesRadians"][0], .3)
            replay_command = [sys.executable, str(Path(__file__).with_name("replay-rest-filtered-continuation.py")),
                              str(output)]
            replay = subprocess.run(replay_command, capture_output=True, text=True, timeout=30)
            self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
            verified = json.loads((output / "verified-replay.json").read_text())
            self.assertEqual(len(verified["states"]), len(report["acceptedStateArtifacts"]))
            self.assertTrue(verified["finalStateVerified"])
            self.assertGreater(verified["exactCertificateLeaves"], 0)
            self.assertEqual(verified["triangleVerifierSha256"], hashlib.sha256(
                Path(__file__).with_name("solver_triangle_sweep.py").read_bytes()).hexdigest())
            self.assertEqual(verified["candidateCoverageVerifierSha256"], hashlib.sha256(
                Path(__file__).with_name("solver_candidate_coverage.py").read_bytes()).hexdigest())
            self.assertEqual(verified["verificationBroadPhase"], "ipctk-HashGrid-explicit-v1")
            for checked in verified["states"]:
                triangle_proof = checked["trianglePathVerification"]
                self.assertEqual(triangle_proof["profile"], "exact-rational-affine-triangle-nondegeneracy-v1")
                self.assertEqual(triangle_proof["triangles"], len(self.fixture()["triangles"]) // 3)
                self.assertGreaterEqual(triangle_proof["verifiedLeaves"], triangle_proof["triangles"])
                coverage = checked["candidateCoverageVerification"]
                self.assertTrue(coverage["verified"])
                self.assertEqual(coverage["status"], "complete")
                self.assertEqual(coverage["profile"], "exact-rational-swept-aabb-v1")
            state = output / report["acceptedStateArtifacts"][-1]["path"]
            data = json.loads(state.read_text())
            data["record"]["step"]["foldTargetsRadians"][0] *= -1
            state.write_text(json.dumps(data))
            replay = subprocess.run(replay_command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(replay.returncode, 0)
            self.assertIn("AssertionError", replay.stderr)

    def test_missing_recipe_opt_in_and_invalid_schedule_reject_before_motion(self):
        original = self.fixture()
        cases = [(original, False)]
        missing = copy.deepcopy(original)
        del missing["foldActuation"]
        cases.append((missing, True))
        for key, value in (("hinges", [[0, 0, 2, 3]]), ("initialAnglesRadians", [.1]),
                           ("targetAnglesRadians", [4.]), ("stiffnessJoules", [0.])):
            changed = copy.deepcopy(original)
            changed["foldActuation"][key] = value
            cases.append((changed, True))
        for source, enabled in cases:
            with self.subTest(source=source, enabled=enabled), tempfile.TemporaryDirectory() as directory:
                result, _, report = self.run_control(Path(directory), source, enabled)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertFalse(report["completed"])
                self.assertTrue(report["terminal"])
                self.assertFalse(report["accepted"])
                self.assertEqual(report["acceptedStateArtifacts"], [])
                self.assertIn("failure", report)

    def test_staged_fold_holds_then_moves_and_replays(self):
        source = self.fixture()
        source["assemblySchedule"] = staged_recipe()
        with tempfile.TemporaryDirectory() as directory:
            result, output, report = self.run_control(Path(directory), source, scheduled=True)
            self.assertEqual(result.returncode, 0, result.stderr + str(report.get("failure")))
            first = json.loads((output / report["acceptedStateArtifacts"][0]["path"]).read_text())
            self.assertEqual(first["record"]["endFraction"], .5)
            self.assertEqual(first["record"]["step"]["foldTargetsRadians"], [0.])
            self.assertGreater(report["finalFoldAnglesRadians"][0], .3)
            replay = subprocess.run([sys.executable, str(Path(__file__).with_name("replay-rest-filtered-continuation.py")),
                                     str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        for missing in ("opt-in", "recipe", "alignment"):
            changed = copy.deepcopy(source)
            if missing == "recipe":
                del changed["assemblySchedule"]
            if missing == "alignment":
                changed["assemblySchedule"]["knots"][1]["fraction"] = .25
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                result, _, report = self.run_control(Path(directory), changed, scheduled=missing != "opt-in")
                self.assertEqual(result.returncode, 1)
                self.assertEqual(report["acceptedStateArtifacts"], [])
                self.assertIn("failure", report)


if __name__ == "__main__":
    unittest.main()
