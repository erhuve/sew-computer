"""Linux capture/replay checks for explicit pending-seam controls."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from solver_sewing_input import sewing_source_identity
from test_solver_sewing_input import fixture


@unittest.skipUnless(sys.platform.startswith("linux"), "Captured worker supervision requires Linux")
class SewingContinuationCliTests(unittest.TestCase):
    @staticmethod
    def source(mode="vector", *, initially_active=False):
        source = fixture(mode)
        source["placedMeters"] = copy.deepcopy(source["restMeters"])
        for point in source["placedMeters"][3:]:
            point[2] += .002
        controls = source["sewingActuation"]
        controls["initialTargetsMeters"] = [[0., 0., -.0019]] * 2 if mode == "vector" else [.0019] * 2
        controls["finalTargetsMeters"] = [[0., 0., -.0018]] * 2 if mode == "vector" else [.0018] * 2
        if initially_active:
            controls["schedule"]["knots"] = [{"fraction": fraction, "activation": [1., 1.]}
                                               for fraction in (0., 1.)]
        else:
            controls["schedule"]["knots"] = [{"fraction": fraction, "activation": [weight, weight]}
                                               for fraction, weight in ((0., 0.), (.5, 0.), (1., 1.))]
        for row in source["embeddedConstraints"]["constraints"]:
            row["complianceMPerN"] = 1e-5
        controls["sourceSha256"] = sewing_source_identity(source)
        return source

    def run_control(self, root, source, *, enabled=True, extra=()):
        canonical, placement, output = root / "canonical.json", root / "placement.json", root / "run"
        canonical.write_text(json.dumps(source, allow_nan=False))
        placement.write_text(json.dumps({"placedMeters": source["placedMeters"],
            "canonicalDigest": hashlib.sha256(canonical.read_bytes()).hexdigest()}))
        command = [sys.executable, str(Path(__file__).with_name("spike-contact-continuation.py")),
            "--canonical", str(canonical), "--placement", str(placement), "--output", str(output),
            "--contact-model", "rest-filtered", "--ccd-profile", "temporal-separation-tight-inclusion",
            "--activation-distance-m", ".0001", "--minimum-distance-m", ".0001", "--pressure-pa", "10000",
            "--target-fraction", "1", "--subdivisions", "4", "--step-seconds", ".004",
            "--sewing-mode", source.get("sewingActuation", {}).get("mode", "vector"),
            "--cpu-limit-seconds", "30", "--wall-limit-seconds", "45"]
        if enabled:
            command.append("--sewing-activation")
        process = subprocess.run([*command, *extra], capture_output=True, text=True, timeout=55)
        report = json.loads((output / "report.json").read_text()) if (output / "report.json").exists() else None
        return process, output, report

    def replay(self, output):
        (output / "verified-replay.json").unlink(missing_ok=True)
        return subprocess.run([sys.executable, str(Path(__file__).with_name("replay-rest-filtered-continuation.py")),
            str(output), "--cpu-limit-seconds", "30"], capture_output=True, text=True, timeout=40)

    def assert_control(self, source, process, output, report):
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr + str(report.get("failure")))
        self.assertTrue(report["completed"])
        self.assertFalse(report["accepted"])
        self.assertEqual(report["sewingBinding"]["sourceSha256"], source["sewingActuation"]["sourceSha256"])
        self.assertEqual(report["sewingBinding"]["complianceMPerN"], 1e-5)
        self.assertIn("no pattern-source proof", report["sewingBinding"]["sourceBindingScope"])
        self.assertIn("services/engine/embedded_constraints.py", report["sourceDigests"])
        self.assertIn("spike-full-shirt.py", report["sourceDigests"])
        self.assertIn("sewingActivationInterpolation", report["adaptive"])
        self.assertEqual(report["sewingWorkSummary"]["acceptedSteps"], len(report["acceptedStateArtifacts"]))
        self.assertEqual(report["sewingWorkSummary"]["sewingReleaseEnergyRemovedJoules"], 0.)
        for step in report["adaptive"]["acceptedSteps"]:
            self.assertIn("energyBalance", step["step"])
        replay = self.replay(output)
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        verified = json.loads((output / "verified-replay.json").read_text())
        self.assertTrue(verified["initialSewingVerification"]["verified"])
        self.assertEqual(verified["sewingVerifierSha256"], hashlib.sha256(
            Path(__file__).with_name("solver_sewing_replay.py").read_bytes()).hexdigest())
        self.assertEqual(verified["verifiedSewingWorkSummary"]["acceptedSteps"], len(report["acceptedStateArtifacts"]))
        for state in verified["states"]:
            self.assertIn("sewingVerification", state)
            self.assertLessEqual(state["recomputedResidualN"], 1e-6)
        return verified

    def test_all_modes_capture_pending_then_engaged_work_and_replay(self):
        for mode in ("vector", "distance", "normal-offset"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                source = self.source(mode)
                process, output, report = self.run_control(Path(directory), source)
                self.assert_control(source, process, output, report)
                self.assertEqual(report["initialSewingEnergyJoules"], 0.)
                self.assertEqual(report["initialSewingRowTargetErrorsM"], [None, None])
                self.assertGreater(report["sewingWorkSummary"]["sewingActivationIncreaseWorkJoules"], 0.)
                for record in report["adaptive"]["acceptedSteps"]:
                    if record["endFraction"] <= .5:
                        self.assertEqual(record["step"]["pendingSewingRows"], [0, 1])
                        self.assertEqual(record["step"]["sewingJoules"], 0.)

    def test_explicit_initial_stored_energy_does_not_require_equilibrium(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source("normal-offset", initially_active=True)
            process, output, report = self.run_control(Path(directory), source)
            self.assert_control(source, process, output, report)
            self.assertGreater(report["initialSewingEnergyJoules"], 0.)
            self.assertTrue(all(value > 0 for value in report["initialSewingRowTargetErrorsM"]))

    def test_opt_in_recipe_and_legacy_target_scaling_cannot_conflict(self):
        for enabled, missing, extra in ((False, False, ()), (True, True, ()),
                                        (True, False, ("--target-fraction", ".9"))):
            with self.subTest(enabled=enabled, missing=missing, extra=extra), tempfile.TemporaryDirectory() as directory:
                source = self.source()
                if missing:
                    source.pop("sewingActuation")
                process, output, report = self.run_control(Path(directory), source, enabled=enabled, extra=extra)
                self.assertNotEqual(process.returncode, 0)
                if extra:
                    self.assertFalse(output.exists())
                else:
                    self.assertFalse(report["completed"])
                    self.assertIn("explicit opt-in", report["failure"]["message"])

    def test_aggregate_and_initial_energy_tampering_rejects_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source()
            process, output, report = self.run_control(Path(directory), source)
            self.assert_control(source, process, output, report)
            # Make only the temporary report writable for the declared attacks.
            (output / "report.json").chmod(0o600)
            for label, mutation in (
                ("missing total", lambda value: value.pop("sewingWorkSummary")),
                ("tiny total", lambda value: value["sewingWorkSummary"].__setitem__("sewingParameterWorkJoules", 1e-30)),
                ("initial energy", lambda value: value.__setitem__("initialSewingEnergyJoules", 1e-30))):
                with self.subTest(label=label):
                    altered = copy.deepcopy(report)
                    mutation(altered)
                    (output / "report.json").write_text(json.dumps(altered, allow_nan=False))
                    result = self.replay(output)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("FileExistsError", result.stderr)
                    self.assertFalse((output / "verified-replay.json").exists())
            (output / "report.json").write_text(json.dumps(report, allow_nan=False))
            self.assertEqual(self.replay(output).returncode, 0)


if __name__ == "__main__":
    unittest.main()
