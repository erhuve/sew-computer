"""Independent combined-control and durable-evidence attacks; synthetic cloth only."""

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from solver_attempt_journal import recover_attempt_journal
from solver_gripper_input import mesh_identity
from solver_process_budget import captured_source_path, json_bytes, verify_capture
from solver_sewing_input import sewing_source_identity
import test_solver_sewing_continuation_cli as cli


class NestedCaptureReviewTests(unittest.TestCase):
    def test_only_declared_flat_and_engine_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("solver_test.py", "solver-contact.requirements.txt", "solver-spike.requirements.txt",
                         "services/engine/embedded_constraints.py"):
                self.assertEqual(captured_source_path(root, name), root / "source-snapshot" / name)
            for name in ("../outside.py", "/tmp/outside.py", "services/../outside.py", "services/engine/../../outside.py",
                         "services//engine/outside.py", "services/engine/outside.py/child.py", "services/engine/.hidden.py",
                         "services\\engine\\outside.py", "services/engine/solver-spike.requirements.txt", "", None, True):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    captured_source_path(root, name)

    def test_nested_capture_rejects_symlink_components_and_final_file(self):
        for component in ("source-snapshot", "source-snapshot/services", "source-snapshot/services/engine",
                          "source-snapshot/services/engine/assembly.py"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "source-snapshot/services/engine/assembly.py"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"# captured source\n")
                report = {"sourceDigests": {"services/engine/assembly.py": hashlib.sha256(path.read_bytes()).hexdigest()}}
                for name, key in (("canonical.json", "canonicalSha256"), ("placement.json", "placementSha256")):
                    (root / name).write_bytes(b"{}\n")
                    report[key] = hashlib.sha256((root / name).read_bytes()).hexdigest()
                verify_capture(root, report)
                target = root / component
                displaced = root / "displaced"
                target.rename(displaced)
                target.symlink_to(displaced, target_is_directory=displaced.is_dir())
                with self.assertRaises((OSError, ValueError)):
                    verify_capture(root, report)


@unittest.skipUnless(sys.platform.startswith("linux"), "Captured worker supervision requires Linux")
class CombinedSewingGripperReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.driver = cli.SewingContinuationCliTests()
        cls.source = cls.combined_source()
        cls.process, cls.output, cls.report = cls.driver.run_control(cls.root, cls.source, extra=("--material-grippers",))
        if cls.process.returncode:
            raise AssertionError(cls.process.stdout + cls.process.stderr + str(cls.report))
        cls.verified = cls.driver.assert_control(cls.source, cls.process, cls.output, cls.report)

    @staticmethod
    def combined_source():
        source = cli.SewingContinuationCliTests.source("vector")
        anchors = [{"id": f"pull-{index}", "instanceId": "a:shell", "triangleIndex": 0,
                    "weights": [float(vertex == index) for vertex in range(3)], "stiffnessNPerM": .1}
                   for index in range(3)]
        initial = np.asarray(source["placedMeters"][:3])
        knots = []
        for fraction, shift, activation in ((0., [.00002, 0., 0.], .25),
                (.25, [.00005, 0., 0.], 1.), (.5, [.00005, .00002, 0.], 1.),
                (.75, [.0001, .00002, 0.], .5), (1., [.0001, .00002, 0.], 0.)):
            knots.append({"fraction": fraction, "targetsMeters": (initial + shift).tolist(), "activation": [activation] * 3})
        source["gripperActuation"] = {"profile": "captured-material-grippers-v1", "accepted": False,
            "meshSha256": mesh_identity(source), "anchors": anchors,
            "schedule": {"profile": "material-gripper-target-activation-v1", "gripperIds": [anchor["id"] for anchor in anchors], "knots": knots}}
        source["sewingActuation"]["sourceSha256"] = sewing_source_identity(source)
        return source

    def copy_run(self, name):
        destination = self.root / name
        shutil.copytree(self.output, destination)
        for path in [destination, *destination.rglob("*")]:
            path.chmod(0o700 if path.is_dir() else 0o600)
        (destination / "verified-replay.json").unlink(missing_ok=True)
        return destination, copy.deepcopy(self.report)

    def replay_rejects_without_new_proof(self, directory):
        (directory / "verified-replay.json").unlink(missing_ok=True)
        replay = self.driver.replay(directory)
        self.assertNotEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        self.assertNotIn("FileExistsError", replay.stderr)
        self.assertFalse((directory / "verified-replay.json").exists())
        return replay

    def rewrite_last_step_and_chain(self, directory, report, mutation):
        descriptor = report["acceptedStateArtifacts"][-1]
        path = directory / descriptor["path"]
        state = json.loads(path.read_bytes())
        mutation(state["record"]["step"])
        path.write_bytes(json_bytes(state))
        replacement = {"path": descriptor["path"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        previous = None
        for path in sorted(directory.glob("attempt-*.json")):
            payload = json.loads(path.read_bytes())["payload"]
            payload["previousSha256"] = previous
            if payload["kind"] == "outcome" and payload["data"]["attemptId"] == state["record"]["attemptId"]:
                payload["data"] = copy.deepcopy(state["record"]) | {"acceptedState": replacement}
            envelope = {"payload": payload, "sha256": hashlib.sha256(json_bytes(payload)).hexdigest()}
            path.write_bytes(json_bytes(envelope))
            previous = hashlib.sha256(path.read_bytes()).hexdigest()
        recovered = recover_attempt_journal(directory, report)
        self.assertEqual(recovered["attemptJournal"]["errors"], [])
        self.assertTrue(recovered["adaptive"]["complete"])
        report.update(recovered)
        # Match forged summaries too, so the attack must fail transition
        # semantics rather than a trivially stale aggregate or broken chain.
        for key in ("sewingWorkSummary", "gripperWorkSummary"):
            for name in report[key]:
                if name.endswith("Joules"):
                    report[key][name] = math.fsum(row["step"]["energyBalance"][name]
                                                  for row in report["adaptive"]["acceptedSteps"])
        (directory / "report.json").write_bytes(json_bytes(report))

    def rebind_input_header_and_chain(self, directory, report):
        previous = None
        for path in sorted(directory.glob("attempt-*.json")):
            payload = json.loads(path.read_bytes())["payload"]
            payload["previousSha256"] = previous
            if payload["kind"] == "header":
                payload["data"]["provenance"] = {key: report[key] for key in
                    ("sourceDigests", "canonicalSha256", "placementSha256", "arguments")}
            envelope = {"payload": payload, "sha256": hashlib.sha256(json_bytes(payload)).hexdigest()}
            path.write_bytes(json_bytes(envelope))
            previous = hashlib.sha256(path.read_bytes()).hexdigest()
        recovered = recover_attempt_journal(directory, report)
        self.assertEqual(recovered["attemptJournal"]["errors"], [])
        self.assertTrue(recovered["adaptive"]["complete"])
        report.update(recovered)
        (directory / "report.json").write_bytes(json_bytes(report))

    def test_combined_engagement_release_work_and_independent_force_replay(self):
        report, verified = self.report, self.verified
        self.assertTrue(report["arguments"]["sewing_activation"])
        self.assertTrue(report["arguments"]["material_grippers"])
        self.assertFalse(report["accepted"] or verified["accepted"])
        self.assertEqual(json.loads((self.output / "canonical.json").read_bytes()), self.source)
        self.assertTrue(verified["initialSewingVerification"]["verified"])
        self.assertTrue(verified["initialGripperVerification"]["verified"])
        self.assertGreater(report["gripperWorkSummary"]["gripperReleaseEnergyRemovedJoules"], 0.)
        self.assertGreater(report["sewingWorkSummary"]["sewingActivationIncreaseWorkJoules"], 0.)
        self.assertEqual(report["sewingWorkSummary"]["sewingReleaseEnergyRemovedJoules"], 0.)
        for record, state in zip(report["adaptive"]["acceptedSteps"], verified["states"]):
            self.assertIn("sewingVerification", state)
            self.assertIn("gripperVerification", state)
            self.assertIn("gripperMomentum", record["step"])
            energy = record["step"]["energyBalance"]
            self.assertAlmostEqual(energy["externalParameterWorkJoules"],
                energy["sewingParameterWorkJoules"] + energy["gripperParameterWorkJoules"], delta=1e-20)
            self.assertLessEqual(state["recomputedResidualN"], 1e-6)
        for summary_name, verified_name in (("sewingWorkSummary", "verifiedSewingWorkSummary"),
                                           ("gripperWorkSummary", "verifiedGripperWorkSummary")):
            for key, value in report[summary_name].items():
                if key.endswith("Joules"):
                    self.assertEqual(value, verified[verified_name][key])

    def test_rehashed_control_work_and_momentum_forgeries_reject_semantically(self):
        mutations = (
            ("sewing-activation", lambda step: step["sewingActivation"].__setitem__(0, .5)),
            ("sewing-work", lambda step: step["energyBalance"].__setitem__("sewingParameterWorkJoules",
                                  step["energyBalance"]["sewingParameterWorkJoules"] + 1e-8)),
            ("gripper-work", lambda step: step["energyBalance"].__setitem__("gripperParameterWorkJoules",
                                  step["energyBalance"]["gripperParameterWorkJoules"] + 1e-8)),
            ("external-impulse", lambda step: step["gripperMomentum"]["externalImpulseNs"].__setitem__(0, 1e-5)))
        for label, mutation in mutations:
            with self.subTest(label=label):
                directory, report = self.copy_run("attack-" + label)
                self.rewrite_last_step_and_chain(directory, report, mutation)
                self.replay_rejects_without_new_proof(directory)

    def test_combined_interrupted_prefix_recomputes_only_durable_accepted_work(self):
        directory, report = self.copy_run("interrupted-prefix")
        starts, truncate = 0, False
        for path in sorted(directory.glob("attempt-*.json")):
            if truncate:
                path.unlink()
            elif json.loads(path.read_bytes())["payload"]["kind"] == "start":
                starts += 1
                truncate = starts == 4
        self.assertTrue(truncate)
        recovered = recover_attempt_journal(directory, report, interruption="synthetic interrupted fourth trial")
        self.assertEqual(recovered["attemptJournal"]["errors"], [])
        self.assertEqual(len(recovered["adaptive"]["acceptedSteps"]), 3)
        self.assertEqual(len(recovered["adaptive"]["interruptedSteps"]), 1)
        report.update(recovered)
        report.update(completed=False, classification="uncompleted-research-diagnostic")
        for key in [*report]:
            if key in ("stateArtifact", "sewingWorkSummary", "gripperWorkSummary") or key.startswith("final"):
                report.pop(key)
        (directory / "report.json").write_bytes(json_bytes(report))
        replay = self.driver.replay(directory)
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        verified = json.loads((directory / "verified-replay.json").read_bytes())
        self.assertFalse(verified["completed"] or verified["finalStateVerified"] or verified["accepted"])
        self.assertEqual(verified["completedFraction"], .75)
        self.assertEqual(len(verified["states"]), 3)
        for name in ("verifiedSewingWorkSummary", "verifiedGripperWorkSummary"):
            self.assertEqual(verified[name]["acceptedSteps"], 3)
            for key, value in verified[name].items():
                if key.endswith("Joules"):
                    self.assertEqual(value, math.fsum(record["step"]["energyBalance"][key]
                                                     for record in report["adaptive"]["acceptedSteps"]))

    def test_duplicate_raw_json_keys_reject_before_capture_motion(self):
        for attacked_name in ("canonical", "placement"):
            with self.subTest(input=attacked_name):
                root = self.root / ("duplicate-capture-" + attacked_name)
                root.mkdir()
                canonical, placement, output = root / "canonical.json", root / "placement.json", root / "run"
                source_bytes = json_bytes(self.source)
                if attacked_name == "canonical":
                    source_bytes = b'{"restMeters":[],' + source_bytes[1:]
                canonical.write_bytes(source_bytes)
                placement_bytes = json_bytes({"placedMeters": self.source["placedMeters"],
                    "canonicalDigest": hashlib.sha256(source_bytes).hexdigest()})
                if attacked_name == "placement":
                    placement_bytes = b'{"canonicalDigest":"discarded",' + placement_bytes[1:]
                placement.write_bytes(placement_bytes)
                command = list(self.process.args)
                for flag, value in (("--canonical", canonical), ("--placement", placement), ("--output", output)):
                    command[command.index(flag) + 1] = str(value)
                result = subprocess.run(command, capture_output=True, text=True, timeout=55)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((output / "report.json").read_bytes())
                self.assertFalse(report["completed"])
                self.assertEqual(report["acceptedStateArtifacts"], [])
                self.assertIn("Duplicate JSON key", report["failure"]["message"])

    def test_duplicate_raw_json_keys_reject_replay_despite_valid_rebound_capture_chain(self):
        for name in ("canonical", "placement"):
            with self.subTest(input=name):
                directory, report = self.copy_run("duplicate-replay-" + name)
                if name == "canonical":
                    path = directory / "canonical.json"
                    path.write_bytes(b'{"restMeters":[],' + path.read_bytes()[1:])
                    report["canonicalSha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    placement = json.loads((directory / "placement.json").read_bytes())
                    placement["canonicalDigest"] = report["canonicalSha256"]
                    (directory / "placement.json").write_bytes(json_bytes(placement))
                else:
                    path = directory / "placement.json"
                    path.write_bytes(b'{"canonicalDigest":"discarded",' + path.read_bytes()[1:])
                report["placementSha256"] = hashlib.sha256((directory / "placement.json").read_bytes()).hexdigest()
                self.rebind_input_header_and_chain(directory, report)
                verify_capture(directory, report)
                result = self.replay_rejects_without_new_proof(directory)
                self.assertIn("Duplicate JSON key", result.stderr)


if __name__ == "__main__":
    unittest.main()
