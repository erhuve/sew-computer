"""Linux capture tests for explicit material-gripper diagnostic continuation."""

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

from solver_gripper_input import mesh_identity, PROFILE
from solver_material_grippers import SCHEDULE_PROFILE
from solver_attempt_journal import recover_attempt_journal
from solver_process_budget import json_bytes


@unittest.skipUnless(sys.platform.startswith("linux"), "Captured worker supervision requires Linux")
class GripperContinuationCliTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rest = [[0., 0., 0.], [.03125, 0., 0.], [0., .03125, 0.]]
        source = {"restMeters": rest, "placedMeters": copy.deepcopy(rest), "triangles": [0, 1, 2],
                  "instanceOffsets": {"cloth:shell": 0}, "embeddedConstraints": {"constraints": []}}
        anchors = [{"id": f"grip-{index}", "instanceId": "cloth:shell", "triangleIndex": 0,
                    "weights": [float(vertex == index) for vertex in range(3)], "stiffnessNPerM": .02}
                   for index in range(3)]
        knots = [{"fraction": fraction, "targetsMeters": (np.asarray(rest) + shift).tolist(),
                  "activation": [activation] * 3} for fraction, shift, activation in (
            (0., [0., 0., .00025], .25), (.25, [0., 0., .00025], 1.),
            (.5, [.0001, 0., .0004], 1.), (.75, [.0001, 0., .0004], .5),
            (1., [.0001, 0., .0004], 0.))]
        source["gripperActuation"] = {"profile": PROFILE, "accepted": False,
            "meshSha256": mesh_identity(source), "anchors": anchors,
            "schedule": {"profile": SCHEDULE_PROFILE, "gripperIds": [item["id"] for item in anchors], "knots": knots}}
        return source

    def run_control(self, root, source, enabled=True, extra=()):
        canonical, placement, output = root / "canonical.json", root / "placement.json", root / "run"
        canonical.write_text(json.dumps(source, allow_nan=False))
        placement.write_text(json.dumps({"placedMeters": source["placedMeters"],
            "canonicalDigest": hashlib.sha256(canonical.read_bytes()).hexdigest()}))
        command = [sys.executable, str(Path(__file__).with_name("spike-contact-continuation.py")),
            "--canonical", str(canonical), "--placement", str(placement), "--output", str(output),
            "--contact-model", "rest-filtered", "--ccd-profile", "temporal-separation-tight-inclusion",
            "--activation-distance-m", ".0001", "--minimum-distance-m", ".0001", "--pressure-pa", "10000",
            "--target-fraction", "1", "--subdivisions", "4", "--step-seconds", ".008",
            "--cpu-limit-seconds", "20", "--wall-limit-seconds", "30", "--max-attempts", "64"]
        if enabled:
            command.append("--material-grippers")
        command.extend(extra)
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        return result, output, json.loads((output / "report.json").read_text())

    def replay(self, output):
        return subprocess.run([sys.executable, str(Path(__file__).with_name("replay-rest-filtered-continuation.py")),
            str(output), "--cpu-limit-seconds", "30"], capture_output=True, text=True, timeout=40)

    def verified_replay(self, output, report):
        self.assertEqual(report["adaptive"]["targetInterpolation"], "linear sewing progress over the original physical interval")
        self.assertEqual(report["adaptive"]["gripperInterpolation"],
            "captured piecewise-linear material-point targets and activation over the original physical interval")
        replay = self.replay(output)
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        verified = json.loads((output / "verified-replay.json").read_text())
        self.assertFalse(verified["accepted"])
        self.assertEqual(verified["completed"], report["completed"])
        self.assertEqual(len(verified["states"]), len(report["acceptedStateArtifacts"]))
        self.assertEqual(verified["reportSha256"], hashlib.sha256((output / "report.json").read_bytes()).hexdigest())
        self.assertEqual(verified["gripperVerifierSha256"], hashlib.sha256(
            Path(__file__).with_name("solver_gripper_replay.py").read_bytes()).hexdigest())
        self.assertTrue(verified["initialGripperVerification"]["verified"])
        self.assertEqual(verified["initialGripperVerification"]["initialEnergyJoules"], report["initialGripperEnergyJoules"])
        for state in verified["states"]:
            self.assertTrue(state["candidateCoverageVerification"]["verified"])
            self.assertTrue(state["physicalPathPass"])
            self.assertLessEqual(state["recomputedResidualN"], 1e-6)
            self.assertIn("gripperVerification", state)
        return verified

    def rehash_final_step(self, output, report, mutate):
        """Alter semantics while retaining a valid state/journal hash chain."""
        descriptor = report["acceptedStateArtifacts"][-1]
        state_path = output / descriptor["path"]
        state = json.loads(state_path.read_text())
        mutate(state["record"]["step"])
        state_path.write_bytes(json_bytes(state))
        changed_descriptor = {"path": descriptor["path"], "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()}
        previous = None
        for path in sorted(output.glob("attempt-*.json")):
            payload = json.loads(path.read_text())["payload"]
            payload["previousSha256"] = previous
            if payload["kind"] == "outcome" and payload["data"]["attemptId"] == state["record"]["attemptId"]:
                payload["data"] = copy.deepcopy(state["record"]) | {"acceptedState": changed_descriptor}
            envelope = {"payload": payload, "sha256": hashlib.sha256(json_bytes(payload)).hexdigest()}
            path.write_bytes(json_bytes(envelope))
            previous = hashlib.sha256(path.read_bytes()).hexdigest()
        recovered = recover_attempt_journal(output, report)
        self.assertEqual(recovered["attemptJournal"]["errors"], [])
        self.assertTrue(recovered["adaptive"]["complete"])
        report.update(recovered)

    def test_engage_move_release_captures_initial_energy_and_accepted_only_work(self):
        source = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            result, output, report = self.run_control(Path(directory), source)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + str(report.get("failure")))
            self.assertTrue(report["terminal"] and report["completed"] and report["captureIntegrityVerified"])
            self.assertFalse(report["accepted"])
            self.assertEqual(report["attemptJournal"]["errors"], [])
            self.assertEqual(report["gripperActuation"], source["gripperActuation"])
            self.assertEqual(report["gripperBinding"]["meshSha256"], mesh_identity(source))
            self.assertEqual(report["initialGripperActivation"], [.25] * 3)
            self.assertAlmostEqual(report["initialGripperEnergyJoules"], 3 * .5 * .02 * .25 * .00025 ** 2, delta=1e-24)
            self.assertIsNone(report["seamGapsMm"])
            self.assertIsNone(report["finalMaximumAnchorGapM"])
            self.assertEqual(json.loads((output / "canonical.json").read_text()), source)
            for name, expected in report["sourceDigests"].items():
                self.assertEqual(hashlib.sha256((output / "source-snapshot" / name).read_bytes()).hexdigest(), expected)
            states = []
            for descriptor in report["acceptedStateArtifacts"]:
                path = output / descriptor["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), descriptor["sha256"])
                state = json.loads(path.read_text())
                states.append(state)
                self.assertFalse(state["accepted"])
                self.assertIn("energyBalance", state["record"]["step"])
                momentum = state["record"]["step"]["gripperMomentum"]
                self.assertLessEqual(max(abs(value) for value in momentum["residualNs"]), momentum["toleranceNs"])
            self.assertGreaterEqual(len(states), 4)
            self.assertEqual(states[-1]["record"]["endFraction"], 1.)
            self.assertEqual(states[-1]["record"]["step"]["gripperActivation"], [0.] * 3)
            self.assertEqual(states[-1]["record"]["step"]["energyBalance"]["gripperAfterJoules"], 0.)
            self.assertGreater(np.linalg.norm(states[-1]["velocitiesMetersPerSecond"]), 0.)
            accepted = report["adaptive"]["acceptedSteps"]
            summary = report["gripperWorkSummary"]
            self.assertEqual(summary["acceptedSteps"], len(accepted))
            for key, value in summary.items():
                if key.endswith("Joules"):
                    self.assertEqual(value, math.fsum(record["step"]["energyBalance"][key] for record in accepted))
            self.assertGreater(summary["gripperReleaseEnergyRemovedJoules"], 0.)
            durable_outcomes = []
            for descriptor in report["attemptJournal"]["events"]:
                path = output / descriptor["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), descriptor["sha256"])
                payload = json.loads(path.read_text())["payload"]
                if payload["kind"] == "outcome" and payload["data"]["outcome"] == "accepted":
                    durable_outcomes.append(payload["data"])
                    self.assertIn("energyBalance", payload["data"]["step"])
                    self.assertIn("gripperMomentum", payload["data"]["step"])
            self.assertEqual(len(durable_outcomes), len(states))
            self.assertEqual(durable_outcomes, accepted)
            verified = self.verified_replay(output, report)
            self.assertTrue(verified["finalStateVerified"])
            for key, value in summary.items():
                if key.endswith("Joules") or key == "acceptedSteps":
                    self.assertEqual(verified["verifiedGripperWorkSummary"][key], value)

    def test_replay_rejects_rehashed_work_controls_and_momentum_evidence(self):
        step_attacks = [
            ("missing work", lambda step: step.pop("energyBalance")),
            ("altered release", lambda step: step["energyBalance"].__setitem__("gripperReleaseEnergyRemovedJoules", 0.)),
            ("altered target", lambda step: step["gripperTargetsMeters"][0].__setitem__(2, .1)),
            ("altered activation", lambda step: step["gripperActivation"].__setitem__(0, 1.)),
            ("inflated impulse tolerance", lambda step: step["gripperMomentum"].__setitem__("toleranceNs", 1.)),
        ]
        report_attacks = [
            ("initial energy", lambda report: report.__setitem__("initialGripperEnergyJoules", 0.)),
            ("missing summary", lambda report: report.pop("gripperWorkSummary")),
            ("altered work total", lambda report: report["gripperWorkSummary"].__setitem__("gripperParameterWorkJoules", 1.)),
            ("one-ulp work total", lambda report: report["gripperWorkSummary"].__setitem__("gripperParameterWorkJoules",
                math.nextafter(report["gripperWorkSummary"]["gripperParameterWorkJoules"], math.inf))),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, output, report = self.run_control(root, self.fixture())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for index, (name, mutate) in enumerate(step_attacks + report_attacks):
                with self.subTest(attack=name):
                    attacked = root / f"attack-{index}"
                    shutil.copytree(output, attacked)
                    changed = copy.deepcopy(report)
                    if index < len(step_attacks):
                        self.rehash_final_step(attacked, changed, mutate)
                    else:
                        mutate(changed)
                    (attacked / "report.json").write_bytes(json_bytes(changed))
                    replay = self.replay(attacked)
                    self.assertNotEqual(replay.returncode, 0, replay.stdout + replay.stderr)
                    self.assertFalse((attacked / "verified-replay.json").exists())

    def test_replay_reconstructs_only_the_durable_interrupted_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output, report = self.run_control(Path(directory), self.fixture())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Simulate a worker lost after the second attempt start: its first
            # accepted outcome is durable; no following outcome is durable.
            starts, truncate = 0, False
            for path in sorted(output.glob("attempt-*.json")):
                if truncate:
                    path.unlink()
                elif json.loads(path.read_text())["payload"]["kind"] == "start":
                    starts += 1
                    truncate = starts == 2
            report.update(recover_attempt_journal(output, report, interruption="synthetic interrupted worker"))
            report["completed"] = False
            report.pop("stateArtifact", None)
            report.pop("gripperWorkSummary", None)
            self.assertEqual(report["attemptJournal"]["errors"], [])
            self.assertEqual(len(report["adaptive"]["acceptedSteps"]), 1)
            self.assertEqual(len(report["adaptive"]["interruptedSteps"]), 1)
            (output / "report.json").write_bytes(json_bytes(report))
            verified = self.verified_replay(output, report)
            self.assertFalse(verified["completed"] or verified["finalStateVerified"])
            self.assertEqual(verified["completedFraction"], .25)
            self.assertEqual(verified["verifiedGripperWorkSummary"]["acceptedSteps"], 1)
            first = report["adaptive"]["acceptedSteps"][0]["step"]["energyBalance"]
            for key, value in verified["verifiedGripperWorkSummary"].items():
                if key.endswith("Joules"):
                    self.assertEqual(value, first[key])

    @staticmethod
    def contact_fixture():
        # A small numerical contact fixture, not a garment or a construction phase.
        rest = np.array([[0., 0., 0.], [.03125, 0., 0.], [0., .03125, 0.],
                         [0., 0., .001953125], [.03125, 0., .001953125], [0., .03125, .001953125]])
        source = {"restMeters": rest.tolist(), "placedMeters": rest.tolist(), "triangles": [0, 1, 2, 3, 4, 5],
                  "instanceOffsets": {"lower:shell": 0, "upper:shell": 3}, "embeddedConstraints": {"constraints": []}}
        anchors = [{"id": f"grip-{vertex}", "instanceId": "lower:shell" if vertex < 3 else "upper:shell",
                    "triangleIndex": vertex // 3, "weights": [float(index == vertex % 3) for index in range(3)],
                    "stiffnessNPerM": .5} for vertex in range(6)]
        knots = []
        for fraction, inward, activation in ((0., 0., 1.), (.25, .00055, 1.),
                (.5, .00075, 1.), (.75, .00075, .5), (1., .00075, 0.)):
            targets = rest.copy()
            targets[:3, 2] += inward
            targets[3:, 2] -= inward
            knots.append({"fraction": fraction, "targetsMeters": targets.tolist(), "activation": [activation] * 6})
        source["gripperActuation"] = {"profile": PROFILE, "accepted": False, "meshSha256": mesh_identity(source),
            "anchors": anchors, "schedule": {"profile": SCHEDULE_PROFILE,
                "gripperIds": [anchor["id"] for anchor in anchors], "knots": knots}}
        return source

    def test_two_free_layers_enter_contact_then_release_and_replay(self):
        source = self.contact_fixture()
        rest = np.asarray(source["restMeters"])
        with tempfile.TemporaryDirectory() as directory:
            result, output, report = self.run_control(Path(directory), source, extra=(
                "--activation-distance-m", ".001", "--step-seconds", ".08", "--subdivisions", "8"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + str(report.get("failure"))
                             + str(report.get("adaptive", {}).get("reason")))
            self.assertEqual(json.loads((output / "canonical.json").read_text()), source)
            self.assertEqual(report["initialGripperEnergyJoules"], 0.)
            states = [json.loads((output / artifact["path"]).read_text()) for artifact in report["acceptedStateArtifacts"]]
            self.assertGreater(max(state["record"]["step"]["energyBalance"]["contactAfterJoules"] for state in states), 0.)
            from solver_rest_filtered_contact import RestFilteredSurfaceContact
            contact = RestFilteredSurfaceContact(rest, np.array([[0, 1, 2], [3, 4, 5]]),
                activation_distance_m=.001, minimum_distance_m=.0001, stiffness=10000,
                ccd_profile="temporal-separation-tight-inclusion")
            self.assertEqual(contact.energy(rest), 0.)
            largest = max(states, key=lambda state: state["record"]["step"]["energyBalance"]["contactAfterJoules"])
            positions = np.asarray(largest["positionsMeters"])
            forces = -contact.gradient(positions)
            self.assertLess(np.sum(forces[:3, 2]), 0.)
            self.assertGreater(np.sum(forces[3:, 2]), 0.)
            np.testing.assert_allclose(np.sum(forces, axis=0), 0., rtol=0, atol=1e-14)
            self.assertLess(np.min(positions[3:, 2]) - np.max(positions[:3, 2]), .0011)
            for state in states:
                q = np.asarray(state["positionsMeters"])
                self.assertGreater(np.min(q[3:, 2]) - np.max(q[:3, 2]), .0001)
            self.assertEqual(states[-1]["record"]["step"]["gripperActivation"], [0.] * 6)
            self.assertGreater(report["gripperWorkSummary"]["gripperReleaseEnergyRemovedJoules"], 0.)
            verified = self.verified_replay(output, report)
            self.assertGreater(max(state["contactEnergyJ"] for state in verified["states"]), 0.)
            np.testing.assert_array_equal(contact.rest_positions, rest)

    def test_missing_flag_recipe_or_both_cannot_run_single_empty_seam_instance(self):
        original = self.fixture()
        missing = copy.deepcopy(original)
        del missing["gripperActuation"]
        for source, enabled in ((original, False), (missing, True), (missing, False)):
            with self.subTest(enabled=enabled, has_recipe="gripperActuation" in source), tempfile.TemporaryDirectory() as directory:
                result, _, report = self.run_control(Path(directory), source, enabled)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertFalse(report["completed"] or report["accepted"])
                self.assertTrue(report["terminal"])
                self.assertEqual(report["acceptedStateArtifacts"], [])
                self.assertIn("failure", report)

    def test_invalid_source_binding_mesh_and_schedule_reject_before_motion(self):
        def mixed_boolean(source, field):
            if field == "triangles":
                source[field][0] = False
            else:
                source[field][0][0] = False
            source["gripperActuation"]["meshSha256"] = mesh_identity(source)

        mutations = [
            ("accepted claim", lambda source: source["gripperActuation"].__setitem__("accepted", True)),
            ("mesh digest", lambda source: source["gripperActuation"].__setitem__("meshSha256", "0" * 64)),
            ("foreign instance", lambda source: source["gripperActuation"]["anchors"][0].__setitem__("instanceId", "foreign:shell")),
            ("foreign triangle", lambda source: source["gripperActuation"]["anchors"][0].__setitem__("triangleIndex", 1)),
            ("wrong support", lambda source: source["gripperActuation"]["anchors"][0].__setitem__("weights", [0., 0., 0.])),
            ("schedule identity", lambda source: source["gripperActuation"]["schedule"]["gripperIds"].__setitem__(0, "foreign")),
            ("misaligned knot", lambda source: source["gripperActuation"]["schedule"]["knots"][1].__setitem__("fraction", .125)),
            ("unknown recipe field", lambda source: source["gripperActuation"].__setitem__("verified", True)),
            ("boolean rest coordinate", lambda source: mixed_boolean(source, "restMeters")),
            ("boolean triangle index", lambda source: mixed_boolean(source, "triangles")),
        ]
        for name, mutate in mutations:
            source = self.fixture()
            mutate(source)
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                result, _, report = self.run_control(Path(directory), source)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertFalse(report["completed"] or report["accepted"])
                self.assertEqual(report["acceptedStateArtifacts"], [])
                self.assertIn("failure", report)

    def test_unsupported_schedule_refinement_bound_rejects_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("spike-contact-continuation.py")),
                "--canonical", "missing", "--placement", "missing", "--output", str(output),
                "--activation-distance-m", ".0001", "--minimum-distance-m", ".0001", "--pressure-pa", "10000",
                "--target-fraction", "1", "--material-grippers", "--subdivisions", "4096",
                "--max-attempts", "4096", "--max-depth", "30"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
