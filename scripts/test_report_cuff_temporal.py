"""Portable reporting regressions; synthetic arrays only, no ignored captures.

Run with the plotting environment used by report-cuff-temporal.py.
"""

import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("cuff_temporal_report", Path(__file__).with_name("report-cuff-temporal.py"))
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)
np = reporter.np


def zero_state():
    return {"q": np.zeros((66, 3)), "v": np.zeros((66, 3)), "a": np.zeros(24)}


def synthetic_pair():
    schedule = {"profile": "sewing-fold-progress-v1", "knots": [
        {"fraction": f, "sewingProgress": s, "foldProgress": a}
        for f, s, a in zip([0., .0625, .25, 1.], [0., 1., 1., 1.], [0., 0., 1., 1.])]}
    args = {"subdivisions": 256, "output": "coarse", "step_seconds": .256,
            "sewing_mode": "normal-offset", "activation_distance_m": .0001,
            "minimum_distance_m": .0001, "pressure_pa": 10000., "assembly_schedule": True,
            "fold_actuation": True, "max_evaluations": 100}
    source = {"assemblySchedule": schedule, "sewingFrames": {
        "recipe": "explicit-facing-source-parent-region-v2", "creaseFrameRegion": "body"}}
    profile = {"filteredPrimitivePairs": 0, "fullActivationM": .0001, "fullMinimumM": .0001, "pressurePa": 10000.}
    report = {"arguments": args, "canonicalSha256": "a" * 64, "placementSha256": "b" * 64,
              "sourceDigests": {"synthetic.py": "c" * 64}, "contactProfile": profile,
              "assemblySchedule": schedule, "foldActuation": {"synthetic": True}}
    summary = {"verificationDigests": {"replayScriptSha256": reporter.REPLAY_SHA256}}
    coarse = {"run": (report, source, None, None, None, None, summary),
              "trajectory": {Fraction(index, 256): zero_state() for index in range(257)}}
    fine = copy.deepcopy(coarse)
    fine["run"][0]["arguments"].update(subdivisions=512, output="fine")
    fine["trajectory"] = {Fraction(index, 512): zero_state() for index in range(513)}
    return coarse, fine


class TemporalEvidenceTests(unittest.TestCase):
    def test_missing_replay_is_mandatory_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "report.json").write_text(json.dumps({}))
            with self.assertRaises(FileNotFoundError) as failure:
                reporter.load_run(directory, directory / "irrelevant.log")
            self.assertEqual(Path(failure.exception.filename).name, "verified-replay.json")

    def test_analytic_hinge_angles_and_rigid_transform(self):
        angle_list = [-1.1, 0., .4, 1.2]
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for angle in angle_list:
            points = np.array([[0., 1., 0.], [0., -np.cos(angle), -np.sin(angle)], [0., 0., 0.], [1., 0., 0.]])
            for coordinates in (points, points @ rotation.T + [.2, -.4, .7]):
                self.assertAlmostEqual(reporter.hinge_angles(coordinates, [[0, 1, 2, 3]])[0], angle, places=14)
        with self.assertRaises(AssertionError):
            reporter.hinge_angles(np.zeros((4, 3)), [[0, 1, 2, 3]])

    def test_world_displacement_and_rms_have_analytic_values(self):
        coarse, fine = zero_state(), zero_state()
        fine["q"][:] = [.003, .004, 0.]
        fine["v"][:] = [-.006, .008, 0.]
        fine["a"][:12], fine["a"][12:] = .2, -.3
        values = reporter.state_difference(coarse, fine, Fraction(1, 2), .256)
        self.assertEqual(values["fraction"], "1/2")
        self.assertEqual(values["timeMs"], 128.)
        for name in ("maximumPositionDifferenceMm", "rmsPositionDifferenceMm"):
            self.assertAlmostEqual(values[name], 5.)  # A rigid alignment would incorrectly erase this.
        for name in ("maximumVelocityVectorDifferenceMPerS", "rmsVelocityVectorDifferenceMPerS", "maximumSpeedDifferenceMPerS"):
            self.assertAlmostEqual(values[name], .01)
        self.assertAlmostEqual(values["rmsAngleDifferenceDegrees"], np.degrees(np.sqrt(.065)))
        self.assertAlmostEqual(values["shellMaximumAngleDifferenceDegrees"], np.degrees(.2))
        self.assertAlmostEqual(values["facingMaximumAngleDifferenceDegrees"], np.degrees(.3))
        self.assertEqual(values["worstAngleHinge"], 12)

    def test_speed_and_velocity_vector_differences_are_distinct(self):
        coarse, fine = zero_state(), zero_state()
        coarse["v"][:] = [.01, 0., 0.]
        fine["v"][:] = [-.01, 0., 0.]
        values = reporter.state_difference(coarse, fine, Fraction(1), .256)
        self.assertEqual(values["maximumSpeedDifferenceMPerS"], 0.)
        self.assertEqual(values["maximumVelocityVectorDifferenceMPerS"], .02)

    def test_exact_shared_times_and_no_interpolation(self):
        coarse, fine = synthetic_pair()
        rows = reporter.compare(coarse, fine)
        self.assertEqual(len(rows), 257)
        self.assertTrue(all(row["maximumPositionDifferenceMm"] == 0 for row in rows))
        nearby = Fraction(1, 256) + Fraction(1, 2 ** 60)
        fine["trajectory"][nearby] = fine["trajectory"].pop(Fraction(1, 256))
        with self.assertRaises(AssertionError):
            reporter.compare(coarse, fine)

    def test_unauthorized_input_or_proof_differences_reject(self):
        changes = [
            lambda run: run["run"][0]["arguments"].__setitem__("fine_only_argument", 1),
            lambda run: run["run"][0]["arguments"].__setitem__("minimum_distance_m", .00005),
            lambda run: run["run"][0]["arguments"].__setitem__("max_evaluations", 101),
            lambda run: run["run"][0].__setitem__("placementSha256", "d" * 64),
            lambda run: run["run"][0]["sourceDigests"].__setitem__("synthetic.py", "e" * 64),
            lambda run: run["run"][0]["contactProfile"].__setitem__("pressurePa", 9999.),
            lambda run: run["run"][1]["sewingFrames"].__setitem__("creaseFrameRegion", "allowance"),
            lambda run: run["run"][-1]["verificationDigests"].__setitem__("replayScriptSha256", "f" * 64),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                coarse, fine = synthetic_pair()
                change(fine)
                with self.assertRaises(AssertionError):
                    reporter.validate_pair(coarse, fine)

    def test_coarse_coverage_and_expected_subdivision_counts_are_required(self):
        coarse, fine = synthetic_pair()
        with self.assertRaises(AssertionError):
            reporter.validate_pair(fine, coarse)
        coarse["trajectory"].pop(Fraction(1, 256))
        with self.assertRaises(AssertionError):
            reporter.validate_pair(coarse, fine)


if __name__ == "__main__":
    unittest.main()
