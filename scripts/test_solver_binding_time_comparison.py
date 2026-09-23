"""Lightweight comparison/window attacks using explicitly mocked reporter data.

These fixtures are NOT validated source units, simulation captures or replay
evidence. They exercise only the comparison layer after its reporter boundary.
Actual source/reference admission is covered by separate reporter tests.
"""

import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("binding_time_comparison_review",
    Path(__file__).with_name("analyze-binding-time-controls.py"))
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def mocked_pair(slower=False):
    count, duration, identity = ((256, .512, "fourfold-512ms-v1") if slower
                                 else (64, .128, "original-128ms-v1"))
    policy = {"profile": "binding-first-turn-time-v1", "id": identity,
              "durationSeconds": duration, "nominalStepSeconds": .002}
    source = {
        "scope": "MOCK ONLY: no source unit or replay is validated by this fixture",
        "restMeters": [[0., 0., 0.]], "triangles": [0, 1, 2],
        "material": {"density": .2, "stiffness": 10000.},
        "sourcePattern": {"identity": "mock-original-pattern"},
        "phasePlan": {"status": "unexecuted"},
        "placedMeters": [[0., 0., .001]],
        "sewingFrames": {"faces": [[0, 1, 2]], "sides": [1]},
        "sewingActuation": {"sourceSha256": "slower-derived-hash" if slower else "original-derived-hash",
                            "initialTargetsMeters": [.001], "finalTargetsMeters": [.001]},
        "gripperActuation": {"anchors": [{"weights": [.25, .25, .5], "stiffnessNPerM": 1.}],
            "schedule": {"knots": [{"fraction": fraction, "targetsMeters": [[.1, .2, .3]],
                                      "activation": [0. if fraction >= .875 else 1.]}
                for fraction in [index/16 for index in range(9)] + [.75, .875, 1.]]}},
        "bindingFirstTurnDiagnostic": {"subdivisions": count, "timePolicy": policy,
            "sleeveInstanceId": "mock-sleeve", "angleDegrees": 3.,
            "frameSelectionBindings": [{"sourceTriangle": 7}],
            "codeDigests": {"scripts/prepare-binding-first-turn.py": "new-generator" if slower else "old-generator",
                            "scripts/solver_material_grippers.py": "unchanged-runtime"},
            "runtime": {"numpy": "captured-version"}}}
    states = []
    for index in range(count+1):
        fraction = index/count
        state = {"fraction": fraction, "timeSeconds": fraction*duration,
            "maximumSpeedMetersPerSecond": 0., "contactEnergyJoules": 0.,
            "bindingGeometry": {"rows": [{"relativeTurnDegrees": 3., "current": {
                "absoluteDistanceErrorMeters": 0., "tangentOffsetMeters": 0., "crossTangentOffsetMeters": 0.}}
                for _ in range(5)],
                "strainByInstance": {"mock-sleeve": {"current": {
                    "principalStretchMinimum": 1., "principalStretchMaximum": 1.}}},
                "rigidFits": {"mock-sleeve": {"rotationDegrees": 0., "translationMeters": [0., 0., 0.],
                                                "maximumResidualMeters": 0.}},
                "relativeRigidMotion": {"rotationDegrees": 3.}},
            "surfaceSeparation": {"distanceMeters": .001},
            "gripperGeometry": {"grippers": [{"trackingErrorMeters": 0.} for _ in range(3)]}}
        if index:
            state["stepDurationSeconds"] = .002
        states.append(state)
    record = {"timePolicy": copy.deepcopy(policy), "nominalTimeStepPreserved": True, "states": states,
        "physicsArguments": {"step_seconds": duration, "subdivisions": count, "max_attempts": count*2,
            "cpu_limit_seconds": 100 if slower else 30, "wall_limit_seconds": 200 if slower else 60,
            "max_depth": 8, "pressure_pa": 10000., "minimum_distance_m": .0001,
            "sewing_mode": "distance", "material_grippers": True},
        "angleDegrees": 3., "sourceUnitSha256": "mock-source", "initialPositionsSha256": "mock-placement",
        "sewingControls": {"targets": [.001], "activation": [1., 0.]},
        "gripperAnchors": [{"weights": [.25, .25, .5]}],
        "gripperActivationSchedule": [{"fraction": 0., "activation": [1.]}, {"fraction": 1., "activation": [0.]}],
        "initialGripperTargets": [[.1, .2, .3]], "sourceDigests": {"solver.py": "same-physics"},
        "sewingPathToleranceMeters": 1e-5, "canonicalSha256": "mock-slower" if slower else "mock-original",
        "directory": "MOCK-slower" if slower else "MOCK-original", "reportSha256": "MOCK-report",
        "replaySha256": "MOCK-not-replayed", "exactContactLeaves": 0,
        "declaration": copy.deepcopy(source["bindingFirstTurnDiagnostic"]),
        "sewingWorkSummary": {"scope": "mock"}, "gripperWorkSummary": {"scope": "mock"}}
    return record, source


class BindingTimeComparisonTests(unittest.TestCase):
    def baseline(self):
        original, original_source = mocked_pair()
        slower, slower_source = mocked_pair(True)
        return original, slower, original_source, slower_source

    def test_normalization_removes_only_timing_derived_hash_and_generator_identity(self):
        original, slower, first, second = self.baseline()
        untouched = copy.deepcopy((original, slower, first, second))
        self.assertEqual(REPORT.time_independent_source(first), REPORT.time_independent_source(second))
        result = REPORT.validate_time_pair(original, slower, first, second)
        self.assertEqual(result["generatorSha256"], ["old-generator", "new-generator"])
        self.assertEqual(result["supervisionArguments"][1]["cpu_limit_seconds"], 100)
        self.assertEqual((original, slower, first, second), untouched)
        first["bindingFirstTurnDiagnostic"].pop("timePolicy")
        self.assertEqual(REPORT.time_independent_source(first), REPORT.time_independent_source(second))

    def test_full_target_source_material_frame_and_dependency_payload_is_retained(self):
        attacks = [
            ("noninitial target ULP", lambda s: s["gripperActuation"]["schedule"]["knots"][4]["targetsMeters"][0].__setitem__(0, math.nextafter(.1, math.inf))),
            ("knot activation", lambda s: s["gripperActuation"]["schedule"]["knots"][8]["activation"].__setitem__(0, .5)),
            ("tool stiffness", lambda s: s["gripperActuation"]["anchors"][0].__setitem__("stiffnessNPerM", 2.)),
            ("target distance", lambda s: s["sewingActuation"]["finalTargetsMeters"].__setitem__(0, .002)),
            ("rest point", lambda s: s["restMeters"][0].__setitem__(0, .0001)),
            ("placement", lambda s: s["placedMeters"][0].__setitem__(2, .002)),
            ("material", lambda s: s["material"].__setitem__("density", .3)),
            ("source", lambda s: s["sourcePattern"].__setitem__("identity", "changed")),
            ("phase", lambda s: s["phasePlan"].__setitem__("status", "complete")),
            ("winding", lambda s: s["sewingFrames"]["faces"][0].reverse()),
            ("frame", lambda s: s["bindingFirstTurnDiagnostic"]["frameSelectionBindings"][0].__setitem__("sourceTriangle", 8)),
            ("runtime", lambda s: s["bindingFirstTurnDiagnostic"]["runtime"].__setitem__("numpy", "different")),
            ("dependency", lambda s: s["bindingFirstTurnDiagnostic"]["codeDigests"].__setitem__("scripts/solver_material_grippers.py", "different")),
            ("unknown optional field", lambda s: s.__setitem__("unrecognizedPhysicalChange", True)),
        ]
        for label, change in attacks:
            original, slower, first, second = self.baseline()
            change(second)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "non-timing declaration"):
                REPORT.validate_time_pair(original, slower, first, second)

    def test_changed_physics_and_observed_identity_are_not_resource_exemptions(self):
        for field in ("pressure_pa", "minimum_distance_m", "max_depth", "material_grippers"):
            original, slower, first, second = self.baseline()
            slower["physicsArguments"][field] = False if field == "material_grippers" else 123
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "other numerical/physical"):
                REPORT.validate_time_pair(original, slower, first, second)
        for field in ("sourceDigests", "sewingPathToleranceMeters", "initialPositionsSha256", "sewingControls",
                      "gripperAnchors", "gripperActivationSchedule", "initialGripperTargets", "sourceUnitSha256"):
            original, slower, first, second = self.baseline()
            slower[field] = "forged"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                REPORT.validate_time_pair(original, slower, first, second)

    def test_complete_grids_reject_missing_duplicate_reordered_and_ulp_changed_states(self):
        attacks = [lambda r: r["states"].pop(10),
                   lambda r: r["states"].__setitem__(10, copy.deepcopy(r["states"][9])),
                   lambda r: r["states"].reverse(),
                   lambda r: r["states"][10].__setitem__("fraction", math.nextafter(r["states"][10]["fraction"], math.inf)),
                   lambda r: r["states"][10].__setitem__("timeSeconds", math.nextafter(r["states"][10]["timeSeconds"], math.inf)),
                   lambda r: r["states"][10].__setitem__("stepDurationSeconds", .001),
                   lambda r: r.__setitem__("nominalTimeStepPreserved", 1)]
        for side in (0, 1):
            for index, attack in enumerate(attacks):
                data = self.baseline()
                attack(data[side])
                with self.subTest(side=side, attack=index), self.assertRaises(ValueError):
                    REPORT.validate_time_pair(*data)

    def test_boolean_saved_grid_values_are_not_numerical_fractions_or_times(self):
        for side, index, field, value in ((0, 0, "fraction", False), (1, -1, "fraction", True),
                                          (0, 0, "timeSeconds", False)):
            data = self.baseline()
            data[side]["states"][index][field] = value
            with self.subTest(side=side, field=field), self.assertRaises(ValueError):
                REPORT.validate_time_pair(*data)

    def test_cross_duration_policy_or_grid_cannot_be_relabeled(self):
        original, slower, first, second = self.baseline()
        with self.assertRaises(ValueError):
            REPORT.validate_time_pair(slower, original, second, first)
        slower["physicsArguments"]["subdivisions"] = 64
        with self.assertRaises(ValueError):
            REPORT.validate_time_pair(original, slower, first, second)

    def test_fixed_sixteen_ms_window_is_separate_from_entire_fourfold_passive_tail(self):
        record, _ = mocked_pair(True)
        for fraction, turn, speed in ((29/32, 7., .4), (63/64, 4., .1)):
            state = record["states"][int(fraction*256)]
            state["bindingGeometry"]["rows"][0]["relativeTurnDegrees"] = turn
            state["maximumSpeedMetersPerSecond"] = speed
            state["gripperGeometry"]["grippers"][0]["trackingErrorMeters"] = speed/100
        result = REPORT.summarize(record)
        passive, fixed = result["passiveTail"], result["finalSixteenMilliseconds"]
        self.assertEqual((passive["startFraction"], passive["savedStates"]), (7/8, 33))
        self.assertEqual((fixed["startFraction"], fixed["savedStates"]), (31/32, 9))
        self.assertAlmostEqual(passive["endTimeSeconds"]-passive["startTimeSeconds"], .064)
        self.assertAlmostEqual(fixed["endTimeSeconds"]-fixed["startTimeSeconds"], .016)
        self.assertEqual(passive["maximumSameRowAngleRangeDegrees"], 4.)
        self.assertEqual(fixed["maximumSameRowAngleRangeDegrees"], 1.)
        self.assertEqual(passive["maximumSameRowEndpointChangeDegrees"], 0.)
        self.assertEqual(fixed["maximumSameRowEndpointChangeDegrees"], 0.)
        self.assertEqual(passive["maximumSpeedMetersPerSecond"], .4)
        self.assertEqual(fixed["maximumSpeedMetersPerSecond"], .1)
        self.assertEqual(passive["maximumAbsoluteCommandErrorDegrees"], 4.)
        self.assertEqual(fixed["maximumAbsoluteCommandErrorDegrees"], 1.)

    def test_original_passive_and_fixed_windows_coincide_and_range_is_not_net_change(self):
        record, _ = mocked_pair()
        for fraction, value in ((7/8, 2.), (15/16, 10.), (1., 5.)):
            record["states"][int(fraction*64)]["bindingGeometry"]["rows"][0]["relativeTurnDegrees"] = value
        result = REPORT.summarize(record)
        self.assertEqual(result["passiveTail"], result["finalSixteenMilliseconds"])
        self.assertEqual(result["passiveTail"]["maximumSameRowAngleRangeDegrees"], 8.)
        self.assertEqual(result["passiveTail"]["maximumSameRowEndpointChangeDegrees"], 3.)

    def test_same_row_range_does_not_measure_dispersion_between_rows(self):
        record, _ = mocked_pair(True)
        for state in record["states"]:
            for index, row in enumerate(state["bindingGeometry"]["rows"]):
                row["relativeTurnDegrees"] = float(index)
        result = REPORT.summarize(record)
        for name in ("allTime", "hold", "passiveTail", "finalSixteenMilliseconds"):
            self.assertEqual(result[name]["maximumSameRowAngleRangeDegrees"], 0.)
            self.assertEqual(result[name]["maximumSameRowEndpointChangeDegrees"], 0.)

    def test_window_requires_saved_exact_endpoints_without_interpolation(self):
        record, _ = mocked_pair(True)
        record["states"].pop(248)  # 31/32, the fixed sixteen-millisecond start.
        with self.assertRaisesRegex(ValueError, "Exact saved window endpoints"):
            REPORT.summarize(record)

    def test_compare_checks_canonical_bytes_again_after_reporter_observation(self):
        original, original_source = mocked_pair()
        slower, slower_source = mocked_pair(True)
        source_reads = [(original_source, original["canonicalSha256"]), (slower_source, "changed-after-observation")]
        with patch.object(REPORT.FIRST, "compare", side_effect=[{"control": original, "driven": original},
                                                               {"control": slower, "driven": slower}]), \
             patch.object(REPORT.FIRST, "document", side_effect=source_reads), \
             self.assertRaisesRegex(ValueError, "changed after observation"):
            REPORT.compare("mock-old-zero", "mock-old-driven", "mock-new-zero", "mock-new-driven")


if __name__ == "__main__":
    unittest.main()
