"""Independent first-turn reporter admission checks; no dynamics or captures.

Fresh Linux source fixtures exercise the actual reference validator. The
comparison tests mock only evidence loading, after independently admitting
the source references; they do not stand in for a replay-proof test.
"""

import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from solver_gripper_replay import derive_grippers
from solver_sewing_replay import derive_sewing


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("first_turn_report_review", SCRIPTS / "analyze-binding-first-turn.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rebind_sewing(source):
    # Repair the enclosing hash so attacks reach the semantic admission gate.
    source["sewingActuation"]["sourceSha256"] = sha(
        {key: value for key, value in source.items() if key != "sewingActuation"})


def gripper_source(weights):
    source = {"restMeters": [[2., 0., 0.], [2., 1., 0.], [2., 0., 1.]],
              "triangles": [0, 1, 2], "instanceOffsets": {"cloth": 0}}
    source["gripperActuation"] = {
        "profile": "captured-material-grippers-v1", "accepted": False,
        "meshSha256": sha(source),
        "anchors": [{"id": "g", "instanceId": "cloth", "triangleIndex": 0,
                     "weights": weights, "stiffnessNPerM": 1e12}],
        "schedule": {"profile": "material-gripper-target-activation-v1", "gripperIds": ["g"],
            "knots": [{"fraction": time, "targetsMeters": [[3., 0., 0.]],
                       "activation": [0. if time >= .875 else 1.]}
                      for time in (0., .75, .875, 1.)]}}
    return source


def timing_fixture(identity):
    # Literal reviewed experiment constants, independent of the implementation.
    duration, count = {"original-128ms-v1": (.128, 64), "fourfold-512ms-v1": (.512, 256)}[identity]
    declaration = {"profile": "binding-first-turn-time-v1", "id": identity,
                   "durationSeconds": duration, "nominalStepSeconds": .002}
    return {"subdivisions": count, "timePolicy": declaration}, {"subdivisions": count, "step_seconds": duration}


class BindingFirstTurnTimingReportTests(unittest.TestCase):
    def valid_baselines(self):
        for identity in ("original-128ms-v1", "fourfold-512ms-v1"):
            metadata, arguments = timing_fixture(identity)
            before = encoded([metadata, arguments])
            for run in (None, arguments):
                self.assertEqual(encoded(REPORT.validate_time_policy(metadata, run)), encoded(metadata["timePolicy"]))
            self.assertEqual(encoded([metadata, arguments]), before)

    def test_explicit_original_and_fourfold_policies_are_admitted_without_mutation(self):
        self.valid_baselines()
        first, first_args = timing_fixture("original-128ms-v1")
        fourth, fourth_args = timing_fixture("fourfold-512ms-v1")
        self.assertEqual(fourth_args["step_seconds"], 4 * first_args["step_seconds"])
        self.assertEqual(fourth["subdivisions"], 4 * first["subdivisions"])
        self.assertEqual(first_args["step_seconds"] / first["subdivisions"], .002)
        self.assertEqual(fourth_args["step_seconds"] / fourth["subdivisions"], .002)

    def test_legacy_missing_policy_is_only_the_original_duration_and_grid(self):
        self.valid_baselines()
        explicit, arguments = timing_fixture("original-128ms-v1")
        legacy = {"subdivisions": 64}
        self.assertEqual(REPORT.validate_time_policy(legacy), explicit["timePolicy"])
        self.assertEqual(REPORT.validate_time_policy(legacy, arguments), explicit["timePolicy"])
        for metadata, run in (({"subdivisions": 256}, None),
                              (legacy, {"subdivisions": 256, "step_seconds": .512}),
                              (legacy, {"subdivisions": 64, "step_seconds": .512}),
                              ({"subdivisions": 64, "timePolicy": None}, arguments)):
            with self.subTest(metadata=metadata, run=run), self.assertRaises(ValueError):
                REPORT.validate_time_policy(metadata, run)

    def test_policy_schema_ids_and_literal_constants_are_strict(self):
        self.valid_baselines()
        for identity in ("original-128ms-v1", "fourfold-512ms-v1"):
            metadata, arguments = timing_fixture(identity)
            declaration = metadata["timePolicy"]
            alternatives = [None, [], identity, False, {}, declaration | {"extra": 1}]
            alternatives.extend({key: value for key, value in declaration.items() if key != missing}
                                for missing in declaration)
            alternatives.extend(declaration | {key: value} for key, values in {
                "profile": ["binding-first-turn-time-v2", False, 1, None],
                "id": ["unknown", "original-128ms-v1 ", "fourfold-512ms-v2", True, 1, None],
                "durationSeconds": [True, "0.128", None, math.nan, math.inf,
                                    math.nextafter(declaration["durationSeconds"], math.inf)],
                "nominalStepSeconds": [False, "0.002", 0, .004, math.nextafter(.002, 0.)],
            }.items() for value in values)
            for number, candidate in enumerate(alternatives):
                with self.subTest(policy=identity, attack=number), self.assertRaises((ValueError, TypeError)):
                    REPORT.validate_time_policy(metadata | {"timePolicy": candidate}, arguments)

    def test_metadata_and_run_counts_require_integers_and_exact_agreement(self):
        self.valid_baselines()
        for identity in ("original-128ms-v1", "fourfold-512ms-v1"):
            metadata, arguments = timing_fixture(identity)
            count = metadata["subdivisions"]
            for value in (None, True, False, str(count), float(count), np.int64(count), count * 2, count // 2):
                with self.subTest(policy=identity, metadata_count=value), self.assertRaises(ValueError):
                    REPORT.validate_time_policy(metadata | {"subdivisions": value}, arguments)
                with self.subTest(policy=identity, run_count=value), self.assertRaises(ValueError):
                    REPORT.validate_time_policy(metadata, arguments | {"subdivisions": value})
            for key in ("subdivisions", "step_seconds"):
                with self.subTest(policy=identity, missing=key), self.assertRaises(ValueError):
                    REPORT.validate_time_policy(metadata, {k: v for k, v in arguments.items() if k != key})

    def test_duration_or_cross_policy_mismatch_rejects_even_at_matching_step_ratio(self):
        self.valid_baselines()
        original, original_run = timing_fixture("original-128ms-v1")
        fourfold, fourfold_run = timing_fixture("fourfold-512ms-v1")
        for metadata, arguments in ((original, fourfold_run), (fourfold, original_run)):
            self.assertEqual(arguments["step_seconds"] / arguments["subdivisions"], .002)
            with self.assertRaises(ValueError):
                REPORT.validate_time_policy(metadata, arguments)
        for metadata, arguments in ((original, original_run), (fourfold, fourfold_run)):
            duration = arguments["step_seconds"]
            for value in (True, False, None, str(duration), math.nan, math.inf, -duration,
                          duration * 2, math.nextafter(duration, 0.), math.nextafter(duration, math.inf)):
                with self.subTest(duration=value), self.assertRaises(ValueError):
                    REPORT.validate_time_policy(metadata, arguments | {"step_seconds": value})


class BindingFirstTurnGripReportTests(unittest.TestCase):
    def check_exact_grip(self, weights):
        source = gripper_source(weights)
        record = derive_grippers(source, 64)
        positions = np.asarray(source["restMeters"])
        before = positions.copy()
        observed = REPORT.grip_observations(record, positions, Fraction(1, 2))
        # Direct original-input rational oracle, independent of reporter helpers.
        anchor = [sum((Fraction(weight) * Fraction(float(positions[vertex, axis]))
                       for vertex, weight in enumerate(weights)), Fraction()) for axis in range(3)]
        error = [value - goal for value, goal in zip(anchor, (3, 0, 0))]
        force = [-Fraction(10 ** 12) * value for value in error]
        weight_sum = sum(map(Fraction, weights), Fraction())
        expected_total = [float(sum((Fraction(weight) * component for weight in weights), Fraction()))
                          for component in force]
        self.assertEqual(observed["grippers"][0]["positionMeters"], list(map(float, anchor)))
        self.assertEqual(observed["grippers"][0]["forceNewtons"], list(map(float, force)))
        self.assertEqual(observed["totalClothForceNewtons"], expected_total)
        self.assertEqual(observed["energyJoules"], float(Fraction(10 ** 12, 2) * sum(x * x for x in error)))
        np.testing.assert_array_equal(positions, before)
        return observed, record, positions, weight_sum

    def test_unnormalized_admitted_weights_use_nodal_force_sum(self):
        self.check_exact_grip([.5, .25, .25])  # Positive exact-normalized baseline.
        weights = [.5, .25, .2500000000005]
        observed, _, _, weight_sum = self.check_exact_grip(weights)
        self.assertNotEqual(weight_sum, 1)
        self.assertLess(abs(float(weight_sum - 1)), 1e-12)
        self.assertNotEqual(observed["totalClothForceNewtons"][0], observed["grippers"][0]["forceNewtons"][0])

    def test_release_observation_requires_exact_zero_controls_force_and_energy(self):
        _, record, positions, _ = self.check_exact_grip([.5, .25, .25])
        for fraction in (Fraction(7, 8), Fraction(15, 16), Fraction(1)):
            observed = REPORT.grip_observations(record, positions, fraction)
            self.assertEqual(observed["energyJoules"], 0.)
            self.assertEqual(observed["totalClothForceNewtons"], [0., 0., 0.])
            self.assertEqual(observed["grippers"][0]["forceNewtons"], [0., 0., 0.])
        source = gripper_source([.5, .25, .25])
        source["gripperActuation"]["schedule"]["knots"][-2]["activation"] = [1.]
        with self.assertRaisesRegex(ValueError, "Released grippers"):
            REPORT.grip_observations(derive_grippers(source, 64), positions, Fraction(7, 8))


@unittest.skipUnless(sys.platform.startswith("linux"), "Fresh source coefficients use the pinned Linux runtime")
class BindingFirstTurnReferenceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

        def run(name, *arguments):
            result = subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, arguments)],
                cwd=directory, env=environment, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)

        run("prepare-cuff-source.py", "--output", directory / "parent")
        run("prepare-cuff-construction.py", "--source-canonical", directory / "parent/canonical.json",
            "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1",
            "--output", directory / "unit")
        cls.timed_sources = {}
        for policy in ("original-128ms-v1", "fourfold-512ms-v1"):
            cls.timed_sources[policy] = {}
            for angle in (0, 3):
                output = directory / f"{policy}-{angle}"
                run("prepare-binding-first-turn.py", "--source-unit", directory / "unit/unit.json",
                    "--output", output, "--angle-degrees", angle, "--time-policy", policy)
                cls.timed_sources[policy][angle] = json.loads((output / "canonical.json").read_bytes())
        cls.sources = cls.timed_sources["original-128ms-v1"]

    def reference(self, source):
        subdivisions = source["bindingFirstTurnDiagnostic"]["subdivisions"]
        return REPORT.validate_reference(source, np.asarray(source["placedMeters"]),
            derive_sewing(source, subdivisions, sewing_mode="distance"), derive_grippers(source, subdivisions))

    def valid_baseline(self):
        zero, driven = (self.reference(self.sources[angle]) for angle in (0, 3))
        self.assertEqual(zero, driven)
        self.assertEqual(len(zero), 64)
        return zero

    def rejected_attacks(self, attacks):
        self.valid_baseline()
        for name, attack in attacks:
            source = copy.deepcopy(self.sources[3])
            attack(source)
            rebind_sewing(source)
            # Both independent records must still derive, preventing a stale
            # source hash or unrelated malformed record from passing this test.
            sewing = derive_sewing(source, 64, sewing_mode="distance")
            grippers = derive_grippers(source, 64)
            with self.subTest(attack=name), self.assertRaises((ValueError, AssertionError)):
                REPORT.validate_reference(source, np.asarray(source["placedMeters"]), sewing, grippers)

    def test_valid_zero_and_three_degree_references_normalize_identically(self):
        before = encoded(self.sources)
        self.valid_baseline()
        self.assertEqual(encoded(self.sources), before)

    def test_fresh_explicit_timing_pairs_and_original_legacy_reference(self):
        self.valid_baseline()
        matched = {}
        for identity, sources in self.timed_sources.items():
            expected, arguments = timing_fixture(identity)
            for angle, source in sources.items():
                with self.subTest(policy=identity, angle=angle):
                    metadata = source["bindingFirstTurnDiagnostic"]
                    self.assertEqual(encoded(metadata["timePolicy"]), encoded(expected["timePolicy"]))
                    self.assertEqual(REPORT.validate_time_policy(metadata, arguments), expected["timePolicy"])
            matched[identity] = self.reference(sources[0])
            self.assertEqual(matched[identity], self.reference(sources[3]))
        # Time is a real experimental difference, not one of the normalized
        # angle-dependent fields allowed within a matched pair.
        self.assertNotEqual(matched["original-128ms-v1"], matched["fourfold-512ms-v1"])
        original = copy.deepcopy(self.sources[3])
        del original["bindingFirstTurnDiagnostic"]["timePolicy"]
        rebind_sewing(original)
        self.assertEqual(len(self.reference(original)), 64)
        fourfold = copy.deepcopy(self.timed_sources["fourfold-512ms-v1"][3])
        del fourfold["bindingFirstTurnDiagnostic"]["timePolicy"]
        rebind_sewing(fourfold)
        with self.assertRaisesRegex(ValueError, "grid differs"):
            self.reference(fourfold)

    def test_reference_axis_tangent_range_and_initial_placement_forgery_reject(self):
        self.rejected_attacks([
            ("source axis", lambda s: s["bindingFirstTurnDiagnostic"]["sourceAxisEndpointsMeters"][0].__setitem__(0, .021)),
            ("receiver axis", lambda s: s["bindingFirstTurnDiagnostic"]["receiverAxisEndpointsMeters"][0].__setitem__(2, .001)),
            ("axis origin", lambda s: s["bindingFirstTurnDiagnostic"]["axisOriginMeters"].__setitem__(2, .002)),
            ("axis direction", lambda s: s["bindingFirstTurnDiagnostic"]["axisDirection"].__setitem__(0, 0.)),
            ("binding tangent", lambda s: s["bindingFirstTurnDiagnostic"]["bindingTangentsRest"][0].__setitem__(1, -1.)),
            ("sleeve tangent", lambda s: s["bindingFirstTurnDiagnostic"]["sleeveTangentsRest"][0].__setitem__(0, 0.)),
            ("instance range", lambda s: s["bindingFirstTurnDiagnostic"]["instanceRanges"]["sleeve_left:shell"].__setitem__(0, 65)),
            ("initial placement", lambda s: s["placedMeters"][0].__setitem__(2, .051)),
        ])

    def test_actual_targets_stiffness_angular_units_and_hold_release_controls_reject(self):
        self.rejected_attacks([
            ("target one ulp", lambda s: s["gripperActuation"]["schedule"]["knots"][4]["targetsMeters"][0].__setitem__(
                0, math.nextafter(s["gripperActuation"]["schedule"]["knots"][4]["targetsMeters"][0][0], math.inf))),
            ("actual stiffness", lambda s: s["gripperActuation"]["anchors"][0].__setitem__("stiffnessNPerM", 2.)),
            ("declared stiffness", lambda s: s["bindingFirstTurnDiagnostic"].__setitem__("gripperStiffnessNPerM", 2.)),
            ("angular units", lambda s: s["bindingFirstTurnDiagnostic"].__setitem__("angleRadians", .1)),
            ("knot time", lambda s: s["gripperActuation"]["schedule"]["knots"][1].__setitem__("fraction", 1 / 32)),
            ("premature release", lambda s: s["gripperActuation"]["schedule"]["knots"][9].__setitem__("activation", [0.] * 3)),
            ("missing release", lambda s: s["gripperActuation"]["schedule"]["knots"][10].__setitem__("activation", [1.] * 3)),
        ])

    def test_normalized_away_clearance_and_contraction_forgery_reject(self):
        self.rejected_attacks([
            ("minimum clearance", lambda s: s["bindingFirstTurnDiagnostic"]["rigidReferenceClearanceMeters"].__setitem__(
                "minimumToSleevePlane", .001)),
            ("maximum clearance", lambda s: s["bindingFirstTurnDiagnostic"]["rigidReferenceClearanceMeters"].__setitem__(
                "maximumAboveSleevePlane", .001)),
            ("contraction", lambda s: s["bindingFirstTurnDiagnostic"]["schedule"].__setitem__("maximumChordRelativeContraction", 0.)),
        ])

    def admitted_comparison_record(self, source):
        metadata = source["bindingFirstTurnDiagnostic"]
        return {"angleDegrees": metadata["angleDegrees"], "sourceUnitSha256": metadata["sourceUnitCanonicalSha256"],
            "physicsArguments": {"subdivisions": 64}, "sourceDigests": {}, "sewingPathToleranceMeters": 1e-5,
            "initialPositionsSha256": sha(source["placedMeters"]),
            "sewingControls": {key: value for key, value in source["sewingActuation"].items() if key != "sourceSha256"},
            "gripperAnchors": source["gripperActuation"]["anchors"],
            "gripperActivationSchedule": [{"fraction": knot["fraction"], "activation": knot["activation"]}
                for knot in source["gripperActuation"]["schedule"]["knots"]],
            "initialGripperTargets": source["gripperActuation"]["schedule"]["knots"][0]["targetsMeters"],
            "matchedSourceSha256": self.reference(source), "declaration": metadata,
            "states": [{"fraction": 0.}, {"fraction": 1.}]}

    def test_frame_and_sewing_controls_are_not_normalized_out_of_matched_identity(self):
        self.valid_baseline()
        control = self.admitted_comparison_record(self.sources[0])
        driven = self.admitted_comparison_record(self.sources[3])
        with patch.object(REPORT, "load_control", side_effect=[control, driven]):
            result = REPORT.compare("fresh-control", "fresh-driven")
        self.assertEqual(result["exactCommonFractions"], [0., 1.])
        self.assertIs(result["accepted"], False)
        attacks = [
            ("binding frame", lambda s: s["bindingFirstTurnDiagnostic"]["bindingFrameFaces"].__setitem__(
                0, s["bindingFirstTurnDiagnostic"]["bindingFrameFaces"][1])),
            ("sleeve winding", lambda s: s["bindingFirstTurnDiagnostic"]["sleeveFrameFaces"][0].reverse()),
            ("target", lambda s: s["sewingActuation"]["finalTargetsMeters"].__setitem__(0, .002)),
            ("activation", lambda s: s["sewingActuation"]["schedule"]["knots"][-1]["activation"].__setitem__(slice(5, 10), [1.] * 5)),
        ]
        for name, attack in attacks:
            source = copy.deepcopy(self.sources[3])
            attack(source)
            rebind_sewing(source)
            altered = self.admitted_comparison_record(source)
            self.assertNotEqual(altered["matchedSourceSha256"], driven["matchedSourceSha256"])
            with self.subTest(attack=name), patch.object(REPORT, "load_control", side_effect=[control, altered]):
                with self.assertRaisesRegex(ValueError, "Matched controls differ"):
                    REPORT.compare("fresh-control", "fresh-altered")


if __name__ == "__main__":
    unittest.main()
