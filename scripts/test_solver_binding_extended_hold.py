"""Independent extended-hold input/report checks, without dynamics or replay.

Portable fixtures exercise timing and exact virtual-tool reporting. Linux
fixtures regenerate the actual source unit and reproduce captured preparation
bytes; none of these tests certifies motion, binding, or construction completion.
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

import numpy as np

from solver_gripper_input import bind_material_grippers
from solver_gripper_replay import derive_grippers
from solver_sewing_input import bind_sewing_activation
from solver_sewing_replay import derive_sewing


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SPEC = importlib.util.spec_from_file_location("binding_extended_hold_review",
                                            SCRIPTS / "analyze-binding-first-turn.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
HOLD_SPEC = importlib.util.spec_from_file_location("binding_extended_hold_comparison_review",
                                                 SCRIPTS / "analyze-binding-hold-control.py")
HOLD = importlib.util.module_from_spec(HOLD_SPEC)
HOLD_SPEC.loader.exec_module(HOLD)
FOURFOLD = "fourfold-512ms-v1"
EXTENDED = "extended-hold-1024ms-v1"
PHASES = {"turnFractions": [0., .25],
          "angularSampleFractions": [index / 32 for index in range(9)],
          "holdFractions": [.25, .875], "releaseFractions": [.875, .9375],
          "passiveTailFractions": [.9375, 1.]}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rebind_sewing(source):
    source["sewingActuation"]["sourceSha256"] = sha(
        {key: value for key, value in source.items() if key != "sewingActuation"})


def timing_fixture():
    return {"subdivisions": 512, "timePolicy": {
        "profile": "binding-first-turn-time-v1", "id": EXTENDED,
        "durationSeconds": 1.024, "nominalStepSeconds": .002}}, {
        "subdivisions": 512, "step_seconds": 1.024}


def tool_fixture():
    source = {"restMeters": [[2., 0., 0.], [2., 1., 0.], [2., 0., 1.]],
              "triangles": [0, 1, 2], "instanceOffsets": {"cloth": 0}}
    source["gripperActuation"] = {
        "profile": "captured-material-grippers-v1", "accepted": False,
        "meshSha256": sha(source),
        "anchors": [{"id": "g", "instanceId": "cloth", "triangleIndex": 0,
                     "weights": [.5, .25, .25], "stiffnessNPerM": 1e12}],
        "schedule": {"profile": "material-gripper-target-activation-v1", "gripperIds": ["g"],
            "knots": [{"fraction": fraction, "targetsMeters": [[3., 0., 0.]],
                       "activation": [activation]}
                      for fraction, activation in ((0., 1.), (.875, 1.), (.9375, 0.), (1., 0.))]}}
    return source


def mocked_comparison_pair(extended):
    """MOCK reporter output only: this is not source or replay evidence."""
    count, duration, identity = (512, 1.024, EXTENDED) if extended else (256, .512, FOURFOLD)
    turn, hold, release = (.25, .875, .9375) if extended else (.5, .75, .875)
    phases = {"turnFractions": [0., turn], "angularSampleFractions": [index * turn / 8 for index in range(9)],
              "holdFractions": [turn, hold], "releaseFractions": [hold, release],
              "passiveTailFractions": [release, 1.]}
    policy = {"profile": "binding-first-turn-time-v1", "id": identity,
              "durationSeconds": duration, "nominalStepSeconds": .002}
    source = {"scope": "MOCK ONLY: no source, cloth motion or replay is validated",
        "restMeters": [[0., 0., 0.]], "triangles": [0, 1, 2],
        "material": {"density": .2}, "sourcePattern": {"revision": "same-original"},
        "phasePlan": {"status": "unexecuted"}, "sewingFrames": {"faces": [[0, 1, 2]], "sides": [1]},
        "sewingActuation": {"sourceSha256": "new-derived-hash" if extended else "old-derived-hash",
                            "initialTargetsMeters": [.001], "finalTargetsMeters": [.001]},
        "gripperActuation": {"anchors": [{"weights": [.25, .25, .5], "stiffnessNPerM": 1.}],
            "schedule": {"knots": [{"fraction": fraction, "targetsMeters": [[.1, .2, .3]],
                                    "activation": [0. if fraction >= release else 1.]}
                                   for fraction in phases["angularSampleFractions"] + [hold, release, 1.]]}},
        "bindingFirstTurnDiagnostic": {"subdivisions": count, "timePolicy": policy, "schedule": phases,
            "codeDigests": {"scripts/prepare-binding-first-turn.py": "new-generator" if extended else "old-generator",
                            "scripts/solver_material_grippers.py": "same-physics"},
            "runtime": {"numpy": "same-runtime"}}}
    states = [{"fraction": index / count, "timeSeconds": index / count * duration,
               **({"stepDurationSeconds": .002} if index else {})} for index in range(count + 1)]
    record = {"timePolicy": copy.deepcopy(policy), "nominalTimeStepPreserved": True, "states": states,
        "physicsArguments": {"step_seconds": duration, "subdivisions": count,
            "max_attempts": count * 2, "cpu_limit_seconds": 100 if extended else 50,
            "wall_limit_seconds": 200 if extended else 100, "max_depth": 8,
            "minimum_distance_m": .0001, "sewing_mode": "distance", "material_grippers": True},
        "angleDegrees": 3., "sourceUnitSha256": "mock-source", "initialPositionsSha256": "mock-placement",
        "sewingControls": {"targets": [.001], "activation": [1., 0.]},
        "gripperAnchors": [{"weights": [.25, .25, .5]}], "initialGripperTargets": [[.1, .2, .3]],
        "sourceDigests": {"solver.py": "same-physics"}, "sewingPathToleranceMeters": 1e-5,
        "canonicalSha256": "mock-extended" if extended else "mock-previous"}
    return record, source


class ExtendedHoldMockedComparisonTests(unittest.TestCase):
    """Only comparison-layer admission, with explicit mocked reporter records."""

    def pair(self):
        previous, previous_source = mocked_comparison_pair(False)
        held, held_source = mocked_comparison_pair(True)
        return previous, held, previous_source, held_source

    def test_only_declared_timing_generator_identity_and_resource_budgets_normalize(self):
        values = self.pair()
        before = encoded(values)
        result = HOLD.validate_hold_pair(*values)
        self.assertEqual(encoded(values), before)
        self.assertEqual(result["generatorSha256"], ["old-generator", "new-generator"])
        self.assertEqual(HOLD.hold_independent_source(values[2]), HOLD.hold_independent_source(values[3]))
        self.assertNotEqual(result["supervisionArguments"][0], result["supervisionArguments"][1])

    def test_each_normalized_phase_and_knot_time_is_separately_admitted(self):
        for source_index in (2, 3):
            for phase in PHASES:
                values = self.pair()
                values[source_index]["bindingFirstTurnDiagnostic"]["schedule"][phase][0] = False
                with self.subTest(source=source_index, phase=phase), self.assertRaises(ValueError):
                    HOLD.validate_hold_pair(*values)
            for knot_index in range(12):
                values = self.pair()
                knot = values[source_index]["gripperActuation"]["schedule"]["knots"][knot_index]
                knot["fraction"] = math.nextafter(knot["fraction"], math.inf)
                with self.subTest(source=source_index, knot=knot_index), self.assertRaises(ValueError):
                    HOLD.validate_hold_pair(*values)
            for knot_index, value in ((0, False), (11, True)):
                values = self.pair()
                values[source_index]["gripperActuation"]["schedule"]["knots"][knot_index]["fraction"] = value
                with self.subTest(source=source_index, boolean_knot=knot_index), self.assertRaises(ValueError):
                    HOLD.validate_hold_pair(*values)

    def test_source_targets_frames_material_and_numerical_settings_cannot_change(self):
        mutations = (
            ("target", lambda source: source["gripperActuation"]["schedule"]["knots"][8]["targetsMeters"][0].__setitem__(0, math.nextafter(.1, 1.))),
            ("activation", lambda source: source["gripperActuation"]["schedule"]["knots"][9].__setitem__("activation", [.5])),
            ("anchor", lambda source: source["gripperActuation"]["anchors"][0].__setitem__("weights", [.5, .25, .25])),
            ("material", lambda source: source["material"].__setitem__("density", .21)),
            ("source", lambda source: source["sourcePattern"].__setitem__("revision", "changed")),
            ("frame", lambda source: source["sewingFrames"].__setitem__("sides", [-1])),
            ("rest", lambda source: source["restMeters"][0].__setitem__(0, 1e-12)),
            ("phase-completion", lambda source: source["phasePlan"].__setitem__("status", "complete")),
            ("solver", lambda source: source["bindingFirstTurnDiagnostic"]["codeDigests"].__setitem__("scripts/solver_material_grippers.py", "changed")),
            ("runtime", lambda source: source["bindingFirstTurnDiagnostic"]["runtime"].__setitem__("numpy", "changed")),
            ("unknown", lambda source: source.__setitem__("unreviewed-input", True)),
        )
        for label, mutation in mutations:
            values = self.pair()
            mutation(values[3])
            with self.subTest(change=label), self.assertRaises(ValueError):
                HOLD.validate_hold_pair(*values)
        for key, value in (("max_depth", 9), ("minimum_distance_m", .0002), ("material_grippers", 1),
                           ("sewing_mode", "vector"), ("new_solver_setting", 1)):
            values = self.pair()
            values[1]["physicsArguments"][key] = value
            with self.subTest(setting=key), self.assertRaises(ValueError):
                HOLD.validate_hold_pair(*values)

    def test_saved_256_and_512_grids_reject_missing_misdated_boolean_states(self):
        for record_index in (0, 1):
            attacks = (
                ("missing", lambda record: record["states"].pop(17)),
                ("fraction", lambda record: record["states"][17].__setitem__("fraction", .5)),
                ("time", lambda record: record["states"][17].__setitem__("timeSeconds", .1)),
                ("duration", lambda record: record["states"][17].__setitem__("stepDurationSeconds", .004)),
                ("bool-zero", lambda record: record["states"][0].__setitem__("fraction", False)),
                ("bool-final", lambda record: record["states"][-1].__setitem__("fraction", True)),
                ("bool-time", lambda record: record["states"][0].__setitem__("timeSeconds", False)),
                ("nominal-flag", lambda record: record.__setitem__("nominalTimeStepPreserved", 1)),
            )
            for label, mutate in attacks:
                values = self.pair()
                mutate(values[record_index])
                with self.subTest(record=record_index, attack=label), self.assertRaises(ValueError):
                    HOLD.validate_hold_pair(*values)

    def test_extended_hold_passive_and_fixed_sixteen_ms_windows_remain_distinct(self):
        # All geometry below is MOCK reporter data: oscillating angle samples
        # test window arithmetic, not a simulation or a settling criterion.
        record, _ = mocked_comparison_pair(True)
        record.update({"directory": "MOCK-no-capture", "reportSha256": "MOCK-report",
            "replaySha256": "MOCK-not-replayed", "exactContactLeaves": 0,
            "declaration": {"sleeveInstanceId": "mock-sleeve"},
            "sewingWorkSummary": {"scope": "MOCK"}, "gripperWorkSummary": {"scope": "MOCK"}})
        for index, state in enumerate(record["states"]):
            values = [3.] * 5
            if index == 400:
                values[0] = 6.  # Within extended hold only.
            if index == 496:
                values[1] = 7.  # Passive tail, before final sixteen ms.
            if index == 508:
                values[2] = 4.  # Final sixteen ms; endpoints still equal.
            state.update({"maximumSpeedMetersPerSecond": 0.,
                "bindingGeometry": {"rows": [{"relativeTurnDegrees": value, "current": {
                    "absoluteDistanceErrorMeters": 0., "tangentOffsetMeters": 0.,
                    "crossTangentOffsetMeters": 0.}} for value in values],
                    "strainByInstance": {"mock-sleeve": {"current": {
                        "principalStretchMinimum": 1., "principalStretchMaximum": 1.}}},
                    "rigidFits": {"mock-sleeve": {"rotationDegrees": 0., "translationMeters": [0., 0., 0.],
                                                    "maximumResidualMeters": 0.}},
                    "relativeRigidMotion": {"rotationDegrees": 3.}},
                "surfaceSeparation": {"distanceMeters": .001},
                "gripperGeometry": {"grippers": [{"trackingErrorMeters": 0.}]}})
        summary = HOLD.TIME.summarize(record, hold_fractions=(Fraction(1, 4), Fraction(7, 8)),
                                     passive_start=Fraction(15, 16))
        for name, start, end, states, movement in (
                ("hold", .25, .875, 321, 3.),
                ("passiveTail", .9375, 1., 33, 4.),
                ("finalSixteenMilliseconds", 63 / 64, 1., 9, 1.)):
            observed = summary[name]
            self.assertEqual(observed["startFraction"], start)
            self.assertEqual(observed["endFraction"], end)
            self.assertEqual(observed["startTimeSeconds"], start * 1.024)
            self.assertEqual(observed["endTimeSeconds"], end * 1.024)
            self.assertEqual(observed["savedStates"], states)
            self.assertEqual(observed["maximumSameRowAngleRangeDegrees"], movement)
            self.assertEqual(observed["maximumSameRowEndpointChangeDegrees"], 0.)
        self.assertIs(summary["accepted"], False)

    def test_prefix_compares_every_q_and_velocity_and_rejects_rehashed_semantic_changes(self):
        previous, held, _, _ = self.pair()
        with tempfile.TemporaryDirectory() as temporary:
            directories = [Path(temporary) / name for name in ("previous", "held")]
            reports = []
            for directory, record, count in zip(directories, (previous, held), (256, 512)):
                directory.mkdir()
                report = {"scope": "MOCK artifact hashes only; no replay was performed", "acceptedStateArtifacts": []}
                for index in range(192):
                    state = {"fraction": (index + 1) / count,
                             "positionsMeters": [[index * .001, 0., 0.]],
                             "velocitiesMetersPerSecond": [[.01, 0., 0.]]}
                    path = f"state-{index}.json"
                    (directory / path).write_bytes(encoded(state))
                    state_hash = sha(state)
                    report["acceptedStateArtifacts"].append({"path": path, "sha256": state_hash})
                    record["states"][index + 1]["stateSha256"] = state_hash
                (directory / "report.json").write_bytes(encoded(report))
                record["reportSha256"] = sha(report)
                reports.append(report)
            evidence = HOLD.verify_prefix(*directories, previous, held)
            self.assertEqual(evidence["transitions"], 192)
            self.assertEqual(len(evidence["witnesses"]), 192)
            self.assertEqual(evidence["durationSeconds"], .384)
            self.assertNotEqual(evidence["witnesses"][0]["stateSha256"][0],
                                evidence["witnesses"][0]["stateSha256"][1])
            # Forge a matching journal hash/report/observed artifact identity;
            # the comparison must still compare all actual q/v components.
            for index, field in ((0, "positionsMeters"), (95, "velocitiesMetersPerSecond"),
                                  (191, "positionsMeters")):
                path = directories[1] / f"state-{index}.json"
                original_bytes = path.read_bytes()
                original_hash = reports[1]["acceptedStateArtifacts"][index]["sha256"]
                changed = json.loads(original_bytes)
                changed[field][0][2] = math.nextafter(0., 1.)
                path.write_bytes(encoded(changed))
                changed_hash = sha(changed)
                reports[1]["acceptedStateArtifacts"][index]["sha256"] = changed_hash
                held["states"][index + 1]["stateSha256"] = changed_hash
                (directories[1] / "report.json").write_bytes(encoded(reports[1]))
                held["reportSha256"] = sha(reports[1])
                with self.subTest(index=index, field=field), self.assertRaisesRegex(ValueError, "prefix differs"):
                    HOLD.verify_prefix(*directories, previous, held)
                path.write_bytes(original_bytes)
                reports[1]["acceptedStateArtifacts"][index]["sha256"] = original_hash
                held["states"][index + 1]["stateSha256"] = original_hash
                (directories[1] / "report.json").write_bytes(encoded(reports[1]))
                held["reportSha256"] = sha(reports[1])
            # Stale bytes cannot use an earlier successful comparison result.
            with (directories[1] / "state-191.json").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "state changed"):
                HOLD.verify_prefix(*directories, previous, held)
            with (directories[0] / "report.json").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "report changed"):
                HOLD.verify_prefix(*directories, previous, held)


class ExtendedHoldPortableTests(unittest.TestCase):
    def test_literal_duration_grid_and_physical_phase_durations(self):
        metadata, arguments = timing_fixture()
        before = encoded([metadata, arguments])
        self.assertEqual(REPORT.validate_time_policy(metadata, arguments), metadata["timePolicy"])
        self.assertEqual(REPORT.validate_time_policy(metadata), metadata["timePolicy"])
        self.assertEqual(encoded([metadata, arguments]), before)
        self.assertEqual(arguments["step_seconds"] / arguments["subdivisions"], .002)
        for field, milliseconds in (("turnFractions", 256), ("holdFractions", 640),
                                    ("releaseFractions", 64), ("passiveTailFractions", 64)):
            low, high = PHASES[field]
            # All boundaries are dyadic; the common binary duration is used
            # explicitly rather than pretending the decimal seconds are exact.
            self.assertEqual(float((Fraction(high) - Fraction(low)) * Fraction(1.024)),
                             milliseconds / 1000)

    def test_timing_metadata_run_and_boolean_mutations_reject(self):
        metadata, arguments = timing_fixture()
        declaration = metadata["timePolicy"]
        attacks = [declaration | {key: value} for key, values in {
            "profile": [None, False, "binding-first-turn-time-v2"],
            "id": [FOURFOLD, True, EXTENDED + " "],
            "durationSeconds": [True, "1.024", .512, math.nextafter(1.024, math.inf)],
            "nominalStepSeconds": [False, ".002", .004, math.nextafter(.002, 0.)],
        }.items() for value in values]
        attacks.extend({key: value for key, value in declaration.items() if key != omitted}
                       for omitted in declaration)
        attacks.extend((None, False, {}, declaration | {"extra": False}))
        for attack in attacks:
            with self.subTest(declaration=attack), self.assertRaises((ValueError, TypeError)):
                REPORT.validate_time_policy(metadata | {"timePolicy": attack}, arguments)
        with self.assertRaises(ValueError):
            REPORT.validate_time_policy({"subdivisions": 512}, arguments)
        for count in (True, False, 512., "512", np.int64(512), 256, 1024):
            with self.subTest(count=count), self.assertRaises(ValueError):
                REPORT.validate_time_policy(metadata | {"subdivisions": count}, arguments)
            with self.subTest(run_count=count), self.assertRaises(ValueError):
                REPORT.validate_time_policy(metadata, arguments | {"subdivisions": count})
        for duration in (True, False, None, "1.024", math.nan, math.inf, .512,
                         math.nextafter(1.024, 0.), math.nextafter(1.024, math.inf)):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                REPORT.validate_time_policy(metadata, arguments | {"step_seconds": duration})
        # A matching nominal step does not excuse the wrong duration/grid.
        with self.assertRaises(ValueError):
            REPORT.validate_time_policy(metadata, {"step_seconds": .512, "subdivisions": 256})

    def test_exact_force_energy_while_held_during_release_and_after_release(self):
        source = tool_fixture()
        record = derive_grippers(source, 512)
        positions = np.asarray(source["restMeters"])
        anchor = [Fraction(2), Fraction(1, 4), Fraction(1, 4)]
        error = [value - target for value, target in zip(anchor, (3, 0, 0))]
        for fraction, activation in ((Fraction(7, 8), Fraction(1)),
                                     (Fraction(29, 32), Fraction(1, 2)),
                                     (Fraction(15, 16), Fraction(0)), (Fraction(1), Fraction(0))):
            with self.subTest(fraction=fraction):
                observed = REPORT.grip_observations(record, positions, fraction, release_fraction=.9375)
                force = [float(-10**12 * activation * value) for value in error]
                energy = float(Fraction(10**12, 2) * activation * sum(value * value for value in error))
                self.assertEqual(observed["grippers"][0]["activation"], float(activation))
                self.assertEqual(observed["grippers"][0]["forceNewtons"], force)
                self.assertEqual(observed["totalClothForceNewtons"], force)
                self.assertEqual(observed["energyJoules"], energy)
                self.assertGreater(observed["grippers"][0]["trackingErrorMeters"], 1.)
        # The old release boundary would reject this still-active extended
        # hold. This checks that callers must route the actual boundary.
        with self.assertRaisesRegex(ValueError, "Released grippers"):
            REPORT.grip_observations(record, positions, Fraction(7, 8))

    def test_release_boundary_validation_rejects_coercion_and_active_release(self):
        source = tool_fixture()
        record = derive_grippers(source, 512)
        positions = np.asarray(source["restMeters"])
        for boundary in (True, False, None, ".9375", Fraction(15, 16), .9, math.nan, math.inf):
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                REPORT.grip_observations(record, positions, 1., release_fraction=boundary)
        source["gripperActuation"]["schedule"]["knots"][-2]["activation"] = [.5]
        record = derive_grippers(source, 512)
        with self.assertRaisesRegex(ValueError, "Released grippers"):
            REPORT.grip_observations(record, positions, Fraction(15, 16), release_fraction=.9375)


@unittest.skipUnless(sys.platform.startswith("linux"), "Fresh source reproduction requires pinned Linux runtime")
class ExtendedHoldFreshSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        cls.command(SCRIPTS / "prepare-cuff-source.py", "--output", cls.directory / "parent")
        cls.command(SCRIPTS / "prepare-cuff-construction.py", "--source-canonical",
                    cls.directory / "parent/canonical.json", "--side", "left",
                    "--attachment-policy", "inner-facing-first-outer-shell-last-v1",
                    "--output", cls.directory / "unit")
        cls.unit_path = cls.directory / "unit/unit.json"
        cls.unit_bytes = cls.unit_path.read_bytes()
        cls.unit = json.loads(cls.unit_bytes)
        cls.sources, cls.outputs = {}, {}
        for policy in (FOURFOLD, EXTENDED):
            for angle in (0, 3):
                output = cls.directory / f"{policy}-{angle}"
                result = cls.command(SCRIPTS / "prepare-binding-first-turn.py",
                    "--source-unit", cls.unit_path, "--output", output,
                    "--angle-degrees", angle, "--time-policy", policy)
                summary = json.loads(result.stdout)
                if summary["solverRun"] is not False or summary["accepted"] is not False:
                    raise AssertionError("Preparation must not claim simulation or acceptance")
                cls.sources[policy, angle] = json.loads((output / "canonical.json").read_bytes())
                cls.outputs[policy, angle] = output

    @classmethod
    def command(cls, entry, *arguments):
        process = subprocess.run([sys.executable, str(entry), *map(str, arguments)],
            cwd=cls.directory, env=cls.environment, capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise AssertionError(process.stdout + process.stderr)
        return process

    def reference(self, source):
        count = source["bindingFirstTurnDiagnostic"]["subdivisions"]
        return REPORT.validate_reference(source, np.asarray(source["placedMeters"]),
            derive_sewing(source, count, sewing_mode="distance"), derive_grippers(source, count))

    def test_fresh_extended_sources_reproduce_captured_snapshot_bytes(self):
        for angle in (0, 3):
            original = self.outputs[EXTENDED, angle]
            source = self.sources[EXTENDED, angle]
            self.reference(source)
            output = self.directory / f"reproduced-extended-{angle}"
            self.command(original / "source-snapshot/scripts/prepare-binding-first-turn.py",
                "--source-unit", original / "source-unit.json", "--output", output,
                "--angle-degrees", angle, "--time-policy", EXTENDED)
            for name in ("canonical.json", "placement.json", "source-unit.json"):
                self.assertEqual((original / name).read_bytes(), (output / name).read_bytes(), name)
            self.assertEqual((original / "source-unit.json").read_bytes(), self.unit_bytes)
            for name, expected in source["bindingFirstTurnDiagnostic"]["codeDigests"].items():
                contents = (original / "source-snapshot" / name).read_bytes()
                self.assertEqual(hashlib.sha256(contents).hexdigest(), expected)
                self.assertEqual(contents, (output / "source-snapshot" / name).read_bytes())
            self.assertFalse((original / "report.json").exists())
            self.assertFalse((original / "verified-replay.json").exists())

    def test_complete_sources_differ_only_in_explicit_timing_and_dependent_hash(self):
        for angle in (0, 3):
            sources = [self.sources[policy, angle] for policy in (FOURFOLD, EXTENDED)]
            normalized = []
            for source in sources:
                self.reference(source)
                for key, value in self.unit.items():
                    self.assertEqual(encoded(source[key]), encoded(value), key)
                value = copy.deepcopy(source)
                value["sewingActuation"].pop("sourceSha256")
                declaration = value["bindingFirstTurnDiagnostic"]
                declaration.pop("timePolicy")
                declaration.pop("subdivisions")
                for phase in PHASES:
                    declaration["schedule"].pop(phase)
                for knot in value["gripperActuation"]["schedule"]["knots"]:
                    knot.pop("fraction")
                normalized.append(value)
            self.assertEqual(encoded(normalized[0]), encoded(normalized[1]))
            self.assertEqual(encoded(sources[0]["gripperActuation"]["anchors"]),
                             encoded(sources[1]["gripperActuation"]["anchors"]))
            self.assertEqual(sources[0]["placedMeters"], sources[1]["placedMeters"])
            metadata = sources[1]["bindingFirstTurnDiagnostic"]
            for phase, values in PHASES.items():
                self.assertEqual(encoded(metadata["schedule"][phase]), encoded(values))
            self.assertEqual([knot["fraction"] for knot in sources[1]["gripperActuation"]["schedule"]["knots"]],
                             [index / 32 for index in range(9)] + [.875, .9375, 1.])

    def test_every_common_two_millisecond_prefix_control_through_384ms_matches(self):
        for angle in (0, 3):
            sources = [self.sources[policy, angle] for policy in (FOURFOLD, EXTENDED)]
            schedules = [bind_material_grippers(source, count)[1]
                         for source, count in zip(sources, (256, 512))]
            sewing = [bind_sewing_activation(source, count, sewing_mode="distance")[0]
                      for source, count in zip(sources, (256, 512))]
            for index in range(193):
                fractions = (Fraction(index, 256), Fraction(index, 512))
                self.assertEqual(fractions[0] * Fraction(.512), fractions[1] * Fraction(1.024))
                first = schedules[0].parameters(fractions[0])
                second = schedules[1].parameters(fractions[1])
                for actual, expected in zip(first, second):
                    np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(first[1], [1.] * 3)
                for controls, fraction in zip(sewing, fractions):
                    np.testing.assert_array_equal(controls.parameters(fraction), [1.] * 5 + [0.] * 35)
            # After the common prefix the short control starts releasing;
            # comparing equal fractions would conflate different physical times.
            self.assertTrue(np.all(schedules[0].parameters(Fraction(193, 256))[1] < 1))
            np.testing.assert_array_equal(schedules[1].parameters(Fraction(193, 512))[1], [1.] * 3)

    def test_each_phase_declaration_tamper_rejects_with_repaired_source_hash(self):
        for field in PHASES:
            for mutation in ("value", "boolean", "extra", "missing"):
                with self.subTest(field=field, mutation=mutation):
                    source = copy.deepcopy(self.sources[EXTENDED, 3])
                    schedule = source["bindingFirstTurnDiagnostic"]["schedule"]
                    if mutation == "missing":
                        schedule.pop(field)
                    elif mutation == "extra":
                        schedule[field].append(1.)
                    else:
                        index = 0 if field in ("turnFractions", "angularSampleFractions") else -1
                        original = schedule[field][index]
                        schedule[field][index] = (bool(original) if mutation == "boolean"
                                                  else math.nextafter(original, math.inf))
                    rebind_sewing(source)
                    with self.assertRaises((ValueError, KeyError)):
                        self.reference(source)

    def test_valid_but_wrong_actual_control_times_and_activation_reject(self):
        mutations = (
            ("late-angle", lambda knots: knots[1].__setitem__("fraction", 3 / 64)),
            ("early-release", lambda knots: knots[9].__setitem__("fraction", 27 / 32)),
            ("partial-hold", lambda knots: knots[9].__setitem__("activation", [.5] * 3)),
            ("unreleased", lambda knots: knots[10].__setitem__("activation", [.5] * 3)),
            ("target-ulp", lambda knots: knots[9]["targetsMeters"][0].__setitem__(0,
                math.nextafter(knots[9]["targetsMeters"][0][0], math.inf))),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                source = copy.deepcopy(self.sources[EXTENDED, 3])
                mutate(source["gripperActuation"]["schedule"]["knots"])
                rebind_sewing(source)
                # These remain valid generic recipes. Rejection must come from
                # the declared experiment reference, not an obsolete hash.
                grippers = derive_grippers(source, 512)
                sewing = derive_sewing(source, 512, sewing_mode="distance")
                with self.assertRaises(ValueError):
                    REPORT.validate_reference(source, np.asarray(source["placedMeters"]), sewing, grippers)

    def test_actual_source_is_still_active_at_896ms_and_exactly_released_at_960ms(self):
        source = self.sources[EXTENDED, 3]
        positions = np.asarray(source["placedMeters"])
        record = derive_grippers(source, 512)
        recipe, schedule, _ = bind_material_grippers(source, 512)
        held = REPORT.grip_observations(record, positions, Fraction(7, 8), release_fraction=.9375)
        self.assertGreater(held["energyJoules"], 0.)
        self.assertTrue(all(item["activation"] == 1. for item in held["grippers"]))
        self.assertTrue(any(any(value != 0 for value in item["forceNewtons"]) for item in held["grippers"]))
        for fraction in (Fraction(15, 16), Fraction(31, 32), Fraction(1)):
            observed = REPORT.grip_observations(record, positions, fraction, release_fraction=.9375)
            self.assertEqual(observed["energyJoules"], 0.)
            self.assertEqual(observed["totalClothForceNewtons"], [0.] * 3)
            for item in observed["grippers"]:
                self.assertEqual(item["activation"], 0.)
                self.assertEqual(item["forceNewtons"], [0.] * 3)
            potential = recipe.potential(*schedule.parameters(fraction))
            self.assertEqual(potential.energy(positions), 0.)
            np.testing.assert_array_equal(potential.gradient(positions), np.zeros_like(positions))
            self.assertEqual(potential.hessian().nnz, 0)


if __name__ == "__main__":
    unittest.main()
