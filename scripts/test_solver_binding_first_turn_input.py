"""Fresh-source first-turn input/reproduction checks; no simulation or phases.

The retained Linux source coefficient convention is deliberate. These tests
prepare and inspect controls only, without importing or running a solver.
"""

import copy
from fractions import Fraction
import hashlib
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
from solver_sewing_input import bind_sewing_activation


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
GENERATOR = SCRIPTS / "prepare-binding-first-turn.py"
BINDING = "opening_binding_left_left:shell"
SLEEVE = "sleeve_left:shell"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def exact_anchor(positions, vertices, weights):
    return np.array([float(sum((Fraction(float(positions[vertex, axis])) * Fraction(weight)
                               for vertex, weight in zip(vertices, weights)), Fraction()))
                     for axis in range(3)])


@unittest.skipUnless(sys.platform.startswith("linux"), "Source coefficient reproduction uses pinned Linux runtime")
class BindingFirstTurnInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        cls.command(SCRIPTS / "prepare-cuff-source.py", "--output", cls.directory / "parent")
        for side in ("left", "right"):
            cls.command(SCRIPTS / "prepare-cuff-construction.py", "--source-canonical",
                        cls.directory / "parent/canonical.json", "--side", side,
                        "--attachment-policy", "inner-facing-first-outer-shell-last-v1",
                        "--output", cls.directory / side)
        cls.unit_path = cls.directory / "left/unit.json"
        cls.unit_bytes = cls.unit_path.read_bytes()
        cls.unit = json.loads(cls.unit_bytes)
        cls.outputs, cls.sources = {}, {}
        for angle in (0, 3):
            output = cls.directory / f"angle-{angle}"
            result = cls.command(GENERATOR, "--source-unit", cls.unit_path, "--output", output,
                                 "--angle-degrees", angle)
            summary = json.loads(result.stdout)
            if summary["solverRun"] is not False or summary["accepted"] is not False:
                raise AssertionError("Input preparation must not claim a solver run or acceptance")
            cls.outputs[angle] = output
            cls.sources[angle] = json.loads((output / "canonical.json").read_bytes())

    @classmethod
    def command(cls, entry, *arguments, succeeds=True):
        process = subprocess.run([sys.executable, str(entry), *map(str, arguments)],
                                 cwd=cls.directory, env=cls.environment,
                                 capture_output=True, text=True, timeout=60)
        if succeeds and process.returncode:
            raise AssertionError(process.stdout + process.stderr)
        if not succeeds and process.returncode == 0:
            raise AssertionError("Unexpected accepted input: " + process.stdout)
        return process

    def test_original_source_and_matched_control_metadata_are_preserved(self):
        for angle, source in self.sources.items():
            with self.subTest(angle=angle):
                for key in self.unit:
                    self.assertEqual(encoded(source[key]), encoded(self.unit[key]), key)
                self.assertEqual(len(source["instances"]), 5)
                self.assertEqual(len(source["restMeters"]), 227)
                self.assertEqual(len(source["triangles"]) // 3, 366)
                rows = source["embeddedConstraints"]["constraints"]
                self.assertEqual(len(rows), 40)
                self.assertEqual({row["complianceMPerN"] for row in rows}, {1e-8})
                self.assertNotIn("sewingFrames", source)
                metadata = source["bindingFirstTurnDiagnostic"]
                self.assertEqual(metadata["profile"], "source-left-binding-first-turn-v1")
                self.assertIs(metadata["accepted"], False)
                self.assertIs(metadata["solverReady"], False)
                self.assertIn("No engagement phase", metadata["scope"])
                self.assertIn("does not execute", metadata["initialAttachmentCondition"])
                self.assertEqual(metadata["preservedSourceFields"], sorted(self.unit))
        zero, turned = self.sources[0], self.sources[3]
        self.assertEqual(zero["placedMeters"], turned["placedMeters"])
        self.assertEqual(zero["gripperActuation"]["anchors"], turned["gripperActuation"]["anchors"])
        # Identify every permitted metadata difference, so unrelated physical
        # settings cannot silently diverge between the matched controls.
        first, second = (copy.deepcopy(s["bindingFirstTurnDiagnostic"]) for s in (zero, turned))
        for value in (first, second):
            for key in ("angleDegrees", "angleRadians", "rigidReferenceClearanceMeters"):
                value.pop(key)
            value["schedule"].pop("maximumChordRelativeContraction")
        self.assertEqual(encoded(first), encoded(second))
        first, second = (copy.deepcopy(s["sewingActuation"]) for s in (zero, turned))
        first.pop("sourceSha256")
        second.pop("sourceSha256")
        self.assertEqual(encoded(first), encoded(second))

    def test_exact_held_and_pending_rows_with_unchanged_scalar_targets(self):
        for source in self.sources.values():
            controls, manifest = bind_sewing_activation(source, 64, sewing_mode="distance")
            metadata = source["bindingFirstTurnDiagnostic"]
            self.assertEqual(list(controls.row_ids[:5]), metadata["heldRowIds"])
            self.assertEqual(metadata["heldRowIndices"], list(range(5)))
            self.assertEqual(metadata["pendingRowIndices"], list(range(5, 40)))
            np.testing.assert_array_equal(controls.initial_targets, controls.final_targets)
            np.testing.assert_array_equal(controls.initial_targets[:5], metadata["heldSeamTargetsMeters"])
            self.assertTrue(np.all(controls.initial_targets > 0))
            self.assertLessEqual(float(np.max(np.abs(controls.initial_targets[:5] - .001))), 1e-12)
            self.assertEqual(manifest["cuffSourceBinding"]["counts"]["starRowSelectors"], 8)
            for fraction in (Fraction(0), Fraction(1, 128), Fraction(1, 2), Fraction(13, 16), Fraction(1)):
                np.testing.assert_array_equal(controls.parameters(fraction), [1.] * 5 + [0.] * 35)
            self.assertEqual(manifest["complianceMPerN"], 1e-8)

    def test_source_grippers_and_turn_hold_release_are_independently_reconstructed(self):
        source = self.sources[3]
        metadata = source["bindingFirstTurnDiagnostic"]
        placed = np.asarray(source["placedMeters"])
        faces = np.asarray(source["triangles"]).reshape(-1, 3)
        recipe, schedule, _ = bind_material_grippers(source, 64)
        anchors = source["gripperActuation"]["anchors"]
        self.assertEqual([anchor["id"] for anchor in anchors], metadata["gripperIds"])
        self.assertEqual([anchor["weights"] for anchor in anchors],
                         [[.5, .25, .25], [.25, .5, .25], [.375, .375, .25]])
        initial = []
        for anchor, support in zip(anchors, metadata["gripperSourceSupports"]):
            self.assertEqual(anchor["instanceId"], BINDING)
            self.assertEqual(anchor["stiffnessNPerM"], 1.)
            local = source["sourceTemplates"]["opening_binding_left_left"]["triangles"][support["sourceTriangleIndex"]]
            self.assertEqual(local, support["instanceLocalVertices"])
            self.assertEqual([vertex + source["instanceOffsets"][BINDING] for vertex in local], support["canonicalVertices"])
            self.assertEqual(faces[anchor["triangleIndex"]].tolist(), support["canonicalVertices"])
            self.assertEqual([str(Fraction(weight)) for weight in anchor["weights"]], support["weightFractions"])
            initial.append(exact_anchor(placed, faces[anchor["triangleIndex"]], anchor["weights"]))
        initial = np.asarray(initial)
        targets, activation = schedule.parameters(0)
        np.testing.assert_array_equal(targets, initial)
        np.testing.assert_array_equal(activation, [1.] * 3)
        self.assertEqual(recipe.potential(targets, activation).energy(placed), metadata["initialGripperEnergyJoules"])
        self.assertGreater(float(np.linalg.norm(np.cross(initial[1] - initial[0], initial[2] - initial[0]))), 1e-5)
        axis, origin = np.asarray(metadata["axisDirection"]), np.asarray(metadata["axisOriginMeters"])
        # Independent axis decomposition rotates the original barycentric
        # material points; rounding barycentres and rotating all vertices may
        # differ by a few ulps, so this is a geometry check, not a byte oracle.
        displacement = initial - origin
        along = np.outer(displacement @ axis, axis)
        across = displacement - along
        expected_fractions = [index / 16 for index in range(9)] + [.75, .875, 1.]
        self.assertEqual([knot["fraction"] for knot in source["gripperActuation"]["schedule"]["knots"]], expected_fractions)
        for index in range(9):
            angle = math.radians(3) * index / 8
            expected = origin + along + math.cos(angle) * across + math.sin(angle) * np.cross(axis, across)
            target, active = schedule.parameters(Fraction(index, 16))
            np.testing.assert_allclose(target, expected, rtol=0, atol=2e-16)
            np.testing.assert_array_equal(active, [1.] * 3)
        left, _ = schedule.parameters(Fraction(1, 16))
        right, _ = schedule.parameters(Fraction(2, 16))
        middle, _ = schedule.parameters(Fraction(3, 32))
        np.testing.assert_allclose(middle, (left + right) / 2, rtol=0, atol=6e-17)
        final, _ = schedule.parameters(Fraction(1, 2))
        for fraction, expected_activation in ((Fraction(3, 4), 1.), (Fraction(13, 16), .5),
                                              (Fraction(7, 8), 0.), (Fraction(1), 0.)):
            target, active = schedule.parameters(fraction)
            np.testing.assert_array_equal(target, final)
            np.testing.assert_array_equal(active, [expected_activation] * 3)
        zero_knots = self.sources[0]["gripperActuation"]["schedule"]["knots"]
        self.assertTrue(all(knot["targetsMeters"] == zero_knots[0]["targetsMeters"] for knot in zero_knots))
        self.assertEqual(metadata["schedule"]["maximumChordRelativeContraction"], 1 - math.cos(math.radians(3) / 16))
        self.assertIn("not a continuous rigid rotation", metadata["schedule"]["betweenKnotTargetMotion"])

    def test_proper_placements_oriented_source_frames_and_research_sides(self):
        source = self.sources[3]
        metadata = source["bindingFirstTurnDiagnostic"]
        rest, placed = np.asarray(source["restMeters"]), np.asarray(source["placedMeters"])
        faces = np.asarray(source["triangles"]).reshape(-1, 3)
        covered = []
        for pose in metadata["rigidPlacements"]:
            begin, end = metadata["instanceRanges"][pose["instanceId"]]
            covered.extend(range(begin, end))
            rotation = np.asarray(pose["rotation"])
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=2e-15)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1., delta=2e-15)
            np.testing.assert_array_equal(placed[begin:end], rest[begin:end] @ rotation.T + pose["translationMeters"])
            self.assertIs(pose["sourceMirrorX"], False)
            np.testing.assert_array_equal(rotation @ [0., 0., 1.], pose["initialPlacedNormal"])
            relevant = faces[np.all((faces >= begin) & (faces < end), axis=1)]
            original_edges = rest[relevant[:, 1:]] - rest[relevant[:, :1]]
            placed_edges = placed[relevant[:, 1:]] - placed[relevant[:, :1]]
            np.testing.assert_allclose(np.linalg.norm(original_edges, axis=2),
                                       np.linalg.norm(placed_edges, axis=2), rtol=0, atol=2e-16)
            self.assertTrue(np.all(np.cross(placed_edges[:, 0], placed_edges[:, 1])[:, 2] > 0))
        self.assertEqual(sorted(covered), list(range(len(rest))))
        self.assertEqual(len(metadata["frameSelectionBindings"]), 10)
        for selection in metadata["frameSelectionBindings"]:
            row_index, identity = selection["rowIndex"], selection["instanceId"]
            row = source["embeddedConstraints"]["constraints"][row_index]
            local_face = source["sourceTemplates"][selection["sourceTemplateId"]]["triangles"][selection["sourceTriangleIndex"]]
            self.assertEqual(selection["instanceLocalVertices"], local_face)
            self.assertEqual(selection["canonicalVertices"], faces[selection["canonicalTriangleIndex"]].tolist())
            support = {term["vertex"] for term in row["terms"] if term["instanceId"] == identity}
            self.assertTrue(support.issubset(local_face))
            self.assertFalse(selection["ambiguousFrameChoice"])
            key = "sleeve" if identity == SLEEVE else "binding"
            self.assertEqual(selection["canonicalVertices"], metadata[key + "FrameFaces"][row_index])
            tangent = np.asarray(metadata[key + "TangentsRest"][row_index])
            self.assertAlmostEqual(float(tangent @ tangent), 1., delta=2e-15)
            self.assertEqual(tangent[2], 0.)
        sides = metadata["textileSidePolicy"]
        self.assertIn("not inferred", sides["classification"])
        self.assertEqual(sides["rightSideRelativeToCanonicalNormal"][SLEEVE], 1)
        self.assertEqual(sides["rightSideRelativeToCanonicalNormal"][BINDING], -1)
        self.assertEqual(sides["initialActiveRightSideNormals"], {SLEEVE: [0., 0., 1.], BINDING: [0., 0., -1.]})
        self.assertAlmostEqual(metadata["rigidReferenceClearanceMeters"]["minimumToSleevePlane"],
                               .001 - .010 * math.sin(math.radians(3)), delta=1e-18)
        self.assertGreater(metadata["rigidReferenceClearanceMeters"]["minimumToSleevePlane"], .0002)

    def test_exact_hashes_and_captured_input_generator_dependency_bytes(self):
        for angle, output in self.outputs.items():
            source = self.sources[angle]
            metadata = source["bindingFirstTurnDiagnostic"]
            self.assertEqual((output / "source-unit.json").read_bytes(), self.unit_bytes)
            self.assertEqual(metadata["sourceUnitBytesSha256"], digest(self.unit_bytes))
            self.assertEqual(metadata["sourceUnitCanonicalSha256"], digest(encoded(self.unit)))
            placement = json.loads((output / "placement.json").read_bytes())
            self.assertEqual(placement["canonicalDigest"], digest((output / "canonical.json").read_bytes()))
            self.assertEqual(placement["placedMeters"], source["placedMeters"])
            expected_mesh = {key: source[key] for key in ("restMeters", "triangles", "instanceOffsets")}
            self.assertEqual(source["gripperActuation"]["meshSha256"], digest(encoded(expected_mesh)))
            expected_source = {key: value for key, value in source.items() if key != "sewingActuation"}
            self.assertEqual(source["sewingActuation"]["sourceSha256"], digest(encoded(expected_source)))
            snapshot = output / "source-snapshot"
            self.assertEqual({str(path.relative_to(snapshot)) for path in snapshot.rglob("*.py")}, set(metadata["codeDigests"]))
            self.assertIn("scripts/prepare-binding-first-turn.py", metadata["codeDigests"])
            for name, expected in metadata["codeDigests"].items():
                self.assertFalse(Path(name).is_absolute())
                self.assertNotIn("..", Path(name).parts)
                content = (snapshot / name).read_bytes()
                self.assertEqual(digest(content), expected, name)
                self.assertEqual(content, (ROOT / name).read_bytes(), name)
            self.assertFalse((output / "report.json").exists())
            self.assertFalse((output / "verified-replay.json").exists())

    def test_both_controls_reproduce_from_captured_repository_shaped_tree(self):
        for angle, original in self.outputs.items():
            with self.subTest(angle=angle):
                output = self.directory / f"reproduced-{angle}"
                self.command(original / "source-snapshot/scripts/prepare-binding-first-turn.py",
                             "--source-unit", original / "source-unit.json", "--output", output,
                             "--angle-degrees", angle)
                for name in ("canonical.json", "placement.json", "source-unit.json"):
                    self.assertEqual((original / name).read_bytes(), (output / name).read_bytes(), name)
                for name in self.sources[angle]["bindingFirstTurnDiagnostic"]["codeDigests"]:
                    self.assertEqual((original / "source-snapshot" / name).read_bytes(),
                                     (output / "source-snapshot" / name).read_bytes(), name)

    def test_invalid_angles_and_stiffness_fail_before_output(self):
        for index, (flag, value) in enumerate((
                ("--angle-degrees", "-0.001"), ("--angle-degrees", "3.000001"),
                ("--angle-degrees", "nan"), ("--angle-degrees", "inf"),
                ("--angle-degrees", "true"), ("--gripper-stiffness-n-per-m", "0"),
                ("--gripper-stiffness-n-per-m", "100.0001"),
                ("--gripper-stiffness-n-per-m", "nan"))):
            with self.subTest(flag=flag, value=value):
                output = self.directory / f"invalid-parameter-{index}"
                self.command(GENERATOR, "--source-unit", self.unit_path, "--output", output,
                             flag, value, succeeds=False)
                self.assertFalse(output.exists())

    def test_unsupported_and_mutated_sources_fail_without_publication(self):
        mutations = [
            ("rest", lambda unit: unit["restMeters"][0].__setitem__(0, math.nextafter(unit["restMeters"][0][0], math.inf))),
            ("coefficient", lambda unit: unit["embeddedConstraints"]["constraints"][0]["terms"][0].__setitem__("coefficient", -.5)),
            ("compliance", lambda unit: unit["embeddedConstraints"]["constraints"][0].__setitem__("complianceMPerN", 2e-8)),
            ("winding", lambda unit: unit["triangles"].__setitem__(0, unit["triangles"][1])),
            ("placement", lambda unit: unit.__setitem__("placedMeters", unit["restMeters"])),
            ("control", lambda unit: unit.__setitem__("gripperActuation", {})),
            ("frame", lambda unit: unit.__setitem__("sewingFrames", {})),
            ("source", lambda unit: unit["sourcePattern"]["drafting"].__setitem__("seamAllowanceMm", 9)),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                unit = copy.deepcopy(self.unit)
                mutate(unit)
                # Repair the available pattern metadata to prevent a stale
                # top-level hash from being the only protection of semantics.
                unit["provenance"]["patternSha256"] = digest(encoded(unit["sourcePattern"]))
                path, output = self.directory / f"changed-{label}.json", self.directory / f"rejected-{label}"
                path.write_bytes(encoded(unit))
                self.command(GENERATOR, "--source-unit", path, "--output", output, succeeds=False)
                self.assertFalse(output.exists())
        output = self.directory / "rejected-right"
        self.command(GENERATOR, "--source-unit", self.directory / "right/unit.json", "--output", output, succeeds=False)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
