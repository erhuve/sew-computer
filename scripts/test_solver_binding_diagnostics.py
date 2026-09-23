"""Geometry counterexamples independent of cloth or actuation implementations."""

import ast
import copy
import math
from pathlib import Path
import unittest

import numpy as np

from solver_binding_diagnostics import analyze_binding_diagnostic
import solver_binding_diagnostics as diagnostics


def rotation(angle, axis=(0., 1., 0.)):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def fixture():
    panel = [[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]]
    rest = np.array(panel + panel)
    initial = rest.copy()
    initial[4:, 2] += 1.
    faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    weights = ((.125, .375, .5), (.25, .5, .25), (.375, .25, .375), (.5, .125, .375), (.25, .25, .5))
    rows = [[*((vertex, weight) for vertex, weight in enumerate(row)),
             *((vertex + 4, -weight) for vertex, weight in enumerate(row))] for row in weights]
    options = dict(instance_ranges={"sleeve": (0, 4), "binding": (4, 8)}, row_ids=[f"row:{i}" for i in range(5)],
        rows=rows, sleeve_instance_id="sleeve", binding_instance_id="binding",
        sleeve_frame_faces=[[0, 1, 2]] * 5, binding_frame_faces=[[4, 5, 6]] * 5,
        sleeve_tangents_rest=[[0., 1., 0.]] * 5, binding_tangents_rest=[[0., 1., 0.]] * 5,
        target_distances=[1.] * 5)
    return rest, initial, faces, options


def measure(current=None, *, initial_override=None, **changes):
    rest, initial, faces, options = fixture()
    if initial_override is not None:
        initial = initial_override
    options.update(changes)
    return analyze_binding_diagnostic(rest, initial, initial if current is None else current, faces, **options)


class BindingDiagnosticsTests(unittest.TestCase):
    def test_independent_dependencies_and_nonmidpoint_anchor_measurements(self):
        tree = ast.parse(Path(diagnostics.__file__).read_text())
        imports = {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names)
        self.assertLessEqual(imports, {"hashlib", "json", "math", "numpy"})
        result = measure()
        first = result["rows"][0]["current"]
        np.testing.assert_array_equal(first["sleeveAnchorMeters"], [.75, 0., 0.])
        np.testing.assert_array_equal(first["bindingAnchorMeters"], [.75, 0., 1.])
        self.assertEqual(first["signedNormalGapMeters"], 1.)
        self.assertEqual(first["tangentOffsetMeters"], 0.)
        self.assertEqual(first["crossTangentOffsetMeters"], 0.)
        self.assertEqual(first["signedDistanceErrorMeters"], 0.)
        self.assertFalse(result["accepted"])

    def test_binding_only_and_sleeve_only_turns_have_opposite_relative_signs(self):
        _, initial, _, _ = fixture()
        angle = math.radians(3.)
        for selected, pivot, sign in ((slice(4, 8), [0., 0., 1.], 1), (slice(0, 4), [0., 0., 0.], -1)):
            current = initial.copy()
            current[selected] = (current[selected] - pivot) @ rotation(angle).T + pivot
            result = measure(current)
            for row in result["rows"]:
                self.assertAlmostEqual(row["relativeTurnRadians"], sign * angle, places=14)
                self.assertAlmostEqual(row["current"]["tangentMisalignmentRadians"], 0., places=14)
            self.assertAlmostEqual(result["relativeRigidMotion"]["rotationRadians"], angle, places=14)
            for fit in result["rigidFits"].values():
                self.assertLess(fit["maximumResidualMeters"], 1e-14)
                self.assertAlmostEqual(np.linalg.det(fit["rotationMatrix"]), 1., places=14)

    def test_initial_angle_is_subtracted_and_common_world_rigid_motion_is_covariant(self):
        rest, initial, faces, options = fixture()
        pivot = np.array([0., 0., 1.])
        initial[4:] = (initial[4:] - pivot) @ rotation(.2).T + pivot
        current = initial.copy()
        current[4:] = (current[4:] - pivot) @ rotation(.05).T + pivot
        base = analyze_binding_diagnostic(rest, initial, current, faces, **options)
        world = rotation(.8, (1., 2., -3.))
        translated = analyze_binding_diagnostic(rest, initial @ world.T + [2., -4., 3.],
            current @ world.T + [2., -4., 3.], faces, **options)
        for first, second in zip(base["rows"], translated["rows"]):
            self.assertAlmostEqual(first["initial"]["signedRollRadians"], .2, places=14)
            self.assertAlmostEqual(first["relativeTurnRadians"], .05, places=14)
            self.assertAlmostEqual(first["relativeTurnRadians"], second["relativeTurnRadians"], places=14)
            for key in ("signedNormalGapMeters", "tangentOffsetMeters", "crossTangentOffsetMeters", "anchorDistanceMeters"):
                self.assertAlmostEqual(first["current"][key], second["current"][key], places=14)
        self.assertAlmostEqual(base["relativeRigidMotion"]["rotationRadians"],
                               translated["relativeRigidMotion"]["rotationRadians"], places=14)
        common = analyze_binding_diagnostic(rest, initial, initial @ world.T + [2., -4., 3.], faces, **options)
        self.assertLess(common["relativeRigidMotion"]["rotationRadians"], 2e-15)
        self.assertTrue(all(abs(row["relativeTurnRadians"]) < 2e-15 for row in common["rows"]))

    def test_local_bending_cannot_be_replaced_by_global_fit_angle(self):
        rest, initial, faces, options = fixture()
        for row in (3, 4):
            options["rows"][row] = [(0, .25), (2, .25), (3, .5), (4, -.25), (6, -.25), (7, -.5)]
            options["sleeve_frame_faces"][row] = [0, 2, 3]
            options["binding_frame_faces"][row] = [4, 6, 7]
        current = initial.copy()
        current[7, 2] += .6
        result = analyze_binding_diagnostic(rest, initial, current, faces, **options)
        turns = [row["relativeTurnRadians"] for row in result["rows"]]
        self.assertEqual(turns[0], 0.)
        self.assertGreater(max(turns) - min(turns), .2)
        fit = result["rigidFits"]["binding"]
        self.assertGreater(fit["rotationRadians"], .05)
        self.assertGreater(fit["maximumResidualMeters"], .1)
        self.assertGreater(result["strainByInstance"]["binding"]["current"]["principalStretchMaximum"], 1.)

    def test_original_rest_principal_stretches_and_area_include_initial_deformation(self):
        rest, initial, faces, options = fixture()
        initial[:, 0] *= 1.5
        current = initial.copy()
        current[:, 0] *= 4. / 3.
        current[:, 1] *= .5
        result = analyze_binding_diagnostic(rest, initial, current, faces, **options)
        for instance in result["strainByInstance"].values():
            self.assertAlmostEqual(instance["initial"]["principalStretchMaximum"], 1.5, places=14)
            self.assertAlmostEqual(instance["initial"]["areaRatioMaximum"], 1.5, places=14)
            self.assertAlmostEqual(instance["current"]["principalStretchMinimum"], .5, places=14)
            self.assertAlmostEqual(instance["current"]["principalStretchMaximum"], 2., places=14)
            self.assertAlmostEqual(instance["current"]["areaRatioMaximum"], 1., places=14)

    def test_material_offsets_change_without_becoming_physical_slip_claims(self):
        _, initial, _, _ = fixture()
        current = initial.copy()
        current[4:] += [.125, .25, .5]
        result = measure(current)
        for row in result["rows"]:
            # +z normal, +y tangent, z cross y = -x cross-tangent.
            self.assertEqual(row["current"]["signedNormalGapMeters"], 1.5)
            self.assertEqual(row["current"]["tangentOffsetMeters"], .25)
            self.assertEqual(row["current"]["crossTangentOffsetMeters"], -.125)
            self.assertEqual(row["offsetChangesMeters"]["signedNormalGapMeters"], .5)
            self.assertEqual(row["relativeTurnRadians"], 0.)
        np.testing.assert_allclose(result["relativeRigidMotion"]["translationMeters"], [.125, .25, .5], atol=1e-15)

    def test_canonical_winding_is_explicit_and_fit_remains_proper(self):
        rest, initial, faces, options = fixture()
        bad = copy.deepcopy(options)
        bad["sleeve_frame_faces"][0] = [0, 2, 1]
        with self.assertRaises(ValueError):
            analyze_binding_diagnostic(rest, initial, initial, faces, **bad)
        reversed_faces = [[a, c, b] for a, b, c in faces]
        options["sleeve_frame_faces"] = [[0, 2, 1]] * 5
        options["binding_frame_faces"] = [[4, 6, 5]] * 5
        result = analyze_binding_diagnostic(rest, initial, initial, reversed_faces, **options)
        self.assertEqual(result["rows"][0]["current"]["signedNormalGapMeters"], -1.)
        current = initial.copy()
        current[4:, 0] *= -1
        result = measure(current)
        fit = result["rigidFits"]["binding"]
        self.assertAlmostEqual(np.linalg.det(fit["rotationMatrix"]), 1., places=14)
        self.assertAlmostEqual(fit["rotationRadians"], math.pi, places=14)
        self.assertLess(fit["maximumResidualMeters"], 1e-14)

    def test_undefined_triangles_projections_and_rigid_fits_fail(self):
        rest, initial, faces, options = fixture()
        collapsed = initial.copy()
        collapsed[6] = collapsed[5]
        with self.assertRaisesRegex(ValueError, "triangle frame"):
            measure(collapsed)
        perpendicular = initial.copy()
        perpendicular[4:] = (perpendicular[4:] - [0., 0., 1.]) @ rotation(math.pi / 2, (1., 0., 0.)).T + [0., 0., 1.]
        with self.assertRaisesRegex(ValueError, "undefined projection"):
            measure(perpendicular)
        folded = initial.copy()
        folded[7] = folded[5]
        with self.assertRaisesRegex(ValueError, "proper rigid fit"):
            measure(folded)
        for tangent in ([0., 0., 0.], [0., 1., 1.]):
            with self.assertRaisesRegex(ValueError, "source tangent"):
                measure(sleeve_tangents_rest=[tangent] * 5)

    def test_raw_admission_source_support_and_inputs_remain_unchanged(self):
        rest, initial, faces, options = fixture()
        untouched = copy.deepcopy((rest, initial, faces, options))
        result = analyze_binding_diagnostic(rest, initial, initial, faces, **options)
        self.assertEqual(faces, untouched[2])
        self.assertEqual(options, untouched[3])
        np.testing.assert_array_equal(rest, untouched[0])
        np.testing.assert_array_equal(initial, untouched[1])
        result["rows"][0]["current"]["sleeveFrame"]["normal"][0] = 123.
        self.assertEqual(measure()["rows"][0]["current"]["sleeveFrame"]["normal"][0], 0.)
        attacks = [{"target_distances": [True] * 5}, {"target_distances": [0.] * 5},
            {"target_distances": [np.inf] * 5}, {"target_distances": [np.int64(2 ** 53 + 1)] * 5},
            {"row_ids": ["duplicate"] * 5}, {"row_ids": ["x"]},
            {"instance_ranges": {"sleeve": (0, 4), "binding": (3, 8)}},
            {"instance_ranges": {"sleeve": (False, 4), "binding": (4, 8)}},
            {"sleeve_frame_faces": [[4, 5, 6]] * 5}, {"sleeve_frame_faces": [[False, 1, 2]] * 5}]
        for change in attacks:
            with self.subTest(change=change), self.assertRaises(ValueError):
                measure(**change)
        changed = copy.deepcopy(options)
        changed["rows"][0][0] = (True, .125)
        with self.assertRaises(ValueError):
            analyze_binding_diagnostic(rest, initial, initial, faces, **changed)
        changed = copy.deepcopy(options)
        changed["rows"][0][0] = (0, .25)
        with self.assertRaises(ValueError):
            analyze_binding_diagnostic(rest, initial, initial, faces, **changed)
        malformed = initial.tolist()
        malformed[0][0] = False
        with self.assertRaises(ValueError):
            analyze_binding_diagnostic(rest, initial, malformed, faces, **options)
        with self.assertRaises(ValueError):
            analyze_binding_diagnostic([np.array(0.)] * 8, initial, initial, faces, **options)


if __name__ == "__main__":
    unittest.main()
