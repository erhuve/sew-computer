"""Independent fresh-source lineage and repaired-descriptor attacks; no solver."""

from collections import Counter
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

from solver_binding_refinement import build_binding_refinement, validate_binding_refinement


SCRIPTS = Path(__file__).resolve().parent


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rational(value):
    result = Fraction(int(value["numerator"]), int(value["denominator"]))
    if float(result) != value["roundedBinary64"]:
        raise AssertionError("Reported rounded rational differs")
    return result


def binary(value):
    return Fraction(float(value))


def twice_area(points):
    a, b, c = [[binary(value) for value in point] for point in points]
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


class BindingRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        commands = [("prepare-cuff-source.py", ["--output", root / "parent"])]
        commands.extend(("prepare-cuff-construction.py", ["--source-canonical", root / "parent/canonical.json",
                         "--side", side, "--attachment-policy", "inner-facing-first-outer-shell-last-v1",
                         "--output", root / side]) for side in ("left", "right"))
        for script, arguments in commands:
            process = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)],
                                     env=environment, cwd=root, capture_output=True, text=True, timeout=60)
            if process.returncode:
                raise AssertionError(process.stdout + process.stderr)
        cls.source = json.loads((root / "left/unit.json").read_bytes())
        cls.right = json.loads((root / "right/unit.json").read_bytes())
        cls.descriptor = build_binding_refinement(cls.source)

    def test_complete_base_is_unchanged_and_descriptor_has_no_executable_rows(self):
        before = encoded(self.source)
        descriptor = build_binding_refinement(self.source)
        self.assertEqual(encoded(self.source), before)
        self.assertEqual(descriptor["sourceUnitSha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(descriptor["originalTemplateSha256"], digest(self.source["sourceTemplates"]["opening_binding_left_left"]))
        self.assertEqual(descriptor["originalEmbeddedConstraintsSha256"], digest(self.source["embeddedConstraints"]))
        self.assertEqual(descriptor["profile"], "source-left-binding-refinement-descriptor-v1")
        for field in ("accepted", "solverReady", "executable"):
            self.assertIs(descriptor[field], False)
        self.assertEqual(descriptor["counts"], {"originalStripVertices": 11, "originalStripTriangles": 12,
            "refinedStripVertices": 29, "refinedStripTriangles": 44, "prospectiveUnitVertices": 245,
            "prospectiveUnitTriangles": 398, "unchangedSourceRows": 40, "fabricInstances": 5})
        self.assertFalse({"restMeters", "triangles", "embeddedConstraints", "sewingFrames", "gripperActuation", "placedMeters"}.intersection(descriptor))
        self.assertEqual(len(descriptor["pending"]), 4)
        self.assertIn("does not prove exact-real", descriptor["reconstructionPolicy"]["domainScope"])
        self.assertEqual(len(descriptor["unchangedOtherInstanceLocalMeshSha256"]), 4)
        for instance in self.source["instances"]:
            if instance["id"] == descriptor["instanceId"]:
                continue
            mesh = self.source["sourceTemplates"][instance["templateId"]]
            offset = self.source["instanceOffsets"][instance["id"]]
            payload = {"restMeters": self.source["restMeters"][offset:offset + len(mesh["restPositions"])],
                       "triangles": mesh["triangles"]}
            self.assertEqual(descriptor["unchangedOtherInstanceLocalMeshSha256"][instance["id"]], digest(payload))
        descriptor["sourceBinding"]["rowGroups"].clear()
        descriptor["originalLocalMesh"]["verticesMeters"][0][0] = 99.
        self.assertEqual(encoded(self.source), before)
        self.assertEqual(encoded(build_binding_refinement(self.source)), encoded(self.descriptor))

    def test_both_stage_weights_compose_exactly_without_normalizing_roundoff(self):
        descriptor = self.descriptor
        first, second = [stage["output"] for stage in descriptor["stages"]]
        original = descriptor["originalLocalMesh"]["verticesMeters"]
        final = descriptor["finalLocalMesh"]["verticesMeters"]
        self.assertEqual(final[:11], original)
        maximum, witnessed = Fraction(), False
        for vertex, entry in enumerate(descriptor["composedVertexWeights"]):
            expected = {}
            for middle, outer in second["sourceWeights"][vertex].items():
                for initial, inner in first["sourceWeights"][int(middle)].items():
                    expected[int(initial)] = expected.get(int(initial), Fraction()) + binary(outer) * binary(inner)
            expected = {index: weight for index, weight in expected.items() if weight}
            actual = {weight["sourceVertex"]: rational(weight["weight"]) for weight in entry["weights"]}
            self.assertEqual(actual, expected)
            self.assertEqual(entry["vertex"], vertex)
            self.assertTrue(all(weight > 0 for weight in actual.values()))
            if vertex < 11:
                self.assertEqual(actual, {vertex: Fraction(1)})
            residual = descriptor["binaryInputReconstruction"]["vertices"][vertex]
            self.assertEqual(rational(residual["weightSumMinusOne"]), sum(expected.values(), Fraction()) - 1)
            for axis in range(2):
                error = binary(final[vertex][axis]) - sum((weight * binary(original[index][axis])
                    for index, weight in expected.items()), Fraction())
                self.assertEqual(rational(residual["storedMinusReconstructedMeters"][axis]), error)
                maximum = max(maximum, abs(error))
                witnessed |= error != 0
        self.assertTrue(witnessed, "This fixture must expose actual binary64 reconstruction residuals")
        self.assertEqual(rational(descriptor["binaryInputReconstruction"]["maximumAbsoluteCoordinateResidualMeters"]), maximum)
        self.assertLess(float(maximum), 1e-16)

    def test_lineage_preserves_arbitrary_nonplanar_original_material_values(self):
        descriptor = self.descriptor
        first, second = [stage["output"] for stage in descriptor["stages"]]
        # Non-affine, out-of-plane original nodal values distinguish material
        # interpolation from merely reconstructing the original flat coordinates.
        original_values = [(Fraction(index, 7), Fraction((-1) ** index * index * index, 5), Fraction(index ** 3, 13))
                           for index in range(11)]
        intermediate = [[sum((binary(weight) * original_values[int(index)][axis]
                             for index, weight in support.items()), Fraction()) for axis in range(3)]
                        for support in first["sourceWeights"]]
        staged = [[sum((binary(weight) * intermediate[int(index)][axis]
                       for index, weight in support.items()), Fraction()) for axis in range(3)]
                  for support in second["sourceWeights"]]
        for vertex, item in enumerate(descriptor["composedVertexWeights"]):
            reconstructed = [sum((rational(weight["weight"]) * original_values[weight["sourceVertex"]][axis]
                                  for weight in item["weights"]), Fraction()) for axis in range(3)]
            self.assertEqual(staged[vertex], reconstructed)

    def test_ultimate_original_faces_have_positive_children_and_full_parent_area(self):
        descriptor = self.descriptor
        first, second = [stage["output"] for stage in descriptor["stages"]]
        original, final = descriptor["originalLocalMesh"], descriptor["finalLocalMesh"]
        self.assertEqual(descriptor["ultimateOriginalTriangleIndices"],
                         [first["parentTriangles"][parent] for parent in second["parentTriangles"]])
        area_sums = [Fraction()] * len(original["triangles"])
        for face, parent in zip(final["triangles"], descriptor["ultimateOriginalTriangleIndices"]):
            area = twice_area([final["verticesMeters"][index] for index in face])
            self.assertGreater(area, 0)
            area_sums[parent] += area
            for vertex in face:
                support = {value["sourceVertex"] for value in descriptor["composedVertexWeights"][vertex]["weights"]}
                self.assertTrue(support.issubset(original["triangles"][parent]))
        for face, child_area in zip(original["triangles"], area_sums):
            original_area = twice_area([original["verticesMeters"][index] for index in face])
            self.assertLess(float(abs(child_area - original_area)), 1e-17)
        edges = Counter(tuple(sorted((face[index], face[(index + 1) % 3])))
                        for face in final["triangles"] for index in range(3))
        self.assertTrue(all(count in (1, 2) for count in edges.values()))
        # Connected disk Euler characteristic is independent of the splitter's
        # Shapely union test and rejects hidden edge/face omission here.
        self.assertEqual(len(final["verticesMeters"]) - len(edges) + len(final["triangles"]), 1)

    def test_both_final_chains_keep_named_orientation_and_full_cut_allowances(self):
        descriptor = self.descriptor
        mesh = descriptor["finalLocalMesh"]
        points = mesh["verticesMeters"]
        edges = Counter(tuple(sorted((face[index], face[(index + 1) % 3])))
                        for face in mesh["triangles"] for index in range(3))
        self.assertEqual([stage["sourcePathName"] for stage in descriptor["stages"]], ["right", "left"])
        self.assertEqual([crease["sourcePathName"] for crease in descriptor["finalCreases"]], ["right", "left"])
        for crease, expected_x, sign in zip(descriptor["finalCreases"], (.020, 0.), (1, -1)):
            chain = crease["vertices"]
            self.assertTrue(all(edges[tuple(edge)] == 2 for edge in crease["edges"]))
            self.assertEqual(set(map(tuple, crease["edges"])),
                             {tuple(sorted(pair)) for pair in zip(chain[:-1], chain[1:])})
            for vertex in chain:
                self.assertLess(abs(points[vertex][0] - expected_x), 1e-17)
            self.assertTrue(all(sign * (points[b][1] - points[a][1]) > 0 for a, b in zip(chain[:-1], chain[1:])))
            ends = [points[chain[0]], points[chain[-1]]]
            self.assertEqual(ends, crease["cutBoundaryEndpointsMeters"])
            original_points = descriptor["originalLocalMesh"]["verticesMeters"]
            for actual, expected in zip(sorted(point[1] for point in ends),
                                        [min(p[1] for p in original_points), max(p[1] for p in original_points)]):
                # Stored affine intersections can differ by one ulp from a
                # horizontal boundary; the descriptor explicitly records this
                # binary64 limitation rather than claiming exact-real equality.
                self.assertLessEqual(abs(actual - expected), 2 * max(math.ulp(actual), math.ulp(expected)))
            source_y = [point[1] for point in crease["sourceLineEndpointsMeters"]]
            self.assertLess(min(point[1] for point in ends), min(source_y))
            self.assertGreater(max(point[1] for point in ends), max(source_y))

    def test_repaired_hash_lineage_domain_and_boolean_substitutions_reject(self):
        attacks = [
            ("base hash", lambda d: d.__setitem__("sourceUnitSha256", "0" * 64)),
            ("domain", lambda d: d["finalLocalMesh"]["verticesMeters"][11].__setitem__(0, .019)),
            ("original prefix", lambda d: d["finalLocalMesh"]["verticesMeters"][0].__setitem__(0, -.009)),
            ("ultimate parent", lambda d: d["ultimateOriginalTriangleIndices"].__setitem__(0, 1)),
            ("stage parent", lambda d: d["stages"][1]["output"]["parentTriangles"].__setitem__(0, 1)),
            ("stage vertex", lambda d: d["stages"][0]["output"]["vertices"][11].__setitem__(0, .019)),
            ("composition", lambda d: d["composedVertexWeights"][11]["weights"][0]["weight"].__setitem__("numerator", "0")),
            ("reported residual", lambda d: d["binaryInputReconstruction"]["vertices"][11]["storedMinusReconstructedMeters"][0].__setitem__("numerator", "0")),
            ("winding", lambda d: d["finalLocalMesh"]["triangles"][0].reverse()),
            ("cut endpoint omitted", lambda d: d["finalCreases"][0]["vertices"].pop()),
            ("boolean parent", lambda d: d["ultimateOriginalTriangleIndices"].__setitem__(0, False)),
            ("boolean acceptance", lambda d: d.__setitem__("accepted", 0)),
            ("pending omitted", lambda d: d["pending"].clear()),
        ]
        for label, mutate in attacks:
            with self.subTest(attack=label):
                changed = copy.deepcopy(self.descriptor)
                mutate(changed)
                prior = changed["stages"][0]["output"]
                changed["stages"][1]["inputMeshSha256"] = digest({"verticesMeters": prior["vertices"], "triangles": prior["triangles"]})
                self.assertNotEqual(encoded(changed), encoded(self.descriptor))
                with self.assertRaisesRegex(ValueError, "differs from source rederivation"):
                    validate_binding_refinement(self.source, changed)

    def test_valid_descriptor_is_repeatable_detached_and_strict_json(self):
        validated = validate_binding_refinement(self.source, copy.deepcopy(self.descriptor))
        self.assertEqual(encoded(validated), encoded(self.descriptor))
        validated["stages"][0]["output"]["vertices"][0][0] = 99.
        self.assertEqual(encoded(build_binding_refinement(self.source)), encoded(self.descriptor))
        for bad in (None, [], {"profile": "other"}, {"profile": "x", "coordinate": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_binding_refinement(self.source, bad)

    def test_base_corruption_placement_controls_and_wrong_side_reject(self):
        for field in ("placedMeters", "sewingActuation", "gripperActuation", "sewingFrames", "assemblySchedule", "foldActuation"):
            source = copy.deepcopy(self.source)
            source[field] = []
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Unplaced, uncontrolled"):
                build_binding_refinement(source)
        with self.assertRaisesRegex(ValueError, "Unplaced, uncontrolled"):
            build_binding_refinement(self.right)
        for mutate in (
                lambda s: s["restMeters"][40].__setitem__(0, math.nextafter(s["restMeters"][40][0], math.inf)),
                lambda s: s["embeddedConstraints"]["constraints"][0].__setitem__("complianceMPerN", 2e-8),
                lambda s: s["sourcePattern"]["drafting"].__setitem__("seamAllowanceMm", 9)):
            source = copy.deepcopy(self.source)
            mutate(source)
            source["provenance"]["patternSha256"] = digest(source["sourcePattern"])
            with self.assertRaises(ValueError):
                build_binding_refinement(source)


if __name__ == "__main__":
    unittest.main()
