"""Independent material-gripper migration checks; no dynamics or replay run.

Fresh source fixtures exercise the producer only to obtain audit inputs. The
verifier is also exercised in an isolated process with standard-library imports.
"""

import copy
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from solver_binding_gripper_remap_replay import verify_binding_gripper_remap, _nearest


SCRIPTS = Path(__file__).resolve().parent
INSTANCE = "opening_binding_left_left:shell"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def mesh_sha(source):
    return sha({key: source[key] for key in ("restMeters", "triangles", "instanceOffsets")})


def rat(value):
    return F(int(value["numerator"]), int(value["denominator"]))


def basis(refinement):
    rows = [{i: F(1)} for i in range(11)]
    for stage in refinement["stages"]:
        previous, rows = rows, []
        for support in stage["output"]["sourceWeights"]:
            row = {}
            for middle, outer in support.items():
                for original, inner in previous[int(middle)].items():
                    row[original] = row.get(original, F()) + F(outer) * inner
            rows.append(row)
    return rows


def gaussian(matrix, target):
    rows = [list(row) + [value] for row, value in zip(matrix, target)]
    for column in range(3):
        pivot = next(index for index in range(column, 3) if rows[index][column])
        rows[pivot], rows[column] = rows[column], rows[pivot]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for index in range(3):
            if index != column:
                scale = rows[index][column]
                rows[index] = [a - scale * b for a, b in zip(rows[index], rows[column])]
    return [row[-1] for row in rows]


class BindingGripperRemapReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from solver_binding_source import build_binding_source
        from solver_binding_gripper_remap import build_binding_gripper_remap
        cls.producer = staticmethod(build_binding_gripper_remap)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        commands = [("prepare-cuff-source.py", ["--output", cls.directory / "parent"]),
            ("prepare-cuff-construction.py", ["--source-canonical", cls.directory / "parent/canonical.json",
                "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1", "--output", cls.directory / "unit"])]
        for script, arguments in commands:
            process = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)], cwd=cls.directory,
                env=env, capture_output=True, text=True, timeout=90)
            if process.returncode:
                raise AssertionError(process.stdout + process.stderr)
        cls.base = json.loads((cls.directory / "unit/unit.json").read_bytes())
        # An explicit test control recipe on the three reviewed source material
        # triangles, not a claimed first-turn trajectory or physical placement.
        # This avoids invoking the intentionally Linux-only input generator in
        # portable source/operator tests, without relaxing its runtime gate.
        cls.original = copy.deepcopy(cls.base)
        faces = [cls.base["triangles"][i:i + 3] for i in range(0, len(cls.base["triangles"]), 3)]
        template = cls.base["sourceTemplates"]["opening_binding_left_left"]
        offset = cls.base["instanceOffsets"][INSTANCE]
        anchors = [{"id": identity, "instanceId": INSTANCE,
                    "triangleIndex": faces.index([v + offset for v in template["triangles"][local]]),
                    "weights": weights, "stiffnessNPerM": 1.}
                   for identity, local, weights in (("binding-wrist-body", 6, [.5, .25, .25]),
                       ("binding-apex-body", 1, [.25, .5, .25]), ("binding-allowance", 10, [.375, .375, .25]))]
        cls.original["gripperActuation"] = {"profile": "captured-material-grippers-v1", "accepted": False,
            "meshSha256": mesh_sha(cls.original), "anchors": anchors,
            "schedule": {"profile": "material-gripper-target-activation-v1",
                "gripperIds": [anchor["id"] for anchor in anchors],
                "knots": [{"fraction": fraction,
                    "targetsMeters": [[.01 + i * .02, .2 + min(k, 8) * .001, .001] for i in range(3)],
                    "activation": [0. if fraction >= .875 else 1.] * 3}
                    for k, fraction in enumerate([i / 16 for i in range(9)] + [.75, .875, 1.])]}}
        cls.refined = build_binding_source(cls.base)
        cls.descriptor = cls.producer(cls.original, cls.refined, subdivisions=64)
        for name, value in (("original", cls.original), ("refined", cls.refined), ("descriptor", cls.descriptor)):
            (cls.directory / f"{name}.json").write_bytes(encoded(value))

    def verify(self, original=None, refined=None, descriptor=None, subdivisions=64):
        return verify_binding_gripper_remap(self.original if original is None else original,
            self.refined if refined is None else refined, self.descriptor if descriptor is None else descriptor,
            subdivisions=subdivisions)

    def test_fresh_source_identity_unchanged_recipe_and_detached_evidence(self):
        before = encoded([self.original, self.refined, self.descriptor])
        evidence = self.verify()
        self.assertIs(evidence["verified"], True)
        self.assertIs(evidence["accepted"], False)
        self.assertEqual(evidence["gripperCount"], 3)
        self.assertEqual(evidence["refinedBindingGrippers"], 3)
        self.assertEqual(evidence["unchangedInstanceGrippers"], 0)
        for key, value in (("originalSourceSha256", self.original), ("refinedSourceSha256", self.refined),
                           ("descriptorSha256", self.descriptor), ("baseUnitSha256", self.base),
                           ("numericalRecipeSha256", self.descriptor["numericalRecipe"])):
            self.assertEqual(evidence[key], sha(value))
        self.assertIn("separately captured source validator", evidence["sourceScope"])
        self.assertIn("No frame-side", evidence["sourceScope"])
        evidence["selectedCanonicalTriangleIndices"].clear()
        self.assertEqual(encoded([self.original, self.refined, self.descriptor]), before)
        self.assertEqual(len(self.verify()["selectedCanonicalTriangleIndices"]), 3)

    def test_verifier_import_and_execution_are_stdlib_only(self):
        command = """import json,sys
sys.path.insert(0,sys.argv[1])
before=set(sys.modules)
from solver_binding_gripper_remap_replay import verify_binding_gripper_remap,REQUIRED_HELPER_FILES
assert REQUIRED_HELPER_FILES==()
documents=[json.load(open(sys.argv[2]+'/'+name+'.json')) for name in ('original','refined','descriptor')]
assert verify_binding_gripper_remap(*documents,subdivisions=64)['verified'] is True
new=set(sys.modules)-before
assert not any(name.startswith(('numpy','scipy','shapely')) for name in new)
assert not any(name in new for name in ('solver_binding_source','solver_binding_refinement','solver_binding_remap',
 'solver_binding_gripper_remap','solver_gripper_input','solver_material_grippers','assembly','cloth_domain','embedded_constraints'))
"""
        process = subprocess.run([sys.executable, "-c", command, str(SCRIPTS), str(self.directory)],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_complete_candidates_and_exact_nonplanar_pullbacks(self):
        self.verify()
        refinement = self.refined["bindingRefinement"]
        lineage = basis(refinement)
        original_probe = [F((-1)**i * (i**3 + 7), 19) for i in range(11)]
        refined_probe = [sum((weight * original_probe[i] for i, weight in row.items()), F()) for row in lineage]
        candidate_count = 0
        for entry in self.descriptor["anchors"]:
            original_face = entry["originalLocalVertices"]
            original_weights = list(map(F, entry["originalAnchor"]["weights"]))
            expected_children = [i for i, parent in enumerate(refinement["ultimateOriginalTriangleIndices"])
                                 if parent == entry["originalLocalTriangleIndex"]]
            self.assertEqual([item["childTriangleIndex"] for item in entry["candidates"]], expected_children)
            for candidate in entry["candidates"]:
                face = candidate["orientedVertices"]
                matrix = [[lineage[vertex].get(original, F()) for vertex in face] for original in original_face]
                expected = gaussian(matrix, original_weights)
                self.assertEqual(list(map(rat, candidate["exactWeights"])), expected)
                self.assertIs(candidate["coefficientNonnegative"], all(weight >= 0 for weight in expected))
                self.assertEqual(sum((weight * refined_probe[v] for v, weight in zip(face, expected)), F()),
                    sum((weight * original_probe[v] for v, weight in zip(original_face, original_weights)), F()))
                candidate_count += 1
        self.assertEqual(self.verify()["enumeratedCandidateCount"], candidate_count)

    def test_shared_edge_grips_preserve_equivalent_operators_without_frame_selection(self):
        self.verify()
        first, second, third = self.descriptor["anchors"]
        self.assertEqual(first["equivalentLocalTriangleIndices"], [27, 28])
        self.assertEqual(second["equivalentLocalTriangleIndices"], [7, 9])
        self.assertEqual(third["equivalentLocalTriangleIndices"], [38])
        self.assertEqual([item["selectedCanonicalTriangleIndex"] for item in (first, second, third)], [75, 55, 86])
        for entry in (first, second):
            maps = [{v: rat(w) for v, w in zip(candidate["orientedVertices"], candidate["exactWeights"]) if rat(w)}
                    for candidate in entry["candidates"] if candidate["coefficientNonnegative"]]
            self.assertEqual(maps[0], maps[1])
            self.assertEqual(sorted(maps[0].values()), [F(1, 2), F(1, 2)])
            self.assertNotIn("frameSide", entry)
            self.assertNotIn("normal", entry)

    def test_unchanged_sleeve_grip_reindexes_global_triangle_preserving_raw_local_recipe(self):
        original = copy.deepcopy(self.original)
        sleeve = "sleeve_left:shell"
        sleeve_start = original["instanceOffsets"][sleeve]
        faces = [original["triangles"][index:index + 3] for index in range(0, len(original["triangles"]), 3)]
        triangle = next(index for index, face in enumerate(faces) if all(v >= sleeve_start for v in face))
        recipe = original["gripperActuation"]
        grip = {"id": "explicit-sleeve-grip", "instanceId": sleeve, "triangleIndex": triangle,
                "weights": [1, 0, 0], "stiffnessNPerM": 7}
        recipe["anchors"].append(grip)
        recipe["schedule"]["gripperIds"].append(grip["id"])
        for knot in recipe["schedule"]["knots"]:
            knot["targetsMeters"].append([.1, .3, .01])
            knot["activation"].append(knot["activation"][0])
        descriptor = self.producer(original, self.refined, subdivisions=64)
        evidence = self.verify(original=original, descriptor=descriptor)
        self.assertEqual(evidence["unchangedInstanceGrippers"], 1)
        last = descriptor["anchors"][-1]
        self.assertEqual(last["migrationKind"], "unchanged-instance")
        self.assertEqual(last["selectedLocalTriangleIndex"], 0)
        self.assertEqual(last["selectedCanonicalTriangleIndex"], triangle + 32)
        self.assertEqual(encoded(last["numericalAnchor"]["weights"]), encoded([1, 0, 0]))
        self.assertNotIn("candidates", last)
        self.assertNotIn("coefficientAudit", last)
        self.assertEqual(list(map(rat, last["storedCoordinateResidualMeters"])), [F(0)] * 3)
        changed = copy.deepcopy(descriptor)
        changed["anchors"][-1]["numericalAnchor"]["triangleIndex"] = triangle
        changed["anchors"][-1]["selectedCanonicalTriangleIndex"] = triangle
        changed["numericalRecipe"]["anchors"][-1]["triangleIndex"] = triangle
        with self.assertRaises(ValueError):
            self.verify(original=original, descriptor=changed)

    def test_nonunit_source_weights_are_not_normalized_and_coordinate_error_is_separate(self):
        original = copy.deepcopy(self.original)
        original["gripperActuation"]["anchors"][0]["weights"] = [.5, .25, .2500000000000005]
        descriptor = self.producer(original, self.refined, subdivisions=64)
        self.verify(original=original, descriptor=descriptor)
        entry = descriptor["anchors"][0]
        source_sum = sum(map(F, original["gripperActuation"]["anchors"][0]["weights"]), F())
        self.assertNotEqual(source_sum, 1)
        self.assertEqual(rat(entry["originalWeightSum"]), source_sum)
        self.assertEqual(rat(entry["coefficientAudit"]["weightSums"]["source"]), source_sum)
        ordinary = self.descriptor["anchors"]
        self.assertTrue(all(rat(item["coefficientAudit"]["pullback"]["residualL1"]) == 0 for item in ordinary))
        self.assertTrue(any(any(map(rat, item["storedCoordinateResidualMeters"])) for item in ordinary))

    def test_nonzero_nearest_conversion_residual_matches_exact_material_action(self):
        original = copy.deepcopy(self.original)
        original["gripperActuation"]["anchors"][2]["weights"] = [.1, .2, .7]
        descriptor = self.producer(original, self.refined, subdivisions=64)
        self.verify(original=original, descriptor=descriptor)
        entry = descriptor["anchors"][2]
        self.assertEqual(entry["selectedLocalTriangleIndex"], 40)
        self.assertEqual(entry["selectedCanonicalTriangleIndex"], 88)
        self.assertEqual(entry["numericalAnchor"]["weights"], [.4, .3999999999999999, .2])
        audit = entry["coefficientAudit"]
        self.assertEqual(rat(audit["pullback"]["residualLInfinity"]), F(1, 2**55))
        self.assertEqual(rat(audit["weightSums"]["binary64DerivedMinusSource"]), -F(1, 2**55))
        lineage = basis(self.refined["bindingRefinement"])
        probe = [F((-1)**i * (i**3 + 11), 17) for i in range(11)]
        child = self.refined["bindingRefinement"]["finalLocalMesh"]["triangles"][40]
        refined_probe = {v: sum((weight * probe[old] for old, weight in lineage[v].items()), F()) for v in child}
        actual = sum((F(weight) * refined_probe[v] for v, weight in zip(child, entry["numericalAnchor"]["weights"])), F())
        expected = sum((F(weight) * probe[v] for v, weight in zip(entry["originalLocalVertices"], [.1, .2, .7])), F())
        recorded_error = sum((rat(item["binary64MinusSource"]) * probe[item["sourceVertex"]]
                              for item in audit["pullback"]["coefficients"]), F())
        self.assertNotEqual(recorded_error, 0)
        self.assertEqual(actual - expected, recorded_error)
        for item in audit["pullback"]["coefficients"]:
            self.assertLessEqual(abs(rat(item["binary64MinusSource"])), rat(item["absoluteRoundingBound"]))

    def test_schedule_stiffness_identity_order_and_release_are_exactly_preserved(self):
        numerical, original = self.descriptor["numericalRecipe"], self.original["gripperActuation"]
        self.assertEqual(encoded(numerical["schedule"]), encoded(original["schedule"]))
        self.assertEqual(numerical["schedule"]["knots"][-2]["activation"], [0.] * 3)
        changes = (
            ("target", lambda recipe: recipe["schedule"]["knots"][3]["targetsMeters"][0].__setitem__(0,
                math.nextafter(recipe["schedule"]["knots"][3]["targetsMeters"][0][0], math.inf))),
            ("activation", lambda recipe: recipe["schedule"]["knots"][-2]["activation"].__setitem__(0, .5)),
            ("knot", lambda recipe: recipe["schedule"]["knots"][1].__setitem__("fraction", 3 / 32)),
            ("stiffness", lambda recipe: recipe["anchors"][0].__setitem__("stiffnessNPerM", 2.)),
            ("id", lambda recipe: recipe["anchors"][0].__setitem__("id", "different")),
            ("instance", lambda recipe: recipe["anchors"][0].__setitem__("instanceId", "sleeve_left:shell")),
            ("order", lambda recipe: recipe["anchors"].reverse()),
        )
        for label, mutate in changes:
            descriptor = copy.deepcopy(self.descriptor)
            mutate(descriptor["numericalRecipe"])
            # Repair duplicated per-anchor numerical data as well, so rejection
            # is tied to the immutable original recipe and derived operator.
            for item, numerical_anchor in zip(descriptor["anchors"], descriptor["numericalRecipe"]["anchors"]):
                item["numericalAnchor"] = copy.deepcopy(numerical_anchor)
            with self.subTest(change=label), self.assertRaises(ValueError):
                self.verify(descriptor=descriptor)

    def test_candidate_completeness_parent_selection_and_repaired_numerical_attacks(self):
        mutations = (
            ("omitted", lambda d: d["anchors"][0]["candidates"].pop()),
            ("duplicate", lambda d: d["anchors"][0]["candidates"].append(copy.deepcopy(d["anchors"][0]["candidates"][0]))),
            ("parent", lambda d: d["anchors"][0]["candidates"][0].__setitem__("sourceParentTriangle", 5)),
            ("winding", lambda d: d["anchors"][0]["candidates"][0]["orientedVertices"].reverse()),
            ("classification", lambda d: d["anchors"][0]["candidates"][0].__setitem__("coefficientNonnegative", True)),
            ("weight", lambda d: d["anchors"][0]["candidates"][0]["exactWeights"][0].__setitem__("numerator", "123")),
            ("frame-equivalence", lambda d: d["anchors"][0].__setitem__("equivalentLocalTriangleIndices", [27])),
            ("chosen-face", lambda d: d["anchors"][0].__setitem__("selectedLocalTriangleIndex", 28)),
            ("numerical-map", lambda d: d["anchors"][0]["coefficientAudit"]["selectedNumericalWeights"][0].__setitem__("weight", .25)),
            ("residual", lambda d: d["anchors"][0]["coefficientAudit"]["pullback"]["coefficients"][0]["binary64MinusSource"].__setitem__("numerator", "1")),
            ("coordinate-residual", lambda d: d["anchors"][0]["storedCoordinateResidualMeters"][0].__setitem__("roundedBinary64", False)),
            ("sum", lambda d: d["anchors"][0]["originalWeightSum"].__setitem__("denominator", "0")),
            ("bool-index", lambda d: d["anchors"][0].__setitem__("gripperIndex", False)),
        )
        for label, mutate in mutations:
            descriptor = copy.deepcopy(self.descriptor)
            mutate(descriptor)
            self.assertTrue(encoded(descriptor) != encoded(self.descriptor), label)
            with self.subTest(change=label), self.assertRaises(ValueError):
                self.verify(descriptor=descriptor)

    def test_changed_original_base_refined_mesh_and_rehashed_lineage_reject(self):
        original = copy.deepcopy(self.original)
        original["sourcePattern"]["drafting"]["seamAllowanceMm"] = 9
        descriptor = copy.deepcopy(self.descriptor)
        descriptor["originalSourceSha256"] = sha(original)
        with self.assertRaises(ValueError):
            self.verify(original=original, descriptor=descriptor)
        for label in ("point", "parent", "basis", "instance-offset"):
            refined, descriptor = copy.deepcopy(self.refined), copy.deepcopy(self.descriptor)
            if label == "point":
                refined["restMeters"][40][0] = math.nextafter(refined["restMeters"][40][0], math.inf)
            elif label == "parent":
                parents = refined["bindingRefinement"]["stages"][1]["output"]["parentTriangles"]
                parents[0] = (parents[0] + 1) % len(refined["bindingRefinement"]["stages"][0]["output"]["triangles"])
            elif label == "basis":
                refined["bindingRefinement"]["stages"][1]["output"]["sourceWeights"][11] = {"0": 1.}
            else:
                refined["instanceOffsets"]["sleeve_left:shell"] -= 1
            descriptor["refinedSourceSha256"] = sha(refined)
            descriptor["refinedMeshSha256"] = mesh_sha(refined)
            descriptor["numericalRecipe"]["meshSha256"] = mesh_sha(refined)
            self.assertTrue(encoded(refined) != encoded(self.refined), label)
            with self.subTest(change=label), self.assertRaises(ValueError):
                self.verify(refined=refined, descriptor=descriptor)

    def test_raw_original_recipe_boolean_nonfinite_bounds_and_hash_repair_reject(self):
        mutations = (
            lambda recipe: recipe["anchors"][0].__setitem__("triangleIndex", True),
            lambda recipe: recipe["anchors"][0]["weights"].__setitem__(0, True),
            lambda recipe: recipe["anchors"][0].__setitem__("stiffnessNPerM", True),
            lambda recipe: recipe["schedule"]["knots"][0].__setitem__("fraction", False),
            lambda recipe: recipe["schedule"]["knots"][0]["targetsMeters"][0].__setitem__(0, True),
            lambda recipe: recipe["schedule"]["knots"][0]["activation"].__setitem__(0, True),
            lambda recipe: recipe["schedule"]["knots"][1].__setitem__("fraction", 1 / 128),
            lambda recipe: recipe["schedule"]["knots"][0]["targetsMeters"][0].__setitem__(0, 101.),
            lambda recipe: recipe["schedule"]["knots"][0]["activation"].__setitem__(0, -1e-12),
        )
        for index, mutate in enumerate(mutations):
            original, descriptor = copy.deepcopy(self.original), copy.deepcopy(self.descriptor)
            mutate(original["gripperActuation"])
            descriptor["originalSourceSha256"] = sha(original)
            descriptor["originalRecipeSha256"] = sha(original["gripperActuation"])
            descriptor["originalRecipe"] = copy.deepcopy(original["gripperActuation"])
            with self.subTest(change=index), self.assertRaises(ValueError):
                self.verify(original=original, descriptor=descriptor)
        original = copy.deepcopy(self.original)
        original["gripperActuation"]["anchors"][0]["weights"][0] = math.nan
        with self.assertRaises(ValueError):
            self.verify(original=original)

    def test_bare_core_exact_profiles_and_closed_descriptor_schema(self):
        for field in ("placedMeters", "gripperActuation", "sewingFrames", "bindingRefinedDiagnostic"):
            refined = copy.deepcopy(self.refined)
            refined[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(refined=refined)
        for target in ("original", "refined", "descriptor"):
            document = copy.deepcopy(getattr(self, target))
            document["profile"] = "generic"
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.verify(**{target: document})
        for field, value in (("accepted", 0), ("solverReady", True), ("executable", True),
                             ("subdivisions", 64.), ("inventedProof", True)):
            descriptor = copy.deepcopy(self.descriptor)
            descriptor[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(descriptor=descriptor)
        original = copy.deepcopy(self.original)
        original["baseUnit"] = copy.deepcopy(self.base)
        with self.assertRaises(ValueError):
            self.verify(original=original)

    def test_conversion_midpoints_underflow_and_resource_limits(self):
        a, b = .5, math.nextafter(.5, 1.)
        c = math.nextafter(b, 1.)
        self.assertEqual(_nearest((F(a) + F(b)) / 2), a)
        self.assertEqual(_nearest((F(b) + F(c)) / 2), c)
        smallest = math.nextafter(0., 1.)
        self.assertEqual(_nearest(F(smallest)), smallest)
        for value in (F(smallest) / 2, -F(smallest), F(2)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _nearest(value)
        for count in (True, False, 64., 0, 3, 8192):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.verify(subdivisions=count)
        original = copy.deepcopy(self.original)
        original["gripperActuation"]["anchors"] = [copy.deepcopy(original["gripperActuation"]["anchors"][0])] * 17
        with self.assertRaisesRegex(ValueError, "sixteen"):
            self.verify(original=original)
        for descriptor in ({"x": (1, 2)}, {1: "bad"}, {"n": 2**64}, {"x": "a" * (24 * 1024**2 + 1)}):
            with self.assertRaises(ValueError):
                self.verify(descriptor=descriptor)


if __name__ == "__main__":
    unittest.main()
