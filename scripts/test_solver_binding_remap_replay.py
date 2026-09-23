"""Independent remap arithmetic, fresh-source consistency and repaired attacks."""

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

from solver_binding_remap_replay import verify_binding_remap, _nearest, _solve, _anchor_candidates


SCRIPTS = Path(__file__).resolve().parent
INSTANCE = "opening_binding_left_left:shell"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rat(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def raw_basis(descriptor):
    rows = [{i: Fraction(1)} for i in range(11)]
    for stage in descriptor["stages"]:
        before = rows
        rows = []
        for support in stage["output"]["sourceWeights"]:
            row = {}
            for intermediate, outer in support.items():
                for original, inner in before[int(intermediate)].items():
                    row[original] = row.get(original, Fraction()) + Fraction(outer) * inner
            rows.append(row)
    return rows


def gaussian(matrix, rhs):
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for column in range(3):
        pivot = next(i for i in range(column, 3) if rows[i][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for i in range(3):
            if i == column:
                continue
            amount = rows[i][column]
            rows[i] = [value - amount * reference for value, reference in zip(rows[i], rows[column])]
    return [row[-1] for row in rows]


class BindingRemapReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        commands = [[sys.executable, str(SCRIPTS / "prepare-cuff-source.py"), "--output", str(root / "parent")],
            [sys.executable, str(SCRIPTS / "prepare-cuff-construction.py"), "--source-canonical", str(root / "parent/canonical.json"),
             "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1", "--output", str(root / "left")],
            [sys.executable, "-c", "import json,pathlib,sys;sys.path.insert(0,sys.argv[1]);"
             "from solver_binding_source import build_binding_source;"
             "p=pathlib.Path(sys.argv[2]);p.write_text(json.dumps(build_binding_source(json.loads(pathlib.Path(sys.argv[3]).read_text()))))",
             str(SCRIPTS), str(root / "derived.json"), str(root / "left/unit.json")]]
        for command in commands:
            process = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=90)
            if process.returncode:
                raise AssertionError(process.stdout + process.stderr)
        cls.source = json.loads((root / "derived.json").read_text())
        cls.source_path = root / "derived.json"

    def test_fresh_five_fabric_unit_and_full_source_digest(self):
        before = encoded(self.source)
        result = verify_binding_remap(self.source)
        self.assertEqual(encoded(self.source), before)
        self.assertIs(result["verified"], True)
        self.assertIs(result["accepted"], False)
        self.assertEqual((result["rowCount"], result["remappedRows"], result["unchangedRows"]), (40, 5, 35))
        self.assertEqual(result["sourceSha256"], digest(self.source))
        self.assertEqual(result["enumeratedCandidateCount"], 23)
        self.assertIn("separately captured source validator", result["scope"])
        changed = copy.deepcopy(self.source)
        changed["sewingActuation"] = {"purpose": "independently admitted by sewing verifier"}
        self.assertEqual(verify_binding_remap(changed)["sourceSha256"], result["sourceSha256"])
        changed["placedMeters"] = changed["restMeters"]
        self.assertNotEqual(verify_binding_remap(changed)["sourceSha256"], result["sourceSha256"])

    def test_current_verifier_imports_no_engine_or_producer_even_during_verification(self):
        command = """import json,sys
sys.path.insert(0,sys.argv[1])
before=set(sys.modules)
from solver_binding_remap_replay import verify_binding_remap,REQUIRED_HELPER_FILES
assert REQUIRED_HELPER_FILES==()
verify_binding_remap(json.load(open(sys.argv[2])))
new=set(sys.modules)-before
assert not any(x in new for x in ('solver_binding_remap','solver_binding_refinement','solver_binding_source','assembly','cloth_domain','embedded_constraints'))
assert not any(x.startswith(('numpy','scipy','shapely')) for x in new)
"""
        process = subprocess.run([sys.executable, "-c", command, str(SCRIPTS), str(self.source_path)],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_exact_dense_oracle_matches_all_candidates_and_nonplanar_residuals(self):
        descriptor = self.source["bindingRefinement"]
        basis = raw_basis(descriptor)
        witness_nonunit, witness_rounding, candidate_count = False, False, 0
        # Non-affine original nodal values expose coefficient errors which a
        # flat rest-coordinate match cannot observe. This is algebra, not cloth.
        probe = [Fraction((-1) ** i * (i ** 3 + 7), 13) for i in range(11)]
        refined_probe = [sum((weight * probe[i] for i, weight in row.items()), Fraction()) for row in basis]
        for row in self.source["bindingSeamRemap"]["rows"]:
            weights = {item["vertex"]: rat(item["weight"]) for item in row["originalWeights"]}
            witness_nonunit |= sum(weights.values(), Fraction()) != 1
            eligible = []
            for candidate in row["candidates"]:
                face = candidate["orientedVertices"]
                parent = descriptor["originalLocalMesh"]["triangles"][candidate["sourceParentTriangle"]]
                matrix = [[basis[v].get(i, Fraction()) for v in face] for i in parent]
                expected = gaussian(matrix, [weights.get(i, Fraction()) for i in parent])
                self.assertEqual(expected, [rat(v) for v in candidate["exactWeights"]])
                self.assertEqual(_solve(matrix, [weights.get(i, Fraction()) for i in parent]), expected)
                self.assertEqual(candidate["coefficientNonnegative"], all(v >= 0 for v in expected))
                if all(v >= 0 for v in expected):
                    eligible.append(candidate["childTriangleIndex"])
                exact_value = sum((weight * refined_probe[v] for v, weight in zip(face, expected)), Fraction())
                source_value = sum((weight * probe[v] for v, weight in weights.items()), Fraction())
                self.assertEqual(exact_value, source_value)
                candidate_count += 1
            self.assertEqual(eligible, row["equivalentChildTriangleIndices"])
            rounded = row["selectedNumericalWeights"]
            numerical_value = sum((Fraction(item["weight"]) * refined_probe[item["vertex"]] for item in rounded), Fraction())
            residual = sum((rat(item["binary64MinusSource"]) * probe[item["sourceVertex"]]
                            for item in row["pullback"]["coefficients"]), Fraction())
            self.assertEqual(numerical_value - source_value, residual)
            witness_rounding |= residual != 0
            for item in row["pullback"]["coefficients"]:
                self.assertLessEqual(abs(rat(item["binary64MinusSource"])), rat(item["absoluteRoundingBound"]))
        self.assertTrue(witness_nonunit)
        self.assertTrue(witness_rounding)
        self.assertEqual(candidate_count, 23)

    def test_nearest_even_midpoints_and_underflow_admission(self):
        a = .5
        b = math.nextafter(a, math.inf)
        c = math.nextafter(b, math.inf)
        self.assertEqual(_nearest((Fraction(a) + Fraction(b)) / 2), a)
        self.assertEqual(_nearest((Fraction(b) + Fraction(c)) / 2), c)
        self.assertEqual(_nearest(Fraction(math.ulp(0.))), math.ulp(0.))
        for value in (Fraction(math.ulp(0.)) / 2, Fraction(-1, 10), Fraction(2)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _nearest(value)

    def test_literal_source_counterexample_uses_coefficients_not_stored_geometry(self):
        descriptor = self.source["bindingRefinement"]
        sparse = raw_basis(descriptor)
        basis = [[row.get(i, Fraction()) for i in range(11)] for row in sparse]
        parents = descriptor["ultimateOriginalTriangleIndices"]
        row = {"terms": [{"instanceId": INSTANCE, "vertex": i, "coefficient": -weight}
                         for i, weight in ((3, .0006965293186517847), (7, .4993034706813482), (8, .5))]}
        source, _, candidates, eligible = _anchor_candidates(row, descriptor, basis, parents)
        self.assertEqual([item[0] for item in eligible], [35])
        original = descriptor["originalLocalMesh"]["verticesMeters"]
        point = [sum((weight * Fraction(vertex[axis]) for weight, vertex in zip(source, original)), Fraction()) for axis in range(2)]
        geometric = []
        for child, face, _, _ in candidates:
            stored = [descriptor["finalLocalMesh"]["verticesMeters"][v] for v in face]
            matrix = [[Fraction(v[0]) for v in stored], [Fraction(v[1]) for v in stored], [Fraction(1)] * 3]
            values = gaussian(matrix, point + [sum(source, Fraction())])
            if all(value >= 0 for value in values):
                geometric.append(child)
        self.assertEqual(geometric, [37])

    def test_boundary_equivalence_preserves_all_candidates_without_frame_choice(self):
        unit = lambda i: [Fraction(int(i == j)) for j in range(11)]
        basis = [unit(0), unit(1), unit(2), unit(2)]
        refinement = {"originalLocalMesh": {"triangles": [[0, 1, 2]]},
                      "finalLocalMesh": {"triangles": [[0, 1, 2], [0, 1, 3]]}}
        row = {"terms": [{"instanceId": INSTANCE, "vertex": 0, "coefficient": -1.}]}
        _, _, _, eligible = _anchor_candidates(row, refinement, basis, [0, 0])
        self.assertEqual([item[0] for item in eligible], [0, 1])
        # Both exact supports reduce to vertex0. Moving into the triangle would
        # expose two different refined-node operators, requiring a policy.
        row = {"terms": [{"instanceId": INSTANCE, "vertex": i, "coefficient": -.25}
                         for i in range(3)]}
        with self.assertRaisesRegex(ValueError, "Different child material operators"):
            _anchor_candidates(row, refinement, basis, [0, 0])

    def test_source_sum_translation_leakage_is_reported_not_normalized_away(self):
        rows = self.source["bindingSeamRemap"]["rows"]
        nonzero = False
        for row in rows:
            errors = [rat(item["binary64MinusSource"]) for item in row["pullback"]["coefficients"]]
            total = sum(errors, Fraction())
            self.assertEqual(total, rat(row["pullback"]["residualSum"]))
            self.assertEqual(total, rat(row["weightSums"]["binary64DerivedMinusSource"]))
            nonzero |= total != 0
            translation = Fraction(7, 16)
            self.assertEqual(sum((error * translation for error in errors), Fraction()), total * translation)
        self.assertTrue(nonzero)

    def test_candidate_completeness_identity_and_nonnegative_attacks(self):
        def other_child(s):
            row = s["bindingSeamRemap"]["rows"][1]
            row["selectedChildTriangleIndex"] = next(item["childTriangleIndex"] for item in row["candidates"]
                if item["childTriangleIndex"] != row["selectedChildTriangleIndex"])
        def flip_classification(s):
            item = s["bindingSeamRemap"]["rows"][0]["candidates"][0]
            item["coefficientNonnegative"] = not item["coefficientNonnegative"]
        mutations = [
            lambda s: s["bindingSeamRemap"]["rows"][0]["candidates"].pop(),
            lambda s: s["bindingSeamRemap"]["rows"][0]["candidates"].reverse(),
            lambda s: s["bindingSeamRemap"]["rows"][0]["candidates"].append(copy.deepcopy(s["bindingSeamRemap"]["rows"][0]["candidates"][0])),
            other_child, flip_classification,
            lambda s: s["bindingSeamRemap"]["rows"][0]["candidates"][0].update(sourceParentTriangle=0),
            lambda s: s["bindingSeamRemap"]["rows"][0]["candidates"][0]["orientedVertices"].reverse(),
            lambda s: s["bindingSeamRemap"]["rows"][0].update(equivalentChildTriangleIndices=[])]
        self.reject_mutations(mutations)

    def test_rounded_weight_clamp_normalization_and_repaired_residual_attacks(self):
        def one_ulp(s):
            item = s["bindingSeamRemap"]["rows"][0]["selectedNumericalWeights"][0]
            item["weight"] = math.nextafter(item["weight"], math.inf)
        def clamp(s):
            values = s["bindingSeamRemap"]["rows"][0]["selectedNumericalWeights"]
            min(values, key=lambda item: item["weight"])["weight"] = 0.
        def normalize(s):
            values = s["bindingSeamRemap"]["rows"][0]["selectedNumericalWeights"]
            smallest = min(values, key=lambda item: item["weight"])
            old = smallest["weight"]
            smallest["weight"] = float(1 - sum((Fraction(item["weight"]) for item in values if item is not smallest), Fraction()))
            self.assertNotEqual(old, smallest["weight"], "Normalization attack must actually change the source operator")
        def forged_residual(s):
            row = s["bindingSeamRemap"]["rows"][0]
            for item in row["pullback"]["coefficients"]:
                item["binary64MinusSource"] = {"numerator": "0", "denominator": "1", "roundedBinary64": 0.}
            row["pullback"]["residualL1"] = {"numerator": "0", "denominator": "1", "roundedBinary64": 0.}
        self.reject_mutations([one_ulp, clamp, normalize, forged_residual,
            lambda s: s["bindingSeamRemap"]["rows"][0]["weightSums"]["source"].update(numerator="1", denominator="1")])

    def test_raw_stage_and_lineage_attacks_despite_repaired_descriptor_digest(self):
        self.reject_mutations([
            lambda s: s["bindingRefinement"]["stages"][0]["output"]["parentTriangles"].__setitem__(0, True),
            lambda s: s["bindingRefinement"]["stages"][1]["output"]["sourceWeights"][0].update({"0": 0.5}),
            lambda s: s["bindingRefinement"]["composedVertexWeights"][0]["weights"][0]["weight"].update(numerator="2"),
            lambda s: s["bindingRefinement"]["ultimateOriginalTriangleIndices"].reverse(),
            lambda s: s["bindingRefinement"]["finalLocalMesh"]["triangles"][0].reverse(),
            lambda s: s["bindingRefinement"]["stages"].reverse()])

    def test_derived_mesh_original_rows_compliance_and_samples_cannot_change(self):
        self.reject_mutations([
            lambda s: s["embeddedConstraints"]["constraints"][5]["terms"][0].update(coefficient=0.),
            lambda s: s["embeddedConstraints"]["constraints"][0].update(complianceMPerN=2e-8),
            lambda s: s["embeddedConstraints"]["constraints"][0]["sourceSamples"][0]["weights"][0].update(weight=.5),
            lambda s: s["restMeters"][0].__setitem__(0, .01),
            lambda s: s["numericalMeshes"][INSTANCE]["verticesMeters"][0].__setitem__(0, .01),
            lambda s: s["instanceOffsets"].update({INSTANCE: True}),
            lambda s: s["instanceTriangleOffsets"].update({INSTANCE: 0}),
            lambda s: s["sourceRowCorrespondence"][0].update(bindingAnchorRemapped=1),
            lambda s: s["embeddedConstraints"].update(originalBundleSha256="0" * 64)])

    def test_profile_downgrade_raw_boolean_malformed_and_budget_rejection(self):
        self.reject_mutations([
            lambda s: s.update(profile="source-cuff-construction-unit-v1"),
            lambda s: s["baseUnit"].update(profile="source-left-binding-refined-unit-v1"),
            lambda s: s["bindingSeamRemap"].update(profile="almost-right", accepted=0),
            lambda s: s["bindingSeamRemap"]["rows"][0].update(anchorSign=True),
            lambda s: s["bindingSeamRemap"]["rows"][0]["pullback"]["residualL1"].update(numerator="00"),
            lambda s: s.update(sewingFrames={}),
            lambda s: s["baseUnit"].update(placedMeters=[]),
            lambda s: s.update(accepted=0),
            lambda s: s.update(instances=tuple(s["instances"])),
            lambda s: s.update(unboundedInteger=2 ** 80),
            lambda s: s.update(nonnumeric=float("nan"))], repair=False)

    def reject_mutations(self, mutations, repair=True):
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = copy.deepcopy(self.source)
                mutate(changed)
                if repair:
                    # A digest is not a mathematical witness. Repair the obvious
                    # claimed identities so rejection exercises rederivation.
                    changed["bindingSeamRemap"]["sourceUnitSha256"] = digest(changed["baseUnit"])
                    changed["bindingSeamRemap"]["refinementDescriptorSha256"] = digest(changed["bindingRefinement"])
                with self.assertRaises(ValueError):
                    verify_binding_remap(changed)


if __name__ == "__main__":
    unittest.main()
