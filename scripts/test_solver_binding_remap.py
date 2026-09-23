"""Fresh-source remap checks with independent exact coefficient-space oracles.

No executable refined source, geometry simulation or construction completion is
tested here. A literal Linux anchor below is an algebraic counterexample only.
"""

import copy
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from solver_binding_refinement import build_binding_refinement
from solver_binding_remap import (POLICY, build_binding_seam_remap_descriptor,
    validate_binding_seam_remap_descriptor, _equivalent_selection,
    _nearest_binary64_weight, _solve_three)


SCRIPTS = Path(__file__).resolve().parent
INSTANCE = "opening_binding_left_left:shell"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rational(value):
    result = F(int(value["numerator"]), int(value["denominator"]))
    if (str(result.numerator) != value["numerator"] or str(result.denominator) != value["denominator"]
            or float(result) != value["roundedBinary64"]):
        raise AssertionError("Noncanonical or incorrectly rounded rational report")
    return result


def det(matrix):
    a, b, c = matrix
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def cramer(matrix, rhs):
    """Independent determinant solve, not production Gaussian elimination."""
    divisor = det(matrix)
    if divisor == 0:
        raise ValueError("Singular independent coefficient basis")
    result = []
    for column in range(3):
        changed = [row.copy() for row in matrix]
        for index in range(3):
            changed[index][column] = rhs[index]
        result.append(det(changed) / divisor)
    return result


def compose(refinement):
    original_count = len(refinement["originalLocalMesh"]["verticesMeters"])
    basis = [[F(int(row == column)) for column in range(original_count)] for row in range(original_count)]
    for stage in refinement["stages"]:
        previous = basis
        basis = [[sum((F(weight) * previous[int(vertex)][column] for vertex, weight in support.items()), F())
                  for column in range(original_count)] for support in stage["output"]["sourceWeights"]]
    return basis


def original_anchor(row, count=11):
    values = [F()] * count
    for term in row["terms"]:
        if term["instanceId"] == INSTANCE:
            values[term["vertex"]] = -F(term["coefficient"])
    return values


class BindingRemapArithmeticTests(unittest.TestCase):
    def test_nearest_even_conversion_midpoints_subnormals_and_forbidden_conversion(self):
        for lower in (.125, math.nextafter(.125, 1.), .25, .5, math.nextafter(.5, 1.)):
            upper = math.nextafter(lower, 1.)
            midpoint = (F(lower) + F(upper)) / 2
            lower_even = struct.unpack(">Q", struct.pack(">d", lower))[0] % 2 == 0
            self.assertEqual(_nearest_binary64_weight(midpoint), lower if lower_even else upper)
            radius = (F(upper) - F(lower)) / 1000
            self.assertEqual(_nearest_binary64_weight(midpoint - radius), lower)
            self.assertEqual(_nearest_binary64_weight(midpoint + radius), upper)
        smallest = math.nextafter(0., 1.)
        self.assertEqual(_nearest_binary64_weight(F(smallest)), smallest)
        self.assertEqual(_nearest_binary64_weight(F(smallest) * F(3, 4)), smallest)
        self.assertEqual(_nearest_binary64_weight(F(0)), 0.)
        self.assertEqual(_nearest_binary64_weight(F(1)), 1.)
        for value in (F(smallest) / 2, F(smallest) / 4, -F(smallest), F(2),
                      F(2**1024), F(2**16385), True, 0., float("nan")):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                _nearest_binary64_weight(value)

    def test_exact_solver_rejects_singular_shape_and_arithmetic_budget(self):
        matrix = [[F(1, 3), F(1, 7), F(0)], [F(1, 4), F(0), F(1, 2)], [F(1), F(1), F(1)]]
        rhs = [F(2, 9), F(3, 5), F(1) - F(1, 2**54)]
        self.assertEqual(_solve_three(matrix, rhs), cramer(matrix, rhs))
        for invalid, target in (([[F(1)] * 3] * 3, rhs), (matrix[:2], rhs),
                                (matrix, rhs[:2]), (matrix, [F(2**16385), F(1), F(1)])):
            with self.assertRaises(ValueError):
                _solve_three(invalid, target)

    def test_only_identical_sparse_child_operators_collapse_without_frame_choice(self):
        first = {"child": 8, "face": [1, 2, 3], "weights": [F(1, 2), F(1, 2), F(0)], "nonnegative": True}
        second = {"child": 4, "face": [4, 2, 1], "weights": [F(0), F(1, 2), F(1, 2)], "nonnegative": True}
        selected, equivalents = _equivalent_selection([first, second])
        self.assertEqual(selected["child"], 4)
        self.assertEqual(equivalents, [4, 8])
        conflicting = copy.deepcopy(second)
        conflicting["face"] = [4, 5, 6]
        with self.assertRaisesRegex(ValueError, "distinct nonnegative"):
            _equivalent_selection([first, conflicting])
        for candidates in ([], [dict(first, nonnegative=False)]):
            with self.assertRaisesRegex(ValueError, "no exactly nonnegative"):
                _equivalent_selection(candidates)


class BindingRemapFreshSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        commands = [("prepare-cuff-source.py", ["--output", root / "parent"]),
            ("prepare-cuff-construction.py", ["--source-canonical", root / "parent/canonical.json",
             "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1", "--output", root / "unit"])]
        for script, arguments in commands:
            process = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)],
                env=environment, cwd=root, capture_output=True, text=True, timeout=60)
            if process.returncode:
                raise AssertionError(process.stdout + process.stderr)
        cls.source = json.loads((root / "unit/unit.json").read_bytes())
        cls.refinement = build_binding_refinement(cls.source)
        cls.remap = build_binding_seam_remap_descriptor(cls.source, cls.refinement)
        cls.basis = compose(cls.refinement)

    def test_five_rows_keep_full_source_identity_and_no_executable_profile(self):
        source_before, refinement_before = encoded(self.source), encoded(self.refinement)
        result = validate_binding_seam_remap_descriptor(self.source, self.refinement, self.remap)
        self.assertEqual(result["sourceUnitSha256"], sha(self.source))
        self.assertEqual(result["refinementDescriptorSha256"], sha(self.refinement))
        self.assertEqual(result["originalEmbeddedConstraintsSha256"], sha(self.source["embeddedConstraints"]))
        self.assertEqual(result["policy"], POLICY)
        self.assertEqual(len(result["rows"]), 5)
        self.assertEqual([row["rowIndex"] for row in result["rows"]], list(range(5)))
        for row in result["rows"]:
            original = self.source["embeddedConstraints"]["constraints"][row["rowIndex"]]
            self.assertEqual(encoded(row["originalRow"]), encoded(original))
            self.assertEqual(row["originalRowSha256"], sha(original))
            fraction = F(original["fraction"])
            self.assertEqual(row["rowId"], "row:" + sha([original["registrationId"], original["memberIndex"],
                                                        fraction.numerator, fraction.denominator]))
        for field in ("accepted", "solverReady", "executable"):
            self.assertIs(result[field], False)
        self.assertFalse({"restMeters", "triangles", "embeddedConstraints", "sewingFrames",
                          "placedMeters", "gripperActuation", "sewingActuation"}.intersection(result))
        self.assertIn("never a frame-side", result["selectionPolicy"])
        self.assertIn("approximations", result["conversionPolicy"])
        self.assertIn("arbitrary independent refined", result["arithmeticScope"])
        result["rows"][0]["originalRow"]["terms"][0]["coefficient"] = 99.
        result["rows"][0]["selectedNumericalWeights"].clear()
        self.assertEqual(encoded(self.source), source_before)
        self.assertEqual(encoded(self.refinement), refinement_before)
        self.assertEqual(encoded(build_binding_seam_remap_descriptor(self.source, self.refinement)), encoded(self.remap))

    def test_all_children_and_five_selections_match_independent_exact_determinants(self):
        original_faces = self.refinement["originalLocalMesh"]["triangles"]
        final_faces = self.refinement["finalLocalMesh"]["triangles"]
        expected_selected = []
        for row in self.remap["rows"]:
            source_weights = original_anchor(row["originalRow"])
            eligible, observed_children = [], []
            for child, parent in enumerate(self.refinement["ultimateOriginalTriangleIndices"]):
                if parent not in row["originalParentTriangles"]:
                    continue
                face, old_face = final_faces[child], original_faces[parent]
                expected = cramer([[self.basis[vertex][index] for vertex in face] for index in old_face],
                                  [source_weights[index] for index in old_face])
                actual = row["candidates"][len(observed_children)]
                observed_children.append(child)
                self.assertEqual(actual["childTriangleIndex"], child)
                self.assertEqual(actual["sourceParentTriangle"], parent)
                self.assertEqual(actual["orientedVertices"], face)
                self.assertEqual(list(map(rational, actual["exactWeights"])), expected)
                self.assertIs(actual["coefficientNonnegative"], all(weight >= 0 for weight in expected))
                if all(weight >= 0 for weight in expected):
                    eligible.append(child)
            self.assertEqual(len(row["candidates"]), len(observed_children))
            self.assertEqual(len(eligible), 1)
            self.assertEqual(row["equivalentChildTriangleIndices"], eligible)
            self.assertEqual(row["selectedChildTriangleIndex"], eligible[0])
            expected_selected.append(eligible[0])
        if sys.platform.startswith("linux"):
            # Literal pinned Linux source convention. Other BLAS runtimes may
            # derive different immutable original coefficients and must solve
            # their own exact operators instead of silently reusing this list.
            self.assertEqual(expected_selected, [25, 35, 40, 14, 5])

    def test_rounding_cells_exact_pullbacks_bounds_and_nonunit_source_sums(self):
        witnessed_nonzero_residual = witnessed_nonunit_sum = False
        for row in self.remap["rows"]:
            source_weights = original_anchor(row["originalRow"])
            ideal = {entry["vertex"]: rational(entry["weight"]) for entry in row["exactSelectedWeights"]}
            numerical = {entry["vertex"]: F(entry["weight"]) for entry in row["selectedNumericalWeights"]}
            self.assertEqual(set(ideal), set(numerical))
            for vertex, expected in ideal.items():
                value = float(numerical[vertex])
                lower, upper = F(math.nextafter(value, -math.inf)), F(math.nextafter(value, math.inf))
                midpoint_low, midpoint_high = (lower + numerical[vertex]) / 2, (numerical[vertex] + upper) / 2
                self.assertLessEqual(midpoint_low, expected)
                self.assertLessEqual(expected, midpoint_high)
                if expected in (midpoint_low, midpoint_high):
                    self.assertEqual(struct.unpack(">Q", struct.pack(">d", value))[0] % 2, 0)
            residuals, bounds = [], []
            for index, weight in enumerate(source_weights):
                self.assertEqual(sum((value * self.basis[v][index] for v, value in ideal.items()), F()), weight)
                residual = sum((value * self.basis[v][index] for v, value in numerical.items()), F()) - weight
                bound = sum((abs((numerical[v] - value) * self.basis[v][index]) for v, value in ideal.items()), F())
                observed = row["pullback"]["coefficients"][index]
                self.assertEqual(observed["sourceVertex"], index)
                self.assertEqual(rational(observed["binary64MinusSource"]), residual)
                self.assertEqual(rational(observed["absoluteRoundingBound"]), bound)
                self.assertLessEqual(abs(residual), bound)
                residuals.append(residual)
                bounds.append(bound)
            for key, expected in (("residualL1", sum(map(abs, residuals), F())),
                                  ("residualLInfinity", max(map(abs, residuals))),
                                  ("roundingBoundL1", sum(bounds, F())), ("roundingBoundLInfinity", max(bounds)),
                                  ("residualSum", sum(residuals, F()))):
                self.assertEqual(rational(row["pullback"][key]), expected)
            sums = [sum(values, F()) for values in (source_weights, ideal.values(), numerical.values())]
            for key, expected in zip(("source", "exactDerived", "binary64Derived"), sums):
                self.assertEqual(rational(row["weightSums"][key]), expected)
            self.assertEqual(rational(row["weightSums"]["exactDerivedMinusSource"]), sums[1] - sums[0])
            self.assertEqual(rational(row["weightSums"]["binary64DerivedMinusSource"]), sums[2] - sums[0])
            witnessed_nonzero_residual |= any(residuals)
            witnessed_nonunit_sum |= sums[0] != 1
        self.assertTrue(witnessed_nonzero_residual)
        self.assertTrue(witnessed_nonunit_sum)

    def test_exact_nonplanar_material_pullback_and_binary_error_remain_distinct(self):
        # Original nodal values are deliberately non-affine and nonplanar.
        source_positions = [[F(index, 7), F((-1)**index * index**2, 11), F(index**3, 13)] for index in range(11)]
        # A translation must retain the actual source weight sum, which need
        # not be exactly one. This also catches silent affine normalization.
        probes = [source_positions, [[point[1] + 3, -point[0] + 5, point[2] - 2] for point in source_positions]]
        for probe in probes:
            refined = [[sum((weight * point[axis] for weight, point in zip(row, probe)), F()) for axis in range(3)]
                       for row in self.basis]
            for row in self.remap["rows"]:
                source_weights = original_anchor(row["originalRow"])
                ideal = {entry["vertex"]: rational(entry["weight"]) for entry in row["exactSelectedWeights"]}
                numerical = {entry["vertex"]: F(entry["weight"]) for entry in row["selectedNumericalWeights"]}
                residuals = [rational(entry["binary64MinusSource"]) for entry in row["pullback"]["coefficients"]]
                for axis in range(3):
                    expected = sum((weight * point[axis] for weight, point in zip(source_weights, probe)), F())
                    self.assertEqual(sum((weight * refined[v][axis] for v, weight in ideal.items()), F()), expected)
                    actual = sum((weight * refined[v][axis] for v, weight in numerical.items()), F())
                    self.assertEqual(actual - expected, sum((error * point[axis] for error, point in zip(residuals, probe)), F()))

    def test_stored_coordinate_residuals_are_separate_from_operator_rounding(self):
        original = self.refinement["originalLocalMesh"]["verticesMeters"]
        final = self.refinement["finalLocalMesh"]["verticesMeters"]
        witnessed = False
        for row in self.remap["rows"]:
            source_weights = original_anchor(row["originalRow"])
            ideal = {entry["vertex"]: rational(entry["weight"]) for entry in row["exactSelectedWeights"]}
            numerical = {entry["vertex"]: F(entry["weight"]) for entry in row["selectedNumericalWeights"]}
            for axis in range(2):
                source = sum((weight * F(point[axis]) for weight, point in zip(source_weights, original)), F())
                exact = sum((weight * F(final[v][axis]) for v, weight in ideal.items()), F())
                binary = sum((weight * F(final[v][axis]) for v, weight in numerical.items()), F())
                for key, expected in (("exactWeightsMinusSource", exact - source),
                                      ("binary64WeightsMinusSource", binary - source),
                                      ("binary64MinusExactWeights", binary - exact)):
                    self.assertEqual(rational(row["storedCoordinateResidualsMeters"][key][axis]), expected)
                witnessed |= exact != source
        self.assertTrue(witnessed)

    def test_literal_linux_row_one_coordinate_containment_selects_wrong_side(self):
        # This literal source coefficient vector is the retained Linux row-1
        # anchor, used as an algebraic test on the same refinement geometry.
        # It is never substituted into or admitted as this runtime's base unit.
        weights = [F()] * 11
        for vertex, value in ((3, .0006965293186517847), (7, .4993034706813482), (8, .5)):
            weights[vertex] = F(value)
        parent = self.refinement["originalLocalMesh"]["triangles"][9]
        original = self.refinement["originalLocalMesh"]["verticesMeters"]
        final = self.refinement["finalLocalMesh"]
        point = [sum((weight * F(vertex[axis]) for weight, vertex in zip(weights, original)), F()) for axis in range(2)]
        choices, geometric_choices = [], []
        for child, ancestor in enumerate(self.refinement["ultimateOriginalTriangleIndices"]):
            if ancestor != 9:
                continue
            face = final["triangles"][child]
            exact = _solve_three([[self.basis[v][i] for v in face] for i in parent], [weights[i] for i in parent])
            geometric = cramer([[F(final["verticesMeters"][v][axis]) for v in face] for axis in range(2)] + [[F(1)] * 3],
                               point + [sum(weights, F())])
            if all(value >= 0 for value in exact):
                choices.append(child)
            if all(value >= 0 for value in geometric):
                geometric_choices.append(child)
            if child == 37:
                self.assertEqual(min(exact), -F(109, 2**63))
                # A flat-coordinate match still changes a nonplanar material
                # operator; the mismatch cannot be excused by tiny XY error.
                pullback = [sum((geometric[j] * self.basis[v][i] for j, v in enumerate(face)), F()) - weights[i]
                            for i in range(11)]
                self.assertTrue(any(pullback))
        self.assertEqual(choices, [35])
        self.assertEqual(geometric_choices, [37])

    def test_tiny_positive_weights_survive_and_clipping_or_normalization_is_rejected(self):
        tiny = [(i, j) for i, row in enumerate(self.remap["rows"])
                for j, entry in enumerate(row["selectedNumericalWeights"]) if 0 < entry["weight"] < 1e-12]
        self.assertEqual(len(tiny), 5)
        for index, entry_index in tiny:
            changed = copy.deepcopy(self.remap)
            changed["rows"][index]["selectedNumericalWeights"].pop(entry_index)
            with self.assertRaisesRegex(ValueError, "differs from source rederivation"):
                validate_binding_seam_remap_descriptor(self.source, self.refinement, changed)
        changed = copy.deepcopy(self.remap)
        index = next(i for i, row in enumerate(changed["rows"]) if rational(row["weightSums"]["source"]) != 1)
        row = changed["rows"][index]
        denominator = sum((rational(entry["weight"]) for entry in row["exactSelectedWeights"]), F())
        for entry in row["exactSelectedWeights"]:
            value = rational(entry["weight"]) / denominator
            entry["weight"] = {"numerator": str(value.numerator), "denominator": str(value.denominator), "roundedBinary64": float(value)}
        with self.assertRaises(ValueError):
            validate_binding_seam_remap_descriptor(self.source, self.refinement, changed)

    def test_repaired_hash_candidate_parent_row_and_residual_substitutions_reject(self):
        attacks = (
            ("accepted", lambda d: d.__setitem__("accepted", 0)),
            ("executable", lambda d: d.__setitem__("executable", True)),
            ("row-index-bool", lambda d: d["rows"][0].__setitem__("rowIndex", False)),
            ("row-id", lambda d: d["rows"][0].__setitem__("rowId", d["rows"][1]["rowId"])),
            ("source-row", lambda d: d["rows"][0]["originalRow"].__setitem__("complianceMPerN", 2e-8)),
            ("selected-child", lambda d: d["rows"][0].__setitem__("selectedChildTriangleIndex", 0)),
            ("child-order", lambda d: d["rows"][0]["candidates"].reverse()),
            ("parent", lambda d: d["rows"][0]["candidates"][0].__setitem__("sourceParentTriangle", 0)),
            ("negative-hidden", lambda d: d["rows"][0]["candidates"][0].__setitem__("coefficientNonnegative", True)),
            ("residual", lambda d: d["rows"][0]["pullback"]["coefficients"][0]["binary64MinusSource"].__setitem__("numerator", "9")),
            ("round-bound", lambda d: d["rows"][0]["pullback"]["roundingBoundL1"].__setitem__("numerator", "0")),
            ("normalized-source-sum", lambda d: d["rows"][0]["weightSums"].__setitem__("source", {"numerator": "1", "denominator": "1", "roundedBinary64": 1.})),
            ("coordinate-error", lambda d: d["rows"][0]["storedCoordinateResidualsMeters"]["exactWeightsMinusSource"][0].__setitem__("roundedBinary64", False)),
            ("scope", lambda d: d.__setitem__("scope", "Exact preserved executable operators")),
            ("missing-row", lambda d: d["rows"].pop()),
            ("extra", lambda d: d.__setitem__("frameSide", "body")),
        )
        for label, mutate in attacks:
            changed = copy.deepcopy(self.remap)
            mutate(changed)
            for row in changed["rows"]:
                row["originalRowSha256"] = sha(row["originalRow"])
            # Changing a field to its pre-existing value would be a vacuous
            # attack; choose a second candidate when the first is valid.
            if encoded(changed) == encoded(self.remap) and label == "negative-hidden":
                next(item for item in changed["rows"][0]["candidates"] if not item["coefficientNonnegative"])["coefficientNonnegative"] = True
            self.assertNotEqual(encoded(changed), encoded(self.remap), label)
            with self.subTest(attack=label), self.assertRaisesRegex(ValueError, "differs from source rederivation"):
                validate_binding_seam_remap_descriptor(self.source, self.refinement, changed)

    def test_source_refinement_raw_types_policy_and_resource_bounds_fail_closed(self):
        for policy in (None, False, [], {}, "", POLICY + " "):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                build_binding_seam_remap_descriptor(self.source, self.refinement, policy=policy)
        for bad in (None, [], {"coordinate": math.nan}, {"x": (1, 2)}, {1: "bad key"}, {"integer": 2**64}, {"text": "x" * (4 * 1024**2 + 1)}):
            with self.subTest(bad_type=type(bad).__name__), self.assertRaises(ValueError):
                validate_binding_seam_remap_descriptor(self.source, self.refinement, bad)
        for target in ("source", "refinement"):
            source, refinement = copy.deepcopy(self.source), copy.deepcopy(self.refinement)
            if target == "source":
                source["embeddedConstraints"]["constraints"][0]["terms"][0]["coefficient"] = -.5
                refinement["sourceUnitSha256"] = sha(source)
                refinement["originalEmbeddedConstraintsSha256"] = sha(source["embeddedConstraints"])
            else:
                refinement["stages"][1]["output"]["sourceWeights"][11] = {"0": 1.}
            with self.subTest(target=target), self.assertRaises(ValueError):
                build_binding_seam_remap_descriptor(source, refinement)


if __name__ == "__main__":
    unittest.main()
