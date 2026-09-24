"""Static source-placement regressions; no contact dynamics or runner changes."""

import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from solver_binding_placement import (POLICY, REQUEST_PROFILE, _affine, build_binding_placement,
    validate_binding_placement)
from solver_binding_source import build_binding_source


SCRIPTS = Path(__file__).resolve().parent
INSTANCE = "opening_binding_left_left:shell"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rat(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def request_for(source):
    angle = math.radians(17.)
    c, s = math.cos(angle), math.sin(angle)
    return {"profile": REQUEST_PROFILE, "accepted": False, "coordinatePolicy": POLICY,
        "minimumSeparationMeters": .0002,
        "poses": [{"instanceId": instance["id"], "rotation": [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]],
            "translationMeters": [.125, -.25, index * .1]}
            for index, instance in enumerate(source["instances"])]}


class BindingPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        for script, arguments in (("prepare-cuff-source.py", ["--output", root / "parent"]),
                ("prepare-cuff-construction.py", ["--source-canonical", root / "parent/canonical.json",
                    "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1", "--output", root / "unit"])):
            completed = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)],
                cwd=root, env=environment, capture_output=True, text=True, timeout=60)
            if completed.returncode:
                raise AssertionError(completed.stdout + completed.stderr)
        cls.source = build_binding_source(json.loads((root / "unit/unit.json").read_bytes()))
        cls.request = request_for(cls.source)
        cls.descriptor = build_binding_placement(cls.source, cls.request)

    def test_all_vertices_once_rounded_against_independent_decimal_oracle(self):
        with localcontext() as context:
            context.prec = 200
            for pose, instance in zip(self.request["poses"], self.descriptor["instances"]):
                start, end = instance["canonicalVertexRange"]
                for vertex in range(start, end):
                    point = self.source["restMeters"][vertex]
                    expected = [float(sum((Decimal(value) * Decimal(coordinate) for value, coordinate in zip(row, point)), Decimal(shift)))
                        for row, shift in zip(pose["rotation"], pose["translationMeters"])]
                    self.assertEqual(encoded(expected), encoded(self.descriptor["placedMeters"][vertex]))
        self.assertEqual(len(self.descriptor["placedMeters"]), 245)
        self.assertEqual(sum(len(item["edges"]) for item in self.descriptor["instances"]), 638)
        self.assertEqual(sum(len(item["triangles"]) for item in self.descriptor["instances"]), 398)
        for key in ("accepted", "solverReady", "executable"):
            self.assertIs(self.descriptor[key], False)

    def test_exact_edge_decomposition_and_source_triangle_orientation(self):
        nonzero_rotation, nonzero_rounding = False, False
        for instance in self.descriptor["instances"]:
            for edge in instance["edges"]:
                rotation = rat(edge["rotationMetricDefectMetersSquared"])
                cross = rat(edge["roundingCrossTermMetersSquared"])
                square = rat(edge["roundingSquaredTermMetersSquared"])
                self.assertEqual(rat(edge["squaredLengthChangeMetersSquared"]), rotation + cross + square)
                self.assertEqual(rat(edge["placedSquaredLengthMetersSquared"]) - rat(edge["restSquaredLengthMetersSquared"]), rotation + cross + square)
                self.assertGreaterEqual(square, 0)
                self.assertLessEqual(abs(rat(edge["relativeSquaredLengthChange"])), Fraction(4096, 2**52))
                nonzero_rotation |= rotation != 0
                nonzero_rounding |= cross != 0
            self.assertTrue(all(rat(face["orientedAreaDotMetersFourth"]) > 0 for face in instance["triangles"]))
        self.assertTrue(nonzero_rotation)
        self.assertTrue(nonzero_rounding)

    def test_raw_lineage_commutation_retains_rounding_without_vertex_repair(self):
        entries = self.descriptor["bindingLineage"]
        self.assertEqual(len(entries), 29)
        self.assertTrue(self.descriptor["originalBindingPrefixPreservedUnderDeclaredEvaluation"])
        for item in entries:
            self.assertEqual([rat(value) for value in item["placedMinusInterpolatedOriginalMeters"]],
                [rat(a) + rat(b) for a, b in zip(item["idealAffineCommutatorMeters"], item["placementRoundingCommutatorMeters"])])
        self.assertTrue(any(rat(value) for item in entries for value in item["placedMinusInterpolatedOriginalMeters"]))
        self.assertTrue(any(rat(value) for item in entries for value in item["sourceCoordinateResidualMeters"]))
        for item in entries[:11]:
            self.assertEqual([rat(value) for value in item["placedMinusInterpolatedOriginalMeters"]], [0, 0, 0])

    def test_separating_planes_cover_every_pair_and_exact_all_vertex_extrema(self):
        identities = [item["id"] for item in self.source["instances"]]
        expected = {(a, b) for i, a in enumerate(identities) for b in identities[i + 1:]}
        self.assertEqual({(item["firstInstanceId"], item["secondInstanceId"]) for item in self.descriptor["separatingPlanes"]}, expected)
        ranges = {item["instanceId"]: item["canonicalVertexRange"] for item in self.descriptor["instances"]}
        for item in self.descriptor["separatingPlanes"]:
            axis, sign = item["axisIndex"], item["directionSign"]
            first = [sign * Fraction(point[axis]) for point in self.descriptor["placedMeters"][slice(*ranges[item["firstInstanceId"]])]]
            second = [sign * Fraction(point[axis]) for point in self.descriptor["placedMeters"][slice(*ranges[item["secondInstanceId"]])]]
            self.assertEqual(rat(item["separatingPlaneGapMeters"]), min(second) - max(first))
            self.assertGreaterEqual(min(second) - max(first), Fraction(self.request["minimumSeparationMeters"]))

    def test_general_three_dimensional_proper_rotation_preserves_scope(self):
        request = copy.deepcopy(self.request)
        c, s = math.cos(.41), math.sin(.41)
        for index, pose in enumerate(request["poses"]):
            pose["rotation"] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
            pose["translationMeters"][2] = float(index)
        result = build_binding_placement(self.source, request)
        self.assertEqual(len(result["separatingPlanes"]), 10)
        self.assertTrue(all(rat(item["rotationDeterminant"]) > 0 for item in result["instances"]))

    def test_affine_cancellation_and_signed_subnormal_rounding_are_not_hidden(self):
        c = Fraction(math.sqrt(.5))
        rotation = [[c, c, Fraction()], [-c, c, Fraction()], [Fraction(), Fraction(), Fraction(1)]]
        exact, numerical, residual = _affine([[Fraction(1), Fraction(1, 2**54), Fraction()]], rotation, [-c, Fraction(), Fraction()])
        self.assertEqual(exact[0][0], c / 2**54)
        self.assertNotEqual(numerical[0][0], 0.)
        tiny_rotation = [[Fraction(1), Fraction(), Fraction()], [Fraction(), Fraction(1), Fraction()], [Fraction(-1, 2**1074), Fraction(), Fraction(1)]]
        _, output, errors = _affine([[Fraction(1, 4), Fraction(), Fraction()]], tiny_rotation, [Fraction()] * 3)
        self.assertEqual(math.copysign(1., output[0][2]), -1.)
        self.assertEqual(errors[0][2], Fraction(1, 2**1076))
        self.assertEqual(Fraction(numerical[0][0]) - exact[0][0], residual[0][0])

    def test_reflection_scaling_shear_and_nonbinary_raw_poses_reject(self):
        matrices = [
            [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            [[1.000001, 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            [[1., .000001, 0.], [0., 1., 0.], [0., 0., 1.]],
            [[True, 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            [[1., 0.], [0., 1., 0.], [0., 0., 1.]],
        ]
        for matrix in matrices:
            request = copy.deepcopy(self.request)
            request["poses"][0]["rotation"] = matrix
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                build_binding_placement(self.source, request)

    def test_exact_gram_bound_rejects_excess_hidden_by_binary64_dot_rounding(self):
        request = copy.deepcopy(self.request)
        scale = 1. + 2.**-47
        self.assertEqual(scale * scale - 1., 64. * 2.**-52)
        self.assertGreater(Fraction(scale)**2 - 1, Fraction(64, 2**52))
        request["poses"][0]["rotation"] = [[scale, 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        with self.assertRaisesRegex(ValueError, "rotation"):
            build_binding_placement(self.source, request)
        request["poses"][0]["rotation"][0][0] = 1. + 2.**-48
        result = build_binding_placement(self.source, request)
        self.assertNotEqual(rat(result["instances"][0]["rotationGramMinusIdentity"][0][0]), 0)

    def test_incomplete_reordered_unknown_and_zero_clearance_requests_reject(self):
        cases = []
        for key, value in (("accepted", True), ("coordinatePolicy", "normalize"), ("minimumSeparationMeters", 0),
                           ("minimumSeparationMeters", True), ("minimumSeparationMeters", float("nan")), ("unknown", 1)):
            request = copy.deepcopy(self.request)
            request[key] = value
            cases.append(request)
        missing = copy.deepcopy(self.request)
        missing["poses"].pop()
        cases.append(missing)
        reordered = copy.deepcopy(self.request)
        reordered["poses"].reverse()
        cases.append(reordered)
        extra_pose = copy.deepcopy(self.request)
        extra_pose["poses"][0]["sourceMirrorX"] = False
        cases.append(extra_pose)
        for index, request in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                build_binding_placement(self.source, request)

    def test_overlapping_or_under_clearance_instances_reject(self):
        request = copy.deepcopy(self.request)
        request["poses"][1] = {**copy.deepcopy(request["poses"][0]), "instanceId": request["poses"][1]["instanceId"]}
        with self.assertRaisesRegex(ValueError, "clearance"):
            build_binding_placement(self.source, request)
        request = copy.deepcopy(self.request)
        request["minimumSeparationMeters"] = .2
        with self.assertRaisesRegex(ValueError, "clearance"):
            build_binding_placement(self.source, request)

    def test_strict_witness_tampering_and_input_immutability(self):
        mutations = [lambda d: d["placedMeters"][0].__setitem__(0, math.nextafter(d["placedMeters"][0][0], math.inf)),
            lambda d: d["instances"][0]["edges"].pop(),
            lambda d: d["instances"][0]["triangles"][0]["orientedAreaDotMetersFourth"].__setitem__("numerator", "0"),
            lambda d: d["bindingLineage"][20]["placementRoundingCommutatorMeters"][1].__setitem__("numerator", "123"),
            lambda d: d["separatingPlanes"].pop(), lambda d: d.__setitem__("accepted", True),
            lambda d: d.__setitem__("sourceSha256", "0" * 64)]
        original_source, original_request = encoded(self.source), encoded(self.request)
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(self.descriptor)
            mutate(changed)
            self.assertNotEqual(encoded(changed), encoded(self.descriptor))
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_binding_placement(self.source, changed)
        self.assertEqual(encoded(validate_binding_placement(self.source, self.descriptor)), encoded(self.descriptor))
        self.assertEqual(encoded(self.source), original_source)
        self.assertEqual(encoded(self.request), original_request)
        for field in ("placedMeters", "foldActuation", "gripperActuation", "sewingActuation", "assemblySchedule", "bindingRefinedDiagnostic"):
            changed = copy.deepcopy(self.source)
            changed[field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_binding_placement(changed, self.request)


if __name__ == "__main__":
    unittest.main()
