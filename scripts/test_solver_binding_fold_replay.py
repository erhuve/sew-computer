"""Independent topology/arithmetic audit tests; no dynamics or acceptance.

Fresh source fixtures use the producer only to create descriptors. The audit
module runs with standard-library imports and independently reconstructs rules.
"""

import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

from solver_binding_fold_replay import verify_binding_fold_control, _sqrt_round, _rail_partition, _native, _edges


SCRIPTS = Path(__file__).resolve().parent
INSTANCE = "opening_binding_left_left:shell"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rational(value):
    return F(int(value["numerator"]), int(value["denominator"]))


def native_topology(source):
    """Test-only incidence construction, independent of native runtime ordering."""
    faces = [source["triangles"][i:i + 3] for i in range(0, len(source["triangles"]), 3)]
    rows = {}
    for face in faces:
        for slot in range(3):
            a, b, opposite = face[slot], face[(slot + 1) % 3], face[(slot + 2) % 3]
            key = tuple(sorted((a, b)))
            if key not in rows:
                rows[key] = [opposite, -1, a, b]
            else:
                assert rows[key][2:] == [b, a] and rows[key][1] == -1
                rows[key][1] = opposite
    return {"vertexCount": len(source["restMeters"]), "triangles": faces, "hinges": list(rows.values())}


def request(folds=None):
    return {"profile": "source-binding-fold-request-v1", "accepted": False,
        "angleConvention": "source-directed-right-hand-reference-region-v1",
        "stiffnessMeasure": "stored-crease-edge-length-v1",
        "folds": folds if folds is not None else [
            {"sourcePathName": "right", "referenceRotationRegion": "body",
             "targetRightHandAngleRadians": .125, "stiffnessJoulesPerMeter": .37},
            {"sourcePathName": "left", "referenceRotationRegion": "allowance",
             "targetRightHandAngleRadians": -.25, "stiffnessJoulesPerMeter": .19}]}


class ExactFoldArithmeticTests(unittest.TestCase):
    def test_independent_square_root_decimal_and_nearest_even_midpoints(self):
        generator = random.Random(317)
        with localcontext() as context:
            context.prec = 200
            for _ in range(100):
                values = [math.ldexp(generator.uniform(-2, 2), generator.randrange(-100, 9)) for _ in range(3)]
                square = sum((F(value) ** 2 for value in values), F())
                decimal = (Decimal(square.numerator) / Decimal(square.denominator)).sqrt()
                self.assertEqual(_sqrt_round(square), float(decimal))
        for below in (0., math.ulp(0.), math.nextafter(1., 0.), 1., math.nextafter(1., math.inf), 16.):
            above = math.nextafter(below, math.inf)
            midpoint = (F(below) + F(above)) / 2
            bits = struct.unpack(">Q", struct.pack(">d", below))[0]
            expected = below if bits % 2 == 0 else above
            if expected == 0:
                with self.assertRaises(ValueError):
                    _sqrt_round(midpoint ** 2)
            else:
                self.assertEqual(_sqrt_round(midpoint ** 2), expected)
            perturbation = (F(above) - F(below)) ** 2 / 16
            self.assertEqual(_sqrt_round(midpoint ** 2 + perturbation), above)
            if below:
                self.assertEqual(_sqrt_round(midpoint ** 2 - perturbation), below)
        for bad in (F(), F(-1), 1., True, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _sqrt_round(bad)

    def test_dual_graph_partition_is_topological_and_directional(self):
        # Two columns, two rows; central chain 1->4->7. No positions enter the
        # classifier, so changing tiny rounded coordinate signs cannot change it.
        faces = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
                 [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]]
        body, allowance, pairs = _rail_partition(faces, [1, 4, 7])
        self.assertEqual(body, [0, 1, 4, 5])
        self.assertEqual(allowance, [2, 3, 6, 7])
        reverse_body, reverse_allowance, _ = _rail_partition(faces, [7, 4, 1])
        self.assertEqual(reverse_body, allowance)
        self.assertEqual(reverse_allowance, body)
        self.assertEqual(len(pairs), 2)
        for bad in ([1, 4], [1, 7], [1, 4, 1], [0, 1, 4, 7], [1, 4, 5, 4, 7]):
            with self.subTest(chain=bad), self.assertRaises((ValueError, StopIteration)):
                _rail_partition(faces, bad)

    def test_all_native_edge_incidence_and_both_valid_orderings(self):
        faces = [[0, 1, 2], [0, 2, 3]]
        source = {"restMeters": [[0., 0., 0.]] * 4, "triangles": sum(faces, [])}
        topology = native_topology(source)
        positions, owners, edges = source["restMeters"], ["one"] * 4, _edges(faces)
        self.assertEqual(len(_native(topology, positions, faces, owners, edges)), 5)
        for index in range(5):
            changed = copy.deepcopy(topology)
            a, b, c, d = changed["hinges"][index]
            changed["hinges"][index] = [b, a, d, c]
            self.assertEqual(len(_native(changed, positions, faces, owners, edges)), 5)
            for permutation in ([b, a, c, d], [a, b, d, c]):
                changed["hinges"][index] = permutation
                with self.assertRaises(ValueError):
                    _native(changed, positions, faces, owners, edges)
        for change in (lambda x: x["hinges"].pop(), lambda x: x["hinges"].append(x["hinges"][0]),
                       lambda x: x["hinges"][0].__setitem__(0, True),
                       lambda x: x["hinges"][0].__setitem__(1, -2),
                       lambda x: x["hinges"][0].__setitem__(2, 999)):
            changed = copy.deepcopy(topology)
            change(changed)
            with self.assertRaises(ValueError):
                _native(changed, positions, faces, owners, edges)
        with self.assertRaises(ValueError):
            _native(topology, positions, faces, ["one", "one", "two", "two"], edges)


class BindingFoldReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from solver_binding_source import build_binding_source
        from solver_binding_fold import build_binding_fold_control
        cls.producer = staticmethod(build_binding_fold_control)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        for script, arguments in (("prepare-cuff-source.py", ["--output", cls.directory / "parent"]),
            ("prepare-cuff-construction.py", ["--source-canonical", cls.directory / "parent/canonical.json",
             "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1",
             "--output", cls.directory / "unit"])):
            process = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)], cwd=cls.directory,
                env=env, capture_output=True, text=True, timeout=90)
            if process.returncode:
                raise AssertionError(process.stdout + process.stderr)
        base = json.loads((cls.directory / "unit/unit.json").read_bytes())
        cls.source = build_binding_source(base)
        cls.topology = native_topology(cls.source)
        cls.request = request()
        cls.descriptor = cls.producer(cls.source, cls.request, cls.topology)
        for name, value in (("source", cls.source), ("descriptor", cls.descriptor)):
            (cls.directory / f"{name}.json").write_bytes(encoded(value))

    def verify(self, descriptor=None, source=None):
        return verify_binding_fold_control(self.source if source is None else source,
                                           self.descriptor if descriptor is None else descriptor)

    def test_fresh_source_complete_identity_and_result_isolation(self):
        before = encoded([self.source, self.descriptor])
        result = self.verify()
        self.assertIs(result["verified"], True)
        self.assertIs(result["accepted"], False)
        self.assertEqual([result["sourceVertexCount"], result["sourceTriangleCount"], result["hingeCount"]], [245, 398, 16])
        self.assertEqual(result["railNames"], ["right", "left"])
        self.assertEqual(result["sourceSha256"], sha(self.source))
        self.assertEqual(result["descriptorSha256"], sha(self.descriptor))
        self.assertEqual(result["foldActuationSha256"], sha(self.descriptor["foldActuation"]))
        self.assertIn("separately captured source validator", result["sourceScope"])
        result["railNames"].clear()
        self.assertEqual(encoded([self.source, self.descriptor]), before)
        self.assertEqual(self.verify()["railNames"], ["right", "left"])

    def test_verifier_execution_imports_standard_library_only(self):
        command = """import json,sys
sys.path.insert(0,sys.argv[1]); before=set(sys.modules)
from solver_binding_fold_replay import verify_binding_fold_control,REQUIRED_HELPER_FILES
assert REQUIRED_HELPER_FILES==()
source=json.load(open(sys.argv[2]+'/source.json')); descriptor=json.load(open(sys.argv[2]+'/descriptor.json'))
assert verify_binding_fold_control(source,descriptor)['verified'] is True
new=set(sys.modules)-before
assert not any(name.startswith(('numpy','scipy','shapely','newton','warp')) for name in new)
assert not any(name in new for name in ('solver_binding_fold','solver_binding_source','solver_binding_refinement',
 'solver_binding_remap','solver_fold_actuation','solver_bending','shirt','assembly'))
"""
        process = subprocess.run([sys.executable, "-c", command, str(SCRIPTS), str(self.directory)],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_actual_native_adjacency_and_reversed_valid_rows(self):
        from newton._src.utils.mesh import MeshAdjacency
        topology = copy.deepcopy(self.topology)
        topology["hinges"] = MeshAdjacency(topology["triangles"]).edge_indices.tolist()
        actual = self.producer(self.source, self.request, topology)
        self.assertEqual(self.verify(actual)["hingeCount"], 16)
        reversed_topology = copy.deepcopy(topology)
        reversed_topology["hinges"] = [[b, a, d, c] for a, b, c, d in topology["hinges"]][::-1]
        reversed_descriptor = self.producer(self.source, self.request, reversed_topology)
        self.assertEqual(self.verify(reversed_descriptor)["hingeCount"], 16)
        for original, reversed_rail in zip(actual["rails"], reversed_descriptor["rails"]):
            for old, new in zip(original["segments"], reversed_rail["segments"]):
                self.assertEqual(old["nativeAngleSign"], new["nativeAngleSign"])
                self.assertEqual(old["targetNativeAngleRadians"], new["targetNativeAngleRadians"])
                self.assertEqual(old["numericalStiffnessJoules"], new["numericalStiffnessJoules"])
                self.assertEqual(old["nativeEdgeDirectionSign"], -new["nativeEdgeDirectionSign"])

    def test_every_rail_hinge_all_twenty_four_permutations(self):
        positions, faces = self.source["restMeters"], self.topology["triangles"]
        owners = [None] * len(positions)
        for name, mesh in self.source["numericalMeshes"].items():
            offset = self.source["instanceOffsets"][name]
            owners[offset:offset + len(mesh["verticesMeters"])] = [name] * len(mesh["verticesMeters"])
        edges = _edges(faces)
        admitted = 0
        for rail in self.descriptor["rails"]:
            for segment in rail["segments"]:
                native_index = segment["nativeHingeIndex"]
                hinge = segment["sourceDirectedHingeCanonical"]
                allowed = {tuple(hinge), (hinge[1], hinge[0], hinge[3], hinge[2])}
                for permutation in itertools.permutations(hinge):
                    changed = copy.deepcopy(self.topology)
                    changed["hinges"][native_index] = list(permutation)
                    if permutation in allowed:
                        _native(changed, positions, faces, owners, edges)
                        admitted += 1
                    else:
                        with self.assertRaises(ValueError):
                            _native(changed, positions, faces, owners, edges)
        self.assertEqual(admitted, 32)

    def test_region_reference_sign_by_independent_rigid_rotations(self):
        def subtract(a, b):
            return [x - y for x, y in zip(a, b)]
        def cross(a, b):
            return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
        def dot(a, b):
            return math.fsum(x*y for x, y in zip(a, b))
        def unit(a):
            length = math.sqrt(dot(a, a))
            return [x / length for x in a]
        def angle(points, hinge):
            first, second, start, end = [points[v] for v in hinge]
            normal1 = unit(cross(subtract(start, first), subtract(end, first)))
            normal2 = unit(cross(subtract(end, second), subtract(start, second)))
            return math.atan2(dot(cross(normal1, normal2), unit(subtract(end, start))), dot(normal1, normal2))
        def rotate(point, start, end, theta):
            axis, vector = unit(subtract(end, start)), subtract(point, start)
            perpendicular = cross(axis, vector)
            return [start[i] + vector[i]*math.cos(theta) + perpendicular[i]*math.sin(theta)
                    + axis[i]*dot(axis, vector)*(1-math.cos(theta)) for i in range(3)]
        theta = .17
        for rail in self.descriptor["rails"]:
            for segment in rail["segments"]:
                hinge = segment["sourceDirectedHingeCanonical"]
                for region, moving, expected in (("body", 0, -theta), ("allowance", 1, theta)):
                    points = copy.deepcopy(self.source["restMeters"])
                    points[hinge[moving]] = rotate(points[hinge[moving]], points[hinge[2]], points[hinge[3]], theta)
                    self.assertAlmostEqual(angle(points, segment["nativeOrderedHingeCanonical"]), expected, delta=2e-13)
                points = copy.deepcopy(self.source["restMeters"])
                for moving in (0, 1):
                    points[hinge[moving]] = rotate(points[hinge[moving]], points[hinge[2]], points[hinge[3]], theta)
                self.assertAlmostEqual(angle(points, segment["nativeOrderedHingeCanonical"]), 0., delta=2e-13)

    def test_exact_lengths_once_rounded_products_and_rail_sums(self):
        self.verify()
        nonzero_length_residual = nonzero_product_residual = False
        with localcontext() as context:
            context.prec = 180
            for rail in self.descriptor["rails"]:
                ideal_total = binary_total = F()
                for segment in rail["segments"]:
                    a, b = [self.source["restMeters"][v] for v in segment["orientedEdgeCanonical"]]
                    square = sum(((F(x)-F(y))**2 for x,y in zip(a,b)), F())
                    length = segment["length"]
                    self.assertEqual(rational(length["exactSquaredLengthMetersSquared"]), square)
                    expected_length = float((Decimal(square.numerator)/Decimal(square.denominator)).sqrt())
                    self.assertEqual(length["roundedLengthMeters"], expected_length)
                    residual = F(expected_length)**2-square
                    self.assertEqual(rational(length["roundedLengthSquaredMinusExactMetersSquared"]), residual)
                    nonzero_length_residual |= residual != 0
                    ideal = F(rail["stiffnessJoulesPerMeter"])*F(expected_length)
                    numerical = segment["numericalStiffnessJoules"]
                    self.assertEqual(numerical, float(Decimal(ideal.numerator)/Decimal(ideal.denominator)))
                    self.assertEqual(rational(segment["stiffnessRoundingResidualJoules"]), F(numerical)-ideal)
                    nonzero_product_residual |= F(numerical) != ideal
                    ideal_total += ideal
                    binary_total += F(numerical)
                self.assertEqual(rational(rail["idealTotalStiffnessFromRoundedLengthsJoules"]), ideal_total)
                self.assertEqual(rational(rail["numericalTotalStiffnessJoules"]), binary_total)
                self.assertEqual(rational(rail["totalStiffnessRoundingResidualJoules"]), binary_total-ideal_total)
        self.assertTrue(nonzero_length_residual)
        self.assertTrue(nonzero_product_residual)

    def test_complete_derived_fields_and_repaired_hash_attacks_reject(self):
        attacks = [
            ("missing rail", lambda d: d["rails"].pop()),
            ("missing segment", lambda d: d["rails"][0]["segments"].pop()),
            ("region", lambda d: d["rails"][0].__setitem__("referenceRotationRegion", "allowance")),
            ("body partition", lambda d: d["rails"][0]["bodyFacesLocal"].pop()),
            ("body normal", lambda d: d["rails"][0]["bodyLeftNormal"].__setitem__(0, 1.)),
            ("source path", lambda d: d["rails"][0].__setitem__("sourcePathSha256", "0"*64)),
            ("native index", lambda d: d["rails"][0]["segments"][0].__setitem__("nativeHingeIndex", 0)),
            ("native sign", lambda d: d["rails"][0]["segments"][0].__setitem__("nativeAngleSign", -1)),
            ("body global", lambda d: d["rails"][0]["segments"][0].__setitem__("bodyFaceCanonical", 0)),
            ("exact length", lambda d: d["rails"][0]["segments"][0]["length"]["exactSquaredLengthMetersSquared"].__setitem__("numerator", "0")),
            ("length cell", lambda d: d["rails"][0]["segments"][0]["length"]["upperMidpointSquaredMetersSquared"].__setitem__("roundedBinary64", 0.)),
            ("rounding witness", lambda d: d["rails"][0]["segments"][0]["stiffnessRoundingResidualJoules"].__setitem__("numerator", "0")),
            ("rail sum", lambda d: d["rails"][0]["numericalTotalStiffnessJoules"].__setitem__("numerator", "0")),
            ("held side claim", lambda d: d["foldActuation"].__setitem__("fixedRegion", "allowance")),
            ("recipe target", lambda d: d["foldActuation"]["targetAnglesRadians"].__setitem__(0, .125)),
            ("initial target", lambda d: d["foldActuation"]["initialAnglesRadians"].__setitem__(0, .01)),
            ("pending bool", lambda d: d.__setitem__("accepted", 0)),
            ("sign bool", lambda d: d["rails"][0]["segments"][0].__setitem__("nativeAngleSign", True)),
            ("units", lambda d: d["units"].__setitem__("length", "mm")),
            ("request target", lambda d: d["request"]["folds"][0].__setitem__("targetRightHandAngleRadians", .2)),
            ("missing native edge", lambda d: d["nativeTopology"]["hinges"].pop()),
            ("boundary corruption", lambda d: d["nativeTopology"]["hinges"][0].__setitem__(1, -2)),
        ]
        for label, mutate in attacks:
            with self.subTest(attack=label):
                changed = copy.deepcopy(self.descriptor)
                mutate(changed)
                self.assertNotEqual(encoded(changed), encoded(self.descriptor))
                changed["requestSha256"] = sha(changed["request"])
                changed["nativeTopologySha256"] = sha(changed["nativeTopology"])
                with self.assertRaises(ValueError):
                    self.verify(changed)

    def test_source_and_topology_substitution_reject_after_hash_repair(self):
        attacks = [
            ("source winding", lambda s: s["numericalMeshes"][INSTANCE]["triangles"][0].reverse()),
            ("offset", lambda s: s["instanceOffsets"].__setitem__(INSTANCE, 41)),
            ("source chain", lambda s: s["bindingRefinement"]["finalCreases"][0]["vertices"].pop()),
            ("chain bool", lambda s: s["bindingRefinement"]["finalCreases"][0]["vertices"].__setitem__(0, True)),
            ("source line", lambda s: s["bindingRefinement"]["finalCreases"][0]["sourceLineEndpointsMeters"][0].__setitem__(0, .019)),
            ("source path", lambda s: s["baseUnit"]["sourcePattern"]["panels"][0].__setitem__("id", "removed")),
            ("source profile", lambda s: s.__setitem__("profile", "synthetic")),
            ("installed controls", lambda s: s.__setitem__("foldActuation", {})),
            ("acceptance bool", lambda s: s.__setitem__("accepted", 0)),
        ]
        for label, mutate in attacks:
            with self.subTest(attack=label):
                source, descriptor = copy.deepcopy(self.source), copy.deepcopy(self.descriptor)
                mutate(source)
                descriptor["sourceSha256"] = sha(source)
                descriptor["baseUnitSha256"] = sha(source["baseUnit"])
                with self.assertRaises(ValueError):
                    self.verify(descriptor, source)

    def test_single_rail_request_order_bounds_and_strict_raw_values(self):
        for folds in ([self.request["folds"][1]], list(reversed(self.request["folds"]))):
            descriptor = self.producer(self.source, request(copy.deepcopy(folds)), self.topology)
            self.assertEqual(self.verify(descriptor)["railNames"], [fold["sourcePathName"] for fold in folds])
        attacks = [("targetRightHandAngleRadians", value) for value in (True, math.pi-1e-8, math.inf, "0", [])]
        attacks += [("stiffnessJoulesPerMeter", value) for value in (False, 0, -1, 1000000.1, math.ulp(0.))]
        for key, value in attacks:
            with self.subTest(key=key, value=value):
                descriptor = copy.deepcopy(self.descriptor)
                descriptor["request"]["folds"][0][key] = value
                if type(value) is not float or math.isfinite(value):
                    descriptor["requestSha256"] = sha(descriptor["request"])
                with self.assertRaises(ValueError):
                    self.verify(descriptor)
        for value in (None, [], {"profile": "wrong"}, {"x": (1, 2)}, {"x": "x" * (24 * 1024**2 + 1)}):
            with self.assertRaises(ValueError):
                verify_binding_fold_control(self.source, value)


if __name__ == "__main__":
    unittest.main()
