"""Independent static placement arithmetic and repaired-evidence attacks.

Fresh source fixtures use the producer to supply claims, never to verify them.
Small helper fixtures are mathematical controls, not accepted garment inputs.
"""

import copy
from decimal import Decimal, localcontext
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

from solver_binding_placement_replay import (verify_binding_placement, _pose, _placed, _round_nearest,
    _gram_error, _edge_terms, _lineage_audit, _separating_planes, _instance_audit)


SCRIPTS = Path(__file__).resolve().parent
INSTANCE = "opening_binding_left_left:shell"
IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rat(value):
    return F(int(value["numerator"]), int(value["denominator"]))


def bits(value):
    return struct.pack(">d", value)


def request(source, *, zero_rotation=False):
    poses = []
    for index, instance in enumerate(source["instances"]):
        theta = .17 + .11*index
        c, s = math.cos(theta), math.sin(theta)
        rotation = copy.deepcopy(IDENTITY) if zero_rotation else [[1., 0., 0.], [0., c, -s], [0., s, c]]
        poses.append({"instanceId": instance["id"], "rotation": rotation,
                      "translationMeters": [.125, -.0625, float(index)]})
    return {"profile": "source-binding-rigid-placement-request-v1", "accepted": False,
        "coordinatePolicy": "exact-affine-once-rounded-binary64-v1", "minimumSeparationMeters": .1, "poses": poses}


class PlacementArithmeticTests(unittest.TestCase):
    def test_exact_affine_conversion_decimal_midpoints_and_signed_zero(self):
        smallest = math.ulp(0.)
        for below in (-1., math.nextafter(-1., 0.), -smallest, 0., smallest, 1., math.nextafter(1., math.inf)):
            above = math.nextafter(below, math.inf)
            midpoint = (F(below)+F(above))/2
            even = struct.unpack(">Q", bits(below))[0] % 2 == 0
            expected = below if even else above
            self.assertEqual(bits(_round_nearest(midpoint)), bits(expected))
        self.assertEqual(bits(_round_nearest(F())), bits(0.))
        self.assertEqual(bits(_round_nearest(-F(smallest)/4)), bits(-0.))
        self.assertEqual(bits(_round_nearest(F(smallest)/4)), bits(0.))
        c, s = math.cos(.713), math.sin(.713)
        matrix = [[F(c), F(-s), F()], [F(s), F(c), F()], [F(), F(), F(1)]]
        points = [[F(.03125), F(.02734375), F(-smallest)/4], [F(1), F(1), F()]]
        translation = [-F(float(F(c)*points[0][0]+F(-s)*points[0][1])), F(.0625), F()]
        exact, numerical, residuals = _placed(points, matrix, translation)
        with localcontext() as context:
            context.prec = 1200
            for i, point in enumerate(points):
                for axis in range(3):
                    expected = sum((Decimal(matrix[axis][j].numerator)/Decimal(matrix[axis][j].denominator)
                                    * Decimal(point[j].numerator)/Decimal(point[j].denominator) for j in range(3)), Decimal())
                    expected += Decimal(translation[axis].numerator)/Decimal(translation[axis].denominator)
                    self.assertEqual(bits(numerical[i][axis]), bits(float(expected)))
                    self.assertEqual(residuals[i][axis], F(numerical[i][axis])-exact[i][axis])
        self.assertNotEqual(exact[0][0], 0, "The test must retain the tiny exact cancellation residual")

    def test_proper_rotation_fixed_boundary_without_normalization(self):
        pose = {"instanceId": "test", "rotation": copy.deepcopy(IDENTITY), "translationMeters": [0., 0., 0.]}
        # The upper admissible binary64 diagonal is not guessed from a float
        # tolerance: compare its exact squared metric defect to the fixed bound.
        bound = F(64, 2**52)
        diagonal = 1.
        while F(math.nextafter(diagonal, math.inf))**2-1 <= bound:
            diagonal = math.nextafter(diagonal, math.inf)
        pose["rotation"][0][0] = diagonal
        matrix, _, gram, determinant = _pose(pose)
        self.assertEqual(matrix[0][0], F(diagonal))
        self.assertEqual(gram[0][0], F(diagonal)**2-1)
        self.assertEqual(determinant, F(diagonal))
        self.assertNotEqual(gram[0][0], 0)
        pose["rotation"][0][0] = math.nextafter(diagonal, math.inf)
        with self.assertRaises(ValueError):
            _pose(pose)
        for matrix in ([[-1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], [[1.,1e-8,0.],[0.,1.,0.],[0.,0.,1.]],
                       [[True,0.,0.],[0.,1.,0.],[0.,0.,1.]], [[float("nan"),0.,0.],[0.,1.,0.],[0.,0.,1.]]):
            pose["rotation"] = matrix
            with self.assertRaises(ValueError):
                _pose(pose)

    def test_exact_edge_decomposition_and_rounding_loss_reject(self):
        matrix = [[F(value) for value in row] for row in IDENTITY]
        matrix[0][0] = F(math.nextafter(1., math.inf))
        points = [[F(.1),F(.2),F()], [F(.3),F(.7),F(.125)]]
        _, placed, errors = _placed(points, matrix, [F(.33),F(-.17),F(.0625)])
        terms = _edge_terms(*points, matrix, _gram_error(matrix), *errors)
        before = sum(((b-a)**2 for a,b in zip(*points)), F())
        after = sum(((F(b)-F(a))**2 for a,b in zip(*placed)), F())
        self.assertEqual(sum(terms, F()), after-before)
        self.assertTrue(all(term != 0 for term in terms))
        # This synthetic local mesh has an edge too short to retain under a
        # one-meter translation. The ordinary near-rigid matrix bound cannot
        # admit its collapsed rounding result or normalize it afterward.
        tiny = math.ulp(0.)
        mesh = {"verticesMeters": [[0.,0.,0.],[tiny,0.,0.],[0.,tiny,0.]], "triangles": [[0,1,2]]}
        pose = _pose({"instanceId":"test","rotation":IDENTITY,"translationMeters":[1.,1.,1.]})
        with self.assertRaises(ValueError):
            _instance_audit("test", mesh, 0, pose)

    def test_nonunit_raw_lineage_requires_translation_commutator(self):
        # An arithmetic-only fixture intentionally has a nonunit source sum.
        # This is not a source-validator bypass or executable garment fixture.
        old = [[i/32, i/64] for i in range(11)]
        weights = [F(.25), F(math.nextafter(.75, math.inf))] + [F()]*9
        basis = [[F(int(i==j)) for j in range(11)] for i in range(11)] + [weights]
        new = [float(sum((w*F(old[i][axis]) for i,w in enumerate(weights)), F())) for axis in range(2)]
        points = [point+[0.] for point in old+[new]]
        source = {"bindingRefinement":{"originalLocalMesh":{"verticesMeters":old}},
                  "numericalMeshes":{INSTANCE:{"verticesMeters":points}}}
        pose = _pose({"instanceId":INSTANCE,"rotation":IDENTITY,"translationMeters":[2.,-1.,.5]})
        exact, placed, errors = _placed([list(map(F, point)) for point in points], pose[0], pose[1])
        audit = _lineage_audit(source, basis, pose, placed, exact, errors)[-1]
        self.assertNotEqual(rat(audit["sourceCoefficientSum"]), 1)
        residual = list(map(rat, audit["sourceCoordinateResidualMeters"]))
        commutator = list(map(rat, audit["idealAffineCommutatorMeters"]))
        self.assertEqual(commutator, [residual[i]+(1-sum(weights))*pose[1][i] for i in range(3)])
        self.assertNotEqual(commutator, residual)
        self.assertEqual(list(map(rat,audit["placedMinusInterpolatedOriginalMeters"])),
            [a+b for a,b in zip(commutator,map(rat,audit["placementRoundingCommutatorMeters"]))])

    def test_all_ten_static_plane_pairs_exact_boundary_and_first_tie(self):
        names = [str(i) for i in range(5)]
        ranges = {name:(i,i+1) for i,name in enumerate(names)}
        points = [[float(i),float(i),float(i)] for i in range(5)]
        result = _separating_planes(names, ranges, points, F(1))
        self.assertEqual(len(result), 10)
        self.assertTrue(all(item["axisIndex"]==0 and item["directionSign"]==1 for item in result))
        reverse = _separating_planes(list(reversed(names)), ranges, points, F(1))
        self.assertTrue(all(item["axisIndex"]==0 and item["directionSign"]==-1 for item in reverse))
        with self.assertRaises(ValueError):
            _separating_planes(names,ranges,points,F(math.nextafter(1.,math.inf)))


class BindingPlacementReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from solver_binding_source import build_binding_source
        from solver_binding_placement import build_binding_placement
        cls.producer = staticmethod(build_binding_placement)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        env = {**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
        for script, arguments in (("prepare-cuff-source.py",["--output",cls.directory/"parent"]),
            ("prepare-cuff-construction.py",["--source-canonical",cls.directory/"parent/canonical.json","--side","left",
             "--attachment-policy","inner-facing-first-outer-shell-last-v1","--output",cls.directory/"unit"])):
            process = subprocess.run([sys.executable,str(SCRIPTS/script),*map(str,arguments)],cwd=cls.directory,
                env=env,capture_output=True,text=True,timeout=90)
            if process.returncode:
                raise AssertionError(process.stdout+process.stderr)
        base = json.loads((cls.directory/"unit/unit.json").read_bytes())
        cls.source = build_binding_source(base)
        cls.request = request(cls.source)
        cls.descriptor = cls.producer(cls.source,cls.request)
        for name,value in (("source",cls.source),("descriptor",cls.descriptor)):
            (cls.directory/f"{name}.json").write_bytes(encoded(value))

    def verify(self,descriptor=None,source=None):
        return verify_binding_placement(self.source if source is None else source,
                                        self.descriptor if descriptor is None else descriptor)

    def test_fresh_source_complete_counts_identity_and_isolation(self):
        before = encoded([self.source,self.descriptor])
        result = self.verify()
        self.assertIs(result["verified"],True)
        self.assertIs(result["accepted"],False)
        self.assertEqual([result[key] for key in ("instanceCount","vertexCount","triangleCount","bindingLineageVertexCount","separatingPlaneCount")],
                         [5,245,398,29,10])
        for key,value in (("sourceSha256",self.source),("descriptorSha256",self.descriptor),
                          ("requestSha256",self.request),("placedMetersSha256",self.descriptor["placedMeters"])):
            self.assertEqual(result[key],sha(value))
        self.assertIn("separately captured source validator",result["sourceScope"])
        self.assertIn("No within-instance contact",result["sourceScope"])
        result["minimumCertifiedGapMeters"]["numerator"]="0"
        self.assertEqual(encoded([self.source,self.descriptor]),before)
        self.assertGreater(rat(self.verify()["minimumCertifiedGapMeters"]),F(.1))

    def test_execution_imports_only_standard_library(self):
        code="""import json,sys
sys.path.insert(0,sys.argv[1]);before=set(sys.modules)
from solver_binding_placement_replay import verify_binding_placement,REQUIRED_HELPER_FILES
assert REQUIRED_HELPER_FILES==()
source=json.load(open(sys.argv[2]+'/source.json'));descriptor=json.load(open(sys.argv[2]+'/descriptor.json'))
assert verify_binding_placement(source,descriptor)['verified'] is True
new=set(sys.modules)-before
assert not any(name.startswith(('numpy','scipy','shapely','newton','warp')) for name in new)
assert not any(name in new for name in ('solver_binding_placement','solver_binding_source','solver_binding_refinement',
 'solver_binding_remap','solver_binding_gripper_remap_replay','shirt','assembly'))
"""
        process=subprocess.run([sys.executable,"-c",code,str(SCRIPTS),str(self.directory)],capture_output=True,text=True,timeout=30)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)

    def test_every_coordinate_has_independent_decimal_exact_affine_rounding(self):
        self.verify()
        with localcontext() as context:
            context.prec=200
            for pose in self.request["poses"]:
                name=pose["instanceId"]
                start=self.source["instanceOffsets"][name]
                for local,point in enumerate(self.source["numericalMeshes"][name]["verticesMeters"]):
                    for axis in range(3):
                        exact=sum((Decimal.from_float(float(pose["rotation"][axis][j]))*Decimal.from_float(float(point[j]))
                                   for j in range(3)),Decimal())+Decimal.from_float(float(pose["translationMeters"][axis]))
                        self.assertEqual(bits(self.descriptor["placedMeters"][start+local][axis]),bits(float(exact)))
        self.assertTrue(any(rat(entry)!=0 for instance in self.descriptor["instances"]
                            for row in instance["rotationGramMinusIdentity"] for entry in row))
        self.assertTrue(any(rat(entry)!=0 for instance in self.descriptor["instances"]
                            for row in instance["coordinateRoundingResidualMeters"] for entry in row))

    def test_all_edge_and_triangle_witnesses_are_complete(self):
        count=0
        for instance in self.descriptor["instances"]:
            name=instance["instanceId"]
            mesh=self.source["numericalMeshes"][name]
            expected=sorted({tuple(sorted((face[i],face[(i+1)%3]))) for face in mesh["triangles"] for i in range(3)})
            self.assertEqual([tuple(edge["verticesLocal"]) for edge in instance["edges"]],expected)
            for edge in instance["edges"]:
                difference=rat(edge["placedSquaredLengthMetersSquared"])-rat(edge["restSquaredLengthMetersSquared"])
                self.assertEqual(difference,rat(edge["squaredLengthChangeMetersSquared"]))
                self.assertEqual(difference,sum((rat(edge[key]) for key in ("rotationMetricDefectMetersSquared",
                    "roundingCrossTermMetersSquared","roundingSquaredTermMetersSquared")),F()))
                self.assertLessEqual(abs(rat(edge["relativeSquaredLengthChange"])),F(4096,2**52))
            self.assertEqual([entry["triangleIndexLocal"] for entry in instance["triangles"]],list(range(len(mesh["triangles"]))))
            self.assertTrue(all(rat(entry["orientedAreaDotMetersFourth"])>0 for entry in instance["triangles"]))
            count+=len(expected)
        self.assertEqual(self.verify()["edgeCount"],count)

    def test_repaired_descriptor_arithmetic_omission_and_boolean_attacks(self):
        attacks=[
            ("placed coordinate",lambda d:d["placedMeters"][0].__setitem__(0,math.nextafter(d["placedMeters"][0][0],math.inf))),
            ("missing vertex",lambda d:d["placedMeters"].pop()),
            ("missing instance",lambda d:d["instances"].pop()),
            ("missing edge",lambda d:d["instances"][0]["edges"].pop()),
            ("duplicated edge",lambda d:d["instances"][0]["edges"].append(d["instances"][0]["edges"][0])),
            ("edge source",lambda d:d["instances"][0]["edges"][0]["verticesLocal"].reverse()),
            ("rotation metric",lambda d:d["instances"][0]["rotationGramMinusIdentity"][1][1].__setitem__("numerator","0")),
            ("coordinate rounding",lambda d:d["instances"][0]["coordinateRoundingResidualMeters"][0][2].__setitem__("numerator","1")),
            ("edge decomposition",lambda d:d["instances"][0]["edges"][0]["roundingCrossTermMetersSquared"].__setitem__("numerator","1")),
            ("triangle orientation",lambda d:d["instances"][0]["triangles"][0]["orientedAreaDotMetersFourth"].__setitem__("numerator","0")),
            ("missing triangle",lambda d:d["instances"][0]["triangles"].pop()),
            ("missing lineage",lambda d:d["bindingLineage"].pop()),
            ("source sum",lambda d:d["bindingLineage"][0]["sourceCoefficientSum"].__setitem__("numerator","2")),
            ("commutator",lambda d:d["bindingLineage"][-1]["idealAffineCommutatorMeters"][0].__setitem__("numerator","1")),
            ("missing pair",lambda d:d["separatingPlanes"].pop()),
            ("duplicated pair",lambda d:d["separatingPlanes"].__setitem__(1,copy.deepcopy(d["separatingPlanes"][0]))),
            ("gap",lambda d:d["separatingPlanes"][0]["separatingPlaneGapMeters"].__setitem__("numerator","0")),
            ("normal sign",lambda d:d["separatingPlanes"][0].__setitem__("directionSign",-1)),
            ("minimum separation",lambda d:d["request"].__setitem__("minimumSeparationMeters",10.)),
            ("pose translation",lambda d:d["request"]["poses"][0]["translationMeters"].__setitem__(0,.25)),
            ("rotation bound",lambda d:d["rotationRepresentationBound"].__setitem__("numerator","128")),
            ("edge bound",lambda d:d["relativeSquaredEdgeRepresentationBound"].__setitem__("numerator","8192")),
            ("prefix boolean",lambda d:d.__setitem__("originalBindingPrefixPreservedUnderDeclaredEvaluation",1)),
            ("accepted boolean",lambda d:d.__setitem__("accepted",0)),
            ("gap sign boolean",lambda d:d["separatingPlanes"][0].__setitem__("directionSign",True)),
        ]
        for label,mutate in attacks:
            with self.subTest(attack=label):
                descriptor=copy.deepcopy(self.descriptor)
                mutate(descriptor)
                self.assertNotEqual(encoded(descriptor),encoded(self.descriptor))
                descriptor["requestSha256"]=sha(descriptor["request"])
                with self.assertRaises(ValueError):
                    self.verify(descriptor)

    def test_raw_lineage_parent_and_canonical_mesh_attacks_with_repaired_hashes(self):
        attacks=[
            ("raw coefficient",lambda s:s["bindingRefinement"]["stages"][1]["output"]["sourceWeights"][-1].__setitem__("0",.125)),
            ("immediate parent",lambda s:s["bindingRefinement"]["stages"][1]["output"]["parentTriangles"].__setitem__(0,1)),
            ("canonical offset",lambda s:s["instanceOffsets"].__setitem__(INSTANCE,41)),
            ("canonical face",lambda s:s["triangles"].__setitem__(0,s["triangles"][1])),
            ("stored rest",lambda s:s["restMeters"][0].__setitem__(0,.125)),
            ("profile",lambda s:s.__setitem__("profile","synthetic")),
            ("base controls",lambda s:s["baseUnit"].__setitem__("placedMeters",[])),
            ("source controls",lambda s:s.__setitem__("gripperActuation",{})),
            ("source flag",lambda s:s.__setitem__("accepted",0)),
        ]
        for label,mutate in attacks:
            with self.subTest(attack=label):
                source,descriptor=copy.deepcopy(self.source),copy.deepcopy(self.descriptor)
                mutate(source)
                self.assertNotEqual(encoded(source),encoded(self.source))
                source["bindingRefinement"]["sourceUnitSha256"]=sha(source["baseUnit"])
                descriptor["sourceSha256"],descriptor["baseUnitSha256"]=sha(source),sha(source["baseUnit"])
                with self.assertRaises(ValueError):
                    self.verify(descriptor,source)

    def test_correct_signed_zero_is_retained_without_changing_rest(self):
        req=request(self.source,zero_rotation=True)
        # A subnormal skew perturbation is inside the declared representation
        # bound; its exact tiny z coordinates legitimately round to signed zero.
        for pose in req["poses"]:
            pose["translationMeters"]=[float(req["poses"].index(pose)),0.,0.]
            pose["rotation"]=[[1.,0.,math.ulp(0.)],[0.,1.,0.],[-math.ulp(0.),0.,1.]]
        descriptor=self.producer(self.source,req)
        self.verify(descriptor)
        negative_zero=next((i,j) for i,point in enumerate(descriptor["placedMeters"]) for j,v in enumerate(point) if bits(v)==bits(-0.))
        i,j=negative_zero
        descriptor["placedMeters"][i][j]=0.
        with self.assertRaises(ValueError):
            self.verify(descriptor)

    def test_strict_pose_fields_order_types_and_resource_bounds(self):
        mutations=[
            lambda d:d["request"]["poses"].reverse(),
            lambda d:d["request"]["poses"].pop(),
            lambda d:d["request"]["poses"][0]["rotation"][0].__setitem__(0,True),
            lambda d:d["request"]["poses"][0]["translationMeters"].__setitem__(0,11.),
            lambda d:d["request"].__setitem__("minimumSeparationMeters",True),
            lambda d:d["request"].__setitem__("minimumSeparationMeters",0.),
            lambda d:d["request"].__setitem__("coordinatePolicy","normalize"),
            lambda d:d["request"]["poses"][0].__setitem__("normalize",True),
        ]
        for mutate in mutations:
            descriptor=copy.deepcopy(self.descriptor)
            mutate(descriptor)
            descriptor["requestSha256"]=sha(descriptor["request"])
            with self.assertRaises(ValueError):
                self.verify(descriptor)
        for value in (None,[],{"bad":float("nan")},{"bad":(0.,0.)},{"bad":2**64},{"bad":"x"*(24*1024**2+1)}):
            with self.assertRaises(ValueError):
                verify_binding_placement(self.source,value)


if __name__=="__main__":
    unittest.main()
