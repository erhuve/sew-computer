"""Independent exact material clipping, tiling and full-descriptor attacks."""

import ast
import copy
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import solver_binding_spatial_seams_replay as audit


SCRIPTS = Path(__file__).resolve().parent


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rat(value):
    assert set(value) == {"numerator", "denominator"}
    result = F(int(value["numerator"]), int(value["denominator"]))
    assert value == {"numerator": str(result.numerator), "denominator": str(result.denominator)}
    return result


def weights(raw):
    return {v["vertex"]: rat(v["weight"]) for v in raw}


def cross(a, b, c):
    return a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1])


def bary(point, points, face):
    a,b,c = [points[v] for v in face]
    area = cross(a,b,c)
    return [cross(point,b,c)/area, cross(a,point,c)/area, cross(a,b,point)/area]


class SpatialMappingArithmeticTests(unittest.TestCase):
    def test_positive_interval_survives_identical_rounded_endpoints(self):
        tiny = F(1, 2**100)
        start, end = [-F(1,2), F(1,2)+tiny, 1-tiny], [F(1,2), -F(1,2)+tiny, 1-tiny]
        interval = audit._clip(start, end)
        self.assertEqual(interval, (F(1,2), F(1,2)+tiny))
        self.assertEqual(float(interval[0]), float(interval[1]))
        self.assertIsNone(audit._clip([-F(1,2**1074), F(1), F(1,2**1074)], [-F(1,2**1074), F(1), F(1,2**1074)]))

    def test_parent_tiling_checks_boundary_order_missing_and_duplicate_children(self):
        original = [[F(),F()], [F(2),F()], [F(),F(2)]]
        points = original+[[F(1),F()]]
        basis = [{0:F(1)}, {1:F(1)}, {2:F(1)}, {0:F(1,2),1:F(1,2)}]
        faces, parents = [[0,3,2],[3,1,2]], [0,0]
        proof = audit._tiling(original, [[0,1,2]], points, faces, basis, parents)
        self.assertEqual(rat(proof[0]["childSumSignedDoubleAreaMmSquared"]), 4)
        self.assertEqual([len(s["pieces"]) for s in proof[0]["boundarySides"]], [2,1,1])
        for changed_faces, changed_parents in ((faces[:1],parents[:1]), (faces+faces,parents+parents), ([[2,3,0],faces[1]],parents)):
            with self.assertRaises(ValueError): audit._tiling(original, [[0,1,2]], points, changed_faces, basis, changed_parents)
        changed = copy.deepcopy(basis)
        changed[3][0] += F(1,2**100)
        with self.assertRaises(ValueError): audit._tiling(original, [[0,1,2]], points, faces, changed, parents)

    @staticmethod
    def member_fixture(shared=False):
        points = [[F(),F()],[F(1),F()],[F(1),F(1)],[F(),F(1)]]
        ends = [[0.,0.],[1.,1.]] if shared else [[.5,0.],[.5,1.]]
        samples = [{"sourceSegment":0,"sourceFraction":float(i),"arcMm":float(i),"restPosition":p} for i,p in enumerate(ends)]
        path = {"name":"line","lengthMm":1.,"samples":samples,"segments":[{"samples":[0,1],"triangle":0}]}
        source = {"baseUnit":{"sourceTemplates":{"t":{"stitchPaths":[path]}},"sourcePattern":{"panels":[{"id":"t","points":ends}]}}}
        mesh = {"templateId":"t","originalPoints":points,"originalFaces":[[0,1,2],[0,2,3]],"points":points,
            "faces":[[0,1,2],[0,2,3]],"lineage":[{i:F(1)}for i in range(4)],"parents":[0,1],"originalTriangleOffset":0,"triangleOffset":7}
        member = {"instanceId":"i","pathName":"line","startArcMm":0.,"endArcMm":1.,"direction":"forward"}
        return source,member,mesh

    def test_exact_all_face_clipping_ignores_tolerant_owner_and_respects_reverse_subinterval(self):
        source,member,mesh = self.member_fixture()
        # Local mathematical fixture, not an independently source-validated unit.
        mapped = audit._member_map(source,member,mesh)
        self.assertEqual(len(mapped["cells"]),2)
        self.assertEqual([c["originalCandidates"][0]["localTriangleIndex"]for c in mapped["cells"]],[0,1])
        member.update(direction="reverse",startArcMm=.25,endArcMm=.75)
        reverse = audit._member_map(source,member,mesh)
        self.assertEqual([c["originalCandidates"][0]["localTriangleIndex"]for c in reverse["cells"]],[1,0])
        self.assertEqual(audit._sample(reverse,F(),"numerical"),{0:F(1,4),2:F(1,2),3:F(1,4)})
        self.assertEqual(audit._sample(reverse,F(1),"numerical"),{0:F(1,2),1:F(1,4),2:F(1,4)})

    def test_shared_edge_operator_equivalence_keeps_both_director_candidates(self):
        source,member,mesh = self.member_fixture(shared=True)
        result = audit._member_map(source,member,mesh)
        self.assertEqual(len(result["cells"]),1)
        cell = result["cells"][0]
        self.assertEqual(len(cell["originalCandidates"]),2)
        self.assertEqual(len(cell["numericalCandidates"]),2)
        self.assertEqual(audit._sample(result,F(1,3),"numerical"),{0:F(2,3),2:F(1,3)})
        self.assertNotIn("selectedCandidateId",cell)

    def test_bounded_raw_inputs_are_checked_before_source_prerequisites(self):
        deep=0
        for _ in range(42):deep=[deep]
        for bad in (deep,{1:0},{"x":math.inf},{"x":(0,1)},{"x":2**64},{"x":"\ud800"}):
            for slot in range(3):
                inputs=[{}, {}, {}];inputs[slot]=bad
                with mock.patch.object(audit,"verify_binding_remap") as prerequisite:
                    with self.assertRaises(ValueError):audit.verify_binding_spatial_seams(*inputs)
                    prerequisite.assert_not_called()


class BindingSpatialSeamsReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_solver_binding_sewing_schedule import BindingSewingScheduleTests
        from solver_binding_spatial_seams import build_binding_spatial_seams
        BindingSewingScheduleTests.setUpClass.__func__(cls)
        cls.descriptor=build_binding_spatial_seams(cls.source,cls.reference_source)

    def verify(self,descriptor=None,*,source=None,reference=None):
        return audit.verify_binding_spatial_seams(self.source if source is None else source,
            self.reference_source if reference is None else reference,self.descriptor if descriptor is None else descriptor)

    def test_full_eight_selectors_forty_rows_and_input_immutability(self):
        before=[encoded(v)for v in(self.source,self.reference_source,self.descriptor)]
        evidence=self.verify()
        self.assertTrue(evidence["verified"]);self.assertFalse(evidence["accepted"])
        self.assertEqual(evidence["selectorCount"],8);self.assertEqual(evidence["rowCount"],40)
        rows=[i for selector in self.descriptor["selectors"]for i in selector["rowIndices"]]
        self.assertEqual(sorted(rows),list(range(40)))
        self.assertEqual(sum(s["held"]for s in self.descriptor["selectors"]),1)
        self.assertEqual(before,[encoded(v)for v in(self.source,self.reference_source,self.descriptor)])

    def test_all_cell_operators_match_independent_area_and_raw_lineage_oracle(self):
        base=self.source["baseUnit"]
        first,second=[stage["output"]for stage in self.source["bindingRefinement"]["stages"]]
        dense=[[sum((F(w)*F(first["sourceWeights"][int(mid)].get(str(i),0))for mid,w in support.items()),F())for i in range(11)]for support in second["sourceWeights"]]
        total=0
        for selector in self.descriptor["selectors"]:
            for member in selector["members"]:
                recipe=member["registrationMember"];name=recipe["instanceId"];template=base["sourceTemplates"][member["templateId"]]
                original=[[F(x)for x in p]for p in template["restPositions"]]
                basis=dense if name==audit.INSTANCE else [[F(int(i==j))for j in range(len(original))]for i in range(len(original))]
                material=[[sum((w*p[axis]for w,p in zip(row,original)),F())for axis in(0,1)]for row in basis]
                path=next(p for p in template["stitchPaths"]if p["name"]==recipe["pathName"])
                for cell in member["cells"]:
                    lo,hi=map(rat,cell["fractionInterval"]);self.assertLess(lo,hi)
                    source_segment=path["segments"][cell["sourcePathSegmentIndex"]];a,z=[path["samples"][v]for v in source_segment["samples"]]
                    for endpoint,suffix in((lo,"Start"),(hi,"End")):
                        u=endpoint if recipe["direction"]=="forward"else 1-endpoint
                        arc=F(recipe["startArcMm"])+(F(recipe["endArcMm"])-F(recipe["startArcMm"]))*u
                        t=(arc-F(a["arcMm"]))/ (F(z["arcMm"])-F(a["arcMm"]))
                        point=[F(x)+t*(F(y)-F(x))for x,y in zip(a["restPosition"],z["restPosition"])]
                        for domain,points in(("original",original),("numerical",material)):
                            expected=weights(cell[domain+suffix+"Weights"])
                            for candidate in cell[domain+"Candidates"]:
                                values=bary(point,points,candidate["localVertices"])
                                actual={v:w for v,w in zip(candidate["localVertices"],values)if w}
                                self.assertEqual(actual,expected)
                                self.assertTrue(all(w>=0 for w in actual.values()))
                                self.assertEqual(sum(actual.values(),F()),1)
                        numerical=weights(cell["numerical"+suffix+"Weights"])
                        pullback={i:sum((w*basis[v][i]for v,w in numerical.items()),F())for i in range(len(original))}
                        self.assertEqual({i:w for i,w in pullback.items()if w},weights(cell["original"+suffix+"Weights"]))
                    total+=1
        self.assertGreater(total,100)

    def test_all_signed_raw_row_residuals_and_translation_sums_are_preserved(self):
        for witness in self.descriptor["rowComparisons"]:
            index=witness["rowIndex"];selector=next(s for s in self.descriptor["selectors"]if index in s["rowIndices"])
            for domain,rows in(("original",self.source["baseUnit"]["embeddedConstraints"]["constraints"]),("numerical",self.source["embeddedConstraints"]["constraints"])):
                row=rows[index];continuum={}
                # Independently locate the enclosing cell and interpolate its endpoints.
                for sign,member in zip((1,-1),selector["members"]):
                    f=F(row["fraction"]);cell=next(c for c in member["cells"]if rat(c["fractionInterval"][0])<=f<=rat(c["fractionInterval"][1]))
                    lo,hi=map(rat,cell["fractionInterval"]);t=(f-lo)/(hi-lo);a,z=weights(cell[domain+"StartWeights"]),weights(cell[domain+"EndWeights"])
                    name=member["registrationMember"]["instanceId"]
                    for v in a.keys()|z.keys():continuum[name,v]=sign*((1-t)*a.get(v,F())+t*z.get(v,F()))
                stored={(term["instanceId"],term["vertex"]):F(term["coefficient"])for term in row["terms"]}
                expected={key:continuum.get(key,F())-stored.get(key,F())for key in continuum.keys()|stored.keys()}
                expected={key:v for key,v in expected.items()if v}
                actual={(term["instanceId"],term["vertex"]):rat(term["coefficient"])for term in witness[domain+"ResidualTerms"]}
                self.assertEqual(actual,expected)
                for anchor in witness["perAnchor"]:
                    name=anchor["instanceId"]
                    self.assertEqual(rat(anchor[domain+"StoredCoefficientSum"]),sum((v for (n,_),v in stored.items()if n==name),F()))
                    self.assertEqual(rat(anchor[domain+"ResidualL1"]),sum((abs(v)for(n,_),v in expected.items()if n==name),F()))
        self.assertTrue(any(r["numericalResidualTerms"]for r in self.descriptor["rowComparisons"]))

    def test_repaired_descriptor_claims_cannot_omit_cells_candidates_or_pending_rows(self):
        mutations=[lambda d:d.update(accepted=0),lambda d:d.update(materialCurveCoverageVerified=1),
            lambda d:d.update(controlsInstalled=True),lambda d:d.update(continuousSpatialStitchingVerified=True),
            lambda d:d["selectors"].pop(),lambda d:d["rowComparisons"].pop(),
            lambda d:d["selectors"][0]["members"][0]["cells"].pop(),
            lambda d:d["selectors"][0]["members"][0]["cells"][0]["originalCandidates"].clear(),
            lambda d:d["selectors"][0]["members"][0]["cells"][0]["numericalCandidates"][0].update(canonicalTriangleIndex=0),
            lambda d:d["selectors"][0]["members"][0]["cells"][0]["numericalCandidates"][0]["localVertices"].reverse(),
            lambda d:d["selectors"][0]["members"][0]["cells"][0].update(pullbackVerified=1),
            lambda d:d["rowComparisons"][0].update(numericalResidualTerms=[]),
            lambda d:d["selectors"][0]["rowIndices"].__setitem__(0,False),
            lambda d:d["selectors"][0]["targetKnots"][0].update(targetMeters=.002),
            lambda d:d["materialMeshes"][2]["originalParentTiling"][0]["boundarySides"][0]["pieces"].pop(),
            lambda d:d["materialMeshes"][2].update(materialCoordinatesSha256="0"*64)]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                altered=copy.deepcopy(self.descriptor);mutation(altered)
                with self.assertRaises(ValueError):self.verify(altered)

    def test_missing_shared_candidate_and_merged_tiny_rational_cell_reject(self):
        shared=next((si,mi,ci)for si,s in enumerate(self.descriptor["selectors"])for mi,m in enumerate(s["members"])for ci,c in enumerate(m["cells"])if len(c["numericalCandidates"])>1)
        altered=copy.deepcopy(self.descriptor);si,mi,ci=shared;altered["selectors"][si]["members"][mi]["cells"][ci]["numericalCandidates"].pop()
        with self.assertRaises(ValueError):self.verify(altered)
        candidates=[(rat(c["fractionInterval"][1])-rat(c["fractionInterval"][0]),si,mi,ci)for si,s in enumerate(self.descriptor["selectors"])for mi,m in enumerate(s["members"])for ci,c in enumerate(m["cells"])]
        width,si,mi,ci=min(candidates);self.assertLess(width,F(1,2**40))
        altered=copy.deepcopy(self.descriptor);cells=altered["selectors"][si]["members"][mi]["cells"]
        if ci:cells[ci-1]["fractionInterval"][1]=cells[ci]["fractionInterval"][1]
        else:cells[1]["fractionInterval"][0]=cells[0]["fractionInterval"][0]
        cells.pop(ci)
        with self.assertRaises(ValueError):self.verify(altered)

    def test_signed_zero_reference_and_stale_source_hashes_are_not_interchangeable(self):
        from solver_binding_spatial_seams import build_binding_spatial_seams
        reference=copy.deepcopy(self.reference_source)
        for knot in reference["sewingActuation"]["schedule"]["knots"]:knot["activation"][5]=-0.
        descriptor=build_binding_spatial_seams(self.source,reference)
        self.verify(descriptor,reference=reference)
        with self.assertRaises(ValueError):self.verify(descriptor)
        descriptor["sewingControlSchedule"]["knots"][0]["activation"][5]=0.
        with self.assertRaises(ValueError):self.verify(descriptor,reference=reference)
        for key in("placedMeters","sewingActuation","sewingFrames"):
            source=copy.deepcopy(self.source);source[key]=None
            with self.assertRaises(ValueError):self.verify(source=source)

    def test_discrepancy_core_never_ignores_extra_owners_or_boolean_vertex_aliases(self):
        # Isolate supplied-row completeness from the separate source validator.
        row_ids=[r["rowId"]for r in self.descriptor["rowComparisons"]]
        selector=next(s for s in self.descriptor["selectors"]if 5 in s["rowIndices"])
        owners={m["registrationMember"]["instanceId"]for m in selector["members"]}
        outsider=next(i["id"]for i in self.source["instances"]if i["id"]not in owners)
        for change in (lambda r:r["terms"][0].update(instanceId=outsider),
                       lambda r:r["terms"][0].update(vertex=False),
                       lambda r:r["terms"][0].update(coefficient=0.),
                       lambda r:r["terms"][0].update(coefficient=-r["terms"][0]["coefficient"]),
                       lambda r:r["terms"].append(copy.deepcopy(r["terms"][0]))):
            altered=copy.deepcopy(self.source)
            change(altered["embeddedConstraints"]["constraints"][5])
            with self.assertRaises(ValueError):audit._comparisons(altered,self.descriptor["selectors"],row_ids)

    def test_stdlib_only_closure_and_isolated_byte_reproduction(self):
        files=["solver_binding_spatial_seams_replay.py",*audit.REQUIRED_HELPER_FILES]
        allowed={name[:-3]for name in files}|set(sys.stdlib_module_names)
        for name in files:
            for node in ast.walk(ast.parse((SCRIPTS/name).read_text())):
                imports=[v.name for v in node.names]if isinstance(node,ast.Import)else[node.module]if isinstance(node,ast.ImportFrom)else[]
                for imported in imports:self.assertIn(imported.split('.')[0],allowed)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            for name in files:(root/name).write_bytes((SCRIPTS/name).read_bytes())
            (root/'inputs.json').write_bytes(encoded([self.source,self.reference_source,self.descriptor]))
            code="import sys,json;sys.path.insert(0,sys.argv[1]);from pathlib import Path;from solver_binding_spatial_seams_replay import verify_binding_spatial_seams;x=json.loads((Path(sys.argv[1])/'inputs.json').read_text());print(json.dumps(verify_binding_spatial_seams(*x),sort_keys=True,separators=(',',':')))"
            completed=subprocess.run([sys.executable,'-I','-S','-c',code,directory],check=True,text=True,capture_output=True)
            self.assertEqual(completed.stdout.strip().encode(),encoded(self.verify()))


if __name__=="__main__":unittest.main()
