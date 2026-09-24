"""Independent source-curve tests; constructed states are not cloth trajectories."""

import copy
from fractions import Fraction as F
import hashlib
import json
import math
import unittest

from solver_binding_spatial_seams import (
    build_binding_spatial_seams, evaluate_binding_spatial_seams,
    validate_binding_spatial_seams,
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rational(value):
    if set(value) != {"numerator", "denominator"}:
        raise AssertionError("A rational witness must have no rounded replacement")
    result = F(int(value["numerator"]), int(value["denominator"]))
    if (str(result.numerator), str(result.denominator)) != (value["numerator"], value["denominator"]):
        raise AssertionError("Rational witnesses must be canonical and reduced")
    return result


def sparse(values):
    vertices = [x["vertex"] for x in values]
    if vertices != sorted(set(vertices)):
        raise AssertionError("Sparse vertex order must be unique and sorted")
    result = {x["vertex"]: rational(x["weight"]) for x in values}
    if any(x <= 0 for x in result.values()) or sum(result.values(), F()) != 1:
        raise AssertionError("A point operator must preserve exact positive support and unit sum")
    return result


def add_scaled(result, operator, scale):
    for key, value in operator.items():
        result[key] = result.get(key, F()) + scale * value
    return {key: value for key, value in result.items() if value}


def independent_material(source):
    """Recompose raw stages, rather than trust composed weights or rest metres."""
    refinement = source["bindingRefinement"]
    lineage = [{i: F(1)} for i in range(len(refinement["originalLocalMesh"]["verticesMeters"]))]
    for stage in refinement["stages"]:
        composed = []
        for raw in stage["output"]["sourceWeights"]:
            operator = {}
            for previous, value in raw.items():
                operator = add_scaled(operator, lineage[int(previous)], F(value))
            composed.append(operator)
        lineage = composed
    result = {}
    for instance in source["instances"]:
        template = source["baseUnit"]["sourceTemplates"][instance["templateId"]]
        original = [[F(x) for x in point] for point in template["restPositions"]]
        basis = lineage if instance["id"] == refinement["instanceId"] else [{i: F(1)} for i in range(len(original))]
        assert all(sum(row.values(), F()) == 1 and all(x > 0 for x in row.values()) for row in basis)
        material = [[sum((weight * original[v][axis] for v, weight in row.items()), F())
                     for axis in range(2)] for row in basis]
        result[instance["id"]] = (template, original, material, basis)
    return result


def barycentric(point, vertices):
    a, b, c = vertices
    determinant = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    assert determinant
    second = ((point[0]-a[0])*(c[1]-a[1])-(point[1]-a[1])*(c[0]-a[0])) / determinant
    third = ((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])) / determinant
    return [1-second-third, second, third]


def path_point(path, member, fraction):
    lo, hi = F(member["startArcMm"]), F(member["endArcMm"])
    arc = lo + (hi-lo) * (fraction if member["direction"] == "forward" else 1-fraction)
    index, (a, b) = next((i, pair) for i, pair in enumerate(zip(path["samples"], path["samples"][1:]))
                         if F(pair[0]["arcMm"]) <= arc <= F(pair[1]["arcMm"]))
    t = (arc-F(a["arcMm"]))/(F(b["arcMm"])-F(a["arcMm"]))
    return index, [(1-t)*F(x)+t*F(y) for x, y in zip(a["restPosition"], b["restPosition"])]


def clipping_breaks(path, member, points, faces):
    """Exhaust all face/segment inequalities, independent of stored segment owners."""
    lo, hi = F(member["startArcMm"]), F(member["endArcMm"])
    def fraction(arc):
        t = (arc-lo)/(hi-lo)
        return t if member["direction"] == "forward" else 1-t
    bounds = {F(), F(1)}
    for a, b in zip(path["samples"], path["samples"][1:]):
        arc0, arc1 = F(a["arcMm"]), F(b["arcMm"])
        lower, upper = max(arc0, lo), min(arc1, hi)
        if lower >= upper:
            continue
        bounds.update((fraction(lower), fraction(upper)))
        p0, p1 = [[F(x) for x in sample["restPosition"]] for sample in (a, b)]
        for face in faces:
            w0, w1 = [barycentric(p, [points[v] for v in face]) for p in (p0, p1)]
            start, end = (lower-arc0)/(arc1-arc0), (upper-arc0)/(arc1-arc0)
            for value, final in zip(w0, w1):
                change = final-value
                if change > 0:
                    start = max(start, -value/change)
                elif change < 0:
                    end = min(end, -value/change)
                elif value < 0:
                    start, end = F(1), F()
            if start < end:
                bounds.update((fraction(arc0+(arc1-arc0)*start), fraction(arc0+(arc1-arc0)*end)))
    return sorted(bounds)


def member_weights(member_map, fraction, numerical=True):
    cell = next(cell for cell in member_map["cells"]
                if rational(cell["fractionInterval"][0]) <= fraction <= rational(cell["fractionInterval"][1]))
    a, b = map(rational, cell["fractionInterval"])
    t = (fraction-a)/(b-a)
    prefix = "numerical" if numerical else "original"
    return add_scaled(add_scaled({}, sparse(cell[prefix+"StartWeights"]), 1-t),
                      sparse(cell[prefix+"EndWeights"]), t)


class BindingSpatialSeamsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_solver_binding_sewing_schedule import BindingSewingScheduleTests
        BindingSewingScheduleTests.setUpClass.__func__(cls)
        cls.sewing_descriptor = cls.descriptor
        cls.descriptor = build_binding_spatial_seams(cls.source, cls.reference_source)
        cls.material = independent_material(cls.source)

    def build(self, source=None, reference=None):
        return build_binding_spatial_seams(self.source if source is None else source,
            self.reference_source if reference is None else reference)

    def validate(self, descriptor):
        return validate_binding_spatial_seams(self.source, self.reference_source, descriptor)

    def test_all_eight_selectors_forty_rows_targets_and_static_scope(self):
        d, base = self.descriptor, self.source["baseUnit"]
        self.assertEqual(d["profile"], "source-binding-spatial-seam-map-v1")
        self.assertEqual(len(d["selectors"]), 8)
        self.assertEqual(sorted(i for s in d["selectors"] for i in s["rowIndices"]), list(range(40)))
        self.assertEqual(d["heldRowIndices"], list(range(5)))
        self.assertEqual(d["pendingRowIndices"], list(range(5, 40)))
        ids = self.reference_source["sewingActuation"]["schedule"]["rowIds"]
        for selector in d["selectors"]:
            self.assertEqual(len(selector["rowIndices"]), 5)
            self.assertEqual(selector["rowIds"], [ids[i] for i in selector["rowIndices"]])
            self.assertEqual([rational(k["fraction"]) for k in selector["targetKnots"]], [F(i,4) for i in range(5)])
            self.assertIs(selector["held"], selector["rowIndices"] == list(range(5)))
            self.assertTrue(all(i in base["phaseConstraintRows"][selector["phaseId"]] for i in selector["rowIndices"]))
            self.assertEqual(encoded([k["targetMeters"] for k in selector["targetKnots"]]),
                encoded([self.reference_source["sewingActuation"]["initialTargetsMeters"][i] for i in selector["rowIndices"]]))
        for key in ("initialTargetsMeters", "finalTargetsMeters"):
            self.assertEqual(encoded(d[key]), encoded(self.reference_source["sewingActuation"][key]))
        self.assertEqual(encoded(d["sewingControlSchedule"]), encoded(self.reference_source["sewingActuation"]["schedule"]))
        self.assertIs(d["materialCurveCoverageVerified"], True)
        for key in ("accepted", "solverReady", "executable", "controlsInstalled", "continuousSpatialStitchingVerified",
                    "constructionPhaseCompleted", "historicalExtrasInherited"):
            self.assertIs(d[key], False)
        self.assertEqual(d["complianceMPerN"], 1e-8)
        for field, value in (("sourceSha256",self.source), ("baseUnitSha256",base),
                ("referenceSourceSha256",self.reference_source), ("numericalBundleSha256",self.source["embeddedConstraints"]),
                ("originalBundleSha256",base["embeddedConstraints"])):
            self.assertEqual(d[field],digest(value))
        self.assertEqual(encoded(self.validate(d)), encoded(d))

    def test_full_triangle_traversal_exact_coverage_pullbacks_and_canonical_offsets(self):
        smallest, different_owner = F(1), False
        meshes = {m["instanceId"]:m for m in self.descriptor["materialMeshes"]}
        for selector in self.descriptor["selectors"]:
            for member_map in selector["members"]:
                member = member_map["registrationMember"]
                instance = member["instanceId"]
                template, original, numerical, basis = self.material[instance]
                path = next(p for p in template["stitchPaths"] if p["name"] == member["pathName"])
                original_faces, numerical_faces = template["triangles"], self.source["numericalMeshes"][instance]["triangles"]
                expected = sorted(set(clipping_breaks(path, member, original, original_faces)) |
                                  set(clipping_breaks(path, member, numerical, numerical_faces)))
                cells = member_map["cells"]
                self.assertEqual([rational(c["fractionInterval"][0]) for c in cells] + [rational(cells[-1]["fractionInterval"][1])], expected)
                for index, cell in enumerate(cells):
                    lo, hi = map(rational, cell["fractionInterval"])
                    self.assertLess(lo, hi); smallest = min(smallest, hi-lo)
                    _, point = path_point(path, member, (lo+hi)/2)
                    for prefix, points, faces in (("original",original,original_faces), ("numerical",numerical,numerical_faces)):
                        expected_faces = [i for i,f in enumerate(faces) if min(barycentric(point,[points[v] for v in f])) >= 0]
                        candidates = cell[prefix+"Candidates"]
                        self.assertEqual([c["localTriangleIndex"] for c in candidates], expected_faces)
                        for candidate in candidates:
                            face_index = candidate["localTriangleIndex"]
                            self.assertEqual(candidate["localVertices"], faces[face_index])
                            offset = meshes[instance]["originalTriangleOffset" if prefix == "original" else "triangleOffset"]
                            self.assertEqual(candidate["canonicalTriangleIndex"], offset+face_index)
                            for endpoint, fraction in (("Start",lo), ("End",hi)):
                                _, endpoint_point = path_point(path,member,fraction)
                                coefficients = barycentric(endpoint_point,[points[v] for v in faces[face_index]])
                                oracle = {v:w for v,w in zip(faces[face_index],coefficients) if w}
                                self.assertEqual(sparse(cell[prefix+endpoint+"Weights"]),oracle)
                        if index:
                            self.assertEqual(cells[index-1][prefix+"EndWeights"],cell[prefix+"StartWeights"])
                    for endpoint in ("Start", "End"):
                        pullback = {}
                        for vertex, weight in sparse(cell["numerical"+endpoint+"Weights"]).items():
                            pullback = add_scaled(pullback,basis[vertex],weight)
                        self.assertEqual(pullback,sparse(cell["original"+endpoint+"Weights"]))
                    self.assertIs(cell["pullbackVerified"],True)
                    segment = cell["sourcePathSegmentIndex"]
                    if "segments" in path and segment < len(path["segments"]):
                        different_owner |= [c["localTriangleIndex"] for c in cell["originalCandidates"]] != [path["segments"][segment]["triangle"]]
        self.assertLess(smallest,F(1,10**12), "Tiny genuine cells must not be epsilon-merged")
        self.assertTrue(different_owner, "The oracle must exercise traversal beyond tolerant stored owners")

    def test_material_coordinate_roundoff_and_parent_tiling_remain_explicit(self):
        defects = []
        for mesh in self.descriptor["materialMeshes"]:
            instance = mesh["instanceId"]
            template, original, material, basis = self.material[instance]
            self.assertEqual(mesh["vertexOffset"],self.source["instanceOffsets"][instance])
            self.assertEqual(mesh["originalVertexOffset"],self.source["baseUnit"]["instanceOffsets"][instance])
            for index, point in enumerate(material):
                stored = self.source["restMeters"][mesh["vertexOffset"]+index]
                expected = [F(stored[axis])-(point[axis]/1000 if axis < 2 else F()) for axis in range(3)]
                actual = list(map(rational,mesh["storedRestMinusMaterialMeters"][index]))
                self.assertEqual(actual,expected); defects.extend(actual)
            tilings = mesh["originalParentTiling"]
            if instance != self.source["bindingRefinement"]["instanceId"]:
                self.assertEqual(tilings,[]); continue
            self.assertEqual(len(tilings),len(template["triangles"]))
            self.assertEqual(sorted(i for t in tilings for i in t["childTriangleIndices"]),list(range(mesh["triangleCount"])))
            for tiling in tilings:
                self.assertGreater(rational(tiling["originalSignedDoubleAreaMmSquared"]),0)
                self.assertEqual(tiling["originalSignedDoubleAreaMmSquared"],tiling["childSumSignedDoubleAreaMmSquared"])
                for side in tiling["boundarySides"]:
                    intervals = [list(map(rational,p["interval"])) for p in side["pieces"]]
                    self.assertEqual(intervals[0][0],0);self.assertEqual(intervals[-1][1],1)
                    self.assertTrue(all(a<b for a,b in intervals))
                    self.assertTrue(all(a[1]==b[0] for a,b in zip(intervals,intervals[1:])))
        self.assertTrue(any(defects), "Stored rest metres must not replace exact material coordinates")

    def test_all_raw_row_discrepancies_reconstructed_without_normalizing_support(self):
        selectors = {i:s for s in self.descriptor["selectors"] for i in s["rowIndices"]}
        counts = [0,0]
        for index, comparison in enumerate(self.descriptor["rowComparisons"]):
            self.assertEqual(comparison["rowIndex"],index)
            fraction = F(self.source["embeddedConstraints"]["constraints"][index]["fraction"])
            for numerical, key, bundle in ((False,"original",self.source["baseUnit"]["embeddedConstraints"]),
                                           (True,"numerical",self.source["embeddedConstraints"])):
                row = bundle["constraints"][index]
                self.assertEqual(comparison[key+"RowSha256"],digest(row))
                residual = {}
                for sign, member in zip((1,-1),selectors[index]["members"]):
                    instance = member["registrationMember"]["instanceId"]
                    residual = add_scaled(residual,{(instance,v):w for v,w in member_weights(member,fraction,numerical).items()},F(sign))
                for term in row["terms"]:
                    residual = add_scaled(residual,{(term["instanceId"],term["vertex"]):F(term["coefficient"])},F(-1))
                actual = {(t["instanceId"],t["vertex"]):rational(t["coefficient"]) for t in comparison[key+"ResidualTerms"]}
                self.assertEqual(actual,residual)
                self.assertEqual(list(actual),sorted(actual))
                for sign,anchor in zip((1,-1),comparison["perAnchor"]):
                    instance = anchor["instanceId"]
                    self.assertEqual(anchor["sign"],sign)
                    coefficient_sum = sum((F(t["coefficient"]) for t in row["terms"] if t["instanceId"] == instance),F())
                    self.assertEqual(rational(anchor[key+"StoredCoefficientSum"]),coefficient_sum)
                    self.assertEqual(rational(anchor[key+"ResidualL1"]),sum((abs(v) for (name,_),v in residual.items() if name == instance),F()))
                counts[numerical] += bool(residual)
        self.assertEqual(len(self.descriptor["rowComparisons"]),40)
        self.assertTrue(all(counts), "The test must not silently assume continuum operators equal stored samples")

    def test_forged_coverage_coordinates_offsets_and_scope_reject(self):
        self.assertEqual(encoded(self.validate(self.descriptor)),encoded(self.descriptor))
        attacks = [
            lambda d:d["selectors"][0]["members"][0]["cells"].pop(1),
            lambda d:d["selectors"][0]["members"][0]["cells"][0]["originalCandidates"][0].update(canonicalTriangleIndex=0),
            lambda d:d["selectors"][0]["members"][0]["cells"][0].update(pullbackVerified=1),
            lambda d:d["materialMeshes"][2]["storedRestMinusMaterialMeters"][11][0].update(numerator="0"),
            lambda d:d["materialMeshes"][2]["originalParentTiling"][0]["boundarySides"][0]["pieces"].pop(),
            lambda d:d["rowComparisons"][0]["numericalResidualTerms"].clear(),
            lambda d:d.update(materialCurveCoverageVerified=1),
            lambda d:d.update(continuousSpatialStitchingVerified=True),
            lambda d:d.update(accepted=0),
            lambda d:d["limitations"].pop(),
        ]
        for index, attack in enumerate(attacks):
            changed = copy.deepcopy(self.descriptor); attack(changed)
            self.assertNotEqual(encoded(changed),encoded(self.descriptor),"The attack must change actual bytes")
            with self.subTest(index=index),self.assertRaises(ValueError):self.validate(changed)

    def test_source_lineage_mesh_rows_and_reference_forgery_fail_closed(self):
        self.assertEqual(encoded(self.build()),encoded(self.descriptor))
        for kind in ("lineage", "face", "coefficient", "offset", "source-control"):
            source = copy.deepcopy(self.source)
            if kind == "lineage":
                row = source["bindingRefinement"]["stages"][0]["output"]["sourceWeights"][-1]
                key = next(iter(row)); row[key] = math.nextafter(row[key],math.inf)
            if kind == "face":source["triangles"][:3] = source["triangles"][:3][::-1]
            if kind == "coefficient":source["embeddedConstraints"]["constraints"][0]["terms"][0]["coefficient"] = 0.
            if kind == "offset":source["instanceOffsets"]["sleeve_left:shell"] -= 18
            if kind == "source-control":source["sewingActuation"] = copy.deepcopy(self.reference_source["sewingActuation"])
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.build(source=source)
        for kind in ("target", "normal", "pending", "rowhash"):
            reference = copy.deepcopy(self.reference_source)
            recipe = reference["sewingActuation"]
            if kind == "target":recipe["initialTargetsMeters"][0] = True;recipe["finalTargetsMeters"][0] = True
            if kind == "normal":recipe["mode"] = "normal-offset"
            if kind == "pending":recipe["schedule"]["knots"][0]["activation"][5] = 1.
            if kind == "rowhash":recipe["schedule"]["rowIds"][0] = recipe["schedule"]["rowIds"][1]
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.build(reference=reference)

    def test_returned_mapping_and_inputs_are_not_aliased(self):
        source, reference = copy.deepcopy(self.source),copy.deepcopy(self.reference_source)
        before = encoded([source,reference])
        first = build_binding_spatial_seams(source,reference)
        second = validate_binding_spatial_seams(source,reference,first)
        second["selectors"][0]["members"][0]["cells"].clear()
        second["rowComparisons"][0]["numericalResidualTerms"].clear()
        second["sewingControlSchedule"]["knots"][0]["activation"][0] = 0.
        self.assertEqual(encoded(first),encoded(self.descriptor))
        self.assertEqual(encoded([source,reference]),before)

    def test_stored_path_vs_panel_interpolation_defects_are_exact_and_not_repaired(self):
        nonzero = 0
        panels = {p["id"]:p for p in self.source["baseUnit"]["sourcePattern"]["panels"]}
        for selector in self.descriptor["selectors"]:
            for member in selector["members"]:
                template = self.source["baseUnit"]["sourceTemplates"][member["templateId"]]
                path = next(p for p in template["stitchPaths"] if p["name"] == member["registrationMember"]["pathName"])
                self.assertEqual(member["pathSha256"],digest(path))
                self.assertEqual(len(member["sourcePathSamples"]),len(path["samples"]))
                for index,(actual,sample) in enumerate(zip(member["sourcePathSamples"],path["samples"])):
                    panel = panels[member["templateId"]]
                    a, b = panel["points"][sample["sourceSegment"]:sample["sourceSegment"]+2]
                    t = F(sample["sourceFraction"])
                    expected = [F(x)-(1-t)*F(first)-t*F(last) for x,first,last in zip(sample["restPosition"],a,b)]
                    self.assertEqual(actual["sampleIndex"],index)
                    self.assertEqual(actual["sourceSampleSha256"],digest(sample))
                    self.assertEqual(list(map(rational,actual["storedPointMinusSourceInterpolationMm"])),expected)
                    nonzero += any(expected)
        self.assertGreater(nonzero,0)

    def test_reverse_member_maps_its_own_arc_range_and_retains_every_tiny_cell(self):
        from solver_binding_spatial_seams import _material, _member
        contexts,_ = _material(self.source)
        original = self.descriptor["selectors"][0]["members"][1]
        raw = copy.deepcopy(original["registrationMember"])
        raw["direction"] = "reverse"
        reversed_map = _member(self.source,contexts[raw["instanceId"]],raw)
        self.assertEqual(reversed_map["registrationMember"],raw)
        self.assertEqual(len(reversed_map["cells"]),len(original["cells"]))
        for first,last in zip(original["cells"],reversed(reversed_map["cells"])):
            lo,hi = map(rational,first["fractionInterval"])
            self.assertEqual(list(map(rational,last["fractionInterval"])),[1-hi,1-lo])
            for prefix in ("original","numerical"):
                self.assertEqual(last[prefix+"StartWeights"],first[prefix+"EndWeights"])
                self.assertEqual(last[prefix+"EndWeights"],first[prefix+"StartWeights"])
                self.assertEqual(last[prefix+"Candidates"],first[prefix+"Candidates"])
        # An explicit sub-arc has a separate origin and scale, not the other
        # member's slightly different source length or the whole stored path.
        start,end = raw["endArcMm"]/4,raw["endArcMm"]*3/4
        raw.update(startArcMm=start,endArcMm=end)
        subset = _member(self.source,contexts[raw["instanceId"]],raw)
        denominator = F(original["registrationMember"]["endArcMm"])
        for f in (F(),F(1,3),F(1)):
            original_fraction = (F(end)+(F(start)-F(end))*f)/denominator
            self.assertEqual(member_weights(subset,f),member_weights(original,original_fraction))

    def test_tiling_primitive_rejects_missing_overlapping_reversed_and_escaped_children(self):
        from solver_binding_spatial_seams import _tiling
        original = [[F(0),F(0)],[F(1),F(0)],[F(0),F(1)]]
        points = original + [[F(1,2),F(0)]]
        basis = [{0:F(1)},{1:F(1)},{2:F(1)},{0:F(1,2),1:F(1,2)}]
        faces = [[0,3,2],[3,1,2]]
        valid = _tiling(original,[[0,1,2]],points,faces,[0,0],basis)
        self.assertEqual(len(valid[0]["childTriangleIndices"]),2)
        for kind in ("gap","overlap","winding","escaped","wrong-parent"):
            p,f,b,parents = copy.deepcopy(points),copy.deepcopy(faces),copy.deepcopy(basis),[0,0]
            if kind == "gap":f.pop();parents.pop()
            if kind == "overlap":f.append(f[0]);parents.append(0)
            if kind == "winding":f[0].reverse()
            if kind == "escaped":p[3] = [F(1,2),F(-1,2)]
            if kind == "wrong-parent":b[3][9] = F(1,100)
            with self.subTest(kind=kind),self.assertRaises(ValueError):_tiling(original,[[0,1,2]],p,f,parents,b)

    def _constructed_nullspace_states(self):
        source = self.source
        selector = self.descriptor["selectors"][0]
        receiver,binding = [m["registrationMember"]["instanceId"] for m in selector["members"]]
        sleeve_template = self.material[receiver][0]
        sleeve_path = next(p for p in sleeve_template["stitchPaths"] if p["name"] == selector["members"][0]["registrationMember"]["pathName"])
        beginning,ending = [sample["restPosition"] for sample in (sleeve_path["samples"][0],sleeve_path["samples"][-1])]
        direction = [b-a for a,b in zip(beginning,ending)]
        length = math.hypot(*direction); tangent = [x/length for x in direction]
        transverse = [tangent[1],-tangent[0]]
        q0 = copy.deepcopy(source["restMeters"])
        binding_offset = source["instanceOffsets"][binding]
        for index,point in enumerate(self.material[binding][2]):
            q0[binding_offset+index] = [float(F(beginning[axis])/1000 + F(transverse[axis])*(point[0]-20)/1000
                + F(tangent[axis])*point[1]/1000) for axis in range(2)] + [.001]
        rows = source["embeddedConstraints"]["constraints"]
        incidence = {}
        for row_index,row in enumerate(rows):
            for term in row["terms"]:
                if term["instanceId"] == binding and term["coefficient"]:
                    incidence.setdefault(term["vertex"],set()).add(row_index)
        candidates = [t for t in rows[2]["terms"] if t["instanceId"] == binding and t["coefficient"]
            and t["vertex"] >= len(self.material[binding][1]) and incidence[t["vertex"]] == {2}]
        self.assertGreaterEqual(len(candidates),2)
        a,b = sorted(candidates,key=lambda term:self.material[binding][2][term["vertex"]][1])[::len(candidates)-1]
        first,second = binding_offset+a["vertex"],binding_offset+b["vertex"]
        exact = [[F(x) for x in point] for point in q0]
        exact[first][2] += F(1,2048)
        exact[second][2] -= F(a["coefficient"])/F(b["coefficient"])/2048
        q1 = [[float(x) for x in point] for point in exact]
        def vector(row,q):
            return [sum((F(t["coefficient"])*F(q[source["instanceOffsets"][t["instanceId"]]+t["vertex"]][axis])
                         for t in row["terms"]),F()) for axis in range(3)]
        # This is the nullspace of the actual source rows, not an interpolated
        # chord through five sample positions. Rounding is disclosed separately.
        for row in rows:
            self.assertEqual(vector(row,q0),vector(row,exact))
            self.assertLess(max(abs(x-y) for x,y in zip(vector(row,q0),vector(row,q1))),F(1,10**17))
        for row in source["baseUnit"]["embeddedConstraints"]["constraints"]:
            self.assertEqual(vector(row,q0),vector(row,q1))
        return q0,q1

    def test_actual_source_nullspace_opens_continuous_seam_with_unchanged_forty_samples(self):
        q0,q1 = self._constructed_nullspace_states()
        before = encoded([self.source,self.reference_source,self.descriptor,q0,q1])
        reports = [evaluate_binding_spatial_seams(self.source,self.reference_source,self.descriptor,q,tolerance_m=.0002)
                   for q in (q0,q1)]
        self.assertIs(reports[0]["activeSpatialDistanceBoundSatisfied"],True)
        self.assertIs(reports[1]["activeSpatialDistanceBoundSatisfied"],False)
        for q,report in zip((q0,q1),reports):
            self.assertEqual(report["positionsSha256"],digest(q))
            self.assertEqual(report["mapDescriptorSha256"],digest(self.descriptor))
            self.assertEqual(len(report["selectors"]),8)
            self.assertEqual(len(report["rowDiagnostics"]),40)
            self.assertIs(report["verified"],True)
            for field in ("accepted","controlsInstalled","constructionPhaseCompleted","continuousSpatialStitchingVerified"):
                self.assertIs(report[field],False)
            self.assertEqual([r["rowIndex"] for r in report["rowDiagnostics"] if r["held"]],list(range(5)))
            self.assertTrue(any(not s["certificate"]["withinTolerance"] for s in report["selectors"] if not s["held"]))
            self._assert_evaluation_geometry(report,q)
        certificate = reports[1]["selectors"][0]["certificate"]
        self.assertLess(rational(certificate["squaredGapMinimum"]["valueMetersSquared"]),F(6,10000)**2)
        self.assertGreater(rational(certificate["squaredGapMaximum"]["valueMetersSquared"]),F(14,10000)**2)
        for row in reports[1]["rowDiagnostics"][:5]:
            self.assertLess(abs(rational(row["rawSquaredGapMetersSquared"])-F(.001)**2),F(1,10**18))
        self.assertEqual(encoded([self.source,self.reference_source,self.descriptor,q0,q1]),before)

    def _assert_evaluation_geometry(self,report,q):
        def gap(selector,fraction):
            result = [F(),F(),F()]
            for sign,member in zip((1,-1),selector["members"]):
                offset = self.source["instanceOffsets"][member["registrationMember"]["instanceId"]]
                for vertex,weight in member_weights(member,fraction).items():
                    for axis in range(3):result[axis] += sign*weight*F(q[offset+vertex][axis])
            return result
        selectors = {i:s for s in self.descriptor["selectors"] for i in s["rowIndices"]}
        for item,row in zip(report["rowDiagnostics"],self.source["embeddedConstraints"]["constraints"]):
            actual = [sum((F(t["coefficient"])*F(q[self.source["instanceOffsets"][t["instanceId"]]+t["vertex"]][axis])
                          for t in row["terms"]),F()) for axis in range(3)]
            continuum = gap(selectors[item["rowIndex"]],F(row["fraction"]))
            self.assertEqual(list(map(rational,item["rawNumericalVectorMeters"])),actual)
            self.assertEqual(list(map(rational,item["continuumVectorMeters"])),continuum)
            self.assertEqual(list(map(rational,item["continuumMinusRawMeters"])),[b-a for a,b in zip(actual,continuum)])
            self.assertEqual(rational(item["rawSquaredGapMetersSquared"]),sum((x*x for x in actual),F()))
            self.assertEqual(rational(item["continuumSquaredGapMetersSquared"]),sum((x*x for x in continuum),F()))
        for selector,result in zip(self.descriptor["selectors"],report["selectors"]):
            bounds = sorted({rational(x) for member in selector["members"] for cell in member["cells"] for x in cell["fractionInterval"]}
                | {rational(k["fraction"]) for k in selector["targetKnots"]})
            certificate = result["certificate"]
            self.assertEqual(certificate["cellCount"],len(bounds)-1)
            extrema = []
            for cell,lo,hi in zip(certificate["cells"],bounds,bounds[1:]):
                self.assertEqual(list(map(rational,cell["interval"])),[lo,hi])
                start,end = gap(selector,lo),gap(selector,hi)
                delta = [b-a for a,b in zip(start,end)]
                a = sum((x*x for x in delta),F())
                b = 2*sum((x*y for x,y in zip(start,delta)),F())
                c = sum((x*x for x in start),F())
                self.assertEqual(list(map(rational,cell["squaredGapPolynomial"])),[a,b,c])
                points = [F(),F(1)]
                if a and 0 < -b/(2*a) < 1:points.append(-b/(2*a))
                extrema.extend((a*t*t+b*t+c,lo+(hi-lo)*t) for t in points)
            minimum = min(extrema)
            maximum = min(extrema,key=lambda x:(-x[0],x[1]))
            for name,expected in (("squaredGapMinimum",minimum),("squaredGapMaximum",maximum)):
                self.assertEqual(rational(certificate[name]["valueMetersSquared"]),expected[0])
                self.assertEqual(rational(certificate[name]["atFraction"]),expected[1])

    def test_evaluation_rejects_raw_shape_boolean_nonfinite_and_repaired_descriptor(self):
        q = copy.deepcopy(self.source["restMeters"])
        valid = evaluate_binding_spatial_seams(self.source,self.reference_source,self.descriptor,q,tolerance_m=0.)
        self.assertIs(valid["verified"],True)
        for kind in ("missing","ragged","bool","nan","wide","range"):
            changed = copy.deepcopy(q)
            if kind == "missing":changed.pop()
            if kind == "ragged":changed[0].pop()
            if kind == "bool":changed[0][0] = False
            if kind == "nan":changed[0][0] = float("nan")
            if kind == "wide":changed[0][0] = 2**53+1
            if kind == "range":changed[0][0] = 1000001.
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                evaluate_binding_spatial_seams(self.source,self.reference_source,self.descriptor,changed,tolerance_m=.001)
        changed = copy.deepcopy(self.descriptor);changed["rowComparisons"].pop()
        with self.assertRaises(ValueError):
            evaluate_binding_spatial_seams(self.source,self.reference_source,changed,q,tolerance_m=.001)
        for value in (True,-.001,float("inf")):
            with self.subTest(tolerance=value),self.assertRaises(ValueError):
                evaluate_binding_spatial_seams(self.source,self.reference_source,self.descriptor,q,tolerance_m=value)


if __name__ == "__main__":
    unittest.main()
