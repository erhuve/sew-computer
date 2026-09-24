"""Fresh-source static frame correspondence regressions; no motion/admission."""
import copy
from fractions import Fraction
import hashlib
import json
import unittest

from solver_binding_frames import build_binding_frames, validate_binding_frames
from solver_sewing_input import bind_sewing_activation, sewing_source_identity


INSTANCE = "opening_binding_left_left:shell"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rational(value):
    if set(value) != {"numerator", "denominator"}:
        raise AssertionError("Exact rational, without a rounded normal, required")
    result = Fraction(int(value["numerator"]), int(value["denominator"]))
    if value != {"numerator": str(result.numerator), "denominator": str(result.denominator)}:
        raise AssertionError("Canonical reduced rational required")
    return result


def subtract(a, b):
    return [x-y for x, y in zip(a, b)]


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Fraction())


def original_faces(source, identity):
    instance = next(x for x in source["instances"] if x["id"] == identity)
    return source["baseUnit"]["sourceTemplates"][instance["templateId"]]["triangles"]


def composed_lineage(source):
    count = len(source["bindingRefinement"]["originalLocalMesh"]["verticesMeters"])
    basis = [{i: Fraction(1)} for i in range(count)]
    for stage in source["bindingRefinement"]["stages"]:
        previous = basis
        basis = []
        for support in stage["output"]["sourceWeights"]:
            row = {}
            for key, raw_weight in support.items():
                for original, coefficient in previous[int(key)].items():
                    row[original] = row.get(original, Fraction()) + Fraction(raw_weight)*coefficient
            basis.append({k:v for k,v in row.items() if v})
    return basis


def candidate_indices(source, row_index, *, prune_below=None):
    original = source["baseUnit"]["embeddedConstraints"]["constraints"][row_index]
    numerical = source["embeddedConstraints"]["constraints"][row_index]
    terms = [t for t in numerical["terms"] if t["coefficient"] < 0]
    identity = terms[0]["instanceId"]
    support = {t["vertex"] for t in terms if prune_below is None or abs(t["coefficient"]) >= prune_below}
    old_support = {t["vertex"] for t in original["terms"] if t["coefficient"] < 0}
    faces = source["numericalMeshes"][identity]["triangles"]
    old_faces = original_faces(source, identity)
    parents = (source["bindingRefinement"]["ultimateOriginalTriangleIndices"] if identity == INSTANCE
               else list(range(len(faces))))
    return [i for i,face in enumerate(faces) if support.issubset(face) and old_support.issubset(old_faces[parents[i]])]


def region_oracle(source, path):
    """Dual components and directed winding, with no centroid/coordinate sign."""
    faces = source["numericalMeshes"][INSTANCE]["triangles"]
    chain = next(x["vertices"] for x in source["bindingRefinement"]["finalCreases"] if x["sourcePathName"] == path)
    cut = {tuple(sorted(pair)) for pair in zip(chain, chain[1:])}
    edge_faces = {}
    for index, face in enumerate(faces):
        for a,b in zip(face, face[1:]+face[:1]):
            edge_faces.setdefault(tuple(sorted((a,b))), []).append((index,a,b))
    adjacency = [set() for _ in faces]
    for edge, uses in edge_faces.items():
        if edge not in cut and len(uses) == 2:
            a,b = uses[0][0],uses[1][0]
            adjacency[a].add(b); adjacency[b].add(a)
    components = []
    unseen = set(range(len(faces)))
    while unseen:
        component, pending = set(), [min(unseen)]
        while pending:
            vertex = pending.pop()
            if vertex in component:
                continue
            component.add(vertex); pending.extend(adjacency[vertex]-component)
        unseen -= component; components.append(component)
    if len(components) != 2:
        raise AssertionError("Declared crease must partition exactly two components")
    seeds = set()
    for a,b in zip(chain,chain[1:]):
        uses = edge_faces[tuple(sorted((a,b)))]
        following = [i for i,x,y in uses if (x,y) == (a,b)]
        if len(uses) != 2 or len(following) != 1:
            raise AssertionError("Directed interior crease adjacency required")
        seeds.update(following)
    body = next(c for c in components if seeds <= c)
    return {i:("body" if i in body else "allowance") for i in range(len(faces))}


def request_for(descriptor):
    """Explicit synthetic choices, not inferred fabric sides or assembly policy."""
    result = copy.deepcopy(descriptor["request"])
    for index,(row,binding) in enumerate(zip(result["rows"],descriptor["rowBindings"])):
        candidate = binding["candidates"][-1]
        row["candidateId"] = candidate["candidateId"]
        row["offsetSide"] = 1 if index % 2 else -1
        row["attachmentRegion"] = candidate["creaseRegions"].get("right")
    for index,row in enumerate(result["instanceTextileSides"]):
        row["positiveNormalTextileSide"] = "right" if index % 2 else "wrong"
    return result


class BindingFramesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Local import avoids duplicate discovery of the older TestCase.
        from test_solver_binding_sewing_schedule import BindingSewingScheduleTests
        BindingSewingScheduleTests.setUpClass.__func__(cls)
        cls.sewing_schedule_descriptor = cls.descriptor
        cls.descriptor = build_binding_frames(cls.source, cls.reference_source)

    def build(self, *, source=None, reference=None, request=None):
        return build_binding_frames(self.source if source is None else source,
                                    self.reference_source if reference is None else reference, request)

    def validate(self, descriptor):
        return validate_binding_frames(self.source, self.reference_source, descriptor)

    def test_all_forty_rows_preserve_raw_operators_targets_compliance_and_phase_identity(self):
        d,base = self.descriptor,self.source["baseUnit"]
        self.assertEqual(len(d["rowBindings"]),40)
        phases = {i:phase for phase,indices in base["phaseConstraintRows"].items() for i in indices}
        originals = base["embeddedConstraints"]["constraints"]
        numerical = self.source["embeddedConstraints"]["constraints"]
        for index,(row,original,new) in enumerate(zip(d["rowBindings"], originals, numerical)):
            with self.subTest(row=index):
                f = Fraction(original["fraction"])
                row_id = "row:"+digest([original["registrationId"],original["memberIndex"],f.numerator,f.denominator])
                self.assertEqual(row["rowIndex"],index); self.assertEqual(row["rowId"],row_id)
                self.assertEqual(row["registrationId"],original["registrationId"])
                self.assertEqual(row["memberIndex"],original["memberIndex"])
                self.assertEqual((row["fractionNumerator"],row["fractionDenominator"]),(f.numerator,f.denominator))
                self.assertEqual(row["originalRowSha256"],digest(original)); self.assertEqual(row["numericalRowSha256"],digest(new))
                self.assertEqual(row["phaseId"],phases[index]); self.assertIs(row["held"],index<5)
                self.assertEqual(encoded(row["targetMeters"]),encoded(self.reference_source["sewingActuation"]["initialTargetsMeters"][index]))
                self.assertEqual(encoded(row["complianceMPerN"]),encoded(new["complianceMPerN"]))
                for field,source,raw in (("originalTerms",base,original),("numericalTerms",self.source,new)):
                    expect = [{"instanceId":t["instanceId"],"localVertex":t["vertex"],
                        "canonicalVertex":source["instanceOffsets"][t["instanceId"]]+t["vertex"],
                        "coefficient":t["coefficient"]} for t in raw["terms"] if t["coefficient"] != 0]
                    self.assertEqual(encoded(row[field]),encoded(expect))
        for key in ("initialTargetsMeters","finalTargetsMeters"):
            self.assertEqual(encoded(d[key]),encoded(self.reference_source["sewingActuation"][key]))
        self.assertIs(type(d["initialTargetsMeters"][-1]),int)
        self.assertEqual(encoded(d["sewingControlSchedule"]),encoded(self.reference_source["sewingActuation"]["schedule"]))
        self.assertEqual(d["heldRowIndices"],list(range(5))); self.assertEqual(d["pendingRowIndices"],list(range(5,40)))
        for field,value in (("sourceSha256",self.source),("baseUnitSha256",base),
                ("refinementDescriptorSha256",self.source["bindingRefinement"]),("seamRemapDescriptorSha256",self.source["bindingSeamRemap"]),
                ("referenceSourceSha256",self.reference_source),("referenceSewingActuationSha256",self.reference_source["sewingActuation"]),
                ("originalBundleSha256",base["embeddedConstraints"]),("numericalBundleSha256",self.source["embeddedConstraints"]),
                ("sourcePhasePlanSha256",base["phasePlan"]),("sourcePhaseConstraintRowsSha256",base["phaseConstraintRows"]),
                ("requestSha256",d["request"])):
            self.assertEqual(d[field],digest(value))

    def test_complete_oriented_candidate_enumeration_keeps_tiny_support(self):
        altered_inventory = []
        for index,row in enumerate(self.descriptor["rowBindings"]):
            expected = candidate_indices(self.source,index)
            self.assertEqual([c["localTriangleIndex"] for c in row["candidates"]],expected)
            for c in row["candidates"]:
                identity = c["instanceId"]; face = self.source["numericalMeshes"][identity]["triangles"][c["localTriangleIndex"]]
                self.assertEqual(c["localVertices"],face)
                self.assertEqual(c["canonicalVertices"],[v+self.source["instanceOffsets"][identity] for v in face])
                self.assertEqual(c["canonicalTriangleIndex"],c["localTriangleIndex"]+self.source["instanceTriangleOffsets"][identity])
                self.assertEqual(c["candidateId"],"face:"+digest([identity,c["localTriangleIndex"],face]))
            if index<5:
                negative=[t for t in row["numericalTerms"] if t["coefficient"]<0]
                self.assertEqual(len(negative),3)
                self.assertTrue(0<min(abs(t["coefficient"]) for t in negative)<1e-12)
                reduced = candidate_indices(self.source,index,prune_below=1e-12)
                if reduced != expected:
                    altered_inventory.append(index)
                    self.assertTrue(set(expected)<set(reduced))
        self.assertTrue(altered_inventory,"A tolerance-pruned support must visibly invent candidates on this fixture")

    def test_exact_cross_products_lineage_parent_support_and_topological_regions(self):
        basis = composed_lineage(self.source)
        regions = {name:region_oracle(self.source,name) for name in ("right","left")}
        for row in self.descriptor["rowBindings"]:
            for c in row["candidates"]:
                identity=c["instanceId"]; local=c["localVertices"]; index=c["localTriangleIndex"]
                parent = (self.source["bindingRefinement"]["ultimateOriginalTriangleIndices"][index]
                          if identity==INSTANCE else index)
                self.assertEqual(c["originalTriangleIndex"],parent)
                self.assertEqual(c["originalVertices"],original_faces(self.source,identity)[parent])
                old_support={t["localVertex"] for t in row["originalTerms"] if t["coefficient"]<0}
                self.assertTrue(old_support.issubset(c["originalVertices"]))
                for vertex,lineage in zip(local,c["vertexLineage"]):
                    self.assertEqual(lineage["localVertex"],vertex)
                    expect=basis[vertex] if identity==INSTANCE else {vertex:Fraction(1)}
                    self.assertEqual([x["vertex"] for x in lineage["originalWeights"]],sorted(expect))
                    self.assertEqual({x["vertex"]:rational(x["coefficient"]) for x in lineage["originalWeights"]},expect)
                    self.assertEqual(sum(expect.values(),Fraction()),1)
                    self.assertTrue(all(v>0 for v in expect.values()))
                    self.assertTrue(set(expect).issubset(c["originalVertices"]))
                p=[[Fraction(x) for x in self.source["restMeters"][v]] for v in c["canonicalVertices"]]
                normal=cross(subtract(p[1],p[0]),subtract(p[2],p[0])); squared=dot(normal,normal)
                self.assertGreater(squared,0)
                self.assertEqual([rational(x) for x in c["restCrossProductMetersSquared"]],normal)
                self.assertEqual(rational(c["restCrossProductSquaredNormMetersFourth"]),squared)
                expected={name:regions[name][index] for name in regions} if identity==INSTANCE else {}
                self.assertEqual(c["creaseRegions"],expected)

    def test_mesh_identities_use_current_offsets_and_original_local_order(self):
        original_triangle_offset=0
        for instance,mesh in zip(self.source["instances"],self.descriptor["meshes"]):
            name=instance["id"]; template=self.source["baseUnit"]["sourceTemplates"][instance["templateId"]]
            offset=self.source["baseUnit"]["instanceOffsets"][name]; count=len(template["restPositions"])
            original={"verticesMeters":self.source["baseUnit"]["restMeters"][offset:offset+count],"triangles":template["triangles"]}
            self.assertEqual(mesh,{"instanceId":name,"templateId":instance["templateId"],
                "originalMeshSha256":digest(original),"numericalMeshSha256":digest(self.source["numericalMeshes"][name]),
                "originalVertexOffset":offset,"vertexOffset":self.source["instanceOffsets"][name],
                "originalTriangleOffset":original_triangle_offset,"triangleOffset":self.source["instanceTriangleOffsets"][name],
                "originalVertexCount":count,"vertexCount":len(self.source["numericalMeshes"][name]["verticesMeters"]),
                "originalTriangleCount":len(template["triangles"]),"triangleCount":len(self.source["numericalMeshes"][name]["triangles"])})
            original_triangle_offset+=len(template["triangles"])
        sleeve=next(x for x in self.descriptor["meshes"] if x["instanceId"].startswith("sleeve_"))
        self.assertEqual(sleeve["vertexOffset"]-sleeve["originalVertexOffset"],18)

    def test_default_records_unique_support_but_retains_ambiguity_and_unknown_sides(self):
        d=self.descriptor; ambiguous=[]
        for index,row in enumerate(d["rowBindings"]):
            self.assertIsNone(row["offsetSide"]);self.assertIsNone(row["attachmentRegion"])
            if len(row["candidates"])==1:
                self.assertEqual(row["selectionAuthority"],"unique-support")
                self.assertEqual(row["selectedCandidateId"],row["candidates"][0]["candidateId"])
            else:
                ambiguous.append(index);self.assertIsNone(row["selectedCandidateId"])
                self.assertEqual(row["selectionAuthority"],"unresolved")
        self.assertEqual(ambiguous,[27,37])
        self.assertEqual(d["unresolvedFrameRowIndices"],ambiguous)
        self.assertEqual(d["unresolvedOffsetSideRowIndices"],list(range(40)))
        self.assertEqual(d["unresolvedTextileInstanceIds"],[x["id"] for x in self.source["instances"]])
        for field in ("frameChoicesComplete","offsetSidesComplete","textileSidesComplete","choicesComplete"):
            self.assertIs(d[field],False)
        self.assertTrue(all(x["positiveNormalTextileSide"] is None for x in d["instanceTextileSides"]))
        for request_row in d["request"]["rows"]:
            self.assertIsNone(request_row["candidateId"]); self.assertIsNone(request_row["offsetSide"])

    def test_ambiguous_flat_frames_develop_different_normals_with_fixed_anchor(self):
        q=[[Fraction(x) for x in p] for p in self.source["restMeters"]]
        for index in (27,37):
            row=self.descriptor["rowBindings"][index]
            support={t["canonicalVertex"] for t in row["numericalTerms"] if t["coefficient"]<0}
            a,b=sorted(support);self.assertEqual(q[a][1:],q[b][1:]);cy,cz=q[a][1:]
            # Constructive two-face metric witness only; no simulated trajectory
            # or claim about the surrounding full surface/contact is made.
            turned=[p[:] if p[1]<=cy else [p[0],cy-(p[2]-cz),cz+(p[1]-cy)] for p in q]
            self.assertTrue(all(turned[v]==q[v] for v in support))
            normals=[]
            for candidate in row["candidates"]:
                face=candidate["canonicalVertices"]
                for u,v in zip(face,face[1:]+face[:1]):
                    self.assertEqual(dot(subtract(q[u],q[v]),subtract(q[u],q[v])),
                                     dot(subtract(turned[u],turned[v]),subtract(turned[u],turned[v])))
                normals.append(cross(subtract(turned[face[1]],turned[face[0]]),subtract(turned[face[2]],turned[face[0]])))
            self.assertGreater(dot(normals[0],normals[0]),0);self.assertGreater(dot(normals[1],normals[1]),0)
            self.assertEqual(dot(normals[0],normals[1]),0)
            self.assertIsNone(row["selectedCandidateId"])

    def test_explicit_complete_choices_do_not_install_controls_or_resolve_mechanics(self):
        request=request_for(self.descriptor); result=self.build(request=request)
        self.assertEqual(encoded(result["request"]),encoded(request))
        for row,choice,old in zip(result["rowBindings"],request["rows"],self.descriptor["rowBindings"]):
            self.assertEqual(row["selectedCandidateId"],choice["candidateId"])
            self.assertEqual(row["selectionAuthority"],"explicit-request")
            self.assertEqual(row["offsetSide"],choice["offsetSide"])
            self.assertEqual(row["attachmentRegion"],choice["attachmentRegion"])
            self.assertEqual(row["candidates"],old["candidates"])
        for flag in ("frameChoicesComplete","offsetSidesComplete","textileSidesComplete","choicesComplete"):
            self.assertIs(result[flag],True)
        for flag in ("accepted","solverReady","executable","controlsInstalled","normalOffsetMechanicsExecuted",
                     "constructionPhaseCompleted","continuousSpatialSeamsVerified","historicalExtrasInherited"):
            self.assertIs(result[flag],False)
        for key in ("initialTargetsMeters","finalTargetsMeters","sewingControlSchedule","complianceMPerN"):
            self.assertEqual(encoded(result[key]),encoded(self.descriptor[key]))
        self.assertEqual(self.validate(result),result)
        with self.assertRaisesRegex(ValueError,"Refined cuff normal-offset"):
            bind_sewing_activation(self.source,64,sewing_mode="normal-offset")

    def test_topological_attachment_region_is_optional_separate_and_never_filters_inventory(self):
        request=copy.deepcopy(self.descriptor["request"])
        for index in range(5):
            request["rows"][index]["attachmentRegion"]=self.descriptor["rowBindings"][index]["candidates"][0]["creaseRegions"]["right"]
        result=self.build(request=request)
        for row,old in zip(result["rowBindings"],self.descriptor["rowBindings"]):
            self.assertEqual(row["candidates"],old["candidates"])
            self.assertEqual(row["selectedCandidateId"],old["selectedCandidateId"])
            self.assertIsNone(row["offsetSide"])
        for index in (0,1,4):
            wrong=copy.deepcopy(request);wanted=wrong["rows"][index]["attachmentRegion"]
            wrong["rows"][index]["attachmentRegion"]="allowance" if wanted=="body" else "body"
            with self.subTest(row=index),self.assertRaises(ValueError):self.build(request=wrong)
        wrong=copy.deepcopy(request);wrong["rows"][27]["attachmentRegion"]="body"
        with self.assertRaises(ValueError):self.build(request=wrong)

    def test_request_rejects_raw_types_incomplete_identity_reorder_and_inferred_textile_side(self):
        request=request_for(self.descriptor)
        attacks=[lambda r:r["rows"][0].update(offsetSide=True),lambda r:r["rows"][0].update(offsetSide=1.),
            lambda r:r["rows"][0].update(offsetSide=0),lambda r:r["rows"][0].update(offsetSide=float("nan")),
            lambda r:r["rows"][0].update(candidateId="face:"+"0"*64),lambda r:r["rows"][0].update(originalRowSha256="0"*64),
            lambda r:r["rows"].pop(),lambda r:r["rows"].reverse(),lambda r:r["instanceTextileSides"].reverse(),
            lambda r:r["instanceTextileSides"][0].update(positiveNormalTextileSide=True),
            lambda r:r["instanceTextileSides"][0].update(positiveNormalTextileSide="+z"),
            lambda r:r.update(sourceSha256="0"*64),lambda r:r.update(referenceSourceSha256="0"*64),
            lambda r:r.update(accepted=False)]
        for index,attack in enumerate(attacks):
            changed=copy.deepcopy(request);attack(changed)
            with self.subTest(attack=index),self.assertRaises(ValueError):self.build(request=changed)

    def test_full_rederivation_rejects_alias_winding_parent_lineage_and_repaired_hash_attacks(self):
        def reverse_face(d):
            c=d["rowBindings"][0]["candidates"][0]
            c["localVertices"][1:]=c["localVertices"][:0:-1]
            c["canonicalVertices"][1:]=c["canonicalVertices"][:0:-1]
            c["candidateId"]="face:"+digest([c["instanceId"],c["localTriangleIndex"],c["localVertices"]])
            d["rowBindings"][0]["selectedCandidateId"]=c["candidateId"]
        def request_swap(d):
            d["request"]["rows"][0]["offsetSide"]=1
            d["requestSha256"]=digest(d["request"])
        attacks=[reverse_face,request_swap,lambda d:d["rowBindings"].pop(),
            lambda d:d["rowBindings"][27]["candidates"].pop(),
            lambda d:d["rowBindings"][0]["candidates"][0].update(originalTriangleIndex=0),
            lambda d:d["rowBindings"][0]["candidates"][0]["canonicalVertices"].__setitem__(0,False),
            lambda d:d["rowBindings"][0]["candidates"][0]["restCrossProductMetersSquared"].__setitem__(2,{"numerator":"1","denominator":"1"}),
            lambda d:d["rowBindings"][0]["candidates"][0]["vertexLineage"][0]["originalWeights"][0]["coefficient"].update(numerator="0"),
            lambda d:d["rowBindings"][27].update(selectedCandidateId=d["rowBindings"][27]["candidates"][0]["candidateId"],selectionAuthority="unique-support"),
            lambda d:d["rowBindings"][0].update(held=1),lambda d:d.update(accepted=0),lambda d:d.update(executable=True),
            lambda d:d["instanceTextileSides"][0].update(positiveNormalTextileSide="right"),
            lambda d:d["meshes"][-1].update(vertexOffset=d["meshes"][-1]["originalVertexOffset"])]
        for index,attack in enumerate(attacks):
            changed=copy.deepcopy(self.descriptor);attack(changed)
            with self.subTest(attack=index),self.assertRaises(ValueError):self.validate(changed)

    def test_source_reference_and_control_extras_cannot_be_repaired_into_new_frame_authority(self):
        attacks=[lambda s:s["embeddedConstraints"]["constraints"][0]["terms"][0].update(coefficient=.5),
            lambda s:s["bindingRefinement"]["ultimateOriginalTriangleIndices"].__setitem__(0,11),
            lambda s:s["instanceOffsets"].update({next(k for k in s["instanceOffsets"] if k.startswith("sleeve_")):62}),
            lambda s:s.update(sewingFrames={}),lambda s:s.update(bindingFrames=self.descriptor)]
        for index,attack in enumerate(attacks):
            changed=copy.deepcopy(self.source);attack(changed)
            with self.subTest(source=index),self.assertRaises(ValueError):self.build(source=changed)
        for kind in ("compliance","normal-mode","boolean-mask"):
            changed=copy.deepcopy(self.reference_source)
            if kind=="compliance":changed["embeddedConstraints"]["constraints"][39]["complianceMPerN"]=2e-8
            if kind=="normal-mode":changed["sewingActuation"]["mode"]="normal-offset"
            if kind=="boolean-mask":changed["sewingActuation"]["schedule"]["knots"][0]["activation"][0]=True
            changed["sewingActuation"]["sourceSha256"]=sewing_source_identity(changed)
            with self.subTest(reference=kind),self.assertRaises(ValueError):self.build(reference=changed)

    def test_historical_textile_extras_stay_provenance_and_unknown_declarations_are_not_inferred(self):
        reference=copy.deepcopy(self.reference_source)
        reference.update(placedMeters=[[99.]],gripperActuation={"opaque":"not installed"},
            bindingFirstTurnDiagnostic={"textileSidePolicy":{"sleeve":"+z","binding":"-z"}})
        reference["sewingActuation"]["sourceSha256"]=sewing_source_identity(reference)
        result=self.build(reference=reference)
        self.assertNotEqual(result["referenceSourceSha256"],self.descriptor["referenceSourceSha256"])
        self.assertEqual(result["rowBindings"],self.descriptor["rowBindings"])
        self.assertEqual(result["instanceTextileSides"],self.descriptor["instanceTextileSides"])
        self.assertIs(result["historicalExtrasInherited"],False)

    def test_inputs_and_outputs_are_detached_without_replacing_raw_target_values(self):
        source=copy.deepcopy(self.source);reference=copy.deepcopy(self.reference_source);request=request_for(self.descriptor)
        before=[encoded(x) for x in (source,reference,request)]
        first=build_binding_frames(source,reference,request)
        self.assertEqual(before,[encoded(x) for x in (source,reference,request)])
        second=validate_binding_frames(source,reference,first)
        self.assertEqual(encoded(first),encoded(second))
        second["request"]["rows"][0]["offsetSide"]=None
        second["rowBindings"][0]["candidates"][0]["localVertices"][0]=-1
        self.assertNotEqual(encoded(first),encoded(second))
        self.assertEqual(before,[encoded(x) for x in (source,reference,request)])
        unchanged=encoded(first)
        source["numericalMeshes"][INSTANCE]["triangles"][0].reverse()
        reference["sewingActuation"]["initialTargetsMeters"][0]=.002
        request["rows"][0]["offsetSide"]=None
        self.assertEqual(encoded(first),unchanged)


if __name__ == "__main__":
    unittest.main()
