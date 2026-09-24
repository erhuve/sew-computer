"""Independent static frame arithmetic and repaired-descriptor attacks."""

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

import solver_binding_frames_replay as audit


SCRIPTS = Path(__file__).resolve().parent


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rational(value):
    assert set(value) == {"numerator", "denominator"}
    result = F(int(value["numerator"]), int(value["denominator"]))
    assert value == {"numerator": str(result.numerator), "denominator": str(result.denominator)}
    return result


class FrameArithmeticTests(unittest.TestCase):
    def test_exact_cross_winding_and_subnormal_squared_norm_without_normalization(self):
        tiny = math.ulp(0.)
        mesh = {"verticesMeters": [[0., 0., 0.], [tiny, 0., 0.], [0., tiny, 0.]], "triangles": [[0, 1, 2]]}
        source = {"numericalMeshes": {"a": mesh}, "instanceOffsets": {"a": 7}, "instanceTriangleOffsets": {"a": 4}}
        candidates = audit._candidates(source, {"a": mesh}, "a", {0}, "a", {0}, [], [], {})
        cross = [rational(v) for v in candidates[0]["restCrossProductMetersSquared"]]
        self.assertEqual(cross, [F(), F(), F(tiny)**2])
        self.assertEqual(rational(candidates[0]["restCrossProductSquaredNormMetersFourth"]), F(tiny)**4)
        self.assertEqual(float(cross[2]), 0.)
        self.assertEqual(candidates[0]["canonicalVertices"], [7, 8, 9])
        mesh["triangles"] = [[0, 2, 1]]
        reverse = audit._candidates(source, {"a": mesh}, "a", {0}, "a", {0}, [], [], {})[0]
        self.assertEqual(rational(reverse["restCrossProductMetersSquared"][2]), -F(tiny)**2)
        self.assertNotEqual(reverse["candidateId"], candidates[0]["candidateId"])
        mesh["verticesMeters"][2] = [2*tiny, 0., 0.]
        with self.assertRaisesRegex(ValueError, "Nondegenerate"):
            audit._candidates(source, {"a": mesh}, "a", {0}, "a", {0}, [], [], {})

    def test_dual_regions_follow_direction_and_reject_incomplete_chain(self):
        # Both faces retain the entire allowance/body cut domain.
        faces = [[0, 1, 2], [2, 1, 3]]
        self.assertEqual(audit._regions(faces, [1, 2]), {0: "body", 1: "allowance"})
        self.assertEqual(audit._regions(faces, [2, 1]), {0: "allowance", 1: "body"})
        for chain in ([0, 1], [1, 2, 1], [1]):
            with self.assertRaises(ValueError):
                audit._regions(faces, chain)

    def test_shared_anchor_and_equal_flat_normals_do_not_make_candidates_equivalent(self):
        # Constructed exact pose only: no forces, solver, source admission or motion.
        original = {"verticesMeters": [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,-1.,0.]],
                    "triangles": [[0,1,2], [1,0,3]]}
        mesh = copy.deepcopy(original)
        source = {"numericalMeshes": {"a": mesh}, "instanceOffsets": {"a": 0}, "instanceTriangleOffsets": {"a": 0}}
        flat = audit._candidates(source, {"a": original}, "a", {0,1}, "a", {0,1}, [], [], {})
        self.assertEqual(flat[0]["restCrossProductMetersSquared"], flat[1]["restCrossProductMetersSquared"])
        mesh["verticesMeters"][2] = [0.,0.,1.]
        bent = audit._candidates(source, {"a": original}, "a", {0,1}, "a", {0,1}, [], [], {})
        crosses = [[rational(v) for v in c["restCrossProductMetersSquared"]] for c in bent]
        self.assertEqual(crosses, [[0,-1,0], [0,0,1]])
        for face in original["triangles"]:
            for i,j in zip(face,face[1:]+face[:1]):
                square = lambda m: sum(((F(a)-F(b))**2 for a,b in zip(m["verticesMeters"][i],m["verticesMeters"][j])),F())
                self.assertEqual(square(mesh), square(original))
        self.assertEqual(mesh["verticesMeters"][:2], original["verticesMeters"][:2])

    def test_raw_budget_admission_precedes_every_prerequisite(self):
        nested = 0
        for _ in range(42):
            nested = [nested]
        for value in (nested, {1: 0}, {"x": (1, 2)}, {"x": math.nan}, {"x": 2**64}, {"x": "\ud800"}):
            for slot in range(3):
                args = [{}, {}, {}]
                args[slot] = value
                with mock.patch.object(audit, "verify_binding_remap") as prerequisite:
                    with self.assertRaises(ValueError):
                        audit.verify_binding_frames(*args)
                    prerequisite.assert_not_called()


class BindingFramesReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fixture sharing is allowed only in tests; avoid duplicate discovery.
        from test_solver_binding_sewing_schedule import BindingSewingScheduleTests
        from solver_binding_frames import build_binding_frames
        BindingSewingScheduleTests.setUpClass.__func__(cls)
        cls.descriptor = build_binding_frames(cls.source, cls.reference_source)

    def verify(self, descriptor=None, *, source=None, reference=None):
        return audit.verify_binding_frames(self.source if source is None else source,
            self.reference_source if reference is None else reference,
            self.descriptor if descriptor is None else descriptor)

    def build(self, *, request=None, reference=None):
        from solver_binding_frames import build_binding_frames
        return build_binding_frames(self.source, self.reference_source if reference is None else reference, request)

    def complete_request(self):
        request = copy.deepcopy(self.descriptor["request"])
        for index, (choice, row) in enumerate(zip(request["rows"], self.descriptor["rowBindings"])):
            choice["candidateId"] = row["candidates"][-1]["candidateId"]
            choice["offsetSide"] = -1 if index % 2 else 1
        for index, side in enumerate(request["instanceTextileSides"]):
            side["positiveNormalTextileSide"] = "wrong" if index % 2 else "right"
        return request

    def test_fresh_source_all_candidates_and_exact_crosses_independent_dense_oracle(self):
        before = [encoded(v) for v in (self.source, self.reference_source, self.descriptor)]
        evidence = self.verify()
        self.assertTrue(evidence["verified"])
        self.assertFalse(evidence["accepted"])
        self.assertEqual(evidence["candidateCount"], 42)
        self.assertEqual(evidence["ambiguousFrameRowIndices"], [27, 37])
        stages = self.source["bindingRefinement"]["stages"]
        first = stages[0]["output"]
        second = stages[1]["output"]
        # Explicit dense two-factor expansion; do not read composed witnesses.
        exact_basis = [[sum((F(weight)*F(first["sourceWeights"][int(mid)].get(str(i),0))
                            for mid,weight in support.items()),F()) for i in range(11)]
                       for support in second["sourceWeights"]]
        exact_parents = [first["parentTriangles"][p] for p in second["parentTriangles"]]
        for index, (row, binding) in enumerate(zip(self.source["embeddedConstraints"]["constraints"], self.descriptor["rowBindings"])):
            negative = [t for t in row["terms"] if F(t["coefficient"]) < 0]
            instance = negative[0]["instanceId"]
            support = {t["vertex"] for t in negative}
            mesh = self.source["numericalMeshes"][instance]
            complete = [i for i, face in enumerate(mesh["triangles"]) if support <= set(face)]
            self.assertEqual([c["localTriangleIndex"] for c in binding["candidates"]], complete)
            for candidate in binding["candidates"]:
                vertices = [mesh["verticesMeters"][i] for i in candidate["localVertices"]]
                # Independent determinant/cofactor form, not helper edge crosses.
                cross = []
                for i, j in ((1, 2), (2, 0), (0, 1)):
                    a, b, c = vertices
                    cross.append(F(a[i])*(F(b[j])-F(c[j])) + F(b[i])*(F(c[j])-F(a[j])) + F(c[i])*(F(a[j])-F(b[j])))
                self.assertEqual([rational(v) for v in candidate["restCrossProductMetersSquared"]], cross)
                self.assertEqual(rational(candidate["restCrossProductSquaredNormMetersFourth"]), sum((v*v for v in cross), F()))
                for vertex in candidate["vertexLineage"]:
                    weights = {v["vertex"]: rational(v["coefficient"]) for v in vertex["originalWeights"]}
                    self.assertEqual(sum(weights.values(), F()), 1)
                    self.assertTrue(all(v > 0 for v in weights.values()))
                    self.assertLessEqual(set(weights), set(candidate["originalVertices"]))
                    expected_weights = ({i:w for i,w in enumerate(exact_basis[vertex["localVertex"]]) if w}
                        if instance == audit.INSTANCE else {vertex["localVertex"]: F(1)})
                    self.assertEqual(weights, expected_weights)
                expected_parent = exact_parents[candidate["localTriangleIndex"]] if instance == audit.INSTANCE else candidate["localTriangleIndex"]
                self.assertEqual(candidate["originalTriangleIndex"], expected_parent)
            self.assertEqual(binding["held"], index < 5)
        self.assertEqual(before, [encoded(v) for v in (self.source, self.reference_source, self.descriptor)])

    def test_raw_lineage_requires_exact_unit_sums_and_immediate_parent_containment(self):
        altered = copy.deepcopy(self.source)
        support = altered["bindingRefinement"]["stages"][0]["output"]["sourceWeights"][11]
        key = next(iter(support))
        support[key] = math.nextafter(support[key], 0.)
        with self.assertRaisesRegex(ValueError, "normalized"):
            audit._lineage(altered)
        altered = copy.deepcopy(self.source)
        first, second = [stage["output"] for stage in altered["bindingRefinement"]["stages"]]
        attack = None
        for child, (face, parent) in enumerate(zip(second["triangles"],second["parentTriangles"])):
            support = {int(i) for v in face for i,w in second["sourceWeights"][v].items() if F(w)}
            for other, candidate in enumerate(first["triangles"]):
                if first["parentTriangles"][other] == first["parentTriangles"][parent] and not support <= set(candidate):
                    attack = child, other
                    break
            if attack is not None: break
        self.assertIsNotNone(attack)
        child, parent = attack
        second["parentTriangles"][child] = parent
        with self.assertRaisesRegex(ValueError, "immediate parent"):
            audit._lineage(altered)

    def test_default_does_not_infer_sides_and_complete_request_still_never_executes(self):
        self.assertEqual(self.descriptor["unresolvedFrameRowIndices"], [27, 37])
        self.assertEqual(self.descriptor["unresolvedOffsetSideRowIndices"], list(range(40)))
        self.assertEqual(len(self.descriptor["unresolvedTextileInstanceIds"]), 5)
        full = self.build(request=self.complete_request())
        self.assertTrue(self.verify(full)["choicesComplete"])
        for key in ("accepted", "solverReady", "executable", "controlsInstalled", "normalOffsetMechanicsExecuted",
                    "constructionPhaseCompleted", "continuousSpatialSeamsVerified", "historicalExtrasInherited"):
            self.assertIs(full[key], False)
        self.assertTrue(all(r["selectionAuthority"] == "explicit-request" for r in full["rowBindings"]))
        self.assertEqual(encoded(full["initialTargetsMeters"]), encoded(self.descriptor["initialTargetsMeters"]))

    def test_every_exact_nonzero_support_including_tiny_allowance_terms_is_mandatory(self):
        small = []
        for i, row in enumerate(self.descriptor["rowBindings"][:5]):
            terms = [t for t in row["numericalTerms"] if t["coefficient"] < 0]
            for term in terms:
                if abs(term["coefficient"]) < 1e-15:
                    small.append((i, term))
                    for c in row["candidates"]:
                        self.assertIn(term["localVertex"], c["localVertices"])
                    changed = copy.deepcopy(self.descriptor)
                    changed["rowBindings"][i]["numericalTerms"].remove(term)
                    with self.assertRaises(ValueError): self.verify(changed)
        self.assertGreaterEqual(len(small), 2)
        for i, row in enumerate(self.descriptor["rowBindings"][:5]):
            region = row["candidates"][0]["creaseRegions"]["right"]
            opposite = "allowance" if region == "body" else "body"
            forged = copy.deepcopy(self.descriptor)
            forged["request"]["rows"][i]["attachmentRegion"] = opposite
            forged["requestSha256"] = sha(forged["request"])
            forged["rowBindings"][i]["attachmentRegion"] = opposite
            with self.assertRaisesRegex(ValueError, "region"):
                self.verify(forged)

    def test_descriptor_field_mutations_cannot_repair_canonical_identity(self):
        mutations = [
            lambda d: d.update(accepted=0), lambda d: d.update(controlsInstalled=True),
            lambda d: d.update(choicesComplete=True), lambda d: d["rowBindings"].pop(),
            lambda d: d["rowBindings"][27]["candidates"].pop(),
            lambda d: d["rowBindings"][0].update(rowIndex=False),
            lambda d: d["rowBindings"][0].update(targetMeters=.002),
            lambda d: d["rowBindings"][0].update(complianceMPerN=1e-7),
            lambda d: d["rowBindings"][0]["candidates"][0].update(canonicalTriangleIndex=0),
            lambda d: d["rowBindings"][0]["candidates"][0]["canonicalVertices"].reverse(),
            lambda d: d["rowBindings"][0]["candidates"][0]["localVertices"].reverse(),
            lambda d: d["rowBindings"][0]["candidates"][0].update(originalTriangleIndex=0),
            lambda d: d["rowBindings"][0]["candidates"][0]["originalVertices"].reverse(),
            lambda d: d["rowBindings"][0]["candidates"][0]["vertexLineage"][0]["originalWeights"][0].update(vertex=False),
            lambda d: d["rowBindings"][0]["candidates"][0].update(restCrossProductMetersSquared=[{"numerator":"0","denominator":"1"}]*2+[{"numerator":"1","denominator":"1"}]),
            lambda d: d["meshes"][2].update(vertexOffset=0),
            lambda d: d["instanceTextileSides"][0].update(positiveNormalTextileSide="right"),
            lambda d: d["rowBindings"][27].update(selectedCandidateId=d["rowBindings"][27]["candidates"][0]["candidateId"], selectionAuthority="unique-support"),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                altered = copy.deepcopy(self.descriptor)
                mutation(altered)
                with self.assertRaises(ValueError): self.verify(altered)

    def test_repaired_request_hash_does_not_authorize_invalid_or_stale_choices(self):
        mutations = [lambda r: r["rows"][0].update(offsetSide=True), lambda r: r["rows"][0].update(offsetSide=1.),
            lambda r: r["rows"][0].update(offsetSide=-0.), lambda r: r["rows"][0].update(candidateId="face:"+"0"*64),
            lambda r: r["rows"][0].update(numericalRowSha256="0"*64), lambda r: r["rows"].reverse(),
            lambda r: r["instanceTextileSides"].reverse(), lambda r: r["rows"][10].update(attachmentRegion="body"),
            lambda r: r["instanceTextileSides"][0].update(positiveNormalTextileSide=1),
            lambda r: r.update(sourceSha256="0"*64), lambda r: r.update(referenceSourceSha256="0"*64),
            lambda r: r.update(unknown=None)]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                altered = copy.deepcopy(self.descriptor)
                mutation(altered["request"])
                altered["requestSha256"] = sha(altered["request"])
                with self.assertRaises(ValueError): self.verify(altered)

    def test_reference_signed_zero_raw_targets_and_stale_reference_are_distinct(self):
        reference = copy.deepcopy(self.reference_source)
        for knot in reference["sewingActuation"]["schedule"]["knots"]:
            knot["activation"][5] = -0.
        targets = reference["sewingActuation"]["initialTargetsMeters"]
        targets[0] = math.ulp(0.)
        targets[1] = 100
        reference["sewingActuation"]["finalTargetsMeters"] = copy.deepcopy(targets)
        reference["sewingActuation"]["sourceSha256"] = sha({k:v for k,v in reference.items() if k != "sewingActuation"})
        changed = self.build(reference=reference)
        self.verify(changed, reference=reference)
        self.assertEqual(encoded(changed["initialTargetsMeters"]), encoded(targets))
        self.assertEqual(encoded(changed["sewingControlSchedule"]), encoded(reference["sewingActuation"]["schedule"]))
        with self.assertRaises(ValueError): self.verify(changed)
        bad = copy.deepcopy(changed)
        bad["sewingControlSchedule"]["knots"][0]["activation"][5] = 0.
        with self.assertRaises(ValueError): self.verify(bad, reference=reference)

    def test_source_controls_downgrade_and_forged_rail_identity_fail_closed(self):
        for field in audit.CONTROL_FIELDS:
            source = copy.deepcopy(self.source)
            source[field] = None
            with self.assertRaises(ValueError): self.verify(source=source)
        for change in (lambda s: s.update(profile="synthetic"),
                       lambda s: s["bindingRefinement"]["finalCreases"][0]["vertices"].reverse(),
                       lambda s: s["bindingRefinement"]["finalCreases"][0]["edges"].pop()):
            source = copy.deepcopy(self.source)
            change(source)
            with self.assertRaises(ValueError): self.verify(source=source)

    def test_only_stdlib_independent_helpers_and_isolated_reproduction(self):
        files = ["solver_binding_frames_replay.py", *audit.REQUIRED_HELPER_FILES]
        allowed = {name[:-3] for name in files} | set(sys.stdlib_module_names)
        for name in files:
            for node in ast.walk(ast.parse((SCRIPTS/name).read_text())):
                imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
                for imported in imports:
                    self.assertIn(imported.split('.')[0], allowed)
                    self.assertNotEqual(imported, "solver_binding_frames")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in files: (target/name).write_bytes((SCRIPTS/name).read_bytes())
            (target/'inputs.json').write_bytes(encoded([self.source, self.reference_source, self.descriptor]))
            code = "import sys,json;sys.path.insert(0,sys.argv[1]);from solver_binding_frames_replay import verify_binding_frames;from pathlib import Path;x=json.loads((Path(sys.argv[1])/'inputs.json').read_text());print(json.dumps(verify_binding_frames(*x),sort_keys=True,separators=(',',':')))"
            result = subprocess.run([sys.executable, '-I', '-S', '-c', code, directory], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip().encode(), encoded(self.verify()))


if __name__ == "__main__":
    unittest.main()
