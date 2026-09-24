"""Independent exact squared-gap and static combined-declaration attacks."""

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

import solver_binding_sewing_schedule_replay as audit
from solver_binding_sewing_schedule_replay import verify_binding_sewing_schedule


SCRIPTS = Path(__file__).resolve().parent


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rat(value):
    if set(value) != {"numerator","denominator"}:
        raise AssertionError("Exact squared witnesses must not project to binary64")
    result = F(int(value["numerator"]),int(value["denominator"]))
    if str(result.numerator) != value["numerator"] or str(result.denominator) != value["denominator"]:
        raise AssertionError("Reduced positive-denominator rationals required")
    return result


def reseal(reference):
    reference["sewingActuation"]["sourceSha256"] = sha({k:v for k,v in reference.items() if k != "sewingActuation"})


class ExactCombinedArithmeticTests(unittest.TestCase):
    def test_original_local_prefix_uses_current_offset_and_nonunit_coefficients(self):
        source={"baseUnit":{"instances":[{"id":"strip","templateId":"a"},{"id":"sleeve","templateId":"b"}],
            "sourceTemplates":{"a":{"restPositions":[0]*3},"b":{"restPositions":[0]*3}},
            "instanceOffsets":{"strip":0,"sleeve":3}},
            "numericalMeshes":{"strip":{"verticesMeters":[0]*4},"sleeve":{"verticesMeters":[0]*3}},
            "instanceOffsets":{"strip":0,"sleeve":4}}
        points=[[.1,.2,.3],[.3,.5,.7],[.9,.2,.4],[99.,98.,97.],[2.,3.,4.],[5.,6.,7.],[8.,9.,10.]]
        row={"terms":[{"instanceId":"sleeve","vertex":0,"coefficient":1.},
            *[{"instanceId":"strip","vertex":i,"coefficient":-weight} for i,weight in enumerate((.1,.2,.7))]]}
        target=F(.001)**2
        operator,square=audit._operator(row,source,{"placedMeters":points},target,original=True)
        expected=[F(points[4][axis])-sum((F(weight)*F(points[i][axis]) for i,weight in enumerate((.1,.2,.7))),F()) for axis in range(3)]
        self.assertNotEqual(sum(map(F,(.1,.2,.7))),1)
        self.assertEqual([rat(value) for value in operator["vectorMeters"]],expected)
        self.assertEqual(square,sum((value*value for value in expected),F()))
        self.assertEqual(rat(operator["squaredGapMinusTargetSquaredMetersSquared"]),square-target)
        wrong=[value-F(points[4][axis])+F(points[3][axis]) for axis,value in enumerate(expected)]
        self.assertNotEqual(expected,wrong)
        altered=copy.deepcopy(row);altered["terms"][1]["vertex"]=3
        with self.assertRaises(ValueError):audit._operator(altered,source,{"placedMeters":points},target,original=True)

    def test_exact_squared_subnormal_is_retained_without_sqrt_or_float_projection(self):
        smallest=math.ulp(0.)
        source={"baseUnit":{"instances":[{"id":"a","templateId":"a"},{"id":"b","templateId":"b"}],
            "sourceTemplates":{key:{"restPositions":[0]*3} for key in ("a","b")}},
            "numericalMeshes":{key:{"verticesMeters":[0]*3} for key in ("a","b")},"instanceOffsets":{"a":0,"b":3}}
        placement={"placedMeters":[[smallest,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]}
        row={"terms":[{"instanceId":"a","vertex":0,"coefficient":1.},{"instanceId":"b","vertex":0,"coefficient":-1.}]}
        with mock.patch.object(math,"sqrt",side_effect=AssertionError("No square root")):
            operator,square=audit._operator(row,source,placement,F(smallest)**2,original=False)
        self.assertGreater(square,0)
        self.assertEqual(float(square),0.)
        self.assertEqual(rat(operator["squaredGapMetersSquared"]),F(smallest)**2)
        self.assertEqual(rat(operator["squaredGapMinusTargetSquaredMetersSquared"]),0)
        for change in (lambda r:r["terms"][0].update(vertex=False),lambda r:r["terms"][0].update(coefficient=True),
                       lambda r:r["terms"][0].update(coefficient=0.)):
            altered=copy.deepcopy(row);change(altered)
            with self.assertRaises(ValueError):audit._operator(altered,source,placement,F(),original=False)

    def test_phase_partition_rejects_boolean_alias_duplicate_and_missing_rows(self):
        phases={"a":list(range(5)),"b":list(range(5,40)),"empty":[]}
        self.assertEqual(len(audit._phase_membership({"phaseConstraintRows":phases})),40)
        for change in (lambda p:p["a"].__setitem__(0,False),lambda p:p["a"].__setitem__(1,1.),
                       lambda p:p["b"].append(0),lambda p:p["b"].pop(),lambda p:p.update(c=[40])):
            altered=copy.deepcopy(phases);change(altered)
            with self.assertRaises(ValueError):audit._phase_membership({"phaseConstraintRows":altered})

    def test_raw_all_input_admission_before_prerequisites(self):
        deep=0
        for _ in range(42):deep=[deep]
        for bad in (deep,{1:"bad"},{"x":(1,2)},{"x":math.nan},{"x":2**64},{"x":"\ud800"}):
            for slot in (0,4,5):
                inputs=[{} for _ in range(6)];inputs[slot]=bad
                with mock.patch.object(audit,"verify_binding_control_schedule") as folded:
                    with self.assertRaises(ValueError):verify_binding_sewing_schedule(*inputs)
                    folded.assert_not_called()


class BindingSewingScheduleReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_solver_binding_sewing_schedule import BindingSewingScheduleTests
        BindingSewingScheduleTests.setUpClass.__func__(cls)

    def verify(self,descriptor=None,*,source=None,fold=None,placement=None,fold_schedule=None,reference=None):
        return verify_binding_sewing_schedule(self.source if source is None else source,
            self.fold if fold is None else fold,self.placement if placement is None else placement,
            self.fold_schedule if fold_schedule is None else fold_schedule,
            self.reference_source if reference is None else reference,self.descriptor if descriptor is None else descriptor)

    def build(self,reference=None,placement=None,fold_schedule=None):
        from solver_binding_sewing_schedule import build_binding_sewing_schedule
        return build_binding_sewing_schedule(self.source,self.fold,self.placement if placement is None else placement,
            self.fold_schedule if fold_schedule is None else fold_schedule,
            self.reference_source if reference is None else reference)

    def test_fresh_complete_evidence_and_independent_original_input_oracle(self):
        inputs=(self.source,self.fold,self.placement,self.fold_schedule,self.reference_source,self.descriptor)
        before=[encoded(value) for value in inputs]
        report=self.verify()
        self.assertEqual((report["rowCount"],report["heldRowCount"],report["pendingRowCount"]),(40,5,35))
        self.assertIs(report["verified"],True);self.assertIs(report["accepted"],False)
        self.assertIs(report["foldScheduleAudit"]["verified"],True);self.assertIs(report["remapAudit"]["verified"],True)
        self.assertEqual(report["descriptorSha256"],sha(self.descriptor))
        self.assertIn("separate strict source validator",report["sourceScope"])
        self.assertEqual(before,[encoded(value) for value in inputs])
        nonzero=0
        for row_index,witness in enumerate(self.descriptor["initialHeldGeometryWitnesses"]):
            squares=[]
            for key,bundle in (("originalOperator",self.source["baseUnit"]["embeddedConstraints"]),
                               ("numericalOperator",self.source["embeddedConstraints"])):
                row=bundle["constraints"][row_index]
                vector=[sum((F(term["coefficient"])*F(self.placement["placedMeters"][self.source["instanceOffsets"][term["instanceId"]]+term["vertex"]][axis])
                             for term in row["terms"]),F()) for axis in range(3)]
                square=sum((value**2 for value in vector),F());squares.append(square)
                self.assertEqual([rat(value) for value in witness[key]["vectorMeters"]],vector)
                self.assertEqual(rat(witness[key]["squaredGapMetersSquared"]),square)
                self.assertEqual(rat(witness[key]["squaredGapMinusTargetSquaredMetersSquared"]),square-F(.001)**2)
            self.assertEqual(rat(witness["numericalMinusOriginalSquaredGapMetersSquared"]),squares[1]-squares[0])
            nonzero+=squares[1]!=squares[0]
        self.assertGreater(nonzero,0)

    def test_reference_types_signed_zero_and_all_pending_targets_preserved(self):
        reference=copy.deepcopy(self.reference_source)
        for key in ("initialTargetsMeters","finalTargetsMeters"):
            reference["sewingActuation"][key][0]=math.ulp(0.)
            reference["sewingActuation"][key][1]=100
        reference["sewingActuation"]["schedule"]["knots"][0]["fraction"]=-0.
        reference["sewingActuation"]["schedule"]["knots"][0]["activation"][7]=-0.
        reference["sewingActuation"]["schedule"]["knots"][1]["activation"][7]=0
        descriptor=self.build(reference)
        self.verify(descriptor,reference=reference)
        self.assertEqual(encoded(descriptor["sewingControlSchedule"]),encoded(reference["sewingActuation"]["schedule"]))
        self.assertIs(type(descriptor["initialTargetsMeters"][-1]),int)
        self.assertIs(type(descriptor["initialTargetsMeters"][1]),int)
        self.assertEqual(rat(descriptor["initialHeldGeometryWitnesses"][0]["targetSquaredMetersSquared"]),F(math.ulp(0.))**2)
        self.assertEqual(float(rat(descriptor["initialHeldGeometryWitnesses"][0]["targetSquaredMetersSquared"])),0.)
        self.assertEqual(descriptor["initialTargetsMeters"][5:],reference["sewingActuation"]["initialTargetsMeters"][5:])
        for change in (lambda d:d["sewingControlSchedule"]["knots"][0].update(fraction=0.),
                       lambda d:d["sewingControlSchedule"]["knots"][0]["activation"].__setitem__(7,0.),
                       lambda d:d["initialTargetsMeters"].__setitem__(39,1.)):
            modified=copy.deepcopy(descriptor);change(modified)
            with self.assertRaises(ValueError):self.verify(modified,reference=reference)

    def test_historical_extras_are_only_hashed_provenance_not_inherited(self):
        reference=copy.deepcopy(self.reference_source)
        reference.update(placedMeters={"unverified":"historical placement"},gripperActuation={"unverified":True},
                         bindingFirstTurnDiagnostic={"textileSide":"unverified historical research"})
        reseal(reference)
        descriptor=self.build(reference)
        self.verify(descriptor,reference=reference)
        self.assertIs(descriptor["historicalExtrasInherited"],False)
        self.assertEqual(descriptor["initialHeldGeometryWitnesses"],self.descriptor["initialHeldGeometryWitnesses"])
        self.assertNotEqual(descriptor["referenceSourceSha256"],self.descriptor["referenceSourceSha256"])
        self.assertEqual(descriptor["placementDescriptorSha256"],self.descriptor["placementDescriptorSha256"])

    def test_resealed_reference_cannot_change_base_raw_recipe_or_held_partition(self):
        mutations=[lambda r:r["limitations"].append("changed base"),lambda r:r.update(unexpected=True),
            lambda r:r["sewingActuation"].update(accepted=0),lambda r:r["sewingActuation"].update(mode="normal-offset"),
            lambda r:r["sewingActuation"]["finalTargetsMeters"].__setitem__(39,1.),
            lambda r:r["sewingActuation"]["initialTargetsMeters"].__setitem__(5,False),
            lambda r:r["sewingActuation"]["initialTargetsMeters"].__setitem__(5,0.),
            lambda r:r["sewingActuation"]["schedule"]["rowIds"].reverse(),
            lambda r:r["sewingActuation"]["schedule"]["knots"][0].update(fraction=False),
            lambda r:r["sewingActuation"]["schedule"]["knots"][0]["activation"].__setitem__(0,True),
            lambda r:r["sewingActuation"]["schedule"]["knots"][1]["activation"].__setitem__(4,0.),
            lambda r:r["sewingActuation"]["schedule"]["knots"][1]["activation"].__setitem__(5,1.),
            lambda r:r["sewingActuation"]["schedule"]["knots"].append({"fraction":1,"activation":[0]*40})]
        for index,change in enumerate(mutations):
            reference=copy.deepcopy(self.reference_source);change(reference);reseal(reference)
            descriptor=copy.deepcopy(self.descriptor)
            descriptor["referenceSourceSha256"]=sha(reference)
            descriptor["referenceSewingActuationSha256"]=sha(reference["sewingActuation"])
            with self.subTest(index=index),self.assertRaises(ValueError):self.verify(descriptor,reference=reference)

    def test_fully_repaired_claims_cannot_replace_exact_geometry_or_scope(self):
        def nudge(value):
            value["numerator"]=str(int(value["numerator"])+1)
        mutations=[lambda d:d.update(accepted=0),lambda d:d.update(executable=True),lambda d:d.update(controlsInstalled=True),
            lambda d:d.update(materialSidesResolved=True),lambda d:d.update(continuousSpatialSeamsVerified=True),
            lambda d:d.update(releasedFoldTailPolicy="all-controls-free"),lambda d:d.update(timePolicy="independent-times"),
            lambda d:d["heldRowIndices"].__setitem__(0,False),lambda d:d["rowBindings"][0].update(held=1),
            lambda d:d["rowBindings"][0].update(originalRowSha256=d["rowBindings"][0]["numericalRowSha256"]),
            lambda d:d["rowBindings"][5].update(targetMeters=.001),
            lambda d:nudge(d["initialHeldGeometryWitnesses"][0]["numericalOperator"]["vectorMeters"][0]),
            lambda d:nudge(d["initialHeldGeometryWitnesses"][1]["numericalMinusOriginalSquaredGapMetersSquared"]),
            lambda d:d["initialHeldGeometryWitnesses"][0]["targetSquaredMetersSquared"].update(roundedBinary64=1e-6),
            lambda d:d["limitations"].__setitem__(0,"Verified motion"),lambda d:d.update(extra=True)]
        for index,change in enumerate(mutations):
            descriptor=copy.deepcopy(self.descriptor);change(descriptor)
            self.assertTrue(encoded(descriptor)!=encoded(self.descriptor),index)
            with self.subTest(index=index),self.assertRaises(ValueError):self.verify(descriptor)
        descriptor=copy.deepcopy(self.descriptor)
        rational=descriptor["initialHeldGeometryWitnesses"][0]["originalOperator"]["squaredGapMetersSquared"]
        rational.update(numerator=str(2*int(rational["numerator"])),denominator=str(2*int(rational["denominator"])))
        with self.assertRaises(ValueError):self.verify(descriptor)

    def test_stale_valid_current_placement_and_fold_timeline_reject(self):
        from solver_binding_placement import build_binding_placement
        from solver_binding_control_schedule import build_binding_control_schedule
        request=copy.deepcopy(self.placement["request"])
        next(pose for pose in request["poses"] if pose["instanceId"]=="sleeve_left:shell")["translationMeters"][2]+=.01
        placement=build_binding_placement(self.source,request)
        folded=build_binding_control_schedule(self.source,self.fold,placement,self.fold_schedule["request"])
        with self.assertRaises(ValueError):self.verify(placement=placement,fold_schedule=folded)
        fresh=self.build(placement=placement,fold_schedule=folded)
        self.verify(fresh,placement=placement,fold_schedule=folded)
        # Geometry is bound to current placement, including the moved sleeve.
        self.assertNotEqual(fresh["initialHeldGeometryWitnesses"],self.descriptor["initialHeldGeometryWitnesses"])
        request=copy.deepcopy(self.fold_schedule["request"]);request["durationSeconds"]*=2
        folded=build_binding_control_schedule(self.source,self.fold,self.placement,request)
        with self.assertRaises(ValueError):self.verify(fold_schedule=folded)

    def test_original_and_numerical_row_compliance_and_selectors_are_strict(self):
        original=copy.deepcopy(self.source["baseUnit"]["embeddedConstraints"]["constraints"][0])
        for change in (lambda r:r.update(memberIndex=True),lambda r:r.update(fraction=False),
                       lambda r:r.update(complianceMPerN=True),lambda r:r.update(complianceMPerN=math.nextafter(1e-8,math.inf)),
                       lambda r:r.update(registrationId="\ninvalid")):
            row=copy.deepcopy(original);change(row)
            with self.assertRaises(ValueError):audit._row_identity(row)
        source=copy.deepcopy(self.source);source["embeddedConstraints"]["constraints"][39]["complianceMPerN"]=2e-8
        descriptor=copy.deepcopy(self.descriptor);descriptor["sourceSha256"]=sha(source)
        descriptor["numericalBundleSha256"]=sha(source["embeddedConstraints"])
        descriptor["rowBindings"][39].update(complianceMPerN=2e-8,numericalRowSha256=sha(source["embeddedConstraints"]["constraints"][39]))
        with self.assertRaises(ValueError):self.verify(descriptor,source=source)

    def test_stdlib_only_import_closure_and_isolated_execution(self):
        files=[SCRIPTS/"solver_binding_sewing_schedule_replay.py",*(SCRIPTS/name for name in audit.REQUIRED_HELPER_FILES)]
        allowed={"fractions","hashlib","json","math","re","struct",
                 *(path.stem for path in files)}
        for path in files:
            for node in ast.walk(ast.parse(path.read_text())):
                names=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
                self.assertTrue(set(names)<=allowed,(path.name,names))
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):self.assertNotIn(node.func.id,("eval","exec","__import__"))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for path in files:(root/path.name).write_bytes(path.read_bytes())
            (root/"input.json").write_bytes(encoded([self.source,self.fold,self.placement,self.fold_schedule,self.reference_source,self.descriptor]))
            program="import sys,json;sys.path.insert(0,sys.argv[1]);from solver_binding_sewing_schedule_replay import verify_binding_sewing_schedule;print(json.dumps(verify_binding_sewing_schedule(*json.load(open(sys.argv[2])))));assert 'numpy' not in sys.modules"
            result=subprocess.run([sys.executable,"-I","-c",program,str(root),str(root/"input.json")],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIs(json.loads(result.stdout)["verified"],True)


if __name__=="__main__":
    unittest.main()
