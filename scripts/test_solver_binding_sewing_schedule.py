"""Fresh original sewing references composed statically; no refined motion."""

import copy
from fractions import Fraction
import hashlib
import json
import math
import unittest

from solver_binding_sewing_schedule import build_binding_sewing_schedule, validate_binding_sewing_schedule
from solver_sewing_input import bind_sewing_activation, derive_sewing_row_ids, sewing_source_identity


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rat(value):
    if set(value) != {"numerator", "denominator"}:
        raise AssertionError("Exact rational witnesses must not add rounded values")
    result = Fraction(int(value["numerator"]), int(value["denominator"]))
    if str(result.numerator) != value["numerator"] or str(result.denominator) != value["denominator"]:
        raise AssertionError("Canonical reduced rational witness required")
    return result


def reseal(reference):
    reference["sewingActuation"]["sourceSha256"] = sewing_source_identity(reference)
    return reference


def reference_for(source, subdivisions):
    reference = copy.deepcopy(source["baseUnit"])
    # Explicit numerical controls, deliberately independent of this fixture's
    # parked geometry. This tests preservation, never a zero-energy placement.
    targets = [.001] * 5 + [index / 128 for index in range(5, 40)]
    targets[-1] = 1  # Preserve raw integer versus floating JSON identity.
    activation = [1] * 5 + [0.] * 35
    recipe = {"profile": "captured-sewing-activation-v1", "accepted": False,
        "sourceSha256": sewing_source_identity(reference), "mode": "distance",
        "initialTargetsMeters": targets, "finalTargetsMeters": copy.deepcopy(targets),
        "schedule": {"profile": "sewing-row-activation-v1", "rowIds": list(derive_sewing_row_ids(reference)),
            "knots": [{"fraction": fraction, "activation": copy.deepcopy(activation)} for fraction in (0, 1.)]}}
    reference["sewingActuation"] = recipe
    controls, _ = bind_sewing_activation(reference, subdivisions, sewing_mode="distance")
    assert list(controls.row_ids) == recipe["schedule"]["rowIds"]
    return reference


def original_input_vector(row, source, placement):
    points = placement["placedMeters"]
    return [sum((Fraction(term["coefficient"]) * Fraction(points[source["instanceOffsets"][term["instanceId"]]
        + term["vertex"]][axis]) for term in row["terms"]), Fraction()) for axis in range(3)]


class BindingSewingScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import locally: unittest must not discover the prior TestCase again.
        from test_solver_binding_control_schedule import BindingControlScheduleTests
        BindingControlScheduleTests.setUpClass.__func__(cls)
        cls.fold_schedule = cls.descriptor
        cls.reference_source = reference_for(cls.source, cls.fold_schedule["initialSubdivisions"])
        cls.descriptor = build_binding_sewing_schedule(cls.source, cls.fold, cls.placement,
            cls.fold_schedule, cls.reference_source)

    def build(self, *, source=None, fold=None, placement=None, fold_schedule=None, reference=None):
        return build_binding_sewing_schedule(self.source if source is None else source,
            self.fold if fold is None else fold, self.placement if placement is None else placement,
            self.fold_schedule if fold_schedule is None else fold_schedule,
            self.reference_source if reference is None else reference)

    def validate(self, descriptor, *, source=None, fold=None, placement=None, fold_schedule=None, reference=None):
        return validate_binding_sewing_schedule(self.source if source is None else source,
            self.fold if fold is None else fold, self.placement if placement is None else placement,
            self.fold_schedule if fold_schedule is None else fold_schedule,
            self.reference_source if reference is None else reference, descriptor)

    def test_all_forty_original_and_numerical_rows_bind_distinct_identities_and_phases(self):
        descriptor, base = self.descriptor, self.source["baseUnit"]
        self.assertEqual(len(descriptor["rowBindings"]), 40)
        self.assertEqual(descriptor["heldSelector"], {"registrationId":"bind_opening_left_left","memberIndex":1})
        self.assertEqual(descriptor["heldRowIndices"], list(range(5)))
        self.assertEqual(descriptor["pendingRowIndices"], list(range(5,40)))
        phases = {i:name for name, indices in base["phaseConstraintRows"].items() for i in indices}
        for index,(binding,original,numerical) in enumerate(zip(descriptor["rowBindings"],
                base["embeddedConstraints"]["constraints"], self.source["embeddedConstraints"]["constraints"])):
            fraction=Fraction(original["fraction"])
            identity="row:"+digest([original["registrationId"],original["memberIndex"],fraction.numerator,fraction.denominator])
            self.assertEqual(binding["rowIndex"],index)
            self.assertEqual(binding["rowId"],identity)
            self.assertEqual(binding["registrationId"],original["registrationId"])
            self.assertEqual(binding["memberIndex"],original["memberIndex"])
            self.assertEqual((binding["fractionNumerator"],binding["fractionDenominator"]),(fraction.numerator,fraction.denominator))
            self.assertEqual(binding["originalRowSha256"],digest(original))
            self.assertEqual(binding["numericalRowSha256"],digest(numerical))
            self.assertEqual(binding["phaseId"],phases[index])
            self.assertIs(binding["held"],index<5)
            self.assertEqual(binding["complianceMPerN"],1e-8)
        self.assertEqual([i for i,b in enumerate(descriptor["rowBindings"]) if b["phaseId"]=="cuff:left:close-shell"],[10,12,14,16,18])
        self.assertEqual([i for i,b in enumerate(descriptor["rowBindings"]) if b["phaseId"]=="cuff:left:attach-facing"],[11,13,15,17,19])
        for field,value in (("sourceSha256",self.source),("baseUnitSha256",base),
                ("originalBundleSha256",base["embeddedConstraints"]),("numericalBundleSha256",self.source["embeddedConstraints"]),
                ("referenceSourceSha256",self.reference_source),("referenceSewingActuationSha256",self.reference_source["sewingActuation"]),
                ("sourcePhasePlanSha256",base["phasePlan"]),("sourcePhaseConstraintRowsSha256",base["phaseConstraintRows"])):
            self.assertEqual(descriptor[field],digest(value))

    def test_five_exact_binary_input_geometry_witnesses_keep_original_and_derived_operators(self):
        self.assertEqual(len(self.descriptor["initialHeldGeometryWitnesses"]),5)
        original=self.source["baseUnit"]["embeddedConstraints"]["constraints"]
        numerical=self.source["embeddedConstraints"]["constraints"]
        for index,witness in enumerate(self.descriptor["initialHeldGeometryWitnesses"]):
            self.assertEqual(witness["rowIndex"],index)
            target=Fraction(self.reference_source["sewingActuation"]["initialTargetsMeters"][index])
            self.assertEqual(rat(witness["targetSquaredMetersSquared"]),target*target)
            squares=[]
            for key,row in (("originalOperator",original[index]),("numericalOperator",numerical[index])):
                vector=original_input_vector(row,self.source,self.placement)
                square=sum((x*x for x in vector),Fraction());squares.append(square)
                self.assertEqual([rat(x) for x in witness[key]["vectorMeters"]],vector)
                self.assertEqual(rat(witness[key]["squaredGapMetersSquared"]),square)
                self.assertEqual(rat(witness[key]["squaredGapMinusTargetSquaredMetersSquared"]),square-target*target)
            self.assertEqual(rat(witness["numericalMinusOriginalSquaredGapMetersSquared"]),squares[1]-squares[0])
        self.assertTrue(any(rat(w["numericalMinusOriginalSquaredGapMetersSquared"]) for w in self.descriptor["initialHeldGeometryWitnesses"]))

    def test_one_ulp_target_change_is_preserved_without_retargeting_geometry(self):
        reference=copy.deepcopy(self.reference_source)
        recipe=reference["sewingActuation"]
        recipe["initialTargetsMeters"][0]=math.nextafter(recipe["initialTargetsMeters"][0],math.inf)
        recipe["finalTargetsMeters"]=copy.deepcopy(recipe["initialTargetsMeters"])
        result=self.build(reference=reference)
        self.assertEqual(encoded(result["initialTargetsMeters"]),encoded(recipe["initialTargetsMeters"]))
        self.assertNotEqual(result["referenceSewingActuationSha256"],self.descriptor["referenceSewingActuationSha256"])
        for field in ("originalOperator","numericalOperator"):
            old,new=self.descriptor["initialHeldGeometryWitnesses"][0][field],result["initialHeldGeometryWitnesses"][0][field]
            self.assertEqual(old["vectorMeters"],new["vectorMeters"])
            self.assertEqual(old["squaredGapMetersSquared"],new["squaredGapMetersSquared"])
            self.assertNotEqual(old["squaredGapMinusTargetSquaredMetersSquared"],new["squaredGapMinusTargetSquaredMetersSquared"])
        self.assertEqual(result["initialHeldGeometryWitnesses"][1:],self.descriptor["initialHeldGeometryWitnesses"][1:])

    def test_distinct_pending_targets_and_raw_signed_zero_and_integer_controls_survive(self):
        reference=copy.deepcopy(self.reference_source)
        for knot in reference["sewingActuation"]["schedule"]["knots"]:
            knot["activation"][5]=-0.
            knot["activation"][1]=1.
        result=self.build(reference=reference)
        recipe=reference["sewingActuation"]
        self.assertEqual(encoded(result["sewingControlSchedule"]),encoded(recipe["schedule"]))
        self.assertEqual(encoded(result["initialTargetsMeters"]),encoded(recipe["initialTargetsMeters"]))
        self.assertEqual(len(set(result["initialTargetsMeters"][5:])),35)
        self.assertIs(type(result["initialTargetsMeters"][-1]),int)
        self.assertEqual(math.copysign(1,result["sewingControlSchedule"]["knots"][0]["activation"][5]),-1.)
        self.assertIs(result["rowBindings"][5]["held"],False)
        self.assertEqual(encoded(result["rowBindings"][-1]["targetMeters"]),b'1')

    def test_shared_time_and_fold_controls_are_copied_without_installing_or_completing(self):
        for field in ("durationSeconds","initialSubdivisions","maxDepth","retryGridDenominator","minimumStepSeconds",
                      "foldControlSchedule","foldStiffnessJoules"):
            self.assertEqual(encoded(self.descriptor[field]),encoded(self.fold_schedule[field]))
        self.assertEqual(self.descriptor["foldDescriptorSha256"],digest(self.fold))
        self.assertEqual(self.descriptor["placementDescriptorSha256"],digest(self.placement))
        self.assertEqual(self.descriptor["foldScheduleDescriptorSha256"],digest(self.fold_schedule))
        self.assertEqual(self.descriptor["timePolicy"],"one-original-fraction-grid-for-all-controls-v1")
        self.assertEqual(self.descriptor["releasedFoldTailPolicy"],"sewing-remains-held-when-folds-release")
        for field in ("accepted","solverReady","executable","historicalExtrasInherited","controlsInstalled",
                      "constructionPhaseCompleted","continuousSpatialSeamsVerified","materialSidesResolved"):
            self.assertIs(self.descriptor[field],False)
        for field,value in (("durationSeconds",2.),("initialSubdivisions",128),("maxDepth",7)):
            changed=copy.deepcopy(self.fold_schedule);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.build(fold_schedule=changed)

    def test_allowed_historical_extras_are_hashed_provenance_and_never_inherited(self):
        reference=copy.deepcopy(self.reference_source)
        reference.update(placedMeters=[[99.]], gripperActuation={"unverified":"must not be installed"},
            bindingFirstTurnDiagnostic={"textileSidePolicy":{"unverified":"must not resolve sides"}})
        reseal(reference)
        result=self.build(reference=reference)
        self.assertNotEqual(result["referenceSourceSha256"],self.descriptor["referenceSourceSha256"])
        for field in ("initialHeldGeometryWitnesses","foldControlSchedule","rowBindings"):
            self.assertEqual(encoded(result[field]),encoded(self.descriptor[field]))
        for field in ("placedMeters","gripperActuation","bindingFirstTurnDiagnostic"):
            self.assertNotIn(field,result)
        for field in ("historicalExtrasInherited","materialSidesResolved","controlsInstalled"):
            self.assertIs(result[field],False)
        for key in ("sewingFrames","foldActuation","newTimeline"):
            changed=copy.deepcopy(self.reference_source);changed[key]={};reseal(changed)
            with self.subTest(key=key),self.assertRaises(ValueError):self.build(reference=changed)

    def test_reference_row_compliance_profile_and_source_hash_mutations_reject(self):
        attacks=[lambda r:r["embeddedConstraints"]["constraints"][0]["terms"][0].update(coefficient=.5),
            lambda r:r["embeddedConstraints"]["constraints"][-1].update(complianceMPerN=2e-8),
            lambda r:r["phaseConstraintRows"]["cuff:left:bind-opening-left"].__setitem__(0,False),
            lambda r:r["sewingActuation"].update(mode="normal-offset"),
            lambda r:r["sewingActuation"].update(sourceSha256="0"*64),
            lambda r:r["sewingActuation"].update(accepted=0)]
        for index,attack in enumerate(attacks):
            changed=copy.deepcopy(self.reference_source);attack(changed)
            if index<3:reseal(changed)
            with self.subTest(index=index),self.assertRaises(ValueError):self.build(reference=changed)

    def test_explicit_constant_held_and_pending_partition_cannot_be_replaced(self):
        for kind in ("held-half","pending-active","row-reorder","extra-knot","boolean-mask"):
            changed=copy.deepcopy(self.reference_source);schedule=changed["sewingActuation"]["schedule"]
            if kind=="held-half":
                for knot in schedule["knots"]:knot["activation"][:5]=[.5]*5
            if kind=="pending-active":
                for knot in schedule["knots"]:knot["activation"][5:10]=[1.]*5
            if kind=="row-reorder":schedule["rowIds"][:2]=schedule["rowIds"][:2][::-1]
            if kind=="extra-knot":schedule["knots"].insert(1,{"fraction":.5,"activation":copy.deepcopy(schedule["knots"][0]["activation"])})
            if kind=="boolean-mask":schedule["knots"][0]["activation"][0]=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.build(reference=changed)

    def test_targets_are_positive_raw_bounded_and_endpoint_bytes_must_match(self):
        for value in (True,0.,-0.,100.000001,float('nan'),[.001]):
            changed=copy.deepcopy(self.reference_source)
            for key in ("initialTargetsMeters","finalTargetsMeters"):changed["sewingActuation"][key][0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.build(reference=changed)
        changed=copy.deepcopy(self.reference_source)
        changed["sewingActuation"]["finalTargetsMeters"][-1]=1.
        with self.assertRaises(ValueError):self.build(reference=changed)
        changed=copy.deepcopy(self.reference_source);changed["sewingActuation"]["initialTargetsMeters"].pop()
        with self.assertRaises(ValueError):self.build(reference=changed)

    def test_source_numerical_rows_and_phase_partition_cannot_be_silently_repaired(self):
        for kind in ("coefficient","compliance","phase-duplicate","phase-missing","phase-bool","controls"):
            changed=copy.deepcopy(self.source)
            if kind=="coefficient":changed["embeddedConstraints"]["constraints"][0]["terms"][0]["coefficient"]*=.99
            if kind=="compliance":changed["embeddedConstraints"]["constraints"][0]["complianceMPerN"]=2e-8
            if kind=="phase-duplicate":changed["baseUnit"]["phaseConstraintRows"]["cuff:left:bind-opening-left"][0]=1
            if kind=="phase-missing":changed["baseUnit"]["phaseConstraintRows"]["cuff:left:bind-opening-left"].pop()
            if kind=="phase-bool":changed["baseUnit"]["phaseConstraintRows"]["cuff:left:bind-opening-left"][0]=False
            if kind=="controls":changed["sewingActuation"]=copy.deepcopy(self.reference_source["sewingActuation"])
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.build(source=changed)

    def test_descriptor_full_rederivation_rejects_repaired_hashes_and_raw_scope_flags(self):
        attacks=[lambda d:d["rowBindings"][0].update(held=1),lambda d:d["rowBindings"][10].update(phaseId="cuff:left:attach-facing"),
            lambda d:d["initialHeldGeometryWitnesses"][0]["originalOperator"]["vectorMeters"][0].update(numerator="0"),
            lambda d:d["initialHeldGeometryWitnesses"][0]["numericalOperator"]["squaredGapMetersSquared"].update(roundedBinary64=0.),
            lambda d:d.update(accepted=0),lambda d:d.update(materialSidesResolved=True),
            lambda d:d["limitations"].pop(),lambda d:d["pendingRowIndices"].pop()]
        for index,attack in enumerate(attacks):
            changed=copy.deepcopy(self.descriptor);attack(changed)
            with self.subTest(index=index),self.assertRaises(ValueError):self.validate(changed)
        changed=copy.deepcopy(self.descriptor)
        changed["initialTargetsMeters"][0]=.01;changed["finalTargetsMeters"][0]=.01
        changed["rowBindings"][0]["targetMeters"]=.01
        forged=copy.deepcopy(self.reference_source)
        forged["sewingActuation"]["initialTargetsMeters"][0]=.01
        forged["sewingActuation"]["finalTargetsMeters"][0]=.01
        changed["referenceSourceSha256"]=digest(forged)
        changed["referenceSewingActuationSha256"]=digest(forged["sewingActuation"])
        with self.assertRaises(ValueError):self.validate(changed)

    def test_inputs_and_each_returned_nested_descriptor_are_isolated(self):
        inputs=list(map(copy.deepcopy,(self.source,self.fold,self.placement,self.fold_schedule,self.reference_source)))
        before=list(map(encoded,inputs))
        result=build_binding_sewing_schedule(*inputs)
        self.assertEqual(list(map(encoded,inputs)),before)
        independent=validate_binding_sewing_schedule(*inputs,result)
        independent["rowBindings"][0]["rowId"]="changed"
        independent["initialHeldGeometryWitnesses"][0]["originalOperator"]["vectorMeters"].clear()
        independent["sewingControlSchedule"]["knots"][0]["activation"][0]=0
        self.assertEqual(result,self.descriptor)
        result["foldControlSchedule"]["hinges"][0][0]=0
        result["initialTargetsMeters"][0]=.01
        self.assertEqual(list(map(encoded,inputs)),before)
        again=build_binding_sewing_schedule(*inputs);saved=encoded(again)
        inputs[-1]["sewingActuation"]["initialTargetsMeters"][0]=99.
        inputs[0]["restMeters"][40][0]=99.
        self.assertEqual(encoded(again),saved)


if __name__=="__main__":
    unittest.main()
