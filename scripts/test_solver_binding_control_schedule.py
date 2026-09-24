"""Fresh-source static schedule regressions; no motion or capture admission."""

import copy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from solver_binding_control_schedule import build_binding_control_schedule, validate_binding_control_schedule
from solver_binding_fold import build_binding_fold_control
from solver_binding_placement import build_binding_placement
from solver_binding_source import build_binding_source
from solver_fold_control_schedule import FoldControlSchedule
from test_solver_binding_fold import model_for, request as fold_request
from test_solver_binding_placement import request_for as placement_request


SCRIPTS = Path(__file__).resolve().parent


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rat(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def request_for(fold):
    return {"profile": "source-binding-fold-schedule-request-v1", "accepted": False,
        "durationSeconds": 1.024, "initialSubdivisions": 64, "maxDepth": 8,
        "gripperPolicy": "none-balanced-relative-folds-v1",
        "railOrder": [{"sourcePathName": rail["sourcePathName"],
            "referenceRotationRegion": rail["referenceRotationRegion"]} for rail in fold["rails"]],
        "knots": [{"fraction": fraction, "sourceRightHandAnglesRadians": angles, "activation": activation}
            for fraction, angles, activation in (
                (0., [0., -0.], [0., 0.]), (.125, [.03, -.04], [.25, 0.]),
                (.25, [.1, -.2], [1., 0.]), (.5, [.1, .3], [1., 1.]),
                (.75, [-.15, .2], [.5, .25]), (.875, [-.15, .2], [0., 0.]),
                (1., [-.15, .2], [0., 0.]))]}


def source_sample(request, fraction):
    knots = request["knots"]
    exact = next((knot for knot in knots if Fraction(knot["fraction"]) == fraction), None)
    if exact is not None:
        angles, activation = exact["sourceRightHandAnglesRadians"], exact["activation"]
    else:
        lower, upper = next((a, b) for a, b in zip(knots, knots[1:])
            if Fraction(a["fraction"]) < fraction < Fraction(b["fraction"]))
        amount = (fraction-Fraction(lower["fraction"]))/(Fraction(upper["fraction"])-Fraction(lower["fraction"]))
        angles, activation = [[(1-amount)*Fraction(a)+amount*Fraction(b) for a, b in zip(lower[key], upper[key])]
            for key in ("sourceRightHandAnglesRadians", "activation")]
    return angles, activation


def expected_sample(fold, request, fraction):
    """Interpolate source controls independently, then apply orientation parity."""
    angles, activation = source_sample(request, fraction)
    count = len(fold["foldActuation"]["hinges"])
    targets, weights, seen = [None]*count, [None]*count, set()
    for rail_index, rail in enumerate(fold["rails"]):
        region_sign = -1 if rail["referenceRotationRegion"] == "body" else 1
        for segment in rail["segments"]:
            native = fold["nativeTopology"]["hinges"][segment["nativeHingeIndex"]]
            source = segment["sourceDirectedHingeCanonical"]
            assert native[:2] in (source[:2], source[:2][::-1])
            assert native[2:] in (source[2:], source[2:][::-1])
            sign = (1 if native[:2] == source[:2] else -1) * (1 if native[2:] == source[2:] else -1)
            index = segment["actuatorIndex"]
            assert index not in seen
            seen.add(index)
            targets[index] = float(region_sign*sign*angles[rail_index])
            weights[index] = float(activation[rail_index])
    assert seen == set(range(count))
    return np.asarray(targets), np.asarray(weights)


class BindingControlScheduleTests(unittest.TestCase):
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
        _, cls.native = model_for(cls.source)
        cls.fold = build_binding_fold_control(cls.source, fold_request(), cls.native)
        cls.placement = build_binding_placement(cls.source, placement_request(cls.source))
        cls.request = request_for(cls.fold)
        cls.descriptor = build_binding_control_schedule(cls.source, cls.fold, cls.placement, cls.request)

    def build(self, request=None, source=None, fold=None, placement=None):
        return build_binding_control_schedule(self.source if source is None else source,
            self.fold if fold is None else fold, self.placement if placement is None else placement,
            self.request if request is None else request)

    def test_complete_native_identity_and_independent_source_orientation_mapping(self):
        numerical = self.descriptor["foldControlSchedule"]
        self.assertEqual(numerical["profile"], "fold-angle-activation-v1")
        self.assertEqual(numerical["hinges"], self.fold["foldActuation"]["hinges"])
        self.assertEqual(len(numerical["hinges"]), 16)
        self.assertEqual([len(rail["segments"]) for rail in self.fold["rails"]], [7, 9])
        for actual, original in zip(numerical["knots"], self.request["knots"]):
            self.assertEqual(actual["fraction"], original["fraction"])
            targets, activation = expected_sample(self.fold, self.request, Fraction(original["fraction"]))
            self.assertEqual(np.asarray(actual["targetsRadians"]).tobytes(), targets.tobytes())
            self.assertEqual(np.asarray(actual["activation"]).tobytes(), activation.tobytes())
        for key in ("accepted", "solverReady", "executable"):
            self.assertIs(self.descriptor[key], False)
        for key in ("sewingControlsInstalled", "gripperControlsInstalled", "constructionPhaseCompleted",
                    "staticFoldReferenceTargetsUsedForMotion"):
            self.assertIs(self.descriptor[key], False)
        for field,value in (("sourceSha256",self.source),("foldDescriptorSha256",self.fold),
                            ("placementDescriptorSha256",self.placement),("requestSha256",self.request)):
            self.assertEqual(self.descriptor[field],digest(value))
        self.assertEqual(encoded(validate_binding_control_schedule(self.source,self.fold,self.placement,self.descriptor)),
                         encoded(self.descriptor))

    def test_exact_original_fraction_retry_sampling_reversal_release_and_isolation(self):
        numerical = self.descriptor["foldControlSchedule"]
        controls = FoldControlSchedule(numerical, 64, hinges=numerical["hinges"])
        fractions = [Fraction(3, 16), Fraction(11, 64), Fraction(21, 128), Fraction(3, 16),
            Fraction(95, 128), Fraction(7, 8), Fraction(15, 16), Fraction(1), Fraction(1, 16384)]
        for fraction in fractions:
            actual = controls.parameters(fraction)
            expected = expected_sample(self.fold, self.request, fraction)
            for values, oracle in zip(actual, expected):
                self.assertEqual(values.tobytes(), oracle.tobytes())
                values[:] = 999
            for values, oracle in zip(controls.parameters(fraction), expected):
                self.assertEqual(values.tobytes(), oracle.tobytes())
        self.assertTrue(np.all(controls.parameters(Fraction(7, 8))[1] == 0))
        self.assertTrue(np.any(controls.parameters(Fraction(3, 4))[1] > 0))

    def test_valid_rail_reordering_changes_full_hinge_order_without_name_inference(self):
        request = fold_request()
        request["folds"].reverse()
        folded = build_binding_fold_control(self.source, request, self.native)
        requested = request_for(folded)
        descriptor = self.build(request=requested, fold=folded)
        self.assertEqual(descriptor["foldControlSchedule"]["hinges"], folded["foldActuation"]["hinges"])
        expected, _ = expected_sample(folded, requested, Fraction(1, 4))
        np.testing.assert_array_equal(descriptor["foldControlSchedule"]["knots"][2]["targetsRadians"], expected)
        with self.assertRaises(ValueError):
            self.build(request=self.request, fold=folded)

    def test_raw_boolean_shapes_unknown_controls_and_principal_branch_reject(self):
        mutations = [lambda r:r.update(accepted=0), lambda r:r.update(worldAxis=[0,1,0]),
            lambda r:r.update(gripperPolicy="fixed-body-grippers"),
            lambda r:r["knots"][1].update(sourceRightHandAnglesRadians=[True,0.]),
            lambda r:r["knots"][1].update(sourceRightHandAnglesRadians=[math.pi,0.]),
            lambda r:r["knots"][1].update(activation=[1.]),
            lambda r:r["knots"][1].update(activation=[float("nan"),0.]),
            lambda r:r["knots"][1].update(activation=[-1e-9,0.]),
            lambda r:r["knots"][1].update(activation=[1.000001,0.]),
            lambda r:r["knots"][1].update(activation=[False,0.])]
        for index, mutation in enumerate(mutations):
            changed=copy.deepcopy(self.request);mutation(changed)
            with self.subTest(index=index),self.assertRaises(ValueError):self.build(request=changed)

    def test_initial_controls_rail_identity_and_knot_grid_are_explicit(self):
        mutations = [lambda r:r["knots"][0]["activation"].__setitem__(0,.1),
            lambda r:r["knots"][0]["sourceRightHandAnglesRadians"].__setitem__(1,.1),
            lambda r:r["railOrder"].reverse(),
            lambda r:r["railOrder"][0].update(referenceRotationRegion="allowance"),
            lambda r:r["knots"][1].update(fraction=.1),
            lambda r:r["knots"][1].update(fraction=True),
            lambda r:r["knots"][1].update(fraction=0.),lambda r:r["knots"].pop()]
        for index,mutation in enumerate(mutations):
            changed=copy.deepcopy(self.request);mutation(changed)
            with self.subTest(index=index),self.assertRaises(ValueError):self.build(request=changed)

    def test_timing_and_retry_budget_are_explicit_and_do_not_underflow(self):
        mutations=[lambda r:r.update(durationSeconds=True),lambda r:r.update(durationSeconds=0.),
            lambda r:r.update(durationSeconds=math.nextafter(3600.,math.inf)),
            lambda r:r.update(durationSeconds=math.nextafter(0.,1.)),
            lambda r:r.update(initialSubdivisions=63),lambda r:r.update(initialSubdivisions=True),
            lambda r:r.update(initialSubdivisions=8192),lambda r:r.update(maxDepth=False),
            lambda r:r.update(maxDepth=-1),lambda r:r.update(maxDepth=31),
            lambda r:r.update(initialSubdivisions=4096,maxDepth=29)]
        for index,mutation in enumerate(mutations):
            changed=copy.deepcopy(self.request);mutation(changed)
            with self.subTest(index=index),self.assertRaises(ValueError):self.build(request=changed)
        request=copy.deepcopy(self.request);request.update(initialSubdivisions=4096,maxDepth=28)
        result=self.build(request=request)
        self.assertEqual(result["retryGridDenominator"],2**40)
        self.assertEqual(rat(result["minimumStepSeconds"]),Fraction(request["durationSeconds"])/2**40)
        for knot,time in zip(request["knots"],result["knotTimesSeconds"]):
            self.assertEqual(rat(time),Fraction(request["durationSeconds"])*Fraction(knot["fraction"]))
        request["durationSeconds"]=math.ldexp(math.nextafter(0.,1.),40)
        result=self.build(request=request)
        self.assertEqual(result["minimumStepSeconds"]["roundedBinary64"],math.nextafter(0.,1.))

    def test_minimum_active_coefficients_match_exhaustive_independent_fine_grid(self):
        request=copy.deepcopy(self.request)
        request.update(initialSubdivisions=8,maxDepth=2)
        # Leave the second rail active at the end: neither an implicit release
        # nor monotone activation is a prerequisite of this declaration.
        request["knots"][-1]["activation"][1]=.3
        result=self.build(request=request)
        grid=32
        witnesses={w["actuatorIndex"]:w for w in result["minimumActiveCoefficientWitnesses"]}
        self.assertEqual(set(witnesses),set(range(16)))
        for column,rail in enumerate(self.fold["rails"]):
            probes=[]
            for tick in range(grid+1):
                fraction=Fraction(tick,grid)
                exact=Fraction(source_sample(request,fraction)[1][column])
                if exact>0:probes.append((float(exact),fraction,exact))
            numerical,fraction,exact=min(probes,key=lambda item:(item[0],item[1]))
            for segment in rail["segments"]:
                index=segment["actuatorIndex"];witness=witnesses[index]
                self.assertIs(witness["everActive"],True)
                self.assertEqual(rat(witness["fraction"]),fraction)
                self.assertEqual(rat(witness["exactActivation"]),exact)
                self.assertEqual(witness["numericalActivation"],numerical)
                product=Fraction(self.fold["foldActuation"]["stiffnessJoules"][index])*Fraction(numerical)
                self.assertEqual(rat(witness["exactCoefficientFromRoundedActivationJoules"]),product)
                self.assertEqual(witness["numericalCoefficientJoules"],float(product))
                self.assertEqual(rat(witness["coefficientRoundingResidualJoules"]),Fraction(float(product))-product)
        for knot in request["knots"]:knot["activation"][1]=0.
        result=self.build(request=request)
        for index in range(7,16):
            self.assertEqual(result["minimumActiveCoefficientWitnesses"][index],{"actuatorIndex":index,"everActive":False})

    def test_positive_activation_and_effective_stiffness_underflow_reject_statically(self):
        tiny=math.nextafter(0.,1.)
        for weight,expected_error in ((tiny,"activation underflows"),(tiny*16384,"coefficient underflows")):
            request=copy.deepcopy(self.request)
            request["knots"]=[{"fraction":0.,"sourceRightHandAnglesRadians":[0.,0.],"activation":[0.,0.]},
                {"fraction":1.,"sourceRightHandAnglesRadians":[.1,0.],"activation":[weight,0.]}]
            with self.subTest(weight=weight),self.assertRaisesRegex(ValueError,expected_error):
                self.build(request=request)
        # This is the same declared grid, with sufficiently representable
        # positive activation and k*activation throughout every possible retry.
        request["knots"][-1]["activation"][0]=math.ldexp(tiny,40)
        result=self.build(request=request)
        self.assertTrue(all(w["numericalCoefficientJoules"]>0 for w in result["minimumActiveCoefficientWitnesses"][:7]))

    def test_changed_source_and_unvalidated_fold_or_placement_cannot_bind(self):
        sources=[]
        for mutation in (lambda s:s["restMeters"][40].__setitem__(0,s["restMeters"][40][0]+1e-8),
                         lambda s:s.update(gripperActuation={}),lambda s:s.update(accepted=0)):
            changed=copy.deepcopy(self.source);mutation(changed);sources.append(changed)
        for changed in sources:
            with self.assertRaises(ValueError):self.build(source=changed)
        folded=copy.deepcopy(self.fold)
        folded["foldActuation"]["hinges"][0].reverse()
        with self.assertRaises(ValueError):self.build(fold=folded)
        placed=copy.deepcopy(self.placement)
        placed["placedMeters"][0][0]=math.nextafter(placed["placedMeters"][0][0],math.inf)
        with self.assertRaises(ValueError):self.build(placement=placed)

    def test_valid_alternate_placement_or_fold_cannot_reuse_stale_schedule_binding(self):
        request=placement_request(self.source)
        request["poses"][0]["translationMeters"][0]+=.01
        placed=build_binding_placement(self.source,request)
        with self.assertRaises(ValueError):
            validate_binding_control_schedule(self.source,self.fold,placed,self.descriptor)
        request=fold_request()
        request["folds"][0]["stiffnessJoulesPerMeter"]*=2
        folded=build_binding_fold_control(self.source,request,self.native)
        with self.assertRaises(ValueError):
            validate_binding_control_schedule(self.source,folded,self.placement,self.descriptor)

    def test_descriptor_controls_and_scope_claims_rederive_strictly(self):
        mutations=[lambda d:d["foldControlSchedule"]["knots"][1]["targetsRadians"].__setitem__(0,.8),
            lambda d:d["foldControlSchedule"]["knots"][1]["activation"].__setitem__(0,0.),
            lambda d:d["foldControlSchedule"]["hinges"].reverse(),
            lambda d:d.update(accepted=0),lambda d:d.update(executable=True),lambda d:d.update(solverReady=True),
            lambda d:d.update(worldAxis=[0.,1.,0.])]
        for index,mutation in enumerate(mutations):
            changed=copy.deepcopy(self.descriptor);mutation(changed)
            with self.subTest(index=index),self.assertRaises(ValueError):
                validate_binding_control_schedule(self.source,self.fold,self.placement,changed)

    def test_inputs_and_returned_nested_recipes_are_detached(self):
        source,fold,placement,request=map(copy.deepcopy,(self.source,self.fold,self.placement,self.request))
        before=[encoded(value) for value in (source,fold,placement,request)]
        descriptor=build_binding_control_schedule(source,fold,placement,request)
        self.assertEqual([encoded(value) for value in (source,fold,placement,request)],before)
        descriptor["foldControlSchedule"]["hinges"][0][0]=0
        descriptor["foldControlSchedule"]["knots"][1]["activation"][0]=0.
        self.assertEqual([encoded(value) for value in (source,fold,placement,request)],before)
        result=self.build(source=source,fold=fold,placement=placement,request=request)
        captured=encoded(result)
        request["knots"][1]["activation"][0]=0.
        source["restMeters"][40][0]=99.
        fold["foldActuation"]["hinges"][0][0]=0
        placement["placedMeters"][0][0]=99.
        self.assertEqual(encoded(result),captured)


if __name__=="__main__":
    unittest.main()
