"""Independent schedule arithmetic, stale bindings and repaired-claim attacks.

The producer fixture only creates claims. Production verification imports no
producer, engine, numerical sampler or mechanics and executes no motion.
"""

import ast
import copy
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import solver_binding_control_schedule_replay as audit
from solver_binding_control_schedule_replay import verify_binding_control_schedule


SCRIPTS = Path(__file__).resolve().parent


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rational(value):
    return F(int(value["numerator"]), int(value["denominator"]))


class ScheduleArithmeticTests(unittest.TestCase):
    def test_integer_extrema_against_independent_exhaustive_fraction_oracle(self):
        rng = random.Random(3771)
        for _ in range(120):
            denominator = 32
            indices = [0, 4, 12, 16, 28, 32]
            values = [[rng.choice([0., .125, .3, .7, 1.])] for _ in indices]
            values[0] = [0.]
            actual = audit._minimum_on_grid(indices, values, 0, denominator)
            exhaustive = []
            for index in range(denominator + 1):
                segment = next((j for j in range(len(indices)-1) if indices[j] <= index <= indices[j+1]))
                amount = F(index-indices[segment], indices[segment+1]-indices[segment])
                alpha = F(values[segment][0]) + amount*(F(values[segment+1][0])-F(values[segment][0]))
                if alpha > 0:
                    exhaustive.append((float(alpha), F(index, denominator), alpha))
            expected = min(exhaustive, key=lambda row: (row[0], row[1])) if exhaustive else None
            self.assertEqual(actual, expected)

    def test_rounding_ties_release_neighbours_and_maximum_grid(self):
        smallest = math.ulp(0.)
        actual = audit._minimum_on_grid([0, 2, 4], [[0.], [3*smallest], [0.]], 0, 4)
        self.assertEqual(actual, (2*smallest, F(1, 4), F(3*smallest)/2))
        self.assertIsNone(audit._minimum_on_grid([0, 2**40], [[0.], [-0.]], 0, 2**40))
        self.assertEqual(audit._minimum_on_grid([0, 2**39, 2**40], [[0.], [1.], [0.]], 0, 2**40),
                         (2.**-39, F(1, 2**40), F(1, 2**39)))
        with self.assertRaisesRegex(ValueError, "activation underflows"):
            audit._minimum_on_grid([0, 2], [[0.], [smallest]], 0, 2)

    def test_raw_budget_admission_precedes_prerequisite_audits(self):
        deep = 0
        for _ in range(42):
            deep = [deep]
        for malformed in (deep, {1: "key"}, {"x": (1, 2)}, {"x": float("inf")},
                          {"x": 2**64}, {"x": "\ud800"}):
            with mock.patch.object(audit, "verify_binding_fold_control") as prerequisite:
                with self.assertRaises(ValueError):
                    verify_binding_control_schedule(malformed, {}, {}, {})
                prerequisite.assert_not_called()


class BindingControlScheduleReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_solver_binding_control_schedule import BindingControlScheduleTests
        cls.fixture = BindingControlScheduleTests
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.fixture.setUpClass()
        cls.source, cls.fold, cls.placement, cls.descriptor = (
            getattr(cls.fixture, key) for key in ("source", "fold", "placement", "descriptor"))

    def verify(self, descriptor=None, source=None, fold=None, placement=None):
        return verify_binding_control_schedule(self.source if source is None else source,
            self.fold if fold is None else fold, self.placement if placement is None else placement,
            self.descriptor if descriptor is None else descriptor)

    def build(self, request=None, fold=None):
        from solver_binding_control_schedule import build_binding_control_schedule
        return build_binding_control_schedule(self.source, self.fold if fold is None else fold,
            self.placement, self.descriptor["request"] if request is None else request)

    def test_complete_fresh_source_static_evidence_and_no_input_mutation(self):
        before = [encoded(value) for value in (self.source, self.fold, self.placement, self.descriptor)]
        report = self.verify()
        self.assertIs(report["verified"], True)
        self.assertIs(report["accepted"], False)
        self.assertEqual((report["railCount"], report["hingeCount"], report["knotCount"]), (2, 16, 7))
        self.assertEqual(report["descriptorSha256"], sha(self.descriptor))
        self.assertEqual(report["sourceSha256"], sha(self.source))
        self.assertIs(report["foldAudit"]["verified"], True)
        self.assertIs(report["placementAudit"]["verified"], True)
        self.assertIn("Pattern/split validity", report["sourceScope"])
        self.assertEqual(before, [encoded(value) for value in (self.source, self.fold, self.placement, self.descriptor)])
        for raw, witness in zip(self.descriptor["request"]["knots"], self.descriptor["knotTimesSeconds"]):
            self.assertEqual(rational(witness), F(raw["fraction"])*F(self.descriptor["durationSeconds"]))

    def test_source_parity_rederived_from_native_pairs_including_reordering(self):
        from solver_binding_fold import build_binding_fold_control
        from test_solver_binding_control_schedule import request_for
        for reverse in (False, True):
            request = copy.deepcopy(self.fold["request"])
            if reverse:
                request["folds"].reverse()
            fold = build_binding_fold_control(self.source, request, self.fixture.native)
            descriptor = self.build(request_for(fold), fold)
            self.verify(descriptor, fold=fold)
            for row, knot in zip(descriptor["foldControlSchedule"]["knots"], descriptor["request"]["knots"]):
                for column, rail in enumerate(fold["rails"]):
                    for segment in rail["segments"]:
                        source = segment["sourceDirectedHingeCanonical"]
                        native = fold["nativeTopology"]["hinges"][segment["nativeHingeIndex"]]
                        parity = (1 if native[0] == source[0] else -1)*(1 if native[2] == source[2] else -1)
                        sign = parity*(-1 if rail["referenceRotationRegion"] == "body" else 1)
                        expected = sign*float(knot["sourceRightHandAnglesRadians"][column])
                        self.assertEqual(encoded(row["targetsRadians"][segment["actuatorIndex"]]), encoded(expected))

    def test_signed_zero_complete_flags_and_full_descriptor_field_attacks(self):
        mutations = [lambda d:d.update(accepted=0), lambda d:d.update(solverReady=True),
            lambda d:d.update(executable=True), lambda d:d.update(staticFoldReferenceTargetsUsedForMotion=True),
            lambda d:d.update(gripperControlsInstalled=True), lambda d:d.update(constructionPhaseCompleted=True),
            lambda d:d.update(extra=True), lambda d:d.pop("limitations"),
            lambda d:d["limitations"].__setitem__(0,"Motion verified"),
            lambda d:d["foldControlSchedule"]["knots"][0]["targetsRadians"].__setitem__(0,0.),
            lambda d:d["railBindings"][0]["sourceRightHandToNativeSigns"].__setitem__(0,1),
            lambda d:d["foldControlSchedule"]["hinges"].reverse()]
        for index, change in enumerate(mutations):
            descriptor = copy.deepcopy(self.descriptor)
            change(descriptor)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify(descriptor)

    def test_repaired_request_hash_rejects_raw_semantic_and_grid_attacks(self):
        changes = [lambda r:r.update(accepted=0), lambda r:r.update(initialSubdivisions=True),
            lambda r:r.update(initialSubdivisions=3), lambda r:r.update(maxDepth=31),
            lambda r:r.update(initialSubdivisions=4096,maxDepth=29), lambda r:r.update(durationSeconds=0.),
            lambda r:r.update(durationSeconds=math.ulp(0.)), lambda r:r.update(gripperPolicy="fixed-region"),
            lambda r:r["railOrder"].reverse(), lambda r:r["knots"][1].update(fraction=.1),
            lambda r:r["knots"][1].update(fraction=True),
            lambda r:r["knots"][1]["activation"].__setitem__(0,True),
            lambda r:r["knots"][1]["activation"].__setitem__(0,-.1),
            lambda r:r["knots"][1]["sourceRightHandAnglesRadians"].__setitem__(0,math.pi-1e-8),
            lambda r:r["knots"][0]["activation"].__setitem__(0,.1),
            lambda r:r["knots"].pop(), lambda r:r["knots"][1].update(fraction=0.)]
        for index, change in enumerate(changes):
            descriptor = copy.deepcopy(self.descriptor)
            change(descriptor["request"])
            descriptor["requestSha256"] = sha(descriptor["request"])
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify(descriptor)
        for bad in (float("nan"), float("inf"), float("-inf")):
            descriptor = copy.deepcopy(self.descriptor)
            descriptor["request"]["durationSeconds"] = bad
            with self.assertRaises(ValueError):
                self.verify(descriptor)

    def test_repaired_expanded_controls_physical_times_and_coefficient_witnesses_reject(self):
        changes = [lambda d:d["foldControlSchedule"]["knots"][2]["targetsRadians"].__setitem__(0,.8),
            lambda d:d["foldControlSchedule"]["knots"][2]["activation"].__setitem__(0,.5),
            lambda d:d["minimumStepSeconds"].update(numerator="1"),
            lambda d:d["knotTimesSeconds"][1].update(roundedBinary64=.5),
            lambda d:d.update(retryGridDenominator=2**40),
            lambda d:d["minimumActiveCoefficientWitnesses"][0].update(everActive=False),
            lambda d:d["minimumActiveCoefficientWitnesses"][0]["exactActivation"].update(numerator="2"),
            lambda d:d["minimumActiveCoefficientWitnesses"][0]["fraction"].update(numerator="2"),
            lambda d:d["minimumActiveCoefficientWitnesses"][0].update(numericalCoefficientJoules=1.),
            lambda d:d["minimumActiveCoefficientWitnesses"][0]["coefficientRoundingResidualJoules"].update(numerator="1"),
            lambda d:d["railBindings"][0]["initialPlacedChainMeters"][0].__setitem__(2,99.)]
        for index, change in enumerate(changes):
            descriptor = copy.deepcopy(self.descriptor)
            change(descriptor)
            self.assertTrue(encoded(descriptor) != encoded(self.descriptor), index)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify(descriptor)

    def test_fully_valid_changed_prerequisites_cannot_reuse_stale_binding(self):
        from solver_binding_placement import build_binding_placement
        from solver_binding_fold import build_binding_fold_control
        request = copy.deepcopy(self.placement["request"])
        request["poses"][0]["translationMeters"][0] += .01
        placement = build_binding_placement(self.source, request)
        with self.assertRaises(ValueError):
            self.verify(placement=placement)
        request = copy.deepcopy(self.fold["request"])
        request["folds"][0]["stiffnessJoulesPerMeter"] *= 2
        fold = build_binding_fold_control(self.source, request, self.fixture.native)
        with self.assertRaises(ValueError):
            self.verify(fold=fold)

    def test_prerequisites_cannot_be_bypassed_by_repaired_schedule_hashes(self):
        for what in ("source", "fold", "placement"):
            source, fold, placement, descriptor = map(copy.deepcopy,(self.source,self.fold,self.placement,self.descriptor))
            if what == "source":
                source["instances"].reverse()
                descriptor["sourceSha256"] = sha(source)
                fold["sourceSha256"] = placement["sourceSha256"] = sha(source)
            elif what == "fold":
                fold["rails"][0]["segments"][0]["nativeAngleSign"] *= -1
            else:
                placement["placedMeters"][0][2] = math.nextafter(placement["placedMeters"][0][2], math.inf)
            descriptor["foldDescriptorSha256"] = sha(fold)
            descriptor["placementDescriptorSha256"] = sha(placement)
            with self.subTest(what=what), self.assertRaises(ValueError):
                self.verify(descriptor, source, fold, placement)

    def test_maximum_grid_inactive_controls_and_reference_targets_not_motion(self):
        request = copy.deepcopy(self.descriptor["request"])
        request.update(initialSubdivisions=4096, maxDepth=28)
        for knot in request["knots"]:
            knot["activation"][1] = 0.
        descriptor = self.build(request)
        result = self.verify(descriptor)
        self.assertEqual(result["retryGridDenominator"], 2**40)
        self.assertEqual(result["everActiveHingeCount"], 7)
        self.assertTrue(all(w == {"actuatorIndex": i, "everActive": False}
                            for i,w in enumerate(descriptor["minimumActiveCoefficientWitnesses"]) if i >= 7))
        self.assertNotEqual(descriptor["foldControlSchedule"]["knots"][2]["targetsRadians"],
                            descriptor["staticFoldReferenceTargetsRadians"])
        self.assertIs(descriptor["staticFoldReferenceTargetsUsedForMotion"], False)

    def test_double_rounding_and_underflow_from_original_activation_inputs(self):
        request = copy.deepcopy(self.descriptor["request"])
        # A small grid is exhaustively enumerable by this test oracle.
        request.update(initialSubdivisions=8, maxDepth=2)
        request["knots"] = [{"fraction": fraction, "sourceRightHandAnglesRadians": [0.,0.],
                             "activation": values} for fraction, values in
                            ((0., [0.,0.]), (.375, [.3,.7]), (1., [0.,0.]))]
        descriptor = self.build(request)
        self.verify(descriptor)
        rounded_activation_cases = 0
        for column, rail in enumerate(self.fold["rails"]):
            candidates = []
            knots = request["knots"]
            for index in range(33):
                t = F(index,32)
                lower,upper = next((a,b) for a,b in zip(knots,knots[1:]) if F(a["fraction"]) <= t <= F(b["fraction"]))
                amount = (t-F(lower["fraction"]))/(F(upper["fraction"])-F(lower["fraction"]))
                alpha = F(lower["activation"][column]) + amount*(F(upper["activation"][column])-F(lower["activation"][column]))
                if alpha > 0:
                    candidates.append((float(alpha),t,alpha))
            minimum = min(candidates,key=lambda row:(row[0],row[1]))
            for segment in rail["segments"]:
                i = segment["actuatorIndex"]
                witness = descriptor["minimumActiveCoefficientWitnesses"][i]
                self.assertEqual(rational(witness["exactActivation"]), minimum[2])
                product = F(self.fold["foldActuation"]["stiffnessJoules"][i])*F(minimum[0])
                self.assertEqual(rational(witness["exactCoefficientFromRoundedActivationJoules"]),product)
                self.assertEqual(witness["numericalCoefficientJoules"],float(product))
                if F(minimum[0]) != minimum[2]:
                    rounded_activation_cases += 1
                    self.assertNotEqual(product,F(self.fold["foldActuation"]["stiffnessJoules"][i])*minimum[2])
        self.assertGreater(rounded_activation_cases,0)
        for alpha, message in ((math.ulp(0.), "activation underflows"), (math.ldexp(1.,-1060), "coefficient underflows")):
            descriptor = copy.deepcopy(self.descriptor)
            for knot in descriptor["request"]["knots"][1:]:
                knot["activation"] = [alpha,0.]
            descriptor["requestSha256"] = sha(descriptor["request"])
            with self.subTest(alpha=alpha),self.assertRaisesRegex(ValueError,message):
                self.verify(descriptor)

    def test_production_import_closure_is_only_independent_standard_library_auditors(self):
        paths = [SCRIPTS/"solver_binding_control_schedule_replay.py",
                 *(SCRIPTS/name for name in audit.REQUIRED_HELPER_FILES)]
        allowed = {"fractions","hashlib","json","math","re","struct",
                   "solver_binding_fold_replay","solver_binding_placement_replay"}
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                modules = [alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
                self.assertTrue(set(modules) <= allowed, (path.name,modules))
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                    self.assertNotIn(node.func.id,("__import__","eval","exec"))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for path in paths:
                (directory/path.name).write_bytes(path.read_bytes())
            (directory/"inputs.json").write_bytes(encoded([self.source,self.fold,self.placement,self.descriptor]))
            program = "import json,sys;sys.path.insert(0,sys.argv[1]);from solver_binding_control_schedule_replay import verify_binding_control_schedule;print(json.dumps(verify_binding_control_schedule(*json.load(open(sys.argv[2])))));assert 'numpy' not in sys.modules"
            result = subprocess.run([sys.executable,"-I","-c",program,str(directory),str(directory/"inputs.json")],
                                    capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIs(json.loads(result.stdout)["verified"],True)


if __name__ == "__main__":
    unittest.main()
