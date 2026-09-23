"""Fresh-source migration regressions, including silently valid wrong faces."""

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

from solver_binding_gripper_remap import build_binding_gripper_remap, validate_binding_gripper_remap
from solver_binding_source import build_binding_source
from solver_gripper_input import bind_material_grippers, mesh_identity


SCRIPTS = Path(__file__).resolve().parent
INSTANCE = "opening_binding_left_left:shell"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rational(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def controls(base):
    source = copy.deepcopy(base)
    anchors = [{"id": identity, "instanceId": INSTANCE, "triangleIndex": 48 + parent,
                "weights": weights, "stiffnessNPerM": 1.}
        for identity, parent, weights in (("wrist", 6, [.5, .25, .25]),
            ("apex", 1, [.25, .5, .25]), ("allowance", 10, [.375, .375, .25]))]
    source["gripperActuation"] = {"profile": "captured-material-grippers-v1", "accepted": False,
        "meshSha256": mesh_identity(source), "anchors": anchors,
        "schedule": {"profile": "material-gripper-target-activation-v1",
            "gripperIds": [anchor["id"] for anchor in anchors],
            "knots": [{"fraction": fraction,
                "targetsMeters": [[index * .01, fraction * .03, -.001] for index in range(3)],
                "activation": [activation] * 3}
                for fraction, activation in ((0., 1.), (.5, 1.), (.75, 1.), (.875, 0.), (1., 0.))]}}
    return source


def add_anchor(source, identity, instance, face, weights):
    recipe = source["gripperActuation"]
    recipe["anchors"].append({"id": identity, "instanceId": instance, "triangleIndex": face,
        "weights": weights, "stiffnessNPerM": 2.})
    recipe["schedule"]["gripperIds"].append(identity)
    for knot in recipe["schedule"]["knots"]:
        knot["targetsMeters"].append([.02, -.03, .04])
        knot["activation"].append(1. if knot["fraction"] < .875 else 0.)


class BindingGripperRemapTests(unittest.TestCase):
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
                env=environment, cwd=root, capture_output=True, text=True, timeout=60)
            if completed.returncode:
                raise AssertionError(completed.stdout + completed.stderr)
        cls.base = json.loads((root / "unit/unit.json").read_bytes())
        cls.refined = build_binding_source(cls.base)
        cls.original = controls(cls.base)
        cls.descriptor = build_binding_gripper_remap(cls.original, cls.refined, subdivisions=64)

    def test_three_point_operators_preserve_exact_coefficients_and_equivalent_edge_support(self):
        anchors = self.descriptor["anchors"]
        self.assertEqual([item["originalLocalTriangleIndex"] for item in anchors], [6, 1, 10])
        self.assertEqual([item["selectedLocalTriangleIndex"] for item in anchors], [27, 7, 38])
        self.assertEqual([item["equivalentLocalTriangleIndices"] for item in anchors], [[27, 28], [7, 9], [38]])
        self.assertEqual([item["selectedCanonicalTriangleIndex"] for item in anchors], [75, 55, 86])
        self.assertEqual([item["numericalAnchor"]["weights"] for item in anchors], [[.5, 0., .5], [0., .5, .5], [.375, .125, .5]])
        for item in anchors:
            audit = item["coefficientAudit"]
            self.assertEqual(rational(audit["pullback"]["residualLInfinity"]), 0)
            self.assertTrue(all(rational(row["binary64MinusExact"]) == 0 for row in audit["weightRoundingErrors"]))
            self.assertEqual({candidate["sourceParentTriangle"] for candidate in item["candidates"]}, {item["originalLocalTriangleIndex"]})
        for key in ("accepted", "solverReady", "executable"):
            self.assertIs(self.descriptor[key], False)

    def test_exact_pullback_does_not_hide_stored_coordinate_residual(self):
        original, refined = self.original, self.refined
        for item in self.descriptor["anchors"]:
            points = []
            for source, anchor in ((original, item["originalAnchor"]), (refined, item["numericalAnchor"])):
                face = source["triangles"][3 * anchor["triangleIndex"]:3 * anchor["triangleIndex"] + 3]
                points.append([sum((Fraction(weight) * Fraction(source["restMeters"][vertex][axis])
                    for vertex, weight in zip(face, anchor["weights"])), Fraction()) for axis in range(3)])
            self.assertEqual([rational(value) for value in item["storedCoordinateResidualMeters"]],
                             [b - a for a, b in zip(*points)])
        self.assertNotEqual([rational(value) for value in self.descriptor["anchors"][2]["storedCoordinateResidualMeters"]], [0, 0, 0])

    def test_only_mesh_identity_triangle_and_weights_change_in_entire_control_recipe(self):
        old, new = self.descriptor["originalRecipe"], copy.deepcopy(self.descriptor["numericalRecipe"])
        new["meshSha256"] = old["meshSha256"]
        for old_anchor, new_anchor in zip(old["anchors"], new["anchors"]):
            for key in ("triangleIndex", "weights"):
                new_anchor[key] = copy.deepcopy(old_anchor[key])
        self.assertEqual(encoded(old), encoded(new))
        self.assertEqual(self.descriptor["originalRecipeSha256"], hashlib.sha256(encoded(old)).hexdigest())
        _, schedule, _ = bind_material_grippers({**self.refined, "gripperActuation": self.descriptor["numericalRecipe"]}, 64)
        self.assertEqual(schedule.parameters(.875)[1].tolist(), [0., 0., 0.])
        self.assertEqual(schedule.parameters(1.)[1].tolist(), [0., 0., 0.])

    def test_unchanged_instances_rebuild_global_indices_without_uniform_shift(self):
        source = copy.deepcopy(self.original)
        add_anchor(source, "sleeve", "sleeve_left:shell", 200, [.5, .25, .25])
        add_anchor(source, "cuff", "cuff_left:shell", 3, [.2, .3, .5])
        descriptor = build_binding_gripper_remap(source, self.refined, subdivisions=64)
        self.assertEqual([item["selectedCanonicalTriangleIndex"] for item in descriptor["anchors"][-2:]], [232, 3])
        for item in descriptor["anchors"][-2:]:
            self.assertEqual(item["migrationKind"], "unchanged-instance")
            self.assertEqual(item["originalAnchor"]["weights"], item["numericalAnchor"]["weights"])
            self.assertEqual([rational(value) for value in item["storedCoordinateResidualMeters"]], [0, 0, 0])
        wrong = copy.deepcopy(descriptor)
        wrong["numericalRecipe"]["anchors"][3]["triangleIndex"] = 200
        # This wrong material location is still a syntactically valid sleeve face.
        bind_material_grippers({**self.refined, "gripperActuation": wrong["numericalRecipe"]}, 64)
        with self.assertRaises(ValueError):
            validate_binding_gripper_remap(source, self.refined, wrong, subdivisions=64)

    def test_nonunit_raw_source_sum_is_preserved_without_normalizing(self):
        source = copy.deepcopy(self.original)
        source["gripperActuation"]["anchors"][2]["weights"] = [.375, .375, math.nextafter(.25, 1.)]
        descriptor = build_binding_gripper_remap(source, self.refined, subdivisions=64)
        item = descriptor["anchors"][2]
        expected = sum(map(Fraction, source["gripperActuation"]["anchors"][2]["weights"]), Fraction())
        self.assertNotEqual(expected, 1)
        self.assertEqual(rational(item["originalWeightSum"]), expected)
        self.assertEqual(rational(item["coefficientAudit"]["weightSums"]["source"]), expected)
        self.assertEqual(encoded(descriptor["originalRecipe"]), encoded(source["gripperActuation"]))

    def test_nonrepresentable_child_weight_records_nonzero_conversion_error(self):
        source = copy.deepcopy(self.original)
        source["gripperActuation"]["anchors"][2]["weights"] = [.1, .2, .7]
        descriptor = build_binding_gripper_remap(source, self.refined, subdivisions=64)
        item = descriptor["anchors"][2]
        self.assertEqual(item["selectedLocalTriangleIndex"], 40)
        self.assertEqual(item["numericalAnchor"]["weights"], [.4, .3999999999999999, .2])
        audit = item["coefficientAudit"]
        self.assertEqual(rational(audit["pullback"]["residualLInfinity"]), Fraction(1, 2**55))
        self.assertEqual(rational(audit["weightSums"]["binary64DerivedMinusSource"]), -Fraction(1, 2**55))
        self.assertTrue(any(rational(row["binary64MinusExact"]) for row in audit["weightRoundingErrors"]))
        self.assertEqual(source["gripperActuation"]["anchors"][2]["weights"], [.1, .2, .7])

    def test_strict_descriptor_rederivation_rejects_each_witness_and_control_tamper(self):
        attacks = [lambda x: x.update(accepted=True), lambda x: x.update(subdivisions=32),
            lambda x: x.update(originalSourceSha256="0" * 64),
            lambda x: x["numericalRecipe"]["schedule"]["knots"][-1].update(activation=[1., 1., 1.]),
            lambda x: x["anchors"][0].update(equivalentLocalTriangleIndices=[27]),
            lambda x: x["anchors"][2]["coefficientAudit"]["pullback"]["residualLInfinity"].update(numerator="1"),
            lambda x: x["anchors"][2]["storedCoordinateResidualMeters"][0].update(numerator="0"),
            lambda x: x["numericalRecipe"]["anchors"][0].update(stiffnessNPerM=2.)]
        for attack in attacks:
            changed = copy.deepcopy(self.descriptor)
            attack(changed)
            with self.subTest(attack=attacks.index(attack)), self.assertRaises(ValueError):
                validate_binding_gripper_remap(self.original, self.refined, changed, subdivisions=64)

    def test_no_source_mutation_and_no_returned_aliases(self):
        before = encoded([self.original, self.refined])
        descriptor = validate_binding_gripper_remap(self.original, self.refined, self.descriptor, subdivisions=64)
        descriptor["originalRecipe"]["anchors"][0]["weights"][0] = 0.
        descriptor["numericalRecipe"]["schedule"]["knots"].clear()
        descriptor["anchors"][0]["originalAnchor"]["weights"][0] = 0.
        self.assertEqual(encoded([self.original, self.refined]), before)
        self.assertEqual(len(self.descriptor["numericalRecipe"]["schedule"]["knots"]), 5)

    def test_subdivision_types_schedule_boundaries_and_gripper_budget(self):
        for value in (True, 64., 0, 3, -1, 8192):
            with self.subTest(subdivisions=value), self.assertRaises(ValueError):
                build_binding_gripper_remap(self.original, self.refined, subdivisions=value)
        with self.assertRaises(ValueError):
            build_binding_gripper_remap(self.original, self.refined, subdivisions=4)
        source = copy.deepcopy(self.original)
        for index in range(13):
            add_anchor(source, f"extra-{index}", "cuff_left:shell", 3, [.25, .25, .5])
        self.assertEqual(len(build_binding_gripper_remap(source, self.refined, subdivisions=64)["anchors"]), 16)
        add_anchor(source, "too-many", "cuff_left:shell", 3, [.25, .25, .5])
        with self.assertRaises(ValueError):
            build_binding_gripper_remap(source, self.refined, subdivisions=64)

    def test_malformed_recipe_and_raw_json_reject_before_migration(self):
        attacks = [lambda x: x["gripperActuation"].update(accepted=0),
            lambda x: x["gripperActuation"].update(meshSha256="0" * 64),
            lambda x: x["gripperActuation"]["anchors"][0].update(triangleIndex=True),
            lambda x: x["gripperActuation"]["anchors"][0].update(weights=[True, 0., 0.]),
            lambda x: x["gripperActuation"]["anchors"][0].update(instanceId="sleeve_left:shell"),
            lambda x: x["gripperActuation"]["anchors"][0].update(weights=[-1e-18, .5, .5]),
            lambda x: x["gripperActuation"]["schedule"]["knots"][1].update(fraction=.3),
            lambda x: x.update(extra=tuple()), lambda x: x.update(extra=float("nan"))]
        for attack in attacks:
            source = copy.deepcopy(self.original)
            attack(source)
            with self.subTest(attack=attacks.index(attack)), self.assertRaises(ValueError):
                build_binding_gripper_remap(source, self.refined, subdivisions=64)

    def test_bare_core_and_exact_original_base_are_required(self):
        for key in ("placedMeters", "gripperActuation", "sewingActuation", "foldActuation", "assemblySchedule", "bindingRefinedDiagnostic"):
            refined = {**self.refined, key: None}
            with self.subTest(control=key), self.assertRaises(ValueError):
                build_binding_gripper_remap(self.original, refined, subdivisions=64)
        source = copy.deepcopy(self.original)
        source["phaseConstraintRows"] = {}
        with self.assertRaises(ValueError):
            build_binding_gripper_remap(source, self.refined, subdivisions=64)
        with self.assertRaises(ValueError):
            build_binding_gripper_remap(self.refined, self.refined, subdivisions=64)


if __name__ == "__main__":
    unittest.main()
