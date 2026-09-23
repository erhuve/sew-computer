"""Captured source identity, sparse anchor, frame and control admission tests."""

import copy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from solver_sewing_input import (PROFILE, MAX_SOURCE_BYTES,
                                bind_sewing_activation, derive_sewing_row_ids,
                                sewing_source_identity)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fixture(mode="vector"):
    source = {"restMeters": [[0., 0., 0.], [.03125, 0., 0.], [0., .03125, 0.],
                            [0., 0., 0.], [.03125, 0., 0.], [0., .03125, 0.]],
              "triangles": [0, 1, 2, 3, 4, 5], "instanceOffsets": {"a:shell": 0, "b:facing": 3},
              "embeddedConstraints": {"constraints": []}, "syntheticNote": "captured JSON test only"}
    for vertex, fraction in ((0, 0.), (1, 1.)):
        source["embeddedConstraints"]["constraints"].append({
            "registrationId": "attachment", "memberIndex": 1, "fraction": fraction,
            "complianceMPerN": 1e-8,
            "terms": [{"instanceId": identity, "vertex": vertex, "coefficient": coefficient}
                      for identity, coefficient in (("a:shell", 1.), ("b:facing", -1.))],
            "sourceSamples": [{"instanceId": identity, "pathName": "edge", "weights": [{"vertex": vertex, "weight": 1.}]}
                              for identity in ("a:shell", "b:facing")]})
    return attach_recipe(source, mode)


def attach_recipe(source, mode="vector", activation=None):
    row_ids = derive_sewing_row_ids(source)
    count = len(row_ids)
    if mode == "normal-offset":
        faces = np.asarray(source["triangles"]).reshape((-1, 3)).tolist()
        source["sewingFrames"] = {"faces": [faces[1].copy() for _ in row_ids], "sides": [-1] * count,
            "bindings": [{"rowId": row_id, "instanceId": "b:facing", "triangleIndex": 1, "side": -1}
                         for row_id in row_ids]}
    source["sewingActuation"] = {"profile": PROFILE, "accepted": False, "sourceSha256": sewing_source_identity(source),
        "mode": mode, "initialTargetsMeters": [[0., 0., -.002] for _ in row_ids] if mode == "vector" else [.002] * count,
        "finalTargetsMeters": [[0., 0., -.0002] for _ in row_ids] if mode == "vector" else [.0002] * count,
        "schedule": {"profile": "sewing-row-activation-v1", "rowIds": list(row_ids),
                     "knots": [{"fraction": fraction, "activation": values} for fraction, values in
                               (activation or [(0., [0.] * count), (.5, [.25] * count), (1., [1.] * count)])]}}
    return source


def rehash(source):
    source["sewingActuation"]["sourceSha256"] = sewing_source_identity(source)
    return source


class SewingInputTests(unittest.TestCase):
    def test_full_source_digest_and_ordered_readable_identity(self):
        source = fixture()
        original = encoded(source)
        controls, binding = bind_sewing_activation(source, 4, sewing_mode="vector")
        payload = {key: value for key, value in source.items() if key != "sewingActuation"}
        self.assertEqual(binding["sourceSha256"], hashlib.sha256(encoded(payload)).hexdigest())
        expected_ids = tuple("row:" + hashlib.sha256(encoded(["attachment", 1, fraction, 1])).hexdigest()
                             for fraction in (0, 1))
        self.assertEqual(controls.row_ids, expected_ids)
        self.assertEqual(binding["rowIds"], list(expected_ids))
        self.assertEqual(controls.rows, [{0: 1., 3: -1.}, {1: 1., 4: -1.}])
        self.assertEqual(controls.compliance, 1e-8)
        self.assertEqual(controls.mode, "vector")
        self.assertEqual([row["rowIndex"] for row in binding["rowBindings"]], [0, 1])
        self.assertEqual([row["fractionNumerator"] for row in binding["rowBindings"]], [0, 1])
        self.assertEqual(binding["sourceBundleSha256"], hashlib.sha256(encoded(source["embeddedConstraints"])).hexdigest())
        self.assertEqual(original, encoded(source))
        self.assertFalse(binding["accepted"])
        self.assertIn("no pattern-source proof", binding["sourceBindingScope"])
        self.assertIn("unexecuted", binding["constructionStatus"])
        self.assertIsNone(binding["cuffSourceBinding"])

    def test_all_three_explicit_modes_and_stateless_original_fraction_samples(self):
        for mode in ("vector", "distance", "normal-offset"):
            with self.subTest(mode=mode):
                source = fixture(mode)
                controls, binding = bind_sewing_activation(source, 4, sewing_mode=mode)
                for fraction, expected in ((1, 1), (0, 0), (Fraction(3, 4), .625), (Fraction(1, 4), .125)):
                    np.testing.assert_array_equal(controls.parameters(fraction), [expected] * 2)
                    np.testing.assert_array_equal(controls.activation_schedule.parameters(fraction), [expected] * 2)
                expected_shape = (2, 3) if mode == "vector" else (2,)
                self.assertEqual(controls.initial_targets.shape, expected_shape)
                self.assertEqual(controls.final_targets.shape, expected_shape)
                self.assertEqual(binding["mode"], mode)
                self.assertEqual(binding["sourceFrameMetadataUsed"], mode == "normal-offset")
                if mode == "normal-offset":
                    np.testing.assert_array_equal(controls.frame_faces, [[3, 4, 5]] * 2)
                    np.testing.assert_array_equal(controls.sides, [-1, -1])
                    self.assertEqual(binding["frameBindings"][0]["instanceLocalVertices"], [0, 1, 2])
                    self.assertIn("not textile right-side", binding["scope"])
                else:
                    self.assertIsNone(controls.frame_faces)
                    self.assertIsNone(controls.sides)

    def test_input_and_returned_values_do_not_mutate_controls_or_other_results(self):
        source = fixture("normal-offset")
        controls, binding = bind_sewing_activation(source, 4, sewing_mode="normal-offset")
        source["embeddedConstraints"]["constraints"][0]["terms"][0]["coefficient"] = .5
        source["sewingActuation"]["initialTargetsMeters"][0] = .09
        source["sewingActuation"]["schedule"]["knots"][-1]["activation"][0] = .5
        source["sewingFrames"]["faces"][0][0] = 0
        binding["rowIds"][0] = "altered"
        binding["frameBindings"][0]["canonicalVertices"][0] = 0
        controls.initial_targets[:] = 90.
        controls.final_targets[:] = 90.
        controls.frame_faces[:] = 0
        controls.sides[:] = 0
        controls.rows[0][0] = 0
        controls.parameters(1)[:] = 0
        controls.schedule_recipe["knots"][-1]["activation"][0] = 0
        np.testing.assert_array_equal(controls.initial_targets, [.002] * 2)
        np.testing.assert_array_equal(controls.final_targets, [.0002] * 2)
        np.testing.assert_array_equal(controls.frame_faces, [[3, 4, 5]] * 2)
        np.testing.assert_array_equal(controls.sides, [-1] * 2)
        np.testing.assert_array_equal(controls.parameters(1), [1.] * 2)
        self.assertEqual(controls.rows[0], {0: 1., 3: -1.})
        with self.assertRaises(AttributeError):
            controls._compliance = .1
        with self.assertRaises(AttributeError):
            del controls._row_ids
        with self.assertRaises(ValueError):
            controls._initial.flags.writeable = True

    def test_source_mutation_rejects_without_rewriting_bound_recipe(self):
        mutations = (
            lambda source: source["restMeters"][0].__setitem__(0, .001),
            lambda source: source.__setitem__("syntheticNote", "altered metadata"),
            lambda source: source.__setitem__("unrelatedMetadata", {"material": "different"}),
            lambda source: source["embeddedConstraints"].__setitem__("note", "changed full bundle"),
            lambda source: source["embeddedConstraints"]["constraints"].reverse())
        for mutation in mutations:
            source = fixture()
            mutation(source)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bind_sewing_activation(source, 4, sewing_mode="vector")

    def test_recipe_field_mode_target_schedule_and_flag_tampering_rejects(self):
        mutations = (
            lambda recipe: recipe.__setitem__("accepted", True),
            lambda recipe: recipe.__setitem__("accepted", 0),
            lambda recipe: recipe.__setitem__("profile", "different"),
            lambda recipe: recipe.__setitem__("mode", "distance"),
            lambda recipe: recipe.__setitem__("sourceSha256", "0" * 64),
            lambda recipe: recipe.__setitem__("extra", 0),
            lambda recipe: recipe.pop("initialTargetsMeters"),
            lambda recipe: recipe.__setitem__("initialTargetsMeters", [0., 0.]),
            lambda recipe: recipe["finalTargetsMeters"][0].__setitem__(0, True),
            lambda recipe: recipe["finalTargetsMeters"][0].__setitem__(0, 101.),
            lambda recipe: recipe["schedule"]["rowIds"].reverse(),
            lambda recipe: recipe["schedule"]["knots"][-1]["activation"].pop())
        for mutation in mutations:
            source = fixture()
            mutation(source["sewingActuation"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bind_sewing_activation(source, 4, sewing_mode="vector")
        for value in (0., -1., True, None, [1.], 101.):
            source = fixture("distance")
            source["sewingActuation"]["initialTargetsMeters"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                bind_sewing_activation(source, 4, sewing_mode="distance")

    def test_raw_json_resource_and_nonfinite_values_reject_before_binding(self):
        invalid = [np.ones(3), {1: "nonstring key"}, {"value": np.float64(.5)}, {"value": float("nan")},
                   {"value": float("inf")}, {"value": 1 << 64}, {"value": "\ud800"},
                   {"value": "x" * (MAX_SOURCE_BYTES + 1)}]
        nested = 0
        for _ in range(42):
            nested = [nested]
        invalid.append({"deep": nested})
        invalid.append({"many": [None] * 1000001})
        for source in invalid:
            with self.subTest(kind=type(source)), self.assertRaises(ValueError):
                sewing_source_identity(source)
        recursive = {}
        recursive["self"] = recursive
        with self.assertRaises(ValueError):
            sewing_source_identity(recursive)

    def test_raw_mesh_and_instance_admission_after_valid_digest_rejects(self):
        mutations = (
            lambda source: source["restMeters"][0].__setitem__(0, True),
            lambda source: source["restMeters"][0].__setitem__(0, 101.),
            lambda source: source["triangles"].__setitem__(0, False),
            lambda source: source["triangles"].__setitem__(0, 1),
            lambda source: source["triangles"].__setitem__(0, 3),
            lambda source: source["triangles"].extend([0, 1, 2]),
            lambda source: source["instanceOffsets"].__setitem__("a:shell", True),
            lambda source: source["instanceOffsets"].__setitem__("b:facing", 0),
            lambda source: source["instanceOffsets"].__setitem__("b:facing", 2),
            lambda source: source["instanceOffsets"].__setitem__("a:shell", 1),
            lambda source: source["restMeters"].append([0., 0., 0.]))
        for mutation in mutations:
            source = fixture()
            mutation(source)
            rehash(source)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bind_sewing_activation(source, 4, sewing_mode="vector")

    def test_rows_require_original_order_distinct_identity_and_uniform_compliance(self):
        mutations = (
            lambda rows: rows.reverse(),
            lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
            lambda rows: rows[0].__setitem__("memberIndex", True),
            lambda rows: rows[0].__setitem__("memberIndex", 0),
            lambda rows: rows[0].__setitem__("fraction", True),
            lambda rows: rows[0].__setitem__("fraction", -1.),
            lambda rows: rows[0].__setitem__("registrationId", ""),
            lambda rows: rows[0].__setitem__("complianceMPerN", 0.),
            lambda rows: rows[0].__setitem__("complianceMPerN", True),
            lambda rows: rows[0].__setitem__("complianceMPerN", 1001.),
            lambda rows: rows[1].__setitem__("complianceMPerN", 1e-7))
        for mutation in mutations:
            source = fixture()
            mutation(source["embeddedConstraints"]["constraints"])
            rehash(source)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bind_sewing_activation(source, 4, sewing_mode="vector")
        source = fixture()
        middle = copy.deepcopy(source["embeddedConstraints"]["constraints"][0])
        middle["registrationId"] = "interleaved"
        source["embeddedConstraints"]["constraints"].insert(1, middle)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            derive_sewing_row_ids(source)

    def test_sparse_anchor_coefficients_instances_and_samples_reject_corruption(self):
        mutations = (
            lambda row: row["terms"].append(copy.deepcopy(row["terms"][0])),
            lambda row: row["terms"][0].__setitem__("vertex", True),
            lambda row: row["terms"][0].__setitem__("vertex", 3),
            lambda row: row["terms"][0].__setitem__("instanceId", "unknown"),
            lambda row: row["terms"][0].__setitem__("coefficient", True),
            lambda row: row["terms"][0].__setitem__("coefficient", 0.),
            lambda row: row["terms"][0].__setitem__("coefficient", .5),
            lambda row: row["terms"][0].__setitem__("coefficient", -1.),
            lambda row: row["terms"][1].update(instanceId="a:shell", vertex=1),
            lambda row: row["sourceSamples"].reverse(),
            lambda row: row["sourceSamples"][0]["weights"][0].__setitem__("weight", .5),
            lambda row: row["sourceSamples"][0]["weights"].append({"vertex": 0, "weight": 0.}))
        for mutation in mutations:
            source = fixture()
            mutation(source["embeddedConstraints"]["constraints"][0])
            rehash(source)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bind_sewing_activation(source, 4, sewing_mode="vector")

    def test_fraction_identity_is_exact_binary_value_and_terms_keep_source_order(self):
        source = fixture()
        row = source["embeddedConstraints"]["constraints"][0]
        row["fraction"] = .1
        row["terms"].reverse()
        attach_recipe(source)
        controls, manifest = bind_sewing_activation(source, 4, sewing_mode="vector")
        fraction = Fraction(.1)
        self.assertEqual(manifest["rowBindings"][0]["fractionNumerator"], fraction.numerator)
        self.assertEqual(manifest["rowBindings"][0]["fractionDenominator"], fraction.denominator)
        self.assertEqual(list(controls.rows[0]), [3, 0])
        self.assertNotEqual(fraction, Fraction(1, 10))

    def test_normalized_anchor_support_must_fit_one_material_triangle(self):
        source = fixture()
        source["restMeters"].insert(3, [.03125, .03125, 0.])
        source["instanceOffsets"]["b:facing"] = 4
        source["triangles"] = [0, 1, 2, 0, 1, 3, 4, 5, 6]
        row = source["embeddedConstraints"]["constraints"][0]
        row["terms"] = [{"instanceId": "a:shell", "vertex": vertex, "coefficient": .5} for vertex in (2, 3)]
        row["terms"].append({"instanceId": "b:facing", "vertex": 0, "coefficient": -1.})
        row["sourceSamples"][0]["weights"] = [{"vertex": vertex, "weight": .5} for vertex in (2, 3)]
        rehash(source)
        with self.assertRaisesRegex(ValueError, "support must lie on a canonical triangle"):
            bind_sewing_activation(source, 4, sewing_mode="vector")

    def test_normal_frame_cannot_select_another_triangle_that_misses_negative_support(self):
        source = fixture("normal-offset")
        source["restMeters"].append([.03125, .03125, 0.])
        source["triangles"].extend([3, 5, 6])
        source["sewingFrames"]["faces"][1] = [3, 5, 6]
        source["sewingFrames"]["bindings"][1]["triangleIndex"] = 2
        rehash(source)
        with self.assertRaisesRegex(ValueError, "negative material anchor support"):
            bind_sewing_activation(source, 4, sewing_mode="normal-offset")

    def test_normal_frame_requires_complete_explicit_binding_support_side_and_winding(self):
        mutations = (
            lambda source: source.pop("sewingFrames"),
            lambda source: source["sewingFrames"].pop("bindings"),
            lambda source: source["sewingFrames"]["faces"][0].reverse(),
            lambda source: source["sewingFrames"]["faces"][0].__setitem__(0, True),
            lambda source: source["sewingFrames"]["sides"].__setitem__(0, True),
            lambda source: source["sewingFrames"]["bindings"][0].__setitem__("side", True),
            lambda source: source["sewingFrames"]["bindings"][0].__setitem__("side", 1),
            lambda source: source["sewingFrames"]["bindings"][0].__setitem__("triangleIndex", True),
            lambda source: source["sewingFrames"]["bindings"][0].__setitem__("triangleIndex", 0),
            lambda source: source["sewingFrames"]["bindings"][0].__setitem__("instanceId", "a:shell"),
            lambda source: source["sewingFrames"]["bindings"].reverse())
        for mutation in mutations:
            source = fixture("normal-offset")
            mutation(source)
            rehash(source)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bind_sewing_activation(source, 4, sewing_mode="normal-offset")

    def test_non_normal_mode_preserves_but_does_not_use_frame_metadata(self):
        source = fixture()
        source["sewingFrames"] = {"unconsumed": "retained explicit metadata"}
        rehash(source)
        controls, manifest = bind_sewing_activation(source, 4, sewing_mode="vector")
        self.assertFalse(manifest["sourceFrameMetadataUsed"])
        self.assertEqual(manifest["frameBindings"], [])
        self.assertIsNone(controls.frame_faces)
        source["sewingFrames"]["unconsumed"] = "changed"
        with self.assertRaises(ValueError):
            bind_sewing_activation(source, 4, sewing_mode="vector")

    def test_cuff_metadata_cannot_downgrade_to_generic_profile(self):
        for field in ("sourcePattern", "sourceConstruction", "sourceInventory", "sourceAssembly", "phasePlan",
                      "phaseConstraintRows", "selectedOperationIds", "excludedOperationIds"):
            for profile in (None, "synthetic", "source-cuff-construction-unit-v0"):
                source = fixture()
                source[field] = {}
                if profile is not None:
                    source["profile"] = profile
                rehash(source)
                with self.subTest(field=field, profile=profile), self.assertRaisesRegex(ValueError, "downgrade"):
                    bind_sewing_activation(source, 4, sewing_mode="vector")
        source = fixture()
        source["provenance"] = {"phasePlanSha256": "0" * 64}
        rehash(source)
        with self.assertRaisesRegex(ValueError, "downgrade"):
            bind_sewing_activation(source, 4, sewing_mode="vector")


class CuffSewingInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        scripts = Path(__file__).resolve().parent
        environment = {"PATH": os.environ.get("PATH", ""), "HOME": str(directory), "LANG": "C.UTF-8",
                       "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

        def command(script, *arguments):
            result = subprocess.run([sys.executable, str(scripts / script), *map(str, arguments)],
                                    capture_output=True, text=True, env=environment, timeout=60)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)

        command("prepare-cuff-source.py", "--output", directory / "parent")
        cls.units = {}
        for side in ("left", "right"):
            command("prepare-cuff-construction.py", "--source-canonical", directory / "parent/canonical.json",
                    "--output", directory / side, "--side", side,
                    "--attachment-policy", "inner-facing-first-outer-shell-last-v1")
            cls.units[side] = json.loads((directory / side / "unit.json").read_bytes())

    def test_both_source_cuffs_bind_all_forty_rows_without_phase_promotion(self):
        for side, unit in self.units.items():
            with self.subTest(side=side):
                source = attach_recipe(copy.deepcopy(unit), "distance")
                original = encoded(source)
                controls, binding = bind_sewing_activation(source, 4, sewing_mode="distance")
                self.assertEqual(len(controls.row_ids), 40)
                self.assertEqual(len(controls.rows), 40)
                self.assertEqual(len(binding["cuffSourceBinding"]["rowGroups"]), 8)
                self.assertFalse(binding["accepted"])
                self.assertFalse(binding["cuffSourceBinding"]["accepted"])
                self.assertFalse(binding["cuffSourceBinding"]["solverReady"])
                self.assertIn("unexecuted", binding["constructionStatus"])
                self.assertNotIn("completed", binding)
                self.assertEqual(original, encoded(source))

    def test_cuff_five_sample_selectors_cannot_split_activation_even_with_valid_hash(self):
        source = attach_recipe(copy.deepcopy(self.units["left"]), "distance")
        source["sewingActuation"]["schedule"]["knots"][1]["activation"][1] = .5
        with self.assertRaisesRegex(ValueError, "five sample rows must share activation"):
            bind_sewing_activation(source, 4, sewing_mode="distance")

    def test_strong_cuff_validator_is_mandatory_and_rejects_rehashed_source_corruption(self):
        source = attach_recipe(copy.deepcopy(self.units["left"]), "distance")
        with patch("solver_cuff_source_binding.validate_cuff_source_binding", side_effect=ValueError("required verifier")) as verify:
            with self.assertRaisesRegex(ValueError, "required verifier"):
                bind_sewing_activation(source, 4, sewing_mode="distance")
            verify.assert_called_once_with(source)
        source["sourceAssembly"]["operations"][0]["dependsOn"].append("invented-completion")
        rehash(source)
        with self.assertRaises(ValueError):
            bind_sewing_activation(source, 4, sewing_mode="distance")

    def test_cuff_normal_frames_report_unique_source_template_triangles(self):
        source = attach_recipe(copy.deepcopy(self.units["left"]), "distance")
        faces = np.asarray(source["triangles"]).reshape((-1, 3)).tolist()
        declarations, frame_faces = [], []
        for row_id, row in zip(source["sewingActuation"]["schedule"]["rowIds"], source["embeddedConstraints"]["constraints"]):
            negative = [term for term in row["terms"] if term["coefficient"] < 0]
            identity = negative[0]["instanceId"]
            support = {source["instanceOffsets"][identity] + term["vertex"] for term in negative}
            # The test author explicitly declares this selected canonical frame;
            # production binding must check the declaration and never select one.
            index = next(index for index, face in enumerate(faces) if support.issubset(face))
            declarations.append({"rowId": row_id, "instanceId": identity, "triangleIndex": index, "side": 1})
            frame_faces.append(faces[index].copy())
        source["sewingFrames"] = {"faces": frame_faces, "sides": [1] * 40, "bindings": declarations}
        source["sewingActuation"]["mode"] = "normal-offset"
        rehash(source)
        controls, manifest = bind_sewing_activation(source, 4, sewing_mode="normal-offset")
        self.assertEqual(len(manifest["frameBindings"]), 40)
        np.testing.assert_array_equal(controls.frame_faces, frame_faces)
        for binding in manifest["frameBindings"]:
            self.assertEqual(source["sourceTemplates"][binding["templateId"]]["triangles"][binding["sourceTriangleIndex"]],
                             binding["instanceLocalVertices"])


if __name__ == "__main__":
    unittest.main()
