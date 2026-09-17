import copy
import unittest

import numpy as np

from cloth_domain import mesh_cloth_domain
from embedded_constraints import build_embedded_constraints, constraint_residuals, project_embedded_constraints, sample_stitch_path, validate_embedded_constraints


def source(identifier, length=100):
    panel = {"id": identifier, "points": [[0, 0], [20, 0], [20, length], [0, length], [0, 0]],
             "draft": {"cutLine": [[-10, -10], [30, -10], [30, length + 10], [-10, length + 10], [-10, -10]],
                       "edges": [{"name": name, "start": index, "end": index + 1, "lengthMm": size} for index, (name, size) in enumerate(
                           (("top", 20), ("right", length), ("bottom", 20), ("left", length)))]}}
    return {"panel": panel, "mesh": mesh_cloth_domain(panel, 20)}


def member(identifier, end=100, direction="forward"):
    return {"instanceId": identifier, "pathName": "right", "startArcMm": 0, "endArcMm": end, "direction": direction}


def registration(members, count=3, compliance=0):
    return [{"id": "sew", "members": members, "sampleCount": count, "complianceMPerN": compliance}]


def state(sources):
    positions, inverse = {}, {}
    for index, (identifier, item) in enumerate(sources.items()):
        rest = np.asarray(item["mesh"]["restPositions"]) / 1000
        positions[identifier] = np.column_stack((rest, np.full(len(rest), index * 0.02)))
        inverse[identifier] = np.full(len(rest), index + 1.0)
    return positions, inverse


class EmbeddedConstraintTests(unittest.TestCase):
    def test_arbitrary_interior_binding_sample_is_not_snapped(self):
        item = source("binding")
        sample = sample_stitch_path(item["mesh"], "right", 37.7)
        np.testing.assert_allclose(sample["restPositionMm"], [20, 37.7], atol=1e-12)
        reconstructed = sum(entry["weight"] * np.asarray(item["mesh"]["restPositions"][entry["vertex"]]) for entry in sample["weights"])
        np.testing.assert_allclose(reconstructed, [20, 37.7], atol=1e-12)
        self.assertGreater(len(sample["weights"]), 1)
        self.assertGreater(min(np.linalg.norm(np.asarray(position) - reconstructed) for position in item["mesh"]["restPositions"]), 0.01)

    def test_reversed_and_gathered_multi_layer_source_intervals(self):
        sources = {"shell": source("shell"), "facing": source("facing"), "gather": source("gather", 200)}
        original = copy.deepcopy(sources)
        bundle = build_embedded_constraints(sources, registration([member("shell"), member("facing", direction="reverse"), member("gather", 200)]))
        self.assertEqual(len(bundle["constraints"]), 6)
        self.assertEqual(bundle["topology"], "independent-star")
        self.assertEqual(bundle["constraints"][0]["sourceSamples"][1]["arcMm"], 100)
        self.assertEqual(bundle["constraints"][1]["sourceSamples"][1]["arcMm"], 0)
        self.assertEqual(bundle["constraints"][3]["sourceSamples"][1]["arcMm"], 100)
        self.assertEqual(bundle["constraints"][-1]["sourceSamples"][1]["arcMm"], 200)
        self.assertEqual(sources, original)
        self.assertTrue(validate_embedded_constraints(sources, bundle)["sourceCorrespondenceAccepted"])

    def test_xpbd_preserves_mass_weighted_center_and_rest_sources(self):
        sources = {"shell": source("shell"), "binding": source("binding")}
        source_snapshot = copy.deepcopy(sources)
        bundle = build_embedded_constraints(sources, registration([member("shell"), member("binding")]))
        positions, inverse = state(sources)
        before = {identifier: array.copy() for identifier, array in positions.items()}
        center = sum(np.sum(array / inverse[identifier][:, None], axis=0) for identifier, array in positions.items())
        result = project_embedded_constraints(bundle, positions, inverse, 1 / 240, iterations=20)
        after_center = sum(np.sum(array / inverse[identifier][:, None], axis=0) for identifier, array in result["positions"].items())
        np.testing.assert_allclose(center, after_center, atol=1e-12)
        self.assertLess(np.max(np.linalg.norm(result["residualsM"], axis=1)), 1e-10)
        for identifier in positions:
            np.testing.assert_array_equal(positions[identifier], before[identifier])
        self.assertEqual(sources, source_snapshot)

    def test_pinned_vertices_and_immovable_residuals(self):
        sources = {"shell": source("shell"), "binding": source("binding")}
        bundle = build_embedded_constraints(sources, registration([member("shell"), member("binding")]))
        positions, inverse = state(sources)
        inverse["shell"][:] = 0
        result = project_embedded_constraints(bundle, positions, inverse, 1 / 240, iterations=10)
        np.testing.assert_array_equal(result["positions"]["shell"], positions["shell"])
        self.assertLess(np.max(np.abs(result["residualsM"])), 1e-10)
        inverse["binding"][:] = 0
        result = project_embedded_constraints(bundle, positions, inverse, 1 / 240)
        self.assertEqual(result["immovableConstraints"], [0, 1, 2])
        self.assertGreater(np.max(np.abs(result["residualsM"])), 0.01)

    def test_compliance_timestep_and_multiplier_equation(self):
        sources = {"shell": source("shell"), "binding": source("binding")}
        bundle = build_embedded_constraints(sources, registration([member("shell"), member("binding")], compliance=1e-5))
        positions, inverse = state(sources)
        timestep = 1 / 240
        result = project_embedded_constraints(bundle, positions, inverse, timestep, iterations=20)
        np.testing.assert_allclose(result["residualsM"] + 1e-5 / timestep ** 2 * result["multipliers"], 0, atol=1e-10)
        shorter = project_embedded_constraints(bundle, positions, inverse, timestep / 2, iterations=20)
        self.assertGreater(np.linalg.norm(shorter["residualsM"]), np.linalg.norm(result["residualsM"]))
        resumed = project_embedded_constraints(bundle, result["positions"], inverse, timestep, multipliers=result["multipliers"])
        for identifier in positions:
            np.testing.assert_allclose(resumed["positions"][identifier], result["positions"][identifier], atol=1e-12)

    def test_sparse_jacobian_matches_finite_difference(self):
        sources = {"shell": source("shell"), "binding": source("binding")}
        bundle = build_embedded_constraints(sources, registration([member("shell"), member("binding")]))
        positions, _ = state(sources)
        term = bundle["constraints"][1]["terms"][0]
        initial = constraint_residuals(bundle, positions)
        positions[term["instanceId"]][term["vertex"], 2] += 1e-7
        difference = (constraint_residuals(bundle, positions) - initial)[1, 2] / 1e-7
        self.assertAlmostEqual(difference, term["coefficient"], places=8)

    def test_mutations_and_invalid_projection_inputs_rejected(self):
        sources = {"shell": source("shell"), "binding": source("binding")}
        bundle = build_embedded_constraints(sources, registration([member("shell"), member("binding")]))
        for mutation in ("weight", "source_arc", "source_hash", "missing_row", "ready"):
            changed = copy.deepcopy(bundle)
            if mutation == "weight":
                changed["constraints"][0]["terms"][0]["coefficient"] += 0.1
            elif mutation == "source_arc":
                changed["constraints"][0]["sourceSamples"][0]["arcMm"] += 1
            elif mutation == "source_hash":
                changed["sourceIdentities"]["shell"]["meshDigest"] = "bad"
            elif mutation == "missing_row":
                changed["constraints"].pop()
            else:
                changed["solverReady"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_embedded_constraints(sources, changed)
        for direction in (None, "backwards"):
            with self.assertRaises(ValueError):
                build_embedded_constraints(sources, registration([member("shell"), member("binding", direction=direction)]))
        positions, inverse = state(sources)
        for timestep in (0, True, float("nan")):
            with self.assertRaises(ValueError):
                project_embedded_constraints(bundle, positions, inverse, timestep)
        inverse["shell"][0] = -1
        with self.assertRaises(ValueError):
            project_embedded_constraints(bundle, positions, inverse, 1 / 240)

    def test_duplicate_physical_registrations_cannot_double_stiffness(self):
        sources = {"shell": source("shell"), "binding": source("binding")}
        original = registration([member("shell"), member("binding")])
        for mutation in ("copy", "member_order", "direction", "sampling"):
            duplicate = copy.deepcopy(original[0])
            duplicate["id"] = "duplicate"
            if mutation == "member_order":
                duplicate["members"].reverse()
            elif mutation == "direction":
                for participant in duplicate["members"]:
                    participant["direction"] = "reverse"
            elif mutation == "sampling":
                duplicate["sampleCount"] += 1
                duplicate["complianceMPerN"] = 1e-5
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "Duplicate physical sewing registration"):
                build_embedded_constraints(sources, original + [duplicate])

    def test_staged_targets_preserve_zero_behavior_and_avoid_initial_impulse(self):
        sources = {"shell": source("shell"), "binding": source("binding")}
        bundle = build_embedded_constraints(sources, registration([member("shell"), member("binding")]))
        positions, inverse = state(sources)
        original = project_embedded_constraints(bundle, positions, inverse, 1 / 240)
        zero = project_embedded_constraints(bundle, positions, inverse, 1 / 240, target_offsets_m=np.zeros((3, 3)))
        for identifier in positions:
            np.testing.assert_array_equal(original["positions"][identifier], zero["positions"][identifier])
        initial = constraint_residuals(bundle, positions)
        result = project_embedded_constraints(bundle, positions, inverse, 1 / 240, target_offsets_m=initial)
        for identifier in positions:
            np.testing.assert_array_equal(result["positions"][identifier], positions[identifier])
        np.testing.assert_array_equal(result["multipliers"], 0)
        np.testing.assert_array_equal(result["targetResidualsM"], 0)
        np.testing.assert_array_equal(result["residualsM"], initial)
        self.assertGreater(np.max(np.abs(result["residualsM"])), 0.01)

    def test_time_varying_targets_preserve_mass_and_source_geometry(self):
        sources = {"shell": source("shell"), "binding": source("binding")}
        source_snapshot = copy.deepcopy(sources)
        bundle = build_embedded_constraints(sources, registration([member("shell"), member("binding")]))
        positions, inverse = state(sources)
        center = sum(np.sum(array / inverse[identifier][:, None], axis=0) for identifier, array in positions.items())
        initial = constraint_residuals(bundle, positions)
        for fraction in (1, 0.9, 0.6, 0.2, 0):
            target = initial * fraction
            target_snapshot = target.copy()
            result = project_embedded_constraints(bundle, positions, inverse, 1 / 240, iterations=10, target_offsets_m=target)
            positions = result["positions"]
            after_center = sum(np.sum(array / inverse[identifier][:, None], axis=0) for identifier, array in positions.items())
            np.testing.assert_allclose(center, after_center, atol=1e-12)
            np.testing.assert_allclose(result["targetResidualsM"], 0, atol=1e-10)
            np.testing.assert_allclose(result["residualsM"], target, atol=1e-10)
            np.testing.assert_array_equal(target, target_snapshot)
        self.assertEqual(sources, source_snapshot)

    def test_malformed_staging_targets_fail_before_projection(self):
        sources = {"shell": source("shell"), "binding": source("binding")}
        bundle = build_embedded_constraints(sources, registration([member("shell"), member("binding")]))
        positions, inverse = state(sources)
        snapshots = {identifier: array.copy() for identifier, array in positions.items()}
        for target in (np.zeros(3), np.zeros((1, 3)), np.full((3, 3), np.nan), np.full((3, 3), np.inf), np.full((3, 3), 101)):
            with self.assertRaisesRegex(ValueError, "staged coupling targets"):
                project_embedded_constraints(bundle, positions, inverse, 1 / 240, target_offsets_m=target)
        for identifier in positions:
            np.testing.assert_array_equal(positions[identifier], snapshots[identifier])


if __name__ == "__main__":
    unittest.main()
