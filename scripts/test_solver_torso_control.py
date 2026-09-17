import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/engine"))
from assembly import compile_assembly, compile_inventory
from cloth_domain import mesh_cloth_domain
from embedded_constraints import build_embedded_constraints, constraint_residuals
from meshing_test import CONSTRUCTION, shirt_pattern
from solver_torso_control import torso_equilibrium_placement


class TorsoControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        cls.instances = [item for item in inventory["instances"] if item["templateId"].startswith(("front_", "back_"))]
        panels = {panel["id"]: panel for panel in cls.pattern["panels"]}
        cls.sources = {item["id"]: {"panel": panels[item["templateId"]], "mesh": mesh_cloth_domain(panels[item["templateId"]], 60, quality_refinement=True)} for item in cls.instances}
        cls.rest = {item["id"]: np.column_stack((np.asarray(cls.sources[item["id"]]["mesh"]["restPositions"]) * [-1 if item["mirrorX"] else 1, 1] / 1000, np.zeros(len(cls.sources[item["id"]]["mesh"]["restPositions"])))) for item in cls.instances}
        operations = [operation for operation in compile_assembly(cls.pattern, inventory)["operations"] if all(member["instanceId"] in cls.rest for member in operation["participants"])]
        registrations = [{"id": operation["id"], "members": [{"instanceId": member["instanceId"], "pathName": member["edgeName"], "startArcMm": 0, "endArcMm": next(path["lengthMm"] for path in cls.sources[member["instanceId"]]["mesh"]["stitchPaths"] if path["name"] == member["edgeName"]), "direction": "forward"} for member in operation["participants"]], "sampleCount": 5, "complianceMPerN": 1e-8} for operation in operations]
        cls.bundle = build_embedded_constraints(cls.sources, registrations)

    def test_exact_source_constraints_and_unchanged_cloth_edges(self):
        placed, metadata = torso_equilibrium_placement(self.pattern, self.instances, self.rest)
        self.assertEqual(metadata["classification"], "diagnostic-only")
        self.assertEqual(len(self.bundle["constraints"]), 25)
        self.assertLess(np.linalg.norm(constraint_residuals(self.bundle, placed), axis=1).max(), 1e-12)
        for identity, source in self.sources.items():
            faces = np.asarray(source["mesh"]["triangles"])
            for edge in range(3):
                first, second = faces[:, edge], faces[:, (edge + 1) % 3]
                np.testing.assert_allclose(np.linalg.norm(placed[identity][first] - placed[identity][second], axis=1), np.linalg.norm(self.rest[identity][first] - self.rest[identity][second], axis=1), rtol=0, atol=1e-15)

    def test_gap_is_rigid_perturbation_and_does_not_mutate_rest(self):
        original = {identity: values.copy() for identity, values in self.rest.items()}
        placed, metadata = torso_equilibrium_placement(self.pattern, self.instances, self.rest, front_gap_mm=5)
        residuals = np.linalg.norm(constraint_residuals(self.bundle, placed), axis=1)
        self.assertAlmostEqual(residuals.max(), 0.005, places=12)
        self.assertEqual(metadata["frontGapMm"], 5)
        for identity in original:
            np.testing.assert_array_equal(self.rest[identity], original[identity])

    def test_rejects_other_inventory_bad_gap_and_incompatible_source(self):
        with self.assertRaises(ValueError):
            torso_equilibrium_placement(self.pattern, self.instances[:-1], self.rest)
        for gap in (True, -1, float("nan"), 101):
            with self.assertRaises(ValueError):
                torso_equilibrium_placement(self.pattern, self.instances, self.rest, gap)
        pattern = copy.deepcopy(self.pattern)
        panel = next(panel for panel in pattern["panels"] if panel["id"] == "front_left")
        edge = next(edge for edge in panel["draft"]["edges"] if edge["name"] == "side")
        panel["points"][edge["start"]][0] += 1
        with self.assertRaisesRegex(ValueError, "translation witness"):
            torso_equilibrium_placement(pattern, self.instances, self.rest)

    def test_source_width_determines_translation(self):
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern(selection={"placketWidthMm": 42})).encode(), CONSTRUCTION)
        instances = [item for item in inventory["instances"] if item["templateId"].startswith(("front_", "back_"))]
        panels = {panel["id"]: panel for panel in pattern["panels"]}
        rest = {item["id"]: np.asarray([[(-1 if item["mirrorX"] else 1) * point[0] / 1000, point[1] / 1000, 0] for point in panels[item["templateId"]]["points"]]) for item in instances}
        placed, metadata = torso_equilibrium_placement(pattern, instances, rest)
        self.assertAlmostEqual(metadata["translationsMeters"]["front_left:shell"][0], 0.021)
        self.assertAlmostEqual(metadata["translationsMeters"]["front_right:shell"][0], -0.021)


if __name__ == "__main__":
    unittest.main()
