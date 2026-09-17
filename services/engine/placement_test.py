import json
import math
import copy
import unittest

from assembly import compile_inventory
from meshing_test import CONSTRUCTION, shirt_pattern
from placement import place_rest_positions, shirt_placement


class RigidPlacementTests(unittest.TestCase):
    def test_all_physical_frames_are_proper_rigid_transforms(self):
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        placement = shirt_placement(pattern, inventory["instances"])
        self.assertEqual({item["instanceId"] for item in placement["frames"]}, {item["id"] for item in inventory["instances"]})
        panels = {item["id"]: item for item in pattern["panels"]}
        instances = {item["id"]: item for item in inventory["instances"]}
        for frame in placement["frames"]:
            basis = frame["basis"]
            for first in range(3):
                for second in range(3):
                    self.assertAlmostEqual(sum(basis[first][axis] * basis[second][axis] for axis in range(3)), 1 if first == second else 0)
            cross = [basis[0][1] * basis[1][2] - basis[0][2] * basis[1][1], basis[0][2] * basis[1][0] - basis[0][0] * basis[1][2], basis[0][0] * basis[1][1] - basis[0][1] * basis[1][0]]
            self.assertAlmostEqual(sum(cross[axis] * basis[2][axis] for axis in range(3)), 1)
            panel = panels[frame["instanceId"].split(":")[0]]
            hand = -1 if instances[frame["instanceId"]]["mirrorX"] else 1
            rest = [[hand * point[0], point[1], 0] for point in panel["points"]]
            placed = place_rest_positions(rest, frame)
            for index in range(len(rest) - 1):
                self.assertAlmostEqual(math.dist(rest[index], rest[index + 1]), math.dist(placed[index], placed[index + 1]), places=8)
            self.assertEqual(placement["classification"], "unvalidated-placement")
            if frame["instanceId"].startswith("front_"):
                self.assertGreater(frame["basis"][2][1], 0)

    def test_bindings_follow_source_edges_with_opposite_spacing(self):
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        original = copy.deepcopy(pattern)
        placement = shirt_placement(pattern, inventory["instances"])
        self.assertEqual(placement["recipe"], "sew-shirt-rigid-staging/2")
        panels = {panel["id"]: panel for panel in pattern["panels"]}
        instances = {instance["templateId"]: instance for instance in inventory["instances"] if instance["role"] == "shell"}
        frames = {frame["instanceId"]: frame for frame in placement["frames"]}
        for side in ("left", "right"):
            binding_positions = []
            sleeve = instances[f"sleeve_{side}"]
            sleeve_frame = frames[sleeve["id"]]
            sleeve_panel = panels[sleeve["templateId"]]
            sleeve_hand = -1 if sleeve["mirrorX"] else 1
            for opening, spacing in (("left", -5), ("right", 5)):
                binding = instances[f"opening_binding_{side}_{opening}"]
                binding_panel = panels[binding["templateId"]]
                hand = -1 if binding["mirrorX"] else 1
                binding_edge = next(edge for edge in binding_panel["draft"]["edges"] if edge["name"] == "right")
                sleeve_edge = next(edge for edge in sleeve_panel["draft"]["edges"] if edge["name"] == f"opening_{opening}")
                placed = place_rest_positions([[hand * point[0], point[1], 0] for point in binding_panel["points"]], frames[binding["id"]])
                binding_positions.append(placed)
                for endpoint in ("start", "end"):
                    source = sleeve_panel["points"][sleeve_edge[endpoint]]
                    target = place_rest_positions([[sleeve_hand * source[0], source[1], 0]], sleeve_frame)[0]
                    actual = placed[binding_edge[endpoint]]
                    for axis in range(3):
                        self.assertAlmostEqual(actual[axis] - target[axis], spacing * sleeve_frame["basis"][2][axis], delta=1e-6)
            self.assertGreaterEqual(min(math.dist(first, second) for first in binding_positions[0] for second in binding_positions[1]), 9.999999)
        self.assertEqual(pattern, original)

    def test_binding_resolution_is_independent_of_inventory_order(self):
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        first = shirt_placement(pattern, inventory["instances"])
        second = shirt_placement(pattern, list(reversed(inventory["instances"])))
        self.assertEqual({frame["instanceId"]: frame for frame in first["frames"]}, {frame["instanceId"]: frame for frame in second["frames"]})

    def test_binding_does_not_resize_or_accept_missing_sleeve(self):
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        with self.assertRaises(ValueError):
            shirt_placement(pattern, [instance for instance in inventory["instances"] if instance["templateId"] != "sleeve_left"])
        binding = next(panel for panel in pattern["panels"] if panel["id"] == "opening_binding_left_left")
        edge = next(edge for edge in binding["draft"]["edges"] if edge["name"] == "right")
        binding["points"][edge["end"]][1] += 1
        with self.assertRaises(ValueError):
            shirt_placement(pattern, inventory["instances"])


if __name__ == "__main__":
    unittest.main()
