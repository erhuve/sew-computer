import json
import math
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
            panel = panels[frame["instanceId"].split(":")[0]]
            hand = -1 if instances[frame["instanceId"]]["mirrorX"] else 1
            rest = [[hand * point[0], point[1], 0] for point in panel["points"]]
            placed = place_rest_positions(rest, frame)
            for index in range(len(rest) - 1):
                self.assertAlmostEqual(math.dist(rest[index], rest[index + 1]), math.dist(placed[index], placed[index + 1]), places=8)
            self.assertEqual(placement["classification"], "unvalidated-placement")
            if frame["instanceId"].startswith("front_"):
                self.assertGreater(frame["basis"][2][1], 0)


if __name__ == "__main__":
    unittest.main()
