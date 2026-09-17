import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services/engine"))

from assembly import compile_assembly, compile_inventory
from meshing_test import CONSTRUCTION, shirt_pattern


class ShirtRegistrationTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specification = importlib.util.spec_from_file_location("shirt_probe", ROOT / "scripts/spike-full-shirt.py")
        cls.module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.module)
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        cls.panels = {panel["id"]: panel for panel in pattern["panels"]}
        cls.operations = compile_assembly(pattern, inventory)["operations"]

    def endpoint(self, identity, edge_name, endpoint):
        panel = self.panels[identity.split(":")[0]]
        edge = next(edge for edge in panel["draft"]["edges"] if edge["name"] == edge_name)
        return identity, tuple(panel["points"][edge[endpoint]])

    def equivalences(self, legacy=False):
        parents = {}

        def root(point):
            parents.setdefault(point, point)
            if parents[point] != point:
                parents[point] = root(parents[point])
            return parents[point]

        for operation in self.operations:
            for endpoint in ("start", "end"):
                points = []
                for index, participant in enumerate(operation["participants"]):
                    if legacy:
                        reverse = index > 0 and operation["id"].startswith(("sleeve_back_", "underarm_"))
                    else:
                        reverse = self.module.shirt_registration_direction(operation, participant) == "reverse"
                    selected = ("end" if endpoint == "start" else "start") if reverse else endpoint
                    points.append(self.endpoint(participant["instanceId"], participant["edgeName"], selected))
                for point in points[1:]:
                    parents[root(point)] = root(points[0])
        return root

    def test_collar_intervals_do_not_collapse_through_garment_junctions(self):
        legacy = self.equivalences(legacy=True)
        corrected = self.equivalences()
        for role in ("shell", "facing"):
            for interval in (1, 3, 5):
                with self.subTest(role=role, interval=interval):
                    start = self.endpoint(f"collar_stand:{role}", f"neck_{interval}", "start")
                    end = self.endpoint(f"collar_stand:{role}", f"neck_{interval}", "end")
                    self.assertEqual(legacy(start), legacy(end))
                    self.assertNotEqual(corrected(start), corrected(end))

    def test_entire_collar_chain_retains_seven_distinct_junctions(self):
        corrected = self.equivalences()
        for role in ("shell", "facing"):
            junctions = [self.endpoint(f"collar_stand:{role}", f"neck_{index}", "start") for index in range(6)]
            junctions.append(self.endpoint(f"collar_stand:{role}", "neck_5", "end"))
            self.assertEqual(len({corrected(point) for point in junctions}), 7)

    def test_sleeve_cap_apex_and_underarm_meet_correct_body_junctions(self):
        corrected = self.equivalences()
        for side in ("left", "right"):
            with self.subTest(side=side):
                apex = corrected(self.endpoint(f"sleeve_{side}:shell", "cap_front", "end"))
                self.assertEqual(apex, corrected(self.endpoint(f"sleeve_{side}:shell", "cap_back", "start")))
                for body in ("front", "back"):
                    self.assertEqual(apex, corrected(self.endpoint(f"{body}_{side}:shell", "shoulder", "start")))
                underarm = corrected(self.endpoint(f"sleeve_{side}:shell", "cap_front", "start"))
                self.assertEqual(underarm, corrected(self.endpoint(f"sleeve_{side}:shell", "cap_back", "end")))
                for body in ("front", "back"):
                    self.assertEqual(underarm, corrected(self.endpoint(f"{body}_{side}:shell", "side", "end")))
                self.assertNotEqual(apex, underarm)

    def test_collar_fall_keeps_shell_and_facing_on_the_same_stand_end(self):
        corrected = self.equivalences()
        for endpoint, opposite in (("start", "end"), ("end", "start")):
            stand = corrected(self.endpoint("collar_stand:shell", "fall", endpoint))
            self.assertEqual(stand, corrected(self.endpoint("collar_stand:facing", "fall", endpoint)))
            for role in ("shell", "facing"):
                self.assertEqual(stand, corrected(self.endpoint(f"collar_fall:{role}", "top", opposite)))


if __name__ == "__main__":
    unittest.main()
