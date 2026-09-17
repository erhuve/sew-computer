import math
import json
import unittest

from meshing import mesh_panel
from meshing_test import CONSTRUCTION, shirt_pattern
from assembly import compile_assembly, compile_inventory
from quality_meshing import quality_triangles
from shapely.geometry import Polygon
from simulation_validation import validate_rest_mesh
from cloth_domain import mesh_cloth_domain, validate_cloth_domain
from shapely.ops import unary_union


class QualityRefinementTests(unittest.TestCase):
    def test_sleeve_cut_domain_excludes_exterior_nearcollinear_cells(self):
        for panel in shirt_pattern()["panels"]:
            if panel["id"] not in ("sleeve_left", "sleeve_right"):
                continue
            with self.subTest(panel=panel["id"]):
                mesh = mesh_cloth_domain(panel, 60, quality_refinement=True)
                source = Polygon(panel["draft"]["cutLine"])
                cells = [Polygon([mesh["restPositions"][index] for index in triangle]) for triangle in mesh["triangles"]]
                self.assertEqual(mesh["restPositions"][:len(panel["draft"]["cutLine"]) - 1], panel["draft"]["cutLine"][:-1])
                self.assertTrue(all(source.contains(cell.centroid) for cell in cells))
                self.assertLess(unary_union(cells).symmetric_difference(source).area, 1e-6)
                self.assertLess(abs(sum(cell.area for cell in cells) - source.area), 1e-6)
                validate_cloth_domain(panel, mesh)

    def test_collar_registration_rounding_does_not_duplicate_boundary(self):
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        graph = compile_assembly(pattern, inventory)
        for panel in pattern["panels"]:
            if panel["id"] not in ("collar_stand", "collar_fall"):
                continue
            with self.subTest(panel=panel["id"]):
                registrations = {member["edgeName"]: [0, .25, .5, .75, 1] for operation in graph["operations"] for member in operation["participants"] if member["instanceId"].split(":")[0] == panel["id"]}
                mesh = mesh_panel(panel, 60, registrations, quality_refinement=True)
                self.assertEqual(mesh["restPositions"][:len(panel["points"]) - 1], panel["points"][:-1])
                self.assertLess(len(mesh["restPositions"]), 1000)
                validate_rest_mesh(panel, mesh)
                for boundary in mesh["boundaries"]:
                    if boundary["name"] in registrations:
                        for fraction in registrations[boundary["name"]]:
                            self.assertLess(min(abs(sample["arcMm"] - fraction * boundary["lengthMm"]) for sample in boundary["samples"]), 1e-7)

    def test_refinement_rejects_zero_length_boundary_before_subdivision(self):
        points = [[0, 0], [10, 0], [10, 10], [0, 10]]
        with self.assertRaisesRegex(ValueError, "zero-length"):
            quality_triangles(Polygon(points), [point[:] for point in points], [points[0], points[0], *points[1:]], 10)

    def test_body_refinement_does_not_create_micron_steiner_clusters(self):
        for panel in shirt_pattern()["panels"][:2]:
            with self.subTest(panel=panel["id"]):
                source = panel["points"][:-1]
                shortest_source = min(math.dist(first, second) for first, second in zip(source, source[1:] + source[:1]))
                mesh = mesh_panel(panel, 60, quality_refinement=True)
                positions = mesh["restPositions"]
                shortest_mesh = min(math.dist(positions[first], positions[second]) for triangle in mesh["triangles"] for first, second in zip(triangle, triangle[1:] + triangle[:1]))
                self.assertEqual(positions[:len(source)], source)
                self.assertGreaterEqual(shortest_mesh, shortest_source / 2)
                self.assertLess(len(positions), 1000)
                report = validate_rest_mesh(panel, mesh)
                self.assertLess(report["boundaryErrorMm"], 1e-7)
                self.assertLess(report["areaErrorMm2"], 1e-7)
                self.assertFalse(report["solverQualityAccepted"])


if __name__ == "__main__":
    unittest.main()
