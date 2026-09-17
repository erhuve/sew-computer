import copy
import math
import unittest

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from cloth_domain import mesh_cloth_domain, validate_cloth_domain
from meshing_test import shirt_pattern


class ClothDomainTests(unittest.TestCase):
    def test_binding_includes_allowances_and_exact_interior_paths(self):
        panel = next(panel for panel in shirt_pattern()["panels"] if panel["id"] == "opening_binding_left_left")
        mesh = mesh_cloth_domain(panel, 20)
        self.assertEqual(mesh["representation"], "cut-line-cloth")
        self.assertFalse(mesh["solverReady"])
        self.assertAlmostEqual(Polygon(panel["points"]).bounds[2] - Polygon(panel["points"]).bounds[0], 20)
        self.assertAlmostEqual(max(point[0] for point in mesh["restPositions"]) - min(point[0] for point in mesh["restPositions"]), 40)
        self.assertEqual(mesh["sourceDomain"], "draft.cutLine")
        self.verify_geometry(panel, mesh)

    def verify_geometry(self, panel, mesh):
        positions = mesh["restPositions"]
        cells = [Polygon([positions[index] for index in triangle]) for triangle in mesh["triangles"]]
        self.assertLess(unary_union(cells).symmetric_difference(Polygon(panel["draft"]["cutLine"])).area, 1e-7)
        for position, weights in zip(positions, mesh["sourceWeights"]):
            reconstructed = [sum(panel["draft"]["cutLine"][item["point"]][axis] * item["weight"] for item in weights) for axis in range(2)]
            self.assertLess(math.dist(position, reconstructed), 1e-7)
        for path in mesh["stitchPaths"]:
            self.assertAlmostEqual(path["samples"][0]["arcMm"], 0)
            self.assertAlmostEqual(path["samples"][-1]["arcMm"], path["lengthMm"])
            length = 0
            for sample in path["samples"]:
                triangle = mesh["triangles"][sample["triangle"]]
                reconstructed = [sum(positions[index][axis] * weight for index, weight in zip(triangle, sample["weights"])) for axis in range(2)]
                source_first, source_second = panel["points"][sample["sourceSegment"]:sample["sourceSegment"] + 2]
                source_position = [first + (second - first) * sample["sourceFraction"] for first, second in zip(source_first, source_second)]
                self.assertLess(math.dist(reconstructed, source_position), 1e-7)
            for segment in path["segments"]:
                endpoints = [path["samples"][index]["restPosition"] for index in segment["samples"]]
                line = LineString(endpoints)
                self.assertLess(line.difference(cells[segment["triangle"]].buffer(1e-7)).length, 1e-7)
                length += line.length
            self.assertAlmostEqual(length, path["lengthMm"])

    def test_curved_and_concave_real_panels_preserve_geometry(self):
        for panel in shirt_pattern()["panels"]:
            with self.subTest(panel=panel["id"]):
                self.verify_geometry(panel, mesh_cloth_domain(panel, 60))

    def test_invalid_cut_and_path_inputs_fail(self):
        original = shirt_pattern()["panels"][0]
        for mutation in ("missing", "outside", "duplicate", "length", "interval"):
            panel = copy.deepcopy(original)
            if mutation == "missing":
                del panel["draft"]["cutLine"]
            elif mutation == "outside":
                panel["draft"]["cutLine"] = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
            elif mutation == "duplicate":
                panel["draft"]["edges"].append(panel["draft"]["edges"][0])
            elif mutation == "length":
                panel["draft"]["edges"][0]["lengthMm"] = float("nan")
            else:
                panel["draft"]["edges"][0]["start"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                mesh_cloth_domain(panel)

    def test_translated_reflected_source_keeps_stitch_correspondence(self):
        panel = next(panel for panel in shirt_pattern()["panels"] if panel["id"] == "opening_binding_left_left")
        original = mesh_cloth_domain(panel, 20)
        changed = copy.deepcopy(panel)
        for points in (changed["points"], changed["draft"]["cutLine"]):
            for point in points:
                point[:] = [-point[0] + 71, point[1] - 83]
        reflected = mesh_cloth_domain(changed, 20)
        self.verify_geometry(changed, reflected)
        self.assertNotEqual(original["sourcePanelDigest"], reflected["sourcePanelDigest"])
        self.assertEqual([path["lengthMm"] for path in original["stitchPaths"]], [path["lengthMm"] for path in reflected["stitchPaths"]])

    def test_runtime_validator_rejects_missing_and_falsified_correspondence(self):
        panel = next(panel for panel in shirt_pattern()["panels"] if panel["id"] == "opening_binding_left_left")
        original = mesh_cloth_domain(panel, 20)
        for mutation in ("path", "segment", "cut_weight", "stitch_weight", "source_fraction", "triangle", "winding", "unused", "readiness", "identity"):
            mesh = copy.deepcopy(original)
            if mutation == "path":
                mesh["stitchPaths"].pop()
            elif mutation == "segment":
                mesh["stitchPaths"][0]["segments"].pop()
            elif mutation == "cut_weight":
                mesh["sourceWeights"][0][0]["point"] = 2
            elif mutation == "stitch_weight":
                mesh["stitchPaths"][0]["samples"][0]["weights"] = [1, 1, 1]
            elif mutation == "source_fraction":
                mesh["stitchPaths"][0]["samples"][0]["sourceFraction"] = 0.5
            elif mutation == "triangle":
                mesh["triangles"].pop()
            elif mutation == "winding":
                mesh["triangles"][-1].reverse()
            elif mutation == "unused":
                mesh["restPositions"].append(mesh["restPositions"][0][:])
                mesh["sourceWeights"].append(copy.deepcopy(mesh["sourceWeights"][0]))
            elif mutation == "readiness":
                mesh["solverReady"] = True
            else:
                mesh["sourcePanelDigest"] = "bad"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_cloth_domain(panel, mesh)
        missing = copy.deepcopy(panel)
        missing["draft"]["edges"].pop()
        with self.assertRaisesRegex(ValueError, "cover every source segment"):
            mesh_cloth_domain(missing, 20)

    def test_concave_support_cannot_cross_outside_cloth(self):
        panel = next(panel for panel in shirt_pattern()["panels"] if panel["id"] == "front_left")
        mesh = mesh_cloth_domain(panel, 60)
        mesh["sourceWeights"][44] = [{"point": index, "weight": weight} for index, weight in zip(
            (0, 1, 3), (0.857958514684747, 0.00004138888668123175, 0.14200009642857142))]
        with self.assertRaisesRegex(ValueError, "Original cut vertex identity changed|support leaves source domain"):
            validate_cloth_domain(panel, mesh)
        mesh = mesh_cloth_domain(panel, 60)
        mesh["sourceWeights"][97] = [{"point": index, "weight": weight} for index, weight in zip(
            (0, 1, 3), (0.06164565004304068, 0.03429219951052795, 0.9040621504464283))]
        with self.assertRaisesRegex(ValueError, "support leaves source domain"):
            validate_cloth_domain(panel, mesh)


if __name__ == "__main__":
    unittest.main()
