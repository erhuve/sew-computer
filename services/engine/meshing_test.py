import copy
import json
import pathlib
import math
import struct
import unittest

from assembly import build_inspection, compile_assembly, compile_inventory, expand_instance_meshes
from inspection_gltf import inspection_glb
from meshing import mesh_panel
from shirt import compile_shirt
from simulation_validation import validate_rest_mesh


CONSTRUCTION = {"collar": "stand-and-fall", "cuff": "button", "sleeves": "long", "opening": "buttons", "frill": "front-opening", "hem": "curved-back-tail"}


def rectangle():
    return {"id": "rectangle", "points": [[0, 0], [120, 0], [120, 80], [0, 80], [0, 0]]}


def shirt_pattern(bust=960, hip=1000, shoulder=400, selection=None):
    inputs = {"bodyMm": {"bust": bust, "hip": hip, "shoulder": shoulder}, "lengthMm": 650, "easeMm": 100, "flare": 1,
        "inputDigest": "a" * 64, "provenance": [], "design": {"seamAllowanceMm": 10, "opening": "buttons", "placketWidthMm": 30,
        "hem": "curved-back-tail", "tailExtensionMm": 100, "buttonSpacingMm": 80, "frill": "front-opening", "frillWidthMm": 35,
        "frillFullness": 1.8, "sleeves": "long", "sleeveLengthMm": 550, "cuff": "button", "cuffDepthMm": 55,
        "cuffCircumferenceMm": 220, "collar": "stand-and-fall", "collarStandMm": 30, "collarFallMm": 60}}
    inputs["design"].update(selection or {})
    return compile_shirt(inputs, "synthetic")


class RestMeshTests(unittest.TestCase):
    def test_quality_refinement_improves_dense_curved_body_without_rest_changes(self):
        for panel in shirt_pattern()["panels"][:2]:
            baseline = validate_rest_mesh(panel, mesh_panel(panel, 60))
            refined = validate_rest_mesh(panel, mesh_panel(panel, 60, quality_refinement=True))
            self.assertGreater(refined["minTriangleQuality"], baseline["minTriangleQuality"] * 10)
            self.assertLess(refined["boundaryErrorMm"], 1e-7)
            self.assertLess(refined["areaErrorMm2"], 1e-7)
            self.assertFalse(refined["solverQualityAccepted"])

    def test_exact_registration_preserves_irregular_source_lengths(self):
        for panel in shirt_pattern()["panels"]:
            if panel["id"] not in ("placket_left", "frill_left", "front_left"):
                continue
            edge = panel["draft"]["edges"][-1]
            fractions = [0, 0.17, 0.5, 0.83, 1]
            mesh = mesh_panel(panel, 40, {edge["name"]: fractions})
            validate_rest_mesh(panel, mesh)
            boundary = next(item for item in mesh["boundaries"] if item["name"] == edge["name"])
            for fraction in fractions:
                error = min(abs(sample["arcMm"] - fraction * boundary["lengthMm"]) for sample in boundary["samples"])
                self.assertLess(error, 1e-7)
        for fractions in ({"unknown": [0]}, {"right": [float("nan")]}, {"right": [True]}, {"right": [-0.1]}):
            panel = next(item for item in shirt_pattern()["panels"] if item["id"] == "placket_left")
            with self.assertRaises(ValueError):
                mesh_panel(panel, 40, fractions)

    def test_assembly_physical_coverage_and_order(self):
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        graph = compile_assembly(pattern, inventory)
        self.assertFalse(graph["solverReady"])
        operations = {item["id"]: item for item in graph["operations"]}
        fixture = json.loads((pathlib.Path(__file__).parents[2] / "packages/test-fixtures/assembly-expected.json").read_text())["fullShirt"]
        for expected in fixture["multiwayAttachments"]:
            side = expected["id"].split("_")[0]
            self.assertEqual({member["instanceId"] for member in operations[f"placket_attach_{side}"]["participants"]}, set(expected["participants"]))
        self.assertEqual(len(graph["closures"]), 10)
        self.assertEqual(len([item for item in graph["operations"] if item["kind"] == "binding"]), 4)
        covered = [(item["instanceId"], item["edgeName"]) for operation in graph["operations"] for item in operation["participants"]]
        covered += [(item["instanceId"], item["edgeName"]) for item in graph["freeBoundaries"]]
        self.assertEqual(len(covered), len(set(covered)))
        by_id = {panel["id"]: panel for panel in pattern["panels"]}
        expected = {(instance["id"], edge["name"]) for instance in inventory["instances"] for edge in by_id[instance["templateId"]]["draft"]["edges"]}
        self.assertEqual(set(covered), expected)
        completed = set()
        for operation in graph["operations"]:
            self.assertTrue(set(operation["dependsOn"]) <= completed)
            completed.add(operation["id"])
        self.assertIn("perimeter:collar_fall:bottom", operations["collar_fall_attach"]["dependsOn"])
        for side in ("left", "right"):
            members = operations[f"underarm_{side}"]["participants"]
            self.assertEqual(members[0]["instanceId"], members[1]["instanceId"])
            self.assertNotEqual(members[0]["edgeName"], members[1]["edgeName"])
            self.assertNotIn("extension", [item["edgeName"] for item in operations[f"cuff_gather_{side}"]["participants"]])

    def test_assembly_rejects_source_graph_and_closure_mutations(self):
        for mutation in ("missing", "duplicate", "edge", "ratio", "stitch", "closure", "closure_outside"):
            pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
            if mutation == "missing":
                pattern["drafting"]["assembly"].pop()
            elif mutation == "duplicate":
                pattern["drafting"]["assembly"][-1] = pattern["drafting"]["assembly"][0]
            elif mutation == "edge":
                pattern["drafting"]["assembly"][0]["sides"][0]["edge"] = 1
            elif mutation == "ratio":
                pattern["drafting"]["assembly"][0]["ratio"] = float("nan")
            elif mutation == "stitch":
                pattern["stitches"].pop()
            else:
                panel = next(item for item in pattern["panels"] if item["id"] == "placket_left")
                if mutation == "closure":
                    panel["draft"]["marks"] = []
                else:
                    panel["draft"]["marks"][0]["point"] = [99999, 0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compile_assembly(pattern, inventory)

    def test_rectangle_fidelity_and_refinement(self):
        for resolution in (20, 40, 100):
            panel = rectangle()
            mesh = mesh_panel(panel, resolution)
            report = validate_rest_mesh(panel, mesh)
            self.assertLessEqual(report["maxEdgeMm"], resolution + 1e-7)
            self.assertLess(report["areaErrorMm2"], 1e-6)
            self.assertFalse(report["solverQualityAccepted"])

    def test_concave_polygon(self):
        panel = {"id": "concave", "points": [[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100], [0, 0]]}
        validate_rest_mesh(panel, mesh_panel(panel, 10))
        mesh = mesh_panel(panel, 100)
        mesh["sourceWeights"][3] = [{"point": 0, "weight": 0.6}, {"point": 1, "weight": 0.2}, {"point": 5, "weight": 0.2}]
        with self.assertRaisesRegex(ValueError, "direct identity"):
            validate_rest_mesh(panel, mesh)

    def test_rigid_source_transform_preserves_area_and_boundaries(self):
        panel = rectangle()
        angle = 0.73
        transformed = {"id": panel["id"], "points": [[point[0] * math.cos(angle) - point[1] * math.sin(angle) + 1000, point[0] * math.sin(angle) + point[1] * math.cos(angle) - 400] for point in panel["points"]]}
        report = validate_rest_mesh(transformed, mesh_panel(transformed))
        self.assertLess(report["areaErrorMm2"], 1e-6)
        self.assertLess(report["boundaryErrorMm"], 1e-6)

    def test_strip_mesh_quality_after_boundary_refinement(self):
        for panel in shirt_pattern()["panels"]:
            if panel["id"] in ("frill_left", "placket_left"):
                report = validate_rest_mesh(panel, mesh_panel(panel))
                self.assertGreater(report["minTriangleQuality"], 0.7)

    def test_reject_invalid_sources(self):
        for points in ([[[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]], [[0, 0], [float("nan"), 0], [1, 1], [0, 0]], [[0, 0], [1, 0], [1, 1], [0, 1]]]):
            with self.assertRaises(ValueError):
                mesh_panel({"id": "bad", "points": points})
        with self.assertRaises(ValueError):
            mesh_panel({**rectangle(), "holes": [[[1, 1], [2, 1], [1, 2]]]})
        with self.assertRaises(ValueError):
            mesh_panel(rectangle(), float("nan"))

    def test_reject_rest_shape_and_topology_mutations(self):
        panel = rectangle()
        mesh = mesh_panel(panel)
        for mutation in ("position", "weight", "duplicate", "missing", "winding", "index", "unused"):
            altered = copy.deepcopy(mesh)
            if mutation == "position":
                altered["restPositions"][0][0] += 1
            elif mutation == "weight":
                altered["sourceWeights"][0][0]["weight"] = 0.5
            elif mutation == "duplicate":
                altered["triangles"].append(altered["triangles"][0])
            elif mutation == "missing":
                altered["triangles"].pop()
            elif mutation == "winding":
                altered["triangles"][0].reverse()
            elif mutation == "index":
                altered["triangles"][0][0] = -1
            else:
                altered["restPositions"].append([0, 0])
                altered["sourceWeights"].append([{"point": 0, "weight": 1}])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_rest_mesh(panel, altered)

    def test_resource_rejection(self):
        panel = {"id": "huge", "points": [[0, 0], [10000, 0], [10000, 10000], [0, 10000], [0, 0]]}
        with self.assertRaisesRegex(ValueError, "resource budget"):
            mesh_panel(panel, 5)

    def test_full_shirt_inventory_and_six_size_fidelity(self):
        for bust, hip, shoulder in [(800, 850, 350), (880, 930, 380), (960, 1000, 400), (1040, 1090, 430), (1160, 1210, 460), (1280, 1330, 490)]:
            with self.subTest(bust=bust):
                pattern = shirt_pattern(bust, hip, shoulder)
                source = json.dumps(pattern).encode()
                result = build_inspection(source, CONSTRUCTION)
                self.assertEqual(len(result["instances"]), 24)
                self.assertEqual(len(result["unresolvedPhysicalRoles"]), 6)
                self.assertEqual(len(result["templates"]), 18)
                self.assertEqual(result["classification"], "placement-inspection")
                self.assertEqual(source, json.dumps(pattern).encode())
                for instance in result["instances"]:
                    expected_mirror = instance["templateId"] in {"front_right", "back_right", "sleeve_right", "placket_right", "cuff_right", "frill_right", "opening_binding_right_left", "opening_binding_right_right"}
                    self.assertEqual(instance["mirrorX"], expected_mirror)

    def test_inventory_mutations(self):
        for mutation in ("missing", "duplicate", "count", "unknown", "component", "grain", "edge", "role"):
            pattern = shirt_pattern()
            if mutation == "missing":
                pattern["panels"].pop()
            elif mutation == "duplicate":
                pattern["panels"][1] = pattern["panels"][0]
            elif mutation == "count":
                pattern["panels"][0]["cutQuantity"] = 2
            elif mutation == "unknown":
                pattern["drafting"]["compiler"] = "unknown"
            elif mutation == "component":
                pattern["drafting"]["components"].remove("cuffs")
            elif mutation == "grain":
                pattern["panels"][0]["draft"]["grainline"] = [[1e100, 0], [1e100, 1e100]]
            elif mutation == "edge":
                pattern["panels"][0]["draft"]["edges"][0]["end"] = 20000
            else:
                pattern["panels"][0]["draft"]["component"] = "collar"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compile_inventory(json.dumps(pattern).encode(), CONSTRUCTION)

    def test_independent_inventory_fixture(self):
        fixture = json.loads((pathlib.Path(__file__).parents[2] / "packages/test-fixtures/assembly-expected.json").read_text())["fullShirt"]
        _, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        expected = {f'{template["id"]}:{role}' for template in fixture["templates"] for role in template["roles"]}
        self.assertEqual({instance["id"] for instance in inventory["instances"]}, expected)
        self.assertEqual({item["templateId"] for item in inventory["unresolvedPhysicalRoles"]}, {template["id"] for template in fixture["templates"] if template.get("interfacing")})

    def test_instance_mirroring_and_glb_units(self):
        inspection = build_inspection(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        templates = {template["templateId"]: template for template in inspection["templates"]}
        expanded = expand_instance_meshes(inspection)
        for instance in expanded:
            template = templates[instance["templateId"]]
            for face, source_face in zip(instance["triangles"], template["triangles"]):
                points = [instance["restPositions"][vertex] for vertex in face]
                first, second, third = points
                self.assertGreater((second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0]), 0)
                for start, end in zip(source_face, source_face[1:] + source_face[:1]):
                    self.assertAlmostEqual(math.dist(instance["restPositions"][start], instance["restPositions"][end]), math.dist(template["restPositions"][start], template["restPositions"][end]), places=8)
            source_grain = next(item["sourceGrainline"] for item in inspection["instances"] if item["id"] == instance["instanceId"])
            self.assertAlmostEqual(math.dist(*instance["grainline"]), math.dist(*source_grain), places=8)
        artifact = inspection_glb(inspection)
        magic, version, length = struct.unpack_from("<III", artifact)
        self.assertEqual((magic, version, length), (0x46546C67, 2, len(artifact)))
        json_length, kind = struct.unpack_from("<II", artifact, 12)
        document = json.loads(artifact[20:20 + json_length])
        self.assertEqual(kind, 0x4E4F534A)
        self.assertEqual(len(document["meshes"]), 24)
        self.assertNotIn("uri", document["buffers"][0])
        self.assertEqual(document["extras"]["patternDigest"], inspection["patternDigest"])
        data_start = 20 + json_length + 8
        for index, instance in enumerate(expanded):
            accessor = document["accessors"][index * 2]
            view = document["bufferViews"][accessor["bufferView"]]
            for vertex, source in enumerate(instance["restPositions"]):
                actual = struct.unpack_from("<3f", artifact, data_start + view["byteOffset"] + vertex * 12)
                self.assertAlmostEqual(actual[0] * 1000, source[0], places=3)
                self.assertAlmostEqual(actual[1] * -1000, source[1], places=3)
                self.assertEqual(actual[2], 0)

    def test_source_edge_mapping_mutations(self):
        panel = shirt_pattern()["panels"][0]
        mesh = mesh_panel(panel)
        for mutation in ("nan", "arc", "reverse", "missing"):
            altered = copy.deepcopy(mesh)
            edge = altered["boundaries"][0]
            if mutation == "nan":
                edge["lengthMm"] = float("nan")
            elif mutation == "arc":
                edge["samples"][1]["arcMm"] = float("inf")
            elif mutation == "reverse":
                edge["samples"].reverse()
            else:
                edge["samples"].pop(1)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_rest_mesh(panel, altered)

    def test_independent_inventory_variants(self):
        fixture = json.loads((pathlib.Path(__file__).parents[2] / "packages/test-fixtures/assembly-expected.json").read_text())
        changes = {"full_without_frills": {"frill": "none"}, "full_without_cuffs": {"cuff": "none"}, "full_stand_only": {"collar": "stand"}, "full_without_collar": {"collar": "none"}, "full_straight_hem": {"hem": "straight"}}
        for variant in fixture["variants"]:
            with self.subTest(variant=variant["id"]):
                construction = {**CONSTRUCTION, **variant.get("selection", changes.get(variant["id"], {}))}
                pattern = shirt_pattern(selection=construction)
                _, inventory = compile_inventory(json.dumps(pattern).encode(), construction)
                self.assertEqual(len(inventory["instances"]), variant["fabricInstanceCount"])
                self.assertEqual(len(inventory["unresolvedPhysicalRoles"]), variant["interfacingRoleCount"])
                expected = set(variant["templates"]) if "templates" in variant else {template["id"] for template in fixture["fullShirt"]["templates"]} - set(variant["omitTemplates"])
                self.assertEqual({instance["templateId"] for instance in inventory["instances"]}, expected)
        with self.assertRaises(ValueError):
            compile_inventory(json.dumps(shirt_pattern()).encode(), {**CONSTRUCTION, "sleeves": "short"})


if __name__ == "__main__":
    unittest.main()
