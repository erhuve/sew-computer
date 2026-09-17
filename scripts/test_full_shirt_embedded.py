import hashlib
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services/engine"))

from assembly import compile_assembly, compile_inventory
from cloth_domain import mesh_cloth_domain, validate_cloth_domain
from embedded_constraints import build_embedded_constraints, validate_embedded_constraints
from meshing_test import CONSTRUCTION, shirt_pattern


class FullShirtEmbeddedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="sew-embedded-shirt-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.output = Path(cls.directory.name) / "probe"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/spike-full-shirt.py"),
            "--output", str(cls.output), "--fixture", "torso", "--embedded-sewing", "--quality-refinement",
            "--shirt-placement", "--disable-contact", "--steps", "3", "--substeps", "2", "--ramp-steps", "2"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise AssertionError(result.stdout[-4000:] + result.stderr[-4000:])
        cls.report = json.loads((cls.output / "report.json").read_text())
        cls.canonical = json.loads((cls.output / "canonical.json").read_text())

    def test_captured_source_and_geometry_identity(self):
        for name, digest in self.report["sourceDigests"].items():
            self.assertEqual(hashlib.sha256((self.output / "source-snapshot" / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashlib.sha256((self.output / "source-geometry.json").read_bytes()).hexdigest(), self.report["sourceGeometrySha256"])
        self.assertIn("cloth_domain.py", self.report["sourceDigests"])
        self.assertIn("embedded_constraints.py", self.report["sourceDigests"])

    def test_full_shirt_source_endpoints_and_invalid_partial_intervals(self):
        specification = importlib.util.spec_from_file_location("full_shirt_probe", ROOT / "scripts/spike-full-shirt.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        graph = compile_assembly(pattern, inventory)
        templates = {panel["id"]: {"panel": panel, "mesh": mesh_cloth_domain(panel, 60, quality_refinement=True)} for panel in pattern["panels"]}
        sources = {instance["id"]: templates[instance["templateId"]] for instance in inventory["instances"]}
        registrations = module.shirt_embedded_registrations(graph["operations"], sources, True)
        bundle = build_embedded_constraints(sources, registrations)
        self.assertEqual(len(bundle["sourceIdentities"]), 24)
        self.assertFalse(validate_embedded_constraints(sources, bundle)["solverReady"])
        for registration in registrations:
            for member in registration["members"]:
                path = next(path for path in sources[member["instanceId"]]["mesh"]["stitchPaths"] if path["name"] == member["pathName"])
                self.assertEqual(member["endArcMm"], path["lengthMm"])
        for mutation in ("source-point", "start", "end", "tiny-start", "tiny-end", "boolean-start"):
            operation = copy.deepcopy(graph["operations"][0])
            participant = operation["participants"][0]
            if mutation == "source-point":
                participant["sourcePointInterval"][0] += 1
            elif mutation == "start":
                participant["intervalMm"][0] += .001
            elif mutation == "tiny-start":
                participant["intervalMm"][0] += 1e-8
            elif mutation == "tiny-end":
                participant["intervalMm"][1] -= 1e-8
            elif mutation == "boolean-start":
                participant["intervalMm"][0] = False
            else:
                participant["intervalMm"][1] += .001
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.shirt_embedded_registrations([operation], sources, True)

    def test_source_cut_domain_and_embedded_registration(self):
        panels = {panel["id"]: panel for panel in shirt_pattern()["panels"]}
        instances = {instance["id"]: instance for instance in self.canonical["inventory"]["instances"]}
        sources = {}
        for identity in self.canonical["instanceOffsets"]:
            template = instances[identity]["templateId"]
            mesh = self.canonical["sourceTemplates"][template]
            validate_cloth_domain(panels[template], mesh)
            sources[identity] = {"panel": panels[template], "mesh": mesh}
        validation = validate_embedded_constraints(sources, self.canonical["embeddedConstraints"])
        self.assertTrue(validation["sourceCorrespondenceAccepted"])
        self.assertFalse(validation["solverReady"])
        self.assertEqual(validation["constraintCount"], 25)

    def test_physical_mirroring_and_winding(self):
        expected_ids = {"front_left:shell", "front_right:shell", "back_left:shell", "back_right:shell"}
        self.assertEqual(set(self.canonical["instanceOffsets"]), expected_ids)
        rest = np.asarray(self.canonical["restMeters"])
        faces = np.asarray(self.canonical["triangles"]).reshape((-1, 3))
        for identity in expected_ids:
            template = identity.split(":")[0]
            mesh = self.canonical["sourceTemplates"][template]
            start = self.canonical["instanceOffsets"][identity]
            actual = rest[start:start + len(mesh["restPositions"])].copy() * 1000
            if "right" in template:
                actual[:, 0] *= -1
            np.testing.assert_allclose(actual[:, :2], mesh["restPositions"], atol=1e-10)
            np.testing.assert_array_equal(actual[:, 2], 0)
        normals = np.cross(rest[faces[:, 1]] - rest[faces[:, 0]], rest[faces[:, 2]] - rest[faces[:, 0]])
        self.assertTrue(np.all(normals[:, 2] > 0))

    def test_independent_residual_and_rejected_classification(self):
        positions = np.asarray(self.canonical["positionsMeters"])
        previous = np.asarray(self.canonical["previousPositionsMeters"])
        velocities = np.asarray(self.canonical["velocitiesMetersPerSecond"])
        np.testing.assert_allclose(velocities, (positions - previous) / self.report["coupling"]["timestepSeconds"], rtol=1e-5, atol=1e-5)
        residuals = []
        for constraint in self.canonical["embeddedConstraints"]["constraints"]:
            residual = np.zeros(3)
            for term in constraint["terms"]:
                index = self.canonical["instanceOffsets"][term["instanceId"]] + term["vertex"]
                residual += term["coefficient"] * positions[index]
            residuals.append(float(np.linalg.norm(residual) * 1000))
        self.assertAlmostEqual(max(residuals), self.report["seamGapMaxMm"], places=8)
        self.assertFalse(self.report["accepted"])
        self.assertEqual(self.report["classification"], "rejected-experimental-assembly")
        self.assertTrue(self.report["restTensorsUnchanged"])
        self.assertEqual(self.report["coupling"]["mode"], "embedded-cut-cloth")
        self.assertTrue(any("reintroduce penetration" in limitation for limitation in self.report["limitations"]))


if __name__ == "__main__":
    unittest.main()
