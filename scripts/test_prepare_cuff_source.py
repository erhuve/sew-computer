"""Fresh-source reproduction and correspondence checks; no dynamics acceptance.

Run with the pinned research Python. All inputs come from the synthetic shirt
compiler, and subprocesses get a clean environment and isolated temporary home.
"""

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/prepare-cuff-source.py"


class PrepareCuffSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.environment = {
            "PATH": os.environ.get("PATH", ""), "HOME": str(cls.directory), "LANG": "C.UTF-8",
            "WARP_CACHE_PATH": str(ROOT / ".planning/solver/source-review-warp-cache"),
            "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        for name in ("first", "second"):
            cls.run_cli("prepare-cuff-source.py", "--output", cls.directory / name)
        cls.canonical_path = cls.directory / "first/canonical.json"
        cls.content = cls.canonical_path.read_bytes()
        cls.source = json.loads(cls.content)

    @classmethod
    def run_cli(cls, name, *arguments, success=True):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
                                cwd=cls.directory, env=cls.environment, capture_output=True, text=True, timeout=60)
        if success and result.returncode:
            raise AssertionError(f"{name} failed: {result.stdout}\n{result.stderr}")
        return result

    def test_reproduction_and_self_contained_provenance(self):
        self.assertEqual(self.content, (self.directory / "second/canonical.json").read_bytes())
        provenance = self.source["provenance"]
        self.assertIs(provenance["accepted"], False)
        self.assertIn("synthetic", provenance["classification"])
        self.assertIn("not historical captured bytes", provenance["classification"])
        self.assertIn("omitted", " ".join(provenance["limitations"]))
        encoded_pattern = json.dumps(self.source["sourcePattern"], sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(provenance["patternSha256"], hashlib.sha256(encoded_pattern).hexdigest())
        self.assertNotIn(str(ROOT), self.content.decode())
        self.assertNotIn(str(self.directory), self.content.decode())
        self.assertEqual(set(provenance["runtime"]), {"python", "numpy", "scipy", "shapely"})
        self.assertEqual(set(provenance["sourceDigests"]), {
            "scripts/prepare-cuff-source.py", "scripts/spike-full-shirt.py", "scripts/solver_process_budget.py",
            "scripts/solver-spike.requirements.txt", "services/engine/shirt.py", "services/engine/assembly.py",
            "services/engine/cloth_domain.py", "services/engine/meshing.py", "services/engine/quality_meshing.py",
            "services/engine/simulation_validation.py", "services/engine/embedded_constraints.py",
            "services/engine/meshing_test.py", "services/engine/inspection_gltf.py"})
        for name, digest in provenance["sourceDigests"].items():
            self.assertEqual(digest, hashlib.sha256((ROOT / name).read_bytes()).hexdigest())
        self.assertEqual(stat.S_IMODE(self.canonical_path.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.canonical_path.stat().st_mode), 0o400)

    def test_cut_geometry_and_layer_units_against_analytic_source(self):
        source = self.source
        self.assertEqual(source["instanceOffsets"], {"cuff_left:shell": 0, "cuff_left:facing": 20})
        rest = np.asarray(source["restMeters"])
        placed = np.asarray(source["placedMeters"])
        triangles = np.asarray(source["triangles"]).reshape((-1, 3))
        self.assertEqual(rest.shape, (40, 3))
        self.assertEqual(triangles.shape, (48, 3))
        np.testing.assert_array_equal(rest[:20], rest[20:])
        np.testing.assert_array_equal(triangles[:24], triangles[24:] - 20)
        np.testing.assert_array_equal(rest[:, 2], 0.)
        np.testing.assert_allclose(placed[:20], rest[:20], rtol=0., atol=0.)
        np.testing.assert_allclose(placed[20:], rest[20:] + [0., 0., .002], rtol=0., atol=0.)
        mesh = source["sourceTemplates"]["cuff_left"]
        self.assertEqual(mesh["units"], "mm")
        self.assertEqual(mesh["sourceDomain"], "draft.cutLine")
        np.testing.assert_array_equal(np.asarray(mesh["restPositions"]) * .001, rest[:20, :2])
        np.testing.assert_array_equal(mesh["triangles"], triangles[:24])
        # The independent fixture is a 240 x 55 mm seam rectangle with 10 mm
        # allowances: 260 x 75 mm cut domain. No mesher report supplies this.
        expected = Polygon([[-.01, -.01], [-.01, .065], [.25, .065], [.25, -.01]])
        cells = [Polygon(rest[triangle, :2]) for triangle in triangles[:24]]
        self.assertLess(unary_union(cells).symmetric_difference(expected).area, 1e-16)
        self.assertAlmostEqual(sum(cell.area for cell in cells), .260 * .075, places=15)
        self.assertTrue(all(cell.area > 0 for cell in cells))

    def test_perimeter_registrations_leave_source_attachment_open(self):
        expected_edges = {
            "extension": (np.array([220., 0.]), np.array([240., 0.])),
            "end": (np.array([240., 0.]), np.array([240., 55.])),
            "outer": (np.array([240., 55.]), np.array([0., 55.])),
            "start": (np.array([0., 55.]), np.array([0., 0.]))}
        operations = self.source["sourceAssemblyOperations"]
        self.assertEqual({item["id"] for item in operations},
                         {f"perimeter:cuff_left:{name}" for name in expected_edges})
        self.assertEqual(len(operations), 4)
        rows = self.source["embeddedConstraints"]["constraints"]
        self.assertEqual(Counter(row["registrationId"] for row in rows),
                         {f"perimeter:cuff_left:{name}": 5 for name in expected_edges})
        for name in expected_edges:
            self.assertEqual(sorted(row["fraction"] for row in rows
                                    if row["registrationId"] == f"perimeter:cuff_left:{name}"),
                             [0., .25, .5, .75, 1.])
        rest = np.asarray(self.source["restMeters"])
        offsets = self.source["instanceOffsets"]
        for row in rows:
            name = row["registrationId"].split(":")[-1]
            first, last = expected_edges[name]
            expected = first + row["fraction"] * (last - first)
            self.assertIn(row["fraction"], [0., .25, .5, .75, 1.])
            self.assertEqual([sample["instanceId"] for sample in row["sourceSamples"]],
                             ["cuff_left:shell", "cuff_left:facing"])
            for sample in row["sourceSamples"]:
                self.assertEqual(sample["pathName"], name)
                np.testing.assert_allclose(sample["restPositionMm"], expected, rtol=0., atol=1e-10)
                actual = sum(weight["weight"] * rest[offsets[sample["instanceId"]] + weight["vertex"]]
                             for weight in sample["weights"])
                np.testing.assert_allclose(actual, [*(expected * .001), 0.], rtol=0., atol=1e-14)
                self.assertFalse(1e-12 < actual[0] < .220 - 1e-12 and abs(actual[1]) < 1e-12)
            residual = sum(term["coefficient"] * rest[offsets[term["instanceId"]] + term["vertex"]]
                           for term in row["terms"])
            np.testing.assert_allclose(residual, 0., rtol=0., atol=1e-14)

    def test_end_to_end_extraction_and_child_triangle_remapping(self):
        sewing, fold, combined = (self.directory / name for name in ("sewing", "fold", "combined"))
        self.run_cli("spike-cuff-sewing-input.py", "--canonical", self.canonical_path, "--output", sewing,
                     "--normal-offset-frames")
        self.run_cli("spike-cuff-fold-input.py", "--canonical", self.canonical_path, "--output", fold,
                     "--crease", "outer-allowance")
        self.run_cli("spike-cuff-sequence-input.py", "--sewing-input", sewing / "canonical.json",
                     "--fold-input", fold / "canonical.json", "--output", combined)
        for directory in (sewing, fold, combined):
            content = (directory / "canonical.json").read_bytes()
            control = json.loads(content)
            placement = json.loads((directory / "placement.json").read_bytes())
            self.assertEqual(placement["canonicalDigest"], hashlib.sha256(content).hexdigest())
            self.assertEqual(placement["placedMeters"], control["placedMeters"])
            self.assertIs(control["provenance"]["accepted"], False)
            self.assertEqual(control["provenance"]["sourceSha256"], hashlib.sha256(self.content).hexdigest())
        control = json.loads((combined / "canonical.json").read_bytes())
        original = np.asarray(self.source["restMeters"])
        rest = np.asarray(control["restMeters"])
        self.assertEqual(rest.shape, (66, 3))
        self.assertEqual(control["instanceOffsets"], {"cuff_left:shell": 0, "cuff_left:facing": 33})
        np.testing.assert_array_equal(rest[:33], rest[33:])
        np.testing.assert_array_equal(rest[:20], original[:20])
        subdivision = control["provenance"]["subdivision"]
        faces = np.asarray(control["triangles"]).reshape((-1, 3))[:48]
        original_faces = np.asarray(self.source["triangles"]).reshape((-1, 3))[:24]
        for vertex, weights in enumerate(subdivision["sourceWeights"]):
            self.assertAlmostEqual(sum(weights.values()), 1., places=14)
            self.assertTrue(all(0. <= value <= 1. for value in weights.values()))
            mapped = sum(value * original[int(source_vertex)] for source_vertex, value in weights.items())
            np.testing.assert_allclose(mapped, rest[vertex], rtol=0., atol=1e-15)
        for parent, triangle in enumerate(original_faces):
            children = [Polygon(rest[child, :2]) for child, source_parent in zip(faces, subdivision["parentTriangles"])
                        if source_parent == parent]
            expected = Polygon(original[triangle, :2])
            self.assertLess(unary_union(children).symmetric_difference(expected).area, 1e-17)
            self.assertAlmostEqual(sum(child.area for child in children), expected.area, places=16)
        for row, original_row in zip(control["embeddedConstraints"]["constraints"],
                                     self.source["embeddedConstraints"]["constraints"]):
            self.assertEqual(row["registrationId"], original_row["registrationId"])
            for sample, original_sample in zip(row["sourceSamples"], original_row["sourceSamples"]):
                self.assertEqual(sample["originalWeights"], original_sample["weights"])
                weights = sample["weights"]
                position = sum(weight["weight"] * rest[weight["vertex"]] for weight in weights)
                np.testing.assert_allclose(position[:2] * 1000, original_sample["restPositionMm"], rtol=0., atol=1e-10)
                parent = sample["parentTriangle"]
                self.assertTrue({weight["vertex"] for weight in sample["originalWeights"]}
                                <= set(original_faces[parent]))
                self.assertTrue(any({weight["vertex"] for weight in weights} <= set(face)
                                    for face, source_parent in zip(faces, subdivision["parentTriangles"])
                                    if source_parent == parent))
        self.assertEqual(len(control["embeddedConstraints"]["constraints"]), 20)
        self.assertEqual(len(control["foldActuation"]["hinges"]), 24)

    def test_existing_output_is_never_overwritten(self):
        result = self.run_cli("prepare-cuff-source.py", "--output", self.canonical_path.parent, success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.canonical_path.read_bytes(), self.content)

    def test_source_changes_before_publication_reject_without_output(self):
        specification = importlib.util.spec_from_file_location("prepare_cuff_source_review", GENERATOR)
        generator = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(generator)
        original_read = Path.read_bytes
        for name in ("services/engine/shirt.py", "scripts/solver_process_budget.py"):
            output = self.directory / ("changed-" + Path(name).stem)
            reads = 0

            def changed_read(path):
                nonlocal reads
                content = original_read(path)
                if path == ROOT / name:
                    reads += 1
                    if reads > 1:
                        return content + b"\n# changed during generation\n"
                return content

            with self.subTest(source=name), patch.object(Path, "read_bytes", changed_read):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    generator.generate(output)
            self.assertGreaterEqual(reads, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
