"""Geometric controls for the cuff diagnostic, not an accepted cuff recipe.

The compiler regenerates a synthetic cuff and its source assembly operations.
These tests use its actual cut triangles and material registrations, without
running dynamics. They establish bounds only for equally oriented, congruent
isometric folds separated by a translation; they do not prove that every
deformable cuff configuration is infeasible.
"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from solver_bending import ElasticDihedralBending
from solver_cuff_sequence import combine_cuff_controls
import test_solver_cuff_crease_cli as cuff_crease_cli

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services/engine"))
from assembly import compile_assembly, compile_inventory
from cloth_domain import mesh_cloth_domain
from embedded_constraints import build_embedded_constraints, validate_embedded_constraints
from meshing_test import CONSTRUCTION, shirt_pattern


class CuffCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specification = importlib.util.spec_from_file_location("cuff_source_registrations", ROOT / "scripts/spike-full-shirt.py")
        full_shirt = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(full_shirt)
        pattern, inventory = compile_inventory(json.dumps(shirt_pattern()).encode(), CONSTRUCTION)
        graph = compile_assembly(pattern, inventory)
        identities = {"cuff_left:shell", "cuff_left:facing"}
        operations = [operation for operation in graph["operations"]
                      if {member["instanceId"] for member in operation["participants"]} == identities]
        if {operation["id"] for operation in operations} != {
                "perimeter:cuff_left:" + name for name in ("extension", "end", "outer", "start")}:
            raise AssertionError("Expected four source cuff perimeter operations")
        source_panel = next(panel for panel in pattern["panels"] if panel["id"] == "cuff_left")
        mesh = mesh_cloth_domain(source_panel, 60, quality_refinement=True)
        sources = {identity: {"panel": source_panel, "mesh": mesh} for identity in sorted(identities)}
        registrations = full_shirt.shirt_embedded_registrations(operations, sources)
        bundle = build_embedded_constraints(sources, registrations)
        validate_embedded_constraints(sources, bundle)
        panel = np.column_stack((np.asarray(mesh["restPositions"]) * .001, np.zeros(len(mesh["restPositions"]))))
        faces = np.asarray(mesh["triangles"])
        source = {"restMeters": np.concatenate((panel, panel)).tolist(),
            "placedMeters": np.concatenate((panel, panel + [0., 0., .002])).tolist(),
            "triangles": np.concatenate((faces, faces + len(panel))).ravel().tolist(),
            "instanceOffsets": {"cuff_left:shell": 0, "cuff_left:facing": len(panel)},
            "sourceTemplates": {"cuff_left": mesh}, "embeddedConstraints": bundle}
        fixture = cuff_crease_cli.CuffCreaseCliTests()
        with tempfile.TemporaryDirectory() as directory:
            result, output = fixture.run_extract(source, directory)
            if result.returncode:
                raise RuntimeError(result.stderr)
            fold = json.loads((output / "canonical.json").read_text())
        sewing = {**source, "provenance": {"sourceSha256": fold["provenance"]["sourceSha256"]}}
        cls.sewing_source, cls.fold_source = sewing, fold
        # These historical incompatibility bounds explicitly use allowance
        # frames at crease anchors, rather than an implicit first child.
        cls.control = combine_cuff_controls(sewing, fold, crease_frame_region="allowance")
        cls.panel = np.asarray(fold["restMeters"])
        cls.faces = np.asarray(fold["triangles"]).reshape((-1, 3))
        cls.crease_y = fold["provenance"]["subdivision"]["sourceStitchPath"]["samples"][0]["restPosition"][1] * .001

    def folded_panel(self, angle):
        """Exact isometry on each side; positive source hinge bends toward -Z."""
        result = self.panel.copy()
        allowance = self.panel[:, 1] > self.crease_y
        distance = self.panel[allowance, 1] - self.crease_y
        result[allowance, 1] = self.crease_y + distance * np.cos(angle)
        result[allowance, 2] = -distance * np.sin(angle)
        return result

    def frame_normals(self, panel):
        faces = np.asarray(self.control["sewingFrames"]["faces"]) - len(panel)
        points = panel[faces]
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        return normals / np.linalg.norm(normals, axis=1)[:, None]

    def test_equal_signed_targets_are_exact_isometries_for_both_layers(self):
        edges = np.unique(np.sort(self.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1), axis=0)
        rest_lengths = np.linalg.norm(self.panel[edges[:, 1]] - self.panel[edges[:, 0]], axis=1)
        hinges = self.control["foldActuation"]["hinges"]
        angles = ElasticDihedralBending(2 * len(self.panel), hinges, np.zeros(len(hinges)),
                                        np.ones(len(hinges)), 1.)
        for angle in (-1.2, 0., 1.2):
            with self.subTest(angle=angle):
                panel = self.folded_panel(angle)
                lengths = np.linalg.norm(panel[edges[:, 1]] - panel[edges[:, 0]], axis=1)
                np.testing.assert_allclose(lengths, rest_lengths, rtol=0., atol=1e-16)
                layers = np.concatenate((panel, panel + [0., 0., .002]))
                np.testing.assert_allclose(angles.angles(layers), angle, rtol=0., atol=2e-14)

    def test_outer_and_side_frames_require_different_layer_translations(self):
        rows = self.control["embeddedConstraints"]["constraints"]
        outer = next(index for index, row in enumerate(rows)
                     if row["sourceSamples"][0]["pathName"] == "outer" and row["fraction"] == .5)
        side = next(index for index, row in enumerate(rows)
                    if row["sourceSamples"][0]["pathName"] == "start" and row["fraction"] == .5)
        for angle in (-1.2, 0., 1.2):
            panel = self.folded_panel(angle)
            normals = self.frame_normals(panel)
            expected_allowance = np.array([0., np.sin(angle), np.cos(angle)])
            np.testing.assert_allclose(normals[outer], expected_allowance, rtol=0., atol=1e-14)
            np.testing.assert_allclose(normals[side], [0., 0., 1.], rtol=0., atol=1e-14)
            for offset in (.00011, .0005):
                with self.subTest(angle=angle, offset=offset):
                    targets = -offset * normals[[outer, side]]
                    # Equally oriented congruent layers have one translation.
                    # By the triangle inequality one of these seam residuals
                    # is >= half their target separation; their mean attains it.
                    lower_bound = offset * abs(np.sin(angle / 2))
                    np.testing.assert_allclose(np.linalg.norm(targets[0] - targets[1]) / 2,
                                               lower_bound, rtol=0., atol=1e-16)
                    best_translation = targets.mean(axis=0)
                    np.testing.assert_allclose(np.linalg.norm(best_translation - targets, axis=1),
                                               lower_bound, rtol=0., atol=1e-16)
                    # Directly reconstruct both remapped material anchors,
                    # independently of the sewing-potential implementation.
                    layers = {"cuff_left:shell": panel + best_translation,
                              "cuff_left:facing": panel}
                    for row in (rows[outer], rows[side]):
                        difference = sum(term["coefficient"] * layers[term["instanceId"]][term["vertex"]]
                                         for term in row["terms"])
                        np.testing.assert_allclose(difference, best_translation, rtol=0., atol=1e-16)
        self.assertGreater(.0005 * np.sin(.6), .00028)

    def test_allowance_normal_offset_loses_body_clearance(self):
        minimum = .0001
        for angle in (-1.2, 1.2):
            allowance_normal = np.array([0., np.sin(angle), np.cos(angle)])
            for offset, feasible_body_gap in ((.00011, False), (.0005, True)):
                with self.subTest(angle=angle, offset=offset):
                    shell = self.folded_panel(angle)
                    facing = shell + offset * allowance_normal
                    body = self.panel[:, 1] < self.crease_y - 1e-12
                    # The overlapping planar body interiors have this exact
                    # perpendicular distance, despite correct outer seam gap.
                    np.testing.assert_allclose(shell[body, 2], 0., rtol=0., atol=1e-16)
                    gap = offset * np.cos(angle)
                    np.testing.assert_allclose(facing[body, 2], gap, rtol=0., atol=1e-16)
                    # A material point far from all boundaries belongs to both
                    # projected body domains after the small rigid translation.
                    probe = np.array([.12, .0275])
                    facing_probe = probe - offset * allowance_normal[:2]
                    body_faces = self.faces[np.all(self.panel[self.faces, 1] <= self.crease_y + 1e-12, axis=1)]
                    for coordinate in (probe, facing_probe):
                        self.assertTrue(any(np.min(np.linalg.solve(
                            np.vstack((self.panel[face, :2].T, np.ones(3))), [*coordinate, 1.])) >= -1e-12
                            for face in body_faces))
                    self.assertEqual(gap > minimum, feasible_body_gap)
        self.assertAlmostEqual(.00011 * np.cos(1.2) * 1000, .0398593529924341)
        self.assertAlmostEqual(np.degrees(np.arccos(minimum / .00011)), 24.619977328657107)


if __name__ == "__main__":
    unittest.main()
