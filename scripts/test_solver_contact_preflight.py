import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from solver_contact_preflight import inspect_contact_placement, staging_seam_gaps


class ContactPreflightTests(unittest.TestCase):
    def test_malformed_staging_inputs_leave_rejected_reports(self):
        vertices = [[0., 0, 0], [.1, 0, 0], [0, .1, 0]] * 2
        for mutation in ("negative", "cross-instance", "container", "nonfinite", "nonrigid"):
            canonical = {"restMeters": vertices, "placedMeters": vertices, "triangles": list(range(6)),
                         "instanceOffsets": {"shell": 0, "facing": 3}, "embeddedConstraints": {
                             "constraints": [{"terms": [
                                 {"instanceId": "shell", "vertex": 0, "coefficient": 1.},
                                 {"instanceId": "facing", "vertex": 0, "coefficient": -1.}]}]}}
            term = canonical["embeddedConstraints"]["constraints"][0]["terms"][0]
            if mutation in ("negative", "cross-instance"):
                term["vertex"] = -1 if mutation == "negative" else 3
            elif mutation == "container":
                canonical["embeddedConstraints"] = [1]
            elif mutation == "nonrigid":
                canonical["restMeters"] = [[coordinate * 1.01 for coordinate in vertex] for vertex in vertices]
            else:
                term["coefficient"] = float("nan")
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                source, output = Path(directory) / "input.json", Path(directory) / "output"
                source.write_text(json.dumps(canonical))
                result = subprocess.run([sys.executable, str(Path(__file__).with_name("solver_contact_preflight.py")),
                    "--canonical", str(source), "--output", str(output), "--activation-distance-m", ".001",
                    "--minimum-distance-m", ".0001", "--stiffness", "1e8", "--rigid-clearance-m", ".002"],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr)
                report = json.loads((output / "report.json").read_text())
                self.assertFalse(report["contactAdmissible"])
                self.assertFalse(report["accepted"])
                self.assertTrue(report["rejection"])
                self.assertFalse((output / "staged-placement.json").exists())
                json.dumps(report, allow_nan=False)

    def test_staging_seam_diagnostics_validate_local_anchors(self):
        original = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0]] * 2)
        staged = original.copy()
        staged[3:, 2] += .002
        canonical = {"instanceOffsets": {"shell": 0, "facing": 3}, "embeddedConstraints": {
            "constraints": [{"terms": [{"instanceId": "shell", "vertex": 0, "coefficient": 1.},
                                       {"instanceId": "facing", "vertex": 0, "coefficient": -1.}]}]}}
        report = staging_seam_gaps(canonical, original, staged)
        self.assertEqual(report["beforeMax"], 0.)
        self.assertEqual(report["afterMax"], 2.)
        term = canonical["embeddedConstraints"]["constraints"][0]["terms"][0]
        for vertex in (-1, 3, True, 0.5):
            term["vertex"] = vertex
            with self.subTest(vertex=vertex), self.assertRaisesRegex(ValueError, "instance-local"):
                staging_seam_gaps(canonical, original, staged)
        term["vertex"] = 0
        for coefficient in (float("nan"), float("inf"), True, 0., 2., "1"):
            term["coefficient"] = coefficient
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                staging_seam_gaps(canonical, original, staged)
        for bundle in ([1], {}, {"constraints": [None]}, {"constraints": [{"terms": []}]}):
            canonical["embeddedConstraints"] = bundle
            with self.subTest(bundle=bundle), self.assertRaises(ValueError):
                staging_seam_gaps(canonical, original, staged)

    def inspect(self, gap):
        vertices = np.array([[0., 0, 0], [.1, 0, 0], [0, .1, 0],
                             [.02, .02, gap], [.12, .02, gap], [.02, .12, gap]])
        return inspect_contact_placement({"restMeters": vertices, "placedMeters": vertices,
                                           "triangles": list(range(6))}, activation_distance_m=.001,
                                          minimum_distance_m=.0001, stiffness=1e8)

    def test_separated_layers_are_admissible_but_never_accepted_garments(self):
        report = self.inspect(.0005)
        self.assertTrue(report["contactAdmissible"])
        self.assertFalse(report["accepted"])
        self.assertAlmostEqual(report["minimumActiveCandidateDistanceM"], .0005)
        self.assertEqual(report["independentSurfaceOracle"]["intersectingPairCount"], 0)

    def test_coplanar_overlap_rejects_with_independent_evidence(self):
        report = self.inspect(0.)
        self.assertFalse(report["contactAdmissible"])
        self.assertEqual(report["independentSurfaceOracle"]["intersectingPairCount"], 1)
        self.assertEqual(report["minimumActiveCandidateDistanceM"], 0.)

    def test_subthickness_gap_rejects_even_without_surface_intersection(self):
        report = self.inspect(.00005)
        self.assertFalse(report["contactAdmissible"])
        self.assertEqual(report["independentSurfaceOracle"]["intersectingPairCount"], 0)
        self.assertIn("minimum surface separation", report["rejection"])

    def test_no_active_candidates_does_not_claim_measured_global_clearance(self):
        report = self.inspect(.1)
        self.assertTrue(report["contactAdmissible"])
        self.assertIsNone(report["minimumActiveCandidateDistanceM"])


if __name__ == "__main__":
    unittest.main()
