import unittest

import numpy as np

from solver_contact_preflight import inspect_contact_placement


class ContactPreflightTests(unittest.TestCase):
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
