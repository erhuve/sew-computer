import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from solver_rigid_staging import separate_rigid_instances


class RigidStagingTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0., 0, 0], [.1, 0, 0], [0, .1, 0]] * 2)
        self.faces = np.arange(6).reshape((2, 3))
        self.offsets = {"shell": 0, "facing": 3}

    def stage(self, **kwargs):
        return separate_rigid_instances(self.points, self.faces, self.offsets, .002, **kwargs)

    def test_coincident_layers_separate_without_changing_edges_or_input(self):
        before = self.points.copy()
        staged, report = self.stage()
        assert_array_equal(self.points, before)
        assert_allclose(staged.mean(axis=0), before.mean(axis=0), atol=1e-12)
        self.assertGreaterEqual(abs(staged[3, 2] - staged[0, 2]), .002)
        self.assertFalse(report["accepted"])
        self.assertLess(report["maxEdgeVectorErrorM"], 1e-12)
        assert_allclose(staged[1:3] - staged[0], before[1:3] - before[0], atol=1e-12)

    def test_already_separated_is_unchanged(self):
        self.points[3:, 2] = .01
        staged, _ = self.stage()
        assert_array_equal(staged, self.points)

    def test_transverse_sheets_and_translation(self):
        self.points[3:] = [[.02, .02, -.02], [.02, .02, .02], [.08, .02, .02]]
        staged, report = self.stage()
        self.assertGreaterEqual(report["minimumSeparatingPlaneGapM"], .002)
        self.points += [10, -20, 30]
        moved, _ = self.stage()
        assert_allclose(moved - [10, -20, 30], staged, atol=1e-10)

    def test_three_identical_layers_have_pairwise_clearance(self):
        self.points = np.tile(self.points[:3], (3, 1))
        self.faces = np.arange(9).reshape((3, 3))
        self.offsets["interfacing"] = 6
        staged, report = self.stage()
        self.assertEqual(report["pairCount"], 3)
        self.assertTrue(np.all(np.diff(np.sort(staged[::3, 2])) >= .002))

    def test_instance_reordering_preserves_identity_translations(self):
        staged, report = self.stage()
        reordered, reordered_report = separate_rigid_instances(self.points, self.faces,
            {"facing": 0, "shell": 3}, .002)
        for identity in self.offsets:
            assert_allclose(report["translationsM"][identity], reordered_report["translationsM"][identity], atol=1e-12)
        assert_allclose(staged[:3], reordered[3:], atol=1e-12)

    def test_budget_rejects(self):
        with self.assertRaisesRegex(ValueError, "translation budget"):
            self.stage(max_translation_m=.0001)

    def test_overflow_and_malformed_distances_reject(self):
        for value in (True, "0.002", 1j, np.inf, np.nan, 1e-300):
            with self.subTest(value=value), self.assertRaises(ValueError):
                separate_rigid_instances(self.points, self.faces, self.offsets, value)
        self.points.fill(1e308)
        with self.assertRaises(ValueError):
            self.stage()

    def test_duplicate_and_degenerate_faces_reject(self):
        self.faces = np.vstack((self.faces, self.faces[0]))
        with self.assertRaisesRegex(ValueError, "nondegenerate"):
            self.stage()

    def test_invalid_partition_and_cross_instance_faces_reject(self):
        for offsets in ({"shell": 1, "facing": 3}, {"shell": 0, "facing": 0},
                        {"shell": False, "facing": 3}, {"shell": 0, "facing": 5}):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                separate_rigid_instances(self.points, self.faces, offsets, .002)
        self.faces[0, 2] = 3
        with self.assertRaises(ValueError):
            self.stage()

    def test_nonplanar_and_nonfinite_input_reject(self):
        self.points[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.stage()
        points = np.array([[0., 0, 0], [.1, 0, 0], [0, .1, 0], [.1, .1, .02]] * 2)
        with self.assertRaisesRegex(ValueError, "planar"):
            separate_rigid_instances(points, np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]]),
                                     {"shell": 0, "facing": 4}, .002)


if __name__ == "__main__":
    unittest.main()
