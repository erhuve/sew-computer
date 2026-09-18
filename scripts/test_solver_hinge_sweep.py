import unittest

import numpy as np

from solver_bending import ElasticDihedralBending
from solver_hinge_sweep import hinge_sweep_safe


class HingeSweepTests(unittest.TestCase):
    def setUp(self):
        self.start = np.array([[0.2, 0.8, 0.], [0.6, -0.7, 0.],
                               [0., 0., 0.], [1., 0., 0.]])
        self.indices = np.array([[0, 1, 2, 3]])

    def test_flat_translation_and_small_deformation(self):
        self.assertTrue(hinge_sweep_safe(self.start, self.start + [20., -4., 9.], self.indices))
        end = self.start.copy()
        end[0] += [0.1, 0.1, 0.2]
        self.assertTrue(hinge_sweep_safe(self.start, end, self.indices))

    def test_triangle_collapses_between_valid_endpoints(self):
        end = self.start.copy()
        end[0, 1] = -0.8
        end[1, 1] = 0.7
        bending = ElasticDihedralBending(4, self.indices, [0.], [1.], [1.])
        np.testing.assert_allclose(bending.angles(self.start), bending.angles(end))
        self.assertFalse(hinge_sweep_safe(self.start, end, self.indices))

    def test_edge_collapses_between_valid_endpoints(self):
        end = self.start.copy()
        end[[2, 3]] = end[[3, 2]]
        self.assertFalse(hinge_sweep_safe(self.start, end, self.indices))

    def test_fold_crossing_with_nearly_identical_endpoint_angles(self):
        start = np.array([[0.6119065446162345, -1.6902081011955592, 1.2257051237720422],
                          [-0.2786332348886038, -0.3440301606606076, 0.4312879755572565],
                          [0.5056388638387653, 1.1400702076934885, 0.5516534390872517],
                          [0.943768121726797, -0.4335550133327649, -2.328041629328143]])
        end = np.array([[-0.314076046152558, 2.5096349759519647, -0.052042420751663925],
                        [1.5178414051402414, 0.8472008877288331, 0.754321342549151],
                        [1.907869935829746, 0.3019193050720776, 1.999160599859372],
                        [0.6080432388466356, -0.4301824502215452, 0.49761380966677254]])
        bending = ElasticDihedralBending(4, self.indices, [0.], [1.], [1.])
        self.assertLess(abs(bending.angles(start)[0] - bending.angles(end)[0]), 4e-5)
        angles = [bending.angles(start + fraction * (end - start))[0]
                  for fraction in np.linspace(0., 1., 201)]
        self.assertGreater(np.max(np.abs(np.diff(angles))), 6.)
        self.assertFalse(hinge_sweep_safe(start, end, self.indices))

    def test_near_pi_safe_step_and_branch_crossing(self):
        start = self.start.copy()
        start[1, 1:] = [0.7, 1e-6]
        end = start.copy()
        end[1, 2] = 2e-6
        self.assertTrue(hinge_sweep_safe(start, end, self.indices))
        end[1, 2] = -1e-6
        self.assertFalse(hinge_sweep_safe(start, end, self.indices))

    def test_subdivision_accepts_safe_rotation_and_budget_fails_closed(self):
        angle = np.deg2rad(150.)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        end = self.start @ rotation.T
        self.assertTrue(hinge_sweep_safe(self.start, end, self.indices))
        self.assertFalse(hinge_sweep_safe(self.start, end, self.indices, max_depth=0))
        self.assertFalse(hinge_sweep_safe(self.start, end, self.indices, max_intervals=1))

    def test_scale_frame_and_orientation_covariance(self):
        rotation, _ = np.linalg.qr(np.random.default_rng(815).normal(size=(3, 3)))
        end = self.start.copy()
        end[0, 2] = 0.3
        bad_end = self.start.copy()
        bad_end[[0, 1], 1] *= -1
        for scale in [1e-100, 1e-5, 1., 1e5, 1e100]:
            for mirror in [1., -1.]:
                transform = lambda points: scale * (mirror * points @ rotation.T + [2., -3., 1.])
                self.assertTrue(hinge_sweep_safe(transform(self.start), transform(end), self.indices))
                self.assertFalse(hinge_sweep_safe(transform(self.start), transform(bad_end), self.indices))

    def test_batch_requires_every_hinge(self):
        start = np.concatenate((self.start, self.start + [3., 0., 0.]))
        end = start.copy()
        end[4:6, 1] *= -1
        self.assertFalse(hinge_sweep_safe(start, end, [[0, 1, 2, 3], [4, 5, 6, 7]]))

    def test_empty_and_invalid_inputs(self):
        self.assertTrue(hinge_sweep_safe(self.start, self.start, np.empty((0, 4), dtype=int)))
        for indices in [[[0, 1, 2, 2]], [[0, 1, 2, 4]], [[-1, 1, 2, 3]],
                        [[0., 1., 2., 3.]], [[0, 1, 2]], []]:
            with self.assertRaises(ValueError):
                hinge_sweep_safe(self.start, self.start, indices)
        for budget in [-1, True, 1.5, 25]:
            with self.assertRaises(ValueError):
                hinge_sweep_safe(self.start, self.start, self.indices, max_depth=budget)
        for budget in [0, True, 1.5]:
            with self.assertRaises(ValueError):
                hinge_sweep_safe(self.start, self.start, self.indices, max_intervals=budget)
        for invalid in [np.full((4, 3), np.nan), np.zeros((4, 2)), np.zeros((3, 3))]:
            with self.assertRaises(ValueError):
                hinge_sweep_safe(self.start, invalid, self.indices)
        self.assertFalse(hinge_sweep_safe(np.zeros((4, 3)), np.zeros((4, 3)), self.indices))


if __name__ == "__main__":
    unittest.main()
