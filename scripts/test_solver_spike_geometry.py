import json
import unittest

import numpy as np

from solver_spike_geometry import state_finiteness, surface_intersections, triangles_intersect


class SurfaceOracleTests(unittest.TestCase):
    def test_nonfinite_state_reports_remain_json_serializable(self):
        positions = np.zeros((2, 3))
        velocities = np.zeros((2, 3))
        self.assertTrue(state_finiteness(positions, velocities)["finite"])
        positions[0, 1] = np.nan
        velocities[1, 0] = np.inf
        report = state_finiteness(positions, velocities)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), {
            "finite": False, "nonfinitePositionValues": 1, "nonfiniteVelocityValues": 1})
        self.assertTrue(np.isnan(positions[0, 1]))
        with self.assertRaises(ValueError):
            state_finiteness(positions, np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            state_finiteness([], [])

    def test_broad_phase_padding_covers_diagonal_scaled_oracle_tolerance(self):
        first = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        second = first + [0, 0, 1.2e-9]
        self.assertTrue(triangles_intersect(first, second))
        self.assertEqual(surface_intersections(np.concatenate([first, second]), [[0, 1, 2], [3, 4, 5]])["intersectingPairCount"], 1)

    def test_broad_phase_preserves_brute_force_intersections(self):
        positions = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                              [.5, .5, -1], [.5, .5, 1], [2, 2, 0],
                              [10, 10, 10], [12, 10, 10], [10, 12, 10]], dtype=float)
        faces = np.arange(9).reshape((3, 3))
        for scale in (.001, 1, 1000):
            moved = positions * scale + [1, -2, 3]
            expected = sum(triangles_intersect(moved[first], moved[second]) for index, first in enumerate(faces) for second in faces[index + 1:])
            result = surface_intersections(moved, faces)
            self.assertEqual(result["intersectingPairCount"], expected)
            self.assertEqual(result["testedCandidates"], 1)
        with self.assertRaises(ValueError):
            surface_intersections(positions, [[0, 0, 1]])
        with self.assertRaises(ValueError):
            surface_intersections(positions, faces, candidate_budget=0)

    def test_crossing_without_inside_vertices(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        second = np.array([[0.5, 0.5, -1.], [0.5, 0.5, 1.], [2., 2., 0.]])
        self.assertTrue(triangles_intersect(first, second))

    def test_coplanar_overlap_and_boundary_only(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        self.assertTrue(triangles_intersect(first, first + [0.1, 0.1, 0.]))
        self.assertFalse(triangles_intersect(first, first + [2., 0., 0.]))

    def test_separated_and_rigid_transform(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        self.assertFalse(triangles_intersect(first, first + [0., 0., 0.01]))
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertTrue(triangles_intersect(first @ rotation + 8, (first + [0.1, 0.1, 0.]) @ rotation + 8))

    def test_millimetre_crossing_scale_and_out_of_plane_rotation(self):
        first = np.array([[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.]])
        second = np.array([[.00025, .00025, -.00001], [.00025, .00025, .00001], [.00075, .00075, 0.]])
        cosine, sine = np.cos(.7), np.sin(.7)
        rotation = np.array([[1., 0., 0.], [0., cosine, -sine], [0., sine, cosine]])
        for scale in (.001, 1., 1000.):
            self.assertTrue(triangles_intersect(first * scale, second * scale))
            self.assertTrue(triangles_intersect(first @ rotation * scale + .5, second @ rotation * scale + .5))
            self.assertFalse(triangles_intersect(first * scale, (first + [0., 0., .00001]) * scale))

    def test_coplanar_tiny_overlap_far_from_origin(self):
        first = np.array([[0., 0., 0.], [1e-6, 0., 0.], [0., 1e-6, 0.]])
        second = first + [1e-7, 1e-7, 0.]
        self.assertTrue(triangles_intersect(first, second))
        self.assertTrue(triangles_intersect(first + [100, 200, 300], second + [100, 200, 300]))


if __name__ == "__main__":
    unittest.main()
