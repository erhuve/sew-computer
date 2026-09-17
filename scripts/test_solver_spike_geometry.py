import unittest

import numpy as np

from solver_spike_geometry import triangles_intersect


class SurfaceOracleTests(unittest.TestCase):
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
