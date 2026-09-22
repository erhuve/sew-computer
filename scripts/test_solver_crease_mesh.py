import unittest

import numpy as np

from solver_crease_mesh import split_straight_crease


class CreaseMeshTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.line = np.array([[0., .75], [2., .75]])

    def test_subdivision_preserves_source_metric_and_parent_correspondence(self):
        result = split_straight_crease(self.points, self.faces, self.line)
        points = np.asarray(result["vertices"])
        np.testing.assert_array_equal(points[:4], self.points)
        for index, weights in enumerate(result["sourceWeights"]):
            np.testing.assert_allclose(points[index], sum(weight * self.points[source]
                for source, weight in weights.items()), atol=1e-14)
        self.assertEqual(len(result["triangles"]), 6)
        self.assertEqual(len(result["creaseEdges"]), 2)
        for face, parent in zip(result["triangles"], result["parentTriangles"]):
            self.assertTrue(all(set(result["sourceWeights"][vertex]).issubset(self.faces[parent]) for vertex in face))

    def test_rotation_translation_and_existing_edge(self):
        baseline = split_straight_crease(self.points, self.faces, self.line)
        angle = .731
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shifted = split_straight_crease(self.points @ rotation + [3., -2.], self.faces,
                                       self.line @ rotation + [3., -2.])
        self.assertEqual(baseline["triangles"], shifted["triangles"])
        np.testing.assert_allclose(np.asarray(baseline["vertices"]) @ rotation + [3., -2.], shifted["vertices"])
        existing = split_straight_crease(self.points, self.faces, self.points[[0, 2]])
        self.assertEqual(existing["triangles"], self.faces.tolist())
        self.assertEqual(existing["creaseEdges"], [(0, 2)])

    def test_invalid_topology_and_noninterior_creases_reject(self):
        for faces in (self.faces.astype(float), [[0, 1, 4]], [[0, 0, 2]],
                      [[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [3, 2, 0]]):
            with self.subTest(faces=faces), self.assertRaises(ValueError):
                split_straight_crease(self.points, faces, self.line)
        for line in ([[0., 2.], [2., 2.]], [[0., 0.], [2., 0.]], [[0., 0.], [0., 0.]],
                     [[0., float("nan")], [2., 0.]]):
            with self.subTest(line=line), self.assertRaises(ValueError):
                split_straight_crease(self.points, self.faces, line)

    def test_t_junction_rejects(self):
        points = [[0., 0.], [2., 0.], [2., 1.], [0., 1.], [1., .5]]
        with self.assertRaisesRegex(ValueError, "shared"):
            split_straight_crease(points, [[0, 1, 2], [0, 4, 3], [4, 2, 3]], self.line)


if __name__ == "__main__":
    unittest.main()
