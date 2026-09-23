"""Independent geometric and admission tests for saved-state surface gaps."""

import copy
from fractions import Fraction
import math
import unittest
from unittest.mock import patch

import numpy as np

import solver_surface_distance_diagnostics as diagnostic


class SurfaceDistanceDiagnosticTests(unittest.TestCase):
    @staticmethod
    def pair(first, second):
        return diagnostic.minimum_surface_distance([*first, *second], [[0, 1, 2], [3, 4, 5]], [0], [1])

    def assert_distance(self, result, expected):
        self.assertFalse(result["accepted"])
        self.assertAlmostEqual(result["distanceMeters"], expected, delta=max(1e-14*abs(expected), 1e-30))
        self.assertAlmostEqual(math.dist(result["closestPointFirstMeters"], result["closestPointSecondMeters"]),
                               expected, delta=max(2e-14*abs(expected), 1e-30))
        value = result["exactSquaredDistanceMeters2"]
        square = Fraction(int(value["numerator"]), int(value["denominator"]))
        self.assertGreaterEqual(square, 0)
        return square

    def test_vertex_face_minimum_is_not_nearest_vertex(self):
        first = [[.25, .25, 1.], [.5, .25, 2.], [.25, .5, 2.]]
        second = [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]
        result = self.pair(first, second)
        self.assertEqual(self.assert_distance(result, 1.), 1)
        self.assertEqual(result["closestPointFirstMeters"], [.25, .25, 1.])
        self.assertEqual(result["closestPointSecondMeters"], [.25, .25, 0.])
        self.assertGreater(min(math.dist(a, b) for a in first for b in second), 1.)

    def test_skew_edge_edge_minimum_has_both_witnesses_in_edge_interiors(self):
        result = self.pair([[-1., 0., 0.], [1., 0., 0.], [0., -1., -1.]],
                           [[0., -1., 1.], [0., 1., 1.], [1., 0., 2.]])
        self.assertEqual(self.assert_distance(result, 1.), 1)
        self.assertEqual(result["closestPointFirstMeters"], [0., 0., 0.])
        self.assertEqual(result["closestPointSecondMeters"], [0., 0., 1.])

    def test_edge_pierces_face_without_a_vertex_or_edge_endpoint_contact(self):
        result = self.pair([[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.]],
                           [[0., 0., -1.], [0., 0., 1.], [3., 0., 1.]])
        self.assertEqual(self.assert_distance(result, 0.), 0)
        self.assertGreater(result["counts"]["segmentFaceTests"], 0)

    def test_coplanar_containment_and_crossing_are_intersections(self):
        for second in ([[.125, .125, 0.], [.25, .125, 0.], [.125, .25, 0.]],
                       [[-2., 1., 0.], [2., 1., 0.], [0., -2., 0.]]):
            with self.subTest(second=second):
                result = self.pair([[-2., -1., 0.], [2., -1., 0.], [0., 2., 0.]], second)
                self.assertEqual(self.assert_distance(result, 0.), 0)

    def test_collinear_overlap_and_coplanar_disjoint_surfaces(self):
        first = [[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]]
        overlap = [[.5, 0., 0.], [1., 0., 0.], [.5, -1., 0.]]
        self.assertEqual(self.assert_distance(self.pair(first, overlap), 0.), 0)
        separate = [[3., 0., 0.], [4., 0., 0.], [3., 1., 0.]]
        self.assertEqual(self.assert_distance(self.pair(first, separate), 1.), 1)

    def test_positive_near_coplanar_gap_is_not_zeroed_by_tolerance(self):
        first = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
        for gap in (2.**-40, 1e-200, math.ulp(0.)):
            second = [[x, y, gap] for x, y, _ in first]
            with self.subTest(gap=gap):
                result = self.pair(first, second)
                self.assertEqual(result["distanceMeters"], gap)
                value = result["exactSquaredDistanceMeters2"]
                self.assertEqual(Fraction(int(value["numerator"]), int(value["denominator"])), Fraction(gap)**2)

    def test_nonrepresentable_positive_gap_rejects_instead_of_reporting_intersection(self):
        tiny = math.ulp(0.)
        first = [[0., 0., 0.], [-1., 0., 0.], [0., -1., 0.]]
        second = [[tiny, 0., 0.], [-2., 1., 0.], [tiny, 0., 1.]]
        with self.assertRaisesRegex(ValueError, "Positive surface distance"):
            self.pair(first, second)

    def test_rigid_covariance_and_group_swap_preserve_distance(self):
        first = np.array([[-1., 0., 0.], [1., 0., 0.], [0., -1., -1.]])
        second = np.array([[0., -1., 1.], [0., 1., 1.], [1., 0., 2.]])
        rotation = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
        translation = np.array([.125, -.25, 4.])
        before = self.pair(first.tolist(), second.tolist())
        after = self.pair((first @ rotation.T + translation).tolist(),
                          (second @ rotation.T + translation).tolist())
        self.assertEqual(self.assert_distance(after, 1.), 1)
        np.testing.assert_array_equal(after["closestPointFirstMeters"],
                                      np.array(before["closestPointFirstMeters"]) @ rotation.T + translation)
        self.assertEqual(self.pair(second.tolist(), first.tolist())["distanceMeters"], 1.)

    def test_complete_pair_search_finds_later_closest_face_and_accounts_for_pruning(self):
        base = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        positions = np.concatenate([base + [0., 0., z] for z in (0., 10., 1., 20.)])
        result = diagnostic.minimum_surface_distance(positions, np.arange(12).reshape((-1, 3)), [0], [3, 1, 2])
        self.assertEqual(result["secondFaceIndex"], 2)
        self.assertEqual(self.assert_distance(result, 1.), 1)
        counts = result["counts"]
        self.assertEqual(counts["requestedTrianglePairs"], 3)
        self.assertEqual(counts["evaluatedTrianglePairs"] + counts["aabbPrunedTrianglePairs"]
                         + counts["zeroDistancePrunedTrianglePairs"], 3)
        self.assertGreater(counts["aabbPrunedTrianglePairs"], 0)

    def test_intersection_early_exit_cannot_hide_a_degenerate_selected_face(self):
        positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]] * 2 + [[0., 0., 10.], [1., 0., 10.], [2., 0., 10.]]
        with self.assertRaisesRegex(ValueError, "degenerate"):
            diagnostic.minimum_surface_distance(positions, [[0, 1, 2], [3, 4, 5], [6, 7, 8]], [0], [1, 2])

    def test_bounds_reject_before_coordinate_or_exact_geometry_work(self):
        with patch.object(diagnostic, "_number", side_effect=AssertionError("coordinate work before budget")):
            with self.assertRaisesRegex(ValueError, "budget"):
                diagnostic.minimum_surface_distance([[0., 0., 0.]] * 3, [[0, 1, 2]] * 2002,
                                                    list(range(1001)), list(range(1001, 2002)))

    def test_malformed_types_ids_shapes_and_nonfinite_coordinates_reject(self):
        positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 1.], [0., 1., 1.]]
        faces = [[0, 1, 2], [3, 4, 5]]
        attacks = [(positions, faces, [True], [1]), (positions, faces, [0.], [1]),
                   (positions, faces, [0, 0], [1]), (positions, faces, [0], [0]),
                   (positions, faces, [], [1]), (positions, faces, [0], [2]),
                   (positions, [[0, 0, 2], [3, 4, 5]], [0], [1]),
                   (positions, [[False, 1, 2], [3, 4, 5]], [0], [1]),
                   (positions, [[0, 1, 2], [3, 4, 6]], [0], [1]),
                   ([[0., 0.]] * 6, faces, [0], [1]),
                   ([np.array(0.)] * 6, faces, [0], [1])]
        for coordinate in (True, float("nan"), float("inf"), np.int64(2**53+1)):
            changed = copy.deepcopy(positions)
            changed[0][0] = coordinate
            attacks.append((changed, faces, [0], [1]))
        for attack in attacks:
            with self.subTest(attack=repr(attack)[:120]):
                with self.assertRaises(ValueError):
                    diagnostic.minimum_surface_distance(*attack)

    def test_input_result_isolation_and_hash_binding(self):
        first = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
        second = [[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]]
        saved = copy.deepcopy((first, second))
        result = self.pair(first, second)
        expected = copy.deepcopy(result)
        self.assertEqual((first, second), saved)
        result["closestPointFirstMeters"][0] = 20.
        self.assertEqual(self.pair(first, second), expected)
        first[0][0] = .125
        self.assertNotEqual(self.pair(first, second)["inputSha256"], expected["inputSha256"])


if __name__ == "__main__":
    unittest.main()
