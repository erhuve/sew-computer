from fractions import Fraction
import unittest

import numpy as np
import ipctk

from solver_ipc_contact import IpcSurfaceContact
from solver_swept_separation import certified_candidates, swept_plane_separated


class SweptPlaneTests(unittest.TestCase):
    def test_random_certificates_pass_exact_rational_support_oracle(self):
        rng = np.random.default_rng(7531)
        first = rng.normal(size=(800, 4, 3)) * .001
        normals = rng.normal(size=(800, 3))
        shifts = normals / np.linalg.norm(normals, axis=1)[:, None] * rng.uniform(.0001, .01, (800, 1))
        second = rng.normal(size=(800, 6, 3)) * .001 + shifts[:, None]
        translations = rng.uniform(-100, 100, (800, 1, 3))
        first += translations
        second += translations
        minimum = .0001
        certified = swept_plane_separated(first, second, normals, minimum)
        self.assertGreater(int(certified.sum()), 300)
        for a, b, normal, safe in zip(first, second, normals, certified):
            if not safe:
                continue
            n = [Fraction(float(x)) for x in normal]
            def support(points):
                return [sum(Fraction(float(x)) * y for x, y in zip(point, n)) for point in points]
            ap, bp = support(a), support(b)
            gap = max(min(ap) - max(bp), min(bp) - max(ap))
            self.assertGreater(gap, 0)
            self.assertGreater(gap * gap, Fraction(minimum) ** 2 * sum(x * x for x in n))

    def test_crossing_and_tangent_paths_cannot_be_certified(self):
        first = np.array([[[0., 0., 0.], [1., 0., 0.]]])
        second = np.array([[[0., 0., .001], [1., 0., .001],
                            [0., 0., -.001], [1., 0., -.001]]])
        for normal in ([0., 0., 1.], [1., 0., 0.], [0., 0., -10.]):
            self.assertFalse(swept_plane_separated(first, second, [normal], .0001)[0])
        for distance in (.0001, np.nextafter(.0001, 0)):
            second[:, :, 2] = distance
            self.assertFalse(swept_plane_separated(first, second, [[0., 0., 1.]], .0001)[0])

    def test_unrepresentable_direction_falls_back_without_certificate(self):
        first = np.zeros((1, 2, 3))
        second = np.ones((1, 2, 3))
        for direction in ([0., 0., 0.], [np.nan, 0., 0.], [np.inf, 0., 0.],
                          [1e300, 1e300, 1e300], [1e-300, 0., 0.]):
            self.assertFalse(swept_plane_separated(first, second, [direction], .0001)[0])

    def test_invalid_shapes_and_distances_rejected(self):
        for minimum in (0., -1., True, np.nan, np.inf):
            with self.assertRaises(ValueError):
                swept_plane_separated(np.zeros((1, 2, 3)), np.ones((1, 2, 3)), [[0., 0., 1.]], minimum)
        with self.assertRaises(ValueError):
            swept_plane_separated(np.zeros((1, 0, 3)), np.ones((1, 2, 3)), [[0., 0., 1.]], .1)


class SweptContactTests(unittest.TestCase):
    def setUp(self):
        ipctk.set_num_threads(1)
        normal = np.array([1., 2., 3.]) / np.sqrt(14.)
        tangent = np.cross(normal, [1., 0., 0.])
        tangent /= np.linalg.norm(tangent)
        other = np.cross(normal, tangent)
        triangle = np.array([np.zeros(3), .01 * tangent, .01 * other])
        self.normal = normal
        self.rest = np.concatenate((triangle, triangle + .00017 * normal))
        self.faces = np.array([[0, 1, 2], [3, 4, 5]])
        self.contact = IpcSurfaceContact(self.rest, self.faces, activation_distance_m=.00002,
            minimum_distance_m=.0001, stiffness=10000., energy_profile="area-improved-max",
            ccd_profile="swept-plane-tight-inclusion")

    def test_oblique_safe_motion_is_certified_without_changing_energy(self):
        end = self.rest.copy()
        end[3:] -= .00001 * self.normal
        pending, report = certified_candidates(self.contact.mesh, self.rest, end, .0001)
        self.assertGreater(report["candidateCount"], 0)
        self.assertEqual(report["remainingCount"], 0)
        self.assertEqual(len(pending), 0)
        self.assertEqual(self.contact.step_limit(self.rest, end), 1.)
        self.assertTrue(self.contact.path_safe(self.rest, end))
        legacy = IpcSurfaceContact(self.rest, self.faces, activation_distance_m=.00002,
            minimum_distance_m=.0001, stiffness=10000., energy_profile="area-improved-max")
        self.assertEqual(self.contact.energy(end), legacy.energy(end))
        np.testing.assert_array_equal(self.contact.gradient(end), legacy.gradient(end))
        self.assertEqual((self.contact.hessian(end) - legacy.hessian(end)).nnz, 0)
        self.assertEqual(self.contact.profile()["ccd"]["conservativeRescaling"], .8)

    def test_crossing_candidates_retained_for_original_ccd(self):
        end = self.rest.copy()
        end[3:] -= .00034 * self.normal
        _, report = certified_candidates(self.contact.mesh, self.rest, end, .0001)
        self.assertGreater(report["remainingCount"], 0)
        limit = self.contact.step_limit(self.rest, end)
        self.assertGreater(limit, 0.)
        self.assertLess(limit, 1.)
        self.assertFalse(self.contact.path_safe(self.rest, end))

    def test_active_contact_force_unchanged(self):
        active = self.rest.copy()
        active[3:] -= .00006 * self.normal
        legacy = IpcSurfaceContact(self.rest, self.faces, activation_distance_m=.00002,
            minimum_distance_m=.0001, stiffness=10000., energy_profile="area-improved-max")
        self.assertGreater(self.contact.energy(active), 0.)
        np.testing.assert_array_equal(self.contact.gradient(active), legacy.gradient(active))
        self.assertEqual((self.contact.hessian(active) - legacy.hessian(active)).nnz, 0)

    def test_profile_is_immutable_and_unknown_profile_rejected(self):
        with self.assertRaises(AttributeError):
            self.contact.ccd_profile = "tight-inclusion"
        with self.assertRaises(ValueError):
            IpcSurfaceContact(self.rest, self.faces, activation_distance_m=.00002,
                minimum_distance_m=.0001, stiffness=10000., ccd_profile="skip-collisions")


if __name__ == "__main__":
    unittest.main()
