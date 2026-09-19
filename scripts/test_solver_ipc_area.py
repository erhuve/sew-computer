import unittest

import ipctk
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from solver_ipc_contact import IpcSurfaceContact


class AreaContactTests(unittest.TestCase):
    def contact(self, rest, faces, pressure=10000.):
        return IpcSurfaceContact(rest, faces, activation_distance_m=.001,
                                 minimum_distance_m=.0001, stiffness=pressure,
                                 energy_profile="area-improved-max")

    def layers(self):
        layer = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
        rest = np.vstack([layer, layer])
        placed = rest.copy()
        placed[3:] += [.0003, .0002, .0005]
        return rest, placed, np.array([[0, 1, 2], [3, 4, 5]])

    def test_pressure_units_and_profile_are_explicit(self):
        rest, placed, faces = self.layers()
        contact = self.contact(rest, faces)
        profile = contact.profile()
        self.assertEqual(profile["stiffnessUnits"], "Pa")
        self.assertTrue(profile["physicalBarrier"])
        self.assertTrue(profile["areaWeighting"])
        self.assertTrue(contact.requires_guarded_metric)
        self.assertFalse(profile["accepted"])
        legacy_potential = ipctk.BarrierPotential(.001, 10000., use_physical_barrier=False)
        _, collisions = contact._collisions(placed)
        squared_gap = .001 * (.001 + 2 * .0001)
        expected = legacy_potential(collisions, contact.mesh, placed) * .001 / squared_gap ** 2
        self.assertAlmostEqual(contact.energy(placed), expected, places=12)

    def test_rebuilt_offset_layer_energy_and_hessian_derivatives(self):
        rest, placed, faces = self.layers()
        contact = self.contact(rest, faces)
        gradient = contact.gradient(placed)
        hessian = contact.hessian(placed)
        generator = np.random.default_rng(5123)
        for _ in range(6):
            direction = generator.normal(size=placed.shape)
            direction /= np.linalg.norm(direction)
            epsilon = 1e-10
            plus, minus = placed + epsilon * direction, placed - epsilon * direction
            derivative = (contact.energy(plus) - contact.energy(minus)) / (2 * epsilon)
            assert_allclose(derivative, np.sum(gradient * direction), rtol=2e-6, atol=1e-7)
            numerical = (contact.gradient(plus) - contact.gradient(minus)).ravel() / (2 * epsilon)
            assert_allclose(numerical, hessian @ direction.ravel(), rtol=2e-5, atol=1e-3)
        assert_allclose(hessian.toarray(), hessian.toarray().T, atol=1e-8)

    def test_rigid_covariance_and_immutable_area_weights(self):
        rest, placed, faces = self.layers()
        contact = self.contact(rest, faces)
        rotation = np.linalg.qr(np.random.default_rng(61).normal(size=(3, 3)))[0]
        transformed = self.contact(rest @ rotation + [.2, -.1, .3], faces)
        moved = placed @ rotation + [.2, -.1, .3]
        assert_allclose(transformed.energy(moved), contact.energy(placed), rtol=1e-9)
        assert_allclose(transformed.gradient(moved), contact.gradient(placed) @ rotation, rtol=1e-8, atol=1e-8)
        assert_array_equal(contact.rest_positions, rest)
        assert_allclose(contact.gradient(placed).sum(axis=0), 0., atol=1e-10)

    def test_separate_layers_and_local_fold_keep_continuous_contact(self):
        rest, placed, faces = self.layers()
        fan_rest = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.], [.002, .002, 0.]])
        fan_placed = fan_rest.copy()
        fan_placed[3] = [.008, -.002, .0005]
        for source, start, topology, moving in (
                (rest, placed, faces, slice(3, None)),
                (fan_rest, fan_placed, np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]]), 3)):
            contact = self.contact(source, topology)
            self.assertGreater(contact.energy(start), 0.)
            end = start.copy()
            end[moving, 2] = -.0005
            self.assertGreater(contact.step_limit(start, end), 0.)
            self.assertLess(contact.step_limit(start, end), 1.)
            self.assertFalse(contact.path_safe(start, end))

    def test_legacy_default_and_invalid_profile(self):
        rest, placed, faces = self.layers()
        legacy = IpcSurfaceContact(rest, faces, activation_distance_m=.001,
                                   minimum_distance_m=.0001, stiffness=1e8)
        self.assertFalse(legacy.requires_guarded_metric)
        self.assertFalse(legacy.profile()["physicalBarrier"])
        self.assertGreater(legacy.energy(placed), 0.)
        with self.assertRaises(ValueError):
            IpcSurfaceContact(rest, faces, activation_distance_m=.001,
                              minimum_distance_m=.0001, stiffness=1e8, energy_profile="unknown")

    def test_energy_mode_cannot_drift_after_native_potential_construction(self):
        rest, _, faces = self.layers()
        contact = self.contact(rest, faces)
        with self.assertRaises(AttributeError):
            contact.energy_profile = "legacy"
        with self.assertRaises(AttributeError):
            contact.requires_guarded_metric = False

    def test_unrepresentable_physical_normalization_rejects(self):
        rest, _, faces = self.layers()
        with self.assertRaises(ValueError), np.errstate(all="raise"):
            IpcSurfaceContact(rest, faces, activation_distance_m=1e-100,
                              minimum_distance_m=1e-100, stiffness=1e300,
                              energy_profile="area-improved-max")


if __name__ == "__main__":
    unittest.main()
