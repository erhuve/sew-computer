"""Independent regressions for the reviewed per-pair contact law and its limits."""

import unittest

import ipctk
import numpy as np

from solver_rest_filtered_contact import RestFilteredSurfaceContact
from solver_temporal_separation import _GROUPS
from test_solver_contact_range_adversarial import square_grid


def contact(rest, faces):
    return RestFilteredSurfaceContact(rest, faces, activation_distance_m=.01,
        minimum_distance_m=.0001, stiffness=10000.,
        ccd_profile="temporal-separation-tight-inclusion")


class RestFilteredLocalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def test_distant_refinement_preserves_existing_energy_force_and_curvature(self):
        rest, faces = square_grid(2)
        positions = rest * .4
        positions[4, 2] = .000015
        original = contact(rest, faces)
        energy, gradient, hessian = (original.energy(positions), original.gradient(positions),
                                     original.hessian(positions).toarray())
        self.assertGreater(energy, 0.)
        self.assertGreater(np.linalg.norm(gradient), 1e-4)
        # These disconnected stationary components never enter contact with the
        # original panel. v1's global minimum scale erased its force entirely.
        for subdivisions, width in ((2, .00001), (4, .00001), (2, .002)):
            with self.subTest(subdivisions=subdivisions, width=width):
                remote, remote_faces = square_grid(subdivisions, width=width)
                remote += [1., 1., 0.]
                combined = contact(np.vstack((rest, remote)),
                                   np.vstack((faces, remote_faces + len(rest))))
                state = np.vstack((positions, remote))
                combined.validate_state(state)
                np.testing.assert_allclose(combined.energy(state), energy, rtol=2e-14, atol=0.)
                np.testing.assert_allclose(combined.gradient(state)[:len(rest)], gradient,
                                           rtol=1e-12, atol=1e-15)
                np.testing.assert_array_equal(combined.gradient(state)[len(rest):], 0.)
                matrix = combined.hessian(state).toarray()
                np.testing.assert_allclose(matrix[:positions.size, :positions.size], hessian,
                                           rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(matrix[:positions.size, positions.size:], 0.)
                np.testing.assert_array_equal(matrix[positions.size:, :positions.size], 0.)
                for key, index in original._filtered_parameter_indices.items():
                    other = combined._filtered_parameter_indices[key]
                    self.assertEqual(original._contact_parameters[index],
                                     combined._contact_parameters[other])
                with self.assertRaisesRegex(ValueError, "minimum separation"):
                    combined.validate_state(np.vstack((rest * .15, remote)))
        with self.assertRaisesRegex(ValueError, "minimum separation"):
            original.validate_state(rest * .15)

    def test_partition_is_complete_and_scales_belong_to_each_rest_pair(self):
        rest, faces = square_grid(2)
        model = contact(rest, faces)
        candidates = model._candidates(rest, rest)
        expected = {}
        for name in _GROUPS:
            for candidate in getattr(candidates, name):
                key = model._key(name, candidate)
                distance = float(np.sqrt(candidate.compute_distance(
                    candidate.dof(rest, model._edges, faces))))
                expected[key] = (distance / 4, min(.0001, distance / 4))
                self.assertEqual(model.minimum_distance_for_candidate(name, candidate), expected[key][1])
        self.assertGreater(len(set(expected.values())), 1)
        observed = {}
        for index, group in model._partition(candidates):
            for name in _GROUPS:
                for candidate in getattr(group, name):
                    key = model._key(name, candidate)
                    self.assertNotIn(key, observed)
                    observed[key] = model._contact_parameters[index]
        self.assertEqual(observed, expected)
        profile = model.profile()
        self.assertEqual(profile["adapter"], "experimental-rest-filtered-contact-v2")
        self.assertEqual(profile["filteredPrimitivePairs"], len(expected))
        self.assertEqual(len(profile["filteredParametersSha256"]), 64)
        for field in ("_filtered_parameter_indices", "_contact_parameters", "_potentials"):
            with self.assertRaises(AttributeError):
                setattr(model, field, None)
        with self.assertRaises(TypeError):
            model._filtered_parameter_indices[next(iter(expected))] = 0

    def test_feature_transition_has_continuous_force_and_piecewise_curvature(self):
        rest, faces = square_grid(2)
        model = contact(rest, faces)
        positions = rest * .4
        positions[4, 2] = .000015
        direction = np.random.default_rng(712).normal(size=positions.shape)
        direction /= np.linalg.norm(direction)
        center_force = model.gradient(positions)
        epsilon = 1e-10
        left = model.gradient(positions - epsilon * direction)
        right = model.gradient(positions + epsilon * direction)
        first_derivative = (model.energy(positions + epsilon * direction)
                            - model.energy(positions - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(first_derivative, np.sum(center_force * direction),
                                   rtol=1e-5, atol=1e-9)
        self.assertLess(np.linalg.norm(right - center_force), 2e-8)
        self.assertLess(np.linalg.norm(left - center_force), 2e-8)
        # At a closest-feature transition there is no unique ordinary Hessian.
        # Keep this limitation visible rather than loosening the smooth test.
        left_slope = (center_force - left).ravel() / epsilon
        right_slope = (right - center_force).ravel() / epsilon
        self.assertGreater(np.linalg.norm(left_slope - right_slope), 1.)
        for sign in (-1., 1.):
            state = positions + sign * 1e-7 * direction
            step = 1e-11
            derivative = (model.gradient(state + step * direction)
                          - model.gradient(state - step * direction)).ravel() / (2 * step)
            np.testing.assert_allclose(model.hessian(state) @ direction.ravel(), derivative,
                                       rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
