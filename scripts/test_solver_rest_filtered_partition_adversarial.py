"""Independent checks of per-pair IPC aggregation and native-state drift."""

import unittest

import ipctk
import numpy as np

from solver_rest_filtered_contact import RestFilteredSurfaceContact
from solver_temporal_separation import _GROUPS
from test_solver_contact_range_adversarial import square_grid


class RestFilteredPartitionAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def fixture(self, profile="temporal-separation-tight-inclusion"):
        rest, faces = square_grid(2)
        contact = RestFilteredSurfaceContact(rest, faces, activation_distance_m=.01,
            minimum_distance_m=.0001, stiffness=10000., ccd_profile=profile)
        positions = rest * .4
        positions[4, 2] = .000015
        return rest, faces, contact, positions

    def test_parameter_buckets_equal_original_candidate_sum_across_feature_changes(self):
        rest, faces, contact, positions = self.fixture()
        direction = np.random.default_rng(712).normal(size=positions.shape)
        direction /= np.linalg.norm(direction)
        features = []
        for offset in (-1e-7, 0., 1e-7):
            with self.subTest(offset=offset):
                state = positions + offset * direction
                candidates = contact._candidates(state, state)
                # This small panel lies entirely inside the full broad-phase
                # range: enumerate every nonincident primitive geometrically,
                # without trusting the adapter's bucket counts.
                expected_faces = {(face, vertex) for face, vertices in enumerate(faces)
                                  for vertex in range(len(rest)) if vertex not in vertices}
                expected_edges = {(first, second) for first in range(len(contact._edges))
                                  for second in range(first + 1, len(contact._edges))
                                  if not set(contact._edges[first]) & set(contact._edges[second])}
                self.assertEqual({(candidate.face_id, candidate.vertex_id)
                                  for candidate in candidates.fv_candidates}, expected_faces)
                self.assertEqual({tuple(sorted((candidate.edge0_id, candidate.edge1_id)))
                                  for candidate in candidates.ee_candidates}, expected_edges)
                self.assertEqual(len(candidates), len(expected_faces) + len(expected_edges))
                energy, gradient = 0., np.zeros_like(state)
                hessian = np.zeros((state.size, state.size))
                reduced_features = {}
                for name in _GROUPS:
                    for candidate in getattr(candidates, name):
                        # Derive parameters from this original rest primitive,
                        # bypassing both the adapter's parameter map and merge.
                        distance = np.sqrt(candidate.compute_distance(candidate.dof(rest, contact._edges, faces)))
                        self.assertGreater(distance, 0.)
                        self.assertLess(distance, .0101)
                        activation, minimum = distance / 4, min(.0001, distance / 4)
                        single = ipctk.Candidates()
                        setattr(single, name, [candidate])
                        collisions = ipctk.NormalCollisions()
                        collisions.use_area_weighting = True
                        collisions.collision_set_type = ipctk.NormalCollisions.IPC
                        collisions.build(single, contact.mesh, state, activation, minimum)
                        potential = ipctk.BarrierPotential(activation, 10000., use_physical_barrier=True)
                        energy += potential(collisions, contact.mesh, state)
                        gradient += np.asarray(potential.gradient(collisions, contact.mesh, state)).reshape(state.shape)
                        hessian += potential.hessian(collisions, contact.mesh, state,
                                                     ipctk.PSDProjectionMethod.NONE).toarray()
                        feature = (name, *(len(getattr(collisions, kind)) for kind in
                                   ("vv_collisions", "ev_collisions", "ee_collisions", "fv_collisions")))
                        reduced_features[feature] = reduced_features.get(feature, 0) + 1
                self.assertGreater(energy, 0.)
                np.testing.assert_allclose(contact.energy(state), energy, rtol=2e-14, atol=0.)
                np.testing.assert_allclose(contact.gradient(state), gradient, rtol=1e-12, atol=1e-16)
                np.testing.assert_allclose(contact.hessian(state).toarray(), hessian, rtol=1e-12, atol=1e-12)
                features.append(reduced_features)
        self.assertNotEqual(features[0], features[1])
        self.assertNotEqual(features[1], features[2])

    def test_native_barrier_mutations_cannot_change_a_recorded_law(self):
        for mutation in ("activation", "barrier"):
            for profile in ("tight-inclusion", "temporal-separation-tight-inclusion"):
                with self.subTest(mutation=mutation, profile=profile):
                    _, _, contact, positions = self.fixture(profile)
                    self.assertGreater(contact.energy(positions), 0.)  # populate cached buckets
                    if mutation == "activation":
                        contact._potentials[1].dhat = .01
                    else:
                        contact._potentials[1].barrier = ipctk.ClampedLogBarrier()
                    evaluations = (lambda: contact.energy(positions), lambda: contact.gradient(positions),
                        lambda: contact.hessian(positions), lambda: contact.energy_change(positions, positions),
                        lambda: contact.validate_state(positions), lambda: contact.step_limit(positions, positions),
                        lambda: contact.path_safe(positions, positions),
                        lambda: contact.path_certificate(positions, positions), contact.profile)
                    for evaluate in evaluations:
                        with self.assertRaisesRegex(ValueError, "Native contact parameters"):
                            evaluate()

    def test_native_ccd_and_declared_profile_cannot_drift(self):
        for parameter, value in (("tolerance", 1e-3), ("max_iterations", 1), ("conservative_rescaling", .99)):
            with self.subTest(parameter=parameter):
                _, _, contact, positions = self.fixture()
                contact.energy(positions)
                setattr(contact.ccd, parameter, value)
                for evaluate in (lambda: contact.energy(positions), lambda: contact.gradient(positions),
                                 lambda: contact.hessian(positions), lambda: contact.path_safe(positions, positions),
                                 contact.profile):
                    with self.assertRaisesRegex(ValueError, "Native contact parameters"):
                        evaluate()
        _, _, contact, _ = self.fixture()
        for parameter, value in (("_configuration_locked", False), ("_energy_profile", "legacy"),
                                 ("_ccd_profile", "tight-inclusion"), ("ccd", None)):
            with self.subTest(parameter=parameter), self.assertRaises(AttributeError):
                setattr(contact, parameter, value)


if __name__ == "__main__":
    unittest.main()
