import unittest
from unittest.mock import patch

import ipctk
import numpy as np

from solver_rest_filtered_contact import RestFilteredSurfaceContact
from test_solver_contact_range_adversarial import square_grid


def filtered(rest, faces, activation=.00002, minimum=.0001, **options):
    return RestFilteredSurfaceContact(rest, faces, activation_distance_m=activation,
                                      minimum_distance_m=minimum, stiffness=10000., **options)


class RestFilteredContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def test_refined_flat_source_has_no_artificial_force(self):
        for count in (4, 6, 8, 16):
            with self.subTest(count=count):
                rest, faces = square_grid(count)
                contact = filtered(rest, faces)
                contact.validate_state(rest)
                self.assertEqual(contact.energy(rest), 0.)
                np.testing.assert_array_equal(contact.gradient(rest), np.zeros_like(rest))
                np.testing.assert_array_equal(contact.rest_positions, rest)
                self.assertFalse(contact.profile()["accepted"])

    def test_separate_coincident_rest_panels_keep_full_thickness(self):
        panel, faces = square_grid(4)
        rest = np.vstack((panel, panel))
        triangles = np.vstack((faces, faces + len(panel)))
        contact = filtered(rest, triangles)
        positions = rest.copy()
        positions[len(panel):, 2] = .00011
        contact.validate_state(positions)
        self.assertGreater(contact.energy(positions), 0.)
        self.assertLess(contact.gradient(positions)[len(panel):, 2].sum(), 0.)
        positions[len(panel):, 2] = .00009
        with self.assertRaises(ValueError):
            contact.validate_state(positions)

    def test_local_barrier_survives_compression_and_rejects_crossing(self):
        rest, faces = square_grid(2)
        contact = filtered(rest, faces, activation=.01)
        positions = rest * .4
        contact.validate_state(positions)
        self.assertGreater(contact.energy(positions), 0.)
        self.assertGreater(np.linalg.norm(contact.gradient(positions)), 0.)
        start = rest.copy()
        end = rest.copy()
        end[0] = rest[8]
        self.assertLess(contact.step_limit(start, end), 1.)

    def test_rebuilt_energy_gradient_and_hessian(self):
        rest, faces = square_grid(2)
        contact = filtered(rest, faces, activation=.01)
        positions = rest * .4
        positions[4, 2] = .000015
        # Closest-feature ties are only piecewise C2; the transition case is
        # covered separately without asserting a nonexistent unique Hessian.
        positions += np.random.default_rng(83).normal(size=positions.shape) * 1e-6
        direction = np.random.default_rng(712).normal(size=positions.shape)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-9
        gradient = contact.gradient(positions)
        self.assertGreater(contact.energy(positions), 0.)
        self.assertGreater(np.linalg.norm(gradient), 1e-6)
        hessian = contact.hessian(positions)
        energy_derivative = (contact.energy(positions + epsilon * direction)
                             - contact.energy(positions - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(energy_derivative, np.sum(gradient * direction), rtol=1e-5, atol=1e-9)
        gradient_derivative = (contact.gradient(positions + epsilon * direction)
                               - contact.gradient(positions - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(gradient_derivative.ravel(), hessian @ direction.ravel(),
                                   rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-12)

    def test_crossing_distinct_layers_retains_continuous_guard(self):
        panel, faces = square_grid(2)
        rest = np.vstack((panel, panel))
        contact = filtered(rest, np.vstack((faces, faces + len(panel))))
        start = rest.copy()
        start[len(panel):, 2] = .002
        end = start.copy()
        end[len(panel):, 2] = -.002
        self.assertEqual(contact.energy(start), 0.)
        self.assertFalse(contact.path_safe(start, end))
        self.assertLess(contact.step_limit(start, end), .475)

    def test_rigid_covariance_and_partition_stability(self):
        rest, faces = square_grid(2)
        rotation = np.linalg.qr(np.random.default_rng(92).normal(size=(3, 3)))[0]
        contact = filtered(rest, faces, activation=.01)
        rotated = filtered(rest @ rotation, faces, activation=.01)
        self.assertEqual(contact.profile()["filteredPairsSha256"], rotated.profile()["filteredPairsSha256"])
        positions = rest * .4
        positions[4, 2] = .000015
        np.testing.assert_allclose(contact.energy(positions), rotated.energy(positions @ rotation), rtol=1e-10)
        np.testing.assert_allclose(contact.gradient(positions) @ rotation,
                                   rotated.gradient(positions @ rotation), atol=1e-10)
        before = contact.profile()["filteredPairsSha256"]
        contact.gradient(positions)
        self.assertEqual(contact.profile()["filteredPairsSha256"], before)

    def test_invalid_geometry_and_candidate_budgets_reject(self):
        rest, faces = square_grid(2)
        for budget in (True, 0, 1000001):
            with self.assertRaises(ValueError):
                filtered(rest, faces, max_candidates=budget)
        with self.assertRaises(ValueError):
            filtered(rest, faces, activation=.01, max_candidates=1)
        rest[4, 2] = .0001
        with self.assertRaisesRegex(ValueError, "planar"):
            filtered(rest, faces)

    def test_local_fan_fold_retains_contact(self):
        rest = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0], [.002, .002, 0]])
        faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
        contact = filtered(rest, faces, activation=.01)
        start = rest.copy()
        start[3] = [.008, -.002, .00013]
        self.assertGreater(contact.energy(start), 0.)
        end = start.copy()
        end[3, 2] *= -1
        self.assertFalse(contact.path_safe(start, end))
        self.assertLess(contact.step_limit(start, end), .5)

    def test_configuration_cannot_drift_from_precomputed_partition(self):
        rest, faces = square_grid(2)
        contact = filtered(rest, faces)
        for field in ("minimum_distance_m", "activation_distance_m", "stiffness"):
            with self.assertRaises(AttributeError):
                setattr(contact, field, 1.)

    def test_parallel_layer_alignment_derivatives(self):
        panel, faces = square_grid(2)
        rest = np.vstack((panel, panel))
        contact = filtered(rest, np.vstack((faces, faces + len(panel))))
        direction = np.zeros_like(rest)
        direction[len(panel):, 0] = 1.
        for offset in (-1e-7, 0., 1e-7, .0001):
            positions = rest.copy()
            positions[len(panel):, 2] = .00011
            positions += offset * direction
            epsilon = 1e-10
            expected = np.sum(contact.gradient(positions) * direction)
            observed = (contact.energy(positions + epsilon * direction)
                        - contact.energy(positions - epsilon * direction)) / (2 * epsilon)
            np.testing.assert_allclose(observed, expected, rtol=1e-4, atol=1e-7)

    def test_temporal_certificate_covers_both_thickness_groups_exactly(self):
        from test_solver_temporal_separation import verify_leaf
        from solver_temporal_separation import _GROUPS

        panel, faces = square_grid(2, width=.0003)
        rest = np.vstack((panel, panel))
        contact = filtered(rest, np.vstack((faces, faces + len(panel))),
                           ccd_profile="temporal-separation-tight-inclusion")
        start = rest.copy()
        start[len(panel):, 2] = .00011
        end = start + [.0003, -.0002, .0004]
        with patch.object(contact, "step_limit", side_effect=AssertionError("Not a predicate")):
            self.assertTrue(contact.path_safe(start, end))
        report = contact.path_certificate(start, end, keep_leaves=True)
        self.assertTrue(report["safe"], report)
        self.assertEqual(report["filteredPairsSha256"], contact.profile()["filteredPairsSha256"])
        candidates = ipctk.Candidates()
        candidates.build(contact.mesh, start, end,
                         inflation_radius=np.nextafter(contact.minimum_distance_m / 2, np.inf),
                         broad_phase=ipctk.BruteForce())
        expected = {}
        for group in _GROUPS:
            for candidate in getattr(candidates, group):
                distance = float(np.sqrt(candidate.compute_distance(candidate.dof(rest, contact._edges, contact.faces))))
                expected[contact._key(group, candidate)] = (min(contact.minimum_distance_m, distance / 4)
                    if contact._key(group, candidate) in contact._filtered else contact.minimum_distance_m)
        intervals = {}
        for leaf in report["certificateLeaves"]:
            identity = contact._ids_key(leaf["group"], leaf["first"], leaf["second"])
            self.assertEqual(leaf["minimumDistanceM"], expected[identity])
            verify_leaf(start, end, leaf, expected[identity])
            intervals.setdefault(identity, []).append((leaf["t0"], leaf["t1"]))
        self.assertEqual(set(intervals), set(expected))
        self.assertEqual(len(expected), len(candidates))
        self.assertEqual(len(expected), report["candidateCount"])
        self.assertIn(contact.minimum_distance_m, expected.values())
        self.assertTrue(any(value < contact.minimum_distance_m for value in expected.values()))
        self.assertEqual(report["filteredParametersSha256"], contact.profile()["filteredParametersSha256"])
        for spans in intervals.values():
            cursor = 0.
            for lower, upper in sorted(spans):
                self.assertEqual(lower, cursor)
                self.assertGreater(upper, lower)
                cursor = upper
            self.assertEqual(cursor, 1.)

    def test_temporal_crossing_and_budget_exhaustion_fail_closed(self):
        panel, faces = square_grid(2)
        rest = np.vstack((panel, panel))
        contact = filtered(rest, np.vstack((faces, faces + len(panel))),
                           ccd_profile="temporal-separation-tight-inclusion")
        start = rest.copy()
        start[len(panel):, 2] = .002
        end = start.copy()
        end[len(panel):, 2] = -.002
        self.assertFalse(contact.path_safe(start, end))
        limited = contact.path_certificate(start, start + [.002, .002, .002], max_nodes=1)
        self.assertFalse(limited["safe"])
        self.assertEqual(limited["reason"], "node-budget-exhausted")
        with patch("solver_temporal_separation.certify_linear_path", return_value={"safe": False}):
            with patch.object(contact, "step_limit", return_value=1.):
                self.assertFalse(contact.path_safe(start, start))

    def test_temporal_profile_preserves_contact_law_and_optimizer_proposal(self):
        rest, faces = square_grid(2)
        original = filtered(rest, faces, activation=.01)
        temporal = filtered(rest, faces, activation=.01,
                            ccd_profile="temporal-separation-tight-inclusion")
        positions = rest * .4
        positions[4, 2] = .000015
        self.assertEqual(original.energy(positions), temporal.energy(positions))
        np.testing.assert_array_equal(original.gradient(positions), temporal.gradient(positions))
        self.assertEqual((original.hessian(positions) - temporal.hessian(positions)).nnz, 0)
        self.assertEqual(original.step_limit(rest, positions), temporal.step_limit(rest, positions))
        with self.assertRaises(ValueError):
            filtered(rest, faces, ccd_profile="swept-plane-tight-inclusion")

    def test_complete_positive_offset_sewing_with_coupled_contact(self):
        import newton
        import warp as wp
        from solver_global_sewing import GlobalSewingSolver

        panel, faces = square_grid(2, width=.01)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for height in (0., .002):
            builder.add_cloth_mesh(pos=wp.vec3(0, 0, height), rot=wp.quat_identity(),
                scale=1, vel=wp.vec3(0, 0, 0), vertices=panel.tolist(), indices=faces.ravel().tolist(),
                density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0)
        builder.particle_mass[:len(panel)] = [0.] * len(panel)
        builder.set_coloring([[vertex] for vertex in range(2 * len(panel))])
        model = builder.finalize(device="cpu")
        rest = model.particle_q.numpy().astype(float)
        contact = filtered(rest, model.tri_indices.numpy())
        boundary = [vertex for vertex in range(len(panel)) if vertex != 4]
        rows = [{vertex: -1., vertex + len(panel): 1.} for vertex in boundary]
        solver = GlobalSewingSolver(model, rows, 1e-8, contact=contact)
        initial = solver.sewing @ rest
        target = np.tile([0., 0., .00011], (len(rows), 1))
        positions, velocities = rest.copy(), np.zeros_like(rest)
        for step in range(1, 17):
            targets = initial + (step / 16) * (target - initial)
            previous = positions.copy()
            positions, velocities, report = solver.step(positions, velocities, targets,
                                                        .001, max_evaluations=100)
            self.assertTrue(report["converged"], report)
            self.assertLessEqual(report["gradientInfinityNorm"], 1e-6)
            self.assertTrue(contact.path_safe(previous, positions))
            np.testing.assert_array_equal(positions[:len(panel)], rest[:len(panel)])
        self.assertGreater(contact.energy(positions), 0.)
        self.assertLess(np.max(np.linalg.norm(solver.sewing @ positions - target, axis=1)), 1e-6)
        self.assertGreater(np.min(positions[len(panel):, 2]), .0001)
        np.testing.assert_array_equal(contact.rest_positions, rest)

    def test_complete_positive_offset_sewing_with_temporal_contact(self):
        original = filtered
        with patch(__name__ + ".filtered", side_effect=lambda *args, **kwargs:
                   original(*args, **kwargs, ccd_profile="temporal-separation-tight-inclusion")):
            self.test_complete_positive_offset_sewing_with_coupled_contact()


if __name__ == "__main__":
    unittest.main()
