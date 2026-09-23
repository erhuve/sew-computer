import unittest

import numpy as np
from scipy.sparse import csr_matrix

from solver_normal_sewing import NormalOffsetSewing


class NormalSewingTests(unittest.TestCase):
    def setUp(self):
        self.rows = csr_matrix([[-.2, -.3, -.5, 1.]])
        self.positions = np.array([[0., 0., 0.], [.02, 0., .003],
                                   [0., .03, -.002], [.008, .017, .004]])
        self.potential = NormalOffsetSewing(self.rows, [.002], .003, [[0, 1, 2]], [1])

    def test_all_derivatives_and_energy_change(self):
        epsilon = 1e-7
        gradient = self.potential.gradient(self.positions)
        jacobian = self.potential.jacobian(self.positions).toarray()
        for index, direction in enumerate(np.eye(12).reshape((-1, 4, 3))):
            plus, minus = self.positions + epsilon * direction, self.positions - epsilon * direction
            upper, lower = self.potential.residual(plus), self.potential.residual(minus)
            np.testing.assert_allclose((upper - lower) / (2 * epsilon), jacobian[:, index], atol=1e-8)
            self.assertAlmostEqual((upper @ upper - lower @ lower) / (4 * epsilon), gradient[index], places=8)
            self.assertAlmostEqual(self.potential.energy_change(minus, plus),
                                   .5 * (upper @ upper - lower @ lower), places=15)
        metric = self.potential.hessian(self.positions).toarray()
        np.testing.assert_allclose(metric, jacobian.T @ jacobian, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(metric).min(), -1e-10)

    def test_rigid_covariance_force_torque_and_frame_reactions(self):
        rotation = np.linalg.qr(np.random.default_rng(374).normal(size=(3, 3)))[0]
        transformed = self.positions @ rotation + [.1, .2, -.1]
        force = self.potential.gradient(self.positions).reshape((-1, 3))
        np.testing.assert_allclose(self.potential.residual(transformed).reshape((-1, 3)),
                                   self.potential.residual(self.positions).reshape((-1, 3)) @ rotation, atol=1e-13)
        np.testing.assert_allclose(self.potential.gradient(transformed).reshape((-1, 3)), force @ rotation, atol=1e-12)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.cross(self.positions, force).sum(axis=0), 0, atol=1e-12)
        frozen_force = np.asarray(self.rows.T @ self.potential.residual(self.positions).reshape((-1, 3))) / np.sqrt(.003)
        self.assertGreater(np.linalg.norm(force - frozen_force), .01)

    def test_exact_curvature_matches_gradient_differences_and_rigid_covariance(self):
        epsilon = 1e-7
        matrix = self.potential.exact_hessian(self.positions).toarray()
        np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=1e-10)
        for index, direction in enumerate(np.eye(12).reshape((-1, 4, 3))):
            difference = (self.potential.gradient(self.positions + epsilon * direction)
                          - self.potential.gradient(self.positions - epsilon * direction)) / (2 * epsilon)
            np.testing.assert_allclose(matrix[:, index], difference, rtol=2e-7, atol=1e-7)
        self.assertGreater(np.max(np.abs(matrix - self.potential.hessian(self.positions).toarray())), 1.)
        rotation = np.linalg.qr(np.random.default_rng(337).normal(size=(3, 3)))[0]
        transform = np.kron(np.eye(4), rotation.T)
        moved = self.potential.exact_hessian(self.positions @ rotation + [.1, -.2, .3]).toarray()
        np.testing.assert_allclose(moved, transform @ matrix @ transform.T, rtol=1e-10, atol=1e-8)

    def test_exact_curvature_accumulates_shared_frames_and_vanishes_at_zero_residual(self):
        rows = csr_matrix([[-.2, -.3, -.5, 1.], [-.4, -.1, -.5, 1.]])
        potential = NormalOffsetSewing(rows, [.002, .003], .003, [[0, 1, 2], [0, 1, 2]], [-1, 1])
        matrix = potential.exact_hessian(self.positions).toarray()
        expected = sum(NormalOffsetSewing(rows[index:index + 1], [potential.targets[index]], .003,
                       [[0, 1, 2]], [potential.sides[index]]).exact_hessian(self.positions).toarray()
                       for index in range(2))
        np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-10)
        direction = np.random.default_rng(28).normal(size=(4, 3))
        epsilon = 1e-7
        difference = (potential.gradient(self.positions + epsilon * direction)
                      - potential.gradient(self.positions - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(matrix @ direction.ravel(), difference, rtol=1e-7, atol=1e-6)
        positions = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .03, 0.], [.006, .015, .002]])
        np.testing.assert_allclose(self.potential.exact_hessian(positions).toarray(),
                                   self.potential.hessian(positions).toarray(), rtol=0, atol=1e-10)
        with self.assertRaises(ValueError):
            self.potential.exact_hessian(np.zeros((4, 3)))

    def test_explicit_side_distinguishes_equal_distance_layers(self):
        positions = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .03, 0.], [.006, .015, .002]])
        np.testing.assert_allclose(self.potential.residual(positions), 0, atol=1e-14)
        opposite = positions.copy()
        opposite[3, 2] *= -1
        self.assertGreater(np.linalg.norm(self.potential.residual(opposite)), .05)
        reversed_side = NormalOffsetSewing(self.rows, [.002], .003, [[0, 1, 2]], [-1])
        np.testing.assert_allclose(reversed_side.residual(opposite), 0, atol=1e-14)
        opposite[3, 0] += .001
        self.assertGreater(np.linalg.norm(reversed_side.residual(opposite)), .01)

    def test_malformed_frames_and_degeneracy_reject(self):
        for faces, sides in (([[0, 0, 2]], [1]), ([[0, 1, 4]], [1]),
                             ([[0., 1., 2.]], [1]), ([[0, 1, 2]], [0]),
                             ([[0, 1, 2]], [True]), ([[0, 1, 3]], [1])):
            with self.assertRaises(ValueError):
                NormalOffsetSewing(self.rows, [.002], .003, faces, sides)
        for target in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                NormalOffsetSewing(self.rows, [target], .003, [[0, 1, 2]], [1])
        with self.assertRaises(ValueError):
            self.potential.residual(np.zeros((4, 3)))
        with self.assertRaises(ValueError):
            NormalOffsetSewing(self.rows * 2, [.002], .003, [[0, 1, 2]], [1])

    def test_coupled_rotated_layer_closure_and_energy_accounting(self):
        import ipctk
        import newton
        import warp as wp
        from solver_adaptive_contact import adaptive_contact_step
        from solver_energy_balance import global_energy_transition
        from solver_global_sewing import GlobalSewingSolver
        from solver_rest_filtered_contact import RestFilteredSurfaceContact
        from solver_spike_geometry import surface_intersections

        ipctk.set_num_threads(1)
        rotation = np.linalg.qr(np.random.default_rng(781).normal(size=(3, 3)))[0]
        panel = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for height in (0., .002):
            builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
                vel=wp.vec3(0, 0, 0), vertices=((panel + [0, 0, height]) @ rotation).tolist(),
                indices=[0, 1, 2], density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0)
        builder.set_coloring([[vertex] for vertex in range(6)])
        model = builder.finalize(device="cpu")
        start = model.particle_q.numpy().astype(float)
        contact = RestFilteredSurfaceContact(start, model.tri_indices.numpy(), activation_distance_m=.00002,
            minimum_distance_m=.0001, stiffness=10000., ccd_profile="temporal-separation-tight-inclusion")
        rows = [{vertex: -1., vertex + 3: 1.} for vertex in range(3)]
        solver = GlobalSewingSolver(model, rows, 1e-8, contact=contact, sewing_mode="normal-offset",
                                   sewing_frame_faces=[[0, 1, 2]] * 3, sewing_sides=[1] * 3)
        accepted = []
        def on_accept(positions, velocities, record):
            accepted.append((positions.copy(), velocities.copy(), record))
        final, _, report = adaptive_contact_step(solver, start, np.zeros_like(start), np.full(3, .002),
            np.full(3, .00011), .016, initial_subdivisions=16, on_accept=on_accept)
        self.assertTrue(report["complete"], report)
        previous, previous_velocity, previous_targets = start, np.zeros_like(start), np.full(3, .002)
        for positions, velocities, record in accepted:
            targets = np.full(3, .002 + record["endFraction"] * (.00011 - .002))
            energy = global_energy_transition(solver, previous, positions, previous_velocity, velocities,
                                              previous_targets, targets, record["durationSeconds"])
            residual = solver.sewing_potential(targets).residual(positions)
            self.assertAlmostEqual(energy["sewingAfterJoules"], .5 * residual @ residual, places=12)
            self.assertAlmostEqual(energy["sewingChangeJoules"],
                                   energy["sewingAfterJoules"] - energy["sewingBeforeJoules"], places=12)
            self.assertTrue(contact.path_safe(previous, positions))
            self.assertFalse(ipctk.has_intersections(contact.mesh, positions, broad_phase=ipctk.BruteForce()))
            self.assertEqual(surface_intersections(positions, solver.faces)["intersectingPairCount"], 0)
            np.testing.assert_allclose(solver.mass @ positions, solver.mass @ start, atol=1e-12)
            previous, previous_velocity, previous_targets = positions, velocities, targets
        self.assertGreater(contact.energy(final), 0.)
        self.assertLess(np.max(np.abs(solver.sewing_potential(np.full(3, .00011)).residual(final))) * 1e-4, 1e-6)
        with self.assertRaises(ValueError):
            GlobalSewingSolver(model, rows, 1e-8, sewing_mode="normal-offset",
                              sewing_frame_faces=[[0, 2, 1]] * 3, sewing_sides=[1] * 3)


if __name__ == "__main__":
    unittest.main()
