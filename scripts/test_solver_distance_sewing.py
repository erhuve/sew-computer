import unittest

import newton
import numpy as np
from scipy.sparse import csr_matrix

from solver_distance_sewing import DistanceSewing
from solver_global_sewing import GlobalSewingSolver


class DistanceSewingTests(unittest.TestCase):
    def setUp(self):
        self.rows = csr_matrix([[.3, .7, -.4, -.6], [1., -1., 0., 0.]])
        self.positions = np.array([[0., 0., 0.], [.02, .01, 0.],
                                   [.03, .02, .01], [.04, -.01, .03]])
        self.sewing = DistanceSewing(self.rows, [.02, .04], .003)

    def test_derivatives_and_stable_energy_change(self):
        positions = self.positions
        epsilon = 1e-7
        basis = np.eye(positions.size).reshape((-1, *positions.shape))
        gradient = self.sewing.gradient(positions)
        hessian = self.sewing.hessian(positions).toarray()
        jacobian = self.sewing.jacobian(positions).toarray()
        for index, direction in enumerate(basis):
            plus, minus = positions + epsilon * direction, positions - epsilon * direction
            residual_plus, residual_minus = self.sewing.residual(plus), self.sewing.residual(minus)
            difference = (residual_plus @ residual_plus - residual_minus @ residual_minus) / (4 * epsilon)
            self.assertAlmostEqual(difference, gradient[index], places=7)
            np.testing.assert_allclose((self.sewing.gradient(plus) - self.sewing.gradient(minus))
                                       / (2 * epsilon), hessian[:, index], atol=1e-7, rtol=1e-8)
            np.testing.assert_allclose((residual_plus - residual_minus) / (2 * epsilon),
                                       jacobian[:, index], atol=1e-8)
            self.assertAlmostEqual(self.sewing.energy_change(minus, plus),
                                   (residual_plus @ residual_plus - residual_minus @ residual_minus) / 2,
                                   places=14)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
        np.testing.assert_allclose(jacobian.T @ self.sewing.residual(positions), gradient, atol=1e-12)
        self.assertLess(np.linalg.eigvalsh(hessian).min(), -1.)
        self.assertGreater(np.linalg.eigvalsh(self.sewing.hessian(positions, True).toarray()).min(), -1e-10)

    def test_rigid_motion_preserves_energy_and_zero_internal_torque(self):
        rotation = np.linalg.qr(np.random.default_rng(462).normal(size=(3, 3)))[0]
        transformed = self.positions @ rotation + [.1, -.2, .3]
        np.testing.assert_allclose(self.sewing.residual(transformed), self.sewing.residual(self.positions), atol=1e-14)
        force = self.sewing.gradient(self.positions).reshape((-1, 3))
        np.testing.assert_allclose(self.sewing.gradient(transformed).reshape((-1, 3)), force @ rotation, atol=1e-12)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.cross(self.positions, force).sum(axis=0), 0, atol=1e-12)

    def test_invalid_distances_and_coincident_anchors_reject(self):
        for targets in ([0., .1], [-1., .1], [np.nan, .1], [np.inf, .1], [[.1, .1]], [.1]):
            with self.assertRaises(ValueError):
                DistanceSewing(self.rows, targets, .003)
        with self.assertRaises(ValueError):
            self.sewing.residual(np.zeros_like(self.positions))

    def test_free_pair_matches_analytical_implicit_step_in_any_direction(self):
        for direction in (np.array([1., 0., 0.]), np.array([1., 2., 3.]) / np.sqrt(14)):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            for position, mass in ((np.zeros(3), .2), (.02 * direction, .3)):
                builder.add_particle(pos=position, vel=(0, 0, 0), mass=mass)
            builder.set_coloring([[0], [1]])
            model = builder.finalize(device="cpu")
            solver = GlobalSewingSolver(model, [{0: -1., 1: 1.}], .001, sewing_mode="distance")
            start = model.particle_q.numpy().astype(float)
            masses = solver.mass
            for target in (.005, .03):
                final, velocity, report = solver.step(start, np.zeros_like(start), [target], .01)
                length = np.linalg.norm(start[1] - start[0])
                expected_length = (length + .01 ** 2 / .001 * np.sum(1 / masses) * target) / (
                    1 + .01 ** 2 / .001 * np.sum(1 / masses))
                self.assertTrue(report["converged"], report)
                self.assertAlmostEqual(np.linalg.norm(final[1] - final[0]), expected_length, places=10)
                np.testing.assert_allclose(masses @ final, masses @ start, atol=1e-12)
                np.testing.assert_allclose(masses @ velocity, 0, atol=1e-10)
                self.assertEqual(report["sewingMode"], "distance")
                self.assertFalse(report["accepted"])
                from solver_energy_balance import global_energy_transition
                energy = global_energy_transition(solver, start, final, np.zeros_like(start),
                                                  velocity, [length], [target], .01)
                expected_energy = (np.linalg.norm(final[1] - final[0]) - target) ** 2 / (.002)
                self.assertAlmostEqual(energy["sewingAfterJoules"], expected_energy, places=12)
                self.assertAlmostEqual(energy["sewingChangeJoules"], expected_energy, places=12)
                self.assertAlmostEqual(energy["targetParameterWorkJoules"], (length - target) ** 2 / .002,
                                       places=12)

    def test_distance_sewing_closes_layers_with_active_contact(self):
        import ipctk
        import warp as wp
        from solver_rest_filtered_contact import RestFilteredSurfaceContact
        from solver_spike_geometry import surface_intersections

        ipctk.set_num_threads(1)
        panel = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for height in (0., .002):
            builder.add_cloth_mesh(pos=wp.vec3(0, 0, height), rot=wp.quat_identity(), scale=1,
                vel=wp.vec3(0, 0, 0), vertices=panel.tolist(), indices=[0, 1, 2],
                density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0)
        builder.set_coloring([[vertex] for vertex in range(6)])
        model = builder.finalize(device="cpu")
        start = model.particle_q.numpy().astype(float)
        contact = RestFilteredSurfaceContact(start, model.tri_indices.numpy(), activation_distance_m=.00002,
            minimum_distance_m=.0001, stiffness=10000., ccd_profile="temporal-separation-tight-inclusion")
        solver = GlobalSewingSolver(model, [{vertex: -1., vertex + 3: 1.} for vertex in range(3)],
                                   1e-8, contact=contact, sewing_mode="distance")
        positions, velocities = start.copy(), np.zeros_like(start)
        for step in range(1, 17):
            target = .002 + step / 16 * (.00011 - .002)
            previous = positions.copy()
            positions, velocities, report = solver.step(positions, velocities, np.full(3, target), .001)
            self.assertTrue(report["converged"], report)
            self.assertTrue(contact.path_safe(previous, positions))
            self.assertEqual(surface_intersections(positions, solver.faces)["intersectingPairCount"], 0)
            self.assertFalse(ipctk.has_intersections(contact.mesh, positions, broad_phase=ipctk.BruteForce()))
            np.testing.assert_allclose(solver.mass @ positions, solver.mass @ start, atol=1e-12)
        self.assertGreater(contact.energy(positions), 0.)
        lengths = np.linalg.norm(solver.sewing @ positions, axis=1)
        self.assertGreater(lengths.min(), .0001)
        self.assertLess(np.max(np.abs(lengths - .00011)), 1e-6)

    def test_tangential_prediction_is_not_locked_to_initial_world_vector(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_particle(pos=(0, 0, 0), vel=(0, 0, 0), mass=0)
        builder.add_particle(pos=(.02, 0, 0), vel=(0, 0, 0), mass=.2)
        builder.set_coloring([[0], [1]])
        solver = GlobalSewingSolver(builder.finalize(device="cpu"), [{0: -1., 1: 1.}],
                                   1e-6, sewing_mode="distance")
        start = np.array([[0., 0., 0.], [.02, 0., 0.]])
        velocity = np.array([[0., 0., 0.], [0., 1., 0.]])
        final, _, report = solver.step(start, velocity, [.02], .01)
        self.assertTrue(report["converged"], report)
        np.testing.assert_array_equal(final[0], start[0])
        self.assertAlmostEqual(final[1, 1] / final[1, 0], .5, places=8)
        self.assertLess(abs(np.linalg.norm(final[1]) - .02), 1e-5)

    def test_scalar_adaptive_targets_preserve_original_schedule_after_rejection(self):
        from types import SimpleNamespace
        from solver_adaptive_contact import adaptive_contact_step

        observed = []

        def step(positions, velocities, targets, duration):
            observed.append((float(targets[0]), duration))
            valid = duration <= .25
            return positions, velocities, {"converged": valid,
                "gradientInfinityNorm": 0. if valid else 1.}

        solver = SimpleNamespace(sewing_mode="distance", step=step)
        positions = np.array([[0., 0., 0.], [.002, 0., 0.]])
        _, _, report = adaptive_contact_step(solver, positions, np.zeros_like(positions),
                                             [.002], [.00011], 1., initial_subdivisions=2)
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["rejectedSteps"]), 2)
        for attempt in report["acceptedSteps"]:
            self.assertEqual(attempt["durationSeconds"], .25)
        np.testing.assert_allclose([target for target, duration in observed if duration == .25],
                                   [.002 + fraction * (.00011 - .002) for fraction in (.25, .5, .75, 1.)])
        for targets in ([0.], [-.001], [[.001, 0., 0.]]):
            with self.assertRaises(ValueError):
                adaptive_contact_step(solver, positions, np.zeros_like(positions), [.002], targets, 1.)


if __name__ == "__main__":
    unittest.main()
