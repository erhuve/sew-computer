import unittest

import newton
import numpy as np
import warp as wp
from numpy.testing import assert_allclose, assert_array_equal

from solver_energy_balance import global_energy_transition
from solver_global_sewing import GlobalSewingSolver, _direct_descent
from solver_ipc_contact import IpcSurfaceContact


class IpcIntegrationTests(unittest.TestCase):
    def fixture(self, fixed=False, gap=.002):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for height in (0., gap):
            builder.add_cloth_mesh(
                pos=wp.vec3(0, 0, height), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0),
                vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0]], indices=[0, 1, 2], density=.2,
                tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0,
            )
        if fixed:
            builder.particle_mass[:3] = [0., 0., 0.]
        builder.set_coloring([[vertex] for vertex in range(6)])
        model = builder.finalize(device="cpu")
        rest = model.particle_q.numpy().astype(float)
        contact = IpcSurfaceContact(rest, model.tri_indices.numpy(), activation_distance_m=.005,
                                    minimum_distance_m=.0001, stiffness=1e5)
        solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}], 1e-6, contact=contact)
        return model, rest, contact, solver, solver.sewing @ rest

    def test_repulsion_momentum_and_energy_accounting(self):
        model, rest, contact, solver, targets = self.fixture()
        velocity = np.zeros_like(rest)
        final, result_velocity, report = solver.step(rest, velocity, targets, .01, max_evaluations=1000)
        self.assertTrue(report["converged"], report)
        self.assertFalse(report["accepted"])
        self.assertGreater(final[3:, 2].mean() - final[:3, 2].mean(), rest[3:, 2].mean())
        assert_allclose(solver.mass @ result_velocity, 0., atol=1e-9)
        self.assertTrue(contact.path_safe(rest, final))
        self.assertLess(contact.energy(final), contact.energy(rest))
        assert_array_equal(model.particle_q.numpy(), rest)
        assert_array_equal(contact.rest_positions, rest)
        balance = global_energy_transition(solver, rest, final, velocity, result_velocity, targets, targets, .01)
        self.assertAlmostEqual(balance["contactChangeJoules"], contact.energy(final) - contact.energy(rest))
        expected = sum(balance[name] for name in ("membraneChangeJoules", "bendingChangeJoules",
                       "foldBarrierChangeJoules", "contactChangeJoules", "kineticChangeJoules", "sewingChangeJoules"))
        self.assertAlmostEqual(balance["mechanicalChangeJoules"], expected)
        self.assertLessEqual(balance["mechanicalChangeJoules"], 1e-10)
        self.assertFalse(balance["accepted"])

    def test_incoming_layers_do_not_cross(self):
        _, rest, contact, solver, targets = self.fixture(gap=.01)
        velocity = np.zeros_like(rest)
        velocity[:3, 2], velocity[3:, 2] = 1., -1.
        self.assertFalse(contact.path_safe(rest, rest + .01 * velocity))
        final, _, report = solver.step(rest, velocity, targets, .01, max_evaluations=1000)
        self.assertTrue(report["converged"], report)
        self.assertTrue(contact.path_safe(rest, final))
        self.assertGreater(final[3:, 2].min() - final[:3, 2].max(), contact.minimum_distance_m)

    def test_fixed_layer_is_unchanged(self):
        _, rest, contact, solver, targets = self.fixture(fixed=True)
        final, velocity, report = solver.step(rest, np.zeros_like(rest), targets, .01, max_evaluations=1000)
        self.assertTrue(report["converged"], report)
        assert_array_equal(final[:3], rest[:3])
        assert_array_equal(velocity[:3], 0.)
        self.assertGreater(final[3:, 2].mean(), rest[3:, 2].mean())
        self.assertTrue(contact.path_safe(rest, final))

    def test_modes_and_intersecting_initial_state_rejected(self):
        _, rest, _, solver, targets = self.fixture()
        for mode in ("lsmr", "shifted"):
            with self.assertRaisesRegex(ValueError, "Contact requires direct"):
                solver.step(rest, np.zeros_like(rest), targets, .01, linear_solver=mode)
        intersecting = rest.copy()
        intersecting[3:] = [[.02, .02, -.01], [.02, .02, .01], [.08, .02, .01]]
        with self.assertRaises(ValueError):
            solver.step(intersecting, np.zeros_like(rest), targets, .01)

    def test_identity_mismatch_rejected(self):
        model, rest, _, _, _ = self.fixture()
        for vertices, faces in ((rest * [1.01, 1., 1.], model.tri_indices.numpy()),
                                (rest, model.tri_indices.numpy()[::-1])):
            contact = IpcSurfaceContact(vertices, faces, activation_distance_m=.005,
                                        minimum_distance_m=.0001, stiffness=1e5)
            with self.assertRaisesRegex(ValueError, "match the solver model"):
                GlobalSewingSolver(model, [{0: 1., 1: -1.}], 1e-6, contact=contact)

    def test_material_rest_accepts_independent_rigid_piece_placement(self):
        model, rest, contact, _, _ = self.fixture()
        rotation = np.linalg.qr(np.random.default_rng(813).normal(size=(3, 3)))[0]
        placed = rest.copy()
        placed[:3] = rest[:3] @ rotation + [.2, .3, .4]
        placed[3:] = rest[3:] @ rotation.T + [-.2, -.3, -.4]
        model.particle_q.assign(placed.astype(np.float32))
        solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}], 1e-6, contact=contact)
        self.assertIs(solver.contact, contact)
        assert_array_equal(contact.rest_positions, rest)
        self.assertFalse(np.array_equal(model.particle_q.numpy(), contact.rest_positions))

    def test_physical_path_is_checked_separately_from_optimizer_path(self):
        _, rest, contact, solver, targets = self.fixture()
        paths = []
        original_path_safe = contact.path_safe

        def recorded_path_safe(start, end):
            paths.append((start.copy(), end.copy()))
            return original_path_safe(start, end)

        contact.path_safe = recorded_path_safe
        _, _, report = solver.step(rest, np.zeros_like(rest), targets, .01, max_evaluations=1000)
        self.assertTrue(report["converged"], report)
        distinct_optimizer_paths = 0
        for index in range(0, len(paths) - 1, 2):
            optimizer_start, optimizer_end = paths[index]
            physical_start, physical_end = paths[index + 1]
            assert_array_equal(physical_start, rest)
            assert_array_equal(physical_end, optimizer_end)
            distinct_optimizer_paths += int(not np.array_equal(optimizer_start, physical_start))
        self.assertGreater(distinct_optimizer_paths, 0)

    def test_step_limit_precedes_candidate_energy_evaluation(self):
        from scipy.sparse import eye

        observed = []

        def objective(positions):
            observed.append(float(positions[0]))
            self.assertLess(positions[0], .5)
            return .5 * (positions[0] - 1) ** 2

        result = _direct_descent(lambda positions: positions - 1, np.array([0.]), 2,
                                 lambda positions: eye(1), objective,
                                 gradient_function=lambda positions: positions - 1,
                                 step_limiter=lambda start, end: .25)
        assert_allclose(result.x, [.25])
        self.assertEqual(observed, [0., .25])


if __name__ == "__main__":
    unittest.main()
