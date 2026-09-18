import unittest

import newton
import numpy as np
import warp as wp

from solver_energy_balance import global_energy_transition
from solver_global_sewing import GlobalSewingSolver
from solver_bending import ElasticDihedralBending


class BendingIntegrationTests(unittest.TestCase):
    def test_newton_nonzero_rest_angle_matches_derivative_convention(self):
        for parity in (-1., 1.):
            points = np.array([[.2, 1., 0.], [.7, -.8, -.6], [0., 0., 0.], [1., 0., 0.]])
            points[:, 0] *= parity
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
                vel=wp.vec3(0, 0, 0), vertices=points.tolist(), indices=[0, 2, 3, 1, 3, 2], density=.2,
                tri_ke=0, tri_ka=0, tri_kd=0, edge_ke=.2, edge_kd=0)
            model = builder.finalize(device='cpu')
            bending = ElasticDihedralBending.from_model(model)
            positions = model.particle_q.numpy().astype(float)
            self.assertGreater(np.abs(bending.rest_angles[0]), .5)
            np.testing.assert_allclose(bending.angles(positions), bending.rest_angles, atol=1e-7)
            np.testing.assert_allclose(bending.gradient(positions), 0, atol=1e-7)

    def fixture(self, pinned=False):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=[[.2, 1., 0.], [.7, -1., 0.], [0., 0., 0.], [1., 0., 0.]],
            indices=[0, 2, 3, 1, 3, 2], density=.2, tri_ke=0, tri_ka=0, tri_kd=0,
            edge_ke=.2, edge_kd=0)
        if pinned:
            builder.particle_mass[2] = 0
        builder.add_particle(pos=(3., 0., 0.), vel=(0, 0, 0), mass=.1)
        builder.add_particle(pos=(3., 0., 0.), vel=(0, 0, 0), mass=.1)
        builder.set_coloring([[vertex] for vertex in range(6)])
        model = builder.finalize(device='cpu')
        positions = model.particle_q.numpy().astype(float)
        positions[1] = [.7, -np.cos(.5), -np.sin(.5)]
        return GlobalSewingSolver(model, [{4: 1., 5: -1.}], 1e-8), positions

    def test_bending_only_descent_momentum_and_energy(self):
        solver, previous = self.fixture()
        targets = np.zeros((1, 3))
        old_velocity = np.zeros_like(previous)
        timestep = .01
        current, velocities, report = solver.step(previous, old_velocity, targets, timestep)
        self.assertTrue(report['converged'], report)
        self.assertFalse(report['accepted'])
        self.assertEqual(report['bendingHinges'], 1)
        self.assertEqual(report['exactSteps'], 0)
        self.assertEqual(report['shiftedSteps'], 0)
        self.assertGreater(report['coupledSteps'], 0)
        self.assertLess(solver.bending.energy(current), solver.bending.energy(previous))
        np.testing.assert_allclose(solver.mass @ velocities, 0, atol=1e-8)
        gradient = solver.mass[:, None] * (current - previous) / timestep ** 2 + solver.bending.gradient(current)
        self.assertLess(np.abs(gradient).max(), 1e-6)
        accounting = global_energy_transition(solver, previous, current, old_velocity, velocities,
                                               targets, targets, timestep)
        expected = solver.bending.energy(current) - solver.bending.energy(previous)
        self.assertAlmostEqual(accounting['bendingChangeJoules'], expected, places=13)
        self.assertAlmostEqual(accounting['mechanicalChangeJoules'],
                               expected + np.sum(solver.mass[:, None] * velocities ** 2) / 2, places=13)
        self.assertLess(accounting['mechanicalChangeMinusTargetWorkJoules'], 0)

    def test_fixed_particle_preserved_and_reaction_excluded(self):
        solver, previous = self.fixture(pinned=True)
        current, velocities, report = solver.step(previous, np.zeros_like(previous), np.zeros((1, 3)), .01)
        self.assertTrue(report['converged'], report)
        np.testing.assert_array_equal(current[2], previous[2])
        np.testing.assert_array_equal(velocities[2], 0)
        self.assertGreater(np.linalg.norm(solver.bending.gradient(current)[2]), 1e-3)

    def test_unsupported_search_modes_cannot_bypass_branch_guard(self):
        solver, previous = self.fixture()
        for mode in ('lsmr', 'shifted'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'branch-aware'):
                solver.step(previous, np.zeros_like(previous), np.zeros((1, 3)), .01, linear_solver=mode)


if __name__ == '__main__':
    unittest.main()
