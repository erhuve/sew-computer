import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

import solver_global_sewing
from solver_energy_balance import global_energy_transition
from solver_global_sewing import GlobalSewingSolver
from solver_hinge_sweep import hinge_sweep_safe


class FoldBarrierIntegrationTests(unittest.TestCase):
    def fixture(self, stiffness=.03, pinned=False, bending=0.):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=[[.2, 1., 0.], [.7, -1., 0.], [0., 0., 0.], [1., 0., 0.]],
            indices=[0, 2, 3, 1, 3, 2], density=.2, tri_ke=0, tri_ka=0, tri_kd=0,
            edge_ke=bending, edge_kd=0)
        if pinned:
            builder.particle_mass[2] = 0
        builder.add_particle(pos=(3., 0., 0.), vel=(0, 0, 0), mass=.1)
        builder.add_particle(pos=(3., 0., 0.), vel=(0, 0, 0), mass=.1)
        builder.set_coloring([[vertex] for vertex in range(6)])
        model = builder.finalize(device="cpu")
        solver = GlobalSewingSolver(model, [{4: 1., 5: -1.}], 1e-8, fold_barrier_joules=stiffness)
        return solver, model.particle_q.numpy().astype(float)

    def test_zero_bending_hinges_receive_barrier_force_and_energy(self):
        solver, previous = self.fixture()
        self.assertFalse(solver.has_bending)
        self.assertEqual(len(solver.fold_barrier.indices), 1)
        previous[1] = [.7, -np.cos(2.3), -np.sin(2.3)]
        targets = np.zeros((1, 3))
        initial_velocity = np.zeros_like(previous)
        timestep = .05
        current, velocities, report = solver.step(previous, initial_velocity, targets, timestep)
        self.assertTrue(report["converged"], report)
        self.assertFalse(report["accepted"])
        self.assertTrue(report["localFoldBarrier"])
        self.assertLess(solver.fold_barrier.energy(current), solver.fold_barrier.energy(previous))
        np.testing.assert_allclose(solver.mass @ velocities, 0, atol=1e-8)
        gradient = solver.mass[:, None] * (current - previous) / timestep ** 2 + solver.fold_barrier.gradient(current)
        self.assertLess(np.abs(gradient).max(), 1e-6)
        accounting = global_energy_transition(solver, previous, current, initial_velocity, velocities,
                                               targets, targets, timestep)
        expected = solver.fold_barrier.energy(current) - solver.fold_barrier.energy(previous)
        self.assertAlmostEqual(accounting["foldBarrierChangeJoules"], expected, places=13)
        self.assertAlmostEqual(accounting["mechanicalChangeJoules"],
            expected + np.sum(solver.mass[:, None] * velocities ** 2) / 2, places=13)
        self.assertEqual(accounting["bendingChangeJoules"], 0.)
        self.assertLess(accounting["mechanicalChangeMinusTargetWorkJoules"], 0.)

    def test_fixed_particle_stays_fixed_despite_nonzero_reaction(self):
        solver, previous = self.fixture(pinned=True)
        previous[1] = [.7, -np.cos(2.3), -np.sin(2.3)]
        current, velocities, report = solver.step(previous, np.zeros_like(previous), np.zeros((1, 3)), .05)
        self.assertTrue(report["converged"], report)
        np.testing.assert_array_equal(current[2], previous[2])
        np.testing.assert_array_equal(velocities[2], 0)
        self.assertGreater(np.linalg.norm(solver.fold_barrier.gradient(current)[2]), 1e-3)

    def test_flat_configuration_and_default_model_are_unchanged(self):
        for stiffness in (None, .03):
            solver, previous = self.fixture(stiffness=stiffness)
            velocity = np.broadcast_to([.2, -.1, .3], previous.shape).copy()
            current, _, report = solver.step(previous, velocity, np.zeros((1, 3)), .01)
            self.assertTrue(report["converged"], report)
            np.testing.assert_allclose(current, previous + .01 * velocity, atol=1e-12)
            self.assertEqual(report["foldBarrierJoules"], 0.)
            self.assertEqual(report["localFoldBarrier"], stiffness is not None)
            if stiffness is None:
                self.assertIsNone(solver.fold_barrier)
                self.assertEqual(report["profile"], "experimental-global-membrane-sewing-reference-v6")

    def test_zero_bending_cannot_bypass_sweep_with_other_search_modes(self):
        solver, previous = self.fixture()
        for mode in ("lsmr", "shifted"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                solver.step(previous, np.zeros_like(previous), np.zeros((1, 3)), .01, linear_solver=mode)

    def test_barrier_and_bending_both_enter_stationarity(self):
        solver, previous = self.fixture(bending=.2)
        previous[1] = [.7, -np.cos(2.3), -np.sin(2.3)]
        timestep = .03
        current, _, report = solver.step(previous, np.zeros_like(previous), np.zeros((1, 3)), timestep)
        self.assertTrue(report["converged"], report)
        gradient = (solver.mass[:, None] * (current - previous) / timestep ** 2
                    + solver.bending.gradient(current) + solver.fold_barrier.gradient(current))
        self.assertLess(np.abs(gradient).max(), 1e-6)
        self.assertGreater(np.linalg.norm(solver.bending.gradient(current)), .1)
        self.assertGreater(np.linalg.norm(solver.fold_barrier.gradient(current)), .01)

    def test_particle_only_model_has_empty_barrier(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for position in ((0., 0., 0.), (1., 0., 0.)):
            builder.add_particle(pos=position, vel=(0, 0, 0), mass=1.)
        builder.set_coloring([[0], [1]])
        model = builder.finalize(device="cpu")
        solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}], .1, fold_barrier_joules=.03)
        previous = model.particle_q.numpy().astype(float)
        timestep = .01
        current, _, report = solver.step(previous, np.zeros_like(previous), np.zeros((1, 3)), timestep)
        self.assertTrue(report["converged"], report)
        self.assertEqual(len(solver.fold_barrier.indices), 0)
        expected_gap = 1 / (1 + 2 * timestep ** 2 / .1)
        self.assertAlmostEqual(current[1, 0] - current[0, 0], expected_gap, places=12)

    def test_invalid_candidate_preserves_residual_dimension(self):
        solver, previous = self.fixture()
        collapsed = previous.copy()
        collapsed[0] = collapsed[2]
        original_descent = solver_global_sewing._direct_descent

        def inspect_descent(evaluate, *args, **kwargs):
            valid = evaluate(previous.ravel())
            invalid = evaluate(collapsed.ravel())
            self.assertEqual(valid.shape, invalid.shape)
            self.assertTrue(np.isinf(invalid).all())
            return original_descent(evaluate, *args, **kwargs)

        with patch.object(solver_global_sewing, "_direct_descent", side_effect=inspect_descent):
            solver.step(previous, np.zeros_like(previous), np.zeros((1, 3)), .01)

    def test_optimizer_and_physical_paths_are_guarded_independently(self):
        solver, previous = self.fixture()
        quarter_turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        middle = previous @ quarter_turn.T
        opposite = middle @ quarter_turn.T
        hinges = solver.fold_barrier.indices
        self.assertTrue(hinge_sweep_safe(middle, opposite, hinges))
        self.assertFalse(hinge_sweep_safe(previous, opposite, hinges))
        self.assertTrue(hinge_sweep_safe(previous, previous, hinges))
        original_descent = solver_global_sewing._direct_descent

        def inspect_descent(*args, **kwargs):
            change = kwargs["energy_change_function"]
            self.assertTrue(np.isinf(change(middle.ravel(), opposite.ravel())))
            self.assertTrue(np.isinf(change(opposite.ravel(), previous.ravel())))
            self.assertEqual(change(previous.ravel(), previous.ravel()), 0.)
            return original_descent(*args, **kwargs)

        with patch.object(solver_global_sewing, "_direct_descent", side_effect=inspect_descent) as inspected:
            _, _, report = solver.step(previous, np.zeros_like(previous), np.zeros((1, 3)), .01)
        self.assertTrue(report["converged"], report)
        self.assertEqual(inspected.call_count, 1)


if __name__ == "__main__":
    unittest.main()
