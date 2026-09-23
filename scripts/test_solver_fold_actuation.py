import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

from solver_adaptive_contact import adaptive_contact_step
from solver_energy_balance import global_energy_transition
from solver_fold_actuation import FoldActuation
from solver_global_sewing import GlobalSewingSolver
from solver_hinge_sweep import hinge_sweep_safe


def fold_fixture(contact_enabled=False, rotation=None):
    vertices = np.array([[.005, .01, 0.], [.005, -.01, 0.], [0., 0., 0.], [.02, 0., 0.]])
    if rotation is not None:
        vertices = vertices @ rotation
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
        vel=wp.vec3(0, 0, 0), vertices=vertices.tolist(), indices=[0, 2, 3, 1, 3, 2],
        density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.001, edge_kd=0)
    builder.set_coloring([[vertex] for vertex in range(4)])
    model = builder.finalize(device="cpu")
    start = model.particle_q.numpy().astype(float)
    hinges = model.edge_indices.numpy()
    hinges = hinges[np.all(hinges >= 0, axis=1)]
    contact = None
    if contact_enabled:
        from solver_rest_filtered_contact import RestFilteredSurfaceContact
        contact = RestFilteredSurfaceContact(start, model.tri_indices.numpy(),
            activation_distance_m=.001, minimum_distance_m=.0001, stiffness=10000,
            ccd_profile="temporal-separation-tight-inclusion")
    solver = GlobalSewingSolver(model, [], 1e-8, fold_hinges=hinges, fold_stiffness_joules=[.02],
                               fold_barrier_joules=.00001, contact=contact)
    return model, solver, start


class FoldActuationTests(unittest.TestCase):
    def test_signed_target_derivatives_covariance_and_balanced_reactions(self):
        _, solver, start = fold_fixture()
        positions = start.copy()
        positions[1, 2] = .002
        potential = solver.fold_actuation.potential([1.2])
        gradient = potential.gradient(positions)
        for index, direction in enumerate(np.eye(12).reshape((-1, 4, 3))):
            epsilon = 1e-8
            difference = (potential.energy(positions + epsilon * direction)
                          - potential.energy(positions - epsilon * direction)) / (2 * epsilon)
            self.assertAlmostEqual(difference, gradient.ravel()[index], places=7)
        np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.cross(positions, gradient).sum(axis=0), 0, atol=1e-12)
        rotation = np.linalg.qr(np.random.default_rng(814).normal(size=(3, 3)))[0]
        transformed = positions @ rotation + [.1, -.2, .3]
        self.assertAlmostEqual(potential.energy(transformed), potential.energy(positions), places=13)
        np.testing.assert_allclose(potential.gradient(transformed), gradient @ rotation, atol=1e-11)
        opposite = solver.fold_actuation.potential([-1.2])
        np.testing.assert_allclose(potential.gradient(start), -opposite.gradient(start), atol=1e-12)

    def test_source_topology_and_explicit_targets_fail_closed(self):
        model, solver, start = fold_fixture()
        hinges = solver.fold_actuation.hinges
        for invalid in (hinges.astype(float), [[0, 0, 2, 3]], [[1, 0, 2, 3]],
                        np.concatenate((hinges, hinges)), [[0, 1, 2, 8]], None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FoldActuation(model, invalid, [.02])
        for invalid in ([0.], [-1.], [np.nan], [True], None):
            with self.assertRaises(ValueError):
                FoldActuation(model, hinges, invalid)
        for targets in (None, [True], [np.inf], [np.pi], [-np.pi], [0., 0.]):
            with self.assertRaises(ValueError):
                solver.step(start, np.zeros_like(start), np.empty((0, 3)), .01, fold_targets=targets)
        for mode in ("lsmr", "shifted"):
            with self.assertRaises(ValueError):
                solver.step(start, np.zeros_like(start), np.empty((0, 3)), .01,
                            fold_targets=[1.], linear_solver=mode)
        with self.assertRaises(ValueError):
            global_energy_transition(solver, start, start, np.zeros_like(start), np.zeros_like(start),
                                     np.empty((0, 3)), np.empty((0, 3)), .01)

    def test_fold_ramp_preserves_rest_geometry_and_replays_forces_and_work(self):
        import ipctk
        from solver_membrane_hessian import membrane_element_derivatives
        from solver_spike_geometry import surface_intersections

        ipctk.set_num_threads(1)
        model, solver, start = fold_fixture(contact_enabled=True)
        rest_angles = solver.bending.rest_angles.copy()
        rest_poses = solver.poses.copy()
        states = []
        final, _, report = adaptive_contact_step(solver, start, np.zeros_like(start),
            np.empty((0, 3)), np.empty((0, 3)), .08, initial_subdivisions=16,
            initial_fold_targets=[0.], fold_targets=[2.6],
            on_accept=lambda positions, velocities, record: states.append((positions, velocities, record)))
        self.assertTrue(report["complete"], report)
        previous, previous_velocity, old_target = start, np.zeros_like(start), np.array([0.])
        for positions, velocities, record in states:
            target = np.array([2.6 * record["endFraction"]])
            potential = solver.fold_actuation.potential(target)
            timestep = record["durationSeconds"]
            _, membrane, _ = membrane_element_derivatives(positions[solver.faces], solver.poses,
                                                          solver.areas, solver.materials[:, :3])
            gradient = solver.mass[:, None] * (positions - previous - timestep * previous_velocity) / timestep ** 2
            np.add.at(gradient, solver.faces, membrane.reshape((-1, 3, 3)))
            gradient += (solver.bending.gradient(positions) + solver.fold_barrier.gradient(positions)
                         + potential.gradient(positions) + solver.contact.gradient(positions))
            self.assertLess(np.max(np.abs(gradient)), 1e-6)
            self.assertTrue(hinge_sweep_safe(previous, positions, potential.indices))
            self.assertTrue(solver.contact.path_safe(previous, positions))
            self.assertFalse(ipctk.has_intersections(solver.contact.mesh, positions, broad_phase=ipctk.BruteForce()))
            self.assertEqual(surface_intersections(positions, solver.faces)["intersectingPairCount"], 0)
            accounting = global_energy_transition(solver, previous, positions, previous_velocity, velocities,
                np.empty((0, 3)), np.empty((0, 3)), timestep,
                previous_fold_targets=old_target, fold_targets=target)
            old_potential = solver.fold_actuation.potential(old_target)
            self.assertAlmostEqual(accounting["foldTargetParameterWorkJoules"],
                potential.energy(previous) - old_potential.energy(previous), places=13)
            self.assertAlmostEqual(accounting["foldActuationChangeJoules"],
                potential.energy(positions) - old_potential.energy(previous), places=13)
            self.assertAlmostEqual(accounting["mechanicalChangeJoules"] - accounting["targetParameterWorkJoules"],
                                   accounting["mechanicalChangeMinusTargetWorkJoules"], places=12)
            np.testing.assert_allclose(solver.mass @ positions, solver.mass @ start, atol=1e-12)
            previous, previous_velocity, old_target = positions, velocities, target
        self.assertGreater(solver.fold_actuation.potential([2.6]).angles(final)[0], 2.5)
        np.testing.assert_array_equal(solver.bending.rest_angles, rest_angles)
        np.testing.assert_array_equal(solver.poses, rest_poses)
        np.testing.assert_array_equal(model.particle_q.numpy(), start)

    def test_adaptive_rejection_keeps_original_fold_schedule(self):
        _, solver, start = fold_fixture()
        original = solver.step
        observed = []

        def step(positions, velocities, targets, dt, **options):
            observed.append((dt, options["fold_targets"].copy()))
            if len(observed) == 1:
                raise ValueError("deliberate subdivision")
            return original(positions, velocities, targets, dt, **options)

        with patch.object(solver, "step", side_effect=step):
            _, _, report = adaptive_contact_step(solver, start, np.zeros_like(start),
                np.empty((0, 3)), np.empty((0, 3)), .01,
                initial_fold_targets=[0.], fold_targets=[.2])
        self.assertTrue(report["complete"], report)
        self.assertEqual(len(report["rejectedSteps"]), 1)
        np.testing.assert_allclose([target[0] for _, target in observed], [.2, .1, .2])
        np.testing.assert_allclose([duration for duration, _ in observed], [.01, .005, .005])

    def test_actuator_guards_optimizer_and_physical_paths_without_barrier(self):
        import solver_global_sewing

        _, solver, start = fold_fixture()
        solver.fold_barrier = None
        quarter_turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        middle, opposite = start @ quarter_turn, -start
        self.assertTrue(hinge_sweep_safe(middle, opposite, solver.fold_actuation.hinges))
        self.assertFalse(hinge_sweep_safe(start, opposite, solver.fold_actuation.hinges))
        original = solver_global_sewing._direct_descent

        def inspect(evaluate, *args, **options):
            collapsed = start.copy()
            collapsed[0] = collapsed[2]
            self.assertEqual(evaluate(start.ravel()).shape, evaluate(collapsed.ravel()).shape)
            self.assertTrue(np.isinf(evaluate(collapsed.ravel())).all())
            self.assertTrue(np.isinf(options["energy_change_function"](middle.ravel(), opposite.ravel())))
            self.assertTrue(np.isinf(options["energy_change_function"](opposite.ravel(), start.ravel())))
            return original(evaluate, *args, **options)

        with patch.object(solver_global_sewing, "_direct_descent", side_effect=inspect):
            _, _, report = solver.step(start, np.zeros_like(start), np.empty((0, 3)), .01, fold_targets=[.1])
        self.assertTrue(report["converged"], report)


if __name__ == "__main__":
    unittest.main()
