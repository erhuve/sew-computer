"""Guarded synthetic cloth integration; no refined source/capture admission."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("WARP_CACHE_PATH", str(Path(__file__).resolve().parents[1] / ".planning/solver/warp-cache"))

import newton
import numpy as np
import warp as wp

from solver_controlled_fold import ControlledFoldActuation
from solver_global_sewing import GlobalSewingSolver


def fixture(count=2, contact_enabled=False, reversed_faces=False):
    base = np.array([[.005, .01, 0.], [.005, -.01, 0.], [0., 0., 0.], [.02, 0., 0.]])
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for index in range(count):
        vertices = (base + [.04 * index, 0., 0.]).tolist()
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=vertices,
            indices=[0, 3, 2, 1, 2, 3] if reversed_faces else [0, 2, 3, 1, 3, 2],
            density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.001, edge_kd=0)
    builder.set_coloring([[vertex] for vertex in range(4 * count)])
    model = builder.finalize(device="cpu")
    q = model.particle_q.numpy().astype(float)
    native = model.edge_indices.numpy()
    hinges = native[np.all(native >= 0, axis=1)]
    recipe = ControlledFoldActuation(model, hinges, [.02] * count)
    contact = None
    if contact_enabled:
        import ipctk
        from solver_rest_filtered_contact import RestFilteredSurfaceContact
        ipctk.set_num_threads(1)
        contact = RestFilteredSurfaceContact(q, model.tri_indices.numpy(),
            activation_distance_m=.001, minimum_distance_m=.0001, stiffness=10000,
            ccd_profile="temporal-separation-tight-inclusion")
    solver = GlobalSewingSolver(model, [], 1e-8, controlled_fold_actuation=recipe, contact=contact)
    return model, solver, q


EMPTY = np.empty((0, 3))
POLICY = "all declared controlled hinges, including inactive; optimizer and physical affine paths"


class ControlledFoldIntegrationTests(unittest.TestCase):
    def test_explicit_model_rebinding_and_legacy_exclusion(self):
        model, solver, q = fixture()
        recipe = solver.controlled_fold_actuation
        self.assertIsNone(solver.fold_actuation)
        rebound = GlobalSewingSolver(model, [], 1e-8, controlled_fold_actuation=recipe)
        self.assertIsNot(rebound.controlled_fold_actuation, recipe)
        np.testing.assert_array_equal(rebound.controlled_fold_actuation.hinges, recipe.hinges)
        for kwargs in ({"fold_hinges": recipe.hinges}, {"fold_stiffness_joules": [.02, .02]}):
            with self.assertRaises(ValueError):
                GlobalSewingSolver(model, [], 1e-8, controlled_fold_actuation=recipe, **kwargs)
        other, _, _ = fixture(count=1)
        with self.assertRaises(ValueError):
            GlobalSewingSolver(other, [], 1e-8, controlled_fold_actuation=recipe)
        reversed_model, _, _ = fixture(reversed_faces=True)
        with self.assertRaises(ValueError):
            GlobalSewingSolver(reversed_model, [], 1e-8, controlled_fold_actuation=recipe)
        for invalid in (True, {}, object()):
            with self.assertRaises(ValueError):
                GlobalSewingSolver(model, [], 1e-8, controlled_fold_actuation=invalid)

    def test_missing_malformed_and_uncaptured_direct_parameters_reject(self):
        model, solver, q = fixture()
        for kwargs in ({}, {"fold_targets": [0., 0.]}, {"fold_activation": [0., 0.]},
                       {"fold_targets": [0., 0.], "fold_activation": [True, 0.]},
                       {"fold_targets": [0., 0.], "fold_activation": [.5]},
                       {"fold_targets": [np.pi, 0.], "fold_activation": [0., 0.]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solver.step(q, np.zeros_like(q), EMPTY, .01, **kwargs)
        for mode in ("lsmr", "shifted"):
            with self.assertRaises(ValueError):
                solver.step(q, np.zeros_like(q), EMPTY, .01, linear_solver=mode,
                            fold_targets=[0., 0.], fold_activation=[0., 0.])
        legacy = GlobalSewingSolver(model, [], 1e-8, fold_hinges=solver.controlled_fold_actuation.hinges,
                                    fold_stiffness_joules=[.02, .02])
        with self.assertRaises(ValueError):
            legacy.step(q, np.zeros_like(q), EMPTY, .01, fold_targets=[0., 0.], fold_activation=[1., 1.])
        with self.assertRaises(ValueError):
            solver.step(q, np.zeros_like(q), EMPTY, True, fold_targets=[0., 0.], fold_activation=[0., 0.])
        for state in ("position", "velocity"):
            original = q.tolist() if state == "position" else np.zeros_like(q).tolist()
            original[0][0] = 2**53 + 1
            with self.assertRaises(ValueError):
                solver.step(original if state == "position" else q,
                            original if state == "velocity" else np.zeros_like(q), EMPTY, .01,
                            fold_targets=[0., 0.], fold_activation=[0., 0.])

    def test_independent_hinges_move_with_full_force_and_contact_reconstruction(self):
        from solver_hinge_sweep import hinge_sweep_safe
        from solver_membrane_hessian import membrane_element_derivatives
        from solver_triangle_sweep import triangle_sweep_safe
        model, solver, q = fixture(contact_enabled=True)
        rest = [model.particle_q.numpy().copy(), solver.poses.copy(), solver.bending.rest_angles.copy()]
        velocity = np.zeros_like(q)
        for targets, activation in (([.3, -.4], [1., 0.]), ([.3, -.4], [1., .5]), ([.3, -.4], [0., 0.])):
            final, v, report = solver.step(q, velocity, EMPTY, .01, fold_targets=targets, fold_activation=activation)
            self.assertTrue(report["converged"], report)
            self.assertFalse(report["accepted"])
            potential = solver.controlled_fold_actuation.potential(targets, activation)
            _, membrane, _ = membrane_element_derivatives(final[solver.faces], solver.poses,
                                                          solver.areas, solver.materials[:, :3])
            gradient = solver.mass[:, None] * (final - q - .01 * velocity) / .01**2
            np.add.at(gradient, solver.faces, membrane.reshape((-1, 3, 3)))
            gradient += solver.bending.gradient(final) + potential.gradient(final) + solver.contact.gradient(final)
            self.assertLessEqual(np.max(np.abs(gradient)), 1e-6)
            self.assertAlmostEqual(np.max(np.abs(gradient)), report["gradientInfinityNorm"], delta=1e-11)
            np.testing.assert_array_equal(v, (final-q)/.01)
            self.assertEqual(report["foldControls"], potential.diagnostics(final))
            self.assertEqual(report["foldHingeSweepPolicy"], POLICY)
            self.assertTrue(hinge_sweep_safe(q, final, solver.controlled_fold_actuation.hinges))
            self.assertTrue(triangle_sweep_safe(q, final, solver.faces))
            self.assertTrue(solver.contact.path_safe(q, final))
            np.testing.assert_allclose(solver.mass @ final, solver.mass @ rest[0], atol=1e-12)
            q, velocity = final, v
        self.assertEqual(report["foldAnglesRadians"], [None, None])
        self.assertEqual(report["foldActuationJoules"], 0.)
        np.testing.assert_array_equal(model.particle_q.numpy(), rest[0])
        np.testing.assert_array_equal(solver.poses, rest[1])
        np.testing.assert_array_equal(solver.bending.rest_angles, rest[2])

    def test_all_inactive_retains_full_hinge_guards_and_active_residual_shape(self):
        import solver_global_sewing
        model, solver, q = fixture()
        original = solver_global_sewing._direct_descent
        seen = []

        def hinge_guard(start, end, hinges):
            seen.append(np.asarray(hinges).copy())
            np.testing.assert_array_equal(hinges, solver.controlled_fold_actuation.hinges)
            return not np.array_equal(end, -q)

        def inspect(evaluate, *args, **kwargs):
            expected = q.size + 7*len(solver.faces) + len(solver.bending.indices)
            self.assertEqual(evaluate(q.ravel()).shape, (expected,))
            collapsed = q.copy(); collapsed[0] = collapsed[2]
            self.assertEqual(evaluate(collapsed.ravel()).shape, (expected,))
            self.assertTrue(np.isinf(evaluate(collapsed.ravel())).all())
            middle = q.copy(); middle[:, 2] += .001
            # Isolate both hinge checks from the independently tested triangle
            # guard. Current optimization chord and original physical chord
            # must each be checked using all released hinges.
            with patch('solver_triangle_sweep.triangle_sweep_safe', return_value=True):
                self.assertEqual(kwargs['energy_change_function'](middle.ravel(), (-q).ravel()), np.inf)
            return original(evaluate, *args, **kwargs)

        with patch('solver_hinge_sweep.hinge_sweep_safe', side_effect=hinge_guard), \
             patch.object(solver_global_sewing, '_direct_descent', side_effect=inspect):
            _, _, report = solver.step(q, np.zeros_like(q), EMPTY, .01,
                fold_targets=[.3, -.4], fold_activation=[0., 0.])
        self.assertTrue(report["converged"])
        self.assertGreaterEqual(len(seen), 3)
        self.assertEqual(report["foldControls"]["inactiveHingeIndices"], [0, 1])

    def test_inactive_initial_and_final_hinge_guard_failures_reject(self):
        _, solver, q = fixture()
        with patch('solver_hinge_sweep.hinge_sweep_safe', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Initial controlled fold'):
                solver.step(q, np.zeros_like(q), EMPTY, .01, fold_targets=[0., 0.], fold_activation=[0., 0.])
        fake = SimpleNamespace(x=q.ravel(), success=True, nfev=1, status=1, message="injected stationary result")
        with patch('solver_global_sewing._direct_descent', return_value=fake), \
             patch('solver_hinge_sweep.hinge_sweep_safe', side_effect=[True, False]):
            with self.assertRaisesRegex(ValueError, 'Physical fold step'):
                solver.step(q, np.zeros_like(q), EMPTY, .01, fold_targets=[0., 0.], fold_activation=[0., 0.])

    def test_released_hinge_physical_branch_crossing_rejects_even_with_safe_optimizer_chord(self):
        from solver_hinge_sweep import hinge_sweep_safe
        from solver_triangle_sweep import triangle_sweep_safe
        _, solver, flat = fixture(count=1)

        def folded(angle):
            q = flat.copy()
            q[1, 1] = -.01 * np.cos(angle)
            q[1, 2] = -.01 * np.sin(angle)
            return q

        previous, end = folded(3.1), folded(-3.1)
        hinges = solver.controlled_fold_actuation.hinges
        self.assertTrue(triangle_sweep_safe(previous, end, solver.faces))
        self.assertTrue(hinge_sweep_safe(flat, end, hinges))
        self.assertFalse(hinge_sweep_safe(previous, end, hinges))

        def inspect(evaluate, *args, **kwargs):
            # The proposed optimization segment is valid, but its endpoint
            # cannot be reached from the previous physical state on this
            # saved affine chord. Activation zero does not waive that guard.
            self.assertEqual(kwargs['energy_change_function'](flat.ravel(), end.ravel()), np.inf)
            return SimpleNamespace(x=previous.ravel(), success=False, nfev=1, status=0, message="guard inspection only")

        with patch('solver_global_sewing._direct_descent', side_effect=inspect):
            _, _, report = solver.step(previous, np.zeros_like(previous), EMPTY, .01,
                fold_targets=[0.], fold_activation=[0.])
        self.assertFalse(report['converged'])

    def test_inactive_final_triangle_and_contact_guards_are_mandatory(self):
        _, solver, q = fixture(contact_enabled=True)
        fake = SimpleNamespace(x=q.ravel(), success=True, nfev=1, status=1, message="injected stationary result")
        with patch('solver_global_sewing._direct_descent', return_value=fake), \
             patch('solver_triangle_sweep.triangle_sweep_safe', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Physical cloth step'):
                solver.step(q, np.zeros_like(q), EMPTY, .01, fold_targets=[0., 0.], fold_activation=[0., 0.])
        with patch('solver_global_sewing._direct_descent', return_value=fake), \
             patch.object(solver.contact, 'path_safe', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Physical contact step'):
                solver.step(q, np.zeros_like(q), EMPTY, .01, fold_targets=[0., 0.], fold_activation=[0., 0.])


if __name__ == '__main__':
    unittest.main()
