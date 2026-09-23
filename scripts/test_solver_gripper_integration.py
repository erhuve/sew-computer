"""Independent mechanics and guard checks for the opt-in global gripper term.

Dense quadratic solutions below are assembled from declared source weights,
not the actuator's matrix, derivatives, or residual implementation.
"""
import unittest
from unittest.mock import patch

import newton
import numpy as np
from scipy.optimize import OptimizeResult
import warp as wp

from solver_global_sewing import GlobalSewingSolver
from solver_material_grippers import MaterialGrippers
from solver_triangle_sweep import triangle_sweep_safe


def make_model(vertices=None, faces=None, *, membrane=0., bending=0.):
    if vertices is None:
        vertices = [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]]
    if faces is None:
        faces = [[0, 1, 2]]
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
        vel=wp.vec3(0, 0, 0), vertices=vertices, indices=np.asarray(faces).ravel().tolist(),
        density=2., tri_ke=membrane, tri_ka=membrane, tri_kd=0,
        edge_ke=bending, edge_kd=0)
    builder.set_coloring([[i] for i in range(len(vertices))])
    return builder.finalize(device="cpu")


def make_recipe(model, weights=(.25, .25, .5), stiffness=8., *, faces=None):
    return MaterialGrippers(["test-panel"] * model.particle_count,
        model.tri_indices.numpy() if faces is None else faces,
        [{"id": "grip", "instanceId": "test-panel", "triangleIndex": 0,
          "weights": list(weights), "stiffnessNPerM": stiffness}])


class GripperIntegrationTests(unittest.TestCase):
    def test_source_triangle_matches_closed_form_motion_energy_and_external_impulse(self):
        model = make_model()
        previous = model.particle_q.numpy().astype(float)
        rest = model.particle_q.numpy().copy()
        recipe = make_recipe(model)
        solver = GlobalSewingSolver(model, [], 1e-3, material_grippers=recipe)
        poses, areas = solver.poses.copy(), solver.areas.copy()
        weights = np.array([.25, .25, .5])
        dt, stiffness, activation = .02, 8., .7
        velocity = np.array([[.01, -.005, .02], [0., .01, -.01], [-.01, .002, .004]])
        target = weights @ previous + [.003, -.002, .004]
        diagonal = np.diag(solver.mass / dt ** 2)
        matrix = diagonal + stiffness * activation * np.outer(weights, weights)
        rhs = diagonal @ (previous + dt * velocity) + stiffness * activation * weights[:, None] * target
        expected = np.linalg.solve(matrix, rhs)
        actual, actual_velocity, report = solver.step(previous, velocity, np.empty((0, 3)), dt,
            gripper_targets=[target], gripper_activation=[activation])
        self.assertTrue(report["converged"], report)
        self.assertFalse(report["accepted"])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-15)
        np.testing.assert_allclose(actual_velocity, (expected - previous) / dt, rtol=0, atol=2e-13)
        anchor = weights @ actual
        force = stiffness * activation * (target - anchor)
        nodal_force = weights[:, None] * force
        reaction = -force
        np.testing.assert_allclose(solver.mass @ (actual_velocity - velocity), dt * force,
                                   rtol=0, atol=3e-16)
        np.testing.assert_allclose(nodal_force.sum(axis=0) + reaction, 0., rtol=0, atol=1e-17)
        np.testing.assert_allclose(np.cross(actual, nodal_force).sum(axis=0)
                                   + np.cross(target, reaction), 0., rtol=0, atol=1e-17)
        diagnostics = report["gripperDiagnostics"]
        for key, expected_diagnostic in (
                ("anchorPositionsMeters", [anchor]), ("anchorForcesNewtons", [force]),
                ("toolReactionsNewtons", [reaction]), ("nodalForcesNewtons", nodal_force),
                ("totalClothForceNewtons", force), ("totalToolReactionNewtons", reaction),
                ("clothTorqueNewtonMeters", np.cross(actual, nodal_force).sum(axis=0)),
                ("toolTorqueNewtonMeters", np.cross(target, reaction)),
                ("netForceResidualNewtons", np.zeros(3)), ("netTorqueResidualNewtonMeters", np.zeros(3))):
            with self.subTest(diagnostic=key):
                np.testing.assert_allclose(diagnostics[key], expected_diagnostic, rtol=0, atol=2e-16)
        expected_energy = .5 * stiffness * activation * np.sum((anchor - target) ** 2)
        self.assertAlmostEqual(report["gripperEnergyJoules"], expected_energy, delta=1e-16)
        inertia = .5 * np.sum(solver.mass[:, None] * (actual - previous - dt * velocity) ** 2) / dt ** 2
        self.assertAlmostEqual(report["finalEnergy"], inertia + expected_energy, delta=1e-16)
        # Backward Euler's quadratic energy loss has two independently known
        # nonnegative terms; no physical damping or zero external force is assumed.
        before = .5 * np.sum(solver.mass[:, None] * velocity ** 2)
        before += .5 * stiffness * activation * np.sum((weights @ previous - target) ** 2)
        after = .5 * np.sum(solver.mass[:, None] * actual_velocity ** 2) + expected_energy
        loss = .5 * np.sum(solver.mass[:, None] * (actual_velocity - velocity) ** 2)
        loss += .5 * stiffness * activation * np.sum((weights @ (actual - previous)) ** 2)
        self.assertAlmostEqual(after - before, -loss, delta=1e-16)
        np.testing.assert_array_equal(model.particle_q.numpy(), rest)
        np.testing.assert_array_equal(solver.poses, poses)
        np.testing.assert_array_equal(solver.areas, areas)
        np.testing.assert_array_equal(previous, rest.astype(float))
        np.testing.assert_array_equal(report["gripperTargetsMeters"], [target])
        np.testing.assert_array_equal(report["gripperActivation"], [activation])

    def test_sewing_and_gripper_solve_the_same_joint_quadratic(self):
        vertices = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.],
                             [.15, 0., 0.], [.25, 0., 0.], [.15, .1, 0.]])
        model = make_model(vertices.tolist(), [[0, 1, 2], [3, 4, 5]])
        previous = model.particle_q.numpy().astype(float)
        coefficients = np.array([[.5, .25, .25, -.25, -.5, -.25]])
        rows = [{i: value for i, value in enumerate(coefficients[0])}]
        compliance, dt = .03, .01
        recipe = make_recipe(model, (.25, .5, .25), 5.)
        solver = GlobalSewingSolver(model, rows, compliance, material_grippers=recipe)
        weights = np.array([.25, .5, .25, 0., 0., 0.])
        target = weights @ previous + [.001, .002, .003]
        seam_target = coefficients @ previous + [[.002, -.001, .001]]
        diagonal = np.diag(solver.mass / dt ** 2)
        matrix = diagonal + coefficients.T @ coefficients / compliance + 5. * np.outer(weights, weights)
        expected = np.linalg.solve(matrix, diagonal @ previous
            + coefficients.T @ seam_target / compliance + 5. * weights[:, None] * target)
        actual, velocity, report = solver.step(previous, np.zeros_like(previous), seam_target, dt,
            gripper_targets=[target], gripper_activation=[1.])
        self.assertTrue(report["converged"], report)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-15)
        # The second panel moves through sewing even though its gripper weights
        # are exactly zero; internal sewing does not alter the external impulse.
        self.assertGreater(np.linalg.norm(actual[3:] - previous[3:]), 1e-6)
        force = 5. * (target - weights @ actual)
        np.testing.assert_allclose(solver.mass @ velocity, dt * force, rtol=0, atol=5e-16)
        expected_sewing = .5 * np.sum((coefficients @ actual - seam_target) ** 2) / compliance
        self.assertAlmostEqual(report["sewingJoules"], expected_sewing, delta=1e-16)

    def test_zero_activation_leaves_free_inertial_motion_even_with_remote_target(self):
        model = make_model()
        previous = model.particle_q.numpy().astype(float)
        solver = GlobalSewingSolver(model, [], 1e-3, material_grippers=make_recipe(model))
        velocity = np.broadcast_to([.01, -.02, .03], previous.shape).copy()
        actual, actual_velocity, report = solver.step(previous, velocity, np.empty((0, 3)), .01,
            gripper_targets=[[100., -100., 100.]], gripper_activation=[0.])
        self.assertTrue(report["converged"], report)
        np.testing.assert_allclose(actual, previous + .01 * velocity, rtol=0, atol=2e-16)
        np.testing.assert_allclose(actual_velocity, velocity, rtol=0, atol=2e-14)
        self.assertEqual(report["gripperEnergyJoules"], 0.)
        np.testing.assert_array_equal(report["gripperDiagnostics"]["nodalForcesNewtons"], np.zeros_like(previous))

    def test_topology_opt_in_target_shape_and_solver_choice_reject(self):
        model = make_model()
        previous = model.particle_q.numpy().astype(float)
        with self.assertRaisesRegex(ValueError, "ordered faces"):
            GlobalSewingSolver(model, [], 1e-3,
                material_grippers=make_recipe(model, faces=[[0, 2, 1]]))
        solver = GlobalSewingSolver(model, [], 1e-3, material_grippers=make_recipe(model))
        base = {"gripper_targets": [[.02, .02, .01]], "gripper_activation": [1.]}
        for changes in ({"gripper_targets": None}, {"gripper_activation": None},
                        {"gripper_targets": [[.02, .02]]}, {"gripper_targets": [[np.nan, 0., 0.]]},
                        {"gripper_activation": [1., 1.]}, {"gripper_activation": [-.01]},
                        {"gripper_activation": [1.01]}, {"linear_solver": "lsmr"},
                        {"linear_solver": "shifted"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                solver.step(previous, np.zeros_like(previous), np.empty((0, 3)), .01, **(base | changes))
        plain = GlobalSewingSolver(model, [{0: 1., 1: -1.}], 1e-3)
        with self.assertRaisesRegex(ValueError, "require a material-gripper recipe"):
            plain.step(previous, np.zeros_like(previous), [[-.1, 0., 0.]], .01, **base)

    def test_fixed_mass_or_flag_cannot_hide_support_reactions(self):
        for kind in ("mass", "flag"):
            with self.subTest(kind=kind):
                model = make_model()
                recipe = make_recipe(model)
                if kind == "mass":
                    values = model.particle_mass.numpy()
                    values[0] = 0.
                    model.particle_mass.assign(values)
                else:
                    values = model.particle_flags.numpy()
                    values[0] = 0
                    model.particle_flags.assign(values)
                with self.assertRaisesRegex(ValueError, "free positive-mass cloth"):
                    GlobalSewingSolver(model, [], 1e-3, material_grippers=recipe)

    def test_gripper_cannot_bypass_optimizer_or_physical_triangle_sweeps(self):
        model = make_model()
        previous = model.particle_q.numpy().astype(float)
        solver = GlobalSewingSolver(model, [], 1e-3, material_grippers=make_recipe(model))
        quarter_turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        middle, end = previous @ quarter_turn, previous @ np.diag([-1., -1., 1.])
        self.assertTrue(triangle_sweep_safe(middle, end, solver.faces))
        self.assertFalse(triangle_sweep_safe(previous, end, solver.faces))
        def malicious_descent(evaluate, start, maximum, hessian, objective, **options):
            self.assertTrue(np.isfinite(evaluate(end.ravel())).all())
            self.assertTrue(np.isfinite(objective(end.ravel())))
            # First trial has a safe physical path and an unsafe optimizer path;
            # the second has the reverse. Both must fail before publication.
            self.assertTrue(np.isinf(options["energy_change_function"](end.ravel(), start)))
            self.assertTrue(np.isinf(options["energy_change_function"](middle.ravel(), end.ravel())))
            collapsed = previous.copy()
            collapsed[1] = collapsed[0]
            self.assertEqual(evaluate(collapsed.ravel()).shape, evaluate(start).shape)
            self.assertTrue(np.isinf(evaluate(collapsed.ravel())).all())
            return OptimizeResult(x=end.ravel(), success=True, status=1, message="adversarial endpoint", nfev=1)
        with patch("solver_global_sewing._direct_descent", malicious_descent), \
                self.assertRaisesRegex(ValueError, "triangle path"):
            solver.step(previous, np.zeros_like(previous), np.empty((0, 3)), .01,
                gripper_targets=[end.mean(axis=0)], gripper_activation=[1.])

    def test_gripper_cannot_bypass_contact_ccd_or_publish_layer_exchange(self):
        import ipctk
        from solver_rest_filtered_contact import RestFilteredSurfaceContact
        ipctk.set_num_threads(1)
        vertices = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.],
                             [0., 0., .01], [.02, 0., .01], [0., .02, .01]])
        model = make_model(vertices.tolist(), [[0, 1, 2], [3, 4, 5]])
        previous = model.particle_q.numpy().astype(float)
        contact = RestFilteredSurfaceContact(previous, model.tri_indices.numpy(),
            activation_distance_m=.001, minimum_distance_m=.0001, stiffness=10000,
            ccd_profile="temporal-separation-tight-inclusion")
        solver = GlobalSewingSolver(model, [], 1e-3, contact=contact, material_grippers=make_recipe(model))
        end = previous.copy()
        end[:3, 2], end[3:, 2] = previous[3:, 2], previous[:3, 2]
        self.assertTrue(triangle_sweep_safe(previous, end, solver.faces))
        contact.validate_state(end)
        self.assertFalse(contact.path_safe(previous, end))
        def malicious_descent(evaluate, start, maximum, hessian, objective, **options):
            self.assertTrue(np.isfinite(objective(end.ravel())))
            self.assertLess(options["step_limiter"](start, end.ravel()), 1.)
            self.assertTrue(np.isinf(options["energy_change_function"](start, end.ravel())))
            return OptimizeResult(x=end.ravel(), success=True, status=1, message="adversarial exchange", nfev=1)
        with patch("solver_global_sewing._direct_descent", malicious_descent), \
                self.assertRaisesRegex(ValueError, "contact step crosses a collision"):
            solver.step(previous, np.zeros_like(previous), np.empty((0, 3)), .01,
                gripper_targets=[end[:3].mean(axis=0)], gripper_activation=[1.])

    def test_combined_fold_barrier_sewing_gripper_residual_derivatives_and_rest_preservation(self):
        import solver_global_sewing
        vertices = [[.005, .01, 0.], [.005, -.01, 0.], [0., 0., 0.], [.02, 0., 0.]]
        model = make_model(vertices, [[0, 2, 3], [1, 3, 2]], membrane=100., bending=.001)
        rest = model.particle_q.numpy().astype(float)
        previous = rest.copy()
        previous[1, 1:] = [-.01 * np.cos(.2), .01 * np.sin(.2)]
        hinges = model.edge_indices.numpy()
        hinges = hinges[np.all(hinges >= 0, axis=1)]
        solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}], .001,
            fold_hinges=hinges, fold_stiffness_joules=[.02],
            fold_barrier_joules=1e-5, fold_activation_angle=.1,
            material_grippers=make_recipe(model, stiffness=.3))
        poses, angles = solver.poses.copy(), solver.bending.rest_angles.copy()
        target = np.array([.25, .25, .5]) @ previous[solver.faces[0]] + [.00002, -.00001, .0001]
        seam_target = previous[[0]] - previous[[1]] + [[0., 0., .00002]]
        real_descent = solver_global_sewing._direct_descent
        inspected = []
        def inspect(evaluate, start, maximum, hessian, objective, **options):
            inspected.append(True)
            residual = evaluate(start)
            jacobian = evaluate(start, True).toarray()
            epsilon = 1e-8
            numeric = np.column_stack([(evaluate(start + epsilon * direction)
                - evaluate(start - epsilon * direction)) / (2 * epsilon) for direction in np.eye(len(start))])
            np.testing.assert_allclose(jacobian, numeric, rtol=1e-5, atol=2e-6)
            np.testing.assert_allclose(jacobian.T @ residual, options["gradient_function"](start),
                                       rtol=2e-11, atol=2e-11)
            return real_descent(evaluate, start, maximum, hessian, objective, **options)
        with patch("solver_global_sewing._direct_descent", inspect):
            actual, velocity, report = solver.step(previous, np.zeros_like(previous), seam_target, .001,
                fold_targets=[.3], gripper_targets=[target], gripper_activation=[.8])
        self.assertEqual(inspected, [True])
        self.assertTrue(report["converged"], report)
        self.assertGreater(np.linalg.norm(actual - previous), 1e-7)
        force = .3 * .8 * (target - np.array([.25, .25, .5]) @ actual[solver.faces[0]])
        np.testing.assert_allclose(solver.mass @ velocity, .001 * force, rtol=0, atol=5e-9)
        np.testing.assert_array_equal(solver.poses, poses)
        np.testing.assert_array_equal(solver.bending.rest_angles, angles)
        np.testing.assert_array_equal(model.particle_q.numpy(), rest)


if __name__ == "__main__":
    unittest.main()
