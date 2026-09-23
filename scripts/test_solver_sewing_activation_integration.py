"""Independent global mechanics and guard checks for optional seam activation."""
import unittest
from unittest.mock import patch

import newton
import numpy as np
from scipy.optimize import OptimizeResult
import warp as wp

from solver_global_sewing import GlobalSewingSolver
from solver_triangle_sweep import triangle_sweep_safe


def particles(points=None, masses=None):
    if points is None:
        points = [[0., 0., 0.], [.2, .1, -.1], [-.1, .2, .05], [.3, -.1, .2]]
    if masses is None:
        masses = [1., 2., 3., 4.]
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for point, mass in zip(points, masses):
        builder.add_particle(pos=point, vel=(0, 0, 0), mass=mass)
    builder.set_coloring([[index] for index in range(len(points))])
    return builder.finalize(device="cpu")


def cloth(points=None):
    if points is None:
        points = [[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.],
                  [0., 0., .01], [.02, 0., .01], [0., .02, .01]]
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
        vel=wp.vec3(0, 0, 0), vertices=points, indices=[0, 1, 2, 3, 4, 5],
        density=2., tri_ke=0, tri_ka=0, tri_kd=0, edge_ke=0, edge_kd=0)
    builder.set_coloring([[index] for index in range(6)])
    return builder.finalize(device="cpu")


def sparse_rows(array):
    return [{index: float(value) for index, value in enumerate(row) if value != 0} for row in array]


class SewingActivationIntegrationTests(unittest.TestCase):
    def test_adaptive_caller_cannot_bypass_captured_schedule_and_work_accounting(self):
        from solver_adaptive_contact import adaptive_contact_step
        model = particles()
        previous = model.particle_q.numpy().astype(float)
        solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}], .1)
        targets = solver.sewing @ previous
        with patch.object(solver, "step") as step, self.assertRaisesRegex(ValueError, "captured activation schedule"):
            adaptive_contact_step(solver, previous, np.zeros_like(previous), targets, targets,
                                  .01, sewing_activation=[0.])
        step.assert_not_called()

    def assert_status(self, report, activation, errors, *, explicit=True):
        np.testing.assert_array_equal(report["sewingActivation"], activation)
        self.assertIs(report["sewingActivationExplicit"], explicit)
        self.assertEqual(report["activeSewingRows"], [i for i, value in enumerate(activation) if value > 0])
        self.assertEqual(report["pendingSewingRows"], [i for i, value in enumerate(activation) if value == 0])
        self.assertEqual(len(report["sewingRowTargetErrorsM"]), len(activation))
        self.assertTrue(isinstance(report["sewingTargetErrorMetric"], str) and report["sewingTargetErrorMetric"])
        for actual, expected, active in zip(report["sewingRowTargetErrorsM"], errors, activation):
            if active == 0:
                self.assertIsNone(actual)
            else:
                self.assertAlmostEqual(actual, expected, delta=2e-14 * max(1., abs(expected)))
        active_errors = [error for error, active in zip(errors, activation) if active > 0]
        self.assertAlmostEqual(report["sewingTargetErrorM"], max(active_errors, default=0.),
                               delta=2e-14 * max(1., max(active_errors, default=0.)))

    def test_mixed_vector_activation_matches_dense_solution_including_initializer_and_hessian(self):
        import solver_global_sewing
        model = particles()
        previous = model.particle_q.numpy().astype(float)
        matrix = np.array([[.25, .75, -.4, -.6], [1., -1., 0., 0.], [0., 0., 1., -1.]])
        activation = np.array([.125, 0., .8])
        targets = np.array([[.1, -.02, .03], [1000., -2000., 3000.], [-.2, .1, -.1]])
        velocity = np.array([[.1, -.2, .03], [0., .02, -.01], [-.02, 0., .01], [.04, -.01, 0.]])
        compliance, dt = .03, .07
        solver = GlobalSewingSolver(model, sparse_rows(matrix), compliance)
        mass_matrix = np.diag(solver.mass / dt ** 2)
        expected_hessian = mass_matrix + matrix.T @ np.diag(activation / compliance) @ matrix
        rhs = mass_matrix @ (previous + dt * velocity) + matrix.T @ ((activation / compliance)[:, None] * targets)
        expected = np.linalg.solve(expected_hessian, rhs)
        original = solver_global_sewing._direct_descent
        starts = []
        def inspect(evaluate, start, maximum, hessian, objective, **options):
            starts.append(start.copy())
            np.testing.assert_allclose(start.reshape((-1, 3)), expected, rtol=0, atol=4e-16)
            np.testing.assert_allclose(hessian(start).toarray(), np.kron(expected_hessian, np.eye(3)), rtol=2e-15, atol=1e-13)
            return original(evaluate, start, maximum, hessian, objective, **options)
        with patch("solver_global_sewing._direct_descent", inspect):
            actual, actual_velocity, report = solver.step(previous, velocity, targets, dt,
                sewing_activation=activation, max_evaluations=2)
        self.assertEqual(len(starts), 1)
        self.assertTrue(report["converged"], report)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-16)
        np.testing.assert_allclose(solver.mass @ (actual_velocity - velocity), 0., rtol=0, atol=1e-13)
        error = matrix @ actual - targets
        energy = np.sum(activation[:, None] * error ** 2) / (2 * compliance)
        self.assertAlmostEqual(report["sewingJoules"], energy, delta=2e-15)
        self.assert_status(report, activation, np.max(np.abs(error), axis=1))
        np.testing.assert_array_equal(solver.sewing.toarray(), matrix)

    def test_zero_vector_rows_leave_free_cloth_motion_despite_huge_target_errors(self):
        model = cloth()
        previous = model.particle_q.numpy().astype(float)
        solver = GlobalSewingSolver(model, [{0: -1., 3: 1.}, {1: -1., 4: 1.}], .001)
        row_values, poses, faces = solver.sewing.toarray().copy(), solver.poses.copy(), solver.faces.copy()
        velocity = np.broadcast_to([.02, -.03, .01], previous.shape).copy()
        targets = np.array([[1e100, -1e100, 1e100], [-1e100, 1e100, -1e100]])
        target_copy = targets.copy()
        actual, new_velocity, report = solver.step(previous, velocity, targets, .01, sewing_activation=[0., 0.])
        self.assertTrue(report["converged"], report)
        np.testing.assert_allclose(actual, previous + .01 * velocity, rtol=0, atol=1e-16)
        np.testing.assert_allclose(new_velocity, velocity, rtol=0, atol=1e-14)
        self.assertEqual(report["sewingJoules"], 0.)
        self.assert_status(report, [0., 0.], [None, None])
        np.testing.assert_array_equal(solver.sewing.toarray(), row_values)
        np.testing.assert_array_equal(solver.poses, poses)
        np.testing.assert_array_equal(solver.faces, faces)
        np.testing.assert_array_equal(model.particle_q.numpy(), previous)
        np.testing.assert_array_equal(targets, target_copy)

    def test_pending_distance_coincidence_is_inert_and_positive_activation_rejects(self):
        model = particles([[.1, .2, .3], [.1, .2, .3]], [1., 2.])
        previous = model.particle_q.numpy().astype(float)
        solver = GlobalSewingSolver(model, [{0: -1., 1: 1.}], .001, sewing_mode="distance")
        for velocity in (np.zeros_like(previous), np.array([[.01, 0., 0.], [-.02, 0., 0.]])):
            actual, new_velocity, report = solver.step(previous, velocity, [100.], .01, sewing_activation=[0.])
            self.assertTrue(report["converged"], report)
            np.testing.assert_allclose(actual, previous + .01 * velocity, rtol=0, atol=1e-16)
            np.testing.assert_allclose(new_velocity, velocity, rtol=0, atol=1e-14)
            self.assertEqual(report["sewingJoules"], 0.)
            self.assert_status(report, [0.], [None])
        for activation in ([1.], [1e-12]):
            with self.subTest(activation=activation), self.assertRaisesRegex(ValueError, "coincident"):
                solver.step(previous, np.zeros_like(previous), [.01], .01, sewing_activation=activation)

    def test_pending_normal_mismatch_is_inert_but_invalid_cloth_frame_still_rejects(self):
        model = cloth()
        previous = model.particle_q.numpy().astype(float)
        rows = [{0: -.2, 1: -.3, 2: -.5, 3: 1.}]
        solver = GlobalSewingSolver(model, rows, .001, sewing_mode="normal-offset",
                                   sewing_frame_faces=[[0, 1, 2]], sewing_sides=[-1])
        velocity = np.broadcast_to([.01, -.02, .03], previous.shape).copy()
        actual, new_velocity, report = solver.step(previous, velocity, [100.], .01, sewing_activation=[0.])
        self.assertTrue(report["converged"], report)
        np.testing.assert_allclose(actual, previous + .01 * velocity, rtol=0, atol=1e-16)
        np.testing.assert_allclose(new_velocity, velocity, rtol=0, atol=1e-14)
        self.assertEqual(report["sewingJoules"], 0.)
        self.assert_status(report, [0.], [None])
        invalid = previous.copy()
        invalid[1] = invalid[0]
        with self.assertRaises(ValueError):
            solver.step(invalid, np.zeros_like(invalid), [100.], .01, sewing_activation=[0.])

    def test_active_normal_frame_reactions_and_global_curvature_match_independent_energy(self):
        import solver_global_sewing
        points = [[0., 0., 0.], [.04, .003, .002], [0., .05, -.002],
                  [.011, .02, .01], [.05, .02, .01], [.011, .06, .01]]
        model = cloth(points)
        previous = model.particle_q.numpy().astype(float)
        matrix = np.array([[-.2, -.3, -.5, 1., 0., 0.], [-.4, -.1, -.5, 0., 1., 0.], [-1., 0., 0., 0., 0., 1.]])
        activation, target, signs = np.array([.25, 0., .8]), np.array([.002, 50., .003]), np.array([1, -1, 1])
        compliance, dt = .003, .001
        solver = GlobalSewingSolver(model, sparse_rows(matrix), compliance, sewing_mode="normal-offset",
                                   sewing_frame_faces=[[0, 1, 2]] * 3, sewing_sides=signs)
        poses, source_rows = solver.poses.copy(), solver.sewing.toarray().copy()
        def residual(q):
            normal = np.cross(q[1] - q[0], q[2] - q[0])
            # Analytic extension, not a conjugating complex norm: supports an
            # independent complex-step derivative of the real source-frame energy.
            normal /= np.sqrt(np.sum(normal * normal))
            return matrix @ q - (target * signs)[:, None] * normal
        def energy(q):
            error = residual(q)
            return np.sum(activation[:, None] * error * error) / (2 * compliance)
        def gradient(q):
            epsilon = 1e-24
            return np.array([np.imag(energy(q.astype(complex) + 1j * epsilon * direction.reshape(q.shape))) / epsilon
                             for direction in np.eye(q.size)])
        potential = solver.sewing_potential(target, activation=activation)
        np.testing.assert_allclose(potential.gradient(previous), gradient(previous), rtol=3e-13, atol=2e-12)
        frozen = matrix.T @ ((activation / compliance)[:, None] * residual(previous))
        self.assertGreater(np.linalg.norm(potential.gradient(previous).reshape((-1, 3)) - frozen), .001)
        np.testing.assert_allclose(potential.gradient(previous).reshape((-1, 3)).sum(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(np.cross(previous, potential.gradient(previous).reshape((-1, 3))).sum(axis=0), 0., atol=1e-12)
        epsilon = 1e-7
        expected_hessian = np.column_stack([(gradient(previous + epsilon * direction.reshape(previous.shape))
            - gradient(previous - epsilon * direction.reshape(previous.shape))) / (2 * epsilon) for direction in np.eye(previous.size)])
        np.testing.assert_allclose(potential.exact_hessian(previous).toarray(), expected_hessian, rtol=2e-7, atol=2e-7)
        original = solver_global_sewing._direct_descent
        def inspect(evaluate, start, maximum, hessian, objective, **options):
            assembled = options["coupled_hessian"](start).toarray()
            expected = expected_hessian + np.diag(np.repeat(solver.mass / dt ** 2, 3))
            np.testing.assert_allclose(assembled, expected, rtol=2e-7, atol=2e-7)
            return original(evaluate, start, maximum, hessian, objective, **options)
        with patch("solver_global_sewing._direct_descent", inspect):
            actual, _, report = solver.step(previous, np.zeros_like(previous), target, dt, sewing_activation=activation)
        self.assertTrue(report["converged"], report)
        total_gradient = (solver.mass[:, None] * (actual - previous) / dt ** 2).ravel() + gradient(actual)
        self.assertLessEqual(np.max(np.abs(total_gradient)), 1e-6)
        self.assertAlmostEqual(report["sewingJoules"], float(energy(actual)), delta=2e-15)
        self.assert_status(report, activation, np.max(np.abs(residual(actual)), axis=1))
        np.testing.assert_array_equal(solver.poses, poses)
        np.testing.assert_array_equal(solver.sewing.toarray(), source_rows)
        np.testing.assert_array_equal(model.particle_q.numpy(), previous)

    def test_invalid_raw_activation_rejects_and_omission_preserves_full_strength(self):
        model = particles()
        previous = model.particle_q.numpy().astype(float)
        solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}, {2: 1., 3: -1.}], .1)
        targets = np.array([[.1, 0., 0.], [.2, 0., 0.]])
        for activation in (True, .5, [True, 0.], [False, 1.], [1., -1e-12], [1., 1.00001],
                           [np.nan, 1.], [np.inf, 0.], [[1.], [0.]], [1.], [1., 0., 0.], ["1", "0"]):
            with self.subTest(activation=activation), self.assertRaises(ValueError):
                solver.step(previous, np.zeros_like(previous), targets, .01, sewing_activation=activation)
        for method in ("lsmr", "shifted"):
            with self.assertRaises(ValueError):
                solver.step(previous, np.zeros_like(previous), targets, .01, sewing_activation=[0., 0.], linear_solver=method)
        omitted, _, original = solver.step(previous, np.zeros_like(previous), targets, .01)
        explicit, _, report = solver.step(previous, np.zeros_like(previous), targets, .01, sewing_activation=[1., 1.])
        np.testing.assert_array_equal(omitted, explicit)
        errors = np.max(np.abs(solver.sewing @ omitted - targets), axis=1)
        self.assert_status(original, [1., 1.], errors, explicit=False)
        self.assert_status(report, [1., 1.], errors)

    def test_small_positive_activation_does_not_hide_active_target_error(self):
        model = particles()
        previous = model.particle_q.numpy().astype(float)
        solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}, {2: 1., 3: -1.}], .1)
        targets = np.array([[1., 2., 3.], [1e10, 0., 0.]])
        actual, _, report = solver.step(previous, np.zeros_like(previous), targets, .01, sewing_activation=[1e-12, 0.])
        errors = np.max(np.abs(solver.sewing @ actual - targets), axis=1)
        self.assert_status(report, [1e-12, 0.], errors)
        self.assertGreater(report["sewingTargetErrorM"], 2.)

    def test_pending_normal_rows_cannot_bypass_optimizer_or_physical_triangle_sweeps(self):
        model = cloth()
        previous = model.particle_q.numpy().astype(float)
        solver = GlobalSewingSolver(model, [{0: -1., 3: 1.}], .001, sewing_mode="normal-offset",
                                   sewing_frame_faces=[[0, 1, 2]], sewing_sides=[1])
        end = previous @ np.diag([-1., -1., 1.])
        middle = previous @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertTrue(triangle_sweep_safe(middle, end, solver.faces))
        self.assertFalse(triangle_sweep_safe(previous, end, solver.faces))
        def malicious(evaluate, start, maximum, hessian, objective, **options):
            self.assertTrue(np.isfinite(evaluate(end.ravel())).all())
            self.assertTrue(np.isinf(options["energy_change_function"](end.ravel(), start)))
            self.assertTrue(np.isinf(options["energy_change_function"](middle.ravel(), end.ravel())))
            return OptimizeResult(x=end.ravel(), success=True, status=1, message="pending-frame collapse probe", nfev=1)
        with patch("solver_global_sewing._direct_descent", malicious), self.assertRaisesRegex(ValueError, "triangle path"):
            solver.step(previous, np.zeros_like(previous), [.002], .01, sewing_activation=[0.])

    def test_pending_sewing_cannot_bypass_contact_ccd_and_physical_publication(self):
        import ipctk
        from solver_rest_filtered_contact import RestFilteredSurfaceContact
        ipctk.set_num_threads(1)
        model = cloth()
        previous = model.particle_q.numpy().astype(float)
        contact = RestFilteredSurfaceContact(previous, model.tri_indices.numpy(), activation_distance_m=.001,
            minimum_distance_m=.0001, stiffness=10000, ccd_profile="temporal-separation-tight-inclusion")
        solver = GlobalSewingSolver(model, [{0: -1., 3: 1.}], .001, contact=contact)
        end = previous.copy()
        end[:3, 2], end[3:, 2] = previous[3:, 2], previous[:3, 2]
        self.assertTrue(triangle_sweep_safe(previous, end, solver.faces))
        contact.validate_state(end)
        def malicious(evaluate, start, maximum, hessian, objective, **options):
            self.assertTrue(np.isfinite(objective(end.ravel())))
            self.assertLess(options["step_limiter"](start, end.ravel()), 1.)
            self.assertTrue(np.isinf(options["energy_change_function"](start, end.ravel())))
            return OptimizeResult(x=end.ravel(), success=True, status=1, message="pending-seam layer exchange", nfev=1)
        with patch("solver_global_sewing._direct_descent", malicious), self.assertRaisesRegex(ValueError, "contact step crosses a collision"):
            solver.step(previous, np.zeros_like(previous), [[0., 0., 0.]], .01, sewing_activation=[0.])


if __name__ == "__main__":
    unittest.main()
