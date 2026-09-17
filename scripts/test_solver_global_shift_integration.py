import unittest
from unittest.mock import patch

import newton
import numpy as np

import solver_global_sewing
from solver_global_sewing import GlobalSewingSolver
import test_solver_global_adversarial


class GlobalShiftIntegrationTests(unittest.TestCase):
    def test_compressed_membrane_uses_shifted_steps_and_converges(self):
        model = test_solver_global_adversarial.GlobalAdversarialTests().make_model()
        solver = GlobalSewingSolver(model, [{vertex: 1, vertex + 3: -1} for vertex in range(3)], 1e-8)
        positions = model.particle_q.numpy().astype(float) * .5
        positions[1, 2] = .005
        result, speed, report = solver.step(positions, np.zeros_like(positions), np.zeros((3, 3)),
                                            1 / 480, linear_solver="shifted")
        self.assertTrue(report["converged"])
        self.assertGreater(report["shiftedSteps"], 0)
        self.assertLessEqual(report["gradientInfinityNorm"], 1e-6)
        masses = model.particle_mass.numpy().astype(float)
        np.testing.assert_allclose(masses @ result, masses @ positions, atol=1e-11, rtol=0)
        np.testing.assert_allclose(masses @ speed, np.zeros(3), atol=1e-10, rtol=0)
        rotation = np.linalg.qr(np.random.default_rng(736).normal(size=(3, 3)))[0]
        moved, _, moved_report = solver.step(positions @ rotation + [1, 2, 3], np.zeros_like(positions),
                                             np.zeros((3, 3)), 1 / 480, linear_solver="shifted")
        self.assertTrue(moved_report["converged"])
        self.assertGreater(moved_report["shiftedSteps"], 0)
        np.testing.assert_allclose(moved, result @ rotation + [1, 2, 3], atol=1e-8, rtol=0)

    def test_shifted_nonlinear_covariance_and_momentum(self):
        model = test_solver_global_adversarial.GlobalAdversarialTests().make_model()
        solver = GlobalSewingSolver(model, [{vertex: 1, vertex + 3: -1} for vertex in range(3)], 1e-8)
        positions = model.particle_q.numpy().astype(float)
        positions[1] += [.02, .01, .03]
        velocities = np.tile([.3, -.1, .2], (6, 1))
        targets = np.array([[.002, 0, 0], [.001, .001, 0], [0, .001, -.001]])
        result, speed, report = solver.step(positions, velocities, targets, 1 / 480, linear_solver="shifted")
        rotation = np.linalg.qr(np.random.default_rng(735).normal(size=(3, 3)))[0]
        moved, _, moved_report = solver.step(positions @ rotation + [1, 2, 3], velocities @ rotation,
                                             targets @ rotation, 1 / 480, linear_solver="shifted")
        self.assertTrue(report["converged"])
        self.assertTrue(moved_report["converged"])
        np.testing.assert_allclose(moved, result @ rotation + [1, 2, 3], atol=1e-8, rtol=0)
        masses = model.particle_mass.numpy().astype(float)
        np.testing.assert_allclose(masses @ speed, masses @ velocities, atol=1e-10, rtol=0)
        self.assertFalse(report["accepted"])
        self.assertTrue(np.all(np.diff(report["energyHistory"]) <= 0))

    def test_no_faces_pinned_gradient_matches_sparse_and_dense_solution(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        positions = np.array([[.2, -.3, .4], [1, .2, .1], [.4, .6, .8], [1, 1, 1]])
        masses = np.array([0., .0002, .0003, .0004])
        for position, mass in zip(positions, masses):
            builder.add_particle(pos=position, vel=(0, 0, 0), mass=mass)
        builder.set_coloring([[vertex] for vertex in range(4)])
        model = builder.finalize(device="cpu")
        coefficients = np.array([.4, .6, -.3, -.7])
        solver = GlobalSewingSolver(model, [dict(enumerate(coefficients))], 1e-8)
        target = np.array([[.1, -.1, 0]])
        descent = solver_global_sewing._direct_descent

        def audited(evaluate, start, budget, hessian, objective, **options):
            for candidate in (start, start + .01):
                expected = evaluate(candidate, True).T @ evaluate(candidate)
                np.testing.assert_allclose(options["gradient_function"](candidate), expected,
                                           rtol=1e-12, atol=1e-8)
                end = candidate + .002
                np.testing.assert_allclose(options["energy_change_function"](candidate, end),
                                           objective(end) - objective(candidate), rtol=1e-11, atol=1e-8)
            return descent(evaluate, start, budget, hessian, objective, **options)

        with patch.object(solver_global_sewing, "_direct_descent", audited):
            actual, _, report = solver.step(positions, np.zeros_like(positions), target, 1 / 480,
                                             linear_solver="shifted")
        inertia = np.diag(model.particle_mass.numpy()[1:].astype(float) * 480 ** 2)
        expected = np.linalg.solve(inertia + np.outer(coefficients[1:], coefficients[1:]) / 1e-8,
                                   inertia @ positions[1:] + coefficients[1:, None]
                                   * (target - coefficients[0] * positions[0]) / 1e-8)
        np.testing.assert_allclose(actual[1:], expected, atol=1e-9)
        np.testing.assert_array_equal(actual[0], positions[0])
        self.assertTrue(report["converged"])

    def test_hessian_cache_survives_projection_and_mutated_position_buffer(self):
        model = test_solver_global_adversarial.GlobalAdversarialTests().make_model()
        solver = GlobalSewingSolver(model, [{vertex: 1, vertex + 3: -1} for vertex in range(3)], 1e-8)
        positions = model.particle_q.numpy().astype(float)
        descent = solver_global_sewing._direct_descent

        def audited(evaluate, start, budget, hessian, objective, **options):
            candidate = start.copy()
            original = options["exact_hessian"](candidate).toarray()
            hessian(candidate)
            np.testing.assert_array_equal(options["exact_hessian"](candidate).toarray(), original)
            candidate[4] += .005
            changed = options["exact_hessian"](candidate).toarray()
            self.assertGreater(np.linalg.norm(changed - original), 1)
            np.testing.assert_array_equal(options["exact_hessian"](start).toarray(), original)
            gradient = options["gradient_function"]
            energy_change = options["energy_change_function"]
            end = candidate + np.random.default_rng(755).normal(size=candidate.shape) * 1e-5
            np.testing.assert_allclose(energy_change(candidate, end), objective(end) - objective(candidate),
                                       rtol=1e-9, atol=1e-9)
            tiny_end = candidate + np.random.default_rng(756).normal(size=candidate.shape) * 1e-12
            np.testing.assert_allclose(energy_change(candidate, tiny_end),
                                       gradient(candidate) @ (tiny_end - candidate), rtol=1e-6, atol=1e-14)
            self.assertEqual(energy_change(candidate, candidate), 0.)
            numerical = np.empty_like(changed)
            for coordinate in range(len(candidate)):
                offset = np.zeros_like(candidate)
                offset[coordinate] = 1e-7
                numerical[:, coordinate] = (gradient(candidate + offset) - gradient(candidate - offset)) / 2e-7
            np.testing.assert_allclose(changed, numerical, rtol=1e-6, atol=.02)
            return descent(evaluate, start, budget, hessian, objective, **options)

        with patch.object(solver_global_sewing, "_direct_descent", audited):
            solver.step(positions, np.zeros_like(positions), np.zeros((3, 3)), 1 / 480,
                        linear_solver="shifted")


if __name__ == "__main__":
    unittest.main()
