import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

from solver_global_sewing import GlobalSewingSolver
import solver_global_sewing


class GlobalAdversarialTests(unittest.TestCase):
    def make_model(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for offset in (0, .1):
            builder.add_cloth_mesh(
                pos=wp.vec3(offset, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0),
                vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0]], indices=[0, 1, 2], density=.2,
                tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0,
            )
        builder.set_coloring([[vertex] for vertex in range(6)])
        return builder.finalize(device="cpu")

    def test_perturbed_sewing_covariance_and_momentum(self):
        model = self.make_model()
        masses = model.particle_mass.numpy().astype(float)
        solver = GlobalSewingSolver(model, [{vertex: 1, vertex + 3: -1} for vertex in range(3)], 1e-8)
        original = model.particle_q.numpy().astype(float)
        original[1] += [.02, .01, .03]
        velocity = np.tile([.3, -.1, .2], (6, 1))
        targets = np.array([[.002, 0, 0], [.001, .001, 0], [0, .001, -.001]])
        actual, actual_velocity, report = solver.step(original, velocity, targets, 1 / 480)
        rotation = np.linalg.qr(np.random.default_rng(135).normal(size=(3, 3)))[0]
        moved, _, moved_report = solver.step(
            original @ rotation + [1, 2, 3], velocity @ rotation, targets @ rotation, 1 / 480,
        )
        self.assertTrue(report["converged"])
        self.assertTrue(moved_report["converged"])
        self.assertFalse(report["accepted"])
        self.assertFalse(moved_report["accepted"])
        np.testing.assert_allclose(moved, actual @ rotation + [1, 2, 3], atol=1e-8, rtol=0)
        np.testing.assert_allclose(masses @ actual_velocity, masses @ velocity, atol=1e-10, rtol=0)

    def test_collapsed_prediction_uses_valid_initial_state(self):
        model = self.make_model()
        solver = GlobalSewingSolver(model, [{vertex: 1, vertex + 3: -1} for vertex in range(3)], 1e-8)
        original = model.particle_q.numpy().astype(float)
        # Only guarded search methods may traverse cloth states. The legacy
        # unconstrained least-squares diagnostic cannot certify its iterates.
        for method in ("direct", "shifted"):
            actual, _, report = solver.step(original, -original * 480, np.zeros((3, 3)), 1 / 480,
                                            linear_solver=method)
            self.assertTrue(np.isfinite(actual).all())
            self.assertFalse(report["accepted"])
            triangles = actual[solver.faces]
            area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                           triangles[:, 2] - triangles[:, 0]), axis=1)
            self.assertTrue(np.all(area > 1e-8))
        with self.assertRaisesRegex(ValueError, "swept nondegeneracy"):
            solver.step(original, -original * 480, np.zeros((3, 3)), 1 / 480,
                        linear_solver="lsmr")

    def test_float32_tolerance_does_not_admit_unbalanced_global_rows(self):
        with self.assertRaisesRegex(ValueError, "preserve translation"):
            GlobalSewingSolver(self.make_model(), [{0: 1, 3: -1 + 5e-8}], 1e-8)

    def test_stable_objective_matches_residual_energy_and_gradient(self):
        model = self.make_model()
        solver = GlobalSewingSolver(model, [{vertex: 1, vertex + 3: -1} for vertex in range(3)], 1e-8)
        original = model.particle_q.numpy().astype(float)
        descent = solver_global_sewing._direct_descent

        def audited_descent(evaluate, start, max_evaluations, hessian, objective, **options):
            shear = solver.materials[:, 0]
            bulk = shear + solver.materials[:, 1]
            alpha = 1 + shear / np.maximum(bulk, 1e-6)
            constant = np.sum(solver.areas * (shear + bulk / 2 * (1 - alpha) ** 2))
            generator = np.random.default_rng(517)
            for _ in range(10):
                candidate = start + generator.normal(size=start.shape) * .003
                residual = evaluate(candidate)
                self.assertAlmostEqual(objective(candidate), residual @ residual / 2 - constant, places=8)
                analytical = evaluate(candidate, True).T @ residual
                np.testing.assert_allclose(options["gradient_function"](candidate), analytical, rtol=1e-11, atol=1e-8)
                numerical = np.zeros_like(candidate)
                for coordinate in range(len(candidate)):
                    offset = np.zeros_like(candidate)
                    offset[coordinate] = 1e-7
                    numerical[coordinate] = (objective(candidate + offset) - objective(candidate - offset)) / 2e-7
                np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-3)
            return descent(evaluate, start, max_evaluations, hessian, objective, **options)

        with patch.object(solver_global_sewing, "_direct_descent", audited_descent):
            solver.step(original, np.zeros_like(original), np.zeros((3, 3)), 1 / 480)


if __name__ == "__main__":
    unittest.main()
