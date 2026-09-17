import unittest

import newton
import numpy as np
import warp as wp

from solver_global_sewing import GlobalSewingSolver


class GlobalSewingTests(unittest.TestCase):
    def test_sewn_triangles_rigid_translation(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for offset in (0, 0.1):
            builder.add_cloth_mesh(
                pos=wp.vec3(offset, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0),
                vertices=[[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], indices=[0, 1, 2], density=0.2,
                tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0,
            )
        builder.set_coloring([[vertex] for vertex in range(6)])
        model = builder.finalize(device="cpu")
        original = model.particle_q.numpy().astype(float)
        masses = model.particle_mass.numpy().astype(float)
        timestep, compliance = 1 / 480, 1e-8
        inertial = masses[0] / timestep ** 2
        gap = 0.1 * inertial / (inertial + 2 / compliance)
        expected = original.copy()
        expected[:3, 0] += (0.1 - gap) / 2
        expected[3:, 0] -= (0.1 - gap) / 2
        solver = GlobalSewingSolver(model, [{vertex: 1, vertex + 3: -1} for vertex in range(3)], compliance)
        actual, _, report = solver.step(original, np.zeros_like(original), np.zeros((3, 3)), timestep)
        np.testing.assert_allclose(actual, expected, atol=2e-8)
        np.testing.assert_allclose(masses @ actual, masses @ original, atol=2e-11)
        self.assertFalse(report["accepted"])

    def test_weighted_anchor_with_pin(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        masses = np.array([0, 0.0002, 0.0003, 0.0004])
        original = np.array([[0, 0, 0], [1, .2, .1], [.4, .6, .8], [1, 1, 1]])
        for position, mass in zip(original, masses):
            builder.add_particle(pos=position, vel=(0, 0, 0), mass=mass)
        builder.set_coloring([[vertex] for vertex in range(4)])
        model = builder.finalize(device="cpu")
        coefficients = np.array([.4, .6, -.3, -.7])
        target = np.array([[.1, -.1, 0]])
        timestep, compliance = 1 / 480, 1e-8
        solver = GlobalSewingSolver(model, [dict(enumerate(coefficients))], compliance)
        actual, _, _ = solver.step(original, np.zeros_like(original), target, timestep)
        diagonal = np.diag(model.particle_mass.numpy()[1:].astype(float) / timestep ** 2)
        expected = np.linalg.solve(diagonal + np.outer(coefficients[1:], coefficients[1:]) / compliance,
                                   diagonal @ original[1:] + coefficients[1:, None] * target / compliance)
        np.testing.assert_allclose(actual[1:], expected, atol=2e-8)
        np.testing.assert_array_equal(actual[0], original[0])

    def test_degenerate_membrane_rejected(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for offset in (0, .1):
            builder.add_cloth_mesh(
                pos=wp.vec3(offset, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0),
                vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0]], indices=[0, 1, 2], density=.2,
                tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0,
            )
        builder.set_coloring([[vertex] for vertex in range(6)])
        model = builder.finalize(device="cpu")
        solver = GlobalSewingSolver(model, [{vertex: 1, vertex + 3: -1} for vertex in range(3)], 1e-8)
        with self.assertRaisesRegex(ValueError, "Degenerate membrane"):
            solver.step(np.zeros((6, 3)), np.zeros((6, 3)), np.zeros((3, 3)), 1 / 480)

    def test_unsupported_gravity_and_all_fixed_rejected(self):
        for mass, gravity, message in ((0, (0, 0, 0), "active positive-mass"),
                                       (1, (0, 0, -9.81), "zero gravity")):
            builder = newton.ModelBuilder(gravity=gravity)
            for position in ((0, 0, 0), (1, 0, 0)):
                builder.add_particle(pos=position, vel=(0, 0, 0), mass=mass)
            builder.set_coloring([[0], [1]])
            with self.assertRaisesRegex(ValueError, message):
                GlobalSewingSolver(builder.finalize(device="cpu"), [{0: 1, 1: -1}], 1e-8)

    def test_direct_nonlinear_descent_and_budget(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for offset in (0, .1):
            builder.add_cloth_mesh(
                pos=wp.vec3(offset, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0),
                vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0]], indices=[0, 1, 2], density=.2,
                tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0,
            )
        builder.set_coloring([[vertex] for vertex in range(6)])
        model = builder.finalize(device="cpu")
        solver = GlobalSewingSolver(model, [{vertex: 1, vertex + 3: -1} for vertex in range(3)], 1e-8)
        original = model.particle_q.numpy().astype(float)
        original[1] += [.02, .01, .03]
        masses = model.particle_mass.numpy().astype(float)
        for budget in (3, 300):
            actual, _, report = solver.step(original, np.zeros_like(original), np.zeros((3, 3)),
                                             1 / 480, max_evaluations=budget, linear_solver="direct")
            self.assertGreater(report["evaluations"], 1)
            self.assertLessEqual(report["evaluations"], budget)
            self.assertLess(report["finalEnergy"], report["initialEnergy"])
            self.assertTrue(np.all(np.diff(report["energyHistory"]) <= 0))
            np.testing.assert_allclose(masses @ actual, masses @ original, atol=2e-10)
            if budget == 3:
                self.assertFalse(report["converged"])


if __name__ == "__main__":
    unittest.main()
