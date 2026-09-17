import sys
from pathlib import Path
import unittest

import newton
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/engine"))
from constraint_coloring import refine_constraint_colors


class SpringColoringDynamicsTests(unittest.TestCase):
    def simulate(self, refined, mass, iterations):
        wp.init()
        wp.set_device("cpu")
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for position in ((-0.1, 0, 0), (0.1, 0, 0)):
            builder.add_particle(position, (0, 0, 0), mass)
        builder.add_spring(0, 1, ke=1e6, kd=0, control=0)
        builder.spring_rest_length[0] = 0
        builder.color(include_bending=True)
        if refined:
            builder.set_coloring(refine_constraint_colors(2, builder.particle_color_groups, [(0, 1)]))
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, iterations=iterations, particle_enable_self_contact=False)
        state, next_state = model.state(), model.state()
        state.clear_forces()
        solver.step(state, next_state, model.control(), None, 1 / 240)
        return next_state.particle_q.numpy().astype(float)

    def test_coloring_improves_convergence_against_exact_implicit_solution(self):
        mass = 1.0
        initial = np.array([[-0.1, 0, 0], [0.1, 0, 0]])
        expected = initial / (1 + 2e6 / (mass * 240 ** 2))
        original = self.simulate(False, mass, 100)
        corrected = self.simulate(True, mass, 100)
        original_error = np.max(np.abs(original - expected))
        corrected_error = np.max(np.abs(corrected - expected))
        self.assertLess(corrected_error, 2e-6)
        self.assertLess(corrected_error, original_error / 100)

    def test_small_gap_does_not_hide_unconverged_mass_drift(self):
        positions = self.simulate(True, 0.0001, 10)
        self.assertTrue(np.isfinite(positions).all())
        self.assertLess(np.linalg.norm(positions[1] - positions[0]), 1e-5)
        self.assertGreater(np.linalg.norm(positions.mean(axis=0)), 0.09)


if __name__ == "__main__":
    unittest.main()
