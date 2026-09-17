from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

from solver_embedded_sewing import _ENERGY, _PARAMETERS, _load, install_embedded_sewing, set_embedded_targets, uninstall_embedded_sewing, validate_rows


def fixture(translation=0.0, pinned=False, iterations=100):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_particle(pos=(translation, 0, 0), vel=(0, 0, 0), mass=0 if pinned else 1)
    builder.add_particle(pos=(translation + 1, 0, 0), vel=(0, 0, 0), mass=1)
    builder.set_coloring([[0], [1]])
    model = builder.finalize(device="cpu")
    return newton.solvers.SolverVBD(model, iterations=iterations)


class EmbeddedSewingTests(unittest.TestCase):
    def run_pair(self, translation=0, pinned=False, target=0):
        solver = fixture(translation, pinned)
        state_in, state_out = solver.model.state(), solver.model.state()
        with tempfile.TemporaryDirectory() as directory:
            report = install_embedded_sewing(solver, [{0: 1.0, 1: -1.0}], 0.01, directory)
            self.assertFalse(report["accepted"])
            set_embedded_targets(solver, [[target, 0, 0]])
            try:
                solver.step(state_in, state_out, None, None, 0.1)
                return state_out.particle_q.numpy()
            finally:
                uninstall_embedded_sewing(solver)

    def test_implicit_solution(self):
        positions = self.run_pair()
        np.testing.assert_allclose(positions[:, 0], [1 / 3, 2 / 3], atol=2e-6)
        np.testing.assert_allclose(positions[:, 1:], 0, atol=1e-7)

    def test_pins_translation_and_ramp(self):
        np.testing.assert_allclose(self.run_pair(pinned=True)[:, 0], [0, 0.5], atol=2e-6)
        np.testing.assert_allclose(self.run_pair(translation=2)[:, 0], [2 + 1 / 3, 2 + 2 / 3], atol=2e-6)
        np.testing.assert_allclose(self.run_pair(target=-1)[:, 0], [0, 1], atol=2e-6)

    def test_hypergraph_and_invalid_rows(self):
        validate_rows([{0: 0.5, 1: 0.5, 2: -1}], 3, [0, 1, 2])
        for row, colors in [({0: 0.5, 1: 0.5, 2: -1}, [0, 0, 1]), ({0: 1, 1: 1}, [0, 1, 2]),
                            ({0: float("nan"), 1: -1}, [0, 1, 2]), ({0: 1, 3: -1}, [0, 1, 2])]:
            with self.assertRaises(ValueError):
                validate_rows([row], 3, colors)

    def test_lifecycle_and_invalid_targets(self):
        solver = fixture()
        original = solver._solve_particle_iteration
        with tempfile.TemporaryDirectory() as directory:
            install_embedded_sewing(solver, [{0: 1, 1: -1}], 0.01, directory)
            with self.assertRaises(ValueError):
                install_embedded_sewing(solver, [{0: 1, 1: -1}], 0.01, directory)
            with self.assertRaises(ValueError):
                set_embedded_targets(solver, [[float("nan"), 0, 0]])
            uninstall_embedded_sewing(solver)
            self.assertEqual(original, solver._solve_particle_iteration)
            with self.assertRaises(ValueError):
                set_embedded_targets(solver, [[0, 0, 0]])

    def test_fractional_anchor_analytic_solution(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for position, mass in [((0, 0, 0), 0), ((1, 0, 0), 1), ((0, 0, 0), 0)]:
            builder.add_particle(pos=position, vel=(0, 0, 0), mass=mass)
        builder.set_coloring([[0], [1], [2]])
        solver = newton.solvers.SolverVBD(builder.finalize(device="cpu"), iterations=1)
        state_in, state_out = solver.model.state(), solver.model.state()
        with tempfile.TemporaryDirectory() as directory:
            install_embedded_sewing(solver, [{0: 0.5, 1: 0.5, 2: -1}], 0.01, directory)
            try:
                solver.step(state_in, state_out, None, None, 0.1)
                np.testing.assert_allclose(state_out.particle_q.numpy()[:, 0], [0, 0.8, 0], atol=1e-6)
            finally:
                uninstall_embedded_sewing(solver)

    def test_reject_changed_runtime_and_bindings(self):
        from newton._src.solvers.vbd import solver_vbd

        solver = fixture()
        with tempfile.TemporaryDirectory() as directory:
            with patch("solver_embedded_sewing._PINNED_KERNEL_DIGEST", "wrong"):
                with self.assertRaises(ValueError):
                    install_embedded_sewing(solver, [{0: 1, 1: -1}], 0.01, directory)
            self.assertFalse(list(Path(directory).iterdir()))
            install_embedded_sewing(solver, [{0: 1, 1: -1}], 0.01, directory)
            try:
                with patch.object(solver_vbd, "accumulate_self_contact_force_and_hessian", object()):
                    with self.assertRaises(ValueError):
                        solver._solve_particle_iteration(None, None, None, 0.1, 0)
            finally:
                uninstall_embedded_sewing(solver)

    def test_six_vertex_sparse_solution_and_derivatives(self):
        rest = np.array([[0, 0, 0], [1, 0.2, 0], [0.3, 1, 0.1], [0, 0, 1], [1, 0.4, 1], [0.2, 1, 1.2]])
        masses = np.array([0.3, 1, 2, 0.4, 1.5, 0.8])
        coefficients = np.array([0.2, 0.3, 0.5, -0.1, -0.6, -0.3])
        target = np.array([0.05, -0.03, 0.01])
        compliance, timestep = 0.03, 0.1
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for position, mass in zip(rest, masses):
            builder.add_particle(pos=position, vel=(0, 0, 0), mass=mass)
        builder.set_coloring([[vertex] for vertex in range(6)])
        solver = newton.solvers.SolverVBD(builder.finalize(device="cpu"), iterations=100)
        state_in, state_out = solver.model.state(), solver.model.state()
        with tempfile.TemporaryDirectory() as directory:
            install_embedded_sewing(solver, [dict(enumerate(coefficients))], compliance, directory, [target])
            try:
                solver.step(state_in, state_out, None, None, timestep)
                diagonal = np.diag(masses / timestep ** 2)
                expected = np.linalg.solve(diagonal + np.outer(coefficients, coefficients) / compliance,
                                           diagonal @ rest + coefficients[:, None] * target / compliance)
                np.testing.assert_allclose(state_out.particle_q.numpy(), expected, atol=2e-6)
                probe_source = "import warp as wp\n@wp.kernel\ndef probe(pos: wp.array[wp.vec3],\n" + _PARAMETERS + "    forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):\n    particle_index = wp.tid()\n    f = wp.vec3(0.0)\n    h = wp.mat33(0.0)\n" + _ENERGY + "    forces[particle_index] = f\n    hessians[particle_index] = h\n"
                probe, _ = _load(probe_source, Path(directory) / "probe.py", "sewing_probe")
                forces = wp.zeros(6, dtype=wp.vec3, device="cpu")
                hessians = wp.zeros(6, dtype=wp.mat33, device="cpu")
                wp.launch(probe.probe, dim=6, inputs=[wp.array(rest, dtype=wp.vec3, device="cpu"), *solver._embedded_sewing_inputs, forces, hessians], device="cpu")
                def energy(positions):
                    residual = coefficients @ positions - target
                    return np.dot(residual, residual) / (2 * compliance)
                step = 1e-4
                numeric_force, numeric_hessian = np.zeros((6, 3)), np.zeros((6, 3, 3))
                for vertex in range(6):
                    for axis in range(3):
                        delta = np.zeros_like(rest)
                        delta[vertex, axis] = step
                        numeric_force[vertex, axis] = -(energy(rest + delta) - energy(rest - delta)) / (2 * step)
                        for second_axis in range(3):
                            second_delta = np.zeros_like(rest)
                            second_delta[vertex, second_axis] = step
                            numeric_hessian[vertex, axis, second_axis] = (energy(rest + delta + second_delta) - energy(rest + delta - second_delta) - energy(rest - delta + second_delta) + energy(rest - delta - second_delta)) / (4 * step ** 2)
                np.testing.assert_allclose(forces.numpy(), numeric_force, atol=3e-6)
                np.testing.assert_allclose(hessians.numpy(), numeric_hessian, atol=3e-6)
            finally:
                uninstall_embedded_sewing(solver)

    def check_point_contact_coexistence(self, augmented):
        from solver_point_contact import install_point_contact, uninstall_point_contact

        vertices = [[0, 0, 0], [0.002, 0, 0], [0, 0.002, 0], [0.002, 0.002, 0]]
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for _ in range(2):
            builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0),
                                   vertices=vertices, indices=[0, 1, 2, 1, 3, 2], density=0.2,
                                   tri_ke=10000, tri_ka=10000, particle_radius=0.001)
        builder.particle_q[4:] = [wp.vec3(point[0], point[1], 0.0005) for point in vertices]
        builder.set_coloring([[vertex] for vertex in range(8)])
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, iterations=5, particle_enable_self_contact=True,
                                         particle_collision_detection_interval=1, particle_topological_contact_filter_threshold=0,
                                         particle_self_contact_margin=0.003, particle_self_contact_gap=0.001)
        with tempfile.TemporaryDirectory() as directory:
            install_point_contact(solver, vertices * 2, model.tri_indices.numpy(), ["shell"] * 4 + ["facing"] * 4, directory, 0.003)
            try:
                install_embedded_sewing(solver, [{0: 1, 4: -1}], 1.0, directory, [[0, 0, -0.0005]], augmented=augmented)
                try:
                    state, next_state = model.state(), model.state()
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts)
                    solver.step(state, next_state, model.control(), contacts, 1 / 240)
                    positions = next_state.particle_q.numpy()
                    self.assertTrue(np.isfinite(positions).all())
                    self.assertGreater(float(positions[4:, 2].mean() - positions[:4, 2].mean()), 0.00051)
                finally:
                    uninstall_embedded_sewing(solver)
            finally:
                uninstall_point_contact(solver)

    def test_point_contact_coexistence(self):
        self.check_point_contact_coexistence(False)

    def test_augmented_point_contact_coexistence(self):
        self.check_point_contact_coexistence(True)

    def test_augmented_hard_pair_preserves_momentum(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for position in [(0, 0, 0), (1, 0, 0)]:
            builder.add_particle(pos=position, vel=(0, 0, 0), mass=1e-4)
        builder.set_coloring([[0], [1]])
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, iterations=50)
        state_in, state_out = model.state(), model.state()
        with tempfile.TemporaryDirectory() as directory:
            report = install_embedded_sewing(solver, [{0: 1, 1: -1}], 1e-8, directory, augmented=True)
            try:
                self.assertEqual(report["algorithm"], "auxiliary-separation-augmented-lagrangian")
                set_embedded_targets(solver, [[0, 0, 0]])
                timestep = 1 / 480
                solver.step(state_in, state_out, None, None, timestep)
                inertial = 1e-4 / timestep ** 2
                expected_gap = inertial / (inertial + 2e8)
                np.testing.assert_allclose(state_out.particle_q.numpy()[:, 0], [(1 - expected_gap) / 2, (1 + expected_gap) / 2], atol=2e-6)
                self.assertLess(abs(float(state_out.particle_qd.numpy()[:, 0].sum())), 0.002)
            finally:
                uninstall_embedded_sewing(solver)

    def test_augmented_pins_and_target_reset(self):
        solver = fixture(pinned=True, iterations=50)
        state_in, state_out = solver.model.state(), solver.model.state()
        with tempfile.TemporaryDirectory() as directory:
            install_embedded_sewing(solver, [{0: 1, 1: -1}], .01, directory, augmented=True)
            try:
                solver.step(state_in, state_out, None, None, .1)
                np.testing.assert_allclose(state_out.particle_q.numpy()[:, 0], [0, .5], atol=2e-6)
                set_embedded_targets(solver, [[-1, 0, 0]])
                state_in, state_out = solver.model.state(), solver.model.state()
                solver.step(state_in, state_out, None, None, .1)
                np.testing.assert_allclose(state_out.particle_q.numpy()[:, 0], [0, 1], atol=2e-6)
            finally:
                uninstall_embedded_sewing(solver)

    def test_augmented_rejects_all_fixed_row(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for position in [(0, 0, 0), (1, 0, 0)]:
            builder.add_particle(pos=position, vel=(0, 0, 0), mass=0)
        builder.set_coloring([[0], [1]])
        solver = newton.solvers.SolverVBD(builder.finalize(device="cpu"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                install_embedded_sewing(solver, [{0: 1, 1: -1}], 1e-8, directory, augmented=True)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_augmented_weighted_stiff_solution(self):
        rest = np.array([[0, 0, 0], [1, .2, .1], [.4, .6, .8], [1, 1, 1]])
        masses = np.array([1e-4, 2e-4, 3e-4, 4e-4])
        coefficients = np.array([.4, .6, -.3, -.7])
        target = np.array([.1, -.1, 0])
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for position, mass in zip(rest, masses):
            builder.add_particle(pos=position, vel=(0, 0, 0), mass=mass)
        builder.set_coloring([[vertex] for vertex in range(4)])
        solver = newton.solvers.SolverVBD(builder.finalize(device="cpu"), iterations=100)
        timestep, compliance = 1 / 480, 1e-8
        with tempfile.TemporaryDirectory() as directory:
            install_embedded_sewing(solver, [dict(enumerate(coefficients))], compliance, directory, [target], augmented=True)
            try:
                state_in, state_out = solver.model.state(), solver.model.state()
                solver.step(state_in, state_out, None, None, timestep)
                diagonal = np.diag(masses / timestep ** 2)
                expected = np.linalg.solve(diagonal + np.outer(coefficients, coefficients) / compliance,
                                           diagonal @ rest + coefficients[:, None] * target / compliance)
                np.testing.assert_allclose(state_out.particle_q.numpy(), expected, atol=2e-6)
                np.testing.assert_allclose(masses @ state_out.particle_q.numpy(), masses @ rest, atol=2e-9)
            finally:
                uninstall_embedded_sewing(solver)


if __name__ == "__main__":
    unittest.main()
