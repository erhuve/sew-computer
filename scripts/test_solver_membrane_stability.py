import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp
from newton._src.solvers.vbd import particle_vbd_kernels, solver_vbd

from solver_membrane_stability import generate_stable_membrane_module, install_stable_membrane, uninstall_stable_membrane


PROBE = '''import warp as wp
from MODULE import evaluate_neo_hookean_membrane_force_hessian

@wp.kernel
def probe(positions: wp.array[wp.vec3], triangles: wp.array2d[int], forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    index = wp.tid()
    force, hessian = evaluate_neo_hookean_membrane_force_hessian(index // 3, index % 3, positions, positions, triangles, wp.mat22(1.0, 0.0, 0.0, 1.0), 0.5, 1.0, 1.0, 0.0, 1.0 / 240.0)
    forces[index] = force
    hessians[index] = hessian
'''


def sample_kernel(module_name, directory, positions):
    source = PROBE.replace("MODULE", module_name)
    digest = hashlib.sha256((str(directory) + source).encode()).hexdigest()[:24]
    path = Path(directory) / f"probe_{digest}.py"
    path.write_text(source)
    specification = importlib.util.spec_from_file_location(f"membrane_probe_{digest}", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    flattened = np.asarray(positions, dtype=np.float32).reshape((-1, 3))
    count = len(flattened)
    forces, hessians = wp.zeros(count, dtype=wp.vec3, device="cpu"), wp.zeros(count, dtype=wp.mat33, device="cpu")
    wp.launch(module.probe, dim=count, inputs=[wp.array(flattened, dtype=wp.vec3, device="cpu"), wp.array(np.arange(count).reshape((-1, 3)), dtype=int, device="cpu"), forces, hessians], device="cpu")
    return forces.numpy().astype(float).reshape((-1, 3, 3)), hessians.numpy().astype(float).reshape((-1, 3, 3, 3))


def energy(positions):
    first, second = positions[1] - positions[0], positions[2] - positions[0]
    area_ratio = np.linalg.norm(np.cross(first, second))
    return 0.5 * (0.5 * (np.dot(first, first) + np.dot(second, second) - 2) + (area_ratio - 1.5) ** 2)


def finite_difference(positions, step):
    result = np.zeros((3, 3))
    for vertex in range(3):
        for axis in range(3):
            positive, negative = positions.copy(), positions.copy()
            positive[vertex, axis] += step
            negative[vertex, axis] -= step
            result[vertex, axis] = -(energy(positive) - energy(negative)) / (2 * step)
    return result


def solver_fixture(self_contact=False):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0), vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], indices=[0, 1, 2], density=0.2, tri_ke=1, tri_ka=1, tri_kd=0, edge_ke=0, edge_kd=0)
    builder.color(include_bending=True)
    model = builder.finalize(device="cpu")
    return newton.solvers.SolverVBD(model, iterations=2, particle_enable_self_contact=self_contact, particle_topological_contact_filter_threshold=0)


class StableMembraneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.set_device("cpu")

    def test_compiled_near_collinear_force_matches_independent_energy_differences(self):
        positions, steps = [], []
        for epsilon in (0.1, 0.001, 0.0003, 0.0001, 0.00001, 0.000001):
            for angle in (0, 0.37, 1.2):
                cosine, sine = np.cos(angle), np.sin(angle)
                rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
                positions.append(np.array([[0, 0, 0], [1, 0, 0], [1, epsilon, 0]]) @ rotation.T)
                steps.append(min(1e-5, epsilon / 100))
        positions = np.asarray(positions).astype(np.float32).astype(float)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as directory:
            module, report = generate_stable_membrane_module(directory)
            forces, hessians = sample_kernel(module.__name__, directory, positions)
            old_forces, _ = sample_kernel(particle_vbd_kernels.__name__, directory, positions)
            reference = np.array([finite_difference(points, step) for points, step in zip(positions, steps)])
            np.testing.assert_allclose(forces, reference, atol=5e-6, rtol=5e-6)
            self.assertGreater(np.max(np.abs(old_forces - reference)), 1e5)
            self.assertTrue(np.isfinite(hessians).all())
            np.testing.assert_allclose(hessians, np.swapaxes(hessians, -1, -2), atol=1e-6)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(hessians).min()), -1e-6)
            np.testing.assert_allclose(forces.sum(axis=1), 0, atol=3e-7)
            generated = (Path(directory) / "generated_stable_membrane.py").read_bytes()
            self.assertEqual(hashlib.sha256(generated).hexdigest(), report["generatedKernelDigest"])
            self.assertTrue(generated.startswith(Path(particle_vbd_kernels.__file__).read_bytes().split(b"\n\n", 1)[0]))
            self.assertFalse(report["accepted"])

    def test_regular_tensile_and_degenerate_states_remain_finite(self):
        positions = [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
            [[0, 0, 0], [1000, 0, 0], [1000, .002, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as directory:
            module, _ = generate_stable_membrane_module(directory)
            forces, hessians = sample_kernel(module.__name__, directory, positions)
            old_forces, old_hessians = sample_kernel(particle_vbd_kernels.__name__, directory, positions)
            self.assertTrue(np.isfinite(forces).all() and np.isfinite(hessians).all())
            np.testing.assert_allclose(forces[[0, 1, 3, 4]], old_forces[[0, 1, 3, 4]], atol=1e-6)
            np.testing.assert_allclose(hessians[[0, 1, 3, 4]], old_hessians[[0, 1, 3, 4]], atol=1e-6)
            np.testing.assert_allclose(forces[0], 0, atol=1e-7)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(hessians).min()), -1e-5)

    def test_general_three_dimensional_rotations_match_energy_gradient(self):
        axis = np.array([1.0, 2.0, 3.0])
        axis /= np.linalg.norm(axis)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) * np.cos(.83) + (1 - np.cos(.83)) * np.outer(axis, axis) + np.sin(.83) * skew
        epsilons = (.1, .01, .001, .0001)
        positions = np.array([np.array([[0, 0, 0], [1, 0, 0], [1, epsilon, 0]]) @ rotation.T for epsilon in epsilons]).astype(np.float32).astype(float)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as directory:
            module, _ = generate_stable_membrane_module(directory)
            forces, hessians = sample_kernel(module.__name__, directory, positions)
            reference = np.array([finite_difference(points, min(1e-5, epsilon / 100)) for points, epsilon in zip(positions, epsilons)])
            np.testing.assert_allclose(forces, reference, atol=4e-5, rtol=4e-5)
            self.assertTrue(np.isfinite(hessians).all())
            self.assertGreaterEqual(float(np.linalg.eigvalsh(hessians).min()), -1e-6)

    def test_installed_cpu_solver_binding_runs_and_restores(self):
        solver = solver_fixture()
        original = solver_vbd.solve_elasticity
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as directory:
            install_stable_membrane(solver, directory)
            try:
                self.assertIsNot(solver_vbd.solve_elasticity, original)
                with self.assertRaisesRegex(ValueError, "Only one"):
                    install_stable_membrane(solver, directory)
                with self.assertRaisesRegex(ValueError, "does not own"):
                    uninstall_stable_membrane(object())
                initial, result = solver.model.state(), solver.model.state()
                solver.step(initial, result, solver.model.control(), None, 1 / 240)
                np.testing.assert_allclose(result.particle_q.numpy(), initial.particle_q.numpy(), atol=1e-6)
            finally:
                uninstall_stable_membrane(solver)
            self.assertIs(solver_vbd.solve_elasticity, original)
            with self.assertRaisesRegex(ValueError, "does not own"):
                uninstall_stable_membrane(solver)

    def test_dynamic_compressed_triangle_uses_corrected_binding_with_contact_adapter(self):
        from newton._src.geometry import tri_mesh_collision
        from solver_point_contact import install_point_contact, uninstall_point_contact

        solver = solver_fixture(self_contact=True)
        compressed = np.array([[0, 0, 0], [1, 0, 0], [1, .0001, 0]], dtype=np.float32)
        original_poses = solver.model.tri_poses.numpy().copy()
        state, baseline = solver.model.state(), solver.model.state()
        state.particle_q.assign(compressed)
        solver.step(state, baseline, solver.model.control(), None, 1 / 240)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as directory:
            install_point_contact(solver, [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], ["shell"] * 3, directory, .003)
            contact_binding = tri_mesh_collision.vertex_triangle_collision_detection_kernel
            try:
                install_stable_membrane(solver, directory)
                try:
                    state, corrected = solver.model.state(), solver.model.state()
                    state.particle_q.assign(compressed)
                    solver.step(state, corrected, solver.model.control(), None, 1 / 240)
                    self.assertTrue(np.isfinite(corrected.particle_q.numpy()).all())
                    self.assertGreater(np.linalg.norm(corrected.particle_q.numpy().astype(float) - baseline.particle_q.numpy()), 1e-6)
                    np.testing.assert_array_equal(solver.model.tri_poses.numpy(), original_poses)
                    self.assertIs(tri_mesh_collision.vertex_triangle_collision_detection_kernel, contact_binding)
                finally:
                    uninstall_stable_membrane(solver)
                self.assertIs(tri_mesh_collision.vertex_triangle_collision_detection_kernel, contact_binding)
            finally:
                uninstall_point_contact(solver)

    def test_unknown_versions_sources_and_tile_solver_reject(self):
        solver = solver_fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as directory:
            with patch.object(newton, "__version__", "unknown"):
                with self.assertRaisesRegex(ValueError, "requires Newton"):
                    install_stable_membrane(solver, directory)
            with patch("solver_membrane_stability._PINNED_KERNEL_DIGEST", "invalid"):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    install_stable_membrane(solver, directory)
            with patch.object(solver, "use_particle_tile_solve", True):
                with self.assertRaisesRegex(ValueError, "direct CPU"):
                    install_stable_membrane(solver, directory)
            with patch.object(solver_vbd, "solve_elasticity", object()):
                with self.assertRaisesRegex(ValueError, "already modified"):
                    install_stable_membrane(solver, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
