import importlib.util
from pathlib import Path
import tempfile
import unittest

import newton
import numpy as np
import warp as wp

from solver_membrane_stability import generate_stable_membrane_module
from solver_strain_diagnostics import membrane_energy_report


PROBE = '''import warp as wp
from MODULE import evaluate_neo_hookean_membrane_force_hessian

@wp.kernel
def probe(positions: wp.array[wp.vec3], triangles: wp.array2d[int], poses: wp.array[wp.mat22], areas: wp.array[float], materials: wp.array2d[float], forces: wp.array[wp.vec3]):
    vertex = wp.tid()
    force, hessian = evaluate_neo_hookean_membrane_force_hessian(0, vertex, positions, positions, triangles, poses[0], areas[0], materials[0, 0], materials[0, 1], 0.0, 1.0 / 480.0)
    forces[vertex] = force
'''


class MembraneEnergyDiagnosticsTests(unittest.TestCase):
    def test_small_bulk_matches_newton_guarded_rest_referenced_energy(self):
        positions = np.array([[0., 0., 0.], [1.2, 0., 0.], [0., .8, 0.]])
        triangles = np.array([[0, 1, 2]])
        poses, areas = np.array([np.eye(2)]), np.array([.5])
        for shear, lame in ((1e-8, 2e-8), (0., 1e-8), (1e-6, 0.)):
            bulk = shear + lame
            alpha = 1 + shear / max(bulk, 1e-6)
            area_ratio = 1.2 * .8
            invariant = 1.2 ** 2 + .8 ** 2
            expected = .5 * (.5 * shear * (invariant - 2) + .5 * bulk * ((area_ratio - alpha) ** 2 - (1 - alpha) ** 2))
            actual = membrane_energy_report(positions, triangles, poses, areas, np.array([[shear, lame, 0, 0, 0]]))["joules"]
            self.assertAlmostEqual(actual, expected, delta=1e-20)

    def test_gradient_matches_compiled_force_with_actual_nonidentity_rest_and_five_material_columns(self):
        wp.init()
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
                               vel=wp.vec3(0, 0, 0), vertices=[[0, 0, 0], [.07, .01, 0], [.02, .05, 0]],
                               indices=[0, 1, 2], density=.2, tri_ke=7000, tri_ka=11000, tri_kd=.01,
                               edge_ke=0, edge_kd=0)
        model = builder.finalize(device="cpu")
        triangles = model.tri_indices.numpy()
        poses, areas, materials = model.tri_poses.numpy(), model.tri_areas.numpy(), model.tri_materials.numpy()
        self.assertEqual(materials.shape, (1, 5))
        self.assertFalse(np.allclose(poses[0], np.eye(2)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as directory:
            module, _ = generate_stable_membrane_module(directory)
            probe_path = Path(directory) / "energy_probe.py"
            probe_path.write_text(PROBE.replace("MODULE", module.__name__))
            specification = importlib.util.spec_from_file_location("energy_diagnostic_probe", probe_path)
            probe = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(probe)
            positions = model.particle_q.numpy().astype(float)
            self.assertAlmostEqual(membrane_energy_report(positions, triangles, poses, areas, materials)["joules"], 0, places=10)
            for transform in (np.array([[1.1, .2, .1], [0, .8, .2], [.1, 0, 1]]),
                              np.array([[.1, 0, .1], [.4, .03, .3], [0, 0, 1]])):
                deformed = (positions @ transform).astype(np.float32)
                forces = wp.zeros(3, dtype=wp.vec3, device="cpu")
                wp.launch(probe.probe, dim=3, inputs=[wp.array(deformed, dtype=wp.vec3, device="cpu"),
                          model.tri_indices, model.tri_poses, model.tri_areas, model.tri_materials, forces], device="cpu")
                numerical = np.zeros((3, 3))
                for vertex in range(3):
                    for axis in range(3):
                        positive, negative = deformed.astype(float), deformed.astype(float)
                        positive[vertex, axis] += 1e-7
                        negative[vertex, axis] -= 1e-7
                        numerical[vertex, axis] = -(membrane_energy_report(positive, triangles, poses, areas, materials)["joules"] - membrane_energy_report(negative, triangles, poses, areas, materials)["joules"]) / 2e-7
                np.testing.assert_allclose(forces.numpy(), numerical, atol=.001, rtol=2e-5)

    def test_energy_is_rigid_motion_invariant_and_zero_material_has_zero_energy(self):
        positions = np.array([[0, 0, 0], [2, 0, .2], [.3, 1, -.1]])
        triangles = np.array([[0, 1, 2]])
        poses, areas = np.array([[[2, -.3], [.1, .7]]]), np.array([.5])
        materials = np.array([[10, 20, .1, 0, 0]])
        rotation, _ = np.linalg.qr(np.random.default_rng(94).normal(size=(3, 3)))
        expected = membrane_energy_report(positions, triangles, poses, areas, materials)["joules"]
        actual = membrane_energy_report(positions @ rotation + [2, -1, 7], triangles, poses, areas, materials)["joules"]
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(membrane_energy_report(positions, triangles, poses, areas, materials * 0)["joules"], 0)


if __name__ == "__main__":
    unittest.main()
