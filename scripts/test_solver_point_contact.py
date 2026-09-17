import importlib.util
import tempfile
import unittest
from pathlib import Path

import newton
import numpy as np
import warp as wp

from solver_point_contact import HELPERS, install_point_contact, reference_surface, uninstall_point_contact


class PointContactTests(unittest.TestCase):
    def test_connected_fold_keeps_remote_contact_response(self):
        rest = [[column * .02, row * .02, 0] for column in range(3) for row in range(2)]
        faces = [[0, 2, 1], [1, 2, 3], [2, 4, 3], [3, 4, 5]]
        placed = [[0 if column != 1 else .02, row * .02, column * .00025] for column in range(3) for row in range(2)]
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0), vertices=rest, indices=np.asarray(faces).flatten().tolist(), density=.2, tri_ke=10000, tri_ka=10000, edge_ke=0, edge_kd=0, particle_radius=.001)
        builder.particle_q[:] = [wp.vec3(*point) for point in placed]
        builder.color(include_bending=True)
        model = builder.finalize(device="cpu")
        gaps = []
        for enabled in (False, True):
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as output:
                solver = newton.solvers.SolverVBD(model, iterations=5, particle_enable_self_contact=enabled, particle_collision_detection_interval=1, particle_topological_contact_filter_threshold=0, particle_self_contact_margin=.003, particle_self_contact_gap=.001)
                if enabled:
                    install_point_contact(solver, rest, faces, ["shell"] * 6, output, .003)
                try:
                    state, next_state = model.state(), model.state()
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts)
                    solver.step(state, next_state, model.control(), contacts, 1 / 240)
                    positions = next_state.particle_q.numpy()
                    self.assertTrue(np.isfinite(positions).all())
                    gaps.append(float(positions[4:6, 2].mean() - positions[:2, 2].mean()))
                finally:
                    if enabled:
                        uninstall_point_contact(solver)
        self.assertGreater(gaps[1], gaps[0] + 1e-5)

    def test_seam_anchor_pairs_are_local_and_paired(self):
        rest = [[0, 0, 0], [.01, 0, 0], [.01, .01, 0], [0, .01, 0]]
        faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
        pairs = [{"first": {"instanceId": "shell", "restMeters": [coordinate, .001]}, "second": {"instanceId": "facing", "restMeters": [coordinate, .001]}} for coordinate in (.001, .009)]
        reference = reference_surface(rest + rest, faces, ["shell"] * 4 + ["facing"] * 4, .003, pairs, .0005)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as output:
            path = Path(output) / "seam_probe.py"
            path.write_text(HELPERS + '''
@wp.kernel
def probe(reference: wp.array[wp.vec3], first: wp.array[wp.vec3], second: wp.array[wp.vec3], descriptors: wp.array[int], result: wp.array[int]):
    index = wp.tid()
    if sewn_material_neighbors(reference, first[index], second[index], reference[descriptors[index]]):
        result[index] = 1
''')
            specification = importlib.util.spec_from_file_location("seam_probe", path)
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            first = [[.001, .001, 0], [.001, .001, 0], [.009, .001, 0], [.005, .001, 0], [.001, .001, 0], [.001, .001, 1]]
            second = [[.001, .001, 1], [.009, .001, 1], [.009, .001, 1], [.005, .001, 1], [.001, .001, 2], [.001, .001, 0]]
            result = wp.zeros(6, dtype=int, device="cpu")
            wp.launch(module.probe, dim=6, inputs=[wp.array(reference, dtype=wp.vec3, device="cpu"), wp.array(first, dtype=wp.vec3, device="cpu"), wp.array(second, dtype=wp.vec3, device="cpu"), wp.array([24, 24, 24, 24, 24, 28], dtype=int, device="cpu"), result], device="cpu")
            np.testing.assert_array_equal(result.numpy(), [1, 0, 1, 0, 0, 1])
        with self.assertRaises(ValueError):
            reference_surface(rest + rest, faces, ["shell"] * 4 + ["facing"] * 4, .003, pairs, .004)
        pairs[0]["first"]["restMeters"] = [.02, .001]
        with self.assertRaisesRegex(ValueError, "outside immutable"):
            reference_surface(rest + rest, faces, ["shell"] * 4 + ["facing"] * 4, .003, pairs, .0005)

    def test_clearance_certificate_is_conservative(self):
        rest = [[0, 0, 0], [.01, 0, 0], [.01, .01, 0], [0, .01, 0], [.005, .005, 0]]
        faces = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
        reference = reference_surface(rest, faces, ["shell"] * 5)
        clearance = reference[5:10, 2]
        self.assertTrue(np.all(clearance[:4] == 0))
        self.assertGreater(clearance[4], .0049)
        self.assertLess(clearance[4], .005)
        for fraction in np.linspace(0, 1, 25):
            point = (1 - fraction) * np.array(rest[0]) + fraction * np.array(rest[4])
            lower_bound = max(0, float(clearance[0]) - np.linalg.norm(point - rest[0]), float(clearance[4]) - np.linalg.norm(point - rest[4]))
            exact = min(point[0], point[1], .01 - point[0], .01 - point[1])
            self.assertLessEqual(lower_bound, exact)

    def test_corner_crossing_notches_remain_contact_candidates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as output:
            path = Path(output) / "material_domain_probe.py"
            path.write_text(HELPERS + '''
@wp.kernel
def probe(reference: wp.array[wp.vec3], first: wp.array[wp.vec3], second: wp.array[wp.vec3], result: wp.array[int]):
    index = wp.tid()
    if material_neighbors(reference, first[index], second[index], wp.vec3(0., 28., 0.), .01):
        result[index] = 1
''')
            specification = importlib.util.spec_from_file_location("material_domain_probe", path)
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            corners = [[0, 0], [10, 0], [10, 4], [8, 4], [8, 2], [7, 1], [6, 2], [6, 4], [4, 4], [4, 2], [3, 1], [2, 2], [2, 4], [0, 4], [0, 0]]
            boundary = [[point[0] * .001, point[1] * .001, 0] for first, second in zip(corners, corners[1:]) for point in (first, second)]
            first = [[.001, .002, 0], [.001, .0005, 0], [.001, .0005, 0]]
            second = [[.009, .002, 0], [.009, .0005, 0], [.009, .0005, 1]]
            result = wp.zeros(3, dtype=int, device="cpu")
            wp.launch(module.probe, dim=3, inputs=[wp.array(boundary, dtype=wp.vec3, device="cpu"), wp.array(first, dtype=wp.vec3, device="cpu"), wp.array(second, dtype=wp.vec3, device="cpu"), result], device="cpu")
            np.testing.assert_array_equal(result.numpy(), [0, 1, 0])

    def test_rest_layer_identity_and_boundary_metadata(self):
        rest = [[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0]]
        reference = reference_surface(rest + rest, [[0, 1, 2], [3, 4, 5]], ["shell"] * 3 + ["facing"] * 3)
        np.testing.assert_array_equal(reference[:3, :2], reference[3:6, :2])
        self.assertTrue(np.all(reference[:3, 2] == 0))
        self.assertTrue(np.all(reference[3:6, 2] == 1))
        self.assertTrue(np.all(reference[6:9, 0] == 24))

    def test_malformed_source_rejected(self):
        rest = [[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0]]
        for faces in ([[0.9, 1.9, 2.9]], [[0, 0, 1]], [[0, 1, 2], [2, 1, 0]]):
            with self.assertRaises(ValueError):
                reference_surface(rest, faces, ["shell"] * 3)
        with self.assertRaises(ValueError):
            reference_surface(rest, [[0, 1, 2]], [1] * 3)

    def test_dynamic_local_rest_and_separate_material_contact(self):
        from newton._src.geometry import tri_mesh_collision

        wp.init()
        wp.set_device("cpu")
        original_vertex = tri_mesh_collision.vertex_triangle_collision_detection_kernel
        original_edge = tri_mesh_collision.edge_colliding_edges_detection_kernel
        try:
            for fixture in ("single", "layers", "remote-same-instance"):
                with self.subTest(fixture=fixture), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".planning") as output:
                    vertices = [[0, 0, 0], [0.002, 0, 0], [0, 0.002, 0], [0.002, 0.002, 0]]
                    faces = [0, 1, 2, 1, 3, 2]
                    rest = list(vertices)
                    builder = newton.ModelBuilder(gravity=(0, 0, 0))
                    builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0), vertices=vertices, indices=faces, density=0.2, tri_ke=10000, tri_ka=10000, particle_radius=0.001)
                    identifiers = ["shell"] * 4
                    if fixture != "single":
                        second = [[point[0] + (0.1 if fixture == "remote-same-instance" else 0), point[1], 0] for point in vertices]
                        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0), vertices=second, indices=faces, density=0.2, tri_ke=10000, tri_ka=10000, particle_radius=0.001)
                        rest.extend(second)
                        identifiers.extend(["facing" if fixture == "layers" else "shell"] * 4)
                        builder.particle_q[4:] = [wp.vec3(point[0], point[1], 0.0005) for point in vertices]
                    builder.color(include_bending=True)
                    model = builder.finalize(device="cpu")
                    solver = newton.solvers.SolverVBD(model, iterations=5, particle_enable_self_contact=True, particle_collision_detection_interval=1, particle_topological_contact_filter_threshold=0, particle_self_contact_margin=0.003, particle_self_contact_gap=0.001)
                    install_point_contact(solver, rest, model.tri_indices.numpy(), identifiers, output, 0.003)
                    state, next_state = model.state(), model.state()
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts)
                    solver.step(state, next_state, model.control(), contacts, 1 / 240)
                    positions = next_state.particle_q.numpy()
                    uninstall_point_contact(solver)
                    self.assertTrue(np.isfinite(positions).all())
                    if fixture == "single":
                        self.assertLess(float(np.linalg.norm(positions - np.asarray(vertices), axis=1).max()), 1e-7)
                    else:
                        self.assertGreater(float(positions[4:, 2].mean() - positions[:4, 2].mean()), 0.00051)
        finally:
            tri_mesh_collision.vertex_triangle_collision_detection_kernel = original_vertex
            tri_mesh_collision.edge_colliding_edges_detection_kernel = original_edge


if __name__ == "__main__":
    unittest.main()
