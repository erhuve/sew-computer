import json
import unittest

import numpy as np

from solver_spike_geometry import snapshot_particle_positions, state_finiteness, surface_intersections, triangles_intersect


class SurfaceOracleTests(unittest.TestCase):
    def test_snapshot_preserves_complete_velocity_when_vbd_mutates_input(self):
        try:
            import newton
            import warp as wp
        except ModuleNotFoundError:
            self.skipTest("Newton research runtime is not installed")
        wp.init()
        wp.set_device("cpu")
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0]], indices=[0, 1, 2],
            density=.2, tri_ke=10000, tri_ka=10000, tri_kd=.01)
        builder.color(include_bending=True)
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, iterations=10, particle_enable_self_contact=False)
        state, next_state = model.state(), model.state()
        compressed = state.particle_q.numpy().copy()
        compressed[1, 0] = .08
        state.particle_q.assign(compressed)
        previous = snapshot_particle_positions(state)
        alias = state.particle_q.numpy()
        timestep = 1 / 240
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, model.control(), contacts, timestep)
        np.testing.assert_array_equal(previous, compressed)
        self.assertFalse(np.shares_memory(previous, alias))
        self.assertGreater(float(np.linalg.norm(alias - previous)), 1e-6)
        expected_velocity = (next_state.particle_q.numpy() - compressed) / timestep
        reconstructed_velocity = (next_state.particle_q.numpy() - previous) / timestep
        np.testing.assert_allclose(reconstructed_velocity, expected_velocity, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(reconstructed_velocity, next_state.particle_qd.numpy(), rtol=1e-5, atol=1e-5)
        aliased_velocity = (next_state.particle_q.numpy() - alias) / timestep
        self.assertGreater(float(np.linalg.norm(reconstructed_velocity - aliased_velocity)), .001)

    def test_nonfinite_state_reports_remain_json_serializable(self):
        positions = np.zeros((2, 3))
        velocities = np.zeros((2, 3))
        self.assertTrue(state_finiteness(positions, velocities)["finite"])
        positions[0, 1] = np.nan
        velocities[1, 0] = np.inf
        report = state_finiteness(positions, velocities)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), {
            "finite": False, "nonfinitePositionValues": 1, "nonfiniteVelocityValues": 1})
        self.assertTrue(np.isnan(positions[0, 1]))
        with self.assertRaises(ValueError):
            state_finiteness(positions, np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            state_finiteness([], [])

    def test_broad_phase_padding_covers_diagonal_scaled_oracle_tolerance(self):
        first = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        second = first + [0, 0, 1.2e-9]
        self.assertTrue(triangles_intersect(first, second))
        self.assertEqual(surface_intersections(np.concatenate([first, second]), [[0, 1, 2], [3, 4, 5]])["intersectingPairCount"], 1)

    def test_broad_phase_preserves_brute_force_intersections(self):
        positions = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                              [.5, .5, -1], [.5, .5, 1], [2, 2, 0],
                              [10, 10, 10], [12, 10, 10], [10, 12, 10]], dtype=float)
        faces = np.arange(9).reshape((3, 3))
        for scale in (.001, 1, 1000):
            moved = positions * scale + [1, -2, 3]
            expected = sum(triangles_intersect(moved[first], moved[second]) for index, first in enumerate(faces) for second in faces[index + 1:])
            result = surface_intersections(moved, faces)
            self.assertEqual(result["intersectingPairCount"], expected)
            self.assertEqual(result["testedCandidates"], 1)
        with self.assertRaises(ValueError):
            surface_intersections(positions, [[0, 0, 1]])
        with self.assertRaises(ValueError):
            surface_intersections(positions, faces, candidate_budget=0)

    def test_crossing_without_inside_vertices(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        second = np.array([[0.5, 0.5, -1.], [0.5, 0.5, 1.], [2., 2., 0.]])
        self.assertTrue(triangles_intersect(first, second))

    def test_coplanar_overlap_and_boundary_only(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        self.assertTrue(triangles_intersect(first, first + [0.1, 0.1, 0.]))
        self.assertFalse(triangles_intersect(first, first + [2., 0., 0.]))

    def test_separated_and_rigid_transform(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        self.assertFalse(triangles_intersect(first, first + [0., 0., 0.01]))
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertTrue(triangles_intersect(first @ rotation + 8, (first + [0.1, 0.1, 0.]) @ rotation + 8))

    def test_millimetre_crossing_scale_and_out_of_plane_rotation(self):
        first = np.array([[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.]])
        second = np.array([[.00025, .00025, -.00001], [.00025, .00025, .00001], [.00075, .00075, 0.]])
        cosine, sine = np.cos(.7), np.sin(.7)
        rotation = np.array([[1., 0., 0.], [0., cosine, -sine], [0., sine, cosine]])
        for scale in (.001, 1., 1000.):
            self.assertTrue(triangles_intersect(first * scale, second * scale))
            self.assertTrue(triangles_intersect(first @ rotation * scale + .5, second @ rotation * scale + .5))
            self.assertFalse(triangles_intersect(first * scale, (first + [0., 0., .00001]) * scale))

    def test_coplanar_tiny_overlap_far_from_origin(self):
        first = np.array([[0., 0., 0.], [1e-6, 0., 0.], [0., 1e-6, 0.]])
        second = first + [1e-7, 1e-7, 0.]
        self.assertTrue(triangles_intersect(first, second))
        self.assertTrue(triangles_intersect(first + [100, 200, 300], second + [100, 200, 300]))


if __name__ == "__main__":
    unittest.main()
