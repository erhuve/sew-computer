import importlib.util
import unittest

import numpy as np

from solver_contact_filters import build_rest_neighbor_filters


def strip(count=5, spacing=0.001):
    positions = [[column * spacing, row * spacing, 0] for column in range(count) for row in range(2)]
    faces = []
    for column in range(count - 1):
        base = column * 2
        faces.extend([[base, base + 2, base + 1], [base + 1, base + 2, base + 3]])
    pairs = sorted({tuple(sorted((face[local], face[(local + 1) % 3]))) for face in faces for local in range(3)})
    edges = [[-1, -1, *pair] for pair in pairs]
    return np.array(positions), np.array(faces), np.array(edges)


class RestNeighborFilterTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("newton"), "Optional pinned Newton research runtime required")
    def test_dynamic_layer_contact_response_is_retained(self):
        import newton
        import warp as wp

        wp.init()
        wp.set_device("cpu")
        rest, faces, _ = strip(4)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for height in (0, 0.0005):
            builder.add_cloth_mesh(
                pos=wp.vec3(0, 0, height), rot=wp.quat_identity(), scale=1,
                vel=wp.vec3(0, 0, 0), vertices=rest.tolist(), indices=faces.flatten().tolist(),
                density=0.2, tri_ke=10000, tri_ka=10000, particle_radius=0.001)
        builder.color(include_bending=True)
        model = builder.finalize(device="cpu")
        vertex_map, edge_map, report = build_rest_neighbor_filters(
            np.concatenate([rest, rest]), model.tri_indices.numpy(), model.edge_indices.numpy(),
            ["shell"] * len(rest) + ["facing"] * len(rest), 0.003)
        self.assertGreater(report["vertexTriangleEntries"], 0)
        initial = model.particle_q.numpy().copy()
        displacements = []
        layer_gaps = []
        for contact_enabled in (False, True):
            solver = newton.solvers.SolverVBD(
                model, iterations=5, particle_enable_self_contact=contact_enabled,
                particle_external_vertex_contact_filtering_map=vertex_map,
                particle_external_edge_contact_filtering_map=edge_map,
                particle_self_contact_margin=0.003, particle_self_contact_gap=0.001)
            state, next_state = model.state(), model.state()
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state.clear_forces()
            pipeline.collide(state, contacts)
            solver.step(state, next_state, model.control(), contacts, 1 / 240)
            positions = next_state.particle_q.numpy()
            self.assertTrue(np.isfinite(positions).all())
            displacements.append(float(np.max(np.abs(positions - initial))))
            layer_gaps.append(float(np.mean(positions[len(rest):, 2]) - np.mean(positions[:len(rest), 2])))
        self.assertLess(displacements[0], 1e-8)
        self.assertGreater(displacements[1], 1e-5)
        self.assertAlmostEqual(layer_gaps[0], 0.0005, places=8)
        self.assertGreater(layer_gaps[1], layer_gaps[0] + 1e-5)

    def test_local_neighborhood_and_remote_fold_candidate_maps(self):
        rest, faces, edges = strip(10)
        original = rest.copy()
        vertex_map, edge_map, report = build_rest_neighbor_filters(rest, faces, edges, ["shell"] * len(rest), 0.0021)
        self.assertIn(0, vertex_map[0])
        self.assertNotIn(len(faces) - 1, vertex_map[0])
        first_edge = next(index for index, edge in enumerate(edges) if list(edge[2:]) == [0, 1])
        last_edge = next(index for index, edge in enumerate(edges) if list(edge[2:]) == [18, 19])
        self.assertNotIn(last_edge, edge_map[first_edge])
        folded = rest.copy()
        folded[-2:] = folded[:2]
        self.assertTrue(np.array_equal(folded[-2:], folded[:2]))
        self.assertTrue(np.array_equal(original, rest))
        for edge_id, neighbors in edge_map.items():
            for neighbor in neighbors:
                self.assertIn(edge_id, edge_map[neighbor])
        self.assertGreater(report["vertexTriangleEntries"], 0)

    def test_near_corner_does_not_exclude_long_primitives(self):
        rest = [[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0], [0.1, 0, 0], [0.1, 0.1, 0]]
        faces = [[0, 1, 2], [1, 3, 4], [1, 4, 2]]
        edges = [[-1, -1, 0, 2], [-1, -1, 1, 3], [-1, -1, 1, 2]]
        vertex_map, edge_map, _ = build_rest_neighbor_filters(rest, faces, edges, ["shell"] * 5, 0.003)
        self.assertIn(0, vertex_map[0])
        self.assertNotIn(1, vertex_map[0])
        self.assertNotIn(2, vertex_map[0])
        self.assertIn(2, edge_map[0])
        self.assertNotIn(1, edge_map[0])
        self.assertNotIn(0, edge_map.get(1, set()))

    def test_coincident_separate_layers_remain_candidates(self):
        rest, faces, edges = strip()
        count = len(rest)
        both_edges = np.concatenate([edges, edges + np.array([0, 0, count, count])])
        vertex_map, edge_map, _ = build_rest_neighbor_filters(
            np.concatenate([rest, rest]), np.concatenate([faces, faces + count]),
            both_edges, ["shell"] * count + ["facing"] * count, 0.003)
        self.assertTrue(all(face_id < len(faces) for vertex in range(count) for face_id in vertex_map[vertex]))
        self.assertTrue(all(neighbor < len(edges) for edge_id in range(len(edges)) for neighbor in edge_map[edge_id]))

    def test_disconnected_same_instance_is_not_spatially_excluded(self):
        rest, faces, edges = strip()
        count = len(rest)
        vertex_map, _, _ = build_rest_neighbor_filters(
            np.concatenate([rest, rest]), np.concatenate([faces, faces + count]),
            np.concatenate([edges, edges + np.array([0, 0, count, count])]), ["shell"] * (count * 2), 0.003)
        self.assertNotIn(len(faces), vertex_map[0])

    def test_rigid_transform_invariance(self):
        rest, faces, edges = strip()
        expected = build_rest_neighbor_filters(rest, faces, edges, ["shell"] * len(rest), 0.0021)
        rotated = rest[:, [2, 0, 1]] + [10, -8, 12]
        actual = build_rest_neighbor_filters(rotated, faces, edges, ["shell"] * len(rest), 0.0021)
        self.assertEqual(expected[:2], actual[:2])

    def test_budgets_and_bad_inputs_reject(self):
        rest, faces, edges = strip()
        identifiers = ["shell"] * len(rest)
        for budget in ({"max_vertices": 1}, {"max_primitives": 1}, {"max_search_visits": 1}, {"max_filter_entries": 1}, {"max_candidate_checks": 1}):
            with self.assertRaises(ValueError):
                build_rest_neighbor_filters(rest, faces, edges, identifiers, 0.003, **budget)
        for margin in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_rest_neighbor_filters(rest, faces, edges, identifiers, margin)
        identifiers[0] = "facing"
        with self.assertRaisesRegex(ValueError, "crosses physical"):
            build_rest_neighbor_filters(rest, faces, edges, identifiers, 0.003)
        with self.assertRaisesRegex(ValueError, "four-slot"):
            build_rest_neighbor_filters(rest, faces, edges[:, 2:], ["shell"] * len(rest), 0.003)


if __name__ == "__main__":
    unittest.main()
