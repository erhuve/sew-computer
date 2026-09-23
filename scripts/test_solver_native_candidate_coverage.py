"""Small exhaustive runtime admission checks for native broad-phase coverage.

The oracle enumerates nonincident surface primitives and compares exact-rational
swept bounding boxes. This is a runtime diagnostic, not a collision certificate
or a substitute for production geometry/contact verification.

The selected explicit HashGrid and native BruteForce are required to pass.
Run this file with --include-platform-default to reproduce the known Linux ARM
default-BVH failure; that diagnostic remains a failure, without expectedFailure.
"""

from fractions import Fraction
import sys
import unittest

import ipctk
import numpy as np

from test_solver_contact_range_adversarial import square_grid
from solver_ipc_broad_phase import contact_broad_phase, CONTACT_BROAD_PHASE_PROFILE
from solver_candidate_coverage import verify_candidate_coverage

INCLUDE_PLATFORM_DEFAULT = False


def exhaustive_swept_candidates(start, end, edges, faces, radius):
    if len(start) > 128 or len(faces) > 256:
        raise ValueError("Exhaustive diagnostic fixture budget exceeded")
    coordinates = [[tuple(Fraction(float(value)) for value in point) for point in state]
                   for state in (start, end)]
    radius = Fraction(float(radius))

    def bounds(vertices):
        points = [coordinates[time][vertex] for time in (0, 1) for vertex in vertices]
        return tuple((min(point[axis] for point in points) - radius,
                      max(point[axis] for point in points) + radius) for axis in range(3))

    vertices = [bounds([vertex]) for vertex in range(len(start))]
    edge_boxes, face_boxes = ([bounds(primitive) for primitive in group] for group in (edges, faces))
    overlaps = lambda first, second: all(a[0] <= b[1] and b[0] <= a[1] for a, b in zip(first, second))
    expected = set()
    for face, face_vertices in enumerate(faces):
        for vertex in range(len(start)):
            if vertex not in face_vertices and overlaps(face_boxes[face], vertices[vertex]):
                expected.add(("fv", face, vertex))
    for first, edge in enumerate(edges):
        for second in range(first + 1, len(edges)):
            if not set(edge) & set(edges[second]) and overlaps(edge_boxes[first], edge_boxes[second]):
                expected.add(("ee", first, second))
    return expected


def actual_candidates(mesh, start, end, radius, method):
    candidates = ipctk.Candidates()
    candidates.build(mesh, start, end, inflation_radius=radius, broad_phase=method)
    found = {("fv", candidate.face_id, candidate.vertex_id) for candidate in candidates.fv_candidates}
    found |= {("ee", *sorted((candidate.edge0_id, candidate.edge1_id))) for candidate in candidates.ee_candidates}
    if len(found) != len(candidates):
        raise AssertionError("Duplicate or unexpected surface candidate kinds")
    return found


class NativeCandidateCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def check_coverage(self, start, end, faces, radius):
        edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1), axis=0)
        expected = exhaustive_swept_candidates(start, end, edges, faces, radius)
        self.assertGreater(len(expected), 0)
        methods = [("selected-HashGrid", contact_broad_phase), ("BruteForce", ipctk.BruteForce)]
        if INCLUDE_PLATFORM_DEFAULT:
            methods.append(("platform-default", lambda: None))
        observations = {}
        for name, constructor in methods:
            with self.subTest(method=name):
                mesh = ipctk.CollisionMesh(start, edges, faces)
                found = actual_candidates(mesh, start, end, radius, constructor())
                missing = expected - found
                self.assertFalse(missing, f"{name}: missing {len(missing)} of {len(expected)} required pairs: {sorted(missing)[:10]}")
                proof = verify_candidate_coverage(start, end, edges, faces, radius, found)
                self.assertTrue(proof["verified"], proof)
                self.assertEqual(proof["counts"]["requiredCandidates"], len(expected))
                observations[name] = found
        self.assertEqual(observations["selected-HashGrid"], observations["BruteForce"])
        return expected

    def test_selected_runtime_identity_and_query_isolation(self):
        self.assertEqual(CONTACT_BROAD_PHASE_PROFILE, "ipctk-HashGrid-explicit-v1")
        first, second = contact_broad_phase(), contact_broad_phase()
        self.assertIsInstance(first, ipctk.HashGrid)
        self.assertIsNot(first, second)

    def test_stationary_source_grid_exhaustive_coverage(self):
        positions, faces = square_grid(2)
        expected = self.check_coverage(positions, positions, faces, .00505)
        self.assertEqual(len(expected), 121)

    def test_compressed_and_rotated_grid_coverage(self):
        rest, faces = square_grid(2)
        start = rest * .4
        start[4, 2] = .000015
        rotation = np.linalg.qr(np.random.default_rng(281).normal(size=(3, 3)))[0]
        for transform in (np.eye(3), rotation):
            with self.subTest(rotated=not np.array_equal(transform, np.eye(3))):
                self.check_coverage(start @ transform, start @ transform, faces, .00505)

    def test_crossing_layers_swept_coverage(self):
        panel, faces = square_grid(2)
        start = np.vstack((panel, panel + [0., 0., .002]))
        end = np.vstack((panel, panel + [.00013, -.00017, -.002]))
        faces = np.vstack((faces, faces + len(panel)))
        self.check_coverage(start, end, faces, .0000500001)

    def test_refined_folded_transformed_and_scaled_sweeps(self):
        rotation = np.linalg.qr(np.random.default_rng(309).normal(size=(3, 3)))[0]
        for subdivisions in (2, 4):
            rest, faces = square_grid(subdivisions)
            start, end = rest.copy(), rest.copy()
            fold = rest[:, 0] > .0005
            displacement = rest[fold, 0] - .0005
            for state, angle in ((start, .1), (end, 2.3)):
                state[fold, 0] = .0005 + displacement * np.cos(angle)
                state[fold, 2] = displacement * np.sin(angle)
            for scale in (2. ** -20, 1., 2. ** 20):
                for transform, translation in ((np.eye(3), np.zeros(3)),
                                                (rotation, np.array([10., -4., 2.]))):
                    with self.subTest(refinement=subdivisions, scale=scale,
                                      translated=bool(np.any(translation))):
                        self.check_coverage((start @ transform + translation) * scale,
                                            (end @ transform + translation) * scale,
                                            faces, .0000100001 * scale)

    def test_compiler_cuff_source_folded_sweeps(self):
        # This source-derived fixture includes the real cut corners, extracted
        # allowance crease, remeshing, and duplicated shell/facing topology.
        # Import the module rather than a TestCase symbol to avoid collecting
        # its separate compatibility tests twice during unittest discovery.
        import test_solver_cuff_compatibility as cuff_source

        cuff_source.CuffCompatibilityTests.setUpClass()
        source = cuff_source.CuffCompatibilityTests()
        panel_start, panel_end = source.folded_panel(.1), source.folded_panel(1.2)
        start = np.vstack((panel_start, panel_start + [0., 0., .002]))
        end = np.vstack((panel_end, panel_end + [0., .0001, -.002]))
        faces = np.vstack((source.faces, source.faces + len(panel_start)))
        self.check_coverage(start, end, faces, .0000500001)


if __name__ == "__main__":
    if "--include-platform-default" in sys.argv:
        INCLUDE_PLATFORM_DEFAULT = True
        sys.argv.remove("--include-platform-default")
    unittest.main()
