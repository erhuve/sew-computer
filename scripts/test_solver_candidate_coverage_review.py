"""Independent inclusive-interval, identity and replay attacks for coverage.

The replay function is extracted from its actual script AST, so these tests
exercise integration without launching its CLI or changing process limits.
"""

import ast
import copy
from pathlib import Path
import random
import types
import unittest

import ipctk
import numpy as np
from fractions import Fraction

from solver_candidate_coverage import verify_candidate_coverage
from solver_ipc_broad_phase import contact_broad_phase
from solver_rest_filtered_contact import RestFilteredSurfaceContact


def mesh_edges(faces):
    return sorted({tuple(sorted((face[index], face[(index + 1) % 3])))
                   for face in faces for index in range(3)})

def nonincident_pairs(vertex_count, edges, faces):
    return ([('fv', face_index, vertex) for face_index, face in enumerate(faces)
             for vertex in range(vertex_count) if vertex not in face]
            + [('ee', first, second) for first in range(len(edges)) for second in range(first + 1, len(edges))
               if not set(edges[first]).intersection(edges[second])])


class CandidateCoverageReviewTests(unittest.TestCase):
    def test_zero_width_intervals_include_every_equal_endpoint_pair(self):
        # Broad-phase coverage accepts arbitrary endpoint boxes; nondegenerate
        # triangles and physical path separation are separate verifier duties.
        faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6]]
        edges = mesh_edges(faces)
        expected = nonincident_pairs(7, edges, faces)
        for axis in range(3):
            # All boxes have zero width on all axes, covering the start-before-
            # end tie rule for face/vertex and same-group edge/edge events.
            coordinate = np.zeros(3)
            coordinate[axis] = 2. ** 500
            positions = np.tile(coordinate, (7, 1))
            result = verify_candidate_coverage(positions, positions, edges, faces, 0., expected)
            self.assertTrue(result["verified"], result)
            self.assertEqual(result["counts"]["requiredCandidates"], len(expected))
            for omitted in expected:
                failure = verify_candidate_coverage(positions, positions, edges, faces, 0.,
                                                    [pair for pair in expected if pair != omitted])
                self.assertEqual(failure["status"], "missing-candidate")
                self.assertEqual(failure["firstMissingCandidate"], list(omitted))

    def test_topology_and_candidate_identity_survive_independent_permutations(self):
        faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6]]
        edges = mesh_edges(faces)
        original = nonincident_pairs(7, edges, faces)
        rng = random.Random(292)
        for trial in range(8):
            face_order, edge_order, vertex_order = list(range(3)), list(range(len(edges))), list(range(7))
            rng.shuffle(face_order)
            rng.shuffle(edge_order)
            rng.shuffle(vertex_order)
            face_ids = {old: new for new, old in enumerate(face_order)}
            edge_ids = {old: new for new, old in enumerate(edge_order)}
            vertex_ids = {old: new for new, old in enumerate(vertex_order)}
            remapped_faces = [[vertex_ids[vertex] for vertex in faces[old]][::-1] for old in face_order]
            remapped_edges = [[vertex_ids[vertex] for vertex in edges[old]][::-1] for old in edge_order]
            remapped_candidates = [(kind, face_ids[first], vertex_ids[second]) if kind == 'fv'
                                   else (kind, edge_ids[second], edge_ids[first])
                                   for kind, first, second in original]
            rng.shuffle(remapped_candidates)
            points = [[0., 0., 0.]] * 7
            with self.subTest(trial=trial):
                result = verify_candidate_coverage(points, points, remapped_edges, remapped_faces, 0.,
                                                    remapped_candidates)
                self.assertTrue(result["verified"], result)
                self.assertEqual(result["counts"]["requiredCandidates"], len(original))
                # Reversing an EE candidate is still the same physical pair.
                duplicate = next(pair for pair in remapped_candidates if pair[0] == 'ee')
                result = verify_candidate_coverage(points, points, remapped_edges, remapped_faces, 0.,
                                                    remapped_candidates + [(duplicate[0], duplicate[2], duplicate[1])])
                self.assertEqual(result["status"], "invalid-input")

    def test_exact_budget_boundaries_cannot_return_partial_success(self):
        faces = [[0, 1, 2], [3, 4, 5]]
        edges = mesh_edges(faces)
        points = [[0., 0., 0.]] * 6
        expected = nonincident_pairs(6, edges, faces)
        result = verify_candidate_coverage(points, points, edges, faces, 0., expected)
        self.assertTrue(result["verified"], result)
        limits = {"max_primitives": 6 + len(edges) + len(faces),
                  "max_candidates": len(expected), "max_comparisons": result["counts"]["comparisons"]}
        self.assertTrue(verify_candidate_coverage(points, points, edges, faces, 0., expected, **limits)["verified"])
        for name, value in limits.items():
            with self.subTest(budget=name):
                exhausted = verify_candidate_coverage(points, points, edges, faces, 0., expected,
                                                       **{**limits, name: value - 1})
                self.assertIs(exhausted["verified"], False)
                self.assertEqual(exhausted["status"], "budget-exceeded")


class ReplayCandidateCoverageReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)
        # The swept AABBs overlap near the diagonal corners, but the triangles
        # are safely separated. Nonempty valid proof leaves can be permuted.
        cls.positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                  [1.00008, 1.00008, 0.], [2.00008, 1.00008, 0.],
                                  [1.00008, 2.00008, 0.]])
        cls.faces = np.array([[0, 1, 2], [3, 4, 5]])
        cls.contact = RestFilteredSurfaceContact(cls.positions, cls.faces, activation_distance_m=.001,
            minimum_distance_m=.0001, stiffness=10000., ccd_profile="temporal-separation-tight-inclusion")
        cls.proof = cls.contact.path_certificate(cls.positions, cls.positions, keep_leaves=True)
        if not cls.proof["safe"] or cls.proof["candidateCount"] < 3:
            raise AssertionError(cls.proof)
        path = Path(__file__).with_name("replay-rest-filtered-continuation.py")
        tree = ast.parse(path.read_text(), filename=str(path))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_certificate")
        cls.code = compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec")

    def verify(self, proof, candidate_factory=ipctk.Candidates):
        namespace = {"ipctk": types.SimpleNamespace(Candidates=candidate_factory),
                     "contact": self.contact, "rest": self.positions, "faces": self.faces,
                     "np": np, "Fraction": Fraction,
                     "broad_phase_verifier": types.SimpleNamespace(contact_broad_phase=contact_broad_phase),
                     "coverage_verifier": types.SimpleNamespace(verify_candidate_coverage=verify_candidate_coverage)}
        exec(self.code, namespace)
        return namespace["verify_certificate"](self.positions, self.positions, proof)

    def test_replay_uses_primitive_identity_across_native_candidate_orderings(self):
        class ReversedCandidates:
            def __init__(self):
                self.native = ipctk.Candidates()

            def build(self, *args, **kwargs):
                self.native.build(*args, **kwargs)

            def __len__(self):
                return len(self.native)

            def __getattr__(self, name):
                return list(reversed(getattr(self.native, name)))

        expected = self.verify(self.proof)
        observed = self.verify(self.proof, ReversedCandidates)
        self.assertEqual(observed, expected)
        changed = copy.deepcopy(self.proof)
        for group in {leaf["group"] for leaf in changed["certificateLeaves"]}:
            labels = sorted({leaf["candidate"] for leaf in changed["certificateLeaves"] if leaf["group"] == group})
            replacement = dict(zip(labels, reversed(labels)))
            for leaf in changed["certificateLeaves"]:
                if leaf["group"] == group:
                    leaf["candidate"] = replacement[leaf["candidate"]]
        self.assertEqual(self.verify(changed, ReversedCandidates), expected)

    def test_replay_rejects_incomplete_geometry_labels_and_interval_proofs(self):
        mutations = ("missing", "count", "index-alias", "out-of-range", "wrong-vertices", "gap", "overrun",
                     "zero-normal", "false-minimum", "zero-span", "boolean-index")
        for mutation in mutations:
            changed = copy.deepcopy(self.proof)
            leaves = changed["certificateLeaves"]
            first = leaves[0]
            if mutation == "missing":
                changed["certificateLeaves"] = leaves[1:]
            elif mutation == "count":
                changed["candidateCount"] -= 1
            elif mutation == "index-alias":
                other = next(leaf for leaf in leaves[1:] if leaf["group"] == first["group"])
                other["candidate"] = first["candidate"]
            elif mutation == "out-of-range":
                first["candidate"] = changed["candidateCount"] + 1
            elif mutation == "wrong-vertices":
                first["first"] = [999]
            elif mutation == "gap":
                first["t0"] = .25
            elif mutation == "overrun":
                first["t1"] = 1.25
            elif mutation == "zero-normal":
                first["normal"] = [0., 0., 0.]
            elif mutation == "false-minimum":
                first["minimumDistanceM"] = 0.
            elif mutation == "zero-span":
                first["t1"] = first["t0"]
            else:
                first["candidate"] = True
            with self.subTest(mutation=mutation), self.assertRaises((AssertionError, KeyError, IndexError, ValueError)):
                self.verify(changed)


if __name__ == "__main__":
    unittest.main()
