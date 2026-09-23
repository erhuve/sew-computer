"""Independent Cartesian controls and adversarial inputs for exact coverage."""

from fractions import Fraction
import math
import random
import unittest

from solver_candidate_coverage import verify_candidate_coverage


def edges_from(faces):
    return sorted({tuple(sorted((face[first], face[second]))) for face in faces
                   for first, second in ((0, 1), (1, 2), (2, 0))})


def all_pairs(vertices, edges, faces):
    return ({("fv", face_id, vertex) for face_id, face in enumerate(faces)
             for vertex in range(len(vertices)) if vertex not in face}
            | {("ee", first, second) for first, edge in enumerate(edges)
               for second in range(first + 1, len(edges)) if not set(edge) & set(edges[second])})


def cartesian_oracle(start, end, edges, faces, radius):
    """Tiny test-only Cartesian reference, independent of the interval sweep."""
    radius = Fraction(float(radius))
    expected = set()
    for kind, first, second in all_pairs(start, edges, faces):
        primitives = (faces[first], [second]) if kind == "fv" else (edges[first], edges[second])
        for axis in range(3):
            values = [[Fraction(float(state[vertex][axis])) for state in (start, end)
                       for vertex in primitive] for primitive in primitives]
            if max(values[0]) + 2 * radius < min(values[1]) or max(values[1]) + 2 * radius < min(values[0]):
                break
        else:
            expected.add((kind, first, second))
    return expected


class CandidateCoverageTests(unittest.TestCase):
    def setUp(self):
        self.start = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                      [0., 0., 2.], [1., 0., 2.], [0., 1., 2.]]
        self.faces = [[0, 1, 2], [3, 4, 5]]
        self.edges = edges_from(self.faces)
        self.pairs = all_pairs(self.start, self.edges, self.faces)

    def verify(self, **kwargs):
        arguments = dict(start=self.start, end=self.start, edges=self.edges,
                         faces=self.faces, inflation_radius_m=1., observed_candidates=self.pairs)
        arguments.update(kwargs)
        return verify_candidate_coverage(**arguments)

    def test_touching_inflated_boxes_and_exact_float_neighbors(self):
        for radius in (math.nextafter(1., 0.), 1., math.nextafter(1., math.inf)):
            with self.subTest(radius=radius):
                expected = cartesian_oracle(self.start, self.start, self.edges, self.faces, radius)
                result = self.verify(inflation_radius_m=radius, observed_candidates=expected)
                self.assertTrue(result["verified"], result)
                self.assertEqual(result["counts"]["requiredCandidates"], len(expected))
                self.assertEqual(bool(expected), radius >= 1.)
                for omitted in expected:
                    failure = self.verify(inflation_radius_m=radius, observed_candidates=expected - {omitted})
                    self.assertFalse(failure["verified"])
                    self.assertEqual(failure["status"], "missing-candidate")
                    self.assertEqual(failure["firstMissingCandidate"], list(omitted))

    def test_interval_sweep_matches_independent_cartesian_random_paths(self):
        rng = random.Random(9083)
        for trial in range(50):
            count = 3 * (2 + trial % 3)
            states = [[[rng.randint(-12, 12) / 4 for axis in range(3)]
                       for vertex in range(count)] for time in range(2)]
            faces = [list(range(first, first + 3)) for first in range(0, count, 3)]
            edges = edges_from(faces)
            radius = rng.choice([0., .125, .5])
            expected = cartesian_oracle(*states, edges, faces, radius)
            with self.subTest(trial=trial):
                proof = verify_candidate_coverage(*states, edges, faces, radius, expected)
                self.assertTrue(proof["verified"], proof)
                self.assertEqual(proof["counts"]["requiredCandidates"], len(expected))
                if expected:
                    omitted = min(expected)
                    failure = verify_candidate_coverage(*states, edges, faces, radius, expected - {omitted})
                    self.assertEqual(failure["status"], "missing-candidate")

    def test_additional_valid_candidates_allowed_but_bad_records_rejected(self):
        self.assertTrue(self.verify(inflation_radius_m=0.)["verified"])
        for record in (("fv", 0, 0), ("ee", 0, 0), ("fv", 2, 1),
                       ("fv", False, 4), ("fv", 0., 4), ("ev", 0, 4),
                       ("fv", 0, 4, 5), ("fv", 0), ("fv", 0, float("nan"))):
            with self.subTest(record=record):
                result = self.verify(observed_candidates=[record])
                self.assertFalse(result["verified"])
                self.assertEqual(result["status"], "invalid-input")
        record = next(iter(self.pairs))
        self.assertEqual(self.verify(observed_candidates=[record, record])["status"], "invalid-input")

    def test_sparse_interval_sweep_avoids_unconditional_cartesian_work(self):
        vertices = [[offset + x, y, 0.] for offset in range(0, 80, 10)
                    for x, y in ((0., 0.), (1., 0.), (0., 1.))]
        faces = [list(range(first, first + 3)) for first in range(0, len(vertices), 3)]
        edges = edges_from(faces)
        proof = verify_candidate_coverage(vertices, vertices, edges, faces, 0., [], max_comparisons=64)
        self.assertTrue(proof["verified"], proof)
        self.assertEqual(proof["counts"]["requiredCandidates"], 0)
        self.assertLess(proof["counts"]["comparisons"], len(vertices) * len(faces))

    def test_invalid_shapes_topology_nonfinite_and_boolean_inputs_fail_closed(self):
        malformed = [dict(start=[]), dict(end=self.start[:-1]),
                     dict(start=[[0., 0.]] * 6), dict(start=[[0., 0., float("nan")]] * 6),
                     dict(start=[[0., 0., float("inf")]] * 6), dict(start=[[False, 0., 0.]] * 6),
                     dict(faces=[[False, 1, 2], [3, 4, 5]]), dict(faces=[[0., 1, 2], [3, 4, 5]]),
                     dict(faces=[[0, 1, 1], [3, 4, 5]]), dict(faces=[[0, 1, 2]] * 2),
                     dict(faces=[[0, 1, 2], [3, 4, 6]]), dict(edges=self.edges[:-1]),
                     dict(edges=self.edges + [self.edges[0]]), dict(inflation_radius_m=-1),
                     dict(inflation_radius_m=float("nan")), dict(inflation_radius_m=True),
                     dict(max_comparisons=True), dict(max_comparisons=0), dict(max_comparisons=2000001)]
        for arguments in malformed:
            with self.subTest(arguments=arguments):
                result = self.verify(**arguments)
                self.assertFalse(result["verified"])
                self.assertEqual(result["status"], "invalid-input", result)

    def test_all_budgets_fail_closed_and_allocation_limits_precede_geometry(self):
        for budget in (dict(max_primitives=1), dict(max_comparisons=1), dict(max_candidates=1)):
            with self.subTest(budget=budget):
                result = self.verify(**budget)
                self.assertFalse(result["verified"])
                self.assertEqual(result["status"], "budget-exceeded")

        class UntouchableGeometry:
            def __len__(self):
                return 6

            def __iter__(self):
                raise AssertionError("Geometry allocated before early budget check")

        for budget in (dict(max_primitives=1), dict(max_candidates=1)):
            result = self.verify(start=UntouchableGeometry(), **budget)
            self.assertEqual(result["status"], "budget-exceeded")
        generator = (candidate for candidate in sorted(self.pairs))
        result = self.verify(observed_candidates=generator, max_candidates=1)
        self.assertEqual(result["status"], "budget-exceeded")
        self.assertEqual(result["counts"]["observedCandidates"], 2)


if __name__ == "__main__":
    unittest.main()
