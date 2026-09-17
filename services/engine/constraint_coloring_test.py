import itertools
import unittest

from constraint_coloring import refine_constraint_colors


class ConstraintColoringTests(unittest.TestCase):
    def test_all_four_particle_graphs_preserve_existing_separations(self):
        pairs = list(itertools.combinations(range(4), 2))
        for mask in range(1 << len(pairs)):
            constraints = [pair for index, pair in enumerate(pairs) if mask & (1 << index)]
            for groups in ([[0, 1, 2, 3]], [[0, 2], [1, 3]], [[0], [1], [2], [3]]):
                result = refine_constraint_colors(4, groups, constraints)
                colors = {vertex: color for color, group in enumerate(result) for vertex in group}
                self.assertEqual(sorted(itertools.chain.from_iterable(result)), list(range(4)))
                for first, second in constraints:
                    self.assertNotEqual(colors[first], colors[second])
                for group in result:
                    self.assertTrue(any(set(group) <= set(original) for original in groups))
                self.assertEqual(result, refine_constraint_colors(4, groups, list(reversed(constraints))))

    def test_multivertex_constraints_and_empty_model(self):
        groups = refine_constraint_colors(5, [[0, 1, 2, 3, 4]], [[0, 1, 2], [2, 3, 4]])
        colors = {vertex: color for color, group in enumerate(groups) for vertex in group}
        for constraint in ((0, 1, 2), (2, 3, 4)):
            self.assertEqual(len({colors[vertex] for vertex in constraint}), 3)
        self.assertEqual(refine_constraint_colors(0, [], []), [])

    def test_invalid_partition_and_constraint_indices_reject(self):
        for groups in ([[0, 0]], [[0]], [[0, 2]], [[0, True]], [[0, 1.0]]):
            with self.assertRaises(ValueError):
                refine_constraint_colors(2, groups, [])
        for constraint in ([0, 0], [0, 2], [0, -1], [0, True], [0, 1.0], [0]):
            with self.assertRaises(ValueError):
                refine_constraint_colors(2, [[0, 1]], [constraint])


if __name__ == "__main__":
    unittest.main()
