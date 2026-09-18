from types import SimpleNamespace
import unittest

import newton
import numpy as np

from solver_fold_barrier import LocalAngularFoldBarrier
from solver_global_sewing import GlobalSewingSolver


class FoldInputAdversarialTests(unittest.TestCase):
    def test_empty_hinges_still_validate_declared_parameters(self):
        for parameters in ({'stiffness_joules': np.nan}, {'stiffness_joules': np.inf},
                           {'stiffness_joules': -1.}, {'activation_angle': np.nan},
                           {'activation_angle': -1.}, {'activation_angle': np.pi}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                LocalAngularFoldBarrier(1, np.empty((0, 4), dtype=int), **parameters)

    def test_missing_bending_properties_cannot_hide_invalid_barrier_topology(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in ((.2, 1., 0.), (.7, -1., 0.), (0., 0., 0.), (1., 0., 0.)):
            builder.add_particle(pos=point, vel=(0, 0, 0), mass=1.)
        builder.set_coloring([[0], [1], [2], [3]])
        model = builder.finalize(device='cpu')
        model.edge_bending_properties = None
        for indices in (np.array([[0, 1, 2, -2]]), np.array([[-2, 1, 2, 3]]),
                        np.array([[0, 1, 2, np.nan]]), np.array([[0, 1, 2, 3.5]]),
                        np.array([[0, 1, 2, 4]]), np.array([])):
            model.edge_indices = SimpleNamespace(numpy=lambda: indices)
            with self.subTest(indices=indices.tolist()), self.assertRaises(ValueError):
                GlobalSewingSolver(model, [{0: 1., 1: -1.}], 1e-8, fold_barrier_joules=.1)


if __name__ == '__main__':
    unittest.main()
