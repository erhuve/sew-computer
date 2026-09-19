import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import diags

from solver_global_sewing import _direct_descent
from solver_global_shift import shifted_positive_definite_direction


class GuardedContactMetricTests(unittest.TestCase):
    def solve(self, primary, fallback, **options):
        return _direct_descent(
            lambda positions: positions.copy(), np.array([1., -1.]), 100,
            lambda positions: diags(fallback),
            lambda positions: .5 * float(positions @ positions),
            gradient_function=lambda positions: positions.copy(),
            coupled_hessian=lambda positions: diags(primary),
            **options)

    def test_indefinite_primary_uses_projected_fallback_without_changing_energy(self):
        result = self.solve([-1., 2.], [1., 1.],
                            guard_assembled_metrics=True, inertia_diagonal=np.array([2., 3.]))
        self.assertTrue(result.success)
        self.assertEqual(result.shifted_steps, 0)
        self.assertEqual(result.direction_history[0]["metric"], "projected")
        self.assertEqual(result.direction_history[0]["shift"]["lambda"], 0)
        self.assertTrue(np.all(np.diff(result.energy_history) < 0))
        self.assertAlmostEqual(result.cost, .5 * float(result.x @ result.x))
        np.testing.assert_allclose(result.x, 0., atol=1e-6)

    def test_indefinite_fallback_is_also_shifted_after_primary_exhaustion(self):
        with patch("solver_global_shift.shifted_positive_definite_direction",
                   wraps=shifted_positive_definite_direction) as guarded_solve:
            result = self.solve([-1e20, 1.], [-1., 2.],
                                guard_assembled_metrics=True, inertia_diagonal=np.array([2., 3.]))
        self.assertTrue(result.success)
        self.assertGreaterEqual(guarded_solve.call_count, 2)
        self.assertTrue(all(entry["metric"] == "projected" for entry in result.direction_history))
        self.assertTrue(all(entry["shift"]["lambda"] > 0 for entry in result.direction_history))
        self.assertTrue(np.all(np.diff(result.energy_history) < 0))

    def test_both_unrepairable_metrics_fail_closed(self):
        result = self.solve([-1e20, 1.], [-1e20, 2.],
                            guard_assembled_metrics=True, inertia_diagonal=np.ones(2))
        self.assertFalse(result.success)
        self.assertEqual(result.status, -2)
        self.assertEqual(result.direction_history, [])
        np.testing.assert_array_equal(result.x, [1., -1.])

    def test_default_coupled_path_does_not_call_shift_helper(self):
        with patch("solver_global_shift.shifted_positive_definite_direction",
                   side_effect=AssertionError("Legacy contact must not use shifts")):
            result = self.solve([-1., 2.], [1., 1.])
        self.assertTrue(result.success)
        self.assertEqual(result.projected_steps, 1)
        self.assertEqual(result.shifted_steps, 0)
        self.assertNotIn("metric", result.direction_history[0])

    def test_guard_requires_physical_inertia(self):
        with self.assertRaisesRegex(ValueError, "physical inertia"):
            self.solve([1., 1.], [1., 1.], guard_assembled_metrics=True)

    def test_solver_passes_free_particle_inertia_and_reports_guarded_profile(self):
        from test_solver_ipc_integration import IpcIntegrationTests
        from solver_global_sewing import GlobalSewingSolver
        from solver_ipc_contact import IpcSurfaceContact

        model, rest, _, _, targets = IpcIntegrationTests().fixture(fixed=True)
        contact = IpcSurfaceContact(rest, model.tri_indices.numpy(), activation_distance_m=.005,
                                    minimum_distance_m=.0001, stiffness=1e4,
                                    energy_profile="area-improved-max")
        solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}], 1e-6, contact=contact)
        expected_inertia = np.repeat(solver.mass, 3)[solver.free] / .01 ** 2
        with patch("solver_global_sewing._direct_descent", wraps=_direct_descent) as descent:
            final, _, report = solver.step(rest, np.zeros_like(rest), targets, .01,
                                          max_evaluations=1000)
        self.assertTrue(report["converged"], report)
        self.assertEqual(descent.call_count, 1)
        np.testing.assert_allclose(descent.call_args.kwargs["inertia_diagonal"], expected_inertia, rtol=1e-15)
        self.assertTrue(descent.call_args.kwargs["guard_assembled_metrics"])
        self.assertEqual(report["profile"], "experimental-global-ipc-guarded-contact-reference-v1")
        self.assertIn("both assembled metrics", report["contactSearchMetric"])
        self.assertFalse(report["accepted"])
        self.assertTrue(contact.path_safe(rest, final))
        np.testing.assert_array_equal(final[:3], rest[:3])


if __name__ == "__main__":
    unittest.main()
