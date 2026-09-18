import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import eye

from solver_global_sewing import _direct_descent


class IpcCouplingAdversarialTests(unittest.TestCase):
    def fixture(self):
        from test_solver_ipc_integration import IpcIntegrationTests

        return IpcIntegrationTests().fixture()

    def test_initial_contact_force_prevents_false_stationarity(self):
        _, rest, contact, solver, targets = self.fixture()
        expected = contact.gradient(rest).ravel()[solver.free]
        self.assertGreater(np.max(np.abs(expected)), 1e-6)
        final, _, report = solver.step(rest, np.zeros_like(rest), targets, .01, max_evaluations=1)
        np.testing.assert_array_equal(final, rest)
        self.assertFalse(report["converged"])
        self.assertAlmostEqual(report["gradientInfinityNorm"], np.max(np.abs(expected)), delta=1e-7)

    def test_optimizer_and_physical_paths_are_separately_required(self):
        _, rest, contact, solver, targets = self.fixture()

        class Inspected(Exception):
            pass

        def inspect_descent(*arguments, **options):
            intermediate = rest + [0.00001, 0., 0.]
            endpoint = rest + [0.00002, 0., 0.]
            energy_change = options["energy_change_function"]
            for rejected_path in ("optimizer", "physical", None):
                calls = []

                def path_safe(start, end):
                    calls.append((start.copy(), end.copy()))
                    physical = np.array_equal(start, rest)
                    return not ((rejected_path == "physical" and physical)
                                or (rejected_path == "optimizer" and not physical))

                with patch.object(contact, "path_safe", side_effect=path_safe):
                    result = energy_change(intermediate.ravel()[solver.free], endpoint.ravel()[solver.free])
                self.assertEqual(np.isfinite(result), rejected_path is None)
                np.testing.assert_array_equal(calls[0][0], intermediate)
                if rejected_path != "optimizer":
                    self.assertEqual(len(calls), 2)
                    np.testing.assert_array_equal(calls[1][0], rest)
            raise Inspected

        with patch("solver_global_sewing._direct_descent", side_effect=inspect_descent):
            with self.assertRaises(Inspected):
                solver.step(rest, np.zeros_like(rest), targets, .01)

    def test_zero_and_unrepresentable_steps_do_not_converge(self):
        for bound in (False, 0., 1e-320):
            result = _direct_descent(lambda positions: positions - 2., np.array([1.]), 4,
                                     lambda positions: eye(1), lambda positions: .5 * (positions[0] - 2.) ** 2,
                                     gradient_function=lambda positions: positions - 2.,
                                     step_limiter=lambda start, end: bound)
            self.assertFalse(result.success)
            np.testing.assert_array_equal(result.x, [1.])

    def test_malformed_step_bounds_reject(self):
        for bound in (-.01, 1.01, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "Invalid search step bound"):
                _direct_descent(lambda positions: positions - 2., np.array([1.]), 4,
                                lambda positions: eye(1), lambda positions: .5 * (positions[0] - 2.) ** 2,
                                gradient_function=lambda positions: positions - 2.,
                                step_limiter=lambda start, end: bound)


if __name__ == "__main__":
    unittest.main()
