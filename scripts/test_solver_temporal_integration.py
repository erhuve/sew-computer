import unittest
from unittest.mock import patch

import ipctk
import numpy as np

from solver_global_sewing import GlobalSewingSolver
from solver_ipc_contact import IpcSurfaceContact
from solver_temporal_separation import certify_linear_path


class StopProbe(Exception):
    pass


class TemporalIntegrationTests(unittest.TestCase):
    def setUp(self):
        ipctk.set_num_threads(1)

    def test_optimizer_and_physical_chord_remain_separate_mandatory_guards(self):
        from test_solver_ipc_integration import IpcIntegrationTests

        for reject_index in (0, 1):
            model, rest, _, _, _ = IpcIntegrationTests().fixture()
            contact = IpcSurfaceContact(rest, model.tri_indices.numpy(), activation_distance_m=.005,
                minimum_distance_m=.0001, stiffness=1e5, ccd_profile="temporal-separation-tight-inclusion")
            solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}], 1e-6, contact=contact)
            calls = []

            def predicate(a, b):
                calls.append((a.copy(), b.copy()))
                return len(calls) - 1 != reject_index

            def probe(*args, **kwargs):
                current = args[1] + 1e-6
                trial = current + 1e-6
                with patch.object(contact, "path_safe", side_effect=predicate):
                    change = kwargs["energy_change_function"](current, trial)
                self.assertEqual(change, float("inf"))
                self.assertEqual(len(calls), reject_index + 1)
                np.testing.assert_array_equal(calls[0][0].ravel(), current)
                if reject_index == 1:
                    np.testing.assert_array_equal(calls[1][0], rest)
                    np.testing.assert_array_equal(calls[0][1], calls[1][1])
                with patch.object(contact, "step_limit", return_value=.125) as limiter:
                    self.assertEqual(kwargs["step_limiter"](current, trial), .125)
                    self.assertEqual(limiter.call_count, 1)
                raise StopProbe()

            with patch("solver_global_sewing._direct_descent", side_effect=probe):
                with self.assertRaises(StopProbe):
                    solver.step(rest, np.zeros_like(rest), solver.sewing @ rest, .01)

    def test_temporal_profile_retains_fold_and_crossing_guards_and_repulsion(self):
        import test_solver_contact_range_adversarial as original

        def contact(rest, faces, activation=.00002, pressure=10000.):
            return IpcSurfaceContact(rest, faces, activation_distance_m=activation,
                minimum_distance_m=.0001, stiffness=pressure, energy_profile="area-improved-max",
                ccd_profile="temporal-separation-tight-inclusion")

        with patch.object(original, "narrow_contact", side_effect=contact):
            case = original.ContactRangeAdversarialTests()
            case.test_local_fan_fold_retains_force_and_continuous_guard()
            case.test_crossing_layers_detected_outside_activation_range()
            case.test_active_narrow_contact_repels_layer_with_pinned_surface()

    def test_endpoint_validation_cannot_be_replaced_with_pair_separation(self):
        positions = np.array([[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.],
                              [0., -.5, -1.], [0., -.5, 1.], [0., .5, 1.]])
        contact = IpcSurfaceContact(positions, np.array([[0, 1, 2], [3, 4, 5]]),
            activation_distance_m=.02, minimum_distance_m=.01, stiffness=1.,
            ccd_profile="temporal-separation-tight-inclusion")
        self.assertTrue(ipctk.has_intersections(contact.mesh, positions))
        self.assertTrue(certify_linear_path(contact.mesh, positions, positions, .01)["safe"])
        with self.assertRaisesRegex(ValueError, "Intersecting contact surface"):
            contact.path_safe(positions, positions)


if __name__ == "__main__":
    unittest.main()
