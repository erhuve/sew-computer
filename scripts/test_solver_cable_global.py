"""Actual generic cable/contact integration; no garment or capture admission."""

import copy
from fractions import Fraction as F
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("WARP_CACHE_PATH", str(Path(__file__).resolve().parents[1] / ".planning/solver/warp-cache"))

import ipctk
import newton
import numpy as np
from scipy.sparse import eye
import warp as wp

from solver_continuous_cable_sewing import ContinuousCableSewing
from solver_cable_integration import CableControl, assemble_gradient, directional_interval, stationarity
from solver_global_sewing import GlobalSewingSolver, _direct_descent
from solver_membrane_hessian import membrane_element_derivatives
from solver_rest_filtered_contact import RestFilteredSurfaceContact
from solver_triangle_sweep import triangle_sweep_safe


EMPTY = np.empty((0, 3))
PRECISION = {
    "energy_tolerance_joules": 1e-14,
    "gradient_tolerance_newtons": 1e-10,
    "hessian_tolerance_newtons_per_meter": 1e-7,
    "absolute_tolerance_joules": 1e-18,
    "max_boundary_depth": 80,
    "max_boundary_panels": 256,
    "moment_max_terms": 128,
    "moment_max_panels": 256,
    "moment_max_depth": 64,
}


def rational(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def fraction(value):
    return F(int(value["numerator"]), int(value["denominator"]))


def anchor(vertex):
    return [{"vertex": vertex, "weight": rational(F(1))}]


def cells(*, activation=1., slack_target=.003):
    return [{"id": name, "positiveStart": anchor(a), "positiveEnd": anchor(b),
             "negativeStart": anchor(a+4), "negativeEnd": anchor(b+4),
             "targetsMeters": [target, target], "referenceLengthMeters": rational(F(1, 50)),
             "stiffnessDensityNPerM2": 100., "activation": activation}
            for name, a, b, target in (("taut-center-edge", 2, 3, .0015),
                                        ("slack-outer-points", 0, 1, slack_target))]


def fixture(*, contact_enabled=True, activation=1., slack_target=.003,
            precision=None, barrier=False):
    """Independent eight-vertex, four-face cloth with no old sewing rows."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    base = np.array([[.005, .01, 0.], [.005, -.01, 0.], [0., 0., 0.], [.02, 0., 0.]])
    for height in (0., .002):
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=(base+[0., 0., height]).tolist(),
            indices=[0, 2, 3, 1, 3, 2], density=.2, tri_ke=10000, tri_ka=10000,
            tri_kd=0, edge_ke=.001, edge_kd=0)
    builder.set_coloring([[vertex] for vertex in range(8)])
    model = builder.finalize(device="cpu")
    q = model.particle_q.numpy().astype(float)
    contact = None
    if contact_enabled:
        ipctk.set_num_threads(1)
        contact = RestFilteredSurfaceContact(q, model.tri_indices.numpy(),
            activation_distance_m=.002, minimum_distance_m=.0001, stiffness=10000,
            ccd_profile="temporal-separation-tight-inclusion")
    recipe = ContinuousCableSewing(8, cells(activation=activation, slack_target=slack_target))
    solver = GlobalSewingSolver(model, [], 1e-8, contact=contact,
        continuous_cable=recipe, cable_precision=copy.deepcopy(PRECISION if precision is None else precision),
        fold_barrier_joules=1e-6 if barrier else None)
    return model, solver, q, recipe


def force_parts(solver, previous, velocity, final, dt, cable):
    _, elements, _ = membrane_element_derivatives(final[solver.faces], solver.poses,
                                                   solver.areas, solver.materials[:, :3])
    membrane = np.zeros_like(final)
    np.add.at(membrane, solver.faces, elements.reshape(-1, 3, 3))
    return {"inertia": solver.mass[:, None]*(final-previous-dt*velocity)/dt**2,
            "membrane": membrane, "bending": solver.bending.gradient(final),
            "contact": solver.contact.gradient(final) if solver.contact is not None else np.zeros_like(final),
            "cable": cable["gradient"].reshape(final.shape)}


class CableGlobalTests(unittest.TestCase):
    def test_actual_taut_cable_and_contact_share_one_guarded_force_balance(self):
        model, solver, q, recipe = fixture()
        dt = .001
        previous_velocity = np.zeros_like(q)
        rest = {"q": model.particle_q.numpy().copy(), "poses": solver.poses.copy(),
                "faces": solver.faces.copy(), "bend": solver.bending.rest_angles.copy()}
        final, velocity, report = solver.step(q, previous_velocity, EMPTY, dt)
        self.assertTrue(report["converged"], report)
        self.assertIs(report["accepted"], False)
        cable = recipe.evaluate(final, **{key: value for key, value in PRECISION.items()
                                         if key != "absolute_tolerance_joules"})
        parts = force_parts(solver, q, previous_velocity, final, dt, cable)
        residual = sum(parts.values())
        cable_error = fraction(cable["certificate"]["gradientMaxAbsoluteErrorBoundNewtons"])
        self.assertLessEqual(np.max(np.abs(residual))+float(cable_error), 1e-6+2e-10)
        self.assertGreater(cable["energy"], 0.)
        self.assertGreater(solver.contact.energy(final), 0.)
        self.assertGreater(np.max(np.abs(parts["cable"])), 1e-7)
        self.assertGreater(np.max(np.abs(parts["contact"])), 1e-7)
        self.assertGreater(np.max(np.abs(final-q)), 0.)
        self.assertEqual(report["sewingJoules"], 0.)
        self.assertEqual(solver.sewing.shape, (0, 8))
        self.assertEqual(report["contact"]["broadPhase"], "ipctk-HashGrid-explicit-v1")
        bound = fraction(report["cableStationarity"]["stationarityUpperBoundNewtons"])
        self.assertLessEqual(bound, F(1e-6))
        np.testing.assert_allclose(solver.mass@(velocity-previous_velocity), 0.,
            atol=8*dt*float(bound)+3e-12, rtol=0)
        np.testing.assert_array_equal(velocity, (final-q)/dt)
        self.assertTrue(triangle_sweep_safe(q, final, solver.faces))
        self.assertTrue(solver.contact.path_safe(q, final))
        for name, value in (("q", model.particle_q.numpy()), ("poses", solver.poses),
                            ("faces", solver.faces), ("bend", solver.bending.rest_angles)):
            np.testing.assert_array_equal(value, rest[name])

    def test_slack_cell_retains_its_target_and_has_no_hidden_compressive_response(self):
        _, solver, q, recipe = fixture()
        final, velocity, report = solver.step(q, np.zeros_like(q), EMPTY, .001)
        self.assertTrue(report["converged"], report)
        pending = ContinuousCableSewing(8, [recipe.cells[1]])
        response = pending.evaluate(final, **{key: value for key, value in PRECISION.items()
                                             if key != "absolute_tolerance_joules"})
        self.assertEqual(response["energy"], 0.)
        np.testing.assert_array_equal(response["gradient"], 0.)
        self.assertEqual(response["hessian"].nnz, 0)
        self.assertGreater(abs(np.linalg.norm(final[0]-final[4])-.003), 1e-4)
        _, changed, changed_q, changed_recipe = fixture(slack_target=.01)
        other, other_velocity, other_report = changed.step(changed_q, np.zeros_like(q), EMPTY, .001)
        self.assertTrue(other_report["converged"], other_report)
        np.testing.assert_array_equal(other, final)
        np.testing.assert_array_equal(other_velocity, velocity)
        self.assertEqual(recipe.cells[1]["targetsMeters"], [.003, .003])
        self.assertEqual(changed_recipe.cells[1]["targetsMeters"], [.01, .01])

    def test_zero_activation_matches_passive_baseline_with_empty_old_rows(self):
        model, solver, q, recipe = fixture(contact_enabled=False, activation=0.)
        final, velocity, report = solver.step(q, np.zeros_like(q), EMPTY, .001)
        self.assertTrue(report["converged"], report)
        # Native float32 rest metrics have a small passive residual. Compare
        # the same passive model, rather than pretending that it is exactly
        # stationary in the double-precision global solver.
        baseline = GlobalSewingSolver(model, [{0: 1., 4: -1.}], 1e-8)
        expected, expected_velocity, expected_report = baseline.step(
            q, np.zeros_like(q), [[.123, .456, .789]], .001, sewing_activation=[0.])
        self.assertTrue(expected_report["converged"], expected_report)
        np.testing.assert_array_equal(final, expected)
        np.testing.assert_array_equal(velocity, expected_velocity)
        self.assertEqual(report["continuousCable"]["energyJoules"], 0.)
        self.assertEqual(fraction(report["continuousCable"]["certificate"]["gradientMaxAbsoluteErrorBoundNewtons"]), 0)
        self.assertEqual(report["sewingJoules"], 0.)
        np.testing.assert_array_equal(model.particle_q.numpy().astype(float), q)
        self.assertEqual(recipe.cells[0]["targetsMeters"], [.0015, .0015])

    def test_raw_state_configuration_and_free_mass_requirements(self):
        model, solver, q, recipe = fixture(contact_enabled=False)
        for mode in ("shifted", "lsmr"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "direct"):
                solver.step(q, np.zeros_like(q), EMPTY, .001, linear_solver=mode)
        for arguments in ({"continuous_cable": recipe}, {"cable_precision": PRECISION},
                          {"continuous_cable": ContinuousCableSewing(9, cells()), "cable_precision": PRECISION}):
            with self.subTest(arguments=list(arguments)), self.assertRaises(ValueError):
                GlobalSewingSolver(model, [], 1e-8, **arguments)
        masses, flags = model.particle_mass.numpy().copy(), model.particle_flags.numpy().copy()
        for attribute, original, replacement in (("particle_mass", masses, 0.), ("particle_flags", flags, 0)):
            altered = original.copy()
            altered[0] = replacement
            getattr(model, attribute).assign(altered)
            try:
                with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, "free positive-mass"):
                    GlobalSewingSolver(model, [], 1e-8, continuous_cable=recipe, cable_precision=PRECISION)
            finally:
                getattr(model, attribute).assign(original)
        for component in ("q", "v", "dt"):
            for value in (True, 2**53+1):
                positions, velocities, dt = q.tolist(), np.zeros_like(q).tolist(), .001
                if component == "q":
                    positions[0][0] = value
                elif component == "v":
                    velocities[0][0] = value
                else:
                    dt = value
                with self.subTest(component=component, value=value), self.assertRaises(ValueError):
                    solver.step(positions, velocities, EMPTY, dt)

    def test_inactive_cable_preserves_optimizer_and_physical_triangle_guards(self):
        _, solver, q, _ = fixture(contact_enabled=False, activation=0.)
        end = q @ np.diag([-1., -1., 1.])
        middle = q @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertTrue(triangle_sweep_safe(middle, end, solver.faces))
        self.assertFalse(triangle_sweep_safe(q, end, solver.faces))

        def attempt(evaluate, start, maximum, hessian, objective, **options):
            self.assertTrue(np.isfinite(objective(end.ravel())))
            self.assertIsNone(options["energy_change_interval_function"](start, end.ravel()))
            # An individually valid optimizer chord cannot waive the saved
            # physical chord from q to end.
            self.assertIsNone(options["energy_change_interval_function"](middle.ravel(), end.ravel()))
            return SimpleNamespace(x=end.ravel(), success=True, nfev=1, status=1, message="guard attack")

        with patch("solver_global_sewing._direct_descent", side_effect=attempt), \
             self.assertRaisesRegex(ValueError, "Physical cloth step"):
            solver.step(q, np.zeros_like(q), EMPTY, .001)

    def test_inactive_cable_does_not_allow_layer_exchange_through_contact(self):
        _, solver, q, _ = fixture(activation=0.)
        end = q.copy()
        end[:4, 2], end[4:, 2] = q[4:, 2], q[:4, 2]
        self.assertTrue(triangle_sweep_safe(q, end, solver.faces))
        solver.contact.validate_state(end)

        def attempt(evaluate, start, maximum, hessian, objective, **options):
            self.assertTrue(np.isfinite(objective(end.ravel())))
            self.assertLess(options["step_limiter"](start, end.ravel()), 1.)
            self.assertIsNone(options["energy_change_interval_function"](start, end.ravel()))
            return SimpleNamespace(x=end.ravel(), success=True, nfev=1, status=1, message="layer exchange attack")

        with patch("solver_global_sewing._direct_descent", side_effect=attempt), \
             self.assertRaisesRegex(ValueError, "Physical contact step"):
            solver.step(q, np.zeros_like(q), EMPTY, .001)

    def test_explicit_fold_barrier_guards_branch_even_with_inactive_cable(self):
        from solver_hinge_sweep import hinge_sweep_safe
        _, solver, flat, _ = fixture(contact_enabled=False, activation=0., barrier=True)

        def folded(angle):
            result = flat.copy()
            radius = -flat[1, 1]
            result[1, 1] = -radius*np.cos(angle)
            result[1, 2] = -radius*np.sin(angle)
            return result

        previous, end = folded(3.1), folded(-3.1)
        self.assertTrue(triangle_sweep_safe(previous, end, solver.faces))
        self.assertTrue(hinge_sweep_safe(flat, end, solver.fold_barrier.indices))
        self.assertFalse(hinge_sweep_safe(previous, end, solver.fold_barrier.indices))

        def attempt(evaluate, start, maximum, hessian, objective, **options):
            self.assertIsNone(options["energy_change_interval_function"](flat.ravel(), end.ravel()))
            return SimpleNamespace(x=end.ravel(), success=True, nfev=1, status=1, message="branch attack")

        with patch("solver_global_sewing._direct_descent", side_effect=attempt), \
             self.assertRaisesRegex(ValueError, "fold-barrier hinge"):
            solver.step(previous, np.zeros_like(previous), EMPTY, .001)

    def test_mutating_helper_or_replacing_control_does_not_publish_state(self):
        _, solver, q, _ = fixture(contact_enabled=False)
        original = CableControl.evaluate
        for mode in ("input", "identity"):
            def corrupt(control, positions):
                response = original(control, positions)
                if mode == "input":
                    positions[0, 0] = np.nextafter(positions[0, 0], np.inf)
                else:
                    solver.continuous_cable = CableControl(ContinuousCableSewing(8, cells()), PRECISION)
                return response

            supplied, velocity = q.copy(), np.zeros_like(q)
            with self.subTest(mode=mode), patch.object(CableControl, "evaluate", corrupt), self.assertRaises(ValueError):
                solver.step(supplied, velocity, EMPTY, .001)
            np.testing.assert_array_equal(supplied, q)
            np.testing.assert_array_equal(velocity, 0.)

    def test_unresolved_active_slack_partition_rejects_without_relaxing_precision(self):
        policy = dict(PRECISION, max_boundary_depth=0)
        _, solver, q, _ = fixture(contact_enabled=False, precision=policy)
        # This genuine cloth state gives the active cell distances 2 mm to
        # 1 mm against a constant 1.5 mm target. One unrefined mixed panel
        # cannot meet the explicit narrow response bounds.
        q[7, 2] = .001
        saved, velocity = q.copy(), np.zeros_like(q)
        with self.assertRaisesRegex(ValueError, "[Bb]oundary|unresolved"):
            solver.step(q, velocity, EMPTY, .001)
        self.assertEqual(solver.continuous_cable.policy, policy)
        np.testing.assert_array_equal(q, saved)
        np.testing.assert_array_equal(velocity, 0.)

    def test_control_clone_output_isolation_and_strict_precision_admission(self):
        original = ContinuousCableSewing(8, cells())
        policy = copy.deepcopy(PRECISION)
        control = CableControl(original, policy)
        self.assertIsNot(control._potential, original)
        policy["gradient_tolerance_newtons"] = 1.
        returned = control.policy
        returned["gradient_tolerance_newtons"] = 2.
        returned_cells = original.cells
        returned_cells[0]["targetsMeters"][0] = 10.
        self.assertEqual(control.policy, PRECISION)
        self.assertEqual(control._potential.cells[0]["targetsMeters"], [.0015, .0015])
        for operation in (lambda: setattr(control, "_sealed", False), lambda: delattr(control, "_sealed")):
            with self.assertRaises(AttributeError):
                operation()
        for key in PRECISION:
            malformed = copy.deepcopy(PRECISION)
            malformed.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                CableControl(original, malformed)
            malformed = copy.deepcopy(PRECISION)
            malformed[key] = True
            with self.subTest(boolean=key), self.assertRaises(ValueError):
                CableControl(original, malformed)
        with self.assertRaises(ValueError):
            CableControl(original, dict(PRECISION, guessed_tolerance=1e-3))

    def test_directional_interval_and_exact_gradient_assembly_preserve_cancellation(self):
        baseline = np.array([1e16, 1.])
        cable = np.array([-1e16, 2.**-53])
        gradient, error, rounding = assemble_gradient(baseline, cable, F(1, 10**12))
        self.assertEqual(gradient[0], 0.)
        self.assertEqual(gradient[1], 1.)
        self.assertEqual(rounding, F(1, 2**53))
        self.assertEqual(error, F(1, 10**12)+rounding)
        start = np.array([2.**53, -1.])
        end = np.array([2.**53+2., np.nextafter(-1., 0.)])
        g = np.array([-2., 3.])
        lo, hi = directional_interval(g, F(1, 10), start, end)
        delta = [F(y)-F(x) for x, y in zip(start, end)]
        center = sum((F(x)*y for x, y in zip(g, delta)), F())
        radius = F(1, 10)*sum(map(abs, delta), F())
        self.assertEqual((lo, hi), (center-radius, center+radius))
        diagnostic = stationarity(np.array([9e-7]), F(2e-7), F(2e-7), F())
        self.assertGreater(fraction(diagnostic["stationarityUpperBoundNewtons"]), F(1e-6))

    def test_interval_search_uses_lower_armijo_slope_and_actual_rounded_displacement(self):
        # These one-dimensional callbacks test only the numerical decision
        # protocol. Actual mechanics are exercised by the cloth tests above.
        def run(start, gradient, uncertainty, change):
            initial = np.array([start])
            calls = []

            def work(first, last):
                calls.append((first.copy(), last.copy()))
                return change, change

            result = _direct_descent(lambda q: np.zeros(1), initial, 2,
                lambda q: eye(1, format="csr"), lambda q: 1.,
                gradient_function=lambda q: np.array([gradient if q[0] == start else 0.]),
                gradient_error_function=lambda q: uncertainty if q[0] == start else F(),
                energy_change_interval_function=work)
            return result, calls

        # True slope can be -5, not just the upper endpoint -3. A work change
        # -4e-4 is therefore insufficient for c=1e-4.
        rejected, calls = run(0., 2., F(1, 2), F(-4, 10000))
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(rejected.x, [0.])
        self.assertFalse(rejected.success)
        accepted, _ = run(0., 2., F(1, 2), F(-6, 10000))
        np.testing.assert_array_equal(accepted.x, [-2.])
        self.assertTrue(accepted.success)
        decision = accepted.direction_history[0]["cableAwareDecision"]
        self.assertEqual(fraction(decision["conditionalSlopeLowerJoules"]), -5)
        self.assertEqual(fraction(decision["conditionalSlopeUpperJoules"]), -3)

        # Nominal direction +1.5 rounds to an actual +2 displacement here.
        # Nominal Armijo would accept -0.00025; actual-displacement Armijo must
        # reject it because its slope is -3 rather than -2.25.
        rejected, calls = run(2.**53, -1.5, F(), F(-1, 4000))
        self.assertEqual(F(calls[0][1][0])-F(calls[0][0][0]), 2)
        np.testing.assert_array_equal(rejected.x, [2.**53])
        self.assertFalse(rejected.success)

    def test_uncertainty_cannot_publish_false_stationarity_or_nominal_descent(self):
        calls = []

        def work(first, last):
            calls.append(1)
            return F(1), F(1)

        result = _direct_descent(lambda q: np.zeros(1), np.zeros(1), 2,
            lambda q: eye(1, format="csr"), lambda q: 1.,
            gradient_function=lambda q: np.array([9e-7]),
            gradient_error_function=lambda q: F(2e-7), energy_change_interval_function=work)
        self.assertFalse(result.success)
        self.assertEqual(len(calls), 1)
        calls.clear()
        result = _direct_descent(lambda q: np.zeros(1), np.zeros(1), 2,
            lambda q: eye(1, format="csr"), lambda q: 1.,
            gradient_function=lambda q: np.array([2e-6]),
            gradient_error_function=lambda q: F(3e-6), energy_change_interval_function=work)
        self.assertFalse(result.success)
        self.assertEqual(calls, [])
        for overrides in ({"gradient_error_function": lambda q: F()},
                          {"energy_change_interval_function": work}):
            with self.assertRaises(ValueError):
                _direct_descent(lambda q: np.zeros(1), np.zeros(1), 2,
                    lambda q: eye(1, format="csr"), lambda q: 1.,
                    gradient_function=lambda q: np.ones(1), **overrides)

    def test_response_certificate_and_data_types_are_checked_before_use(self):
        control = CableControl(ContinuousCableSewing(8, cells()), copy.deepcopy(PRECISION))
        q = np.zeros((8, 3))
        q[4:, 2] = .002
        response = control.evaluate(q)
        saved = response["gradient"].copy()
        isolated = control.validate_response(q, response)
        isolated["gradient"][:] = 7.
        isolated["hessian"].data[:] = 8.
        isolated["certificate"]["accepted"] = True
        np.testing.assert_array_equal(response["gradient"], saved)
        self.assertIs(response["certificate"]["accepted"], False)
        for mode in ("accepted-type", "verified-type", "state", "bound", "gradient", "curvature"):
            altered = copy.deepcopy(response)
            if mode == "accepted-type":
                altered["certificate"]["accepted"] = 0
            elif mode == "verified-type":
                altered["certificate"]["verified"] = 1
            elif mode == "state":
                altered["certificate"]["positionsSha256"] = "0"*64
            elif mode == "bound":
                altered["certificate"]["gradientMaxAbsoluteErrorBoundNewtons"] = rational(F(1))
            elif mode == "gradient":
                altered["gradient"] = altered["gradient"].astype(bool)
            else:
                matrix = altered["hessian"].tolil()
                matrix[0, 1] = 1.
                matrix[1, 0] = 0.
                altered["hessian"] = matrix.tocsr()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                control.validate_response(q, altered)


if __name__ == "__main__":
    unittest.main()
