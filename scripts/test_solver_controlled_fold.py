"""Explicit fold activation/work mechanics, separate from captured execution."""

import copy
from fractions import Fraction
import math
import os
from pathlib import Path
import unittest

os.environ.setdefault("WARP_CACHE_PATH", str(Path(__file__).resolve().parents[1] / ".planning/solver/warp-cache"))

import newton
import numpy as np
import warp as wp

from solver_controlled_fold import ControlledFoldActuation, COEFFICIENT_POLICY, PARAMETER_ORDER
from solver_fold_actuation import FoldActuation
from solver_fold_control_schedule import FoldControlSchedule
from solver_triangle_sweep import triangle_sweep_safe


def rat(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def fixture(count=2):
    base = [[.25, 1., 0.], [.75, -1., 0.], [0., 0., 0.], [1., 0., 0.]]
    builder = newton.ModelBuilder(gravity=(0., 0., 0.))
    for index in range(count):
        vertices = (np.array(base) + [3. * index, 0., 0.]).tolist()
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=vertices, indices=[0, 2, 3, 1, 3, 2],
            density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.001, edge_kd=0)
    builder.set_coloring([[vertex] for vertex in range(4 * count)])
    model = builder.finalize(device="cpu")
    native = model.edge_indices.numpy()
    hinges = native[np.all(native >= 0, axis=1)]
    return model, model.particle_q.numpy().astype(float), hinges


class ControlledFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.flat, cls.hinges = fixture()
        cls.recipe = ControlledFoldActuation(cls.model, cls.hinges, [1., 2.])

    def test_weighted_energy_gradient_jacobian_and_gauss_newton_metric(self):
        q = self.flat.copy()
        q[0, 2], q[4, 2] = .13, -.17
        potential = self.recipe.potential([.3, -.2], [.25, .75])
        gradient = potential.gradient(q)
        jacobian = potential.jacobian(q).toarray()
        hessian = potential.hessian(q).toarray()
        self.assertEqual(potential.coefficients.tolist(), [.25, 1.5])
        for index, direction in enumerate(np.eye(q.size).reshape((-1, *q.shape))):
            epsilon = 1e-6
            observed = (potential.energy(q + epsilon * direction) - potential.energy(q - epsilon * direction)) / (2 * epsilon)
            np.testing.assert_allclose(observed, gradient.ravel()[index], rtol=2e-8, atol=2e-10)
            residual_change = (potential.residual(q + epsilon * direction) - potential.residual(q - epsilon * direction)) / (2 * epsilon)
            np.testing.assert_allclose(residual_change, jacobian[:, index], rtol=2e-8, atol=2e-10)
        np.testing.assert_allclose(hessian, jacobian.T @ jacobian, atol=5e-16, rtol=2e-15)
        np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=3e-15, rtol=0)
        np.testing.assert_allclose(np.cross(q, gradient).sum(axis=0), 0., atol=3e-15, rtol=0)

    def test_rotation_covariance_and_balanced_reactions(self):
        q = self.flat.copy()
        q[0, 2], q[4, 2] = .1, -.2
        rotation = np.linalg.qr(np.random.default_rng(611).normal(size=(3, 3)))[0]
        moved = q @ rotation + [.3, -.5, .7]
        potential = self.recipe.potential([.3, -.4], [.4, .6])
        self.assertAlmostEqual(potential.energy(q), potential.energy(moved), places=14)
        np.testing.assert_allclose(potential.gradient(q) @ rotation, potential.gradient(moved), rtol=1e-13, atol=1e-14)

    def test_inactive_potential_skips_undefined_angles_but_keeps_complete_identity(self):
        collapsed = self.flat.copy()
        collapsed[:4] = 0.
        potential = self.recipe.potential([.3, .2], [0., .5])
        diagnostic = potential.diagnostics(collapsed)
        self.assertEqual(diagnostic["hinges"], self.hinges.tolist())
        self.assertIsNone(diagnostic["sampledActiveAnglesRadians"][0])
        self.assertEqual(diagnostic["activeHingeIndices"], [1])
        np.testing.assert_array_equal(potential.gradient(collapsed)[:4], 0.)
        self.assertFalse(triangle_sweep_safe(self.flat, collapsed, self.model.tri_indices.numpy()))
        released = self.recipe.potential([.3, .2], [0., 0.])
        self.assertEqual(released.energy(np.zeros_like(collapsed)), 0.)
        self.assertEqual(released.energy_change(self.flat, np.zeros_like(collapsed)), 0.)
        np.testing.assert_array_equal(released.gradient(collapsed), np.zeros_like(collapsed))
        self.assertEqual(released.residual(collapsed).size, 0)
        self.assertEqual(released.jacobian(collapsed).shape, (0, collapsed.size))
        self.assertEqual(released.hessian(collapsed).nnz, 0)
        with self.assertRaises(ValueError):
            self.recipe.potential([.3, .2], [1., .5]).energy(collapsed)

    def test_parameter_work_uses_union_geometry_for_release_and_engagement(self):
        collapsed = self.flat.copy()
        collapsed[:4] = 0.
        for old, new in (([1., 0.], [0., 0.]), ([0., 0.], [1., 0.])):
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                self.recipe.parameter_energy_change(collapsed, [0., 0.], old, [1., 1.], new)
        work = self.recipe.parameter_energy_change(collapsed, [0., 0.], [0., 0.], [1., 1.], [0., 0.])
        self.assertEqual(work["totalParameterWorkJoules"], 0.)
        self.assertEqual(work["sampledAnglesRadians"], [None, None])
        self.assertEqual(work["perHinge"], [])

    def test_target_first_order_records_release_despite_zero_total(self):
        recipe = ControlledFoldActuation(self.model, self.hinges, [2., 2.])
        work = recipe.parameter_energy_change(self.flat, [0., 0.], [1., 0.], [1., 0.], [0., 0.])
        self.assertEqual(work["parameterOrder"], PARAMETER_ORDER)
        self.assertEqual(work["targetParameterWorkJoules"], 1.)
        self.assertEqual(work["activationParameterWorkJoules"], -1.)
        self.assertEqual(work["releaseEnergyRemovedJoules"], 1.)
        self.assertEqual(work["totalParameterWorkJoules"], 0.)

    def test_opposing_target_changes_retain_tiny_exact_total(self):
        recipe = ControlledFoldActuation(self.model, self.hinges, [1., 1.])
        targets = [1. + 2.**-52, 1. - 2.**-52]
        work = recipe.parameter_energy_change(self.flat, [1., 1.], [1., 1.], targets, [1., 1.])
        self.assertEqual(work["totalParameterWorkJoules"], 2.**-104)
        self.assertEqual(rat(work["exactWorkJoules"]["total"]), Fraction(1, 2**104))
        old = np.array([1., 1.]); change = np.array(targets) - old
        legacy = float(np.sum(change * (old + .5 * change)))
        self.assertEqual(legacy, 2.**-105)

    def test_cancelling_target_and_activation_components_do_not_erase_total(self):
        recipe = ControlledFoldActuation(self.model, self.hinges, [1., 1.])
        work = recipe.parameter_energy_change(self.flat, [1., 0.], [1., 0.], [2., 0.], [math.nextafter(.25, 1.), 0.])
        self.assertEqual(work["totalParameterWorkJoules"], 2.**-53)
        self.assertEqual(work["targetParameterWorkJoules"] + work["activationParameterWorkJoules"], 0.)
        self.assertGreaterEqual(work["roundedComponentSumErrorBoundJoules"], abs(work["totalParameterWorkJoules"]))

    def test_rounded_coefficient_plateau_has_zero_activation_work(self):
        recipe = ControlledFoldActuation(self.model, self.hinges, [.75, 1.])
        first, second = 1. - 2.**-52, 1. - 3. * 2.**-53
        work = recipe.parameter_energy_change(self.flat, [1., 0.], [first, 0.], [1., 0.], [second, 0.])
        before, after = work["previousCoefficientWitnesses"][0], work["coefficientWitnesses"][0]
        self.assertEqual(before["numericalCoefficientJoules"], after["numericalCoefficientJoules"])
        self.assertNotEqual(rat(before["exactProductJoules"]), rat(after["exactProductJoules"]))
        self.assertEqual(work["activationParameterWorkJoules"], 0.)
        self.assertEqual(work["totalParameterWorkJoules"], 0.)
        self.assertEqual(rat(before["roundedMinusExactProductJoules"]), -Fraction(1, 2**54))
        self.assertEqual(rat(after["roundedMinusExactProductJoules"]), Fraction(1, 2**55))
        self.assertEqual(work["coefficientPolicy"], COEFFICIENT_POLICY)

    def test_positive_coefficient_underflow_rejects_but_declared_zero_remains_zero(self):
        recipe = ControlledFoldActuation(self.model, self.hinges, [math.ulp(0.), 1.])
        with self.assertRaisesRegex(ValueError, "underflow"):
            recipe.potential([0., 0.], [.5, 0.])
        self.assertEqual(recipe.potential([0., 0.], [0., 0.]).coefficients.tolist(), [0., 0.])

    def test_force_contraction_retains_representable_result_after_tiny_coefficient(self):
        q = self.flat.copy()
        q[:4] = np.array([[0., 1., 0.], [0., -1., 0.], [0., 0., 0.], [1., 0., 0.]]) * 1e-50
        beta = math.ulp(0.)
        potential = ControlledFoldActuation(self.model, self.hinges, [beta, 1.]).potential([.1, 0.], [1., 0.])
        # For a flat hinge these derivatives are analytic. Intermediate
        # beta*error rounds to zero, while the final four forces do not.
        self.assertEqual(beta * -.1, 0.)
        derivative = np.zeros_like(q)
        derivative[:4, 2] = [-1e50, -1e50, 2e50, 0.]
        expected = np.array([float(Fraction(beta) * -Fraction(.1) * Fraction(float(value)))
                             for value in derivative.flat]).reshape(q.shape)
        np.testing.assert_array_equal(potential.gradient(q), expected)
        self.assertNotEqual(expected[0, 2], 0.)

    def test_metric_contraction_does_not_underflow_asymmetrically(self):
        q = self.flat.copy()
        q[:4] = [[.25, 1e-8, 0.], [.75, -10., 0.], [0., 0., 0.], [1., 0., 0.]]
        beta = math.ulp(0.)
        potential = ControlledFoldActuation(self.model, self.hinges, [beta, 1.]).potential([.1, 0.], [1., 0.])
        metric = potential.hessian(q).toarray()
        expected = float(Fraction(beta) * Fraction(-1e8) * Fraction(-.1))
        self.assertGreater(expected, 0.)
        self.assertEqual(metric[2, 5], expected)
        np.testing.assert_array_equal(metric, metric.T)

    def test_shared_vertex_force_cancellation_accumulates_before_rounding(self):
        q = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])
        builder = newton.ModelBuilder(gravity=(0., 0., 0.))
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=q.tolist(), indices=[0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1],
            density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.001, edge_kd=0)
        builder.set_coloring([[vertex] for vertex in range(5)])
        model = builder.finalize(device="cpu")
        hinges = [[2, 4, 0, 1], [2, 4, 3, 0]]
        targets = [1., math.nextafter(-1., 0.)]
        potential = ControlledFoldActuation(model, hinges, [1e12, 1e12]).potential(targets, [1., 1.])
        force = -Fraction(1e12) * (Fraction(targets[0]) + Fraction(targets[1]))
        expected = np.zeros_like(q)
        expected[:, 2] = [float(force * value) for value in [2, 0, -1, 0, -1]]
        np.testing.assert_array_equal(potential.gradient(q), expected)

    def test_active_tiny_motion_is_retained_when_legacy_endpoint_subtraction_is_zero(self):
        q = self.flat.copy()
        q[:4] = [[.2, .8, .3], [.6, -.7, -.1], [0., 0., 0.], [1., 0., 0.]]
        moved = q.copy(); moved[0, 1] = math.nextafter(q[0, 1], math.inf)
        potential = self.recipe.potential([.1, 0.], [1., 0.])
        delta = potential.energy_change(q, moved)
        expected = float(np.sum(potential.gradient(q) * (moved - q)))
        self.assertNotEqual(delta, 0.)
        np.testing.assert_allclose(delta, expected, rtol=2e-12, atol=1e-31)
        np.testing.assert_allclose(potential.energy_change(moved, q), -delta, rtol=2e-12, atol=1e-31)

    def test_motion_change_matches_endpoint_energy_for_resolved_updates(self):
        q = self.flat.copy(); q[0, 2], q[4, 2] = .2, -.1
        moved = q.copy(); moved[0, 2] += .01; moved[4, 1] -= .05
        potential = self.recipe.potential([.3, -.4], [.25, .75])
        np.testing.assert_allclose(potential.energy_change(q, moved), potential.energy(moved) - potential.energy(q), rtol=1e-12, atol=1e-15)

    def test_immutable_snapshots_include_nested_force_evaluator(self):
        targets, activation = [.3, -.2], [.4, .6]
        potential = self.recipe.potential(targets, activation)
        before = (potential.energy(self.flat), potential.gradient(self.flat).copy())
        targets[0] = 1.; activation[0] = 0.
        potential.targets[:] = 9.; potential.activation[:] = 0.; potential.coefficients[:] = 99.
        potential.all_hinges[:] = 0; self.recipe.hinges[:] = 0; self.recipe.stiffness[:] = 99.
        potential._geometry.weights = np.full(2, 4.)
        potential._geometry.indices = np.zeros((2, 4), dtype=int)
        self.assertEqual(potential.energy(self.flat), before[0])
        np.testing.assert_array_equal(potential.gradient(self.flat), before[1])
        for field in ("_targets", "_activation", "_coefficients", "_active"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                getattr(potential, field).setflags(write=True)
            with self.assertRaises(AttributeError):
                setattr(potential, field, np.zeros(2))

    def test_original_scalar_integer_precision_is_checked_before_numpy_promotion(self):
        for values in ([2**53 + 1, 1.], np.array([2**53 + 1, 1], dtype=np.int64)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ControlledFoldActuation(self.model, self.hinges, values)
        for q in (self.flat.tolist(), np.zeros_like(self.flat, dtype=np.int64)):
            q[0][0] = 2**53 + 1
            with self.assertRaises(ValueError):
                self.recipe.potential([0., 0.], [0., 0.]).energy(q)
        valid = ControlledFoldActuation(self.model, self.hinges, [2**53, 1.])
        self.assertEqual(int(valid.stiffness[0]), 2**53)

    def test_malformed_targets_activation_topology_and_branch_limits_reject(self):
        for targets in ([True, .1], [float("nan"), 0.], [math.pi, 0.], [-math.pi, 0.], [0.], None):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.recipe.potential(targets, [0., 0.])
        for activation in ([True, 0.], [-.1, 0.], [1.01, 0.], [float("inf"), 0.], [0.], None):
            with self.subTest(activation=activation), self.assertRaises(ValueError):
                self.recipe.potential([0., 0.], activation)
        edge = math.pi - 1e-8
        with self.assertRaises(ValueError):
            self.recipe.potential([edge, 0.], [0., 0.])
        self.recipe.potential([math.nextafter(edge, 0.), 0.], [0., 0.])
        for hinges in (self.hinges.astype(float), self.hinges[::-1, ::-1], self.hinges[:1].repeat(2, axis=0)):
            with self.assertRaises(ValueError):
                ControlledFoldActuation(self.model, hinges, [1., 1.])

    def test_schedule_drives_independent_hinges_then_releases_without_rest_changes(self):
        original_rest = self.model.particle_q.numpy().copy()
        recipe = {"profile": "fold-angle-activation-v1", "hinges": self.hinges.tolist(), "knots": [
            {"fraction": 0., "targetsRadians": [0., 0.], "activation": [1., 0.]},
            {"fraction": .25, "targetsRadians": [.1, 0.], "activation": [1., 0.]},
            {"fraction": .5, "targetsRadians": [.1, -.2], "activation": [1., 1.]},
            {"fraction": .75, "targetsRadians": [.1, -.2], "activation": [0., 0.]},
            {"fraction": 1., "targetsRadians": [.1, -.2], "activation": [0., 0.]}]}
        schedule = FoldControlSchedule(recipe, 8, hinges=self.hinges)
        original = copy.deepcopy(recipe)
        for fraction in (0., .125, .25, .375, .5, .625, .75, .875, 1.):
            targets, activation = schedule.parameters(fraction)
            potential = self.recipe.potential(targets, activation)
            if fraction < .25:
                self.assertEqual(potential.diagnostics(self.flat)["inactiveHingeIndices"], [1])
            if fraction >= .75:
                self.assertEqual(potential.energy(self.flat), 0.)
                np.testing.assert_array_equal(potential.gradient(self.flat), 0.)
        self.assertEqual(recipe, original)
        np.testing.assert_array_equal(self.model.particle_q.numpy(), original_rest)

    def test_legacy_fixed_fold_api_remains_unchanged(self):
        legacy = FoldActuation(self.model, self.hinges, [1., 2.])
        q = self.flat.copy(); q[0, 2], q[4, 2] = .13, -.17
        old = legacy.potential([.3, -.2]); new = self.recipe.potential([.3, -.2], [1., 1.])
        np.testing.assert_allclose(old.energy(q), new.energy(q), rtol=5e-16, atol=1e-16)
        np.testing.assert_allclose(old.gradient(q), new.gradient(q), rtol=3e-15, atol=1e-15)
        self.assertIsInstance(old, type(legacy.potential([0., 0.])))
        with self.assertRaises(TypeError):
            legacy.potential([0., 0.], activation=[1., 1.])


if __name__ == "__main__":
    unittest.main()
