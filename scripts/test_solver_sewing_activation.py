"""Portable mathematical checks for pending/active canonical sewing rows."""

from decimal import Decimal, localcontext
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from solver_distance_sewing import DistanceSewing
from solver_normal_sewing import NormalOffsetSewing
from solver_sewing_activation import validate_sewing_activation


def energy(potential, positions):
    residual = potential.residual(positions)
    return .5 * residual @ residual


class SewingActivationTests(unittest.TestCase):
    def setUp(self):
        self.rows = csr_matrix([[-.2, -.3, -.5, 1., 0., 0., 0., 0.],
                                [0., 0., 0., 0., -.25, -.25, -.5, 1.]])
        self.positions = np.array([[0., 0., 0.], [.02, 0., .003],
            [0., .03, -.002], [.008, .017, .004],
            [.1, .02, -.01], [.12, .022, -.008],
            [.103, .055, -.007], [.105, .041, -.006]])
        self.frames = np.array([[0, 1, 2], [4, 5, 6]])
        self.sides = np.array([1, -1])
        self.targets = np.array([.002, .003])
        self.activation = np.array([.25, .7])
        self.compliance = .003

    def potential(self, kind, activation=None, **overrides):
        common = dict(sewing=self.rows, targets=self.targets, compliance=self.compliance,
                      activation=activation)
        if kind == "normal":
            common.update(frame_faces=self.frames, sides=self.sides)
        common.update(overrides)
        return (NormalOffsetSewing if kind == "normal" else DistanceSewing)(**common)

    def test_validation_defaults_bounds_and_input_isolation(self):
        np.testing.assert_array_equal(validate_sewing_activation(None, 2), [1., 1.])
        self.assertEqual(validate_sewing_activation(None, 0).shape, (0,))
        self.assertEqual(validate_sewing_activation([], 0).shape, (0,))
        self.assertEqual(validate_sewing_activation(None, 32768).shape, (32768,))
        for original in ([0., .3], np.array([0., .3]), (0, 1)):
            with self.subTest(original=original):
                result = validate_sewing_activation(original, 2)
                self.assertEqual(result.dtype, np.float64)
                result[:] = .5
                np.testing.assert_array_equal(original, [0, 1] if isinstance(original, tuple) else [0., .3])
        for count in (-1, 32769, 2., True, np.int64(2)):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_sewing_activation(None, count)

    def test_raw_boolean_shape_type_and_nonfinite_activation_reject(self):
        invalid = (True, .5, [True, 1.], [np.bool_(False), 1.], np.array([True, False]),
            [], [.5], [.5, .5, .5], [[.5], [.5]], np.ones((2, 1)),
            [np.array(.5), .5], ["0.5", "1"], [0, 1j], np.array([.5, 1.], dtype=object),
            [np.nan, 1.], [np.inf, 0.], [-1e-30, 1.], [0., np.nextafter(1., 2.)],
            {0: .5, 1: .5})
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_sewing_activation(values, 2)
            for kind in ("distance", "normal"):
                with self.subTest(kind=kind, values=values), self.assertRaises(ValueError):
                    self.potential(kind, values)

    def test_positive_extended_precision_cannot_become_inactive(self):
        smallest = np.nextafter(0., 1.)
        np.testing.assert_array_equal(validate_sewing_activation([0., smallest], 2), [0., smallest])
        for kind in ("distance", "normal"):
            np.testing.assert_array_equal(self.potential(kind, [0., smallest]).activation, [0., smallest])
        extended = np.array([0., np.nextafter(np.longdouble(0), np.longdouble(1))], dtype=np.longdouble)
        with np.errstate(under="ignore"):
            loses_positive = extended[1] > 0 and extended.astype(float)[1] == 0
        if loses_positive:
            for values in (extended, list(extended)):
                with self.subTest(values=values), np.errstate(all="raise"):
                    with self.assertRaisesRegex(ValueError, "underflow to inactive"):
                        validate_sewing_activation(values, 2)
                    for kind in ("distance", "normal"):
                        with self.assertRaisesRegex(ValueError, "underflow to inactive"):
                            self.potential(kind, values)
        else:
            # macOS ARM longdouble aliases float64; the smallest positive is representable.
            np.testing.assert_array_equal(validate_sewing_activation(extended, 2), [0., smallest])

    def test_weighted_energies_match_independent_source_geometry(self):
        vectors = self.rows.toarray() @ self.positions
        triangles = self.positions[self.frames]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals = cross / np.linalg.norm(cross, axis=1)[:, None]
        errors = {"distance": (np.linalg.norm(vectors, axis=1) - self.targets) ** 2,
                  "normal": np.sum((vectors - (self.targets * self.sides)[:, None] * normals) ** 2, axis=1)}
        for kind in errors:
            with self.subTest(kind=kind):
                potential = self.potential(kind, self.activation)
                expected = np.dot(self.activation, errors[kind]) / (2 * self.compliance)
                self.assertAlmostEqual(energy(potential, self.positions), expected, places=16)

    def test_weighted_derivatives_and_motion_work(self):
        epsilon = 1e-7
        for kind in ("distance", "normal"):
            with self.subTest(kind=kind):
                potential = self.potential(kind, self.activation)
                gradient = potential.gradient(self.positions)
                jacobian = potential.jacobian(self.positions).toarray()
                exact = potential.exact_hessian(self.positions).toarray()
                np.testing.assert_allclose(exact, exact.T, atol=1e-10, rtol=0)
                for index, direction in enumerate(np.eye(24).reshape((-1, 8, 3))):
                    plus, minus = self.positions + epsilon * direction, self.positions - epsilon * direction
                    np.testing.assert_allclose((potential.residual(plus) - potential.residual(minus))
                                               / (2 * epsilon), jacobian[:, index], atol=1e-8, rtol=2e-8)
                    self.assertAlmostEqual((energy(potential, plus) - energy(potential, minus))
                                           / (2 * epsilon), gradient[index], places=8)
                    np.testing.assert_allclose((potential.gradient(plus) - potential.gradient(minus))
                                               / (2 * epsilon), exact[:, index], atol=2e-7, rtol=3e-7)
                    self.assertAlmostEqual(potential.energy_change(minus, plus),
                                           energy(potential, plus) - energy(potential, minus), places=15)
                np.testing.assert_allclose(jacobian.T @ potential.residual(self.positions), gradient, atol=1e-13)
                if kind == "normal":
                    np.testing.assert_allclose(potential.hessian(self.positions).toarray(),
                                               jacobian.T @ jacobian, atol=1e-12)

    def test_fractional_weights_scale_each_retained_row(self):
        for kind in ("distance", "normal"):
            with self.subTest(kind=kind):
                potential = self.potential(kind, self.activation)
                full = self.potential(kind)
                residual_scale = np.sqrt(self.activation) if kind == "distance" else np.repeat(np.sqrt(self.activation), 3)
                np.testing.assert_allclose(potential.residual(self.positions), full.residual(self.positions) * residual_scale)
                np.testing.assert_allclose(potential.jacobian(self.positions).toarray(),
                                           full.jacobian(self.positions).toarray() * residual_scale[:, None])
                gradient = np.zeros(24)
                exact = np.zeros((24, 24))
                hessian = np.zeros((24, 24))
                for index in range(2):
                    isolated = self.potential(kind, np.eye(2)[index])
                    gradient += self.activation[index] * isolated.gradient(self.positions)
                    exact += self.activation[index] * isolated.exact_hessian(self.positions).toarray()
                    hessian += self.activation[index] * isolated.hessian(self.positions).toarray()
                np.testing.assert_allclose(potential.gradient(self.positions), gradient, atol=1e-13)
                np.testing.assert_allclose(potential.exact_hessian(self.positions).toarray(), exact, atol=1e-10)
                np.testing.assert_allclose(potential.hessian(self.positions).toarray(), hessian, atol=1e-10)
                np.testing.assert_array_equal(potential.sewing.toarray(), self.rows.toarray())
                np.testing.assert_array_equal(potential.targets, self.targets)

    def test_covariance_and_internal_force_torque(self):
        rotation = np.linalg.qr(np.random.default_rng(825).normal(size=(3, 3)))[0]
        transformed = self.positions @ rotation + [.1, -.2, .3]
        transform = np.kron(np.eye(8), rotation.T)
        for kind in ("distance", "normal"):
            with self.subTest(kind=kind):
                potential = self.potential(kind, self.activation)
                gradient = potential.gradient(self.positions).reshape((-1, 3))
                self.assertAlmostEqual(energy(potential, transformed), energy(potential, self.positions), places=15)
                np.testing.assert_allclose(potential.gradient(transformed).reshape((-1, 3)), gradient @ rotation, atol=1e-12)
                np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-12)
                np.testing.assert_allclose(np.cross(self.positions, gradient).sum(axis=0), 0., atol=1e-12)
                np.testing.assert_allclose(potential.exact_hessian(transformed).toarray(),
                    transform @ potential.exact_hessian(self.positions).toarray() @ transform.T, atol=1e-9, rtol=1e-10)

    def test_pending_coincident_or_collapsed_geometry_is_not_evaluated(self):
        collapsed = self.positions.copy()
        collapsed[4:] = 0.
        for kind in ("distance", "normal"):
            with self.subTest(kind=kind):
                potential = self.potential(kind, [1., 0.])
                row_slice = slice(1, 2) if kind == "distance" else slice(3, 6)
                np.testing.assert_array_equal(potential.residual(collapsed)[row_slice], 0.)
                self.assertEqual(potential.jacobian(collapsed)[row_slice].nnz, 0)
                np.testing.assert_array_equal(potential.gradient(collapsed).reshape((-1, 3))[4:], 0.)
                self.assertEqual(potential.hessian(collapsed)[12:].nnz, 0)
                self.assertEqual(potential.exact_hessian(collapsed)[12:].nnz, 0)
                self.assertEqual(potential.energy_change(self.positions, collapsed), 0.)
                with self.assertRaises(ValueError):
                    self.potential(kind, [1., np.nextafter(0., 1.)]).residual(collapsed)
                geometry = potential.geometry(collapsed)
                if kind == "distance":
                    np.testing.assert_array_equal(geometry[0][1], 0.)
                    self.assertEqual(geometry[1][1], 1.)
                else:
                    for value in geometry[:3]:
                        np.testing.assert_array_equal(value[1], 0.)
                    self.assertEqual(geometry[3][1], 1.)

    def test_all_pending_gives_exact_zero_without_inactive_arithmetic(self):
        for kind in ("distance", "normal"):
            with self.subTest(kind=kind):
                potential = self.potential(kind, [0., 0.])
                for positions in (np.zeros((8, 3)), np.full((8, 3), 1e308)):
                    with np.errstate(all="raise"):
                        self.assertEqual(energy(potential, positions), 0.)
                        np.testing.assert_array_equal(potential.residual(positions), 0.)
                        np.testing.assert_array_equal(potential.gradient(positions), 0.)
                        self.assertEqual(potential.jacobian(positions).nnz, 0)
                        self.assertEqual(potential.hessian(positions).nnz, 0)
                        self.assertEqual(potential.exact_hessian(positions).nnz, 0)
                        self.assertEqual(potential.energy_change(positions, -positions), 0.)
                with self.assertRaises(ValueError):
                    potential.residual(np.full((8, 3), np.nan))
                with self.assertRaises(ValueError):
                    potential.residual(np.zeros((7, 3)))

    def test_inactive_unrelated_overflow_does_not_affect_active_work(self):
        start, end = self.positions.copy(), self.positions.copy()
        start[4:] = 1e308
        end[4:] = -1e308
        end[3, 2] += 1e-7
        for kind in ("distance", "normal"):
            with self.subTest(kind=kind), np.errstate(all="raise"):
                potential = self.potential(kind, [1., 0.])
                work = potential.energy_change(start, end)
                clean_end = self.positions.copy()
                clean_end[3, 2] += 1e-7
                self.assertEqual(work, potential.energy_change(self.positions, clean_end))

    def test_inactive_rows_still_require_explicit_valid_targets_and_frames(self):
        for kind in ("distance", "normal"):
            for targets in ([0., .003], [-1., .003], [True, .003], [np.nan, .003], [np.inf, .003]):
                with self.subTest(kind=kind, targets=targets), self.assertRaises(ValueError):
                    self.potential(kind, [0., 0.], targets=targets)
            for compliance in (True, 0., -1., np.nan, np.inf):
                with self.subTest(kind=kind, compliance=compliance), self.assertRaises(ValueError):
                    self.potential(kind, [0., 0.], compliance=compliance)
        for frames in ([[0, 0, 2], [4, 5, 6]], [[0, 1, 2], [4, 5, 8]],
                       [[0, 1, 3], [4, 5, 6]], [[False, 1, 2], [4, 5, 6]]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                self.potential("normal", [0., 0.], frame_faces=frames)
        for sides in ([1, 0], [1, True], [1., -1.]):
            with self.subTest(sides=sides), self.assertRaises(ValueError):
                self.potential("normal", [0., 0.], sides=sides)
        with self.assertRaises(ValueError):
            self.potential("normal", [0., 0.], sewing=self.rows * 2)

    def test_captured_inputs_and_returned_activation_are_isolated(self):
        for kind in ("distance", "normal"):
            with self.subTest(kind=kind):
                rows, targets, activation = self.rows.copy(), self.targets.copy(), self.activation.copy()
                frames, sides = self.frames.copy(), self.sides.copy()
                extra = {"frame_faces": frames, "sides": sides} if kind == "normal" else {}
                potential = self.potential(kind, activation, sewing=rows, targets=targets, **extra)
                initial = potential.gradient(self.positions).copy()
                rows.data[:] = 0.
                targets[:] = 1.
                activation[:] = 0.
                frames[:] = 0
                sides[:] = 0
                potential.activation[:] = 0.
                np.testing.assert_array_equal(potential.activation, self.activation)
                np.testing.assert_array_equal(potential.gradient(self.positions), initial)
                with self.assertRaises(ValueError):
                    potential._activation.flags.writeable = True

    def test_default_matches_explicit_ones(self):
        for kind in ("distance", "normal"):
            first, second = self.potential(kind), self.potential(kind, [1., 1.])
            for method in ("residual", "gradient"):
                np.testing.assert_array_equal(getattr(first, method)(self.positions), getattr(second, method)(self.positions))
            for method in ("jacobian", "hessian", "exact_hessian"):
                np.testing.assert_array_equal(getattr(first, method)(self.positions).toarray(),
                                               getattr(second, method)(self.positions).toarray())
            moved = self.positions.copy()
            moved[3, 0] += 1e-5
            self.assertEqual(first.energy_change(self.positions, moved), second.energy_change(self.positions, moved))

    def test_empty_canonical_rows_keep_full_vertex_derivative_shapes(self):
        for kind in ("distance", "normal"):
            extra = {"frame_faces": np.empty((0, 3), dtype=int), "sides": np.empty(0, dtype=int)} if kind == "normal" else {}
            potential = self.potential(kind, [], sewing=csr_matrix((0, 4)), targets=[], **extra)
            positions = np.zeros((4, 3))
            self.assertEqual(potential.residual(positions).shape, (0,))
            self.assertEqual(potential.gradient(positions).shape, (12,))
            self.assertEqual(potential.jacobian(positions).shape, (0, 12))
            self.assertEqual(potential.hessian(positions).shape, (12, 12))
            self.assertEqual(potential.exact_hessian(positions).shape, (12, 12))
            self.assertEqual(potential.energy_change(positions, positions), 0.)

    def test_tiny_motion_work_survives_equal_rounded_energies(self):
        positions = np.array([[0., 0., 0.], [.03125, 0., 0.],
                              [0., .03125, 0.], [.0078125, .015625, .03125]])
        end = positions.copy()
        end[3, 2] = np.nextafter(end[3, 2], 1.)
        rows = csr_matrix([[-.25, -.25, -.5, 1.]])
        for kind in ("distance", "normal"):
            extra = {"frame_faces": [[0, 1, 2]], "sides": [1]} if kind == "normal" else {}
            potential = self.potential(kind, [.25], sewing=rows, targets=[1e6], **extra)
            self.assertEqual(energy(potential, positions), energy(potential, end))
            with localcontext() as context:
                context.prec = 100
                old, new = Decimal.from_float(positions[3, 2]), Decimal.from_float(end[3, 2])
                target = Decimal.from_float(1e6)
                expected = float(Decimal.from_float(.25) * ((new - target) ** 2 - (old - target) ** 2)
                                 / (2 * Decimal.from_float(self.compliance)))
            actual = potential.energy_change(positions, end)
            self.assertNotEqual(actual, 0.)
            self.assertLess(abs(actual - expected), abs(expected) * 3e-16)


if __name__ == "__main__":
    unittest.main()
