"""Independent analytical observations and strict-admission regressions."""

import copy
from fractions import Fraction
import hashlib
import json
import math
import unittest

import numpy as np

from solver_rigid_velocity_diagnostics import analyze_rigid_velocity


def square():
    return np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])


class RigidVelocityDiagnosticsTests(unittest.TestCase):
    def test_exact_translation_has_no_angular_or_residual_motion(self):
        q = square()
        velocity = np.tile([2., -3., 4.], (4, 1))
        observed = analyze_rigid_velocity(q, velocity, np.ones(4))
        self.assertEqual(observed["totalMassKg"], 4.)
        self.assertEqual(observed["centerOfMassMeters"], [0., 0., 0.])
        self.assertEqual(observed["translationVelocityMetersPerSecond"], [2., -3., 4.])
        self.assertEqual(observed["linearMomentumKgMetersPerSecond"], [8., -12., 16.])
        self.assertEqual(observed["angularVelocityRadiansPerSecond"], [0., 0., 0.])
        np.testing.assert_array_equal(observed["residualVelocitiesMetersPerSecond"], np.zeros((4, 3)))
        self.assertEqual(observed["kineticJoules"], 58.)
        self.assertEqual(observed["translationKineticJoules"], 58.)
        self.assertEqual(observed["rotationKineticJoules"], 0.)
        self.assertEqual(observed["residualKineticJoules"], 0.)
        self.assertEqual(observed["checks"]["kineticPartitionResidualJoules"], 0.)

    def test_exact_planar_rigid_rotation_and_translation(self):
        q = square()
        velocity = np.column_stack((-2 * q[:, 1], 2 * q[:, 0], np.zeros(4))) + [2., -3., 4.]
        observed = analyze_rigid_velocity(q, velocity, [1.] * 4, normal=[0., 0., 1.])
        np.testing.assert_array_equal(observed["inertiaKgMetersSquared"], np.diag([4., 4., 8.]))
        self.assertEqual(observed["angularVelocityRadiansPerSecond"], [0., 0., 2.])
        self.assertEqual(observed["angularMomentumAboutCOMKgMetersSquaredPerSecond"], [0., 0., 16.])
        # Scale-protected binary64 energy need not round identically to the
        # integer scalar oracle; the rigid field itself is exact here.
        self.assertAlmostEqual(observed["kineticJoules"], 74., delta=math.ulp(74.))
        self.assertEqual(observed["rotationKineticJoules"], 16.)
        self.assertEqual(observed["residualKineticJoules"], 0.)
        self.assertEqual(observed["conditioning"]["inertiaEigenvalueRatio"], .5)
        self.assertEqual(observed["normalObservation"]["residualNormalKineticJoules"], 0.)

    def test_orthogonal_saddle_mode_has_independent_closed_form_partition(self):
        q = square()
        # z=x*y has zero mean and zero moments against x and y on these four
        # corners. Its normal motion is orthogonal to all six rigid modes.
        saddle = np.column_stack((np.zeros((4, 2)), q[:, 0] * q[:, 1]))
        rigid = np.column_stack((-2 * q[:, 1], 2 * q[:, 0], np.zeros(4))) + [2., -3., 4.]
        observed = analyze_rigid_velocity(q, rigid + saddle, [1.] * 4, normal=[0., 0., 1.])
        self.assertEqual(observed["translationVelocityMetersPerSecond"], [2., -3., 4.])
        self.assertEqual(observed["angularVelocityRadiansPerSecond"], [0., 0., 2.])
        np.testing.assert_array_equal(observed["rigidVelocitiesMetersPerSecond"], rigid)
        np.testing.assert_array_equal(observed["residualVelocitiesMetersPerSecond"], saddle)
        self.assertEqual(observed["kineticJoules"], 76.)
        self.assertEqual(observed["translationKineticJoules"], 58.)
        self.assertEqual(observed["rotationKineticJoules"], 16.)
        self.assertEqual(observed["residualKineticJoules"], 2.)
        self.assertEqual(observed["normalObservation"]["residualNormalKineticFraction"], 1.)
        for key, value in observed["checks"].items():
            with self.subTest(check=key):
                np.testing.assert_array_equal(value, np.zeros_like(value))

    def test_finite_rigid_pose_rotation_can_leave_an_instantaneous_residual(self):
        start = square()
        quarter_turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        end = start @ quarter_turn.T
        # These endpoints differ by an exact proper rigid rotation. Their
        # half-second backward difference includes an outward radial term at
        # the new positions; a velocity residual alone is not strain evidence.
        velocity = 2 * (end - start)
        observed = analyze_rigid_velocity(end, velocity, [1.] * 4)
        self.assertEqual(observed["angularVelocityRadiansPerSecond"], [0., 0., 2.])
        np.testing.assert_array_equal(observed["residualVelocitiesMetersPerSecond"], 2 * end)
        self.assertEqual(observed["residualKineticJoules"], 16.)
        self.assertEqual(observed["rotationKineticJoules"], 16.)
        self.assertEqual(observed["kineticJoules"], 32.)

    def test_nonuniform_masses_set_com_and_preserve_rigid_motion(self):
        q = np.array([[0., 0., 0.], [2., 0., 0.], [0., 4., 0.], [0., 0., 8.]])
        mass = np.array([1., 2., 3., 4.])
        center = np.array([.4, 1.2, 3.2])  # Direct weighted coordinate sums /10.
        omega = np.array([.5, -1., 2.])
        translation = np.array([3., 2., -4.])
        velocity = translation + np.cross(omega, q - center)
        observed = analyze_rigid_velocity(q, velocity, mass)
        np.testing.assert_allclose(observed["centerOfMassMeters"], center, rtol=0, atol=5e-16)
        np.testing.assert_allclose(observed["translationVelocityMetersPerSecond"], translation, rtol=0, atol=2e-15)
        np.testing.assert_allclose(observed["angularVelocityRadiansPerSecond"], omega, rtol=0, atol=2e-15)
        np.testing.assert_allclose(observed["residualVelocitiesMetersPerSecond"], 0., rtol=0, atol=8e-15)
        self.assertLess(abs(observed["checks"]["kineticPartitionResidualJoules"]), 3e-13)

    def test_mass_scaling_changes_energy_and_momentum_but_not_motion(self):
        q = square()
        velocity = np.array([[2., 3., 1.], [1., -2., 3.], [4., 1., -1.], [-2., 2., .5]])
        mass = np.array([1., 2., 3., 4.])
        original = analyze_rigid_velocity(q, velocity, mass)
        scaled = analyze_rigid_velocity(q, velocity, mass * 8)
        for key in ("centerOfMassMeters", "translationVelocityMetersPerSecond", "angularVelocityRadiansPerSecond",
                    "rigidVelocitiesMetersPerSecond", "residualVelocitiesMetersPerSecond"):
            np.testing.assert_array_equal(scaled[key], original[key])
        for key in ("totalMassKg", "linearMomentumKgMetersPerSecond", "inertiaKgMetersSquared",
                    "angularMomentumAboutCOMKgMetersSquaredPerSecond", "kineticJoules",
                    "translationKineticJoules", "rotationKineticJoules", "residualKineticJoules"):
            np.testing.assert_array_equal(scaled[key], np.asarray(original[key]) * 8)
        self.assertNotEqual(scaled["inputSha256"], original["inputSha256"])

    def test_proper_rotations_and_coordinate_translations_preserve_the_observation(self):
        q = square()
        velocity = np.array([[1., 3., 2.], [2., 1., -2.], [-1., 3., 2.], [4., -2., -2.]])
        mass = [1., 2., 3., 4.]
        base = analyze_rigid_velocity(q, velocity, mass, normal=[0., 0., 1.])
        axis = np.array([1., 2., 3.]) / math.sqrt(14)
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        rotation = np.eye(3) + math.sin(.73) * cross + (1 - math.cos(.73)) * cross @ cross
        self.assertAlmostEqual(np.linalg.det(rotation), 1., places=14)
        translation = np.array([3., -7., 2.])
        changed = analyze_rigid_velocity(q @ rotation.T + translation, velocity @ rotation.T,
                                        mass, normal=rotation[:, 2])
        np.testing.assert_allclose(changed["centerOfMassMeters"], rotation @ base["centerOfMassMeters"] + translation,
                                   rtol=0, atol=2e-15)
        for key in ("translationVelocityMetersPerSecond", "angularVelocityRadiansPerSecond",
                    "linearMomentumKgMetersPerSecond", "angularMomentumAboutCOMKgMetersSquaredPerSecond"):
            np.testing.assert_allclose(changed[key], rotation @ base[key], rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(changed["inertiaKgMetersSquared"], rotation @ base["inertiaKgMetersSquared"] @ rotation.T,
                                   rtol=3e-14, atol=3e-14)
        for key in ("rigidVelocitiesMetersPerSecond", "residualVelocitiesMetersPerSecond"):
            np.testing.assert_allclose(changed[key], np.asarray(base[key]) @ rotation.T, rtol=3e-14, atol=3e-14)
        for key in ("kineticJoules", "translationKineticJoules", "rotationKineticJoules", "residualKineticJoules"):
            self.assertAlmostEqual(changed[key], base[key], delta=3e-13)
        self.assertAlmostEqual(changed["normalObservation"]["residualNormalKineticJoules"],
                               base["normalObservation"]["residualNormalKineticJoules"], delta=3e-13)

    def test_scale_normalized_conditioning_admits_small_and_large_planar_support(self):
        q = square()
        velocity = np.column_stack((-q[:, 1], q[:, 0], q[:, 0] * q[:, 1]))
        base = analyze_rigid_velocity(q, velocity, [1.] * 4)
        for scale in (1e-80, 1e80):
            with self.subTest(scale=scale):
                changed = analyze_rigid_velocity(q * scale, velocity * scale, [1.] * 4)
                np.testing.assert_allclose(changed["angularVelocityRadiansPerSecond"], [0., 0., 1.], rtol=2e-15, atol=0.)
                self.assertEqual(changed["conditioning"]["inertiaEigenvalueRatio"], .5)
                for key in ("kineticJoules", "rotationKineticJoules", "residualKineticJoules"):
                    np.testing.assert_allclose(changed[key], base[key] * scale ** 2, rtol=3e-15, atol=0.)

    def test_collinear_coincident_and_ill_conditioned_support_fail_closed(self):
        # A planar, noncollinear baseline is valid before the rejection family.
        analyze_rigid_velocity(square(), np.zeros((4, 3)), [1.] * 4)
        cases = [np.zeros((3, 3)), [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
                 [[0., 0., 0.], [1., 2., 3.], [2., 4., 6.]],
                 [[0., 0., 0.], [1., 0., 0.], [2., 1e-8, 0.]]]
        for q in cases:
            with self.subTest(positions=q), self.assertRaisesRegex(ValueError, "support|inertia"):
                analyze_rigid_velocity(q, np.zeros((3, 3)), [1.] * 3)

    def test_input_hash_is_independent_and_no_inputs_or_returned_values_alias(self):
        q = square()
        v = np.column_stack((q[:, 1], q[:, 0], q[:, 0] * q[:, 1]))
        mass = np.ones(4)
        normal = np.array([0., 0., 1.])
        before = [array.copy() for array in (q, v, mass, normal)]
        result = analyze_rigid_velocity(q, v, mass, normal=normal)
        payload = {"profile": "mass-weighted-rigid-velocity-observation-v1", "positionsMeters": q.tolist(),
                   "velocitiesMetersPerSecond": v.tolist(), "massesKg": mass.tolist(), "normal": normal.tolist()}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(result["inputSha256"], expected)
        self.assertIs(result["accepted"], False)
        self.assertNotIn("verified", result)
        result["residualVelocitiesMetersPerSecond"][0][0] = 123.
        result["normalObservation"]["normal"][2] = -1.
        for value, snapshot in zip((q, v, mass, normal), before):
            np.testing.assert_array_equal(value, snapshot)
        fresh = analyze_rigid_velocity(q, v, mass, normal=normal)
        self.assertEqual(fresh["inputSha256"], expected)
        self.assertNotEqual(fresh["residualVelocitiesMetersPerSecond"][0][0], 123.)
        flipped = analyze_rigid_velocity(q, v, mass, normal=-normal)
        self.assertNotEqual(flipped["inputSha256"], expected)
        self.assertEqual(flipped["normalObservation"]["residualNormalKineticJoules"],
                         fresh["normalObservation"]["residualNormalKineticJoules"])

    def test_unit_normal_admission_and_projection_do_not_silently_rescale_energy(self):
        q = square()
        v = np.column_stack((np.zeros((4, 2)), q[:, 0] * q[:, 1]))
        base = analyze_rigid_velocity(q, v, [1.] * 4, normal=[0., 0., 1.])
        admitted = [0., 0., math.nextafter(1., 2.)]
        near = analyze_rigid_velocity(q, v, [1.] * 4, normal=admitted)
        self.assertEqual(near["normalObservation"]["normal"], admitted)
        self.assertAlmostEqual(near["normalObservation"]["residualNormalKineticJoules"],
                               base["normalObservation"]["residualNormalKineticJoules"], delta=1e-15)
        for normal in ([0., 0., 0.], [0., 0., 2.], [0., 0., 1. + 1e-10], [0., 0.], [False, 0., 1.], [0., math.nan, 1.]):
            with self.subTest(normal=normal), self.assertRaises(ValueError):
                analyze_rigid_velocity(q, v, [1.] * 4, normal=normal)

    def test_strict_raw_types_shapes_finiteness_and_positive_mass_admission(self):
        q, v, mass = square().tolist(), np.zeros((4, 3)).tolist(), [1.] * 4
        analyze_rigid_velocity(q, v, mass)
        attacks = [lambda x: x[0][0].__setitem__(0, False), lambda x: x[1][0].__setitem__(0, True),
                   lambda x: x[2].__setitem__(0, False), lambda x: x[0][0].__setitem__(0, float("nan")),
                   lambda x: x[1][0].__setitem__(0, float("inf")), lambda x: x[2].__setitem__(0, float("inf")),
                   lambda x: x[2].__setitem__(0, 0.), lambda x: x[2].__setitem__(0, -1.),
                   lambda x: x[0][0].__setitem__(0, 1e101), lambda x: x[1][0].__setitem__(0, 1e101),
                   lambda x: x[2].__setitem__(0, 1e101), lambda x: x[0][0].__setitem__(0, np.int64(2 ** 53 + 1)),
                   lambda x: x[0][0].__setitem__(0, "1"), lambda x: x[0][0].append(0.),
                   lambda x: x[0].pop(), lambda x: x[1].pop(), lambda x: x[2].pop(),
                   lambda x: x.__setitem__(0, np.array(1.)), lambda x: x.__setitem__(2, 1.)]
        if np.dtype(np.longdouble).itemsize > 8:
            attacks.append(lambda x: x[0][0].__setitem__(0, np.longdouble(1.)))
        for number, attack in enumerate(attacks):
            values = copy.deepcopy([q, v, mass])
            attack(values)
            with self.subTest(attack=number), self.assertRaises(ValueError):
                analyze_rigid_velocity(*values)
        with self.assertRaises(ValueError):
            analyze_rigid_velocity([[0., 0., 0.]] * 25001, [], [])
        with self.assertRaises(ValueError):
            analyze_rigid_velocity([[0., 0., 0.]] * 2, [], [])

    def test_unrepresentable_mass_ratios_inertia_and_energy_reject(self):
        q = square()
        analyze_rigid_velocity(q, np.zeros((4, 3)), [1.] * 4)
        with self.assertRaisesRegex(ValueError, "Mass ratios"):
            analyze_rigid_velocity(q, np.zeros((4, 3)), [1e100, math.ulp(0.), 1., 1.])
        with self.assertRaisesRegex(ValueError, "inertia"):
            analyze_rigid_velocity(q * 1e-200, np.zeros((4, 3)), [1.] * 4)
        with self.assertRaisesRegex(ValueError, "kinetic energy"):
            analyze_rigid_velocity(q, np.full((4, 3), 1e-200), [1.] * 4)

    def test_subnormal_mass_normalization_cannot_silently_lose_finite_energy(self):
        q = square()
        masses = [4 * math.ulp(0.), 1., 1., 1.]
        velocities = [[1e100, 0., 0.]] + [[1e-62, 0., 0.]] * 3
        oracle = sum((Fraction(m) * Fraction(v[0]) ** 2 / 2 for m, v in zip(masses, velocities)), Fraction())
        self.assertGreater(float(oracle), 1.13e-123)
        self.assertLess(float(oracle), 1.14e-123)
        # Earlier admission returned energy 35% low and a misleading zero
        # partition residual. This dynamic range must explicitly fail closed.
        with self.assertRaisesRegex(ValueError, "Mass ratios are subnormal"):
            analyze_rigid_velocity(q, velocities, masses)

    def test_tiny_majority_velocity_is_not_lost_by_an_extreme_first_sample(self):
        masses = [1e-200, 1., 1., 1.]
        velocities = [[1e100, 0., 0.]] + [[1e-62, 0., 0.]] * 3
        result = analyze_rigid_velocity(square(), velocities, masses)
        expected = float(sum((Fraction(m) * Fraction(v[0]) for m, v in zip(masses, velocities)), Fraction())
                         / sum(map(Fraction, masses), Fraction()))
        np.testing.assert_allclose(result["translationVelocityMetersPerSecond"][0], expected, rtol=2e-15, atol=0.)
        self.assertGreater(result["translationKineticJoules"], 0.)
        oracle = float(sum((Fraction(m) * Fraction(v[0]) ** 2 / 2 for m, v in zip(masses, velocities)), Fraction()))
        np.testing.assert_allclose(result["kineticJoules"], oracle, rtol=2e-15, atol=0.)

    def test_speed_and_residual_speed_preserve_tiny_representable_magnitudes(self):
        q = square()
        translation = np.tile([1e-200, 0., 0.], (4, 1))
        moved = analyze_rigid_velocity(q, translation, [1e100] * 4)
        self.assertEqual(moved["maximumSpeedMetersPerSecond"], 1e-200)
        self.assertEqual(moved["translationVelocityMetersPerSecond"], [1e-200, 0., 0.])
        np.testing.assert_allclose(moved["kineticJoules"], 2e-300, rtol=2e-15, atol=0.)
        saddle = np.column_stack((np.zeros((4, 2)), (q[:, 0] * q[:, 1]) * 1e-200))
        bent = analyze_rigid_velocity(q, saddle, [1e100] * 4, normal=[0., 0., 1.])
        self.assertEqual(bent["maximumSpeedMetersPerSecond"], 1e-200)
        self.assertEqual(bent["maximumResidualSpeedMetersPerSecond"], 1e-200)
        np.testing.assert_allclose(bent["residualKineticJoules"], 2e-300, rtol=2e-15, atol=0.)


if __name__ == "__main__":
    unittest.main()
