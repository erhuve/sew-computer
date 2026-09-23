"""Independent adversarial controls for the scalar sampled-row path lemma."""

import copy
from fractions import Fraction
import math
import unittest

import numpy as np

from solver_sewing_sweep import verify_control_interval, verify_distance_sewing_sweep_exact


class SewingSweepReviewTests(unittest.TestCase):
    @staticmethod
    def check(start, end=None, **changes):
        options = dict(row_ids=["row:independent"], initial_targets=[1.],
                       final_targets=[1.], initial_activation=[1.],
                       final_activation=[1.], tolerance_m=.25)
        rows = changes.pop("rows", [[(0, 1.), (1, -1.)]])
        options.update(changes)
        return verify_distance_sewing_sweep_exact(start, start if end is None else end,
                                                  rows, **options)

    def test_endpoint_matching_chord_fails_without_material_collapse(self):
        # A rigid translation of a second small triangle can realize these
        # anchors while every triangle remains unchanged and far separated.
        start = [[1., 0., 0.], [0., 0., 0.]]
        end = [[0., 1., 0.], [0., 0., 0.]]
        self.assertEqual(math.dist(*start), 1.)
        self.assertEqual(math.dist(*end), 1.)
        self.assertEqual(Fraction(1, 2) - Fraction(3, 4) ** 2, -Fraction(1, 16))
        with self.assertRaises(ValueError):
            self.check(start, end)

    def test_changing_positive_target_has_interior_nonmidpoint_failure(self):
        # For v=(1-s, 2s, 0), t=1+s and eps=1/4, the lower
        # squared margin is minimized at s=7/16 with value -21/64.
        s = Fraction(7, 16)
        self.assertEqual((1-s)**2 + 4*s*s - (1+s-Fraction(1, 4))**2,
                         -Fraction(21, 64))
        with self.assertRaises(ValueError):
            self.check([[1., 0., 0.], [0., 0., 0.]],
                       [[0., 2., 0.], [0., 0., 0.]], final_targets=[2.])

    def test_negative_lower_threshold_is_not_squared_as_a_constraint(self):
        result = self.check([[.0625, 0., 0.], [0., 0., 0.]],
                            initial_targets=[.125], final_targets=[.5625],
                            tolerance_m=.5)
        self.assertTrue(result["verified"])

    def test_closed_error_boundary_and_one_ulp_outside(self):
        for distance in (.75, 1.25):
            with self.subTest(distance=distance):
                self.assertTrue(self.check([[distance, 0., 0.], [0., 0., 0.]])["verified"])
        for distance in (math.nextafter(.75, 0.), math.nextafter(1.25, math.inf)):
            with self.subTest(distance=distance):
                with self.assertRaises(ValueError):
                    self.check([[distance, 0., 0.], [0., 0., 0.]])

    def test_collapse_is_rejected_even_when_error_tolerance_would_allow_it(self):
        # Zero is reached at 1/4, not at a midpoint sampling location.
        with self.assertRaises(ValueError):
            self.check([[.25, 0., 0.], [0., 0., 0.]],
                       [[-.75, 0., 0.], [0., 0., 0.]], tolerance_m=2.)

    def test_pending_collapse_is_skipped_but_engagement_is_conservative(self):
        state = [[0., 0., 0.], [0., 0., 0.]]
        result = self.check(state, initial_activation=[0.], final_activation=[0.])
        self.assertTrue(result["verified"])
        self.assertEqual(result["rows"][0]["status"], "pending")
        with self.assertRaises(ValueError):
            self.check(state, [[1., 0., 0.], [0., 0., 0.]],
                       initial_activation=[0.], final_activation=[1.], tolerance_m=2.)

    def test_positive_tiny_activation_does_not_hide_unweighted_error(self):
        with self.assertRaises(ValueError):
            self.check([[2., 0., 0.], [0., 0., 0.]],
                       initial_activation=[math.ulp(0.)], final_activation=[math.ulp(0.)])

    def test_original_products_preserve_anchor_lost_by_rounded_accumulation(self):
        # 0.5*1 + 0.5*nextafter(1,+inf) - 1 = 2^-53 exactly,
        # while sequential binary64 sums erase the nonzero anchor.
        state = [[1., 0., 0.], [math.nextafter(1., math.inf), 0., 0.], [1., 0., 0.]]
        target = 2.**-53
        self.assertEqual(.5*state[0][0] + .5*state[1][0] - state[2][0], 0.)
        result = self.check(state, rows=[[(0, .5), (1, .5), (2, -1.)]],
                            initial_targets=[target], final_targets=[target], tolerance_m=math.ulp(0.))
        self.assertTrue(result["verified"])
        self.assertEqual(result["rows"][0]["status"], "checked")

    def test_numpy_integer_coercion_cannot_erase_a_unit_target_error(self):
        # float == np.int64 would itself round this integer and report equal.
        # Admission must compare the integral values without NumPy promotion.
        state = [[float(2**53), 0., 0.], [0., 0., 0.]]
        for integer in (np.int64(2**53+1), np.uint64(2**53+1)):
            with self.subTest(integer_type=type(integer).__name__):
                self.assertEqual(float(integer), float(2**53))
                self.assertNotEqual(int(float(integer)), int(integer))
                with self.assertRaises(ValueError):
                    self.check(state, initial_targets=[integer], final_targets=[integer])

    def test_certificate_has_exact_extremum_not_a_sampled_bound(self):
        result = self.check([[1., 0., 0.], [0., 0., 0.]],
                            [[0., 1., 0.], [0., 0., 0.]], tolerance_m=.3)
        def rational(value):
            self.assertIs(type(value["numerator"]), str)
            self.assertIs(type(value["denominator"]), str)
            return Fraction(int(value["numerator"]), int(value["denominator"]))
        row = result["rows"][0]
        self.assertEqual(rational(row["distanceSquaredMinimum"]["value"]), Fraction(1, 2))
        self.assertEqual(rational(row["distanceSquaredMinimum"]["atLocalFraction"]), Fraction(1, 2))
        self.assertEqual(rational(row["lowerErrorMarginMinimum"]["value"]),
                         Fraction(1, 2) - (1-Fraction(.3))**2)
        self.assertEqual(rational(row["lowerErrorMarginMinimum"]["atLocalFraction"]), Fraction(1, 2))

    def test_exact_binary_rescaling_and_axis_permutation_preserve_decision(self):
        for exponent in (-400, -20, 0, 6):
            scale = 2.**exponent
            for axis in range(3):
                state = np.zeros((2, 3))
                state[0, axis] = 1.125*scale
                with self.subTest(exponent=exponent, axis=axis):
                    result = self.check(state, initial_targets=[scale], final_targets=[scale],
                                        tolerance_m=.125*scale)
                    self.assertTrue(result["verified"])

    def test_zero_motion_changing_targets_checks_both_ends(self):
        with self.assertRaises(ValueError):
            self.check([[1., 0., 0.], [0., 0., 0.]],
                       initial_targets=[.5], final_targets=[1.])

    def test_raw_malformed_inputs_reject_even_when_pending(self):
        state = [[1., 0., 0.], [0., 0., 0.]]
        attacks = [dict(initial_targets=[True]), dict(final_targets=[0.]),
                   dict(initial_activation=[False]), dict(final_activation=[float("nan")]),
                   dict(tolerance_m=True), dict(tolerance_m=float("inf")),
                   dict(rows=[[(0, 1.), (0, -1.)]]), dict(rows=[[(True, 1.), (1, -1.)]]),
                   dict(row_ids=[True])]
        for attack in attacks:
            with self.subTest(attack=attack):
                options = dict(initial_activation=[0.], final_activation=[0.])
                options.update(attack)
                with self.assertRaises(ValueError):
                    self.check(state, **options)

    def test_result_and_input_mutation_do_not_change_subsequent_proof(self):
        start = [[1., 0., 0.], [0., 0., 0.]]
        options = dict(rows=[[(0, 1.), (1, -1.)]], initial_targets=[1.], final_targets=[1.])
        saved_start, saved_options = copy.deepcopy(start), copy.deepcopy(options)
        first = self.check(start, **options)
        expected = copy.deepcopy(first)
        self.assertEqual(start, saved_start)
        self.assertEqual(options, saved_options)
        first["rows"][0]["status"] = "forged"
        self.assertEqual(self.check(start, **options), expected)

    def test_both_schedule_knot_sets_are_checked_with_exact_boundaries(self):
        knots = [0., .5, 1., Fraction(1, 4), Fraction(3, 4), .5]
        for left, right in ((0., .25), (.25, .5), (.5, .75), (.75, 1.)):
            self.assertEqual(verify_control_interval(left, right, knots),
                             (Fraction(left), Fraction(right)))
        for left, right in ((0., .5), (.25, .75), (.5, 1.)):
            with self.subTest(left=left, right=right):
                with self.assertRaises(ValueError):
                    verify_control_interval(left, right, knots)

    def test_control_interval_cannot_silently_round_or_reverse_fractions(self):
        for left, right, knots in ((False, .5, []), (.5, .5, []), (1., 0., []),
                                   (0., 1., [True]), (0., Fraction(1, 3), []),
                                   (0., Fraction(1, 2**41), []), (0., 1., [0.]*131)):
            with self.subTest(left=left, right=right, knots=knots):
                with self.assertRaises(ValueError):
                    verify_control_interval(left, right, knots)


if __name__ == "__main__":
    unittest.main()
