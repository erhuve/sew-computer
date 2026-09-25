"""Independent binary64 coordinate admission boundaries, using stdlib only."""
from fractions import Fraction
import math
import struct
import sys
import unittest

from solver_contact_work import _distance_choices, _distance_choice_integers


class FloatSubclass(float):
    def as_integer_ratio(self):
        raise AssertionError("Non-builtin float must be rejected before conversion")


class NumericTripwire:
    def __float__(self):
        raise AssertionError("Coordinate must not be coerced")

    def __hash__(self):
        raise AssertionError("Invalid coordinate must not reach memo hashing")

    def __eq__(self, other):
        raise AssertionError("Invalid coordinate must not reach memo equality")


class CoordinateAdmissionTests(unittest.TestCase):
    def setUp(self):
        _distance_choice_integers.cache_clear()

    def tearDown(self):
        _distance_choice_integers.cache_clear()

    def assert_cold_and_warm_distance(self, points, expected):
        self.assertEqual(_distance_choices('vv', points), {'P_P': expected})
        self.assertEqual(_distance_choices('vv', points), {'P_P': expected})
        info = _distance_choice_integers.cache_info()
        self.assertEqual((info.misses, info.hits, info.currsize), (1, 1, 1))

    def test_opposite_maximum_finite_and_minimum_subnormal_exact_distance(self):
        largest = sys.float_info.max
        smallest = math.ulp(0.0)
        points = ((largest, smallest, -0.0), (-largest, -smallest, 0.0))
        # IEEE binary64 extrema derived as integers, without producer conversion.
        largest_integer = ((1 << 53) - 1) << 971
        expected = 4 * (Fraction(largest_integer**2) + Fraction(1, 1 << 2148))
        self.assert_cold_and_warm_distance(points, expected)

    def test_subnormal_and_normal_boundary_differences_remain_exact(self):
        unit = math.ulp(0.0)
        normal = sys.float_info.min
        points = ((unit, math.nextafter(unit, math.inf), normal),
                  (0.0, 0.0, math.nextafter(normal, 0.0)))
        # Coordinate differences are one, two and one minimum subnormal units.
        self.assert_cold_and_warm_distance(points, Fraction(6, 1 << 2148))

    def test_signed_zeros_keep_existing_equal_key_and_input_signs(self):
        points = ((0.0, -0.0, 0.0), (-0.0, 0.0, -0.0))
        flipped = tuple(tuple(-value for value in row) for row in points)
        before = tuple(struct.pack('<d', value) for row in points for value in row)
        self.assertEqual(_distance_choices('vv', points), {'P_P': Fraction(0)})
        value = _distance_choices('vv', flipped)['P_P']
        self.assertEqual((value.numerator, value.denominator), (0, 1))
        self.assertEqual(tuple(struct.pack('<d', value) for row in points for value in row), before)
        info = _distance_choice_integers.cache_info()
        self.assertEqual((info.misses, info.hits, info.currsize), (1, 1, 1))

    def test_invalid_equal_value_types_reject_before_lookup_cold_and_warm(self):
        from decimal import Decimal
        valid = ((0.0, 0.0, 0.0), (1.0, 2.0, 2.0))
        values = (False, 0, Fraction(0), Decimal(0), FloatSubclass(0.0),
                  NumericTripwire(), 0j, None, '0')
        for warm in (False, True):
            for value in values:
                with self.subTest(warm=warm, coordinate_type=type(value).__name__):
                    _distance_choice_integers.cache_clear()
                    if warm:
                        _distance_choices('vv', valid)
                    before = _distance_choice_integers.cache_info()
                    with self.assertRaisesRegex(ValueError, '^Finite binary64 input required$'):
                        _distance_choices('vv', ((value, 0.0, 0.0), valid[1]))
                    self.assertEqual(_distance_choice_integers.cache_info(), before)

    def test_nonfinite_coordinates_reject_cold_and_warm_without_lookup(self):
        valid = ((0.0, 0.0, 0.0), (1.0, 2.0, 2.0))
        for warm in (False, True):
            for value in (math.inf, -math.inf, math.nan, -math.nan):
                with self.subTest(warm=warm, value=value):
                    _distance_choice_integers.cache_clear()
                    if warm:
                        _distance_choices('vv', valid)
                    before = _distance_choice_integers.cache_info()
                    with self.assertRaisesRegex(ValueError, '^Finite binary64 input required$'):
                        _distance_choices('vv', (valid[0], (1.0, 2.0, value)))
                    self.assertEqual(_distance_choice_integers.cache_info(), before)

    def test_invalid_last_coordinate_precedes_degenerate_geometry(self):
        warm_points = ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        for warm in (False, True):
            for value in (False, FloatSubclass(0.0), math.inf):
                with self.subTest(warm=warm, coordinate_type=type(value).__name__):
                    _distance_choice_integers.cache_clear()
                    if warm:
                        _distance_choices('ev', warm_points)
                    before = _distance_choice_integers.cache_info()
                    # If the last value were admitted as zero, this edge degenerates.
                    invalid = (warm_points[0], warm_points[1], (0.0, 0.0, value))
                    with self.assertRaisesRegex(ValueError, '^Finite binary64 input required$'):
                        _distance_choices('ev', invalid)
                    self.assertEqual(_distance_choice_integers.cache_info(), before)

    def test_shape_admission_still_precedes_coordinate_validation(self):
        good = ((0.0, 0.0, 0.0), (1.0, 2.0, 2.0))
        for warm in (False, True):
            with self.subTest(warm=warm):
                _distance_choice_integers.cache_clear()
                if warm:
                    _distance_choices('vv', good)
                before = _distance_choice_integers.cache_info()
                with self.assertRaisesRegex(ValueError, '^Supported ordered contact stencil required$'):
                    _distance_choices('vv', ((NumericTripwire(), 0.0, 0.0),))
                with self.assertRaisesRegex(ValueError, '^Three-dimensional immutable contact coordinates required$'):
                    _distance_choices('vv', ((NumericTripwire(), 0.0, 0.0), (0.0, 0.0)))
                self.assertEqual(_distance_choice_integers.cache_info(), before)

    def test_nonstandard_kind_keeps_uncached_extrema_and_strict_admission(self):
        class Kind(str):
            pass
        unit = math.ulp(0.0)
        points = ((unit, -unit, 0.0), (-unit, unit, -0.0))
        expected = {'P_P': Fraction(8, 1 << 2148)}
        for warm in (False, True):
            with self.subTest(warm=warm):
                _distance_choice_integers.cache_clear()
                if warm:
                    _distance_choices('vv', points)
                before = _distance_choice_integers.cache_info()
                self.assertEqual(_distance_choices(Kind('vv'), points), expected)
                self.assertEqual(_distance_choices(Kind('vv'), points), expected)
                for value in (FloatSubclass(unit), math.inf):
                    with self.assertRaisesRegex(ValueError, '^Finite binary64 input required$'):
                        _distance_choices(Kind('vv'), ((value, -unit, 0.0), points[1]))
                self.assertEqual(_distance_choice_integers.cache_info(), before)


if __name__ == '__main__':
    unittest.main()
