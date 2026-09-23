"""Independent sampling and admission checks; no sewing execution is implied."""

import copy
from fractions import Fraction
import math
import unittest
from unittest import mock

import numpy as np

from solver_sewing_activation_schedule import MAX_KNOTS, MAX_ROWS, PROFILE, SewingActivationSchedule


def recipe():
    return {"profile": PROFILE, "rowIds": ["pending", "held", "part-active", "later"], "knots": [
        {"fraction": 0., "activation": [0., 1., .25, 0.]},
        {"fraction": .25, "activation": [0., 1., .5, 0.]},
        {"fraction": .75, "activation": [0., 1., 1., .5]},
        {"fraction": 1., "activation": [0., 1., 1., 1.]}]}


class SewingActivationScheduleTests(unittest.TestCase):
    def test_exact_independent_curves_and_out_of_order_retry_sampling(self):
        source = recipe()
        schedule = SewingActivationSchedule(source, 4, row_ids=source["rowIds"])
        # Closed-form rational curves, independent of the interpolation loop.
        fractions = [Fraction(tick, 128) for tick in range(129)]
        fractions = fractions[::2] + list(reversed(fractions[1::2])) + [Fraction(3, 8)] * 3
        for fraction in fractions:
            expected = [0., 1., float(min(Fraction(1), Fraction(1, 4) + fraction)),
                        float(max(Fraction(0), fraction - Fraction(1, 4)) if fraction <= Fraction(3, 4)
                              else 2 * fraction - 1)]
            with self.subTest(fraction=fraction):
                np.testing.assert_array_equal(schedule.parameters(fraction), expected)
        for knot in source["knots"]:
            np.testing.assert_array_equal(schedule.parameters(knot["fraction"]), knot["activation"])

    def test_capture_and_returned_arrays_are_immutable_and_independent(self):
        source = recipe()
        ids = list(source["rowIds"])
        source["knots"][1]["activation"] = np.array(source["knots"][1]["activation"])
        schedule = SewingActivationSchedule(source, 4, row_ids=ids)
        expected = schedule.parameters(Fraction(3, 8))
        source["knots"][1]["activation"][:] = 99
        source["knots"][-1]["fraction"] = .8
        source["rowIds"][0] = ids[0] = "changed"
        first, second = schedule.parameters(Fraction(3, 8)), schedule.parameters(Fraction(3, 8))
        first[:] = -1
        np.testing.assert_array_equal(second, expected)
        for fraction in (0, 1):
            output = schedule.parameters(fraction)
            output[:] = 99
            self.assertTrue(np.all(schedule.parameters(fraction) <= 1))
        self.assertEqual(schedule.row_ids, ("pending", "held", "part-active", "later"))
        for field in ("_row_ids", "_fractions", "_activation", "_subdivisions", "row_ids", "unexpected"):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(schedule, field, None)
            with self.subTest(delete=field), self.assertRaises(AttributeError):
                delattr(schedule, field)
        with self.assertRaises(ValueError):
            schedule._activation[0, 0] = .5
        with self.assertRaises(ValueError):
            schedule._activation.flags.writeable = True
        with self.assertRaises(ValueError):
            schedule._activation.base.flags.writeable = True

    def test_inactive_partial_and_fully_held_rows_remain_exact(self):
        values = [0., float(np.nextafter(0., 1.)), .1, math.nextafter(1., 0.), 1.]
        ids = [f"row-{index}" for index in range(len(values))]
        source = {"profile": PROFILE, "rowIds": ids, "knots": [
            {"fraction": fraction, "activation": values.copy()} for fraction in (0., .0625, 1.)]}
        schedule = SewingActivationSchedule(source, 16, row_ids=tuple(ids))
        for tick in range(257):
            np.testing.assert_array_equal(schedule.parameters(Fraction(tick, 256)), values)
        self.assertEqual(schedule.parameters(Fraction(1, 2 ** 40))[-1], 1.)

    def test_positive_interior_activation_cannot_underflow_to_pending(self):
        smallest = float(np.nextafter(0., 1.))
        source = {"profile": PROFILE, "rowIds": ["pending", "engaging", "held"], "knots": [
            {"fraction": 0., "activation": [0., 0., smallest]},
            {"fraction": 1., "activation": [0., smallest, smallest]}]}
        schedule = SewingActivationSchedule(source, 1, row_ids=source["rowIds"])
        self.assertGreater(Fraction(smallest) / 2, 0)
        for fraction in (Fraction(1, 2), Fraction(1, 2 ** 40)):
            # Admission does not depend on the caller's NumPy underflow mode.
            with self.subTest(fraction=fraction), np.errstate(under="raise"), \
                    self.assertRaisesRegex(ValueError, "underflows to inactive"):
                schedule.parameters(fraction)
        np.testing.assert_array_equal(schedule.parameters(0), [0., 0., smallest])
        np.testing.assert_array_equal(schedule.parameters(1), [0., smallest, smallest])
        source["knots"][-1]["activation"][1] = 2 * smallest
        representable = SewingActivationSchedule(source, 1, row_ids=source["rowIds"])
        np.testing.assert_array_equal(representable.parameters(.5), [0., smallest, smallest])

    def test_bad_schema_ordered_identities_and_raw_subdivisions_reject(self):
        source = recipe()
        malformed = [None, {}, {**source, "profile": "unknown"}, {**source, "accepted": False},
                     {**source, "rowIds": tuple(source["rowIds"])}, {**source, "rowIds": source["rowIds"][::-1]},
                     {**source, "rowIds": ["pending"] * 4}, {**source, "knots": tuple(source["knots"])},
                     {**source, "knots": []}, {**source, "knots": source["knots"][:1]}]
        for key in source:
            malformed.append({name: value for name, value in source.items() if name != key})
        for bad in malformed:
            with self.subTest(recipe=bad), self.assertRaises(ValueError):
                SewingActivationSchedule(bad, 4, row_ids=source["rowIds"])
        for ids in (None, [], "row", [True], [""], ["white space"], ["-leading"], ["x" * 129], ["é"],
                    ["repeated", "repeated"], [1], np.array(["pending"])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                SewingActivationSchedule(source, 4, row_ids=ids)
        for subdivisions in (True, False, np.int64(4), 4., 0, -1, 3, 8192, None):
            with self.subTest(subdivisions=subdivisions), self.assertRaises(ValueError):
                SewingActivationSchedule(source, subdivisions, row_ids=source["rowIds"])

    def test_bad_activation_raw_types_shape_finiteness_and_release_reject(self):
        bad_values = [None, 0., [], [0.] * 3, [0.] * 5, [[0.]] * 4, [np.array(0.)] * 4,
                      [False, 1., .5, 0.], [0., np.bool_(True), .5, 0.], np.array([False] * 4),
                      np.array([False, 1., .5, 0.], dtype=object), ["0", "1", ".5", "0"],
                      [0j, 1., .5, 0.], [float("nan"), 1., .5, 0.], [float("inf"), 1., .5, 0.],
                      [-.01, 1., .5, 0.], [0., math.nextafter(1., math.inf), .5, 0.],
                      [0., math.nextafter(1., 0.), .5, 0.], [0., 1., math.nextafter(.25, 0.), 0.]]
        for values in bad_values:
            source = recipe()
            source["knots"][1]["activation"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                SewingActivationSchedule(source, 4, row_ids=source["rowIds"])

    def test_bad_knots_and_sample_fractions_reject(self):
        for fraction in (True, np.bool_(False), "0.25", None, float("nan"), float("inf"), -.25, 1.25,
                         Fraction(1, 3), Fraction(1, 2 ** 41), .125, 0., 1.):
            source = recipe()
            source["knots"][1]["fraction"] = fraction
            with self.subTest(knot=fraction), self.assertRaises(ValueError):
                SewingActivationSchedule(source, 4, row_ids=source["rowIds"])
        for position, value in ((0, .25), (-1, .75)):
            source = recipe()
            source["knots"][position]["fraction"] = value
            with self.subTest(endpoint=position), self.assertRaises(ValueError):
                SewingActivationSchedule(source, 4, row_ids=source["rowIds"])
        for bad in ({"fraction": .25}, {"activation": [0., 1., .5, 0.]},
                    {"fraction": .25, "activation": [0., 1., .5, 0.], "completed": True}, []):
            source = recipe()
            source["knots"][1] = bad
            with self.subTest(knot=bad), self.assertRaises(ValueError):
                SewingActivationSchedule(source, 4, row_ids=source["rowIds"])
        source = recipe()
        schedule = SewingActivationSchedule(source, 4, row_ids=source["rowIds"])
        for fraction in (True, np.bool_(False), "0.25", None, float("nan"), float("inf"), -.25, 1.25,
                         Fraction(1, 3), Fraction(1, 2 ** 41)):
            with self.subTest(sample=fraction), self.assertRaises(ValueError):
                schedule.parameters(fraction)

    def test_maximum_budgets_and_deep_refinement_are_supported(self):
        ids = [f"seam/member:1/fraction:{index}" for index in range(MAX_ROWS)]
        source = {"profile": PROFILE, "rowIds": ids, "knots": [
            {"fraction": index / 64, "activation": [index / 64] * MAX_ROWS} for index in range(MAX_KNOTS)]}
        schedule = SewingActivationSchedule(source, 4096, row_ids=ids)
        for fraction in (0, Fraction(1, 2 ** 40), Fraction(33, 128), Fraction(2 ** 40 - 1, 2 ** 40), 1):
            actual = schedule.parameters(fraction)
            self.assertEqual(actual.shape, (MAX_ROWS,))
            np.testing.assert_array_equal(actual, float(fraction))

    def test_budgets_and_shapes_reject_before_array_coercion(self):
        source = recipe()
        cases = [(source, [f"row-{index}" for index in range(MAX_ROWS + 1)]),
                 ({**source, "knots": source["knots"] * 17}, source["rowIds"])]
        for bad_shape in ([[0.] * 10000] * 4, [0.] * (MAX_ROWS + 1), np.zeros((4, 2))):
            changed = copy.deepcopy(source)
            changed["knots"][0]["activation"] = bad_shape
            cases.append((changed, source["rowIds"]))
        with mock.patch("solver_sewing_activation_schedule.np.asarray", side_effect=AssertionError("Unexpected coercion")):
            for changed, ids in cases:
                with self.subTest(rows=len(ids)), self.assertRaises(ValueError):
                    SewingActivationSchedule(changed, 4, row_ids=ids)


if __name__ == "__main__":
    unittest.main()
