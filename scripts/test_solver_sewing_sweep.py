"""Adversarial exact temporal bounds, independent of numerical cloth solvers."""

import ast
import copy
from fractions import Fraction
import json
from pathlib import Path
import unittest

import numpy as np

import solver_sewing_sweep as sweep


def rational(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def check(start, end=None, **changes):
    options = dict(rows=[((0, 1.), (1, -1.))], row_ids=["source:row-0"],
                   initial_targets=[1.], final_targets=[1.], initial_activation=[1.],
                   final_activation=[1.], tolerance_m=.25)
    options.update(changes)
    return sweep.verify_distance_sewing_sweep_exact(start, start if end is None else end, **options)


class SewingSweepTests(unittest.TestCase):
    def test_no_captured_solver_dependency(self):
        tree = ast.parse(Path(sweep.__file__).read_text())
        imports = {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names)
        self.assertLessEqual(imports, {"fractions", "hashlib", "json", "math", "numpy"})

    def test_held_endpoint_success_does_not_hide_interior_compression(self):
        first = [[-3., 4., 0.], [0., 0., 0.]]
        last = [[3., 4., 0.], [0., 0., 0.]]
        self.assertEqual(np.linalg.norm(first[0]), 5.)
        self.assertEqual(np.linalg.norm(last[0]), 5.)
        with self.assertRaisesRegex(ValueError, "lower scalar-distance"):
            check(first, last, initial_targets=[5.], final_targets=[5.], tolerance_m=.5)

    def test_nondyadic_interior_witness_and_exact_tolerance_boundary(self):
        first, last = [[-1., 1., 0.], [0., 0., 0.]], [[2., 1., 0.], [0., 0., 0.]]
        report = check(first, last, initial_targets=[2.], final_targets=[2.], tolerance_m=1.)
        row = report["rows"][0]
        self.assertEqual(rational(row["distanceSquaredMinimum"]["value"]), 1)
        self.assertEqual(rational(row["distanceSquaredMinimum"]["atLocalFraction"]), Fraction(1, 3))
        self.assertEqual(rational(row["lowerErrorMarginMinimum"]["value"]), 0)
        self.assertEqual(rational(row["lowerErrorMarginMinimum"]["atLocalFraction"]), Fraction(1, 3))
        self.assertEqual(rational(row["upperErrorMarginMinimum"]["value"]), 4)
        self.assertTrue(report["verified"])
        self.assertFalse(report["accepted"])
        with self.assertRaises(ValueError):
            check(first, last, initial_targets=[2.], final_targets=[2.], tolerance_m=np.nextafter(1., 0.))
        check(first, last, initial_targets=[2.], final_targets=[2.], tolerance_m=np.nextafter(1., 2.))

    def test_changing_targets_and_signed_lower_threshold_intervals(self):
        first, last = [[1., 0., 0.], [0., 0., 0.]], [[0., 2., 0.], [0., 0., 0.]]
        with self.assertRaisesRegex(ValueError, "lower scalar-distance"):
            check(first, last, initial_targets=[1.], final_targets=[2.], tolerance_m=.25)
        report = check(first, last, initial_targets=[1.], final_targets=[2.], tolerance_m=.5)
        margin = report["rows"][0]["lowerErrorMarginMinimum"]
        self.assertEqual(rational(margin["value"]), Fraction(3, 16))
        self.assertEqual(rational(margin["atLocalFraction"]), Fraction(3, 8))
        positions = [[.25, 0., 0.], [0., 0., 0.]]
        for before, after, interval in ((.25, 1.25, (Fraction(3, 4), Fraction(1))),
                                         (1.25, .25, (Fraction(0), Fraction(1, 4)))):
            report = check(positions, initial_targets=[before], final_targets=[after], tolerance_m=1.)
            margin = report["rows"][0]["lowerErrorMarginMinimum"]
            self.assertEqual(tuple(map(rational, margin["interval"])), interval)
            self.assertEqual(rational(margin["value"]), 0)
        report = check(positions, initial_targets=[.125], final_targets=[.25], tolerance_m=1.)
        self.assertIsNone(report["rows"][0]["lowerErrorMarginMinimum"])

    def test_active_closed_path_noncoincidence_is_separate_and_conservative(self):
        root = 2. ** -40
        first, last = [[-root, 0., 0.], [0., 0., 0.]], [[1. - root, 0., 0.], [0., 0., 0.]]
        for before in (0., 1.):
            with self.assertRaisesRegex(ValueError, "nonpositive distance"):
                check(first, last, initial_activation=[before], tolerance_m=2.)
        with self.assertRaisesRegex(ValueError, "nonpositive distance"):
            check([[0., 0., 0.], [0., 0., 0.]], last, initial_activation=[0.], tolerance_m=2.)
        pending = check(first, last, initial_activation=[0.], final_activation=[0.], tolerance_m=2.)
        self.assertEqual(pending["rows"], [{"rowIndex": 0, "rowId": "source:row-0", "status": "pending"}])

    def test_tiny_positive_activation_never_scales_geometric_error(self):
        tiny = float(np.nextafter(0., 1.))
        with self.assertRaisesRegex(ValueError, "upper scalar-distance"):
            check([[2., 0., 0.], [0., 0., 0.]], initial_activation=[tiny], final_activation=[tiny])

    def test_original_products_retain_cancellation_and_underflowed_distances(self):
        self.assertEqual(.1 * 10. - 1., 0.)
        positions = [[10., 0., 0.], [0., 0., 0.], [1., 0., 0.]]
        distance = 2. ** -54
        report = check(positions, rows=[((0, .1), (1, .9), (2, -1.))],
                       initial_targets=[distance], final_targets=[distance], tolerance_m=2. ** -60)
        self.assertEqual(rational(report["rows"][0]["distanceSquaredMinimum"]["value"]), Fraction(distance) ** 2)
        tiny = float(np.nextafter(0., 1.))
        self.assertEqual(tiny * .5, 0.)
        report = check([[tiny, 0., 0.], [0., 0., 0.]], rows=[((0, .5), (1, -.5))],
                       initial_targets=[tiny], final_targets=[tiny], tolerance_m=tiny)
        self.assertEqual(rational(report["rows"][0]["distanceSquaredMinimum"]["value"]), Fraction(tiny) ** 2 / 4)

    def test_pending_rows_keep_full_identity_validation_and_digest_binding(self):
        positions = [[0., 0., 0.], [0., 0., 0.]]
        changes = dict(rows=[((0, 1.), (1, -1.)), ((0, .5), (1, -.5))], row_ids=["first", "second"],
                       initial_targets=[1., 2.], final_targets=[2., 3.], initial_activation=[0., 0.],
                       final_activation=[0., 0.])
        baseline = check(positions, **changes)
        self.assertEqual((baseline["checkedRows"], baseline["pendingRows"]), (0, 2))
        self.assertEqual([item["rowId"] for item in baseline["rows"]], ["first", "second"])
        for mutation in ({"final_targets": [2., 4.]}, {"tolerance_m": .5}, {"row_ids": ["second", "first"]},
                         {"rows": [((0, .75), (1, -.75)), ((0, .5), (1, -.5))]}):
            self.assertNotEqual(check(positions, **dict(changes, **mutation))["inputSha256"], baseline["inputSha256"])
        moved = [[1., 0., 0.], [1., 0., 0.]]
        self.assertNotEqual(check(positions, moved, **changes)["inputSha256"], baseline["inputSha256"])
        json.dumps(baseline, allow_nan=False)
        baseline["rows"][0]["rowId"] = "mutated-output"
        self.assertEqual(check(positions, **changes)["rows"][0]["rowId"], "first")
        with self.assertRaises(ValueError):
            check(positions, **dict(changes, initial_targets=[0., 2.]))

    def test_witnesses_reconstruct_exact_geometry_across_seeded_paths(self):
        rng = np.random.default_rng(1821)
        for _ in range(20):
            first, last = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
            before, after, epsilon = 8., 9., 10.
            report = check(first, last, initial_targets=[before], final_targets=[after], tolerance_m=epsilon)
            x0 = [Fraction(float(a)) - Fraction(float(b)) for a, b in zip(*first)]
            x1 = [Fraction(float(a)) - Fraction(float(b)) for a, b in zip(*last)]
            for name in ("distanceSquaredMinimum", "upperErrorMarginMinimum"):
                witness = report["rows"][0][name]
                u = rational(witness["atLocalFraction"])

                def value(t):
                    q = sum(((a + t * (b - a)) ** 2 for a, b in zip(x0, x1)), Fraction())
                    return q if name == "distanceSquaredMinimum" else (Fraction(before + epsilon) + t) ** 2 - q

                self.assertEqual(value(u), rational(witness["value"]))
                self.assertTrue(all(value(Fraction(i, 10)) >= value(u) for i in range(11)))

    def test_raw_types_shape_bounds_and_nonexact_integer_conversion_reject(self):
        positions = [[1., 0., 0.], [0., 0., 0.]]
        attacks = [dict(tolerance_m=value) for value in (True, 0., -1., np.inf, np.nan)]
        attacks += [dict(initial_targets=[value]) for value in (False, 0., -1., np.inf, np.nan, np.int64(2 ** 53 + 1))]
        attacks += [dict(initial_activation=[value]) for value in (True, -1., 2., np.nan)]
        attacks += [dict(initial_activation=[.5], final_activation=[.25]), dict(final_targets=[]),
                    dict(row_ids=[]), dict(row_ids=[False]), dict(row_ids=["x" * 129]),
                    dict(rows=[((False, 1.), (1, -1.))]), dict(rows=[((0., 1.), (1, -1.))]),
                    dict(rows=[((0, True), (1, -1.))]), dict(rows=[((0, 0.), (1, -1.))]),
                    dict(rows=[((0, 1.), (0, -1.))]), dict(rows=[((2, 1.), (1, -1.))]),
                    dict(rows=[[(0, 1.)] * 7]), dict(rows=[[(0, 1.)]] * 4097)]
        for options in attacks:
            with self.subTest(options=str(options)[:150]), self.assertRaises(ValueError):
                check(positions, **options)
        for malformed in ([[False, 0., 0.], [0., 0., 0.]], [[np.inf, 0., 0.], [0., 0., 0.]],
                          [np.array(1.)], [[1., 0.]], [[1., 0., 0.]] * 25001):
            with self.subTest(shape=str(malformed)[:100]), self.assertRaises(ValueError):
                check(malformed)
        if np.dtype(np.longdouble).itemsize > 8:
            with self.assertRaises(ValueError):
                check(positions, tolerance_m=np.longdouble(.25))

    def test_exact_control_knot_guard_and_dyadic_bound(self):
        self.assertEqual(sweep.verify_control_interval(.25, .5, [1., .5, 0., .25, .5]),
                         (Fraction(1, 4), Fraction(1, 2)))
        self.assertEqual(sweep.verify_control_interval(0, Fraction(1, 2 ** 40), []),
                         (Fraction(0), Fraction(1, 2 ** 40)))
        for start, end, knots in ((0, 1, [.5]), (.25, .75, [.5]), (0, 1, [2. ** -40]),
                                  (False, 1, []), (0, True, []), (0, 1, [False]),
                                  (0, 0, []), (1, 0, []), (-1, 1, []), (0, 2, []),
                                  (0, Fraction(1, 3), []), (0, 1, [Fraction(1, 3)]),
                                  (0, 2. ** -41, []), (0, 1, [np.nan]), (0, 1, [0] * 131)):
            with self.subTest(start=start, end=end, knots=knots[:3]), self.assertRaises(ValueError):
                sweep.verify_control_interval(start, end, knots)


if __name__ == "__main__":
    unittest.main()
