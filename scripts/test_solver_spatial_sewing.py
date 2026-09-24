"""Independent spatial lemmas; no solver, source mapper, or simulated fixture."""

import ast
import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random
import unittest
from unittest import mock

import solver_spatial_sewing as spatial


def rat(value):
    value = F(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def value(encoded):
    return F(int(encoded["numerator"]), int(encoded["denominator"]))


def cell(first=(1, 0, 0), last=None, target0=1, target1=None, lo=0, hi=1):
    return {"interval": [rat(lo), rat(hi)], "startDifferenceMeters": list(map(rat, first)),
            "endDifferenceMeters": list(map(rat, first if last is None else last)),
            "startTargetMeters": rat(target0), "endTargetMeters": rat(target0 if target1 is None else target1)}


def run(cells=None, epsilon=0):
    return spatial.verify_spatial_distance_exact([cell()] if cells is None else cells, tolerance_m=epsilon)


def direct(item, u, kind, epsilon):
    first = list(map(value, item["startDifferenceMeters"]))
    last = list(map(value, item["endDifferenceMeters"]))
    gap = sum(((1 - u) * a + u * b) ** 2 for a, b in zip(first, last))
    d = (1 - u) * value(item["startTargetMeters"]) + u * value(item["endTargetMeters"])
    return {"gap": gap, "error": gap - d ** 2, "upper": (d + epsilon) ** 2 - gap,
            "lower": gap - (d - epsilon) ** 2}[kind]


def independent_extrema(item, kind, epsilon, lo=F(0), hi=F(1)):
    # Recover the quadratic from three direct geometric evaluations, independently
    # of the producer's coefficient construction.
    f0, fm, f1 = [direct(item, u, kind, epsilon) for u in (F(0), F(1, 2), F(1))]
    a = 2 * (f1 + f0 - 2 * fm)
    b = f1 - f0 - a
    points = [lo, hi]
    if a and lo < -b / (2 * a) < hi:
        points.append(-b / (2 * a))
    samples = [(direct(item, u, kind, epsilon), u) for u in points]
    return min(samples), min(samples, key=lambda pair: (-pair[0], pair[1]))


class SpatialSewingTests(unittest.TestCase):
    def test_independent_stdlib_only_and_scope(self):
        tree = ast.parse(Path(spatial.__file__).read_text())
        imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names)
        self.assertLessEqual(imports, {"fractions", "hashlib", "json", "math", "re"})
        report = run()
        self.assertIs(report["verified"], True)
        self.assertIs(report["accepted"], False)
        self.assertTrue(report["withinTolerance"])
        text = " ".join(report["limitations"])
        for limitation in ("at one state", "source correspondence", "space-time", "force", "garment acceptance"):
            self.assertIn(limitation, text)

    def test_hidden_bowing_between_five_correct_samples(self):
        # Eight exact affine cells represent two continuous curves. The second
        # has four bows unseen by the five conventional quarter-fraction rows.
        gap, excess = F(1, 1024), F(1, 128)
        heights = [gap + (excess if i % 2 else 0) for i in range(9)]
        cells = [cell((0, 0, heights[i]), (0, 0, heights[i + 1]), gap, gap, F(i, 8), F(i + 1, 8))
                 for i in range(8)]
        self.assertEqual(heights[::2], [gap] * 5)
        report = run(cells)
        self.assertFalse(report["withinTolerance"])
        self.assertEqual(value(report["squaredGapMinimum"]["valueMetersSquared"]), gap ** 2)
        self.assertEqual(value(report["squaredGapMaximum"]["valueMetersSquared"]), (gap + excess) ** 2)
        self.assertEqual(value(report["squaredGapMaximum"]["atFraction"]), F(1, 8))
        self.assertTrue(run(cells, float(excess))["withinTolerance"])
        self.assertFalse(run(cells, math.nextafter(float(excess), 0))["withinTolerance"])

    def test_matching_endpoints_hide_interior_collapse(self):
        report = run([cell((1, 0, 0), (-1, 0, 0))], .25)
        self.assertFalse(report["withinTolerance"])
        self.assertEqual(value(report["squaredGapMinimum"]["valueMetersSquared"]), 0)
        self.assertEqual(value(report["squaredGapMinimum"]["atFraction"]), F(1, 2))
        lower = report["cells"][0]["lowerMargin"]["minimum"]
        self.assertEqual(value(lower["valueMetersSquared"]), -F(9, 16))
        self.assertGreaterEqual(value(report["cells"][0]["upperMargin"]["minimum"]["valueMetersSquared"]), 0)
        # This geometric diagnostic deliberately does not impose an independent
        # nonzero-distance/contact condition when the requested tolerance allows zero.
        self.assertTrue(run([cell((1, 0, 0), (-1, 0, 0))], 1)["withinTolerance"])

    def test_nonconstant_target_zero_tolerance_and_signed_squared_error(self):
        for first, last in ((F(1, 3), F(7, 5)), (F(7, 5), F(1, 3))):
            report = run([cell((first, 0, 0), (last, 0, 0), first, last)])
            self.assertTrue(report["withinTolerance"])
            item = report["cells"][0]
            self.assertEqual(list(map(value, item["signedSquaredTargetErrorPolynomial"])), [0, 0, 0])
            self.assertEqual(value(item["upperMargin"]["minimum"]["valueMetersSquared"]), 0)
            self.assertEqual(value(item["lowerMargin"]["minimum"]["valueMetersSquared"]), 0)
        report = run([cell((2, 0, 0))])
        self.assertEqual(value(report["signedSquaredTargetErrorMaximum"]["valueMetersSquared"]), 3)
        self.assertIn("not (sqrt(Q)-d)^2", report["signedSquaredTargetErrorConvention"])

    def test_changing_targets_have_interior_margin_witness(self):
        item = run([cell((1, 0, 0), (0, 2, 0), 1, 2)], .5)["cells"][0]
        witness = item["lowerMargin"]["minimum"]
        self.assertEqual(value(witness["valueMetersSquared"]), F(3, 16))
        self.assertEqual(value(witness["atLocalFraction"]), F(3, 8))
        self.assertTrue(item["withinTolerance"])
        self.assertFalse(run([cell((1, 0, 0), (0, 2, 0), 1, 2)], .25)["withinTolerance"])

    def test_lower_threshold_splits_and_reverses_exactly(self):
        for d0, d1, expected in ((F(1, 4), F(5, 4), (F(3, 4), F(1))),
                                 (F(5, 4), F(1, 4), (F(0), F(1, 4)))):
            report = run([cell((F(1, 4), 0, 0), target0=d0, target1=d1)], 1)
            lower = report["cells"][0]["lowerMargin"]
            self.assertEqual(tuple(map(value, lower["localInterval"])), expected)
            self.assertEqual(tuple(map(value, lower["interval"])), expected)
            self.assertEqual(value(lower["minimum"]["valueMetersSquared"]), 0)
            self.assertTrue(report["withinTolerance"])
        self.assertIsNone(run([cell((0, 0, 0), target0=F(1, 4), target1=F(1, 2))], 1)["cells"][0]["lowerMargin"])

    def test_equality_threshold_keeps_closed_singleton_and_full_intervals(self):
        for d0, d1, expected in ((1, F(1, 2), (0, 0)), (F(1, 2), 1, (1, 1)), (1, 1, (0, 1))):
            report = run([cell((0, 0, 0), target0=d0, target1=d1)], 1)
            lower = report["cells"][0]["lowerMargin"]
            self.assertEqual(tuple(map(value, lower["localInterval"])), expected)
            self.assertEqual(value(lower["minimum"]["valueMetersSquared"]), 0)
            self.assertTrue(report["withinTolerance"])

    def test_exact_tiny_cell_not_collapsed_to_binary64_parameter(self):
        a, b = F(1, 2), F(1, 2) + F(1, 2 ** 200)
        self.assertEqual(float(a), float(b))
        cells = [cell(lo=0, hi=a), cell((1, 0, 0), (-1, 0, 0), lo=a, hi=b),
                 cell((-1, 0, 0), lo=b, hi=1)]
        report = run(cells, .25)
        self.assertFalse(report["withinTolerance"])
        self.assertEqual(value(report["squaredGapMinimum"]["atFraction"]), (a + b) / 2)
        self.assertEqual(report["squaredGapMinimum"]["cellIndex"], 1)
        self.assertEqual(value(report["squaredGapMinimum"]["atLocalFraction"]), F(1, 2))

    def test_subnormal_exact_gap_and_tolerance_are_not_rounded_to_zero(self):
        tiny = F(1, 2 ** 1200)
        report = run([cell((tiny, 0, 0), target0=tiny)])
        self.assertEqual(float(tiny), 0.)
        self.assertEqual(value(report["squaredGapMinimum"]["valueMetersSquared"]), tiny ** 2)
        self.assertTrue(report["withinTolerance"])
        epsilon = math.nextafter(0., 1.)
        self.assertEqual(value(run(epsilon=epsilon)["exactToleranceMeters"]), F(1, 2 ** 1074))

    def test_independent_exact_and_decimal_geometry_oracle(self):
        rng = random.Random(72941)
        for _ in range(40):
            first = [F(rng.randint(-12, 12), rng.randint(1, 17)) for _ in range(3)]
            last = [F(rng.randint(-12, 12), rng.randint(1, 17)) for _ in range(3)]
            d0, d1 = F(rng.randint(1, 10), 7), F(rng.randint(1, 10), 7)
            epsilon = .5
            raw = cell(first, last, d0, d1)
            item = run([raw], epsilon)["cells"][0]
            for kind, low_name, high_name in (("gap", "squaredGapMinimum", "squaredGapMaximum"),
                                             ("error", "signedSquaredTargetErrorMinimum", "signedSquaredTargetErrorMaximum")):
                expected = independent_extrema(raw, kind, F(epsilon))
                for name, pair in zip((low_name, high_name), expected):
                    witness = item[name]
                    self.assertEqual((value(witness["valueMetersSquared"]), value(witness["atLocalFraction"])), pair)
            upper = independent_extrema(raw, "upper", F(epsilon))[0]
            self.assertEqual(value(item["upperMargin"]["minimum"]["valueMetersSquared"]), upper[0])
            lower = item["lowerMargin"]
            if lower is not None:
                lo, hi = map(value, lower["localInterval"])
                expected = independent_extrema(raw, "lower", F(epsilon), lo, hi)[0]
                self.assertEqual((value(lower["minimum"]["valueMetersSquared"]), value(lower["minimum"]["atLocalFraction"])), expected)
            # Independent high-precision evaluations bound the exact extrema at
            # several interior values; this is test evidence, not runtime sampling.
            with localcontext() as context:
                context.prec = 120
                dec = lambda x: Decimal(x.numerator) / Decimal(x.denominator)
                for u in (F(1, 11), F(5, 11), F(10, 11)):
                    squared = sum(((1 - dec(u)) * dec(a) + dec(u) * dec(b)) ** 2 for a, b in zip(first, last))
                    self.assertLessEqual(dec(value(item["squaredGapMinimum"]["valueMetersSquared"])) - Decimal("1e-110"), squared)
                    self.assertGreaterEqual(dec(value(item["squaredGapMaximum"]["valueMetersSquared"])) + Decimal("1e-110"), squared)

    def test_reverse_parameter_and_rigid_difference_covariance(self):
        first, last = [F(-1), F(1), F(2)], [F(2), F(4), F(-1)]
        before = run([cell(first, last, 2, 3)], .75)
        after = run([cell(last, first, 3, 2)], .75)
        rotated = run([cell([-first[1], first[0], first[2]], [-last[1], last[0], last[2]], 2, 3)], .75)
        for key in ("squaredGapMinimum", "squaredGapMaximum", "signedSquaredTargetErrorMinimum", "signedSquaredTargetErrorMaximum"):
            self.assertEqual(before[key]["valueMetersSquared"], after[key]["valueMetersSquared"])
            self.assertEqual(before[key], rotated[key])
        self.assertEqual(before["withinTolerance"], after["withinTolerance"])
        self.assertEqual(value(before["squaredGapMinimum"]["atFraction"]), 1 - value(after["squaredGapMinimum"]["atFraction"]))

    def test_full_partition_and_exact_continuity_not_tolerance(self):
        valid = [cell(lo=0, hi=F(1, 2)), cell(lo=F(1, 2), hi=1)]
        run(valid)
        attacks = []
        for path, replacement in (([0, "interval", 0], rat(F(1, 8))),
                                   ([1, "interval", 1], rat(F(7, 8))),
                                   ([1, "interval", 0], rat(F(1, 2) + F(1, 2 ** 200))),
                                   ([1, "interval", 0], rat(F(1, 4))),
                                   ([1, "interval", 1], rat(F(1, 2))),
                                   ([1, "startDifferenceMeters", 0], rat(1 + F(1, 2 ** 200))),
                                   ([1, "startTargetMeters"], rat(1 + F(1, 2 ** 200)))):
            changed = copy.deepcopy(valid)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
            attacks.append(changed)
        attacks.extend((valid[::-1], [valid[0]], [valid[1]], [valid[0], valid[0], valid[1]]))
        for malformed in attacks:
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                run(malformed)

    def test_raw_rational_aliases_types_shapes_and_targets_reject(self):
        aliases = [{"numerator": n, "denominator": d} for n, d in
                   (("2", "2"), ("0", "2"), ("-0", "1"), ("+1", "1"), ("01", "1"),
                    ("1", "01"), ("1", "-1"), ("1", "0"), (" 1", "1"), ("1\n", "1"),
                    ("1.0", "1"), ("1e0", "1"), ("１", "1"), (True, "1"), (1, "1"), ("1", 1))]
        aliases += [True, 1., ["1", "1"], {"numerator": "1"}, dict(rat(1), extra=False)]
        for malformed in aliases:
            raw = cell()
            raw["startDifferenceMeters"][0] = malformed
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                run([raw])
        for target in (0, -1, 101):
            with self.assertRaises(ValueError):
                run([cell(target0=target)])
        for malformed in (None, [], (), [None], [dict(cell(), extra=0)], [dict(cell(), interval=(rat(0), rat(1)))],
                          [dict(cell(), startDifferenceMeters=[rat(1)])], [dict(cell(), endTargetMeters=None)]):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                spatial.verify_spatial_distance_exact(malformed, tolerance_m=0)
        # Omission and surplus fields are never repaired.
        for key in spatial.CELL_FIELDS:
            raw = cell()
            del raw[key]
            with self.assertRaises(ValueError):
                run([raw])

    def test_raw_tolerance_admission_and_raw_identity(self):
        for epsilon in (True, False, "0", F(0), None, -1., math.inf, math.nan, 100.00001, 2 ** 53 + 1):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                run(epsilon=epsilon)
        self.assertEqual(value(run(epsilon=100)["exactToleranceMeters"]), 100)
        hashes = [run(epsilon=epsilon)["inputSha256"] for epsilon in (0, 0., -0.)]
        self.assertEqual(len(set(hashes)), 3)
        self.assertEqual(math.copysign(1, run(epsilon=-0.)["toleranceMeters"]), -1)

    def test_bounded_input_and_fail_closed_output_resources(self):
        with self.assertRaises(ValueError):
            run([cell()] * 4097)
        for raw in ({"numerator": "1" * 1236, "denominator": "1"}, rat(F(1, 2 ** 4096))):
            item = cell()
            item["startDifferenceMeters"][0] = raw
            with self.assertRaises(ValueError):
                run([item])
        # Exercise each independent fixed budget before expensive geometric work.
        # Reduced caps make these allocation regressions bounded, without weakening
        # production caps or relying on a multi-megabyte test fixture.
        for cap, limit in (("MAX_JSON_BYTES", 32), ("MAX_TOTAL_INTEGER_BITS", 8),
                           ("MAX_OUTPUT_BYTES", 1024), ("MAX_RESULT_INTEGER_BITS", 8)):
            with self.subTest(cap=cap), mock.patch.object(spatial, cap, limit), self.assertRaises(ValueError):
                run([cell((F(1, 257), 0, 0), target0=F(1, 257))])

    def test_input_hash_and_result_isolation(self):
        cells = [cell(lo=0, hi=F(1, 2)), cell(lo=F(1, 2), hi=1)]
        before = copy.deepcopy(cells)
        report = run(cells, .25)
        payload = {"profile": spatial.PROFILE, "cells": cells, "toleranceMeters": .25}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(report["inputSha256"], digest)
        self.assertEqual(cells, before)
        report["cells"][0]["interval"][0]["numerator"] = "999"
        report["limitations"].append("forged")
        report["squaredGapMaximum"]["valueMetersSquared"]["numerator"] = "999"
        self.assertEqual(cells, before)
        fresh = run(cells, .25)
        self.assertEqual(value(fresh["squaredGapMaximum"]["valueMetersSquared"]), 1)
        self.assertNotIn("forged", fresh["limitations"])
        self.assertEqual(fresh["squaredGapMinimum"]["cellIndex"], 0)
        self.assertEqual(value(fresh["squaredGapMinimum"]["atFraction"]), 0)
        self.assertNotEqual(run(cells, .5)["inputSha256"], digest)
        json.dumps(fresh, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
