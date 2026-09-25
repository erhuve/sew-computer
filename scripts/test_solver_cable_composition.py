"""Independent affine-polynomial oracles and exact cable-response regressions.

The math class extracts only four stdlib functions, without importing the native
module. The response class is for the locked numerical runtime and imports it
lazily. The binomial oracle is independent of the production Horner/convolution
algorithm; the retained previous Horner body supplies a separate behavior check.
"""
import ast
from decimal import Decimal
from fractions import Fraction as F
import json
import math
from pathlib import Path
import random
import struct
import unittest
from unittest import mock

SOURCE = Path(__file__).with_name('solver_continuous_cable_sewing.py')
BASELINE_COMPOSE = '''def _compose(poly, start, width):
    result = (F(),)
    for coefficient in reversed(poly):
        result = _add(_mul(result, (start, width)), (coefficient,))
    return result
'''


def extracted_math():
    """Execute stdlib-only definitions, never the production module's imports."""
    names = {'_trim', '_add', '_mul', '_compose'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names or len(nodes) != len(names):
        raise AssertionError('Unique source math definitions required')
    namespace = {'F': F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


def binomial_composition(poly, start, width):
    """Coefficient of x**j in sum a[k]*(start+width*x)**k."""
    a = tuple(F(value) for value in poly)
    start, width = F(start), F(width)
    terms = {j: sum((a[k] * math.comb(k, j) * start**(k-j) * width**j
                     for k in range(j, len(a))), F())
             for j in range(len(a))}
    degree = max((j for j, value in terms.items() if value), default=0)
    return tuple(terms.get(j, F()) for j in range(degree+1))


def direct_value(poly, value):
    return sum((F(coefficient) * F(value)**degree
                for degree, coefficient in enumerate(poly)), F())


def signature(poly):
    """Preserve numeric types and signed binary64 zeros in fallback comparison."""
    return tuple((type(value), struct.pack('>d', value) if type(value) is float else value)
                 for value in poly)


class CableCompositionMathTests(unittest.TestCase):
    def setUp(self):
        self.math = extracted_math()
        old = dict(self.math)
        exec(compile(BASELINE_COMPOSE, '<retained-baseline-compose>', 'exec'), old)
        self.baseline = old['_compose']
        self.compose = self.math['_compose']

    def assert_exact(self, poly, start, width):
        want = binomial_composition(poly, start, width)
        got = self.compose(poly, start, width)
        self.assertEqual(got, want)
        self.assertEqual(self.baseline(poly, start, width), want)
        self.assertIs(type(got), tuple)
        self.assertTrue(all(type(value) is F for value in got))
        self.assertTrue(len(got) == 1 or got[-1] != 0)
        for x in (F(-3, 2), F(), F(1, 7), F(1), F(5, 3)):
            self.assertEqual(direct_value(got, x), direct_value(poly, start+width*x))

    def test_identity_zero_large_subnormal_and_trailing_zero_oracles(self):
        tiny = F(math.ulp(0.))
        largest = F(float.fromhex('0x1.fffffffffffffp+1023'))
        for poly in ((), (F(),), (F(), F(), F()), (F(-0.0),),
                     (F(3, 7), F(-2, 9), F(), F()),
                     (tiny, -tiny, F(1, 2**1200)),
                     (largest, -largest, F(2**4096+1, 2**2053+3)),
                     (F(-17, 23), F(), F(19, 31), F())):
            with self.subTest(poly=poly):
                self.assert_exact(poly, F(), F(1))

    def test_nonidentity_translation_scaling_reversal_and_collapse(self):
        poly = (F(2, 7), F(-3, 5), F(), F(11, 13), F(-1, 17), F())
        for start, width in ((F(2, 3), F(5, 7)), (F(-5, 2), F(-3, 4)),
                             (F(1), F(-1)), (F(3, 11), F()),
                             (F(math.ulp(0.)), F(1)),
                             (F(), F(1)+F(1, 2**1074))):
            with self.subTest(start=start, width=width):
                self.assert_exact(poly, start, width)

    def test_seeded_rational_coefficient_and_evaluation_oracles(self):
        rng = random.Random(487091)
        for index in range(40):
            poly = tuple(F(rng.randrange(-31, 32), rng.randrange(1, 24))
                         for _ in range(rng.randrange(0, 8)))
            start, width = (F(rng.randrange(-7, 8), rng.randrange(1, 10))
                            for _ in range(2))
            with self.subTest(index=index):
                self.assert_exact(poly, F(), F(1))
                self.assert_exact(poly, start, width)

    def test_identity_fast_path_avoids_polynomial_arithmetic(self):
        poly = (F(7, 19), F(-5, 13), F(), F())
        expected = binomial_composition(poly, F(), F(1))
        def forbidden(*args, **kwargs):
            raise AssertionError('Identity composition dispatched polynomial arithmetic')
        with mock.patch.dict(self.math, {'_add': forbidden, '_mul': forbidden}), \
             mock.patch.object(F, '__add__', forbidden), \
             mock.patch.object(F, '__mul__', forbidden):
            result = self.compose(poly, F(), F(1))
        self.assertEqual(result, expected)
        self.assertTrue(all(value is not source for value in result for source in poly))

    def test_container_and_numeric_types_use_unchanged_fallback(self):
        class TupleSubclass(tuple):
            pass
        class FractionSubclass(F):
            pass
        class IntSubclass(int):
            pass
        cases = [([F(1), F(2), F()], F(), F(1)),
                 (TupleSubclass((F(1), F(2))), F(), F(1)),
                 ((F(1), F(2)), 0, 1), ((F(1), F(2)), -0.0, 1.0),
                 ((F(1), F(2)), False, True),
                 ((F(1), F(2)), FractionSubclass(0), F(1)),
                 ((F(1), F(2)), F(), FractionSubclass(1)),
                 ((FractionSubclass(1, 3), F(2)), F(), F(1)),
                 ((1, IntSubclass(2), False), F(), F(1)),
                 ((-0.0, .5, 0.0), F(), F(1))]
        original_mul = self.math['_mul']
        for poly, start, width in cases:
            with self.subTest(poly=poly, start=start, width=width):
                with mock.patch.dict(self.math, {'_mul': mock.Mock(wraps=original_mul)}) as _:
                    got = self.compose(poly, start, width)
                    self.assertGreater(self.math['_mul'].call_count, 0)
                self.assertEqual(signature(got), signature(self.baseline(poly, start, width)))

    def test_unsupported_numeric_failures_match_retained_baseline(self):
        cases = [((object(),), F(), F(1)), ((Decimal('0.1'),), F(), F(1)),
                 ((F(1),), None, F(1)), ((F(1),), F(), None),
                 ((F(1),), 'zero', F(1)), (None, F(), F(1))]
        for poly, start, width in cases:
            outcomes = []
            for fn in (self.baseline, self.compose):
                try:
                    value = fn(poly, start, width)
                except Exception as error:
                    outcomes.append(('error', type(error), str(error)))
                else:
                    outcomes.append(('return', signature(value)))
            with self.subTest(poly=poly, start=start, width=width):
                self.assertEqual(outcomes[0], outcomes[1])
                self.assertEqual(outcomes[0][0], 'error')

    def test_result_fraction_instances_do_not_alias_inputs_or_siblings(self):
        shared = F(3, 7)
        source = (shared, shared, F())
        first = self.compose(source, F(), F(1))
        second = self.compose(source, F(), F(1))
        self.assertEqual(first, (F(3, 7), F(3, 7)))
        self.assertIsNot(first, source)
        self.assertIsNot(first[0], first[1])
        self.assertTrue(all(value is not original for value in first for original in source))
        self.assertTrue(all(a is not b for a in first for b in second))
        # Fractions present an immutable API but their private slots are writable;
        # deliberately alter them here to catch newly shared internal references.
        object.__setattr__(first[0], '_numerator', 99)
        self.assertEqual(source, (F(3, 7), F(3, 7), F()))
        self.assertEqual(first[1], F(3, 7))
        self.assertEqual(second, (F(3, 7), F(3, 7)))
        object.__setattr__(shared, '_numerator', 5)
        self.assertEqual(second, (F(3, 7), F(3, 7)))

    def test_nonidentity_also_preserves_old_copy_behavior(self):
        source = (F(2, 5), F(7, 11), F())
        before = tuple((value.numerator, value.denominator) for value in source)
        result = self.compose(source, F(1, 3), F(2, 7))
        self.assertTrue(all(value is not old for value in result for old in source))
        object.__setattr__(result[0], '_numerator', 987)
        self.assertEqual(tuple((v.numerator, v.denominator) for v in source), before)


def response_cell():
    def anchor(vertex):
        return [{'vertex': vertex, 'weight': {'numerator': '1', 'denominator': '1'}}]
    return {'id': 'independent-composition-response',
            'negativeStart': anchor(0), 'negativeEnd': anchor(1),
            'positiveStart': anchor(2), 'positiveEnd': anchor(3),
            'targetsMeters': [.5, .5],
            'referenceLengthMeters': {'numerator': '1', 'denominator': '1'},
            'stiffnessDensityNPerM2': 2., 'activation': 1.}


class CableCompositionResponseTests(unittest.TestCase):
    """Locked numerical runtime only; host source-extraction run excludes these."""
    def setUp(self):
        import solver_continuous_cable_sewing as cable
        self.cable = cable
        old = extracted_math()
        exec(compile(BASELINE_COMPOSE, '<retained-baseline-compose>', 'exec'), old)
        self.baseline = old['_compose']
        self.potential = cable.ContinuousCableSewing(4, [response_cell()])
        self.precision = {'energy_tolerance_joules': 1e-8,
                          'gradient_tolerance_newtons': 1e-8,
                          'hessian_tolerance_newtons_per_meter': 1e-8}

    def assert_response_identical(self, first, second):
        self.assertEqual(set(first), set(second))
        self.assertEqual(struct.pack('>d', first['energy']), struct.pack('>d', second['energy']))
        a, b = first['gradient'], second['gradient']
        self.assertEqual((a.shape, a.dtype.str, a.tobytes()), (b.shape, b.dtype.str, b.tobytes()))
        a, b = first['hessian'], second['hessian']
        self.assertEqual((a.format, a.shape), (b.format, b.shape))
        for field in ('data', 'indices', 'indptr'):
            x, y = getattr(a, field), getattr(b, field)
            self.assertEqual((x.dtype.str, x.tobytes()), (y.dtype.str, y.tobytes()))
        self.assertEqual(json.dumps(first['certificate'], sort_keys=True, allow_nan=False),
                         json.dumps(second['certificate'], sort_keys=True, allow_nan=False))

    def test_complete_response_matches_binomial_and_previous_compose(self):
        # Fully taut, slack, and a crossing at exact material fraction1/4.
        fixtures = [([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], True),
                    ([[0., 0., 0.], [0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]], False),
                    ([[0., 0., 0.], [0., 0., 0.], [.25, 0., 0.], [1.25, 0., 0.]], True)]
        observed_maps = []
        for positions, composed in fixtures:
            with self.subTest(positions=positions):
                original = self.cable._compose
                def observed(poly, start, width):
                    observed_maps.append((start, width))
                    return original(poly, start, width)
                count = len(observed_maps)
                with mock.patch.object(self.cable, '_compose', observed):
                    actual = self.potential.evaluate(positions, **self.precision)
                self.assertEqual(len(observed_maps) > count, composed)
                for oracle in (binomial_composition, self.baseline):
                    with mock.patch.object(self.cable, '_compose', oracle):
                        expected = self.potential.evaluate(positions, **self.precision)
                    self.assert_response_identical(actual, expected)
        self.assertIn((F(), F(1)), observed_maps)
        self.assertTrue(any(pair != (F(), F(1)) for pair in observed_maps))

    def test_complete_work_record_matches_binomial_and_previous_compose(self):
        first = [[0., 0., 0.], [0., 0., 0.], [.25, 0., 0.], [1.25, 0., 0.]]
        last = [[0., 0., 0.], [0., 0., 0.], [.5, 0., 0.], [1.5, 0., 0.]]
        actual = self.potential.energy_change(first, last, absolute_tolerance_joules=1e-8)
        self.assertNotEqual(actual['changeJoules'], 0.)
        self.assertIs(actual['certificate']['stats']['normInvariant'], False)
        for oracle in (binomial_composition, self.baseline):
            with mock.patch.object(self.cable, '_compose', oracle):
                expected = self.potential.energy_change(first, last, absolute_tolerance_joules=1e-8)
            self.assertEqual(json.dumps(actual, sort_keys=True, allow_nan=False),
                             json.dumps(expected, sort_keys=True, allow_nan=False))

        # Parameter work imports _compose into its own module namespace. Patching
        # only the cable module would leave that actual shared callsite untested.
        import solver_cable_parameters as parameters
        self.assertIs(parameters._compose, self.cable._compose)
        recipe = parameters.CableParameterRecipe(self.potential)
        parameter_maps = []
        for positions in ([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], first):
            arguments = (positions, [[.5, .5]], [1.], [[.75, .75]], [.5])
            original_alias = parameters._compose
            def observed_alias(poly, start, width):
                parameter_maps.append((start, width))
                return original_alias(poly, start, width)
            with mock.patch.object(parameters, '_compose', observed_alias):
                parameter_actual = recipe.parameter_energy_change(
                    *arguments, absolute_tolerance_joules=1e-8)
            self.assertNotEqual(parameter_actual['totalWorkJoules'], 0.)
            self.assertGreater(parameter_actual['releaseEnergyRemovedJoules'], 0.)
            for oracle in (binomial_composition, self.baseline):
                with mock.patch.object(self.cable, '_compose', oracle), \
                     mock.patch.object(parameters, '_compose', oracle):
                    parameter_expected = recipe.parameter_energy_change(
                        *arguments, absolute_tolerance_joules=1e-8)
                self.assertEqual(json.dumps(parameter_actual, sort_keys=True, allow_nan=False),
                                 json.dumps(parameter_expected, sort_keys=True, allow_nan=False))
        self.assertIn((F(), F(1)), parameter_maps)
        self.assertTrue(any(pair != (F(), F(1)) for pair in parameter_maps))
        self.assertIs(parameters._compose, self.cable._compose)


if __name__ == '__main__':
    unittest.main()
