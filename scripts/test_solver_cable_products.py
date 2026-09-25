"""Independent product/force coefficient oracles and lazy native regressions.

Host math tests extract only stdlib definitions. Retained old functions supply a
behavior comparison; independent distributive and analytic oracles never call
the production product or functional builders. Native imports occur in setUp of
the response class only, for the declared locked-runtime test suite.
"""
import ast
from decimal import Decimal
from fractions import Fraction as F
import itertools
import json
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace
import unittest
from unittest import mock

SOURCE = Path(__file__).with_name('solver_continuous_cable_sewing.py')
BASELINE_MUL = '''def _mul(first, second):
    values = [F()]*(len(first)+len(second)-1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            values[i+j] += a*b
    return _trim(values)
'''
BASELINE_FUNCTIONALS = '''def _functionals(self, index, vectors, q, derivatives):
    beta, d = self._beta[index], self._targets[index]
    result = [(('e',), _scale(_add(q, _mul(d, d)), beta/2),
               _scale(_mul(d, q), -beta), (F(),))]
    if not derivatives:
        return result
    dofs = [(3*v+a, row, a) for v, row in self._rows[index] for a in range(3)]
    for dof, row, axis in dofs:
        first = _scale(_mul(row, vectors[axis]), beta)
        result.append((('g', dof), first, _scale(_mul(first, d), -1), (F(),)))
    for offset, (a, left, axis) in enumerate(dofs):
        for b, right, other in dofs[offset:]:
            product = _scale(_mul(left, right), beta)
            polynomial = product if axis == other else (F(),)
            half = _scale(_mul(product, d), -1) if axis == other else (F(),)
            three = _mul(_mul(_mul(product, d), vectors[axis]), vectors[other])
            result.append((('h', a, b), polynomial, half, three))
    return result
'''


def extracted_math(*, baseline=False):
    tree = ast.parse(SOURCE.read_text())
    names = {'_trim', '_add', '_scale', '_mul'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ContinuousCableSewing')
    nodes.append(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_functionals'))
    if len(nodes) != 5 or {n.name for n in nodes} != names | {'_functionals'}:
        raise AssertionError('Unique source definitions required')
    namespace = {'F': F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    if baseline:
        exec(compile(BASELINE_MUL + BASELINE_FUNCTIONALS, '<retained-original-products>', 'exec'), namespace)
    return namespace


def coefficients(terms):
    """Canonical coefficients from a degree-to-exact-value dictionary."""
    degree = max((k for k, v in terms.items() if v), default=0)
    return tuple(F(terms.get(k, 0)) for k in range(degree+1))


def product_oracle(*polynomials):
    """Distributive monomial enumeration, independent of iterative convolution."""
    terms = {}
    for selected in itertools.product(*(tuple(enumerate(p)) for p in polynomials)):
        degree = sum(k for k, _ in selected)
        coefficient = math.prod((F(v) for _, v in selected), start=F(1))
        terms[degree] = terms.get(degree, F()) + coefficient
    return coefficients(terms)


def linear_oracle(*weighted_polynomials):
    terms = {}
    for weight, polynomial in weighted_polynomials:
        for degree, value in enumerate(polynomial):
            terms[degree] = terms.get(degree, F()) + F(weight)*F(value)
    return coefficients(terms)


def value(poly, point):
    return sum((F(v)*F(point)**k for k, v in enumerate(poly)), F())


def functional_oracle(self, index, vectors, q, derivatives):
    """Coefficients of analytic energy, gradient and upper Hessian entries.

    E=beta*(q+d*d)/2-beta*d*sqrt(q); g_i=beta*w_i*v_a*(1-d/sqrt(q));
    H_ij=beta*w_i*w_j*((1-d/sqrt(q))*delta_ab+d*v_a*v_b/q**(3/2)).
    Each returned triple multiplies 1, q**(-1/2), q**(-3/2).
    """
    beta, d = self._beta[index], self._targets[index]
    zero = (F(),)
    records = [(('e',), linear_oracle((beta/2, q), (beta/2, product_oracle(d, d))),
                linear_oracle((-beta, product_oracle(d, q))), zero)]
    if not derivatives:
        return records
    dofs = [(3*vertex+axis, row, axis) for vertex, row in self._rows[index] for axis in range(3)]
    for dof, row, axis in dofs:
        records.append((('g', dof), linear_oracle((beta, product_oracle(row, vectors[axis]))),
                        linear_oracle((-beta, product_oracle(d, row, vectors[axis]))), (F(),)))
    for i, (a, left, axis) in enumerate(dofs):
        for b, right, other in dofs[i:]:
            records.append((('h', a, b),
                linear_oracle((beta, product_oracle(left, right))) if axis == other else (F(),),
                linear_oracle((-beta, product_oracle(d, left, right))) if axis == other else (F(),),
                linear_oracle((beta, product_oracle(d, left, right, vectors[axis], vectors[other])))))
    return records


def specimen(rows, beta=F(7, 11), target=(F(1, 3), F(2, 5))):
    return SimpleNamespace(_rows=[rows], _beta=[beta], _targets=[target])


def geometry(vectors):
    return linear_oracle(*((F(1), product_oracle(v, v)) for v in vectors))


def signature(poly):
    return tuple((type(v), struct.pack('>d', v) if type(v) is float else v) for v in poly)


class CableProductMathTests(unittest.TestCase):
    def setUp(self):
        self.current = extracted_math()
        self.old = extracted_math(baseline=True)

    def test_product_distributive_and_evaluation_oracles(self):
        tiny = F(math.ulp(0.));huge = F(float.fromhex('0x1.fffffffffffffp+1023'))
        polys = [(), (F(),), (F(), F()), (F(-0.0),), (F(1),),
                 (F(-3, 7), F(5, 11), F()), (tiny, -tiny, F(1, 2**1200)),
                 (huge, -huge), (F(2**2048+1, 2**1021+3), F(7, 13))]
        rng = random.Random(853107)
        polys += [tuple(F(rng.randrange(-20, 21), rng.randrange(1, 15))
                        for _ in range(rng.randrange(1, 5))) for _ in range(8)]
        for i, first in enumerate(polys):
            for second in (polys[(i+3) % len(polys)], (F(),), (F(1),)):
                expected = product_oracle(first, second)
                actual = self.current['_mul'](first, second)
                self.assertEqual(actual, expected)
                self.assertEqual(self.old['_mul'](first, second), expected)
                self.assertTrue(all(type(v) is F for v in actual))
                for point in (F(-2, 3), F(), F(5, 7)):
                    self.assertEqual(value(actual, point), value(first, point)*value(second, point))

    def test_zero_shortcut_has_no_arithmetic_and_narrow_admission(self):
        class TupleSubclass(tuple):
            pass
        class FractionSubclass(F):
            pass
        def forbidden(*args, **kwargs):
            raise AssertionError('Fraction multiplication dispatched')
        zero = (F(),);other = (F(2, 7), F(-5, 13), F())
        noncanonical_zero = F()
        object.__setattr__(noncanonical_zero, '_denominator', 2)
        with mock.patch.object(F, '__mul__', forbidden):
            for first, second in ((zero, other), (other, zero), (zero, ()), ((), zero), (zero, zero)):
                self.assertEqual(self.current['_mul'](first, second), (F(),))
            for first, second in (((F(), F()), other), (other, (F(), F())), ([F()], other),
                                  (TupleSubclass(zero), other), ((FractionSubclass(0),), other),
                                  (zero, (FractionSubclass(2),)), ((noncanonical_zero,), other)):
                with self.assertRaisesRegex(AssertionError, 'multiplication dispatched'):
                    self.current['_mul'](first, second)

    def test_other_types_and_invalid_operands_retain_baseline_behavior(self):
        class TupleSubclass(tuple):
            pass
        class FractionSubclass(F):
            pass
        cases = [([F()], (F(2),)), (TupleSubclass((F(),)), (F(2),)),
                 ((FractionSubclass(0),), (F(2),)), ((F(),), (FractionSubclass(2),)),
                 ((0,), (2, 3)), ((False,), (True,)), ((-0.0,), (2., -3.)),
                 ((F(),), (2.,)), ((F(),), (float('inf'),)), ((F(),), (float('nan'),)),
                 ((F(),), (Decimal('0.1'),)), ((F(),), (object(),)),
                 ((F(),), None), (None, (F(),)), ((F(),), (complex(1, 2),))]
        for first, second in cases:
            outcomes = []
            for namespace in (self.old, self.current):
                try:
                    result = namespace['_mul'](first, second)
                except Exception as error:
                    outcomes.append(('error', type(error), str(error)))
                else:
                    outcomes.append(('return', signature(result)))
            with self.subTest(first=first, second=second):
                self.assertEqual(*outcomes)

    def test_product_coefficients_do_not_alias_inputs_outputs_or_calls(self):
        for first, second in (((F(),), (F(2, 3), F(5, 7))),
                              ((F(1, 3), F(2, 5)), (F(7, 11), F(-3, 2)))):
            before = (tuple(first), tuple(second))
            a = self.current['_mul'](first, second);b = self.current['_mul'](first, second)
            expected = product_oracle(first, second)
            self.assertTrue(all(v is not original for v in a for original in first+second))
            self.assertEqual(len({id(v) for v in a}), len(a))
            self.assertTrue(all(x is not y for x in a for y in b))
            object.__setattr__(a[0], '_numerator', 991)
            self.assertEqual((first, second), before)
            self.assertEqual(b, expected)

    def test_functional_coefficients_zero_one_many_and_duplicate_vertex_rows(self):
        vectors = ((F(3, 7), F(-2, 9)), (F(5, 13), F(7, 19)), (F(), F(11, 17)))
        q = geometry(vectors)
        rows = [(4, (F(1, 2), F(2, 3))), (4, (F(-3, 5), F(7, 11))),
                (2, (F(), F(5, 17))), (8, (F(13, 19), F(-2, 23)))]
        for selected in ([], rows[:1], rows[:2], rows):
            for beta, target in ((F(), (F(1), F())), (F(7, 11), (F(2, 3), F(-5, 7))),
                                 (F(2**1024+1, 2**509+3), (F(math.ulp(0.)), F(1, 2**1200)))):
                obj = specimen(selected, beta, target)
                expected = functional_oracle(obj, 0, vectors, q, True)
                actual = self.current['_functionals'](obj, 0, vectors, q, True)
                self.assertEqual(actual, expected)
                self.assertEqual(self.old['_functionals'](obj, 0, vectors, q, True), expected)
                n = len(selected)*3
                self.assertEqual(len(actual), 1+n+n*(n+1)//2)
                for _, *polys in actual:
                    for poly in polys:
                        self.assertTrue(all(type(v) is F for v in poly))
                        self.assertTrue(len(poly) == 1 or poly[-1])

    def test_analytic_energy_force_hessian_values_and_energy_only(self):
        vectors = ((F(3), F(1)), (F(4), F(4, 3)), (F(),))
        q = geometry(vectors);rows = [(0, (F(1, 2), F(2, 3))), (1, (F(-3, 5), F(1, 7)))]
        obj = specimen(rows, F(7, 11), (F(1, 3), F(1, 5)))
        records = self.current['_functionals'](obj, 0, vectors, q, True)
        for point in (F(), F(1, 7), F(1)):
            v = [value(p, point) for p in vectors];radius = F(5)*(1+point/3)
            self.assertEqual(sum(x*x for x in v), radius*radius)
            beta, d = obj._beta[0], value(obj._targets[0], point)
            w = {3*vertex+a: (value(row, point), a) for vertex, row in rows for a in range(3)}
            for key, polynomial, half, three in records:
                got = value(polynomial, point)+value(half, point)/radius+value(three, point)/radius**3
                if key[0] == 'e':
                    want = beta*(radius-d)**2/2
                elif key[0] == 'g':
                    weight, axis = w[key[1]];want = beta*weight*v[axis]*(1-d/radius)
                else:
                    left, axis = w[key[1]];right, other = w[key[2]]
                    want = beta*left*right*((1-d/radius)*(axis == other)+d*v[axis]*v[other]/radius**3)
                self.assertEqual(got, want)
        # Energy-only must not access even the private row collection.
        energy_obj = SimpleNamespace(_beta=obj._beta, _targets=obj._targets)
        self.assertEqual(self.current['_functionals'](energy_obj, 0, vectors, q, False), records[:1])

    def test_returned_functionals_have_no_new_cross_axis_or_call_aliases(self):
        obj = specimen([(0, (F(1, 2), F(2, 3))), (1, (F(3, 5), F(-1, 7)))])
        vectors = ((F(2), F(1, 3)), (F(3), F(2, 5)), (F(5), F(1, 7)))
        q = geometry(vectors);expected = functional_oracle(obj, 0, vectors, q, True)
        actual = self.current['_functionals'](obj, 0, vectors, q, True)
        second = self.current['_functionals'](obj, 0, vectors, q, True)
        old = self.old['_functionals'](obj, 0, vectors, q, True)
        flat = lambda rs: [v for record in rs for poly in record[1:] for v in poly]
        for results in (actual, old):
            values = flat(results)
            self.assertEqual(len(values), len({id(v) for v in values}))
        inputs = [obj._beta[0], *obj._targets[0], *q, *(x for row in vectors for x in row),
                  *(x for _, row in obj._rows[0] for x in row)]
        self.assertTrue(all(v is not original for v in flat(actual) for original in inputs))
        self.assertTrue(all(v is not other for v in flat(actual) for other in flat(second)))
        first_hessian = next(row for row in actual if row[0] == ('h', 0, 0))
        object.__setattr__(first_hessian[1][0], '_numerator', 1001)
        self.assertEqual(second, expected)
        for got, want in zip(actual, expected):
            if got[0] != ('h', 0, 0):
                self.assertEqual(got, want)
        self.assertEqual(functional_oracle(obj, 0, vectors, q, True), expected)

    def test_dispatch_reduction_and_cache_lifetime(self):
        rows = [(i, (F(i+1, 7), F(i+2, 11))) for i in range(4)]
        vectors = ((F(2), F(1, 3)), (F(3), F(2, 5)), (F(5), F(1, 7)));q = geometry(vectors)
        obj = specimen(rows)
        counts = []
        for namespace in (self.old, self.current):
            counter = mock.Mock(wraps=namespace['_mul'])
            with mock.patch.dict(namespace, {'_mul': counter}):
                self.assertEqual(namespace['_functionals'](obj, 0, vectors, q, True), functional_oracle(obj, 0, vectors, q, True))
            counts.append(counter.call_count)
        row_pairs = 4*5//2;dofs = 3*4;dof_pairs = dofs*(dofs+1)//2
        old_expected = 2+2*dofs+4*dof_pairs+3*row_pairs
        new_expected = 2+2*dofs+2*row_pairs+2*dof_pairs
        self.assertEqual((old_expected, new_expected), (368, 202))
        self.assertEqual(counts, [old_expected, new_expected])
        # Reuse the same owner/index with changed parameters and row order.
        obj._rows[0] = list(reversed(rows));obj._targets[0] = (F(5, 7), F(9, 11));obj._beta[0] = F(13, 17)
        self.assertEqual(self.current['_functionals'](obj, 0, vectors, q, True), functional_oracle(obj, 0, vectors, q, True))


class CableProductResponseTests(unittest.TestCase):
    """Locked runtime only. Both defining and imported multiplication aliases tested."""
    def setUp(self):
        import solver_continuous_cable_sewing as cable
        import solver_cable_parameters as parameters
        import test_solver_cable_composition as composition
        self.cable, self.parameters, self.composition = cable, parameters, composition
        self.old = extracted_math(baseline=True)
        self.potential = cable.ContinuousCableSewing(4, [composition.response_cell()])
        self.precision = {'energy_tolerance_joules':1e-8, 'gradient_tolerance_newtons':1e-8,
                          'hessian_tolerance_newtons_per_meter':1e-8}

    def alternatives(self):
        return ((self.old['_mul'], self.old['_functionals']), (product_oracle, functional_oracle))

    def test_complete_responses_match_analytic_and_retained_builders(self):
        fixtures = [[[0., 0., 0.], [0., 0., 0.], [1., .2, .1], [2., .4, .2]],
                    [[0., 0., 0.], [0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]],
                    [[0., 0., 0.], [0., 0., 0.], [.25, 0., 0.], [1.25, 0., 0.]]]
        for positions in fixtures:
            actual = self.potential.evaluate(positions, **self.precision)
            for multiply, functional in self.alternatives():
                with mock.patch.object(self.cable, '_mul', multiply), \
                     mock.patch.object(self.cable.ContinuousCableSewing, '_functionals', functional):
                    expected = self.potential.evaluate(positions, **self.precision)
                self.composition.CableCompositionResponseTests.assert_response_identical(self, actual, expected)

    def test_fixed_and_parameter_work_match_both_multiplication_aliases(self):
        self.assertIs(self.parameters._mul, self.cable._mul)
        first = [[0., 0., 0.], [0., 0., 0.], [.25, 0., 0.], [1.25, 0., 0.]]
        last = [[0., 0., 0.], [0., 0., 0.], [.5, 0., 0.], [1.5, 0., 0.]]
        actual = self.potential.energy_change(first, last, absolute_tolerance_joules=1e-8)
        self.assertNotEqual(actual['changeJoules'], 0.)
        recipe = self.parameters.CableParameterRecipe(self.potential)
        for multiply, functional in self.alternatives():
            with mock.patch.object(self.cable, '_mul', multiply), \
                 mock.patch.object(self.parameters, '_mul', multiply), \
                 mock.patch.object(self.cable.ContinuousCableSewing, '_functionals', functional):
                expected = self.potential.energy_change(first, last, absolute_tolerance_joules=1e-8)
            self.assertEqual(json.dumps(actual, sort_keys=True, allow_nan=False), json.dumps(expected, sort_keys=True, allow_nan=False))
        for positions in (first, [[0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]):
            arguments = (positions, [[.5, .5]], [1.], [[.75, .75]], [.5])
            observed = mock.Mock(wraps=self.parameters._mul)
            with mock.patch.object(self.parameters, '_mul', observed):
                actual = recipe.parameter_energy_change(*arguments, absolute_tolerance_joules=1e-8)
            self.assertGreater(observed.call_count, 0)
            self.assertNotEqual(actual['totalWorkJoules'], 0.)
            self.assertGreater(actual['releaseEnergyRemovedJoules'], 0.)
            for multiply, functional in self.alternatives():
                with mock.patch.object(self.cable, '_mul', multiply), \
                     mock.patch.object(self.parameters, '_mul', multiply), \
                     mock.patch.object(self.cable.ContinuousCableSewing, '_functionals', functional):
                    expected = recipe.parameter_energy_change(*arguments, absolute_tolerance_joules=1e-8)
                self.assertEqual(json.dumps(actual, sort_keys=True, allow_nan=False), json.dumps(expected, sort_keys=True, allow_nan=False))
        self.assertIs(self.parameters._mul, self.cable._mul)


if __name__ == '__main__':
    unittest.main()
