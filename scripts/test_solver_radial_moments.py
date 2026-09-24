"""Independent exact and Decimal antiderivative checks of rational moments."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import math
import unittest

from solver_radial_moments import radial_moment_bounds


POWERS = (F(-1, 2), F(-3, 2), F(-1), F(1, 2))


def decimal(value):
    value = F(value)
    return D(value.numerator)/D(value.denominator)


def atan(value):
    """Independent Decimal half-angle reduction followed by alternating series."""
    x = value
    multiplier = 1
    while abs(x) > D('0.1'):
        x = x/(1+(1+x*x).sqrt())
        multiplier *= 2
    total, term, n = x, x, 1
    while True:
        term *= -x*x
        following = total+term/(2*n+1)
        if following == total:
            return total*multiplier
        total = following
        n += 1


def one_plus_square_reference(degree):
    """Elementary antiderivatives/recurrences, not binomial quadrature."""
    with localcontext() as context:
        context.prec = 140
        root = D(2).sqrt()
        inverse_root = [(1+root).ln(), root-1]
        for k in range(2, degree+3):
            inverse_root.append((root-(k-1)*inverse_root[k-2])/k)
        cube = [1/root, 1-1/root]
        inverse = [atan(D(1)), D(2).ln()/2]
        for k in range(2, degree+1):
            cube.append(inverse_root[k-2]-cube[k-2])
            inverse.append(D(1)/(k-1)-inverse[k-2])
        return {F(-1, 2): inverse_root[:degree+1], F(-3, 2): cube[:degree+1],
                F(-1): inverse[:degree+1], F(1, 2): [inverse_root[k]+inverse_root[k+2] for k in range(degree+1)]}


class RadialMomentTests(unittest.TestCase):
    def assert_contains(self, interval, reference):
        lo, hi = interval
        self.assertIs(type(lo), F)
        self.assertIs(type(hi), F)
        expected = F(reference)
        self.assertLessEqual(lo, expected)
        self.assertGreaterEqual(hi, expected)

    def assert_certificate(self, result, tolerance):
        self.assertIs(result['verified'], True)
        self.assertEqual(result['absoluteTolerance'], tolerance)
        self.assertGreater(result['minimumQ'], 0)
        previous = F()
        for a, b, low, high in result['panelRanges']:
            self.assertEqual(a, previous)
            self.assertLess(a, b)
            self.assertGreater(low, 0)
            self.assertLessEqual(4*(high-low), high+low)
            previous = b
        self.assertEqual(previous, 1)
        self.assertEqual(result['panels'], len(result['panelRanges']))
        for values in result['moments'].values():
            for lo, hi in values:
                self.assertGreaterEqual(lo, 0)
                self.assertLessEqual(lo, hi)
                self.assertLessEqual(hi-lo, tolerance)

    def test_constant_rational_square_is_exact_for_every_degree_and_power(self):
        tolerance = F(1, 10**60)
        for root in (F(1), F(3, 7), F(2**150), F(1, 2**150)):
            result = radial_moment_bounds((root*root, F(), F()), POWERS, 5, tolerance)
            self.assert_certificate(result, tolerance)
            self.assertEqual(result['panels'], 1)
            self.assertEqual(result['maxTerms'], 1)
            for power in POWERS:
                for k, interval in enumerate(result['moments'][power]):
                    value = root**int(2*power)/F(k+1)
                    self.assertEqual(interval, (value, value))

    def test_constant_irrational_scale_has_directed_normalization_error(self):
        tolerance = F(1, 10**55)
        result = radial_moment_bounds((F(2), F(), F()), POWERS, 5, tolerance)
        self.assert_certificate(result, tolerance)
        with localcontext() as context:
            context.prec = 140
            root = D(2).sqrt()
            for p in POWERS:
                for k, interval in enumerate(result['moments'][p]):
                    # p=-1 has an exact rational answer; squaring a rounded
                    # Decimal sqrt(2) would fabricate an oracle discrepancy.
                    if p == -1:
                        self.assertEqual(interval, (F(1, 2*(k+1)),)*2)
                    else:
                        self.assert_contains(interval, root**int(2*p)/(k+1))

    def test_all_moments_match_independent_elementary_antiderivatives(self):
        tolerance = F(1, 10**40)
        result = radial_moment_bounds((F(1), F(), F(1)), POWERS, 5, tolerance)
        self.assert_certificate(result, tolerance)
        self.assertGreater(result['panels'], 1)
        expected = one_plus_square_reference(5)
        for p in POWERS:
            for k, interval in enumerate(result['moments'][p]):
                with self.subTest(power=str(p), degree=k):
                    self.assert_contains(interval, expected[p][k])

    def test_perfect_square_linear_norm_preserves_logarithmic_limits(self):
        tolerance = F(1, 10**38)
        result = radial_moment_bounds((F(1), F(2), F(1)), POWERS, 5, tolerance)
        self.assert_certificate(result, tolerance)
        with localcontext() as context:
            context.prec = 140
            for p in POWERS:
                for k, interval in enumerate(result['moments'][p]):
                    expected = D(0)
                    for j in range(k+1):
                        exponent = j+int(2*p)+1
                        integral = D(2).ln() if exponent == 0 else (D(2)**exponent-1)/exponent
                        expected += math.comb(k, j)*(-1)**(k-j)*integral
                    self.assert_contains(interval, expected)

    def test_nearly_collinear_quadratic_keeps_small_discriminant(self):
        epsilon = F(1, 2**40)
        tolerance = F(1, 10**42)
        result = radial_moment_bounds((1+epsilon*epsilon, F(2), F(1)), POWERS, 0, tolerance)
        self.assert_certificate(result, tolerance)
        with localcontext() as context:
            context.prec = 140
            e = decimal(epsilon)
            r1, r2 = (1+e*e).sqrt(), (4+e*e).sqrt()
            logarithm = ((2+r2)/(1+r1)).ln()
            expected = {F(-1, 2): logarithm, F(-3, 2): (2/r2-1/r1)/(e*e),
                        F(-1): (atan(e)-atan(e/2))/e,
                        F(1, 2): (2*r2-r1+e*e*logarithm)/2}
            for p in POWERS:
                self.assert_contains(result['moments'][p][0], expected[p])
            # A thresholded exact-collinear branch would incorrectly return
            # log(2); the enclosure resolves the genuine epsilon^2 difference.
            self.assertLess(result['moments'][F(-1, 2)][0][1], F(D(2).ln()))

    def test_narrow_positive_interior_minimum_and_exact_stationary_split(self):
        epsilon = F(1, 64)
        tolerance = F(1, 10**25)
        result = radial_moment_bounds((F(1, 4)+epsilon**2, F(-1), F(1)), POWERS, 0, tolerance)
        self.assert_certificate(result, tolerance)
        self.assertEqual(result['minimumQ'], epsilon**2)
        self.assertEqual(result['minimumQAt'], F(1, 2))
        self.assertIn(F(1, 2), [p[1] for p in result['panelRanges']])
        with localcontext() as context:
            context.prec = 140
            e = decimal(epsilon);r=(D('.25')+e*e).sqrt()
            logarithm = ((D('.5')+r)/e).ln()
            expected = {F(-1, 2): 2*logarithm, F(-3, 2): 1/(e*e*r),
                        F(-1): 2*atan(D('.5')/e)/e,
                        F(1, 2): D('.5')*r+e*e*logarithm}
            for p in POWERS:
                self.assert_contains(result['moments'][p][0], expected[p])

    def test_collapsed_closed_interval_rejects_even_off_sample_grid(self):
        cases = [(F(1, 9), F(-2, 3), F(1)), (F(), F(), F(1)),
                 (F(1), F(-2), F(1)), (F(1), F(-5), F(1)), (F(-1), F(), F())]
        for q in cases:
            with self.subTest(q=q), self.assertRaisesRegex(ValueError, 'strictly positive'):
                radial_moment_bounds(q, POWERS, 3, F(1, 10**20))

    def test_concave_positive_quadratic_includes_interior_maximum(self):
        tolerance = F(1, 10**35)
        result = radial_moment_bounds((F(1), F(1), F(-1)), (F(-1),), 0, tolerance)
        self.assert_certificate(result, tolerance)
        self.assertEqual(result['minimumQ'], 1)
        self.assertEqual(result['maximumQ'], F(5, 4))
        with localcontext() as context:
            context.prec = 140
            root = D(5).sqrt()
            expected = 2*((root+1)/(root-1)).ln()/root
            self.assert_contains(result['moments'][F(-1)][0], expected)

    def test_tiny_nonconstant_term_below_binary64_range_is_not_discarded(self):
        epsilon = F(1, 2**1100)
        tolerance = F(1, 2**1150)
        result = radial_moment_bounds((F(1), epsilon, F()), (F(-1),), 0, tolerance)
        self.assert_certificate(result, tolerance)
        lo, hi = result['moments'][F(-1)][0]
        self.assertLess(hi, 1)
        # Independent alternating-log bounds for log(1+epsilon)/epsilon.
        lower = 1-epsilon/2+epsilon**2/3-epsilon**3/4
        upper = 1-epsilon/2+epsilon**2/3
        self.assertLessEqual(lo, lower)
        self.assertGreaterEqual(hi, upper)

    def test_large_and_small_power_of_two_scaling_preserves_exact_bounds(self):
        tolerance = F(1, 10**28)
        references = one_plus_square_reference(0)
        for exponent in (-200, 200):
            scale = F(2)**exponent
            for p in POWERS:
                factor = scale**int(2*p)
                result = radial_moment_bounds((scale**2, F(), scale**2), (p,), 0, tolerance*factor)
                self.assert_certificate(result, tolerance*factor)
                with localcontext() as context:
                    context.prec = 140
                    self.assert_contains(result['moments'][p][0], references[p][0]*decimal(factor))

    def test_first_two_series_tail_bounds_are_exact_and_budgeted(self):
        q = (F(1), F(1, 4), F())
        first = radial_moment_bounds(q, (F(-1),), 0, F(1), max_terms=1)
        self.assertEqual(first['maxTerms'], 1)
        self.assertEqual(first['moments'][F(-1)][0], (F(104, 135), F(136, 135)))
        second = radial_moment_bounds(q, (F(-1),), 0, F(1, 10), max_terms=2)
        self.assertEqual(second['maxTerms'], 2)
        self.assertEqual(second['moments'][F(-1)][0], (F(1064, 1215), F(1096, 1215)))
        with localcontext() as context:
            context.prec = 140
            reference = 4*D('1.25').ln()
            self.assert_contains(first['moments'][F(-1)][0], reference)
            self.assert_contains(second['moments'][F(-1)][0], reference)
        with self.assertRaisesRegex(ValueError, 'term budget'):
            radial_moment_bounds(q, (F(-1),), 0, F(1, 10**30), max_terms=2)

    def test_panel_and_depth_budgets_fail_closed_without_tolerance_change(self):
        q = (F(1), F(), F(10000))
        for options in ({'max_panels': 1}, {'max_depth': 0}, {'max_depth': 1}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'budget exhausted'):
                radial_moment_bounds(q, POWERS, 2, F(1, 10**25), **options)

    def test_raw_type_bounds_and_unsupported_requests_reject(self):
        q = (F(1), F(), F(1)); tolerance = F(1, 10**20)
        for value in ([F(1), F(), F(1)], (1, F(), F(1)), (True, F(), F(1)),
                      (1., F(), F(1)), (F(1), F()), (F(1 << 4096), F(), F())):
            with self.subTest(q=value), self.assertRaises(ValueError):
                radial_moment_bounds(value, POWERS, 2, tolerance)
        for powers in ([], [], (), (F(-2),), (F(-1), F(-1)), (-1,), (True,), ([F(-1)],)):
            with self.subTest(powers=powers), self.assertRaises(ValueError):
                radial_moment_bounds(q, powers, 2, tolerance)
        for degree in (True, -1, 6, 2.):
            with self.subTest(degree=degree), self.assertRaises(ValueError):
                radial_moment_bounds(q, POWERS, degree, tolerance)
        for tol in (F(), F(-1), 0., True, 1e-20, F(1, 1 << 4096)):
            with self.subTest(tolerance=tol), self.assertRaises(ValueError):
                radial_moment_bounds(q, POWERS, 2, tol)
        for options in ({'max_panels': 0}, {'max_panels': True}, {'max_panels': 4097},
                        {'max_terms': 0}, {'max_terms': 257}, {'max_depth': -1}, {'max_depth': 129}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                radial_moment_bounds(q, POWERS, 2, tolerance, **options)

    def test_global_monomial_moments_obey_reversal_and_leave_inputs_unchanged(self):
        q, powers, tolerance = (F(1), F(1), F(1)), POWERS, F(1, 10**25)
        result = radial_moment_bounds(q, powers, 5, tolerance)
        reverse = radial_moment_bounds((F(3), F(-3), F(1)), powers, 5, tolerance)
        for p in powers:
            for k, (lo, hi) in enumerate(reverse['moments'][p]):
                expected_lo = expected_hi = F()
                for j in range(k+1):
                    coefficient = F((-1)**j*math.comb(k, j))
                    a, b = result['moments'][p][j]
                    expected_lo += min(coefficient*a, coefficient*b)
                    expected_hi += max(coefficient*a, coefficient*b)
                self.assertLessEqual(lo, expected_hi)
                self.assertGreaterEqual(hi, expected_lo)
        self.assertEqual(q, (F(1), F(1), F(1)))
        self.assertEqual(powers, POWERS)
        result['moments'].clear()
        self.assertEqual(radial_moment_bounds((F(1), F(), F()), powers, 0, tolerance)['moments'][F(-1)], ((F(1), F(1)),))


if __name__ == '__main__':
    unittest.main()
