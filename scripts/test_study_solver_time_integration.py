"""Independent stdlib analytic oracles; no cloth/native imports or jobs."""
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal as D, localcontext, ROUND_HALF_EVEN
from fractions import Fraction as F
from functools import lru_cache
import unittest

import importlib.util
from pathlib import Path

_PRODUCER_SPEC = importlib.util.spec_from_file_location(
    'study_solver_time_integration', Path(__file__).with_name('study_solver_time_integration.py'))
producer = importlib.util.module_from_spec(_PRODUCER_SPEC)
_PRODUCER_SPEC.loader.exec_module(producer)


@dataclass(frozen=True)
class Root3:
    """Exact Q(sqrt(3)); independent of the producer's Decimal elimination."""
    a: F = F(0)
    b: F = F(0)

    @staticmethod
    def of(value):
        return value if isinstance(value, Root3) else Root3(F(value))

    def __add__(self, other):
        other = self.of(other)
        return Root3(self.a + other.a, self.b + other.b)

    __radd__ = __add__

    def __neg__(self):
        return Root3(-self.a, -self.b)

    def __sub__(self, other):
        return self + -self.of(other)

    def __rsub__(self, other):
        return self.of(other) + -self

    def __mul__(self, other):
        other = self.of(other)
        return Root3(self.a*other.a + 3*self.b*other.b,
                     self.a*other.b + self.b*other.a)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.of(other)
        norm = other.a*other.a - 3*other.b*other.b
        if norm == 0:
            raise ZeroDivisionError
        return self * Root3(other.a/norm, -other.b/norm)

    def __pow__(self, exponent):
        assert type(exponent) is int and exponent >= 0
        result = Root3(F(1))
        for _ in range(exponent):
            result *= self
        return result

    def decimal(self):
        return decimal(self.a) + decimal(self.b)*D(3).sqrt()


def decimal(value):
    value = F(value)
    return D(value.numerator)/D(value.denominator)


def solve_exact(matrix, rhs):
    """Full augmented Gaussian elimination; no A-squared stage elimination."""
    n = len(rhs)
    rows = [[Root3.of(v) for v in row] + [Root3.of(y)]
            for row, y in zip(matrix, rhs)]
    for column in range(n):
        pivot = next(i for i in range(column, n) if rows[i][column] != Root3())
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [v/scale for v in rows[column]]
        for i in range(n):
            if i != column:
                scale = rows[i][column]
                rows[i] = [v-scale*w for v, w in zip(rows[i], rows[column])]
    return tuple(row[-1] for row in rows)


def exact_stages(method, x, v, t, h, omega, duration, amplitude):
    # Derive the collocation tableau separately in the exact quadratic field.
    if method == 'midpoint':
        a, weights = ((Root3(F(1, 2)),),), (F(1),)
    elif method == 'gauss4':
        a = ((Root3(F(1, 4)), Root3(F(1, 4), F(-1, 6))),
             (Root3(F(1, 4), F(1, 6)), Root3(F(1, 4))))
        weights = (F(1, 2), F(1, 2))
    else:
        raise ValueError(method)
    n = len(weights)
    times = tuple(t + h*sum(row, Root3()) for row in a)
    # f(s) from q''+omega^2*q, expressed as a polynomial in physical time.
    coefficient = amplitude/duration**5
    loads = tuple(coefficient*(omega**2*s**5 + 20*s**3) for s in times)
    derivatives = tuple(coefficient*(5*omega**2*s**4 + 60*s**2) for s in times)
    matrix = [[Root3() for _ in range(2*n)] for _ in range(2*n)]
    rhs = []
    for i in range(n):
        matrix[i][i] = Root3(F(1))
        for j in range(n):
            matrix[i][n+j] = -h*a[i][j]
        rhs.append(Root3.of(x))
    for i in range(n):
        matrix[n+i][n+i] = Root3(F(1))
        for j in range(n):
            matrix[n+i][j] = h*omega**2*a[i][j]
        rhs.append(v + h*sum((a[i][j]*loads[j] for j in range(n)), Root3()))
    solution = solve_exact(matrix, rhs)
    positions, velocities = solution[:n], solution[n:]
    x1 = x + h*sum((weights[i]*velocities[i] for i in range(n)), Root3())
    v1 = v + h*sum((weights[i]*(-omega**2*positions[i]+loads[i]) for i in range(n)), Root3())
    work = -h*sum((weights[i]*derivatives[i]*positions[i] for i in range(n)), Root3())
    split = None
    if n == 1:
        def force(s):
            return coefficient*(omega**2*s**5+20*s**3)
        split = -(loads[0]-force(t))*x-(force(t+h)-loads[0])*x1
    return dict(x=x1, v=v1, stagePositions=positions, stageVelocities=velocities,
                stageTimes=times, quadratureWork=work, symmetricSplitWork=split)


def exact_rotation(method, z):
    z = F(z)
    if method == 'midpoint':
        return (4-z*z)/(4+z*z), 4*z/(4+z*z)
    if method == 'gauss4':
        denominator = z**4+12*z*z+144
        return (z**4-60*z*z+144)/denominator, (144*z-12*z**3)/denominator
    raise ValueError(method)


@lru_cache(maxsize=1)
def pi_rational():
    # Independent Machin construction. Alternating remainders bound pi error.
    def arctan_inverse(denominator, terms):
        return sum((F((-1)**k, (2*k+1)*denominator**(2*k+1))
                    for k in range(terms)), F())
    value = 16*arctan_inverse(5, 100)-4*arctan_inverse(239, 32)
    bound = F(16, 201*5**201)+F(4, 65*239**65)
    assert bound < F(1, 10**130)
    return value


def reference_trig(angle):
    # Range reduction using independent pi, then direct Taylor; no halving or
    # double-angle recurrence from the producer. Remainder <1e-113 here.
    with localcontext() as context:
        context.prec = 120
        x = decimal(angle)
        period = 2*decimal(pi_rational())
        x -= (x/period).to_integral_value(rounding=ROUND_HALF_EVEN)*period
        cosine, sine = D(1), x
        cterm, sterm = D(1), x
        for k in range(1, 256):
            cterm = -cterm*x*x/D((2*k-1)*(2*k))
            sterm = -sterm*x*x/D((2*k)*(2*k+1))
            cosine += cterm
            sine += sterm
            if max(abs(cterm), abs(sterm)) < D('1e-115'):
                return cosine, sine
        raise AssertionError('independent reference failed to converge')


class TimeIntegrationOracleTests(unittest.TestCase):
    def assert_close(self, actual, expected, tolerance=D('1e-85')):
        self.assertLessEqual(abs(actual-expected), tolerance*(1+abs(expected)))

    def test_rational_maps_energy_reversal_and_gauss_branch(self):
        with localcontext() as context:
            context.prec = 100
            for method in producer.METHODS:
                for z in (F(0), F(1, 7), F(-2), F(34641, 10000),
                          F(17321, 5000), F(4), F(100), F(-100)):
                    with self.subTest(method=method, z=z):
                        c, s = exact_rotation(method, z)
                        self.assertEqual(c*c+s*s, 1)
                        self.assertEqual(exact_rotation(method, -z), (c, -s))
                        actual = producer.rotation(method, decimal(z))
                        for got, wanted in zip(actual, (c, s)):
                            self.assert_close(got, decimal(wanted))
            self.assertGreater(exact_rotation('gauss4', F(34641, 10000))[1], 0)
            self.assertLess(exact_rotation('gauss4', F(17321, 5000))[1], 0)

    def test_independent_trig_small_negative_and_large_phase(self):
        with localcontext() as context:
            context.prec = 100
            for angle in (F(0), F(1, 3), F(-7, 2), F(320),
                          F(687928, 125), F(-687928, 125)):
                expected = reference_trig(angle)
                actual = producer.sincos(decimal(angle))
                for got, wanted in zip(actual, expected):
                    self.assert_close(got, wanted)
            with self.assertRaises(ValueError):
                producer.sincos(D(2)**31)

    def test_full_coupled_stages_work_and_conjugation_oracle(self):
        cases = [
            (F(0), F(0), F(0), F(1, 100), F(500), F(8, 125), F(1, 10000)),
            (F(3, 10000), F(-7, 1000), F(3, 1000), F(7, 10000), F(13), F(8, 125), F(1, 10000)),
            (F(-1, 100), F(2, 100), F(1, 125), F(3, 250), F(7), F(8, 125), F(0)),
        ]
        with localcontext() as context:
            context.prec = 100
            for method in producer.METHODS:
                for values in cases:
                    with self.subTest(method=method, values=values):
                        expected = exact_stages(method, *values)
                        actual = producer.forced_step(method, *(decimal(v) for v in values))
                        for key in ('x', 'v', 'quadratureWork'):
                            self.assertEqual(expected[key].b, 0)
                            self.assert_close(actual[key], expected[key].decimal())
                        for key in ('stagePositions', 'stageVelocities', 'stageTimes'):
                            for got, wanted in zip(actual[key], expected[key]):
                                self.assert_close(got, wanted.decimal())
                        if method == 'midpoint':
                            self.assert_close(actual['symmetricSplitWork'], expected['symmetricSplitWork'].decimal())
                        else:
                            self.assertIsNone(actual['symmetricSplitWork'])
                            for key in ('stagePositions', 'stageVelocities', 'stageTimes'):
                                left, right = expected[key]
                                self.assertEqual((left.a, left.b), (right.a, -right.b))
                        self.assertLessEqual(actual['stagePositionDefect'], D('1e-85'))
                        self.assertLessEqual(actual['stageVelocityDefect'], D('1e-85'))

    def test_coupled_gauss_is_not_two_midpoints_or_stage_velocity(self):
        values = (F(1, 100), F(3, 100), F(0), F(1, 8), F(7), F(8, 125), F(0))
        exact = exact_stages('gauss4', *values)
        x, v, t, h, w, duration, amplitude = values
        half = exact_stages('midpoint', x, v, t, h/2, w, duration, amplitude)
        twice = exact_stages('midpoint', half['x'].a, half['v'].a, t+h/2, h/2, w, duration, amplitude)
        self.assertNotEqual((exact['x'], exact['v']), (twice['x'], twice['v']))
        self.assertNotEqual(exact['v'], (exact['x']-x)/h)
        c, s = exact_rotation('gauss4', w*h)
        self.assertEqual(exact['x'], Root3(c*x+s*v/w))
        self.assertEqual(exact['v'], Root3(-s*w*x+c*v))

    def test_polynomial_motion_forcing_and_exact_work_integral(self):
        w, duration, amplitude = F(500), F(8, 125), F(1, 10000)
        coefficient = amplitude/duration**5
        # Multiply q by derivative of f as polynomials, then integrate exactly.
        q = {5: coefficient}
        force = {3: 20*coefficient, 5: w*w*coefficient}
        integrand = {}
        for i, qi in q.items():
            for j, fj in force.items():
                integrand[i+j-1] = integrand.get(i+j-1, F())-qi*j*fj
        work = sum((coefficient*duration**(degree+1)/(degree+1)
                    for degree, coefficient in integrand.items()), F())
        self.assertEqual(work, -w*w*amplitude**2/2-F(15, 2)*amplitude**2/duration**2)
        with localcontext() as context:
            context.prec = 100
            for t in (F(0), duration/7, duration):
                args = (decimal(t), decimal(w), decimal(duration), decimal(amplitude))
                expected_force = sum((value*t**power for power, value in force.items()), F())
                expected_derivative = sum((power*value*t**(power-1) for power, value in force.items()), F())
                self.assert_close(producer.forcing(*args), decimal(expected_force))
                self.assert_close(producer.forcing_derivative(*args), decimal(expected_derivative))
                position, velocity = producer.polynomial_motion(decimal(t), decimal(duration), decimal(amplitude))
                self.assert_close(position, decimal(coefficient*t**5))
                self.assert_close(velocity, decimal(5*coefficient*t**4))
            row = producer.forced('midpoint', 2)
            self.assert_close(D(row['exactContinuousParameterWorkJoules']), decimal(work))

    def test_exact_discrete_energy_does_not_prove_true_work(self):
        t, duration, amplitude, omega = F(0), F(8, 125), F(1, 10000), F(7)
        exact = exact_stages('midpoint', F(0), F(0), t, duration, omega, duration, amplitude)
        x, v = exact['x'], exact['v']
        force_final = omega**2*amplitude+20*amplitude/duration**2
        energy = v*v/2+omega**2*x*x/2-force_final*x
        continuous_work = -omega**2*amplitude**2/2-F(15, 2)*amplitude**2/duration**2
        self.assertEqual(energy, exact['symmetricSplitWork'])
        self.assertNotEqual(energy, Root3(continuous_work))
        self.assertNotEqual(exact['quadratureWork'], Root3(continuous_work))
        self.assertNotEqual(x, Root3(amplitude))
        with localcontext() as context:
            context.prec = 100
            actual = producer.forced_step('midpoint', D(0), D(0), D(0), decimal(duration), decimal(omega), decimal(duration), decimal(amplitude))
            self.assert_close(actual['symmetricSplitWork'], energy.decimal())
            self.assert_close(actual['quadratureWork'], exact['quadratureWork'].decimal())

    def test_all_grid_worst_phase_and_raw_doubling_independent_reference(self):
        with localcontext() as context:
            context.prec = 100
            duration, amplitude, omega, count = F(8, 125), F(1, 100), F(1000), 32
            for method in producer.METHODS:
                c, s = exact_rotation(method, omega*duration/count)
                real, imag = F(1), F(0)
                peaks = []
                for k in range(1, count+1):
                    real, imag = real*c-imag*s, real*s+imag*c
                    rc, rs = reference_trig(omega*duration*k/count)
                    peaks.append(((decimal(real)-rc)**2+(decimal(imag)-rs)**2).sqrt()*decimal(amplitude))
                expected = max(peaks)
                row = producer.oscillator(method, decimal(omega), count)
                self.assert_close(D(row['maximumWorstPhaseNativeVelocityErrorMPerSecond']), expected)
                self.assert_close(D(row['maximumWorstPhasePositionErrorMetres']), expected/decimal(omega))
                self.assertEqual(row['maximumErrorSampleIndex'], peaks.index(expected)+1)
                self.assert_close(D(row['finalWorstPhaseNativeVelocityErrorMPerSecond']), peaks[-1])
                coarse_c, coarse_s = exact_rotation(method, 2*omega*duration/count)
                local_squared = (coarse_c-(c*c-s*s))**2+(coarse_s-2*c*s)**2
                self.assert_close(D(row['localDoublingWorstPhaseVelocityIndicatorMPerSecond']), decimal(amplitude)*decimal(local_squared).sqrt())
                self.assertLess(D(row['maximumArithmeticRelativeEnergyDefect']), D('1e-85'))
                if method == 'midpoint':
                    self.assertGreater(expected, peaks[-1])
                    self.assertFalse(row['sampledAccuracyPass'])

    def test_fast_mode_position_can_pass_with_wrong_native_velocity(self):
        with localcontext() as context:
            context.prec = 100
            row = producer.oscillator('midpoint', D(85991), 8)
            self.assertLess(D(row['maximumWorstPhasePositionErrorMetres']), D(producer.CONFIG['positionGoalMetres']))
            self.assertGreater(D(row['maximumWorstPhaseNativeVelocityErrorMPerSecond']), D(producer.CONFIG['nativeVelocityGoalMPerSecond']))
            self.assertFalse(row['sampledAccuracyPass'])

    def test_budget_edge_counts_all_trials_and_coupled_stages(self):
        with localcontext() as context:
            context.prec = 100
            for method, count in [('midpoint', 2730), ('gauss4', 2730), ('gauss4', 2732)]:
                row = producer.oscillator(method, D(50), count)
                stages = 1 if method == 'midpoint' else 2
                self.assertEqual(row['hypotheticalDoublingTransactions'], count//2)
                self.assertEqual(row['hypotheticalPhysicalTrials'], 3*(count//2))
                self.assertEqual(row['minimumStageForceEvaluations'], stages*3*(count//2))
                self.assertEqual(row['coupledPositionUnknownsPerMechanicalDof'], stages)
                self.assertEqual(row['withinTrialBudget'], count == 2730)
            self.assertEqual(producer.CONFIG['maximumTrials'], 4096)

    def test_invalid_method_grid_and_frequency_reject(self):
        for method in ('gauss2', 'two-midpoints', None):
            with self.assertRaises(ValueError):
                producer.rotation(method, D(1))
            with self.assertRaises(ValueError):
                producer.tableau(method)
            with self.assertRaises(ValueError):
                producer.oscillator(method, D(50), 8)
        for count in (0, -2, 1, 3, True, 2.0, 32770):
            with self.assertRaises(ValueError):
                producer.oscillator('midpoint', D(50), count)
        for omega in (D(0), D(-1), D('NaN'), D('Infinity'), 50):
            with self.assertRaises(ValueError):
                producer.oscillator('midpoint', omega, 8)

    def test_near_threshold_unresolved_and_precision_decision_guards(self):
        with localcontext() as context:
            context.prec = 100
            guard = D(producer.CONFIG['decisionGuardAbsolute'])
            goal = D('0.00025')
            self.assertTrue(producer.diagnostic_pass([(goal-2*guard, goal)]))
            self.assertFalse(producer.diagnostic_pass([(goal+2*guard, goal)]))
            for delta in (-guard, D(0), guard):
                self.assertIsNone(producer.diagnostic_pass([(goal+delta, goal)]))
            self.assertFalse(producer.diagnostic_pass([(goal, goal), (2*goal, goal)]))
            sample = {'oscillators': [{'method': 'midpoint', 'fineIntervals': 8,
                                      'error': '0.1', 'sampledAccuracyPass': None}], 'forced': []}
            repeated = deepcopy(sample)
            repeated['oscillators'][0]['error'] = str(D('0.1')+D('1e-60'))
            comparison = producer.compare_precision(sample, repeated)
            self.assertTrue(comparison['decisionsAndIdentitiesAgree'])
            self.assertFalse(comparison['certifiedIntervalArithmetic'])
            for key, value in [('sampledAccuracyPass', True), ('method', 'gauss4'),
                               ('fineIntervals', 10), ('error', '0.1000001')]:
                mutated = deepcopy(sample)
                mutated['oscillators'][0][key] = value
                with self.assertRaises(ArithmeticError):
                    producer.compare_precision(sample, mutated)
            with self.assertRaises(ValueError):
                producer.compare_precision(sample, {'oscillators': [], 'forced': []})
            missing = deepcopy(sample)
            del missing['oscillators'][0]['error']
            with self.assertRaises(ValueError):
                producer.compare_precision(sample, missing)


if __name__ == '__main__':
    unittest.main()
