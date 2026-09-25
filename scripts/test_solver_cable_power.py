"""Independent supplied-state actuator-power and control-rounding checks.

The expectations use exact 1D positive-part polynomial integrals and Decimal
radial antiderivatives. They do not call producer integration helpers, solve a
timestep, or admit a garment, source construction, or physical trajectory.
"""
import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
import math
import unittest
from unittest import mock

import numpy as np

from solver_continuous_cable_sewing import ContinuousCableSewing
from solver_cable_parameters import CableParameterRecipe
from solver_cable_parameter_schedule import CableParameterSchedule
from solver_cable_power import CableSchedulePower
import solver_cable_power as module


FIELDS = ('targetPowerWatts', 'activationPowerWatts', 'totalPowerWatts')
BUDGETS = dict(max_boundary_depth=80, max_boundary_panels=256,
               moment_max_terms=128, moment_max_panels=256, moment_max_depth=64)


def rat(value):
    value = F(value)
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def rational(value):
    assert type(value) is dict and set(value) == {'numerator', 'denominator'}
    assert type(value['numerator']) is str and type(value['denominator']) is str
    result = F(int(value['numerator']), int(value['denominator']))
    assert rat(result) == value
    return result


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def cell(name='span', *, density=1., length=F(1)):
    def anchor(vertex):
        return [{'vertex': vertex, 'weight': rat(1)}]
    return {'id': name, 'positiveStart': anchor(0), 'positiveEnd': anchor(1),
            'negativeStart': anchor(2), 'negativeEnd': anchor(3),
            'targetsMeters': [1., 1.], 'activation': 1.,
            'stiffnessDensityNPerM2': density, 'referenceLengthMeters': rat(length)}


def make(before=(1., 1.), after=(.5, .5), a0=.25, a1=.75, *, duration=1.,
         cells=None, middle=None):
    cells = [cell()] if cells is None else cells
    recipe = CableParameterRecipe(ContinuousCableSewing(4, cells))
    count = len(cells)
    knots = [dict(fraction=0., targetsMeters=[list(before) for _ in range(count)], activation=[a0]*count),
             dict(fraction=1., targetsMeters=[list(after) for _ in range(count)], activation=[a1]*count)]
    if middle is not None:
        knots.insert(1, middle)
    raw = dict(profile='cable-target-activation-v1', geometrySha256=recipe.geometry_sha256,
               cellIds=list(recipe.cell_ids), knots=knots)
    schedule = CableParameterSchedule(raw, 4, cable_recipe=recipe)
    return CableSchedulePower(schedule, duration_seconds=duration), schedule, recipe, raw


def positions(radius=(2., 2.)):
    return np.array([[radius[0], 0., 0.], [radius[1], 0., 0.],
                     [0., 0., 0.], [0., 0., 0.]])


def original_sample(raw, fraction, duration, side, index=0):
    """Choose the original segment and differentiate its exact interpolant."""
    fraction = F(fraction)
    knots = raw['knots']
    for number, (first, last) in enumerate(zip(knots, knots[1:])):
        left, right = F(first['fraction']), F(last['fraction'])
        belongs = left <= fraction < right if side == 'right' else left < fraction <= right
        if not belongs:
            continue
        amount = (fraction-left)/(right-left)
        initial, final = first['targetsMeters'][index], last['targetsMeters'][index]
        targets = tuple((1-amount)*F(a)+amount*F(b) for a, b in zip(initial, final))
        rates = tuple((F(b)-F(a))/((right-left)*F(duration)) for a, b in zip(initial, final))
        a0, a1 = F(first['activation'][index]), F(last['activation'][index])
        return targets, (1-amount)*a0+amount*a1, rates, (a1-a0)/((right-left)*F(duration)), number
    raise AssertionError('Oracle requires an admitted one-sided sample')


def taut_segments(radius, targets):
    """Split affine signed radius at zero and its positive residual at zero."""
    r0, r1 = map(F, radius)
    d0, d1 = map(F, targets)
    dr, dd = r1-r0, d1-d0
    cuts = {F(), F(1)}
    if dr and 0 < -r0/dr < 1:
        cuts.add(-r0/dr)
    cuts = sorted(cuts)
    for lo, hi in zip(cuts, cuts[1:]):
        sign = 1 if r0+dr*(lo+hi)/2 >= 0 else -1
        e0, de = sign*r0-d0, sign*dr-dd
        local = [lo, hi]
        if de and lo < -e0/de < hi:
            local.insert(1, -e0/de)
        for left, right in zip(local, local[1:]):
            if e0+de*(left+right)/2 > 0:
                yield left, right, e0, de, sign


def integral_product(first, second, lo, hi):
    result = F()
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            degree = i+j+1
            result += a*b*(hi**degree-lo**degree)/degree
    return result


def exact_power(radius, targets, activation, target_rates, activation_rate, measure=F(1)):
    square, target = F(), F()
    rate = F(target_rates[0]), F(target_rates[1])-F(target_rates[0])
    for lo, hi, e0, de, _ in taut_segments(radius, targets):
        square += integral_product((e0, de), (e0, de), lo, hi)
        target += integral_product((e0, de), rate, lo, hi)
    target = -F(measure)*F(activation)*target
    activation = F(measure)*F(activation_rate)*square/2
    return dict(zip(FIELDS, (target, activation, target+activation)))


def exact_response(radius, targets, activation, measure=F(1)):
    energy, rows = F(), [F() for _ in range(4)]
    for lo, hi, e0, de, sign in taut_segments(radius, targets):
        energy += integral_product((e0, de), (e0, de), lo, hi)/2
        for index, row in enumerate(((F(1), F(-1)), (F(), F(1)),
                                     (F(-1), F(1)), (F(), F(-1)))):
            rows[index] += sign*integral_product((e0, de), row, lo, hi)
    factor = F(measure)*F(activation)
    return energy*factor, tuple(value*factor for value in rows)


class CablePowerTests(unittest.TestCase):
    def assert_enclosed(self, result, expected, tolerance, components=None):
        certificate = result['certificate']
        self.assertIs(certificate['verified'], True)
        self.assertIs(certificate['accepted'], False)
        self.assertEqual(set(certificate['errorsWatts']), set(FIELDS))
        requested = dict.fromkeys(FIELDS, tolerance)
        if components is not None:
            requested.update(components)
        for field in FIELDS:
            self.assertIs(type(result[field]), float, field)
            self.assertTrue(math.isfinite(result[field]), field)
            bound = rational(certificate['errorsWatts'][field])
            self.assertGreaterEqual(bound, 0, field)
            self.assertLessEqual(bound, F(requested[field]), field)
            self.assertLessEqual(abs(F(result[field])-expected[field]), bound, field)
        sum_error = sum((rational(certificate['errorsWatts'][key]) for key in FIELDS), F())
        self.assertLessEqual(abs(F(result[FIELDS[2]])-F(result[FIELDS[0]])-F(result[FIELDS[1]])), sum_error)

    def test_original_binary64_knots_exact_interpolation_and_physical_rates(self):
        middle = dict(fraction=.75, targetsMeters=[[.67, .42]], activation=[.83])
        power, schedule, _, raw = make((.13, .71), (.89, 1.125), .1, .12,
                                       duration=.064, middle=middle)
        for fraction, side in ((F(), 'right'), (F(1, 2**40), 'right'),
                               (F(1, 2), 'left'), (F(3, 4), 'left'),
                               (F(3, 4), 'right'), (F(1), 'left')):
            with self.subTest(fraction=fraction, side=side):
                sample = power.sample(fraction, side=side)
                targets, activation, rates, adot, _ = original_sample(raw, fraction, .064, side)
                self.assertEqual(tuple(map(rational, sample['exactTargetsMeters'][0])), targets)
                self.assertEqual(rational(sample['exactActivation'][0]), activation)
                self.assertEqual(tuple(map(rational, sample['targetRatesMetersPerSecond'][0])), rates)
                self.assertEqual(rational(sample['activationRatesPerSecond'][0]), adot)
                np.testing.assert_array_equal(sample['targetsMeters'], schedule.parameters(fraction)[0])
                np.testing.assert_array_equal(sample['activation'], schedule.parameters(fraction)[1])
                self.assertEqual(sample['scheduleSha256'], hashlib.sha256(encoded(raw)).hexdigest())
        left, right = power.sample(F(3, 4), side='left'), power.sample(F(3, 4), side='right')
        np.testing.assert_array_equal(left['targetsMeters'], right['targetsMeters'])
        self.assertNotEqual(left['targetRatesMetersPerSecond'], right['targetRatesMetersPerSecond'])
        self.assertNotEqual(left['sampleSha256'], right['sampleSha256'])

    def test_trial_uses_right_start_left_end_and_rejects_interior_knot(self):
        middle = dict(fraction=.5, targetsMeters=[[.25, .75]], activation=[1.])
        power, _, _, raw = make(duration=.064, middle=middle)
        for start, end in ((F(), F(1, 2)), (F(1, 2), F(1)), (F(1, 8), F(3, 8))):
            trial = power.trial(start, end)
            self.assertIs(trial['accepted'], False)
            self.assertEqual(rational(trial['durationSeconds']), (end-start)*F(.064))
            self.assertEqual(len(trial['samples']), 3)
            for sample, fraction, side in zip(trial['samples'], (start, (start+end)/2, end),
                                              ('right', 'right', 'left')):
                expected = original_sample(raw, fraction, .064, side)
                self.assertEqual(tuple(map(rational, sample['targetRatesMetersPerSecond'][0])), expected[2])
                self.assertEqual(rational(sample['fraction']), fraction)
        for start, end in ((0, 1), (F(1, 4), F(3, 4)), (F(1, 2), F(1, 2)), (1, 0)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                power.trial(start, end)
        # Endpoint admission alone does not establish midpoint-grid admission.
        with self.assertRaises(ValueError):
            power.trial(F(), F(1, 2**40))

    def test_signed_power_matches_exact_positive_part_polynomial_oracles(self):
        cases = [((.5, 3.), (.75, 2.), (1.5, .5), .25, .75),
                 ((3., .5), (2., .75), (.5, 1.5), .75, .25),
                 ((-2., 3.), (.5, 1.25), (1.5, .25), .25, .75),
                 ((1., 1.), (1., 1.), (.5, 1.5), .75, .25),
                 ((-1., 1.), (.25, .25), (.5, .5), .75, .25)]
        measure = F(.1)/7
        for radius, before, after, a0, a1 in cases:
            power, _, _, raw = make(before, after, a0, a1, duration=.75,
                                    cells=[cell(density=.1, length=F(1, 7))])
            for fraction, side in ((F(), 'right'), (F(1, 2), 'right'), (F(1), 'left')):
                with self.subTest(radius=radius, fraction=fraction):
                    targets, alpha, rates, adot, _ = original_sample(raw, fraction, .75, side)
                    result = power.evaluate(positions(radius), fraction, side=side, absolute_tolerance_watts=1e-11)
                    self.assert_enclosed(result, exact_power(radius, targets, alpha, rates, adot, measure), 1e-11)

    def test_activation_at_zero_has_nonzero_power_and_target_term_is_zero(self):
        power, _, _, raw = make((1., 1.), (.5, .75), 0., 1., duration=.5)
        result = power.evaluate(positions(), 0, side='right', absolute_tolerance_watts=1e-13)
        args = original_sample(raw, 0, .5, 'right')
        self.assert_enclosed(result, exact_power((2., 2.), *args[:4]), 1e-13)
        self.assertEqual(result['targetPowerWatts'], 0.)
        self.assertEqual(result['activationPowerWatts'], 1.)
        self.assertGreater(result['totalPowerWatts'], 0.)

    def test_zero_activation_rate_and_slack_coincident_geometry_are_exact_zero(self):
        for a0, a1 in ((0., 0.), (.25, .75), (.75, .25)):
            power, _, _, _ = make((100., .001), (.001, 100.), a0, a1)
            result = power.evaluate(np.zeros((4, 3)), F(1, 2), side='right', absolute_tolerance_watts=1e-30)
            self.assert_enclosed(result, dict.fromkeys(FIELDS, F()), 1e-30)
            self.assertTrue(all(result[field] == 0. for field in FIELDS))
        power, _, _, _ = make((1., 1.), (.5, .5), 0., 0.)
        result = power.evaluate(positions((3., 3.)), F(1, 2), side='right', absolute_tolerance_watts=1e-30)
        self.assert_enclosed(result, dict.fromkeys(FIELDS, F()), 1e-30)

    def test_nonsquare_radial_power_matches_decimal_antiderivatives(self):
        power, _, _, raw = make((.75, .75), (.5, 1.), .25, .75, duration=.75)
        q = np.array([[0., 1., 0.], [1., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
        targets, alpha, rates, adot, _ = original_sample(raw, 0, .75, 'right')
        with localcontext() as context:
            context.prec = 150
            def dec(value):
                return Decimal(value.numerator)/Decimal(value.denominator)
            root = Decimal(2).sqrt()
            radial0 = (root+(1+root).ln())/2
            radial1 = (2*root-1)/3
            d, a, b0, b1, ap = map(dec, (targets[0], alpha, rates[0], rates[1], adot))
            target = -a*(b0*radial0+(b1-b0)*radial1-d*(b0+b1)/2)
            activation = ap*(Decimal(4)/3+d*d-2*d*radial0)/2
            expected = dict(zip(FIELDS, map(F, (target, activation, target+activation))))
        result = power.evaluate(q, 0, side='right', absolute_tolerance_watts=1e-12)
        self.assert_enclosed(result, expected, 1e-12)

    def test_two_irrational_slack_boundaries_match_decimal_antiderivatives(self):
        target = 65/64
        power, _, _, _ = make((target, target), (target-1/64, target+3/64), .25, .75, duration=.5)
        q = np.array([[-.5, 1., 0.], [.5, 1., 0.], [0., 0., 0.], [0., 0., 0.]])
        with localcontext() as context:
            context.prec = 150
            d, end = Decimal.from_float(target), Decimal('.5')
            root = (d*d-1).sqrt()
            def radial(t):
                r = (1+t*t).sqrt()
                return (t*r+(t+r).ln())/2
            excess = 2*(radial(end)-radial(root)-d*(end-root))
            square = 2*((1+d*d)*(end-root)+(end**3-root**3)/3-2*d*(radial(end)-radial(root)))
            target_power = -Decimal('.25')*Decimal(1)/32*excess
            activation_power = square/2
            expected = dict(zip(FIELDS, map(F, (target_power, activation_power, target_power+activation_power))))
        result = power.evaluate(q, 0, side='right', absolute_tolerance_watts=1e-12)
        self.assert_enclosed(result, expected, 1e-12)

    def test_control_rounding_bounds_cover_exact_energy_and_gradient_discrepancy(self):
        middle = dict(fraction=.75, targetsMeters=[[.67, .42]], activation=[.83])
        power, _, _, raw = make((.13, .71), (.89, 1.125), .1, .12, middle=middle)
        positive_energy_bound, positive_gradient_bound = False, False
        for radius in ((2., 3.), (-1., 2.), (.1, 1.)):
            for fraction in (F(1, 8), F(1, 4), F(1, 2)):
                with self.subTest(radius=radius, fraction=fraction):
                    sample = power.sample(fraction, side='right')
                    targets, alpha, _, _, _ = original_sample(raw, fraction, 1., 'right')
                    original = exact_response(radius, targets, alpha)
                    rounded = exact_response(radius, sample['targetsMeters'][0], sample['activation'][0])
                    result = power.evaluate(positions(radius), fraction, side='right', absolute_tolerance_watts=1e-10)
                    bounds = result['certificate']['controlRoundingBounds']
                    eb, gb = rational(bounds['energyJoules']), rational(bounds['gradientMaxAbsoluteNewtons'])
                    self.assertGreaterEqual(eb, abs(original[0]-rounded[0]))
                    self.assertGreaterEqual(gb, max(abs(a-b) for a, b in zip(original[1], rounded[1])))
                    positive_energy_bound |= eb > 0
                    positive_gradient_bound |= gb > 0
        self.assertTrue(positive_energy_bound)
        self.assertTrue(positive_gradient_bound)
        # At original knots the controls themselves have no rounding defect.
        for fraction, side in ((0, 'right'), (F(3, 4), 'left'), (1, 'left')):
            result = power.evaluate(positions(), fraction, side=side, absolute_tolerance_watts=1e-10)
            bounds = result['certificate']['controlRoundingBounds']
            self.assertEqual(rational(bounds['energyJoules']), 0)
            self.assertEqual(rational(bounds['gradientMaxAbsoluteNewtons']), 0)

    def test_exact_power_uses_continuous_parameters_before_sample_rounding(self):
        adjacent = math.nextafter(1., math.inf)
        middle = dict(fraction=.75, targetsMeters=[[adjacent, adjacent]], activation=[1.])
        power, _, _, raw = make((1., 1.), (adjacent, adjacent), 0., 1., middle=middle)
        fraction = F(1, 2)
        targets, alpha, rates, adot, _ = original_sample(raw, fraction, 1., 'right')
        sample = power.sample(fraction, side='right')
        self.assertNotEqual(targets, tuple(map(F, sample['targetsMeters'][0])))
        self.assertNotEqual(alpha, F(sample['activation'][0]))
        result = power.evaluate(positions(), fraction, side='right', absolute_tolerance_watts=1e-15)
        self.assert_enclosed(result, exact_power((2., 2.), targets, alpha, rates, adot), 1e-15)

    def test_duration_scaling_and_subdivision_do_not_redifferentiate_rounded_samples(self):
        args = ((.13, .71), (.67, .42), .1, .83)
        fast, _, _, raw = make(*args, duration=.125)
        slow, _, _, _ = make(*args, duration=.5)
        fraction = F(3, 8)
        for power, duration in ((fast, .125), (slow, .5)):
            expected = exact_power((2., 3.), *original_sample(raw, fraction, duration, 'right')[:4])
            result = power.evaluate(positions((2., 3.)), fraction, side='right', absolute_tolerance_watts=1e-12)
            self.assert_enclosed(result, expected, 1e-12)
        # A tiny trial still carries the original slope without substituting
        # the subinterval duration for the full schedule duration.
        for sample in fast.trial(F(1, 2), F(1, 2)+F(1, 2**39))['samples']:
            self.assertEqual(tuple(map(rational, sample['targetRatesMetersPerSecond'][0])),
                             original_sample(raw, rational(sample['fraction']), .125, 'right')[2])

    def test_direct_total_preserves_cancellation_below_component_rounding(self):
        power, _, _, _ = make((1., 1.), (1., 1.), 0., 1., cells=[cell(length=F(1, 3))])
        with self.assertRaises(ValueError):
            power.evaluate(positions(), 0, side='right', absolute_tolerance_watts=1e-30)
        alpha = math.nextafter(.5, 0.)
        power, _, _, raw = make((1., 1.), (1.5, 1.5), alpha, 1., duration=.5)
        expected = exact_power((2., 2.), *original_sample(raw, 0, .5, 'right')[:4])
        self.assertEqual(expected['totalPowerWatts'], F(1, 2**53))
        with self.assertRaises(ValueError):
            power.evaluate(positions(), 0, side='right', absolute_tolerance_watts=1e-30)
        components = {FIELDS[0]: 1e-12, FIELDS[1]: 1e-12}
        result = power.evaluate(positions(), 0, side='right', absolute_tolerance_watts=1e-30,
                                component_tolerances_watts=components)
        self.assert_enclosed(result, expected, 1e-30, components)
        self.assertEqual(result['totalPowerWatts'], 2.**-53)
        self.assertNotEqual(F(result['targetPowerWatts'])+F(result['activationPowerWatts']),
                            F(result['totalPowerWatts']))

    def test_stiff_opposite_cells_do_not_erase_small_third_cell_power(self):
        cells = [cell('plus', density=1e12, length=F(100)),
                 cell('minus', density=1e12, length=F(100)), cell('tiny')]
        _, _, recipe, raw = make((1., 1.), (1., 1.), 1., 1., cells=cells)
        adjacent = math.nextafter(1., math.inf)
        raw['knots'][1]['targetsMeters'] = [[.5, .5], [1.5, 1.5], [adjacent, adjacent]]
        power = CableSchedulePower(CableParameterSchedule(raw, 4, cable_recipe=recipe), duration_seconds=1.)
        expected = dict.fromkeys(FIELDS, F())
        for index, measure in enumerate((F(1e14), F(1e14), F(1))):
            contribution = exact_power((2., 2.), *original_sample(raw, 0, 1., 'right', index)[:4], measure)
            for field in FIELDS:
                expected[field] += contribution[field]
        result = power.evaluate(positions(), 0, side='right', absolute_tolerance_watts=1e-30)
        self.assert_enclosed(result, expected, 1e-30)
        self.assertEqual(result['totalPowerWatts'], -2.**-52)

    def test_actual_boundary_budget_exhaustion_fails_closed(self):
        power, _, _, _ = make((1., 1.), (.5, 1.5), .25, .75)
        for budget in ({'max_boundary_depth': 0}, {'max_boundary_panels': 1}):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                power.evaluate(positions((.5, 2.)), 0, side='right', absolute_tolerance_watts=1e-14, **budget)

    def test_component_tolerances_require_complete_explicit_finite_values(self):
        power, _, _, raw = make()
        components = {FIELDS[0]: 1e-10, FIELDS[1]: 1e-9}
        result = power.evaluate(positions(), F(1, 2), side='right', absolute_tolerance_watts=1e-12,
                                component_tolerances_watts=components)
        self.assert_enclosed(result, exact_power((2., 2.), *original_sample(raw, F(1, 2), 1., 'right')[:4]),
                             1e-12, components)
        attacks = [{}, {FIELDS[0]: 1e-10}, dict(components, totalPowerWatts=1e-10), list(components.items())]
        for value in (True, 0., -1., math.nan, math.inf, np.float64(1e-10)):
            attack = components.copy(); attack[FIELDS[0]] = value; attacks.append(attack)
        for attack in attacks:
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                power.evaluate(positions(), 0, side='right', absolute_tolerance_watts=1e-12,
                               component_tolerances_watts=attack)

    def test_constructor_and_sampling_reject_ambiguous_or_out_of_domain_inputs(self):
        power, schedule, _, _ = make()
        for duration in (True, 0., -1., math.inf, math.nan, F(1, 3), np.float64(1.)):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                CableSchedulePower(schedule, duration_seconds=duration)
        for schedule_like in (None, object(), schedule.description()):
            with self.assertRaises(ValueError):
                CableSchedulePower(schedule_like, duration_seconds=1.)
        for fraction in (True, -.1, 1.1, math.inf, F(1, 3), F(1, 2**41), np.float64(.5)):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                power.sample(fraction, side='right')
        for fraction, side in ((0, 'left'), (1, 'right'), (.5, None), (.5, 'both'), (.5, True)):
            with self.subTest(fraction=fraction, side=side), self.assertRaises(ValueError):
                power.sample(fraction, side=side)

    def test_invalid_positions_tolerances_and_budgets_reject(self):
        power, _, _, _ = make(); q = positions()
        for attacked in (q[:2], np.full_like(q, math.nan), [[True, 0., 0.]]+q[1:].tolist()):
            with self.assertRaises(ValueError):
                power.evaluate(attacked, 0, side='right', absolute_tolerance_watts=1e-12)
        if np.dtype(np.longdouble).itemsize > 8:
            with self.assertRaises(ValueError):
                power.evaluate(np.ones((4, 3), dtype=np.longdouble), 0, side='right', absolute_tolerance_watts=1e-12)
        for value in (True, 0., -1., math.inf, math.nan, np.float64(1e-12)):
            with self.subTest(tolerance=value), self.assertRaises(ValueError):
                power.evaluate(q, 0, side='right', absolute_tolerance_watts=value)
        for key, value in (('max_boundary_depth', True), ('max_boundary_panels', 0),
                           ('moment_max_terms', 257), ('moment_max_panels', 1.5), ('moment_max_depth', -1)):
            with self.subTest(budget=key), self.assertRaises(ValueError):
                power.evaluate(q, 0, side='right', absolute_tolerance_watts=1e-12, **{key: value})

    def test_capture_and_all_public_mutations_leave_subsequent_results_unchanged(self):
        power, schedule, _, raw = make()
        q = positions(); before = q.copy(); declaration = copy.deepcopy(raw)
        first = power.sample(F(1, 2), side='right')
        expected = power.evaluate(q, F(1, 2), side='right', absolute_tolerance_watts=1e-12)
        raw['knots'][0]['targetsMeters'][0][0] = 99.
        first['targetsMeters'][0][0] = 88.
        first['activation'][0] = 0.
        first['exactTargetsMeters'][0][0]['numerator'] = '999'
        first['targetRatesMetersPerSecond'][0][0]['numerator'] = '999'
        self.assertEqual(schedule.description()['capturedSchedule'], declaration)
        self.assertEqual(power.evaluate(q, F(1, 2), side='right', absolute_tolerance_watts=1e-12), expected)
        np.testing.assert_array_equal(q, before)
        for name in ('_schedule', '_recipe', '_duration'):
            with self.assertRaises(AttributeError):
                setattr(power, name, None)
            with self.assertRaises(AttributeError):
                delattr(power, name)
        with self.assertRaises(AttributeError):
            power.__init__(schedule, duration_seconds=2.)
        expected['certificate']['errorsWatts'][FIELDS[2]] = rat(999)
        self.assertNotEqual(power.evaluate(q, F(1, 2), side='right', absolute_tolerance_watts=1e-12), expected)

    def test_corrupt_or_failed_schedule_samples_are_counted_and_rejected(self):
        power, _, _, _ = make()
        original = CableParameterSchedule.parameters
        def corrupt(schedule, fraction):
            targets, activation = original(schedule, fraction)
            targets[0, 0] = math.nextafter(targets[0, 0], math.inf)
            return targets, activation
        for effect in (corrupt, RuntimeError('injected sampling failure')):
            with self.subTest(effect=effect), mock.patch.object(CableParameterSchedule, 'parameters', side_effect=effect,
                                                               autospec=True):
                with self.assertRaises(module.CablePowerFailure) as caught:
                    power.evaluate(positions(), 0, side='right', absolute_tolerance_watts=1e-12)
                self.assertEqual(caught.exception.costs['scheduleSamples'], 1)
                self.assertEqual(caught.exception.costs['geometryCalls'], 0)
                self.assertEqual(caught.exception.costs['smoothCalls'], 0)

    def test_geometry_helper_mutation_is_rejected_without_changing_callers_array(self):
        power, _, _, _ = make(); q = positions(); before = q.copy()
        original = ContinuousCableSewing._geometry
        def mutate(cable, index, supplied):
            supplied[0, 0] += 1.
            return original(cable, index, supplied)
        with mock.patch.object(ContinuousCableSewing, '_geometry', mutate):
            with self.assertRaises(module.CablePowerFailure) as caught:
                power.evaluate(q, 0, side='right', absolute_tolerance_watts=1e-12)
        self.assertIn('mutated', str(caught.exception))
        self.assertEqual(caught.exception.costs['geometryCalls'], 1)
        self.assertEqual(caught.exception.costs['smoothCalls'], 0)
        np.testing.assert_array_equal(q, before)

    def test_failed_mutated_or_malformed_integrator_outputs_reject_with_attempt_counts(self):
        power, _, _, _ = make()
        def mutate(q, lo, hi, functionals, tolerances, budgets, stats):
            functionals[FIELDS[0]] = ((F(999),), (F(),))
            raise RuntimeError('injected integration failure after mutation')
        effects = [RuntimeError('injected integration failure'), mutate]
        for record in ({}, {key: (F(1), F()) for key in FIELDS},
                       {key: (0., 0.) for key in FIELDS}):
            effects.append(lambda *args, record=record: record)
        for effect in effects:
            with self.subTest(effect=effect), mock.patch.object(module, '_smooth', side_effect=effect):
                with self.assertRaises(module.CablePowerFailure) as caught:
                    power.evaluate(positions(), 0, side='right', absolute_tolerance_watts=1e-12)
                self.assertEqual(caught.exception.costs['scheduleSamples'], 1)
                self.assertEqual(caught.exception.costs['geometryCalls'], 1)
                self.assertEqual(caught.exception.costs['smoothCalls'], 1)


if __name__ == '__main__':
    unittest.main()
