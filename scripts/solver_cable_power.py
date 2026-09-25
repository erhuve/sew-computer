"""Bounded continuous cable parameter power on an original time schedule.

Power differentiates the exact interpolation of the original binary64 knots,
not the rounded samples used by a native force evaluation. Separate bounds
compare those two controls at the same supplied geometry. This does not advance
time, certify quadrature in time, or admit a timestep or garment.
"""
from bisect import bisect_left, bisect_right
import copy
from fractions import Fraction as F
import hashlib
import json
import math

import numpy as np

from solver_cable_parameter_schedule import CableParameterSchedule, _fraction
from solver_cable_parameters import CableParameterRecipe, _smooth
from solver_continuous_cable_sewing import (
    _add, _scale, _mul, _at, _quadratic_range, _plus_interval, _times_interval,
    _rounded_interval, _error_bound, _budgets,
)
from solver_continuous_normal_sewing import _number, _rat, _rational
from solver_temporal_control import problem_identity


PROFILE = 'original-schedule-cable-power-v1'
POWER_FIELDS = ('targetPowerWatts', 'activationPowerWatts', 'totalPowerWatts')
COMPONENT_FIELDS = POWER_FIELDS[:2]
ZERO = (F(),)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _array_state(value):
    return value.dtype.str, value.shape, value.strides, value.tobytes()


def _ceiling(value):
    """Nonnegative rational bound rounded outward, never down to zero."""
    if value < 0:
        raise ValueError('Nonnegative control-rounding bound required')
    rounded = float(value)
    if not math.isfinite(rounded):
        raise ValueError('Control-rounding bound is outside binary64 range')
    if F(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded):
        raise ValueError('Control-rounding bound ceiling overflow')
    return _rat(F(rounded))


class CablePowerFailure(ValueError):
    """Failed evaluation with detached helper and partition work counts."""
    def __init__(self, reason, costs):
        super().__init__(reason)
        self.costs = copy.deepcopy(costs)


class CableSchedulePower:
    __slots__ = ('_schedule', '_recipe', '_duration', '_knots', '_fractions',
                 '_description_bytes', '_identity', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('Scheduled cable power is immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('Scheduled cable power is immutable')

    def __init__(self, schedule, duration_seconds):
        if getattr(self, '_sealed', False):
            raise AttributeError('Scheduled cable power cannot be reinitialized')
        if type(schedule) is not CableParameterSchedule or type(schedule._recipe) is not CableParameterRecipe:
            raise ValueError('An exact immutable cable schedule and recipe are required')
        if type(duration_seconds) is not float or not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError('A positive finite binary64 total duration is required')
        self._schedule, self._recipe = schedule, schedule._recipe
        self._duration = F(duration_seconds)
        self._identity = problem_identity(None, (self._schedule, self._recipe))
        description = schedule.description()
        self._fractions = schedule.fractions
        self._knots = tuple((
            tuple(tuple(F(float(v)) for v in pair) for pair in knot['targetsMeters']),
            tuple(F(float(a)) for a in knot['activation']))
            for knot in description['capturedSchedule']['knots'])
        self._description_bytes = _encoded({
            'profile': PROFILE, 'scheduleSha256': description['scheduleSha256'],
            'geometrySha256': self._recipe.geometry_sha256, 'cellIds': list(self._recipe.cell_ids),
            'vertexCount': self._recipe.vertex_count, 'durationSeconds': _rat(self._duration),
            'law': 'kL*(activationRate*integral(positivePart(r-d)^2)/2-activation*integral(positivePart(r-d)*targetRate))',
            'ratePolicy': 'Exact original-knot interpolation differentiated with respect to physical seconds; explicit one-sided rates at knots',
            'powerPolicy': 'Original exact interpolants, not the discontinuous binary64 sample map',
            'roundingPolicy': 'Separate fixed-geometry bounds for original versus rounded native control energy and gradient; does not include response integration or constitutive error',
            'accepted': False,
            'scope': 'Generic supplied-cell instantaneous parameter power only; no time-quadrature error bound, trajectory, contact/path admission, source construction or garment acceptance',
        })
        self._check()
        self._sealed = True

    def _check(self):
        if problem_identity(None, (self._schedule, self._recipe)) != self._identity:
            raise ValueError('Scheduled cable geometry or control identity changed')

    def description(self):
        return json.loads(self._description_bytes)

    def _sample(self, fraction, side, costs):
        self._check()
        fraction = _fraction(fraction)
        if type(side) is not str or side not in ('left', 'right'):
            raise ValueError('Explicit left or right one-sided rate required')
        if fraction == 0 and side == 'left' or fraction == 1 and side == 'right':
            raise ValueError('Requested one-sided rate lies outside the original schedule')
        index = (bisect_left(self._fractions, fraction) if side == 'left'
                 else bisect_right(self._fractions, fraction))-1
        lo, hi = self._fractions[index:index+2]
        amount, seconds = (fraction-lo)/(hi-lo), self._duration*(hi-lo)
        before_d, before_a = self._knots[index]
        after_d, after_a = self._knots[index+1]
        exact_d = tuple(tuple((1-amount)*first+amount*last for first, last in zip(a, b))
                        for a, b in zip(before_d, after_d))
        exact_a = tuple((1-amount)*a+amount*b for a, b in zip(before_a, after_a))
        rates_d = tuple(tuple((last-first)/seconds for first, last in zip(a, b))
                        for a, b in zip(before_d, after_d))
        rates_a = tuple((b-a)/seconds for a, b in zip(before_a, after_a))
        costs['scheduleSamples'] += 1
        try:
            rounded_d, rounded_a = self._schedule.parameters(fraction)
        finally:
            self._check()
        expected_d = np.array([[float(v) for v in row] for row in exact_d])
        expected_a = np.array([float(a) for a in exact_a])
        if fraction in self._fractions:
            # Original knot samples preserve their bytes, including signed
            # zero activation. Exact rational arithmetic has no signed zero.
            knot = self._schedule.description()['capturedSchedule']['knots'][self._fractions.index(fraction)]
            expected_d = np.array(knot['targetsMeters'], dtype=np.float64)
            expected_a = np.array(knot['activation'], dtype=np.float64)
        for actual, expected in ((rounded_d, expected_d), (rounded_a, expected_a)):
            if (type(actual) is not np.ndarray or actual.dtype != np.dtype(np.float64)
                    or actual.shape != expected.shape or actual.tobytes() != expected.tobytes()):
                raise ValueError('Native schedule sample differs from original interpolation and one rounding')
        target_error = max(abs(F(float(v))-exact_d[i][j]) for i, row in enumerate(rounded_d)
                           for j, v in enumerate(row))
        activation_error = max(abs(F(float(v))-exact_a[i]) for i, v in enumerate(rounded_a))
        record = {
            'profile': PROFILE, 'scheduleSha256': self.description()['scheduleSha256'],
            'geometrySha256': self._recipe.geometry_sha256,
            'fraction': _rat(fraction), 'timeSeconds': _rat(fraction*self._duration),
            'segmentFractions': [_rat(lo), _rat(hi)], 'side': side,
            'targetsMeters': rounded_d.tolist(), 'activation': rounded_a.tolist(),
            'exactTargetsMeters': [[_rat(v) for v in row] for row in exact_d],
            'exactActivation': [_rat(v) for v in exact_a],
            'targetRatesMetersPerSecond': [[_rat(v) for v in row] for row in rates_d],
            'activationRatesPerSecond': [_rat(v) for v in rates_a],
            'controlRounding': {'targetMaxAbsoluteMetres': _rat(target_error),
                                'activationMaxAbsolute': _rat(activation_error)},
            'accepted': False,
        }
        record['sampleSha256'] = _sha(record)
        record['targetsMeters'], record['activation'] = rounded_d.copy(), rounded_a.copy()
        return record, exact_d, exact_a, rates_d, rates_a

    def sample(self, fraction, *, side):
        try:
            return self._sample(fraction, side, {'scheduleSamples': 0})[0]
        finally:
            self._check()

    def trial(self, start, end):
        """Three original samples on an interval without an interior knot."""
        start, end = _fraction(start), _fraction(end)
        if start >= end or any(start < knot < end for knot in self._fractions):
            raise ValueError('Increasing trial fractions without an interior control knot required')
        # Both endpoints can be legal while their midpoint exceeds the grid.
        midpoint = _fraction((start+end)/2)
        return {'profile': PROFILE, 'accepted': False,
                'durationSeconds': _rat((end-start)*self._duration),
                'samples': [self.sample(start, side='right'), self.sample(midpoint, side='right'),
                            self.sample(end, side='left')],
                'scope': 'Original control samples and rates only; no force evaluation, quadrature, timestep or state publication'}

    def evaluate(self, positions, fraction, *, side, absolute_tolerance_watts,
                 component_tolerances_watts=None, max_boundary_depth=80, max_boundary_panels=256,
                 moment_max_terms=128, moment_max_panels=256, moment_max_depth=64):
        costs = {'scheduleSamples': 0, 'geometryCalls': 0, 'smoothCalls': 0,
                 'visitedBoundaryPanels': 0, 'boundaryLeaves': 0, 'boundaryMaxDepth': 0,
                 'slackCells': 0, 'constantControlCells': 0, 'smoothPanels': 0,
                 'momentPanels': 0, 'maxMomentTerms': 0}
        try:
            return self._evaluate(positions, fraction, side, absolute_tolerance_watts,
                component_tolerances_watts, (max_boundary_depth, max_boundary_panels,
                    moment_max_terms, moment_max_panels, moment_max_depth), costs)
        except Exception as failure:
            raise CablePowerFailure(str(failure), costs) from failure

    def _evaluate(self, positions, fraction, side, total_tolerance, component_tolerances,
                  budget_values, costs):
        self._check()
        if (type(positions) is not np.ndarray or positions.dtype != np.dtype(np.float64)
                or positions.shape != (self._recipe.vertex_count, 3)
                or not np.isfinite(positions).all() or np.any(np.abs(positions) > 1e6)):
            raise ValueError('Finite raw binary64 geometry bounded by 1e6 metres required')
        original = _array_state(positions)
        q = positions.copy()
        snapshot = _array_state(q)
        tolerance = _number(total_tolerance, 0, 1e6, positive=True)
        tolerances = dict.fromkeys(POWER_FIELDS, tolerance)
        if component_tolerances is not None:
            if type(component_tolerances) is not dict or set(component_tolerances) != set(COMPONENT_FIELDS):
                raise ValueError('Explicit target and activation power tolerances required')
            for field, value in component_tolerances.items():
                tolerances[field] = _number(value, 0, 1e6, positive=True)
        budgets = _budgets(*budget_values)
        count = len(self._recipe.cell_ids)
        shares = {field: value/(32*count*budgets['max_boundary_panels']) for field, value in tolerances.items()}
        sums = dict.fromkeys(POWER_FIELDS, (F(), F()))
        energy_rounding, vertex_rounding = F(), {}
        def intact():
            if _array_state(q) != snapshot or _array_state(positions) != original:
                raise ValueError('Cable power helper mutated supplied geometry')
            self._check()

        try:
            sample, targets, activation, target_rates, activation_rates = self._sample(fraction, side, costs)
            for i, measure in enumerate(self._recipe.stiffness_measures):
                costs['geometryCalls'] += 1
                supplied = q.copy()
                supplied_snapshot = _array_state(supplied)
                try:
                    vectors, squared = self._recipe._base._geometry(i, supplied)
                finally:
                    if _array_state(supplied) != supplied_snapshot:
                        raise ValueError('Cable geometry helper mutated supplied geometry')
                    intact()
                d = (targets[i][0], targets[i][1]-targets[i][0])
                dd = (target_rates[i][0], target_rates[i][1]-target_rates[i][0])
                a, ad = activation[i], activation_rates[i]
                rd = tuple(F(float(v)) for v in sample['targetsMeters'][i])
                ra = F(float(sample['activation'][i]))
                # Norm of an affine gap is convex; endpoint L1 norms bound it
                # everywhere without a square root or normalization near r=0.
                rmax = max(sum(abs(_at(v, s)) for v in vectors) for s in (F(), F(1)))
                emax, rounded_emax = max(F(), rmax-min(targets[i])), max(F(), rmax-min(rd))
                delta_d, delta_a = max(abs(x-y) for x, y in zip(targets[i], rd)), abs(a-ra)
                energy_rounding += measure*(delta_a*emax**2+ra*delta_d*(emax+rounded_emax))/2
                for vertex, coefficients in self._recipe._base._rows[i]:
                    cmax = max(abs(_at(coefficients, F())), abs(_at(coefficients, F(1))))
                    bound = measure*cmax*(delta_a*emax+ra*delta_d)
                    vertex_rounding[vertex] = vertex_rounding.get(vertex, F())+bound
                if not ad and (not a or not any(dd)):
                    costs['constantControlCells'] += 1
                    continue
                margin = _add(squared, _scale(_mul(d, d), -1))
                target_poly = _scale(_mul(d, dd), measure*a)
                target_radial = _scale(dd, -measure*a)
                activation_poly = _scale(_add(squared, _mul(d, d)), measure*ad/2)
                activation_radial = _scale(d, -measure*ad)
                functionals = dict(zip(POWER_FIELDS, (
                    (target_poly, target_radial), (activation_poly, activation_radial),
                    (_add(target_poly, activation_poly), _add(target_radial, activation_radial)))))
                pending, leaves, classified = [(F(), F(1), 0)], 1, []
                while pending:
                    lo, hi, depth = pending.pop()
                    costs['visitedBoundaryPanels'] += 1
                    costs['boundaryMaxDepth'] = max(costs['boundaryMaxDepth'], depth)
                    low, high = _quadratic_range(margin, lo, hi)
                    if high <= 0:
                        if lo == 0 and hi == 1:
                            costs['slackCells'] += 1
                        continue
                    if low >= 0:
                        if classified and classified[-1][1] == lo:
                            classified[-1] = classified[-1][0], hi
                        else:
                            classified.append((lo, hi))
                        continue
                    extension = high/(2*min(_at(d, lo), _at(d, hi)))
                    # A mixed panel has e in [0, extension]. Preserve signed
                    # rates; activation power can be nonzero at a=0.
                    rate_lo, rate_hi = sorted((_at(dd, lo), _at(dd, hi)))
                    product = min(F(), extension*rate_lo), max(F(), extension*rate_hi)
                    target_interval = _times_interval(product, -measure*a*(hi-lo))
                    activation_interval = _times_interval((F(), extension**2/2), measure*ad*(hi-lo))
                    intervals = dict(zip(POWER_FIELDS, (target_interval, activation_interval,
                                                       _plus_interval(target_interval, activation_interval))))
                    if any(b-c > shares[field] for field, (c, b) in intervals.items()):
                        if depth >= budgets['max_boundary_depth'] or leaves >= budgets['max_boundary_panels']:
                            raise ValueError('Continuous power active-boundary budget unresolved')
                        middle = (lo+hi)/2
                        pending.extend(((middle, hi, depth+1), (lo, middle, depth+1)))
                        leaves += 1
                    else:
                        costs['boundaryLeaves'] += 1
                        for field, interval in intervals.items():
                            sums[field] = _plus_interval(sums[field], interval)
                for lo, hi in classified:
                    costs['smoothCalls'] += 1
                    costs['smoothPanels'] += 1
                    working = copy.deepcopy(functionals)
                    working_identity = problem_identity(None, (working,))
                    stats = {'momentPanels': 0, 'maxMomentTerms': 0}
                    try:
                        intervals = _smooth(squared, lo, hi, working, dict(shares), dict(budgets), stats)
                    finally:
                        costs['momentPanels'] += stats['momentPanels']
                        costs['maxMomentTerms'] = max(costs['maxMomentTerms'], stats['maxMomentTerms'])
                        if problem_identity(None, (working,)) != working_identity:
                            raise ValueError('Power integration helper mutated its functionals')
                        intact()
                    if (type(intervals) is not dict or set(intervals) != set(POWER_FIELDS)
                            or any(type(pair) is not tuple or len(pair) != 2
                                   or any(type(v) is not F for v in pair) or pair[0] > pair[1]
                                   for pair in intervals.values())):
                        raise ValueError('Exact ordered power integral enclosures required')
                    for field, interval in intervals.items():
                        sums[field] = _plus_interval(sums[field], interval)
            values, errors = {}, {}
            for field in POWER_FIELDS:
                values[field], error = _rounded_interval(sums[field], tolerances[field])
                errors[field] = _error_bound(error, tolerances[field])
            return dict(values, certificate={
                'profile': PROFILE, 'verified': True, 'accepted': False,
                'scheduleSha256': sample['scheduleSha256'], 'sampleSha256': sample['sampleSha256'],
                'geometrySha256': self._recipe.geometry_sha256, 'positionsSha256': _sha(q.tolist()),
                'fraction': sample['fraction'], 'side': side, 'durationSeconds': _rat(self._duration),
                'errorsWatts': errors, 'requestedToleranceWatts': _rat(tolerance),
                'requestedTolerancesWatts': {field: _rat(v) for field, v in tolerances.items()},
                'controlRoundingBounds': {'energyJoules': _ceiling(energy_rounding),
                    'gradientMaxAbsoluteNewtons': _ceiling(max(vertex_rounding.values(), default=F()))},
                'budgets': budgets, 'costs': copy.deepcopy(costs),
                'costScope': 'Schedule, geometry and smooth-helper dispatches are attempted-call counts. Moment-panel/term statistics describe completed helper returns; an internal radial failure can leave only its enclosing smooth-call attempt recorded.',
                'scope': 'Instantaneous power for exact original interpolated controls, with bounded material integration and final rounding. Control-rounding bounds compare exact and rounded control energy/gradient at this geometry only, excluding physical-response numerical error. No bound on quadrature in time, trajectory accuracy or timestep/garment acceptance',
            })
        finally:
            intact()
