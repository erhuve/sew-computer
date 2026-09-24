"""Immutable supplied-cell parameters and bounded discrete cable work.

No control is installed or moved here. Work is a target-first parameter jump
at one supplied position, not a continuous actuator trajectory or seam proof.
"""
from fractions import Fraction as F
import hashlib
import json

import numpy as np

from solver_continuous_cable_sewing import (
    ContinuousCableSewing, PROFILE, _add, _scale, _mul, _at, _compose,
    _integral, _quadratic_range, _plus_interval, _times_interval,
    _rounded_interval, _error_bound, _budgets,
)
from solver_continuous_normal_sewing import (
    _capture, _encoded, _number, _positive_float, _rat, _rational,
)
from solver_sewing_activation import _contains_bool
from solver_radial_moments import radial_moment_bounds


WORK_FIELDS = ('targetWorkJoules', 'activationWorkJoules', 'totalWorkJoules',
               'activationIncreaseWorkJoules', 'releaseEnergyRemovedJoules')
COMPONENT_FIELDS = tuple(name for name in WORK_FIELDS if name != 'totalWorkJoules')
PARAMETER_ORDER = 'target-first-at-old-activation-then-activation-at-new-target'
ZERO = (F(),)


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _array(value, shape, *, activation):
    if _contains_bool(value):
        raise ValueError('Boolean cable parameters are not numerical controls')
    try:
        raw = np.asarray(value)
        if (raw.shape != shape or raw.dtype.kind not in 'fiu'
                or raw.dtype.kind == 'f' and raw.dtype.itemsize > 8):
            raise ValueError('Matching real cable parameters without extended-precision narrowing required')
        result = np.array(raw, dtype=np.float64, copy=True)
        if raw.dtype.kind in 'iu' and any(int(a) != F(float(b)) for a, b in zip(raw.flat, result.flat)):
            raise ValueError('Cable integer parameters must be exactly representable in binary64')
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Matching finite binary64 cable parameter arrays required') from error
    lower_ok = result >= 0 if activation else result > 0
    if not np.isfinite(result).all() or not lower_ok.all() or np.any(result > (1 if activation else 100)):
        raise ValueError('Cable targets must be in (0,100]m and activation in [0,1]')
    return result


def _polynomials(q, targets, states, weights):
    """Combine shared radial coefficients before interval multiplication."""
    result = {}
    for field, pair in weights.items():
        polynomial, radial = ZERO, ZERO
        for d, state, weight in zip(targets, states, pair):
            if state == 1 and weight:
                polynomial = _add(polynomial, _scale(_add(q, _mul(d, d)), weight/2))
                radial = _add(radial, _scale(d, -weight))
        result[field] = polynomial, radial
    return result


def _merge_functionals(destination, source):
    for field, (poly, radial) in source.items():
        old_poly, old_radial = destination.get(field, (ZERO, ZERO))
        destination[field] = _add(old_poly, poly), _add(old_radial, radial)


def _boundary(q, targets, margins, states, weights, lo, hi):
    # U_j = 1/2 integral (sqrt(Q)-d_j)_+^2.  The exact inequality
    # (sqrt(Q)-d)_+ <= max(Q-d^2,0)/(2 min(d)) needs no square root.
    bounds = []
    for d, margin, state in zip(targets, margins, states):
        maximum = max(F(), _quadratic_range(margin, lo, hi)[1])
        minimum = min(_at(d, lo), _at(d, hi))
        bounds.append(F() if state == 0 else (hi-lo)*maximum**2/(8*minimum**2))
    return {field: _plus_interval(_times_interval((F(), bounds[0]), pair[0]),
                                  _times_interval((F(), bounds[1]), pair[1]))
            for field, pair in weights.items()}


def _smooth(q, lo, hi, functionals, tolerances, budgets, stats):
    width = hi-lo
    local_q = _compose(q, lo, width)
    local_q = local_q+(F(),)*(3-len(local_q))
    local = {field: (_compose(poly, lo, width), _compose(radial, lo, width))
             for field, (poly, radial) in functionals.items()}
    requests = [tolerances[field]/(8*width*sum(map(abs, radial), F()))
                for field, (_, radial) in local.items() if any(radial)]
    if requests:
        degree = max(len(radial)-1 for _, radial in local.values() if any(radial))
        moments = radial_moment_bounds(local_q, (F(1, 2),), degree, min(requests),
            max_panels=budgets['moment_max_panels'], max_terms=budgets['moment_max_terms'],
            max_depth=budgets['moment_max_depth'])
        if moments.get('verified') is not True:
            raise ValueError('Verified parameter-work radial moments required')
        stats['momentPanels'] += moments['panels']
        stats['maxMomentTerms'] = max(stats['maxMomentTerms'], moments['maxTerms'])
    result = {}
    for field, (polynomial, radial) in local.items():
        value = _integral(polynomial)
        interval = value, value
        for i, coefficient in enumerate(radial):
            if coefficient:
                interval = _plus_interval(interval, _times_interval(moments['moments'][F(1, 2)][i], coefficient))
        result[field] = _times_interval(interval, width)
    return result


class CableParameterRecipe:
    """Separate immutable geometry; effective potentials remain immutable too."""
    __slots__ = ('_base', '_geometry_bytes', '_description_bytes', '_measures', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('Cable parameter recipes are immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('Cable parameter recipes are immutable')

    def __init__(self, potential):
        if type(potential) is not ContinuousCableSewing:
            raise ValueError('An explicit continuous cable potential is required')
        self._base = ContinuousCableSewing(potential.vertex_count, potential.cells)
        cells = self._base.cells
        geometry = {'vertexCount': self.vertex_count, 'cells': [
            {key: value for key, value in cell.items() if key not in ('targetsMeters', 'activation')}
            for cell in cells]}
        self._geometry_bytes = _capture(geometry)
        self._measures = tuple(_number(cell['stiffnessDensityNPerM2'], 0, 1e12, positive=True)
                               *_rational(cell['referenceLengthMeters']) for cell in cells)
        self._description_bytes = _encoded({
            'profile': 'supplied-cell-cable-parameter-recipe-v1', 'law': PROFILE,
            'geometrySha256': self.geometry_sha256, 'basePotentialSha256': self._base.description()['inputSha256'],
            'vertexCount': self.vertex_count, 'cellCount': len(cells), 'cellIds': list(self.cell_ids),
            'initialParameterSha256': self.parameter_sha256(*self.initial_parameters),
            'betaPolicy': self._base.description()['betaPolicy'], 'parameterOrder': PARAMETER_ORDER,
            'anchorPolicy': 'Preserve independently rounded original unit anchors without renormalization; numerical translation/net-force defects remain.',
            'identityPolicy': 'Geometry retains original raw JSON fields; numerical parameter hashes bind normalized binary64 arrays, separately from raw base/schedule provenance.',
            'accepted': False, 'sourceControlsInstalled': False, 'sourceAdmissionGranted': False,
            'scope': 'Supplied immutable geometry and discrete parameter work only; no schedule execution, motion, contact, source construction or physical seam acceptance',
        })
        self._sealed = True

    @property
    def vertex_count(self):
        return self._base.vertex_count

    @property
    def cell_ids(self):
        return tuple(cell['id'] for cell in self._base.cells)

    @property
    def geometry_sha256(self):
        return hashlib.sha256(self._geometry_bytes).hexdigest()

    @property
    def stiffness_measures(self):
        return self._measures

    @property
    def initial_parameters(self):
        cells = self._base.cells
        return (np.array([cell['targetsMeters'] for cell in cells], dtype=np.float64),
                np.array([cell['activation'] for cell in cells], dtype=np.float64))

    def description(self):
        return json.loads(self._description_bytes)

    def parameters(self, targets, activation):
        count = len(self._measures)
        d = _array(targets, (count, 2), activation=False)
        a = _array(activation, (count,), activation=True)
        for alpha, measure in zip(a, self._measures):
            beta = F(float(alpha))*measure
            if beta:
                _positive_float(beta)
        return d, a

    def parameter_sha256(self, targets, activation):
        d, a = self.parameters(targets, activation)
        return _sha({'targetsMeters': d.tolist(), 'activation': a.tolist()})

    def potential(self, targets, activation):
        d, a = self.parameters(targets, activation)
        cells = self._base.cells
        for cell, distances, alpha in zip(cells, d, a):
            cell['targetsMeters'], cell['activation'] = distances.tolist(), float(alpha)
        return ContinuousCableSewing(self.vertex_count, cells)

    def parameter_energy_change(self, positions, before_targets, before_activation,
                                after_targets, after_activation, *, absolute_tolerance_joules,
                                component_tolerances_joules=None,
                                max_boundary_depth=80, max_boundary_panels=256,
                                moment_max_terms=128, moment_max_panels=256, moment_max_depth=64):
        q = self._base._positions(positions)
        d0, a0 = self.parameters(before_targets, before_activation)
        d1, a1 = self.parameters(after_targets, after_activation)
        before, after = self.potential(d0, a0), self.potential(d1, a1)
        total_tolerance = _number(absolute_tolerance_joules, 0, 1e6, positive=True)
        tolerances = {field: total_tolerance for field in WORK_FIELDS}
        if component_tolerances_joules is not None:
            if type(component_tolerances_joules) is not dict or set(component_tolerances_joules) != set(COMPONENT_FIELDS):
                raise ValueError('Explicit tolerances for all four work components required')
            for field, value in component_tolerances_joules.items():
                tolerances[field] = _number(value, 0, 1e6, positive=True)
        budgets = _budgets(max_boundary_depth, max_boundary_panels, moment_max_terms,
                           moment_max_panels, moment_max_depth)
        count = max(1, sum(bool(x or y) for x, y in zip(a0, a1)))
        shares = {field: value/(32*count) for field, value in tolerances.items()}
        sums = {field: (F(), F()) for field in WORK_FIELDS}
        groups = {}
        stats = {'inactiveCells': 0, 'unchangedCells': 0, 'boundaryLeaves': 0,
                 'boundaryMaxDepth': 0, 'classifiedSmoothPanels': 0, 'combinedSmoothPanels': 0,
                 'momentPanels': 0, 'maxMomentTerms': 0}
        for i, (beta0, beta1) in enumerate(zip(before._beta, after._beta)):
            if not beta0 and not beta1:
                stats['inactiveCells'] += 1
                continue
            if before._targets[i] == after._targets[i] and beta0 == beta1:
                stats['unchangedCells'] += 1
                continue
            targets = before._targets[i], after._targets[i]
            weights = {
                'targetWorkJoules': (-beta0, beta0),
                'activationWorkJoules': (F(), beta1-beta0),
                'totalWorkJoules': (-beta0, beta1),
                'activationIncreaseWorkJoules': (F(), max(F(), beta1-beta0)),
                'releaseEnergyRemovedJoules': (F(), max(F(), beta0-beta1)),
            }
            # Equal target functions refer to the same energy, including its
            # boundary slivers. Combine before bounding so a known zero target
            # change stays exact and tiny activation differences stay small.
            if targets[0] == targets[1]:
                weights = {field: (sum(pair, F()), F()) for field, pair in weights.items()}
            relevant = tuple(any(pair[j] for pair in weights.values()) for j in range(2))
            _, squared = self._base._geometry(i, q)
            margins = tuple(_add(squared, _scale(_mul(d, d), -1)) for d in targets)
            pending, classified = [(F(), F(1), 0)], []
            leaves, mixed = 1, 0
            while pending:
                lo, hi, depth = pending.pop()
                stats['boundaryMaxDepth'] = max(stats['boundaryMaxDepth'], depth)
                ranges = [_quadratic_range(margin, lo, hi) if needed else (F(), F())
                          for margin, needed in zip(margins, relevant)]
                states = tuple(0 if high <= 0 else 1 if low >= 0 else -1 for low, high in ranges)
                if -1 in states:
                    enclosure = _boundary(squared, targets, margins, states, weights, lo, hi)
                    if any(high-low > shares[field] for field, (low, high) in enclosure.items()):
                        if depth >= budgets['max_boundary_depth'] or leaves >= budgets['max_boundary_panels']:
                            raise ValueError('Cable parameter-work active-boundary budget unresolved')
                        middle = (lo+hi)/2
                        pending.extend(((middle, hi, depth+1), (lo, middle, depth+1)))
                        leaves += 1
                        continue
                    mixed += 1
                    if mixed > 4:
                        raise ValueError('Two quadratic cable margins require at most four boundary leaves')
                    stats['boundaryLeaves'] += 1
                    for field, interval in enclosure.items():
                        sums[field] = _plus_interval(sums[field], interval)
                    classified.append((None, lo, hi))
                elif classified and classified[-1][0] == states and classified[-1][2] == lo:
                    classified[-1] = states, classified[-1][1], hi
                else:
                    classified.append((states, lo, hi))
            for states, lo, hi in classified:
                if states is None or states == (0, 0):
                    continue
                stats['classifiedSmoothPanels'] += 1
                functionals = _polynomials(squared, targets, states, weights)
                _merge_functionals(groups.setdefault((squared, lo, hi), {}), functionals)
        for (squared, lo, hi), functionals in sorted(groups.items()):
            intervals = _smooth(squared, lo, hi, functionals, shares, budgets, stats)
            stats['combinedSmoothPanels'] += 1
            for field, interval in intervals.items():
                sums[field] = _plus_interval(sums[field], interval)
        for field in ('activationIncreaseWorkJoules', 'releaseEnergyRemovedJoules'):
            lo, hi = sums[field]
            if hi < 0:
                raise ValueError('Nonnegative activation-stage work enclosure inconsistent')
            sums[field] = max(F(), lo), hi
        values, errors = {}, {}
        for field in WORK_FIELDS:
            values[field], error = _rounded_interval(sums[field], tolerances[field])
            errors[field] = _error_bound(error, tolerances[field])
        return dict(values, certificate={
            'profile': 'continuous-cable-parameter-work-enclosure-v1', 'law': PROFILE,
            'geometrySha256': self.geometry_sha256, 'positionsSha256': _sha(q.tolist()),
            'beforeParameterSha256': self.parameter_sha256(d0, a0),
            'afterParameterSha256': self.parameter_sha256(d1, a1),
            'beforePotentialSha256': before.description()['inputSha256'],
            'afterPotentialSha256': after.description()['inputSha256'],
            'parameterOrder': PARAMETER_ORDER, 'errorsJoules': errors,
            'requestedToleranceJoules': _rat(total_tolerance),
            'requestedTolerancesJoules': {field: _rat(value) for field, value in tolerances.items()},
            'budgets': budgets, 'stats': stats, 'verified': True, 'accepted': False,
            'sourceControlsInstalled': False, 'sourceAdmissionGranted': False,
            'scope': 'Bounded discrete target-first work at one supplied state, including output rounding; no continuous actuator work, dynamics, contact, source or physical acceptance',
        })
