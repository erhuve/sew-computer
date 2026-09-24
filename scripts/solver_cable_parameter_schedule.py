"""Immutable supplied-cell cable controls on original dyadic time fractions.

This standalone schedule supplies neither forces nor work and grants no source,
construction, solver or capture admission. The fixed CableControl is unchanged.
"""
from bisect import bisect_left
from fractions import Fraction as F
import hashlib
import json
import math

import numpy as np

from solver_cable_parameters import CableParameterRecipe


PROFILE = 'cable-target-activation-v1'
MAX_CELLS = 4096
MAX_KNOTS = 65
MAX_FRACTION_DENOMINATOR = 2**40
MAX_RAW_BYTES = 32*1024**2
COEFFICIENT_POLICY = ('Exact product of sampled binary64 activation, binary64 density and '
                      'rational reference length; positive beta must round finite and positive, '
                      'but beta is not rounded in the cable law')


def _encoded(value):
    try:
        result = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError('Finite raw JSON cable schedule required') from error
    if len(result) > MAX_RAW_BYTES:
        raise ValueError('Cable schedule exceeds byte budget')
    return result


def _rat(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def _fraction(value):
    if type(value) not in (int, float, F):
        raise ValueError('Raw finite non-Boolean dyadic fraction required')
    if type(value) is float and not math.isfinite(value):
        raise ValueError('Raw finite non-Boolean dyadic fraction required')
    if type(value) is F and max(value.numerator.bit_length(), value.denominator.bit_length()) > 41:
        raise ValueError('Fraction exceeds the bounded dyadic grid')
    if not 0 <= value <= 1:
        raise ValueError('Control fraction must be in [0,1]')
    result = F(value)
    if result.denominator > MAX_FRACTION_DENOMINATOR or result.denominator & (result.denominator-1):
        raise ValueError('Control fraction must lie on the bounded dyadic grid')
    return result


def _number(value, *, target):
    if type(value) not in (int, float):
        raise ValueError('Raw non-Boolean finite binary64 control number required')
    if not (0 < value <= 100 if target else 0 <= value <= 1):
        raise ValueError('Positive targets at most100m and activation in [0,1] required')
    number = float(value)
    if not math.isfinite(number) or type(value) is int and int(number) != value:
        raise ValueError('Exactly representable finite binary64 control number required')
    return number


def _raw_values(targets, activation, count):
    if (type(targets) is not list or len(targets) != count
            or any(type(row) is not list or len(row) != 2 for row in targets)
            or type(activation) is not list or len(activation) != count):
        raise ValueError('Exactly two raw targets and one activation per ordered cell required')
    for row in targets:
        for value in row:
            _number(value, target=True)
    for value in activation:
        _number(value, target=False)


def _packed(value):
    array = np.asarray(value, dtype=np.float64)
    # Even a read-only ndarray permits shape/strides/dtype mutation. Retain
    # only immutable bytes and tuple metadata; construct disposable local
    # views for each operation instead of retaining an ndarray object.
    return array.tobytes(), tuple(array.shape)


class CableParameterSchedule:
    """Stateless controls; an optional preflight proves only its named retry grid."""

    __slots__ = ('_recipe', '_recipe_bytes', '_description_bytes', '_fractions',
                 '_target_bytes', '_target_shape', '_activation_bytes', '_activation_shape',
                 '_cell_ids', '_geometry_sha256', '_subdivisions')

    def __setattr__(self, name, value):
        raise AttributeError('Cable parameter schedules are immutable')

    def __delattr__(self, name):
        raise AttributeError('Cable parameter schedules are immutable')

    def __init__(self, raw, subdivisions, *, cable_recipe):
        if hasattr(self, '_recipe_bytes'):
            raise AttributeError('Cable parameter schedules cannot be reinitialized')
        if type(cable_recipe) is not CableParameterRecipe:
            raise ValueError('An exact immutable CableParameterRecipe is required')
        if (type(subdivisions) is not int or not 1 <= subdivisions <= 4096
                or subdivisions & (subdivisions-1)):
            raise ValueError('Power-of-two initial subdivisions in [1,4096] required')
        ids = cable_recipe.cell_ids
        if (type(raw) is not dict or set(raw) != {'profile','geometrySha256','cellIds','knots'}
                or type(raw['profile']) is not str or raw['profile'] != PROFILE
                or type(raw['geometrySha256']) is not str
                or raw['geometrySha256'] != cable_recipe.geometry_sha256
                or type(raw['cellIds']) is not list or not 1 <= len(raw['cellIds']) <= MAX_CELLS
                or any(type(item) is not str or not 1 <= len(item) <= 128 for item in raw['cellIds'])
                or tuple(raw['cellIds']) != ids or len(set(raw['cellIds'])) != len(ids)
                or type(raw['knots']) is not list or not 2 <= len(raw['knots']) <= MAX_KNOTS):
            raise ValueError('Exact cable schedule profile, geometry identity and complete ordered cells required')
        fractions = []
        for knot in raw['knots']:
            if (type(knot) is not dict or set(knot) != {'fraction','targetsMeters','activation'}
                    or type(knot['fraction']) not in (int,float)):
                raise ValueError('Each raw knot requires fraction, targetsMeters and activation only')
            fraction = _fraction(knot['fraction'])
            if (fraction*subdivisions).denominator != 1 or fractions and fraction <= fractions[-1]:
                raise ValueError('Strictly ordered knots on the original subdivision grid required')
            _raw_values(knot['targetsMeters'], knot['activation'], len(ids))
            fractions.append(fraction)
        if fractions[0] != 0 or fractions[-1] != 1:
            raise ValueError('Cable control knots must span exactly [0,1]')
        # Shape and every raw scalar are bounded before JSON allocation/capture.
        captured = _encoded(raw)
        detached = json.loads(captured)
        targets, activation = [], []
        for knot in detached['knots']:
            values, weights = cable_recipe.parameters(knot['targetsMeters'], knot['activation'])
            targets.append(values.copy()); activation.append(weights.copy())
        object.__setattr__(self, '_recipe', cable_recipe)
        object.__setattr__(self, '_recipe_bytes', captured)
        object.__setattr__(self, '_fractions', tuple(fractions))
        target_bytes, target_shape = _packed(targets)
        activation_bytes, activation_shape = _packed(activation)
        object.__setattr__(self, '_target_bytes', target_bytes)
        object.__setattr__(self, '_target_shape', target_shape)
        object.__setattr__(self, '_activation_bytes', activation_bytes)
        object.__setattr__(self, '_activation_shape', activation_shape)
        object.__setattr__(self, '_cell_ids', tuple(ids))
        object.__setattr__(self, '_geometry_sha256', cable_recipe.geometry_sha256)
        object.__setattr__(self, '_subdivisions', subdivisions)
        description = {
            'profile': PROFILE, 'scheduleSha256': hashlib.sha256(captured).hexdigest(),
            'geometrySha256': self._geometry_sha256, 'cellIds': list(ids),
            'vertexCount': cable_recipe.vertex_count, 'subdivisions': subdivisions,
            'capturedSchedule': detached,
            'interpolationPolicy': 'Original exact dyadic fraction; exact interpolation of original binary64 endpoints followed by one nearest-even rounding; knot bytes preserved',
            'coefficientPolicy': COEFFICIENT_POLICY,
            'accepted': False, 'sourceAdmissionGranted': False, 'controlsInstalled': False,
            'scope': 'Generic supplied-cell parameters only; no force, work, motion, source correspondence, construction or captured-run admission. Initial parameters are explicitly supplied; an initial jump is neither executed nor accounted here.',
        }
        object.__setattr__(self, '_description_bytes', _encoded(description))

    @property
    def cell_ids(self):
        return self._cell_ids

    @property
    def geometry_sha256(self):
        return self._geometry_sha256

    @property
    def fractions(self):
        return self._fractions

    @property
    def subdivisions(self):
        return self._subdivisions

    def description(self):
        return json.loads(self._description_bytes)

    def _arrays(self):
        return (np.frombuffer(self._target_bytes, dtype=np.float64).reshape(self._target_shape),
                np.frombuffer(self._activation_bytes, dtype=np.float64).reshape(self._activation_shape))

    def parameters(self, fraction):
        fraction = _fraction(fraction)
        target_array, activation_array = self._arrays()
        upper = bisect_left(self._fractions, fraction)
        if upper < len(self._fractions) and fraction == self._fractions[upper]:
            values, weights = target_array[upper].copy(), activation_array[upper].copy()
        else:
            lower = upper-1
            amount = (fraction-self._fractions[lower])/(self._fractions[upper]-self._fractions[lower])
            outputs = []
            for is_activation, array in ((False,target_array),(True,activation_array)):
                result = np.empty(array.shape[1:], dtype=np.float64)
                for index in np.ndindex(result.shape):
                    first, last = float(array[(lower,)+index]), float(array[(upper,)+index])
                    exact = (1-amount)*F(first)+amount*F(last)
                    converted = float(exact)
                    if is_activation and exact > 0 and converted == 0:
                        raise ValueError('Positive interpolated cable activation underflows to inactive')
                    result[index] = converted
                outputs.append(result)
            values, weights = outputs
        # Revalidate every sample, including those on a deeper grid than any
        # prior preflight. Exact beta admission belongs to the shared recipe.
        values, weights = self._recipe.parameters(values, weights)
        return values.copy(), weights.copy()

    def preflight(self, max_depth):
        if (type(max_depth) is not int or not 0 <= max_depth <= 30
                or self._subdivisions.bit_length()-1+max_depth > 40):
            raise ValueError('Retry depth in [0,30] and dyadic grid denominator at most2^40 required')
        denominator = self._subdivisions*(2**max_depth)
        increment = F(1,denominator)
        target_array, activation_array = self._arrays()
        minimum_values = np.zeros(len(self._cell_ids), dtype=np.float64)
        witnesses = []
        for column, identity in enumerate(self._cell_ids):
            candidates = {}
            for index, (lower,upper) in enumerate(zip(self._fractions,self._fractions[1:])):
                first, last = F(float(activation_array[index,column])), F(float(activation_array[index+1,column]))
                if not first and not last:
                    continue
                probes = [lower,upper]
                if upper-lower > increment:
                    if not first: probes.append(lower+increment)
                    if not last: probes.append(upper-increment)
                for fraction in probes:
                    amount = (fraction-lower)/(upper-lower)
                    exact = (1-amount)*first+amount*last
                    if exact > 0:
                        numerical = float(exact)
                        if numerical == 0:
                            raise ValueError('Positive cable activation underflows on the declared retry grid')
                        candidates[fraction] = (numerical,exact)
            item = {'cellId':identity, 'activeOnGrid':bool(candidates),
                    'positiveCandidateCount':len(candidates), 'minimumPositiveActivation':None}
            if candidates:
                fraction, (numerical,exact) = min(candidates.items(), key=lambda pair:(pair[1][0],pair[0]))
                minimum_values[column] = numerical
                beta = F(numerical)*self._recipe.stiffness_measures[column]
                admitted = float(beta)
                if not math.isfinite(admitted) or admitted <= 0:
                    raise ValueError('Positive cable beta underflows on the declared retry grid')
                item['minimumPositiveActivation'] = {
                    'fraction':_rat(fraction), 'exactInterpolatedActivation':_rat(exact),
                    'numericalActivation':numerical,
                    'interpolationRoundingResidual':_rat(F(numerical)-exact),
                    'exactEffectiveStiffnessNPerM':_rat(beta),
                    'binary64AdmissionValueNPerM':admitted,
                    'admissionRoundingResidualNPerM':_rat(F(admitted)-beta),
                }
            witnesses.append(item)
        # Per-cell minima may occur at different fractions. This vector is a
        # coefficient-admission probe only, not an actual sampled control state.
        self._recipe.parameters(target_array[0].copy(), minimum_values)
        return {
            'profile':'cable-parameter-grid-preflight-v1', 'verified':True, 'accepted':False,
            'sourceAdmissionGranted':False, 'controlsInstalled':False,
            'scheduleSha256':hashlib.sha256(self._recipe_bytes).hexdigest(),
            'geometrySha256':self._geometry_sha256, 'subdivisions':self._subdivisions,
            'maxDepth':max_depth, 'gridDenominator':denominator, 'gridPointCount':denominator+1,
            'minimumFractionStep':_rat(increment), 'cellWitnesses':witnesses,
            'coefficientPolicy':COEFFICIENT_POLICY,
            'scope':'Minimum positive rounded activation and exact beta admission only on the named finite dyadic retry grid. Endpoint/adjacent-zero witnesses avoid grid enumeration; a selected witness is not necessarily the earliest full-grid minimizer on a rounding plateau. No time integration, force, work or construction proof.',
        }
