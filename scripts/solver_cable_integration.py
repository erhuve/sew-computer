"""Validated fixed cable controls and conditional numerical decision bounds.

Cable uncertainty is certified. Other solver terms remain their existing
binary64 numerical evaluations; this does not certify all physical forces.
"""
import copy
from fractions import Fraction as F
import hashlib
import math

import numpy as np
from scipy.sparse import isspmatrix_csr

from solver_continuous_cable_sewing import ContinuousCableSewing, PROFILE, _budgets
from solver_continuous_normal_sewing import _capture, _encoded, _number, _rat, _rational


TOLERANCES = ('energy_tolerance_joules', 'gradient_tolerance_newtons',
              'hessian_tolerance_newtons_per_meter', 'absolute_tolerance_joules')
BUDGETS = ('max_boundary_depth', 'max_boundary_panels', 'moment_max_terms',
           'moment_max_panels', 'moment_max_depth')
STATIONARITY_SCOPE = ('Cable-error-aware stationarity relative to existing binary64 non-cable '
                     'gradient terms; includes cable addition rounding, not a certificate of all physical forces')


def positions_sha256(positions):
    return hashlib.sha256(_encoded(positions.tolist())).hexdigest()


def round_sum(values, error=F()):
    """One final rounding of finite binary64 terms, with conditional error."""
    values = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in values) or error < 0:
        raise ValueError('Finite summands and nonnegative error required')
    exact = sum((F(value) for value in values), F())
    try:
        rounded = float(exact)
    except OverflowError as failure:
        raise ValueError('Assembled cable response is outside binary64 range') from failure
    if not math.isfinite(rounded):
        raise ValueError('Assembled cable response is outside binary64 range')
    return rounded, error+abs(F(rounded)-exact)


def assemble_gradient(baseline, cable, cable_error):
    if baseline.shape != cable.shape or baseline.ndim != 1:
        raise ValueError('Matching flat cable and baseline gradients required')
    values, rounding = [], F()
    for first, second in zip(baseline, cable):
        value, error = round_sum((first, second))
        values.append(value)
        rounding = max(rounding, error)
    return np.asarray(values), cable_error+rounding, rounding


def stationarity(gradient, error, cable_error, assembly_error, *, tolerance_newtons=1e-6):
    from solver_stationarity import stationarity_tolerance
    tolerance = stationarity_tolerance(tolerance_newtons)
    if (not np.isfinite(gradient).all() or min(error, cable_error, assembly_error) < 0
            or error != cable_error+assembly_error):
        raise ValueError('Consistent finite cable gradient bounds required')
    norm = float(np.max(np.abs(gradient), initial=0))
    upper = F(norm)+error
    return {'gradientInfinityNorm': norm, 'cableGradientErrorBoundNewtons': _rat(cable_error),
            'assemblyRoundingBoundNewtons': _rat(assembly_error),
            'totalGradientErrorBoundNewtons': _rat(error),
            'stationarityUpperBoundNewtons': _rat(upper), 'toleranceNewtons': tolerance,
            'scope': STATIONARITY_SCOPE}


def directional_interval(gradient, error, start, end):
    """Use the exact difference of actual binary endpoints, not rounded delta."""
    if (gradient.shape != start.shape or start.shape != end.shape or gradient.ndim != 1
            or not all(np.isfinite(value).all() for value in (gradient, start, end)) or error < 0):
        raise ValueError('Finite matching directional states and error required')
    delta = [F(float(last))-F(float(first)) for first, last in zip(start, end)]
    midpoint = sum((F(float(value))*step for value, step in zip(gradient, delta)), F())
    radius = error*sum(map(abs, delta), F())
    return midpoint-radius, midpoint+radius


class CableControl:
    """Immutable recipe and fixed explicit precision; unresolved decisions reject."""
    __slots__ = ('_potential', '_policy_bytes', '_description_bytes', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('Integrated cable controls are immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('Integrated cable controls are immutable')

    def __init__(self, potential, precision):
        if type(potential) is not ContinuousCableSewing:
            raise ValueError('An explicit standalone continuous cable recipe is required')
        if type(precision) is not dict or set(precision) != set(TOLERANCES+BUDGETS):
            raise ValueError('All four cable tolerances and five resource budgets must be explicit')
        self._policy_bytes = _capture(precision)
        for field in TOLERANCES:
            _number(precision[field], 0, 1e6, positive=True)
        _budgets(**{field: precision[field] for field in BUDGETS})
        self._potential = ContinuousCableSewing(potential.vertex_count, potential.cells)
        self._description_bytes = _encoded({
            'profile': 'guarded-fixed-cable-control-v1', 'law': PROFILE,
            'inputSha256': self._potential.description()['inputSha256'],
            'precisionSha256': hashlib.sha256(self._policy_bytes).hexdigest(),
            'vertexCount': potential.vertex_count, 'precision': self.policy,
            'numericalConversions': self._potential.description()['conversions'],
            'anchorPolicy': 'Independently rounded unit anchors without renormalization; signed coefficient-sum defects can produce tiny translation and force-balance defects in the numerical law',
            'parametersFixed': True, 'accepted': False, 'sourceAdmissionGranted': False,
            'precisionPolicy': 'Fixed explicit absolute limits; unresolved integration or decisions reject without relaxing a tolerance',
            'hessianPolicy': 'slack-sided-generalized-curvature; entrywise bounded, rounded PSD not certified',
            'scope': 'Generic model vertex membership only; no source ownership, construction, material calibration or garment acceptance',
        })
        self._sealed = True

    @property
    def policy(self):
        import json
        return json.loads(self._policy_bytes)

    @property
    def vertex_count(self):
        return self._potential.vertex_count

    def description(self):
        import json
        return json.loads(self._description_bytes)

    def positions(self, positions):
        return self._potential._positions(positions)

    def _certificate(self, value, expected):
        if type(value) is not dict:
            raise ValueError('Complete cable certificate required')
        _capture(value)
        for key, target in expected.items():
            if key not in value or _encoded(value[key]) != _encoded(target):
                raise ValueError('Cable certificate identity/policy mismatch: '+key)

    def _bound(self, certificate, field, tolerance):
        if field not in certificate:
            raise ValueError('Missing cable error bound: '+field)
        bound = _rational(certificate[field])
        if not 0 <= bound <= F(float(tolerance)):
            raise ValueError('Cable error outside declared precision: '+field)
        return bound

    def _response_certificate(self, positions, certificate):
        q, policy = self.positions(positions), self.policy
        self._certificate(certificate, {
            'profile': 'continuous-cable-response-enclosure-v1', 'law': PROFILE,
            'inputSha256': self.description()['inputSha256'], 'positionsSha256': positions_sha256(q),
            'hessianPolicy': 'slack-sided-generalized-curvature', 'verified': True,
            'accepted': False, 'sourceControlsInstalled': False,
            'requestedTolerances': {key: _rat(F(float(policy[field]))) for key, field in zip('egh', TOLERANCES)},
            'budgets': {field: policy[field] for field in BUDGETS},
        })
        for field, tolerance in zip(('energyErrorBoundJoules', 'gradientMaxAbsoluteErrorBoundNewtons',
                                    'hessianMaxEntryErrorBoundNewtonsPerMeter'), TOLERANCES):
            self._bound(certificate, field, policy[tolerance])

    def validate_diagnostics(self, positions, diagnostics):
        if (type(diagnostics) is not dict or set(diagnostics) != {'definition', 'energyJoules', 'certificate'}
                or _encoded(diagnostics['definition']) != self._description_bytes
                or type(diagnostics['energyJoules']) is not float
                or not math.isfinite(diagnostics['energyJoules']) or diagnostics['energyJoules'] < 0):
            raise ValueError('Complete finite cable diagnostics with matching definition required')
        self._response_certificate(positions, diagnostics['certificate'])
        return copy.deepcopy(diagnostics)

    def validate_response(self, positions, response):
        if type(response) is not dict or set(response) != {'energy', 'gradient', 'hessian', 'certificate'}:
            raise ValueError('Complete cable energy/gradient/curvature response required')
        certificate = response['certificate']
        self._response_certificate(positions, certificate)
        energy, gradient, hessian = response['energy'], response['gradient'], response['hessian']
        size = 3*self.vertex_count
        if (type(energy) is not float or not math.isfinite(energy) or energy < 0
                or type(gradient) is not np.ndarray or gradient.dtype != np.dtype(np.float64)
                or gradient.shape != (size,) or not np.isfinite(gradient).all()
                or not isspmatrix_csr(hessian) or hessian.dtype != np.dtype(np.float64)
                or hessian.shape != (size, size) or not hessian.has_canonical_format
                or not np.isfinite(hessian.data).all() or (hessian-hessian.T).nnz):
            raise ValueError('Finite correctly shaped symmetric cable response required')
        return {'energy': energy, 'gradient': gradient.copy(), 'hessian': hessian.copy(),
                'certificate': copy.deepcopy(certificate)}

    def evaluate(self, positions):
        q, policy = self.positions(positions), self.policy
        response = self._potential.evaluate(q, **{key: value for key, value in policy.items()
                                                  if key != 'absolute_tolerance_joules'})
        return self.validate_response(q, response)

    def validate_change(self, start, end, response):
        first, last, policy = self.positions(start), self.positions(end), self.policy
        if type(response) is not dict or set(response) != {'changeJoules', 'certificate'}:
            raise ValueError('Complete fixed cable work response required')
        value = response['changeJoules']
        if type(value) is not float or not math.isfinite(value):
            raise ValueError('Finite cable work required')
        certificate = response['certificate']
        self._certificate(certificate, {
            'profile': 'continuous-cable-fixed-parameter-work-enclosure-v1', 'law': PROFILE,
            'inputSha256': self.description()['inputSha256'],
            'startPositionsSha256': positions_sha256(first), 'endPositionsSha256': positions_sha256(last),
            'requestedToleranceJoules': _rat(F(float(policy['absolute_tolerance_joules']))),
            'budgets': {field: policy[field] for field in BUDGETS}, 'verified': True, 'accepted': False,
        })
        self._bound(certificate, 'changeErrorBoundJoules', policy['absolute_tolerance_joules'])
        return copy.deepcopy(response)

    def energy_change(self, start, end):
        first, last, policy = self.positions(start), self.positions(end), self.policy
        response = self._potential.energy_change(first, last, absolute_tolerance_joules=policy['absolute_tolerance_joules'],
                                                **{field: policy[field] for field in BUDGETS})
        return self.validate_change(first, last, response)

    def diagnostics(self, positions):
        response = self.validate_response(positions, self.evaluate(positions))
        return {'definition': self.description(), 'energyJoules': response['energy'],
                'certificate': response['certificate']}
