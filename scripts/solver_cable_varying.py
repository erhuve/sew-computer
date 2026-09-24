"""Immutable varying-cable recipe/policies and discrete parameter-work checks.

Effective controls are fixed during each solver trial. Numerical validators here
check structure, identity, precision and algebra; accepted publication also
requires a fresh numerical evaluation of each complete work/energy record.
"""
import copy
from fractions import Fraction as F
import hashlib
import json
import math

from solver_cable_integration import CableControl, BUDGETS, positions_sha256
from solver_cable_parameters import CableParameterRecipe, WORK_FIELDS, COMPONENT_FIELDS, PARAMETER_ORDER
from solver_continuous_cable_sewing import PROFILE, _budgets
from solver_continuous_normal_sewing import _capture, _encoded, _number, _rat, _rational


WORK_POLICY_FIELDS = ('absolute_tolerance_joules', 'component_tolerances_joules', *BUDGETS)
WORK_SCOPE = ('Bounded discrete target-first work at one supplied state, including output rounding; '
              'no continuous actuator work, dynamics, contact, source or physical acceptance')
WORK_STATS = ('inactiveCells', 'unchangedCells', 'boundaryLeaves', 'boundaryMaxDepth',
              'classifiedSmoothPanels', 'combinedSmoothPanels', 'momentPanels', 'maxMomentTerms')


def _array_state(value):
    return value.shape, value.strides, value.dtype.str, value.tobytes()


class VaryingCableControl:
    """One supplied geometry and two fixed continuation-wide precision policies."""
    __slots__ = ('_recipe', '_initial', '_work_policy_bytes', '_description_bytes', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('Varying cable controls are immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('Varying cable controls are immutable')

    def __init__(self, recipe, response_precision, parameter_work_precision):
        if getattr(self, '_sealed', False):
            raise AttributeError('Varying cable controls cannot be reinitialized')
        if type(recipe) is not CableParameterRecipe:
            raise ValueError('An exact immutable cable parameter recipe is required')
        if type(parameter_work_precision) is not dict or set(parameter_work_precision) != set(WORK_POLICY_FIELDS):
            raise ValueError('Complete explicit cable parameter-work precision is required')
        captured = _capture(parameter_work_precision)
        policy = json.loads(captured)
        _number(policy['absolute_tolerance_joules'], 0, 1e6, positive=True)
        components = policy['component_tolerances_joules']
        if components is not None:
            if type(components) is not dict or set(components) != set(COMPONENT_FIELDS):
                raise ValueError('All four explicit cable component tolerances or null required')
            for value in components.values():
                _number(value, 0, 1e6, positive=True)
        _budgets(**{field: policy[field] for field in BUDGETS})
        initial = CableControl(recipe.potential(*recipe.initial_parameters), response_precision)
        self._recipe, self._initial = recipe, initial
        self._work_policy_bytes = captured
        self._description_bytes = _encoded({
            'profile': 'guarded-varying-cable-control-v1', 'law': PROFILE,
            'recipe': recipe.description(), 'vertexCount': recipe.vertex_count,
            'geometrySha256': recipe.geometry_sha256, 'cellIds': list(recipe.cell_ids),
            'responsePrecision': initial.policy,
            'responsePrecisionSha256': initial.description()['precisionSha256'],
            'parameterWorkPrecision': policy,
            'parameterWorkPrecisionSha256': hashlib.sha256(captured).hexdigest(),
            'parameterOrder': PARAMETER_ORDER, 'parametersFixed': False,
            'precisionPolicy': 'Both response/fixed-motion and parameter-work policies are immutable across all trials and retries; unresolved decisions reject without relaxation',
            'trialPolicy': 'One immutable effective potential per trial; parameters fixed throughout that nonlinear solve',
            'accepted': False, 'sourceAdmissionGranted': False, 'sourceControlsInstalled': False,
            'scope': 'Generic supplied-cell control only; no source ownership, physical joint, construction, calibrated material or garment acceptance',
        })
        self._sealed = True

    @property
    def recipe(self):
        return self._recipe

    @property
    def vertex_count(self):
        return self._recipe.vertex_count

    @property
    def response_precision(self):
        return self._initial.policy

    @property
    def parameter_work_precision(self):
        return json.loads(self._work_policy_bytes)

    def description(self):
        return json.loads(self._description_bytes)

    def positions(self, positions):
        return self._initial.positions(positions)

    def parameters(self, targets, activation):
        return self._recipe.parameters(targets, activation)

    def parameter_sha256(self, targets, activation):
        return self._recipe.parameter_sha256(targets, activation)

    def effective(self, targets, activation):
        values, weights = self.parameters(targets, activation)
        snapshots = _array_state(values), _array_state(weights)
        potential = self._recipe.potential(values, weights)
        if snapshots != (_array_state(values), _array_state(weights)):
            raise ValueError('Cable potential helper mutated supplied parameters')
        return CableControl(potential, self.response_precision)

    def parameter_record(self, targets, activation):
        values, weights = self.parameters(targets, activation)
        control = self.effective(values, weights)
        return {'definition': self.description(),
                'parameterSha256': self.parameter_sha256(values, weights),
                'targetsMeters': values.tolist(), 'activation': weights.tolist(),
                'effectivePotentialSha256': control.description()['inputSha256']}

    def validate_parameter_work(self, positions, before_targets, before_activation,
                                after_targets, after_activation, result):
        q = self.positions(positions)
        d0, a0 = self.parameters(before_targets, before_activation)
        d1, a1 = self.parameters(after_targets, after_activation)
        if type(result) is not dict or set(result) != {*WORK_FIELDS, 'certificate'}:
            raise ValueError('Complete five-field cable parameter work and certificate required')
        _capture(result)
        for field in WORK_FIELDS:
            if type(result[field]) is not float or not math.isfinite(result[field]):
                raise ValueError('Finite binary64 cable parameter work required')
        if min(result['activationIncreaseWorkJoules'], result['releaseEnergyRemovedJoules']) < 0:
            raise ValueError('Activation increase and release removal must be nonnegative')
        policy = self.parameter_work_precision
        tolerances = {field: _number(policy['absolute_tolerance_joules'], 0, 1e6, positive=True)
                      for field in WORK_FIELDS}
        if policy['component_tolerances_joules'] is not None:
            tolerances.update({field: _number(value, 0, 1e6, positive=True)
                               for field, value in policy['component_tolerances_joules'].items()})
        expected = {
            'profile': 'continuous-cable-parameter-work-enclosure-v1', 'law': PROFILE,
            'geometrySha256': self._recipe.geometry_sha256, 'positionsSha256': positions_sha256(q),
            'beforeParameterSha256': self.parameter_sha256(d0, a0),
            'afterParameterSha256': self.parameter_sha256(d1, a1),
            'beforePotentialSha256': self.effective(d0, a0).description()['inputSha256'],
            'afterPotentialSha256': self.effective(d1, a1).description()['inputSha256'],
            'parameterOrder': PARAMETER_ORDER,
            'requestedToleranceJoules': _rat(tolerances['totalWorkJoules']),
            'requestedTolerancesJoules': {field: _rat(value) for field, value in tolerances.items()},
            'budgets': {field: policy[field] for field in BUDGETS},
            'verified': True, 'accepted': False, 'sourceControlsInstalled': False,
            'sourceAdmissionGranted': False, 'scope': WORK_SCOPE,
        }
        certificate = result['certificate']
        if type(certificate) is not dict or set(certificate) != {*expected, 'errorsJoules', 'stats'}:
            raise ValueError('Complete cable parameter-work certificate required')
        for field, value in expected.items():
            if _encoded(certificate[field]) != _encoded(value):
                raise ValueError('Cable parameter-work identity or precision mismatch: '+field)
        errors, stats = certificate['errorsJoules'], certificate['stats']
        if type(errors) is not dict or set(errors) != set(WORK_FIELDS):
            raise ValueError('All five cable parameter-work error bounds required')
        bounds = {field: _rational(errors[field]) for field in WORK_FIELDS}
        if any(not 0 <= bounds[field] <= tolerances[field] for field in WORK_FIELDS):
            raise ValueError('Cable parameter-work error exceeds the fixed policy')
        if (type(stats) is not dict or set(stats) != set(WORK_STATS)
                or any(type(value) is not int or value < 0 for value in stats.values())):
            raise ValueError('Complete nonnegative integer cable work statistics required')
        numerical = {field: F(result[field]) for field in WORK_FIELDS}
        if (abs(numerical['totalWorkJoules']-numerical['targetWorkJoules']-numerical['activationWorkJoules'])
                > bounds['totalWorkJoules']+bounds['targetWorkJoules']+bounds['activationWorkJoules']):
            raise ValueError('Cable target and activation work do not enclose total work')
        if (abs(numerical['activationWorkJoules']-numerical['activationIncreaseWorkJoules']+numerical['releaseEnergyRemovedJoules'])
                > bounds['activationWorkJoules']+bounds['activationIncreaseWorkJoules']+bounds['releaseEnergyRemovedJoules']):
            raise ValueError('Cable activation increase and release do not enclose signed work')
        return copy.deepcopy(result)

    def parameter_work(self, positions, before_targets, before_activation, after_targets, after_activation):
        q = self.positions(positions)
        d0, a0 = self.parameters(before_targets, before_activation)
        d1, a1 = self.parameters(after_targets, after_activation)
        inputs = q, d0, a0, d1, a1
        snapshots = tuple(_array_state(value) for value in inputs)
        result = self._recipe.parameter_energy_change(*inputs, **self.parameter_work_precision)
        if snapshots != tuple(_array_state(value) for value in inputs):
            raise ValueError('Cable parameter-work helper mutated its supplied inputs')
        return self.validate_parameter_work(*inputs, result)
