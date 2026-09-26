"""Bounded fixed-control equilibrium candidates for assembly research.

No inertia, timestep, velocity, physical damping or new constitutive law is
introduced. Search-matrix shifts change only the optimizer. Force stationarity
does not establish stable equilibrium, construction correctness or a garment.
"""
from copy import deepcopy
from fractions import Fraction as F

import numpy as np
from scipy.sparse import diags, issparse

from solver_cable_integration import positions_sha256
from solver_continuous_normal_sewing import _rat, _rational
from solver_global_sewing import GlobalSewingSolver, _positive_definite_direction
from solver_physical_response import (FixedPhysicalPotential, PhysicalResponseFailure,
                                      PROFILE as RESPONSE_PROFILE, ERROR_SCOPE)
from solver_temporal_control import problem_identity
from solver_triangle_sweep import triangle_sweep_safe
from solver_hinge_sweep import hinge_sweep_safe


PROFILE = 'fixed-control-static-equilibrium-v1'


class StaticEquilibriumFailure(ValueError):
    def __init__(self, reason, costs, trace):
        super().__init__(reason)
        self.costs = deepcopy(costs)
        self.trace = deepcopy(trace)


class _InvariantError(ValueError):
    pass


def _snapshot(array):
    return array.dtype.str, array.shape, array.strides, array.tobytes()


def solve_static_equilibrium(solver, potential, positions, *,
                             force_tolerance_newtons=1e-8, max_responses=300,
                             max_iterations=100, max_backtracks=24):
    """Return a detached stationary candidate and the guarded optimizer path.

    The path is piecewise affine in optimizer order, with no physical-time
    interpretation. The caller still owns source/construction admission,
    stability, strain, resolution and full-garment acceptance.
    """
    costs = dict(responseAttempts=0, responseSuccesses=0, componentCalls={},
                 guardCalls={}, factorizationAttempts=0, iterations=0,
                 candidateAttempts=0, rejectedCandidates=0,
                 freshValidationAttempts=0)
    trace = []
    try:
        return _solve(solver, potential, positions, force_tolerance_newtons,
                      max_responses, max_iterations, max_backtracks, costs, trace)
    except Exception as failure:
        raise StaticEquilibriumFailure(str(failure), costs, trace) from failure


def _solve(solver, potential, positions, tolerance, max_responses,
           max_iterations, max_backtracks, costs, trace):
    if (type(solver) is not GlobalSewingSolver
            or type(potential) is not FixedPhysicalPotential
            or potential._solver is not solver):
        raise ValueError('A fixed physical potential of this exact native solver is required')
    if (type(tolerance) not in (float, np.float64) or not np.isfinite(tolerance)
            or not 0 < tolerance <= 1e-8):
        raise ValueError('Positive binary64 tightening-only 1e-8 N force tolerance required')
    for value, low, high in ((max_responses, 2, 10000), (max_iterations, 1, 1000),
                             (max_backtracks, 1, 40)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError('Bounded integer equilibrium work budgets required')
    shape = (len(solver.mass), 3)
    if (type(positions) is not np.ndarray or positions.dtype != np.dtype(np.float64)
            or positions.shape != shape or not np.isfinite(positions).all()):
        raise ValueError('Finite raw binary64 positions of the native model are required')
    if (type(solver.mass) is not np.ndarray or solver.mass.shape != (shape[0],)
            or type(solver.active) is not np.ndarray or solver.active.dtype != np.dtype(bool)
            or solver.active.shape != (shape[0],)
            or type(solver.free) is not np.ndarray or solver.free.ndim != 1
            or solver.free.dtype.kind not in 'iu'
            or not np.any(solver.active) or not np.isfinite(solver.mass).all()
            or np.any(solver.mass < 0) or np.any(solver.mass[solver.active] <= 0)
            or not np.array_equal(solver.free, np.flatnonzero(np.repeat(solver.active, 3)))):
        raise ValueError('Consistent native fixed/free degrees are required')
    original = _snapshot(positions)
    q0 = positions.copy()
    free = solver.free.copy()
    fixed = ~solver.active.copy()
    identity = problem_identity(solver)
    description = potential.description()
    threshold = F(float(tolerance))

    def context():
        if _snapshot(positions) != original:
            raise _InvariantError('Equilibrium helper mutated caller positions')
        if (potential._solver is not solver or problem_identity(solver) != identity
                or potential.description() != description):
            raise _InvariantError('Equilibrium model or physical control identity changed')
        potential._check()

    def components(values):
        if (type(values) is not dict
                or any(type(k) is not str or type(v) is not int or v < 0 for k, v in values.items())):
            raise _InvariantError('Invalid physical component costs')
        for key, value in values.items():
            costs['componentCalls'][key] = costs['componentCalls'].get(key, 0) + value

    def response(q, role):
        context()
        # Keep one response reserved for independent fresh final validation.
        if costs['responseAttempts'] >= max_responses - (role != 'fresh'):
            raise ValueError('Equilibrium response budget exhausted')
        supplied = q.copy()
        snapshot = _snapshot(supplied)
        costs['responseAttempts'] += 1
        record = dict(kind='response', role=role, positionsSha256=positions_sha256(q), successful=False)
        trace.append(record)
        try:
            try:
                result = potential.evaluate(supplied)
            except PhysicalResponseFailure as failure:
                components(failure.component_calls)
                raise
            components(result['componentCalls'])
            result = deepcopy(result)
            g = result['gradientNewtons']
            matrix = result['searchMatrixNewtonsPerMetre']
            if (result['profile'] != RESPONSE_PROFILE
                    or result['positionsSha256'] != record['positionsSha256']
                    or result['controlSha256'] != description['controlSha256']
                    or result['includesInertia'] is not False
                    or result['accepted'] is not False
                    or result['configurationIdentityChecked'] is not True
                    or result['knownErrorBounds']['scope'] != ERROR_SCOPE
                    or result['componentSearchMatrixKinds'] != description['componentSearchMatrixKinds']
                    or type(g) is not np.ndarray or g.dtype != np.dtype(np.float64)
                    or g.shape != (q.size,) or not np.isfinite(g).all()
                    or not issparse(matrix) or matrix.shape != (q.size, q.size)
                    or matrix.dtype != np.dtype(np.float64) or not np.isfinite(matrix.data).all()
                    or type(result['energyJoules']) not in (float, np.float64)
                    or not np.isfinite(result['energyJoules'])):
                raise _InvariantError('Physical response is not bound to this equilibrium state')
            for key in ('energyJoules', 'gradientMaxAbsoluteNewtons'):
                if _rational(result['knownErrorBounds'][key]) < 0:
                    raise _InvariantError('Negative physical response error radius')
        finally:
            if _snapshot(supplied) != snapshot:
                raise _InvariantError('Physical response mutated supplied geometry')
            context()
        costs['responseSuccesses'] += 1
        record['successful'] = True
        return result

    hinges = [('bending', solver.bending.indices)]
    for label, recipe, field in (
            ('controlledFold', solver.controlled_fold_actuation, 'hinges'),
            ('legacyFold', solver.fold_actuation, 'hinges'),
            ('foldBarrier', solver.fold_barrier, 'indices')):
        if recipe is not None:
            hinges.append((label, getattr(recipe, field)))

    def guard(label, function, *arguments):
        context()
        supplied = tuple(value.copy() if type(value) is np.ndarray else value for value in arguments)
        snapshots = [(value, _snapshot(value)) for value in supplied if type(value) is np.ndarray]
        costs['guardCalls'][label] = costs['guardCalls'].get(label, 0) + 1
        try:
            value = function(*supplied)
            if type(value) is not bool:
                raise _InvariantError('Physical path guard must return a raw Boolean')
            return value
        except _InvariantError:
            raise
        except (ValueError, ArithmeticError):
            return False
        finally:
            if any(_snapshot(value) != old for value, old in snapshots):
                raise _InvariantError('Physical path guard mutated supplied inputs')
            context()

    def path(start, end):
        if not np.array_equal(end[fixed], q0[fixed]):
            raise _InvariantError('Equilibrium search moved fixed coordinates')
        if not guard('triangles', triangle_sweep_safe, start, end, solver.faces):
            return False
        for label, indices in hinges:
            if len(indices) and not guard(label, hinge_sweep_safe, start, end, indices):
                return False
        return solver.contact is None or guard('contact', solver.contact.path_safe, start, end)

    def force_upper(r):
        error = _rational(r['knownErrorBounds']['gradientMaxAbsoluteNewtons'])
        return max(F(abs(float(x))) for x in r['gradientNewtons'][free]) + error

    def slope_upper(r, start, end):
        delta = [F(float(b)) - F(float(a)) for a, b in zip(start.ravel()[free], end.ravel()[free])]
        central = sum((F(float(g))*d for g, d in zip(r['gradientNewtons'][free], delta)), F())
        radius = _rational(r['knownErrorBounds']['gradientMaxAbsoluteNewtons']) * sum(map(abs, delta), F())
        return central - radius, central + radius

    context()
    if not path(q0, q0):
        raise ValueError('Initial equilibrium geometry failed physical guards')
    q = q0.copy()
    current = response(q, 'initial')
    states = [q.copy()]
    predecessor = None
    while True:
        if force_upper(current) <= threshold:
            costs['freshValidationAttempts'] += 1
            if not path(q, q):
                raise ValueError('Fresh equilibrium endpoint failed physical guards')
            fresh = response(q, 'fresh')
            upper = force_upper(fresh)
            if upper > threshold:
                raise ValueError('Fresh equilibrium force validation failed')
            if (fresh['energyJoules'] != current['energyJoules']
                    or fresh['componentEnergyJoules'] != current['componentEnergyJoules']
                    or fresh['knownErrorBounds'] != current['knownErrorBounds']
                    or not np.array_equal(fresh['gradientNewtons'], current['gradientNewtons'])):
                raise ValueError('Fresh same-state physical response changed its numerical evidence')
            if predecessor is not None:
                slope_low, slope_high = slope_upper(predecessor, states[-2], q)
                radius = (_rational(predecessor['knownErrorBounds']['energyJoules'])
                          + _rational(fresh['knownErrorBounds']['energyJoules']))
                change = F(float(fresh['energyJoules'])) - F(float(predecessor['energyJoules']))
                if slope_high >= 0 or change + radius > F(1, 10000) * slope_low:
                    raise ValueError('Fresh final energy does not retain the admitted decrease')
            context()
            return dict(profile=PROFILE, positions=q.copy(), pathPositions=states,
                initialPositionsSha256=positions_sha256(q0),
                controlSha256=description['controlSha256'], physicalResponse=fresh,
                convergedForce=True, forceResidualUpperNewtons=_rat(upper),
                forceToleranceNewtons=float(tolerance), includesInertia=False, accepted=False,
                stabilityEstablished=False, costs=deepcopy(costs), trace=deepcopy(trace),
                errorScope=ERROR_SCOPE,
                pathScope='Piecewise affine optimizer path; triangle, hinge and configured contact guards. No physical-time or curved-path claim.',
                scope='Fixed-control force-stationary candidate only. No stable-equilibrium, source-construction, strain, resolution, full-garment or drape acceptance.')
        if costs['iterations'] >= max_iterations:
            raise ValueError('Equilibrium iteration budget exhausted')
        costs['iterations'] += 1
        matrix = current['searchMatrixNewtonsPerMetre'][free][:, free].tocsc()
        gradient = current['gradientNewtons'][free].copy()
        scale = max(1., float(np.max(np.abs(matrix.data), initial=0.)))
        found = False
        # Absolute shifts have units N/m and affect only the search direction.
        # This deliberately avoids mass or a fictitious large time increment.
        for relative_shift in (0., *(10.**power for power in range(-9, 4))):
            shift = relative_shift * scale
            candidate_matrix = matrix if shift == 0 else matrix + diags(np.full(len(free), shift))
            context()
            costs['factorizationAttempts'] += 1
            supplied_gradient = gradient.copy()
            inputs = [(value, _snapshot(value)) for value in
                      (supplied_gradient, candidate_matrix.data,
                       candidate_matrix.indices, candidate_matrix.indptr)]
            try:
                direction = _positive_definite_direction(candidate_matrix, supplied_gradient)
            finally:
                if any(_snapshot(value) != before for value, before in inputs):
                    raise _InvariantError('Equilibrium metric helper mutated its supplied inputs')
                context()
            if direction is None:
                continue
            if (type(direction) is not np.ndarray or direction.shape != (len(free),)
                    or direction.dtype != np.dtype(np.float64) or not np.isfinite(direction).all()):
                raise _InvariantError('Invalid equilibrium search direction')
            direction = direction.copy()
            for backtrack in range(max_backtracks):
                if costs['responseAttempts'] >= max_responses - 1:
                    raise ValueError('Equilibrium response budget exhausted')
                candidate = q.copy()
                with np.errstate(over='ignore', invalid='ignore'):
                    candidate.ravel()[free] += (2.**-backtrack) * direction
                costs['candidateAttempts'] += 1
                finite = np.isfinite(candidate).all()
                record = dict(kind='candidate', positionsSha256=positions_sha256(candidate) if finite else None,
                              shiftNewtonsPerMetre=shift, scale=2.**-backtrack, admitted=False)
                trace.append(record)
                if (not finite or np.array_equal(candidate, q)
                        or not path(q, candidate)):
                    record['reason'] = 'invalid, unchanged or rejected geometry path'
                    costs['rejectedCandidates'] += 1
                    continue
                slope_low, slope_high = slope_upper(current, q, candidate)
                if slope_high >= 0:
                    record['reason'] = 'conditional descent not established'
                    costs['rejectedCandidates'] += 1
                    continue
                try:
                    trial = response(candidate, 'candidate')
                except PhysicalResponseFailure as failure:
                    record['reason'] = str(failure)
                    costs['rejectedCandidates'] += 1
                    continue
                radius = (_rational(current['knownErrorBounds']['energyJoules'])
                          + _rational(trial['knownErrorBounds']['energyJoules']))
                change = F(float(trial['energyJoules'])) - F(float(current['energyJoules']))
                record.update(conditionalEnergyChangeUpperJoules=_rat(change + radius),
                              conditionalSlopeLowerJoules=_rat(slope_low))
                if change + radius <= F(1, 10000) * slope_low:
                    record['admitted'] = True
                    predecessor = current
                    q, current = candidate, trial
                    states.append(q.copy())
                    found = True
                    break
                record['reason'] = 'conditional energy decrease not established'
                costs['rejectedCandidates'] += 1
            if found:
                break
        if not found:
            raise ValueError('Equilibrium search exhausted bounded metric/backtrack attempts')
