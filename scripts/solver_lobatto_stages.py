"""Native coupled Lobatto IIIA stage candidates, without timestep publication.

The three supplied physical potentials are fixed samples at nodes 0, 1/2, 1.
This module does not establish their schedule provenance or integrate parameter
power. Convergence here is a conditional force test, not an accepted timestep.
All path checks concern affine segments, never the collocation polynomial.
"""
import copy
from fractions import Fraction as F

import numpy as np
from scipy.sparse import bmat, diags, issparse
from scipy.sparse.linalg import splu

from solver_cable_integration import positions_sha256
from solver_global_sewing import GlobalSewingSolver
from solver_hinge_sweep import hinge_sweep_safe
from solver_physical_response import FixedPhysicalPotential, PhysicalResponseFailure, PROFILE as RESPONSE_PROFILE
from solver_temporal_control import problem_identity, rational, _fraction
from solver_triangle_sweep import triangle_sweep_safe


PROFILE = 'experimental-coupled-lobatto-stages-v1'
A = ((F(), F(), F()), (F(5, 24), F(1, 3), F(-1, 24)),
     (F(1, 6), F(2, 3), F(1, 6)))
NODES = (F(), F(1, 2), F(1))


class LobattoStageFailure(ValueError):
    """No published state; attempted work remains available to the owner."""
    def __init__(self, reason, costs, trace):
        super().__init__(reason)
        self.costs = copy.deepcopy(costs)
        self.trace = copy.deepcopy(trace)


class _InvariantError(ValueError):
    pass


def _snapshot(value):
    return value.dtype.str, value.shape, value.strides, value.tobytes()


def _raw_state(value, shape):
    if (type(value) is not np.ndarray or value.dtype != np.dtype(np.float64)
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError('Finite raw binary64 position and velocity arrays required')
    return value.copy()


def _float(value):
    result = float(value)
    if not np.isfinite(result):
        raise ArithmeticError('Coupled-stage binary64 formation overflow')
    return result


def solve_lobatto_stages(solver, potentials, positions, velocities, dt, *,
                        max_responses=300, max_iterations=100, max_backtracks=24,
                        force_tolerance_newtons=1e-8):
    """Return detached candidate stages and complete local attempted-work counts.

    The caller must additionally validate schedule samples, work/energy accuracy,
    trial reservations and temporal error before any timestep can be committed.
    Fixed vertices are supported with zero input velocity and retained reactions.
    """
    costs = {'responseAttempts': 0, 'responseSuccesses': 0, 'componentCalls': {},
             'guardCalls': {}, 'factorizationAttempts': 0, 'linearSolveAttempts': 0,
             'iterations': 0, 'candidateAttempts': 0, 'rejectedCandidates': 0,
             'freshValidationAttempts': 0}
    trace = []
    try:
        return _solve(solver, potentials, positions, velocities, dt,
                      max_responses, max_iterations, max_backtracks,
                      force_tolerance_newtons, costs, trace)
    except Exception as failure:
        raise LobattoStageFailure(str(failure), costs, trace) from failure


def _solve(solver, potentials, positions, velocities, dt, max_responses,
           max_iterations, max_backtracks, tolerance, costs, trace):
    if type(solver) is not GlobalSewingSolver:
        raise ValueError('Explicit native GlobalSewingSolver required')
    if (type(potentials) is not tuple or len(potentials) != 3
            or any(type(p) is not FixedPhysicalPotential or p._solver is not solver for p in potentials)):
        raise ValueError('Three fixed physical samples of the same solver are required')
    if (type(dt) not in (float, np.float64) or not np.isfinite(dt) or dt <= 0
            or type(tolerance) not in (float, np.float64) or not np.isfinite(tolerance)
            or not 0 < tolerance <= 1e-8):
        raise ValueError('Positive binary64 duration and tightening-only 1e-8 N force tolerance required')
    for value, minimum, maximum in ((max_responses, 6, 10000), (max_iterations, 1, 1000),
                                     (max_backtracks, 1, 40)):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError('Explicit bounded integer stage-work budgets required')
    shape = (len(solver.mass), 3)
    q0, v0 = _raw_state(positions, shape), _raw_state(velocities, shape)
    external = [(positions, _snapshot(positions)), (velocities, _snapshot(velocities))]
    initial = [(q0, _snapshot(q0)), (v0, _snapshot(v0))]
    if np.any(v0[~solver.active] != 0):
        raise ValueError('Fixed vertices require exactly zero input velocity')
    if (solver.mass.shape != (shape[0],) or not np.isfinite(solver.mass).all()
            or np.any(solver.mass < 0) or not np.any(solver.active)
            or np.any(solver.mass[solver.active] <= 0)
            or not np.array_equal(solver.free, np.flatnonzero(np.repeat(solver.active, 3)))):
        raise ValueError('Consistent free degrees and finite nonnegative physical masses required')
    identity = problem_identity(solver)
    descriptions = tuple(p.description() for p in potentials)

    def context():
        if any(_snapshot(value) != before for value, before in external+initial):
            raise _InvariantError('Coupled stage helper mutated caller or initial state')
        if problem_identity(solver) != identity:
            raise _InvariantError('Coupled stage model identity changed')
        for p, description in zip(potentials, descriptions):
            if p._solver is not solver or p.description() != description:
                raise _InvariantError('Coupled stage potential identity changed')
            p._check()

    context()
    free = solver.free.copy()
    masses = np.repeat(solver.mass, 3)[free]
    mf = tuple(F(float(m)) for m in masses)
    h, threshold = F(float(dt)), F(float(tolerance))
    qf = tuple(F(float(x)) for x in q0.ravel()[free])
    vf = tuple(F(float(x)) for x in v0.ravel()[free])
    scaled_mass = np.array([_float(m/h**2) for m in mf])
    if np.any(scaled_mass <= 0):
        raise ValueError('Mass over squared duration is not representable for the search matrix')
    mass_matrix = diags(scaled_mass, format='csc')
    hinge_sets = [('bending', solver.bending.indices)]
    for name, recipe, field in (
            ('controlledFold', solver.controlled_fold_actuation, 'hinges'),
            ('legacyFold', solver.fold_actuation, 'hinges'),
            ('foldBarrier', solver.fold_barrier, 'indices')):
        if recipe is not None:
            hinge_sets.append((name, getattr(recipe, field)))

    def invoke_guard(name, function, *args):
        context()
        # A guard cannot retain a live stage/topology view and mutate it during
        # a later guard call after the original call's snapshot was checked.
        supplied = tuple(value.copy() if type(value) is np.ndarray else value for value in args)
        saved = [(value, _snapshot(value)) for value in supplied if type(value) is np.ndarray]
        costs['guardCalls'][name] = costs['guardCalls'].get(name, 0)+1
        try:
            result = function(*supplied)
            if type(result) is not bool:
                raise _InvariantError('A path guard did not return a raw Boolean')
            return result
        except _InvariantError:
            raise
        except (ValueError, ArithmeticError):
            # Native guards also reject geometrically invalid candidate
            # endpoints by exception. They remain failed paths, while mutation
            # and configuration checks in finally remain hard failures.
            return False
        finally:
            if any(_snapshot(value) != before for value, before in saved):
                raise _InvariantError('Path guard mutated supplied geometry or topology')
            context()

    def path(start, end, role):
        if not invoke_guard(role+'.triangles', triangle_sweep_safe, start, end, solver.faces):
            return False
        for name, indices in hinge_sets:
            if len(indices) and not invoke_guard(role+'.'+name, hinge_sweep_safe, start, end, indices):
                return False
        if solver.contact is not None:
            if not invoke_guard(role+'.contact', solver.contact.path_safe, start, end):
                return False
        return True

    def stage_paths(stages, previous=None, role='candidate'):
        if previous is not None:
            for j in range(2):
                if not path(previous[j], stages[j], role+f'.optimizer{j+1}'):
                    return False
        return (path(q0, stages[0], role+'.startToMidpoint')
                and path(stages[0], stages[1], role+'.midpointToEndpoint')
                and path(q0, stages[1], role+'.physicalEndpointChord'))

    def add_components(values):
        if (type(values) is not dict or any(type(k) is not str or type(v) is not int or v < 0
                                          for k, v in values.items())):
            raise _InvariantError('Invalid physical component work counts')
        for name, count in values.items():
            costs['componentCalls'][name] = costs['componentCalls'].get(name, 0)+count

    def reserve_responses(count, *, final=False):
        # Always retain capacity for all three final fresh responses. A pair or
        # final group cannot begin when only a partial group fits the budget.
        if costs['responseAttempts']+count+(0 if final else 3) > max_responses:
            raise _InvariantError('Complete coupled response group exceeds the reserved work budget')

    def response(index, q, role):
        context()
        if costs['responseAttempts'] >= max_responses:
            raise _InvariantError('Coupled stage physical-response budget exhausted')
        costs['responseAttempts'] += 1
        entry = {'kind': 'response', 'role': role, 'stage': index,
                 'positionsSha256': positions_sha256(q), 'successful': False}
        trace.append(entry)
        supplied = q.copy()
        before = _snapshot(supplied)
        try:
            try:
                value = potentials[index].evaluate(supplied)
            except PhysicalResponseFailure as failure:
                add_components(failure.component_calls)
                raise
            if type(value) is not dict:
                raise _InvariantError('Physical response must be an explicit record')
            add_components(value['componentCalls'])
            gradient, matrix = value['gradientNewtons'], value['searchMatrixNewtonsPerMetre']
            if (value['profile'] != RESPONSE_PROFILE or value['includesInertia'] is not False
                    or value['accepted'] is not False or value['configurationIdentityChecked'] is not True
                    or value['positionsSha256'] != entry['positionsSha256']
                    or value['controlSha256'] != descriptions[index]['controlSha256']
                    or value['componentSearchMatrixKinds'] != descriptions[index]['componentSearchMatrixKinds']
                    or type(value['energyJoules']) not in (float, np.float64)
                    or not np.isfinite(value['energyJoules'])
                    or type(gradient) is not np.ndarray or gradient.dtype != np.dtype(np.float64)
                    or gradient.shape != (q0.size,) or not np.isfinite(gradient).all()
                    or not issparse(matrix) or matrix.shape != (q0.size, q0.size)
                    or matrix.dtype != np.dtype(np.float64) or not np.isfinite(matrix.data).all()):
                raise _InvariantError('Fresh finite physical response with matching state/control identity required')
            _fraction(value['knownErrorBounds']['gradientMaxAbsoluteNewtons'])
            _fraction(value['knownErrorBounds']['energyJoules'])
            value = copy.deepcopy(value)
        except Exception as failure:
            entry['reason'] = str(failure)
            raise
        finally:
            try:
                if _snapshot(supplied) != before:
                    raise _InvariantError('Physical response mutated its supplied stage')
                context()
            except Exception as failure:
                entry['reason'] = str(failure)
                raise
        costs['responseSuccesses'] += 1
        entry['successful'] = True
        return value

    def residual(stages, responses):
        gradients = [tuple(F(float(x)) for x in r['gradientNewtons'][free]) for r in responses]
        errors = [_fraction(r['knownErrorBounds']['gradientMaxAbsoluteNewtons']) for r in responses]
        coordinates = [tuple(F(float(x)) for x in q.ravel()[free]) for q in stages]
        exact = [[], []]
        for k, mass in enumerate(mf):
            d1 = coordinates[0][k]-qf[k]-h*vf[k]/2
            d2 = coordinates[1][k]-qf[k]-h*vf[k]
            exact[0].append(3*mass*d2/h**2+gradients[1][k]+gradients[0][k]/2)
            exact[1].append(mass*(-48*d1+12*d2)/h**2+gradients[2][k]-gradients[0][k])
        bounds = (errors[1]+errors[0]/2, errors[2]+errors[0])
        norms = tuple(max(map(abs, row)) for row in exact)
        upper = tuple(norm+bound for norm, bound in zip(norms, bounds))
        rounded = np.array([[_float(x) for x in row] for row in exact])
        rounding = max(abs(F(float(value))-exact[i][j]) for i, row in enumerate(rounded)
                       for j, value in enumerate(row))
        report = {'centralMaxAbsoluteNewtons': list(map(rational, norms)),
                  'knownErrorMaxAbsoluteNewtons': list(map(rational, bounds)),
                  'upperMaxAbsoluteNewtons': list(map(rational, upper)),
                  'searchResidualRoundingMaxAbsoluteNewtons': rational(rounding),
                  'toleranceNewtons': rational(threshold), 'converged': max(upper) <= threshold,
                  'scope': 'Exact rational assembly of binary64 states, masses, duration and component gradients; physical response uncertainty is conditional on its documented non-cable evaluation scope'}
        return rounded, max(upper), report

    try:
        stages = [q0.copy(), q0.copy()]
        reserve_responses(3)
        if not stage_paths(stages, role='initial'):
            raise ValueError('Initial coupled stage geometry failed path admission')
        r0 = response(0, q0, 'initial')
        responses = [r0, response(1, stages[0], 'initial'), response(2, stages[1], 'initial')]
        residual_values, merit, report = residual(stages, responses)
        for iteration in range(max_iterations+1):
            if report['converged']:
                reserve_responses(3, final=True)
                costs['freshValidationAttempts'] += 1
                if not stage_paths(stages, role='fresh'):
                    raise ValueError('Fresh final stage or physical endpoint path failed')
                # Neither the known stage nor a repeated identical state bypasses
                # a fresh native physical response at this publication boundary.
                responses = [response(0, q0, 'fresh'), response(1, stages[0], 'fresh'),
                             response(2, stages[1], 'fresh')]
                residual_values, merit, report = residual(stages, responses)
                if not report['converged']:
                    raise ValueError('Fresh coupled force residual failed convergence')
                break
            if iteration == max_iterations:
                raise ValueError('Coupled nonlinear iteration budget exhausted')
            costs['iterations'] += 1
            h1, h2 = [r['searchMatrixNewtonsPerMetre'][free][:, free] for r in responses[1:]]
            matrix = bmat([[h1, 3*mass_matrix], [-48*mass_matrix, h2+12*mass_matrix]], format='csc')
            if not np.isfinite(matrix.data).all():
                raise ValueError('Nonfinite coupled search matrix')
            costs['factorizationAttempts'] += 1
            factor = splu(matrix)
            context()
            costs['linearSolveAttempts'] += 1
            direction = factor.solve(-residual_values.ravel())
            context()
            if (type(direction) is not np.ndarray or direction.dtype != np.dtype(np.float64)
                    or direction.shape != (2*len(free),) or not np.isfinite(direction).all()):
                raise _InvariantError('Finite correctly shaped coupled direction required')
            direction = direction.reshape(2, -1)
            found = False
            for backtrack in range(max_backtracks):
                reserve_responses(2)
                costs['candidateAttempts'] += 1
                scale = F(1, 2**backtrack)
                candidate = [q0.copy(), q0.copy()]
                event = {'kind': 'candidate', 'iteration': iteration, 'backtrack': backtrack,
                         'scale': rational(scale), 'admitted': False}
                trace.append(event)
                try:
                    for j in range(2):
                        candidate[j].ravel()[free] = [_float(F(float(x))+scale*F(float(d)))
                            for x, d in zip(stages[j].ravel()[free], direction[j])]
                    event['positionsSha256'] = [positions_sha256(q) for q in candidate]
                    if not stage_paths(candidate, stages):
                        event['reason'] = 'Affine stage or optimizer path rejected'
                    else:
                        candidate_responses = [responses[0], response(1, candidate[0], 'candidate'),
                                               response(2, candidate[1], 'candidate')]
                        next_values, next_merit, next_report = residual(candidate, candidate_responses)
                        event['upperMaxAbsoluteNewtons'] = rational(next_merit)
                        if next_report['converged'] or next_merit <= (1-scale/F(10000))*merit:
                            stages, responses = candidate, candidate_responses
                            residual_values, merit, report = next_values, next_merit, next_report
                            event['admitted'] = found = True
                            break
                        event['reason'] = 'Conditional force residual did not decrease sufficiently'
                except (PhysicalResponseFailure, ArithmeticError) as failure:
                    event['reason'] = str(failure)
                    context()
                costs['rejectedCandidates'] += 1
            if not found:
                raise ValueError('Coupled residual search exhausted its backtrack budget')

        # Recover actual RK stage velocities, not displacement secants. Exact
        # binary-input arithmetic isolates each final binary64 rounding error.
        gradients = [[F(float(x)) for x in r['gradientNewtons'][free]] for r in responses]
        errors = [_fraction(r['knownErrorBounds']['gradientMaxAbsoluteNewtons']) for r in responses]
        stage_velocities = [v0.copy(), np.zeros_like(v0), np.zeros_like(v0)]
        velocity_errors = [[], [], []]
        formation_rounding = [F(), F(), F()]
        for j in range(3):
            for k, mass in enumerate(mf):
                exact = vf[k]-h*sum(A[j][i]*gradients[i][k] for i in range(3))/mass
                rounded = _float(exact)
                rounding = abs(F(rounded)-exact)
                formation_rounding[j] = max(formation_rounding[j], rounding)
                velocity_errors[j].append(rounding+h*sum(abs(A[j][i])*errors[i] for i in range(3))/mass)
                stage_velocities[j].ravel()[free[k]] = rounded
        all_positions = [q0, *stages]
        kinematic, impulses = [], []
        for j in (1, 2):
            kp, ip, kb, ib = F(), F(), F(), F()
            for k, mass in enumerate(mf):
                vv = [F(float(v.ravel()[free[k]])) for v in stage_velocities]
                delta = F(float(all_positions[j].ravel()[free[k]]))-qf[k]
                kr = abs(delta-h*sum(A[j][i]*vv[i] for i in range(3)))
                ir = abs(mass*(vv[j]-vf[k])+h*sum(A[j][i]*gradients[i][k] for i in range(3)))
                kp, ip = max(kp, kr), max(ip, ir)
                kb = max(kb, kr+h*sum(abs(A[j][i])*velocity_errors[i][k] for i in range(3)))
                ib = max(ib, ir+h*sum(abs(A[j][i])*errors[i] for i in range(3)))
            kinematic.append({'centralMaxAbsoluteMetres': rational(kp),
                              'knownUpperMaxAbsoluteMetres': rational(kb)})
            impulses.append({'centralMaxAbsoluteNewtonSeconds': rational(ip),
                             'knownUpperMaxAbsoluteNewtonSeconds': rational(ib)})
        context()
        return {'profile': PROFILE, 'accepted': False, 'convergedCoupledForce': True,
                'positions': stages[1].copy(), 'velocities': stage_velocities[2].copy(),
                'stagePositions': [q.copy() for q in all_positions],
                'stageVelocities': [v.copy() for v in stage_velocities],
                'physicalResponses': copy.deepcopy(responses),
                'forceResidual': report, 'kinematicDefects': kinematic, 'impulseDefects': impulses,
                'velocityFormationRoundingMaxAbsoluteMetresPerSecond': list(map(rational, formation_rounding)),
                'velocityKnownErrorMaxAbsoluteMetresPerSecond': [rational(max(row)) for row in velocity_errors],
                'costs': copy.deepcopy(costs), 'trace': copy.deepcopy(trace),
                'durationSeconds': rational(h), 'nodes': list(map(rational, NODES)),
                'stageControlSha256': [d['controlSha256'] for d in descriptions],
                'pathScope': 'Numerical triangle and hinge guards plus native contact guards over initial, optimizer, start-to-midpoint, midpoint-to-endpoint and physical endpoint affine chords; no collocation-polynomial swept proof',
                'matrixScope': 'General nonsymmetric sparse LU with labelled component search matrices; no exact total Hessian or linear-solve error certificate',
                'scope': 'Research coupled-stage candidate only. Schedule provenance, continuous parameter power, work/energy accuracy, temporal/spatial convergence, source admission and garment/application publication remain required'}
    finally:
        context()
