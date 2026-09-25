"""Independent RK equations and adversarial native stage-boundary tests.

Synthetic mechanics only. A stage candidate is never timestep acceptance.
"""
from fractions import Fraction as F
import unittest
from unittest.mock import patch

import ipctk
import numpy as np

import solver_lobatto_stages as stages
from solver_physical_response import FixedPhysicalPotential, PhysicalResponseFailure
from solver_temporal_control import problem_identity
from test_solver_physical_response import (EMPTY, contact_cloth, folded_control,
                                          normal_fixture, particle_solver)


TABLEAU = ((F(), F(), F()), (F(5, 24), F(1, 3), F(-1, 24)),
           (F(1, 6), F(2, 3), F(1, 6)))


def frac(value):
    return F(int(value['numerator']), int(value['denominator']))


def rat(value):
    value = F(value)
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def linear_fixture(*, fixed=False, varying=False, inactive=False):
    solver, q, coefficients = particle_solver(fixed=fixed)
    targets = [np.array([[.02, -.01, .03]]) for _ in range(3)]
    weights = [.625]*3
    if varying:
        targets = [np.array([[.02+.003*j, -.01-.002*j, .03]]) for j in range(3)]
        weights = [.5, .625, .75]
    if inactive:
        weights = [0.]*3
    potentials = tuple(FixedPhysicalPotential(solver, target, sewing_activation=[weight])
                       for target, weight in zip(targets, weights))
    v = np.arange(q.size, dtype=float).reshape(q.shape)*.001-.004
    v[~solver.active] = 0.
    matrices = [weight/solver.compliance*np.kron(np.outer(coefficients, coefficients), np.eye(3))
                for weight in weights]
    rhs = [(weight/solver.compliance*np.outer(coefficients, target[0])).ravel()
           for weight, target in zip(weights, targets)]
    return solver, q, v, potentials, matrices, rhs


def full_rk_oracle(solver, q, v, dt, matrices, rhs):
    """Solve four RK position/velocity equations, without A² or B^-1.

    Known fixed coordinates enter each force's constant vector explicitly.
    """
    free = solver.free
    fixed = np.setdiff1d(np.arange(q.size), free)
    x0, v0 = q.ravel()[free], v.ravel()[free]
    mass = np.repeat(solver.mass, 3)[free]
    n = len(free)
    eye, zero = np.eye(n), np.zeros((n, n))
    g = [matrix[np.ix_(free, free)]/mass[:, None] for matrix in matrices]
    c = [(matrix[np.ix_(free, fixed)]@q.ravel()[fixed]-b[free])/mass
         for matrix, b in zip(matrices, rhs)]
    initial_acceleration = -(g[0]@x0+c[0])
    matrix = np.block([
        [eye, zero, -dt/3*eye, dt/24*eye],
        [zero, eye, -2*dt/3*eye, -dt/6*eye],
        [dt/3*g[1], -dt/24*g[2], eye, zero],
        [2*dt/3*g[1], dt/6*g[2], zero, eye]])
    right = np.concatenate((x0+dt*5/24*v0, x0+dt/6*v0,
        v0+dt*5/24*initial_acceleration-dt/3*c[1]+dt/24*c[2],
        v0+dt/6*initial_acceleration-2*dt/3*c[1]-dt/6*c[2]))
    xm, xe, vm, ve = np.linalg.solve(matrix, right).reshape(4, n)
    positions, velocities = [q.copy(), q.copy(), q.copy()], [v.copy(), np.zeros_like(v), np.zeros_like(v)]
    for j, (x, velocity) in enumerate(((xm, vm), (xe, ve)), 1):
        positions[j].ravel()[free] = x
        velocities[j].ravel()[free] = velocity
    return positions, velocities


class LobattoStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def assert_candidate(self, result):
        self.assertEqual(result['profile'], stages.PROFILE)
        self.assertIs(result['accepted'], False)
        self.assertIs(result['convergedCoupledForce'], True)
        self.assertEqual([frac(x) for x in result['nodes']], [F(), F(1, 2), F(1)])
        self.assertIn('no collocation-polynomial', result['pathScope'])
        self.assertIn('Schedule provenance', result['scope'])
        self.assertLessEqual(max(map(frac, result['forceResidual']['upperMaxAbsoluteNewtons'])), F(1e-8))
        costs = result['costs']
        self.assertEqual(costs['responseAttempts'], costs['responseSuccesses'])
        records = [r for r in result['trace'] if r['kind'] == 'response']
        self.assertEqual(len(records), costs['responseAttempts'])
        self.assertEqual([(r['role'], r['stage']) for r in records[-3:]], [('fresh', 0), ('fresh', 1), ('fresh', 2)])
        self.assertTrue(all(r['successful'] for r in records))

    def assert_exact_records(self, solver, q, v, dt, result):
        """Reconstruct full RK kinematics/impulses from returned native values."""
        h = F(dt)
        free = solver.free
        mass = [F(float(x)) for x in np.repeat(solver.mass, 3)[free]]
        xs = [[F(float(x)) for x in row.ravel()[free]] for row in result['stagePositions']]
        vs = [[F(float(x)) for x in row.ravel()[free]] for row in result['stageVelocities']]
        gs = [[F(float(x)) for x in row['gradientNewtons'][free]] for row in result['physicalResponses']]
        errors = [frac(row['knownErrorBounds']['gradientMaxAbsoluteNewtons']) for row in result['physicalResponses']]
        velocity_errors, exact_velocities = [], []
        for j in range(3):
            exact = [vs[0][k]-h*sum(TABLEAU[j][i]*gs[i][k] for i in range(3))/m for k, m in enumerate(mass)]
            exact_velocities.append(exact)
            rounded = [F(float(value)) for value in exact]
            self.assertEqual(vs[j], rounded)
            rounding = [abs(value-observed) for value, observed in zip(exact, rounded)]
            bounds = [r+h*sum(abs(TABLEAU[j][i])*errors[i] for i in range(3))/m
                      for r, m in zip(rounding, mass)]
            velocity_errors.append(bounds)
            self.assertEqual(frac(result['velocityFormationRoundingMaxAbsoluteMetresPerSecond'][j]), max(rounding))
            self.assertEqual(frac(result['velocityKnownErrorMaxAbsoluteMetresPerSecond'][j]), max(bounds))
        for j in (1, 2):
            kr = [abs(xs[j][k]-xs[0][k]-h*sum(TABLEAU[j][i]*vs[i][k] for i in range(3))) for k in range(len(free))]
            ir = [abs(m*(vs[j][k]-vs[0][k])+h*sum(TABLEAU[j][i]*gs[i][k] for i in range(3))) for k, m in enumerate(mass)]
            kb = [value+h*sum(abs(TABLEAU[j][i])*velocity_errors[i][k] for i in range(3)) for k, value in enumerate(kr)]
            ib = [value+h*sum(abs(TABLEAU[j][i])*errors[i] for i in range(3)) for value in ir]
            self.assertEqual(frac(result['kinematicDefects'][j-1]['centralMaxAbsoluteMetres']), max(kr))
            self.assertEqual(frac(result['kinematicDefects'][j-1]['knownUpperMaxAbsoluteMetres']), max(kb))
            self.assertEqual(frac(result['impulseDefects'][j-1]['centralMaxAbsoluteNewtonSeconds']), max(ir))
            self.assertEqual(frac(result['impulseDefects'][j-1]['knownUpperMaxAbsoluteNewtonSeconds']), max(ib))
        # Form the complete RK position defects before transforming them to
        # force units; do not evaluate the producer's displacement residual.
        defects = [[xs[j][k]-xs[0][k]-h*sum(TABLEAU[j][i]*exact_velocities[i][k] for i in range(3))
                    for k in range(len(free))] for j in (1, 2)]
        force = [[3*m*defects[1][k]/h**2 for k, m in enumerate(mass)],
                 [m*(-48*defects[0][k]+12*defects[1][k])/h**2 for k, m in enumerate(mass)]]
        central = [max(map(abs, row)) for row in force]
        bounds = [errors[1]+errors[0]/2, errors[2]+errors[0]]
        report = result['forceResidual']
        self.assertEqual(list(map(frac, report['centralMaxAbsoluteNewtons'])), central)
        self.assertEqual(list(map(frac, report['knownErrorMaxAbsoluteNewtons'])), bounds)
        self.assertEqual(list(map(frac, report['upperMaxAbsoluteNewtons'])), [a+b for a, b in zip(central, bounds)])
        search_rounding = max(abs(F(float(value))-value) for row in force for value in row)
        self.assertEqual(frac(report['searchResidualRoundingMaxAbsoluteNewtons']), search_rounding)

    def test_mass_weighted_full_rk_oracle_fixed_and_varying_stage_controls(self):
        for fixed, varying in [(False, False), (True, False), (False, True)]:
            with self.subTest(fixed=fixed, varying=varying):
                solver, q, v, potentials, matrices, rhs = linear_fixture(fixed=fixed, varying=varying)
                expected_x, expected_v = full_rk_oracle(solver, q, v, .04, matrices, rhs)
                original_q, original_v, identity = q.copy(), v.copy(), problem_identity(solver)
                result = stages.solve_lobatto_stages(solver, potentials, q, v, .04)
                self.assert_candidate(result)
                for actual, expected in zip(result['stagePositions'], expected_x):
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-13)
                for actual, expected in zip(result['stageVelocities'], expected_v):
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-12)
                np.testing.assert_array_equal(q, original_q); np.testing.assert_array_equal(v, original_v)
                self.assertEqual(problem_identity(solver), identity)
                self.assert_exact_records(solver, q, v, .04, result)
                self.assertGreater(np.max(np.abs(result['velocities']-(result['positions']-q)/.04)), 1e-4)
                if fixed:
                    for x, velocity in zip(result['stagePositions'], result['stageVelocities']):
                        np.testing.assert_array_equal(x[~solver.active], q[~solver.active])
                        np.testing.assert_array_equal(velocity[~solver.active], 0.)
                    self.assertGreater(np.linalg.norm(result['physicalResponses'][-1]['gradientNewtons'][:3]), 0.)

    def test_rank_one_oscillator_exact_pade_phase_and_native_velocity(self):
        solver, q, _, potentials, matrices, rhs = linear_fixture()
        coefficients = np.array([.75, -.25, .25, -.75])
        mode = coefficients/solver.mass
        omega = np.sqrt(.625/solver.compliance*np.dot(coefficients, mode))
        equilibrium = q.copy()
        target = coefficients@q
        potentials = tuple(FixedPhysicalPotential(solver, [target], sewing_activation=[.625]) for _ in range(3))
        amplitude, h = .007, .13
        q[:, 0] += amplitude*mode
        v = np.zeros_like(q)
        z = F(float(omega*h))
        # Independent [2/2] rational rotation, not a stage residual formula.
        a, b = 1-z*z/12, z/2
        cosine, sine = float((a*a-b*b)/(a*a+b*b)), float(2*a*b/(a*a+b*b))
        result = stages.solve_lobatto_stages(solver, potentials, q, v, h)
        np.testing.assert_allclose(result['positions'][:, 0], equilibrium[:, 0]+amplitude*cosine*mode, atol=2e-13, rtol=0)
        np.testing.assert_allclose(result['velocities'][:, 0], -amplitude*omega*sine*mode, atol=2e-12, rtol=0)
        self.assert_candidate(result)

    def test_free_flight_and_stationary_six_response_boundary(self):
        solver, q, v, potentials, _, _ = linear_fixture(inactive=True)
        result = stages.solve_lobatto_stages(solver, potentials, q, v, .125)
        for c, x, velocity in zip((0., .5, 1.), result['stagePositions'], result['stageVelocities']):
            np.testing.assert_allclose(x, q+c*.125*v, atol=2e-15, rtol=0)
            np.testing.assert_array_equal(velocity, v)
        stationary = stages.solve_lobatto_stages(solver, potentials, q, np.zeros_like(v), .125, max_responses=6)
        self.assert_candidate(stationary)
        self.assertEqual(stationary['costs']['responseAttempts'], 6)
        self.assertEqual(stationary['costs']['factorizationAttempts'], 0)

    def test_actual_contact_cable_normal_and_fold_routes_retain_fresh_evidence(self):
        cases = []
        for route in ('contact', 'fixed', 'varying'):
            solver, q = contact_cloth(route)
            controls = dict(cable_targets=[[.0015, .0015]], cable_activation=[.8]) if route == 'varying' else {}
            cases.append((route, solver, q, EMPTY, controls, 1e-4))
        for kind in ('legacy', 'controlled'):
            solver, q, controls = folded_control(kind)
            cases.append((kind, solver, q, EMPTY, controls, 1e-4))
        solver, q = normal_fixture(); cases.append(('normal', solver, q, [.17], {}, .001))
        for label, solver, q, targets, controls, h in cases:
            with self.subTest(route=label):
                potentials = tuple(FixedPhysicalPotential(solver, targets, **controls) for _ in range(3))
                result = stages.solve_lobatto_stages(solver, potentials, q, np.zeros_like(q), h)
                self.assert_candidate(result)
                self.assert_exact_records(solver, q, np.zeros_like(q), h, result)
                attempts = result['costs']['responseAttempts']
                calls = result['costs']['componentCalls']
                if solver.contact is not None:
                    self.assertEqual(calls['contact.validate_state'], attempts)
                    self.assertIn('fresh.physicalEndpointChord.contact', result['costs']['guardCalls'])
                if label in ('fixed', 'varying'):
                    self.assertEqual(calls['cable.evaluate'], attempts)
                    self.assertEqual(calls['cable.validate_response'], attempts)
                    self.assertTrue(all(r['cableCertificate'] for r in result['physicalResponses']))
                if label in ('legacy', 'controlled'):
                    fold_name = 'legacyFold' if label == 'legacy' else 'controlledFold'
                    for suffix in (fold_name, 'foldBarrier', 'bending'):
                        self.assertIn('fresh.physicalEndpointChord.'+suffix, result['costs']['guardCalls'])

    def test_fresh_known_stage_force_and_final_path_cannot_reuse_earlier_success(self):
        solver, q, v, potentials, _, _ = linear_fixture(inactive=True)
        original = FixedPhysicalPotential.evaluate
        count = 0
        def alter_fresh(potential, values):
            nonlocal count
            count += 1
            response = original(potential, values)
            if count == 4:
                response['gradientNewtons'][0] += 1.
            return response
        with patch.object(FixedPhysicalPotential, 'evaluate', alter_fresh), self.assertRaisesRegex(stages.LobattoStageFailure, 'Fresh coupled force') as failure:
            stages.solve_lobatto_stages(solver, potentials, q, np.zeros_like(v), .1, max_responses=6)
        self.assertEqual(failure.exception.costs['responseAttempts'], 6)
        self.assertEqual(failure.exception.costs['freshValidationAttempts'], 1)
        count = 0
        def reject_fresh(*args):
            nonlocal count
            count += 1
            return count < 4
        with patch.object(stages, 'triangle_sweep_safe', reject_fresh), self.assertRaisesRegex(stages.LobattoStageFailure, 'Fresh final') as failure:
            stages.solve_lobatto_stages(solver, potentials, q, np.zeros_like(v), .1)
        self.assertEqual(failure.exception.costs['responseAttempts'], 3)
        self.assertEqual(failure.exception.costs['guardCalls']['fresh.startToMidpoint.triangles'], 1)

    def test_reserved_final_response_capacity_rejects_before_partial_pair(self):
        solver, q, v, potentials, _, _ = linear_fixture()
        with self.assertRaisesRegex(stages.LobattoStageFailure, 'reserved work budget') as caught:
            stages.solve_lobatto_stages(solver, potentials, q, v, .04, max_responses=7)
        self.assertEqual(caught.exception.costs['responseAttempts'], 3)
        self.assertEqual(caught.exception.costs['candidateAttempts'], 0)
        self.assertEqual(caught.exception.costs['factorizationAttempts'], 1)
        self.assertEqual(caught.exception.costs['linearSolveAttempts'], 1)

    def test_response_failure_retains_partial_attempts_and_component_counts(self):
        solver, q, v, potentials, _, _ = linear_fixture()
        actual = FixedPhysicalPotential.evaluate
        calls = 0
        def fail_second(potential, values):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise PhysicalResponseFailure('forced failed component', {'independent.attempt': 2})
            return actual(potential, values)
        with patch.object(FixedPhysicalPotential, 'evaluate', fail_second), self.assertRaisesRegex(stages.LobattoStageFailure, 'forced failed') as caught:
            stages.solve_lobatto_stages(solver, potentials, q, v, .04)
        self.assertEqual(caught.exception.costs['responseAttempts'], 2)
        self.assertEqual(caught.exception.costs['responseSuccesses'], 1)
        self.assertEqual(caught.exception.costs['componentCalls']['independent.attempt'], 2)
        self.assertFalse(caught.exception.trace[-1]['successful'])
        def native_error(component, values):
            raise RuntimeError('ordinary native helper failure')
        with patch.object(type(solver.bending), 'energy', native_error), self.assertRaisesRegex(stages.LobattoStageFailure, 'ordinary native helper') as caught:
            stages.solve_lobatto_stages(solver, potentials, q, v, .04)
        self.assertEqual(caught.exception.costs['responseAttempts'], 1)
        self.assertEqual(caught.exception.costs['responseSuccesses'], 0)
        calls = caught.exception.costs['componentCalls']
        self.assertEqual(calls['membrane.derivatives'], 1)
        self.assertEqual(calls['bending.energy'], 1)
        self.assertNotIn('bending.gradient', calls)
        self.assertIsInstance(caught.exception.__cause__, PhysicalResponseFailure)
        self.assertIsInstance(caught.exception.__cause__.__cause__, RuntimeError)

    def test_mutated_response_input_cannot_be_counted_as_success(self):
        solver, q, v, potentials, _, _ = linear_fixture()
        actual = FixedPhysicalPotential.evaluate
        before = q.copy()
        def mutate(potential, values):
            response = actual(potential, values)
            values[0, 0] += .125
            return response
        with patch.object(FixedPhysicalPotential, 'evaluate', mutate), self.assertRaisesRegex(stages.LobattoStageFailure, 'mutated') as caught:
            stages.solve_lobatto_stages(solver, potentials, q, v, .04)
        self.assertEqual(caught.exception.costs['responseAttempts'], 1)
        self.assertEqual(caught.exception.costs['responseSuccesses'], 0)
        self.assertFalse(caught.exception.trace[-1]['successful'])
        np.testing.assert_array_equal(q, before)

    def test_guard_retained_array_cannot_mutate_live_stage_after_earlier_check(self):
        solver, q, v, potentials, _, _ = linear_fixture(inactive=True)
        retained, calls = [], 0
        def retaining_guard(start, end, faces):
            nonlocal calls
            calls += 1
            if calls == 1:
                retained.append(end)
            if calls == 3:
                retained[0][0, 0] += .125
            return True
        with patch.object(stages, 'triangle_sweep_safe', retaining_guard):
            result = stages.solve_lobatto_stages(solver, potentials, q, np.zeros_like(v), .1, max_responses=6)
        self.assert_candidate(result)
        self.assertEqual(result['costs']['factorizationAttempts'], 0)
        for position in result['stagePositions']:
            np.testing.assert_array_equal(position, q)

    def test_immediate_guard_mutation_wrong_type_and_model_changes_are_fatal(self):
        for kind in ('geometry', 'model', 'boolean'):
            solver, q, v, potentials, _, _ = linear_fixture()
            before = q.copy()
            def corrupt(start, end, faces):
                if kind == 'geometry': end[0, 0] += .125
                if kind == 'model': solver.mass[0] *= 2
                return np.bool_(True) if kind == 'boolean' else True
            with patch.object(stages, 'triangle_sweep_safe', corrupt), self.assertRaises(stages.LobattoStageFailure) as caught:
                stages.solve_lobatto_stages(solver, potentials, q, v, .04)
            self.assertEqual(caught.exception.costs['responseAttempts'], 0)
            self.assertEqual(caught.exception.costs['guardCalls']['initial.startToMidpoint.triangles'], 1)
            np.testing.assert_array_equal(q, before)

    def test_sparse_factorization_and_solve_failures_keep_cost_and_identity(self):
        solver, q, v, potentials, _, _ = linear_fixture()
        def fail_factor(matrix): raise RuntimeError('independent singular matrix')
        with patch.object(stages, 'splu', fail_factor), self.assertRaisesRegex(stages.LobattoStageFailure, 'singular') as caught:
            stages.solve_lobatto_stages(solver, potentials, q, v, .04)
        self.assertEqual(caught.exception.costs['factorizationAttempts'], 1)
        self.assertEqual(caught.exception.costs['linearSolveAttempts'], 0)
        self.assertEqual(caught.exception.costs['responseAttempts'], 3)
        class CorruptSolve:
            def solve(self, rhs):
                solver.mass[0] *= 2
                raise RuntimeError('failed after mutation')
        with patch.object(stages, 'splu', lambda matrix: CorruptSolve()), self.assertRaisesRegex(stages.LobattoStageFailure, 'identity') as caught:
            stages.solve_lobatto_stages(solver, potentials, q, v, .04)
        self.assertEqual(caught.exception.costs['linearSolveAttempts'], 1)

    def test_iteration_limit_retains_one_admitted_nonconverged_correction(self):
        solver, q, v, potentials, _, _ = linear_fixture()
        actual = stages.splu
        class HalfCorrection:
            def __init__(self, factor): self.factor = factor
            def solve(self, rhs): return self.factor.solve(rhs)*.5
        with patch.object(stages, 'splu', lambda matrix: HalfCorrection(actual(matrix))), self.assertRaisesRegex(stages.LobattoStageFailure, 'iteration budget') as caught:
            stages.solve_lobatto_stages(solver, potentials, q, v, .04, max_iterations=1)
        costs = caught.exception.costs
        self.assertEqual(costs['iterations'], 1)
        self.assertEqual(costs['factorizationAttempts'], 1)
        self.assertEqual(costs['responseAttempts'], 5)
        self.assertEqual(costs['candidateAttempts'], 1)
        self.assertEqual(costs['rejectedCandidates'], 0)
        self.assertEqual(costs['freshValidationAttempts'], 0)
        self.assertTrue(any(row.get('admitted') is True for row in caught.exception.trace))

    def test_backtrack_budget_and_candidate_guards_precede_response_dispatch(self):
        solver, q, v, potentials, _, _ = linear_fixture()
        calls = 0
        def only_initial(*args):
            nonlocal calls
            calls += 1
            return calls <= 3
        with patch.object(stages, 'triangle_sweep_safe', only_initial), self.assertRaisesRegex(stages.LobattoStageFailure, 'backtrack budget') as caught:
            stages.solve_lobatto_stages(solver, potentials, q, v, .04, max_backtracks=2)
        costs = caught.exception.costs
        self.assertEqual(costs['responseAttempts'], 3)
        self.assertEqual(costs['candidateAttempts'], 2)
        self.assertEqual(costs['rejectedCandidates'], 2)
        self.assertEqual(costs['guardCalls']['candidate.optimizer1.triangles'], 2)

    def test_nonnegative_canonical_force_radius_is_part_of_acceptance(self):
        solver, q, v, potentials, _, _ = linear_fixture(inactive=True)
        actual = FixedPhysicalPotential.evaluate
        def uncertain(potential, values):
            response = actual(potential, values)
            response['knownErrorBounds']['gradientMaxAbsoluteNewtons'] = rat(F(1, 10**6))
            return response
        with patch.object(FixedPhysicalPotential, 'evaluate', uncertain), self.assertRaises(stages.LobattoStageFailure) as caught:
            stages.solve_lobatto_stages(solver, potentials, q, np.zeros_like(v), .1, max_responses=6)
        self.assertEqual(caught.exception.costs['freshValidationAttempts'], 0)
        for bad in ({'numerator': '-1', 'denominator': '100'}, {'numerator': '0', 'denominator': '2'}):
            def forged(potential, values):
                response = actual(potential, values)
                response['knownErrorBounds']['gradientMaxAbsoluteNewtons'] = bad
                return response
            with patch.object(FixedPhysicalPotential, 'evaluate', forged), self.assertRaises(stages.LobattoStageFailure):
                stages.solve_lobatto_stages(solver, potentials, q, v, .1)

    def test_state_control_and_duration_admission_rejects_without_motion(self):
        solver, q, v, potentials, _, _ = linear_fixture(fixed=True)
        cases = [(q.tolist(), v, .1, {}), (q.astype(np.float32), v, .1, {}),
                 (q, v, True, {}), (q, v, 1, {}), (q, v, 0., {}), (q, v, float('inf'), {}),
                 (q, v, .1, {'max_responses': 5}), (q, v, .1, {'max_iterations': True}),
                 (q, v, .1, {'max_backtracks': 0}), (q, v, .1, {'force_tolerance_newtons': 1.01e-8})]
        moving_fixed = v.copy(); moving_fixed[0, 0] = .001; cases.append((q, moving_fixed, .1, {}))
        for x, velocity, dt, options in cases:
            with self.assertRaises(stages.LobattoStageFailure) as caught:
                stages.solve_lobatto_stages(solver, potentials, x, velocity, dt, **options)
            self.assertEqual(caught.exception.costs['responseAttempts'], 0)
        other, _, _, other_potentials, _, _ = linear_fixture()
        for invalid in (list(potentials), potentials[:2], (potentials[0], potentials[1], other_potentials[2])):
            with self.assertRaises(stages.LobattoStageFailure):
                stages.solve_lobatto_stages(solver, invalid, q, v, .1)

    def test_returned_arrays_and_records_do_not_alias_inputs_or_each_other(self):
        solver, q, v, potentials, _, _ = linear_fixture(inactive=True)
        before_q, before_v = q.copy(), v.copy()
        result = stages.solve_lobatto_stages(solver, potentials, q, v, .125)
        expected = result['stagePositions'][2].copy()
        result['positions'][:] = 99.
        np.testing.assert_array_equal(result['stagePositions'][2], expected)
        result['velocities'][:] = 77.
        np.testing.assert_array_equal(result['stageVelocities'][2], v)
        result['physicalResponses'][0]['gradientNewtons'][:] = 66.
        np.testing.assert_array_equal(result['physicalResponses'][1]['gradientNewtons'], 0.)
        np.testing.assert_array_equal(q, before_q); np.testing.assert_array_equal(v, before_v)


if __name__ == '__main__':
    unittest.main()
