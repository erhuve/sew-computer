"""Independent static equilibria and bounded candidate-boundary counterexamples.

No time integration or garment acceptance is tested. The force oracle below is
an explicitly declared two-spring energy, independent of native derivatives.
"""
import copy
from fractions import Fraction as F
import hashlib
import json
import unittest
from unittest.mock import patch

import ipctk
import newton
import numpy as np
import warp as wp

import solver_static_equilibrium as static
from solver_global_sewing import GlobalSewingSolver
from solver_physical_response import FixedPhysicalPotential, PhysicalResponseFailure
from solver_temporal_control import problem_identity
from solver_triangle_sweep import verify_triangle_sweep_exact
from test_solver_physical_response import EMPTY, contact_cloth, folded_control
from test_solver_sewing_activation_integration import particles


def frac(value):
    answer = F(int(value['numerator']), int(value['denominator']))
    assert value == rat(answer)
    return answer


def rat(value):
    value = F(value)
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def state_sha(q):
    return hashlib.sha256(json.dumps(q.tolist(), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def pinned(mass=1., *, stationary=False, inactive=False):
    model = particles(points=[[0., 0., 0.], [.125, .25, -.125], [3., 0., 0.]],
                      masses=[0., mass, 0.])
    solver = GlobalSewingSolver(model, [{0: -1., 1: 1.}, {1: -1., 2: 1.}], .125)
    q = model.particle_q.numpy().astype(np.float64)
    if stationary:
        q[1] = [1.75, 0., 0.]
    potential = FixedPhysicalPotential(solver, [[1., 0., 0.], [1., 0., 0.]],
        sewing_activation=[0., 0.] if inactive else [.25, .75])
    return solver, potential, q


def spring_oracle(q):
    # U = (|q1-q0-ex|² + 3|q2-q1-ex|²), compliance=1/8.
    x = [[F(float(a)) for a in point] for point in q]
    unit = [F(1), F(), F()]
    left = [x[1][i]-x[0][i]-unit[i] for i in range(3)]
    right = [x[2][i]-x[1][i]-unit[i] for i in range(3)]
    energy = sum(a*a+3*b*b for a,b in zip(left,right))
    gradient = [[-2*a for a in left], [2*a-6*b for a,b in zip(left,right)],
                [6*b for b in right]]
    return energy, gradient


def cloth_fixture():
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
        vel=wp.vec3(0, 0, 0), vertices=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        indices=[0, 1, 2], density=2., tri_ke=16., tri_ka=16., tri_kd=0., edge_ke=0., edge_kd=0.)
    builder.particle_mass[:2] = [0., 0.]
    builder.set_coloring([[0], [1], [2]])
    model = builder.finalize(device='cpu')
    solver = GlobalSewingSolver(model, [{0: -1., 2: 1.}], .125)
    q = model.particle_q.numpy().astype(np.float64)
    q[2] = [0., 1.125, .03125]
    potential = FixedPhysicalPotential(solver, [[0., 1., 0.]])
    return solver, potential, q


class StaticEquilibriumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def assert_candidate(self, solver, potential, initial, result, tolerance=1e-8):
        self.assertEqual(result['profile'], 'fixed-control-static-equilibrium-v3')
        self.assertEqual(result['searchMetricPolicy'], 'row-maximum-diagonal-congruence-v1')
        self.assertEqual(result['searchRegularizationPolicy'], 'best-force-guided-shift-order-v1')
        self.assertIs(result['accepted'], False)
        self.assertIs(result['includesInertia'], False)
        self.assertIs(result['convergedForce'], True)
        self.assertIs(result['stabilityEstablished'], False)
        self.assertEqual(result['initialPositionsSha256'], state_sha(initial))
        self.assertEqual(result['controlSha256'], potential.description()['controlSha256'])
        self.assertTrue(set(result).isdisjoint({'velocities', 'durationSeconds', 'dt', 'timeSeconds'}))
        response = result['physicalResponse']
        self.assertEqual(response['positionsSha256'], state_sha(result['positions']))
        self.assertEqual(response['controlSha256'], result['controlSha256'])
        self.assertIs(response['accepted'], False)
        self.assertIs(response['includesInertia'], False)
        upper = max((abs(F(float(x))) for x in response['gradientNewtons'][solver.free]), default=F())
        upper += frac(response['knownErrorBounds']['gradientMaxAbsoluteNewtons'])
        self.assertEqual(frac(result['forceResidualUpperNewtons']), upper)
        self.assertLessEqual(upper, F(tolerance))
        self.assertIn('not certified', response['knownErrorBounds']['scope'])
        self.assertEqual(result['positions'].dtype, np.dtype(np.float64))
        self.assertEqual(result['positions'].shape, initial.shape)
        self.assertTrue(np.isfinite(result['positions']).all())
        self.assertGreaterEqual(len(result['pathPositions']), 1)
        np.testing.assert_array_equal(result['pathPositions'][0], initial)
        np.testing.assert_array_equal(result['pathPositions'][-1], result['positions'])
        for q in result['pathPositions']:
            self.assertEqual(q[~solver.active].tobytes(), initial[~solver.active].tobytes())
        costs = result['costs']
        self.assertEqual(costs['responseAttempts'], costs['responseSuccesses'])
        self.assertEqual(costs['freshValidationAttempts'], 1)
        self.assertEqual(costs['candidateAttempts']-costs['rejectedCandidates'], len(result['pathPositions'])-1)

    def test_closed_form_pinned_equilibrium_retains_nonzero_fixed_reactions(self):
        solver, potential, q = pinned()
        before = problem_identity(solver, (q,)); original = q.copy()
        result = static.solve_static_equilibrium(solver, potential, q, force_tolerance_newtons=1e-12)
        self.assert_candidate(solver, potential, original, result, tolerance=1e-12)
        np.testing.assert_allclose(result['positions'][1], [1.75, 0., 0.], rtol=0, atol=2e-9)
        energy, gradient = spring_oracle(result['positions'])
        np.testing.assert_allclose(result['physicalResponse']['gradientNewtons'],
            np.asarray(gradient, dtype=float).ravel(), rtol=0, atol=2e-14)
        self.assertAlmostEqual(result['physicalResponse']['energyJoules'], float(energy), delta=2e-14)
        self.assertAlmostEqual(float(energy), .75, delta=2e-14)
        np.testing.assert_allclose(result['physicalResponse']['gradientNewtons'].reshape(-1, 3)[[0, 2], 0],
                                   [-1.5, 1.5], rtol=0, atol=2e-9)
        self.assertEqual(problem_identity(solver, (q,)), before)
        energies = [spring_oracle(x)[0] for x in result['pathPositions']]
        self.assertTrue(all(after < before for before, after in zip(energies, energies[1:])))

    def test_positive_mass_magnitude_cannot_change_static_solution(self):
        outcomes = []
        for mass in (2.**-16, 1., 2.**16):
            with self.subTest(mass=mass):
                solver, potential, q = pinned(mass)
                result = static.solve_static_equilibrium(solver, potential, q)
                self.assert_candidate(solver, potential, q, result)
                outcomes.append(result['positions'])
        for q in outcomes:
            np.testing.assert_allclose(q[1], [1.75, 0., 0.], rtol=0, atol=2e-9)
        for q in outcomes[1:]:
            np.testing.assert_allclose(q, outcomes[0], rtol=0, atol=2e-9)

    def test_all_free_singular_translation_mode_has_valid_force_equilibrium(self):
        model = particles(points=[[0., 0., 0.], [2., .25, -.125]], masses=[1., 1024.])
        solver = GlobalSewingSolver(model, [{0: -1., 1: 1.}], .125)
        q = model.particle_q.numpy().astype(np.float64)
        potential = FixedPhysicalPotential(solver, [[1., 0., 0.]])
        result = static.solve_static_equilibrium(solver, potential, q)
        self.assert_candidate(solver, potential, q, result)
        # The equilibrium has a free translation. Do not impose a gauge or
        # assert that Newton's numerical choice is a physical centre of mass.
        np.testing.assert_allclose(result['positions'][1]-result['positions'][0], [1., 0., 0.], rtol=0, atol=2e-9)

    def test_native_membrane_and_swept_triangle_reach_known_rest_equilibrium(self):
        solver, potential, q = cloth_fixture()
        before = problem_identity(solver, (q,))
        segments = []; actual = static.triangle_sweep_safe
        def observe(first, last, faces):
            segments.append((first.tobytes(), last.tobytes()))
            return actual(first, last, faces)
        with patch.object(static, 'triangle_sweep_safe', observe):
            result = static.solve_static_equilibrium(solver, potential, q)
        self.assert_candidate(solver, potential, q, result)
        np.testing.assert_allclose(result['positions'], [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], rtol=0, atol=3e-9)
        self.assertLess(abs(result['physicalResponse']['energyJoules']), 1e-14)
        self.assertGreater(result['costs']['componentCalls']['membrane.derivatives'], 0)
        self.assertEqual(problem_identity(solver, (q,)), before)
        for first, last in zip(result['pathPositions'], result['pathPositions'][1:]):
            self.assertIn((first.tobytes(), last.tobytes()), segments)
            verify_triangle_sweep_exact(first, last, solver.faces)

    def test_stationary_and_inactive_inputs_still_require_fresh_response(self):
        for stationary, inactive in ((True, False), (False, True)):
            solver, potential, q = pinned(stationary=stationary, inactive=inactive)
            seen = []; original = FixedPhysicalPotential.evaluate
            def record(p, x):
                seen.append(x.copy()); return original(p, x)
            with self.subTest(stationary=stationary), patch.object(FixedPhysicalPotential, 'evaluate', record):
                result = static.solve_static_equilibrium(solver, potential, q, max_responses=2)
            self.assert_candidate(solver, potential, q, result)
            self.assertEqual(len(seen), 2)
            self.assertEqual(result['costs']['factorizationAttempts'], 0)
            self.assertEqual(len(result['pathPositions']), 1)
            np.testing.assert_array_equal(result['positions'], q)

    def test_response_budget_reserves_final_validation_without_partial_publication(self):
        solver, potential, q = pinned()
        before = q.copy()
        with self.assertRaises(static.StaticEquilibriumFailure) as caught:
            static.solve_static_equilibrium(solver, potential, q, max_responses=2)
        self.assertLessEqual(caught.exception.costs['responseAttempts'], 2)
        self.assertEqual(caught.exception.costs['responseAttempts'], caught.exception.costs['responseSuccesses'])
        self.assertEqual(caught.exception.costs['freshValidationAttempts'], 0)
        np.testing.assert_array_equal(q, before)

    def test_fresh_force_radius_cannot_be_ignored_at_stationary_center(self):
        solver, potential, q = pinned(stationary=True)
        original = FixedPhysicalPotential.evaluate; calls = 0
        def uncertain(p, x):
            nonlocal calls
            calls += 1
            response = original(p, x)
            if calls == 2:
                response['knownErrorBounds']['gradientMaxAbsoluteNewtons'] = rat(F(1, 1000000))
            return response
        with patch.object(FixedPhysicalPotential, 'evaluate', uncertain):
            with self.assertRaises(static.StaticEquilibriumFailure) as caught:
                static.solve_static_equilibrium(solver, potential, q)
        self.assertEqual(calls, 2)
        self.assertEqual(caught.exception.costs['freshValidationAttempts'], 1)
        self.assertEqual(caught.exception.costs['responseAttempts'], 2)

    def test_fresh_energy_or_energy_radius_change_cannot_erase_admission_check(self):
        original = FixedPhysicalPotential.evaluate
        for mode in ('energy', 'energy-radius'):
            solver, potential, q = pinned()
            seen, changed = set(), []
            initial_sha = state_sha(q)
            def corrupt(p, x):
                response = original(p, x)
                identity = state_sha(x)
                if identity in seen and identity != initial_sha:
                    changed.append(identity)
                    # Leave every force value unchanged: the independent final
                    # response must also preserve the accepted energy decision.
                    if mode == 'energy': response['energyJoules'] += 10.
                    else: response['knownErrorBounds']['energyJoules'] = rat(F(1, 4))
                seen.add(identity)
                return response
            with self.subTest(mode=mode), patch.object(FixedPhysicalPotential, 'evaluate', corrupt):
                with self.assertRaises(static.StaticEquilibriumFailure) as caught:
                    static.solve_static_equilibrium(solver, potential, q)
            self.assertEqual(len(changed), 1)
            self.assertEqual(caught.exception.costs['freshValidationAttempts'], 1)
            self.assertGreaterEqual(caught.exception.costs['responseSuccesses'], 3)

    def test_unknown_or_rounded_away_energy_drop_cannot_admit_a_moved_candidate(self):
        original = FixedPhysicalPotential.evaluate
        for mode in ('uncertain', 'rounded-away'):
            solver, potential, q = pinned()
            def corrupt(p, x):
                response = original(p, x)
                if mode == 'uncertain':
                    response['knownErrorBounds']['energyJoules'] = rat(1000)
                else:
                    response['energyJoules'] = 1e100
                return response
            with self.subTest(mode=mode), patch.object(FixedPhysicalPotential, 'evaluate', corrupt):
                with self.assertRaises(static.StaticEquilibriumFailure) as caught:
                    static.solve_static_equilibrium(solver, potential, q, max_backtracks=3)
            self.assertEqual(caught.exception.costs['freshValidationAttempts'], 0)
            self.assertGreater(caught.exception.costs['rejectedCandidates'], 0)
            self.assertLessEqual(caught.exception.costs['candidateAttempts'],
                                 3*caught.exception.costs['factorizationAttempts'])

    def test_triangle_hinge_and_contact_path_rejections_are_hard_boundaries(self):
        for route in ('triangle', 'hinge', 'contact'):
            if route == 'triangle':
                solver, potential, q = cloth_fixture()
                owner, name = static, 'triangle_sweep_safe'
            elif route == 'hinge':
                solver, q, controls = folded_control()
                potential = FixedPhysicalPotential(solver, EMPTY, **controls)
                owner, name = static, 'hinge_sweep_safe'
            else:
                solver, q = contact_cloth()
                potential = FixedPhysicalPotential(solver, EMPTY)
                owner, name = type(solver.contact), 'path_safe'
            before = problem_identity(solver, (q,))
            with self.subTest(route=route), patch.object(owner, name, return_value=False) as guard:
                with self.assertRaises(static.StaticEquilibriumFailure) as caught:
                    static.solve_static_equilibrium(solver, potential, q)
            self.assertGreater(guard.call_count, 0)
            self.assertGreater(sum(caught.exception.costs['guardCalls'].values()), 0)
            self.assertEqual(caught.exception.costs['responseAttempts'], 0)
            self.assertEqual(problem_identity(solver, (q,)), before)

    def test_path_guard_mutation_or_nonboolean_result_cannot_publish(self):
        for mode in ('caller', 'supplied-state', 'nonboolean'):
            solver, potential, q = pinned()
            def bad(first, last, faces):
                if mode == 'caller': q[1, 0] += 1.
                elif mode == 'supplied-state': last[1, 0] += 1.
                else: return np.bool_(True)
                return True
            with self.subTest(mode=mode), patch.object(static, 'triangle_sweep_safe', bad):
                with self.assertRaises(static.StaticEquilibriumFailure) as caught:
                    static.solve_static_equilibrium(solver, potential, q)
            self.assertEqual(caught.exception.costs['guardCalls']['triangles'], 1)
            self.assertEqual(caught.exception.costs['responseAttempts'], 0)

    def test_bounded_metric_failure_mutation_and_nondescent_preserve_attempts(self):
        for mode in ('unresolved', 'nonfinite', 'mutation', 'nondescent'):
            solver, potential, q = pinned()
            def bad(matrix, gradient):
                if mode == 'unresolved': return None
                if mode == 'nonfinite': return np.full(gradient.shape, np.nan)
                if mode == 'mutation': matrix.data[:] = 999.
                return gradient.copy()
            with self.subTest(mode=mode), patch.object(static, '_positive_definite_direction', side_effect=bad) as dispatch:
                with self.assertRaises(static.StaticEquilibriumFailure) as caught:
                    static.solve_static_equilibrium(solver, potential, q, max_backtracks=2)
            costs = caught.exception.costs
            self.assertEqual(costs['factorizationAttempts'], dispatch.call_count)
            self.assertGreater(dispatch.call_count, 0)
            self.assertLessEqual(dispatch.call_count, 20)
            self.assertEqual(costs['responseAttempts'], 1)
            self.assertEqual(costs['freshValidationAttempts'], 0)
            self.assertLessEqual(costs['candidateAttempts'], 2*dispatch.call_count)

    def test_forged_physical_response_identity_and_shape_reject_with_work_counted(self):
        original = FixedPhysicalPotential.evaluate
        attacks = [lambda r: r.update(positionsSha256='0'*64),
                   lambda r: r.update(controlSha256='0'*64),
                   lambda r: r.update(includesInertia=True),
                   lambda r: r.update(accepted=True),
                   lambda r: r.update(configurationIdentityChecked=False),
                   lambda r: r.update(gradientNewtons=r['gradientNewtons'][:-1]),
                   lambda r: r['knownErrorBounds'].update(gradientMaxAbsoluteNewtons=rat(-1))]
        for attack in attacks:
            solver, potential, q = pinned()
            def corrupt(p, x):
                result = original(p, x); attack(result); return result
            with self.subTest(attack=attacks.index(attack)), patch.object(FixedPhysicalPotential, 'evaluate', corrupt):
                with self.assertRaises(static.StaticEquilibriumFailure) as caught:
                    static.solve_static_equilibrium(solver, potential, q)
            self.assertEqual(caught.exception.costs['responseAttempts'], 1)
            self.assertEqual(caught.exception.costs['responseSuccesses'], 0)
            self.assertGreater(caught.exception.costs['componentCalls']['sewing.vector_response'], 0)

    def test_model_caller_and_potential_mutations_reject_after_counting_native_work(self):
        original = FixedPhysicalPotential.evaluate
        for mode in ('caller', 'model', 'potential', 'equivalent-owner', 'supplied-state'):
            solver, potential, q = pinned()
            def corrupt(p, x):
                response = original(p, x)
                if mode == 'caller': q[1, 0] += 1.
                elif mode == 'model': solver.mass[1] *= 2.
                elif mode == 'potential': object.__setattr__(p, '_description_bytes', b'{}')
                elif mode == 'equivalent-owner': object.__setattr__(p, '_solver', copy.copy(solver))
                else: x[1, 0] += 1.
                return response
            with self.subTest(mode=mode), patch.object(FixedPhysicalPotential, 'evaluate', corrupt):
                with self.assertRaises(static.StaticEquilibriumFailure) as caught:
                    static.solve_static_equilibrium(solver, potential, q)
            self.assertEqual(caught.exception.costs['responseAttempts'], 1)
            self.assertGreater(caught.exception.costs['componentCalls']['sewing.vector_response'], 0)
            self.assertEqual(caught.exception.costs['responseSuccesses'], 0)

    def test_physical_failure_costs_survive_post_call_mutation_guard(self):
        solver, potential, q = pinned()
        work = {'membrane.derivatives': 1, 'sewing.vector_response': 1}
        def fail(p, x):
            q[1, 0] += 1.
            raise PhysicalResponseFailure('injected after physical work', work)
        with patch.object(FixedPhysicalPotential, 'evaluate', fail):
            with self.assertRaises(static.StaticEquilibriumFailure) as caught:
                static.solve_static_equilibrium(solver, potential, q)
        self.assertEqual(caught.exception.costs['responseAttempts'], 1)
        self.assertEqual(caught.exception.costs['componentCalls'], work)
        self.assertEqual(caught.exception.costs['responseSuccesses'], 0)
        work['membrane.derivatives'] = 999
        self.assertEqual(caught.exception.costs['componentCalls']['membrane.derivatives'], 1)

    def test_wrong_potential_owner_raw_states_and_work_limits_reject(self):
        solver, potential, q = pinned(); other, foreign, _ = pinned()
        for bad in (foreign, object()):
            with self.subTest(potential=type(bad).__name__), self.assertRaises(static.StaticEquilibriumFailure) as caught:
                static.solve_static_equilibrium(solver, bad, q)
            self.assertEqual(caught.exception.costs['responseAttempts'], 0)
        invalid = (q.tolist(), q.astype(np.float32), q[:-1], np.full(q.shape, np.nan))
        for state in invalid:
            with self.subTest(state_type=type(state).__name__), self.assertRaises(static.StaticEquilibriumFailure) as caught:
                static.solve_static_equilibrium(solver, potential, state)
            self.assertEqual(caught.exception.costs['responseAttempts'], 0)
        for options in ({'force_tolerance_newtons': True}, {'force_tolerance_newtons': 1e-7},
                        {'force_tolerance_newtons': 0.}, {'max_responses': True},
                        {'max_responses': 1}, {'max_iterations': True}, {'max_backtracks': 0}):
            with self.subTest(options=options), self.assertRaises(static.StaticEquilibriumFailure) as caught:
                static.solve_static_equilibrium(solver, potential, q, **options)
            self.assertEqual(caught.exception.costs['responseAttempts'], 0)

    def test_result_arrays_and_failure_records_are_detached(self):
        solver, potential, q = pinned()
        description = potential.description(); before = q.copy()
        result = static.solve_static_equilibrium(solver, potential, q)
        self.assertFalse(np.shares_memory(result['positions'], q))
        for stage in result['pathPositions']:
            self.assertFalse(np.shares_memory(stage, q))
        result['positions'][:] = 999.
        result['physicalResponse']['gradientNewtons'][:] = 999.
        result['pathPositions'][0][:] = 999.
        np.testing.assert_array_equal(q, before)
        self.assertEqual(potential.description(), description)
        costs, trace = {'responseAttempts': 2, 'componentCalls': {'sample': 1}}, [{'kind': 'sample'}]
        failure = static.StaticEquilibriumFailure('detachment witness', costs, trace)
        costs['componentCalls']['sample'] = 999; trace[0]['kind'] = 'changed'
        self.assertEqual(failure.costs['componentCalls']['sample'], 1)
        self.assertEqual(failure.trace, [{'kind': 'sample'}])


if __name__ == '__main__':
    unittest.main()
