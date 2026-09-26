"""Physical counterexamples for bounded static search-order adaptation.

Two native vector springs have the exact energy 2|x|^2 + |y|^2/2.
Only the search metric/directions or an explicitly enlarged error radius are
injected; native energies, gradients, geometry guards and fresh validation run.
"""
from fractions import Fraction as F
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import eye

import solver_static_equilibrium as static
from solver_global_sewing import GlobalSewingSolver
from solver_physical_response import FixedPhysicalPotential
from solver_temporal_control import problem_identity
from test_solver_sewing_activation_integration import particles


def fraction(record):
    return F(int(record['numerator']), int(record['denominator']))


def springs():
    model = particles(points=[[0., 0., 0.], [.125, 0., 0.], [1., 0., 0.]],
                      masses=[0., 1., 1.])
    solver = GlobalSewingSolver(model, [{0: -1., 1: 1.}, {0: -1., 2: 1.}], .25)
    potential = FixedPhysicalPotential(solver, np.zeros((2, 3)), sewing_activation=[1., .25])
    return solver, potential, model.particle_q.numpy().astype(np.float64)


def oracle(q):
    x = [F(float(a)) - F(float(b)) for a, b in zip(q[1], q[0])]
    y = [F(float(a)) - F(float(b)) for a, b in zip(q[2], q[0])]
    energy = 2 * sum(a*a for a in x) + sum(b*b for b in y) / 2
    free_force = max([abs(4*a) for a in x] + [abs(b) for b in y])
    return energy, free_force


def identity_metric_response(callback=None):
    original = FixedPhysicalPotential.evaluate
    seen = []

    def evaluate(potential, q):
        response = original(potential, q)
        response['searchMatrixNewtonsPerMetre'] = eye(q.size, format='csr', dtype=np.float64)
        if callback is not None:
            callback(response, q, len(seen) + 1)
        seen.append((q.copy(), response))
        return response

    return evaluate, seen


def state(initial, first, second=0.):
    q = initial.copy()
    q[1] = [first, 0., 0.]
    q[2] = [second, 0., 0.]
    return q


class StaticAdaptationTests(unittest.TestCase):
    def assert_candidate(self, initial, result):
        self.assertEqual(result['profile'], 'fixed-control-static-equilibrium-v3')
        self.assertEqual(result['searchRegularizationPolicy'], 'best-force-guided-shift-order-v1')
        self.assertIs(result['accepted'], False)
        self.assertIs(result['stabilityEstablished'], False)
        self.assertIs(result['includesInertia'], False)
        np.testing.assert_array_equal(result['positions'][0], initial[0])
        np.testing.assert_array_equal(result['pathPositions'][0], initial)
        np.testing.assert_array_equal(result['pathPositions'][-1], result['positions'])
        expected_force = oracle(result['positions'])[1]
        radius = fraction(result['physicalResponse']['knownErrorBounds']['gradientMaxAbsoluteNewtons'])
        self.assertEqual(fraction(result['forceResidualUpperNewtons']), expected_force + radius)
        self.assertLessEqual(expected_force + radius, F(1e-8))
        self.assertEqual(result['costs']['freshValidationAttempts'], 1)
        for first, last in zip(result['pathPositions'], result['pathPositions'][1:]):
            self.assertLess(oracle(last)[0], oracle(first)[0])

    def test_native_energy_decrease_with_force_growth_changes_next_search(self):
        solver, potential, initial = springs()
        before = problem_identity(solver, (initial,))
        evaluate, seen = identity_metric_response()
        with patch.object(FixedPhysicalPotential, 'evaluate', evaluate):
            result = static.solve_static_equilibrium(solver, potential, initial)
        self.assert_candidate(initial, result)
        admitted = [row for row in result['trace'] if row['kind'] == 'candidate' and row['admitted']]
        # Actual sparse directions, not a scripted solve: H_search=I produces
        # x=-3/8,y=0 from x=1/8,y=1. Energy falls while the free force grows.
        np.testing.assert_array_equal(result['pathPositions'][1], state(initial, -.375))
        self.assertEqual(oracle(initial), (F(17, 32), F(1)))
        self.assertEqual(oracle(result['pathPositions'][1]), (F(9, 32), F(3, 2)))
        self.assertIs(admitted[0]['improvedBestForce'], False)
        self.assertEqual(admitted[0]['nextPreferredDimensionlessShift'], 1e-9)
        self.assertEqual(admitted[1]['preferredDimensionlessShift'], 1e-9)
        self.assertEqual(admitted[1]['dimensionlessDiagonalShift'], 1e-9)
        self.assertIs(admitted[1]['improvedBestForce'], True)
        self.assertEqual(problem_identity(solver, (initial,)), before)
        self.assertEqual(len(seen), result['costs']['responseAttempts'])

    def test_recovery_to_previous_best_does_not_undo_regularization(self):
        solver, potential, initial = springs()
        path = [initial, state(initial, -.375), state(initial, -.25),
                state(initial, -.125), state(initial, 0.)]
        dispatched = []

        def direction(matrix, gradient):
            index = len(dispatched)
            dispatched.append(matrix.copy())
            return (path[index + 1] - path[index]).ravel()[solver.free].copy()

        evaluate, _ = identity_metric_response()
        with patch.object(FixedPhysicalPotential, 'evaluate', evaluate), \
             patch.object(static, '_positive_definite_direction', side_effect=direction):
            result = static.solve_static_equilibrium(solver, potential, initial)
        self.assert_candidate(initial, result)
        admitted = [row for row in result['trace'] if row['kind'] == 'candidate' and row['admitted']]
        self.assertEqual([oracle(q)[1] for q in path], [F(1), F(3, 2), F(1), F(1, 2), F(0)])
        self.assertEqual([fraction(row['priorBestForceResidualUpperNewtons']) for row in admitted],
                         [F(1), F(1), F(1), F(1, 2)])
        self.assertEqual([row['improvedBestForce'] for row in admitted], [False, False, True, True])
        self.assertEqual([row['nextPreferredDimensionlessShift'] for row in admitted],
                         [1e-9, 1e-8, 1e-9, 0.])
        self.assertEqual(result['costs']['factorizationAttempts'], len(dispatched))
        self.assertEqual(len(dispatched), 4)

    def test_energy_rejected_zero_force_trial_cannot_update_best_or_publish(self):
        solver, potential, initial = springs()
        zero, spike = state(initial, 0.), state(initial, -.375)
        starts, ends = [initial, initial, spike], [zero, spike, zero]
        dispatched = []

        def direction(matrix, gradient):
            index = len(dispatched)
            dispatched.append(matrix.copy())
            return (ends[index] - starts[index]).ravel()[solver.free].copy()

        def uncertain_energy(response, q, call):
            if call == 2:
                # Native force is exactly zero, but its energy certificate is
                # insufficient to establish a moved candidate's decrease.
                self.assertEqual(oracle(q)[1], 0)
                response['knownErrorBounds']['energyJoules'] = {'numerator': '1', 'denominator': '1'}

        evaluate, seen = identity_metric_response(uncertain_energy)
        with patch.object(FixedPhysicalPotential, 'evaluate', evaluate), \
             patch.object(static, '_positive_definite_direction', side_effect=direction):
            result = static.solve_static_equilibrium(solver, potential, initial, max_backtracks=1)
        self.assert_candidate(initial, result)
        candidates = [row for row in result['trace'] if row['kind'] == 'candidate']
        self.assertEqual(len(candidates), 3)
        self.assertFalse(candidates[0]['admitted'])
        self.assertEqual(candidates[0]['reason'], 'conditional energy decrease not established')
        self.assertNotIn('improvedBestForce', candidates[0])
        self.assertEqual(fraction(candidates[1]['priorBestForceResidualUpperNewtons']), 1)
        self.assertIs(candidates[1]['improvedBestForce'], False)
        self.assertEqual(candidates[1]['nextPreferredDimensionlessShift'], 1e-8)
        self.assertEqual(result['costs']['rejectedCandidates'], 1)
        self.assertEqual(len(seen), 5)  # Initial, rejected, two admitted, fresh.

    def test_upper_preference_is_bounded_and_lower_shifts_remain_fallbacks(self):
        solver, potential, initial = springs()
        spike, zero = state(initial, -.375), state(initial, 0.)
        grid = (0., 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
                .001, .01, .1, 1., 10., 100., 1000.)
        seen_shifts = []

        def direction(matrix, gradient):
            seen_shifts.append(float(matrix.diagonal()[0]) - 1.)
            if len(seen_shifts) == len(grid):
                return (spike - initial).ravel()[solver.free].copy()
            if len(seen_shifts) == len(grid) + 2:
                return (zero - spike).ravel()[solver.free].copy()
            return None

        evaluate, _ = identity_metric_response()
        with patch.object(FixedPhysicalPotential, 'evaluate', evaluate), \
             patch.object(static, '_positive_definite_direction', side_effect=direction):
            result = static.solve_static_equilibrium(solver, potential, initial, max_backtracks=1)
        self.assert_candidate(initial, result)
        np.testing.assert_allclose(seen_shifts, [*grid, 1000., 0.], rtol=1e-12, atol=2e-13)
        admitted = [row for row in result['trace'] if row['kind'] == 'candidate' and row['admitted']]
        self.assertEqual(admitted[0]['nextPreferredDimensionlessShift'], 1000.)
        self.assertEqual(admitted[1]['preferredDimensionlessShift'], 1000.)
        self.assertEqual(admitted[1]['dimensionlessDiagonalShift'], 0.)
        self.assertEqual(admitted[1]['nextPreferredDimensionlessShift'], 0.)
        self.assertEqual(result['costs']['factorizationAttempts'], 16)
        self.assertEqual(result['costs']['candidateAttempts'], 2)

    def test_iteration_failure_preserves_work_and_next_solve_resets_preference(self):
        solver, potential, initial = springs()
        before = problem_identity(solver, (initial,))
        evaluate, seen = identity_metric_response()
        with patch.object(FixedPhysicalPotential, 'evaluate', evaluate):
            with self.assertRaisesRegex(static.StaticEquilibriumFailure, 'iteration budget') as caught:
                static.solve_static_equilibrium(solver, potential, initial, max_iterations=1)
            first_count = len(seen)
            result = static.solve_static_equilibrium(solver, potential, initial)
        failure = caught.exception
        self.assertEqual(failure.costs['iterations'], 1)
        self.assertEqual(failure.costs['factorizationAttempts'], 1)
        self.assertEqual(failure.costs['responseAttempts'], first_count)
        self.assertEqual(first_count, 2)
        self.assertEqual(failure.costs['freshValidationAttempts'], 0)
        old = [row for row in failure.trace if row['kind'] == 'candidate' and row['admitted']]
        self.assertEqual(old[0]['nextPreferredDimensionlessShift'], 1e-9)
        new = [row for row in result['trace'] if row['kind'] == 'candidate']
        self.assertEqual(new[0]['preferredDimensionlessShift'], 0.)
        self.assertEqual(new[0]['dimensionlessDiagonalShift'], 0.)
        self.assert_candidate(initial, result)
        self.assertEqual(problem_identity(solver, (initial,)), before)


if __name__ == '__main__':
    unittest.main()
