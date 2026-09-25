"""Independent coupled-stage, mechanical-unit and failure controls; stdlib only."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


def load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


producer = load_sibling('_lobatto_under_test', 'study_solver_lobatto.py')
prior = load_sibling('_exact_time_oracle_for_lobatto', 'test_study_solver_time_integration.py')
Q, solve_exact = prior.Root3, prior.solve_exact
ORACLE_DEPENDENCIES = ('test_study_solver_time_integration.py', 'study_solver_time_integration.py')


def dec(value):
    value = F(value)
    return D(value.numerator)/D(value.denominator)


def exact_tableau(method):
    if method == 'lobatto3a':
        a = ((F(0), F(0), F(0)), (F(5, 24), F(1, 3), F(-1, 24)),
             (F(1, 6), F(2, 3), F(1, 6)))
        return tuple(tuple(Q(v) for v in row) for row in a), (F(1, 6), F(2, 3), F(1, 6)), (1, 2)
    if method == 'gauss4':
        return ((Q(F(1, 4)), Q(F(1, 4), F(-1, 6))),
                (Q(F(1, 4), F(1, 6)), Q(F(1, 4)))), (F(1, 2), F(1, 2)), (0, 1)
    raise ValueError(method)


def polynomial_oracle(method, x0, v0, t0, h, mass, omega, duration, amplitude):
    """Four original RK X,V equations; never uses A² or producer functions."""
    a, weights, unknown = exact_tableau(method)
    n = len(weights)
    times = tuple(t0+h*sum(row, Q()) for row in a)
    coefficient = amplitude/duration**5
    loads = tuple(coefficient*(omega**2*t**5+20*t**3) for t in times)
    powers = tuple(coefficient*(5*omega**2*t**4+60*t**2) for t in times)
    known = tuple(i for i in range(n) if i not in unknown)
    matrix = [[Q() for _ in range(4)] for _ in range(4)]
    rhs = []
    for row, i in enumerate(unknown):
        matrix[row][row] = Q(F(1))
        for column, j in enumerate(unknown):
            matrix[row][2+column] = -h*a[i][j]
        rhs.append(x0+h*sum((a[i][j]*v0 for j in known), Q()))
    for row, i in enumerate(unknown):
        matrix[2+row][2+row] = Q(F(1))
        for column, j in enumerate(unknown):
            matrix[2+row][column] = h*omega**2*a[i][j]
        rhs.append(v0-h*sum((a[i][j]*(omega**2*x0-loads[j]) for j in known), Q())
                   + h*sum((a[i][j]*loads[j] for j in unknown), Q()))
    solved = solve_exact(matrix, rhs)
    positions = tuple(Q(x0) if i in known else solved[unknown.index(i)] for i in range(n))
    velocities = tuple(Q(v0) if i in known else solved[2+unknown.index(i)] for i in range(n))
    gradients = tuple(mass*(omega**2*positions[i]-loads[i]) for i in range(n))
    endpoint_x = x0+h*sum((weights[i]*velocities[i] for i in range(n)), Q())
    endpoint_v = v0-h/mass*sum((weights[i]*gradients[i] for i in range(n)), Q())
    work = -mass*h*sum((weights[i]*powers[i]*positions[i] for i in range(n)), Q())
    return dict(positions=positions, velocities=velocities, times=times, gradients=gradients,
                x=endpoint_x, v=endpoint_v, work=work)


class SampleSpy:
    def __init__(self, model, action=None):
        self.model, self.action, self.calls = model, action, []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def sample(self, x, t):
        self.calls.append((x, t))
        value = self.model.sample(x, t)
        return value if self.action is None else self.action(len(self.calls), value)


class LobattoIndependentTests(unittest.TestCase):
    def assert_close(self, got, wanted, tolerance=D('1e-65')):
        self.assertLessEqual(abs(got-wanted), tolerance*(1+abs(wanted)))

    def polynomial(self, mass=D(1)):
        return producer.Model('polynomial', mass, D('0.064'), omega=D(500), amplitude=D('0.0001'))

    def test_exact_tableau_order_four_stability_and_non_symplecticity(self):
        a = ((F(0), F(0), F(0)), (F(5, 24), F(1, 3), F(-1, 24)),
             (F(1, 6), F(2, 3), F(1, 6)))
        b, c = a[-1], (F(0), F(1, 2), F(1))
        ac = tuple(sum(a[i][j]*c[j] for j in range(3)) for i in range(3))
        ac2 = tuple(sum(a[i][j]*c[j]**2 for j in range(3)) for i in range(3))
        aac = tuple(sum(a[i][j]*ac[j] for j in range(3)) for i in range(3))
        self.assertEqual(tuple(sum(row) for row in a), c)
        self.assertEqual(sum(b), 1)
        self.assertEqual(sum(b[i]*c[i] for i in range(3)), F(1, 2))
        self.assertEqual(sum(b[i]*c[i]**2 for i in range(3)), F(1, 3))
        self.assertEqual(sum(b[i]*ac[i] for i in range(3)), F(1, 6))
        self.assertEqual(sum(b[i]*c[i]**3 for i in range(3)), F(1, 4))
        self.assertEqual(sum(b[i]*c[i]*ac[i] for i in range(3)), F(1, 8))
        self.assertEqual(sum(b[i]*ac2[i] for i in range(3)), F(1, 12))
        self.assertEqual(sum(b[i]*aac[i] for i in range(3)), F(1, 24))
        self.assertEqual(2*b[0]*a[0][0]-b[0]**2, F(-1, 36))
        for z in (F(0), F(1, 7), F(-1), F(-4), F(3)):
            stages = solve_exact([[1-z/3, z/24], [-2*z/3, 1-z/6]], [1+5*z/24, 1+z/6])
            self.assertEqual(stages[1], Q((1+z/2+z*z/12)/(1-z/2+z*z/12)))
        with localcontext() as ctx:
            ctx.prec = 100
            actual_a, actual_c, actual_b, unknown = producer.tableau('lobatto3a')
            self.assertEqual(unknown, (1, 2))
            for expected, actual in zip(a, actual_a):
                for x, y in zip(expected, actual):
                    self.assert_close(y, dec(x))
            for expected, actual in zip(c+b, actual_c+actual_b):
                self.assert_close(actual, dec(expected))

    def test_polynomial_full_four_equation_oracle_and_actual_stage_work(self):
        cases = [(F(0), F(0), F(0), F(1, 100), F(1)),
                 (F(3, 10000), F(-7, 1000), F(3, 1000), F(7, 10000), F(3, 2))]
        with localcontext() as ctx:
            ctx.prec = 100
            for method in producer.METHODS:
                for x, v, t, h, mass in cases:
                    expected = polynomial_oracle(method, x, v, t, h, mass, F(500), F(8, 125), F(1, 10000))
                    actual = producer.trial(method, self.polynomial(dec(mass)), dec(x), dec(v), dec(t), dec(h))
                    for key, oracle in [('stagePositionsMetres', 'positions'), ('stageVelocitiesMPerSecond', 'velocities'),
                                        ('stageTimesSeconds', 'times'), ('stageGradientsNewtons', 'gradients')]:
                        self.assertEqual(len(actual[key]), len(expected[oracle]))
                        for got, wanted in zip(actual[key], expected[oracle]):
                            self.assert_close(got, wanted.decimal())
                    for key, oracle in [('xMetres', 'x'), ('nativeVelocityMPerSecond', 'v'), ('parameterWorkJoules', 'work')]:
                        self.assertEqual(expected[oracle].b, 0)
                        self.assert_close(actual[key], expected[oracle].decimal())
                    self.assertEqual(actual['counts']['newtonUpdates'], 1)
                    self.assertLessEqual(max(map(abs, actual['forceResidualNewtons'])), D('1e-35'))
                    if method == 'lobatto3a':
                        self.assertEqual(actual['xMetres'], actual['stagePositionsMetres'][-1])
                        self.assertEqual(actual['nativeVelocityMPerSecond'], actual['stageVelocitiesMPerSecond'][-1])

    def test_models_mass_units_power_and_exact_inverse_solution(self):
        with localcontext() as ctx:
            ctx.prec = 100
            mass, a, b, x = F(3, 2), F(1, 10000), F(1, 20), F(1, 5000)
            model = producer.Model('inverse', dec(mass), D('0.064'), a=dec(a), b=dec(b))
            k = mass*a*a*b*b
            expected = (k/(2*x*x), -k/x**3, 3*k/x**4, F(0))
            for actual, wanted in zip(model.sample(dec(x), D('0.01')), expected):
                self.assert_close(actual, dec(wanted))
            for time in (D(0), D('0.064')/17, D('0.064')):
                px, pv = model.exact(time)
                energy, gradient, _, power = model.sample(px, time)
                self.assert_close(px*px, dec(a*a)+dec(b*b)*time*time)
                self.assert_close(pv*px, dec(b*b)*time)
                self.assert_close(dec(mass)*pv*pv/2+energy, dec(mass*b*b/2))
                self.assert_close(-gradient/dec(mass), dec(a*a*b*b)/px**3)
                self.assertEqual(power, 0)
            low, high = self.polynomial(D(1)), self.polynomial(D(3))
            for pos in (D(0), D('-0.0002'), D('0.0004')):
                one, three = low.sample(pos, D('0.02')), high.sample(pos, D('0.02'))
                for first, second in zip(one, three):
                    self.assert_close(second, 3*first)
                t, w, amp, duration = F(1, 50), F(500), F(1, 10000), F(8, 125)
                coefficient = amp/duration**5
                fp = coefficient*(5*w*w*t**4+60*t**2)
                self.assert_close(one[3], dec(-fp*F(pos)))
            self.assert_close(high.exact_work(), 3*low.exact_work())

    def test_inverse_coupled_residual_reconstruction_and_native_endpoint(self):
        with localcontext() as ctx:
            ctx.prec = 100
            model = producer.Model('inverse', D(1), D('0.064'), a=D('0.0001'), b=D('0.05'))
            x0, v0, h = model.a, D(0), D('0.00025')
            for method in producer.METHODS:
                row = producer.trial(method, model, x0, v0, D(0), h)
                x = row['stagePositionsMetres']
                gradient = tuple(-model.mass*model.a**2*model.b**2/q**3 for q in x)
                if method == 'lobatto3a':
                    residual = (3*model.mass*(x[2]-x0-h*v0)/h**2+gradient[1]+gradient[0]/2,
                                model.mass*(-48*(x[1]-x0-h*v0/2)+12*(x[2]-x0-h*v0))/h**2+gradient[2]-gradient[0])
                    interior = 1
                else:
                    root = D(3).sqrt()
                    c = (D(1)/2-root/6, D(1)/2+root/6)
                    displacement = tuple(x[i]-x0-h*c[i]*v0 for i in range(2))
                    residual = (model.mass*(6*displacement[0]+(-18+12*root)*displacement[1])/h**2+gradient[0],
                                model.mass*((-18-12*root)*displacement[0]+6*displacement[1])/h**2+gradient[1])
                    interior = 0
                self.assertTrue(all(q > 0 for q in x))
                for got, wanted in zip(row['forceResidualNewtons'], residual):
                    self.assert_close(got, wanted)
                self.assertLessEqual(max(map(abs, residual)), D('1.0000000001e-35'))
                self.assertNotEqual(row['nativeVelocityMPerSecond'], (row['xMetres']-x0)/h)
                self.assertNotEqual(row['nativeVelocityMPerSecond'], row['stageVelocitiesMPerSecond'][interior])
                self.assertEqual(row['parameterWorkJoules'], 0)
                if method == 'lobatto3a':
                    self.assertEqual(row['xMetres'], x[-1])
                    self.assertEqual(row['nativeVelocityMPerSecond'], row['stageVelocitiesMPerSecond'][-1])
                self.assertLess(abs(row['endpointPositionFormulaDefectMetres']), D('1e-38'))
                self.assertLess(abs(row['endpointVelocityFormulaDefectMPerSecond']), D('1e-38'))

    def test_initial_and_model_specific_domain_checks_are_counted(self):
        with localcontext() as ctx:
            ctx.prec = 100
            inverse = producer.Model('inverse', D(1), D('0.064'), a=D('0.0001'), b=D('0.05'))
            for method in producer.METHODS:
                for bad in (D(0), D('-0.0001')):
                    spy = SampleSpy(inverse)
                    with self.assertRaises(producer.TrialFailure) as caught:
                        producer.trial(method, spy, bad, D(0), D(0), D('0.001'))
                    self.assertEqual(caught.exception.reason, 'invalid-or-singular-trial')
                    self.assertEqual(caught.exception.counts['initialStateCalls'], 1)
                    self.assertEqual(caught.exception.counts['potentialCalls'], 1)
                    self.assertEqual(len(spy.calls), 1)
            self.assertIsInstance(self.polynomial().sample(D(0), D(0)), tuple)
            self.assertIsInstance(self.polynomial().sample(D(-1), D(0)), tuple)

    def test_fresh_stage_endpoint_budget_edges_and_initial_call(self):
        with localcontext() as ctx:
            ctx.prec = 100
            for method, total, stages in [('gauss4', 8, 2), ('lobatto3a', 9, 3)]:
                for maximum in (total-2, total-1, total):
                    policy = dict(producer.CONFIG['newton'], maximumPotentialCalls=maximum)
                    spy = SampleSpy(self.polynomial())
                    if maximum == total:
                        row = producer.trial(method, spy, D(0), D(0), D(0), D('0.001'), policy=policy)
                        self.assertEqual(row['counts']['potentialCalls'], total)
                        self.assertEqual(row['counts']['freshStageCalls'], stages)
                        self.assertEqual(row['counts']['freshEndpointCalls'], 1)
                        self.assertEqual(row['counts']['freshResidualValidations'], 1)
                        self.assertEqual(row['counts']['initialStateCalls'], 1)
                        self.assertEqual(row['counts']['budgetDeniedCalls'], 0)
                        self.assertEqual(len(spy.calls), total)
                    else:
                        with self.assertRaises(producer.TrialFailure) as caught:
                            producer.trial(method, spy, D(0), D(0), D(0), D('0.001'), policy=policy)
                        self.assertEqual(caught.exception.reason, 'potential-call-budget')
                        self.assertEqual(caught.exception.counts['potentialCalls'], maximum)
                        self.assertEqual(caught.exception.counts['budgetDeniedCalls'], 1)
                        self.assertEqual(caught.exception.counts['freshEndpointCalls'], 0)
                        self.assertEqual(len(spy.calls), maximum)

    def test_fresh_force_recheck_cannot_reuse_converged_old_samples(self):
        with localcontext() as ctx:
            ctx.prec = 100
            def perturb_fresh(number, value):
                return (value[0], value[1]+1, value[2], value[3]) if number == 6 else value
            for method, stages in [('gauss4', 2), ('lobatto3a', 3)]:
                spy = SampleSpy(self.polynomial(), perturb_fresh)
                with self.assertRaises(producer.TrialFailure) as caught:
                    producer.trial(method, spy, D(0), D(0), D(0), D('0.001'))
                self.assertEqual(caught.exception.reason, 'fresh-force-residual')
                self.assertEqual(caught.exception.counts['newtonUpdates'], 1)
                self.assertEqual(caught.exception.counts['freshStageCalls'], stages)
                self.assertEqual(caught.exception.counts['freshResidualValidations'], 1)
                self.assertEqual(caught.exception.counts['freshEndpointCalls'], 0)

    def test_rejected_line_search_counts_and_no_small_update_acceptance(self):
        with localcontext() as ctx:
            ctx.prec = 100
            def reject_first_candidate(number, value):
                if number == 4:
                    raise ValueError('deliberate candidate-domain witness')
                return value
            for method in producer.METHODS:
                spy = SampleSpy(self.polynomial(), reject_first_candidate)
                policy = dict(producer.CONFIG['newton'], maximumBacktracks=1)
                with self.assertRaises(producer.TrialFailure) as caught:
                    producer.trial(method, spy, D(0), D(0), D(0), D('0.001'), policy=policy)
                self.assertEqual(caught.exception.reason, 'backtracking-budget')
                self.assertEqual(caught.exception.counts['lineSearchTrials'], 1)
                self.assertEqual(caught.exception.counts['domainRejectedCandidates'], 1)
                self.assertEqual(caught.exception.counts['potentialCalls'], 4)
                self.assertEqual(caught.exception.counts['newtonUpdates'], 0)
                real_inverse = producer.inverse2
                invocations = []
                def stalled_inverse(matrix):
                    invocations.append(matrix)
                    return real_inverse(matrix) if len(invocations) == 1 else ((D(0), D(0)), (D(0), D(0)))
                with patch.object(producer, 'inverse2', side_effect=stalled_inverse):
                    with self.assertRaises(producer.TrialFailure) as stalled:
                        producer.trial(method, self.polynomial(), D(0), D(0), D(0), D('0.001'),
                                       policy=dict(producer.CONFIG['newton'], maximumBacktracks=2))
                self.assertEqual(stalled.exception.reason, 'backtracking-budget')
                self.assertEqual(stalled.exception.counts['residualRejectedCandidates'], 2)
                self.assertEqual(stalled.exception.counts['newtonUpdates'], 0)
                self.assertEqual(stalled.exception.counts['potentialCalls'], 7)

    def test_newton_budget_preserves_incomplete_force_residual(self):
        with localcontext() as ctx:
            ctx.prec = 100
            model = producer.Model('inverse', D(1), D('0.064'), a=D('0.0001'), b=D('0.05'))
            for method in producer.METHODS:
                with self.assertRaises(producer.TrialFailure) as caught:
                    producer.trial(method, model, model.a, D(0), D(0), D('0.001'),
                                   policy=dict(producer.CONFIG['newton'], maximumNewtonUpdates=1))
                self.assertEqual(caught.exception.reason, 'newton-update-budget')
                self.assertEqual(caught.exception.counts['newtonUpdates'], 1)
                self.assertEqual(caught.exception.counts['freshStageCalls'], 0)
                self.assertGreater(max(abs(D(r)) for r in caught.exception.detail), D('1e-35'))

    def test_failed_prefix_does_not_commit_attempt_or_advance_time(self):
        with localcontext() as ctx:
            ctx.prec = 100
            real_trial = producer.trial
            for failure_at in (1, 3):
                calls = []
                completed = []
                def injected(method, model, x, v, t, h, **options):
                    calls.append((x, v, t, h))
                    if len(calls) == failure_at:
                        raise producer.TrialFailure('deliberate-budget-witness', {'potentialCalls': 3}, 'retained')
                    answer = real_trial(method, model, x, v, t, h, **options)
                    completed.append(answer)
                    return answer
                with patch.object(producer, 'trial', side_effect=injected):
                    row = producer.trajectory('lobatto3a', self.polynomial(), 4)
                self.assertFalse(row['complete'])
                self.assertEqual(row['completedIntervals'], failure_at-1)
                self.assertEqual(D(row['committedTimeSeconds']), D('0.016')*(failure_at-1))
                self.assertEqual(len(row['history']), failure_at-1)
                self.assertEqual(row['failedTrial']['intervalIndex'], failure_at)
                self.assertTrue(row['failedTrial']['uncommitted'])
                self.assertEqual(row['failedTrial']['detail'], 'retained')
                self.assertIsNone(row['fullDurationWorkErrorJoules'])
                self.assertFalse(row['positionGoalPass'])
                self.assertFalse(row['nativeVelocityGoalPass'])
                self.assertEqual(len(calls), failure_at)
                self.assertEqual(row['counts']['potentialCalls'], 3+sum(x['counts']['potentialCalls'] for x in completed))
                if completed:
                    self.assertEqual(calls[-1][:2], (completed[-1]['xMetres'], completed[-1]['nativeVelocityMPerSecond']))
                    self.assertEqual(row['history'][-1]['xMetres'], str(completed[-1]['xMetres']))

    def test_invalid_policy_method_grid_and_diagnostic_boundary(self):
        base = producer.CONFIG['newton']
        for key, value in [('maximumNewtonUpdates', 0), ('maximumBacktracks', True),
                           ('maximumPotentialCalls', 301), ('forceResidualNewtons', '1e-34'),
                           ('forceResidualNewtons', 'NaN'), ('armijoFraction', '1')]:
            with self.assertRaises(ValueError):
                producer.checked_policy(dict(base, **{key: value}))
        with self.assertRaises(ValueError):
            producer.tableau('two-backward-euler')
        for n in (0, -2, 1, 3, True, 2.0, 2732):
            with self.assertRaises(ValueError):
                producer.trajectory('lobatto3a', self.polynomial(), n)
        with localcontext() as ctx:
            ctx.prec = 100
            target, guard = D('0.000001'), D(producer.CONFIG['decisionGuardAbsolute'])
            self.assertTrue(producer.diagnostic_pass(target-2*guard, target))
            self.assertFalse(producer.diagnostic_pass(target+2*guard, target))
            for value in (target-guard, target, target+guard):
                self.assertIsNone(producer.diagnostic_pass(value, target))

    def test_precision_history_counters_are_separate_from_state_values(self):
        comparison = {'maximumAbsoluteDifference': D(0), 'numericFieldsCompared': 0,
                      'stepsWithDifferentCounters': 0, 'sameHistorySchema': True}
        primary = {'x': '0.1', 'stages': ['0.05', '0.1'], 'counts': {'potentialCalls': 8}}
        repeated = {'x': '0.10000000000000000001', 'stages': ['0.05', '0.1'], 'counts': {'potentialCalls': 10}}
        producer.compare_history(repeated, primary, comparison)
        self.assertEqual(comparison['stepsWithDifferentCounters'], 1)
        self.assertEqual(comparison['numericFieldsCompared'], 3)
        self.assertEqual(comparison['maximumAbsoluteDifference'], D('1e-20'))
        self.assertTrue(comparison['sameHistorySchema'])
        self.assertEqual(comparison['maximumDifferenceWitness'],
                         {'field': 'x', 'element': 0, 'actual': repeated['x'], 'expected': primary['x']})
        for malformed in ({'x': '0.1'}, dict(repeated, stages=['0.05'])):
            changed = dict(comparison, sameHistorySchema=True)
            producer.compare_history(malformed, primary, changed)
            self.assertFalse(changed['sameHistorySchema'])

    def test_precision_disagreement_retains_unmatched_success_and_failure_outcomes(self):
        with localcontext() as ctx:
            ctx.prec = 100
            model = self.polynomial()
            with patch.object(producer, 'trial', side_effect=producer.TrialFailure(
                    'primary-first-step-failure', {'potentialCalls': 3}, 'primary retained')):
                primary = producer.trajectory('lobatto3a', model, 2)
            self.assertEqual(primary['history'], [])
            repeated = producer.trajectory('lobatto3a', model, 2, expected=primary)
            comparison = repeated['historyComparison']
            self.assertTrue(repeated['complete'])
            self.assertFalse(comparison['sameCompletedPrefix'])
            self.assertFalse(comparison['sameFailureReason'])
            self.assertTrue(comparison['sameHistorySchema'])
            self.assertEqual(comparison['numericFieldsCompared'], 0)
            self.assertEqual([item['intervalIndex'] for item in comparison['unmatchedSuccessfulSteps']], [1, 2])
            final = comparison['unmatchedSuccessfulSteps'][-1]['record']
            self.assertIn('stagePositionsMetres', final)
            self.assertIn('nativeVelocityMPerSecond', final)
            self.assertEqual(D(repeated['committedTimeSeconds']), model.duration)

            complete = producer.trajectory('lobatto3a', model, 2)
            changed_decision = dict(complete, positionGoalPass=not complete['positionGoalPass'])
            repeated = producer.trajectory('lobatto3a', model, 2, expected=changed_decision)
            self.assertTrue(repeated['historyComparison']['sameCompletedPrefix'])
            self.assertFalse(repeated['historyComparison']['sameDecisions'])
            self.assertTrue(repeated['historyComparison']['sameFailureReason'])

            with patch.object(producer, 'trial', side_effect=producer.TrialFailure(
                    'different-repeat-failure', {'potentialCalls': 5}, 'repeat retained')):
                failed = producer.trajectory('lobatto3a', model, 2, expected=primary)
            self.assertTrue(failed['historyComparison']['sameCompletedPrefix'])
            self.assertFalse(failed['historyComparison']['sameFailureReason'])
            self.assertEqual(failed['failedTrial']['reason'], 'different-repeat-failure')
            self.assertEqual(failed['failedTrial']['counts'], {'potentialCalls': 5})
            self.assertEqual(D(failed['committedTimeSeconds']), 0)
            self.assertEqual(failed['historyComparison']['unmatchedSuccessfulSteps'], [])


if __name__ == '__main__':
    unittest.main()
