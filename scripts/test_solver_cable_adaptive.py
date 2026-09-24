"""Fixed-cable adaptive publication tests with explicitly mocked equilibrium.

The fixture calls real cable diagnostics and energy/work integration but supplies
its convergence claim. These tests verify orchestration, not cloth mechanics,
contact paths, captured source admission or garment construction.
"""
import copy
from fractions import Fraction as F
import unittest
from unittest import mock

import numpy as np
from scipy.sparse import csr_matrix

from solver_adaptive_contact import adaptive_contact_step
from solver_bending import ElasticDihedralBending
from solver_cable_integration import CableControl, _rat, _rational, stationarity
from solver_continuous_cable_sewing import ContinuousCableSewing
import solver_energy_balance


def control(*, strict=False):
    anchor = lambda vertex: [{'vertex': vertex, 'weight': {'numerator': '1', 'denominator': '1'}}]
    potential = ContinuousCableSewing(2, [{
        'id': 'synthetic-fixed-cable',
        'positiveStart': anchor(0), 'positiveEnd': anchor(0),
        'negativeStart': anchor(1), 'negativeEnd': anchor(1),
        'targetsMeters': [.001, .001], 'referenceLengthMeters': {'numerator': '1', 'denominator': '8'},
        'stiffnessDensityNPerM2': 8., 'activation': 1.,
    }])
    return CableControl(potential, {
        'energy_tolerance_joules': 1e-80 if strict else 1e-16,
        'gradient_tolerance_newtons': 1e-80 if strict else 1e-12,
        'hessian_tolerance_newtons_per_meter': 1e-80 if strict else 1e-10,
        'absolute_tolerance_joules': 1e-80 if strict else 1e-16,
        'max_boundary_depth': 80, 'max_boundary_panels': 256,
        'moment_max_terms': 128, 'moment_max_panels': 256, 'moment_max_depth': 64,
    })


class ReportFixture:
    """No force solve: real fixed-cable mechanics, mocked total equilibrium."""
    def __init__(self):
        self.continuous_cable = control()
        self.material_grippers = self.fold_actuation = self.controlled_fold_actuation = None
        self.mass, self.active = np.ones(2), np.ones(2, dtype=bool)
        self.sewing = csr_matrix((0, 2)); self.compliance = .04; self.sewing_mode = 'vector'
        self.poses, self.faces = np.empty((0, 2, 2)), np.empty((0, 3), dtype=int)
        self.areas, self.materials = np.empty(0), np.empty((0, 3))
        self.bending = ElasticDihedralBending(2, np.empty((0, 4), dtype=int), [], [], [])
        self.calls, self.after_step, self.reject_above, self.interrupt_at = [], None, None, None
        self.interruption = KeyboardInterrupt

    def step(self, q, v, target, duration, **options):
        self.calls.append((q.copy(), v.copy(), target.copy(), duration))
        if self.interrupt_at == len(self.calls):
            q[:], v[:] = 999., 888.
            raise self.interruption('synthetic cable trial interruption')
        if self.reject_above is not None and duration > self.reject_above:
            q[:], v[:] = 999., 888.
            target.resize((1, 3), refcheck=False); target[:] = 777.
            return q, v, {'converged': False, 'gradientInfinityNorm': 1.}
        candidate = q+duration*v
        velocity = (candidate-q)/duration
        report = {'converged': True, 'gradientInfinityNorm': 0.}
        if self.continuous_cable is not None:
            diagnostic = self.continuous_cable.diagnostics(candidate)
            error = _rational(diagnostic['certificate']['gradientMaxAbsoluteErrorBoundNewtons'])
            report.update(continuousCable=diagnostic,
                cableStationarity=stationarity(np.array([0.]), error, error, F()))
        self.last_report = report
        if self.after_step is not None and len(self.calls) == 2:
            changed = self.after_step(self, q, v, target, candidate, velocity, report)
            if changed is not None:
                candidate, velocity = changed
        return candidate, velocity, report


class Journal:
    def __init__(self, events):
        self.events, self.rows = events, []

    def start(self, row):
        self.events.append(('start', row['attemptId']))

    def outcome(self, row, q=None, v=None):
        self.rows.append((copy.deepcopy(row), None if q is None else q.copy(), None if v is None else v.copy()))
        self.events.append(('outcome', row['outcome']))

    def finish(self, reason, complete):
        self.events.append(('finish', complete))


class CableAdaptiveTests(unittest.TestCase):
    def setup(self):
        return ReportFixture(), np.array([[0., 0., 0.], [.002, 0., 0.]]), np.full((2, 3), .001)

    def run_case(self, *, step_attack=None, work_attack=None, solver=None, max_depth=0, subdivisions=4, journal_type=Journal):
        default, q, v = self.setup()
        solver = default if solver is None else solver
        solver.after_step = step_attack
        original_q, original_v = q.copy(), v.copy()
        events, accepted, work_calls = [], [], []
        journal = journal_type(events)
        real_energy = solver_energy_balance.global_energy_transition
        def work(*args, **kwargs):
            result = real_energy(*args, **kwargs)
            work_calls.append(([x.copy() for x in args[1:7]], copy.deepcopy(kwargs)))
            events.append(('work', len(work_calls)))
            if len(work_calls) == 2 and work_attack is not None:
                work_attack(solver, args, kwargs, result)
            return result
        def accept(q, v, row):
            self.assertEqual(events[-1], ('outcome', 'accepted'))
            accepted.append((q.copy(), v.copy(), copy.deepcopy(row)))
            q[:], v[:] = 99., 88.
            row['step']['gradientInfinityNorm'] = 77.
            events.append(('callback', row['attemptId']))
        with mock.patch.object(solver_energy_balance, 'global_energy_transition', side_effect=work):
            final, velocity, report = adaptive_contact_step(solver, q, v, np.empty((0, 3)), np.empty((0, 3)), .4,
                max_depth=max_depth, initial_subdivisions=subdivisions, on_accept=accept, attempt_journal=journal)
        np.testing.assert_array_equal(q, original_q); np.testing.assert_array_equal(v, original_v)
        return final, velocity, report, journal, accepted, work_calls, events, solver

    def assert_prefix(self, result):
        final, velocity, report, journal, accepted, _, _, _ = result
        self.assertFalse(report['complete']); self.assertEqual(report['completedFraction'], .25)
        self.assertEqual(len(report['acceptedSteps']), 1); self.assertEqual(len(report['rejectedSteps']), 1)
        self.assertEqual(len(accepted), 1)
        self.assertEqual([row[0]['outcome'] for row in journal.rows], ['accepted', 'rejected'])
        self.assertIsNone(journal.rows[-1][1]); self.assertIsNone(journal.rows[-1][2])
        np.testing.assert_array_equal(final, accepted[0][0]); np.testing.assert_array_equal(velocity, accepted[0][1])
        self.assertIs(report['accepted'], False)

    def test_fixed_only_work_precedes_every_journal_and_callback(self):
        final, velocity, report, journal, accepted, work, events, solver = self.run_case()
        self.assertTrue(report['complete']); self.assertEqual(len(work), 4); self.assertEqual(len(accepted), 4)
        self.assertEqual(report['continuousCable'], solver.continuous_cable.description())
        for index, event in enumerate(events):
            if event == ('outcome', 'accepted'):
                self.assertEqual(events[index-1][0], 'work')
        for row in report['acceptedSteps']:
            energy = row['step']['energyBalance']
            self.assertEqual(energy['cableParameterWorkJoules'], 0.)
            self.assertIn('continuousCableEnergy', energy)
            self.assertEqual(row['step']['gradientInfinityNorm'], 0.)
        np.testing.assert_array_equal(final, accepted[-1][0]); np.testing.assert_array_equal(velocity, accepted[-1][1])

    def test_fresh_candidate_energy_definition_and_certificate_forgery_reject(self):
        attacks = [lambda d:d.__setitem__('energyJoules', d['energyJoules']+1e-8),
            lambda d:d['definition'].__setitem__('accepted', True),
            lambda d:d['certificate'].__setitem__('positionsSha256', '0'*64),
            lambda d:d['certificate'].__setitem__('inputSha256', '0'*64),
            lambda d:d['certificate'].__setitem__('verified', 1),
            lambda d:d['certificate'].__setitem__('sourceControlsInstalled', True),
            lambda d:d['certificate'].pop('gradientMaxAbsoluteErrorBoundNewtons')]
        for i, mutate in enumerate(attacks):
            with self.subTest(attack=i):
                result = self.run_case(step_attack=lambda s,q,v,t,c,w,r:mutate(r['continuousCable']))
                self.assert_prefix(result); self.assertEqual(len(result[5]), 1)

    def test_stationarity_includes_error_and_requires_exact_complete_fields(self):
        attacks = [lambda r:r.pop('cableStationarity'),
            lambda r:r['cableStationarity'].pop('scope'),
            lambda r:r['cableStationarity'].__setitem__('toleranceNewtons', 1.),
            lambda r:r['cableStationarity'].__setitem__('gradientInfinityNorm', False),
            lambda r:r['cableStationarity'].__setitem__('totalGradientErrorBoundNewtons', _rat(F(1))),
            lambda r:r['cableStationarity'].__setitem__('assemblyRoundingBoundNewtons', _rat(-F(1))),
            lambda r:r['cableStationarity'].__setitem__('stationarityUpperBoundNewtons', _rat(F())),
            lambda r:r['cableStationarity'].__setitem__('cableGradientErrorBoundNewtons', {'numerator': False, 'denominator': '1'}),
            lambda r:r.__setitem__('gradientInfinityNorm', float('nan'))]
        def unresolved(r):
            cable = _rational(r['continuousCable']['certificate']['gradientMaxAbsoluteErrorBoundNewtons'])
            r['gradientInfinityNorm'] = 9e-7
            r['cableStationarity'] = stationarity(np.array([9e-7]), cable+F(2e-7), cable, F(2e-7))
        attacks.append(unresolved)
        for i, mutate in enumerate(attacks):
            with self.subTest(attack=i):
                result = self.run_case(step_attack=lambda s,q,v,t,c,w,r:mutate(r))
                self.assert_prefix(result); self.assertEqual(len(result[5]), 1)
        # The raw midpoint residual meets the legacy threshold; its valid
        # conditional upper bound does not. No tolerance is relaxed.
        self.assertLess(9e-7, 1e-6)

    def test_control_replacement_and_post_work_diagnostic_mutation_reject(self):
        result = self.run_case(step_attack=lambda s,q,v,t,c,w,r:setattr(s, 'continuous_cable', control()))
        self.assert_prefix(result); self.assertEqual(len(result[5]), 1)
        for mutate in (lambda s:setattr(s, 'continuous_cable', control()),
                       lambda s:s.last_report['continuousCable'].__setitem__('energyJoules', 88.),
                       lambda s:s.last_report['cableStationarity'].__setitem__('scope', 'forged'),
                       lambda s:s.last_report.__setitem__('converged', False),
                       lambda s:s.last_report.__setitem__('extraAfterWork', 1)):
            self.assert_prefix(self.run_case(work_attack=lambda s,a,k,e:mutate(s)))

    def test_work_input_mutations_and_exceptions_preserve_accepted_prefix(self):
        for index in range(1, 7):
            with self.subTest(argument=index):
                def mutate(s, args, kwargs, energy):
                    if args[index].size:
                        args[index].flat[0] = 777.
                    else:
                        args[index].resize((1, 3), refcheck=False); args[index][:] = 777.
                self.assert_prefix(self.run_case(work_attack=mutate))
        def mutation_raise(s, args, kwargs, energy):
            args[1][:] = 999.
            raise ValueError('synthetic work failure after mutation')
        self.assert_prefix(self.run_case(work_attack=mutation_raise))

    def test_complete_work_fields_and_nested_certificate_forgery_reject(self):
        fields = ('cableBeforeJoules', 'cableAfterJoules', 'cableFixedParameterChangeJoules', 'cableParameterWorkJoules',
                  'targetParameterWorkJoules', 'externalParameterWorkJoules')
        for field in fields:
            for kind in ('missing', 'boolean', 'nonfinite'):
                def attack(s, args, kwargs, energy):
                    if kind == 'missing': energy.pop(field)
                    else: energy[field] = False if kind == 'boolean' else float('inf')
                with self.subTest(field=field, kind=kind):
                    self.assert_prefix(self.run_case(work_attack=attack))
        for mutate in (lambda e:e.pop('continuousCableEnergy'),
                       lambda e:e.__setitem__('accepted', True),
                       lambda e:e.__setitem__('accepted', 0),
                       lambda e:e.__setitem__('cableParameterWorkJoules', 1.),
                       lambda e:e['continuousCableEnergy'].pop('work'),
                       lambda e:e['continuousCableEnergy']['work']['certificate'].__setitem__('endPositionsSha256', '0'*64),
                       lambda e:e['continuousCableEnergy']['after']['definition'].__setitem__('parametersFixed', False)):
            self.assert_prefix(self.run_case(work_attack=lambda s,a,k,e:mutate(e)))

    def test_propagated_work_uncertainty_and_exact_aggregation_cannot_be_repaired_partially(self):
        def payload(energy):
            return energy['continuousCableEnergy']
        attacks = [
            lambda e:payload(e)['errorBoundsJoules'].pop('cableBeforeJoules'),
            lambda e:payload(e)['errorBoundsJoules'].__setitem__('cableFixedParameterChangeJoules', _rat(F(1))),
            lambda e:payload(e)['assemblyRoundingBoundsJoules'].__setitem__('mechanicalChangeJoules', _rat(F(1))),
            lambda e:payload(e)['aggregationTermsJoules']['mechanicalChangeJoules'].__setitem__('cableFixedParameterChangeJoules', 0.7),
            lambda e:payload(e)['aggregationTermsJoules']['mechanicalChangeMinusTargetWorkJoules'].pop('contactChangeJoules'),
            lambda e:payload(e)['work']['certificate'].__setitem__('requestedToleranceJoules', _rat(F(1))),
        ]
        for index, mutate in enumerate(attacks):
            with self.subTest(attack=index):
                self.assert_prefix(self.run_case(work_attack=lambda s,a,k,e:mutate(e)))

    def test_diagnostic_helper_mutation_is_isolated_before_acceptance(self):
        real = CableControl.diagnostics
        calls = []
        def diagnostics(control, positions):
            result = real(control, positions)
            calls.append(1)
            # Each accepted step calls fixture diagnostics, then adaptive
            # before-work and after-work diagnostics, then fresh prior-state
            # energy diagnostics. Attack second step's before-work
            # validation, not the mocked solver itself.
            if len(calls) == 6:
                positions[:] = 777.
            return result
        with mock.patch.object(CableControl, 'diagnostics', diagnostics):
            result = self.run_case()
        self.assert_prefix(result)
        self.assertEqual(len(result[5]), 1)

    def test_fresh_work_binding_rejects_consistent_energy_shift_and_plausible_work_forgery(self):
        from solver_cable_integration import round_sum
        validated_metadata = []
        def shifted(solver, args, kwargs, energy):
            payload = energy['continuousCableEnergy']
            for name, field in (('before', 'cableBeforeJoules'), ('after', 'cableAfterJoules')):
                value = payload[name]['energyJoules']+2.**-40
                payload[name]['energyJoules'] = energy[field] = value
            # Endpoint difference/intervals and declared identities remain
            # consistent. Publication must nevertheless use fresh numbers.
            solver_energy_balance.validate_continuous_cable_energy(solver.continuous_cable, args[1], args[2], energy)
            validated_metadata.append('shift')
        self.assert_prefix(self.run_case(work_attack=shifted))
        def forged_work(solver, args, kwargs, energy):
            payload = energy['continuousCableEnergy']; work = payload['work']
            value = float(np.nextafter(work['changeJoules'], np.inf))
            self.assertNotEqual(value, work['changeJoules'])
            work['changeJoules'] = energy['cableFixedParameterChangeJoules'] = value
            uncertainty = _rational(work['certificate']['changeErrorBoundJoules'])
            for field, terms in payload['aggregationTermsJoules'].items():
                terms['cableFixedParameterChangeJoules'] = value
                energy[field], bound = round_sum(terms.values(), uncertainty)
                payload['errorBoundsJoules'][field] = _rat(bound)
                payload['assemblyRoundingBoundsJoules'][field] = _rat(bound-uncertainty)
            # The forged one-ulp work still intersects endpoint uncertainty;
            # changed sums and their rounding records are fully repaired.
            solver_energy_balance.validate_continuous_cable_energy(solver.continuous_cable, args[1], args[2], energy)
            validated_metadata.append('work')
        self.assert_prefix(self.run_case(work_attack=forged_work))
        self.assertEqual(validated_metadata, ['shift', 'work'])

    def test_agreeing_forged_helper_and_step_certificate_still_rejects(self):
        original = CableControl.diagnostics
        for field, value in (('verified', 1), ('accepted', True), ('inputSha256', '0'*64)):
            def forged(control, positions):
                result = original(control, positions)
                result['certificate'][field] = value
                return result
            with self.subTest(field=field), mock.patch.object(CableControl, 'diagnostics', forged):
                result = self.run_case()
            self.assertFalse(result[2]['complete'])
            self.assertEqual(result[2]['completedFraction'], 0.)
            self.assertEqual(result[4], []); self.assertEqual(result[5], [])
            self.assertEqual(result[3].rows[0][0]['error']['type'], 'ValueError')
            self.assertIsNone(result[3].rows[0][1])

    def test_fresh_endpoint_and_work_evaluation_mutations_preserve_prefix(self):
        original_diagnostics = CableControl.diagnostics
        original_change = CableControl.energy_change
        for channel in ('before', 'work'):
            diagnostics_calls, work_calls = [], []
            def diagnostics(control, positions):
                result = original_diagnostics(control, positions)
                diagnostics_calls.append(1)
                if channel == 'before' and len(diagnostics_calls) == 8:
                    positions[:] = 777.
                return result
            def change(control, start, end):
                result = original_change(control, start, end)
                work_calls.append(1)
                if channel == 'work' and len(work_calls) == 4:
                    start[:], end[:] = 777., 888.
                return result
            with self.subTest(channel=channel), mock.patch.object(CableControl, 'diagnostics', diagnostics), \
                    mock.patch.object(CableControl, 'energy_change', change):
                self.assert_prefix(self.run_case())

    def test_last_fresh_work_cannot_change_journal_core_or_authored_energy(self):
        original_change = CableControl.energy_change
        for channel in ('core', 'energy', 'replace-energy'):
            journals, calls = [], []
            class HoldingJournal(Journal):
                def __init__(self, events):
                    super().__init__(events); journals.append(self)
                def start(self, row):
                    self.live = row
                    super().start(row)
            def change(control, start, end):
                result = original_change(control, start, end)
                calls.append(1)
                if len(calls) == 4:
                    step = journals[0].live['step']
                    if channel == 'core': step['gradientInfinityNorm'] = .5
                    elif channel == 'energy': step['energyBalance']['cableAfterJoules'] = 900.
                    else: step['energyBalance'] = dict(step['energyBalance'], cableAfterJoules=900.)
                return result
            with self.subTest(channel=channel), mock.patch.object(CableControl, 'energy_change', change):
                result = self.run_case(journal_type=HoldingJournal)
            self.assert_prefix(result)
            self.assertEqual(len(result[5]), 2)
            self.assertIn('changed during', result[2]['rejectedSteps'][0]['error']['message'])

    def test_resource_failure_is_fatal_and_preserves_accepted_prefix(self):
        for error in (TimeoutError, RuntimeError, MemoryError):
            def fail(s, args, kwargs, energy):
                args[1][:] = 999.
                raise error('synthetic cable work resource failure')
            with self.subTest(error=error.__name__):
                result = self.run_case(work_attack=fail)
                self.assert_prefix(result)
                self.assertEqual(result[2]['reason'], 'solver-resource-or-runtime-failure')
                self.assertTrue(result[2]['rejectedSteps'][0]['fatal'])

    def test_retries_do_not_advance_state_or_accumulate_rejected_work(self):
        solver, q, v = self.setup(); solver.reject_above = .05
        result = self.run_case(solver=solver, max_depth=1)
        final, velocity, report, _, accepted, work, _, solver = result
        self.assertTrue(report['complete']); self.assertEqual(len(accepted), 8)
        self.assertEqual(len(work), 8); self.assertEqual(len(report['rejectedSteps']), 4)
        states = {0.: (q, v), **{row[2]['endFraction']: row[:2] for row in accepted}}
        for row, call in zip(report['attempts'], solver.calls):
            np.testing.assert_array_equal(call[0], states[row['startFraction']][0])
            np.testing.assert_array_equal(call[1], states[row['startFraction']][1])
            self.assertEqual(call[2].shape, (0, 3))
            if row['outcome'] == 'rejected': self.assertNotIn('energyBalance', row['step'])
        np.testing.assert_array_equal(final, accepted[-1][0]); np.testing.assert_array_equal(velocity, accepted[-1][1])

    def test_unresolved_precision_rejects_without_work_or_publication(self):
        solver, q, v = self.setup(); solver.continuous_cable = control(strict=True)
        # The admitted positive tolerance is much smaller than final output
        # roundoff of this nonzero quadratic energy. No budget retry may waive it.
        events = []; journal = Journal(events)
        with mock.patch.object(solver_energy_balance, 'global_energy_transition') as work:
            final, velocity, report = adaptive_contact_step(solver, q, v, np.empty((0,3)), np.empty((0,3)), .1,
                max_depth=1, attempt_journal=journal)
        self.assertFalse(report['complete']); self.assertEqual(report['completedFraction'], 0.)
        self.assertEqual(len(report['acceptedSteps']), 0); self.assertEqual(work.call_count, 0)
        self.assertTrue(all(row[1] is None for row in journal.rows))
        np.testing.assert_array_equal(final, q); np.testing.assert_array_equal(velocity, v)

    def test_interrupted_step_or_work_keeps_only_accepted_prefix(self):
        for location in ('step', 'work'):
            solver, q, v = self.setup(); events = []; journal = Journal(events); accepted = []
            original_q, original_v = q.copy(), v.copy()
            real_work = solver_energy_balance.global_energy_transition; calls = []
            if location == 'step': solver.interrupt_at = 2
            def work(*args, **kwargs):
                calls.append(1)
                if location == 'work' and len(calls) == 2:
                    args[1][:] = 777.
                    raise KeyboardInterrupt('synthetic work interruption')
                return real_work(*args, **kwargs)
            with mock.patch.object(solver_energy_balance, 'global_energy_transition', side_effect=work):
                with self.assertRaises(KeyboardInterrupt):
                    adaptive_contact_step(solver, q, v, np.empty((0,3)), np.empty((0,3)), .4,
                        initial_subdivisions=4, max_depth=0, attempt_journal=journal,
                        on_accept=lambda q,v,r:accepted.append((q.copy(),v.copy(),copy.deepcopy(r))))
            self.assertEqual(len(accepted), 1)
            self.assertEqual([row[0]['outcome'] for row in journal.rows], ['accepted', 'interrupted'])
            self.assertIsNone(journal.rows[-1][1]); self.assertIsNone(journal.rows[-1][2])
            np.testing.assert_array_equal(journal.rows[0][1], accepted[0][0])
            np.testing.assert_array_equal(q, original_q); np.testing.assert_array_equal(v, original_v)

    def test_raw_state_candidate_and_fixed_precision_override_admission(self):
        for channel in ('q', 'v'):
            for bad in (True, 2**53+1, float('nan')):
                solver, q, v = self.setup(); q, v = q.tolist(), v.tolist()
                (q if channel == 'q' else v)[0][0] = bad
                with self.subTest(channel=channel, bad=bad), self.assertRaises(ValueError):
                    adaptive_contact_step(solver, q, v, np.empty((0,3)), np.empty((0,3)), .1)
                self.assertEqual(len(solver.calls), 0)
        for channel in ('q', 'v'):
            def candidate(s, q, v, t, c, w, r):
                c, w = c.tolist(), w.tolist(); (c if channel == 'q' else w)[0][0] = True
                return c, w
            self.assert_prefix(self.run_case(step_attack=candidate))
        for kwargs in ({'linear_solver': 'cg'}, {'cable_precision': {}}, {'continuous_cable': control()}):
            solver, q, v = self.setup()
            with self.subTest(kwargs=tuple(kwargs)), self.assertRaises(ValueError):
                adaptive_contact_step(solver, q, v, np.empty((0,3)), np.empty((0,3)), .1, **kwargs)
            self.assertEqual(len(solver.calls), 0)

    def test_raw_duration_rejects_boolean_and_inexact_values_before_subdivision(self):
        values = [True, False, np.bool_(True), 2**53+1, np.int64(2**53+1),
                  '0.1', F(1, 10), float('inf'), float('nan')]
        if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
            values.append(np.nextafter(np.longdouble(1.), np.longdouble(2.)))
        for value in values:
            solver, q, v = self.setup(); events = []; journal = Journal(events)
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                adaptive_contact_step(solver, q, v, np.empty((0,3)), np.empty((0,3)), value,
                    initial_subdivisions=4, attempt_journal=journal)
            self.assertEqual(solver.calls, [])
            self.assertEqual(events, [])

    def test_actual_generic_cable_contact_and_existing_seam_publish_fixed_work(self):
        from test_solver_cable_global import fixture, PRECISION
        from solver_global_sewing import GlobalSewingSolver
        from solver_triangle_sweep import triangle_sweep_safe
        model, original, q, recipe = fixture()
        solver = GlobalSewingSolver(model, [{2: 1., 6: -1.}], 1e-6, sewing_mode='distance',
            contact=original.contact, continuous_cable=recipe, cable_precision=copy.deepcopy(PRECISION))
        target = np.array([np.linalg.norm(q[2]-q[6])])
        frozen = (model.particle_q.numpy().copy(), solver.poses.copy(), solver.faces.copy(), solver.sewing.copy())
        states = []
        final, velocity, report = adaptive_contact_step(solver, q, np.zeros_like(q), target, target, .002,
            initial_subdivisions=2, max_depth=2, max_attempts=16,
            on_accept=lambda q,v,r:states.append((q.copy(),v.copy(),copy.deepcopy(r))))
        self.assertTrue(report['complete'], report)
        self.assertGreaterEqual(len(states), 2)
        previous = q
        for current, v, row in states:
            diagnostic = row['step']; energy = diagnostic['energyBalance']
            self.assertGreater(diagnostic['continuousCable']['energyJoules'], 0.)
            self.assertGreater(diagnostic['contactJoules'], 0.)
            self.assertGreater(diagnostic['sewingJoules'], 0.)
            self.assertLessEqual(_rational(diagnostic['cableStationarity']['stationarityUpperBoundNewtons']), F(1e-6))
            self.assertEqual(energy['cableParameterWorkJoules'], 0.)
            self.assertIn('continuousCableEnergy', energy)
            self.assertTrue(triangle_sweep_safe(previous, current, solver.faces))
            self.assertTrue(solver.contact.path_safe(previous, current))
            np.testing.assert_array_equal(v, (current-previous)/row['durationSeconds'])
            previous = current
        np.testing.assert_array_equal(final, states[-1][0]); np.testing.assert_array_equal(velocity, states[-1][1])
        np.testing.assert_array_equal(model.particle_q.numpy(), frozen[0])
        np.testing.assert_array_equal(solver.poses, frozen[1]); np.testing.assert_array_equal(solver.faces, frozen[2])
        self.assertEqual((solver.sewing != frozen[3]).nnz, 0)
        self.assertEqual(solver.sewing.shape, (1, 8))

    def test_no_cable_keeps_legacy_no_schedule_publication_path(self):
        solver, q, v = self.setup(); solver.continuous_cable = None
        with mock.patch.object(solver_energy_balance, 'global_energy_transition') as work:
            _, _, report = adaptive_contact_step(solver, q, v, np.empty((0,3)), np.empty((0,3)), .1)
        self.assertTrue(report['complete']); self.assertEqual(work.call_count, 0)
        self.assertNotIn('continuousCable', report)
        self.assertNotIn('continuousCableScope', report)
        self.assertNotIn('energyBalance', report['acceptedSteps'][0]['step'])


if __name__ == '__main__':
    unittest.main()
