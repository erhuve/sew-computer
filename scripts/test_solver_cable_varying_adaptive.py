"""Varying-cable publication attacks; mock equilibrium is explicitly scoped.

ReportFixture has real cell/work integration but supplies its stationarity
claim. Separate real solver cases exercise generic mechanics and guards; no
garment source, captured runner, calibration or construction is admitted.
"""
import copy
from fractions import Fraction as F
import unittest
from unittest import mock

import numpy as np

from solver_adaptive_contact import adaptive_contact_step
import solver_adaptive_contact as adaptive_module
from solver_cable_integration import CableControl, _rat, _rational, stationarity
from solver_cable_parameters import CableParameterRecipe
from solver_cable_parameter_schedule import CableParameterSchedule
from solver_cable_varying import VaryingCableControl
from solver_continuous_cable_sewing import ContinuousCableSewing
import solver_energy_balance
from test_solver_cable_adaptive import ReportFixture as FixedReportFixture, Journal


def control(*, activation=0., strict=False, density=8., targets=None):
    anchor = lambda vertex: [{'vertex': vertex, 'weight': _rat(F(1))}]
    recipe = CableParameterRecipe(ContinuousCableSewing(2, [{
        'id': 'generic-varying-cell', 'positiveStart': anchor(0), 'positiveEnd': anchor(0),
        'negativeStart': anchor(1), 'negativeEnd': anchor(1),
        'targetsMeters': [.001, .001] if targets is None else targets,
        'referenceLengthMeters': _rat(F(1, 8)), 'stiffnessDensityNPerM2': density,
        'activation': activation,
    }]))
    budgets = dict(max_boundary_depth=80, max_boundary_panels=256,
                   moment_max_terms=128, moment_max_panels=256, moment_max_depth=64)
    precision = dict(energy_tolerance_joules=1e-80 if strict else 1e-16,
                     gradient_tolerance_newtons=1e-80 if strict else 1e-12,
                     hessian_tolerance_newtons_per_meter=1e-80 if strict else 1e-10,
                     absolute_tolerance_joules=1e-80 if strict else 1e-16, **budgets)
    work = dict(absolute_tolerance_joules=1e-80 if strict else 1e-16,
                component_tolerances_joules=None, **budgets)
    return VaryingCableControl(recipe, precision, work)


def schedule(control, *, constant=False):
    d, a = control.recipe.initial_parameters
    if constant:
        knots = [(0., d.tolist(), a.tolist()), (1., d.tolist(), a.tolist())]
    else:
        knots = [(0., d.tolist(), a.tolist()), (.25, [[.001, .001]], [1.]),
                 (.5, [[.0015, .0015]], [1.]), (.75, [[.001, .001]], [0.]),
                 (1., [[.005, .005]], [0.])]
    return {'profile': 'cable-target-activation-v1', 'geometrySha256': control.recipe.geometry_sha256,
            'cellIds': list(control.recipe.cell_ids), 'knots': [
                {'fraction': f, 'targetsMeters': t, 'activation': w} for f, t, w in knots]}


class ReportFixture(FixedReportFixture):
    def __init__(self, cable=None):
        super().__init__()
        self.continuous_cable = None
        self.cable_parameter_control = control() if cable is None else cable
        self.reject_first = False

    def step(self, q, v, target, duration, **options):
        self.calls.append((q.copy(), v.copy(), duration, copy.deepcopy(options)))
        if self.interrupt_at == len(self.calls):
            q[:], v[:] = 999., 888.
            raise self.interruption('varying cable synthetic interruption')
        if self.reject_first and len(self.calls) == 1:
            q[:], v[:], options['cable_targets'][:], options['cable_activation'][:] = 999., 888., 77., 0.
            return q, v, {'converged': False, 'gradientInfinityNorm': 1.}
        candidate, velocity = q+duration*v, v.copy()
        cable = self.cable_parameter_control
        effective = cable.effective(options['cable_targets'], options['cable_activation'])
        diagnostic = effective.diagnostics(candidate)
        error = _rational(diagnostic['certificate']['gradientMaxAbsoluteErrorBoundNewtons'])
        report = {'profile': 'experimental-global-varying-cable-reference-v1',
                  'converged': True, 'gradientInfinityNorm': 0., 'continuousCable': diagnostic,
                  'cableStationarity': stationarity(np.array([0.]), error, error, F()),
                  'varyingCable': cable.parameter_record(options['cable_targets'], options['cable_activation'])}
        self.last_report, self.last_options = report, options
        if self.after_step is not None and len(self.calls) == 2:
            changed = self.after_step(self, q, v, target, candidate, velocity, report, options)
            if changed is not None:
                candidate, velocity = changed
        return candidate, velocity, report


class CableVaryingAdaptiveTests(unittest.TestCase):
    def run_case(self, *, solver=None, raw=None, step_attack=None, work_attack=None,
                 max_depth=0, subdivisions=4, journal_type=Journal):
        solver = ReportFixture() if solver is None else solver
        raw = schedule(solver.cable_parameter_control) if raw is None else raw
        solver.after_step = step_attack
        q = np.array([[0., 0., 0.], [.002, 0., 0.]])
        v = np.zeros_like(q)
        original_q, original_v, original_raw = q.copy(), v.copy(), copy.deepcopy(raw)
        accepted, work_calls, events = [], [], []
        journal = journal_type(events)
        real = solver_energy_balance.global_energy_transition
        def work(*args, **kwargs):
            result = real(*args, **kwargs)
            work_calls.append((tuple(value.copy() for value in args[1:7]), copy.deepcopy(kwargs)))
            events.append(('work', len(work_calls)))
            if work_attack is not None and len(work_calls) == 2:
                work_attack(solver, args, kwargs, result)
            return result
        def accept(q1, v1, row):
            self.assertEqual(events[-1], ('outcome', 'accepted'))
            accepted.append((q1.copy(), v1.copy(), copy.deepcopy(row)))
            q1[:], v1[:] = 999., 888.
            row['step']['varyingCable']['activation'][0] = 77.
            events.append(('callback', row['attemptId']))
        with mock.patch.object(solver_energy_balance, 'global_energy_transition', side_effect=work):
            final, velocity, report = adaptive_contact_step(solver, q, v, np.empty((0, 3)), np.empty((0, 3)), .4,
                max_depth=max_depth, max_attempts=32, initial_subdivisions=subdivisions,
                cable_parameter_schedule=raw, attempt_journal=journal, on_accept=accept)
        np.testing.assert_array_equal(q, original_q); np.testing.assert_array_equal(v, original_v)
        self.assertEqual(raw, original_raw)
        return final, velocity, report, journal, accepted, work_calls, events, solver

    def assert_prefix(self, result):
        final, velocity, report, journal, accepted, _, _, _ = result
        self.assertFalse(report['complete']); self.assertEqual(report['completedFraction'], .25)
        self.assertEqual(len(report['acceptedSteps']), 1); self.assertEqual(len(report['rejectedSteps']), 1)
        self.assertEqual(len(accepted), 1)
        self.assertEqual([item[0]['outcome'] for item in journal.rows], ['accepted', 'rejected'])
        self.assertIsNone(journal.rows[-1][1]); self.assertIsNone(journal.rows[-1][2])
        np.testing.assert_array_equal(final, accepted[0][0]); np.testing.assert_array_equal(velocity, accepted[0][1])
        self.assertEqual(report['finalCableParameters'], accepted[0][2]['step']['varyingCable'])
        totals = report['varyingCableAcceptedWorkTotals']
        self.assertEqual(totals['acceptedStepCount'], 1)
        for field, value in totals['valuesJoules'].items():
            self.assertEqual(value, accepted[0][2]['step']['energyBalance'][field])
        self.assertIs(report['accepted'], False)

    def test_target_first_work_precedes_journal_and_telescope(self):
        result = self.run_case()
        _, _, report, _, accepted, work, events, solver = result
        self.assertTrue(report['complete']); self.assertEqual(len(accepted), 4); self.assertEqual(len(work), 4)
        self.assertEqual(report['varyingCableControl'], solver.cable_parameter_control.description())
        controls = CableParameterSchedule(schedule(solver.cable_parameter_control), 4,
                                           cable_recipe=solver.cable_parameter_control.recipe)
        totals = []
        for row in report['acceptedSteps']:
            p0, p1 = controls.parameters(row['startFraction']), controls.parameters(row['endFraction'])
            r = F(.002)
            def energy(d, a):
                return F(float(a[0]))*max(r-F(float(d[0, 0])), F())**2/2
            target = energy(p1[0], p0[1])-energy(*p0)
            activation = energy(*p1)-energy(p1[0], p0[1])
            balance = row['step']['energyBalance']; payload = balance['varyingCableEnergy']['parameterWork']
            for field, exact in (('targetWorkJoules', target), ('activationWorkJoules', activation),
                                 ('totalWorkJoules', target+activation)):
                self.assertLessEqual(abs(F(payload[field])-exact), _rational(payload['certificate']['errorsJoules'][field]))
            totals.append((F(payload['totalWorkJoules']), _rational(payload['certificate']['errorsJoules']['totalWorkJoules'])))
            self.assertEqual(balance['cableFixedParameterChangeJoules'], 0.)
            self.assertEqual(row['step']['varyingCable'], solver.cable_parameter_control.parameter_record(*p1))
        self.assertLessEqual(abs(sum((value for value, _ in totals), F())), sum((bound for _, bound in totals), F()))
        for i, event in enumerate(events):
            if event == ('outcome', 'accepted'): self.assertEqual(events[i-1][0], 'work')
        # Release follows the changed target; it is not the old endpoint energy.
        release = accepted[2][2]['step']['energyBalance']
        self.assertLess(release['cableActivationParameterWorkJoules'], 0.)
        self.assertGreater(release['cableTargetParameterWorkJoules'], 0.)
        totals = report['varyingCableAcceptedWorkTotals']
        self.assertEqual(totals['acceptedStepCount'], 4)
        for field, value in totals['valuesJoules'].items():
            exact = sum((F(row['step']['energyBalance'][field]) for row in report['acceptedSteps']), F())
            radius = sum((_rational(row['step']['energyBalance']['varyingCableEnergy']['errorBoundsJoules'][field])
                          for row in report['acceptedSteps']), F())
            self.assertEqual(value, float(exact))
            self.assertEqual(_rational(totals['errorBoundsJoules'][field]), radius+abs(F(value)-exact))
            self.assertEqual(_rational(totals['assemblyRoundingBoundsJoules'][field]), abs(F(value)-exact))

    def test_rejected_mutated_controls_retry_from_original_fractions(self):
        solver = ReportFixture(); solver.reject_first = True
        result = self.run_case(solver=solver, max_depth=1)
        self.assertTrue(result[2]['complete']); self.assertEqual(len(result[2]['rejectedSteps']), 1)
        self.assertEqual(len(result[2]['acceptedSteps']), 5); self.assertEqual(len(result[5]), 5)
        controls = CableParameterSchedule(schedule(solver.cable_parameter_control), 4,
                                           cable_recipe=solver.cable_parameter_control.recipe)
        for row, call in zip(result[2]['attempts'], solver.calls):
            expected = controls.parameters(row['endFraction'])
            np.testing.assert_array_equal(call[3]['cable_targets'], expected[0])
            np.testing.assert_array_equal(call[3]['cable_activation'], expected[1])
            self.assertEqual(row['cableParameterInterval']['before'],
                             solver.cable_parameter_control.parameter_record(*controls.parameters(row['startFraction'])))
            self.assertEqual(row['cableParameterInterval']['after'], solver.cable_parameter_control.parameter_record(*expected))
        np.testing.assert_array_equal(solver.calls[0][0], solver.calls[1][0])
        self.assertEqual(result[2]['acceptedSteps'][0]['endFraction'], .125)

    def test_pairing_initial_identity_and_preflight_reject_before_journal(self):
        solver = ReportFixture(); raw = schedule(solver.cable_parameter_control)
        bad = []
        for mutate in (lambda x:x.__setitem__('geometrySha256', '0'*64),
                       lambda x:x['cellIds'].__setitem__(0, 'stale'),
                       lambda x:x['knots'][0]['activation'].__setitem__(0, .5),
                       lambda x:x['knots'][0]['activation'].__setitem__(0, -0.0),
                       lambda x:x['knots'][0]['targetsMeters'][0].__setitem__(0, np.nextafter(.001, 1.))):
            changed = copy.deepcopy(raw); mutate(changed); bad.append(changed)
        for changed in [None, *bad]:
            journal = Journal([])
            with self.assertRaises(ValueError):
                adaptive_contact_step(solver, np.zeros((2, 3)), np.zeros((2, 3)), np.empty((0, 3)), np.empty((0, 3)), .4,
                    initial_subdivisions=4, cable_parameter_schedule=changed, attempt_journal=journal)
            self.assertEqual(journal.events, [])
        missing = ReportFixture(); missing.cable_parameter_control = None
        with self.assertRaises(ValueError):
            self.run_case(solver=missing, raw=raw)
        from test_solver_cable_adaptive import control as fixed_control
        both = ReportFixture(); both.continuous_cable = fixed_control()
        with self.assertRaises(ValueError): self.run_case(solver=both)
        tiny = control(density=float(np.nextafter(0., 1.)*8))
        solver = ReportFixture(tiny)
        raw = schedule(tiny, constant=True)
        raw['knots'][-1]['activation'] = [1.]
        # Endpoint beta is minsubnormal; the first retry-grid midpoint rounds
        # beta to zero. Failure occurs in preflight, before any attempted solve.
        with self.assertRaises(ValueError): self.run_case(solver=solver, raw=raw, max_depth=1)
        self.assertEqual(solver.calls, [])

    def test_raw_admission_and_per_step_override_rejections(self):
        for raw_dt in (True, 2**53+1, np.longdouble('0.10000000000000000001')):
            if isinstance(raw_dt, np.longdouble) and np.finfo(np.longdouble).nmant == 52: continue
            solver = ReportFixture()
            with self.subTest(dt=repr(raw_dt)), self.assertRaises(ValueError):
                adaptive_contact_step(solver, np.zeros((2, 3)), np.zeros((2, 3)), np.empty((0, 3)), np.empty((0, 3)), raw_dt,
                    initial_subdivisions=4, cable_parameter_schedule=schedule(solver.cable_parameter_control))
        for bad in ([[False, 0., 0.], [1., 0., 0.]], [[2**53+1, 0., 0.], [1., 0., 0.]]):
            for velocity in (False, True):
                solver = ReportFixture()
                with self.assertRaises(ValueError):
                    adaptive_contact_step(solver, np.zeros((2, 3)) if velocity else bad,
                        bad if velocity else np.zeros((2, 3)), np.empty((0, 3)), np.empty((0, 3)), .4,
                        initial_subdivisions=4, cable_parameter_schedule=schedule(solver.cable_parameter_control))
        for options in ({'cable_targets': [[.001, .001]]}, {'cable_activation': [0.]},
                        {'cable_precision': {}}, {'linear_solver': 'cg'}):
            solver = ReportFixture()
            with self.assertRaises(ValueError):
                adaptive_contact_step(solver, np.zeros((2, 3)), np.zeros((2, 3)), np.empty((0, 3)), np.empty((0, 3)), .4,
                    initial_subdivisions=4, cable_parameter_schedule=schedule(solver.cable_parameter_control), **options)

    def test_step_parameter_response_and_stationarity_attacks_preserve_prefix(self):
        def unresolved(report):
            err = _rational(report['continuousCable']['certificate']['gradientMaxAbsoluteErrorBoundNewtons'])
            report['gradientInfinityNorm'] = 9e-7
            report['cableStationarity'] = stationarity(np.array([9e-7]), err+F(2e-7), err, F(2e-7))
        attacks = [lambda r:r.pop('varyingCable'), lambda r:r.__setitem__('profile', 'fixed'),
            lambda r:r['varyingCable'].__setitem__('parameterSha256', '0'*64),
            lambda r:r['varyingCable']['activation'].__setitem__(0, False),
            lambda r:r['varyingCable'].__setitem__('effectivePotentialSha256', '0'*64),
            lambda r:r['continuousCable'].__setitem__('energyJoules', 1.),
            lambda r:r['continuousCable']['certificate'].__setitem__('inputSha256', '0'*64),
            lambda r:r['cableStationarity'].pop('scope'), unresolved]
        for i, attack in enumerate(attacks):
            with self.subTest(attack=i):
                result = self.run_case(step_attack=lambda s,q,v,t,c,w,r,o:attack(r))
                self.assert_prefix(result); self.assertEqual(len(result[5]), 1)
        for field in ('cable_targets', 'cable_activation'):
            self.assert_prefix(self.run_case(step_attack=lambda s,q,v,t,c,w,r,o:o[field].__setitem__(slice(None), .75)))

    def test_work_input_control_and_model_mutations_reject(self):
        for index in (1, 2, 3, 4):
            self.assert_prefix(self.run_case(work_attack=lambda s,a,k,e:a[index].__setitem__(slice(None), 999.)))
        for key in ('previous_cable_targets', 'previous_cable_activation', 'cable_targets', 'cable_activation'):
            self.assert_prefix(self.run_case(work_attack=lambda s,a,k,e:k[key].__setitem__(slice(None), .5)))
        for change in (lambda s:setattr(s, 'cable_parameter_control', control()),
                       lambda s:setattr(s, 'compliance', .5),
                       lambda s:s.last_report.__setitem__('unexpected', 1),
                       lambda s:s.last_report.__setitem__('converged', False)):
            self.assert_prefix(self.run_case(work_attack=lambda s,a,k,e:change(s)))

    def test_missing_raw_work_fields_and_identity_forgeries_reject(self):
        fields = ('cableTargetParameterWorkJoules', 'cableActivationParameterWorkJoules', 'cableParameterWorkJoules',
                  'cableFixedParameterChangeJoules', 'cableChangeJoules', 'targetParameterWorkJoules', 'externalParameterWorkJoules')
        for field in fields:
            for kind in ('missing', 'bool', 'nonfinite'):
                def attack(s, a, k, e):
                    if kind == 'missing': e.pop(field)
                    else: e[field] = False if kind == 'bool' else float('inf')
                with self.subTest(field=field, kind=kind): self.assert_prefix(self.run_case(work_attack=attack))
        for change in (lambda e:e.pop('varyingCableEnergy'),
            lambda e:e.__setitem__('accepted', 0),
            lambda e:e['varyingCableEnergy'].pop('motionWork'),
            lambda e:e['varyingCableEnergy']['parameterWork']['certificate'].__setitem__('afterParameterSha256', '0'*64),
            lambda e:e['varyingCableEnergy']['beforeParameters']['activation'].__setitem__(0, False),
            lambda e:e['varyingCableEnergy']['motionWork']['certificate'].__setitem__('endPositionsSha256', '0'*64)):
            self.assert_prefix(self.run_case(work_attack=lambda s,a,k,e:change(e)))

    def test_every_unconditional_baseline_energy_field_is_required_before_publication(self):
        # Independent public-report list: missing zero-valued old physics is
        # still a malformed full report. Nested correct cable sums do not
        # authorize silently dropping any unconditional baseline scalar.
        names = (
            'membraneChange', 'bendingChange', 'bendingBefore', 'bendingAfter',
            'foldBarrierChange', 'foldBarrierBefore', 'foldBarrierAfter',
            'contactChange', 'contactBefore', 'contactAfter', 'kineticChange',
            'sewingChange', 'sewingBefore', 'sewingAfter',
            'foldActuationChange', 'foldActuationBefore', 'foldActuationAfter', 'foldTargetParameterWork',
            'gripperBefore', 'gripperAfter', 'gripperFixedPositionAfter', 'gripperFixedParameterChange',
            'gripperChange', 'gripperParameterWork', 'gripperTargetParameterWork',
            'gripperActivationParameterWork', 'gripperActivationIncreaseWork',
            'gripperReleaseEnergyRemoved', 'gripperParameterWorkComponentSumErrorBound',
            'targetParameterWork', 'externalParameterWork', 'mechanicalChange',
            'mechanicalChangeMinusTargetWork', 'mechanicalChangeMinusParameterWork',
        )
        self.assertEqual(len(names), 34)
        for name in names:
            field = name+'Joules'
            with self.subTest(field=field):
                result = self.run_case(work_attack=lambda s,a,k,e:e.pop(field))
                self.assert_prefix(result)
                self.assertEqual(len(result[5]), 2)
                self.assertEqual(result[2]['rejectedSteps'][0]['error']['type'], 'ValueError')

    def test_coherent_endpoint_shift_passes_structural_checks_but_not_fresh_numerics(self):
        structural_passes = []
        def attack(s, args, kwargs, e):
            for position in ('before', 'after'):
                nested = e['varyingCableEnergy'][position]
                nested['energyJoules'] += 1e-6
                e['cable'+position.title()+'Joules'] = nested['energyJoules']
            solver_energy_balance.validate_varying_cable_energy(s.cable_parameter_control, args[1], args[2],
                kwargs['previous_cable_targets'], kwargs['previous_cable_activation'],
                kwargs['cable_targets'], kwargs['cable_activation'], e)
            structural_passes.append(True)
        result = self.run_case(work_attack=attack)
        self.assert_prefix(result); self.assertEqual(structural_passes, [True])
        self.assertIn('fresh', result[2]['rejectedSteps'][0]['error']['message'])

    def test_fresh_helper_mutation_and_late_core_attack_reject(self):
        real = VaryingCableControl.parameter_work
        normalized = []
        normalizer = adaptive_module.diagnostic_json
        def retain_normalized(*args, **kwargs):
            value = normalizer(*args, **kwargs)
            if isinstance(value[0], dict) and 'varyingCable' in value[0]: normalized.append(value[0])
            return value
        calls = []
        def corrupt(control, q, *parameters):
            value = real(control, q, *parameters)
            calls.append(1)
            if len(calls) == 4:  # helper's step2 work then adaptive fresh work
                normalized[-1]['gradientInfinityNorm'] = 99.
            return value
        with mock.patch.object(VaryingCableControl, 'parameter_work', corrupt), \
                mock.patch.object(adaptive_module, 'diagnostic_json', retain_normalized):
            self.assert_prefix(self.run_case())
        calls.clear()
        def mutate(control, q, *parameters):
            value = real(control, q, *parameters); calls.append(1)
            if len(calls) == 4: parameters[0][:] = .02
            return value
        with mock.patch.object(VaryingCableControl, 'parameter_work', mutate):
            self.assert_prefix(self.run_case())

    def test_one_ulp_work_forgery_with_repaired_bounds_and_sums_is_rejected(self):
        structural_passes = []
        def attack(s, args, kwargs, e):
            p = e['varyingCableEnergy']
            forged = float(np.nextafter(0., 1.))
            p['motionWork']['changeJoules'] = e['cableFixedParameterChangeJoules'] = forged
            radius = F(1e-16)
            p['motionWork']['certificate']['changeErrorBoundJoules'] = _rat(radius)
            p['errorBoundsJoules']['cableFixedParameterChangeJoules'] = _rat(radius)
            # Independently repair every affected public copy and exact
            # addition witness. The error remains within the fixed policy.
            errors = p['parameterWork']['certificate']['errorsJoules']
            bases = {'cableChangeJoules': _rational(errors['totalWorkJoules'])+radius,
                     'mechanicalChangeJoules': _rational(errors['totalWorkJoules'])+radius,
                     'mechanicalChangeMinusTargetWorkJoules': _rational(errors['activationWorkJoules'])+radius,
                     'mechanicalChangeMinusParameterWorkJoules': radius}
            for name, base in bases.items():
                terms = p['aggregationTermsJoules'][name]
                terms['cableFixedParameterChangeJoules'] = forged
                exact = sum(map(F, terms.values()), F()); e[name] = float(exact)
                rounding = abs(F(e[name])-exact)
                p['assemblyRoundingBoundsJoules'][name] = _rat(rounding)
                p['errorBoundsJoules'][name] = _rat(base+rounding)
            solver_energy_balance.validate_varying_cable_energy(s.cable_parameter_control, args[1], args[2],
                kwargs['previous_cable_targets'], kwargs['previous_cable_activation'],
                kwargs['cable_targets'], kwargs['cable_activation'], e)
            structural_passes.append(True)
        result = self.run_case(work_attack=attack)
        self.assert_prefix(result); self.assertEqual(structural_passes, [True])
        self.assertIn('motionWork', result[2]['rejectedSteps'][0]['error']['message'])

    def test_last_motion_helper_cannot_mutate_prior_helper_inputs_or_report(self):
        real = CableControl.energy_change
        normalizer = adaptive_module.diagnostic_json
        for channel in ('live-core', 'old-work-input', 'input', 'report-accounting'):
            calls, active, old_inputs = [], [], []
            def retain_normalized(*args, **kwargs):
                value = normalizer(*args, **kwargs)
                if isinstance(value[0], dict) and 'varyingCable' in value[0]: active.append(value[0])
                return value
            def retain(s, args, kwargs, energy): old_inputs.append(args[1])
            def change(effective, q0, q1):
                result = real(effective, q0, q1); calls.append(1)
                if len(calls) == 4:
                    if channel == 'live-core': active[-1]['gradientInfinityNorm'] = .5
                    elif channel == 'old-work-input': old_inputs[0][:] = 99.
                    elif channel == 'input': q0[:] = 99.
                    else: active[-1]['energyBalance']['cableAfterJoules'] = 99.
                return result
            with self.subTest(channel=channel), mock.patch.object(CableControl, 'energy_change', change), \
                    mock.patch.object(adaptive_module, 'diagnostic_json', retain_normalized):
                self.assert_prefix(self.run_case(work_attack=retain))

    def test_journal_metadata_and_state_mutation_cannot_change_accepted_steps(self):
        class MutatingJournal(Journal):
            def start(self, row):
                super().start(row)
                for field in ('attemptId', 'startFraction', 'endFraction', 'durationSeconds', 'depth'):
                    row[field] = 777.
                self.last_start = row
            def outcome(self, row, q=None, v=None):
                super().outcome(row, q, v)
                row['attemptId'] = 999.
                row['step']['varyingCable']['activation'][0] = 999.
                if q is not None: q[:], v[:] = 999., 888.
                self.last_start['endFraction'] = -1.
        result = self.run_case(journal_type=MutatingJournal)
        self.assertTrue(result[2]['complete']); self.assertEqual(len(result[4]), 4)
        np.testing.assert_array_equal(result[0], [[0.,0.,0.],[.002,0.,0.]])
        np.testing.assert_array_equal(result[1], np.zeros((2,3)))
        for i, row in enumerate(result[2]['acceptedSteps']):
            self.assertEqual((row['attemptId'],row['startFraction'],row['endFraction']), (i+1,i/4,(i+1)/4))
            self.assertEqual(row['durationSeconds'], .1)
            self.assertEqual(row, result[3].rows[i][0])

    def test_interruption_and_precision_failure_publish_no_false_acceptance(self):
        solver = ReportFixture(); solver.interrupt_at = 2; solver.interruption = TimeoutError
        result = self.run_case(solver=solver)
        self.assert_prefix(result); self.assertEqual(result[2]['reason'], 'solver-resource-or-runtime-failure')
        self.assertTrue(result[2]['rejectedSteps'][0]['fatal'])
        strict = ReportFixture(control(strict=True))
        result = self.run_case(solver=strict)
        self.assertFalse(result[2]['complete']); self.assertEqual(len(result[4]), 0)
        self.assertEqual(result[2]['completedFraction'], 0.)
        self.assertEqual(result[2]['varyingCableControl'], strict.cable_parameter_control.description())
        totals = result[2]['varyingCableAcceptedWorkTotals']
        self.assertEqual(totals['acceptedStepCount'], 0)
        self.assertTrue(all(value == 0. for value in totals['valuesJoules'].values()))
        self.assertTrue(all(_rational(value) == 0 for value in totals['errorBoundsJoules'].values()))

    def test_keyboard_interrupt_journals_only_completed_prefix(self):
        for origin in ('step', 'work'):
            solver = ReportFixture(); journal = Journal([]); accepted = []
            if origin == 'step': solver.interrupt_at = 2
            real = solver_energy_balance.global_energy_transition
            calls = []
            def interrupt(*args, **kwargs):
                calls.append(1)
                if origin == 'work' and len(calls) == 2:
                    args[1][:] = 99.
                    raise KeyboardInterrupt('explicit varying work interruption')
                return real(*args, **kwargs)
            q = np.array([[0., 0., 0.], [.002, 0., 0.]]); original = q.copy()
            with mock.patch.object(solver_energy_balance, 'global_energy_transition', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    adaptive_contact_step(solver, q, np.zeros_like(q), np.empty((0, 3)), np.empty((0, 3)), .4,
                        initial_subdivisions=4, max_depth=0, attempt_journal=journal,
                        cable_parameter_schedule=schedule(solver.cable_parameter_control),
                        on_accept=lambda q,v,r:accepted.append((q.copy(),v.copy(),copy.deepcopy(r))))
            self.assertEqual([row[0]['outcome'] for row in journal.rows], ['accepted', 'interrupted'])
            self.assertEqual(len(accepted), 1); self.assertIsNone(journal.rows[-1][1])
            np.testing.assert_array_equal(q, original)

    def test_actual_constant_schedule_matches_fixed_continuation(self):
        from test_solver_cable_varying_global import particle_fixture
        from solver_global_sewing import GlobalSewingSolver
        model, varying, q, recipe = particle_fixture(activation=.5)
        c = varying.cable_parameter_control
        fixed = GlobalSewingSolver(model, [], 1., continuous_cable=recipe.potential(*recipe.initial_parameters),
                                   cable_precision=c.response_precision)
        raw = schedule(c, constant=True); empty = np.empty((0, 3)); v = np.zeros_like(q)
        result = adaptive_contact_step(varying, q, v, empty, empty, .5, initial_subdivisions=2,
            max_depth=0, cable_parameter_schedule=raw)
        legacy = adaptive_contact_step(fixed, q, v, empty, empty, .5, initial_subdivisions=2, max_depth=0)
        self.assertTrue(result[2]['complete'], result[2]); self.assertTrue(legacy[2]['complete'], legacy[2])
        np.testing.assert_array_equal(result[0], legacy[0]); np.testing.assert_array_equal(result[1], legacy[1])
        for current, old in zip(result[2]['acceptedSteps'], legacy[2]['acceptedSteps']):
            for field in ('continuousCable', 'cableStationarity', 'directionHistory', 'gradientInfinityNorm'):
                self.assertEqual(current['step'][field], old['step'][field])
            self.assertEqual(current['step']['energyBalance']['cableParameterWorkJoules'], 0.)
            self.assertEqual(current['step']['energyBalance']['cableFixedParameterChangeJoules'],
                             old['step']['energyBalance']['cableFixedParameterChangeJoules'])
        self.assertNotIn('varyingCableControl', legacy[2])

    def test_actual_cloth_contact_held_sewing_and_release_keep_guards(self):
        from test_solver_cable_varying_global import cloth_fixture
        from solver_global_sewing import GlobalSewingSolver
        from solver_triangle_sweep import triangle_sweep_safe
        model, base, q, recipe = cloth_fixture(barrier=True)
        solver = GlobalSewingSolver(model, [{2: 1., 6: -1.}], 1e-3, sewing_mode='distance',
            contact=base.contact, fold_barrier_joules=1e-6, cable_parameter_recipe=recipe,
            cable_precision=base.cable_parameter_control.response_precision,
            cable_parameter_work_precision=base.cable_parameter_control.parameter_work_precision)
        d, a = recipe.initial_parameters
        raw = {'profile': 'cable-target-activation-v1', 'geometrySha256': recipe.geometry_sha256,
               'cellIds': list(recipe.cell_ids), 'knots': [
                   {'fraction': f, 'targetsMeters': targets, 'activation': weights}
                   for f, targets, weights in ((0.,d.tolist(),a.tolist()), (.25,d.tolist(),a.tolist()),
                       (.5, [[.0016,.0016], [.003,.003]], [1.,1.]),
                       (.75, [[.0017,.0017], [.004,.004]], [0.,0.]),
                       (1., [[.002,.002], [.005,.005]], [0.,0.]))]}
        sew = {'profile':'sewing-row-activation-v1', 'rowIds':['held-synthetic-point'],
               'knots':[{'fraction':f, 'activation':[1.]} for f in (0.,1.)]}
        targets = np.array([np.linalg.norm(q[2]-q[6])]); states=[]
        final, v, report = adaptive_contact_step(solver, q, np.zeros_like(q), targets, targets, .008,
            initial_subdivisions=4, max_depth=2, cable_parameter_schedule=raw,
            sewing_schedule=sew, sewing_row_ids=sew['rowIds'], on_accept=lambda q,v,r:states.append((q,v,r)))
        self.assertTrue(report['complete'], report); self.assertGreater(len(states), 0)
        previous = q
        self.assertGreater(solver.contact.energy(q), 0.)
        self.assertTrue(any(row['step']['continuousCable']['energyJoules'] > 0 for row in report['acceptedSteps']))
        for current, velocity, row in states:
            self.assertTrue(triangle_sweep_safe(previous, current, solver.faces))
            self.assertTrue(solver.contact.path_safe(previous, current))
            self.assertEqual(row['step']['activeSewingRows'], [0])
            self.assertIn('sewingParameterWorkJoules', row['step']['energyBalance'])
            previous = current
        self.assertEqual(report['acceptedSteps'][-1]['step']['continuousCable']['energyJoules'], 0.)
        self.assertTrue(any(row['step']['energyBalance']['cableReleaseEnergyRemovedJoules'] > 0
                            for row in report['acceptedSteps']))


if __name__ == '__main__':
    unittest.main()
