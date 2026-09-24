"""Actual generic contact motion, independent error accounting and rejection."""
import copy
from dataclasses import asdict
from fractions import Fraction as F
import unittest
from unittest.mock import patch

import ipctk
import newton
import numpy as np
from scipy.sparse import eye
import warp as wp

from solver_adaptive_contact import adaptive_contact_step
from solver_cable_parameters import CableParameterRecipe
from solver_continuous_cable_sewing import ContinuousCableSewing
from solver_contact_work import WorkPolicy
from solver_contact_work_control import checked_change, validate_contact_energy
from solver_energy_balance import global_energy_transition, validate_contact_mechanical, _VARYING_BASE_SCALARS
from solver_global_sewing import GlobalSewingSolver, _direct_descent
from solver_temporal_control import energy_defect, problem_identity
from test_solver_contact_work_native import fixture as native_fixture
from test_solver_cable_global import PRECISION, anchor
from test_solver_cable_varying_control import WORK_PRECISION
from test_solver_temporal_control import declaration


POLICY = asdict(WorkPolicy())
EMPTY = np.empty((0, 3))
DT = 1e-5
MECHANICAL = ('mechanicalChangeJoules', 'mechanicalChangeMinusTargetWorkJoules',
              'mechanicalChangeMinusParameterWorkJoules')
PARAMETER = ('targetParameterWorkJoules', 'externalParameterWorkJoules')


def fraction(value): return F(int(value['numerator']), int(value['denominator']))


def cloth(route='contact'):
    """Two rotated squares with actual native contact, no source/garment admission."""
    contact, q = native_fixture()
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
        vel=wp.vec3(0, 0, 0), vertices=contact.rest_positions.tolist(),
        indices=contact.faces.reshape(-1).tolist(), density=.2,
        tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.001, edge_kd=0)
    builder.set_coloring([[i] for i in range(len(q))])
    model = builder.finalize(device='cpu')
    options = dict(contact=contact, contact_work_policy=copy.deepcopy(POLICY))
    if route != 'contact':
        cable = ContinuousCableSewing(8, [{'id': 'square-edge', 'positiveStart': anchor(0),
            'positiveEnd': anchor(1), 'negativeStart': anchor(4), 'negativeEnd': anchor(5),
            'targetsMeters': [.0015, .0015], 'referenceLengthMeters': {'numerator': '1', 'denominator': '100'},
            'stiffnessDensityNPerM2': 100., 'activation': .5}])
        options['cable_precision'] = copy.deepcopy(PRECISION)
        if route == 'fixed': options['continuous_cable'] = cable
        elif route == 'varying':
            options.update(cable_parameter_recipe=CableParameterRecipe(cable),
                           cable_parameter_work_precision=copy.deepcopy(WORK_PRECISION))
        else: raise ValueError(route)
    return GlobalSewingSolver(model, [], 1e-8, **options), q


def transition(solver, q, *, temporal=True):
    step_options, energy_options = {}, {}
    if solver.cable_parameter_control is not None:
        step_options = dict(cable_targets=[[.0015, .0015]], cable_activation=[.8])
        energy_options = dict(previous_cable_targets=[[.0015, .0015]], previous_cable_activation=[.5], **step_options)
    v = np.zeros_like(q)
    end, velocity, report = solver.step(q, v, EMPTY, DT, **step_options)
    energy = global_energy_transition(solver, q, end, v, velocity, EMPTY, EMPTY, DT,
        include_temporal_motion=temporal, **energy_options)
    return end, velocity, report, energy


class ContactWorkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): ipctk.set_num_threads(1)

    def test_actual_contact_and_both_cable_routes_use_one_bounded_motion_record(self):
        for route in ('contact', 'fixed', 'varying'):
            with self.subTest(route=route):
                solver, q = cloth(route)
                identity = problem_identity(solver)
                end, velocity, step, energy = transition(solver, q)
                self.assertTrue(step['converged'], step)
                self.assertGreater(np.max(np.abs(end-q)), 0.)
                self.assertEqual(step['boundedContactWork'], energy['boundedContactWork'])
                self.assertEqual(step['boundedContactWork'], checked_change(solver.contact_work_control, solver.contact, q, end))
                self.assertGreater(energy['contactBeforeJoules'], 0.)
                self.assertLess(energy['contactChangeJoules'], 0.)
                self.assertEqual(solver.sewing.shape, (0, 8))
                np.testing.assert_array_equal(velocity, (end-q)/DT)
                self.assertEqual(problem_identity(solver), identity)
                self.assertTrue(solver.contact.path_safe(q, end))
                from solver_triangle_sweep import triangle_sweep_safe
                self.assertTrue(triangle_sweep_safe(q, end, solver.faces))
                self.assertGreater(len(step['directionHistory']), 0)
                for row in step['directionHistory']:
                    decision = row['boundedWorkDecision']
                    self.assertLess(fraction(decision['conditionalSlopeUpperJoules']), 0)
                    self.assertLessEqual(fraction(decision['conditionalChangeUpperJoules']),
                                         F(1e-4)*fraction(decision['conditionalSlopeLowerJoules']))

    def test_independent_exact_sum_and_temporal_radii_count_contact_once(self):
        for route in ('contact', 'fixed', 'varying'):
            with self.subTest(route=route):
                solver, q = cloth(route)
                end, velocity, _, energy = transition(solver, q)
                contact = F(energy['boundedContactWork']['changeErrorBoundJoules'])
                self.assertGreater(contact, 0)
                motion = target = parameter = activation = F()
                if route == 'fixed':
                    motion = fraction(energy['continuousCableEnergy']['work']['certificate']['changeErrorBoundJoules'])
                elif route == 'varying':
                    record = energy['varyingCableEnergy']
                    motion = fraction(record['motionWork']['certificate']['changeErrorBoundJoules'])
                    errors = record['parameterWork']['certificate']['errorsJoules']
                    target, parameter, activation = (fraction(errors[k]) for k in
                        ('targetWorkJoules', 'totalWorkJoules', 'activationWorkJoules'))
                expected_bases = dict(zip(MECHANICAL+PARAMETER,
                    (contact+motion+parameter, contact+motion+activation, contact+motion, target, parameter)))
                sums = energy['boundedContactMechanical']
                for name, base in expected_bases.items():
                    terms = sums['aggregationTermsJoules'][name]
                    exact = sum((F(v) for v in terms.values()), F())
                    rounded = float(exact)
                    self.assertEqual(energy[name].hex(), rounded.hex())
                    self.assertEqual(fraction(sums['errorBoundsJoules'][name]), base+abs(F(rounded)-exact))
                    self.assertEqual(fraction(sums['assemblyRoundingBoundsJoules'][name]), abs(F(rounded)-exact))
                defect = energy_defect(solver.mass, np.zeros_like(q), velocity, energy, F(1))
                self.assertEqual(fraction(defect['knownErrorBoundJoules']), contact+motion)
                kinetic = sum((F(float(m))/2*sum((F(float(v))**2 for v in row), F())
                               for m, row in zip(solver.mass, velocity)), F())
                self.assertEqual(fraction(defect['exactStoredKineticChangeJoules']), kinetic)

    def test_temporal_omitted_doubled_and_altered_contact_terms_reject(self):
        solver, q = cloth()
        _, velocity, _, original = transition(solver, q)
        radius = F(original['boundedContactWork']['changeErrorBoundJoules'])
        for value in (F(), 2*radius):
            energy = copy.deepcopy(original)
            energy['temporalMotion']['knownErrorBoundJoules'] = {
                'numerator': str(value.numerator), 'denominator': str(value.denominator)}
            with self.subTest(radius=value), self.assertRaisesRegex(ValueError, 'uncertainty'):
                energy_defect(solver.mass, q*0, velocity, energy, F(1))
        energy = copy.deepcopy(original)
        energy['temporalMotion']['termsJoules']['contactChangeJoules'] = 0.
        with self.assertRaises(ValueError): energy_defect(solver.mass, q*0, velocity, energy, F(1))

    def test_every_unconditional_field_and_parameter_sum_is_required_raw(self):
        solver, q = cloth()
        _, _, _, original = transition(solver, q)
        for key in _VARYING_BASE_SCALARS:
            for value in (None, False, np.longdouble(original[key])):
                report = copy.deepcopy(original)
                if value is None: del report[key]
                else: report[key] = value
                with self.subTest(key=key, value=type(value)), self.assertRaises(ValueError):
                    validate_contact_mechanical(report)
        for key in PARAMETER:
            report = copy.deepcopy(original); report[key] = 1.
            with self.subTest(parameter=key), self.assertRaises(ValueError): validate_contact_mechanical(report)

    def test_bound_crossing_armijo_rejects_a_favorable_midpoint(self):
        def evaluate(x, jacobian=False): return eye(1) if jacobian else x
        result = _direct_descent(evaluate, np.array([1.]), 6, lambda x: eye(1),
            lambda x: float(x@x)/2, gradient_function=lambda x: x,
            gradient_error_function=lambda x: F(),
            energy_change_interval_function=lambda start, end: (F(-1001,1000), F(999,1000)),
            decision_record_key='boundedWorkDecision')
        self.assertFalse(result.success)
        np.testing.assert_array_equal(result.x, [1.])
        self.assertEqual(result.direction_history, [])

    def test_raw_states_and_timesteps_reject_all_three_entrypoints(self):
        solver, q = cloth()
        for dt in (True, 2**53+1, np.longdouble('.00001')):
            calls = (lambda: solver.step(q, q*0, EMPTY, dt),
                     lambda: global_energy_transition(solver, q, q, q*0, q*0, EMPTY, EMPTY, dt),
                     lambda: adaptive_contact_step(solver, q, q*0, EMPTY, EMPTY, dt))
            for i, call in enumerate(calls):
                with self.subTest(dt=dt, route=i), self.assertRaises(ValueError): call()

    def test_contact_only_adaptive_and_temporal_paths_commit_validated_motion(self):
        for temporal in (False, True):
            with self.subTest(temporal=temporal):
                solver, q = cloth()
                options = {'temporal_policy': declaration()} if temporal else {}
                end, velocity, report = adaptive_contact_step(solver, q, q*0, EMPTY, EMPTY, DT,
                    max_depth=1, max_attempts=3, max_evaluations=64, **options)
                self.assertTrue(report['complete'], report)
                self.assertEqual(len(report['acceptedSteps']), 2 if temporal else 1)
                self.assertGreater(np.max(np.abs(end-q)), 0.)
                self.assertEqual(report['boundedContactControl'], solver.contact_work_control.description())
                for row in report['acceptedSteps']:
                    validate_contact_mechanical(row['step']['energyBalance'])
                    self.assertEqual(row['step']['boundedContactWork'], row['step']['energyBalance']['boundedContactWork'])

    def test_corrupt_energy_never_commits_or_reaches_accept_callback(self):
        import solver_energy_balance
        real = solver_energy_balance.global_energy_transition
        for attack in ('missing', 'understated', 'incomplete', 'raw', 'parameter', 'state', 'model'):
            solver, q = cloth(); callbacks = []
            def corrupt(*args, **kwargs):
                energy = real(*args, **kwargs)
                if attack == 'missing': del energy['boundedContactWork']
                elif attack == 'understated': energy['boundedContactWork']['changeErrorBoundJoules'] = 0.
                elif attack == 'incomplete': del energy['externalParameterWorkJoules']
                elif attack == 'raw': energy['externalParameterWorkJoules'] = np.longdouble(0.)
                elif attack == 'parameter': energy['externalParameterWorkJoules'] = 1.
                elif attack == 'state': args[1][:] = 777.
                else: solver.mass[0] *= 2
                return energy
            with self.subTest(attack=attack), patch.object(solver_energy_balance, 'global_energy_transition', side_effect=corrupt):
                end, velocity, report = adaptive_contact_step(solver, q, q*0, EMPTY, EMPTY, DT,
                    max_depth=0, max_attempts=1, on_accept=lambda *args: callbacks.append(args))
            self.assertFalse(report['complete'], (attack, report))
            self.assertEqual(callbacks, [])
            np.testing.assert_array_equal(end, q)
            np.testing.assert_array_equal(velocity, q*0)

    def test_late_validation_mutation_preserves_the_already_accepted_prefix(self):
        import solver_contact_work_control
        real = solver_contact_work_control.validate_contact_energy
        solver, q = cloth(); callbacks = []; calls = 0
        def attack(control, contact, first, last, report):
            nonlocal calls
            result = real(control, contact, first, last, report)
            calls += 1
            # Each trial validates in energy accounting, then at the final
            # adaptive boundary. Attack the second trial's final boundary.
            if calls == 4: first[:] = 99.
            return result
        with patch.object(solver_contact_work_control, 'validate_contact_energy', side_effect=attack):
            end, velocity, report = adaptive_contact_step(solver, q, q*0, EMPTY, EMPTY, DT,
                initial_subdivisions=2, max_depth=0, max_attempts=2,
                on_accept=lambda *args: callbacks.append(args))
        self.assertEqual(calls, 4)
        self.assertFalse(report['complete'], report)
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(len(report['acceptedSteps']), 1)
        np.testing.assert_array_equal(end, callbacks[0][0])
        np.testing.assert_array_equal(velocity, callbacks[0][1])


if __name__ == '__main__': unittest.main()
