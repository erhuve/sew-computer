"""Generic varying-cable direct solves; no source, capture or garment admission.

The five two-particle cases are embedded exact inputs from the independent
backward-Euler design oracle. Expectations below follow a separate reduced-mass
quadratic minimization, not the global solver's returned energy report.
"""
import copy
from fractions import Fraction as F
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('WARP_CACHE_PATH', str(Path(__file__).resolve().parents[1] / '.planning/solver/warp-cache'))

import newton
import numpy as np

from solver_cable_integration import CableControl
from solver_cable_parameters import CableParameterRecipe
from solver_cable_varying import VaryingCableControl
from solver_continuous_cable_sewing import ContinuousCableSewing
from solver_global_sewing import GlobalSewingSolver
from solver_triangle_sweep import triangle_sweep_safe
from test_solver_cable_varying_control import (
    RESPONSE_PRECISION, WORK_PRECISION, pair_recipe, pair_cell, rational, encoded,
)

EMPTY = np.empty((0, 3))
# name, initial gap, initial relative velocity, old d, old alpha, new d, new alpha.
ANALYTIC_CASES = (
    ('engage-from-rest', F(3), F(), F(1), F(), F(1), F(1)),
    ('tighten-while-moving', F(3), F(-1), F(2), F(1), F(1), F(1)),
    ('retarget-and-release', F(3), F(-1), F(1), F(1), F(2), F()),
    ('slack-to-taut', F(1,2), F(2), F(1), F(1), F(1), F(1)),
    ('taut-to-slack', F(5,4), F(-2), F(1), F(1), F(1), F(1)),
)


def particle_fixture(gap=3., *, target=1., activation=1., response=None, work=None):
    q = np.array([[0., 0., 0.], [gap, 0., 0.]])
    builder = newton.ModelBuilder(gravity=(0., 0., 0.))
    for position in q:
        builder.add_particle(pos=position, vel=(0., 0., 0.), mass=2.)
    builder.set_coloring([[0], [1]])
    model = builder.finalize(device='cpu')
    recipe = pair_recipe(target=target, activation=activation)
    solver = GlobalSewingSolver(model, [], 1., cable_parameter_recipe=recipe,
        cable_precision=copy.deepcopy(RESPONSE_PRECISION if response is None else response),
        cable_parameter_work_precision=copy.deepcopy(WORK_PRECISION if work is None else work))
    return model, solver, q, recipe


def cloth_fixture(*, contact=True, barrier=False, response=None):
    from test_solver_cable_global import fixture, PRECISION
    model, fixed, q, base = fixture(contact_enabled=contact, barrier=barrier)
    recipe = CableParameterRecipe(base)
    precision = copy.deepcopy(PRECISION if response is None else response)
    solver = GlobalSewingSolver(model, [], 1e-8, contact=fixed.contact,
        fold_barrier_joules=1e-6 if barrier else None, cable_parameter_recipe=recipe,
        cable_precision=precision, cable_parameter_work_precision=copy.deepcopy(WORK_PRECISION))
    return model, solver, q, recipe


class CableVaryingGlobalTests(unittest.TestCase):
    def test_five_independent_exact_backward_euler_cases(self):
        h, reduced_mass = F(1,2), F(1)
        for name, r0, speed0, d0, a0, d1, a1 in ANALYTIC_CASES:
            with self.subTest(case=name):
                _, solver, q, recipe = particle_fixture(float(r0), target=float(d0), activation=float(a0))
                v = np.array([[-float(speed0)/2, 0., 0.], [float(speed0)/2, 0., 0.]])
                predicted = r0+h*speed0; beta = 4*a1
                expected_gap = ((reduced_mass*predicted+h*h*beta*d1)/(reduced_mass+h*h*beta)
                                if beta and predicted>d1 else predicted)
                expected_q = np.array([[float((r0-expected_gap)/2), 0., 0.],
                                       [float((r0+expected_gap)/2), 0., 0.]])
                result, velocity, report = solver.step(q, v, EMPTY, float(h),
                    cable_targets=[[float(d1), float(d1)]], cable_activation=[float(a1)])
                self.assertTrue(report['converged'], report)
                self.assertIs(report['accepted'], False)
                self.assertEqual(report['profile'], 'experimental-global-varying-cable-reference-v1')
                np.testing.assert_allclose(result, expected_q, rtol=0, atol=2e-11)
                np.testing.assert_array_equal(velocity, (result-q)/float(h))
                np.testing.assert_allclose(solver.mass@(velocity-v), 0., atol=2e-12, rtol=0)
                error = rational(report['cableStationarity']['stationarityUpperBoundNewtons'])
                self.assertLessEqual(error, F(1e-6))
                expected_force = float(beta*max(expected_gap-d1, F()))
                observed = solver.mass[:, None]*(result-q-float(h)*v)/float(h)**2
                np.testing.assert_allclose(observed[:, 0], [expected_force, -expected_force], atol=1e-9, rtol=0)
                expected_energy = beta*max(expected_gap-d1, F())**2/2
                energy = report['continuousCable']['energyJoules']
                bound = rational(report['continuousCable']['certificate']['energyErrorBoundJoules'])
                # State solve roundoff is separate from the fixed-state certificate.
                self.assertLessEqual(abs(energy-float(expected_energy)), float(bound)+1e-9)
                control = solver.cable_parameter_control
                self.assertEqual(report['varyingCable'], control.parameter_record([[float(d1)]*2], [float(a1)]))
                self.assertEqual(recipe.initial_parameters[0].tolist(), [[float(d0)]*2])
                for decision in report.get('directionHistory') or []:
                    interval = decision['cableAwareDecision']
                    self.assertLess(rational(interval['conditionalSlopeUpperJoules']), 0)
                    self.assertLessEqual(rational(interval['conditionalChangeUpperJoules']),
                                        F(1e-4)*rational(interval['conditionalSlopeLowerJoules']))

    def test_constant_control_matches_existing_fixed_solver_and_records(self):
        model, varying, q, recipe = particle_fixture(target=1., activation=.5)
        fixed = GlobalSewingSolver(model, [], 1., continuous_cable=recipe.potential([[1.,1.]], [.5]),
                                   cable_precision=RESPONSE_PRECISION)
        v = np.array([[.125, 0., 0.], [-.125, 0., 0.]])
        state, velocity, report = varying.step(q, v, EMPTY, .5, cable_targets=[[1.,1.]], cable_activation=[.5])
        original, original_velocity, original_report = fixed.step(q, v, EMPTY, .5)
        self.assertTrue(report['converged']); self.assertTrue(original_report['converged'])
        np.testing.assert_array_equal(state, original); np.testing.assert_array_equal(velocity, original_velocity)
        for key in ('continuousCable', 'cableStationarity', 'directionHistory', 'initialEnergy', 'finalEnergy', 'gradientInfinityNorm'):
            self.assertEqual(report[key], original_report[key], key)
        self.assertNotIn('varyingCable', original_report)
        self.assertEqual(original_report['profile'], 'experimental-global-fixed-cable-reference-v1')

    def test_existing_vector_sewing_and_varying_cable_are_both_retained(self):
        model, _, q, recipe = particle_fixture()
        solver = GlobalSewingSolver(model, [{0:1.,1:-1.}], 1., cable_parameter_recipe=recipe,
            cable_precision=RESPONSE_PRECISION, cable_parameter_work_precision=WORK_PRECISION)
        # Reduced mass1, h1/2, old vector spring k1 and rest gap5/2,
        # cable beta4 and target1: minimizer gap=(12+5/2+4)/(4+1+4).
        expected_gap = F(37,18)
        expected = np.array([[float((3-expected_gap)/2),0.,0.],
                             [float((3+expected_gap)/2),0.,0.]])
        state, velocity, report = solver.step(q,np.zeros_like(q),[[-2.5,0.,0.]],.5,
            cable_targets=[[1.,1.]],cable_activation=[1.])
        self.assertTrue(report['converged'],report)
        np.testing.assert_allclose(state,expected,atol=2e-11,rtol=0)
        self.assertGreater(report['sewingJoules'],.09)
        self.assertGreater(report['continuousCable']['energyJoules'],2.)
        self.assertEqual(solver.sewing.shape,(1,2))
        np.testing.assert_allclose(solver.mass@velocity,0.,atol=2e-12,rtol=0)

    def test_actual_cloth_contact_and_varying_cable_balance_share_one_state(self):
        from test_solver_cable_global import force_parts
        model, solver, q, recipe = cloth_fixture()
        d, a = np.array([[.0016,.0016],[.003,.003]]), np.array([.75,0.])
        saved_d, saved_a = d.copy(), a.copy(); saved_rest = model.particle_q.numpy().copy()
        v = np.zeros_like(q); dt = .001
        state, velocity, report = solver.step(q, v, EMPTY, dt, cable_targets=d, cable_activation=a)
        self.assertTrue(report['converged'], report)
        effective = solver.cable_parameter_control.effective(d, a)
        response = effective.evaluate(state)
        parts = force_parts(solver, q, v, state, dt, response)
        self.assertGreater(response['energy'], 0.); self.assertGreater(solver.contact.energy(state), 0.)
        self.assertGreater(np.max(np.abs(parts['cable'])), 1e-7)
        self.assertGreater(np.max(np.abs(parts['contact'])), 1e-7)
        self.assertLessEqual(np.max(np.abs(sum(parts.values())))+
            float(rational(response['certificate']['gradientMaxAbsoluteErrorBoundNewtons'])), 1e-6+2e-10)
        stationarity = rational(report['cableStationarity']['stationarityUpperBoundNewtons'])
        np.testing.assert_allclose(solver.mass@velocity, 0., atol=8*dt*float(stationarity)+3e-12, rtol=0)
        self.assertTrue(triangle_sweep_safe(q, state, solver.faces)); self.assertTrue(solver.contact.path_safe(q, state))
        self.assertEqual(report['contact']['broadPhase'], 'ipctk-HashGrid-explicit-v1')
        np.testing.assert_array_equal(d, saved_d); np.testing.assert_array_equal(a, saved_a)
        np.testing.assert_array_equal(model.particle_q.numpy(), saved_rest)
        self.assertEqual(recipe.initial_parameters[1].tolist(), [1.,1.])
        pending = recipe.potential(d, [0.,0.]).evaluate(state,
            **{key:value for key,value in effective.policy.items() if key!='absolute_tolerance_joules'})
        self.assertEqual(pending['energy'], 0.); np.testing.assert_array_equal(pending['gradient'], 0.)

    def test_every_trial_uses_one_frozen_potential_and_reports_detached_controls(self):
        _, solver, q, _ = particle_fixture()
        d, a = np.array([[1.,1.]]), np.array([.5])
        source = solver.cable_parameter_control; before = encoded(source.description())
        original = CableControl.evaluate; identities = set()
        def observe(control, positions):
            identities.add(control.description()['inputSha256'])
            self.assertTrue(control.description()['parametersFixed'])
            self.assertEqual(control.description()['inputSha256'], source.parameter_record(d,a)['effectivePotentialSha256'])
            return original(control, positions)
        with patch.object(CableControl, 'evaluate', observe):
            state, velocity, report = solver.step(q, np.zeros_like(q), EMPTY, .5, cable_targets=d, cable_activation=a)
        self.assertTrue(report['converged']); self.assertEqual(len(identities), 1)
        report['varyingCable']['targetsMeters'][0][0] = 99.; report['varyingCable']['definition'].clear()
        d[:]=2.; a[:]=0.
        self.assertEqual(encoded(source.description()), before)
        self.assertEqual(source.recipe.initial_parameters[0].tolist(), [[1.,1.]])
        self.assertEqual(source.recipe.initial_parameters[1].tolist(), [1.])

    def test_configuration_pairing_and_step_controls_are_explicit(self):
        model, solver, q, recipe = particle_fixture()
        configured = dict(cable_parameter_recipe=recipe, cable_precision=RESPONSE_PRECISION,
                          cable_parameter_work_precision=WORK_PRECISION)
        for missing in configured:
            malformed = dict(configured); malformed.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                GlobalSewingSolver(model, [], 1., **malformed)
        with self.assertRaises(ValueError):
            GlobalSewingSolver(model, [], 1., continuous_cable=recipe.potential([[1.,1.]], [1.]), **configured)
        bad_recipe = CableParameterRecipe(ContinuousCableSewing(3, [pair_cell()]))
        with self.assertRaises(ValueError): GlobalSewingSolver(model, [], 1., **dict(configured, cable_parameter_recipe=bad_recipe))
        for options in ({}, {'cable_targets':[[1.,1.]]}, {'cable_activation':[1.]},
                        {'cable_targets':[[1.,1.]],'cable_activation':[1.],'linear_solver':'lsmr'},
                        {'cable_targets':[[1.,1.]],'cable_activation':[1.],'linear_solver':'shifted'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                solver.step(q, np.zeros_like(q), EMPTY, .5, **options)
        fixed = GlobalSewingSolver(model, [], 1., continuous_cable=recipe.potential([[1.,1.]], [1.]),
                                  cable_precision=RESPONSE_PRECISION)
        plain = GlobalSewingSolver(model, [{0:1.,1:-1.}], 1.)
        for other, targets in ((fixed, EMPTY), (plain, np.array([[-3.,0.,0.]]))):
            with self.assertRaises(ValueError): other.step(q, np.zeros_like(q), targets, .5,
                                                          cable_targets=[[1.,1.]], cable_activation=[1.])

    def test_original_raw_states_controls_and_free_mass_admission(self):
        model, solver, q, recipe = particle_fixture()
        for key, value in (('q', True), ('v', True), ('dt', True), ('q',2**53+1), ('v',2**53+1),
                           ('dt',2**53+1), ('targets',True), ('activation',True)):
            pos, vel, dt = q.tolist(), np.zeros_like(q).tolist(), .5
            targets, activation = [[1.,1.]], [1.]
            if key=='q': pos[0][0]=value
            elif key=='v': vel[0][0]=value
            elif key=='dt': dt=value
            elif key=='targets': targets[0][0]=value
            else: activation[0]=value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                solver.step(pos, vel, EMPTY, dt, cable_targets=targets, cable_activation=activation)
        if np.dtype(np.longdouble).itemsize>8:
            with self.assertRaises(ValueError): solver.step(q.astype(np.longdouble), np.zeros_like(q), EMPTY, .5,
                                                           cable_targets=[[1.,1.]], cable_activation=[1.])
        for attribute, replacement in (('particle_mass',0.),('particle_flags',0)):
            original = getattr(model,attribute).numpy().copy(); changed=original.copy(); changed[0]=replacement
            getattr(model,attribute).assign(changed)
            try:
                with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                    GlobalSewingSolver(model, [], 1., cable_parameter_recipe=recipe,
                        cable_precision=RESPONSE_PRECISION, cable_parameter_work_precision=WORK_PRECISION)
            finally: getattr(model,attribute).assign(original)

    def test_failed_precision_preserves_input_and_complete_policies(self):
        from test_solver_cable_global import PRECISION
        policy = dict(PRECISION, max_boundary_depth=0)
        _, solver, q, _ = cloth_fixture(contact=False, response=policy)
        q[7,2]=.001; saved=q.copy(); velocity=np.zeros_like(q)
        d,a=np.array([[.0015,.0015],[.003,.003]]),np.array([1.,0.])
        snapshots=d.copy(),a.copy()
        with self.assertRaisesRegex(ValueError, '[Bb]oundary|unresolved'):
            solver.step(q,velocity,EMPTY,.001,cable_targets=d,cable_activation=a)
        np.testing.assert_array_equal(q,saved);np.testing.assert_array_equal(velocity,0.)
        np.testing.assert_array_equal(d,snapshots[0]);np.testing.assert_array_equal(a,snapshots[1])
        self.assertEqual(solver.cable_parameter_control.response_precision,policy)
        self.assertEqual(solver.cable_parameter_control.parameter_work_precision,WORK_PRECISION)

    def test_inactive_new_parameters_preserve_triangle_and_contact_paths(self):
        for contact in (False,True):
            _,solver,q,_=cloth_fixture(contact=contact)
            if contact:
                end=q.copy();end[:4,2],end[4:,2]=q[4:,2],q[:4,2]
            else: end=q@np.diag([-1.,-1.,1.])
            def fabricated(evaluate,start,maximum,hessian,objective,**options):
                self.assertTrue(np.isfinite(objective(end.ravel())))
                self.assertIsNone(options['energy_change_interval_function'](start,end.ravel()))
                return SimpleNamespace(x=end.ravel(),success=True,nfev=1,status=1,message='path attack')
            with self.subTest(contact=contact), patch('solver_global_sewing._direct_descent',side_effect=fabricated), \
                    self.assertRaisesRegex(ValueError,'Physical contact|Physical cloth'):
                solver.step(q,np.zeros_like(q),EMPTY,.001,
                    cable_targets=[[.0015,.0015],[.003,.003]],cable_activation=[0.,0.])

    def test_inactive_new_parameters_preserve_explicit_fold_barrier_branch(self):
        from solver_hinge_sweep import hinge_sweep_safe
        _,solver,flat,_=cloth_fixture(contact=False,barrier=True)
        def folded(angle):
            q=flat.copy();radius=-flat[1,1];q[1,1]=-radius*np.cos(angle);q[1,2]=-radius*np.sin(angle);return q
        q,end=folded(3.1),folded(-3.1)
        self.assertTrue(triangle_sweep_safe(q,end,solver.faces))
        self.assertFalse(hinge_sweep_safe(q,end,solver.fold_barrier.indices))
        def fabricated(evaluate,start,maximum,hessian,objective,**options):
            self.assertIsNone(options['energy_change_interval_function'](flat.ravel(),end.ravel()))
            return SimpleNamespace(x=end.ravel(),success=True,nfev=1,status=1,message='branch attack')
        with patch('solver_global_sewing._direct_descent',side_effect=fabricated), \
                self.assertRaisesRegex(ValueError,'fold-barrier hinge'):
            solver.step(q,np.zeros_like(q),EMPTY,.001,
                cable_targets=[[.0015,.0015],[.003,.003]],cable_activation=[0.,0.])

    def test_helper_mutation_and_replaced_varying_identity_cannot_return_state(self):
        original=CableControl.evaluate
        for mode in ('positions','identity'):
            _,solver,q,recipe=particle_fixture();targets=np.array([[1.,1.]]);activation=np.array([1.])
            definition=encoded(solver.cable_parameter_control.description())
            def corrupt(control,positions):
                result=original(control,positions)
                if mode=='positions': positions.flat[0]=math.nextafter(float(positions.flat[0]),math.inf)
                else: solver.cable_parameter_control=VaryingCableControl(recipe,RESPONSE_PRECISION,WORK_PRECISION)
                return result
            with self.subTest(mode=mode),patch.object(CableControl,'evaluate',corrupt),self.assertRaises(ValueError):
                solver.step(q,np.zeros_like(q),EMPTY,.5,cable_targets=targets,cable_activation=activation)
            np.testing.assert_array_equal(q,[[0.,0.,0.],[3.,0.,0.]])
            np.testing.assert_array_equal(targets,[[1.,1.]]);np.testing.assert_array_equal(activation,[1.])
            self.assertEqual(encoded(solver.cable_parameter_control.description()),definition)


if __name__=='__main__': unittest.main()
