"""Tightened force criteria, analytic solves and publication adversaries.

Tests exercise numerical admission only, not time accuracy or garment capture.
"""
import copy
from fractions import Fraction as F
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import eye

from solver_adaptive_contact import adaptive_contact_step
from solver_attempt_journal import AttemptJournal
from solver_cable_integration import stationarity
from solver_global_sewing import _direct_descent
from solver_stationarity import stationarity_tolerance, validate_tightened_report


TIGHT = 1e-8
EMPTY = np.empty((0, 3))


def report(norm=0., tolerance=TIGHT):
    return {"converged": True, "gradientInfinityNorm": norm, "stationarityToleranceN": tolerance}


def run_fake(step, **kwargs):
    return adaptive_contact_step(SimpleNamespace(step=step), np.zeros((1, 3)), np.ones((1, 3)),
        EMPTY, EMPTY, 1., stationarity_tolerance_newtons=TIGHT,
        max_attempts=kwargs.pop('max_attempts', 1), max_depth=kwargs.pop('max_depth', 0), **kwargs)


class StationarityCriterionTests(unittest.TestCase):
    def test_tightening_only_original_binary64_scalar_admission(self):
        for value in (1e-6, TIGHT, np.float32(1e-8), float.fromhex('0x0.0000000000001p-1022')):
            with self.subTest(valid=repr(value)):
                self.assertEqual(stationarity_tolerance(value), float(value))
        for value in (True, np.bool_(False), None, '1e-8', [TIGHT], np.array(TIGHT),
                      F(TIGHT), 0., -TIGHT, 1e-5, float('inf'), float('nan'), complex(TIGHT),
                      np.nextafter(1e-6, np.inf)):
            with self.subTest(invalid=repr(value)), self.assertRaises(ValueError):
                stationarity_tolerance(value)
        wider = np.longdouble(TIGHT) + np.finfo(np.longdouble).eps * np.longdouble(TIGHT)
        if wider != float(wider):
            with self.assertRaises(ValueError):stationarity_tolerance(wider)

    def test_quadratic_search_does_not_stop_at_the_legacy_criterion(self):
        def solve(tolerance):
            return _direct_descent(lambda q:q.copy(), np.array([1e-7]), 3, lambda q:eye(1),
                lambda q:float(q@q/2), gradient_function=lambda q:q.copy(),
                stationarity_tolerance_newtons=tolerance)
        old, tight = solve(1e-6), solve(TIGHT)
        self.assertTrue(old.success);self.assertTrue(tight.success)
        self.assertEqual(old.x[0], 1e-7);self.assertEqual(tight.x[0], 0.)
        self.assertEqual(old.nfev, 1);self.assertEqual(tight.nfev, 2)

    def test_exact_error_cannot_round_away_at_the_requested_boundary(self):
        error = F(1, 2**1074)
        self.assertEqual(float(F(TIGHT)+error), TIGHT)
        result = _direct_descent(lambda q:q.copy(), np.array([TIGHT]), 1, lambda q:eye(1),
            lambda q:float(q@q/2), gradient_function=lambda q:q.copy(),
            gradient_error_function=lambda q:error, energy_change_interval_function=lambda a,b:(F(),F()),
            stationarity_tolerance_newtons=TIGHT)
        self.assertFalse(result.success)
        bounded = stationarity(np.array([TIGHT]), error, error, F(), tolerance_newtons=TIGHT)
        with self.assertRaises(ValueError):
            validate_tightened_report(dict(report(TIGHT), cableStationarity=bounded), TIGHT)

    def test_raw_criterion_cannot_be_narrowed_to_the_requested_float(self):
        for field in ('top', 'nested'):
            bounds = stationarity(np.array([0.]), F(), F(), F(), tolerance_newtons=TIGHT)
            row = dict(report(), cableStationarity=bounds)
            value = F(TIGHT)+F(1,2**200)
            self.assertEqual(float(value), TIGHT)
            if field == 'top':row['stationarityToleranceN']=value
            else:bounds['toleranceNewtons']=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_tightened_report(row,TIGHT)

    def test_adaptive_rejects_raw_criterion_before_json_narrowing(self):
        def step(q,v,t,h,**options):return q+h*v,v,report(tolerance=F(TIGHT)+F(1,2**200))
        q,v,result=run_fake(step)
        self.assertFalse(result['complete']);np.testing.assert_array_equal(q,0.)
        self.assertEqual(result['rejectedSteps'][0]['error']['type'],'ValueError')

    def test_strict_report_requires_the_requested_top_and_nested_criteria(self):
        for field in ('stationarityToleranceN','toleranceNewtons'):
            for invalid in (None,True,1e-6,TIGHT/2):
                bounds=stationarity(np.array([0.]),F(),F(),F(),tolerance_newtons=TIGHT)
                row=dict(report(),cableStationarity=bounds)
                target=row if field=='stationarityToleranceN' else bounds
                if invalid is None:del target[field]
                else:target[field]=invalid
                with self.subTest(field=field,invalid=invalid),self.assertRaises(ValueError):
                    validate_tightened_report(row,TIGHT)

    def test_final_bound_must_bind_norm_errors_and_exact_upper(self):
        for field in ('gradientInfinityNorm','totalGradientErrorBoundNewtons','stationarityUpperBoundNewtons'):
            bounds=stationarity(np.array([0.]),F(),F(),F(),tolerance_newtons=TIGHT)
            bounds[field]=(TIGHT if field=='gradientInfinityNorm' else {'numerator':'1','denominator':'1000000000'})
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_tightened_report(dict(report(),cableStationarity=bounds),TIGHT)

    def test_loose_success_cannot_advance_a_strict_adaptive_request(self):
        def step(q,v,t,h,**options):return q+999,v,report(norm=1e-7)
        q,v,result=run_fake(step)
        self.assertFalse(result['complete']);np.testing.assert_array_equal(q,0.)
        self.assertEqual(result['stationarityToleranceN'],TIGHT)

    def test_retries_retain_criterion_and_last_accepted_state(self):
        calls=[]
        def step(q,v,t,h,**options):
            calls.append((h,options['stationarity_tolerance_newtons'],q.copy()))
            if h>.5:return q+999,v,report(norm=1e-7)
            return q+h*v,v,report(norm=TIGHT/2)
        q,v,result=run_fake(step,max_attempts=3,max_depth=1)
        self.assertTrue(result['complete']);np.testing.assert_array_equal(q,1.)
        self.assertEqual([(h,t) for h,t,_ in calls],[(1.,TIGHT),(.5,TIGHT),(.5,TIGHT)])
        np.testing.assert_array_equal(calls[1][2],0.)
        self.assertEqual(len(result['rejectedSteps']),1)

    def test_strict_journal_callbacks_cannot_mutate_accepted_records_or_states(self):
        class Journal:
            def start(self,row):row['endFraction']=.25
            def outcome(self,row,q=None,v=None):
                row['step']['stationarityToleranceN']=1e-6
                if q is not None:q[:]=999;v[:]=888
            def finish(self,*args):pass
        def step(q,v,t,h,**options):return q+h*v,v,report()
        q,v,result=run_fake(step,attempt_journal=Journal())
        self.assertTrue(result['complete']);np.testing.assert_array_equal(q,1.)
        np.testing.assert_array_equal(v,1.)
        self.assertEqual(result['acceptedSteps'][0]['endFraction'],1.)
        self.assertEqual(result['acceptedSteps'][0]['step']['stationarityToleranceN'],TIGHT)

    def test_legacy_capture_journal_and_subclasses_reject_before_any_activity(self):
        class Child(AttemptJournal):pass
        calls=[]
        for cls in (AttemptJournal,Child):
            journal=object.__new__(cls)
            journal.start=lambda *args:calls.append('start')
            with self.subTest(cls=cls),self.assertRaisesRegex(ValueError,'Legacy captured'):
                run_fake(lambda *args,**kwargs:calls.append('step'),attempt_journal=journal)
        self.assertEqual(calls,[])

    def test_non_direct_strict_request_is_rejected_before_a_trial(self):
        for mode in ('shifted','lsmr'):
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'direct'):
                run_fake(lambda *args,**kwargs:self.fail('trial executed'),linear_solver=mode)

    def test_default_adaptive_request_does_not_add_trial_options_or_change_bytes(self):
        def step(q,v,t,h):return q+h*v,v,{'converged':True,'gradientInfinityNorm':1e-7}
        args=(SimpleNamespace(step=step),np.zeros((1,3)),np.ones((1,3)),EMPTY,EMPTY,1.)
        a=adaptive_contact_step(*args,max_depth=0,max_attempts=1)
        b=adaptive_contact_step(*args,max_depth=0,max_attempts=1,stationarity_tolerance_newtons=1e-6)
        self.assertEqual(a[0].tobytes(),b[0].tobytes());self.assertEqual(a[1].tobytes(),b[1].tobytes())
        self.assertEqual(json.dumps(a[2],sort_keys=True),json.dumps(b[2],sort_keys=True))

    def test_gripper_only_work_cannot_replace_or_narrow_a_strict_criterion(self):
        from test_solver_gripper_continuation import QuadraticSolver,control_schedule
        import solver_energy_balance
        class StrictQuadratic(QuadraticSolver):
            def step(self,*args,**options):
                q,v,row=super().step(*args,**options)
                row['stationarityToleranceN']=options['stationarity_tolerance_newtons']
                self.last_report=row
                return q,v,row
        for field,value in (('stationarityToleranceN',1e-6),('gradientInfinityNorm',TIGHT*2),
                            ('stationarityToleranceN',F(TIGHT)+F(1,2**200))):
            solver=StrictQuadratic();q=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]])
            real=solver_energy_balance.global_energy_transition
            def work(*args,**kwargs):
                result=real(*args,**kwargs);solver.last_report[field]=value;return result
            with self.subTest(field=field,value=value),patch.object(solver_energy_balance,'global_energy_transition',side_effect=work):
                q1,v1,result=adaptive_contact_step(solver,q,np.zeros_like(q),EMPTY,EMPTY,.1,
                    initial_subdivisions=4,max_attempts=4,max_depth=0,
                    gripper_schedule=control_schedule(),stationarity_tolerance_newtons=TIGHT)
                self.assertFalse(result['complete']);self.assertEqual(result['acceptedSteps'],[])
                np.testing.assert_array_equal(q1,q)

    def test_gripper_only_momentum_uses_requested_force_scale(self):
        from test_solver_gripper_continuation import QuadraticSolver,control_schedule
        class StrictQuadratic(QuadraticSolver):
            def step(self,*args,**options):
                q,v,row=super().step(*args,**options)
                row['stationarityToleranceN']=options['stationarity_tolerance_newtons']
                return q,v,row
        solver=StrictQuadratic();q=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]])
        q1,v1,result=adaptive_contact_step(solver,q,np.zeros_like(q),EMPTY,EMPTY,.1,
            initial_subdivisions=4,max_attempts=4,max_depth=0,
            gripper_schedule=control_schedule(),stationarity_tolerance_newtons=TIGHT)
        self.assertTrue(result['complete'],result)
        for step in result['acceptedSteps']:
            validate_tightened_report(step['step'],TIGHT)
            self.assertEqual(step['step']['gripperMomentum']['toleranceNs'],
                3*step['durationSeconds']*TIGHT+64*np.finfo(float).eps)


class ActualStationarityTests(unittest.TestCase):
    def test_five_analytic_particle_controls_satisfy_tightened_bounds(self):
        from test_solver_cable_varying_global import ANALYTIC_CASES,particle_fixture
        h=F(1,2)
        for name,r0,speed0,d0,a0,d1,a1 in ANALYTIC_CASES:
            with self.subTest(name=name):
                _,solver,q,_=particle_fixture(float(r0),target=float(d0),activation=float(a0))
                v=np.array([[-float(speed0)/2,0.,0.],[float(speed0)/2,0.,0.]])
                prediction=r0+h*speed0;beta=4*a1
                gap=(prediction+h*h*beta*d1)/(1+h*h*beta) if beta and prediction>d1 else prediction
                expected=np.array([[float((r0-gap)/2),0.,0.],[float((r0+gap)/2),0.,0.]])
                actual,velocity,row=solver.step(q,v,EMPTY,float(h),cable_targets=[[float(d1)]*2],
                    cable_activation=[float(a1)],stationarity_tolerance_newtons=TIGHT)
                validate_tightened_report(row,TIGHT)
                np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-11)
                np.testing.assert_array_equal(velocity,(actual-q)/float(h))

    def test_fixed_cloth_contact_tightens_residual_without_changing_rest(self):
        from test_solver_cable_global import fixture,force_parts,PRECISION
        model,solver,q,recipe=fixture()
        frozen=model.particle_q.numpy().copy();v=np.zeros_like(q)
        q1,v1,row=solver.step(q,v,EMPTY,.001,stationarity_tolerance_newtons=TIGHT)
        validate_tightened_report(row,TIGHT)
        response=recipe.evaluate(q1,**{k:v for k,v in PRECISION.items() if k!='absolute_tolerance_joules'})
        parts=force_parts(solver,q,v,q1,.001,response)
        self.assertLessEqual(np.max(np.abs(sum(parts.values()))),TIGHT+2e-10)
        np.testing.assert_array_equal(model.particle_q.numpy(),frozen)
        np.testing.assert_array_equal(v1,(q1-q)/.001)

    def test_omitted_and_explicit_default_have_identical_state_and_report_bytes(self):
        from test_solver_cable_varying_global import particle_fixture
        _,solver,q,_=particle_fixture()
        kwargs={'cable_targets':[[1.,1.]],'cable_activation':[1.]}
        a=solver.step(q,np.zeros_like(q),EMPTY,.5,**kwargs)
        b=solver.step(q,np.zeros_like(q),EMPTY,.5,stationarity_tolerance_newtons=1e-6,**kwargs)
        self.assertEqual(a[0].tobytes(),b[0].tobytes());self.assertEqual(a[1].tobytes(),b[1].tobytes())
        self.assertEqual(json.dumps(a[2],sort_keys=True),json.dumps(b[2],sort_keys=True))

    def test_actual_varying_adaptive_controls_preserve_strict_criterion(self):
        from test_solver_cable_varying_global import particle_fixture
        _,solver,q,recipe=particle_fixture(activation=0.)
        raw={'profile':'cable-target-activation-v1','geometrySha256':recipe.geometry_sha256,
             'cellIds':list(recipe.cell_ids),'knots':[
                 {'fraction':0.,'targetsMeters':[[1.,1.]],'activation':[0.]},
                 {'fraction':.5,'targetsMeters':[[1.,1.]],'activation':[1.]},
                 {'fraction':1.,'targetsMeters':[[1.,1.]],'activation':[0.]}]}
        q1,v1,result=adaptive_contact_step(solver,q,np.zeros_like(q),EMPTY,EMPTY,.5,
            initial_subdivisions=4,max_attempts=8,max_depth=1,cable_parameter_schedule=raw,
            stationarity_tolerance_newtons=TIGHT)
        self.assertTrue(result['complete'],result)
        self.assertEqual(result['stationarityToleranceN'],TIGHT)
        for step in result['acceptedSteps']:validate_tightened_report(step['step'],TIGHT)
        self.assertFalse(result['accepted'])

    def test_actual_fixed_cable_adaptive_preserves_tightened_bounds(self):
        from test_solver_cable_global import fixture
        _,solver,q,_=fixture()
        q1,v1,result=adaptive_contact_step(solver,q,np.zeros_like(q),EMPTY,EMPTY,.001,
            initial_subdivisions=1,max_attempts=1,max_depth=0,stationarity_tolerance_newtons=TIGHT)
        self.assertTrue(result['complete'],result)
        validate_tightened_report(result['acceptedSteps'][0]['step'],TIGHT)

    def test_strict_global_modes_and_scalar_validation_precede_control_evaluation(self):
        from test_solver_cable_varying_global import particle_fixture
        _,solver,q,_=particle_fixture()
        for options in ({'stationarity_tolerance_newtons':True},
                        {'stationarity_tolerance_newtons':1e-5},
                        {'stationarity_tolerance_newtons':TIGHT,'linear_solver':'lsmr'}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                solver.step(q,np.zeros_like(q),EMPTY,.5,**options)


if __name__=='__main__':unittest.main()
