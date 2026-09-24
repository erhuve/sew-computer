"""Independent varying-cable discrete accounting; no source motion or solver run."""
import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from solver_cable_integration import CableControl
from solver_cable_parameters import CableParameterRecipe
from solver_cable_varying import VaryingCableControl
from solver_continuous_cable_sewing import ContinuousCableSewing
from solver_energy_balance import global_energy_transition, validate_varying_cable_energy
from test_solver_cable_energy import fixture as fixed_fixture, cell, policy, rat, rational, anchor


def work_policy(**changes):
    value={"absolute_tolerance_joules":1e-12,"component_tolerances_joules":None,
           **{key:value for key,value in policy().items() if key.startswith(('max_','moment_'))}}
    value.update(changes)
    return value


def fixture(*,cells=None,response=None,work=None):
    solver,q=fixed_fixture()
    solver.continuous_cable=None
    solver.cable_parameter_control=VaryingCableControl(
        CableParameterRecipe(ContinuousCableSewing(4,[cell()] if cells is None else cells)),
        policy() if response is None else response,work_policy() if work is None else work)
    return solver,q


def controls(d0=1.,a0=1.,d1=1.,a1=1.):
    return dict(previous_cable_targets=[[d0,d0]],previous_cable_activation=[a0],
                cable_targets=[[d1,d1]],cable_activation=[a1])


def transition(solver,q,end=None,*,options=None,dt=1.,previous_velocity=None):
    end=q if end is None else end
    return global_energy_transition(solver,q,end,np.zeros_like(q) if previous_velocity is None else previous_velocity,
        (np.asarray(end)-np.asarray(q))/dt,np.empty((0,3)),np.empty((0,3)),dt,
        **(controls() if options is None else options))


def expected_constant(r0,r1,d0,a0,d1,a1,stiffness=2.,measure=F(1)):
    potential=lambda r,d,a:F(stiffness)*measure*F(a)*max(F(),F(r)-F(d))**2/2
    old=potential(r0,d0,a0);middle=potential(r0,d1,a0)
    changed=potential(r0,d1,a1);after=potential(r1,d1,a1)
    return {"cableBeforeJoules":old,"cableAfterJoules":after,
            "cableTargetParameterWorkJoules":middle-old,"cableActivationParameterWorkJoules":changed-middle,
            "cableParameterWorkJoules":changed-old,"cableActivationIncreaseWorkJoules":max(F(),changed-middle),
            "cableReleaseEnergyRemovedJoules":max(F(),middle-changed),
            "cableFixedParameterChangeJoules":after-changed,"cableChangeJoules":after-old}


class VaryingCableEnergyTests(unittest.TestCase):
    def assert_enclosed(self,report,expected):
        bounds=report['varyingCableEnergy']['errorBoundsJoules']
        for key,value in expected.items():
            self.assertLessEqual(abs(F(report[key])-value),rational(bounds[key]),key)

    def assert_assembly(self,report):
        record=report['varyingCableEnergy'];p=record['parameterWork']['certificate']['errorsJoules']
        eP,eT,eA=(rational(p[name]) for name in ('totalWorkJoules','targetWorkJoules','activationWorkJoules'))
        eM=rational(record['motionWork']['certificate']['changeErrorBoundJoules'])
        errors={'mechanicalChangeJoules':eP+eM,'mechanicalChangeMinusTargetWorkJoules':eA+eM,
                'mechanicalChangeMinusParameterWorkJoules':eM,'targetParameterWorkJoules':eT,
                'externalParameterWorkJoules':eP,'cableChangeJoules':eP+eM}
        for name,terms in record['aggregationTermsJoules'].items():
            exact=sum(map(F,terms.values()),F());rounded=float(exact)
            self.assertEqual(report[name],rounded,name)
            rounding=abs(F(rounded)-exact)
            self.assertEqual(rational(record['assemblyRoundingBoundsJoules'][name]),rounding,name)
            self.assertEqual(rational(record['errorBoundsJoules'][name]),errors[name]+rounding,name)

    def test_exact_parameter_then_motion_accounting_and_direct_remainders(self):
        solver,q=fixture();q[1,0]=3.;end=q.copy();end[1,0]=2.5
        result=transition(solver,q,end,options=controls(2.,.25,1.,.75))
        expected=expected_constant(3.,2.5,2.,.25,1.,.75)
        for key,value in expected.items():self.assertEqual(result[key],float(value),key)
        kinetic=F(1,8)
        self.assertEqual(result['mechanicalChangeJoules'],float(kinetic+expected['cableChangeJoules']))
        self.assertEqual(result['mechanicalChangeMinusTargetWorkJoules'],
            float(kinetic+expected['cableActivationParameterWorkJoules']+expected['cableFixedParameterChangeJoules']))
        self.assertEqual(result['mechanicalChangeMinusParameterWorkJoules'],float(kinetic+expected['cableFixedParameterChangeJoules']))
        self.assert_assembly(result);self.assertIs(result['accepted'],False)
        self.assertNotIn('continuousCableEnergy',result)
        self.assertIn('conditional',result['varyingCableEnergy']['scope'])

    def test_retarget_before_release_uses_new_target_and_no_second_release_subtraction(self):
        solver,q=fixture();result=transition(solver,q,options=controls(1.,1.,.5,0.))
        self.assert_enclosed(result,expected_constant(2.,2.,1.,1.,.5,0.))
        self.assertEqual(result['cableReleaseEnergyRemovedJoules'],2.25)
        self.assertEqual(result['cableBeforeJoules'],1.)
        self.assertEqual(result['mechanicalChangeJoules'],-1.)
        self.assertEqual(result['mechanicalChangeMinusTargetWorkJoules'],-2.25)
        self.assertEqual(result['mechanicalChangeMinusParameterWorkJoules'],0.)
        self.assert_assembly(result)

    def test_direct_total_keeps_cancellation_and_does_not_double_parameter_uncertainty(self):
        components={key:1e-12 for key in ('targetWorkJoules','activationWorkJoules',
                    'activationIncreaseWorkJoules','releaseEnergyRemovedJoules')}
        solver,q=fixture(cells=[cell(stiffnessDensityNPerM2=1.)],
                         work=work_policy(absolute_tolerance_joules=1e-30,component_tolerances_joules=components))
        q[1,0]=3.;a1=math.nextafter(.25,math.inf)
        result=transition(solver,q,options=controls(2.,1.,1.,a1))
        self.assertEqual(result['cableParameterWorkJoules'],2.**-53)
        self.assertEqual(result['cableTargetParameterWorkJoules']+result['cableActivationParameterWorkJoules'],0.)
        self.assertEqual(result['mechanicalChangeJoules'],2.**-53)
        self.assertEqual(result['externalParameterWorkJoules'],2.**-53)
        bounds=result['varyingCableEnergy']['errorBoundsJoules']
        self.assertGreater(rational(bounds['cableActivationParameterWorkJoules']),0)
        self.assertEqual(rational(bounds['mechanicalChangeMinusParameterWorkJoules']),0)
        self.assertEqual(result['mechanicalChangeMinusParameterWorkJoules'],0.)
        self.assert_assembly(result)

    def test_nonsquare_radius_parameter_and_motion_errors_have_distinct_contributions(self):
        solver,q=fixture();q[1]=[1.,1.,0.];end=q.copy();end[1]=[1.25,1.,0.]
        result=transition(solver,q,end,options=controls(.25,.75,.5,.25))
        with localcontext() as context:
            context.prec=160
            def energy(radius,target,activation):return Decimal(activation)*(radius-Decimal(target))**2
            r0=Decimal(2).sqrt();r1=(Decimal('1.25')**2+1).sqrt()
            old=energy(r0,'.25','.75');mid=energy(r0,'.5','.75')
            new=energy(r0,'.5','.25');after=energy(r1,'.5','.25')
            expected={'cableBeforeJoules':F(old),'cableAfterJoules':F(after),
                'cableTargetParameterWorkJoules':F(mid-old),'cableActivationParameterWorkJoules':F(new-mid),
                'cableParameterWorkJoules':F(new-old),'cableFixedParameterChangeJoules':F(after-new),
                'cableChangeJoules':F(after-old)}
        self.assert_enclosed(result,expected);self.assert_assembly(result)
        bounds=result['varyingCableEnergy']['errorBoundsJoules']
        self.assertGreater(rational(bounds['cableParameterWorkJoules']),0)
        motion_bound=rational(result['varyingCableEnergy']['motionWork']['certificate']['changeErrorBoundJoules'])
        assembly_bound=rational(result['varyingCableEnergy']['assemblyRoundingBoundsJoules']['mechanicalChangeMinusParameterWorkJoules'])
        self.assertEqual(rational(bounds['mechanicalChangeMinusParameterWorkJoules']),motion_bound+assembly_bound)

    def test_tiny_target_work_is_not_reconstructed_from_rounded_endpoint_energies(self):
        solver,q=fixture(work=work_policy(absolute_tolerance_joules=1e-28));q[1,0]=100.
        after=math.nextafter(.001,math.inf)
        result=transition(solver,q,options=controls(.001,1.,after,1.))
        self.assertEqual(result['cableBeforeJoules'],result['cableAfterJoules'])
        self.assertLess(result['cableParameterWorkJoules'],0.)
        self.assert_enclosed(result,expected_constant(100.,100.,.001,1.,after,1.))
        self.assert_assembly(result)

    def test_tiny_motion_survives_equal_endpoint_diagnostics_with_constant_parameters(self):
        terms=anchor({1:1-F(1,2**40),2:F(1,2**40)})
        solver,q=fixture(cells=[cell(positiveStart=terms,positiveEnd=terms)],
                         response=policy(absolute_tolerance_joules=1e-35))
        q[2]=q[1];end=q.copy();end[2,0]=math.nextafter(q[2,0],math.inf)
        result=transition(solver,q,end)
        d=(F(end[2,0])-F(q[2,0]))/2**40
        self.assertEqual(result['cableBeforeJoules'],result['cableAfterJoules'])
        self.assert_enclosed(result,{'cableFixedParameterChangeJoules':2*d+d*d})
        self.assertGreater(result['cableFixedParameterChangeJoules'],0.)
        self.assertEqual(result['cableParameterWorkJoules'],0.)
        self.assert_assembly(result)

    def test_positive_work_below_binary64_is_retained_in_error_budget(self):
        tiny=math.ulp(0.)
        solver,q=fixture(cells=[cell(stiffnessDensityNPerM2=2*tiny)],
                         work=work_policy(absolute_tolerance_joules=tiny))
        new=math.nextafter(.5,1.)
        result=transition(solver,q,options=controls(1.,.5,1.,new))
        expected=F(2*tiny)*(F(new)-F(.5))/2
        self.assertGreater(expected,0);self.assertEqual(result['cableParameterWorkJoules'],0.)
        self.assert_enclosed(result,{'cableParameterWorkJoules':expected,'cableChangeJoules':expected})
        self.assertGreater(rational(result['varyingCableEnergy']['errorBoundsJoules']['cableParameterWorkJoules']),0)
        self.assertEqual(rational(result['varyingCableEnergy']['errorBoundsJoules']['mechanicalChangeMinusParameterWorkJoules']),0)
        self.assert_assembly(result)

    def test_pending_and_slack_collapse_are_zero_without_claiming_path_validity(self):
        solver,q=fixture();q[:]=0.
        for options in (controls(.25,0.,100.,0.),controls(.25,0.,100.,1.),controls(.25,1.,100.,0.)):
            result=transition(solver,q,options=options)
            for key,value in result.items():
                if key.startswith('cable') and key.endswith('Joules'):self.assertEqual(value,0.,key)
            self.assert_assembly(result);self.assertIs(result['accepted'],False)

    def test_coupled_sewing_fold_gripper_work_is_preserved_for_all_sewing_modes(self):
        from test_solver_controlled_fold_energy import fixture as coupled_fixture,report as coupled_report
        for mode in ('vector','distance','normal-offset'):
            for weighted in (False,True):
                with self.subTest(mode=mode,weighted=weighted):
                    solver,q,_=coupled_fixture(sewing=True,grippers=True,mode=mode)
                    end=q+[.125,0.,.0625]
                    old_sew=np.zeros((2,3)) if mode=='vector' else np.array([1.,1.5])
                    new_sew=np.array([[.25,.5,.125],[-.125,.25,0.]]) if mode=='vector' else np.array([.5,1.25])
                    args=dict(old_targets=[.5,-.25],targets=[1.,.5],old_activation=[.75,.25],activation=[.25,1.],
                        old_sewing=old_sew,sewing=new_sew,previous_gripper_targets=[[0.,0.,0.],[3.,0.,0.]],
                        gripper_targets=[[.5,.125,.25],[3.25,-.25,.5]],previous_gripper_activation=[.75,.5],gripper_activation=[.25,1.])
                    if weighted:args.update(previous_sewing_activation=[.75,.5],sewing_activation=[.25,1.])
                    baseline=coupled_report(solver,q,end,**args)
                    solver.continuous_cable=None
                    solver.cable_parameter_control=VaryingCableControl(CableParameterRecipe(ContinuousCableSewing(8,[cell()])),policy(),work_policy())
                    result=coupled_report(solver,q,end,**args,**controls(1.,1.,.5,.5))
                    expected=expected_constant(2.,2.,1.,1.,.5,.5)
                    self.assert_enclosed(result,expected);self.assert_assembly(result)
                    for key in baseline:
                        if key not in ('scope','mechanicalChangeJoules','mechanicalChangeMinusTargetWorkJoules',
                                       'mechanicalChangeMinusParameterWorkJoules','targetParameterWorkJoules','externalParameterWorkJoules'):
                            self.assertEqual(result[key],baseline[key],key)
                    for key,extra in (('mechanicalChangeJoules',expected['cableChangeJoules']),
                                      ('externalParameterWorkJoules',expected['cableParameterWorkJoules']),
                                      ('targetParameterWorkJoules',expected['cableTargetParameterWorkJoules']),
                                      ('mechanicalChangeMinusTargetWorkJoules',expected['cableActivationParameterWorkJoules'])):
                        self.assertEqual(result[key],float(F(baseline[key])+extra),key)
                    self.assertEqual(result['mechanicalChangeMinusParameterWorkJoules'],baseline['mechanicalChangeMinusParameterWorkJoules'])

    def test_fixed_and_no_cable_dictionaries_do_not_gain_varying_fields(self):
        from test_solver_cable_energy import transition as fixed_transition
        solver,q=fixed_fixture();before=fixed_transition(solver,q)
        solver.cable_parameter_control=None
        self.assertEqual(before,fixed_transition(solver,q))
        solver.continuous_cable=None;before=fixed_transition(solver,q)
        del solver.cable_parameter_control
        self.assertEqual(before,fixed_transition(solver,q))
        self.assertNotIn('varyingCableEnergy',before)
        with self.assertRaises(ValueError):transition(solver,q)

    def test_all_four_controls_exclusive_model_and_raw_admission(self):
        solver,q=fixture()
        for key in controls():
            options=controls();options[key]=None
            with self.subTest(missing=key),self.assertRaises(ValueError):transition(solver,q,options=options)
        for field,value in (('previous_cable_targets',[[True,1.]]),('cable_targets',[[0.,1.]]),
                            ('previous_cable_activation',[True]),('cable_activation',[1.1])):
            options=controls();options[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):transition(solver,q,options=options)
        for bad in (True,2**53+1,float('nan')):
            raw=q.tolist();raw[0][0]=bad
            with self.assertRaises(ValueError):transition(solver,raw)
            velocity=np.zeros_like(q).tolist();velocity[0][0]=bad
            with self.assertRaises(ValueError):transition(solver,q,previous_velocity=velocity)
            with self.assertRaises(ValueError):transition(solver,q,dt=bad)
        if np.dtype(np.longdouble).itemsize>8:
            with self.assertRaises(ValueError):transition(solver,q.astype(np.longdouble))
        saved=solver.cable_parameter_control
        solver.continuous_cable=CableControl(ContinuousCableSewing(4,[cell()]),policy())
        with self.assertRaises(ValueError):transition(solver,q)
        solver.continuous_cable=None;solver.cable_parameter_control=SimpleNamespace(vertex_count=4)
        with self.assertRaises(ValueError):transition(solver,q)
        solver.cable_parameter_control=saved;solver.active[0]=False
        with self.assertRaises(ValueError):transition(solver,q)
        solver.active[:]=True;solver.mass[0]=0.
        with self.assertRaises(ValueError):transition(solver,q)

    def test_structural_records_uncertainty_and_parameter_identity_attacks_reject(self):
        solver,q=fixture();options=controls(.5,.75,1.,.25);valid=transition(solver,q,options=options)
        values=(options['previous_cable_targets'],options['previous_cable_activation'],options['cable_targets'],options['cable_activation'])
        checked=validate_varying_cable_energy(solver.cable_parameter_control,q,q,*values,valid)
        self.assertEqual(checked,valid['varyingCableEnergy']);checked['definition'].clear()
        self.assertTrue(valid['varyingCableEnergy']['definition'])
        attacks=[lambda x:x.pop('varyingCableEnergy'),lambda x:x.update(accepted=0),
            lambda x:x.update(continuousCableEnergy={}),lambda x:x.update(cableParameterWorkJoules=True),
            lambda x:x.update(externalParameterWorkJoules=float('nan')),
            lambda x:x['varyingCableEnergy'].update(extra=1),
            lambda x:x['varyingCableEnergy']['beforeParameters'].update(parameterSha256='0'*64),
            lambda x:x['varyingCableEnergy']['definition'].update(accepted=0),
            lambda x:x['varyingCableEnergy']['before']['certificate'].update(positionsSha256='0'*64),
            lambda x:x['varyingCableEnergy']['motionWork']['certificate'].update(inputSha256='0'*64),
            lambda x:x['varyingCableEnergy']['parameterWork']['certificate'].update(verified=1),
            lambda x:x['varyingCableEnergy']['parameterWork']['certificate']['errorsJoules'].pop('targetWorkJoules'),
            lambda x:x['varyingCableEnergy']['errorBoundsJoules'].update(mechanicalChangeMinusParameterWorkJoules=rat(math.ulp(0.))),
            lambda x:x['varyingCableEnergy']['assemblyRoundingBoundsJoules'].update(cableChangeJoules=rat(1)),
            lambda x:x['varyingCableEnergy']['aggregationTermsJoules']['mechanicalChangeJoules'].pop('cableParameterWorkJoules'),
            lambda x:x['varyingCableEnergy']['aggregationTermsJoules']['mechanicalChangeMinusParameterWorkJoules'].update(cableParameterWorkJoules=0.)]
        for index,attack in enumerate(attacks):
            forged=copy.deepcopy(valid);attack(forged)
            with self.subTest(attack=index),self.assertRaises(ValueError):
                validate_varying_cable_energy(solver.cable_parameter_control,q,q,*values,forged)

    def test_corrupted_helper_records_reject_at_integration_boundary(self):
        solver,q=fixture();original=VaryingCableControl.parameter_work
        attacks=[lambda x:x.pop('certificate'),lambda x:x.update(totalWorkJoules=1.),
            lambda x:x['certificate'].update(beforePotentialSha256='0'*64),
            lambda x:x['certificate']['errorsJoules'].update(totalWorkJoules=rat(-1)),
            lambda x:x['certificate']['budgets'].update(moment_max_depth=1),
            lambda x:x['certificate'].update(sourceControlsInstalled=0)]
        for index,attack in enumerate(attacks):
            def changed(control,*args):
                result=original(control,*args);attack(result);return result
            with self.subTest(attack=index),patch.object(VaryingCableControl,'parameter_work',changed),self.assertRaises(ValueError):
                transition(solver,q)
        evaluate=CableControl.evaluate
        def forged(control,state):
            result=evaluate(control,state);result['certificate']['inputSha256']='0'*64;return result
        with patch.object(CableControl,'evaluate',forged),self.assertRaises(ValueError):transition(solver,q)

    def test_every_unconditional_public_energy_scalar_is_required_and_finite(self):
        # This public report inventory is independent of the implementation's
        # summand map. Endpoint diagnostics are required even when a controller
        # is absent and they do not enter a mechanical sum.
        fields=('membraneChangeJoules','bendingChangeJoules','bendingBeforeJoules','bendingAfterJoules',
            'foldBarrierChangeJoules','foldBarrierBeforeJoules','foldBarrierAfterJoules',
            'contactChangeJoules','contactBeforeJoules','contactAfterJoules','kineticChangeJoules',
            'sewingChangeJoules','sewingBeforeJoules','sewingAfterJoules',
            'foldActuationChangeJoules','foldActuationBeforeJoules','foldActuationAfterJoules',
            'foldTargetParameterWorkJoules','gripperBeforeJoules','gripperAfterJoules',
            'gripperFixedPositionAfterJoules','gripperFixedParameterChangeJoules','gripperChangeJoules',
            'gripperParameterWorkJoules','gripperTargetParameterWorkJoules','gripperActivationParameterWorkJoules',
            'gripperActivationIncreaseWorkJoules','gripperReleaseEnergyRemovedJoules',
            'gripperParameterWorkComponentSumErrorBoundJoules','targetParameterWorkJoules','externalParameterWorkJoules',
            'mechanicalChangeJoules','mechanicalChangeMinusTargetWorkJoules','mechanicalChangeMinusParameterWorkJoules')
        solver,q=fixture();end=q.copy();end[1,0]+=.5
        valid=transition(solver,q,end);control=solver.cable_parameter_control;values=tuple(controls().values())
        self.assertGreater(valid['kineticChangeJoules'],0.)
        for field in ('sewingFixedParameterChangeJoules','sewingMotionAndActivationJoules',
                      'foldFixedParameterChangeJoules','foldActivationParameterWorkJoules'):
            self.assertNotIn(field,valid)
        validate_varying_cable_energy(control,q,end,*values,valid)
        for field in fields:
            for attack in ('delete','boolean','nonfinite'):
                forged=copy.deepcopy(valid)
                if attack=='delete':forged.pop(field)
                else:forged[field]=False if attack=='boolean' else float('nan')
                with self.subTest(field=field,attack=attack),self.assertRaises(ValueError):
                    validate_varying_cable_energy(control,q,end,*values,forged)

    def test_helper_input_mutation_replacement_and_interrupt_leave_caller_inputs_unchanged(self):
        solver,q=fixture();saved=q.copy();options=controls();snapshot=copy.deepcopy(options)
        original=VaryingCableControl.parameter_work
        def mutate(control,state,*args):
            result=original(control,state,*args);state[0,0]+=1.;return result
        with patch.object(VaryingCableControl,'parameter_work',mutate),self.assertRaisesRegex(ValueError,'mutated'):
            transition(solver,q,options=options)
        for metadata in ('shape','strides','dtype'):
            def mutate_metadata(control,state,*args):
                result=original(control,state,*args)
                if metadata=='shape':state.shape=(2,6)
                elif metadata=='strides':state.strides=(0,state.strides[1])
                else:state.dtype=np.uint64
                return result
            with self.subTest(metadata=metadata),patch.object(VaryingCableControl,'parameter_work',mutate_metadata),self.assertRaisesRegex(ValueError,'mutated'):
                transition(solver,q,options=options)
        def replace(control,*args):
            result=original(control,*args);solver.cable_parameter_control=None;return result
        with patch.object(VaryingCableControl,'parameter_work',replace),self.assertRaisesRegex(ValueError,'identity'):
            transition(solver,q,options=options)
        solver.cable_parameter_control=fixture()[0].cable_parameter_control
        with patch.object(VaryingCableControl,'parameter_work',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):
            transition(solver,q,options=options)
        np.testing.assert_array_equal(q,saved);self.assertEqual(options,snapshot)

    def test_unresolved_work_rejects_without_relaxing_declared_policy(self):
        solver,q=fixture();definition=solver.cable_parameter_control.description()
        with patch.object(VaryingCableControl,'parameter_work',side_effect=ValueError('unresolved parameter budget')):
            with self.assertRaisesRegex(ValueError,'unresolved'):transition(solver,q)
        with patch.object(CableControl,'energy_change',side_effect=ValueError('unresolved motion budget')):
            with self.assertRaisesRegex(ValueError,'unresolved'):transition(solver,q)
        self.assertEqual(solver.cable_parameter_control.description(),definition)


if __name__=='__main__':unittest.main()
