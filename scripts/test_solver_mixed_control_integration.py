"""Actual synthetic sewing/fold/contact solves; no garment source or capture."""

import copy
from fractions import Fraction
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault("WARP_CACHE_PATH",str(Path(__file__).resolve().parents[1]/".planning/solver/warp-cache"))

import ipctk
import newton
import numpy as np
import warp as wp

from solver_adaptive_contact import adaptive_contact_step, CONTROLLED_FOLD_SWEEP_POLICY
from solver_controlled_fold import ControlledFoldActuation
from solver_fold_control_schedule import FoldControlSchedule
from solver_global_sewing import GlobalSewingSolver
from solver_hinge_sweep import hinge_sweep_safe
from solver_membrane_hessian import membrane_element_derivatives
from solver_rest_filtered_contact import RestFilteredSurfaceContact
from solver_sewing_activation_schedule import SewingActivationSchedule
from solver_triangle_sweep import triangle_sweep_safe


ROW_IDS=["synthetic:held:2-6","synthetic:pending:3-7"]


def fixture():
    """Independent two-layer diamond construction, with two complete seam rows."""
    builder=newton.ModelBuilder(gravity=(0,0,0))
    base=np.array([[.005,.01,0.],[.005,-.01,0.],[0.,0.,0.],[.02,0.,0.]])
    for height in (0.,.002):
        builder.add_cloth_mesh(pos=wp.vec3(0,0,0),rot=wp.quat_identity(),scale=1,
            vel=wp.vec3(0,0,0),vertices=(base+[0.,0.,height]).tolist(),indices=[0,2,3,1,3,2],
            density=.2,tri_ke=10000,tri_ka=10000,tri_kd=0,edge_ke=.001,edge_kd=0)
    builder.set_coloring([[index] for index in range(8)])
    model=builder.finalize(device="cpu")
    q=model.particle_q.numpy().astype(np.float64)
    native=model.edge_indices.numpy();hinges=native[np.all(native>=0,axis=1)]
    fold=ControlledFoldActuation(model,hinges,[.02,.02])
    ipctk.set_num_threads(1)
    contact=RestFilteredSurfaceContact(q,model.tri_indices.numpy(),activation_distance_m=.001,
        minimum_distance_m=.0001,stiffness=10000,ccd_profile="temporal-separation-tight-inclusion")
    solver=GlobalSewingSolver(model,[{2:1.,6:-1.},{3:1.,7:-1.}],1e-8,sewing_mode="distance",
        controlled_fold_actuation=fold,contact=contact)
    targets=np.array([np.linalg.norm(q[2]-q[6]),.003])
    return model,solver,q,targets


def schedules(hinges,angle=.03):
    sewing={"profile":"sewing-row-activation-v1","rowIds":ROW_IDS.copy(),"knots":[
        {"fraction":fraction,"activation":[1.,0.]} for fraction in (0.,1.)]}
    fold={"profile":"fold-angle-activation-v1","hinges":hinges.tolist(),"knots":[
        {"fraction":fraction,"targetsRadians":targets,"activation":activation} for fraction,targets,activation in (
            (0.,[0.,0.],[0.,0.]),(.125,[0.,0.],[1.,1.]),(.375,[angle,0.],[1.,1.]),
            (.625,[angle,-angle],[1.,1.]),(.75,[angle,-angle],[1.,1.]),
            (.875,[angle,-angle],[0.,0.]),(1.,[angle,-angle],[0.,0.]))]}
    return sewing,fold


def sewing_gradient(q,target,compliance):
    """Analytic force on the single active vertex pair; pending row absent."""
    vector=q[2]-q[6];length=np.linalg.norm(vector)
    value=vector*((1-target/length)/compliance)
    result=np.zeros_like(q);result[2]=value;result[6]=-value
    return result


def residual_parts(solver,previous,velocity,q,dt,targets,fold_targets,fold_activation):
    _,elements,_=membrane_element_derivatives(q[solver.faces],solver.poses,solver.areas,solver.materials[:,:3])
    membrane=np.zeros_like(q);np.add.at(membrane,solver.faces,elements.reshape(-1,3,3))
    return {"inertia":solver.mass[:,None]*(q-previous-dt*velocity)/dt**2,
        "membrane":membrane,"passiveBending":solver.bending.gradient(q),
        "sewing":sewing_gradient(q,targets[0],solver.compliance),
        "fold":solver.controlled_fold_actuation.potential(fold_targets,fold_activation).gradient(q),
        "contact":solver.contact.gradient(q)}


class MixedControlIntegrationTests(unittest.TestCase):
    def assert_transition(self,solver,previous,velocity,q,v,dt,targets,fold_targets,fold_activation,report):
        self.assertTrue(report["converged"],report)
        self.assertIs(report["accepted"],False)
        parts=residual_parts(solver,previous,velocity,q,dt,targets,fold_targets,fold_activation)
        total=sum(parts.values())
        self.assertLessEqual(np.max(np.abs(total)),1e-6)
        self.assertAlmostEqual(np.max(np.abs(total)),report["gradientInfinityNorm"],delta=2e-11)
        np.testing.assert_array_equal(v,(q-previous)/dt)
        potential=solver.sewing_potential(targets,activation=[1.,0.])
        np.testing.assert_allclose(potential.gradient(q).reshape(q.shape),parts["sewing"],atol=0,rtol=0)
        self.assertEqual(potential.residual(q)[1],0.)
        self.assertEqual(potential.jacobian(q).getrow(1).nnz,0)
        self.assertGreater(abs(np.linalg.norm(q[3]-q[7])-targets[1]),1e-5)
        self.assertEqual(report["sewingActivation"],[1.,0.])
        self.assertEqual(report["activeSewingRows"],[0]);self.assertEqual(report["pendingSewingRows"],[1])
        self.assertIsNone(report["sewingRowTargetErrorsM"][1])
        self.assertEqual(report["foldHingeSweepPolicy"],CONTROLLED_FOLD_SWEEP_POLICY)
        self.assertTrue(hinge_sweep_safe(previous,q,solver.controlled_fold_actuation.hinges))
        self.assertTrue(triangle_sweep_safe(previous,q,solver.faces))
        self.assertTrue(solver.contact.path_safe(previous,q))
        return parts

    def test_direct_coupled_force_balance_has_nonzero_sewing_and_inert_pending_row(self):
        _,solver,q,targets=fixture();velocity=np.zeros_like(q)
        final,v,report=solver.step(q,velocity,targets,.002,fold_targets=[.03,-.03],
            fold_activation=[1.,1.],sewing_activation=[1.,0.])
        parts=self.assert_transition(solver,q,velocity,final,v,.002,targets,[.03,-.03],[1.,1.],report)
        self.assertGreater(np.max(np.abs(parts["sewing"])),1e-8)
        self.assertGreater(np.max(np.abs(parts["fold"])),1e-4)
        altered=targets.copy();altered[1]=.1
        second,second_v,second_report=solver.step(q,velocity,altered,.002,fold_targets=[.03,-.03],
            fold_activation=[1.,1.],sewing_activation=[1.,0.])
        np.testing.assert_array_equal(second,final);np.testing.assert_array_equal(second_v,v)
        self.assertEqual(second_report["sewingJoules"],report["sewingJoules"])

    def test_initial_approach_loads_contact_sewing_and_fold_in_same_real_solve(self):
        _,solver,q,targets=fixture()
        # Explicit initial momentum, not a fixed support or hidden external
        # force: the upper layer approaches while the held sample reacts.
        velocity=np.zeros_like(q);velocity[4:,2]=-.5
        final,v,report=solver.step(q,velocity,targets,.002,fold_targets=[.3,-.3],
            fold_activation=[1.,1.],sewing_activation=[1.,0.])
        parts=self.assert_transition(solver,q,velocity,final,v,.002,targets,[.3,-.3],[1.,1.],report)
        for field in ('contact','sewing','fold','membrane','passiveBending'):
            self.assertGreater(np.max(np.abs(parts[field])),1e-8,field)
        self.assertGreater(report['contactJoules'],0.)
        self.assertGreater(report['sewingJoules'],0.)
        self.assertGreater(report['foldActuationJoules'],0.)
        np.testing.assert_allclose(solver.mass@(v-velocity),0.,atol=2e-10,rtol=0)

    def test_actual_adaptive_controls_release_fold_but_retain_sewing_and_rest_identity(self):
        model,solver,q,targets=fixture();sewing,fold=schedules(solver.controlled_fold_actuation.hinges)
        frozen={"modelRest":model.particle_q.numpy().copy(),"poses":solver.poses.copy(),
            "faces":solver.faces.copy(),"rows":solver.sewing.toarray().copy(),
            "bendRest":solver.bending.rest_angles.copy(),"foldHinges":solver.controlled_fold_actuation.hinges.copy()}
        frames=[]
        with patch('solver_hinge_sweep.hinge_sweep_safe',wraps=hinge_sweep_safe) as hinge_guard, \
             patch('solver_triangle_sweep.triangle_sweep_safe',wraps=triangle_sweep_safe) as triangle_guard, \
             patch.object(solver.contact,'path_safe',wraps=solver.contact.path_safe) as contact_guard:
            final,velocity,report=adaptive_contact_step(solver,q,np.zeros_like(q),targets,targets,.016,
                initial_subdivisions=8,sewing_schedule=sewing,sewing_row_ids=ROW_IDS,fold_control_schedule=fold,
                on_accept=lambda a,b,r:frames.append((a,b,r)))
            self.assertTrue(report["complete"],report)
            self.assertEqual(len(frames),8);self.assertEqual(report["rejectedSteps"],[])
            self.assertGreaterEqual(hinge_guard.call_count,16)
            self.assertGreaterEqual(triangle_guard.call_count,8)
            self.assertGreaterEqual(contact_guard.call_count,8)
            for call in hinge_guard.call_args_list:
                np.testing.assert_array_equal(call.args[2],frozen["foldHinges"])
            for call in triangle_guard.call_args_list:
                np.testing.assert_array_equal(call.args[2],frozen["faces"])
        controls=FoldControlSchedule(fold,8,hinges=frozen["foldHinges"])
        previous,old_velocity=q,np.zeros_like(q);sewing_forces=[]
        for positions,v,record in frames:
            ft,fa=controls.parameters(Fraction(record["endFraction"]))
            parts=self.assert_transition(solver,previous,old_velocity,positions,v,record["durationSeconds"],targets,ft,fa,record["step"])
            sewing_forces.append(np.max(np.abs(parts["sewing"])))
            energy=record["step"]["energyBalance"]
            self.assertEqual(energy["sewingParameterWorkJoules"],0.)
            self.assertEqual(energy["sewingTargetParameterWorkJoules"],0.)
            self.assertEqual(energy["sewingActivationParameterWorkJoules"],0.)
            self.assertTrue(np.isfinite(energy["foldParameterWorkJoules"]))
            previous,old_velocity=positions,v
        self.assertGreater(max(sewing_forces),1e-8)
        last=frames[-1][2]["step"]
        self.assertEqual(last["foldAnglesRadians"],[None,None]);self.assertEqual(last["foldActuationJoules"],0.)
        self.assertEqual(last["sewingActivation"],[1.,0.]);self.assertGreater(last["sewingJoules"],0.)
        self.assertGreater(np.max(np.abs(sewing_gradient(final,targets[0],solver.compliance))),1e-8)
        self.assertGreater(np.max(np.abs(velocity)),0.)
        for name,value in (("modelRest",model.particle_q.numpy()),("poses",solver.poses),("faces",solver.faces),
                           ("rows",solver.sewing.toarray()),("bendRest",solver.bending.rest_angles),
                           ("foldHinges",solver.controlled_fold_actuation.hinges)):
            np.testing.assert_array_equal(value,frozen[name])
        np.testing.assert_array_equal(final,frames[-1][0]);np.testing.assert_array_equal(velocity,frames[-1][1])

    def test_rejected_mutating_trial_resamples_both_schedules_before_real_retry(self):
        _,solver,q,targets=fixture();sewing,fold=schedules(solver.controlled_fold_actuation.hinges)
        real_step=solver.step;calls=[];frames=[];events=[]
        fold_oracle=FoldControlSchedule(fold,8,hinges=solver.controlled_fold_actuation.hinges)
        sewing_oracle=SewingActivationSchedule(sewing,8,row_ids=ROW_IDS)
        def step(positions,velocity,target,duration,**options):
            calls.append((positions.copy(),velocity.copy(),target.copy(),duration,
                options['fold_targets'].copy(),options['fold_activation'].copy(),options['sewing_activation'].copy()))
            result=real_step(positions,velocity,target,duration,**options)
            if len(calls)==2:
                # This trial genuinely solved, then its convergence claim is
                # withheld and its caller-owned inputs deliberately corrupted.
                positions[:]=99.;velocity[:]=88.;target[:]=77.
                options['fold_targets'][:]=66.;options['fold_activation'][:]=55.;options['sewing_activation'][:]=44.
                result=(positions,velocity,dict(result[2],converged=False))
            return result
        import solver_energy_balance
        real_work=solver_energy_balance.global_energy_transition
        def work(*args,**kwargs):
            events.append({key:value.copy() for key,value in kwargs.items()})
            return real_work(*args,**kwargs)
        with patch.object(solver,'step',side_effect=step),patch.object(solver_energy_balance,'global_energy_transition',side_effect=work):
            final,velocity,report=adaptive_contact_step(solver,q,np.zeros_like(q),targets,targets,.016,
                initial_subdivisions=8,sewing_schedule=sewing,sewing_row_ids=ROW_IDS,fold_control_schedule=fold,
                on_accept=lambda a,b,r:frames.append((a,b,r)))
        self.assertTrue(report['complete'],report)
        self.assertEqual(len(report['rejectedSteps']),1)
        self.assertEqual(len(report['acceptedSteps']),9)
        self.assertEqual(len(events),9)
        rejected=report['rejectedSteps'][0]
        self.assertEqual((rejected['startFraction'],rejected['endFraction']),(.125,.25))
        self.assertNotIn('energyBalance',rejected['step'])
        prior={0.:(q,np.zeros_like(q))};prior.update({r['endFraction']:(a,b) for a,b,r in frames})
        for record,call in zip(report['attempts'],calls):
            np.testing.assert_array_equal(call[0],prior[record['startFraction']][0])
            np.testing.assert_array_equal(call[1],prior[record['startFraction']][1])
            np.testing.assert_array_equal(call[2],targets)
            ft,fa=fold_oracle.parameters(Fraction(record['endFraction']))
            np.testing.assert_array_equal(call[4],ft);np.testing.assert_array_equal(call[5],fa)
            np.testing.assert_array_equal(call[6],sewing_oracle.parameters(Fraction(record['endFraction'])))
        for record,work in zip(report['acceptedSteps'],events):
            old=fold_oracle.parameters(Fraction(record['startFraction']));new=fold_oracle.parameters(Fraction(record['endFraction']))
            for key,value in zip(('previous_fold_targets','previous_fold_activation','fold_targets','fold_activation'),(*old,*new)):
                np.testing.assert_array_equal(work[key],value)
            np.testing.assert_array_equal(work['previous_sewing_activation'],[1.,0.])
            np.testing.assert_array_equal(work['sewing_activation'],[1.,0.])
        np.testing.assert_array_equal(final,frames[-1][0]);np.testing.assert_array_equal(velocity,frames[-1][1])

    def test_mixed_released_fold_cannot_skip_complete_geometry_or_contact_guards(self):
        _,solver,q,targets=fixture()
        cases=[('solver_hinge_sweep.hinge_sweep_safe','Initial controlled fold'),
               ('solver_triangle_sweep.triangle_sweep_safe','Physical cloth step')]
        for function,message in cases:
            with self.subTest(function=function),patch(function,return_value=False),self.assertRaisesRegex(ValueError,message):
                solver.step(q,np.zeros_like(q),targets,.002,fold_targets=[.03,-.03],fold_activation=[0.,0.],sewing_activation=[1.,0.])
        with patch.object(solver.contact,'path_safe',return_value=False),self.assertRaisesRegex(ValueError,'Physical contact step'):
            solver.step(q,np.zeros_like(q),targets,.002,fold_targets=[.03,-.03],fold_activation=[0.,0.],sewing_activation=[1.,0.])

    def test_post_work_sewing_tampering_rejects_before_second_state_publication(self):
        import solver_energy_balance
        real_work=solver_energy_balance.global_energy_transition
        for mode in ('activation','rows','explicit','missing-work','boolean-work'):
            _,solver,q,targets=fixture();sewing,fold=schedules(solver.controlled_fold_actuation.hinges)
            original_step=solver.step;raw_reports=[];frames=[];work_calls=[]
            def step(*args,**kwargs):
                result=original_step(*args,**kwargs);raw_reports.append(result[2]);return result
            def work(*args,**kwargs):
                result=real_work(*args,**kwargs);work_calls.append(1)
                if len(work_calls)==2:
                    if mode=='activation':raw_reports[-1]['sewingActivation']=[0.,0.]
                    if mode=='rows':raw_reports[-1]['activeSewingRows']=[];raw_reports[-1]['pendingSewingRows']=[0,1]
                    if mode=='explicit':raw_reports[-1]['sewingActivationExplicit']=False
                    if mode=='missing-work':result.pop('sewingParameterWorkJoules')
                    if mode=='boolean-work':result['sewingParameterWorkJoules']=False
                return result
            with self.subTest(mode=mode),patch.object(solver,'step',side_effect=step), \
                 patch.object(solver_energy_balance,'global_energy_transition',side_effect=work):
                final,velocity,report=adaptive_contact_step(solver,q,np.zeros_like(q),targets,targets,.016,
                    initial_subdivisions=8,max_depth=0,sewing_schedule=sewing,sewing_row_ids=ROW_IDS,
                    fold_control_schedule=fold,on_accept=lambda a,b,r:frames.append((a,b,r)))
                self.assertFalse(report['complete'])
                self.assertEqual(len(frames),1);self.assertEqual(len(report['acceptedSteps']),1)
                self.assertEqual(len(report['rejectedSteps']),1);self.assertEqual(len(work_calls),2)
                np.testing.assert_array_equal(final,frames[0][0]);np.testing.assert_array_equal(velocity,frames[0][1])


if __name__=="__main__":
    unittest.main()
