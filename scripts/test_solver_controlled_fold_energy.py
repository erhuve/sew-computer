"""Synthetic controlled-fold accounting, with independent quadratic oracles.

These are energy-state tests, not cloth solves or refined-source trajectories.
"""

import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from solver_bending import ElasticDihedralBending
from solver_controlled_fold import ControlledFoldActuation
from solver_distance_sewing import DistanceSewing
from solver_energy_balance import global_energy_transition
from solver_fold_actuation import FoldActuation
from solver_material_grippers import MaterialGrippers
from solver_normal_sewing import NormalOffsetSewing
from test_solver_dihedral_increment import decimal_angle


def fixture(stiffness=(2.,4.), *, sewing=False, grippers=False, mode="vector"):
    q=np.array([[0.,1.,0.],[0.,-1.,0.],[0.,0.,0.],[1.,0.,0.],
                [3.,1.,0.],[3.,-1.,0.],[3.,0.,0.],[4.,0.,0.]])
    hinges=np.arange(8).reshape(2,4)
    def array(value): return SimpleNamespace(numpy=lambda:np.asarray(value))
    model=SimpleNamespace(edge_indices=array(hinges),particle_mass=array(np.ones(8)),particle_q=array(q))
    recipe=ControlledFoldActuation(model,hinges,stiffness)
    rows=csr_matrix([[1.,-1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,-1.,0.,0.]] if sewing else np.empty((0,8)))
    solver=SimpleNamespace(mass=np.ones(8),active=np.ones(8,dtype=bool),sewing=rows,compliance=.5,
        sewing_mode=mode,poses=np.empty((0,2,2)),faces=np.empty((0,3),dtype=int),areas=np.empty(0),
        materials=np.empty((0,3)),bending=ElasticDihedralBending(8,np.empty((0,4),dtype=int),[],[],[]),
        controlled_fold_actuation=recipe,fold_actuation=None,material_grippers=None)
    if mode=="distance":
        solver.sewing_potential=lambda target,*,activation=None:DistanceSewing(rows,target,.5,activation=activation)
    elif mode=="normal-offset":
        solver.sewing_frames=np.array([[1,3,2],[5,7,6]])
        solver.sewing_sides=np.ones(2,dtype=int)
        solver.sewing_potential=lambda target,*,activation=None:NormalOffsetSewing(
            rows,target,.5,solver.sewing_frames,solver.sewing_sides,activation=activation)
    if grippers:
        solver.material_grippers=MaterialGrippers(["first"]*4+["second"]*4,[[0,2,3],[4,6,7]],[
            {"id":"first-grip","instanceId":"first","triangleIndex":0,"weights":[1.,0.,0.],"stiffnessNPerM":8.},
            {"id":"second-grip","instanceId":"second","triangleIndex":1,"weights":[1.,0.,0.],"stiffnessNPerM":16.}])
    return solver,q,model


def report(solver,q,end=None,*,old_targets=(0.,0.),targets=(0.,0.),old_activation=(1.,1.),activation=(1.,1.),
           old_sewing=None,sewing=None,**kwargs):
    end=q if end is None else end
    if old_sewing is None:
        shape=(solver.sewing.shape[0],3) if solver.sewing_mode=="vector" else (solver.sewing.shape[0],)
        old_sewing=sewing=np.zeros(shape)
    return global_energy_transition(solver,q,end,np.zeros_like(q),np.asarray(end)-np.asarray(q),old_sewing,sewing,1.,
        previous_fold_targets=old_targets,fold_targets=targets,
        previous_fold_activation=old_activation,fold_activation=activation,**kwargs)


def quadratic(beta0,beta1,old_error,new_error):
    before=F(beta0)*sum((F(v)**2 for v in old_error),F())/2
    fixed=F(beta1)*sum((F(v)**2 for v in new_error),F())/2
    target=F(beta0)*sum((F(b)**2-F(a)**2 for a,b in zip(old_error,new_error)),F())/2
    activation=(F(beta1)-F(beta0))*sum((F(v)**2 for v in new_error),F())/2
    return {"before":before,"fixed":fixed,"target":target,"activation":activation,"total":fixed-before,
            "release":max(-activation,F()),"increase":max(activation,F())}


def fold_oracle(stiffness,old_targets,targets,old_activation,activation):
    # These fixture hinges are exactly flat. Coefficient products are rounded
    # once before evaluating an independent exact endpoint quadratic.
    result={key:F() for key in ("before","fixed","target","activation","total","release","increase")}
    for k,t0,t1,a0,a1 in zip(stiffness,old_targets,targets,old_activation,activation):
        beta0,beta1=float(F(k)*F(a0)),float(F(k)*F(a1))
        for key,value in quadratic(beta0,beta1,[-F(t0)],[-F(t1)]).items(): result[key]+=value
    return result


class ControlledFoldEnergyTests(unittest.TestCase):
    def test_target_first_release_and_engagement_have_separate_signed_work(self):
        solver,q,_=fixture()
        before,after,old,new=[0.,.5],[2.,-.25],[1.,.5],[0.,1.]
        actual=report(solver,q,old_targets=before,targets=after,old_activation=old,activation=new)
        expected=fold_oracle([2.,4.],before,after,old,new)
        for key,field in (("before","foldActuationBeforeJoules"),("fixed","foldFixedPositionAfterJoules"),
                          ("fixed","foldActuationAfterJoules"),("target","foldTargetParameterWorkJoules"),
                          ("activation","foldActivationParameterWorkJoules"),("total","foldParameterWorkJoules"),
                          ("total","foldActuationChangeJoules"),("release","foldReleaseEnergyRemovedJoules"),
                          ("increase","foldActivationIncreaseWorkJoules")):
            self.assertEqual(actual[field],float(expected[key]),field)
        self.assertEqual(actual["foldFixedParameterChangeJoules"],0.)
        self.assertEqual(actual["targetParameterWorkJoules"],float(expected["target"]))
        self.assertEqual(actual["externalParameterWorkJoules"],float(expected["total"]))
        self.assertEqual(actual["mechanicalChangeMinusTargetWorkJoules"],float(expected["activation"]))
        self.assertEqual(actual["mechanicalChangeMinusParameterWorkJoules"],0.)
        self.assertGreater(actual["foldReleaseEnergyRemovedJoules"],0.)
        self.assertGreater(actual["foldActivationIncreaseWorkJoules"],0.)
        self.assertIs(actual["accepted"],False)

    def test_direct_parameter_total_survives_rounded_component_cancellation(self):
        solver,q,_=fixture((1.,4.))
        active=math.nextafter(.25,math.inf)
        actual=report(solver,q,old_targets=[1.,0.],targets=[2.,0.],old_activation=[1.,0.],activation=[active,0.])
        with localcontext() as context:
            context.prec=180
            expected=float((4*Decimal.from_float(active)-1)/2)
        self.assertEqual(math.fsum((actual["foldTargetParameterWorkJoules"],actual["foldActivationParameterWorkJoules"])),0.)
        self.assertEqual(expected,2.**-53)
        for field in ("foldParameterWorkJoules","foldActuationChangeJoules","mechanicalChangeJoules","externalParameterWorkJoules"):
            self.assertEqual(actual[field],expected,field)
        self.assertLessEqual(expected,actual["foldParameterWorkComponentSumErrorBoundJoules"])
        self.assertEqual(actual["mechanicalChangeMinusParameterWorkJoules"],0.)

    def test_work_uses_rounded_coefficient_not_ideal_stiffness_activation_product(self):
        solver,q,_=fixture((1e12,4.));active=math.nextafter(.25,math.inf)
        actual=report(solver,q,old_targets=[1.,0.],targets=[2.,0.],old_activation=[1.,0.],activation=[active,0.])
        beta=F(float(F(1e12)*F(active)))
        expected=float((4*beta-F(1e12))/2)
        ideal=float(F(1e12)*(4*F(active)-1)/2)
        self.assertNotEqual(expected,ideal)
        self.assertEqual(actual["foldParameterWorkJoules"],expected)

    def test_inactive_geometry_skips_angle_but_reengagement_needs_previous_geometry(self):
        solver,q,_=fixture();collapsed=np.zeros_like(q)
        actual=report(solver,collapsed,old_targets=[3.,-3.],targets=[-3.,3.],old_activation=[0.,0.],activation=[0.,0.])
        for key,value in actual.items():
            if key.endswith("Joules") and "ErrorBound" not in key:self.assertEqual(value,0.,key)
        self.assertIn("without waiving",actual["scope"])
        with self.assertRaisesRegex(ValueError,"Degenerate"):
            report(solver,collapsed,old_activation=[0.,0.],activation=[1.,0.])
        released=report(solver,q,collapsed,old_targets=[1.,0.],targets=[2.,0.],old_activation=[1.,0.],activation=[0.,0.])
        self.assertEqual(released["foldActuationAfterJoules"],0.)
        self.assertEqual(released["foldFixedParameterChangeJoules"],0.)
        self.assertEqual(released["foldReleaseEnergyRemovedJoules"],4.)

    def test_sub_ulp_motion_retains_work_with_equal_endpoint_energy_diagnostics(self):
        solver,q,_=fixture((3.,4.))
        q[:4]=[[.2,.8,.3],[.6,-.7,-.1],[0.,0.,0.],[1.,0.,0.]]
        end=q.copy();end[0,1]=np.nextafter(.8,math.inf)
        actual=report(solver,q,end,old_targets=[.1,0.],targets=[.1,0.],old_activation=[1.,0.],activation=[1.,0.])
        with localcontext() as context:
            context.prec=160
            before=decimal_angle(q[:4])-Decimal.from_float(.1)
            after=decimal_angle(end[:4])-Decimal.from_float(.1)
            expected=float(Decimal('1.5')*(after**2-before**2))
        self.assertLess(expected,0.)
        self.assertEqual(actual["foldActuationBeforeJoules"],actual["foldActuationAfterJoules"])
        self.assertAlmostEqual(actual["foldFixedParameterChangeJoules"],expected,delta=1e-30)
        self.assertEqual(actual["foldParameterWorkJoules"],0.)
        self.assertAlmostEqual(actual["mechanicalChangeMinusParameterWorkJoules"],expected,delta=1e-30)

    def test_stationary_target_activation_cycle_telescopes_after_release(self):
        solver,q,_=fixture();targets=[0.,0.];activation=[0.,0.];records=[]
        for next_targets,next_activation in (([.5,-.25],[1.,.5]),([1.,-.5],[.5,1.]),([.25,.5],[0.,0.])):
            records.append(report(solver,q,old_targets=targets,targets=next_targets,
                                  old_activation=activation,activation=next_activation))
            targets,activation=next_targets,next_activation
        self.assertEqual(math.fsum(row["foldActuationChangeJoules"] for row in records),0.)
        self.assertEqual(math.fsum(row["externalParameterWorkJoules"] for row in records),0.)
        self.assertEqual(records[-1]["foldActuationAfterJoules"],0.)

    def test_sewing_and_gripper_combinations_use_total_vs_target_work(self):
        for mode in ("vector","distance","normal-offset"):
            for weighted in (False,True):
                with self.subTest(mode=mode,weighted=weighted):
                    solver,q,_=fixture(sewing=True,grippers=True,mode=mode)
                    # A common dyadic translation changes gripper and kinetic
                    # energy but preserves both signed angles and seam vectors.
                    end=q+[.125,0.,.0625]
                    if mode=="vector":
                        old_sew=np.zeros((2,3));new_sew=np.array([[.25,.5,.125],[-.125,.25,0.]])
                    else:
                        old_sew=np.array([1.,1.5]) if mode=="distance" else np.array([.25,.5])
                        new_sew=np.array([.5,1.25]) if mode=="distance" else np.array([.5,.125])
                    old_sa,new_sa=([.75,.5],[.25,1.]) if weighted else ([1.,1.],[1.,1.])
                    old_g=np.array([[0.,0.,0.],[3.,0.,0.]]);new_g=np.array([[.5,.125,.25],[3.25,-.25,.5]])
                    old_ga,new_ga=[.75,.5],[.25,1.]
                    old_ft,new_ft,old_fa,new_fa=[.5,-.25],[1.,.5],[.75,.25],[.25,1.]
                    extra={"previous_gripper_targets":old_g,"gripper_targets":new_g,
                           "previous_gripper_activation":old_ga,"gripper_activation":new_ga}
                    if weighted:extra.update(previous_sewing_activation=old_sa,sewing_activation=new_sa)
                    actual=report(solver,q,end,old_targets=old_ft,targets=new_ft,old_activation=old_fa,activation=new_fa,
                                  old_sewing=old_sew,sewing=new_sew,**extra)
                    fold=fold_oracle([2.,4.],old_ft,new_ft,old_fa,new_fa)
                    target,total,activation=fold["target"],fold["total"],fold["activation"]
                    motion=F()
                    for i in range(2):
                        if mode=="vector":
                            old_error=[F(v)-F(t) for v,t in zip([0.,2.,0.],old_sew[i])]
                            error=[F(v)-F(t) for v,t in zip([0.,2.,0.],new_sew[i])]
                        elif mode=="distance":old_error,error=[2-F(old_sew[i])],[2-F(new_sew[i])]
                        else:old_error,error=[F(),F(2),-F(old_sew[i])],[F(),F(2),-F(new_sew[i])]
                        seam=quadratic(F(old_sa[i])/F(.5),F(new_sa[i])/F(.5),old_error,error)
                        target+=seam["target"];total+=seam["total"];activation+=seam["activation"]
                        vertex=4*i;k=F([8.,16.][i])
                        old_error=[F(x)-F(t) for x,t in zip(q[vertex],old_g[i])]
                        fixed_error=[F(x)-F(t) for x,t in zip(q[vertex],new_g[i])]
                        end_error=[F(x)-F(t) for x,t in zip(end[vertex],new_g[i])]
                        grip=quadratic(k*F(old_ga[i]),k*F(new_ga[i]),old_error,fixed_error)
                        target+=grip["target"];total+=grip["total"];activation+=grip["activation"]
                        motion+=k*F(new_ga[i])*sum((b*b-a*a for a,b in zip(fixed_error,end_error)),F())/2
                    kinetic=sum((F(v)**2 for row in end-q for v in row),F())/2
                    motion+=kinetic
                    for key,expected in (("targetParameterWorkJoules",target),("externalParameterWorkJoules",total),
                                         ("mechanicalChangeJoules",total+motion),
                                         ("mechanicalChangeMinusParameterWorkJoules",motion),
                                         ("mechanicalChangeMinusTargetWorkJoules",motion+activation)):
                        self.assertAlmostEqual(actual[key],float(expected),delta=3e-14,msg=key)

    def test_argument_pairing_mutual_exclusion_and_raw_coordinate_admission(self):
        solver,q,model=fixture()
        values={"previous_fold_targets":[0.,0.],"fold_targets":[0.,0.],
                "previous_fold_activation":[1.,1.],"fold_activation":[1.,1.]}
        for key in values:
            incomplete=dict(values);incomplete[key]=None
            with self.subTest(missing=key),self.assertRaises(ValueError):
                global_energy_transition(solver,q,q,np.zeros_like(q),np.zeros_like(q),np.empty((0,3)),np.empty((0,3)),1.,**incomplete)
        solver.fold_actuation=FoldActuation(model,np.arange(8).reshape(2,4),[2.,4.])
        with self.assertRaises(ValueError):report(solver,q)
        solver.controlled_fold_actuation=None
        with self.assertRaises(ValueError):report(solver,q)
        solver.fold_actuation=None
        with self.assertRaises(ValueError):report(solver,q)
        solver.controlled_fold_actuation=object()
        with self.assertRaises(ValueError):report(solver,q)
        solver,_,_=fixture()
        for value in (True,2**53+1):
            raw=q.tolist();raw[0][0]=value
            with self.subTest(raw=value),self.assertRaises(ValueError):
                report(solver,raw,old_activation=[0.,0.],activation=[0.,0.])

    def test_raw_velocity_and_timestep_admission_precedes_global_coercion(self):
        solver,q,_=fixture()
        options={"previous_fold_targets":[0.,0.],"fold_targets":[0.,0.],
                 "previous_fold_activation":[0.,0.],"fold_activation":[0.,0.]}
        for value in (True,2**53+1):
            velocity=np.zeros_like(q).tolist();velocity[0][0]=value
            with self.subTest(previousVelocity=value),self.assertRaises(ValueError):
                global_energy_transition(solver,q,q,velocity,np.zeros_like(q),np.empty((0,3)),np.empty((0,3)),1.,**options)
            with self.subTest(dt=value),self.assertRaises(ValueError):
                global_energy_transition(solver,q,q,np.zeros_like(q),np.zeros_like(q),np.empty((0,3)),np.empty((0,3)),value,**options)

    def test_work_policy_missing_nonfinite_and_boolean_values_fail_closed(self):
        solver,q,_=fixture();original=solver.controlled_fold_actuation.parameter_energy_change(q,[0.,0.],[1.,1.],[.5,.5],[.5,.5])
        changes=[lambda x:x.pop("totalParameterWorkJoules"),lambda x:x.__setitem__("activationParameterWorkJoules",True),
                 lambda x:x.__setitem__("totalParameterWorkJoules",float("nan")),
                 lambda x:x.__setitem__("releaseEnergyRemovedJoules",-1.),
                 lambda x:x.__setitem__("parameterOrder","activation-first"),
                 lambda x:x.__setitem__("coefficientPolicy","unrounded-product")]
        for change in changes:
            malformed=copy.deepcopy(original);change(malformed)
            with patch.object(ControlledFoldActuation,"parameter_energy_change",return_value=malformed):
                with self.assertRaises(ValueError):report(solver,q,targets=[.5,.5],activation=[.5,.5])


if __name__=="__main__":unittest.main()
