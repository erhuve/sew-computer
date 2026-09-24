"""Synthetic adaptive orchestration tests, not physical convergence evidence.

The report fixture deliberately supplies convergence claims while exercising
control/state/work publication rules. Its fold diagnostics and work are real
primitive evaluations; the final test also exercises actual synthetic global
mechanics. None of these fixtures admits a garment construction or capture.
"""

import copy
from fractions import Fraction
import unittest
from unittest import mock

import numpy as np
from scipy.sparse import csr_matrix

from solver_adaptive_contact import adaptive_contact_step, CONTROLLED_FOLD_SWEEP_POLICY
from solver_bending import ElasticDihedralBending
from solver_controlled_fold import ControlledFoldActuation
from solver_fold_control_schedule import FoldControlSchedule
import solver_energy_balance
from test_solver_controlled_fold import fixture


def schedule(hinges):
    return {"profile": "fold-angle-activation-v1", "hinges": hinges.tolist(), "knots": [
        {"fraction": f, "targetsRadians": t, "activation": a} for f, t, a in (
            (0., [0., -0.], [0., 1.]), (.25, [.1, -0.], [1., .5]),
            (.5, [-.1, .2], [1., 0.]), (.75, [.2, -.1], [0., .5]), (1., [0., 0.], [0., 0.]))]}


class ReportFixture:
    """An explicitly mocked step reporter; it does not solve cloth dynamics."""
    def __init__(self, model, hinges):
        self.controlled_fold_actuation = ControlledFoldActuation(model, hinges, [.01, .02])
        self.fold_actuation = self.material_grippers = None
        self.sewing_mode = "vector"
        self.mass = np.ones(len(model.particle_mass.numpy()))
        self.active = np.ones(len(self.mass), dtype=bool)
        self.sewing, self.compliance = csr_matrix((0, len(self.mass))), .04
        self.poses, self.faces = np.empty((0, 2, 2)), np.empty((0, 3), dtype=int)
        self.areas, self.materials = np.empty(0), np.empty((0, 3))
        self.bending = ElasticDihedralBending(len(self.mass), np.empty((0, 4), dtype=int), [], [], [])
        self.calls, self.reject_above, self.mutate, self.interrupt_at = [], None, None, None

    def step(self, positions, velocities, targets, duration, *, fold_targets, fold_activation, **options):
        self.calls.append((positions.copy(), velocities.copy(), targets.copy(), duration,
                           fold_targets.copy(), fold_activation.copy()))
        if self.interrupt_at == len(self.calls):
            positions[:], velocities[:], fold_targets[:], fold_activation[:] = 99, 88, 77, 66
            raise KeyboardInterrupt("synthetic controlled-fold interruption")
        if self.reject_above is not None and duration > self.reject_above:
            positions[:], velocities[:], fold_targets[:], fold_activation[:] = 99, 88, 77, 66
            return positions, velocities, {"converged": False, "gradientInfinityNorm": 1.}
        candidate = positions + duration*velocities
        potential = self.controlled_fold_actuation.potential(fold_targets, fold_activation)
        diagnostic = potential.diagnostics(candidate)
        report = {"converged": True, "gradientInfinityNorm": 0., "foldTargetsRadians": fold_targets.tolist(),
            "foldAnglesRadians": diagnostic["sampledActiveAnglesRadians"], "foldControls": diagnostic,
            "foldHingeSweepPolicy": CONTROLLED_FOLD_SWEEP_POLICY, "foldActuation": True,
            "foldActuationJoules": diagnostic["energyJoules"]}
        self.last_report = report
        if self.mutate is not None:
            self.mutate(positions, velocities, targets, fold_targets, fold_activation, report)
        return candidate, velocities.copy(), report


class Journal:
    def __init__(self, testcase, events):
        self.testcase, self.events, self.outcomes = testcase, events, []

    def start(self, record):
        self.events.append(("start", record["attemptId"]))

    def outcome(self, record, positions=None, velocities=None):
        if record["outcome"] == "accepted":
            self.testcase.assertEqual(self.events[-1][0], "energy")
            self.testcase.assertIn("foldParameterWorkJoules", record["step"]["energyBalance"])
            self.testcase.assertIsNotNone(positions)
        else:
            self.testcase.assertIsNone(positions)
            self.testcase.assertIsNone(velocities)
        self.outcomes.append(copy.deepcopy(record))
        self.events.append(("outcome", record["attemptId"]))

    def finish(self, reason, complete):
        self.events.append(("finish", complete))


class ControlledFoldAdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.positions, cls.hinges = fixture()

    def setup(self):
        return ReportFixture(self.model, self.hinges), self.positions.copy(), np.full_like(self.positions, .002)

    def run_control(self, solver, positions, velocities, **options):
        return adaptive_contact_step(solver, positions, velocities, np.empty((0, 3)), np.empty((0, 3)), .4,
            initial_subdivisions=4, fold_control_schedule=schedule(self.hinges), **options)

    def test_fold_only_work_precedes_journal_and_callback_through_retries(self):
        solver, positions, velocities = self.setup()
        solver.reject_above = .05
        events, captures = [], {}
        journal = Journal(self, events)
        real_energy = solver_energy_balance.global_energy_transition
        energy_calls = []
        def energy(*args, **kwargs):
            events.append(("energy", len(energy_calls)))
            energy_calls.append({key: value.copy() for key,value in kwargs.items()})
            return real_energy(*args, **kwargs)
        def accept(q, v, record):
            self.assertEqual(events[-1], ("outcome", record["attemptId"]))
            captures[record["endFraction"]] = (q.copy(),v.copy())
            q[:], v[:] = 333, 444
            record["step"]["energyBalance"]["foldParameterWorkJoules"] = 999
            events.append(("callback", record["attemptId"]))
        original_q, original_v = positions.copy(), velocities.copy()
        with mock.patch.object(solver_energy_balance,"global_energy_transition",side_effect=energy):
            final, velocity, report = self.run_control(solver,positions,velocities,attempt_journal=journal,on_accept=accept)
        self.assertTrue(report["complete"])
        self.assertIs(report["accepted"],False)
        self.assertEqual(len(report["acceptedSteps"]),8)
        self.assertEqual(len(report["rejectedSteps"]),4)
        self.assertEqual(len(energy_calls),8)
        oracle=FoldControlSchedule(schedule(self.hinges),4,hinges=self.hinges)
        for record,call in zip(report["attempts"],solver.calls):
            start=(original_q,original_v) if record["startFraction"]==0 else captures[record["startFraction"]]
            np.testing.assert_array_equal(call[0],start[0]);np.testing.assert_array_equal(call[1],start[1])
            expected=oracle.parameters(Fraction(record["endFraction"]))
            self.assertEqual(call[4].tobytes(),expected[0].tobytes())
            self.assertEqual(call[5].tobytes(),expected[1].tobytes())
            if record["outcome"]=="rejected":
                self.assertNotIn("energyBalance",record["step"])
        for record,call in zip(report["acceptedSteps"],energy_calls):
            old,new=oracle.parameters(Fraction(record["startFraction"])),oracle.parameters(Fraction(record["endFraction"]))
            for key,value in zip(("previous_fold_targets","previous_fold_activation","fold_targets","fold_activation"),(*old,*new)):
                self.assertEqual(call[key].tobytes(),value.tobytes())
            self.assertNotEqual(record["step"]["energyBalance"]["foldParameterWorkJoules"],999)
        np.testing.assert_array_equal(positions,original_q);np.testing.assert_array_equal(velocities,original_v)
        np.testing.assert_array_equal(final,captures[1.][0]);np.testing.assert_array_equal(velocity,captures[1.][1])
        self.assertEqual(report["acceptedSteps"][-1]["step"]["foldAnglesRadians"],[None,None])
        self.assertEqual(report["acceptedSteps"][-1]["step"]["foldControls"]["hinges"],self.hinges.tolist())

    def test_fresh_passed_control_mutations_reject_without_work_or_commit(self):
        attacks=[lambda q,v,t,f,a,r:f.__setitem__(0,.7), lambda q,v,t,f,a,r:a.__setitem__(0,0.),
                 lambda q,v,t,f,a,r:f.__setitem__(1,0.),
                 lambda q,v,t,f,a,r:setattr(f,"dtype",np.int64),
                 lambda q,v,t,f,a,r:t.resize((1,3),refcheck=False)]
        for mutate in attacks:
            solver,q,v=self.setup();solver.mutate=mutate
            with mock.patch.object(solver_energy_balance,"global_energy_transition") as energy:
                final,velocity,report=self.run_control(solver,q,v,max_depth=0)
            self.assertFalse(report["complete"]);self.assertEqual(report["acceptedSteps"],[])
            self.assertIn("mutated",report["rejectedSteps"][0]["error"]["message"])
            energy.assert_not_called()
            np.testing.assert_array_equal(final,q);np.testing.assert_array_equal(velocity,v)

    def test_forged_full_diagnostics_masks_identity_and_policy_reject(self):
        attacks=[lambda r:r.pop("foldControls"),lambda r:r.pop("foldAnglesRadians"),
            lambda r:r.__setitem__("foldActuation",False), lambda r:r.__setitem__("foldActuation",1),
            lambda r:r.__setitem__("foldActuationJoules",1.),
            lambda r:r.__setitem__("foldHingeSweepPolicy","active only"),
            lambda r:r["foldTargetsRadians"].__setitem__(0,.2),
            lambda r:r["foldAnglesRadians"].__setitem__(0,False),
            lambda r:r["foldControls"].__setitem__("accepted",0),
            lambda r:r["foldControls"]["activeHingeIndices"].__setitem__(0,False),
            lambda r:r["foldControls"]["activation"].__setitem__(0,True),
            lambda r:r["foldControls"]["hinges"][0].reverse(),
            lambda r:r["foldControls"]["coefficientWitnesses"][0].__setitem__("numericalCoefficientJoules",0.),
            lambda r:r["foldControls"].__setitem__("energyJoules",0.)]
        for index,attack in enumerate(attacks):
            solver,q,v=self.setup();solver.mutate=lambda q,v,t,f,a,r:attack(r)
            with self.subTest(attack=index),mock.patch.object(solver_energy_balance,"global_energy_transition") as energy:
                _,_,report=self.run_control(solver,q,v,max_depth=0)
                self.assertFalse(report["complete"]);self.assertEqual(report["acceptedSteps"],[])
                self.assertIn("diagnostics differ",report["rejectedSteps"][0]["error"]["message"])
                energy.assert_not_called()

    def test_failed_missing_nonfinite_boolean_or_mutating_work_preserves_prefix(self):
        real_energy=solver_energy_balance.global_energy_transition
        for mode in ("exception","none","missing","missing-global","nonfinite","boolean","global-boolean","accepted","mutation","mutation-raise",
                     "diagnostic-mutation","convergence-mutation","residual-mutation"):
            solver,q,v=self.setup();events=[];journal=Journal(self,events)
            def fail(*args,**kwargs):
                events.append(("energy",None))
                if mode=="exception":raise ValueError("synthetic invalid fold work")
                if mode=="none":return None
                if mode=="missing":return {"accepted":False}
                result=real_energy(*args,**kwargs)
                if mode=="missing-global":result.pop("externalParameterWorkJoules")
                if mode=="nonfinite":result["foldParameterWorkJoules"]=float("nan")
                if mode=="boolean":result["foldActivationParameterWorkJoules"]=False
                if mode=="global-boolean":result["mechanicalChangeMinusParameterWorkJoules"]=False
                if mode=="accepted":result["accepted"]=0
                if mode.startswith("mutation"):
                    args[1][:]=999;kwargs["fold_activation"][:]=0
                    if mode=="mutation-raise":raise ValueError("work mutated then failed")
                if mode=="diagnostic-mutation":solver.last_report["foldActuation"]=False
                if mode=="convergence-mutation":solver.last_report["converged"]=False
                if mode=="residual-mutation":solver.last_report["gradientInfinityNorm"]=1.
                return result
            with self.subTest(mode=mode),mock.patch.object(solver_energy_balance,"global_energy_transition",side_effect=fail):
                final,velocity,report=self.run_control(solver,q,v,max_depth=0,attempt_journal=journal)
            self.assertFalse(report["complete"]);self.assertEqual(report["acceptedSteps"],[])
            self.assertEqual(journal.outcomes[0]["outcome"],"rejected")
            np.testing.assert_array_equal(final,q);np.testing.assert_array_equal(velocity,v)

    def test_raw_schedule_mutation_does_not_change_later_original_fraction_controls(self):
        solver,q,v=self.setup();source=schedule(self.hinges);original=copy.deepcopy(source)
        def mutate(q,v,t,f,a,r):
            source["knots"][-1]["activation"][:]=[1.,1.]
            source["knots"][2]["targetsRadians"][:]=[2.,2.]
        solver.mutate=mutate
        _,_,report=adaptive_contact_step(solver,q,v,np.empty((0,3)),np.empty((0,3)),.4,
            initial_subdivisions=4,fold_control_schedule=source)
        self.assertTrue(report["complete"])
        oracle=FoldControlSchedule(original,4,hinges=self.hinges)
        for record,call in zip(report["attempts"],solver.calls):
            targets,activation=oracle.parameters(record["endFraction"])
            np.testing.assert_array_equal(call[4],targets);np.testing.assert_array_equal(call[5],activation)

    def test_interrupted_trial_keeps_only_accepted_journal_prefix(self):
        solver,q,v=self.setup();solver.interrupt_at=2
        events=[];journal=Journal(self,events);captures=[]
        real_energy=solver_energy_balance.global_energy_transition
        def energy(*args,**kwargs):events.append(("energy",None));return real_energy(*args,**kwargs)
        with mock.patch.object(solver_energy_balance,"global_energy_transition",side_effect=energy):
            with self.assertRaises(KeyboardInterrupt):
                self.run_control(solver,q,v,attempt_journal=journal,on_accept=lambda a,b,r:captures.append((a,b,r)))
        self.assertEqual([row["outcome"] for row in journal.outcomes],["accepted","interrupted"])
        self.assertEqual(len(captures),1)
        np.testing.assert_array_equal(captures[0][0],solver.calls[1][0])
        np.testing.assert_array_equal(captures[0][1],solver.calls[1][1])
        self.assertNotIn("energyBalance",journal.outcomes[1]["step"] if "step" in journal.outcomes[1] else {})

    def test_missing_ambiguous_legacy_raw_and_fraction_budget_configuration(self):
        solver,q,v=self.setup();empty=np.empty((0,3))
        configurations=[{}, {"fold_control_schedule":schedule(self.hinges),"initial_fold_targets":[0.,0.]},
            {"fold_control_schedule":schedule(self.hinges),"fold_targets":[0.,0.]},
            {"fold_control_schedule":schedule(self.hinges),"assembly_schedule":{}},
            {"fold_control_schedule":schedule(self.hinges),"fold_activation":[1.,1.]},
            {"fold_control_schedule":schedule(self.hinges),"linear_solver":"shifted"},
            {"fold_control_schedule":schedule(self.hinges),"linear_solver":"lsmr"},
            {"fold_control_schedule":schedule(self.hinges),"initial_subdivisions":4096,"max_attempts":4096,"max_depth":29}]
        for options in configurations:
            with self.subTest(options=options),self.assertRaises(ValueError):
                adaptive_contact_step(solver,q,v,empty,empty,.4,**options)
        self.assertEqual(solver.calls,[])
        solver.controlled_fold_actuation=None
        with self.assertRaises(ValueError):self.run_control(solver,q,v)

    def test_original_state_scalar_and_duration_admission_precedes_step(self):
        for where,value in (("q",True),("q",2**53+1),("v",False),("v",2**53+1),("dt",True),("dt",2**53+1)):
            solver,q,v=self.setup();q,v=q.tolist(),v.tolist();duration=.4
            if where=="q":q[0][0]=value
            if where=="v":v[0][0]=value
            if where=="dt":duration=value
            with self.subTest(where=where,value=value),self.assertRaises(ValueError):
                adaptive_contact_step(solver,q,v,np.empty((0,3)),np.empty((0,3)),duration,
                    initial_subdivisions=4,fold_control_schedule=schedule(self.hinges))
            self.assertEqual(solver.calls,[])

    def test_recipe_replacement_during_step_cannot_change_accepted_identity(self):
        solver,q,v=self.setup()
        replacement=ControlledFoldActuation(self.model,self.hinges,[.01,.02])
        solver.mutate=lambda q,v,t,f,a,r:setattr(solver,"controlled_fold_actuation",replacement)
        _,_,report=self.run_control(solver,q,v,max_depth=0)
        self.assertFalse(report["complete"])
        self.assertIn("identity changed",report["rejectedSteps"][0]["error"]["message"])
        solver.controlled_fold_actuation=object()
        with self.assertRaises(ValueError):self.run_control(solver,q,v)
        solver.controlled_fold_actuation=ControlledFoldActuation(self.model,self.hinges,[.01,.02])
        solver.fold_actuation=object()
        with self.assertRaises(ValueError):self.run_control(solver,q,v)

    def test_raw_candidate_state_scalar_admission_precedes_work(self):
        for index,value in ((0,True),(0,2**53+1),(1,False),(1,2**53+1)):
            solver,q,v=self.setup();real_step=solver.step
            def malformed(*args,**kwargs):
                result=list(real_step(*args,**kwargs))
                result[index]=result[index].tolist()
                result[index][0][0]=value
                return result
            solver.step=malformed
            with self.subTest(index=index,value=value),mock.patch.object(solver_energy_balance,"global_energy_transition") as work:
                final,velocity,report=self.run_control(solver,q,v,max_depth=0)
                self.assertFalse(report["complete"]);self.assertEqual(report["acceptedSteps"],[])
                work.assert_not_called()
                np.testing.assert_array_equal(final,q);np.testing.assert_array_equal(velocity,v)

    def test_actual_synthetic_contact_solver_ramp_release_and_work(self):
        from test_solver_controlled_fold_integration import fixture as global_fixture
        from solver_hinge_sweep import hinge_sweep_safe
        from solver_triangle_sweep import triangle_sweep_safe
        _,solver,q=global_fixture(contact_enabled=True)
        recipe=solver.controlled_fold_actuation
        source=schedule(recipe.hinges)
        for knot in source["knots"]:
            knot["targetsRadians"]=[value*.01 for value in knot["targetsRadians"]]
        saved=[]
        final,velocity,report=adaptive_contact_step(solver,q,np.zeros_like(q),np.empty((0,3)),np.empty((0,3)),.0008,
            initial_subdivisions=4,fold_control_schedule=source,on_accept=lambda a,b,r:saved.append((a,b,r)))
        self.assertTrue(report["complete"],report)
        self.assertIs(report["accepted"],False)
        self.assertEqual(len(saved),4)
        self.assertEqual(report["rejectedSteps"],[])
        self.assertGreater(np.max(np.abs(final-q)),0.)
        self.assertGreater(np.max(np.abs(velocity)),0.)
        oracle=FoldControlSchedule(source,4,hinges=recipe.hinges)
        previous=q
        for state,speed,record in saved:
            targets,activation=oracle.parameters(record["endFraction"])
            self.assertEqual(record["step"]["foldControls"],recipe.potential(targets,activation).diagnostics(state))
            self.assertLessEqual(record["step"]["gradientInfinityNorm"],1e-6)
            self.assertIn("foldParameterWorkJoules",record["step"]["energyBalance"])
            self.assertEqual(record["step"]["foldHingeSweepPolicy"],CONTROLLED_FOLD_SWEEP_POLICY)
            self.assertTrue(hinge_sweep_safe(previous,state,recipe.hinges))
            self.assertTrue(triangle_sweep_safe(previous,state,solver.faces))
            self.assertTrue(solver.contact.path_safe(previous,state))
            np.testing.assert_array_equal(speed,(state-previous)/record["durationSeconds"])
            previous=state
        self.assertEqual(saved[-1][2]["step"]["foldAnglesRadians"],[None,None])
        self.assertEqual(saved[-1][2]["step"]["foldActuationJoules"],0.)


if __name__=="__main__":
    unittest.main()
