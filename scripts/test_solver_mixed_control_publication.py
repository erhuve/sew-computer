"""Mixed-control publication attacks using explicitly mocked solver reports.

Primitive work is real; convergence is deliberately supplied by this fixture.
These tests establish orchestration/rollback only, never physical equilibrium,
captured admission, source construction or garment motion.
"""

import copy
import unittest
from unittest import mock

import numpy as np
from scipy.sparse import csr_matrix, kron

from solver_adaptive_contact import adaptive_contact_step
from solver_global_sewing import GlobalSewingSolver
import solver_energy_balance
from test_solver_controlled_fold_adaptive import ReportFixture, schedule
from test_solver_controlled_fold import fixture


SEWING_FIELDS = ("sewingBeforeJoules", "sewingAfterJoules", "sewingFixedPositionAfterJoules",
    "sewingFixedParameterChangeJoules", "sewingChangeJoules", "sewingTargetParameterWorkJoules",
    "sewingActivationParameterWorkJoules", "sewingParameterWorkJoules",
    "sewingActivationIncreaseWorkJoules", "sewingReleaseEnergyRemovedJoules",
    "sewingParameterWorkComponentSumErrorBoundJoules")
NONNEGATIVE_SEWING_FIELDS = ("sewingBeforeJoules", "sewingAfterJoules", "sewingFixedPositionAfterJoules",
    "sewingActivationIncreaseWorkJoules", "sewingReleaseEnergyRemovedJoules",
    "sewingParameterWorkComponentSumErrorBoundJoules")


class MixedReportFixture(ReportFixture):
    """No solve: keep actual primitive controls/work and forge convergence."""
    sewing_potential = GlobalSewingSolver.sewing_potential

    def __init__(self, model, hinges, mode="vector"):
        super().__init__(model, hinges)
        self.sewing = csr_matrix(([1., -1., 1., -1.], ([0, 0, 1, 1], [0, 4, 1, 5])),
                                 shape=(2, len(self.mass)))
        self.sewing_xyz = kron(self.sewing, np.eye(3), format="csr")
        self.sewing_mode = mode
        self.sewing_frame_faces = np.array([[4, 6, 7], [5, 7, 6]]) if mode == "normal-offset" else None
        self.sewing_sides = np.ones(2,dtype=int) if mode == "normal-offset" else None
        self.after_step = None

    def step(self, *args, **kwargs):
        positions, velocities, report = super().step(*args, **kwargs)
        weights = kwargs["sewing_activation"]
        report.update(sewingActivationExplicit=True, sewingActivation=weights.tolist(),
                      activeSewingRows=np.flatnonzero(weights > 0).tolist(),
                      pendingSewingRows=np.flatnonzero(weights == 0).tolist())
        self.last_report = report
        if self.after_step is not None and len(self.calls) == 2:
            self.after_step(self, args, kwargs, report)
        return positions, velocities, report


class RecordingJournal:
    def __init__(self, events):
        self.events, self.outcomes = events, []

    def start(self, row):
        self.events.append(("start", row["attemptId"]))

    def outcome(self, row, positions=None, velocities=None):
        self.outcomes.append((copy.deepcopy(row), None if positions is None else positions.copy(),
                              None if velocities is None else velocities.copy()))
        self.events.append(("outcome", row["outcome"]))

    def finish(self, reason, complete):
        self.events.append(("finish", complete))


class MixedControlPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.positions, cls.hinges = fixture()

    def run_case(self, *, mode="vector", work_attack=None, step_attack=None):
        solver = MixedReportFixture(self.model, self.hinges, mode)
        solver.after_step = step_attack
        q, v = self.positions.copy(), np.full_like(self.positions, .002)
        original_q, original_v = q.copy(), v.copy()
        target = solver.sewing @ q if mode == "vector" else np.full(2, .001)
        events, callbacks, work_calls = [], [], []
        journal = RecordingJournal(events)
        real_work = solver_energy_balance.global_energy_transition
        def work(*args, **kwargs):
            result = real_work(*args, **kwargs)
            work_calls.append({key:value.copy() for key,value in kwargs.items()})
            events.append(("work", len(work_calls)))
            if len(work_calls) == 2 and work_attack is not None:
                work_attack(solver, args, kwargs, result)
            return result
        def on_accept(q, v, row):
            self.assertEqual(events[-1], ("outcome", "accepted"))
            callbacks.append((q.copy(), v.copy(), copy.deepcopy(row)))
            events.append(("callback", row["attemptId"]))
        sewing = {"profile":"sewing-row-activation-v1", "rowIds":["held", "pending"],
                  "knots":[{"fraction":fraction, "activation":[1., 0.]} for fraction in (0., 1.)]}
        with mock.patch.object(solver_energy_balance, "global_energy_transition", side_effect=work):
            final, velocity, report = adaptive_contact_step(solver, q, v, target, target, .4,
                initial_subdivisions=4, max_depth=0, fold_control_schedule=schedule(self.hinges),
                sewing_schedule=sewing, sewing_row_ids=sewing["rowIds"], attempt_journal=journal, on_accept=on_accept)
        np.testing.assert_array_equal(q, original_q)
        np.testing.assert_array_equal(v, original_v)
        return final, velocity, report, journal, callbacks, work_calls, events

    def assert_prefix_only(self, result):
        final, velocity, report, journal, callbacks, _, _ = result
        self.assertFalse(report["complete"])
        self.assertEqual(report["completedFraction"], .25)
        self.assertEqual(len(report["acceptedSteps"]), 1)
        self.assertEqual(len(report["rejectedSteps"]), 1)
        self.assertEqual(len(callbacks), 1)
        self.assertEqual([row[0]["outcome"] for row in journal.outcomes], ["accepted", "rejected"])
        self.assertIsNotNone(journal.outcomes[0][1])
        self.assertIsNone(journal.outcomes[1][1])
        self.assertIsNone(journal.outcomes[1][2])
        np.testing.assert_array_equal(final, callbacks[0][0])
        np.testing.assert_array_equal(velocity, callbacks[0][1])

    def test_valid_mixed_work_precedes_every_journal_outcome_and_callback(self):
        for mode in ("vector", "distance", "normal-offset"):
            with self.subTest(mode=mode):
                final, velocity, report, journal, callbacks, work, events = self.run_case(mode=mode)
                self.assertTrue(report["complete"])
                self.assertEqual(len(callbacks), 4)
                self.assertEqual(len(work), 4)
                for index, row in enumerate(report["acceptedSteps"]):
                    self.assertEqual(row["step"]["activeSewingRows"], [0])
                    self.assertEqual(row["step"]["pendingSewingRows"], [1])
                    energy = row["step"]["energyBalance"]
                    for key in SEWING_FIELDS:
                        self.assertIn(key, energy)
                        self.assertNotIsInstance(energy[key], bool)
                        self.assertTrue(np.isfinite(energy[key]))
                    np.testing.assert_array_equal(work[index]["previous_sewing_activation"], [1.,0.])
                    np.testing.assert_array_equal(work[index]["sewing_activation"], [1.,0.])
                for index, event in enumerate(events):
                    if event == ("outcome", "accepted"):
                        self.assertEqual(events[index-1][0], "work")
                np.testing.assert_array_equal(final, callbacks[-1][0])
                np.testing.assert_array_equal(velocity, callbacks[-1][1])

    def test_post_work_sewing_diagnostic_mutations_cannot_publish(self):
        mutations = [lambda r:r.update(sewingActivationExplicit=False),
            lambda r:r.update(sewingActivation=[0.,1.]), lambda r:r.update(activeSewingRows=[]),
            lambda r:r.update(pendingSewingRows=[0,1]), lambda r:r.pop("sewingActivation"),
            lambda r:r.update(activeSewingRows=[False]), lambda r:r.update(pendingSewingRows=[True])]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                result = self.run_case(work_attack=lambda solver,args,kwargs,energy:mutate(solver.last_report))
                self.assert_prefix_only(result)

    def test_every_mandatory_sewing_work_field_rejects_missing_boolean_or_nonfinite(self):
        for field in SEWING_FIELDS:
            for kind in ("missing", "boolean", "nonfinite"):
                def attack(solver,args,kwargs,energy):
                    if kind == "missing": energy.pop(field)
                    elif kind == "boolean": energy[field] = False
                    else: energy[field] = float("nan")
                with self.subTest(field=field,kind=kind):
                    self.assert_prefix_only(self.run_case(work_attack=attack))

    def test_negative_potential_release_or_bound_quantities_reject(self):
        for field in NONNEGATIVE_SEWING_FIELDS:
            with self.subTest(field=field):
                self.assert_prefix_only(self.run_case(work_attack=lambda solver,args,kwargs,energy:energy.__setitem__(field,-1.)))

    def test_sewing_row_cached_operator_or_compliance_drift_during_step_rejects(self):
        mutations = [lambda s:s.sewing.data.__setitem__(0,.5),
            lambda s:s.sewing.indices.__setitem__(0,2), lambda s:s.sewing.indptr.__setitem__(1,1),
            lambda s:setattr(s,"sewing",2*s.sewing), lambda s:s.sewing_xyz.data.__setitem__(0,.5),
            lambda s:setattr(s,"compliance",.08), lambda s:setattr(s,"sewing_mode","distance")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                result = self.run_case(step_attack=lambda solver,args,kwargs,report:mutate(solver))
                self.assert_prefix_only(result)
                self.assertEqual(len(result[5]),1,"Reject model drift before evaluating changed-model work")

    def test_sewing_row_cached_operator_compliance_and_frame_drift_during_work_rejects(self):
        mutations = [("vector",lambda s:s.sewing.data.__setitem__(0,.5)),
            ("vector",lambda s:setattr(s,"sewing",2*s.sewing)),
            ("vector",lambda s:s.sewing_xyz.data.__setitem__(0,.5)),
            ("vector",lambda s:setattr(s,"compliance",.08)),
            ("vector",lambda s:setattr(s,"sewing_mode","distance")),
            ("normal-offset",lambda s:s.sewing_frame_faces.__setitem__((0,0),1)),
            ("normal-offset",lambda s:s.sewing_sides.__setitem__(0,-1.))]
        for index, (mode, mutate) in enumerate(mutations):
            with self.subTest(index=index):
                self.assert_prefix_only(self.run_case(mode=mode,work_attack=lambda solver,args,kwargs,energy:mutate(solver)))

    def test_mixed_work_input_mutations_reject_and_preserve_accepted_state(self):
        mutations = [lambda args,kwargs:args[1].__setitem__((0,0),999.),
            lambda args,kwargs:args[2].__setitem__((0,0),999.),
            lambda args,kwargs:args[5].__setitem__((0,0),999.),
            lambda args,kwargs:kwargs["previous_sewing_activation"].__setitem__(0,0.),
            lambda args,kwargs:kwargs["sewing_activation"].__setitem__(1,1.),
            lambda args,kwargs:kwargs["fold_activation"].__setitem__(0,.7)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.assert_prefix_only(self.run_case(work_attack=lambda solver,args,kwargs,energy:mutate(args,kwargs)))

    def test_step_cannot_replace_original_fraction_sewing_controls_and_report(self):
        def mutate(solver,args,kwargs,report):
            kwargs["sewing_activation"][:]=[0.,1.]
            report.update(sewingActivation=[0.,1.],activeSewingRows=[1],pendingSewingRows=[0])
        result=self.run_case(step_attack=mutate)
        self.assert_prefix_only(result)
        self.assertEqual(len(result[5]),1)

    def test_sewing_only_work_isolation_and_required_accounting_preserve_prefix(self):
        from test_solver_sewing_activation_adaptive import QuadraticSewingSolver, ROW_IDS, sewing_schedule
        real_work=solver_energy_balance.global_energy_transition
        for kind in ("control","old-state","new-state","velocity","target","old-activation","new-activation",
                     "mutation-raise","missing","boolean","nonfinite"):
            solver=QuadraticSewingSolver()
            q=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[.1,.1,0.]])
            v=np.zeros_like(q)
            original_q,original_v=q.copy(),v.copy()
            events,callbacks,work_calls=[],[],[]
            journal=RecordingJournal(events)
            def work(*args,**kwargs):
                result=real_work(*args,**kwargs)
                work_calls.append(copy.deepcopy(kwargs));events.append(("work",len(work_calls)))
                if len(work_calls)==2:
                    if kind in ("old-state","mutation-raise"):args[1][:]=999.
                    elif kind=="new-state":args[2][:]=999.
                    elif kind=="velocity":args[3][:]=999.
                    elif kind=="target":args[6][:]=999.
                    elif kind=="old-activation":kwargs["previous_sewing_activation"][:]=0.
                    elif kind=="new-activation":kwargs["sewing_activation"][:]=0.
                    elif kind=="missing":result.pop("sewingParameterWorkJoules")
                    elif kind=="boolean":result["sewingParameterWorkJoules"]=False
                    elif kind=="nonfinite":result["sewingParameterWorkJoules"]=float("nan")
                    if kind=="mutation-raise":raise ValueError("Synthetic work mutation then failure")
                return result
            def accepted(q,v,row):
                callbacks.append((q.copy(),v.copy(),copy.deepcopy(row)))
                events.append(("callback",row["attemptId"]))
            with self.subTest(kind=kind),mock.patch.object(solver_energy_balance,"global_energy_transition",side_effect=work):
                final,velocity,report=adaptive_contact_step(solver,q,v,[[.02,0.,0.],[.03,.04,0.]],
                    [[.01,.02,0.],[.02,.01,.01]],.4,initial_subdivisions=4,max_depth=0,
                    sewing_schedule=sewing_schedule(),sewing_row_ids=ROW_IDS,attempt_journal=journal,on_accept=accepted)
                self.assertIsNone(getattr(solver,"controlled_fold_actuation",None))
                self.assertIsNone(getattr(solver,"fold_actuation",None))
                np.testing.assert_array_equal(q,original_q);np.testing.assert_array_equal(v,original_v)
                if kind=="control":
                    self.assertTrue(report["complete"]);self.assertEqual(len(callbacks),4)
                else:
                    self.assert_prefix_only((final,velocity,report,journal,callbacks,work_calls,events))


if __name__=="__main__":
    unittest.main()
