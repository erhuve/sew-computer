"""Opt-in temporal control exercised with actual generic numerical mechanics."""
import copy
from fractions import Fraction as F
import math
import unittest
from unittest.mock import patch

import newton
import numpy as np

from solver_adaptive_contact import adaptive_contact_step
from solver_global_sewing import GlobalSewingSolver
from solver_temporal_control import problem_identity
from test_solver_temporal_control import declaration


EMPTY = np.empty((0, 3))


def oscillator(omega=1.):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for x in (-.5, .5): builder.add_particle(pos=(x, 0., 0.), vel=(0., 0., 0.), mass=2.)
    builder.set_coloring([[0], [1]])
    model = builder.finalize(device="cpu")
    solver = GlobalSewingSolver(model, [{0: -1., 1: 1.}], 1/(omega*omega))
    return solver, model.particle_q.numpy().astype(float)


def solve(solver, q, *, dt=.125, thresholds=None, **options):
    target = np.zeros((solver.sewing.shape[0], 3))
    return adaptive_contact_step(solver, q, np.zeros_like(q), target, target, dt,
        temporal_policy=declaration(**(thresholds or {})),
        **{"max_depth": 8, "max_attempts": 128, "max_evaluations": 32, **options})


class TemporalGlobalTests(unittest.TestCase):
    def test_execution_handle_requires_temporal_mode_before_any_solver_call(self):
        from solver_temporal_execution import TemporalExecution
        solver, q = oscillator()
        target = np.zeros((1, 3))
        with patch.object(solver, "step", side_effect=AssertionError("must not solve")):
            with self.assertRaisesRegex(ValueError, "explicit temporal policy"):
                adaptive_contact_step(solver, q, q*0, target, target, .125,
                                      temporal_execution=TemporalExecution())

    def test_retained_adaptive_context_and_unused_handle_preserve_real_mechanics(self):
        from solver_temporal_execution import TemporalExecution
        solver, q = oscillator()
        baseline = solve(solver, q)
        handle = TemporalExecution()
        result = solve(solver, q, temporal_execution=handle)
        for a, b in zip(baseline[:2], result[:2]): self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(result[2]["acceptedSteps"], baseline[2]["acceptedSteps"])
        snapshot = handle.snapshot()
        context = snapshot["context"]["adaptiveContext"]
        self.assertEqual(context["stationarityToleranceN"], 1e-6)
        self.assertEqual(context["initialTargets"]["shape"], [1, 3])
        self.assertEqual(context["defaultProgress"], "original-fraction-linear")
        self.assertEqual(snapshot["enrichment"], "ready")
        self.assertTrue(snapshot["complete"])

    def test_returned_cleanup_error_cannot_mask_stop_in_adaptive_wrapper(self):
        from solver_temporal_execution import TemporalExecution, TemporalStopRequested
        for replacement in (ValueError, RuntimeError, MemoryError):
            solver, q = oscillator()
            handle = TemporalExecution("resource-stop")
            original, count = solver.step, 0
            def step(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 4:
                    handle.request_stop()
                    try: raise TemporalStopRequested("first")
                    finally: raise replacement("cleanup")
                return original(*args, **kwargs)
            with self.subTest(replacement=replacement), patch.object(solver, "step", side_effect=step):
                with self.assertRaises(TemporalStopRequested):
                    solve(solver, q, temporal_execution=handle, initial_subdivisions=2)
            snapshot = handle.snapshot()
            self.assertEqual(count, 4)
            self.assertEqual(snapshot["completedFraction"], .5)
            self.assertEqual(snapshot["enrichment"], "ready")
            self.assertTrue(any(row["type"] == replacement.__name__ for row in snapshot["failures"]))

    def test_pre_requested_handle_captures_admitted_sewing_gripper_and_assembly_knots(self):
        from solver_temporal_execution import TemporalExecution, TemporalStopRequested
        from test_solver_sewing_activation_adaptive import SewingActivationAdaptiveTests, gripper_schedule
        fixture = SewingActivationAdaptiveTests()
        solver, q, v = fixture.fixture(coupled=True)
        handle = TemporalExecution(); handle.request_stop()
        assembly = {"profile": "sewing-fold-progress-v1", "knots": [
            {"fraction": 0., "sewingProgress": 0., "foldProgress": 0.},
            {"fraction": .5, "sewingProgress": 1., "foldProgress": .25},
            {"fraction": 1., "sewingProgress": 1., "foldProgress": 1.}]}
        with patch.object(solver, "step", side_effect=AssertionError("must not solve")), self.assertRaises(TemporalStopRequested):
            fixture.run_schedule(solver, q, v, temporal_policy=declaration(), temporal_execution=handle,
                initial_fold_targets=[-.0], fold_targets=[.1], gripper_schedule=gripper_schedule(),
                assembly_schedule=assembly, max_depth=2)
        context = handle.snapshot()["context"]["adaptiveContext"]
        self.assertEqual(context["assemblySchedule"]["knots"]["knots"][1], [.5, 1., .25])
        self.assertEqual(context["sewingSchedule"]["knots"]["rowIds"],
                         ["synthetic:row:first", "synthetic:row:second"])
        self.assertEqual(context["gripperSchedule"]["knots"]["activation"]["values"], [[1.], [0.]])
        self.assertEqual(context["initialFoldTargets"]["values"][0].hex(), "-0x0.0p+0")
        assembly["knots"][1]["foldProgress"] = .99
        self.assertEqual(handle.snapshot()["context"]["adaptiveContext"], context)

    def test_pre_requested_handle_captures_controlled_fold_fixed_cable_and_bounded_contact(self):
        from dataclasses import asdict
        from solver_temporal_execution import TemporalExecution, TemporalStopRequested
        from solver_contact_work import WorkPolicy
        from solver_contact_work_control import ContactWorkControl
        from test_solver_cable_global import fixture as cable_fixture
        from test_solver_controlled_fold_integration import fixture as fold_fixture
        from test_solver_controlled_fold_adaptive import schedule
        for kind in ("fold", "cable"):
            if kind == "fold":
                _, solver, q = fold_fixture(contact_enabled=True)
                raw = schedule(solver.controlled_fold_actuation.hinges)
                options = {"fold_control_schedule": raw}
            else:
                _, solver, q, _ = cable_fixture()
                options = {}
            solver.contact_work_control = ContactWorkControl(solver.contact, asdict(WorkPolicy()))
            handle = TemporalExecution(); handle.request_stop()
            with self.subTest(kind=kind), patch.object(solver, "step", side_effect=AssertionError("must not solve")), \
                 self.assertRaises(TemporalStopRequested):
                adaptive_contact_step(solver, q, np.zeros_like(q), EMPTY, EMPTY, .0008,
                    temporal_policy=declaration(), temporal_execution=handle,
                    initial_subdivisions=4, max_depth=2, **options)
            context = handle.snapshot()["context"]["adaptiveContext"]
            self.assertEqual(context["boundedContactControl"], solver.contact_work_control.description())
            if kind == "fold":
                self.assertEqual(context["foldSchedule"]["knots"]["hinges"], raw["hinges"])
                self.assertIn("fold", context["temporalSchedulePreflight"])
            else:
                self.assertEqual(context["fixedCableControl"], solver.continuous_cable.description())

    def test_varying_cable_enrichment_failure_keeps_raw_context_and_primary_stop(self):
        from solver_temporal_execution import TemporalExecution, TemporalStopRequested
        from test_solver_cable_varying_global import particle_fixture
        for stopped in (False, True):
            _, solver, q, recipe = particle_fixture(activation=0.)
            raw = {"profile": "cable-target-activation-v1", "geometrySha256": recipe.geometry_sha256,
                   "cellIds": list(recipe.cell_ids), "knots": [
                {"fraction": 0., "targetsMeters": [[1., 1.]], "activation": [0.]},
                {"fraction": .5, "targetsMeters": [[1., 1.]], "activation": [1.]},
                {"fraction": 1., "targetsMeters": [[1., 1.]], "activation": [0.]}]}
            handle, original, count = TemporalExecution(), solver.step, 0
            def step(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 4 and stopped:
                    handle.request_stop(); raise TemporalStopRequested("trial stop")
                return original(*args, **kwargs)
            with self.subTest(stopped=stopped), patch.object(solver, "step", side_effect=step), \
                 patch("solver_adaptive_contact._varying_cable_totals", side_effect=ValueError("totals fail")):
                with self.assertRaises(TemporalStopRequested if stopped else ValueError):
                    adaptive_contact_step(solver, q, np.zeros_like(q), EMPTY, EMPTY, .125,
                        temporal_policy=declaration(), temporal_execution=handle,
                        initial_subdivisions=2, max_depth=2, max_evaluations=32, cable_parameter_schedule=raw)
            snapshot = handle.snapshot()
            self.assertEqual(snapshot["completedFraction"], .5 if stopped else 1.)
            self.assertEqual(count, 4 if stopped else 6)
            self.assertEqual(snapshot["enrichment"], "failed")
            context = snapshot["context"]["adaptiveContext"]
            self.assertIsNotNone(context["cableParameterPreflight"])
            self.assertEqual(context["varyingCableControl"], solver.cable_parameter_control.description())
            self.assertTrue(any(row["stage"] == "enrichment" and row["type"] == "ValueError"
                                for row in snapshot["failures"]))

    def test_smooth_oscillator_refines_and_matches_independent_discrete_recurrence(self):
        solver, q = oscillator()
        original = problem_identity(solver)
        actual, velocity, row = solve(solver, q, thresholds={
            "positionToleranceM": 1e-5, "velocityToleranceMPerS": 1e-5,
            "numericalEnergyBudgetJ": .001})
        self.assertTrue(row["complete"], row)
        self.assertGreater(len(row["acceptedSteps"]), 2)
        self.assertEqual(problem_identity(solver), original)
        gap, speed = F(1), F()
        for record in row["acceptedSteps"]:
            h = F(record["durationSeconds"])
            gap, speed = (gap+h*speed)/(1+h*h), (speed-h*gap)/(1+h*h)
        self.assertAlmostEqual(actual[1, 0]-actual[0, 0], float(gap), delta=2e-11)
        self.assertAlmostEqual(velocity[1, 0]-velocity[0, 0], float(speed), delta=2e-11)
        self.assertLess(abs(float(gap)-math.cos(.125)), .001)
        self.assertLess(abs(float(speed)+math.sin(.125)), .001)
        self.assertTrue(any(item["outcome"] == "temporally-rejected" for item in row["temporalAssessments"]))

    def test_stiff_oscillator_state_agreement_does_not_bypass_energy_budget(self):
        solver, q = oscillator(64.)
        _, _, row = solve(solver, q, dt=1., max_depth=1, thresholds={
            "positionToleranceM": .01, "velocityToleranceMPerS": 1., "numericalEnergyBudgetJ": .01})
        self.assertFalse(row["complete"])
        assessment = row["temporalAssessments"][0]
        self.assertTrue(assessment["position"]["withinThreshold"])
        self.assertTrue(assessment["velocity"]["withinThreshold"])
        self.assertTrue(any(item["outcome"] == "exceeds-budget" for item in assessment["fineEnergy"]))
        self.assertEqual(len(row["acceptedSteps"]), 0)

    def test_default_report_and_motion_are_identical_with_explicit_none(self):
        solver, q = oscillator()
        target = np.zeros((1, 3))
        first = adaptive_contact_step(solver, q, q*0, target, target, .125)
        second = adaptive_contact_step(solver, q, q*0, target, target, .125, temporal_policy=None)
        self.assertEqual(first[2], second[2])
        for a, b in zip(first[:2], second[:2]): self.assertEqual(a.tobytes(), b.tobytes())
        self.assertNotIn("temporalMotion", first[2]["acceptedSteps"][0]["step"].get("energyBalance", {}))

    def test_no_callback_or_legacy_journal_is_called(self):
        solver, q = oscillator()
        for keyword in ("on_accept", "attempt_journal"):
            with self.subTest(keyword=keyword), self.assertRaisesRegex(ValueError, "in memory"):
                solve(solver, q, **{keyword: lambda *args: self.fail("callback invoked")})

    def test_raw_lossy_state_and_extra_half_depth_reject_before_step(self):
        solver, q = oscillator()
        bad = q.astype(object); bad[0, 0] = 2**53+1
        with patch.object(solver, "step", side_effect=AssertionError("must not solve")):
            with self.assertRaises(ValueError): solve(solver, bad)
            with self.assertRaises(ValueError): solve(solver, q, max_depth=0)

    def test_mutating_model_in_second_half_stops_at_prior_prefix(self):
        solver, q = oscillator()
        original = solver.step
        count = 0
        def attack(*args, **kwargs):
            nonlocal count
            result = original(*args, **kwargs)
            count += 1
            if count == 3: solver.mass[0] *= 2
            return result
        with patch.object(solver, "step", side_effect=attack):
            try:
                result = solve(solver, q)
            except RuntimeError as error:
                result = error.temporal_result
        self.assertFalse(result[2]["complete"])
        self.assertEqual(len(result[2]["acceptedSteps"]), 0)
        np.testing.assert_array_equal(result[0], q)

    def test_candidate_narrowing_and_work_input_mutation_reject(self):
        import solver_energy_balance
        for attack in ("candidate", "work"):
            with self.subTest(attack=attack):
                solver, q = oscillator()
                if attack == "candidate":
                    original = solver.step
                    def mutate(*args, **kwargs):
                        x, v, r = original(*args, **kwargs)
                        x = x.astype(object); x[0, 0] = 2**53+1
                        return x, v, r
                    context = patch.object(solver, "step", side_effect=mutate)
                else:
                    original = solver_energy_balance.global_energy_transition
                    def mutate(*args, **kwargs):
                        result = original(*args, **kwargs)
                        args[1][:] = 777.
                        return result
                    context = patch.object(solver_energy_balance, "global_energy_transition", side_effect=mutate)
                with context:
                    result = solve(solver, q, max_depth=1)
                self.assertFalse(result[2]["complete"])
                np.testing.assert_array_equal(result[0], q)

    def test_varying_cable_preserves_knots_fine_work_and_exact_schedule(self):
        from test_solver_cable_varying_global import particle_fixture
        _, solver, q, recipe = particle_fixture(activation=0.)
        schedule = {"profile": "cable-target-activation-v1", "geometrySha256": recipe.geometry_sha256,
                    "cellIds": list(recipe.cell_ids), "knots": [
            {"fraction": 0., "targetsMeters": [[1., 1.]], "activation": [0.]},
            {"fraction": .5, "targetsMeters": [[1., 1.]], "activation": [1.]},
            {"fraction": 1., "targetsMeters": [[1., 1.]], "activation": [0.]}]}
        result = adaptive_contact_step(solver, q, np.zeros_like(q), EMPTY, EMPTY, .125,
            temporal_policy=declaration(), initial_subdivisions=2, max_depth=2, max_attempts=12,
            max_evaluations=32, cable_parameter_schedule=copy.deepcopy(schedule))
        row = result[2]
        self.assertTrue(row["complete"], row)
        self.assertEqual([item["endFraction"] for item in row["acceptedSteps"]], [.25, .5, .75, 1.])
        self.assertEqual(row["varyingCableAcceptedWorkTotals"]["acceptedStepCount"], 4)
        self.assertEqual(row["resources"]["mechanicalTrials"], 6)
        for step in row["acceptedSteps"]:
            self.assertIn("temporalMotion", step["step"]["energyBalance"])

    def test_actual_contact_cache_branch_order_is_deterministic(self):
        from test_solver_cable_global import fixture
        _, solver, q, _ = fixture()
        options = dict(temporal_policy=declaration(), max_depth=1, max_attempts=3, max_evaluations=64)
        original = problem_identity(solver)
        first = adaptive_contact_step(solver, q, np.zeros_like(q), EMPTY, EMPTY, .000125, **options)
        # Exercise another actual trial/cache state before repeating from q.
        solver.step(q, np.zeros_like(q), EMPTY, .00003125, max_evaluations=64)
        second = adaptive_contact_step(solver, q, np.zeros_like(q), EMPTY, EMPTY, .000125, **options)
        self.assertTrue(first[2]["complete"], first[2])
        self.assertTrue(second[2]["complete"], second[2])
        self.assertEqual(problem_identity(solver), original)
        for a, b in zip(first[:2], second[:2]): np.testing.assert_array_equal(a, b)
        self.assertEqual(first[2]["conditionalAbsoluteEnergyBoundJoules"], second[2]["conditionalAbsoluteEnergyBoundJoules"])
        from solver_triangle_sweep import triangle_sweep_safe
        previous = q
        for row in first[2]["acceptedSteps"]:
            state = np.array(row["state"]["positionsMeters"])
            self.assertTrue(triangle_sweep_safe(previous, state, solver.faces))
            self.assertTrue(solver.contact.path_safe(previous, state))
            np.testing.assert_array_equal(np.array(row["state"]["velocitiesMPerS"]),
                                          (state-previous)/row["durationSeconds"])
            previous = state

    def test_actual_fold_contact_controls_keep_both_fine_path_guards(self):
        from test_solver_controlled_fold_integration import fixture
        from test_solver_controlled_fold_adaptive import schedule
        from solver_hinge_sweep import hinge_sweep_safe
        from solver_triangle_sweep import triangle_sweep_safe
        _, solver, q = fixture(contact_enabled=True)
        raw = schedule(solver.controlled_fold_actuation.hinges)
        for knot in raw["knots"]: knot["targetsRadians"] = [value*.01 for value in knot["targetsRadians"]]
        _, _, report = adaptive_contact_step(solver, q, np.zeros_like(q), EMPTY, EMPTY, .0008,
            temporal_policy=declaration(), initial_subdivisions=4, max_attempts=24, max_depth=2,
            max_evaluations=64, fold_control_schedule=raw)
        self.assertTrue(report["complete"], report)
        self.assertEqual(len(report["acceptedSteps"]), 8)
        self.assertIn("fold", report["temporalSchedulePreflight"])
        previous = q
        for row in report["acceptedSteps"]:
            state = np.array(row["state"]["positionsMeters"])
            self.assertTrue(hinge_sweep_safe(previous, state, solver.controlled_fold_actuation.hinges))
            self.assertTrue(triangle_sweep_safe(previous, state, solver.faces))
            self.assertTrue(solver.contact.path_safe(previous, state))
            previous = state

    def test_weighted_sewing_gripper_and_legacy_fold_share_original_schedule(self):
        from test_solver_sewing_activation_adaptive import SewingActivationAdaptiveTests, gripper_schedule
        fixture = SewingActivationAdaptiveTests()
        solver, q, v = fixture.fixture(coupled=True)
        result = fixture.run_schedule(solver, q, v, temporal_policy=declaration(
            positionToleranceM=.1, velocityToleranceMPerS=1., numericalEnergyBudgetJ=10.),
            max_depth=2, max_attempts=24, max_evaluations=32,
            initial_fold_targets=[0.], fold_targets=[.1], gripper_schedule=gripper_schedule())
        self.assertTrue(result[2]["complete"], result[2])
        self.assertEqual(len(result[2]["acceptedSteps"]), 8)
        self.assertEqual(set(result[2]["temporalSchedulePreflight"]), {"sewing", "gripper"})
        for record in result[2]["acceptedSteps"]:
            energy = record["step"]["energyBalance"]
            self.assertIn("gripperParameterWorkJoules", energy)
            self.assertIn("foldFixedParameterChangeJoules", energy)

    def test_added_half_grid_activation_underflow_rejects_before_solve(self):
        from test_solver_cable_varying_global import particle_fixture
        _, solver, q, recipe = particle_fixture(activation=0.)
        raw = {"profile": "cable-target-activation-v1", "geometrySha256": recipe.geometry_sha256,
               "cellIds": list(recipe.cell_ids), "knots": [
            {"fraction": 0., "targetsMeters": [[1., 1.]], "activation": [0.]},
            {"fraction": 1., "targetsMeters": [[1., 1.]], "activation": [2**-1074]}]}
        with patch.object(solver, "step", side_effect=AssertionError("no solve permitted")):
            with self.assertRaises(ValueError):
                adaptive_contact_step(solver, q, np.zeros_like(q), EMPTY, EMPTY, .125,
                    temporal_policy=declaration(), max_depth=1, cable_parameter_schedule=raw)

    def test_temporal_payload_forgery_cannot_hide_actual_dissipation(self):
        import solver_energy_balance
        solver, q = oscillator()
        real = solver_energy_balance.global_energy_transition
        def forge(*args, **kwargs):
            report = real(*args, **kwargs)
            report["temporalMotion"]["termsJoules"]["sewingFixedParameterChangeJoules"] = -report["kineticChangeJoules"]
            return report
        with patch.object(solver_energy_balance, "global_energy_transition", side_effect=forge):
            _, _, report = solve(solver, q, max_depth=1, thresholds={"numericalEnergyBudgetJ": 1e-12})
        self.assertFalse(report["complete"])
        self.assertEqual(report["temporalAssessments"][0]["outcome"], "invalid-temporal-evidence")

    def test_incomplete_parameter_work_cannot_be_committed(self):
        import solver_energy_balance
        real = solver_energy_balance.global_energy_transition
        for field in ("targetParameterWorkJoules", "externalParameterWorkJoules", "kineticChangeJoules",
                      "mechanicalChangeMinusParameterWorkJoules", "contactBeforeJoules"):
            with self.subTest(field=field):
                solver, q = oscillator()
                def remove(*args, **kwargs):
                    report = real(*args, **kwargs); del report[field]; return report
                with patch.object(solver_energy_balance, "global_energy_transition", side_effect=remove):
                    _, _, report = solve(solver, q, max_depth=1)
                self.assertFalse(report["complete"])
                self.assertFalse(report["acceptedSteps"])

    def test_interrupted_varying_report_has_same_declared_context_and_prefix_totals(self):
        from test_solver_cable_varying_global import particle_fixture
        _, solver, q, recipe = particle_fixture(activation=0.)
        raw = {"profile": "cable-target-activation-v1", "geometrySha256": recipe.geometry_sha256,
               "cellIds": list(recipe.cell_ids), "knots": [
            {"fraction": 0., "targetsMeters": [[1., 1.]], "activation": [0.]},
            {"fraction": 1., "targetsMeters": [[1., 1.]], "activation": [1.]}]}
        original = solver.step; count = 0
        def interrupt(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 6: raise KeyboardInterrupt("last provisional half")
            return original(*args, **kwargs)
        with patch.object(solver, "step", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt) as caught:
                adaptive_contact_step(solver, q, np.zeros_like(q), EMPTY, EMPTY, .125,
                    temporal_policy=declaration(), initial_subdivisions=2, max_evaluations=32,
                    cable_parameter_schedule=raw)
        report = caught.exception.temporal_result[2]
        self.assertEqual(report["completedFraction"], .5)
        self.assertEqual(report["stationarityToleranceN"], 1e-6)
        self.assertEqual(report["cableParameterSchedule"]["cellIds"], list(recipe.cell_ids))
        self.assertEqual(report["varyingCableAcceptedWorkTotals"]["acceptedStepCount"], 2)


if __name__ == "__main__": unittest.main()
