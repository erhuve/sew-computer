"""Independent fixed-cable energy accounting; no dynamics or source admission."""
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
from solver_cable_integration import CableControl
from solver_continuous_cable_sewing import ContinuousCableSewing
from solver_energy_balance import global_energy_transition, validate_continuous_cable_energy


def rat(value):
    value = F(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def rational(value):
    result = F(int(value["numerator"]), int(value["denominator"]))
    assert rat(result) == value
    return result


def anchor(values):
    return [{"vertex": vertex, "weight": rat(weight)} for vertex, weight in sorted(values.items())]


def cell(**changes):
    value = {"id": "fixed-continuum", "negativeStart": anchor({0: 1}), "negativeEnd": anchor({0: 1}),
             "positiveStart": anchor({1: 1}), "positiveEnd": anchor({1: 1}),
             "targetsMeters": [1., 1.], "referenceLengthMeters": rat(1),
             "stiffnessDensityNPerM2": 2., "activation": 1.}
    value.update(changes)
    return value


def policy(**changes):
    value = {"energy_tolerance_joules": 1e-10, "gradient_tolerance_newtons": 1e-10,
             "hessian_tolerance_newtons_per_meter": 1e-10, "absolute_tolerance_joules": 1e-12,
             "max_boundary_depth": 80, "max_boundary_panels": 256,
             "moment_max_terms": 128, "moment_max_panels": 256, "moment_max_depth": 64}
    value.update(changes)
    return value


def fixture(*, cells=None, precision=None, count=4):
    q = np.array([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]])
    if count != 4:
        q = np.zeros((count, 3)); q[1, 0] = 2.
    recipe = ContinuousCableSewing(count, [cell()] if cells is None else cells)
    control = CableControl(recipe, policy() if precision is None else precision)
    solver = SimpleNamespace(mass=np.ones(count), active=np.ones(count, dtype=bool),
        sewing=csr_matrix((0, count)), compliance=.5, sewing_mode="vector",
        poses=np.empty((0, 2, 2)), faces=np.empty((0, 3), dtype=int), areas=np.empty(0),
        materials=np.empty((0, 3)), bending=ElasticDihedralBending(count, np.empty((0, 4), dtype=int), [], [], []),
        continuous_cable=control)
    return solver, q


def transition(solver, q, end=None, **kwargs):
    end = q if end is None else end
    return global_energy_transition(solver, q, end, np.zeros_like(q), np.asarray(end)-np.asarray(q),
                                    np.empty((0, 3)), np.empty((0, 3)), 1., **kwargs)


class CableEnergyTests(unittest.TestCase):
    def test_exact_taut_motion_and_fixed_parameter_accounting(self):
        solver, q = fixture(); end = q.copy(); end[1, 0] = 2.5
        result = transition(solver, q, end)
        expected = {"cableBeforeJoules": 1., "cableAfterJoules": 2.25,
                    "cableFixedParameterChangeJoules": 1.25, "cableParameterWorkJoules": 0.,
                    "kineticChangeJoules": .125, "mechanicalChangeJoules": 1.375,
                    "mechanicalChangeMinusTargetWorkJoules": 1.375,
                    "mechanicalChangeMinusParameterWorkJoules": 1.375}
        for field, value in expected.items(): self.assertEqual(result[field], value, field)
        self.assertEqual(result["externalParameterWorkJoules"], 0.)
        self.assertIs(result["accepted"], False)
        record = validate_continuous_cable_energy(solver.continuous_cable, q, end, result)
        self.assertEqual(record, result["continuousCableEnergy"])
        self.assertIn("non-cable", record["scope"])
        for bound in record["errorBoundsJoules"].values(): self.assertEqual(rational(bound), 0)
        record["definition"]["accepted"] = True
        self.assertIs(result["continuousCableEnergy"]["definition"]["accepted"], False)
        self.assertIs(solver.continuous_cable.description()["accepted"], False)

    def test_slack_then_taut_crossing_and_pending_geometry(self):
        solver, q = fixture(); q[1, 0] = .5
        for distance, change in ((.75, 0.), (1., 0.), (1.5, .25)):
            with self.subTest(distance=distance):
                end = q.copy(); end[1, 0] = distance
                result = transition(solver, q, end)
                self.assertEqual(result["cableBeforeJoules"], 0.)
                self.assertEqual(result["cableAfterJoules"], change)
                self.assertEqual(result["cableFixedParameterChangeJoules"], change)
        pending, _ = fixture(cells=[cell(activation=0.)])
        collapsed = np.zeros_like(q)
        result = transition(pending, collapsed)
        for key in ("cableBeforeJoules", "cableAfterJoules", "cableFixedParameterChangeJoules"):
            self.assertEqual(result[key], 0.)
        self.assertIs(result["accepted"], False)  # no independent cloth/path claim

    def test_tiny_work_survives_identical_rounded_endpoint_energies(self):
        terms = anchor({1: F(1)-F(1, 2**40), 2: F(1, 2**40)})
        solver, q = fixture(cells=[cell(positiveStart=terms, positiveEnd=terms)],
                            precision=policy(absolute_tolerance_joules=1e-35))
        q[2] = q[1]
        for direction in (-math.inf, math.inf):
            with self.subTest(direction=direction):
                end = q.copy(); end[2, 0] = math.nextafter(q[2, 0], direction)
                delta = (F(end[2, 0])-F(q[2, 0]))/2**40
                expected = 2*delta+delta**2
                result = transition(solver, q, end)
                self.assertEqual(result["cableBeforeJoules"], result["cableAfterJoules"])
                actual = F(result["cableFixedParameterChangeJoules"])
                bound = rational(result["continuousCableEnergy"]["errorBoundsJoules"]["cableFixedParameterChangeJoules"])
                self.assertLessEqual(abs(actual-expected), bound)
                self.assertEqual(math.copysign(1., float(actual)), math.copysign(1., float(expected)))

    def test_rational_and_decimal_energy_oracles_lie_inside_reported_bounds(self):
        solver, q = fixture(); q[1] = [1., 1., 0.]
        end = q.copy(); end[1] = [1., 2., 0.]
        result = transition(solver, q, end)
        record = result["continuousCableEnergy"]
        with localcontext() as context:
            context.prec = 100
            before = (Decimal(2).sqrt()-1)**2
            after = (Decimal(5).sqrt()-1)**2
            expected = {"cableBeforeJoules": before, "cableAfterJoules": after,
                        "cableFixedParameterChangeJoules": after-before,
                        "mechanicalChangeJoules": after-before+Decimal('.5'),
                        "mechanicalChangeMinusTargetWorkJoules": after-before+Decimal('.5'),
                        "mechanicalChangeMinusParameterWorkJoules": after-before+Decimal('.5')}
            for field, reference in expected.items():
                error = abs(F(result[field])-F(reference))
                bound = rational(record["errorBoundsJoules"][field])
                self.assertLessEqual(error, bound+F(1, 10**95), field)
            self.assertGreater(rational(record["errorBoundsJoules"]["cableFixedParameterChangeJoules"]), 0)

    def test_sum_cancellation_retains_individual_legacy_terms(self):
        solver, q = fixture()
        solver.bending = SimpleNamespace(energy_change=lambda *_: 1., energy=lambda _: 0.)
        solver.contact = SimpleNamespace(energy_change=lambda *_: -1e16, energy=lambda _: 0.)
        with patch("solver_energy_balance.membrane_energy_change", return_value=1e16):
            result = transition(solver, q)
            cable = solver.continuous_cable; solver.continuous_cable = None
            legacy = transition(solver, q); solver.continuous_cable = cable
        for field in ("mechanicalChangeJoules", "mechanicalChangeMinusTargetWorkJoules",
                      "mechanicalChangeMinusParameterWorkJoules"):
            self.assertEqual(result[field], 1.)
            self.assertEqual(legacy[field], 0.)
        self.assertNotIn("continuousCableEnergy", legacy)

    def test_final_rounding_bound_is_exact_and_separate_from_cable_error(self):
        solver, q = fixture()
        solver.bending = SimpleNamespace(energy_change=lambda *_: 2.**-54, energy=lambda _: 0.)
        with patch("solver_energy_balance.membrane_energy_change", return_value=1.):
            result = transition(solver, q)
        record = result["continuousCableEnergy"]
        for field in record["assemblyRoundingBoundsJoules"]:
            self.assertEqual(result[field], 1.)
            self.assertEqual(rational(record["assemblyRoundingBoundsJoules"][field]), F(1, 2**54))
            self.assertEqual(rational(record["errorBoundsJoules"][field]), F(1, 2**54))

    def test_legacy_missing_control_and_none_are_identical(self):
        solver, q = fixture(); solver.continuous_cable = None
        end = q.copy(); end[1, 0] = 2.125
        explicit = transition(solver, q, end)
        del solver.continuous_cable
        missing = transition(solver, q, end)
        self.assertEqual(explicit, missing)
        self.assertFalse(any(key.startswith("cable") or key == "continuousCableEnergy" for key in explicit))

    def test_combined_fold_sewing_and_gripper_work_is_not_reclassified_as_cable_work(self):
        from test_solver_controlled_fold_energy import fixture as coupled_fixture, report as coupled_report
        solver, q, _ = coupled_fixture(sewing=True, grippers=True)
        options = dict(old_targets=[.5, -.25], targets=[1., .5], old_activation=[.75, .25], activation=[.25, 1.],
            old_sewing=np.zeros((2, 3)), sewing=np.ones((2, 3))*.125,
            previous_sewing_activation=[.75, .5], sewing_activation=[.25, 1.],
            previous_gripper_targets=[[0., 0., 0.], [3., 0., 0.]],
            gripper_targets=[[.5, .125, .25], [3.25, -.25, .5]],
            previous_gripper_activation=[.75, .5], gripper_activation=[.25, 1.])
        end = q.copy(); end[0, 1] += .125
        baseline = coupled_report(solver, q, end, **options)
        recipe = ContinuousCableSewing(8, [cell()])
        solver.continuous_cable = CableControl(recipe, policy())
        result = coupled_report(solver, q, end, **options)
        for field in ("externalParameterWorkJoules", "targetParameterWorkJoules", "foldParameterWorkJoules",
                      "sewingParameterWorkJoules", "gripperParameterWorkJoules"):
            self.assertEqual(result[field], baseline[field], field)
        for field, terms in result["continuousCableEnergy"]["aggregationTermsJoules"].items():
            self.assertEqual(result[field], float(sum(map(F, terms.values()), F())))
        self.assertEqual(result["cableParameterWorkJoules"], 0.)

    def test_raw_state_timestep_model_and_control_types_reject(self):
        solver, q = fixture()
        bad_states = [[[False, 0., 0.]]+q[1:].tolist(), np.full_like(q, math.nan), q[:2]]
        if np.dtype(np.longdouble).itemsize > 8:
            bad_states.append(q.astype(np.longdouble))
        for bad in bad_states:
            with self.subTest(raw=repr(bad)[:60]), self.assertRaises(ValueError): transition(solver, bad)
        for bad in (True, 2**53+1, 0., math.nan):
            with self.subTest(dt=bad), self.assertRaises(ValueError):
                global_energy_transition(solver, q, q, np.zeros_like(q), np.zeros_like(q), [], [], bad)
        admitted = global_energy_transition(solver, q, q, np.zeros_like(q), np.zeros_like(q),
                                           np.empty((0, 3)), np.empty((0, 3)), np.float64(1.))
        self.assertEqual(admitted["cableFixedParameterChangeJoules"], 0.)
        old = solver.continuous_cable
        solver.continuous_cable = SimpleNamespace(vertex_count=4)
        with self.assertRaises(ValueError): transition(solver, q)
        solver.continuous_cable = CableControl(ContinuousCableSewing(3, [cell()]), policy())
        with self.assertRaises(ValueError): transition(solver, q)
        solver.continuous_cable = old

    def test_corrupted_primitive_records_reject_at_energy_boundary(self):
        solver, q = fixture()
        original_response, original_change = CableControl.evaluate, CableControl.energy_change
        response_attacks = [lambda x: x.pop("certificate"),
            lambda x: x["certificate"].update(positionsSha256="0"*64),
            lambda x: x["certificate"].update(verified=1),
            lambda x: x["certificate"].update(energyErrorBoundJoules=rat(-1)),
            lambda x: x["certificate"].update(energyErrorBoundJoules=rat(1)),
            lambda x: x["certificate"]["budgets"].update(moment_max_terms=64),
            lambda x: x.update(energy=True), lambda x: x.update(gradient=np.full(12, math.inf))]
        for attack in response_attacks:
            def corrupted(control, state):
                result = original_response(control, state); attack(result); return result
            with self.subTest(attack=attack), patch.object(CableControl, "evaluate", corrupted), self.assertRaises(ValueError):
                transition(solver, q)
        work_attacks = [lambda x: x.pop("certificate"),
            lambda x: x["certificate"].update(endPositionsSha256="0"*64),
            lambda x: x["certificate"].update(inputSha256="0"*64),
            lambda x: x["certificate"].update(accepted=0),
            lambda x: x["certificate"].update(changeErrorBoundJoules=rat(1)),
            lambda x: x["certificate"].update(changeErrorBoundJoules={"numerator":"0","denominator":"2"}),
            lambda x: x.update(changeJoules=1.),
            lambda x: x.update(changeJoules=math.inf)]
        for attack in work_attacks:
            def corrupted(control, start, end):
                result = original_change(control, start, end); attack(result); return result
            with self.subTest(attack=attack), patch.object(CableControl, "energy_change", corrupted), self.assertRaises(ValueError):
                transition(solver, q)

    def test_structured_bound_and_identity_tampering_rejects(self):
        solver, q = fixture(); valid = transition(solver, q)
        validate_continuous_cable_energy(solver.continuous_cable, q, q, valid)
        attacks = [lambda x: x.pop("continuousCableEnergy"),
            lambda x: x.update(accepted=0),
            lambda x: x.update(externalParameterWorkJoules=math.inf),
            lambda x: x.update(targetParameterWorkJoules=True),
            lambda x: x.update(cableParameterWorkJoules=True),
            lambda x: x.update(cableFixedParameterChangeJoules=math.ulp(0.)),
            lambda x: x["continuousCableEnergy"].update(extra="unreviewed"),
            lambda x: x["continuousCableEnergy"]["definition"].update(accepted=0),
            lambda x: x["continuousCableEnergy"]["before"]["certificate"].update(positionsSha256="0"*64),
            lambda x: x["continuousCableEnergy"]["errorBoundsJoules"].update(mechanicalChangeJoules=rat(math.ulp(0.))),
            lambda x: x["continuousCableEnergy"]["assemblyRoundingBoundsJoules"].update(mechanicalChangeJoules=rat(1)),
            lambda x: x["continuousCableEnergy"]["aggregationTermsJoules"]["mechanicalChangeJoules"].pop("bendingChangeJoules"),
            lambda x: x["continuousCableEnergy"]["aggregationTermsJoules"]["mechanicalChangeJoules"].update(cableFixedParameterChangeJoules=1.)]
        for attack in attacks:
            value = copy.deepcopy(valid); attack(value)
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                validate_continuous_cable_energy(solver.continuous_cable, q, q, value)

    def test_input_mutation_and_recipe_replacement_are_rejected_without_caller_mutation(self):
        solver, q = fixture(); saved = q.copy(); original = CableControl.evaluate
        def mutated(control, state):
            response = original(control, state); state[0, 0] += 1.; return response
        with patch.object(CableControl, "evaluate", mutated), self.assertRaisesRegex(ValueError, "mutated"):
            transition(solver, q)
        np.testing.assert_array_equal(q, saved)
        def replaced(control, state):
            response = original(control, state)
            solver.continuous_cable = CableControl(ContinuousCableSewing(4, [cell(activation=0.)]), policy())
            return response
        with patch.object(CableControl, "evaluate", replaced), self.assertRaisesRegex(ValueError, "identity"):
            transition(solver, q)
        np.testing.assert_array_equal(q, saved)

    def test_unresolved_work_does_not_return_partial_accounting(self):
        solver, q = fixture(); saved = q.copy()
        with patch.object(CableControl, "energy_change", side_effect=ValueError("Unresolved fixed precision")):
            with self.assertRaisesRegex(ValueError, "Unresolved"): transition(solver, q)
        np.testing.assert_array_equal(q, saved)
        # A genuine non-dyadic slack/taut boundary also fails with a fixed
        # zero subdivision budget; no retry changes its declared precision.
        limited, q = fixture(cells=[cell(positiveEnd=anchor({2: 1}))],
                             precision=policy(max_boundary_depth=0))
        q[1] = [.5, 0., 0.]; q[2] = [2., 0., 0.]
        definition = limited.continuous_cable.description()
        with self.assertRaises(ValueError): transition(limited, q)
        self.assertEqual(limited.continuous_cable.description(), definition)


if __name__ == "__main__":
    unittest.main()
