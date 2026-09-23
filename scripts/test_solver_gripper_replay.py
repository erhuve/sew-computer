"""Portable adversarial replay tests with independently constructed evidence."""
import ast
import copy
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from solver_gripper_replay import derive_grippers, parameters, verify_initial, verify_gripper_step


BINDING_SCOPE = ("Canonical material-triangle anchors; complete canonical digest and captured source lineage remain in the enclosing run manifest. "
                 "Targets are virtual controls, not collision geometry or construction proof.")
MOMENTUM_SCOPE = ("Backward-Euler force at the new state on free cloth; virtual gripper impulse, "
                  "not isolated-cloth momentum conservation")


def bind_digest(source):
    payload = {key: source[key] for key in ("restMeters", "triangles", "instanceOffsets")}
    source["gripperActuation"]["meshSha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fixture():
    source = {"restMeters": [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
              "triangles": [0, 1, 2], "instanceOffsets": {"cloth": 0},
              "gripperActuation": {"profile": "captured-material-grippers-v1", "accepted": False,
                  "anchors": [{"id": "g", "instanceId": "cloth", "triangleIndex": 0,
                               "weights": [1., 0., 0.], "stiffnessNPerM": 2.}],
                  "schedule": {"profile": "material-gripper-target-activation-v1", "gripperIds": ["g"],
                      "knots": [{"fraction": 0., "targetsMeters": [[1.5, 0., 0.]], "activation": [0.]},
                                {"fraction": 1., "targetsMeters": [[1.5, 0., 0.]], "activation": [1.]}]}}}
    bind_digest(source)
    previous = np.array(source["restMeters"])
    final = previous.copy()
    final[0, 0] = .5
    velocity = np.zeros_like(final)
    velocity[0, 0] = 1.
    diagnostics = {"profile": "source-material-compliant-grippers-v1", "gripperIds": ["g"], "energyJoules": 1.,
        "anchorPositionsMeters": [[.5, 0., 0.]], "targetsMeters": [[1.5, 0., 0.]], "activation": [1.],
        "anchorWeightSums": [1.], "weightSumAdmissionTolerance": 1e-12,
        "anchorForcesNewtons": [[2., 0., 0.]], "toolReactionsNewtons": [[-2., 0., 0.]],
        "nodalForcesNewtons": [[2., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        "totalClothForceNewtons": [2., 0., 0.], "totalToolReactionNewtons": [-2., 0., 0.],
        "clothTorqueNewtonMeters": [0., 0., 0.], "toolTorqueNewtonMeters": [0., 0., 0.],
        "netForceResidualNewtons": [0., 0., 0.], "netTorqueResidualNewtonMeters": [0., 0., 0.]}
    bound = math.fsum(math.ulp(value) for value in (2.25, 0., 2.25)) + math.ulp(2.25)
    balance = {"accepted": False, "gripperBeforeJoules": 0., "gripperAfterJoules": 1.,
        "gripperFixedPositionAfterJoules": 2.25, "gripperFixedParameterChangeJoules": -1.25,
        "gripperChangeJoules": 1., "gripperParameterWorkJoules": 2.25,
        "gripperTargetParameterWorkJoules": 0., "gripperActivationParameterWorkJoules": 2.25,
        "gripperActivationIncreaseWorkJoules": 2.25, "gripperReleaseEnergyRemovedJoules": 0.,
        "gripperParameterWorkComponentSumErrorBoundJoules": bound}
    step = {"materialGrippers": True, "accepted": False, "converged": True,
        "gripperTargetsMeters": [[1.5, 0., 0.]], "gripperActivation": [1.], "gripperEnergyJoules": 1.,
        "gripperDiagnostics": diagnostics, "energyBalance": balance,
        "gripperMomentum": {"changeKgMPerS": [1., 0., 0.], "externalImpulseNs": [1., 0., 0.],
            "residualNs": [0., 0., 0.], "toleranceNs": 3 * .5 * 1e-6 + 64 * np.finfo(float).eps,
            "scope": MOMENTUM_SCOPE}}
    binding = {"profile": "captured-material-grippers-v1", "accepted": False,
        "meshSha256": source["gripperActuation"]["meshSha256"], "scope": BINDING_SCOPE,
        "anchorBindings": [dict(source["gripperActuation"]["anchors"][0], canonicalVertices=[0, 1, 2], restAnchorMeters=[0., 0., 0.])]}
    run = {"accepted": False, "gripperActuation": copy.deepcopy(source["gripperActuation"]), "gripperBinding": binding,
           "initialGripperEnergyJoules": 0., "initialGripperTargetsMeters": [[1.5, 0., 0.]], "initialGripperActivation": [0.]}
    arguments = (previous, final, np.zeros_like(previous), velocity, np.ones(3), 0., 1., .5)
    return source, run, step, arguments


class IndependentGripperReplayTests(unittest.TestCase):
    def test_hand_solved_backward_euler_step_and_initial_binding(self):
        source, run, step, arguments = fixture()
        record = derive_grippers(source, 1)
        initial = verify_initial(record, arguments[0], run)
        gradient, evidence = verify_gripper_step(record, *arguments, step)
        self.assertTrue(initial["verified"])
        self.assertTrue(evidence["verified"])
        self.assertEqual(evidence["gripperParameterWorkJoules"], 2.25)
        np.testing.assert_array_equal(gradient, [[-2., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        np.testing.assert_array_equal(evidence["momentumResidualNs"], [0., 0., 0.])

    def test_barycentric_force_distribution_and_nonzero_tool_torque(self):
        source, _, step, arguments = fixture()
        source["gripperActuation"]["anchors"][0]["weights"] = [.25, .25, .5]
        target = [[1.75, .5, 0.]]
        for knot in source["gripperActuation"]["schedule"]["knots"]:
            knot["targetsMeters"] = target
        previous = arguments[0]
        final = previous + [.5, 0., 0.]
        velocity = np.broadcast_to([1., 0., 0.], final.shape).copy()
        step["gripperTargetsMeters"] = target
        step["gripperDiagnostics"].update(anchorPositionsMeters=[[.75, .5, 0.]], targetsMeters=target,
            nodalForcesNewtons=[[.5, 0., 0.], [.5, 0., 0.], [1., 0., 0.]],
            clothTorqueNewtonMeters=[0., 0., -1.], toolTorqueNewtonMeters=[0., 0., 1.])
        gradient, result = verify_gripper_step(derive_grippers(source, 1), previous, final,
            np.zeros_like(previous), velocity, [.25, .25, .5], 0., 1., .5, step)
        np.testing.assert_array_equal(gradient, [[-.5, 0., 0.], [-.5, 0., 0.], [-1., 0., 0.]])
        self.assertTrue(result["verified"])
        step["gripperDiagnostics"]["toolTorqueNewtonMeters"] = [0., 0., 0.]
        with self.assertRaisesRegex(ValueError, "toolTorqueNewtonMeters"):
            verify_gripper_step(derive_grippers(source, 1), previous, final,
                np.zeros_like(previous), velocity, [.25, .25, .5], 0., 1., .5, step)

    def test_all_required_step_fields_and_each_numeric_diagnostic_are_checked(self):
        source, _, step, arguments = fixture()
        record = derive_grippers(source, 1)
        for key in step:
            damaged = copy.deepcopy(step)
            del damaged[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                verify_gripper_step(record, *arguments, damaged)
        for group in ("gripperDiagnostics", "energyBalance", "gripperMomentum"):
            for key, value in step[group].items():
                with self.subTest(group=group, missing=key), self.assertRaises(ValueError):
                    damaged = copy.deepcopy(step)
                    del damaged[group][key]
                    verify_gripper_step(record, *arguments, damaged)
                if type(value) in (float, int) or isinstance(value, list) and key != "gripperIds":
                    damaged = copy.deepcopy(step)
                    if isinstance(value, list):
                        item = damaged[group][key]
                        while isinstance(item[0], list):
                            item = item[0]
                        item[0] += .01
                    else:
                        damaged[group][key] += .01
                    with self.subTest(group=group, altered=key), self.assertRaises(ValueError):
                        verify_gripper_step(record, *arguments, damaged)
        for key, value in (("materialGrippers", False), ("accepted", True), ("converged", False),
                           ("gripperActivation", [0.]), ("gripperTargetsMeters", [[1.6, 0., 0.]]),
                           ("gripperEnergyJoules", 1.01)):
            with self.subTest(altered=key), self.assertRaises(ValueError):
                verify_gripper_step(record, *arguments, step | {key: value})

    def test_binding_missing_or_changed_source_identity_and_initial_energy_reject(self):
        source, run, _, arguments = fixture()
        record = derive_grippers(source, 1)
        for key in run:
            changed = copy.deepcopy(run)
            del changed[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                verify_initial(record, arguments[0], changed)
        attacks = [lambda run: run.update(initialGripperEnergyJoules=.001),
                   lambda run: run.update(initialGripperActivation=[1.]),
                   lambda run: run["gripperBinding"].update(meshSha256="0" * 64),
                   lambda run: run["gripperBinding"].update(accepted=True),
                   lambda run: run["gripperBinding"]["anchorBindings"][0].update(canonicalVertices=[0., 1, 2]),
                   lambda run: run["gripperBinding"]["anchorBindings"][0].update(restAnchorMeters=[.001, 0., 0.]),
                   lambda run: run["gripperBinding"]["anchorBindings"][0].update(weights=[.5, .5, 0.]),
                   lambda run: run["gripperActuation"]["anchors"][0].update(id="different")]
        for index, attack in enumerate(attacks):
            changed = copy.deepcopy(run)
            attack(changed)
            with self.subTest(attack=index), self.assertRaises(ValueError):
                verify_initial(record, arguments[0], changed)

    def test_raw_boolean_mesh_stale_digest_cross_instance_and_recipe_attacks_reject(self):
        source, _, _, _ = fixture()
        attacks = [lambda source: source["restMeters"][0].__setitem__(0, False),
                   lambda source: source["triangles"].__setitem__(0, False),
                   lambda source: source["triangles"].__setitem__(0, .0),
                   lambda source: source["triangles"].__setitem__(1, 0),
                   lambda source: source["gripperActuation"]["anchors"][0].update(triangleIndex=True),
                   lambda source: source["gripperActuation"]["anchors"][0].update(instanceId="other"),
                   lambda source: source["gripperActuation"]["anchors"][0].update(weights=[.5, .4, 0.]),
                   lambda source: source["gripperActuation"]["anchors"][0].update(stiffnessNPerM=True),
                   lambda source: source["gripperActuation"]["anchors"].append(copy.deepcopy(source["gripperActuation"]["anchors"][0])),
                   lambda source: source["gripperActuation"]["schedule"].update(gripperIds=["different"]),
                   lambda source: source["gripperActuation"]["schedule"]["knots"][1].update(activation=[1.01]),
                   lambda source: source["gripperActuation"].update(accepted=True)]
        for index, attack in enumerate(attacks):
            changed = copy.deepcopy(source)
            attack(changed)
            bind_digest(changed)  # Type/semantic validation cannot rely on digest mismatch.
            with self.subTest(attack=index), self.assertRaises(ValueError):
                derive_grippers(changed, 1)
        changed = copy.deepcopy(source)
        changed["restMeters"][0][0] = .01
        with self.assertRaisesRegex(ValueError, "digest"):
            derive_grippers(changed, 1)
        changed = copy.deepcopy(source)
        changed["restMeters"] += [[2., 0., 0.], [3., 0., 0.], [2., 1., 0.]]
        changed["instanceOffsets"]["other"] = 3
        changed["triangles"] = [0, 1, 3, 3, 4, 5]
        bind_digest(changed)
        with self.assertRaisesRegex(ValueError, "cross instances"):
            derive_grippers(changed, 1)

    def test_schedule_sampling_is_stateless_held_bound_and_immutable(self):
        source, _, _, _ = fixture()
        source["gripperActuation"]["schedule"]["knots"] = [
            {"fraction": 0., "targetsMeters": [[100., -1., 2.]], "activation": [1.]},
            {"fraction": .75, "targetsMeters": [[100., 2., -1.]], "activation": [.25]},
            {"fraction": 1., "targetsMeters": [[100., 2., -1.]], "activation": [0.]}]
        record = derive_grippers(source, 4)
        for fraction in (1., .5, .625, 0., .75, .25):
            target, activation = parameters(record, fraction)
            self.assertEqual(target[0, 0], 100.)
            if fraction <= .75:
                np.testing.assert_array_equal(target, [[100., -1. + 4. * fraction, 2. - 4. * fraction]])
                np.testing.assert_array_equal(activation, [1. - fraction])
        expected = parameters(record, .5)
        source["gripperActuation"]["schedule"]["knots"][0]["targetsMeters"][0][0] = -100.
        target, activation = parameters(record, .5)
        target[:], activation[:] = 0., 0.
        for actual, wanted in zip(parameters(record, .5), expected):
            np.testing.assert_array_equal(actual, wanted)
        with self.assertRaises(AttributeError):
            record.ids = ("changed",)
        with self.assertRaises(AttributeError):
            del record.ids
        for fraction in (True, -.1, 1.1, .1, float("nan")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                parameters(record, fraction)

    def test_nonphysical_momentum_cannot_be_hidden_by_forged_tolerance(self):
        source, _, step, arguments = fixture()
        record = derive_grippers(source, 1)
        changed = list(arguments)
        changed[2] = np.ones((3, 3))
        damaged = copy.deepcopy(step)
        damaged["gripperMomentum"]["toleranceNs"] = 1e6
        with self.assertRaises(ValueError):
            verify_gripper_step(record, *changed, damaged)
        changed = list(arguments)
        changed[4] = [0., 1., 1.]
        with self.assertRaisesRegex(ValueError, "positive-mass"):
            verify_gripper_step(record, *changed, step)
        changed = list(arguments)
        changed[3] = arguments[3].copy()
        changed[3][0, 0] += 1e-12
        with self.assertRaisesRegex(ValueError, "velocity continuity"):
            verify_gripper_step(record, *changed, step)

    def test_decimal_cancellation_work_is_not_replaced_by_rounded_component_sum(self):
        source, _, step, arguments = fixture()
        active = float(np.nextafter(.25, np.inf))
        recipe = source["gripperActuation"]
        recipe["anchors"] = [dict(recipe["anchors"][0], id=f"g{i}", stiffnessNPerM=3.) for i in range(2)]
        recipe["schedule"]["gripperIds"] = ["g0", "g1"]
        recipe["schedule"]["knots"] = [
            {"fraction": 0., "targetsMeters": [[1., 0., 0.], [-1., 0., 0.]], "activation": [1., 1.]},
            {"fraction": 1., "targetsMeters": [[2., 0., 0.], [-2., 0., 0.]], "activation": [active, active]}]
        with localcontext() as context:
            context.prec = 150
            exact = Decimal.from_float(active)
            total, activation_work, release = [float(value) for value in (12 * exact - 3, 12 * exact - 12, 12 - 12 * exact)]
        self.assertNotEqual(total, 9. + activation_work)
        force, energy = 6. * active, 12. * active
        diagnostics = step["gripperDiagnostics"]
        diagnostics.update(gripperIds=["g0", "g1"], energyJoules=energy,
            anchorPositionsMeters=[[0., 0., 0.], [0., 0., 0.]],
            targetsMeters=[[2., 0., 0.], [-2., 0., 0.]], activation=[active, active], anchorWeightSums=[1., 1.],
            anchorForcesNewtons=[[force, 0., 0.], [-force, 0., 0.]],
            toolReactionsNewtons=[[-force, 0., 0.], [force, 0., 0.]],
            nodalForcesNewtons=np.zeros((3, 3)).tolist(), totalClothForceNewtons=[0., 0., 0.],
            totalToolReactionNewtons=[0., 0., 0.])
        step.update(gripperTargetsMeters=diagnostics["targetsMeters"], gripperActivation=[active, active], gripperEnergyJoules=energy)
        bound = math.fsum(math.ulp(value) for value in (total, 9., activation_work)) + math.ulp(math.fsum((9., activation_work)))
        step["energyBalance"].update(gripperBeforeJoules=3., gripperAfterJoules=energy,
            gripperFixedPositionAfterJoules=energy, gripperFixedParameterChangeJoules=0., gripperChangeJoules=total,
            gripperParameterWorkJoules=total, gripperTargetParameterWorkJoules=9.,
            gripperActivationParameterWorkJoules=activation_work, gripperActivationIncreaseWorkJoules=0.,
            gripperReleaseEnergyRemovedJoules=release, gripperParameterWorkComponentSumErrorBoundJoules=bound)
        for key in ("changeKgMPerS", "externalImpulseNs", "residualNs"):
            step["gripperMomentum"][key] = [0., 0., 0.]
        q = arguments[0]
        record = derive_grippers(source, 1)
        verify_gripper_step(record, q, q, np.zeros_like(q), np.zeros_like(q), np.ones(3), 0., 1., .5, step)
        step["energyBalance"]["gripperParameterWorkJoules"] = 9. + activation_work
        with self.assertRaisesRegex(ValueError, "exact gripper mismatch"):
            verify_gripper_step(record, q, q, np.zeros_like(q), np.zeros_like(q), np.ones(3), 0., 1., .5, step)

    def test_cancelled_rounded_forces_cannot_hide_a_false_stationary_state(self):
        source, _, step, arguments = fixture()
        goals = [[100., 0., 0.], [float(np.nextafter(-100., 0.)), 0., 0.], [-1.5625e-14, 0., 0.]]
        recipe = source["gripperActuation"]
        recipe["anchors"] = [dict(recipe["anchors"][0], id=f"g{i}", stiffnessNPerM=1e12) for i in range(3)]
        recipe["schedule"]["gripperIds"] = ["g0", "g1", "g2"]
        for knot in recipe["schedule"]["knots"]:
            knot.update(targetsMeters=goals, activation=[1., 1., 1.])
        force = 1e12 * np.array(goals)
        self.assertEqual(np.sum(force[:, 0]), 0.)  # Known naive binary64 failure.
        with localcontext() as context:
            context.prec = 150
            exact_force = sum(Decimal.from_float(point[0]) * Decimal(10) ** 12 for point in goals)
            self.assertGreater(abs(exact_force), Decimal(".001"))
        energy = float(.5e12 * np.sum(np.array(goals) ** 2))
        step.update(gripperTargetsMeters=goals, gripperActivation=[1., 1., 1.], gripperEnergyJoules=energy)
        step["gripperDiagnostics"].update(gripperIds=["g0", "g1", "g2"], energyJoules=energy,
            anchorPositionsMeters=np.zeros((3, 3)).tolist(), targetsMeters=goals,
            activation=[1., 1., 1.], anchorWeightSums=[1., 1., 1.],
            anchorForcesNewtons=force.tolist(), toolReactionsNewtons=(-force).tolist(),
            nodalForcesNewtons=np.zeros((3, 3)).tolist(), totalClothForceNewtons=[0., 0., 0.],
            totalToolReactionNewtons=[0., 0., 0.])
        for key in step["energyBalance"]:
            if key != "accepted":
                step["energyBalance"][key] = 0.
        for key in ("gripperBeforeJoules", "gripperAfterJoules", "gripperFixedPositionAfterJoules"):
            step["energyBalance"][key] = energy
        step["energyBalance"]["gripperParameterWorkComponentSumErrorBoundJoules"] = 4 * math.ulp(0.)
        for key in ("changeKgMPerS", "externalImpulseNs", "residualNs"):
            step["gripperMomentum"][key] = [0., 0., 0.]
        record, q = derive_grippers(source, 1), arguments[0]
        with self.assertRaisesRegex(ValueError, "free-cloth momentum balance failed"):
            verify_gripper_step(record, q, q, np.zeros_like(q), np.zeros_like(q), np.ones(3), 0., 1., .5, step)

    def test_verifier_imports_no_snapshot_solver_and_loads_without_registered_alias(self):
        path = Path(__file__).with_name("solver_gripper_replay.py")
        code = path.read_text()
        allowed = {"fractions", "hashlib", "json", "math", "numbers", "re", "numpy"}
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                self.assertTrue({name.name for name in node.names} <= allowed)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
        namespace = {"__name__": "independent_test_unregistered_verifier"}
        exec(compile(code, str(path), "exec"), namespace)
        source, run, _, arguments = fixture()
        record = namespace["derive_grippers"](source, 1)
        self.assertTrue(namespace["verify_initial"](record, arguments[0], run)["verified"])


if __name__ == "__main__":
    unittest.main()
