"""Independent sewing replay attacks and cross-implementation geometry checks."""

import ast
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from solver_bending import ElasticDihedralBending
from solver_distance_sewing import DistanceSewing
from solver_energy_balance import global_energy_transition
from solver_normal_sewing import NormalOffsetSewing
from solver_sewing_input import bind_sewing_activation
import solver_sewing_replay as verifier


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rehash(source):
    source["sewingActuation"]["sourceSha256"] = digest({k: v for k, v in source.items() if k != "sewingActuation"})
    return source


def fixture(mode="normal-offset"):
    source = {"restMeters": [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]] * 2,
        "triangles": [0, 1, 2, 3, 4, 5], "instanceOffsets": {"shell": 0, "facing": 3},
        "embeddedConstraints": {"constraints": []}, "scope": "synthetic captured JSON only"}
    row_ids = []
    for row_index, (positive, negative) in enumerate((((.2, .3, .5), (.125, .25, .625)),
                                                     ((.7, .1, .2), (.3, .4, .3)))):
        row_ids.append("row:" + digest(["registration", 1, row_index, 1]))
        source["embeddedConstraints"]["constraints"].append({"registrationId": "registration", "memberIndex": 1,
            "fraction": float(row_index), "complianceMPerN": .003,
            "sourceSamples": [{"instanceId": name, "pathName": "synthetic-path", "arcMm": row_index * 10.,
                "weights": [{"vertex": i, "weight": value} for i, value in enumerate(weights)]}
                for name, weights in (("shell", positive), ("facing", negative))],
            "terms": [{"instanceId": name, "vertex": i, "coefficient": sign * value}
                for name, sign, weights in (("shell", 1, positive), ("facing", -1, negative))
                for i, value in enumerate(weights)]})
    if mode == "normal-offset":
        source["sewingFrames"] = {"faces": [[3, 4, 5], [3, 4, 5]], "sides": [-1, 1],
            "bindings": [{"rowId": identity, "instanceId": "facing", "triangleIndex": 1, "side": side}
                         for identity, side in zip(row_ids, (-1, 1))]}
    source["sewingActuation"] = {"profile": "captured-sewing-activation-v1", "accepted": False,
        "sourceSha256": "", "mode": mode,
        "initialTargetsMeters": [[.002, -.001, .003], [.003, .002, -.001]] if mode == "vector" else [.002, .003],
        "finalTargetsMeters": [[.001, .002, .001], [.002, -.001, .003]] if mode == "vector" else [.001, .002],
        "schedule": {"profile": "sewing-row-activation-v1", "rowIds": row_ids,
            "knots": [{"fraction": 0., "activation": [0., .25]},
                      {"fraction": .5, "activation": [.25, .5]},
                      {"fraction": 1., "activation": [1., 1.]}]}}
    return rehash(source)


def production(source):
    mode = source["sewingActuation"]["mode"]
    controls, binding = bind_sewing_activation(source, 4, sewing_mode=mode)
    matrix = np.zeros((len(controls.rows), len(source["restMeters"])))
    for i, row in enumerate(controls.rows):
        for vertex, weight in row.items():
            matrix[i, vertex] = weight
    solver = SimpleNamespace(sewing=csr_matrix(matrix), compliance=controls.compliance, sewing_mode=mode,
        sewing_sides=controls.sides, mass=np.ones(len(matrix.T)), active=np.ones(len(matrix.T), dtype=bool),
        poses=np.empty((0, 2, 2)), faces=np.empty((0, 3), dtype=int), areas=np.empty(0), materials=np.empty((0, 3)),
        bending=ElasticDihedralBending(len(matrix.T), np.empty((0, 4), dtype=int), [], [], []))
    if mode == "distance":
        solver.sewing_potential = lambda t, *, activation=None: DistanceSewing(solver.sewing, t, solver.compliance, activation=activation)
    elif mode == "normal-offset":
        solver.sewing_potential = lambda t, *, activation=None: NormalOffsetSewing(solver.sewing, t, solver.compliance,
            controls.frame_faces, controls.sides, activation=activation)
    return controls, binding, solver, matrix


def diagnostics(solver, q, target, active):
    mask = active > 0
    errors = [None] * len(active)
    if solver.sewing_mode == "vector":
        raw = solver.sewing[mask] @ q - target[mask]
        residual = raw * np.sqrt(active[mask, None] / solver.compliance)
        values = np.max(np.abs(raw), axis=1)
    else:
        potential = solver.sewing_potential(target, activation=active)
        residual = potential.residual(q)
        if solver.sewing_mode == "distance":
            values = abs(potential.geometry(q)[1][mask] - target[mask])
        else:
            normal = potential.geometry(q)[2][mask]
            values = np.max(np.abs(solver.sewing[mask] @ q - (target[mask] * solver.sewing_sides[mask])[:, None] * normal), axis=1)
    for index, value in zip(np.flatnonzero(mask), values):
        errors[index] = float(value)
    return {"sewingMode": solver.sewing_mode, "sewingActivationExplicit": True,
        "sewingActivationLimitations": verifier.LIMITATIONS, "sewingActivation": active.tolist(),
        "activeSewingRows": np.flatnonzero(mask).tolist(), "pendingSewingRows": np.flatnonzero(~mask).tolist(),
        "sewingRowTargetErrorsM": errors, "sewingTargetErrorM": max((v for v in errors if v is not None), default=0.),
        "sewingTargetErrorMetric": verifier.METRIC, "sewingJoules": float(.5 * np.sum(residual ** 2))}


def states(seed=123):
    rng = np.random.default_rng(seed)
    q = np.array([[0., 0., 0.], [.1, 0., .01], [0., .1, -.02],
                  [.01, .005, .003], [.11, .015, .013], [.02, .105, -.017]])
    q += rng.normal(size=q.shape) * .001
    return q, q + rng.normal(size=q.shape) * .0001


def evidence(source, start=0., end=.25, seed=123):
    controls, binding, solver, matrix = production(source)
    q0, q1 = states(seed)
    old_target = controls.initial_targets + start * (controls.final_targets - controls.initial_targets)
    new_target = controls.final_targets if end == 1 else controls.initial_targets + end * (controls.final_targets - controls.initial_targets)
    old_a, new_a = controls.parameters(start), controls.parameters(end)
    step = diagnostics(solver, q1, new_target, new_a)
    step["energyBalance"] = global_energy_transition(solver, q0, q1, np.zeros_like(q0), q1 - q0,
        old_target, new_target, 1., previous_sewing_activation=old_a, sewing_activation=new_a)
    initial = diagnostics(solver, q0, controls.initial_targets, controls.parameters(0))
    report = {"sewingActuation": copy.deepcopy(source["sewingActuation"]), "sewingBinding": binding,
        "initialSewingTargetsMeters": controls.initial_targets.tolist(), "initialSewingActivation": controls.parameters(0).tolist(),
        "initialSewingEnergyJoules": initial["sewingJoules"], "initialSewingRowTargetErrorsM": initial["sewingRowTargetErrorsM"],
        "initialSewingTargetErrorMetric": verifier.METRIC}
    return solver, matrix, q0, q1, old_target, new_target, new_a, step, report


class SewingReplayTests(unittest.TestCase):
    def test_module_imports_only_stdlib_and_numpy(self):
        tree = ast.parse(Path(verifier.__file__).read_text())
        imports = {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertLessEqual(imports, {"fractions", "hashlib", "json", "math", "numpy"})

    def test_varied_sparse_weights_match_captured_geometry_and_exact_work(self):
        for mode in ("vector", "distance", "normal-offset"):
            source = fixture(mode)
            record = verifier.derive_sewing(source, 4, sewing_mode=mode)
            for seed in range(20):
                with self.subTest(mode=mode, seed=seed):
                    solver, _, q0, q1, old, new, active, step, report = evidence(source, seed=seed)
                    verifier.verify_initial(record, q0, report)
                    gradient, checked = verifier.verify_sewing_step(record, q0, q1, 0., .25, old, new, step)
                    if mode == "vector":
                        expected = solver.sewing.T @ (active[:, None] * (solver.sewing @ q1 - new) / solver.compliance)
                    else:
                        expected = solver.sewing_potential(new, activation=active).gradient(q1).reshape((-1, 3))
                    np.testing.assert_allclose(gradient, expected, rtol=2e-13, atol=2e-14)
                    self.assertEqual(checked["sewingParameterWorkJoules"], step["energyBalance"]["sewingParameterWorkJoules"])
                    self.assertFalse(checked["accepted"])

    def test_normal_frame_reactions_match_independent_complex_step_energy(self):
        source = fixture()
        record = verifier.derive_sewing(source, 4, sewing_mode="normal-offset")
        solver, matrix, q0, q1, old, new, active, step, _ = evidence(source)
        gradient, _ = verifier.verify_sewing_step(record, q0, q1, 0., .25, old, new, step)

        def energy(q):
            area = np.cross(q[4] - q[3], q[5] - q[3])
            normal = area / np.sqrt(sum(area * area))
            errors = matrix @ q - (new * [-1, 1])[:, None] * normal
            return np.sum(active[:, None] * errors * errors) / (2 * solver.compliance)

        numerical = np.zeros_like(q1)
        for index in range(q1.size):
            q = q1.astype(complex)
            q.flat[index] += 1j * 1e-25
            numerical.flat[index] = energy(q).imag / 1e-25
        np.testing.assert_allclose(gradient, numerical, rtol=2e-13, atol=3e-14)
        np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=3e-14)
        np.testing.assert_allclose(np.cross(q1, gradient).sum(axis=0), 0., atol=3e-15)

    def test_all_energy_fields_and_tiny_work_tampering_reject(self):
        source = fixture("distance")
        record = verifier.derive_sewing(source, 4, sewing_mode="distance")
        _, _, q0, q1, old, new, _, step, _ = evidence(source)
        for key in [key for key in step["energyBalance"] if key.startswith("sewing")]:
            for attack in ("missing", "ulp"):
                changed = copy.deepcopy(step)
                if attack == "missing":
                    del changed["energyBalance"][key]
                else:
                    changed["energyBalance"][key] = float(np.nextafter(changed["energyBalance"][key], np.inf))
                with self.subTest(key=key, attack=attack), self.assertRaises(ValueError):
                    verifier.verify_sewing_step(record, q0, q1, 0., .25, old, new, changed)

    def test_sub_absolute_tolerance_target_work_cannot_be_zeroed_or_changed_one_ulp(self):
        for mode in ("vector", "distance", "normal-offset"):
            source = fixture(mode)
            recipe = source["sewingActuation"]
            first = np.array(recipe["initialTargetsMeters"])
            last = first.copy()
            last.flat[0] = np.nextafter(last.flat[0], np.inf)
            recipe["finalTargetsMeters"] = last.tolist()
            recipe["schedule"]["knots"] = [{"fraction": 0., "activation": [.25, .5]},
                                             {"fraction": 1., "activation": [.25, .5]}]
            record = verifier.derive_sewing(source, 4, sewing_mode=mode)
            controls, _, solver, _ = production(source)
            q, _ = states()
            active = controls.parameters(1.)
            step = diagnostics(solver, q, last, active)
            step["energyBalance"] = global_energy_transition(solver, q, q, np.zeros_like(q), np.zeros_like(q), first, last,
                1., previous_sewing_activation=active, sewing_activation=active)
            _, checked = verifier.verify_sewing_step(record, q, q, 0., 1., first, last, step)
            expected = checked["sewingTargetParameterWorkJoules"]
            self.assertGreater(abs(expected), 0.)
            self.assertLess(abs(expected), 1e-14)
            for wrong in (0., float(np.nextafter(expected, np.inf))):
                altered = copy.deepcopy(step)
                altered["energyBalance"]["sewingTargetParameterWorkJoules"] = wrong
                with self.subTest(mode=mode, wrong=wrong), self.assertRaises(ValueError):
                    verifier.verify_sewing_step(record, q, q, 0., 1., first, last, altered)

    def test_all_pending_energy_is_structurally_zero_without_subnormal_tolerance(self):
        for mode in ("vector", "distance", "normal-offset"):
            source = fixture(mode)
            for knot in source["sewingActuation"]["schedule"]["knots"]:
                knot["activation"] = [0., 0.]
            record = verifier.derive_sewing(source, 4, sewing_mode=mode)
            _, _, q0, q1, old, new, _, step, report = evidence(source)
            self.assertTrue(verifier.verify_initial(record, q0, report)["verified"])
            gradient, checked = verifier.verify_sewing_step(record, q0, q1, 0., .25, old, new, step)
            np.testing.assert_array_equal(gradient, 0.)
            self.assertTrue(checked["verified"])
            for forged in (float(np.nextafter(0., 1.)), float(np.nextafter(0., -1.))):
                altered_initial, altered_step = copy.deepcopy(report), copy.deepcopy(step)
                altered_initial["initialSewingEnergyJoules"] = forged
                altered_step["sewingJoules"] = forged
                with self.subTest(mode=mode, forged=forged, initial=True), self.assertRaises(ValueError):
                    verifier.verify_initial(record, q0, altered_initial)
                with self.subTest(mode=mode, forged=forged, initial=False), self.assertRaises(ValueError):
                    verifier.verify_sewing_step(record, q0, q1, 0., .25, old, new, altered_step)

    def test_sampled_residual_energy_is_separate_from_original_input_work(self):
        # An exactly satisfied rounded CSR target need not equal the original
        # binary-input rational anchor. Both diagnostics have valid, distinct
        # meanings; the exact helper work must retain the latter.
        source = fixture("vector")
        _, _, solver, _ = production(source)
        q, _ = states(7)
        target = solver.sewing @ q
        recipe = source["sewingActuation"]
        recipe["initialTargetsMeters"] = target.tolist()
        recipe["finalTargetsMeters"] = target.tolist()
        recipe["schedule"]["knots"] = [{"fraction": 0., "activation": [.25, .5]},
                                         {"fraction": 1., "activation": [.25, .5]}]
        controls, binding, solver, _ = production(source)
        record = verifier.derive_sewing(source, 4, sewing_mode="vector")
        active = controls.parameters(0.)
        step = diagnostics(solver, q, target, active)
        step["energyBalance"] = global_energy_transition(solver, q, q, np.zeros_like(q), np.zeros_like(q),
            target, target, 1., previous_sewing_activation=active, sewing_activation=active)
        report = {"sewingActuation": copy.deepcopy(recipe), "sewingBinding": binding,
            "initialSewingTargetsMeters": target.tolist(), "initialSewingActivation": active.tolist(),
            "initialSewingEnergyJoules": step["sewingJoules"],
            "initialSewingRowTargetErrorsM": step["sewingRowTargetErrorsM"],
            "initialSewingTargetErrorMetric": verifier.METRIC}
        self.assertEqual(step["sewingJoules"], 0.)
        original = step["energyBalance"]["sewingAfterJoules"]
        self.assertGreater(original, 1e-40)
        initial = verifier.verify_initial(record, q, report)
        _, checked = verifier.verify_sewing_step(record, q, q, 0., 1., target, target, step)
        self.assertEqual(initial["initialEnergyJoules"], 0.)
        self.assertEqual(initial["initialOriginalInputEnergyJoules"], original)
        self.assertEqual(checked["sampledSewingJoules"], 0.)
        self.assertEqual(checked["sewingFixedParameterChangeJoules"], 0.)
        altered_initial, altered_step = copy.deepcopy(report), copy.deepcopy(step)
        altered_initial["initialSewingEnergyJoules"] = original
        altered_step["sewingJoules"] = original
        with self.assertRaises(ValueError):
            verifier.verify_initial(record, q, altered_initial)
        with self.assertRaises(ValueError):
            verifier.verify_sewing_step(record, q, q, 0., 1., target, target, altered_step)
        altered_step = copy.deepcopy(step)
        altered_step["energyBalance"]["sewingAfterJoules"] = 0.
        with self.assertRaises(ValueError):
            verifier.verify_sewing_step(record, q, q, 0., 1., target, target, altered_step)

    def test_controls_pending_errors_and_initial_binding_tampering_reject(self):
        source = fixture()
        record = verifier.derive_sewing(source, 4, sewing_mode="normal-offset")
        _, _, q0, q1, old, new, _, step, report = evidence(source)
        for key, value in (("sewingActivation", [True, .375]), ("activeSewingRows", [False, 1]),
                           ("pendingSewingRows", [0]), ("sewingActivationExplicit", 1),
                           ("sewingRowTargetErrorsM", [None, None]), ("sewingJoules", 0.),
                           ("sewingTargetErrorMetric", "weighted error")):
            changed = dict(step, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                verifier.verify_sewing_step(record, q0, q1, 0., .25, old, new, changed)
        for key in report:
            changed = copy.deepcopy(report)
            del changed[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                verifier.verify_initial(record, q0, changed)
        changed = copy.deepcopy(report)
        changed["sewingBinding"]["rowBindings"][0]["canonicalTerms"][0]["coefficient"] += 1e-12
        with self.assertRaises(ValueError):
            verifier.verify_initial(record, q0, changed)

    def test_full_source_hash_raw_types_order_and_frames_fail_closed(self):
        source = fixture()
        attacks = []
        changed = copy.deepcopy(source); changed["scope"] += " altered"; attacks.append(changed)
        changed = copy.deepcopy(source); changed["triangles"][0] = False; attacks.append(rehash(changed))
        changed = copy.deepcopy(source); changed["embeddedConstraints"]["constraints"].reverse(); attacks.append(rehash(changed))
        changed = copy.deepcopy(source); changed["embeddedConstraints"]["constraints"][0]["terms"][0]["coefficient"] += 1e-8; attacks.append(rehash(changed))
        changed = copy.deepcopy(source); changed["sewingFrames"]["faces"][0].reverse(); attacks.append(rehash(changed))
        changed = copy.deepcopy(source); changed["sewingFrames"]["bindings"][0]["instanceId"] = "shell"; attacks.append(rehash(changed))
        changed = copy.deepcopy(source); changed["sewingActuation"]["schedule"]["knots"][1]["activation"][1] = .1; attacks.append(changed)
        changed = copy.deepcopy(source); changed["sewingActuation"]["initialTargetsMeters"][0] = 0.; attacks.append(changed)
        changed = copy.deepcopy(source); changed["provenance"] = {"phasePlanSha256": "0" * 64}; attacks.append(rehash(changed))
        for index, changed in enumerate(attacks):
            with self.subTest(attack=index), self.assertRaises(ValueError):
                verifier.derive_sewing(changed, 4, sewing_mode="normal-offset")

    def test_record_and_outputs_are_isolated_and_original_fraction_sampling_matches(self):
        source = fixture("vector")
        record = verifier.derive_sewing(source, 4, sewing_mode="vector")
        controls, _ = bind_sewing_activation(source, 4, sewing_mode="vector")
        for fraction in (0, Fraction(1, 8), Fraction(7, 16), Fraction(5, 8), 1):
            np.testing.assert_array_equal(verifier.parameters(record, fraction), controls.parameters(fraction))
        for progress in (0., .12345, .5, 1.):
            expected = controls.final_targets if progress == 1 else controls.initial_targets + progress * (controls.final_targets - controls.initial_targets)
            np.testing.assert_array_equal(verifier.targets(record, progress), expected)
        source["sewingActuation"]["schedule"]["knots"][0]["activation"] = [1., 1.]
        result = verifier.parameters(record, 0.); result[:] = 1.
        np.testing.assert_array_equal(verifier.parameters(record, 0.), [0., .25])
        with self.assertRaises(AttributeError):
            record.mode = "distance"
        with self.assertRaises(AttributeError):
            del record.rows


if __name__ == "__main__":
    unittest.main()
