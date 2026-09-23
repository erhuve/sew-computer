"""Current, independent verification of captured virtual material actuators.

This module deliberately imports no captured solver code. Fractions evaluate
the exact binary-input quadratic, including cancelling parameter work. The
returned gradient is independently assembled for the enclosing stationarity
check; no construction, tool-collision, or continuous-work proof is claimed.
"""
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
import re

import numpy as np


PROFILE = "independent-source-material-gripper-replay-v1"
INPUT_PROFILE = "captured-material-grippers-v1"
POTENTIAL_PROFILE = "source-material-compliant-grippers-v1"
SCHEDULE_PROFILE = "material-gripper-target-activation-v1"
_EPS = np.finfo(float).eps
_TINY = float(np.nextafter(0., 1.))
_BINDING_SCOPE = ("Canonical material-triangle anchors; complete canonical digest and captured source lineage remain in the enclosing run manifest. "
                  "Targets are virtual controls, not collision geometry or construction proof.")
_MOMENTUM_SCOPE = ("Backward-Euler force at the new state on free cloth; virtual gripper impulse, "
                   "not isolated-cloth momentum conservation")


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Gripper evidence must contain finite canonical JSON values") from error


def _identity(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value) is not None


def _boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b" or (value.dtype.kind == "O" and any(_boolean(item) for item in value.flat))
    return isinstance(value, (list, tuple)) and any(_boolean(item) for item in value)


def _array(value, shape, label, *, bound=None, integers=False):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {label}") from error
    if (_boolean(value) or raw.shape != shape or raw.dtype.kind not in ("iu" if integers else "fiu")
            or not np.isfinite(raw).all() or (bound is not None and np.any(np.abs(raw) > bound))):
        raise ValueError(f"Invalid finite bounded {label}")
    return raw.astype(np.int64 if integers else float, copy=True)


def _number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"Invalid finite {label}")
    return float(value)


def _fraction(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, Fraction)):
        raise ValueError("Gripper fractions must be bounded dyadic numbers")
    try:
        result = Fraction(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Finite gripper fraction required") from error
    if not 0 <= result <= 1 or result.denominator > 2 ** 40 or result.denominator & (result.denominator - 1):
        raise ValueError("Gripper fractions must lie on the bounded dyadic grid")
    return result


class GripperRecord:
    """Immutable tuples only; safe for loading without a sys.modules alias."""
    __slots__ = ("rest", "faces", "vertex_ids", "anchors", "ids", "fractions", "targets",
                 "activation", "mesh_sha256", "recipe_json")

    def __setattr__(self, key, value):
        raise AttributeError("Independent gripper source bindings are immutable")

    def __delattr__(self, key):
        raise AttributeError("Independent gripper source bindings are immutable")


def derive_grippers(source, subdivisions):
    """Validate the exact captured mesh, ordered material bindings and schedule."""
    if not isinstance(source, dict) or type(subdivisions) is not int or not 1 <= subdivisions <= 4096 or subdivisions & (subdivisions - 1):
        raise ValueError("Captured gripper source and bounded power-of-two subdivisions required")
    try:
        recipe = source["gripperActuation"]
        mesh = {key: source[key] for key in ("restMeters", "triangles", "instanceOffsets")}
    except KeyError as error:
        raise ValueError("Missing captured gripper source field") from error
    digest = hashlib.sha256(_json(mesh).encode()).hexdigest()
    if (not isinstance(recipe, dict) or set(recipe) != {"profile", "accepted", "meshSha256", "anchors", "schedule"}
            or recipe["profile"] != INPUT_PROFILE or recipe["accepted"] is not False or recipe["meshSha256"] != digest):
        raise ValueError("Gripper recipe must match the exact captured mesh digest and unaccepted profile")
    try:
        vertex_count = len(mesh["restMeters"])
        raw_faces = np.asarray(mesh["triangles"])
    except (TypeError, ValueError) as error:
        raise ValueError("Canonical gripper mesh required") from error
    if not 3 <= vertex_count <= 25000 or raw_faces.ndim not in (1, 2) or raw_faces.size % 3:
        raise ValueError("Canonical gripper mesh budget or shape invalid")
    if raw_faces.ndim == 2 and raw_faces.shape[1] != 3:
        raise ValueError("Canonical faces must have three vertices")
    rest = _array(mesh["restMeters"], (vertex_count, 3), "source rest positions", bound=100.)
    faces = _array(mesh["triangles"], raw_faces.shape, "canonical faces", integers=True).reshape((-1, 3))
    if (not 1 <= len(faces) <= 50000 or np.any(faces < 0) or np.any(faces >= vertex_count)
            or any(len(set(face)) != 3 for face in faces)
            or len({tuple(sorted(face)) for face in faces}) != len(faces)):
        raise ValueError("Distinct bounded canonical face identities required")
    offsets = mesh["instanceOffsets"]
    if (not isinstance(offsets, dict) or not 1 <= len(offsets) <= 64
            or any(not _identity(key) or type(value) is not int or not 0 <= value < vertex_count for key, value in offsets.items())):
        raise ValueError("Explicit bounded physical instance offsets required")
    ordered = sorted(offsets.items(), key=lambda item: item[1])
    if ordered[0][1] != 0 or len(set(offsets.values())) != len(offsets):
        raise ValueError("Physical instances must partition the captured vertices")
    vertex_ids = []
    for index, (identity, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else vertex_count
        if end - start < 3:
            raise ValueError("Each physical instance requires at least three vertices")
        vertex_ids.extend([identity] * (end - start))
    if any(len({vertex_ids[vertex] for vertex in face}) != 1 for face in faces):
        raise ValueError("A canonical material face cannot cross instances")
    anchors = recipe["anchors"]
    if not isinstance(anchors, list) or not 1 <= len(anchors) <= 256:
        raise ValueError("Bounded nonempty material anchors required")
    bindings, ids = [], []
    for anchor in anchors:
        if (not isinstance(anchor, dict) or set(anchor) != {"id", "instanceId", "triangleIndex", "weights", "stiffnessNPerM"}
                or not _identity(anchor["id"]) or anchor["id"] in ids or not _identity(anchor["instanceId"])
                or type(anchor["triangleIndex"]) is not int or not 0 <= anchor["triangleIndex"] < len(faces)):
            raise ValueError("Unique ordered material anchor identities required")
        triangle = anchor["triangleIndex"]
        if any(vertex_ids[vertex] != anchor["instanceId"] for vertex in faces[triangle]):
            raise ValueError("Gripper support differs from its declared source instance")
        weights = _array(anchor["weights"], (3,), "barycentric weights", bound=1.)
        stiffness = _number(anchor["stiffnessNPerM"], "gripper stiffness")
        if np.any(weights < 0) or abs(math.fsum(weights) - 1.) > 1e-12 or not 0 < stiffness <= 1e12:
            raise ValueError("Unmodified normalized weights and positive bounded stiffness required")
        bindings.append((anchor["id"], anchor["instanceId"], triangle, tuple(weights), stiffness))
        ids.append(anchor["id"])
    schedule = recipe["schedule"]
    if (not isinstance(schedule, dict) or set(schedule) != {"profile", "gripperIds", "knots"}
            or schedule["profile"] != SCHEDULE_PROFILE or schedule["gripperIds"] != ids
            or not isinstance(schedule["knots"], list) or not 2 <= len(schedule["knots"]) <= 65):
        raise ValueError("Explicit schedule with the same ordered gripper identities required")
    fractions, targets, activation = [], [], []
    for knot in schedule["knots"]:
        if not isinstance(knot, dict) or set(knot) != {"fraction", "targetsMeters", "activation"}:
            raise ValueError("Explicit gripper target and activation knots required")
        fraction = _fraction(knot["fraction"])
        target = _array(knot["targetsMeters"], (len(ids), 3), "schedule targets", bound=100.)
        active = _array(knot["activation"], (len(ids),), "schedule activation", bound=1.)
        if (fraction * subdivisions).denominator != 1 or (fractions and fraction <= fractions[-1]) or np.any(active < 0):
            raise ValueError("Ordered initial-grid knots and activation in [0,1] required")
        fractions.append(fraction)
        targets.append(tuple(tuple(point) for point in target))
        activation.append(tuple(active))
    if fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError("Gripper controls must span exactly [0,1]")
    record = GripperRecord()
    values = {"rest": tuple(tuple(point) for point in rest), "faces": tuple(tuple(int(i) for i in face) for face in faces),
              "vertex_ids": tuple(vertex_ids), "anchors": tuple(bindings), "ids": tuple(ids),
              "fractions": tuple(fractions), "targets": tuple(targets), "activation": tuple(activation),
              "mesh_sha256": digest, "recipe_json": _json(recipe)}
    for key, value in values.items():
        object.__setattr__(record, key, value)
    return record


def parameters(record, fraction):
    fraction = _fraction(fraction)
    if fraction in record.fractions:
        index = record.fractions.index(fraction)
        return np.array(record.targets[index]), np.array(record.activation[index])
    upper = next(index for index, knot in enumerate(record.fractions) if knot > fraction)
    amount = float((fraction - record.fractions[upper - 1]) / (record.fractions[upper] - record.fractions[upper - 1]))
    # The declared v1 binary64 sampling operation uses an endpoint difference,
    # a product and an addition. Held targets are preserved exactly.
    def interpolate(first, second):
        return float(first) + amount * (float(second) - float(first))
    targets = [[interpolate(a, b) for a, b in zip(first, second)]
               for first, second in zip(record.targets[upper - 1], record.targets[upper])]
    active = [interpolate(a, b) for a, b in zip(record.activation[upper - 1], record.activation[upper])]
    return np.array(targets), np.array(active)


def _close(observed, expected, label, scale=None, *, operations=64, floor=0.):
    expected = np.asarray(expected, dtype=float)
    observed = _array(observed, expected.shape, label)
    scale = np.abs(expected) if scale is None else np.asarray(scale, dtype=float)
    # No unit-sized absolute tolerance: small forces and tiny work remain
    # testable. Absolute floors only cover binary64 subnormal underflow.
    tolerance = operations * _EPS * scale + operations * _TINY + floor
    if not np.isfinite(expected).all() or not np.isfinite(tolerance).all() or np.any(np.abs(observed - expected) > tolerance):
        raise ValueError(f"Independent gripper mismatch: {label}")


def _exact_number(observed, expected, label):
    if _number(observed, label) != expected:
        raise ValueError(f"Independent exact gripper mismatch: {label}")


def _cross(first, second):
    return [first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0]]


def _evaluate(record, positions, targets, activation):
    q = _array(positions, (len(record.rest), 3), "gripper positions", bound=100.)
    targets = _array(targets, (len(record.ids), 3), "gripper targets", bound=100.)
    activation = _array(activation, (len(record.ids),), "gripper activation", bound=1.)
    if np.any(activation < 0):
        raise ValueError("Negative gripper activation")
    zero = Fraction(0)
    nodal = [[zero] * 3 for _ in q]
    nodal_scale = np.zeros_like(q)
    anchors, forces, energy_terms, errors, coefficients = [], [], [], [], []
    for (identity, instance, triangle, weights, stiffness), target, active in zip(record.anchors, targets, activation):
        if active > 0 and stiffness * active == 0:
            raise ValueError("Positive gripper activation coefficient underflows")
        face = record.faces[triangle]
        anchor = [sum((Fraction(float(w)) * Fraction(float(q[v, axis])) for v, w in zip(face, weights)), zero)
                  for axis in range(3)]
        error = [value - Fraction(float(goal)) for value, goal in zip(anchor, target)]
        coefficient = Fraction(stiffness) * Fraction(float(active))
        force = [-coefficient * value for value in error]
        for vertex, weight in zip(face, weights):
            for axis in range(3):
                contribution = Fraction(float(weight)) * force[axis]
                nodal[vertex][axis] += contribution
                nodal_scale[vertex, axis] += abs(float(contribution))
        anchors.append(anchor)
        forces.append(force)
        errors.append(error)
        coefficients.append(coefficient)
        energy_terms.append(coefficient * sum((value * value for value in error), zero) / 2)
    cloth = [sum((point[axis] for point in nodal), zero) for axis in range(3)]
    tool = [-sum((force[axis] for force in forces), zero) for axis in range(3)]
    cloth_cross = [_cross([Fraction(float(value)) for value in point], force) for point, force in zip(q, nodal)]
    tool_cross = [_cross([Fraction(float(value)) for value in target], [-value for value in force])
                  for target, force in zip(targets, forces)]
    cloth_torque = [sum((point[axis] for point in cloth_cross), zero) for axis in range(3)]
    tool_torque = [sum((point[axis] for point in tool_cross), zero) for axis in range(3)]
    def array(values):
        return np.array([[float(item) for item in row] for row in values])
    anchor_values, force_values, nodal_values = array(anchors), array(forces), array(nodal)
    diagnostics = {"profile": POTENTIAL_PROFILE, "gripperIds": list(record.ids), "energyJoules": float(sum(energy_terms, zero)),
        "anchorPositionsMeters": anchor_values, "targetsMeters": targets, "activation": activation,
        "anchorWeightSums": [math.fsum(anchor[3]) for anchor in record.anchors], "weightSumAdmissionTolerance": 1e-12,
        "anchorForcesNewtons": force_values, "toolReactionsNewtons": -force_values, "nodalForcesNewtons": nodal_values,
        "totalClothForceNewtons": [float(x) for x in cloth], "totalToolReactionNewtons": [float(x) for x in tool],
        "clothTorqueNewtonMeters": [float(x) for x in cloth_torque], "toolTorqueNewtonMeters": [float(x) for x in tool_torque],
        "netForceResidualNewtons": [float(a + b) for a, b in zip(cloth, tool)],
        "netTorqueResidualNewtonMeters": [float(a + b) for a, b in zip(cloth_torque, tool_torque)]}
    force_scale = np.sum(np.abs(force_values), axis=0) + np.sum(nodal_scale, axis=0)
    # Cross-product magnitudes before cancellation bound both product/sum
    # rounding and propagation of the independently bounded nodal force error.
    def torque_scale(points, force_magnitudes):
        p = np.abs(points)
        return np.sum(np.array([p[:, 1] * force_magnitudes[:, 2] + p[:, 2] * force_magnitudes[:, 1],
                                p[:, 2] * force_magnitudes[:, 0] + p[:, 0] * force_magnitudes[:, 2],
                                p[:, 0] * force_magnitudes[:, 1] + p[:, 1] * force_magnitudes[:, 0]]), axis=1)
    cloth_scale = torque_scale(q, nodal_scale)
    tool_scale = torque_scale(targets, np.abs(force_values))
    scales = {"nodalForcesNewtons": nodal_scale, "totalClothForceNewtons": np.sum(nodal_scale, axis=0),
        "totalToolReactionNewtons": np.sum(np.abs(force_values), axis=0),
        "netForceResidualNewtons": force_scale, "clothTorqueNewtonMeters": cloth_scale,
        "toolTorqueNewtonMeters": tool_scale, "netTorqueResidualNewtonMeters": cloth_scale + tool_scale}
    return {"diagnostics": diagnostics, "scales": scales, "gradient": -nodal_values,
            "energy": sum(energy_terms, zero), "energy_terms": energy_terms, "errors": errors,
            "coefficients": coefficients, "cloth_force": cloth,
            "underflow": 64 * _TINY * max(1., sum(float(x) for x in coefficients))}


def verify_initial(record, positions, run_report):
    """Require the captured binding and initially stored potential energy."""
    if not isinstance(run_report, dict) or run_report.get("accepted") is not False:
        raise ValueError("Unaccepted captured gripper run report required")
    try:
        if _json(run_report["gripperActuation"]) != record.recipe_json:
            raise ValueError("Captured gripper declaration differs from source")
        binding = run_report["gripperBinding"]
        if (not isinstance(binding, dict) or set(binding) != {"profile", "accepted", "meshSha256", "anchorBindings", "scope"}
                or binding["profile"] != INPUT_PROFILE or binding["accepted"] is not False
                or binding["meshSha256"] != record.mesh_sha256 or binding["scope"] != _BINDING_SCOPE
                or not isinstance(binding["anchorBindings"], list) or len(binding["anchorBindings"]) != len(record.ids)):
            raise ValueError("Complete source-bound gripper binding evidence required")
        for declared, observed in zip(json.loads(record.recipe_json)["anchors"], binding["anchorBindings"]):
            if not isinstance(observed, dict) or set(observed) != set(declared) | {"canonicalVertices", "restAnchorMeters"}:
                raise ValueError("Complete ordered gripper anchor binding required")
            if _json({key: observed[key] for key in declared}) != _json(declared):
                raise ValueError("Gripper source anchor declaration changed")
            face = record.faces[declared["triangleIndex"]]
            if _json(observed["canonicalVertices"]) != _json(list(face)):
                raise ValueError("Resolved canonical gripper vertices changed")
            terms = np.array(record.rest)[list(face)] * np.array(declared["weights"])[:, None]
            expected = [float(sum((Fraction(float(w)) * Fraction(float(record.rest[v][axis]))
                                  for v, w in zip(face, declared["weights"])), Fraction(0))) for axis in range(3)]
            _close(observed["restAnchorMeters"], expected, "rest anchor", np.sum(np.abs(terms), axis=0))
        targets, active = parameters(record, 0.)
        evaluation = _evaluate(record, positions, targets, active)
        if not np.array_equal(_array(run_report["initialGripperTargetsMeters"], targets.shape, "initial targets"), targets):
            raise ValueError("Initial gripper targets differ from schedule")
        if not np.array_equal(_array(run_report["initialGripperActivation"], active.shape, "initial activation"), active):
            raise ValueError("Initial gripper activation differs from schedule")
        _close(run_report["initialGripperEnergyJoules"], float(evaluation["energy"]), "initial gripper energy",
               floor=evaluation["underflow"])
    except KeyError as error:
        raise ValueError(f"Missing initial gripper evidence: {error.args[0]}") from error
    return {"verified": True, "profile": PROFILE, "meshSha256": record.mesh_sha256,
            "gripperCount": len(record.ids), "initialEnergyJoules": float(evaluation["energy"]),
            "scope": "Captured virtual source-material controls only; not a construction or tool-collision proof."}


def verify_gripper_step(record, previous, positions, previous_velocity, velocity, mass,
                        start_fraction, end_fraction, duration, step_report):
    """Check a saved accepted transition and independently return its gradient."""
    start, end = _fraction(start_fraction), _fraction(end_fraction)
    duration = _number(duration, "step duration")
    if start >= end or duration <= 0 or not isinstance(step_report, dict):
        raise ValueError("Positive ordered gripper transition required")
    shape = (len(record.rest), 3)
    previous = _array(previous, shape, "previous positions", bound=100.)
    positions = _array(positions, shape, "positions", bound=100.)
    old_velocity = _array(previous_velocity, shape, "previous velocity")
    velocity = _array(velocity, shape, "velocity")
    mass = _array(mass, (len(record.rest),), "free cloth mass")
    if np.any(mass <= 0) or not np.array_equal(velocity, (positions - previous) / duration):
        raise ValueError("Free positive-mass cloth and exact velocity continuity required")
    old_targets, old_active = parameters(record, start)
    targets, active = parameters(record, end)
    old = _evaluate(record, previous, old_targets, old_active)
    fixed = _evaluate(record, previous, targets, active)
    final = _evaluate(record, positions, targets, active)
    zero = Fraction(0)
    target_work, activation_work, increase, release = zero, zero, zero, zero
    for index, anchor in enumerate(record.anchors):
        before_norm = sum((value ** 2 for value in old["errors"][index]), zero)
        after_norm = sum((value ** 2 for value in fixed["errors"][index]), zero)
        k = Fraction(anchor[4]) / 2
        before, after = Fraction(float(old_active[index])), Fraction(float(active[index]))
        target_work += k * before * (after_norm - before_norm)
        changed = k * (after - before) * after_norm
        activation_work += changed
        increase += max(zero, changed)
        release += max(zero, -changed)
    work = fixed["energy"] - old["energy"]
    fixed_terms = [coefficient * (after ** 2 - before ** 2) / 2
                   for coefficient, before_row, after_row in zip(final["coefficients"], fixed["errors"], final["errors"])
                   for before, after in zip(before_row, after_row)]
    fixed_change = sum(fixed_terms, zero)
    stable_work = float(work)
    components = (stable_work, float(target_work), float(activation_work))
    rounding_bound = math.fsum(math.ulp(value) for value in components) + math.ulp(math.fsum(components[1:]))
    exact_fields = {"gripperParameterWorkJoules": stable_work,
        "gripperTargetParameterWorkJoules": float(target_work), "gripperActivationParameterWorkJoules": float(activation_work),
        "gripperActivationIncreaseWorkJoules": float(increase), "gripperReleaseEnergyRemovedJoules": float(release),
        "gripperParameterWorkComponentSumErrorBoundJoules": rounding_bound}
    numeric_fields = {"gripperBeforeJoules": float(old["energy"]), "gripperAfterJoules": float(final["energy"]),
        "gripperFixedPositionAfterJoules": float(fixed["energy"]), "gripperFixedParameterChangeJoules": float(fixed_change),
        "gripperChangeJoules": float(work + fixed_change)}
    fixed_scale = math.fsum(abs(float(value)) for value in fixed_terms)
    operations = 64 + 8 * len(record.ids)
    floor = old["underflow"] + fixed["underflow"] + final["underflow"]
    try:
        if step_report["materialGrippers"] is not True or step_report["accepted"] is not False or step_report["converged"] is not True:
            raise ValueError("Explicit unaccepted converged gripper step required")
        for key, value in (("gripperTargetsMeters", targets), ("gripperActivation", active)):
            if not np.array_equal(_array(step_report[key], value.shape, key), value):
                raise ValueError(f"Captured gripper controls differ from schedule: {key}")
        observed = step_report["gripperDiagnostics"]
        if not isinstance(observed, dict) or set(observed) != set(final["diagnostics"]):
            raise ValueError("Complete gripper force/reaction diagnostics required")
        for key, value in final["diagnostics"].items():
            if key in ("profile", "gripperIds"):
                if observed[key] != value:
                    raise ValueError(f"Captured gripper diagnostic identity changed: {key}")
            elif key in ("targetsMeters", "activation", "anchorWeightSums"):
                if not np.array_equal(_array(observed[key], np.asarray(value).shape, key), value):
                    raise ValueError(f"Captured gripper diagnostic parameters changed: {key}")
            elif key == "weightSumAdmissionTolerance":
                _exact_number(observed[key], value, key)
            else:
                _close(observed[key], value, key, final["scales"].get(key), operations=operations, floor=floor)
        _close(step_report["gripperEnergyJoules"], float(final["energy"]), "step gripper energy", floor=floor)
        balance = step_report["energyBalance"]
        if not isinstance(balance, dict) or balance.get("accepted") is not False:
            raise ValueError("Complete unaccepted energy balance required")
        for key, expected in exact_fields.items():
            _exact_number(balance[key], expected, key)
        for key, expected in numeric_fields.items():
            scale = (fixed_scale if key == "gripperFixedParameterChangeJoules" else
                     abs(stable_work) + fixed_scale if key == "gripperChangeJoules" else abs(expected))
            _close(balance[key], expected, key, scale, floor=floor)
        momentum = step_report["gripperMomentum"]
        if not isinstance(momentum, dict) or set(momentum) != {"changeKgMPerS", "externalImpulseNs", "residualNs", "toleranceNs", "scope"} or momentum["scope"] != _MOMENTUM_SCOPE:
            raise ValueError("Complete free-cloth external impulse evidence required")
        change_exact = [sum((Fraction(float(m)) * (Fraction(float(v[axis])) - Fraction(float(old_v[axis])))
                             for m, v, old_v in zip(mass, velocity, old_velocity)), zero) for axis in range(3)]
        impulse_exact = [Fraction(duration) * value for value in final["cloth_force"]]
        change, impulse = [np.array([float(value) for value in vector]) for vector in (change_exact, impulse_exact)]
        residual = np.array([float(a - b) for a, b in zip(change_exact, impulse_exact)])
        tolerance = len(mass) * duration * 1e-6 + 64 * _EPS * max(1., float(np.max(np.abs(change))), float(np.max(np.abs(impulse))))
        change_scale = np.sum(np.abs(mass[:, None] * (velocity - old_velocity)), axis=0)
        impulse_scale = duration * final["scales"]["totalClothForceNewtons"]
        _close(momentum["changeKgMPerS"], change, "momentum change", change_scale, operations=8 * len(mass) + 64)
        _close(momentum["externalImpulseNs"], impulse, "external impulse", impulse_scale, operations=operations, floor=duration * floor)
        _close(momentum["residualNs"], residual, "momentum residual", change_scale + impulse_scale,
               operations=max(operations, 8 * len(mass) + 64), floor=duration * floor)
        _close(momentum["toleranceNs"], tolerance, "momentum tolerance", operations=8)
        if np.max(np.abs(residual)) > tolerance:
            raise ValueError("Independent gripper free-cloth momentum balance failed")
    except KeyError as error:
        raise ValueError(f"Missing gripper step evidence: {error.args[0]}") from error
    evidence = {"verified": True, "profile": PROFILE, "gripperCount": len(record.ids),
        "startFraction": float(start), "endFraction": float(end), **numeric_fields, **exact_fields,
        "momentumResidualNs": residual.tolist(), "momentumToleranceNs": tolerance,
        "scope": "Independent virtual-gripper quadratic and discrete target-first work; enclosing replay verifies other forces and cloth paths."}
    return final["gradient"], evidence
