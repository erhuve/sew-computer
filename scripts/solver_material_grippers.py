"""Bounded source-material grippers and a stateless research control schedule.

These are external compliant tools, not zero-sum seam rows or moving pins.
No source rest geometry, material side, placement or physical acceptance is set.
Parameter work is a discrete fixed-position potential jump, not the continuous
actuator work along a prescribed trajectory.
"""

from fractions import Fraction
import math
from numbers import Real
import re

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

PROFILE = "source-material-compliant-grippers-v1"
SCHEDULE_PROFILE = "material-gripper-target-activation-v1"
MAX_VERTICES = 25000
MAX_FACES = 50000
MAX_GRIPPERS = 256
MAX_COORDINATE_METERS = 100.
MAX_STIFFNESS_N_PER_M = 1e12
WEIGHT_SUM_TOLERANCE = 1e-12
PARAMETER_ORDER = "target-first-at-old-activation-then-activation-at-new-target"
_CANCELLATION_RATIO = 2. ** -20


def _cancels(values, magnitudes):
    values, magnitudes = np.asarray(values), np.asarray(magnitudes)
    return bool(np.any((magnitudes > 0) & (np.abs(values) <= _CANCELLATION_RATIO * magnitudes)))


def _identity(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value) is not None


def _has_bool(value):
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b" or (value.dtype.kind == "O" and any(_has_bool(item) for item in value.flat))
    if isinstance(value, (list, tuple)):
        return any(_has_bool(item) for item in value)
    return isinstance(value, (bool, np.bool_))


def _array(value, shape, name, *, bound=None, integer=False):
    # Bound shape before copying/coercing caller containers. A malformed large
    # nested list must not allocate an unbounded numerical array first.
    def matching_shape(item, expected):
        if isinstance(item, np.ndarray):
            return item.shape == expected
        if not expected:
            return not isinstance(item, (list, tuple))
        return (isinstance(item, (list, tuple)) and len(item) == expected[0]
                and all(matching_shape(child, expected[1:]) for child in item))

    if not matching_shape(value, shape):
        raise ValueError(f"Finite bounded {name} with shape {shape} required")
    if _has_bool(value):
        raise ValueError(f"{name} cannot contain Boolean values")
    try:
        result = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Finite bounded {name} required") from error
    if (result.shape != shape or result.dtype.kind not in ("iu" if integer else "iuf")
            or not np.isfinite(result).all() or (bound is not None and np.any(np.abs(result) > bound))):
        raise ValueError(f"Finite bounded {name} with shape {shape} required")
    return result.astype(np.int64 if integer else np.float64, copy=True)


def _frozen(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _sum_products(pairs):
    """Compensated dot product; no dependence on extended longdouble precision."""
    parts = []
    splitter = 134217729.  # 2**27 + 1; operands are bounded well below overflow.
    for first, second in pairs:
        first, second = float(first), float(second)
        product = first * second
        first_split, second_split = splitter * first, splitter * second
        first_high = first_split - (first_split - first)
        second_high = second_split - (second_split - second)
        first_low, second_low = first - first_high, second - second_high
        error = ((first_high * second_high - product) + first_high * second_low
                 + first_low * second_high) + first_low * second_low
        parts.extend((product, error))
    return math.fsum(parts)


class _Immutable:
    __slots__ = ()

    def __setattr__(self, name, value):
        raise AttributeError("Captured material-gripper inputs are immutable")

    def __delattr__(self, name):
        raise AttributeError("Captured material-gripper inputs are immutable")


class MaterialGrippers(_Immutable):
    """Immutable material anchors; caller binds canonical revision provenance."""
    __slots__ = ("_vertex_instance_ids", "_faces", "_anchor_faces", "_weights", "_stiffness",
                 "_gripper_ids", "_instances", "_triangle_indices", "_row_sums")

    def __init__(self, vertex_instance_ids, faces, anchors):
        if (not isinstance(vertex_instance_ids, (list, tuple))
                or not 3 <= len(vertex_instance_ids) <= MAX_VERTICES
                or not all(_identity(value) for value in vertex_instance_ids)):
            raise ValueError("Bounded explicit per-vertex instance identities required")
        try:
            face_count = len(faces)
        except TypeError as error:
            raise ValueError("Canonical triangle faces required") from error
        if not 1 <= face_count <= MAX_FACES:
            raise ValueError("Canonical triangle face budget exceeded")
        face_array = _array(faces, (face_count, 3), "canonical triangle faces", integer=True,
                            bound=len(vertex_instance_ids) - 1)
        if np.any(face_array < 0) or np.any(np.diff(np.sort(face_array, axis=1), axis=1) == 0):
            raise ValueError("Distinct canonical face vertices required")
        if len(np.unique(np.sort(face_array, axis=1), axis=0)) != face_count:
            raise ValueError("Duplicate canonical material faces are ambiguous")
        identity_numbers = {value: index for index, value in enumerate(dict.fromkeys(vertex_instance_ids))}
        numbers = np.array([identity_numbers[value] for value in vertex_instance_ids])
        if np.any(numbers[face_array] != numbers[face_array[:, :1]]):
            raise ValueError("Every canonical face must belong to one source instance")
        if not isinstance(anchors, (list, tuple)) or not 1 <= len(anchors) <= MAX_GRIPPERS:
            raise ValueError("Bounded nonempty source material anchors required")
        identities, instances, triangles, weights, stiffness, sums = [], [], [], [], [], []
        fields = {"id", "instanceId", "triangleIndex", "weights", "stiffnessNPerM"}
        for anchor in anchors:
            if (not isinstance(anchor, dict) or set(anchor) != fields or not _identity(anchor["id"])
                    or anchor["id"] in identities or not _identity(anchor["instanceId"])
                    or type(anchor["triangleIndex"]) is not int
                    or not 0 <= anchor["triangleIndex"] < face_count):
                raise ValueError("Unique gripper and canonical source triangle identities required")
            triangle = anchor["triangleIndex"]
            if any(vertex_instance_ids[vertex] != anchor["instanceId"] for vertex in face_array[triangle]):
                raise ValueError("Gripper face support differs from declared source instance")
            weight = _array(anchor["weights"], (3,), "barycentric weights", bound=1.)
            total = math.fsum(weight)
            if np.any(weight < 0) or abs(total - 1.) > WEIGHT_SUM_TOLERANCE:
                raise ValueError("Nonnegative normalized barycentric weights required; no renormalization is performed")
            coefficient = anchor["stiffnessNPerM"]
            if (isinstance(coefficient, (bool, np.bool_)) or not isinstance(coefficient, Real)
                    or not math.isfinite(coefficient) or not 0 < coefficient <= MAX_STIFFNESS_N_PER_M):
                raise ValueError("Positive bounded gripper stiffness required")
            identities.append(anchor["id"])
            instances.append(anchor["instanceId"])
            triangles.append(triangle)
            weights.append(weight)
            stiffness.append(float(coefficient))
            sums.append(total)
        for name, value in (("_vertex_instance_ids", tuple(vertex_instance_ids)), ("_faces", _frozen(face_array)),
                            ("_anchor_faces", _frozen(face_array[triangles])), ("_weights", _frozen(np.array(weights))),
                            ("_stiffness", _frozen(np.array(stiffness))), ("_gripper_ids", tuple(identities)),
                            ("_instances", tuple(instances)), ("_triangle_indices", tuple(triangles)),
                            ("_row_sums", _frozen(np.array(sums)))):
            object.__setattr__(self, name, value)

    @property
    def vertex_count(self):
        return len(self._vertex_instance_ids)

    @property
    def faces(self):
        return self._faces.copy()

    @property
    def vertex_instance_ids(self):
        return self._vertex_instance_ids

    @property
    def gripper_ids(self):
        return self._gripper_ids

    @property
    def anchors(self):
        return [{"id": identity, "instanceId": instance, "triangleIndex": triangle,
                 "weights": weight.tolist(), "stiffnessNPerM": float(stiffness)}
                for identity, instance, triangle, weight, stiffness in zip(self._gripper_ids, self._instances,
                    self._triangle_indices, self._weights, self._stiffness)]

    def _positions(self, positions):
        return _array(positions, (self.vertex_count, 3), "world positions", bound=MAX_COORDINATE_METERS)

    def _parameters(self, targets, activation):
        count = len(self._gripper_ids)
        targets = _array(targets, (count, 3), "world targets", bound=MAX_COORDINATE_METERS)
        activation = _array(activation, (count,), "gripper activation", bound=1.)
        if np.any(activation < 0):
            raise ValueError("Gripper activation must lie in [0, 1]")
        coefficients = self._stiffness * activation
        if np.any((activation > 0) & (coefficients == 0)):
            raise ValueError("Nonzero gripper stiffness/activation product underflows")
        return targets, activation

    def _anchor_positions(self, positions):
        return np.array([[_sum_products(zip(weight, positions[face, axis])) for axis in range(3)]
                         for face, weight in zip(self._anchor_faces, self._weights)])

    def _residuals(self, positions, targets):
        return np.array([[_sum_products([*zip(weight, positions[face, axis]), (-1., target[axis])])
                          for axis in range(3)]
                         for face, weight, target in zip(self._anchor_faces, self._weights, targets)])

    def potential(self, targets, activation):
        return MaterialGripperPotential(self, targets, activation)

    def parameter_energy_change(self, positions, before_targets, before_activation, after_targets, after_activation):
        """Exact binary-input discrete work, rounded only at report boundaries.

        First move the target at OLD activation, then change activation at the
        NEW target. Releasing removes stored potential. This says nothing about
        continuous actuator work between those parameter samples.
        """
        positions = self._positions(positions)
        before_targets, before_activation = self._parameters(before_targets, before_activation)
        after_targets, after_activation = self._parameters(after_targets, after_activation)
        totals = {key: Fraction(0) for key in ("total", "target", "activation", "increase", "release")}
        for index, (face, weights, stiffness) in enumerate(zip(self._anchor_faces, self._weights, self._stiffness)):
            anchor = [sum((Fraction(float(weight)) * Fraction(float(positions[vertex, axis]))
                           for vertex, weight in zip(face, weights)), Fraction(0)) for axis in range(3)]
            before_norm = sum((coordinate - Fraction(float(target))) ** 2
                              for coordinate, target in zip(anchor, before_targets[index]))
            after_norm = sum((coordinate - Fraction(float(target))) ** 2
                             for coordinate, target in zip(anchor, after_targets[index]))
            coefficient = Fraction(float(stiffness)) / 2
            old, new = Fraction(float(before_activation[index])), Fraction(float(after_activation[index]))
            target_work = coefficient * old * (after_norm - before_norm)
            activation_work = coefficient * (new - old) * after_norm
            totals["total"] += coefficient * (new * after_norm - old * before_norm)
            totals["target"] += target_work
            totals["activation"] += activation_work
            totals["increase"] += max(activation_work, 0)
            totals["release"] += max(-activation_work, 0)
        values = {key: float(value) for key, value in totals.items()}
        # Conservative bound includes separately rounded components, the stable
        # total, and one floating addition. Large cancelling components cannot
        # reconstruct the stable total more accurately than their own ulps.
        bound = math.fsum(math.ulp(values[key]) for key in ("total", "target", "activation"))
        bound += math.ulp(math.fsum((values["target"], values["activation"])))
        return {"totalParameterWorkJoules": values["total"], "targetParameterWorkJoules": values["target"],
                "activationParameterWorkJoules": values["activation"],
                "activationIncreaseWorkJoules": values["increase"], "releaseEnergyRemovedJoules": values["release"],
                "roundedComponentSumErrorBoundJoules": bound, "parameterOrder": PARAMETER_ORDER,
                "workScope": "Discrete fixed-position parameter jump; not continuous actuator work."}


class MaterialGripperPotential(_Immutable):
    __slots__ = ("_recipe", "_targets", "_activation", "_coefficients", "_exact_coefficients", "_coefficient_roundoff")

    def __init__(self, recipe, targets, activation):
        if not isinstance(recipe, MaterialGrippers):
            raise ValueError("Validated material-gripper recipe required")
        targets, activation = recipe._parameters(targets, activation)
        object.__setattr__(self, "_recipe", recipe)
        object.__setattr__(self, "_targets", _frozen(targets))
        object.__setattr__(self, "_activation", _frozen(activation))
        object.__setattr__(self, "_coefficients", _frozen(recipe._stiffness * activation))
        exact = tuple(Fraction(float(stiffness)) * Fraction(float(active))
                      for stiffness, active in zip(recipe._stiffness, activation))
        object.__setattr__(self, "_exact_coefficients", exact)
        object.__setattr__(self, "_coefficient_roundoff", any(Fraction(float(value)) != expected
                    for value, expected in zip(self._coefficients, exact)))

    @property
    def targets(self):
        return self._targets.copy()

    @property
    def activation(self):
        return self._activation.copy()

    def energy(self, positions):
        positions = self._recipe._positions(positions)
        residuals = self._recipe._residuals(positions, self._targets)
        return math.fsum(.5 * coefficient * _sum_products(zip(residual, residual))
                         for coefficient, residual in zip(self._coefficients, residuals))

    def residual(self, positions):
        positions = self._recipe._positions(positions)
        return (np.sqrt(self._coefficients)[:, None]
                * self._recipe._residuals(positions, self._targets)).ravel()

    def gradient(self, positions):
        positions = self._recipe._positions(positions)
        return -self._force_data(positions)["nodalForcesNewtons"]

    def _anchor_cancellation(self, positions, residuals):
        scale = np.sum(self._recipe._weights[:, :, None] * np.abs(positions[self._recipe._anchor_faces]), axis=1) + np.abs(self._targets)
        return _cancels(residuals[self._activation > 0], scale[self._activation > 0])

    def _exact_errors(self, positions):
        # Only source support is visited; no rational Cartesian cloth matrix.
        values = {int(vertex): tuple(Fraction(float(value)) for value in positions[vertex])
                  for vertex in np.unique(self._recipe._anchor_faces)}
        anchors, residuals = [], []
        for face, weights, target in zip(self._recipe._anchor_faces, self._recipe._weights, self._targets):
            weights = [Fraction(float(weight)) for weight in weights]
            anchor = tuple(sum((weight * values[int(vertex)][axis] for vertex, weight in zip(face, weights)), Fraction())
                           for axis in range(3))
            anchors.append(anchor)
            residuals.append(tuple(value - Fraction(float(goal)) for value, goal in zip(anchor, target)))
        return values, anchors, residuals

    def _exact_force_data(self, positions):
        values, anchors, residuals = self._exact_errors(positions)
        forces = [tuple(-coefficient * value for value in residual)
                  for coefficient, residual in zip(self._exact_coefficients, residuals)]
        nodal = {vertex: [Fraction(), Fraction(), Fraction()] for vertex in values}
        for face, weights, force in zip(self._recipe._anchor_faces, self._recipe._weights, forces):
            for vertex, weight in zip(face, weights):
                weight = Fraction(float(weight))
                for axis in range(3):
                    nodal[int(vertex)][axis] += weight * force[axis]
        cloth = [sum((force[axis] for force in nodal.values()), Fraction()) for axis in range(3)]
        tool = [-sum((force[axis] for force in forces), Fraction()) for axis in range(3)]

        def cross(first, second):
            return (first[1] * second[2] - first[2] * second[1],
                    first[2] * second[0] - first[0] * second[2],
                    first[0] * second[1] - first[1] * second[0])

        cloth_cross = [cross(values[vertex], force) for vertex, force in nodal.items()]
        tool_cross = [cross(tuple(Fraction(float(value)) for value in target), tuple(-value for value in force))
                      for target, force in zip(self._targets, forces)]
        cloth_torque = [sum((value[axis] for value in cloth_cross), Fraction()) for axis in range(3)]
        tool_torque = [sum((value[axis] for value in tool_cross), Fraction()) for axis in range(3)]
        nodal_array = np.zeros_like(positions)
        for vertex, force in nodal.items():
            nodal_array[vertex] = [float(value) for value in force]
        force_array = np.array([[float(value) for value in force] for force in forces])
        return {"anchorPositionsMeters": np.array([[float(value) for value in anchor] for anchor in anchors]),
                "anchorForcesNewtons": force_array, "toolReactionsNewtons": -force_array,
                "nodalForcesNewtons": nodal_array, "totalClothForceNewtons": np.array([float(value) for value in cloth]),
                "totalToolReactionNewtons": np.array([float(value) for value in tool]),
                "clothTorqueNewtonMeters": np.array([float(value) for value in cloth_torque]),
                "toolTorqueNewtonMeters": np.array([float(value) for value in tool_torque]),
                "netForceResidualNewtons": np.array([float(first + second) for first, second in zip(cloth, tool)]),
                "netTorqueResidualNewtonMeters": np.array([float(first + second) for first, second in zip(cloth_torque, tool_torque)])}

    def _force_data(self, positions):
        residuals = self._recipe._residuals(positions, self._targets)
        if self._coefficient_roundoff or self._anchor_cancellation(positions, residuals):
            return self._exact_force_data(positions)
        anchors = self._recipe._anchor_positions(positions)
        forces = -self._coefficients[:, None] * residuals
        nodal, nodal_scale = np.zeros_like(positions), np.zeros_like(positions)
        for face, weight, force in zip(self._recipe._anchor_faces, self._recipe._weights, forces):
            contributions = weight[:, None] * force
            np.add.at(nodal, face, contributions)
            np.add.at(nodal_scale, face, np.abs(contributions))
        cloth_force, tool_force = nodal.sum(axis=0), -forces.sum(axis=0)
        cloth_cross, tool_cross = np.cross(positions, nodal), np.cross(self._targets, -forces)
        cloth_torque, tool_torque = cloth_cross.sum(axis=0), tool_cross.sum(axis=0)
        def cross_scale(first, second):
            first, second = np.abs(first), np.abs(second)
            return np.stack((first[:, 1] * second[:, 2] + first[:, 2] * second[:, 1],
                             first[:, 2] * second[:, 0] + first[:, 0] * second[:, 2],
                             first[:, 0] * second[:, 1] + first[:, 1] * second[:, 0]), axis=1)
        cloth_cross_scale, tool_cross_scale = cross_scale(positions, nodal_scale), cross_scale(self._targets, forces)
        # Cancellation is an arithmetic trigger, not a relaxed mechanics
        # tolerance. Exact fallback starts from original binary inputs, before
        # anchor subtraction, k*activation products or individual force rounding.
        if (_cancels(nodal, nodal_scale) or _cancels(cloth_force, nodal_scale.sum(axis=0))
                or _cancels(tool_force, np.abs(forces).sum(axis=0))
                or _cancels(cloth_cross, cloth_cross_scale) or _cancels(tool_cross, tool_cross_scale)
                or _cancels(cloth_torque, cloth_cross_scale.sum(axis=0))
                or _cancels(tool_torque, tool_cross_scale.sum(axis=0))
                or _cancels(cloth_force + tool_force, np.abs(cloth_force) + np.abs(tool_force))
                or _cancels(cloth_torque + tool_torque, np.abs(cloth_torque) + np.abs(tool_torque))):
            return self._exact_force_data(positions)
        return {"anchorPositionsMeters": anchors, "anchorForcesNewtons": forces, "toolReactionsNewtons": -forces,
                "nodalForcesNewtons": nodal, "totalClothForceNewtons": cloth_force, "totalToolReactionNewtons": tool_force,
                "clothTorqueNewtonMeters": cloth_torque, "toolTorqueNewtonMeters": tool_torque,
                "netForceResidualNewtons": cloth_force + tool_force,
                "netTorqueResidualNewtonMeters": cloth_torque + tool_torque}

    def jacobian(self, positions=None):
        if positions is not None:
            self._recipe._positions(positions)
        count = len(self._coefficients)
        shape = (count * 3, self._recipe.vertex_count * 3)
        if not np.any(self._coefficients):
            return csr_matrix(shape)
        values = np.broadcast_to((np.sqrt(self._coefficients)[:, None] * self._recipe._weights)[:, :, None], (count, 3, 3))
        rows = np.broadcast_to(np.arange(count)[:, None, None] * 3 + np.arange(3)[None, None, :], values.shape)
        columns = self._recipe._anchor_faces[:, :, None] * 3 + np.arange(3)[None, None, :]
        result = coo_matrix((values.ravel(), (rows.ravel(), columns.ravel())), shape=shape).tocsr()
        result.eliminate_zeros()
        return result

    def hessian(self, positions=None):
        if positions is not None:
            self._recipe._positions(positions)
        count = len(self._coefficients)
        shape = (self._recipe.vertex_count * 3,) * 2
        if not np.any(self._coefficients):
            return csr_matrix(shape)
        coefficients = self._coefficients[:, None, None] * self._recipe._weights[:, :, None] * self._recipe._weights[:, None, :]
        values = np.broadcast_to(coefficients[:, :, :, None], (count, 3, 3, 3))
        rows = np.broadcast_to(self._recipe._anchor_faces[:, :, None, None] * 3 + np.arange(3)[None, None, None, :], values.shape)
        columns = np.broadcast_to(self._recipe._anchor_faces[:, None, :, None] * 3 + np.arange(3)[None, None, None, :], values.shape)
        result = coo_matrix((values.ravel(), (rows.ravel(), columns.ravel())), shape=shape).tocsr()
        result.eliminate_zeros()
        return result

    def energy_change(self, start, end):
        start, end = self._recipe._positions(start), self._recipe._positions(end)
        if np.array_equal(start[self._recipe._anchor_faces], end[self._recipe._anchor_faces]):
            return 0.
        values = []
        scale = 0.
        exact_needed = self._coefficient_roundoff
        for face, weight, target, coefficient in zip(self._recipe._anchor_faces, self._recipe._weights,
                                                      self._targets, self._coefficients):
            if coefficient == 0:
                continue
            difference, total = [], []
            for axis in range(3):
                # Compensated products retain small anchor displacements even
                # when the world coordinates and target are much larger.
                difference.append(_sum_products([*zip(weight, end[face, axis]), *zip(-weight, start[face, axis])]))
                total.append(_sum_products([*zip(weight, end[face, axis]), *zip(weight, start[face, axis]), (-2., target[axis])]))
            dot = _sum_products(zip(difference, total))
            dot_scale = math.fsum(abs(first * second) for first, second in zip(difference, total))
            exact_needed |= _cancels(dot, dot_scale)
            values.append(.5 * coefficient * dot)
            scale += .5 * coefficient * dot_scale
        result = math.fsum(values)
        exact_needed |= _cancels(result, scale)
        exact_needed |= self._anchor_cancellation(start, self._recipe._residuals(start, self._targets))
        exact_needed |= self._anchor_cancellation(end, self._recipe._residuals(end, self._targets))
        if exact_needed:
            _, _, before = self._exact_errors(start)
            _, _, after = self._exact_errors(end)
            difference = sum((coefficient * sum(((last - first) * (last + first)
                for first, last in zip(old, new)), Fraction()) / 2
                for coefficient, old, new in zip(self._exact_coefficients, before, after)), Fraction())
            return float(difference)
        return result

    def diagnostics(self, positions):
        positions = self._recipe._positions(positions)
        return {"profile": PROFILE, "gripperIds": list(self._recipe.gripper_ids), "energyJoules": self.energy(positions),
                "targetsMeters": self.targets, "activation": self.activation,
                "anchorWeightSums": self._recipe._row_sums.copy(), "weightSumAdmissionTolerance": WEIGHT_SUM_TOLERANCE,
                **self._force_data(positions)}


class MaterialGripperSchedule(_Immutable):
    """Stateless dyadic piecewise-linear parameters, including release."""
    __slots__ = ("_ids", "_fractions", "_targets", "_activation", "_subdivisions")

    def __init__(self, recipe, subdivisions, *, gripper_ids):
        if (type(subdivisions) is not int or not 1 <= subdivisions <= 4096 or subdivisions & (subdivisions - 1)
                or not isinstance(gripper_ids, (list, tuple)) or not 1 <= len(gripper_ids) <= MAX_GRIPPERS
                or not all(_identity(value) for value in gripper_ids) or len(set(gripper_ids)) != len(gripper_ids)):
            raise ValueError("Bounded dyadic subdivisions and explicit ordered gripper identities required")
        if (not isinstance(recipe, dict) or set(recipe) != {"profile", "gripperIds", "knots"}
                or recipe["profile"] != SCHEDULE_PROFILE or recipe["gripperIds"] != list(gripper_ids)
                or not isinstance(recipe["knots"], list) or not 2 <= len(recipe["knots"]) <= 65):
            raise ValueError("Explicit source-bound material-gripper schedule required")
        fractions, targets, activation = [], [], []
        for knot in recipe["knots"]:
            if not isinstance(knot, dict) or set(knot) != {"fraction", "targetsMeters", "activation"}:
                raise ValueError("Explicit target and activation schedule knots required")
            fraction = self._fraction(knot["fraction"])
            if (fraction * subdivisions).denominator != 1 or (fractions and fraction <= fractions[-1]):
                raise ValueError("Ordered dyadic schedule knots must coincide with initial interval boundaries")
            target = _array(knot["targetsMeters"], (len(gripper_ids), 3), "schedule targets", bound=MAX_COORDINATE_METERS)
            active = _array(knot["activation"], (len(gripper_ids),), "schedule activation", bound=1.)
            if np.any(active < 0):
                raise ValueError("Schedule activation must lie in [0, 1]")
            fractions.append(fraction)
            targets.append(target)
            activation.append(active)
        if fractions[0] != 0 or fractions[-1] != 1:
            raise ValueError("Gripper schedule must span exactly [0, 1]")
        object.__setattr__(self, "_ids", tuple(gripper_ids))
        object.__setattr__(self, "_fractions", tuple(fractions))
        object.__setattr__(self, "_targets", _frozen(np.array(targets)))
        object.__setattr__(self, "_activation", _frozen(np.array(activation)))
        object.__setattr__(self, "_subdivisions", subdivisions)

    @staticmethod
    def _fraction(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, Fraction)):
            raise ValueError("Bounded finite dyadic schedule fraction required")
        try:
            fraction = Fraction(value)
        except (ValueError, OverflowError) as error:
            raise ValueError("Bounded finite dyadic schedule fraction required") from error
        if (not 0 <= fraction <= 1 or fraction.denominator > 2 ** 40
                or fraction.denominator & (fraction.denominator - 1)):
            raise ValueError("Schedule fraction must be in [0, 1] on a bounded dyadic grid")
        return fraction

    @property
    def gripper_ids(self):
        return self._ids

    def parameters(self, fraction):
        fraction = self._fraction(fraction)
        for index, value in enumerate(self._fractions):
            if fraction == value:
                return self._targets[index].copy(), self._activation[index].copy()
        for index, (lower, upper) in enumerate(zip(self._fractions[:-1], self._fractions[1:])):
            if lower < fraction < upper:
                amount = float((fraction - lower) / (upper - lower))
                # Increment form preserves a held target exactly, including
                # at the admission bounds. A convex weighted sum can round a
                # constant 100 m target to 100.00000000000001 m.
                return (self._targets[index] + amount * (self._targets[index + 1] - self._targets[index]),
                        self._activation[index] + amount * (self._activation[index + 1] - self._activation[index]))
        raise ValueError("Schedule does not cover the requested fraction")
