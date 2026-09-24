"""Explicit per-hinge fold controls, separate from the legacy fixed actuator.

The numerical coefficient is RN(binary64 stiffness * binary64 activation).
All mechanics and work use that same coefficient. Inactive contributions do
not evaluate undefined angles; complete source topology is retained separately
and callers must preserve their full cloth/contact/triangle/path guards.
"""

from fractions import Fraction
import math

import numpy as np
from scipy.sparse import csr_matrix, diags

from solver_bending import ElasticDihedralBending
from solver_fold_actuation import FoldActuation


PROFILE = "explicit-source-fold-angle-activation-v1"
COEFFICIENT_POLICY = "rounded-binary64-stiffness-times-activation-v1"
PARAMETER_ORDER = "target-first-at-old-activation-then-activation-at-new-target"
MAX_HINGES = 4096
_WORK_SCOPE = "Discrete parameter jump at fixed positions and sampled binary64 signed angles; not exact-real angle evaluation, continuous actuator work, dissipation or construction acceptance."


def _frozen(array):
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _booleans(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b" or value.dtype.kind == "O" and any(_booleans(item) for item in value.flat)
    return isinstance(value, (list, tuple)) and any(_booleans(item) for item in value)


def _vector(value, count, label, *, minimum=None, maximum=None):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or isinstance(value, np.ndarray) and value.shape != (count,)
            or isinstance(value, (list, tuple)) and (len(value) != count or any(isinstance(x, (list, tuple, np.ndarray)) for x in value))
            or _booleans(value)):
        raise ValueError("One explicit non-Boolean scalar per hinge required: " + label)
    try:
        result = np.array([_binary64(item) for item in value], dtype=np.float64)
        if (minimum is not None and np.any(result < minimum)
                or maximum is not None and np.any(result > maximum)):
            raise ValueError("Representable binary64 fold parameters within declared bounds required: " + label)
    except (TypeError, OverflowError) as error:
        raise ValueError("Finite numeric fold parameters required: " + label) from error
    return result


def _binary64(value):
    # Validate original scalar values before array promotion can round a large
    # integer in a mixed float/integer list or an equality comparison.
    integer = type(value) is int or isinstance(value, np.integer) and not isinstance(value, np.bool_)
    floating = type(value) is float or isinstance(value, np.floating)
    if not (integer or floating):
        raise ValueError("Original non-Boolean binary64 numeric scalar required")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Representable finite binary64 scalar required") from error
    if not math.isfinite(result) or (int(result) != int(value) if integer else value != result):
        raise ValueError("Numeric scalar must be exactly representable as binary64")
    return result


def _float(value):
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Finite controlled-fold energy/work required") from error
    if not math.isfinite(result):
        raise ValueError("Finite controlled-fold energy/work required")
    return result


def _rat(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator), "roundedBinary64": _float(value)}


class _Immutable:
    __slots__ = ()

    def __setattr__(self, name, value):
        raise AttributeError("Explicit fold controls are immutable")

    def __delattr__(self, name):
        raise AttributeError("Explicit fold controls are immutable")


class ControlledFoldActuation(_Immutable):
    """Complete actual native hinge binding; parameters remain independently explicit."""

    __slots__ = ("_vertex_count", "_hinges", "_stiffness")

    def __init__(self, model, hinges, stiffness_joules):
        if (not isinstance(hinges, (list, tuple, np.ndarray))
                or isinstance(hinges, np.ndarray) and (hinges.ndim != 2 or hinges.shape[1] != 4 or not 1 <= len(hinges) <= MAX_HINGES)
                or isinstance(hinges, (list, tuple)) and (not 1 <= len(hinges) <= MAX_HINGES
                    or any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in hinges))
                or _booleans(hinges)):
            raise ValueError("Non-Boolean native hinge indices required")
        raw = np.asarray(hinges)
        if raw.ndim != 2 or raw.shape[1] != 4 or not 1 <= len(raw) <= MAX_HINGES:
            raise ValueError("Bounded complete explicit native hinges required")
        stiffness = _vector(stiffness_joules, len(raw), "stiffness", minimum=0)
        if np.any(stiffness == 0):
            raise ValueError("Base external fold stiffness must be positive; release uses explicit activation")
        legacy = FoldActuation(model, hinges, stiffness)
        object.__setattr__(self, "_vertex_count", legacy.vertex_count)
        object.__setattr__(self, "_hinges", _frozen(legacy.hinges))
        object.__setattr__(self, "_stiffness", _frozen(legacy.stiffness))

    @property
    def vertex_count(self):
        return self._vertex_count

    @property
    def hinges(self):
        return self._hinges.copy()

    @property
    def stiffness(self):
        return self._stiffness.copy()

    def _parameters(self, targets, activation):
        targets = _vector(targets, len(self._hinges), "signed target radians")
        if np.any(np.abs(targets) >= math.pi - 1e-8):
            raise ValueError("Explicit signed target angles must stay inside the existing principal branch")
        activation = _vector(activation, len(self._hinges), "activation", minimum=0, maximum=1)
        exact = tuple(Fraction(float(k)) * Fraction(float(a)) for k, a in zip(self._stiffness, activation))
        coefficient = np.array([_float(value) for value in exact])
        if any(value > 0 and rounded == 0 for value, rounded in zip(exact, coefficient)):
            raise ValueError("Positive controlled-fold coefficient underflows to inactive")
        return targets, activation, coefficient, exact

    def _positions(self, positions):
        if (not isinstance(positions, (list, tuple, np.ndarray))
                or isinstance(positions, np.ndarray) and positions.shape != (self.vertex_count, 3)
                or isinstance(positions, (list, tuple)) and (len(positions) != self.vertex_count
                    or any(not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 3 for row in positions))):
            raise ValueError("Complete finite cloth positions required")
        return np.array([[_binary64(value) for value in row] for row in positions])

    def potential(self, targets, activation):
        return ControlledFoldPotential(self, targets, activation)

    def parameter_energy_change(self, positions, previous_targets, previous_activation, targets, activation):
        """Exact quadratic parameter work conditional on one sampled angle per needed hinge."""
        positions = self._positions(positions)
        old = self.potential(previous_targets, previous_activation)
        new = self.potential(targets, activation)
        union = np.flatnonzero((old._coefficients > 0) | (new._coefficients > 0))
        geometry = ElasticDihedralBending(self.vertex_count, self._hinges[union], np.zeros(len(union)),
                                          np.ones(len(union)), np.ones(len(union)))
        sampled = geometry.angles(positions)
        angles = [None] * len(self._hinges)
        zero = Fraction()
        totals = {name: zero for name in ("before", "after", "target", "activation", "total", "increase", "release")}
        per_hinge = []
        for index, angle in zip(union, sampled):
            index = int(index)
            angles[index] = float(angle)
            theta = Fraction(float(angle))
            beta0, beta1 = Fraction(float(old._coefficients[index])), Fraction(float(new._coefficients[index]))
            e0, e1 = theta - Fraction(float(old._targets[index])), theta - Fraction(float(new._targets[index]))
            before, after = beta0 * e0**2 / 2, beta1 * e1**2 / 2
            target_work = beta0 * (e1**2 - e0**2) / 2
            activation_work = (beta1 - beta0) * e1**2 / 2
            total = after - before
            if total != target_work + activation_work:
                raise ValueError("Exact target-first fold work identity failed")
            values = {"before": before, "after": after, "target": target_work, "activation": activation_work,
                "total": total, "increase": max(activation_work, zero), "release": max(-activation_work, zero)}
            for key, value in values.items():
                totals[key] += value
            per_hinge.append({"hingeIndex": index, "workJoules": {key: _rat(value) for key, value in values.items()}})
        values = {key: _float(value) for key, value in totals.items()}
        # Compute the total independently, then disclose the precision with
        # which separately rounded components can reconstruct that total.
        try:
            component_sum = math.fsum((values["target"], values["activation"]))
            bound = math.fsum(math.ulp(values[key]) for key in ("total", "target", "activation")) + math.ulp(component_sum)
        except OverflowError as error:
            raise ValueError("Finite parameter-work component error bound required") from error
        if not math.isfinite(bound):
            raise ValueError("Finite parameter-work component error bound required")
        return {"profile": PROFILE, "accepted": False, "coefficientPolicy": COEFFICIENT_POLICY,
            "totalParameterWorkJoules": values["total"], "targetParameterWorkJoules": values["target"],
            "activationParameterWorkJoules": values["activation"],
            "activationIncreaseWorkJoules": values["increase"], "releaseEnergyRemovedJoules": values["release"],
            "beforeJoules": values["before"], "fixedPositionAfterJoules": values["after"],
            "roundedComponentSumErrorBoundJoules": bound, "parameterOrder": PARAMETER_ORDER,
            "sampledAnglesRadians": angles, "perHinge": per_hinge,
            "exactWorkJoules": {key: _rat(value) for key, value in totals.items()},
            "previousCoefficientWitnesses": old.coefficient_witnesses(), "coefficientWitnesses": new.coefficient_witnesses(),
            "workScope": _WORK_SCOPE}


class ControlledFoldPotential(_Immutable):
    """Active force evaluation and complete immutable parameter identity are separate."""

    __slots__ = ("_recipe", "_targets", "_activation", "_coefficients", "_exact_coefficients", "_active")

    def __init__(self, recipe, targets, activation):
        if not isinstance(recipe, ControlledFoldActuation):
            raise ValueError("Validated complete native controlled-fold recipe required")
        targets, activation, coefficients, exact = recipe._parameters(targets, activation)
        active = np.flatnonzero(coefficients > 0)
        for name, value in (("_recipe", recipe), ("_targets", _frozen(targets)), ("_activation", _frozen(activation)),
                            ("_coefficients", _frozen(coefficients)), ("_exact_coefficients", exact),
                            ("_active", _frozen(active))):
            object.__setattr__(self, name, value)

    @property
    def _geometry(self):
        # Do not retain a mutable nested evaluator whose weight/indices can
        # drift from the immutable numerical control law between calls.
        count = len(self._active)
        return ElasticDihedralBending(self._recipe.vertex_count, self._recipe._hinges[self._active],
                                      np.zeros(count), np.ones(count), np.ones(count))

    @property
    def targets(self):
        return self._targets.copy()

    @property
    def activation(self):
        return self._activation.copy()

    @property
    def coefficients(self):
        return self._coefficients.copy()

    @property
    def all_hinges(self):
        return self._recipe.hinges

    @property
    def active_hinge_indices(self):
        return self._active.copy()

    def coefficient_witnesses(self):
        return [{"hingeIndex": index, "stiffnessJoules": float(k), "activation": float(a),
            "numericalCoefficientJoules": float(beta), "exactProductJoules": _rat(exact),
            "roundedMinusExactProductJoules": _rat(Fraction(float(beta)) - exact)}
            for index, (k, a, beta, exact) in enumerate(zip(self._recipe._stiffness, self._activation, self._coefficients, self._exact_coefficients))]

    def _angles_errors(self, positions):
        positions = self._recipe._positions(positions)
        angles = self._geometry.angles(positions)
        errors = [Fraction(float(angle)) - Fraction(float(self._targets[index])) for index, angle in zip(self._active, angles)]
        return positions, angles, errors

    def energy(self, positions):
        _, _, errors = self._angles_errors(positions)
        return _float(sum((Fraction(float(self._coefficients[index])) * error**2 / 2
                          for index, error in zip(self._active, errors)), Fraction()))

    def residual(self, positions):
        _, _, errors = self._angles_errors(positions)
        result = np.sqrt(self._coefficients[self._active]) * np.array([_float(value) for value in errors])
        if not np.isfinite(result).all():
            raise ValueError("Finite controlled-fold residual required")
        return result

    def jacobian(self, positions):
        positions = self._recipe._positions(positions)
        if not len(self._active):
            return csr_matrix((0, 3 * self._recipe.vertex_count))
        result = (diags(np.sqrt(self._coefficients[self._active])) @ self._geometry.jacobian(positions)).tocsr()
        if not np.isfinite(result.data).all():
            raise ValueError("Finite controlled-fold residual Jacobian required")
        return result

    def gradient(self, positions):
        positions, _, errors = self._angles_errors(positions)
        jacobian = self._geometry.jacobian(positions).tocsr()
        result = [Fraction()] * (3 * self._recipe.vertex_count)
        # Round only after contracting the sampled binary64 angle Jacobian.
        # Rounding beta*error first can erase a representable force on a small
        # hinge, or lose cancellation between forces on shared vertices.
        for row, (index, error) in enumerate(zip(self._active, errors)):
            force = Fraction(float(self._coefficients[index])) * error
            for entry in range(jacobian.indptr[row], jacobian.indptr[row + 1]):
                result[jacobian.indices[entry]] += force * Fraction(float(jacobian.data[entry]))
        return np.array([_float(value) for value in result]).reshape(self._recipe.vertex_count, 3)

    def hessian(self, positions):
        positions = self._recipe._positions(positions)
        if not len(self._active):
            return csr_matrix((3 * self._recipe.vertex_count, 3 * self._recipe.vertex_count))
        jacobian = self._geometry.jacobian(positions).tocsr()
        upper = {}
        # At most twelve nonzeros per hinge. Accumulate each upper-triangular
        # entry exactly, then mirror the final rounded value. Two successive
        # sparse products can otherwise underflow asymmetrically.
        for row, index in enumerate(self._active):
            beta = Fraction(float(self._coefficients[index]))
            entries = [(int(jacobian.indices[entry]), Fraction(float(jacobian.data[entry])))
                       for entry in range(jacobian.indptr[row], jacobian.indptr[row + 1])
                       if jacobian.data[entry] != 0]
            entries.sort()
            for local, (first, a) in enumerate(entries):
                for second, b in entries[local:]:
                    key = first, second
                    upper[key] = upper.get(key, Fraction()) + beta * a * b
        rows, columns, values = [], [], []
        for (first, second), value in sorted(upper.items()):
            rounded = _float(value)
            if rounded != 0:
                rows.append(first); columns.append(second); values.append(rounded)
                if first != second:
                    rows.append(second); columns.append(first); values.append(rounded)
        size = 3 * self._recipe.vertex_count
        return csr_matrix((values, (rows, columns)), shape=(size, size))

    def energy_change(self, start, end):
        from solver_dihedral_increment import dihedral_angle_increment
        start, end = self._recipe._positions(start), self._recipe._positions(end)
        angles, delta = dihedral_angle_increment(self._geometry, start, end)
        result = Fraction()
        for index, angle, change in zip(self._active, angles, delta):
            coefficient = Fraction(float(self._coefficients[index]))
            residual = Fraction(float(angle)) - Fraction(float(self._targets[index]))
            change = Fraction(float(change))
            result += coefficient * change * (residual + change / 2)
        return _float(result)

    def diagnostics(self, positions):
        _, angles, _ = self._angles_errors(positions)
        sampled = [None] * len(self._targets)
        for index, angle in zip(self._active, angles):
            sampled[int(index)] = float(angle)
        return {"profile": PROFILE, "accepted": False, "coefficientPolicy": COEFFICIENT_POLICY,
            "hinges": self._recipe.hinges.tolist(), "targetsRadians": self._targets.tolist(),
            "activation": self._activation.tolist(), "activeHingeIndices": self._active.tolist(),
            "inactiveHingeIndices": np.flatnonzero(self._coefficients == 0).tolist(),
            "sampledActiveAnglesRadians": sampled, "coefficientWitnesses": self.coefficient_witnesses(),
            "energyJoules": self.energy(positions),
            "limitations": [
                "External relative-angle actuation with balanced cloth reactions; no fixed side, material bending calibration or physical tool geometry.",
                "Inactive hinges retain source identity but contribute exact zero without evaluating their angle. Null diagnostics mean unevaluated, not zero-angle geometry.",
                "Energy and work use the same rounded stiffness-times-activation coefficient; exact product residuals are numerical observations, not extra physical work.",
                "Endpoint angle and stable-increment evaluation does not prove continuous geometry validity. Callers must preserve complete source triangle, contact and relevant hinge-path guards.",
                "No source profile, motion schedule, runner capture, replay admission or construction operation is installed by this primitive."]}
