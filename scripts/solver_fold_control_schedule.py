"""Immutable, independently sampled per-hinge angle and activation controls.

The ordered native hinge identity must be bound to an admitted source/model by
the caller. This schedule supplies no source proof, force, time integration,
fixed-region motion or construction acceptance. Angles are never wrapped and
activation may engage, hold, release and engage again.
"""

from bisect import bisect_left
from fractions import Fraction
import math

import numpy as np


PROFILE = "fold-angle-activation-v1"
MAX_HINGES = 4096
MAX_KNOTS = 65
MAX_FRACTION_DENOMINATOR = 2**40
ANGLE_LIMIT = math.pi - 1e-8


def _fraction(value):
    if type(value) not in (int, float, Fraction):
        raise ValueError("Finite non-Boolean dyadic control fraction required")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Finite non-Boolean dyadic control fraction required")
    if type(value) is Fraction and (value.numerator.bit_length() > 41 or value.denominator.bit_length() > 41):
        raise ValueError("Control fraction exceeds the bounded dyadic grid")
    result = Fraction(value)
    if (not 0 <= result <= 1 or result.denominator > MAX_FRACTION_DENOMINATOR
            or result.denominator & (result.denominator-1)):
        raise ValueError("Control fraction must lie on the bounded dyadic grid in [0,1]")
    return result


def _hinges(value, *, native_array=False):
    if native_array and isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1] != 4 or not 1 <= value.shape[0] <= MAX_HINGES or value.dtype.kind not in "iu":
            raise ValueError("Bounded integer native hinge array required")
        if np.any(value < 0) or np.any(value > 2**63-1):
            raise ValueError("Nonnegative bounded native vertex indices required")
        rows = value.tolist()
    else:
        allowed = (list, tuple) if native_array else (list,)
        if type(value) not in allowed or not 1 <= len(value) <= MAX_HINGES:
            raise ValueError("One to 4096 explicitly ordered native hinges required")
        rows = value
    result = []
    for row in rows:
        allowed = (list, tuple) if native_array else (list,)
        if (type(row) not in allowed or len(row) != 4
                or any(type(vertex) is not int or not 0 <= vertex <= 2**63-1 for vertex in row)
                or len(set(row)) != 4):
            raise ValueError("Four distinct nonnegative raw integer vertices per hinge required")
        result.append(tuple(row))
    if len(set(result)) != len(result):
        raise ValueError("Unique complete ordered native hinge tuples required")
    return tuple(result)


def _values(value, count, *, activation):
    if type(value) is not list or len(value) != count:
        raise ValueError("Exactly one raw control scalar per ordered hinge required")
    for item in value:
        if type(item) not in (int, float):
            raise ValueError("Finite raw non-Boolean control numbers required")
        if activation and not 0 <= item <= 1 or not activation and abs(item) >= ANGLE_LIMIT:
            raise ValueError("Activation must lie in [0,1] and angles strictly inside the principal branch")
        if not math.isfinite(item):
            raise ValueError("Finite raw non-Boolean control numbers required")
    return np.asarray(value, dtype=np.float64)


def _frozen(values):
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


class FoldControlSchedule:
    """Stateless exact interpolation on original fractions; knot bytes retained."""

    __slots__ = ("_hinges", "_fractions", "_targets", "_activation", "_subdivisions")

    def __setattr__(self, name, value):
        raise AttributeError("Fold control schedules are immutable")

    def __delattr__(self, name):
        raise AttributeError("Fold control schedules are immutable")

    def __init__(self, recipe, subdivisions, *, hinges):
        if (type(subdivisions) is not int or not 1 <= subdivisions <= 4096
                or subdivisions & (subdivisions-1)):
            raise ValueError("Power-of-two initial subdivisions in [1,4096] required")
        identity = _hinges(hinges, native_array=True)
        if (type(recipe) is not dict or set(recipe) != {"profile", "hinges", "knots"}
                or recipe["profile"] != PROFILE or _hinges(recipe["hinges"]) != identity
                or type(recipe["knots"]) is not list or not 2 <= len(recipe["knots"]) <= MAX_KNOTS):
            raise ValueError("Exact fold control recipe with complete ordered native hinge identity required")
        fractions, targets, activation = [], [], []
        for knot in recipe["knots"]:
            if type(knot) is not dict or set(knot) != {"fraction", "targetsRadians", "activation"}:
                raise ValueError("Each knot requires only fraction, targetsRadians and activation")
            if type(knot["fraction"]) not in (int, float):
                raise ValueError("Raw finite JSON knot fraction required")
            fraction = _fraction(knot["fraction"])
            if (fraction*subdivisions).denominator != 1 or fractions and fraction <= fractions[-1]:
                raise ValueError("Strictly ordered knots on the original subdivision grid required")
            fractions.append(fraction)
            targets.append(_values(knot["targetsRadians"], len(identity), activation=False))
            activation.append(_values(knot["activation"], len(identity), activation=True))
        if fractions[0] != 0 or fractions[-1] != 1:
            raise ValueError("Fold control knots must span exactly [0,1]")
        object.__setattr__(self, "_hinges", identity)
        object.__setattr__(self, "_fractions", tuple(fractions))
        object.__setattr__(self, "_targets", _frozen(targets))
        object.__setattr__(self, "_activation", _frozen(activation))
        object.__setattr__(self, "_subdivisions", subdivisions)

    @property
    def hinges(self):
        return self._hinges

    @property
    def fractions(self):
        return self._fractions

    @property
    def subdivisions(self):
        return self._subdivisions

    def parameters(self, fraction):
        fraction = _fraction(fraction)
        upper = bisect_left(self._fractions, fraction)
        if upper < len(self._fractions) and fraction == self._fractions[upper]:
            return self._targets[upper].copy(), self._activation[upper].copy()
        if upper == 0 or upper == len(self._fractions):
            raise ValueError("Fold control schedule does not cover the original fraction")
        lower = upper-1
        amount = (fraction-self._fractions[lower])/(self._fractions[upper]-self._fractions[lower])
        outputs = []
        for activation, values in ((False, self._targets), (True, self._activation)):
            result = np.empty(len(self._hinges), dtype=np.float64)
            for index, (start, end) in enumerate(zip(values[lower], values[upper])):
                exact = (1-amount)*Fraction(float(start)) + amount*Fraction(float(end))
                converted = float(exact)
                if activation and exact > 0 and converted == 0:
                    raise ValueError("Positive interior fold activation underflows to inactive")
                # Fraction-to-float is one nearest-even conversion. In
                # particular, no rounded fraction or rounded difference enters
                # the interpolation; exact zero has canonical positive sign.
                result[index] = converted
            outputs.append(result)
        return tuple(outputs)
