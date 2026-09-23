"""Stateless, monotone sewing-row activation on the original time interval.

The caller must bind the ordered row IDs to the complete source sewing rows.
Matching these IDs alone establishes neither source correspondence nor phase
execution, seam closure, construction completion, or physical acceptance.
"""

from fractions import Fraction
import re

import numpy as np


PROFILE = "sewing-row-activation-v1"
MAX_ROWS = 4096
MAX_KNOTS = 65
MAX_FRACTION_DENOMINATOR = 2 ** 40


def _identity(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value) is not None


def _fraction(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, Fraction)):
        raise ValueError("Finite bounded dyadic schedule fraction required")
    try:
        fraction = Fraction(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Finite bounded dyadic schedule fraction required") from error
    if (not 0 <= fraction <= 1 or fraction.denominator > MAX_FRACTION_DENOMINATOR
            or fraction.denominator & (fraction.denominator - 1)):
        raise ValueError("Schedule fraction must lie in [0, 1] on the bounded dyadic grid")
    return fraction


def _activation(value, count):
    # Check raw shape and Boolean scalars before coercion can hide a Boolean
    # among numeric values or allocate from an unbounded nested input.
    if isinstance(value, np.ndarray):
        valid_shape = value.shape == (count,)
    else:
        valid_shape = (isinstance(value, (list, tuple)) and len(value) == count
                       and all(not isinstance(item, (list, tuple, np.ndarray)) for item in value))
    if not valid_shape:
        raise ValueError("Activation must have exactly one bounded scalar per sewing row")
    if any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError("Activation cannot contain Boolean values")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Finite numeric sewing activation required") from error
    if (raw.dtype.kind not in "iuf" or not np.isfinite(raw).all()
            or np.any(raw < 0) or np.any(raw > 1)):
        raise ValueError("Finite numeric sewing activation in [0, 1] required")
    result = raw.astype(np.float64, copy=True)
    if np.any((raw > 0) & (result == 0)):
        raise ValueError("Positive sewing activation cannot underflow to inactive")
    return result


class SewingActivationSchedule:
    """Immutable inputs; independently sampled activation, with no release."""

    __slots__ = ("_row_ids", "_fractions", "_activation", "_subdivisions")

    def __setattr__(self, name, value):
        raise AttributeError("Captured sewing activation schedules are immutable")

    def __delattr__(self, name):
        raise AttributeError("Captured sewing activation schedules are immutable")

    def __init__(self, recipe, subdivisions, *, row_ids):
        if (type(subdivisions) is not int or not 1 <= subdivisions <= 4096
                or subdivisions & (subdivisions - 1)):
            raise ValueError("Bounded power-of-two initial subdivisions required")
        if (not isinstance(row_ids, (list, tuple)) or not 1 <= len(row_ids) <= MAX_ROWS
                or not all(_identity(value) for value in row_ids) or len(set(row_ids)) != len(row_ids)):
            raise ValueError("Bounded unique ordered sewing-row identities required")
        if (not isinstance(recipe, dict) or set(recipe) != {"profile", "rowIds", "knots"}
                or recipe["profile"] != PROFILE or not isinstance(recipe["rowIds"], list)
                or len(recipe["rowIds"]) != len(row_ids)
                or not all(_identity(value) for value in recipe["rowIds"])
                or recipe["rowIds"] != list(row_ids)
                or not isinstance(recipe["knots"], list) or not 2 <= len(recipe["knots"]) <= MAX_KNOTS):
            raise ValueError("Explicit sewing activation recipe with matching ordered row IDs required")
        fractions, activation = [], []
        for knot in recipe["knots"]:
            if not isinstance(knot, dict) or set(knot) != {"fraction", "activation"}:
                raise ValueError("Each sewing activation knot requires only fraction and activation")
            fraction = _fraction(knot["fraction"])
            if (fraction * subdivisions).denominator != 1 or (fractions and fraction <= fractions[-1]):
                raise ValueError("Ordered dyadic knots must coincide with initial interval boundaries")
            active = _activation(knot["activation"], len(row_ids))
            if activation and np.any(active < activation[-1]):
                raise ValueError("Sewing activation must be nondecreasing for every row; held rows cannot release")
            fractions.append(fraction)
            activation.append(active)
        if fractions[0] != 0 or fractions[-1] != 1:
            raise ValueError("Sewing activation schedule must span exactly [0, 1]")
        values = np.array(activation, dtype=np.float64)
        # Immutable bytes back the array: callers cannot re-enable writes to
        # a captured view by changing its NumPy writeable flag.
        frozen = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
        object.__setattr__(self, "_row_ids", tuple(row_ids))
        object.__setattr__(self, "_fractions", tuple(fractions))
        object.__setattr__(self, "_activation", frozen)
        object.__setattr__(self, "_subdivisions", subdivisions)

    @property
    def row_ids(self):
        return self._row_ids

    def parameters(self, fraction):
        fraction = _fraction(fraction)
        for index, knot in enumerate(self._fractions):
            if fraction == knot:
                return self._activation[index].copy()
        for index, (lower, upper) in enumerate(zip(self._fractions[:-1], self._fractions[1:])):
            if lower < fraction < upper:
                amount = float((fraction - lower) / (upper - lower))
                # Increment interpolation preserves inactive and held values
                # exactly, including nonzero arbitrary initial activation.
                with np.errstate(under="ignore"):
                    result = self._activation[index] + amount * (self._activation[index + 1] - self._activation[index])
                # At an interior fraction, a positive upper endpoint and
                # nonnegative lower endpoint imply strictly positive exact
                # activation. Do not silently turn it into a pending row.
                if np.any((self._activation[index + 1] > 0) & (result == 0)):
                    raise ValueError("Positive interpolated sewing activation underflows to inactive")
                return result
        raise ValueError("Sewing activation schedule does not cover the requested fraction")
