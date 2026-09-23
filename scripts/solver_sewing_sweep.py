"""Exact temporal bounds for supplied sampled scalar-distance sewing rows.

This current verifier has no dependency on captured numerical solver modules.
The caller binds rows to source, verifies endpoint controls and schedule knots,
and separately verifies contact, material triangles and construction semantics.
"""

from fractions import Fraction
import hashlib
import json
import math

import numpy as np


PROFILE = "exact-rational-affine-sampled-distance-sewing-v1"
MAX_ROWS = 4096
MAX_VERTICES = 25000
MAX_TERMS = 6
MAX_JSON_BYTES = 8 * 1024 ** 2


def _number(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, np.floating) and value.dtype.itemsize > 8):
        raise ValueError("Finite non-Boolean binary64-or-narrower numeric scalars required")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError("Finite binary64 scalar required") from error
    if (not math.isfinite(result)
            or isinstance(value, (int, np.integer)) and int(result) != int(value)):
        raise ValueError("Finite exactly representable binary64 input required")
    return result


def _points(value):
    if isinstance(value, np.ndarray):
        shape = value.shape
        if len(shape) != 2 or shape[1] != 3 or not 1 <= shape[0] <= MAX_VERTICES:
            raise ValueError("Bounded N by 3 endpoint positions required")
    else:
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_VERTICES:
            raise ValueError("Bounded N by 3 endpoint positions required")
        for row in value:
            if (isinstance(row, np.ndarray) and row.shape != (3,)
                    or not isinstance(row, (list, tuple, np.ndarray))
                    or isinstance(row, (list, tuple)) and len(row) != 3):
                raise ValueError("Bounded N by 3 endpoint positions required")
    return tuple(tuple(_number(component) for component in row) for row in value)


def _vector(value, count, label):
    if isinstance(value, np.ndarray):
        valid = value.shape == (count,)
    else:
        valid = isinstance(value, (list, tuple)) and len(value) == count
    if not valid:
        raise ValueError("One " + label + " per complete source row required")
    return tuple(_number(item) for item in value)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _rational(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _minimum(coefficients, lower=Fraction(0), upper=Fraction(1)):
    """Exact minimum of A*u*u+B*u+C on a closed rational interval."""
    a, b, c = coefficients
    points = [lower, upper]
    if a > 0:
        vertex = -b / (2 * a)
        if lower < vertex < upper:
            points.append(vertex)
    return min((((a * u + b) * u + c, u) for u in points), key=lambda item: (item[0], item[1]))


def _witness(coefficients, lower=Fraction(0), upper=Fraction(1)):
    value, u = _minimum(coefficients, lower, upper)
    return value, {"value": _rational(value), "atLocalFraction": _rational(u),
                   "interval": [_rational(lower), _rational(upper)]}


def verify_control_interval(start, end, knots):
    """Reject a saved interval crossing any independently validated control knot.

    ``knots`` may concatenate both source schedules, with repeated boundaries.
    This helper does not validate schedule contents, order or control values.
    """
    def fraction(value):
        result = value if isinstance(value, Fraction) else Fraction(_number(value))
        if (not 0 <= result <= 1 or result.denominator > 2 ** 40
                or result.denominator & (result.denominator - 1)):
            raise ValueError("Bounded dyadic original control fractions required")
        return result

    lower, upper = fraction(start), fraction(end)
    if lower >= upper:
        raise ValueError("Strictly increasing original control interval required")
    if not isinstance(knots, (list, tuple)) or len(knots) > 130:
        raise ValueError("At most 130 captured control knots required")
    for value in knots:
        knot = fraction(value)
        if lower < knot < upper:
            raise ValueError("Saved interval crosses a captured control knot")
    return lower, upper


def verify_distance_sewing_sweep_exact(start, end, rows, *, row_ids,
        initial_targets, final_targets, initial_activation, final_activation, tolerance_m):
    """Certify unweighted distance error throughout each active affine row path.

    All products/sums below use the original admitted binary inputs as exact
    rationals, not rounded CSR anchors. Targets interpolate the supplied endpoint
    scalars; this is a geometric reference, not an integration of numerical
    forces. Either positive endpoint weight checks the entire closed interval,
    including a zero-weight engagement endpoint. No rows are normalized.
    """
    start, end = _points(start), _points(end)
    if len(start) != len(end):
        raise ValueError("Matching endpoint vertex counts required")
    if not isinstance(rows, (tuple, list)) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("Between one and 4096 ordered sewing rows required")
    count = len(rows)
    if (not isinstance(row_ids, (tuple, list)) or len(row_ids) != count
            or any(type(identity) is not str or not 1 <= len(identity) <= 128
                   or any(not (character.isascii() and (character.isalnum() or character in "_.:/-"))
                          for character in identity) for identity in row_ids)
            or len(set(row_ids)) != count):
        raise ValueError("Complete unique bounded ordered row identities required")
    ordered_rows = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or not 1 <= len(row) <= MAX_TERMS:
            raise ValueError("One to six distinct nonzero terms per sewing row required")
        terms, seen = [], set()
        for term in row:
            if not isinstance(term, (tuple, list)) or len(term) != 2:
                raise ValueError("Each sewing term requires a vertex and coefficient")
            vertex, weight = term
            if (isinstance(vertex, (bool, np.bool_)) or not isinstance(vertex, (int, np.integer))
                    or not 0 <= vertex < len(start) or int(vertex) in seen):
                raise ValueError("Distinct bounded integer sewing vertices required")
            weight = _number(weight)
            if not 0 < abs(weight) <= 1:
                raise ValueError("Nonzero sewing coefficients within [-1, 1] required")
            seen.add(int(vertex))
            terms.append((int(vertex), weight))
        ordered_rows.append(tuple(terms))
    before_d = _vector(initial_targets, count, "initial target")
    after_d = _vector(final_targets, count, "final target")
    before_a = _vector(initial_activation, count, "initial activation")
    after_a = _vector(final_activation, count, "final activation")
    tolerance = _number(tolerance_m)
    if tolerance <= 0 or any(value <= 0 for value in (*before_d, *after_d)):
        raise ValueError("Positive scalar targets and explicit positive tolerance required")
    if any(not 0 <= lower <= upper <= 1 for lower, upper in zip(before_a, after_a)):
        raise ValueError("Monotone activation in [0, 1] required")
    payload = {"profile": PROFILE, "start": start, "end": end, "rows": ordered_rows,
               "rowIds": list(row_ids), "initialTargets": before_d, "finalTargets": after_d,
               "initialActivation": before_a, "finalActivation": after_a, "toleranceM": tolerance}
    encoded = _json(payload)
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("Sewing path input exceeds bounded JSON budget")
    input_digest = hashlib.sha256(encoded).hexdigest()
    del payload, encoded
    epsilon = Fraction(tolerance)
    checked, pending, evidence, evidence_size = 0, 0, [], 0
    # Cache only used coordinates; pending rows never evaluate anchor geometry.
    coordinates = {}

    def anchor(state_index, row):
        values = [Fraction(), Fraction(), Fraction()]
        state = start if state_index == 0 else end
        for vertex, weight in row:
            key = state_index, vertex
            if key not in coordinates:
                coordinates[key] = tuple(Fraction(value) for value in state[vertex])
            coefficient = Fraction(weight)
            for axis in range(3):
                values[axis] += coefficient * coordinates[key][axis]
        return values

    for index, row in enumerate(ordered_rows):
        item = {"rowIndex": index, "rowId": row_ids[index]}
        if before_a[index] == after_a[index] == 0:
            pending += 1
            item["status"] = "pending"
        else:
            checked += 1
            x0, x1 = anchor(0, row), anchor(1, row)
            delta = [b - a for a, b in zip(x0, x1)]
            q = (sum((x * x for x in delta), Fraction()),
                 2 * sum((x * dx for x, dx in zip(x0, delta)), Fraction()),
                 sum((x * x for x in x0), Fraction()))
            value, distance_witness = _witness(q)
            if value <= 0:
                raise ValueError(f"Row {index} has nonpositive distance on the active closed path")
            d0, d1 = Fraction(before_d[index]), Fraction(after_d[index])
            change = d1 - d0
            upper = (change * change - q[0], 2 * (d0 + epsilon) * change - q[1],
                     (d0 + epsilon) ** 2 - q[2])
            value, upper_witness = _witness(upper)
            if value < 0:
                raise ValueError(f"Row {index} exceeds the upper scalar-distance error bound")
            lower_witness = None
            if max(d0, d1) > epsilon:
                lower, upper_time = Fraction(0), Fraction(1)
                if d0 <= epsilon:
                    lower = (epsilon - d0) / change
                elif d1 <= epsilon:
                    upper_time = (epsilon - d0) / change
                polynomial = (q[0] - change * change, q[1] - 2 * (d0 - epsilon) * change,
                              q[2] - (d0 - epsilon) ** 2)
                value, lower_witness = _witness(polynomial, lower, upper_time)
                if value < 0:
                    raise ValueError(f"Row {index} exceeds the lower scalar-distance error bound")
            item.update(status="checked", distanceSquaredMinimum=distance_witness,
                        upperErrorMarginMinimum=upper_witness, lowerErrorMarginMinimum=lower_witness)
        evidence_size += len(_json(item))
        if evidence_size > MAX_JSON_BYTES - 4096:
            raise ValueError("Sewing path certificate exceeds bounded JSON budget")
        evidence.append(item)
    return {"profile": PROFILE, "accepted": False, "verified": True,
            "inputSha256": input_digest, "toleranceM": tolerance,
            "rows": evidence, "checkedRows": checked, "pendingRows": pending,
            "geometry": "Exact binary-input coefficients and endpoint positions; no rounded CSR anchors",
            "targetReference": "Affine interpolation of supplied binary64 endpoint target scalars",
            "activationPolicy": "Either positive endpoint checks the complete closed path with strict nonzero anchor distance; both zero are pending",
            "limitations": [
                "Caller must independently bind complete ordered rows, endpoint controls, source identity, original fractions and schedule-knot coverage.",
                "Closed-path engagement checking is conservative: even a zero-activation endpoint must have nonzero anchor distance when the other endpoint is active.",
                "Temporal bound for sampled material anchors only; no spatial seam coverage, continuous stitched joint or finite-thickness seam proof.",
                "Affine saved-state motion and endpoint-target reference only; no continuous numerical force evaluation or integration accuracy claim.",
                "No contact, triangle, material-side, wrap, turning, construction completion or garment acceptance follows from this certificate."]}
