"""Exact spatial gap diagnostics for supplied continuous affine seam curves.

This module does not derive or authorize a material/source correspondence. It
checks one supplied saved-state curve pair and an explicit scalar reference.
All admitted rationals and all geometric arithmetic remain exact; no square
root, binary64 seam-parameter conversion, clipping, or normalization is used.
"""

from fractions import Fraction
import hashlib
import json
import math
import re


PROFILE = "exact-rational-continuous-spatial-distance-v1"
MAX_CELLS = 4096
MAX_JSON_BYTES = 8 * 1024 ** 2
MAX_OUTPUT_BYTES = 16 * 1024 ** 2
MAX_INTEGER_BITS = 4096
MAX_INTEGER_DIGITS = 1234
MAX_TOTAL_INTEGER_BITS = 2 ** 21
MAX_RESULT_INTEGER_BITS = 12288
CELL_FIELDS = frozenset(("interval", "startDifferenceMeters", "endDifferenceMeters",
                         "startTargetMeters", "endTargetMeters"))
_INTEGER = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z", re.ASCII)
_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z", re.ASCII)
LIMITATIONS = (
    "Exact diagnostic of the supplied continuous piecewise-affine difference curve and scalar target reference at one state only.",
    "Caller must independently derive and bind complete material paths, source identities, triangle traversal, registration maps, and all spatial breakpoints.",
    "Targets interpolate the supplied rational endpoint scalars on each cell; this is an explicit diagnostic reference, not a continuous force or stitching law.",
    "No source correspondence, stored sewing-row equivalence, activation policy, normal frame, material side, or control admission is established.",
    "No temporal or joint space-time bound, contact proof, nonintersection, integration accuracy, physical seam, construction completion, or garment acceptance follows.",
    "verified means the supplied partition and exact arithmetic were checked; withinTolerance is the independent geometric outcome and may be false.",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _rational(value):
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > MAX_RESULT_INTEGER_BITS:
        raise ValueError("Spatial diagnostic unresolved: derived rational complexity budget exceeded")
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _raw_rational(value):
    if type(value) is not dict or set(value) != {"numerator", "denominator"}:
        raise ValueError("Each exact rational requires only numerator and denominator strings")
    numerator, denominator = value["numerator"], value["denominator"]
    if (type(numerator) is not str or type(denominator) is not str
            or len(numerator) > MAX_INTEGER_DIGITS + 1 or len(denominator) > MAX_INTEGER_DIGITS
            or not _INTEGER.fullmatch(numerator) or not _POSITIVE_INTEGER.fullmatch(denominator)):
        raise ValueError("Bounded canonical decimal rational strings required")


def _parse_rational(value):
    numerator, denominator = int(value["numerator"]), int(value["denominator"])
    bits = abs(numerator).bit_length() + denominator.bit_length()
    if max(abs(numerator).bit_length(), denominator.bit_length()) > MAX_INTEGER_BITS:
        raise ValueError("Raw rational integer complexity budget exceeded")
    if math.gcd(numerator, denominator) != 1:
        raise ValueError("Canonical reduced rationals required; aliases are not admitted")
    return Fraction(numerator, denominator), bits


def _tolerance(value):
    if type(value) not in (int, float):
        raise ValueError("Tolerance must be a raw finite non-Boolean binary64 number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Finite exactly representable binary64 tolerance required") from error
    if (not math.isfinite(number) or not 0 <= number <= 100
            or type(value) is int and int(number) != value):
        raise ValueError("Exactly representable finite tolerance in [0, 100] meters required")
    return Fraction(number)


def _input(cells, tolerance_m):
    epsilon = _tolerance(tolerance_m)
    if type(cells) is not list or not 1 <= len(cells) <= MAX_CELLS:
        raise ValueError("A nonempty bounded JSON list of at most 4096 cells is required")
    parsed, encoded_size, total_bits = [], 256, 0
    previous = None
    for index, cell in enumerate(cells):
        if type(cell) is not dict or set(cell) != CELL_FIELDS:
            raise ValueError("Each spatial cell requires exactly the declared five fields")
        for key, length in (("interval", 2), ("startDifferenceMeters", 3), ("endDifferenceMeters", 3)):
            if type(cell[key]) is not list or len(cell[key]) != length:
                raise ValueError("Exact interval and three-component difference lists required")
        raw = [*cell["interval"], *cell["startDifferenceMeters"], *cell["endDifferenceMeters"],
               cell["startTargetMeters"], cell["endTargetMeters"]]
        for value in raw:
            _raw_rational(value)
        # Validate the bounded strings before serialization or large-int parsing.
        encoded_size += len(_json(cell)) + 1
        if encoded_size > MAX_JSON_BYTES:
            raise ValueError("Spatial diagnostic input JSON budget exceeded")
        values = []
        for value in raw:
            result, bits = _parse_rational(value)
            total_bits += bits
            if total_bits > MAX_TOTAL_INTEGER_BITS:
                raise ValueError("Spatial diagnostic aggregate rational complexity budget exceeded")
            values.append(result)
        lo, hi = values[:2]
        first, last = tuple(values[2:5]), tuple(values[5:8])
        d0, d1 = values[8:]
        if not 0 <= lo < hi <= 1 or (index == 0 and lo != 0):
            raise ValueError("Cells must be a strictly positive ordered exact partition of [0, 1]")
        if not 0 < d0 <= 100 or not 0 < d1 <= 100:
            raise ValueError("Positive rational targets of at most 100 meters are required")
        if previous is not None and (lo != previous[1] or first != previous[3] or d0 != previous[5]):
            raise ValueError("Adjacent intervals, differences, and targets must match exactly")
        previous = (lo, hi, first, last, d0, d1)
        parsed.append(previous)
    if previous[1] != 1:
        raise ValueError("The complete exact partition must end at 1")
    payload = {"profile": PROFILE, "cells": cells, "toleranceMeters": tolerance_m}
    return parsed, epsilon, hashlib.sha256(_json(payload)).hexdigest()


def _dot(first, second):
    return sum((a * b for a, b in zip(first, second)), Fraction())


def _extrema(coefficients, lower=Fraction(0), upper=Fraction(1)):
    """Return (value, local fraction), with earliest-fraction ties, for both extrema."""
    a, b, c = coefficients
    points = [lower, upper]
    if a:
        stationary = -b / (2 * a)
        if lower < stationary < upper:
            points.append(stationary)
    values = [((a * u + b) * u + c, u) for u in points]
    return min(values, key=lambda pair: (pair[0], pair[1])), min(values, key=lambda pair: (-pair[0], pair[1]))


def _witness(pair, interval):
    value, u = pair
    lo, hi = interval
    return {"valueMetersSquared": _rational(value), "atLocalFraction": _rational(u),
            "atFraction": _rational(lo + u * (hi - lo))}


def _polynomial(coefficients):
    return [_rational(value) for value in coefficients]


def verify_spatial_distance_exact(cells, *, tolerance_m):
    """Check |norm(D(s)) - d(s)| <= tolerance over every supplied spatial cell.

    ``cells`` must be a complete exact continuous partition of [0, 1]. Every
    rational uses reduced decimal-string numerator/denominator form. Tolerance
    failure returns a complete report with ``withinTolerance=False``. Invalid
    input or exhausted fixed resource bounds raises ValueError without a proof.
    This intentionally allows a zero gap when the explicit tolerance permits it.
    """
    parsed, epsilon, digest = _input(cells, tolerance_m)
    evidence, evidence_size = [], 0
    minimum, maximum = None, None
    signed_minimum, signed_maximum = None, None
    all_within = True
    for index, (lo, hi, first, last, d0, d1) in enumerate(parsed):
        interval = (lo, hi)
        delta, change = tuple(b - a for a, b in zip(first, last)), d1 - d0
        q = (_dot(delta, delta), 2 * _dot(first, delta), _dot(first, first))
        error = (q[0] - change ** 2, q[1] - 2 * d0 * change, q[2] - d0 ** 2)
        upper = (change ** 2 - q[0], 2 * (d0 + epsilon) * change - q[1], (d0 + epsilon) ** 2 - q[2])
        qmin, qmax = _extrema(q)
        emin, emax = _extrema(error)
        upper_min = _extrema(upper)[0]
        lower_report = None
        lower_min = None
        if max(d0, d1) >= epsilon:
            lower_u, upper_u = Fraction(0), Fraction(1)
            if d0 < epsilon:
                lower_u = (epsilon - d0) / change
            elif d1 < epsilon:
                upper_u = (epsilon - d0) / change
            lower = (q[0] - change ** 2, q[1] - 2 * (d0 - epsilon) * change, q[2] - (d0 - epsilon) ** 2)
            lower_min = _extrema(lower, lower_u, upper_u)[0]
            lower_report = {"polynomial": _polynomial(lower),
                            "localInterval": [_rational(lower_u), _rational(upper_u)],
                            "interval": [_rational(lo + lower_u * (hi - lo)), _rational(lo + upper_u * (hi - lo))],
                            "minimum": _witness(lower_min, interval)}
        within = upper_min[0] >= 0 and (lower_min is None or lower_min[0] >= 0)
        all_within = all_within and within
        item = {"cellIndex": index, "interval": [_rational(lo), _rational(hi)], "withinTolerance": within,
                "squaredGapPolynomial": _polynomial(q),
                "squaredGapMinimum": _witness(qmin, interval), "squaredGapMaximum": _witness(qmax, interval),
                "signedSquaredTargetErrorPolynomial": _polynomial(error),
                "signedSquaredTargetErrorMinimum": _witness(emin, interval),
                "signedSquaredTargetErrorMaximum": _witness(emax, interval),
                "upperMargin": {"polynomial": _polynomial(upper), "minimum": _witness(upper_min, interval)},
                "lowerMargin": lower_report}
        evidence_size += len(_json(item)) + 1
        if evidence_size > MAX_OUTPUT_BYTES - 65536:
            raise ValueError("Spatial diagnostic unresolved: output JSON budget exceeded")
        evidence.append(item)
        # Carry exact comparisons, not serialized values or rounded fractions.
        for kind, pair in (("qmin", qmin), ("qmax", qmax), ("emin", emin), ("emax", emax)):
            entry = (pair[0], lo + pair[1] * (hi - lo), index, pair)
            if kind == "qmin" and (minimum is None or entry[:3] < minimum[:3]):
                minimum = entry
            elif kind == "qmax" and (maximum is None or (-entry[0], *entry[1:3]) < (-maximum[0], *maximum[1:3])):
                maximum = entry
            elif kind == "emin" and (signed_minimum is None or entry[:3] < signed_minimum[:3]):
                signed_minimum = entry
            elif kind == "emax" and (signed_maximum is None or (-entry[0], *entry[1:3]) < (-signed_maximum[0], *signed_maximum[1:3])):
                signed_maximum = entry

    def global_witness(entry):
        index = entry[2]
        return dict(cellIndex=index, **_witness(entry[3], parsed[index][:2]))

    report = {"profile": PROFILE, "verified": True, "accepted": False, "withinTolerance": all_within,
              "inputSha256": digest, "toleranceMeters": tolerance_m, "exactToleranceMeters": _rational(epsilon),
              "cellCount": len(evidence), "polynomialConvention": "A*u^2+B*u+C; u=(s-cellStart)/(cellEnd-cellStart)",
              "signedSquaredTargetErrorConvention": "Q-d^2 in meters squared; not (sqrt(Q)-d)^2",
              "tiePolicy": "Earliest exact common fraction, then earliest cell index",
              "squaredGapMinimum": global_witness(minimum), "squaredGapMaximum": global_witness(maximum),
              "signedSquaredTargetErrorMinimum": global_witness(signed_minimum),
              "signedSquaredTargetErrorMaximum": global_witness(signed_maximum),
              "cells": evidence, "limitations": list(LIMITATIONS)}
    if len(_json(report)) > MAX_OUTPUT_BYTES:
        raise ValueError("Spatial diagnostic unresolved: output JSON budget exceeded")
    return report
