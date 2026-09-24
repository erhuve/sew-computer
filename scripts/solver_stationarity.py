"""Explicit tightening of the existing numerical force stopping criterion.

This criterion bounds reported numerical stationarity, not temporal error or
all physical-force error. It cannot loosen the legacy acceptance ceiling.
"""
import math

import numpy as np


DEFAULT_STATIONARITY_NEWTONS = 1e-6


def stationarity_tolerance(value):
    from solver_controlled_fold import _binary64
    result = _binary64(value)
    if not 0 < result <= DEFAULT_STATIONARITY_NEWTONS:
        raise ValueError("Stationarity tolerance must satisfy 0 < tolerance <= 1e-6 N")
    return result


def validate_tightened_report(report, tolerance):
    """Bind the strict request to the final public numerical diagnostics."""
    tolerance = stationarity_tolerance(tolerance)
    if tolerance == DEFAULT_STATIONARITY_NEWTONS:
        return  # Historical mock/legacy reports need not declare this field.
    norm = report.get("gradientInfinityNorm")
    declared = report.get("stationarityToleranceN")
    if (report.get("converged") is not True
            or isinstance(norm, (bool, np.bool_)) or not isinstance(norm, (int, float, np.integer, np.floating))
            or not math.isfinite(float(norm)) or not 0 <= norm <= tolerance
            or stationarity_tolerance(declared) != tolerance):
        raise ValueError("Final stationarity diagnostics do not satisfy the requested tightened criterion")
    if "cableStationarity" in report:
        from fractions import Fraction
        from solver_attempt_journal import same
        from solver_cable_integration import _rational, stationarity
        bounds = report["cableStationarity"]
        if (type(bounds) is not dict
                or stationarity_tolerance(bounds.get("toleranceNewtons")) != tolerance):
            raise ValueError("Cable stationarity does not declare the requested tightened criterion")
        try:
            expected = stationarity(np.array([float(norm)]),
                _rational(bounds["totalGradientErrorBoundNewtons"]),
                _rational(bounds["cableGradientErrorBoundNewtons"]),
                _rational(bounds["assemblyRoundingBoundNewtons"]), tolerance_newtons=tolerance)
        except KeyError as error:
            raise ValueError("Complete tightened cable stationarity bounds required") from error
        if (not same(bounds, expected)
                or _rational(expected["stationarityUpperBoundNewtons"]) > Fraction(tolerance)):
            raise ValueError("Final cable bounds do not satisfy the requested tightened criterion")
