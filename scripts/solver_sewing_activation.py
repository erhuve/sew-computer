"""Strict per-canonical-row sewing activation; pending rows exert no force."""

import numpy as np


def _contains_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b" or (value.dtype.kind == "O" and any(_contains_bool(item) for item in value.flat))
    return isinstance(value, (list, tuple)) and any(_contains_bool(item) for item in value)


def validate_sewing_activation(value, count):
    """Return an isolated float64 vector without normalization or broadcasting."""
    if type(count) is not int or not 0 <= count <= 32768:
        raise ValueError("Sewing activation requires a bounded canonical row count")
    if value is None:
        return np.ones(count, dtype=float)
    if ((isinstance(value, np.ndarray) and value.shape != (count,))
            or (isinstance(value, np.ndarray) and value.dtype.kind not in "fiu")
            or (isinstance(value, (list, tuple)) and len(value) != count)
            or (isinstance(value, (list, tuple)) and any(isinstance(item, (list, tuple, np.ndarray)) for item in value))
            or not isinstance(value, (list, tuple, np.ndarray)) or _contains_bool(value)):
        raise ValueError("One finite non-Boolean activation in [0, 1] per canonical sewing row required")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("One finite non-Boolean activation in [0, 1] per canonical sewing row required") from error
    if (raw.shape != (count,) or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all()
            or np.any(raw < 0) or np.any(raw > 1)):
        raise ValueError("One finite non-Boolean activation in [0, 1] per canonical sewing row required")
    with np.errstate(under="ignore"):
        result = raw.astype(float, copy=True)
    if np.any((raw > 0) & (result == 0)):
        raise ValueError("Positive sewing activation cannot underflow to inactive")
    return result
