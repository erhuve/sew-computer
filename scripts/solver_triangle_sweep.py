"""Bounded nondegeneracy checks for every cloth triangle along affine motion.

Collision candidates omit incident primitives. A nonintersecting pair of endpoint
meshes therefore does not establish that material frames exist between them.
The fast guard uses normalized Bernstein coefficients and numerical margins;
the independent replay check uses exact rational endpoint coordinates.
"""

from fractions import Fraction

import numpy as np

def _inputs(start, end, faces, max_depth, max_intervals):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if any(isinstance(value, (bool, np.bool_)) for value in np.asarray(faces, dtype=object).ravel()):
        raise ValueError("Boolean triangle indices are not material vertex identities")
    faces = np.asarray(faces)
    if (start.ndim != 2 or start.shape[1] != 3 or end.shape != start.shape
            or not np.isfinite(start).all() or not np.isfinite(end).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or np.any(faces < 0) or np.any(faces >= len(start))
            or any(len(set(face)) != 3 for face in faces)
            or type(max_depth) is not int or not 0 <= max_depth <= 24
            or type(max_intervals) is not int or not 1 <= max_intervals <= 1000000):
        raise ValueError("Finite states, distinct integer triangles and bounded sweep budgets required")
    return start, end, faces


def triangle_sweep_safe(start, end, faces, *, max_depth=12, max_intervals=65536):
    """False includes paths unresolved within the numerical margin/budget."""
    from solver_hinge_sweep import _nonzero, _product, _split
    start, end, faces = _inputs(start, end, faces, max_depth, max_intervals)
    if not len(faces):
        return True
    if len(faces) > max_intervals:
        return False
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        points = np.stack((start[faces], end[faces]), axis=1)
        edges = points[:, :, 1:] - points[:, :, :1]
        scale = np.max(np.abs(edges), axis=(1, 2, 3))
        if not np.isfinite(edges).all() or np.any(scale <= 0):
            return False
        edges = edges / scale[:, None, None, None]
        normals = _product(edges[:, :, 0], edges[:, :, 1], np.cross)
        pending, visited = [(0, normals)], 0
        while pending:
            depth, coefficients = pending.pop()
            visited += len(coefficients)
            if visited > max_intervals:
                return False
            safe = _nonzero(coefficients)
            if np.all(safe):
                continue
            if depth == max_depth:
                return False
            left, right = _split(coefficients[~safe])
            pending.extend(((depth + 1, right), (depth + 1, left)))
    return True


def verify_triangle_sweep_exact(start, end, faces, *, max_depth=16, max_intervals=65536):
    """Prove nonzero area using exact float endpoints, independently of the guard.

    Each accepted interval has a fixed direction with strictly positive dot
    product against every quadratic Bernstein normal coefficient. Convexity
    then proves nonzero area throughout that interval. This is not a lower
    physical area bound, strain test, layer-order check or contact certificate.
    """
    start, end, faces = _inputs(start, end, faces, max_depth, max_intervals)
    if len(faces) > max_intervals:
        raise ValueError("Triangle path exact verification interval budget exhausted")

    def subtract(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def cross(a, b):
        return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])

    def midpoint(a, b):
        return tuple((x + y) / 2 for x, y in zip(a, b))

    visited, leaves = 0, 0
    for face in faces:
        endpoints = [[tuple(Fraction(float(x)) for x in point) for point in state[face]]
                     for state in (start, end)]
        first, second = [(subtract(p[1], p[0]), subtract(p[2], p[0])) for p in endpoints]
        coefficients = (cross(*first), midpoint(cross(first[0], second[1]),
                                               cross(second[0], first[1])), cross(*second))
        pending = [(0, coefficients)]
        while pending:
            depth, (a, b, c) = pending.pop()
            visited += 1
            if visited > max_intervals:
                raise ValueError("Triangle path exact verification interval budget exhausted")
            ab, bc = midpoint(a, b), midpoint(b, c)
            center = midpoint(ab, bc)
            if all(sum(x * y for x, y in zip(coefficient, center)) > 0
                   for coefficient in (a, b, c)):
                leaves += 1
                continue
            if depth == max_depth:
                raise ValueError("Triangle path degenerates or exceeds exact verification depth")
            pending.extend(((depth + 1, (center, bc, c)), (depth + 1, (a, ab, center))))
    return {"profile": "exact-rational-affine-triangle-nondegeneracy-v1",
            "triangles": len(faces), "verifiedLeaves": leaves, "visitedIntervals": visited}
