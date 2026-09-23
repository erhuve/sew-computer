"""Bounded minimum distance between two saved-state triangle surface subsets.

All feature parameters and comparisons use exact rationals of the admitted
binary64 coordinates. Distances and closest-point witnesses are rounded for
reporting. This is an endpoint diagnostic, not a motion/contact certificate,
an admission threshold, a stitch-joint model or garment acceptance.
"""

from fractions import Fraction
import hashlib
import json
import math

import numpy as np


PROFILE = "saved-state-triangle-surface-distance-v1"
MAX_VERTICES = 25000
MAX_FACES = 50000
MAX_PAIRS = 1000000


def _number(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, np.floating) and value.dtype.itemsize > 8):
        raise ValueError("Finite binary64-or-narrower non-Boolean coordinates required")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Finite binary64 coordinates required") from error
    if (not math.isfinite(result)
            or isinstance(value, (int, np.integer)) and int(result) != int(value)):
        raise ValueError("Coordinates must be exactly representable finite binary64 values")
    return result


def _matrix(value, maximum, label):
    if isinstance(value, np.ndarray):
        valid = value.ndim == 2 and value.shape[1] == 3 and 1 <= len(value) <= maximum
    else:
        valid = isinstance(value, (list, tuple)) and 1 <= len(value) <= maximum
        if valid:
            valid = all((isinstance(row, np.ndarray) and row.shape == (3,))
                        or (isinstance(row, (list, tuple)) and len(row) == 3) for row in value)
    if not valid:
        raise ValueError("Bounded nonempty three-column " + label + " required")
    return value


def _index(value, count):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 0 <= value < count):
        raise ValueError("Bounded non-Boolean integer identities required")
    return int(value)


def _group(value, count):
    if isinstance(value, np.ndarray):
        valid = value.ndim == 1 and 1 <= len(value) <= count
    else:
        valid = isinstance(value, (list, tuple)) and 1 <= len(value) <= count
    if not valid:
        raise ValueError("Bounded nonempty face identity groups required")
    result = tuple(_index(item, count) for item in value)
    if len(set(result)) != len(result):
        raise ValueError("Face identity groups cannot contain duplicates")
    return tuple(sorted(result))


def _subtract(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _add_scaled(a, vector, scale):
    return tuple(x+scale*y for x, y in zip(a, vector))


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Fraction())


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _square(a, b):
    delta = _subtract(a, b)
    return _dot(delta, delta)


def _point_segment(point, start, end):
    edge = _subtract(end, start)
    amount = _dot(_subtract(point, start), edge) / _dot(edge, edge)
    return _add_scaled(start, edge, max(Fraction(), min(Fraction(1), amount)))


def _inside(point, triangle, normal):
    return all(_dot(_cross(_subtract(triangle[(i+1) % 3], triangle[i]),
                          _subtract(point, triangle[i])), normal) >= 0 for i in range(3))


def _point_face(point, triangle, normal):
    offset = _dot(_subtract(point, triangle[0]), normal) / _dot(normal, normal)
    projection = _add_scaled(point, normal, -offset)
    if _inside(projection, triangle, normal):
        return projection
    choices = [_point_segment(point, triangle[i], triangle[(i+1) % 3]) for i in range(3)]
    return min(choices, key=lambda candidate: _square(point, candidate))


def _segment_face(start, end, triangle, normal):
    first = _dot(_subtract(start, triangle[0]), normal)
    second = _dot(_subtract(end, triangle[0]), normal)
    if first * second > 0 or first == second:
        # Coplanar containment/crossings are handled by vertex-face and
        # edge-edge closest features, including collinear edge overlap.
        return None
    point = _add_scaled(start, _subtract(end, start), first / (first-second))
    return point if _inside(point, triangle, normal) else None


def _segment_segment(p0, p1, q0, q1):
    first, second, between = _subtract(p1, p0), _subtract(q1, q0), _subtract(p0, q0)
    a, b, c = _dot(first, first), _dot(first, second), _dot(second, second)
    d, e = _dot(first, between), _dot(second, between)
    determinant = a*c-b*b
    candidates = [(p0, _point_segment(p0, q0, q1)), (p1, _point_segment(p1, q0, q1)),
                  (_point_segment(q0, p0, p1), q0), (_point_segment(q1, p0, p1), q1)]
    if determinant > 0:
        s, t = (b*e-c*d)/determinant, (a*e-b*d)/determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((_add_scaled(p0, first, s), _add_scaled(q0, second, t)))
    return min(candidates, key=lambda pair: _square(*pair))


def _triangle_pair(first, second, first_normal, second_normal, counts):
    for left, right, normal in ((first, second, second_normal), (second, first, first_normal)):
        for edge in range(3):
            counts["segmentFaceTests"] += 1
            intersection = _segment_face(left[edge], left[(edge+1) % 3], right, normal)
            if intersection is not None:
                return Fraction(), intersection, intersection
    candidates = []
    for point in first:
        counts["vertexFaceTests"] += 1
        candidates.append((point, _point_face(point, second, second_normal)))
    for point in second:
        counts["vertexFaceTests"] += 1
        candidates.append((_point_face(point, first, first_normal), point))
    for i in range(3):
        for j in range(3):
            counts["edgeEdgeTests"] += 1
            candidates.append(_segment_segment(first[i], first[(i+1) % 3],
                                               second[j], second[(j+1) % 3]))
    p, q = min(candidates, key=lambda pair: _square(*pair))
    return _square(p, q), p, q


def _sqrt_fraction(value):
    if value == 0:
        return 0.
    # Normalize before binary64 conversion so a representable distance is not
    # lost by first underflowing its squared distance (or overflowing it).
    exponent = 2 * ((value.numerator.bit_length()-value.denominator.bit_length()) // 2)
    scaled = value / (Fraction(2) ** exponent)
    try:
        result = math.ldexp(math.sqrt(float(scaled)), exponent // 2)
    except OverflowError as error:
        raise ValueError("Surface distance is outside binary64 reporting range") from error
    if not math.isfinite(result) or result == 0:
        raise ValueError("Positive surface distance is outside binary64 reporting range")
    return result


def _rational(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def minimum_surface_distance(positions, faces, first_face_indices, second_face_indices):
    """Return the complete minimum over two disjoint triangle identity groups.

    Every selected face must be nondegenerate. All canonical face identities
    are validated. Exact AABB lower bounds and the nonnegativity of distance
    may skip pairs; counts account for every requested pair. No selected
    surface point is approximated by a nearest vertex.
    """
    positions = _matrix(positions, MAX_VERTICES, "positions")
    faces = _matrix(faces, MAX_FACES, "triangles")
    first_ids, second_ids = _group(first_face_indices, len(faces)), _group(second_face_indices, len(faces))
    if set(first_ids) & set(second_ids):
        raise ValueError("Surface groups must have disjoint canonical face identities")
    pair_count = len(first_ids) * len(second_ids)
    if pair_count > MAX_PAIRS:
        raise ValueError("Surface triangle-pair budget exceeded")
    # Resource admission precedes coordinate conversion/exact arithmetic.
    points = tuple(tuple(_number(item) for item in row) for row in positions)
    topology = tuple(tuple(_index(item, len(points)) for item in face) for face in faces)
    if any(len(set(face)) != 3 for face in topology):
        raise ValueError("Each triangle requires three distinct material vertices")
    payload = {"profile": PROFILE, "positions": points, "faces": topology,
               "firstFaceIndices": first_ids, "secondFaceIndices": second_ids}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    input_hash = hashlib.sha256(encoded).hexdigest()
    del payload, encoded
    exact_points, geometry, boxes = {}, {}, {}
    for index in sorted(set(first_ids) | set(second_ids)):
        for vertex in topology[index]:
            if vertex not in exact_points:
                exact_points[vertex] = tuple(Fraction(value) for value in points[vertex])
        triangle = tuple(exact_points[vertex] for vertex in topology[index])
        normal = _cross(_subtract(triangle[1], triangle[0]), _subtract(triangle[2], triangle[0]))
        if _dot(normal, normal) == 0:
            raise ValueError(f"Selected triangle {index} is degenerate")
        geometry[index] = triangle, normal
        boxes[index] = tuple((min(point[axis] for point in triangle), max(point[axis] for point in triangle))
                             for axis in range(3))
    counts = {"requestedTrianglePairs": pair_count, "evaluatedTrianglePairs": 0,
              "aabbPrunedTrianglePairs": 0, "zeroDistancePrunedTrianglePairs": 0,
              "triangleAabbTests": 0, "validatedFaces": len(geometry),
              "vertexFaceTests": 0, "edgeEdgeTests": 0, "segmentFaceTests": 0}
    best = None
    for first_id in first_ids:
        for second_id in second_ids:
            if best is not None and best[0] == 0:
                counts["zeroDistancePrunedTrianglePairs"] += 1
                continue
            counts["triangleAabbTests"] += 1
            lower = sum((max(a[0]-b[1], b[0]-a[1], Fraction()) ** 2
                         for a, b in zip(boxes[first_id], boxes[second_id])), Fraction())
            if best is not None and lower > best[0]:
                counts["aabbPrunedTrianglePairs"] += 1
                continue
            counts["evaluatedTrianglePairs"] += 1
            first, first_normal = geometry[first_id]
            second, second_normal = geometry[second_id]
            squared, p, q = _triangle_pair(first, second, first_normal, second_normal, counts)
            if best is None or squared < best[0]:
                best = squared, first_id, second_id, p, q
    squared, first_id, second_id, p, q = best
    return {"profile": PROFILE, "accepted": False, "inputSha256": input_hash,
            "distanceMeters": _sqrt_fraction(squared), "exactSquaredDistanceMeters2": _rational(squared),
            "firstFaceIndex": first_id, "secondFaceIndex": second_id,
            "closestPointFirstMeters": [float(value) for value in p],
            "closestPointSecondMeters": [float(value) for value in q],
            "exactClosestPointFirstMeters": [_rational(value) for value in p],
            "exactClosestPointSecondMeters": [_rational(value) for value in q],
            "counts": counts,
            "method": "Complete vertex-face, edge-edge and segment-face features; exact rational binary-input predicates, feature parameters and AABB lower bounds; binary64 distance and rounded closest-point outputs",
            "scope": "Saved-state endpoint diagnostic for the supplied triangle subsets only. No continuous motion, contact admission, stitch coverage, material-side, construction completion or garment acceptance is certified."}
