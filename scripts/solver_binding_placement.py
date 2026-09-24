"""Source-bound, explicitly rounded rigid staging; never run or install controls.

Binary64 rotations are representations of proper rigid poses, not exact
orthogonal rational matrices. Their metric defects and coordinate rounding
are retained separately. Static inter-instance separation says nothing about
the path to this placement, within-instance contact or garment construction.
"""

import copy
from fractions import Fraction
import hashlib
import math

from solver_binding_source import CONTROL_FIELDS, INSTANCE, _bounded, _encoded, validate_binding_source
from solver_binding_remap import _basis, _rational


PROFILE = "source-binding-rigid-placement-v1"
REQUEST_PROFILE = "source-binding-rigid-placement-request-v1"
POLICY = "exact-affine-once-rounded-binary64-v1"
ROTATION_BOUND = Fraction(64, 2**52)
EDGE_BOUND = Fraction(4096, 2**52)
LIMITATIONS = [
    "Explicit static staging only; no controls are installed and no solver trajectory, contact path or construction phase is executed.",
    "Raw binary64 rotation matrices are checked against fixed representation bounds, never normalized or orthogonalized. Their nonzero metric defects remain reported.",
    "Every coordinate is the once-rounded exact affine expression from unchanged numerical rest coordinates. This can differ from earlier BLAS-evaluated placements and is a new declared numerical input.",
    "Exact separating coordinate planes cover all ten cross-instance pairs at the requested distance. They are conservative and do not certify within-instance contact or a collision-free staging path.",
    "The requested geometric gap is not a solver contact profile. Captured motion must independently check the complete placement against its configured thickness and contact activation distances.",
    "Original pattern and refinement validity require the separate strict source validator. Lineage commutation includes raw coefficient sums, source-coordinate residuals and placement rounding without repair.",
    "Source winding supplies geometric orientation only. Textile sides, layer semantics, material calibration, sewing frames and motion controls require separate declarations and verification.",
]


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _number(value, maximum):
    if (type(value) not in (int, float) or not math.isfinite(value) or abs(value) > maximum
            or float(value) != value):
        raise ValueError("Finite bounded raw binary64 placement number required")
    return Fraction(value)


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Fraction())


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _vector(values):
    return [_rational(value) for value in values]


def _pose(raw):
    if type(raw) is not dict or set(raw) != {"instanceId", "rotation", "translationMeters"}:
        raise ValueError("Exact explicit instance pose fields required")
    r, t = raw["rotation"], raw["translationMeters"]
    if (type(r) is not list or len(r) != 3 or any(type(row) is not list or len(row) != 3 for row in r)
            or type(t) is not list or len(t) != 3):
        raise ValueError("Three-dimensional rigid pose required")
    rotation = [[_number(value, 2) for value in row] for row in r]
    translation = [_number(value, 10) for value in t]
    gram = [[sum((rotation[k][i] * rotation[k][j] for k in range(3)), Fraction()) - int(i == j)
             for j in range(3)] for i in range(3)]
    determinant = _dot(rotation[0], _cross(rotation[1], rotation[2]))
    if max(abs(value) for row in gram for value in row) > ROTATION_BOUND or abs(determinant - 1) > ROTATION_BOUND:
        raise ValueError("Proper rigid rotation representation required; scaling, shear and reflection reject")
    return rotation, translation, gram, determinant


def _affine(points, rotation, translation):
    exact = [[_dot(row, point) + shift for row, shift in zip(rotation, translation)] for point in points]
    numerical = [[float(value) for value in point] for point in exact]
    if any(not math.isfinite(value) or abs(value) > 100 for point in numerical for value in point):
        raise ValueError("Bounded placed coordinates required")
    residuals = [[Fraction(value) - ideal for value, ideal in zip(point, reference)]
                 for point, reference in zip(numerical, exact)]
    return exact, numerical, residuals


def build_binding_placement(source, request):
    """Derive a complete standalone placement descriptor from an explicit request."""
    source_bytes, request_bytes = _bounded(source), _bounded(request)
    validate_binding_source(source)
    if CONTROL_FIELDS.intersection(source):
        raise ValueError("Bare refined core required before placement derivation")
    identities = [instance["id"] for instance in source["instances"]]
    if (type(request) is not dict or set(request) != {"profile", "accepted", "coordinatePolicy", "minimumSeparationMeters", "poses"}
            or request["profile"] != REQUEST_PROFILE or request["accepted"] is not False
            or request["coordinatePolicy"] != POLICY or type(request["poses"]) is not list
            or len(request["poses"]) != len(identities)):
        raise ValueError("Exact bounded rigid-placement request required")
    minimum = _number(request["minimumSeparationMeters"], 10)
    if minimum < Fraction(1, 10**8):
        raise ValueError("Explicit positive placement separation of at least 1e-8 m required")
    placed, instances, ranges = [], [], {}
    for identity, raw in zip(identities, request["poses"]):
        if type(raw) is not dict or raw.get("instanceId") != identity:
            raise ValueError("Exactly one explicit pose per instance in immutable source order required")
        rotation, translation, gram, determinant = _pose(raw)
        mesh = source["numericalMeshes"][identity]
        points = [[Fraction(value) for value in point] for point in mesh["verticesMeters"]]
        exact, numerical, residuals = _affine(points, rotation, translation)
        start = source["instanceOffsets"][identity]
        if start != len(placed):
            raise ValueError("Complete canonical instance partition required")
        ranges[identity] = (start, start + len(points))
        placed.extend(numerical)
        positions = [list(map(Fraction, point)) for point in numerical]
        edges = sorted({tuple(sorted((face[i], face[(i + 1) % 3]))) for face in mesh["triangles"] for i in range(3)})
        edge_audits = []
        for a, b in edges:
            rest_edge, ideal_edge = _sub(points[b], points[a]), _sub(exact[b], exact[a])
            actual_edge, delta = _sub(positions[b], positions[a]), _sub(residuals[b], residuals[a])
            rest_squared, final_squared = _dot(rest_edge, rest_edge), _dot(actual_edge, actual_edge)
            transform = _dot(rest_edge, [_dot(row, rest_edge) for row in gram])
            rounding_cross, rounding_squared = 2 * _dot(ideal_edge, delta), _dot(delta, delta)
            difference = final_squared - rest_squared
            if rest_squared <= 0 or abs(difference) > EDGE_BOUND * rest_squared:
                raise ValueError("Placed edge exceeds fixed numerical representation bound")
            if difference != transform + rounding_cross + rounding_squared:
                raise ValueError("Exact squared metric decomposition failed")
            edge_audits.append({"verticesLocal": [a, b], "restSquaredLengthMetersSquared": _rational(rest_squared),
                "placedSquaredLengthMetersSquared": _rational(final_squared),
                "rotationMetricDefectMetersSquared": _rational(transform),
                "roundingCrossTermMetersSquared": _rational(rounding_cross),
                "roundingSquaredTermMetersSquared": _rational(rounding_squared),
                "squaredLengthChangeMetersSquared": _rational(difference),
                "relativeSquaredLengthChange": _rational(difference / rest_squared)})
        # A near-rigid matrix cannot justify accepting a rounded collapsed or
        # reversed triangle. Retain an exact orientation witness for each face.
        face_audits = []
        for index, (a, b, c) in enumerate(mesh["triangles"]):
            ideal_normal = _cross(_sub(exact[b], exact[a]), _sub(exact[c], exact[a]))
            final_normal = _cross(_sub(positions[b], positions[a]), _sub(positions[c], positions[a]))
            orientation = _dot(ideal_normal, final_normal)
            if orientation <= 0:
                raise ValueError("Rounded placement collapses or reverses a source triangle")
            face_audits.append({"triangleIndexLocal": index, "orientedAreaDotMetersFourth": _rational(orientation)})
        instances.append({"instanceId": identity, "canonicalVertexRange": list(ranges[identity]),
            "rotationGramMinusIdentity": [_vector(row) for row in gram],
            "rotationDeterminant": _rational(determinant),
            "coordinateRoundingResidualMeters": [_vector(row) for row in residuals],
            "edges": edge_audits, "triangles": face_audits})
    binding_index = identities.index(INSTANCE)
    rotation, translation, _, _ = _pose(request["poses"][binding_index])
    refinement = source["bindingRefinement"]
    basis = _basis(refinement)
    old_rest = [[Fraction(value) for value in point] + [Fraction()] for point in refinement["originalLocalMesh"]["verticesMeters"]]
    old_exact, old_numerical, old_rounding = _affine(old_rest, rotation, translation)
    start, end = ranges[INSTANCE]
    final_rest = [list(map(Fraction, point)) for point in source["restMeters"][start:end]]
    final_placed = [list(map(Fraction, point)) for point in placed[start:end]]
    exact, _, rounding = _affine(final_rest, rotation, translation)
    lineage = []
    for index, weights in enumerate(basis):
        weight_sum = sum(weights, Fraction())
        rest_residual = [final_rest[index][axis] - sum((w * p[axis] for w, p in zip(weights, old_rest)), Fraction()) for axis in range(3)]
        commutator = [_dot(row, rest_residual) + (1 - weight_sum) * shift for row, shift in zip(rotation, translation)]
        rounding_residual = [rounding[index][axis] - sum((w * p[axis] for w, p in zip(weights, old_rounding)), Fraction()) for axis in range(3)]
        observed = [final_placed[index][axis] - sum((w * Fraction(p[axis]) for w, p in zip(weights, old_numerical)), Fraction()) for axis in range(3)]
        if observed != [x + y for x, y in zip(commutator, rounding_residual)]:
            raise ValueError("Exact source-lineage placement decomposition failed")
        # Independent direct expression also guards affine-translation sign.
        if commutator != [exact[index][axis] - sum((w * p[axis] for w, p in zip(weights, old_exact)), Fraction()) for axis in range(3)]:
            raise ValueError("Exact affine commutation identity failed")
        lineage.append({"vertexLocal": index, "sourceCoefficientSum": _rational(weight_sum),
            "sourceCoordinateResidualMeters": _vector(rest_residual),
            "idealAffineCommutatorMeters": _vector(commutator),
            "placementRoundingCommutatorMeters": _vector(rounding_residual),
            "placedMinusInterpolatedOriginalMeters": _vector(observed)})
    prefix_preserved = _encoded(placed[start:start + len(old_numerical)]) == _encoded(old_numerical)
    if not prefix_preserved:
        raise ValueError("Original binding vertices must retain this same declared pose evaluation")
    separation = []
    for first_index, first in enumerate(identities):
        for second in identities[first_index + 1:]:
            a0, a1 = ranges[first]
            b0, b1 = ranges[second]
            candidates = []
            for axis in range(3):
                for sign in (1, -1):
                    upper = max(sign * Fraction(placed[v][axis]) for v in range(a0, a1))
                    lower = min(sign * Fraction(placed[v][axis]) for v in range(b0, b1))
                    candidates.append((lower - upper, axis, sign, upper, lower))
            gap, axis, sign, upper, lower = max(candidates, key=lambda item: item[0])
            if gap < minimum:
                raise ValueError("Requested all-instance static separating-plane clearance is unavailable")
            separation.append({"firstInstanceId": first, "secondInstanceId": second, "axisIndex": axis,
                "directionSign": sign, "firstMaximumProjectionMeters": _rational(upper),
                "secondMinimumProjectionMeters": _rational(lower), "separatingPlaneGapMeters": _rational(gap)})
    descriptor = {"profile": PROFILE, "accepted": False, "solverReady": False, "executable": False,
        "sourceSha256": hashlib.sha256(source_bytes).hexdigest(), "baseUnitSha256": _sha(source["baseUnit"]),
        "requestSha256": hashlib.sha256(request_bytes).hexdigest(), "request": copy.deepcopy(request),
        "coordinatePolicy": POLICY, "rotationRepresentationBound": _rational(ROTATION_BOUND),
        "relativeSquaredEdgeRepresentationBound": _rational(EDGE_BOUND),
        "placedMeters": placed, "instances": instances, "bindingLineage": lineage,
        "originalBindingPrefixPreservedUnderDeclaredEvaluation": prefix_preserved,
        "separatingPlanes": separation, "limitations": list(LIMITATIONS)}
    if _bounded(source) != source_bytes or _bounded(request) != request_bytes:
        raise ValueError("Placement inputs changed during derivation")
    _bounded(descriptor)
    return descriptor


def validate_binding_placement(source, descriptor):
    """Strictly rederive every input, coordinate and exact witness."""
    before = _bounded(descriptor)
    if type(descriptor) is not dict or "request" not in descriptor:
        raise ValueError("Complete placement descriptor required")
    expected = build_binding_placement(source, descriptor["request"])
    if before != _encoded(expected):
        raise ValueError("Placement differs from complete source rederivation")
    return expected
