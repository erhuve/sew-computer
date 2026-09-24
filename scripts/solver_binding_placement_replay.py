"""Independent standard-library audit of supplied source-bound rigid placement.

Exact input arithmetic and recorded rounding are checked here. Pattern and
split-geometry validity remain the separate captured source validator's scope;
this descriptor does not execute or accept a construction operation.
"""

from fractions import Fraction
import hashlib
import json
import math
import re
import struct

PROFILE = "independent-binding-placement-verification-v1"
REQUIRED_HELPER_FILES = ()
INSTANCE = "opening_binding_left_left:shell"
TEMPLATE = "opening_binding_left_left"
MAX_BYTES = 24 * 1024 ** 2
CORE_FIELDS = {"profile", "accepted", "solverReady", "units", "sourceArcUnits", "baseUnit", "bindingRefinement",
    "bindingSeamRemap", "instances", "numericalMeshes", "instanceOffsets", "instanceTriangleOffsets", "restMeters",
    "triangles", "embeddedConstraints", "sourceRowCorrespondence", "derivationCodeDigests", "limitations"}
ROTATION_BOUND = Fraction(64, 2**52)
EDGE_BOUND = Fraction(4096, 2**52)


def _identity(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value) is not None

def _encoded(value):
    pending, remaining, text_bytes = [(value, 0)], 1000000, 0
    while pending:
        item, depth = pending.pop()
        remaining -= 1
        if remaining < 0 or depth > 40:
            raise ValueError("Independent placement structure budget exceeded")
        if type(item) is dict:
            if any(type(key) is not str for key in item) or len(item) * 2 > remaining:
                raise ValueError("Bounded string-keyed raw JSON required")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif type(item) is list:
            if len(item) > remaining:
                raise ValueError("Independent placement array budget exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > MAX_BYTES:
                raise ValueError("Independent placement string budget exceeded")
            try:
                text_bytes += len(item.encode())
            except UnicodeEncodeError as error:
                raise ValueError("UTF-8 placement strings required") from error
            if text_bytes > MAX_BYTES:
                raise ValueError("Independent placement string budget exceeded")
        elif type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Independent placement integer budget exceeded")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Finite raw JSON numbers required")
        elif item is not None and type(item) is not bool:
            raise ValueError("Raw JSON placement values required")
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MAX_BYTES:
        raise ValueError("Independent placement input byte budget exceeded")
    return data

def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()

def _same(actual, expected, label):
    if _encoded(actual) != _encoded(expected):
        raise ValueError("Independent placement mismatch: " + label)

def _number(value, *, maximum=100.):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > maximum:
        raise ValueError("Finite bounded non-Boolean binary64 number required")
    converted = float(value)
    if converted != value:
        raise ValueError("Input number must be exactly representable as binary64")
    return Fraction(converted)

def _index(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError("Bounded non-Boolean triangle/vertex index required")
    return value

def _rat(value):
    if value.numerator.bit_length() > 16384 or value.denominator.bit_length() > 16384:
        raise ValueError("Independent placement rational budget exceeded")
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "roundedBinary64": float(value)}

def _points(value, dimensions, maximum):
    if type(value) is not list or not 3 <= len(value) <= maximum:
        raise ValueError("Bounded complete vertex array required")
    for point in value:
        if type(point) is not list or len(point) != dimensions:
            raise ValueError("Explicit material-coordinate dimensions required")
        for coordinate in point:
            _number(coordinate)
    return value

def _local_faces(value, count, maximum=50000):
    if type(value) is not list or not 1 <= len(value) <= maximum:
        raise ValueError("Bounded complete face array required")
    for face in value:
        if type(face) is not list or len(face) != 3 or len(set(face)) != 3:
            raise ValueError("Three distinct oriented material vertices required")
        for vertex in face:
            _index(vertex, count)
    if len({tuple(sorted(face)) for face in value}) != len(value):
        raise ValueError("Duplicate material triangle")
    return value

def _mesh(source):
    points = _points(source["restMeters"], 3, 25000)
    flat = source["triangles"]
    if type(flat) is not list or not 3 <= len(flat) <= 150000 or len(flat) % 3:
        raise ValueError("Bounded flat canonical triangle array required")
    faces = _local_faces([flat[index:index + 3] for index in range(0, len(flat), 3)], len(points))
    offsets = source["instanceOffsets"]
    if (type(offsets) is not dict or not 1 <= len(offsets) <= 64
            or any(not _identity(name) for name in offsets)):
        raise ValueError("Explicit physical instance offsets required")
    for value in offsets.values():
        _index(value, len(points))
    ordered = sorted(offsets, key=offsets.get)
    if offsets[ordered[0]] != 0 or len(set(offsets.values())) != len(offsets):
        raise ValueError("Instance offsets must partition material vertices")
    ranges, owner = {}, []
    for index, name in enumerate(ordered):
        end = offsets[ordered[index + 1]] if index + 1 < len(ordered) else len(points)
        if end - offsets[name] < 3:
            raise ValueError("At least three vertices per material instance required")
        ranges[name] = (offsets[name], end)
        owner.extend([name] * (end - offsets[name]))
    local, lookup = {name: [] for name in ordered}, []
    for face in faces:
        identity = owner[face[0]]
        if any(owner[v] != identity for v in face):
            raise ValueError("Material triangle crosses physical instances")
        lookup.append((identity, len(local[identity])))
        local[identity].append([vertex - offsets[identity] for vertex in face])
    return {"points": points, "faces": faces, "ranges": ranges, "local": local, "lookup": lookup,
            "meshSha256": _sha({key: source[key] for key in ("restMeters", "triangles", "instanceOffsets")})}

def _reconstruction(points, original, basis):
    vertices, maximum = [], Fraction()
    for index, (point, weights) in enumerate(zip(points, basis)):
        residual = [_number(point[axis]) - sum((weight * _number(vertex[axis]) for weight, vertex in zip(weights, original)), Fraction())
                    for axis in range(2)]
        maximum = max(maximum, *map(abs, residual))
        vertices.append({"vertex": index, "storedMinusReconstructedMeters": [_rat(value) for value in residual],
                         "weightSumMinusOne": _rat(sum(weights, Fraction()) - 1)})
    return {"vertices": vertices, "maximumAbsoluteCoordinateResidualMeters": _rat(maximum)}

def _lineage(base, refinement):
    if refinement["profile"] != "source-left-binding-refinement-descriptor-v1":
        raise ValueError("Exact two-crease refinement profile required")
    for field in ("accepted", "solverReady", "executable"):
        if refinement[field] is not False:
            raise ValueError("Unaccepted refinement descriptor required")
    for key, expected in (("sourceUnitSha256", _sha(base)), ("instanceId", INSTANCE), ("sourceTemplateId", TEMPLATE),
                          ("originalTemplateSha256", _sha(base["sourceTemplates"][TEMPLATE])),
                          ("originalEmbeddedConstraintsSha256", _sha(base["embeddedConstraints"]))):
        _same(refinement[key], expected, "lineage " + key)
    original = refinement["originalLocalMesh"]
    points = _points(original["verticesMeters"], 2, 128)
    faces = _local_faces(original["triangles"], len(points), 128)
    if len(points) != 11 or len(faces) != 12:
        raise ValueError("Original eleven-vertex/twelve-triangle binding required")
    start = _index(base["instanceOffsets"][INSTANCE], len(base["restMeters"]))
    _same([point + [0.] for point in points], base["restMeters"][start:start + 11], "original material coordinates")
    _same(faces, base["sourceTemplates"][TEMPLATE]["triangles"], "original oriented local faces")
    basis = [[Fraction(int(i == j)) for j in range(11)] for i in range(11)]
    parents = list(range(12))
    stages = refinement["stages"]
    if type(stages) is not list or len(stages) != 2:
        raise ValueError("Both original raw split stages required")
    for stage, path in zip(stages, ("right", "left")):
        _same(stage["sourcePathName"], path, "ordered source-line stages")
        _same(stage["inputMeshSha256"], _sha({"verticesMeters": points, "triangles": faces}), "stage input mesh")
        output = stage["output"]
        next_points = _points(output["vertices"], 2, 128)
        next_faces = _local_faces(output["triangles"], len(next_points), 128)
        _same(next_points[:len(points)], points, "retained stage vertex prefix")
        raw = output["sourceWeights"]
        if type(raw) is not list or len(raw) != len(next_points):
            raise ValueError("Complete raw stage interpolation required")
        stage_basis, composed = [], []
        for vertex, support in enumerate(raw):
            if type(support) is not dict or not 1 <= len(support) <= 3:
                raise ValueError("Bounded stage vertex support required")
            weights = [Fraction()] * len(points)
            for key, value in support.items():
                if type(key) is not str or not key.isascii() or not key.isdigit() or len(key) > 3 or str(int(key)) != key:
                    raise ValueError("Canonical raw stage vertex index required")
                index = _index(int(key), len(points))
                weight = _number(value, maximum=1)
                if weight < 0:
                    raise ValueError("Nonnegative raw lineage coefficients required")
                weights[index] = weight
            if vertex < len(points):
                _same(support, {str(vertex): 1.}, "old stage vertex operator")
            stage_basis.append(weights)
            composed.append([sum((weight * basis[index][old] for index, weight in enumerate(weights)), Fraction()) for old in range(11)])
        _same(stage["binaryInputReconstruction"], _reconstruction(next_points, points, stage_basis), "stage reconstruction residuals")
        raw_parents = output["parentTriangles"]
        if type(raw_parents) is not list or len(raw_parents) != len(next_faces):
            raise ValueError("Complete raw child parent identities required")
        for face, parent in zip(next_faces, raw_parents):
            allowed = faces[_index(parent, len(faces))]
            if any(weight and previous not in allowed for vertex in face
                   for previous, weight in enumerate(stage_basis[vertex])):
                raise ValueError("Raw stage child support escapes its immediate parent")
        parents = [parents[_index(parent, len(faces))] for parent in raw_parents]
        points, faces, basis = next_points, next_faces, composed
    _same(refinement["finalLocalMesh"], {"verticesMeters": points, "triangles": faces}, "final local mesh")
    _same(refinement["ultimateOriginalTriangleIndices"], parents, "ultimate original parent triangles")
    expected_basis = [{"vertex": vertex, "weights": [{"sourceVertex": old, "weight": _rat(weight)}
                       for old, weight in enumerate(row) if weight]} for vertex, row in enumerate(basis)]
    _same(refinement["composedVertexWeights"], expected_basis, "independently composed source operators")
    _same(refinement["binaryInputReconstruction"], _reconstruction(points, original["verticesMeters"], basis), "full reconstruction residuals")
    for face, parent in zip(faces, parents):
        allowed = original["triangles"][parent]
        if any(weight and old not in allowed for vertex in face for old, weight in enumerate(basis[vertex])):
            raise ValueError("Child material coefficients escape the declared original parent")
    if len(points) != 29 or len(faces) != 44:
        raise ValueError("Reviewed twenty-nine-vertex/forty-four-triangle refinement required")
    return basis, parents

def _canonical_meshes(base, refined, refinement, original_mesh, refined_mesh):
    instances = base["instances"]
    if type(instances) is not list or len(instances) != 5 or len({item["id"] for item in instances}) != 5:
        raise ValueError("All five immutable physical instances required")
    _same(refined["instances"], instances, "immutable material instances")
    meshes, offsets, face_offsets, rest, faces = {}, {}, {}, [], []
    for instance in instances:
        name, template = instance["id"], base["sourceTemplates"][instance["templateId"]]
        start, end = original_mesh["ranges"][name]
        _same(original_mesh["local"][name], template["triangles"], "original local triangle identities")
        if end - start != len(template["restPositions"]):
            raise ValueError("Original template vertex count differs")
        if name == INSTANCE:
            mesh = {"verticesMeters": [point + [0.] for point in refinement["finalLocalMesh"]["verticesMeters"]],
                    "triangles": refinement["finalLocalMesh"]["triangles"]}
        else:
            mesh = {"verticesMeters": original_mesh["points"][start:end], "triangles": original_mesh["local"][name]}
        offsets[name], face_offsets[name] = len(rest), len(faces)
        meshes[name] = mesh
        rest.extend(mesh["verticesMeters"])
        faces.extend([[vertex + offsets[name] for vertex in face] for face in mesh["triangles"]])
    for key, value in (("numericalMeshes", meshes), ("instanceOffsets", offsets), ("instanceTriangleOffsets", face_offsets),
                       ("restMeters", rest), ("triangles", [vertex for face in faces for vertex in face])):
        _same(refined[key], value, "refined canonical " + key)
    return face_offsets

def _det(matrix):
    a, b, c = matrix
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def _dot(first, second):
    return sum((a * b for a, b in zip(first, second)), Fraction())


def _matrix_vector(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _gram_error(matrix):
    return [[sum((matrix[k][i] * matrix[k][j] for k in range(3)), Fraction()) - int(i == j)
             for j in range(3)] for i in range(3)]


def _round_nearest(value):
    """Check the entire exact nearest-even rounding cell, including cancellation."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Finite binary64 placement coordinate required")
    center = Fraction(result)
    if center != value:
        previous, following = math.nextafter(result, -math.inf), math.nextafter(result, math.inf)
        if not math.isfinite(previous) or not math.isfinite(following):
            raise ValueError("Bounded placement conversion required")
        lower, upper = (Fraction(previous) + center) / 2, (Fraction(following) + center) / 2
        even = struct.unpack(">Q", struct.pack(">d", result))[0] % 2 == 0
        if not lower <= value <= upper or value in (lower, upper) and not even:
            raise ValueError("Incorrect once-rounded nearest-even placement conversion")
    return result


def _placed(points, matrix, translation):
    exact = [[coordinate + shift for coordinate, shift in zip(_matrix_vector(matrix, point), translation)]
             for point in points]
    rounded = [[_round_nearest(coordinate) for coordinate in point] for point in exact]
    errors = [[Fraction(actual) - expected for actual, expected in zip(point, ideal)]
              for point, ideal in zip(rounded, exact)]
    return exact, rounded, errors


def _edge_terms(first, second, matrix, gram, first_error, second_error):
    edge = [b - a for a, b in zip(first, second)]
    transformed = _matrix_vector(matrix, edge)
    error = [b - a for a, b in zip(first_error, second_error)]
    gram_term = _dot(edge, _matrix_vector(gram, edge))
    cross_term = 2 * _dot(transformed, error)
    squared_error = _dot(error, error)
    return gram_term, cross_term, squared_error


def _cross(first, second):
    return [first[1]*second[2]-first[2]*second[1], first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0]]


def _sub(first, second):
    return [a-b for a, b in zip(first, second)]


def _vector(values):
    return [_rat(value) for value in values]


def _pose(raw):
    if type(raw) is not dict or set(raw) != {"instanceId", "rotation", "translationMeters"}:
        raise ValueError("Complete explicit instance pose required")
    matrix, shift = raw["rotation"], raw["translationMeters"]
    if (type(matrix) is not list or len(matrix) != 3
            or any(type(row) is not list or len(row) != 3 for row in matrix)
            or type(shift) is not list or len(shift) != 3):
        raise ValueError("Explicit three-dimensional pose arrays required")
    matrix = [[_number(value, maximum=2) for value in row] for row in matrix]
    shift = [_number(value, maximum=10) for value in shift]
    gram, determinant = _gram_error(matrix), _det(matrix)
    if max(abs(value) for row in gram for value in row) > ROTATION_BOUND or abs(determinant-1) > ROTATION_BOUND:
        raise ValueError("Pose exceeds fixed proper rotation representation bound")
    return matrix, shift, gram, determinant


def _request(raw, identities):
    if (type(raw) is not dict or set(raw) != {"profile", "accepted", "coordinatePolicy", "minimumSeparationMeters", "poses"}
            or raw["profile"] != "source-binding-rigid-placement-request-v1" or raw["accepted"] is not False
            or raw["coordinatePolicy"] != "exact-affine-once-rounded-binary64-v1"
            or type(raw["poses"]) is not list or len(raw["poses"]) != 5):
        raise ValueError("Exact five-instance placement request required")
    minimum = _number(raw["minimumSeparationMeters"], maximum=10)
    if minimum < Fraction(1, 10**8):
        raise ValueError("Requested static separation must be at least 1e-8 meters")
    poses = []
    for identity, pose in zip(identities, raw["poses"]):
        if type(pose) is not dict or pose.get("instanceId") != identity:
            raise ValueError("Exactly one pose per immutable ordered source instance required")
        poses.append(_pose(pose))
    return poses, minimum


def _instance_audit(name, mesh, start, pose):
    matrix, translation, gram, determinant = pose
    points = [list(map(Fraction, point)) for point in mesh["verticesMeters"]]
    ideal, numerical, errors = _placed(points, matrix, translation)
    if any(abs(coordinate) > 100 for point in numerical for coordinate in point):
        raise ValueError("Placed coordinate exceeds fixed numerical bound")
    placed = [list(map(Fraction, point)) for point in numerical]
    edges = sorted({tuple(sorted((face[i], face[(i+1) % 3]))) for face in mesh["triangles"] for i in range(3)})
    edge_audits = []
    for a, b in edges:
        original_edge, placed_edge = _sub(points[b], points[a]), _sub(placed[b], placed[a])
        before, after = _dot(original_edge, original_edge), _dot(placed_edge, placed_edge)
        if before <= 0 or abs(after-before) > EDGE_BOUND*before:
            raise ValueError("Placed edge violates the fixed relative squared-length bound")
        rotation, cross, squared = _edge_terms(points[a], points[b], matrix, gram, errors[a], errors[b])
        if after-before != rotation+cross+squared:
            raise ValueError("Inconsistent exact edge metric decomposition")
        edge_audits.append({"verticesLocal": [a, b], "restSquaredLengthMetersSquared": _rat(before),
            "placedSquaredLengthMetersSquared": _rat(after), "rotationMetricDefectMetersSquared": _rat(rotation),
            "roundingCrossTermMetersSquared": _rat(cross), "roundingSquaredTermMetersSquared": _rat(squared),
            "squaredLengthChangeMetersSquared": _rat(after-before), "relativeSquaredLengthChange": _rat((after-before)/before)})
    face_audits = []
    for index, (a, b, c) in enumerate(mesh["triangles"]):
        ideal_normal = _cross(_sub(ideal[b], ideal[a]), _sub(ideal[c], ideal[a]))
        placed_normal = _cross(_sub(placed[b], placed[a]), _sub(placed[c], placed[a]))
        orientation = _dot(ideal_normal, placed_normal)
        if orientation <= 0:
            raise ValueError("Rounded placement collapses or reverses an oriented triangle")
        face_audits.append({"triangleIndexLocal": index, "orientedAreaDotMetersFourth": _rat(orientation)})
    return {"instanceId": name, "canonicalVertexRange": [start, start+len(points)],
        "rotationGramMinusIdentity": [_vector(row) for row in gram], "rotationDeterminant": _rat(determinant),
        "coordinateRoundingResidualMeters": [_vector(row) for row in errors],
        "edges": edge_audits, "triangles": face_audits}, numerical, (ideal, errors)


def _lineage_audit(source, basis, pose, placed, ideal, errors):
    matrix, shift, _, _ = pose
    refinement = source["bindingRefinement"]
    old_points = [[Fraction(value) for value in point] + [Fraction()]
                  for point in refinement["originalLocalMesh"]["verticesMeters"]]
    old_ideal, old_placed, old_errors = _placed(old_points, matrix, shift)
    points = [list(map(Fraction, point)) for point in source["numericalMeshes"][INSTANCE]["verticesMeters"]]
    if _encoded(placed[:len(old_points)]) != _encoded(old_placed):
        raise ValueError("Original binding prefix differs under the same exact pose evaluation")
    result = []
    for vertex, weights in enumerate(basis):
        total = sum(weights, Fraction())
        stored_residual = [points[vertex][axis] - sum((weight*old_points[i][axis] for i, weight in enumerate(weights)), Fraction())
                           for axis in range(3)]
        transformed_residual = _matrix_vector(matrix, stored_residual)
        expected_commutator = [transformed_residual[axis] + (1-total)*shift[axis] for axis in range(3)]
        # Derive each commutator directly first, then check the independent
        # decomposition. A total-only comparison could hide cancelling errors.
        ideal_commutator = [ideal[vertex][axis] - sum((weight*old_ideal[i][axis] for i, weight in enumerate(weights)), Fraction())
                            for axis in range(3)]
        rounding_commutator = [errors[vertex][axis] - sum((weight*old_errors[i][axis] for i, weight in enumerate(weights)), Fraction())
                               for axis in range(3)]
        observed = [Fraction(placed[vertex][axis]) - sum((weight*Fraction(old_placed[i][axis]) for i, weight in enumerate(weights)), Fraction())
                    for axis in range(3)]
        if ideal_commutator != expected_commutator or observed != [a+b for a, b in zip(ideal_commutator, rounding_commutator)]:
            raise ValueError("Exact raw-lineage placement commutation identity failed")
        result.append({"vertexLocal": vertex, "sourceCoefficientSum": _rat(total),
            "sourceCoordinateResidualMeters": _vector(stored_residual),
            "idealAffineCommutatorMeters": _vector(ideal_commutator),
            "placementRoundingCommutatorMeters": _vector(rounding_commutator),
            "placedMinusInterpolatedOriginalMeters": _vector(observed)})
    return result


def _separating_planes(identities, ranges, placed, minimum):
    extrema = {}
    for identity in identities:
        start, end = ranges[identity]
        extrema[identity] = [(min(Fraction(point[axis]) for point in placed[start:end]),
                              max(Fraction(point[axis]) for point in placed[start:end])) for axis in range(3)]
    result = []
    for i, first in enumerate(identities):
        for second in identities[i+1:]:
            best = None
            for axis in range(3):
                a_min, a_max = extrema[first][axis]
                b_min, b_max = extrema[second][axis]
                for sign, upper, lower in ((1, a_max, b_min), (-1, -a_min, -b_max)):
                    candidate = (lower-upper, axis, sign, upper, lower)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            gap, axis, sign, upper, lower = best
            if gap < minimum:
                raise ValueError("Complete pairwise static separating-plane clearance required")
            result.append({"firstInstanceId": first, "secondInstanceId": second, "axisIndex": axis,
                "directionSign": sign, "firstMaximumProjectionMeters": _rat(upper),
                "secondMinimumProjectionMeters": _rat(lower), "separatingPlaneGapMeters": _rat(gap)})
    if len(result) != 10:
        raise ValueError("All ten distinct physical instance pairs required")
    return result


def verify_binding_placement(source, descriptor):
    """Independently audit this static descriptor without executing placement."""
    before = [_encoded(value) for value in (source, descriptor)]
    try:
        if (type(source) is not dict or set(source) != CORE_FIELDS
                or source["profile"] != "source-left-binding-refined-unit-v1"
                or source["accepted"] is not False or source["solverReady"] is not False
                or source["units"] != "m" or source["sourceArcUnits"] != "mm"):
            raise ValueError("Bare explicitly unaccepted refined source core required")
        base = source["baseUnit"]
        if (base["profile"] != "source-cuff-construction-unit-v1" or base["side"] != "left"
                or base["accepted"] is not False or base["solverReady"] is not False
                or base["units"] != "m" or base["sourceArcUnits"] != "mm"
                or {"placedMeters", "bindingFirstTurnDiagnostic", "sewingActuation", "gripperActuation", "foldActuation",
                    "assemblySchedule", "sewingFrames", "bindingRefinement", "baseUnit", "bindingSeamRemap"}.intersection(base)
                or any(instance["mirrorX"] is not False for instance in base["instances"])):
            raise ValueError("Immutable left five-fabric original source required")
        original_mesh, mesh = _mesh(base), _mesh(source)
        basis, _ = _lineage(base, source["bindingRefinement"])
        _canonical_meshes(base, source, source["bindingRefinement"], original_mesh, mesh)
        identities = [instance["id"] for instance in source["instances"]]
        fields = {"profile", "accepted", "solverReady", "executable", "sourceSha256", "baseUnitSha256",
            "requestSha256", "request", "coordinatePolicy", "rotationRepresentationBound", "relativeSquaredEdgeRepresentationBound",
            "placedMeters", "instances", "bindingLineage", "originalBindingPrefixPreservedUnderDeclaredEvaluation",
            "separatingPlanes", "limitations"}
        if (type(descriptor) is not dict or set(descriptor) != fields
                or descriptor["profile"] != "source-binding-rigid-placement-v1"
                or any(descriptor[flag] is not False for flag in ("accepted", "solverReady", "executable"))
                or descriptor["coordinatePolicy"] != "exact-affine-once-rounded-binary64-v1"):
            raise ValueError("Complete unaccepted placement descriptor required")
        request = descriptor["request"]
        for field, value in (("sourceSha256", source), ("baseUnitSha256", base), ("requestSha256", request)):
            _same(descriptor[field], _sha(value), field)
        _same(descriptor["rotationRepresentationBound"], _rat(ROTATION_BOUND), "fixed rotation bound")
        _same(descriptor["relativeSquaredEdgeRepresentationBound"], _rat(EDGE_BOUND), "fixed edge bound")
        poses, minimum = _request(request, identities)
        placed, audits, auxiliary = [], [], {}
        for identity, pose in zip(identities, poses):
            start = source["instanceOffsets"][identity]
            if start != len(placed):
                raise ValueError("Canonical instance ranges do not partition placed vertices")
            audit, points, intermediate = _instance_audit(identity, source["numericalMeshes"][identity], start, pose)
            placed.extend(points)
            audits.append(audit)
            auxiliary[identity] = (points, intermediate)
        _same(descriptor["placedMeters"], placed, "every exactly-once rounded coordinate")
        _same(descriptor["instances"], audits, "complete instance, edge and triangle witnesses")
        binding_points, (binding_ideal, binding_errors) = auxiliary[INSTANCE]
        lineage = _lineage_audit(source, basis, poses[identities.index(INSTANCE)], binding_points, binding_ideal, binding_errors)
        _same(descriptor["bindingLineage"], lineage, "complete exact raw-lineage commutators")
        if descriptor["originalBindingPrefixPreservedUnderDeclaredEvaluation"] is not True:
            raise ValueError("Explicit unchanged prefix evaluation witness required")
        separation = _separating_planes(identities, mesh["ranges"], placed, minimum)
        _same(descriptor["separatingPlanes"], separation, "all ten exact static separating planes")
        if (type(descriptor["limitations"]) is not list or not 1 <= len(descriptor["limitations"]) <= 16
                or any(type(text) is not str or not text or len(text) > 4096 for text in descriptor["limitations"])):
            raise ValueError("Bounded explicit limitation statements required")
        if before != [_encoded(value) for value in (source, descriptor)]:
            raise ValueError("Placement verification inputs changed")
        return {"profile": PROFILE, "verified": True, "accepted": False,
            "sourceSha256": _sha(source), "descriptorSha256": _sha(descriptor), "baseUnitSha256": _sha(base),
            "requestSha256": _sha(request), "placedMetersSha256": _sha(placed),
            "instanceCount": 5, "vertexCount": len(placed), "triangleCount": len(mesh["faces"]),
            "edgeCount": sum(len(audit["edges"]) for audit in audits), "bindingLineageVertexCount": len(lineage),
            "separatingPlaneCount": len(separation), "minimumCertifiedGapMeters": _rat(min(
                Fraction(int(item["separatingPlaneGapMeters"]["numerator"]), int(item["separatingPlaneGapMeters"]["denominator"]))
                for item in separation)),
            "sourceScope": "Independent supplied canonical mesh and raw two-stage lineage, complete exact pose/metric/rounding/commutation witnesses and all ten static cross-instance coordinate-plane gaps. Pattern and split geometry require the separately captured source validator. Textual limitations are not evidence. No within-instance contact, staging path, dynamics, material calibration, textile-side, sewing frame or construction acceptance."}
    except (KeyError, TypeError, IndexError, StopIteration, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Malformed or incomplete independent placement audit input") from error
