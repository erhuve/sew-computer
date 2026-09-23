"""Independent stdlib audit of original-to-refined material gripper migration.

This verifies source/recipe identity, raw lineage arithmetic and numerical
anchor correspondence. Pattern and split-geometry admission remain the
responsibility of the separately captured source validator.
"""

from fractions import Fraction
import hashlib
import json
import math
import re
import struct


PROFILE = "independent-binding-gripper-remap-verification-v1"
REQUIRED_HELPER_FILES = ()
INSTANCE = "opening_binding_left_left:shell"
TEMPLATE = "opening_binding_left_left"
MAX_BYTES = 24 * 1024 ** 2
CORE_FIELDS = {"profile", "accepted", "solverReady", "units", "sourceArcUnits", "baseUnit", "bindingRefinement",
    "bindingSeamRemap", "instances", "numericalMeshes", "instanceOffsets", "instanceTriangleOffsets", "restMeters",
    "triangles", "embeddedConstraints", "sourceRowCorrespondence", "derivationCodeDigests", "limitations"}


def _encoded(value):
    pending, remaining, text_bytes = [(value, 0)], 1000000, 0
    while pending:
        item, depth = pending.pop()
        remaining -= 1
        if remaining < 0 or depth > 40:
            raise ValueError("Independent gripper-remap structure budget exceeded")
        if type(item) is dict:
            if any(type(key) is not str for key in item) or len(item) * 2 > remaining:
                raise ValueError("Bounded string-keyed raw JSON required")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif type(item) is list:
            if len(item) > remaining:
                raise ValueError("Independent gripper-remap array budget exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > MAX_BYTES:
                raise ValueError("Independent gripper-remap string budget exceeded")
            try:
                text_bytes += len(item.encode())
            except UnicodeEncodeError as error:
                raise ValueError("UTF-8 gripper-remap strings required") from error
            if text_bytes > MAX_BYTES:
                raise ValueError("Independent gripper-remap string budget exceeded")
        elif type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Independent gripper-remap integer budget exceeded")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Finite raw JSON numbers required")
        elif item is not None and type(item) is not bool:
            raise ValueError("Raw JSON gripper-remap values required")
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MAX_BYTES:
        raise ValueError("Independent gripper-remap input byte budget exceeded")
    return data


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _same(actual, expected, label):
    if _encoded(actual) != _encoded(expected):
        raise ValueError("Independent gripper-remap mismatch: " + label)


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


def _identity(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value) is not None


def _fraction(value):
    result = _number(value, maximum=1)
    if result < 0 or result.denominator > 2**40 or result.denominator & (result.denominator - 1):
        raise ValueError("Bounded nonnegative dyadic control fraction required")
    return result


def _rat(value):
    if value.numerator.bit_length() > 16384 or value.denominator.bit_length() > 16384:
        raise ValueError("Independent gripper-remap rational budget exceeded")
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


def _recipe(source, mesh, subdivisions):
    recipe = source["gripperActuation"]
    if (type(recipe) is not dict or set(recipe) != {"profile", "accepted", "meshSha256", "anchors", "schedule"}
            or recipe["profile"] != "captured-material-grippers-v1" or recipe["accepted"] is not False
            or recipe["meshSha256"] != mesh["meshSha256"]):
        raise ValueError("Exact original gripper recipe and mesh identity required")
    anchors = recipe["anchors"]
    if type(anchors) is not list or not 1 <= len(anchors) <= 16:
        raise ValueError("One to sixteen original grippers required")
    identities = []
    for anchor in anchors:
        if (type(anchor) is not dict or set(anchor) != {"id", "instanceId", "triangleIndex", "weights", "stiffnessNPerM"}
                or not _identity(anchor["id"]) or anchor["id"] in identities or not _identity(anchor["instanceId"])):
            raise ValueError("Unique ordered original gripper identities required")
        identities.append(anchor["id"])
        triangle = _index(anchor["triangleIndex"], len(mesh["faces"]))
        if mesh["lookup"][triangle][0] != anchor["instanceId"]:
            raise ValueError("Original triangle does not belong to gripper instance")
        weights = anchor["weights"]
        if (type(weights) is not list or len(weights) != 3
                or any(_number(value, maximum=1) < 0 for value in weights)
                or abs(math.fsum(weights) - 1) > 1e-12):
            raise ValueError("Unmodified nonnegative normalized gripper weights required")
        if not 0 < _number(anchor["stiffnessNPerM"], maximum=1e12):
            raise ValueError("Positive bounded original gripper stiffness required")
    schedule = recipe["schedule"]
    if (type(schedule) is not dict or set(schedule) != {"profile", "gripperIds", "knots"}
            or schedule["profile"] != "material-gripper-target-activation-v1"):
        raise ValueError("Exact material-gripper schedule fields required")
    _same(schedule["gripperIds"], identities, "ordered schedule identities")
    knots = schedule["knots"]
    if type(knots) is not list or not 2 <= len(knots) <= 65:
        raise ValueError("Bounded complete original schedule required")
    fractions = []
    for knot in knots:
        if type(knot) is not dict or set(knot) != {"fraction", "targetsMeters", "activation"}:
            raise ValueError("Exact target/activation knot fields required")
        fraction = _fraction(knot["fraction"])
        if (fraction * subdivisions).denominator != 1 or fractions and fraction <= fractions[-1]:
            raise ValueError("Ordered initial-grid gripper knots required")
        fractions.append(fraction)
        if (type(knot["targetsMeters"]) is not list or len(knot["targetsMeters"]) != len(anchors)
                or type(knot["activation"]) is not list or len(knot["activation"]) != len(anchors)):
            raise ValueError("Complete target and activation arrays required")
        for target in knot["targetsMeters"]:
            if type(target) is not list or len(target) != 3:
                raise ValueError("Three target coordinates required")
            for coordinate in target:
                _number(coordinate)
        for active in knot["activation"]:
            if not 0 <= _number(active, maximum=1):
                raise ValueError("Activation must lie in [0,1]")
    if fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError("Original gripper recipe must span [0,1]")
    return recipe


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


def _solve(matrix, rhs):
    determinant = _det(matrix)
    if determinant == 0:
        raise ValueError("Singular exact gripper child coefficient basis")
    return [_det([[rhs[row] if column == j else matrix[row][j] for j in range(3)] for row in range(3)]) / determinant
            for column in range(3)]


def _nearest(value):
    rounded = float(value)
    if value < 0 or not math.isfinite(rounded) or not 0 <= rounded <= 1 or value > 0 and rounded == 0:
        raise ValueError("Unrepresentable nonnegative gripper weight")
    center = Fraction(rounded)
    if center != value:
        low = (Fraction(math.nextafter(rounded, -math.inf)) + center) / 2
        high = (Fraction(math.nextafter(rounded, math.inf)) + center) / 2
        even = struct.unpack(">Q", struct.pack(">d", rounded))[0] % 2 == 0
        if not low <= value <= high or value in (low, high) and not even:
            raise ValueError("Incorrect nearest-even gripper conversion")
    return rounded


def _binding_anchor(anchor, local_face, parent, refinement, basis, parents):
    source = [Fraction()] * 11
    for vertex, weight in zip(local_face, anchor["weights"]):
        source[vertex] = _number(weight, maximum=1)
    candidates = []
    for child, (face, ancestor) in enumerate(zip(refinement["finalLocalMesh"]["triangles"], parents)):
        if ancestor != parent:
            continue
        weights = _solve([[basis[vertex][old] for vertex in face] for old in local_face], [source[old] for old in local_face])
        if any(sum((weights[j] * basis[face[j]][old] for j in range(3)), Fraction()) != source[old] for old in range(11)):
            raise ValueError("Exact child solve does not preserve the original material operator")
        candidates.append((child, face, weights, all(weight >= 0 for weight in weights)))
    admitted = [candidate for candidate in candidates if candidate[3]]
    if not admitted:
        raise ValueError("No exactly nonnegative child gripper anchor")
    supports = [{vertex: weight for vertex, weight in zip(face, weights) if weight} for _, face, weights, _ in admitted]
    if any(support != supports[0] for support in supports[1:]):
        raise ValueError("Distinct refined gripper operators require explicit unresolved selection")
    chosen, face, exact, _ = min(admitted, key=lambda item: item[0])
    rounded = [_nearest(weight) for weight in exact]
    if abs(math.fsum(rounded) - 1.) > 1e-12:
        raise ValueError("Derived gripper violates unchanged normalized-weight admission")
    binary = list(map(Fraction, rounded))
    errors = [actual - ideal for actual, ideal in zip(binary, exact)]
    residual = [sum((binary[j] * basis[face[j]][old] for j in range(3)), Fraction()) - source[old] for old in range(11)]
    bounds = [sum((abs(errors[j] * basis[face[j]][old]) for j in range(3)), Fraction()) for old in range(11)]
    sums = [sum(values, Fraction()) for values in (source, exact, binary)]
    original = refinement["originalLocalMesh"]["verticesMeters"]
    stored = refinement["finalLocalMesh"]["verticesMeters"]
    source_point = [sum((weight * _number(point[axis]) for weight, point in zip(source, original)), Fraction()) for axis in range(2)]
    ideal_point = [sum((weight * _number(stored[vertex][axis]) for vertex, weight in zip(face, exact)), Fraction()) for axis in range(2)]
    binary_point = [sum((weight * _number(stored[vertex][axis]) for vertex, weight in zip(face, binary)), Fraction()) for axis in range(2)]
    audit = {
        "exactSelectedWeights": [{"vertex": vertex, "weight": _rat(weight)} for vertex, weight in sorted(zip(face, exact)) if weight],
        "selectedNumericalWeights": [{"vertex": vertex, "weight": weight} for vertex, weight in sorted(zip(face, rounded)) if weight],
        "weightRoundingErrors": [{"vertex": vertex, "binary64MinusExact": _rat(error)} for vertex, error in zip(face, errors)],
        "pullback": {"coefficients": [{"sourceVertex": old, "binary64MinusSource": _rat(value), "absoluteRoundingBound": _rat(bound)}
                                      for old, (value, bound) in enumerate(zip(residual, bounds))],
            "residualL1": _rat(sum(map(abs, residual), Fraction())), "residualLInfinity": _rat(max(map(abs, residual))),
            "roundingBoundL1": _rat(sum(bounds, Fraction())), "roundingBoundLInfinity": _rat(max(bounds)),
            "residualSum": _rat(sum(residual, Fraction()))},
        "weightSums": {"source": _rat(sums[0]), "exactDerived": _rat(sums[1]), "binary64Derived": _rat(sums[2]),
            "exactDerivedMinusSource": _rat(sums[1] - sums[0]), "binary64DerivedMinusSource": _rat(sums[2] - sums[0])},
        "storedCoordinateResidualsMeters": {
            "exactWeightsMinusSource": [_rat(a - b) for a, b in zip(ideal_point, source_point)],
            "binary64WeightsMinusSource": [_rat(a - b) for a, b in zip(binary_point, source_point)],
            "binary64MinusExactWeights": [_rat(a - b) for a, b in zip(binary_point, ideal_point)]}}
    displayed = [{"childTriangleIndex": child, "sourceParentTriangle": parent, "orientedVertices": vertices,
                  "coefficientNonnegative": nonnegative, "exactWeights": [_rat(weight) for weight in weights]}
                 for child, vertices, weights, nonnegative in candidates]
    return chosen, [candidate[0] for candidate in admitted], rounded, displayed, audit


def verify_binding_gripper_remap(original_source, refined_source, descriptor, *, subdivisions):
    """Audit unchanged controls and exact original-to-refined material lineage."""
    inputs = [_encoded(value) for value in (original_source, refined_source, descriptor)]
    if type(subdivisions) is not int or not 1 <= subdivisions <= 4096 or subdivisions & (subdivisions - 1):
        raise ValueError("Bounded power-of-two subdivisions required")
    try:
        result = _verify(original_source, refined_source, descriptor, subdivisions)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, StopIteration) as error:
        raise ValueError("Malformed independent gripper-remap input") from error
    if inputs != [_encoded(value) for value in (original_source, refined_source, descriptor)]:
        raise ValueError("Gripper-remap inputs changed during verification")
    return result


def _verify(original, refined, descriptor, subdivisions):
    if type(refined) is not dict or set(refined) != CORE_FIELDS or refined["profile"] != "source-left-binding-refined-unit-v1":
        raise ValueError("Bare strict refined source core required")
    base = refined["baseUnit"]
    if type(base) is not dict or base.get("profile") != "source-cuff-construction-unit-v1" or base.get("side") != "left":
        raise ValueError("Immutable original left cuff source required")
    if type(original) is not dict or original.get("profile") != "source-cuff-construction-unit-v1":
        raise ValueError("Original v1 cuff source profile required")
    if {"baseUnit", "bindingRefinement", "bindingSeamRemap"}.intersection(original):
        raise ValueError("Original source cannot contain a downgraded refinement namespace")
    if {"placedMeters", "bindingFirstTurnDiagnostic", "sewingActuation", "gripperActuation", "foldActuation",
            "assemblySchedule", "sewingFrames", "bindingRefinement"}.intersection(base):
        raise ValueError("Immutable base must remain uncontrolled")
    for document in (original, refined, base):
        if document.get("accepted") is not False or document.get("solverReady") is not False:
            raise ValueError("Unaccepted source declarations required")
        if document.get("units") != "m" or document.get("sourceArcUnits") != "mm":
            raise ValueError("Explicit metre/mm source units required")
    if any(instance.get("mirrorX") is not False for instance in base["instances"]):
        raise ValueError("Unmirrored original physical instances required")
    for key, value in base.items():
        _same(original[key], value, "original immutable base field " + key)
    original_mesh, refined_mesh = _mesh(original), _mesh(refined)
    recipe = _recipe(original, original_mesh, subdivisions)
    refinement = refined["bindingRefinement"]
    basis, parents = _lineage(base, refinement)
    face_offsets = _canonical_meshes(base, refined, refinement, original_mesh, refined_mesh)
    anchors, numerical = [], []
    for index, anchor in enumerate(recipe["anchors"]):
        instance = anchor["instanceId"]
        original_index = anchor["triangleIndex"]
        local_index = original_mesh["lookup"][original_index][1]
        local_face = original_mesh["local"][instance][local_index]
        if instance == INSTANCE:
            chosen, equivalent, weights, candidates, audit = _binding_anchor(anchor, local_face, local_index, refinement, basis, parents)
            extra = {"candidates": candidates, "coefficientAudit": audit}
        else:
            chosen, equivalent, weights, extra = local_index, [local_index], anchor["weights"], {}
            _same(refined_mesh["local"][instance][chosen], local_face, "unchanged-instance oriented triangle")
        canonical_index = face_offsets[instance] + chosen
        numerical_anchor = {**anchor, "triangleIndex": canonical_index, "weights": weights}
        original_point = [sum((_number(weight, maximum=1) * _number(original_mesh["points"][vertex][axis])
                              for vertex, weight in zip(original_mesh["faces"][original_index], anchor["weights"])), Fraction())
                          for axis in range(3)]
        numerical_point = [sum((_number(weight, maximum=1) * _number(refined_mesh["points"][vertex][axis])
                               for vertex, weight in zip(refined_mesh["faces"][canonical_index], weights)), Fraction())
                           for axis in range(3)]
        anchors.append({"gripperIndex": index, "id": anchor["id"], "instanceId": instance,
            "migrationKind": "refined-binding" if instance == INSTANCE else "unchanged-instance",
            "originalAnchor": anchor, "originalLocalTriangleIndex": local_index, "originalLocalVertices": local_face,
            "selectedLocalTriangleIndex": chosen, "equivalentLocalTriangleIndices": equivalent,
            "selectedCanonicalTriangleIndex": canonical_index, "numericalAnchor": numerical_anchor,
            "originalWeightSum": _rat(sum((_number(weight, maximum=1) for weight in anchor["weights"]), Fraction())),
            "storedCoordinateResidualMeters": [_rat(a - b) for a, b in zip(numerical_point, original_point)], **extra})
        numerical.append(numerical_anchor)
    numerical_recipe = {**recipe, "meshSha256": refined_mesh["meshSha256"], "anchors": numerical}
    header = {"profile": "source-binding-gripper-remap-v1", "policy": "same-control-exact-coefficient-nearest-binary64-v1",
        "accepted": False, "solverReady": False, "executable": False, "subdivisions": subdivisions,
        "originalSourceSha256": _sha(original), "refinedSourceSha256": _sha(refined), "baseUnitSha256": _sha(base),
        "originalRecipeSha256": _sha(recipe), "originalMeshSha256": original_mesh["meshSha256"],
        "refinedMeshSha256": refined_mesh["meshSha256"], "originalRecipe": recipe,
        "numericalRecipe": numerical_recipe, "anchors": anchors}
    if type(descriptor) is not dict or set(descriptor) != set(header) | {"limitations"}:
        raise ValueError("Complete exact gripper-remap descriptor fields required")
    for key, value in header.items():
        _same(descriptor[key], value, "descriptor " + key)
    if (type(descriptor["limitations"]) is not list or not 1 <= len(descriptor["limitations"]) <= 16
            or any(type(value) is not str or not value for value in descriptor["limitations"])):
        raise ValueError("Explicit bounded migration limitations required")
    return {"profile": PROFILE, "verified": True, "accepted": False,
        "originalSourceSha256": _sha(original), "refinedSourceSha256": _sha(refined),
        "descriptorSha256": _sha(descriptor), "baseUnitSha256": _sha(base),
        "originalRecipeSha256": _sha(recipe), "numericalRecipeSha256": _sha(numerical_recipe),
        "subdivisions": subdivisions, "gripperCount": len(anchors),
        "refinedBindingGrippers": sum(anchor["migrationKind"] == "refined-binding" for anchor in anchors),
        "unchangedInstanceGrippers": sum(anchor["migrationKind"] == "unchanged-instance" for anchor in anchors),
        "enumeratedCandidateCount": sum(len(anchor.get("candidates", [])) for anchor in anchors),
        "selectedCanonicalTriangleIndices": [anchor["selectedCanonicalTriangleIndex"] for anchor in anchors],
        "sourceScope": "Independent canonical/base identity, raw two-stage lineage, original-parent candidate completeness, exact coefficient pullback, prescribed nearest-even conversion, recorded residuals, canonical reindexing and unchanged control recipe. Pattern and split-geometry validity require the separately captured source validator. No frame-side, independent refined-motion, material, construction or physical acceptance proof."}
