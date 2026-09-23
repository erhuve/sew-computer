"""Independent binary-input coefficient audit for a derived binding source.

No engine, source validator, splitter or remap implementation is imported here.
The separately captured source validator must rederive the pattern and split
geometry. This audit reconstructs the raw lineage and numerical seam operators.
"""

from fractions import Fraction
import hashlib
import json
import math
import struct


PROFILE = "source-left-binding-refined-unit-v1"
REQUIRED_HELPER_FILES = ()
INSTANCE = "opening_binding_left_left:shell"
TEMPLATE = "opening_binding_left_left"
POLICY = "exact-coefficient-nearest-binary64-v1"
MAX_BYTES = 24 * 1024 ** 2


def _encoded(value):
    pending, remaining = [(value, 0)], 1000000
    while pending:
        item, depth = pending.pop()
        remaining -= 1
        if remaining < 0 or depth > 40:
            raise ValueError("Independent remap structure budget exceeded")
        if type(item) is dict:
            if any(type(key) is not str for key in item) or len(item) * 2 > remaining:
                raise ValueError("Bounded string-keyed raw JSON required")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif type(item) is list:
            if len(item) > remaining:
                raise ValueError("Independent remap array budget exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > MAX_BYTES:
                raise ValueError("Independent remap string budget exceeded")
        elif type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Independent remap integer budget exceeded")
        elif item is None or type(item) in (bool, float):
            pass
        else:
            raise ValueError("Raw JSON values required for independent remap")
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError, OverflowError, UnicodeEncodeError) as error:
        raise ValueError("Bounded finite JSON remap input required") from error
    if len(data) > MAX_BYTES:
        raise ValueError("Independent remap input byte budget exceeded")
    return data


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _same(value, expected, label):
    if _encoded(value) != _encoded(expected):
        raise ValueError("Independent remap mismatch: " + label)


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite non-Boolean binary64 value required")
    rounded = float(value)
    if rounded != value:
        raise ValueError("Input integer is not exactly representable as binary64")
    return Fraction(rounded)


def _index(value, size):
    if type(value) is not int or not 0 <= value < size:
        raise ValueError("Bounded non-Boolean remap index required")
    return value


def _rat(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "roundedBinary64": float(value)}


def _read_rat(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator", "roundedBinary64"}:
        raise ValueError("Canonical rational witness required")
    n, d = value["numerator"], value["denominator"]
    if type(n) is not str or type(d) is not str or max(len(n), len(d)) > 2048:
        raise ValueError("Rational witness budget exceeded")
    try:
        result = Fraction(int(n), int(d))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Invalid rational witness") from error
    _same(value, _rat(result), "canonical rational witness")
    return result


def _faces(raw, count):
    if not isinstance(raw, list) or not 1 <= len(raw) <= 64:
        raise ValueError("Bounded local face array required")
    result = []
    for face in raw:
        if not isinstance(face, list) or len(face) != 3:
            raise ValueError("Three local indices required")
        values = [_index(index, count) for index in face]
        if len(set(values)) != 3:
            raise ValueError("Repeated local face vertex")
        result.append(values)
    return result


def _points(raw, dimensions=2, maximum=64):
    if not isinstance(raw, list) or not 3 <= len(raw) <= maximum:
        raise ValueError("Bounded local vertex array required")
    for point in raw:
        if not isinstance(point, list) or len(point) != dimensions:
            raise ValueError("Invalid local point dimension")
        if any(abs(_number(value)) > 100 for value in point):
            raise ValueError("Local point coordinate budget exceeded")
    return raw


def _det(matrix):
    a, b, c = matrix
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def _solve(matrix, rhs):
    """Cramer's rule, independently of the producer's elimination algorithm."""
    determinant = _det(matrix)
    if determinant == 0:
        raise ValueError("Singular exact child coefficient basis")
    result = []
    for column in range(3):
        changed = [[rhs[i] if j == column else matrix[i][j] for j in range(3)] for i in range(3)]
        result.append(_det(changed) / determinant)
    return result


def _nearest(value):
    """Check round-to-nearest/even against exact adjacent-float midpoint cells."""
    rounded = float(value)
    if not math.isfinite(rounded) or value < 0 or not 0 <= rounded <= 1:
        raise ValueError("Unrepresentable exact remap weight")
    if value > 0 and rounded == 0:
        raise ValueError("Positive remap support cannot underflow to zero")
    center = Fraction(rounded)
    if center == value:
        return rounded
    before = math.nextafter(rounded, -math.inf)
    after = math.nextafter(rounded, math.inf)
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError("Unbounded rounding neighborhood")
    lo, hi = (Fraction(before) + center) / 2, (Fraction(after) + center) / 2
    even = (struct.unpack(">Q", struct.pack(">d", rounded))[0] & 1) == 0
    if not lo <= value <= hi or (value in (lo, hi) and not even):
        raise ValueError("Incorrect nearest-even binary64 conversion")
    return rounded


def _reconstruction(points, original, basis):
    vertices, maximum = [], Fraction()
    for vertex, (point, row) in enumerate(zip(points, basis)):
        differences = [_number(point[axis]) - sum((weight * _number(original[index][axis])
                       for index, weight in enumerate(row)), Fraction()) for axis in range(2)]
        maximum = max(maximum, *(abs(value) for value in differences))
        vertices.append({"vertex": vertex, "storedMinusReconstructedMeters": [_rat(value) for value in differences],
                         "weightSumMinusOne": _rat(sum(row, Fraction()) - 1)})
    return {"vertices": vertices, "maximumAbsoluteCoordinateResidualMeters": _rat(maximum)}


def _compose(base, refinement):
    if refinement.get("profile") != "source-left-binding-refinement-descriptor-v1":
        raise ValueError("Exact refinement profile required")
    for field in ("accepted", "solverReady", "executable"):
        if refinement.get(field) is not False:
            raise ValueError("Unaccepted non-executable refinement required")
    _same(refinement["sourceUnitSha256"], _sha(base), "immutable base digest")
    _same(refinement["originalEmbeddedConstraintsSha256"], _sha(base["embeddedConstraints"]), "source rows digest")
    _same(refinement["originalTemplateSha256"], _sha(base["sourceTemplates"][TEMPLATE]), "source template digest")
    _same(refinement["instanceId"], INSTANCE, "binding instance")
    _same(refinement["sourceTemplateId"], TEMPLATE, "binding template")
    original = refinement["originalLocalMesh"]
    points = _points(original["verticesMeters"])
    if len(points) != 11:
        raise ValueError("Original eleven binding vertices required")
    faces = _faces(original["triangles"], 11)
    if len(faces) != 12:
        raise ValueError("Original twelve binding faces required")
    start = base["instanceOffsets"][INSTANCE]
    _index(start, len(base["restMeters"]))
    _same([point + [0.] for point in points], base["restMeters"][start:start + 11], "original binding rest positions")
    _same(faces, base["sourceTemplates"][TEMPLATE]["triangles"], "original local faces")
    basis = [[Fraction(int(i == j)) for j in range(11)] for i in range(11)]
    parents = list(range(12))
    stages = refinement["stages"]
    if not isinstance(stages, list) or len(stages) != 2:
        raise ValueError("Both ordered raw split stages required")
    for stage, name in zip(stages, ("right", "left")):
        _same(stage["sourcePathName"], name, "ordered source cuts")
        _same(stage["inputMeshSha256"], _sha({"verticesMeters": points, "triangles": faces}), "stage input mesh")
        output = stage["output"]
        new_points = _points(output["vertices"])
        new_faces = _faces(output["triangles"], len(new_points))
        _same(new_points[:len(points)], points, "unchanged stage vertex prefix")
        weights = output["sourceWeights"]
        if not isinstance(weights, list) or len(weights) != len(new_points):
            raise ValueError("Complete raw stage weights required")
        composed, stage_basis = [], []
        for vertex, support in enumerate(weights):
            if not isinstance(support, dict) or not 1 <= len(support) <= 3:
                raise ValueError("Bounded raw stage support required")
            terms = []
            for raw, weight in support.items():
                if type(raw) is not str or not raw.isascii() or not raw.isdigit() or str(int(raw)) != raw:
                    raise ValueError("Canonical raw support index required")
                index = _index(int(raw), len(points))
                rational = _number(weight)
                if not 0 <= rational <= 1:
                    raise ValueError("Negative or oversized raw lineage weight")
                terms.append((index, rational))
            if vertex < len(points):
                _same(support, {str(vertex): 1.}, "original stage vertex operator")
            stage_basis.append([sum((weight for index, weight in terms if index == i), Fraction()) for i in range(len(points))])
            composed.append([sum((weight * basis[index][i] for index, weight in terms), Fraction()) for i in range(11)])
        _same(stage["binaryInputReconstruction"], _reconstruction(new_points, points, stage_basis), "raw-stage reconstruction residuals")
        raw_parents = output["parentTriangles"]
        if not isinstance(raw_parents, list) or len(raw_parents) != len(new_faces):
            raise ValueError("Complete raw stage parent lineage required")
        parents = [parents[_index(parent, len(faces))] for parent in raw_parents]
        basis, points, faces = composed, new_points, new_faces
    _same(refinement["finalLocalMesh"], {"verticesMeters": points, "triangles": faces}, "final local mesh")
    _same(refinement["ultimateOriginalTriangleIndices"], parents, "ultimate original parents")
    expected = [{"vertex": vertex, "weights": [{"sourceVertex": index, "weight": _rat(weight)}
                  for index, weight in enumerate(row) if weight]} for vertex, row in enumerate(basis)]
    _same(refinement["composedVertexWeights"], expected, "raw-composed coefficient basis")
    _same(refinement["binaryInputReconstruction"], _reconstruction(points, original["verticesMeters"], basis), "composed reconstruction residuals")
    for face, parent in zip(faces, parents):
        allowed = original["triangles"][parent]
        if any(weight and index not in allowed for vertex in face for index, weight in enumerate(basis[vertex])):
            raise ValueError("Child coefficient support escapes original parent")
    if len(points) != 29 or len(faces) != 44:
        raise ValueError("Expected bounded twenty-nine vertex/forty-four face refinement")
    return basis, parents


def _anchor_candidates(row, refinement, basis, parents):
    source = [Fraction()] * 11
    seen = set()
    for term in row["terms"]:
        if term["instanceId"] != INSTANCE:
            continue
        index = _index(term["vertex"], 11)
        weight = -_number(term["coefficient"])
        if weight <= 0 or index in seen:
            raise ValueError("Distinct strictly negative source binding terms required")
        source[index] = weight
        seen.add(index)
    choices = [index for index, face in enumerate(refinement["originalLocalMesh"]["triangles"])
               if seen and seen <= set(face)]
    if not choices:
        raise ValueError("Original source parent required for bounded seam remap")
    candidates = []
    for index, (face, owner) in enumerate(zip(refinement["finalLocalMesh"]["triangles"], parents)):
        if owner not in choices:
            continue
        original_face = refinement["originalLocalMesh"]["triangles"][owner]
        matrix = [[basis[vertex][original] for vertex in face] for original in original_face]
        exact = _solve(matrix, [source[original] for original in original_face])
        if any(sum((exact[j] * basis[face[j]][i] for j in range(3)), Fraction()) != source[i] for i in range(11)):
            raise ValueError("Exact coefficient solve does not reproduce full original operator")
        candidates.append((index, face, exact, all(value >= 0 for value in exact)))
    admissible = [candidate for candidate in candidates if candidate[3]]
    if not admissible:
        raise ValueError("No nonnegative exact coefficient child")
    operators = [{vertex: weight for vertex, weight in zip(face, weights) if weight}
                 for _, face, weights, _ in admissible]
    if any(operator != operators[0] for operator in operators[1:]):
        raise ValueError("Different child material operators need explicit selection policy")
    return source, choices, candidates, admissible


def verify_binding_remap(source):
    """Independently verify remap arithmetic and derived mesh/row consistency.

    This is intentionally not a replacement for captured source-pattern and
    splitter rederivation, a frame-side choice, or physical phase acceptance.
    """
    _encoded(source)
    if not isinstance(source, dict) or source.get("profile") != PROFILE:
        raise ValueError("Exact refined binding source profile required")
    try:
        return _verify(source)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, StopIteration) as error:
        raise ValueError("Malformed independent binding remap input") from error


def _verify(source):
    core_fields = {"profile", "accepted", "solverReady", "units", "sourceArcUnits", "baseUnit", "bindingRefinement",
        "bindingSeamRemap", "instances", "numericalMeshes", "instanceOffsets", "instanceTriangleOffsets", "restMeters",
        "triangles", "embeddedConstraints", "sourceRowCorrespondence", "derivationCodeDigests", "limitations"}
    control_fields = {"placedMeters", "sewingActuation", "gripperActuation", "foldActuation", "assemblySchedule",
                      "bindingRefinedDiagnostic", "sewingFrames"}
    if set(source) - control_fields != core_fields:
        raise ValueError("Exact versioned derived source fields required")
    base = source["baseUnit"]
    if not isinstance(base, dict) or base.get("profile") != "source-cuff-construction-unit-v1":
        raise ValueError("Immutable original cuff source profile required")
    for document in (source, base):
        if document.get("accepted") is not False or document.get("solverReady") is not False:
            raise ValueError("Unaccepted source declaration required")
        if document.get("units") != "m" or document.get("sourceArcUnits") != "mm":
            raise ValueError("Explicit metre/mm source units required")
    if base.get("side") != "left" or any(instance.get("mirrorX") is not False for instance in base["instances"]):
        raise ValueError("Unmirrored left base required")
    forbidden = {"placedMeters", "bindingFirstTurnDiagnostic", "sewingActuation", "gripperActuation",
                 "foldActuation", "assemblySchedule", "sewingFrames", "bindingRefinement"}
    if forbidden.intersection(base) or "sourceTemplates" in source or "sewingFrames" in source:
        raise ValueError("Uncontrolled immutable base and no inferred refined source frames required")
    refinement = source["bindingRefinement"]
    basis, parents = _compose(base, refinement)
    return _verify_rows_and_mesh(source, base, refinement, basis, parents)


def _verify_rows_and_mesh(source, base, refinement, basis, parents):
    original_rows = base["embeddedConstraints"]["constraints"]
    if not isinstance(original_rows, list) or len(original_rows) != 40:
        raise ValueError("All forty immutable original rows required")
    indices = []
    for index, row in enumerate(original_rows):
        if (type(row["memberIndex"]) is not int or type(row["registrationId"]) is not str
                or not 0 <= _number(row["fraction"]) <= 1 or _number(row["complianceMPerN"]) <= 0):
            raise ValueError("Strict original row metadata required")
        if any(term["instanceId"] == INSTANCE for term in row["terms"]):
            indices.append(index)
    if indices != list(range(5)):
        raise ValueError("Exactly the five original left-binding rows required")
    if len({_number(row["complianceMPerN"]) for row in original_rows}) != 1:
        raise ValueError("Homogeneous original compliance required")
    _same([original_rows[i]["fraction"] for i in indices], [0., .25, .5, .75, 1.], "five distinct original samples")
    expected_rows = []
    for index in indices:
        row = original_rows[index]
        weights, owners, candidates, admissible = _anchor_candidates(row, refinement, basis, parents)
        chosen, face, exact, _ = min(admissible, key=lambda candidate: candidate[0])
        rounded = [_nearest(value) for value in exact]
        if abs(math.fsum(rounded) - 1.) > 1e-12:
            raise ValueError("Rounded anchor violates existing normalized-row admission")
        binary = [Fraction(value) for value in rounded]
        delta = [actual - ideal for actual, ideal in zip(binary, exact)]
        # Compute the complete original-node residual, rather than reconstructing
        # it from separately rounded coordinates or the producer's witnesses.
        residual = [sum((binary[j] * basis[face[j]][i] for j in range(3)), Fraction()) - weights[i]
                    for i in range(11)]
        bounds = [sum((abs(delta[j]) * abs(basis[face[j]][i]) for j in range(3)), Fraction()) for i in range(11)]
        if any(abs(error) > bound for error, bound in zip(residual, bounds)):
            raise ValueError("Coefficient rounding residual exceeds exact bound")
        sums = [sum(values, Fraction()) for values in (weights, exact, binary)]
        original = refinement["originalLocalMesh"]["verticesMeters"]
        stored = refinement["finalLocalMesh"]["verticesMeters"]
        source_point = [sum((weight * _number(point[axis]) for weight, point in zip(weights, original)), Fraction())
                        for axis in range(2)]
        ideal_point = [sum((exact[j] * _number(stored[face[j]][axis]) for j in range(3)), Fraction()) for axis in range(2)]
        binary_point = [sum((binary[j] * _number(stored[face[j]][axis]) for j in range(3)), Fraction()) for axis in range(2)]
        fraction = _number(row["fraction"])
        expected_rows.append({
            "rowIndex": index, "rowId": "row:" + _sha([row["registrationId"], row["memberIndex"], fraction.numerator, fraction.denominator]),
            "registrationId": row["registrationId"], "memberIndex": row["memberIndex"], "sourceFraction": row["fraction"],
            "originalRowSha256": _sha(row), "originalRow": row, "anchorInstanceId": INSTANCE, "anchorSign": -1,
            "originalParentTriangles": owners,
            "originalWeights": [{"vertex": i, "weight": _rat(weight)} for i, weight in enumerate(weights) if weight],
            "candidates": [{"childTriangleIndex": child, "sourceParentTriangle": parents[child], "orientedVertices": vertices,
                            "coefficientNonnegative": positive, "exactWeights": [_rat(value) for value in values]}
                           for child, vertices, values, positive in candidates],
            "selectedChildTriangleIndex": chosen, "equivalentChildTriangleIndices": [item[0] for item in admissible],
            "exactSelectedWeights": [{"vertex": vertex, "weight": _rat(value)} for vertex, value in sorted(zip(face, exact)) if value],
            "selectedNumericalWeights": [{"vertex": vertex, "weight": value} for vertex, value in sorted(zip(face, rounded)) if value],
            "weightRoundingErrors": [{"vertex": vertex, "binary64MinusExact": _rat(value)} for vertex, value in zip(face, delta)],
            "pullback": {"coefficients": [{"sourceVertex": i, "binary64MinusSource": _rat(error), "absoluteRoundingBound": _rat(bound)}
                                           for i, (error, bound) in enumerate(zip(residual, bounds))],
                "residualL1": _rat(sum(map(abs, residual), Fraction())), "residualLInfinity": _rat(max(map(abs, residual))),
                "roundingBoundL1": _rat(sum(bounds, Fraction())), "roundingBoundLInfinity": _rat(max(bounds)),
                "residualSum": _rat(sum(residual, Fraction()))},
            "weightSums": {"source": _rat(sums[0]), "exactDerived": _rat(sums[1]), "binary64Derived": _rat(sums[2]),
                           "exactDerivedMinusSource": _rat(sums[1] - sums[0]), "binary64DerivedMinusSource": _rat(sums[2] - sums[0])},
            "storedCoordinateResidualsMeters": {
                "exactWeightsMinusSource": [_rat(x - y) for x, y in zip(ideal_point, source_point)],
                "binary64WeightsMinusSource": [_rat(x - y) for x, y in zip(binary_point, source_point)],
                "binary64MinusExactWeights": [_rat(x - y) for x, y in zip(binary_point, ideal_point)]}})
    remap = source["bindingSeamRemap"]
    _same(remap["rows"], expected_rows, "complete independently rederived seam arithmetic")
    expected_header = {"profile": "source-left-binding-seam-remap-descriptor-v1", "policy": POLICY,
        "accepted": False, "solverReady": False, "executable": False, "sourceProfile": base["profile"],
        "refinementProfile": refinement["profile"], "sourceUnitSha256": _sha(base),
        "refinementDescriptorSha256": _sha(refinement), "originalEmbeddedConstraintsSha256": _sha(base["embeddedConstraints"]),
        "instanceId": INSTANCE, "localRefinedVertexCount": 29}
    for key, value in expected_header.items():
        _same(remap[key], value, "remap " + key)
    text_keys = {"selectionPolicy", "conversionPolicy", "arithmeticScope", "scope", "pending"}
    if set(remap) != set(expected_header) | {"rows"} | text_keys:
        raise ValueError("Exact remap descriptor fields required")
    if any(type(remap[key]) is not str or not remap[key] for key in text_keys - {"pending"}):
        raise ValueError("Explicit remap scope declarations required")
    if not isinstance(remap["pending"], list) or len(remap["pending"]) != 3 or any(type(value) is not str for value in remap["pending"]):
        raise ValueError("Explicit unresolved frame and execution gates required")
    _verify_mesh_and_bundle(source, base, refinement, expected_rows)
    return {"profile": "independent-binding-remap-verification-v1", "verified": True, "accepted": False,
        "sourceProfile": PROFILE, "sourceSha256": _sha({key: value for key, value in source.items() if key != "sewingActuation"}),
        "baseUnitSha256": _sha(base), "refinementSha256": _sha(refinement), "remapSha256": _sha(remap),
        "rowCount": 40, "remappedRows": 5, "unchangedRows": 35,
        "enumeratedCandidateCount": sum(len(row["candidates"]) for row in expected_rows),
        "selectedChildTriangleIndices": [row["selectedChildTriangleIndex"] for row in expected_rows],
        "scope": "Independent raw-stage exact coefficient composition, complete original-parent child enumeration, nonnegativity, nearest-even binary64 conversion, residual bounds and derived mesh/row consistency. Pattern and split-geometry rederivation require the separately captured source validator; no frame-side, independent refined-motion, construction or physical acceptance proof."}


def _verify_mesh_and_bundle(source, base, refinement, remapped):
    instances = base["instances"]
    if not isinstance(instances, list) or len(instances) != 5 or len({item["id"] for item in instances}) != 5:
        raise ValueError("Exactly five distinct source instances required")
    _same(source["instances"], instances, "all original fabric instances")
    meshes, offsets, face_offsets, rest, triangles = {}, {}, {}, [], []
    for instance in instances:
        name = instance["id"]
        original = base["sourceTemplates"][instance["templateId"]]
        start = _index(base["instanceOffsets"][name], len(base["restMeters"]))
        if name == INSTANCE:
            mesh = {"verticesMeters": [point + [0.] for point in refinement["finalLocalMesh"]["verticesMeters"]],
                    "triangles": refinement["finalLocalMesh"]["triangles"]}
        else:
            mesh = {"verticesMeters": base["restMeters"][start:start + len(original["restPositions"])],
                    "triangles": original["triangles"]}
        _points(mesh["verticesMeters"], dimensions=3, maximum=25000)
        for face in mesh["triangles"]:
            if not isinstance(face, list) or len(face) != 3 or len(set(face)) != 3:
                raise ValueError("Valid ordered local faces required")
            for vertex in face:
                _index(vertex, len(mesh["verticesMeters"]))
        meshes[name] = mesh
        offsets[name], face_offsets[name] = len(rest), len(triangles) // 3
        rest.extend(mesh["verticesMeters"])
        triangles.extend(offsets[name] + vertex for face in mesh["triangles"] for vertex in face)
    for key, expected in (("numericalMeshes", meshes), ("instanceOffsets", offsets),
                          ("instanceTriangleOffsets", face_offsets), ("restMeters", rest), ("triangles", triangles)):
        _same(source[key], expected, "canonical " + key)
    original_bundle = base["embeddedConstraints"]
    expected_rows = json.loads(_encoded(original_bundle["constraints"]))
    for evidence in remapped:
        row = expected_rows[evidence["rowIndex"]]
        samples = [sample for sample in row["sourceSamples"] if sample["instanceId"] == INSTANCE]
        if len(samples) != 1:
            raise ValueError("Unique binding-side source sample required")
        weights = evidence["selectedNumericalWeights"]
        samples[0]["weights"] = weights
        row["terms"] = sorted([term for term in row["terms"] if term["instanceId"] != INSTANCE] +
                              [{"instanceId": INSTANCE, "vertex": item["vertex"], "coefficient": -item["weight"]} for item in weights],
                              key=lambda term: (term["instanceId"], term["vertex"]))
    bundle = source["embeddedConstraints"]
    expected = {"kind": "derived-embedded-sewing-coupling", "solverReady": False,
        **{key: original_bundle[key] for key in ("units", "sourceArcUnits", "topology", "registrations")},
        "constraints": expected_rows, "originalBundleSha256": _sha(original_bundle),
        "sourceIdentities": {name: {"originalPanelDigest": original_bundle["sourceIdentities"][name]["panelDigest"],
            "originalTemplateId": original_bundle["sourceIdentities"][name]["templateId"],
            "numericalMeshSha256": _sha(mesh), "vertexCount": len(mesh["verticesMeters"])} for name, mesh in meshes.items()}}
    if set(bundle) != set(expected) | {"limitations"}:
        raise ValueError("Exact derived coupling fields required")
    for key, value in expected.items():
        _same(bundle[key], value, "derived coupling " + key)
    correspondence = [{"rowIndex": i, "registrationId": old["registrationId"], "memberIndex": old["memberIndex"],
        "fraction": old["fraction"], "originalRowSha256": _sha(old), "numericalRowSha256": _sha(new),
        "bindingAnchorRemapped": i < 5} for i, (old, new) in enumerate(zip(original_bundle["constraints"], expected_rows))]
    _same(source["sourceRowCorrespondence"], correspondence, "all forty original/numerical row witnesses")
