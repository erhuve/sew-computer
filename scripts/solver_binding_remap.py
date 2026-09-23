"""Source-bound approximate seam anchors for the two-crease strip descriptor.

Exact coefficient pullback chooses the material anchor. Binary64 conversion is
a separately reported approximation, never a coordinate-containment repair.
This descriptor is not an executable source profile or a frame-side policy.
"""

from fractions import Fraction
import hashlib
import json
import math

from solver_binding_refinement import INSTANCE, validate_binding_refinement


PROFILE = "source-left-binding-seam-remap-descriptor-v1"
POLICY = "exact-coefficient-nearest-binary64-v1"
MAX_BYTES = 4 * 1024 ** 2
MAX_RATIONAL_BITS = 16384
MAX_REFINED_VERTICES = 128
MAX_REFINED_TRIANGLES = 128


def _encoded(value):
    remaining, text_size = 250000, 0

    def visit(item, depth):
        nonlocal remaining, text_size
        remaining -= 1
        if remaining < 0 or depth > 40:
            raise ValueError("Bounded raw JSON remap data required")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Bounded JSON integer required")
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Finite JSON remap numbers required")
            return
        if type(item) is str:
            if len(item) > MAX_BYTES:
                raise ValueError("Remap JSON byte budget exceeded")
            try:
                text_size += len(item.encode())
            except UnicodeEncodeError as error:
                raise ValueError("UTF-8 remap strings required") from error
            if text_size > MAX_BYTES:
                raise ValueError("Remap JSON byte budget exceeded")
            return
        if type(item) is list:
            if len(item) > remaining:
                raise ValueError("Remap JSON node budget exceeded")
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict:
            if len(item) * 2 > remaining or any(type(key) is not str for key in item):
                raise ValueError("Bounded string-keyed JSON remap objects required")
            for key, child in item.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        raise ValueError("Raw JSON remap values required")

    visit(value, 0)
    result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(result) > MAX_BYTES:
        raise ValueError("Remap JSON byte budget exceeded")
    return result


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _copy(value):
    return json.loads(_encoded(value))


def _bounded_fraction(value):
    if (type(value) is not Fraction or value.numerator.bit_length() > MAX_RATIONAL_BITS
            or value.denominator.bit_length() > MAX_RATIONAL_BITS):
        raise ValueError("Exact remap rational arithmetic budget exceeded")
    return value


def _rational(value):
    _bounded_fraction(value)
    # Rational strings are normative: a tiny nonzero reporting value may round
    # to zero, but no inference of exact equality is made from this float.
    try:
        rounded = float(value)
    except OverflowError as error:
        raise ValueError("Finite remap rational reporting value required") from error
    if not math.isfinite(rounded):
        raise ValueError("Finite remap rational reporting value required")
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "roundedBinary64": rounded}


def _solve_three(matrix, rhs):
    """Exact bounded Gaussian elimination; no geometric tolerance or clipping."""
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix) or len(rhs) != 3:
        raise ValueError("Three-dimensional source coefficient basis required")
    augmented = [[_bounded_fraction(value) for value in row] + [_bounded_fraction(target)]
                 for row, target in zip(matrix, rhs)]
    for column in range(3):
        pivot = next((index for index in range(column, 3) if augmented[index][column]), None)
        if pivot is None:
            raise ValueError("Singular source coefficient child basis")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [_bounded_fraction(value / divisor) for value in augmented[column]]
        for index in range(3):
            if index != column:
                factor = augmented[index][column]
                augmented[index] = [_bounded_fraction(value - factor * reference)
                                    for value, reference in zip(augmented[index], augmented[column])]
    return [row[-1] for row in augmented]


def _nearest_binary64_weight(value):
    """Once-round a nonnegative exact coefficient; retain all positive support."""
    _bounded_fraction(value)
    if value < 0:
        raise ValueError("Exactly nonnegative remap weights required before conversion")
    try:
        rounded = float(value)
    except OverflowError as error:
        raise ValueError("Finite bounded binary64 remap weight required") from error
    if not math.isfinite(rounded) or not 0 <= rounded <= 1:
        raise ValueError("Finite bounded binary64 remap weight required")
    if value > 0 and rounded == 0:
        raise ValueError("Positive remap weight cannot underflow to zero")
    return rounded


def _equivalent_selection(candidates):
    eligible = [item for item in candidates if item["nonnegative"]]
    if not eligible:
        raise ValueError("Unresolved remap: no exactly nonnegative source-coefficient child")
    maps = [tuple(sorted((vertex, weight) for vertex, weight in zip(item["face"], item["weights"])
                         if weight != 0)) for item in eligible]
    if any(value != maps[0] for value in maps[1:]):
        raise ValueError("Unresolved remap: distinct nonnegative refined-vertex anchor operators")
    # Index chooses an equivalent identity for reporting only. It has no effect
    # on the sparse anchor and must never select a normal or crease-side frame.
    return min(eligible, key=lambda item: item["child"]), sorted(item["child"] for item in eligible)


def _basis(refinement):
    count = len(refinement["originalLocalMesh"]["verticesMeters"])
    values = [[Fraction(int(vertex == original)) for original in range(count)] for vertex in range(count)]
    for stage in refinement["stages"]:
        previous = values
        values = [[_bounded_fraction(sum((Fraction(weight) * previous[int(vertex)][original]
                    for vertex, weight in support.items()), Fraction())) for original in range(count)]
                  for support in stage["output"]["sourceWeights"]]
    return values


def _audit_selected(original, stored, basis, source_weights, selected):
    face, exact = selected["face"], selected["weights"]
    rounded = [_nearest_binary64_weight(weight) for weight in exact]
    if abs(math.fsum(rounded) - 1.) > 1e-12:
        raise ValueError("Derived remap retains the existing normalized-row admission bound")
    binary = list(map(Fraction, rounded))
    errors = [value - ideal for value, ideal in zip(binary, exact)]
    residuals, bounds = [], []
    for original_vertex, source_weight in enumerate(source_weights):
        ideal_pullback = sum((weight * basis[vertex][original_vertex] for vertex, weight in zip(face, exact)), Fraction())
        if ideal_pullback != source_weight:
            raise ValueError("Exact source coefficient pullback failed")
        residual = sum((weight * basis[vertex][original_vertex] for vertex, weight in zip(face, binary)), Fraction()) - source_weight
        bound = sum((abs(error * basis[vertex][original_vertex]) for vertex, error in zip(face, errors)), Fraction())
        if abs(residual) > bound:
            raise ValueError("Exact remap rounding bound failed")
        residuals.append(residual)
        bounds.append(bound)
    source_point = [sum((weight * Fraction(point[axis]) for weight, point in zip(source_weights, original)), Fraction())
                    for axis in range(2)]
    ideal_point = [sum((weight * Fraction(stored[vertex][axis]) for vertex, weight in zip(face, exact)), Fraction())
                   for axis in range(2)]
    binary_point = [sum((weight * Fraction(stored[vertex][axis]) for vertex, weight in zip(face, binary)), Fraction())
                    for axis in range(2)]
    sums = [sum(values, Fraction()) for values in (source_weights, exact, binary)]
    return {
        "exactSelectedWeights": [{"vertex": vertex, "weight": _rational(weight)}
                                 for vertex, weight in sorted(zip(face, exact)) if weight != 0],
        "selectedNumericalWeights": [{"vertex": vertex, "weight": weight}
                                     for vertex, weight in sorted(zip(face, rounded)) if weight != 0],
        "weightRoundingErrors": [{"vertex": vertex, "binary64MinusExact": _rational(error)}
                                 for vertex, error in zip(face, errors)],
        "pullback": {"coefficients": [{"sourceVertex": index, "binary64MinusSource": _rational(residual),
                                       "absoluteRoundingBound": _rational(bound)}
                                      for index, (residual, bound) in enumerate(zip(residuals, bounds))],
            "residualL1": _rational(sum(map(abs, residuals), Fraction())),
            "residualLInfinity": _rational(max(map(abs, residuals))),
            "roundingBoundL1": _rational(sum(bounds, Fraction())),
            "roundingBoundLInfinity": _rational(max(bounds)),
            "residualSum": _rational(sum(residuals, Fraction()))},
        "weightSums": {"source": _rational(sums[0]), "exactDerived": _rational(sums[1]),
            "binary64Derived": _rational(sums[2]), "exactDerivedMinusSource": _rational(sums[1] - sums[0]),
            "binary64DerivedMinusSource": _rational(sums[2] - sums[0])},
        "storedCoordinateResidualsMeters": {
            "exactWeightsMinusSource": [_rational(a - b) for a, b in zip(ideal_point, source_point)],
            "binary64WeightsMinusSource": [_rational(a - b) for a, b in zip(binary_point, source_point)],
            "binary64MinusExactWeights": [_rational(a - b) for a, b in zip(binary_point, ideal_point)]}}


def build_binding_seam_remap_descriptor(source_unit, refinement_descriptor, *, policy=POLICY):
    """Derive five local approximate anchors, retaining every original row."""
    if type(policy) is not str or policy != POLICY:
        raise ValueError("Explicit supported coefficient-remap policy required")
    source_bytes, refinement_bytes = _encoded(source_unit), _encoded(refinement_descriptor)
    refinement = validate_binding_refinement(source_unit, refinement_descriptor)
    source = json.loads(source_bytes)
    original = refinement["originalLocalMesh"]
    final = refinement["finalLocalMesh"]
    if (len(final["verticesMeters"]) > MAX_REFINED_VERTICES
            or len(final["triangles"]) > MAX_REFINED_TRIANGLES):
        raise ValueError("Bounded two-crease remap geometry required")
    basis = _basis(refinement)
    source_rows = source["embeddedConstraints"]["constraints"]
    indices = [index for index, row in enumerate(source_rows)
               if any(term["instanceId"] == INSTANCE for term in row["terms"])]
    expected_group = next(group for group in refinement["sourceBinding"]["rowGroups"]
                          if group["registrationId"] == "bind_opening_left_left" and group["memberIndex"] == 1)
    if len(indices) != 5 or indices != expected_group["rowIndices"]:
        raise ValueError("Exactly five original left-binding source seam anchors required")
    rows = []
    for row_index in indices:
        row = source_rows[row_index]
        source_weights = [Fraction()] * len(original["verticesMeters"])
        for term in row["terms"]:
            if term["instanceId"] == INSTANCE:
                if term["coefficient"] >= 0:
                    raise ValueError("Original binding-side negative source anchor required")
                source_weights[term["vertex"]] = -Fraction(term["coefficient"])
        support = {index for index, value in enumerate(source_weights) if value != 0}
        parents = [index for index, face in enumerate(original["triangles"]) if support.issubset(face)]
        candidates = []
        for child, (face, parent) in enumerate(zip(final["triangles"], refinement["ultimateOriginalTriangleIndices"])):
            if parent not in parents:
                continue
            original_face = original["triangles"][parent]
            weights = _solve_three([[basis[vertex][index] for vertex in face] for index in original_face],
                                   [source_weights[index] for index in original_face])
            if any(sum((weight * basis[vertex][index] for vertex, weight in zip(face, weights)), Fraction()) != target
                   for index, target in enumerate(source_weights)):
                raise ValueError("Candidate leaves the original material coefficient support")
            candidates.append({"child": child, "parent": parent, "face": face, "weights": weights,
                               "nonnegative": all(weight >= 0 for weight in weights)})
        selected, equivalent = _equivalent_selection(candidates)
        fraction = Fraction(row["fraction"])
        row_id = "row:" + _digest([row["registrationId"], row["memberIndex"], fraction.numerator, fraction.denominator])
        rows.append({"rowIndex": row_index, "rowId": row_id, "registrationId": row["registrationId"],
            "memberIndex": row["memberIndex"], "sourceFraction": row["fraction"],
            "originalRowSha256": _digest(row), "originalRow": _copy(row),
            "anchorInstanceId": INSTANCE, "anchorSign": -1, "originalParentTriangles": parents,
            "originalWeights": [{"vertex": index, "weight": _rational(weight)}
                                for index, weight in enumerate(source_weights) if weight != 0],
            "candidates": [{"childTriangleIndex": item["child"], "sourceParentTriangle": item["parent"],
                "orientedVertices": item["face"], "coefficientNonnegative": item["nonnegative"],
                "exactWeights": [_rational(weight) for weight in item["weights"]]} for item in candidates],
            "selectedChildTriangleIndex": selected["child"], "equivalentChildTriangleIndices": equivalent,
            **_audit_selected(original["verticesMeters"], final["verticesMeters"], basis, source_weights, selected)})
    result = {"profile": PROFILE, "policy": POLICY, "accepted": False, "solverReady": False, "executable": False,
        "sourceProfile": source["profile"], "refinementProfile": refinement["profile"],
        "sourceUnitSha256": hashlib.sha256(source_bytes).hexdigest(),
        "refinementDescriptorSha256": hashlib.sha256(refinement_bytes).hexdigest(),
        "originalEmbeddedConstraintsSha256": _digest(source["embeddedConstraints"]),
        "instanceId": INSTANCE, "localRefinedVertexCount": len(final["verticesMeters"]),
        "rows": rows,
        "selectionPolicy": "Exhaust original-parent children in exact source coefficient space; exclude every negative weight. Multiple candidates must have identical exact sparse refined-vertex operators. The least equivalent child index is an anchor identity only, never a frame-side choice.",
        "conversionPolicy": "Once-round each nonnegative rational weight to nearest binary64, ties to even. No clamping or normalization; preserve positive support and existing finite [0,1], normalized-row admission bounds. Numerical weights are approximations with exactly rederived residuals, not adjustable tolerances.",
        "arithmeticScope": "Ideal rational anchors pull back exactly under the recorded coarse-to-refined lineage for arbitrary original nodal values. Binary64 anchors incur the reported coefficient residuals. Neither equality extends to arbitrary independent refined nodal values. Stored-coordinate residuals also include subdivision reconstruction error.",
        "scope": "Five source-bound approximate binding-side seam anchors only; original seam rows remain immutable. No canonical mesh, executable source profile, normal frame, gripper, placement, control or construction completion is supplied.",
        "pending": ["Versioned derived source unit, global index rebuilding and unchanged opposite-side/other-row binding.",
            "Separate explicit crease-side frame selection and captured original gripper rebinding.",
            "Independent derived-profile capture/replay validation and mechanical/contact checks before numerical execution."]}
    if _encoded(source_unit) != source_bytes or _encoded(refinement_descriptor) != refinement_bytes:
        raise ValueError("Source or refinement changed while deriving seam remap")
    return _copy(result)


def validate_binding_seam_remap_descriptor(source_unit, refinement_descriptor, descriptor, *, policy=POLICY):
    """Require complete strict rederivation, including every rounding residual."""
    claimed = _encoded(descriptor)
    expected = build_binding_seam_remap_descriptor(source_unit, refinement_descriptor, policy=policy)
    if claimed != _encoded(expected):
        raise ValueError("Binding seam remap descriptor differs from source rederivation")
    return expected
