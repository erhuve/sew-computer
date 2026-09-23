"""Unaccepted lineage descriptor for two source-line binding subdivisions.

This is preprocessing evidence, not an executable canonical source profile.
It neither remaps mechanical rows/tools nor creates a construction phase.
"""

from collections import Counter
from fractions import Fraction
import hashlib
import json

from solver_crease_mesh import split_straight_crease
from solver_cuff_source_binding import validate_cuff_source_binding


PROFILE = "source-left-binding-refinement-descriptor-v1"
INSTANCE = "opening_binding_left_left:shell"
TEMPLATE = "opening_binding_left_left"
MAX_DESCRIPTOR_BYTES = 4 * 1024 ** 2
_CONTROLS = {"placedMeters", "bindingFirstTurnDiagnostic", "sewingActuation", "gripperActuation",
             "foldActuation", "assemblySchedule", "sewingFrames", "bindingRefinement"}


def _encoded(value):
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("Finite bounded JSON refinement data required") from error
    if len(result) > MAX_DESCRIPTOR_BYTES:
        raise ValueError("Refinement descriptor byte budget exceeded")
    return result


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _copy(value):
    return json.loads(_encoded(value))


def _rational(value):
    """Expose the exact arithmetic value and its rounded reporting value."""
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "roundedBinary64": float(value)}


def _weights(raw):
    return [{int(vertex): Fraction(float(weight)) for vertex, weight in entry.items()}
            for entry in raw]


def _residuals(vertices, original, weights):
    values, maximum = [], Fraction()
    for vertex, (point, support) in enumerate(zip(vertices, weights)):
        differences = [Fraction(float(point[axis])) - sum(
            (weight * Fraction(float(original[index][axis])) for index, weight in support.items()), Fraction())
            for axis in range(2)]
        maximum = max(maximum, *(abs(value) for value in differences))
        values.append({"vertex": vertex, "storedMinusReconstructedMeters": [_rational(value) for value in differences],
                       "weightSumMinusOne": _rational(sum(support.values(), Fraction()) - 1)})
    return {"vertices": values, "maximumAbsoluteCoordinateResidualMeters": _rational(maximum)}


def build_binding_refinement(source):
    """Rederive a bounded left-strip descriptor without changing its v1 base."""
    original_bytes = _encoded(source)
    binding = validate_cuff_source_binding(source)
    if (source["side"] != "left" or _CONTROLS.intersection(source)
            or any(instance["mirrorX"] is not False for instance in source["instances"])):
        raise ValueError("Unplaced, uncontrolled, unmirrored left v1 source unit required")
    mesh = source["sourceTemplates"][TEMPLATE]
    if len(mesh["restPositions"]) != 11 or len(mesh["triangles"]) != 12:
        raise ValueError("This bounded descriptor requires the original eleven-vertex/twelve-face binding")
    panel = next(panel for panel in source["sourcePattern"]["panels"] if panel["id"] == TEMPLATE)
    offset = source["instanceOffsets"][INSTANCE]
    original = [point[:2] for point in source["restMeters"][offset:offset + 11]]
    original_faces = _copy(mesh["triangles"])
    vertices, faces, stages = original, original_faces, []
    for name, expected_start, expected_end, expected_x in (("right", 1, 2, .020), ("left", 3, 4, 0.)):
        edge = next(edge for edge in panel["draft"]["edges"] if edge["name"] == name)
        if (edge["start"], edge["end"]) != (expected_start, expected_end):
            raise ValueError("Named source binding path endpoints differ from the bounded layout")
        line = [[coordinate * .001 for coordinate in panel["points"][index]]
                for index in (edge["start"], edge["end"])]
        if line[0][0] != expected_x or line[1][0] != expected_x or line[0][1] == line[1][1]:
            raise ValueError("Expected named straight right/x20 and left/x0 source paths")
        before = {"verticesMeters": vertices, "triangles": faces}
        result = _copy(split_straight_crease(vertices, faces, line))
        stages.append({"sourcePathName": name, "sourceStart": edge["start"], "sourceEnd": edge["end"],
            "sourceLineEndpointsMeters": line, "sourcePathSha256": _digest(edge),
            "inputMeshSha256": _digest(before), "output": result,
            "binaryInputReconstruction": _residuals(result["vertices"], vertices, _weights(result["sourceWeights"]))})
        vertices, faces = result["vertices"], result["triangles"]
    if vertices[:11] != original:
        raise ValueError("Original source vertex prefix changed during subdivision")
    first, second = (stage["output"] for stage in stages)
    first_weights, second_weights = _weights(first["sourceWeights"]), _weights(second["sourceWeights"])
    composed = []
    for support in second_weights:
        result = {}
        for intermediate, weight in support.items():
            for original_vertex, original_weight in first_weights[intermediate].items():
                result[original_vertex] = result.get(original_vertex, Fraction()) + weight * original_weight
        composed.append({index: weight for index, weight in sorted(result.items()) if weight != 0})
    parents = [first["parentTriangles"][intermediate] for intermediate in second["parentTriangles"]]
    for face, parent in zip(faces, parents):
        if not all(set(composed[vertex]).issubset(original_faces[parent]) for vertex in face):
            raise ValueError("Refined child leaves its original parent material support")
    edges = Counter(tuple(sorted((face[index], face[(index + 1) % 3])))
                    for face in faces for index in range(3))
    creases = []
    for stage in stages:
        output = stage["output"]
        chain, segments = output["creaseVertices"], output["creaseEdges"]
        if (any(edges[tuple(segment)] != 2 for segment in segments)
                or set(map(tuple, segments)) != {tuple(sorted(pair)) for pair in zip(chain[:-1], chain[1:])}
                or any(not any(vertex in edge and count == 1 for edge, count in edges.items())
                       for vertex in (chain[0], chain[-1]))):
            raise ValueError("Both source-line crease chains must survive to the final cut boundary")
        creases.append({"sourcePathName": stage["sourcePathName"],
            "sourceLineEndpointsMeters": stage["sourceLineEndpointsMeters"],
            "vertices": chain, "edges": segments,
            "cutBoundaryEndpointsMeters": [vertices[chain[0]], vertices[chain[-1]]]})
    other_meshes = {}
    for instance in source["instances"]:
        if instance["id"] == INSTANCE:
            continue
        template = source["sourceTemplates"][instance["templateId"]]
        start = source["instanceOffsets"][instance["id"]]
        count = len(template["restPositions"])
        other_meshes[instance["id"]] = _digest({"restMeters": source["restMeters"][start:start + count],
                                                "triangles": template["triangles"]})
    descriptor = {"profile": PROFILE, "accepted": False, "solverReady": False, "executable": False,
        "scope": "Unaccepted deterministic preprocessing lineage only; original source rows, frames, grippers and numerical integration remain unmodified and unimplemented here.",
        "sourceUnitSha256": hashlib.sha256(original_bytes).hexdigest(), "sourceBinding": binding,
        "instanceId": INSTANCE, "sourceTemplateId": TEMPLATE, "units": "m",
        "originalTemplateSha256": _digest(mesh), "originalEmbeddedConstraintsSha256": _digest(source["embeddedConstraints"]),
        "unchangedOtherInstanceLocalMeshSha256": other_meshes,
        "originalLocalMesh": {"verticesMeters": original, "triangles": original_faces},
        "cutPolicy": {"id": "named-right-then-left-infinite-source-lines-v1", "sourcePathOrder": ["right", "left"],
            "extension": "Extend each named straight source path as an infinite line through the full existing cut domain, including end allowances; no trimming or new boundary.",
            "status": "Explicit research crease choice, not an executed fold or an inferred source construction milestone."},
        "stages": stages, "finalLocalMesh": {"verticesMeters": vertices, "triangles": faces},
        "ultimateOriginalTriangleIndices": parents,
        "composedVertexWeights": [{"vertex": vertex, "weights": [
            {"sourceVertex": index, "weight": _rational(weight)} for index, weight in support.items()]}
            for vertex, support in enumerate(composed)],
        "binaryInputReconstruction": _residuals(vertices, original, composed),
        "finalCreases": creases,
        "counts": {"originalStripVertices": 11, "originalStripTriangles": 12,
            "refinedStripVertices": len(vertices), "refinedStripTriangles": len(faces),
            "prospectiveUnitVertices": len(source["restMeters"]) + len(vertices) - 11,
            "prospectiveUnitTriangles": len(source["triangles"]) // 3 + len(faces) - 12,
            "unchangedSourceRows": len(source["embeddedConstraints"]["constraints"]), "fabricInstances": len(source["instances"])},
        "reconstructionPolicy": {"id": "exact-binary-input-composition-no-renormalization-v1",
            "arithmetic": "Fractions compose the actual binary64 stage weights without clamping or normalization; rational residuals compare stored binary64 coordinates with that exact binary-input interpolation. Rounded values are reporting only.",
            "admission": "Rederive both deterministic split stages and every lineage/residual field from the validated base; require strict canonical JSON equality. Residual metadata is not an independently adjustable acceptance tolerance.",
            "domainScope": "The existing splitter uses tolerance-based line classification and Shapely parent-union checks. This descriptor does not prove exact-real domain preservation or supply an exact geometric predicate."},
        "pending": ["Source-anchor numerical remapping with coefficient-pullback audit for the five binding-side rows.",
            "Explicit crease-side source-frame selection and independent gripper material-point rebinding.",
            "Global instance/index rebuilding, executable source-profile validation, source capture and replay integration.",
            "Mechanical/contact validation of the refined discretization; no phase, fold, wrap, stitch-down or apex completion."]}
    if _encoded(source) != original_bytes:
        raise ValueError("Source unit changed while deriving refinement")
    return _copy(descriptor)


def validate_binding_refinement(source, descriptor):
    """Reject even repaired-hash substitutions by strict complete rederivation."""
    claimed = _encoded(descriptor)
    expected = build_binding_refinement(source)
    if claimed != _encoded(expected):
        raise ValueError("Binding refinement descriptor differs from source rederivation")
    return expected
