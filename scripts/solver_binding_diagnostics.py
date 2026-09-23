"""Independent binary64 geometry for a five-sample first binding turn.

No solver, force, contact, continuous-path or construction acceptance is
implemented here. Oriented geometric normals do not assign textile right sides.
"""

import hashlib
import json
import math

import numpy as np


PROFILE = "sampled-source-binding-geometry-v1"
_RELATIVE_FRAME_FLOOR = 1e-12


def _number(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, np.floating) and value.dtype.itemsize > 8):
        raise ValueError("Finite non-Boolean binary64 inputs required")
    try:
        result = float(value)
    except (ValueError, OverflowError, TypeError) as error:
        raise ValueError("Finite binary64 inputs required") from error
    if (not math.isfinite(result)
            or isinstance(value, (int, np.integer)) and int(value) != int(result)):
        raise ValueError("Finite exactly representable binary64 inputs required")
    return result


def _array(value, shape, *, indices=False):
    def visit(node, dimensions):
        if not dimensions:
            if indices:
                if isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)):
                    raise ValueError("Non-Boolean integer material indices required")
                return int(node)
            return _number(node)
        if (not isinstance(node, (tuple, list, np.ndarray))
                or isinstance(node, np.ndarray) and node.ndim == 0 or len(node) != dimensions[0]):
            raise ValueError("Exact bounded input shapes required")
        return [visit(item, dimensions[1:]) for item in node]
    try:
        return np.array(visit(value, shape), dtype=np.int64 if indices else np.float64)
    except (OverflowError, TypeError) as error:
        raise ValueError("Representable bounded input arrays required") from error


def _count(value, lower, upper, label):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or isinstance(value, np.ndarray) and value.ndim == 0
            or not lower <= len(value) <= upper):
        raise ValueError("Bounded " + label + " required")
    return len(value)


def _identity(value):
    return type(value) is str and 1 <= len(value) <= 128 and all(
        character.isascii() and (character.isalnum() or character in "_.:/-") for character in value)


def _triangle_geometry(points, faces):
    edges = np.stack((points[faces[:, 1]] - points[faces[:, 0]],
                      points[faces[:, 2]] - points[faces[:, 0]]), axis=2)
    area_vector = np.cross(edges[:, :, 0], edges[:, :, 1])
    twice_area = np.linalg.norm(area_vector, axis=1)
    scale = np.max(np.linalg.norm(edges, axis=1), axis=1)
    if (not np.isfinite(edges).all() or not np.isfinite(twice_area).all()
            or np.any(scale <= 0) or np.any(twice_area <= _RELATIVE_FRAME_FLOOR * scale ** 2)):
        raise ValueError("Undefined or ill-conditioned material triangle frame")
    return edges, twice_area, area_vector / twice_area[:, None]


def _rotation_angle(rotation):
    sine = .5 * np.linalg.norm([rotation[2, 1] - rotation[1, 2],
                                rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    cosine = np.clip(.5 * (np.trace(rotation) - 1), -1., 1.)
    return float(np.arctan2(sine, cosine))


def _fit(initial, current, weights):
    weights = weights / np.sum(weights)
    first = np.sum(weights[:, None] * initial, axis=0)
    second = np.sum(weights[:, None] * current, axis=0)
    a, b = initial - first, current - second
    covariance = (a * weights[:, None]).T @ b
    left, singular, right = np.linalg.svd(covariance)
    if singular[0] <= 0 or singular[1] <= _RELATIVE_FRAME_FLOOR * singular[0]:
        raise ValueError("Undefined or ill-conditioned proper rigid fit")
    correction = np.eye(3)
    correction[2, 2] = 1. if np.linalg.det(right.T @ left.T) >= 0 else -1.
    rotation = right.T @ correction @ left.T
    translation = second - rotation @ first
    errors = np.linalg.norm(b - a @ rotation.T, axis=1)
    angle = _rotation_angle(rotation)
    return rotation, translation, {"rotationMatrix": rotation.tolist(), "translationMeters": translation.tolist(),
        "rotationRadians": angle, "rotationDegrees": math.degrees(angle),
        "rmsResidualMeters": float(np.sqrt(np.sum(weights * errors ** 2))),
        "maximumResidualMeters": float(np.max(errors)), "vertexCount": len(initial),
        "secondSingularRatio": float(singular[1] / singular[0])}


def _frame(tangent, face_index, basis, deformation, normals):
    first, second, rest_normal = basis
    length = np.linalg.norm(tangent)
    if length <= 0 or abs(np.dot(tangent, rest_normal[face_index])) > _RELATIVE_FRAME_FLOOR * length:
        raise ValueError("Nonzero source tangent in the declared rest triangle required")
    components = np.array([np.dot(tangent, first[face_index]), np.dot(tangent, second[face_index])])
    current = deformation[face_index] @ components
    norm = np.linalg.norm(current)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Undefined transported material tangent")
    current /= norm
    normal = normals[face_index]
    return current, np.cross(normal, current), normal


def _roll(sleeve, binding):
    tangent, _, normal = sleeve
    other = binding[2] - np.dot(binding[2], tangent) * tangent
    norm = np.linalg.norm(other)
    if norm <= _RELATIVE_FRAME_FLOOR:
        raise ValueError("Binding normal has undefined projection about the sleeve tangent")
    other /= norm
    return float(np.arctan2(np.dot(tangent, np.cross(normal, other)), np.dot(normal, other)))


def _anchor(points, row, sign):
    terms = [(vertex, sign * coefficient) for vertex, coefficient in row if sign * coefficient > 0]
    return np.array([math.fsum(weight * float(points[vertex, axis]) for vertex, weight in terms)
                     for axis in range(3)])


def _measure(points, row, target, sleeve_frame, binding_frame):
    sleeve, binding = _anchor(points, row, 1), _anchor(points, row, -1)
    # Direct signed products avoid a second subtraction of separately rounded
    # anchor coordinates; source coefficients remain completely unchanged.
    difference = np.array([-math.fsum(weight * float(points[vertex, axis]) for vertex, weight in row)
                           for axis in range(3)])
    distance = float(np.linalg.norm(difference))
    if distance <= 0:
        raise ValueError("Undefined coincident active material anchors")
    tangent, cross_tangent, normal = sleeve_frame
    misalignment = float(np.arctan2(np.linalg.norm(np.cross(tangent, binding_frame[0])),
                                    np.dot(tangent, binding_frame[0])))
    return {"sleeveAnchorMeters": sleeve.tolist(), "bindingAnchorMeters": binding.tolist(),
        "anchorDistanceMeters": distance, "signedDistanceErrorMeters": distance - target,
        "absoluteDistanceErrorMeters": abs(distance - target),
        "signedNormalGapMeters": float(np.dot(normal, difference)),
        "tangentOffsetMeters": float(np.dot(tangent, difference)),
        "crossTangentOffsetMeters": float(np.dot(cross_tangent, difference)),
        "signedRollRadians": _roll(sleeve_frame, binding_frame),
        "tangentMisalignmentRadians": misalignment,
        "sleeveFrame": {"tangent": tangent.tolist(), "crossTangent": cross_tangent.tolist(), "normal": normal.tolist()},
        "bindingFrame": {"tangent": binding_frame[0].tolist(), "crossTangent": binding_frame[1].tolist(),
                         "normal": binding_frame[2].tolist()}}


def analyze_binding_diagnostic(rest, initial, positions, faces, *, instance_ranges,
        row_ids, rows, sleeve_instance_id, binding_instance_id, sleeve_frame_faces,
        binding_frame_faces, sleeve_tangents_rest, binding_tangents_rest, target_distances):
    """Measure five held source rows; input arrays and coefficients are unmodified.

    Frames must be exact oriented canonical triangles containing their signed
    supports. Positive row terms belong to sleeve, negative terms to binding.
    Per-instance ranges partition every vertex. Source correspondence itself
    remains the caller's independently validated responsibility.
    """
    n = _count(rest, 6, 25000, "source vertices")
    rest, initial, positions = [_array(value, (n, 3)) for value in (rest, initial, positions)]
    f = _count(faces, 2, 50000, "source triangles")
    faces = _array(faces, (f, 3), indices=True)
    if (np.any(faces < 0) or np.any(faces >= n) or len(np.unique(faces)) != n
            or any(len(set(face)) != 3 for face in faces)
            or len({tuple(sorted(face)) for face in faces}) != f):
        raise ValueError("Distinct nonduplicate canonical triangles must cover every vertex")
    if not isinstance(instance_ranges, dict) or not 2 <= len(instance_ranges) <= 64:
        raise ValueError("Bounded complete physical-instance partition required")
    ranges = {}
    for identity, bounds in instance_ranges.items():
        if not _identity(identity):
            raise ValueError("Bounded physical-instance identities required")
        lower, upper = _array(bounds, (2,), indices=True).tolist()
        if not 0 <= lower < upper <= n or upper - lower < 3:
            raise ValueError("Each physical instance requires at least three vertices")
        ranges[identity] = (lower, upper)
    owners = np.empty(n, dtype=int)
    cursor, identities = 0, []
    for identity, (lower, upper) in sorted(ranges.items(), key=lambda item: item[1][0]):
        if lower != cursor:
            raise ValueError("Instance ranges must partition vertices without overlap or omission")
        owners[lower:upper] = len(identities)
        identities.append(identity)
        cursor = upper
    if cursor != n or np.any(owners[faces] != owners[faces[:, :1]]):
        raise ValueError("Every canonical face must lie within its physical instance")
    if (not _identity(sleeve_instance_id) or not _identity(binding_instance_id)
            or sleeve_instance_id not in ranges or binding_instance_id not in ranges
            or sleeve_instance_id == binding_instance_id):
        raise ValueError("Distinct sleeve and binding source instances required")
    if (not isinstance(row_ids, (tuple, list)) or len(row_ids) != 5
            or not all(_identity(value) for value in row_ids) or len(set(row_ids)) != 5
            or not isinstance(rows, (tuple, list)) or len(rows) != 5):
        raise ValueError("Exactly five complete unique ordered source row identities required")
    targets = _array(target_distances, (5,))
    if np.any(targets <= 0):
        raise ValueError("Positive scalar source target distances required")
    frame_arrays = [_array(value, (5, 3), indices=True) for value in (sleeve_frame_faces, binding_frame_faces)]
    tangents = [_array(value, (5, 3)) for value in (sleeve_tangents_rest, binding_tangents_rest)]
    face_lookup = {tuple(face): index for index, face in enumerate(faces)}
    ordered_rows, frame_indices = [], [[], []]
    for index, row in enumerate(rows):
        if not isinstance(row, (tuple, list)) or not 2 <= len(row) <= 6:
            raise ValueError("Two to six signed source terms per row required")
        terms, seen = [], set()
        for term in row:
            if not isinstance(term, (tuple, list)) or len(term) != 2:
                raise ValueError("Vertex and coefficient source terms required")
            vertex = int(_array([term[0]], (1,), indices=True)[0])
            weight = _number(term[1])
            if vertex in seen or not 0 <= vertex < n or not 0 < abs(weight) <= 1:
                raise ValueError("Distinct source vertices and finite nonzero coefficients required")
            seen.add(vertex)
            terms.append((vertex, weight))
        for group, (sign, identity) in enumerate(((1, sleeve_instance_id), (-1, binding_instance_id))):
            support = {vertex for vertex, weight in terms if sign * weight > 0}
            total = math.fsum(sign * weight for _, weight in terms if sign * weight > 0)
            face = tuple(frame_arrays[group][index])
            lower, upper = ranges[identity]
            if (not 1 <= len(support) <= 3 or abs(total - 1.) > 1e-12
                    or face not in face_lookup or not support <= set(face)
                    or not all(lower <= vertex < upper for vertex in face)):
                raise ValueError("Oriented source frame must contain the normalized signed anchor on its instance")
            frame_indices[group].append(face_lookup[face])
        ordered_rows.append(terms)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            re, ra, rn = _triangle_geometry(rest, faces)
            ie, ia, inn = _triangle_geometry(initial, faces)
            ce, ca, cn = _triangle_geometry(positions, faces)
            first = re[:, :, 0] / np.linalg.norm(re[:, :, 0], axis=1)[:, None]
            second = np.cross(rn, first)
            basis = first, second, rn
            metric = np.zeros((f, 2, 2))
            metric[:, 0, 0] = np.linalg.norm(re[:, :, 0], axis=1)
            metric[:, 0, 1] = np.sum(re[:, :, 1] * first, axis=1)
            metric[:, 1, 1] = np.sum(re[:, :, 1] * second, axis=1)
            initial_deformation, deformation = [np.linalg.solve(metric.swapaxes(1, 2), edges.swapaxes(1, 2)).swapaxes(1, 2)
                                                for edges in (ie, ce)]
            measured = []
            for index, row in enumerate(ordered_rows):
                samples = []
                for state, transform, normals in ((initial, initial_deformation, inn), (positions, deformation, cn)):
                    frames = [_frame(tangents[group][index], frame_indices[group][index], basis, transform, normals)
                              for group in range(2)]
                    samples.append(_measure(state, row, targets[index], *frames))
                before, after = samples
                difference = after["signedRollRadians"] - before["signedRollRadians"]
                angle = math.atan2(math.sin(difference), math.cos(difference))
                measured.append({"rowIndex": index, "rowId": row_ids[index], "targetDistanceMeters": float(targets[index]),
                    "sleeveFrameFaceIndex": frame_indices[0][index], "bindingFrameFaceIndex": frame_indices[1][index],
                    "initial": before, "current": after, "relativeTurnRadians": angle,
                    "relativeTurnDegrees": math.degrees(angle),
                    "offsetChangesMeters": {key: after[key] - before[key] for key in
                        ("signedNormalGapMeters", "tangentOffsetMeters", "crossTangentOffsetMeters")}})
            strain = {}
            for owner, identity in enumerate(identities):
                selection = owners[faces[:, 0]] == owner
                summaries = {}
                for label, transform, area in (("initial", initial_deformation, ia), ("current", deformation, ca)):
                    stretches = np.linalg.svd(transform[selection], compute_uv=False)
                    ratios = area[selection] / ra[selection]
                    summaries[label] = {"principalStretchMinimum": float(stretches.min()),
                        "principalStretchMaximum": float(stretches.max()), "areaRatioMinimum": float(ratios.min()),
                        "areaRatioMaximum": float(ratios.max()),
                        "maximumAbsolutePrincipalStrain": float(np.max(abs(stretches - 1.)))}
                strain[identity] = {"triangleCount": int(selection.sum()), **summaries}
            weights = np.zeros(n)
            np.add.at(weights, faces.ravel(), np.repeat(ra / 6, 3))
            fits, fit_arrays = {}, []
            for identity in (sleeve_instance_id, binding_instance_id):
                lower, upper = ranges[identity]
                rotation, translation, data = _fit(initial[lower:upper], positions[lower:upper], weights[lower:upper])
                fits[identity] = data
                fit_arrays.append((rotation, translation))
            (sr, st), (br, bt) = fit_arrays
            relative = sr.T @ br
            angle = _rotation_angle(relative)
            relative_fit = {"rotationMatrix": relative.tolist(), "translationMeters": (sr.T @ (bt - st)).tolist(),
                            "rotationRadians": angle, "rotationDegrees": math.degrees(angle),
                            "convention": "inverse(sleeve initial-to-current fit) composed with binding initial-to-current fit"}
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Undefined or nonfinite binary64 binding geometry") from error
    payload = {"rest": rest.tolist(), "initial": initial.tolist(), "positions": positions.tolist(), "faces": faces.tolist(),
        "instanceRanges": ranges, "rowIds": list(row_ids), "rows": ordered_rows,
        "sleeveInstanceId": sleeve_instance_id, "bindingInstanceId": binding_instance_id,
        "sleeveFrameFaces": frame_arrays[0].tolist(), "bindingFrameFaces": frame_arrays[1].tolist(),
        "sleeveTangentsRest": tangents[0].tolist(), "bindingTangentsRest": tangents[1].tolist(), "targets": targets.tolist()}
    result = {"profile": PROFILE, "accepted": False,
        "inputSha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "rows": measured, "rigidFits": fits, "relativeRigidMotion": relative_fit, "strainByInstance": strain,
        "rigidFitWeighting": "Unchanged source triangle area divided equally among its vertices; normalized within each fitted instance",
        "limitations": ["Binary64 saved-state diagnostic only; no continuous path or contact certificate.",
            "Caller must validate original source correspondence, complete rows, frame identities and declared material-side policy.",
            "Normals follow explicit canonical triangle winding; no textile right-side assignment is inferred.",
            "Relative local angles use projected binding normals and principal initial-subtracted angles in [-pi, pi]; no temporal unwrapping is inferred.",
            "Kabsch fits summarize instance motion and can hide local deformation; inspect local rows and fit residuals together.",
            "Five sampled correspondences do not establish spatial seam coverage, finite-thickness stitch behavior, wrapping, stitch-down, apex treatment or construction acceptance."]}
    # This also catches infinities escaping an operation that did not raise a
    # NumPy floating-point exception. No NaN placeholders are valid evidence.
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, OverflowError) as error:
        raise ValueError("Nonfinite binding diagnostics") from error
    return result
