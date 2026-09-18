from numbers import Real

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def separate_rigid_instances(positions, faces, instance_offsets, clearance_m, max_translation_m=.25):
    points = np.asarray(positions, dtype=float)
    triangles = np.asarray(faces)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not 3 <= len(points) <= 25000 or np.max(np.abs(points)) > 100):
        raise ValueError("Finite bounded staging vertices required")
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu"
            or not 1 <= len(triangles) <= 50000 or np.any(triangles < 0)
            or np.any(triangles >= len(points))):
        raise ValueError("Bounded integer staging triangles required")
    for value in (clearance_m, max_translation_m):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or not 1e-8 <= value <= 10:
            raise ValueError("Staging distances must be finite metres between 1e-8 and 10")
    if not isinstance(instance_offsets, dict) or not 2 <= len(instance_offsets) <= 64:
        raise ValueError("Two to 64 physical instances required")
    if any(not isinstance(identity, str) or not identity or isinstance(offset, bool)
           or not isinstance(offset, int) or not 0 <= offset < len(points)
           for identity, offset in instance_offsets.items()):
        raise ValueError("Valid instance identities and integer offsets required")
    ordered = sorted(instance_offsets.items(), key=lambda entry: entry[1])
    starts = [entry[1] for entry in ordered]
    if starts[0] != 0 or len(set(starts)) != len(starts):
        raise ValueError("Instances must partition all staging vertices")
    ends = starts[1:] + [len(points)]
    owners = np.empty(len(points), dtype=int)
    normals, centers, pieces = [], [], []
    for index, (start, end) in enumerate(zip(starts, ends)):
        if end - start < 3:
            raise ValueError("Each instance requires at least three vertices")
        owners[start:end] = index
        piece = points[start:end]
        center = piece.mean(axis=0)
        _, singular, directions = np.linalg.svd(piece - center, full_matrices=False)
        if singular[1] <= 1e-12 or singular[2] > max(1e-12, singular[0] * 1e-10):
            raise ValueError("Rigid staging requires nondegenerate planar pieces")
        normal = directions[-1]
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        normals.append(normal)
        centers.append(center)
        pieces.append(piece)
    if np.any(owners[triangles] != owners[triangles[:, :1]]) or len(np.unique(triangles)) != len(points):
        raise ValueError("Triangles must cover vertices and stay within their physical instance")
    sorted_faces = np.sort(triangles, axis=1)
    face_edges = points[triangles[:, 1:]] - points[triangles[:, :1]]
    face_areas = np.linalg.norm(np.cross(face_edges[:, 0], face_edges[:, 1]), axis=1)
    edge_scales = np.max(np.linalg.norm(face_edges, axis=2), axis=1)
    if (np.any(np.diff(sorted_faces, axis=1) == 0) or len(np.unique(sorted_faces, axis=0)) != len(triangles)
            or np.any(face_areas <= 1e-12 * edge_scales ** 2)):
        raise ValueError("Unique nondegenerate staging triangles required")
    rows, bounds, pairs = [], [], []
    guard = max(1e-10, clearance_m * 1e-6)
    identities = sorted(range(len(pieces)), key=lambda index: ordered[index][0])
    for pair_index, first in enumerate(identities):
        for second in identities[pair_index + 1:]:
            candidates = []
            for direction in (normals[first], normals[second], *np.eye(3)):
                axis = direction.copy()
                if np.dot(centers[second] - centers[first], axis) < 0:
                    axis = -axis
                gap = float(np.min((pieces[second] - centers[first]) @ axis)
                            - np.max((pieces[first] - centers[first]) @ axis))
                candidates.append((gap, axis))
            gap, axis = max(candidates, key=lambda entry: entry[0])
            row = np.zeros((len(pieces), 3))
            row[first], row[second] = -axis, axis
            rows.append(row.ravel())
            bounds.append(clearance_m + guard - gap)
            pairs.append((first, second, axis))
    matrix, lower = np.asarray(rows), np.asarray(bounds)
    result = minimize(lambda values: .5 * np.dot(values, values), np.zeros(3 * len(pieces)),
                      jac=lambda values: values, method="SLSQP",
                      constraints=[LinearConstraint(matrix, lower, np.inf)],
                      options={"maxiter": 500, "ftol": 1e-12})
    translations = result.x.reshape((-1, 3))
    if not result.success or not np.isfinite(translations).all():
        raise ValueError("Rigid separation optimization did not converge")
    if np.max(np.linalg.norm(translations, axis=1)) > max_translation_m:
        raise ValueError("Rigid staging exceeds the declared translation budget")
    staged = points + translations[owners]
    gaps = [float(np.min((staged[starts[second]:ends[second]] - centers[first]) @ axis)
                  - np.max((staged[starts[first]:ends[first]] - centers[first]) @ axis))
            for first, second, axis in pairs]
    if not np.isfinite(staged).all() or min(gaps) < clearance_m:
        raise ValueError("Independent separating-plane clearance check failed")
    original_edges = points[triangles[:, 1:]] - points[triangles[:, :1]]
    staged_edges = staged[triangles[:, 1:]] - staged[triangles[:, :1]]
    error = float(np.max(np.abs(original_edges - staged_edges)))
    if error > 1e-12:
        raise ValueError("Rigid staging lost source edge precision")
    return staged, {"recipe": "experimental-minimum-translation-staging-v1", "accepted": False,
                    "clearanceM": clearance_m, "minimumSeparatingPlaneGapM": min(gaps),
                    "maxTranslationM": float(np.max(np.linalg.norm(translations, axis=1))),
                    "maxEdgeVectorErrorM": error, "pairCount": len(pairs),
                    "translationsM": {identity: translations[index].tolist()
                                      for index, (identity, _) in enumerate(ordered)},
                    "limitations": ["Static initialization only; movement from the original placement is not simulated.",
                                    "Fixed separating planes can reject feasible arrangements; no global packing claim.",
                                    "Layer order, turning, sewing thickness and assembly paths remain unresolved.",
                                    "Cross-instance separation does not certify within-instance contact."]}
