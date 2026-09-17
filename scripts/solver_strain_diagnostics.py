import numpy as np


def mass_motion_report(initial, positions, velocities, masses):
    initial, positions, velocities, masses = [np.asarray(values, dtype=float) for values in (initial, positions, velocities, masses)]
    if initial.ndim != 2 or initial.shape[1] != 3 or positions.shape != initial.shape or velocities.shape != initial.shape or masses.shape != (len(initial),):
        raise ValueError("Matching particle states and masses required")
    if any(not np.isfinite(values).all() for values in (initial, positions, velocities, masses)) or np.any(masses < 0) or masses.sum() <= 0:
        raise ValueError("Finite states and nonnegative masses with positive total required")
    total = float(masses.sum())
    displacement = np.sum(masses[:, None] * (positions - initial), axis=0) / total
    momentum = np.sum(masses[:, None] * velocities, axis=0)
    return {"scope": "Dynamic-particle mass motion; pins or external forces invalidate isolated momentum conservation",
            "dynamicMassKg": total, "pinnedVertexCount": int(np.count_nonzero(masses == 0)),
            "centerOfMassDisplacementMeters": displacement.tolist(),
            "centerOfMassDisplacementMm": float(np.linalg.norm(displacement) * 1000),
            "linearMomentumKgMetersPerSecond": momentum.tolist(),
            "centerOfMassSpeedMetersPerSecond": float(np.linalg.norm(momentum) / total)}


def edge_strain_report(rest, positions, triangles, instance_ids, pinned_vertices=()):
    rest = np.asarray(rest, dtype=float)
    positions = np.asarray(positions, dtype=float)
    faces = np.asarray(triangles)
    if rest.ndim != 2 or rest.shape[1] != 3 or len(rest) == 0 or positions.shape != rest.shape:
        raise ValueError("Matching nonempty rest and current positions required")
    if not np.isfinite(rest).all() or not np.isfinite(positions).all():
        raise ValueError("Finite positions required for strain diagnostics")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= len(rest)):
        raise ValueError("Indexed triangles required for strain diagnostics")
    if len(instance_ids) != len(rest) or any(not isinstance(identity, str) or not identity for identity in instance_ids):
        raise ValueError("Physical identity required for every vertex")
    if any(type(vertex) is not int or not 0 <= vertex < len(rest) for vertex in pinned_vertices):
        raise ValueError("Invalid pinned vertex")
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    if np.any(lengths == 0):
        raise ValueError("Zero rest edge length")
    identities = np.asarray(instance_ids)
    if np.any(identities[edges[:, 0]] != identities[edges[:, 1]]):
        raise ValueError("Cloth triangles must remain within a physical piece")
    current = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
    ratios = current / lengths
    if not np.isfinite(ratios).all():
        raise ValueError("Nonfinite edge strain")
    per_instance = {}
    for identity in sorted(set(instance_ids)):
        selected = ratios[identities[edges[:, 0]] == identity]
        if not len(selected):
            raise ValueError("Physical piece has no cloth edges")
        per_instance[identity] = {"edgeCount": len(selected), "min": float(selected.min()),
                                  "max": float(selected.max()), "p95": float(np.percentile(selected, 95))}
    worst = []
    pins = set(pinned_vertices)
    for index in np.argsort(-ratios, kind="stable")[:10]:
        vertices = edges[index].tolist()
        worst.append({"vertices": vertices, "instanceId": instance_ids[vertices[0]],
                      "restLengthMm": float(lengths[index] * 1000), "currentLengthMm": float(current[index] * 1000),
                      "ratio": float(ratios[index]), "touchesPin": bool(pins.intersection(vertices))})
    return {"scope": "Unique cloth edges relative to unchanged rest positions; diagnostic only, not acceptance",
            "edgeCount": len(edges), "min": float(ratios.min()), "max": float(ratios.max()),
            "p95": float(np.percentile(ratios, 95)), "p99": float(np.percentile(ratios, 99)),
            "perInstance": per_instance, "worstEdges": worst}
