import numpy as np

from solver_ipc_broad_phase import CONTACT_BROAD_PHASE_PROFILE, contact_broad_phase


def swept_plane_separated(first, second, normals, minimum_distance):
    first, second, normals = (np.asarray(value, dtype=float) for value in (first, second, normals))
    if (first.ndim != 3 or second.ndim != 3 or first.shape[2] != 3 or second.shape[2] != 3
            or first.shape[0] != second.shape[0] or not first.shape[1] or not second.shape[1]
            or normals.shape != (len(first), 3)
            or not np.isfinite(first).all() or not np.isfinite(second).all()
            or isinstance(minimum_distance, (bool, np.bool_)) or not np.isscalar(minimum_distance)
            or not np.isfinite(minimum_distance) or minimum_distance <= 0):
        raise ValueError("Finite swept primitive endpoints, matching directions and positive separation required")
    down = lambda value: np.nextafter(value, -np.inf)
    up = lambda value: np.nextafter(value, np.inf)
    origin = first[:, :1]

    def projections(points):
        differences = points - origin
        lower, upper = down(differences), up(differences)
        first_product, second_product = lower * normals[:, None], upper * normals[:, None]
        lower_product = down(np.minimum(first_product, second_product))
        upper_product = up(np.maximum(first_product, second_product))
        return (down(down(lower_product[:, :, 0] + lower_product[:, :, 1]) + lower_product[:, :, 2]),
                up(up(upper_product[:, :, 0] + upper_product[:, :, 1]) + upper_product[:, :, 2]))

    # Every affine primitive lies in its endpoint convex hull. Outward-rounded
    # support intervals prove separation for the entire motion, not sampled times.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        first_lower, first_upper = projections(first)
        second_lower, second_upper = projections(second)
        gap = np.maximum(down(first_lower.min(axis=1) - second_upper.max(axis=1)),
                         down(second_lower.min(axis=1) - first_upper.max(axis=1)))
        squares = up(normals * normals)
        norm_upper = up(np.sqrt(up(up(squares[:, 0] + squares[:, 1]) + squares[:, 2])))
        threshold = up(minimum_distance * norm_upper)
    return (np.isfinite(normals).all(axis=1) & np.isfinite(gap) & np.isfinite(threshold)
            & (norm_upper > 0) & (gap > threshold))


def certified_candidates(mesh, start, end, minimum_distance):
    import ipctk

    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if (start.shape != (mesh.num_vertices, 3) or end.shape != start.shape or mesh.dim != 3
            or not np.isfinite(start).all() or not np.isfinite(end).all()
            or isinstance(minimum_distance, (bool, np.bool_)) or not np.isscalar(minimum_distance)
            or not np.isfinite(minimum_distance) or minimum_distance <= 0):
        raise ValueError("Finite matching linear motion and positive separation required")
    candidates = ipctk.Candidates()
    candidates.build(mesh, start, end, inflation_radius=np.nextafter(minimum_distance / 2, np.inf),
                     broad_phase=contact_broad_phase())
    total, certified = len(candidates), 0
    edges, faces = np.asarray(mesh.edges), np.asarray(mesh.faces)
    for name in ("vv_candidates", "ev_candidates", "ee_candidates", "fv_candidates"):
        group = list(getattr(candidates, name))
        if not group:
            continue
        first_ids, second_ids, directions = [], [], []
        for candidate in group:
            if name == "vv_candidates":
                first_ids.append([candidate.vertex0_id])
                second_ids.append([candidate.vertex1_id])
            elif name == "ev_candidates":
                first_ids.append([candidate.vertex_id])
                second_ids.append(edges[candidate.edge_id])
            elif name == "ee_candidates":
                first_ids.append(edges[candidate.edge0_id])
                second_ids.append(edges[candidate.edge1_id])
            else:
                first_ids.append([candidate.vertex_id])
                second_ids.append(faces[candidate.face_id])
            directions.append(candidate.compute_distance_vector(candidate.dof(start, edges, faces)))
        first_ids, second_ids = np.asarray(first_ids), np.asarray(second_ids)
        first = np.concatenate((start[first_ids], end[first_ids]), axis=1)
        second = np.concatenate((start[second_ids], end[second_ids]), axis=1)
        separated = swept_plane_separated(first, second, directions, minimum_distance)
        certified += int(separated.sum())
        setattr(candidates, name, [candidate for candidate, safe in zip(group, separated) if not safe])
    return candidates, {"candidateCount": total, "certifiedCount": certified,
                        "remainingCount": len(candidates), "accepted": False,
                        "broadPhase": CONTACT_BROAD_PHASE_PROFILE}
