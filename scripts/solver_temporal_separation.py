from numbers import Real

import numpy as np


_GROUPS = ("vv_candidates", "ev_candidates", "ee_candidates", "fv_candidates")


def _positive_distance(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0 or float(value) != value):
        raise ValueError("Positive finite float64-exact separation required")
    return float(value)


def _motion(mesh, start, end, minimum_distance):
    minimum_distance = _positive_distance(minimum_distance)
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if (start.shape != (mesh.num_vertices, 3) or end.shape != start.shape
            or not len(start) or mesh.dim != 3
            or not np.isfinite(start).all() or not np.isfinite(end).all()):
        raise ValueError("Finite three-dimensional motion matching the collision mesh required")
    return start, end, minimum_distance


def temporal_support_bounds(first_start, second_start, first_end, second_end,
                            normals, start_fraction, end_fraction, minimum_distance):
    minimum_distance = _positive_distance(minimum_distance)
    a0, b0, a1, b1, normals = (np.asarray(value, dtype=float) for value in
                              (first_start, second_start, first_end, second_end, normals))
    if (a0.ndim != 3 or b0.ndim != 3 or a0.shape[2] != 3 or b0.shape[2] != 3
            or a0.shape != a1.shape or b0.shape != b1.shape or a0.shape[0] != b0.shape[0]
            or not a0.shape[1] or not b0.shape[1] or normals.shape != (len(a0), 3)
            or not all(np.isfinite(value).all() for value in (a0, b0, a1, b1))
            or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, Real)
                   or not np.isfinite(t) or float(t) != t for t in (start_fraction, end_fraction))
            or not 0 <= start_fraction < end_fraction <= 1):
        raise ValueError("Matching finite primitive endpoints and an ordered unit interval required")
    down = lambda value: np.nextafter(value, -np.inf)
    up = lambda value: np.nextafter(value, np.inf)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        delta0 = a0[:, :, None, :] - b0[:, None, :, :]
        delta1 = a1[:, :, None, :] - b1[:, None, :, :]
        lo0, hi0, lo1, hi1 = down(delta0), up(delta0), down(delta1), up(delta1)
        direction = normals[:, None, None, :]
        lows, highs = [], []
        for t in (float(start_fraction), float(end_fraction)):
            # Enclose exact original affine motion, not rounded subinterval vertices.
            weight_lo, weight_hi = max(0., down(1. - t)), min(1., up(1. - t))
            left_lo = down(np.minimum(weight_lo * lo0, weight_hi * lo0))
            left_hi = up(np.maximum(weight_lo * hi0, weight_hi * hi0))
            lower = down(left_lo + down(t * lo1))
            upper = up(left_hi + up(t * hi1))
            p, q = lower * direction, upper * direction
            p, q = down(np.minimum(p, q)), up(np.maximum(p, q))
            lows.append(down(down(p[..., 0] + p[..., 1]) + p[..., 2]))
            highs.append(up(up(q[..., 0] + q[..., 1]) + q[..., 2]))
        # One orientation must separate every pair at both times. Affinity and
        # convexity then prove the whole interval, including a moving separator.
        gap = np.maximum(np.minimum(lows[0].min(axis=(1, 2)), lows[1].min(axis=(1, 2))),
                         np.minimum(-highs[0].max(axis=(1, 2)), -highs[1].max(axis=(1, 2))))
        squared = up(normals * normals)
        norm = up(np.sqrt(up(up(squared[:, 0] + squared[:, 1]) + squared[:, 2])))
        threshold = up(minimum_distance * norm)
        bound = down(gap / norm)
    safe = (np.isfinite(normals).all(axis=1) & np.isfinite(gap) & np.isfinite(norm)
            & np.isfinite(threshold) & np.isfinite(bound) & (norm > 0) & (gap > threshold))
    return safe, bound


def _primitive_ids(name, candidate, edges, faces):
    if name == "vv_candidates":
        return [candidate.vertex0_id], [candidate.vertex1_id]
    if name == "ev_candidates":
        return [candidate.vertex_id], edges[candidate.edge_id].tolist()
    if name == "ee_candidates":
        return edges[candidate.edge0_id].tolist(), edges[candidate.edge1_id].tolist()
    return [candidate.vertex_id], faces[candidate.face_id].tolist()


def certify_linear_path(mesh, start, end, minimum_distance, *, max_depth=12,
                        max_nodes=100000, keep_leaves=False):
    import ipctk

    start, end, minimum_distance = _motion(mesh, start, end, minimum_distance)
    if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in (max_depth, max_nodes)) or not 0 <= max_depth <= 30
            or not 1 <= max_nodes <= 1000000 or not isinstance(keep_leaves, bool)):
        raise ValueError("Bounded integer temporal depth/node budgets required")
    candidates = ipctk.Candidates()
    candidates.build(mesh, start, end, inflation_radius=np.nextafter(minimum_distance / 2, np.inf))
    edges, faces = np.asarray(mesh.edges), np.asarray(mesh.faces)
    total = len(candidates)
    report = {"method": "outward-rounded-temporal-separation-v1", "safe": False,
              "candidateCount": total, "certifiedCount": 0, "unresolvedCount": total,
              "nodeCount": 0, "leafCount": 0, "deepest": 0, "lowerBoundM": None,
              "maxDepth": int(max_depth), "maxNodes": int(max_nodes),
              "reason": "unresolved", "accepted": False}
    if keep_leaves:
        report["certificateLeaves"] = []

    def record(name, index, ids, normal, lo, hi, depth, bound):
        report["leafCount"] += 1
        report["deepest"] = max(report["deepest"], depth)
        report["lowerBoundM"] = (float(bound) if report["lowerBoundM"] is None
                                 else min(report["lowerBoundM"], float(bound)))
        if keep_leaves:
            report["certificateLeaves"].append({"group": name, "candidate": int(index),
                "first": ids[0], "second": ids[1], "normal": np.asarray(normal).tolist(),
                "t0": lo, "t1": hi, "lowerBoundM": float(bound)})

    if total > max_nodes:
        report["reason"] = "node-budget-exhausted"
        return report
    for name in _GROUPS:
        group = list(getattr(candidates, name))
        if not group:
            continue
        ids = [_primitive_ids(name, candidate, edges, faces) for candidate in group]
        ia, ib = np.asarray([row[0] for row in ids]), np.asarray([row[1] for row in ids])
        a0, b0, a1, b1 = start[ia], start[ib], end[ia], end[ib]
        normals = np.asarray([candidate.compute_distance_vector(candidate.dof(start, edges, faces))
                              for candidate in group])
        if report["nodeCount"] + len(group) > max_nodes:
            report["reason"] = "node-budget-exhausted"
            return report
        report["nodeCount"] += len(group)
        safe, bounds = temporal_support_bounds(a0, b0, a1, b1, normals, 0., 1., minimum_distance)
        for i in np.flatnonzero(safe):
            record(name, i, ids[i], normals[i], 0., 1., 0, bounds[i])
        report["certifiedCount"] += int(safe.sum())
        report["unresolvedCount"] = total - report["certifiedCount"]
        for i in np.flatnonzero(~safe):
            pending, proven = [(0., 1., 0)], True
            while pending:
                if report["nodeCount"] == max_nodes:
                    report["reason"] = "node-budget-exhausted"
                    return report
                lo, hi, depth = pending.pop()
                report["nodeCount"] += 1
                report["deepest"] = max(report["deepest"], depth)
                separated = False
                for t in ((lo + hi) / 2, lo, hi):
                    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                        y = (1. - t) * start + t * end
                    if not np.isfinite(y).all():
                        continue
                    normal = group[i].compute_distance_vector(group[i].dof(y, edges, faces))
                    check, bound = temporal_support_bounds(a0[i:i+1], b0[i:i+1], a1[i:i+1],
                        b1[i:i+1], [normal], lo, hi, minimum_distance)
                    if check[0]:
                        record(name, i, ids[i], normal, lo, hi, depth, bound[0])
                        separated = True
                        break
                if separated:
                    continue
                if depth == max_depth:
                    proven = False
                    break
                mid = (lo + hi) / 2
                pending.extend([(mid, hi, depth + 1), (lo, mid, depth + 1)])
            if proven:
                report["certifiedCount"] += 1
                report["unresolvedCount"] = total - report["certifiedCount"]
    report["safe"] = report["unresolvedCount"] == 0
    report["reason"] = "certified" if report["safe"] else "unresolved"
    return report
