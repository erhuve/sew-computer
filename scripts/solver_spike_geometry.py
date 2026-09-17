import numpy as np


def snapshot_particle_positions(state):
    return state.particle_q.numpy().copy()


def state_finiteness(positions, velocities):
    positions, velocities = np.asarray(positions), np.asarray(velocities)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0 or velocities.shape != positions.shape:
        raise ValueError("Expected matching nonempty particle position and velocity arrays")
    invalid_positions = int(np.count_nonzero(~np.isfinite(positions)))
    invalid_velocities = int(np.count_nonzero(~np.isfinite(velocities)))
    return {"finite": invalid_positions == 0 and invalid_velocities == 0,
            "nonfinitePositionValues": invalid_positions, "nonfiniteVelocityValues": invalid_velocities}


def segment_triangle(start, end, triangle, tolerance=1e-9):
    direction = end - start
    edge_first = triangle[1] - triangle[0]
    edge_second = triangle[2] - triangle[0]
    cross = np.cross(direction, edge_second)
    determinant = np.dot(edge_first, cross)
    determinant_scale = np.linalg.norm(direction) * np.linalg.norm(edge_first) * np.linalg.norm(edge_second)
    if determinant_scale == 0 or abs(determinant) <= tolerance * determinant_scale:
        return None
    inverse = 1.0 / determinant
    offset = start - triangle[0]
    bary_first = np.dot(offset, cross) * inverse
    cross_offset = np.cross(offset, edge_first)
    bary_second = np.dot(direction, cross_offset) * inverse
    distance = np.dot(edge_second, cross_offset) * inverse
    if min(bary_first, bary_second, 1 - bary_first - bary_second, distance, 1 - distance) < -tolerance:
        return None
    return start + direction * distance


def coplanar_overlap(first, second, normal, tolerance=1e-9):
    drop = int(np.argmax(np.abs(normal)))
    origin = first[0]
    polygon = [np.delete(point - origin, drop) for point in first]
    clip = np.delete(second - origin, drop, axis=1)
    length_scale = max(np.linalg.norm(triangle[(edge + 1) % 3] - triangle[edge]) for triangle in (first, second) for edge in range(3))
    area_tolerance = tolerance * length_scale**2
    cross = lambda left, right: left[0] * right[1] - left[1] * right[0]
    direction = np.sign(cross(clip[1] - clip[0], clip[2] - clip[0]))
    for edge in range(3):
        start, end = clip[edge], clip[(edge + 1) % 3]
        inside = lambda point: direction * cross(end - start, point - start)
        previous = polygon[-1] if polygon else None
        result = []
        for point in polygon:
            previous_side, side = inside(previous), inside(point)
            if (side >= -area_tolerance) != (previous_side >= -area_tolerance):
                result.append(previous + (point - previous) * previous_side / (previous_side - side))
            if side >= -area_tolerance:
                result.append(point)
            previous = point
        polygon = result
    area = abs(sum(cross(polygon[index], polygon[(index + 1) % len(polygon)]) for index in range(len(polygon))) / 2)
    return area > area_tolerance


def triangles_intersect(first, second, tolerance=1e-9):
    length_scale = max(np.linalg.norm(triangle[(edge + 1) % 3] - triangle[edge]) for triangle in (first, second) for edge in range(3))
    distance_tolerance = tolerance * length_scale
    if np.any(first.max(axis=0) < second.min(axis=0) - distance_tolerance) or np.any(second.max(axis=0) < first.min(axis=0) - distance_tolerance):
        return False
    normal = np.cross(first[1] - first[0], first[2] - first[0])
    norm = np.linalg.norm(normal)
    if norm <= tolerance * np.linalg.norm(first[1] - first[0]) * np.linalg.norm(first[2] - first[0]):
        raise ValueError("Degenerate triangle in collision oracle")
    if np.max(np.abs((second - first[0]) @ (normal / norm))) <= distance_tolerance:
        return coplanar_overlap(first, second, normal, tolerance)
    for left, right in ((first, second), (second, first)):
        for index in range(3):
            if segment_triangle(left[index], left[(index + 1) % 3], right, tolerance) is not None:
                return True
    return False


def surface_intersections(positions, triangles, candidate_budget=2000000):
    if type(candidate_budget) is not int or not 1 <= candidate_budget <= 2000000:
        raise ValueError("Invalid surface intersection budget")
    positions = np.asarray(positions, dtype=float)
    faces = np.asarray(triangles)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("Finite surface positions required")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= len(positions)) or len(faces) > 50000:
        raise ValueError("Bounded indexed surface triangles required")
    vertices = positions[faces]
    lengths = np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1) * np.linalg.norm(vertices[:, 2] - vertices[:, 0], axis=1)
    areas = np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1)
    if np.any(areas <= lengths * 1e-9):
        raise ValueError("Degenerate surface triangle")
    lower, upper = vertices.min(axis=1), vertices.max(axis=1)
    padding = max(float(np.linalg.norm(upper - lower, axis=1).max()) * 1e-9, 1e-12)
    order = np.argsort(lower[:, 0])
    sorted_lower = lower[order, 0]
    count, tested, broad_candidates, examples = 0, 0, 0, []
    for first_index, first in enumerate(faces):
        stop = np.searchsorted(sorted_lower, upper[first_index, 0] + padding, side="right")
        potential = order[:stop]
        broad_candidates += len(potential)
        if broad_candidates > candidate_budget * 8:
            raise ValueError("Surface broad-phase candidate budget exceeded")
        potential = potential[potential > first_index]
        candidates = potential[np.all(upper[first_index] + padding >= lower[potential], axis=1) &
                               np.all(upper[potential] + padding >= lower[first_index], axis=1)]
        for second_index in candidates:
            if set(first) & set(faces[second_index]):
                continue
            tested += 1
            if tested > candidate_budget:
                raise ValueError("Surface intersection candidate budget exceeded")
            if triangles_intersect(vertices[first_index], vertices[second_index]):
                count += 1
                if len(examples) < 30:
                    examples.append([first_index, int(second_index)])
    return {"intersectingPairCount": count, "testedCandidates": tested, "broadPhaseCandidates": broad_candidates, "intersectingPairs": examples,
            "scope": "Final nonadjacent surfaces; noncoplanar touches may count; coplanar boundary-only contact omitted; no swept or thickness test"}


def inspect_geometry(rest, placed, positions, triangles, mappings, seam_paths):
    positions = np.asarray(positions)
    rest = np.asarray(rest)
    faces = np.asarray(triangles).reshape((-1, 3))
    source_normals = np.cross(rest[faces[:, 1]] - rest[faces[:, 0]], rest[faces[:, 2]] - rest[faces[:, 0]])
    normals = np.cross(positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]])
    area_ratios = np.linalg.norm(normals, axis=1) / np.linalg.norm(source_normals, axis=1)
    intersections = []
    tested = 0
    for first_index, first in enumerate(faces):
        for second_index in range(first_index + 1, len(faces)):
            second = faces[second_index]
            if set(first) & set(second):
                continue
            tested += 1
            if triangles_intersect(positions[first], positions[second]):
                intersections.append([first_index, second_index])
    seam_lengths = []
    for side in (0, 1):
        vertices = seam_paths[side]
        rest_length = sum(np.linalg.norm(rest[right] - rest[left]) for left, right in zip(vertices, vertices[1:]))
        deformed_length = sum(np.linalg.norm(positions[right] - positions[left]) for left, right in zip(vertices, vertices[1:]))
        seam_lengths.append({"side": side, "restLengthMm": float(rest_length * 1000), "deformedLengthMm": float(deformed_length * 1000), "ratio": float(deformed_length / rest_length)})
    panel_normals = []
    for panel_id in (0, 1):
        selected = [index for index, face in enumerate(faces) if mappings[face[0]]["panel"] == panel_id]
        mean = normals[selected].sum(axis=0)
        panel_normals.append(mean / np.linalg.norm(mean))
    return {"oracle": "all nonadjacent triangle pairs; segment-triangle plus positive-area coplanar clipping; no seam exclusions", "testedPairs": tested, "intersectingPairCount": len(intersections), "intersectingPairs": intersections[:30], "deformedAreaRatioMin": float(area_ratios.min()), "deformedAreaRatioMax": float(area_ratios.max()), "collapsedTriangleCount": int(np.sum(np.linalg.norm(normals, axis=1) < 1e-12)), "meanPanelNormalAngleDegrees": float(np.degrees(np.arccos(np.clip(np.dot(*panel_normals), -1, 1)))), "seamPolylineLengths": seam_lengths, "limitations": ["Discrete final surface only; no swept collision or nonzero-thickness penetration proof", "Topologically adjacent triangle pairs omitted", "Mean normal angle is diagnostic, not a layer-turning certificate"]}
