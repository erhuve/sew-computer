import heapq
import math

import numpy as np


def build_rest_neighbor_filters(rest_positions, triangles, newton_edges, instance_ids, margin,
                                max_vertices=50000, max_primitives=200000,
                                max_search_visits=2000000, max_filter_entries=2000000,
                                max_candidate_checks=4000000):
    rest = np.asarray(rest_positions, dtype=float)
    faces = np.asarray(triangles)
    edges = np.asarray(newton_edges)
    if rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all():
        raise ValueError("Finite immutable rest positions with shape (N, 3) required")
    if not 0 < len(rest) <= max_vertices or len(instance_ids) != len(rest):
        raise ValueError("Rest vertex or instance count invalid")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Positive finite rest-neighbor margin required")
    if any(not isinstance(instance, str) or not instance for instance in instance_ids):
        raise ValueError("Explicit physical instance IDs required for every vertex")
    for primitives, width in ((faces, 3), (edges, 4)):
        if primitives.ndim != 2 or primitives.shape[1] != width:
            raise ValueError("Triangle triples and Newton four-slot bending edges required")
        if len(primitives) > max_primitives or primitives.dtype.kind not in "iu":
            raise ValueError("Primitive budget or integer indexing invalid")
    endpoints = edges[:, 2:4]
    for primitives in (faces, endpoints):
        if np.any(primitives < 0) or np.any(primitives >= len(rest)):
            raise ValueError("Primitive vertex index outside rest mesh")
        for primitive in primitives:
            if len(set(int(vertex) for vertex in primitive)) != len(primitive):
                raise ValueError("Repeated vertex in primitive")
            if len({instance_ids[int(vertex)] for vertex in primitive}) != 1:
                raise ValueError("Primitive crosses physical instance boundary")
    adjacency = [dict() for _ in rest]
    vertex_faces = [set() for _ in rest]
    vertex_edges = [set() for _ in rest]
    for face_id, face in enumerate(faces):
        for vertex in face:
            vertex_faces[int(vertex)].add(face_id)
        for first, second in zip(face, np.roll(face, -1)):
            first, second = int(first), int(second)
            distance = float(np.linalg.norm(rest[first] - rest[second]))
            if distance <= 0:
                raise ValueError("Zero-length rest edge")
            adjacency[first][second] = distance
            adjacency[second][first] = distance
    for edge_id, (first, second) in enumerate(endpoints):
        first, second = int(first), int(second)
        if second not in adjacency[first]:
            raise ValueError("Newton edge endpoints absent from triangle topology")
        vertex_edges[first].add(edge_id)
        vertex_edges[second].add(edge_id)
    vertex_filters = {}
    edge_filters = {}
    visits = 0
    entries = 0
    candidate_checks = 0
    neighborhoods = []

    def check_candidate_budget():
        nonlocal candidate_checks
        candidate_checks += 1
        if candidate_checks > max_candidate_checks:
            raise ValueError("Rest-neighbor candidate check budget exceeded")

    def add_filter(mapping, primitive, neighbor):
        nonlocal entries
        selected = mapping.setdefault(primitive, set())
        if neighbor not in selected:
            entries += 1
            if entries > max_filter_entries:
                raise ValueError("Rest-neighbor filter entry budget exceeded")
            selected.add(neighbor)

    for source in range(len(rest)):
        distances = {source: 0.0}
        pending = [(0.0, source)]
        while pending:
            distance, vertex = heapq.heappop(pending)
            if distance != distances[vertex]:
                continue
            visits += 1
            if visits > max_search_visits:
                raise ValueError("Rest-neighbor search visit budget exceeded")
            for neighbor, length in adjacency[vertex].items():
                check_candidate_budget()
                candidate = distance + length
                if candidate <= margin and candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    heapq.heappush(pending, (candidate, neighbor))
        neighborhood = set(distances)
        neighborhoods.append(neighborhood)
        candidate_faces = set().union(*(vertex_faces[vertex] for vertex in neighborhood))
        for face_id in candidate_faces:
            check_candidate_budget()
            if all(int(corner) in neighborhood for corner in faces[face_id]):
                add_filter(vertex_filters, source, face_id)
    for edge_id, (first, second) in enumerate(endpoints):
        if int(second) not in neighborhoods[int(first)] or int(first) not in neighborhoods[int(second)]:
            continue
        common_neighborhood = neighborhoods[int(first)] & neighborhoods[int(second)]
        candidate_edges = set().union(*(vertex_edges[vertex] for vertex in common_neighborhood))
        for neighbor_id in candidate_edges:
            if neighbor_id <= edge_id:
                continue
            check_candidate_budget()
            neighbor_endpoints = endpoints[neighbor_id]
            if (all(int(endpoint) in common_neighborhood for endpoint in neighbor_endpoints)
                    and all(int(endpoint) in neighborhoods[int(neighbor)]
                            for endpoint in (first, second) for neighbor in neighbor_endpoints)):
                add_filter(edge_filters, edge_id, neighbor_id)
                add_filter(edge_filters, neighbor_id, edge_id)
    return vertex_filters, edge_filters, {
        "profile": "same-instance-whole-primitive-rest-geodesic-v2",
        "marginMeters": margin,
        "searchVisits": visits,
        "candidateChecks": candidate_checks,
        "vertexTriangleEntries": sum(map(len, vertex_filters.values())),
        "directedEdgeEdgeEntries": sum(map(len, edge_filters.values())),
        "limitations": [
            "Research-only whole-primitive neighborhood exclusions, not dynamic contact acceptance.",
            "Uses immutable rest-mesh edge paths; may retain close primitive interiors.",
            "Excludes only same-instance local neighborhoods; separate layers remain contact candidates.",
        ],
    }
