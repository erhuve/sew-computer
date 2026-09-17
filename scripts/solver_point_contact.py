import hashlib
import heapq
import importlib.util
import inspect
from pathlib import Path

import numpy as np

_ACTIVE_INSTALLATION = None


def reference_surface(rest_positions, triangles, instance_ids, radius=0.003, seam_contact_pairs=None, seam_radius=None):
    if not 0 < radius < 0.1:
        raise ValueError("Bounded positive contact radius required")
    rest = np.array(rest_positions, dtype=np.float32, copy=True)
    faces = np.asarray(triangles)
    if rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all() or len(rest) > 50000:
        raise ValueError("Bounded finite flat rest positions required")
    if not len(rest) or len(instance_ids) != len(rest) or np.any(rest[:, 2] != 0):
        raise ValueError("Flat XY rest positions and physical instance IDs required")
    if any(not isinstance(instance, str) or not instance for instance in instance_ids):
        raise ValueError("Explicit nonempty physical instance IDs required")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or len(faces) > 200000 or faces.dtype.kind not in "iu":
        raise ValueError("Bounded triangle array required")
    if np.any(faces < 0) or np.any(faces >= len(rest)):
        raise ValueError("Invalid triangle index")
    instances = list(dict.fromkeys(instance_ids))
    edges = {instance: {} for instance in instances}
    observed_faces = set()
    for face in faces:
        identity = tuple(sorted(int(vertex) for vertex in face))
        if len(set(identity)) != 3 or identity in observed_faces:
            raise ValueError("Repeated triangle or triangle vertex")
        observed_faces.add(identity)
        if np.linalg.norm(np.cross(rest[face[1]] - rest[face[0]], rest[face[2]] - rest[face[0]])) <= 0:
            raise ValueError("Degenerate rest triangle")
        identities = {instance_ids[int(vertex)] for vertex in face}
        if len(identities) != 1:
            raise ValueError("Mixed physical instance triangle")
        instance = identities.pop()
        for first, second in zip(face, np.roll(face, -1)):
            pair = tuple(sorted((int(first), int(second))))
            edges[instance][pair] = edges[instance].get(pair, 0) + 1
    descriptors = np.zeros_like(rest)
    graph_descriptors = np.zeros_like(rest)
    seam_descriptors = np.zeros_like(rest)
    boundary_positions = []
    for ordinal, instance in enumerate(instances):
        selected = np.array([identity == instance for identity in instance_ids])
        rest[selected, 2] = ordinal
        boundary = [pair for pair, count in edges[instance].items() if count == 1]
        if not boundary or len(boundary) > 2048 or any(count > 2 for count in edges[instance].values()):
            raise ValueError("Bounded manifold physical-instance boundary required")
        start = 4 * len(rest) + len(boundary_positions)
        for first, second in boundary:
            boundary_positions.extend([rest[first].copy(), rest[second].copy()])
        descriptors[selected] = [start, start + len(boundary) * 2, 0]
        boundary_array = np.asarray(boundary, dtype=int)
        starts = rest[boundary_array[:, 0], :2].astype(np.float64)
        ends = rest[boundary_array[:, 1], :2].astype(np.float64)
        directions = ends - starts
        lengths_squared = np.sum(directions * directions, axis=1)
        selected_indices = np.flatnonzero(selected)
        error_bound = 64 * np.finfo(np.float32).eps * max(float(np.max(np.abs(rest[selected, :2]))), 1.0)
        for chunk_start in range(0, len(selected_indices), 128):
            indices = selected_indices[chunk_start:chunk_start + 128]
            offsets = rest[indices, None, :2].astype(np.float64) - starts
            fractions = np.clip(np.sum(offsets * directions, axis=2) / lengths_squared, 0, 1)
            distances = np.linalg.norm(offsets - fractions[:, :, None] * directions, axis=2)
            clearance = np.maximum(0, distances.min(axis=1) - error_bound)
            descriptors[indices, 2] = np.nextafter(clearance.astype(np.float32), np.float32(0))
    adjacency = [dict() for _ in rest]
    for instance_edges in edges.values():
        for first, second in instance_edges:
            length = float(np.linalg.norm(rest[first].astype(np.float64) - rest[second]))
            adjacency[first][second] = length
            adjacency[second][first] = length
    graph_entries = []
    visits = 0
    relaxations = 0
    for source in range(len(rest)):
        distances = {source: 0.0}
        pending = [(0.0, source)]
        while pending:
            distance, vertex = heapq.heappop(pending)
            if distances[vertex] != distance:
                continue
            visits += 1
            if visits > 2000000:
                raise ValueError("Material graph visit budget exceeded")
            for neighbor, length in adjacency[vertex].items():
                relaxations += 1
                if relaxations > 12000000:
                    raise ValueError("Material graph relaxation budget exceeded")
                candidate = distance + length
                if candidate < radius and candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    heapq.heappush(pending, (candidate, neighbor))
        start = 4 * len(rest) + len(boundary_positions) + len(graph_entries)
        graph_entries.extend([neighbor, float(np.nextafter(np.float32(distance), np.float32(np.inf))), 0] for neighbor, distance in sorted(distances.items()))
        if len(graph_entries) > 2000000:
            raise ValueError("Material graph entry budget exceeded")
        graph_descriptors[source] = [start, start + len(distances), 0]
    seam_contact_pairs = [] if seam_contact_pairs is None else seam_contact_pairs
    if not isinstance(seam_contact_pairs, list) or len(seam_contact_pairs) > 2048:
        raise ValueError("Bounded explicit seam anchor pairs required")
    if seam_contact_pairs and (seam_radius is None or not 0 < seam_radius <= radius):
        raise ValueError("Seam anchor radius must be explicit and no greater than contact radius")
    anchors_by_instance = {instance: [] for instance in instances}
    original = np.asarray(rest_positions, dtype=np.float64)
    anchor_checks = 0
    for pair in seam_contact_pairs:
        anchors = []
        identities = []
        if not isinstance(pair, dict) or set(pair) != {"first", "second"}:
            raise ValueError("Explicit first/second seam anchor endpoints required")
        for side in ("first", "second"):
            endpoint = pair[side]
            if not isinstance(endpoint, dict) or set(endpoint) != {"instanceId", "restMeters"}:
                raise ValueError("Explicit seam anchor instance and rest coordinates required")
            instance = endpoint["instanceId"]
            point = np.asarray(endpoint["restMeters"], dtype=np.float64)
            if instance not in instances or point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("Invalid seam anchor identity or rest point")
            containing = False
            for face in faces:
                anchor_checks += 1
                if anchor_checks > 4000000:
                    raise ValueError("Seam anchor containment check budget exceeded")
                if instance_ids[int(face[0])] != instance:
                    continue
                triangle = original[face, :2]
                directions = np.roll(triangle, -1, axis=0) - triangle
                offsets = point - triangle
                signs = directions[:, 0] * offsets[:, 1] - directions[:, 1] * offsets[:, 0]
                if np.all(signs >= 0) or np.all(signs <= 0):
                    containing = True
                    break
            if not containing:
                raise ValueError("Seam anchor outside immutable source triangles")
            anchors.append([float(point[0]), float(point[1]), instances.index(instance)])
            identities.append(instance)
        anchors_by_instance[identities[0]].extend(anchors)
        anchors_by_instance[identities[1]].extend(reversed(anchors))
    seam_entries = []
    for instance in instances:
        selected = np.array([identity == instance for identity in instance_ids])
        start = 4 * len(rest) + len(boundary_positions) + len(graph_entries) + len(seam_entries)
        seam_entries.extend(anchors_by_instance[instance])
        seam_descriptors[selected] = [start, start + len(anchors_by_instance[instance]), seam_radius or 0]
    return np.concatenate([rest, descriptors, graph_descriptors, seam_descriptors, np.asarray(boundary_positions), np.asarray(graph_entries), np.asarray(seam_entries).reshape((-1, 3))])


HELPERS = '''
from newton._src.geometry.kernels import *

@wp.func
def material_cross(first: wp.vec3, second: wp.vec3, third: wp.vec3):
    return (second[0]-first[0])*(third[1]-first[1])-(second[1]-first[1])*(third[0]-first[0])

@wp.func
def material_path(reference: wp.array[wp.vec3], count: int, first_vertex: int, second_vertex: int, first: wp.vec3, second: wp.vec3, radius: float):
    if first[2] != second[2]:
        return False
    descriptor = reference[2*count+first_vertex]
    lower = int(descriptor[0])
    upper = int(descriptor[1])
    while lower < upper:
        middle = (lower+upper)//2
        entry = reference[middle]
        if int(entry[0]) < second_vertex:
            lower = middle+1
        else:
            upper = middle
    if lower < int(descriptor[1]):
        entry = reference[lower]
        if int(entry[0]) == second_vertex:
            distance = wp.length(first-reference[first_vertex])+entry[1]+wp.length(second-reference[second_vertex])+1.0e-6
            return distance < radius
    return False

@wp.func
def sewn_material_neighbors(reference: wp.array[wp.vec3], first: wp.vec3, second: wp.vec3, descriptor: wp.vec3):
    for cursor in range(int(descriptor[0]), int(descriptor[1]), 2):
        anchor_first = reference[cursor]
        anchor_second = reference[cursor+1]
        if first[2] == anchor_first[2] and second[2] == anchor_second[2]:
            if wp.length(first-anchor_first)+1.0e-6 < descriptor[2] and wp.length(second-anchor_second)+1.0e-6 < descriptor[2]:
                return True
    return False

@wp.func
def material_neighbors(reference: wp.array[wp.vec3], first: wp.vec3, second: wp.vec3, descriptor: wp.vec3, radius: float):
    if first[2] != second[2] or wp.length(first-second) >= radius:
        return False
    if wp.length(first-second) < descriptor[2]:
        return True
    middle = 0.5*(first+second)
    inside = bool(False)
    on_boundary = bool(False)
    for cursor in range(int(descriptor[0]), int(descriptor[1]), 2):
        start = reference[cursor]
        end = reference[cursor+1]
        side_start = material_cross(first, second, start)
        side_end = material_cross(first, second, end)
        side_first = material_cross(start, end, first)
        side_second = material_cross(start, end, second)
        if side_start*side_end < 0.0 and side_first*side_second < 0.0:
            return False
        direction = second-first
        squared_length = wp.dot(direction, direction)
        if squared_length > 0.0 and (side_start != 0.0 or side_end != 0.0):
            start_fraction = wp.dot(start-first, direction)/squared_length
            end_fraction = wp.dot(end-first, direction)/squared_length
            if wp.abs(side_start) <= 1.0e-12 and start_fraction > 0.0 and start_fraction < 1.0:
                return False
            if wp.abs(side_end) <= 1.0e-12 and end_fraction > 0.0 and end_fraction < 1.0:
                return False
        edge = end-start
        projection = wp.dot(middle-start, edge) / wp.dot(edge, edge)
        if projection >= 0.0 and projection <= 1.0 and material_cross(start, end, middle) == 0.0:
            on_boundary = True
        if (start[1] > middle[1]) != (end[1] > middle[1]):
            crossing = start[0]+(middle[1]-start[1])*(end[0]-start[0])/(end[1]-start[1])
            if middle[0] < crossing:
                inside = not inside
    return inside or on_boundary

'''


def install_point_contact(solver, rest_positions, triangles, instance_ids, output_directory, radius, seam_contact_pairs=None, seam_radius=None):
    global _ACTIVE_INSTALLATION
    import newton
    import warp as wp
    from newton._src.geometry import kernels, tri_mesh_collision

    if _ACTIVE_INSTALLATION is not None:
        raise ValueError("Research adapter permits only one installed solver per process")
    if newton.__version__ != "1.6.0" or wp.__version__ != "1.17.0" or not 0 < radius < 0.1:
        raise ValueError("Pinned Newton 1.6.0 and bounded positive radius required")
    reference = reference_surface(rest_positions, triangles, instance_ids, radius, seam_contact_pairs, seam_radius)
    if len(rest_positions) != solver.model.particle_count or not np.array_equal(triangles, solver.model.tri_indices.numpy()):
        raise ValueError("Immutable rest topology must match the finalized model exactly")
    if solver.collision_pipeline is not None or solver.particle_topological_contact_filter_threshold != 0:
        raise ValueError("Research adapter requires direct detector and no broad topological exclusions")
    vertex_source = inspect.getsource(kernels.vertex_triangle_collision_detection_kernel.func)
    edge_source = inspect.getsource(kernels.edge_colliding_edges_detection_kernel.func)
    old_vertex = '''                    closest_p_ref, _, __ = triangle_closest_point(
                        min_distance_filtering_ref_pos[t1],
                        min_distance_filtering_ref_pos[t2],
                        min_distance_filtering_ref_pos[t3],
                        min_distance_filtering_ref_pos[v_index],
                    )
                    dist_ref = wp.length(closest_p_ref - min_distance_filtering_ref_pos[v_index])

                    if dist_ref < min_query_radius:
                        continue'''
    new_vertex = '''                    closest_p_ref = _bary[0]*min_distance_filtering_ref_pos[t1]+_bary[1]*min_distance_filtering_ref_pos[t2]+_bary[2]*min_distance_filtering_ref_pos[t3]
                    closest_p_ref[2] = min_distance_filtering_ref_pos[t1][2]
                    vertex_ref = min_distance_filtering_ref_pos[v_index]
                    if sewn_material_neighbors(min_distance_filtering_ref_pos, vertex_ref, closest_p_ref, min_distance_filtering_ref_pos[3*pos.shape[0]+v_index]):
                        continue
                    if material_path(min_distance_filtering_ref_pos, pos.shape[0], v_index, t1, vertex_ref, closest_p_ref, min_query_radius) or material_path(min_distance_filtering_ref_pos, pos.shape[0], v_index, t2, vertex_ref, closest_p_ref, min_query_radius) or material_path(min_distance_filtering_ref_pos, pos.shape[0], v_index, t3, vertex_ref, closest_p_ref, min_query_radius):
                        continue
                    if material_neighbors(min_distance_filtering_ref_pos, min_distance_filtering_ref_pos[v_index], closest_p_ref, min_distance_filtering_ref_pos[pos.shape[0]+v_index], min_query_radius):
                        continue'''
    old_edge = '''                    std_ref = wp.closest_point_edge_edge(
                        e0_v0_pos_ref, e0_v1_pos_ref, e1_v0_pos_ref, e1_v1_pos_ref, edge_edge_parallel_epsilon
                    )

                    dist_ref = std_ref[2]
                    if dist_ref < min_query_radius:
                        continue'''
    new_edge = '''                    first_ref = (1.0-std[0])*e0_v0_pos_ref+std[0]*e0_v1_pos_ref
                    second_ref = (1.0-std[1])*e1_v0_pos_ref+std[1]*e1_v1_pos_ref
                    first_ref[2] = e0_v0_pos_ref[2]
                    second_ref[2] = e1_v0_pos_ref[2]
                    if sewn_material_neighbors(min_distance_filtering_ref_pos, first_ref, second_ref, min_distance_filtering_ref_pos[3*pos.shape[0]+e0_v0]):
                        continue
                    if material_path(min_distance_filtering_ref_pos, pos.shape[0], e0_v0, e1_v0, first_ref, second_ref, min_query_radius) or material_path(min_distance_filtering_ref_pos, pos.shape[0], e0_v0, e1_v1, first_ref, second_ref, min_query_radius) or material_path(min_distance_filtering_ref_pos, pos.shape[0], e0_v1, e1_v0, first_ref, second_ref, min_query_radius) or material_path(min_distance_filtering_ref_pos, pos.shape[0], e0_v1, e1_v1, first_ref, second_ref, min_query_radius):
                        continue
                    descriptor = min_distance_filtering_ref_pos[pos.shape[0]+e0_v0]
                    other_descriptor = min_distance_filtering_ref_pos[pos.shape[0]+e0_v1]
                    descriptor[2] = wp.max(0.0, wp.max(descriptor[2]-wp.length(first_ref-e0_v0_pos_ref), other_descriptor[2]-wp.length(first_ref-e0_v1_pos_ref)))
                    if material_neighbors(min_distance_filtering_ref_pos, first_ref, second_ref, descriptor, min_query_radius):
                        continue'''
    if vertex_source.count(old_vertex) != 1 or edge_source.count(old_edge) != 1:
        raise ValueError("Pinned Newton kernel source changed")
    source = HELPERS + vertex_source.replace(old_vertex, new_vertex) + "\n" + edge_source.replace(old_edge, new_edge)
    path = Path(output_directory).resolve() / "generated_material_contact.py"
    path.write_text(source)
    specification = importlib.util.spec_from_file_location("experimental_material_contact", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    _ACTIVE_INSTALLATION = (solver, tri_mesh_collision.vertex_triangle_collision_detection_kernel,
                            tri_mesh_collision.edge_colliding_edges_detection_kernel,
                            solver.particle_q_rest, solver.particle_rest_shape_contact_exclusion_radius)
    tri_mesh_collision.vertex_triangle_collision_detection_kernel = module.vertex_triangle_collision_detection_kernel
    tri_mesh_collision.edge_colliding_edges_detection_kernel = module.edge_colliding_edges_detection_kernel
    solver.particle_q_rest = wp.array(reference, dtype=wp.vec3, device=solver.model.device)
    solver.particle_rest_shape_contact_exclusion_radius = radius
    source_paths = [Path(__file__), Path(kernels.__file__), Path(tri_mesh_collision.__file__), Path(inspect.getfile(type(solver)))]
    return {"profile": "experimental-current-contact-material-locality-v2", "generatedKernelDigest": hashlib.sha256(source.encode()).hexdigest(),
            "sourceDigests": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            "referenceDigest": hashlib.sha256(reference.tobytes()).hexdigest(), "seamAnchorPairs": len(seam_contact_pairs or []), "seamRadiusMeters": seam_radius,
            "triangleDigest": hashlib.sha256(np.asarray(triangles, dtype=np.int32).tobytes()).hexdigest(), "accepted": False}


def uninstall_point_contact(solver):
    global _ACTIVE_INSTALLATION
    from newton._src.geometry import tri_mesh_collision

    if _ACTIVE_INSTALLATION is None or _ACTIVE_INSTALLATION[0] is not solver:
        raise ValueError("Solver does not own the research adapter installation")
    _, vertex_kernel, edge_kernel, reference, radius = _ACTIVE_INSTALLATION
    tri_mesh_collision.vertex_triangle_collision_detection_kernel = vertex_kernel
    tri_mesh_collision.edge_colliding_edges_detection_kernel = edge_kernel
    solver.particle_q_rest = reference
    solver.particle_rest_shape_contact_exclusion_radius = radius
    _ACTIVE_INSTALLATION = None
