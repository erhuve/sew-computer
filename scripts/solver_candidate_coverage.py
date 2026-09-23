"""Bounded, native-independent completeness check for surface candidate sets.

This proves coverage of exact swept primitive AABBs for affine vertex motion.
It does not prove separation, validate a narrow-phase distance, or certify any
contact force. Additional nonincident candidates are allowed. Only topological
incidence is excluded; all vertices must belong to the supplied triangle mesh.
"""

from fractions import Fraction
import math
import numbers

PROFILE = "exact-rational-swept-aabb-v1"


class _Rejected(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def verify_candidate_coverage(start, end, edges, faces, inflation_radius_m,
                              observed_candidates, *, max_primitives=200000,
                              max_comparisons=2000000, max_candidates=1000000):
    """Check (kind, first_id, second_id) FV/EE records against exact overlaps.

    FV IDs are (face, vertex); EE IDs are two edge indices in either order.
    Each endpoint AABB is inflated by ``inflation_radius_m`` on every side,
    matching the native broad-phase inflation convention. Starts sort before
    ends, so touching boxes are included. A sorted-axis sweep only visits pairs
    whose intervals overlap on the chosen axis. All budgets are explicit; a
    rejected or exhausted check returns verified=False, never partial success.
    Counts describe visited work and are totals only when verified is true.
    """
    result = {"verified": False, "status": "invalid-input", "profile": PROFILE,
              "counts": {"vertices": 0, "edges": 0, "faces": 0,
                         "observedCandidates": 0, "comparisons": 0,
                         "requiredCandidates": 0}}
    counts = result["counts"]

    def reject(status, message):
        raise _Rejected(status, message)

    def index(value, bound):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            reject("invalid-input", "Primitive identities must be integers")
        value = int(value)
        if not 0 <= value < bound:
            reject("invalid-input", "Primitive identity out of range")
        return value

    def finite(value):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            reject("invalid-input", "Finite real coordinates and radius required")
        value = float(value)
        if not math.isfinite(value):
            reject("invalid-input", "Finite real coordinates and radius required")
        return Fraction(value)

    try:
        for budget, ceiling in ((max_primitives, 200000),
                                (max_comparisons, 2000000),
                                (max_candidates, 1000000)):
            if type(budget) is not int or not 1 <= budget <= ceiling:
                reject("invalid-input", "Positive bounded integer budgets required")
        vertex_count, edge_count, face_count = len(start), len(edges), len(faces)
        counts.update(vertices=vertex_count, edges=edge_count, faces=face_count)
        if vertex_count + edge_count + face_count > max_primitives:
            reject("budget-exceeded", "Primitive allocation budget exhausted")
        if hasattr(observed_candidates, "__len__") and len(observed_candidates) > max_candidates:
            reject("budget-exceeded", "Observed candidate allocation budget exhausted")
        if len(end) != vertex_count or vertex_count < 3 or face_count < 1:
            reject("invalid-input", "Matching nonempty surface endpoint meshes required")
        radius = finite(inflation_radius_m)
        if radius < 0:
            reject("invalid-input", "Inflation radius must be nonnegative")
        coordinates = []
        for state in (start, end):
            converted = []
            for point in state:
                if len(point) != 3:
                    reject("invalid-input", "Three coordinates per material vertex required")
                converted.append(tuple(finite(value) for value in point))
            coordinates.append(converted)
        topology, face_keys, used_vertices, required_edges = [], set(), set(), set()
        for face in faces:
            if len(face) != 3:
                reject("invalid-input", "Triangle topology required")
            face = tuple(index(vertex, vertex_count) for vertex in face)
            key = tuple(sorted(face))
            if len(set(face)) != 3 or key in face_keys:
                reject("invalid-input", "Distinct vertices and unique triangles required")
            face_keys.add(key)
            used_vertices.update(face)
            required_edges.update(tuple(sorted((face[a], face[b]))) for a, b in ((0, 1), (1, 2), (2, 0)))
            topology.append(face)
        if len(used_vertices) != vertex_count:
            reject("invalid-input", "All material vertices must belong to triangles")
        edge_topology, edge_keys = [], set()
        for edge in edges:
            if len(edge) != 2:
                reject("invalid-input", "Two identities per edge required")
            edge = tuple(index(vertex, vertex_count) for vertex in edge)
            key = tuple(sorted(edge))
            if edge[0] == edge[1] or key in edge_keys:
                reject("invalid-input", "Distinct vertices and unique edges required")
            edge_keys.add(key)
            edge_topology.append(edge)
        if edge_keys != required_edges:
            reject("invalid-input", "Edge topology must equal triangle-derived edges")
        observed = set()
        for candidate in observed_candidates:
            counts["observedCandidates"] += 1
            if counts["observedCandidates"] > max_candidates:
                reject("budget-exceeded", "Observed candidate budget exhausted")
            if len(candidate) != 3 or candidate[0] not in ("fv", "ee"):
                reject("invalid-input", "Only FV/EE surface candidate records are supported")
            kind, first, second = candidate
            if kind == "fv":
                first, second = index(first, face_count), index(second, vertex_count)
                incident = second in topology[first]
            else:
                first, second = sorted((index(first, edge_count), index(second, edge_count)))
                incident = bool(set(edge_topology[first]) & set(edge_topology[second]))
            identity = (kind, first, second)
            if incident or identity in observed:
                reject("invalid-input", "Incident or duplicate candidate record")
            observed.add(identity)

        def bounds(vertices):
            points = [coordinates[time][vertex] for time in (0, 1) for vertex in vertices]
            return tuple((min(point[axis] for point in points) - radius,
                          max(point[axis] for point in points) + radius) for axis in range(3))

        vertex_boxes = [bounds((vertex,)) for vertex in range(vertex_count)]
        face_boxes = [bounds(face) for face in topology]
        edge_boxes = [bounds(edge) for edge in edge_topology]
        # The widest scene axis is a deterministic inexpensive sweep choice.
        axis = max(range(3), key=lambda dimension:
                   max(box[dimension][1] for box in vertex_boxes)
                   - min(box[dimension][0] for box in vertex_boxes))
        result["sweepAxis"] = axis

        def visit(kind, first, second, first_box, second_box):
            if counts["comparisons"] >= max_comparisons:
                reject("budget-exceeded", "Interval comparison budget exhausted")
            counts["comparisons"] += 1
            if ((kind == "fv" and second in topology[first]) or
                    (kind == "ee" and set(edge_topology[first]) & set(edge_topology[second]))):
                return
            if not all(a[0] <= b[1] and b[0] <= a[1] for a, b in zip(first_box, second_box)):
                return
            counts["requiredCandidates"] += 1
            if counts["requiredCandidates"] > max_candidates:
                reject("budget-exceeded", "Required candidate budget exhausted")
            identity = (kind, first, second) if kind == "fv" else (kind, *sorted((first, second)))
            if identity not in observed:
                result["firstMissingCandidate"] = list(identity)
                reject("missing-candidate", "Exact swept AABB overlap lacks a candidate")

        def sweep(kind, groups):
            # No pair matrix is allocated. Events and active sets are bounded
            # by max_primitives; comparisons are charged before topology tests.
            events = [(box[axis][side], side, group, identifier)
                      for group, boxes in enumerate(groups)
                      for identifier, box in enumerate(boxes) for side in (0, 1)]
            events.sort()
            active = [{} for group in groups]
            for _, side, group, identifier in events:
                if side:
                    del active[group][identifier]
                    continue
                opposite = 0 if len(groups) == 1 else 1 - group
                for other in active[opposite]:
                    first, second = ((identifier, other) if group == 0 else (other, identifier))
                    visit(kind, first, second, groups[0][first], groups[-1][second])
                active[group][identifier] = None

        sweep("fv", (face_boxes, vertex_boxes))
        sweep("ee", (edge_boxes,))
        result.update(verified=True, status="complete")
    except _Rejected as error:
        result.update(status=error.status, reason=error.message)
    except (TypeError, ValueError, OverflowError, IndexError, KeyError) as error:
        result.update(status="invalid-input", reason=f"Malformed bounded input: {type(error).__name__}")
    return result
