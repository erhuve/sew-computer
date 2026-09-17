import math

from shapely import constrained_delaunay_triangles
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree


MAX_POINTS = 2000
MAX_VERTICES = 60000
MAX_TRIANGLES = 100000


def source_polygon(panel):
    points = panel.get("points")
    if not isinstance(points, list) or not 4 <= len(points) <= MAX_POINTS:
        raise ValueError("Invalid source boundary budget")
    for point in points:
        if not isinstance(point, list) or len(point) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or abs(value) > 10000 for value in point
        ):
            raise ValueError("Invalid source coordinate")
    if points[0] != points[-1] or len(set(map(tuple, points[:-1]))) != len(points) - 1:
        raise ValueError("Source boundary must close without duplicate vertices")
    if panel.get("holes"):
        raise ValueError("Pattern hole semantics are not supported by this mesh version")
    shape = Polygon(points)
    if not shape.is_valid or shape.area < 1:
        raise ValueError("Invalid source polygon")
    return shape


def mesh_panel(panel, max_edge_mm=40, boundary_fractions=None, quality_refinement=False):
    if isinstance(max_edge_mm, bool) or not isinstance(max_edge_mm, (int, float)) or not math.isfinite(max_edge_mm) or not 5 <= max_edge_mm <= 100:
        raise ValueError("Mesh edge budget must be between 5 and 100 mm")
    shape = source_polygon(panel)
    points = panel["points"][:-1]
    positions = [list(point) for point in points]
    lookup = {tuple(point): index for index, point in enumerate(points)}
    original_cells = list(constrained_delaunay_triangles(shape).geoms)
    original_indices = [[lookup[tuple(point)] for point in list(cell.exterior.coords)[:-1]] for cell in original_cells]
    cell_tree = STRtree(original_cells)
    insertions = {}
    if boundary_fractions is not None:
        edges = {edge["name"]: edge for edge in panel.get("draft", {}).get("edges", [])}
        if not isinstance(boundary_fractions, dict) or set(boundary_fractions) - set(edges):
            raise ValueError("Unknown assembly registration boundary")
        for name, fractions in boundary_fractions.items():
            if not isinstance(fractions, list) or len(fractions) > 256 or any(type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1 for fraction in fractions):
                raise ValueError("Invalid assembly registration fractions")
            edge = edges[name]
            segments = [(index, math.dist(panel["points"][index], panel["points"][index + 1])) for index in range(edge["start"], edge["end"])]
            length = sum(distance for _, distance in segments)
            for fraction in fractions:
                target = fraction * length
                traversed = 0
                for index, distance in segments:
                    if target <= traversed + distance + 1e-9:
                        local = min(1, max(0, (target - traversed) / distance))
                        if 1e-10 < local < 1 - 1e-10:
                            insertions.setdefault(index, set()).add(local)
                        break
                    traversed += distance
    boundary = []
    for segment_index, (first, second) in enumerate(zip(points, points[1:] + points[:1])):
        segments = max(1, math.ceil(math.dist(first, second) / max_edge_mm))
        fractions = sorted(set(step / segments for step in range(segments)) | insertions.get(segment_index, set()))
        if len(boundary) + len(fractions) > MAX_VERTICES:
            raise ValueError("Boundary refinement exceeds resource budget")
        for fraction in fractions:
            point = tuple(first[axis] + (second[axis] - first[axis]) * fraction for axis in range(2))
            if point not in lookup:
                if len(positions) >= MAX_VERTICES:
                    raise ValueError("Boundary refinement exceeds resource budget")
                lookup[point] = len(positions)
                positions.append(list(point))
            boundary.append(point)
    triangles = []
    for triangle in constrained_delaunay_triangles(Polygon(boundary)).geoms:
        indices = [lookup[tuple(point)] for point in list(triangle.exterior.coords)[:-1]]
        triangles.append(indices)
    if quality_refinement:
        from quality_meshing import quality_triangles
        triangles = quality_triangles(shape, positions, boundary, max_edge_mm)

    def midpoint(first, second):
        point = [(positions[first][axis] + positions[second][axis]) / 2 for axis in range(2)]
        positions.append(point)
        return len(positions) - 1

    for iteration in range(16):
        split = {}
        for triangle in triangles:
            for first, second in zip(triangle, triangle[1:] + triangle[:1]):
                key = tuple(sorted((first, second)))
                if key not in split and math.dist(positions[first], positions[second]) > max_edge_mm:
                    split[key] = None
        if not split:
            break
        if len(positions) + len(split) > MAX_VERTICES or len(triangles) * 4 > MAX_TRIANGLES:
            raise ValueError("Mesh refinement exceeds resource budget")
        for key in split:
            split[key] = midpoint(*key)
        refined = []
        for triangle in triangles:
            mids = [split.get(tuple(sorted((first, second)))) for first, second in zip(triangle, triangle[1:] + triangle[:1])]
            count = sum(value is not None for value in mids)
            if count == 0:
                refined.append(triangle)
            elif count == 3:
                first, second, third = triangle
                first_mid, second_mid, third_mid = mids
                refined.extend([[first, first_mid, third_mid], [first_mid, second, second_mid], [third_mid, second_mid, third], [first_mid, second_mid, third_mid]])
            else:
                pivot = next(index for index in range(3) if (mids[index] is not None if count == 1 else mids[index] is None))
                first, second, third = triangle[pivot:] + triangle[:pivot]
                rotated = mids[pivot:] + mids[:pivot]
                if count == 1:
                    refined.extend([[first, rotated[0], third], [rotated[0], second, third]])
                else:
                    refined.extend([[third, rotated[2], rotated[1]], [first, second, rotated[2]], [second, rotated[1], rotated[2]]])
        triangles = refined
    else:
        raise ValueError("Mesh refinement failed to converge within budget")
    for triangle in triangles:
        first, second, third = (positions[index] for index in triangle)
        cross = (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])
        if cross < 0:
            triangle.reverse()
    sources = []
    for vertex, position in enumerate(positions):
        if vertex < len(points):
            sources.append({vertex: 1.0})
            continue
        found = None
        for cell_index in cell_tree.query(Point(position).buffer(1e-7)):
            indices = original_indices[cell_index]
            first, second, third = (points[index] for index in indices)
            denominator = (second[1] - third[1]) * (first[0] - third[0]) + (third[0] - second[0]) * (first[1] - third[1])
            first_weight = ((second[1] - third[1]) * (position[0] - third[0]) + (third[0] - second[0]) * (position[1] - third[1])) / denominator
            second_weight = ((third[1] - first[1]) * (position[0] - third[0]) + (first[0] - third[0]) * (position[1] - third[1])) / denominator
            weights = [first_weight, second_weight, 1 - first_weight - second_weight]
            if min(weights) >= -1e-10:
                found = {index: max(0, weight) for index, weight in zip(indices, weights) if weight > 1e-12}
                total = sum(found.values())
                found = {index: weight / total for index, weight in found.items()}
                break
        if found is None:
            raise ValueError("Refined vertex has no source triangle correspondence")
        sources.append(found)
    boundaries = []
    for edge in panel.get("draft", {}).get("edges", []):
        start, end = edge.get("start"), edge.get("end")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end < len(panel["points"]):
            raise ValueError("Invalid source edge interval")
        path = LineString(panel["points"][start:end + 1])
        if path.length <= 0 or abs(path.length - edge["lengthMm"]) > 1e-4:
            raise ValueError("Source edge length mismatch")
        candidates = []
        for vertex, position in enumerate(positions):
            point = Point(position)
            if path.distance(point) <= 1e-7:
                candidates.append({"vertex": vertex, "arcMm": path.project(point)})
        candidates.sort(key=lambda item: item["arcMm"])
        boundaries.append({"name": edge["name"], "sourceStart": start, "sourceEnd": end, "lengthMm": path.length, "samples": candidates})
    return {
        "schemaVersion": 1, "units": "mm", "templateId": panel["id"],
        "representation": "seam-line-shell", "maxEdgeMm": max_edge_mm,
        "restPositions": positions, "triangles": triangles,
        "boundaries": boundaries,
        "sourceWeights": [[{"point": index, "weight": weight} for index, weight in sorted(weights.items())] for weights in sources],
        "limitations": ["Seam allowances and their bulk are omitted.", "Triangulation preserves the sampled source, not an unsampled curve.", "Element quality is reported, not certified for a cloth solver."],
    }
