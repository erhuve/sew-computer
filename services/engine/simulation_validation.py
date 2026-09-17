import math
from collections import Counter

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from meshing import MAX_TRIANGLES, MAX_VERTICES, source_polygon


def validate_rest_mesh(panel, mesh):
    source = source_polygon(panel)
    if mesh.get("schemaVersion") != 1 or mesh.get("units") != "mm" or mesh.get("templateId") != panel["id"] or mesh.get("representation") != "seam-line-shell":
        raise ValueError("Mesh source identity mismatch")
    positions = mesh.get("restPositions")
    triangles = mesh.get("triangles")
    weights = mesh.get("sourceWeights")
    if not isinstance(positions, list) or not 3 <= len(positions) <= MAX_VERTICES or not isinstance(triangles, list) or not 1 <= len(triangles) <= MAX_TRIANGLES:
        raise ValueError("Mesh resource budget exceeded")
    if not isinstance(weights, list) or len(weights) != len(positions):
        raise ValueError("Missing source correspondence")
    limit = mesh.get("maxEdgeMm")
    if isinstance(limit, bool) or not isinstance(limit, (float, int)) or not math.isfinite(limit) or not 5 <= limit <= 100:
        raise ValueError("Invalid mesh resolution")
    for vertex_index, (position, mapping) in enumerate(zip(positions, weights)):
        if not isinstance(position, list) or len(position) != 2 or any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in position):
            raise ValueError("Invalid mesh position")
        if not isinstance(mapping, list) or not 1 <= len(mapping) <= 3:
            raise ValueError("Invalid source interpolation")
        indices = set()
        reconstructed = [0.0, 0.0]
        total = 0
        for item in mapping:
            index, weight = item.get("point"), item.get("weight")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(panel["points"]) - 1 or index in indices or isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 < weight <= 1:
                raise ValueError("Invalid interpolation weight")
            indices.add(index)
            total += weight
            for axis in range(2):
                reconstructed[axis] += panel["points"][index][axis] * weight
        if abs(total - 1) > 1e-10 or math.dist(reconstructed, position) > 1e-7:
            raise ValueError("Rest positions do not match source interpolation")
        if vertex_index < len(panel["points"]) - 1 and mapping != [{"point": vertex_index, "weight": 1.0}]:
            raise ValueError("Original boundary vertices must retain direct identity")
        support = [panel["points"][index] for index in indices]
        domain = Polygon(support) if len(support) == 3 else LineString(support) if len(support) == 2 else Point(support[0])
        if not source.covers(domain):
            raise ValueError("Source interpolation crosses outside the source domain")
    shapes = []
    edges = Counter()
    used = set()
    max_edge = 0
    min_quality = 1
    for triangle in triangles:
        if not isinstance(triangle, list) or len(triangle) != 3 or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(positions) for index in triangle) or len(set(triangle)) != 3:
            raise ValueError("Invalid triangle indices")
        coordinates = [positions[index] for index in triangle]
        face = Polygon(coordinates)
        first, second, third = coordinates
        signed_area = ((second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])) / 2
        if signed_area <= 1e-10:
            raise ValueError("Degenerate or inverted triangle")
        lengths = [math.dist(start, end) for start, end in zip(coordinates, coordinates[1:] + coordinates[:1])]
        max_edge = max(max_edge, *lengths)
        min_quality = min(min_quality, 4 * math.sqrt(3) * face.area / sum(length * length for length in lengths))
        shapes.append(face)
        used.update(triangle)
        for first, second in zip(triangle, triangle[1:] + triangle[:1]):
            edges[tuple(sorted((first, second)))] += 1
    if len(used) != len(positions) or any(count > 2 for count in edges.values()) or max_edge > limit + 1e-7:
        raise ValueError("Nonmanifold, unused or overlong mesh element")
    union = unary_union(shapes)
    area_error = union.symmetric_difference(source).area
    overlap_area = sum(shape.area for shape in shapes) - union.area
    boundary = unary_union([LineString([positions[first], positions[second]]) for (first, second), count in edges.items() if count == 1])
    boundary_error = boundary.hausdorff_distance(source.boundary)
    if area_error > max(1e-5, source.area * 1e-10) or overlap_area > max(1e-5, source.area * 1e-10) or boundary_error > 1e-6:
        raise ValueError("Mesh does not cover source exactly without overlaps or cracks")
    source_edges = panel.get("draft", {}).get("edges", [])
    if len(mesh.get("boundaries", [])) != len(source_edges):
        raise ValueError("Missing source edge mappings")
    names = set()
    for edge, mapping in zip(source_edges, mesh.get("boundaries", [])):
        if not isinstance(mapping, dict) or not isinstance(mapping.get("samples"), list) or not 2 <= len(mapping["samples"]) <= len(positions):
            raise ValueError("Invalid edge mapping")
        length = mapping.get("lengthMm")
        if isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length):
            raise ValueError("Invalid source edge length")
        if mapping["name"] != edge["name"] or mapping["name"] in names or mapping["sourceStart"] != edge["start"] or mapping["sourceEnd"] != edge["end"]:
            raise ValueError("Source edge identity mismatch")
        names.add(mapping["name"])
        path = LineString(panel["points"][edge["start"]:edge["end"] + 1])
        samples = mapping["samples"]
        if len(samples) < 2 or abs(mapping["lengthMm"] - path.length) > 1e-7 or abs(samples[0]["arcMm"]) > 1e-7 or abs(samples[-1]["arcMm"] - path.length) > 1e-7:
            raise ValueError("Incomplete edge correspondence")
        previous = None
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError("Invalid source edge sample")
            vertex, arc = sample["vertex"], sample["arcMm"]
            if isinstance(vertex, bool) or not isinstance(vertex, int) or not 0 <= vertex < len(positions) or isinstance(arc, bool) or not isinstance(arc, (int, float)) or not math.isfinite(arc) or not 0 <= arc <= path.length or path.interpolate(arc).distance(Point(positions[vertex])) > 1e-7:
                raise ValueError("Incorrect source edge sample")
            if previous is not None and (arc <= previous["arcMm"] or edges[tuple(sorted((vertex, previous["vertex"])))] != 1):
                raise ValueError("Edge samples skip or reverse a mesh boundary")
            previous = sample
    return {"profile": "rest-fidelity/1", "areaErrorMm2": area_error, "overlapAreaMm2": overlap_area, "boundaryErrorMm": boundary_error, "maxEdgeMm": max_edge, "minTriangleQuality": min_quality, "vertices": len(positions), "triangles": len(triangles), "solverQualityAccepted": False}
