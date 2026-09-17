import hashlib
import json
import math

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from meshing import MAX_POINTS, mesh_panel, source_polygon


MAX_PATH_SAMPLES = 100000


def _embedding(position, positions, triangles, tree):
    for triangle_index in tree.query(Point(position).buffer(1e-7)):
        indices = triangles[triangle_index]
        first, second, third = [positions[index] for index in indices]
        denominator = (second[1] - third[1]) * (first[0] - third[0]) + (third[0] - second[0]) * (first[1] - third[1])
        first_weight = ((second[1] - third[1]) * (position[0] - third[0]) + (third[0] - second[0]) * (position[1] - third[1])) / denominator
        second_weight = ((third[1] - first[1]) * (position[0] - third[0]) + (first[0] - third[0]) * (position[1] - third[1])) / denominator
        weights = [first_weight, second_weight, 1 - first_weight - second_weight]
        if min(weights) >= -1e-9:
            weights = [max(0, weight) for weight in weights]
            total = sum(weights)
            return {"triangle": int(triangle_index), "weights": [weight / total for weight in weights]}
    raise ValueError("Interior stitching point has no cloth correspondence")


def mesh_cloth_domain(panel, max_edge_mm=40, quality_refinement=False):
    seam_shape = source_polygon(panel)
    cut_points = panel.get("draft", {}).get("cutLine")
    cut_panel = {"id": panel["id"], "points": cut_points}
    cut_shape = source_polygon(cut_panel)
    if not cut_shape.covers(seam_shape):
        raise ValueError("Cut domain must contain the complete source seam domain")
    edges = panel.get("draft", {}).get("edges", [])
    if not isinstance(edges, list) or len(edges) > MAX_POINTS:
        raise ValueError("Invalid stitching path budget")
    names = set()
    paths = []
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Invalid stitching path record")
        start, end = edge.get("start"), edge.get("end")
        name, length = edge.get("name"), edge.get("lengthMm")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Stitching path names must be unique")
        if type(start) is not int or type(end) is not int or not 0 <= start < end < len(panel["points"]):
            raise ValueError("Invalid stitching path source interval")
        if type(length) not in (int, float) or not math.isfinite(length):
            raise ValueError("Invalid stitching path source length")
        path = LineString(panel["points"][start:end + 1])
        if abs(path.length - length) > 1e-4:
            raise ValueError("Stitching path source length mismatch")
        names.add(name)
        paths.append((edge, path))
    mesh = mesh_panel(cut_panel, max_edge_mm, quality_refinement=quality_refinement)
    positions, triangles = mesh["restPositions"], mesh["triangles"]
    cells = [Polygon([positions[index] for index in triangle]) for triangle in triangles]
    tree = STRtree(cells)
    stitch_paths = []
    sample_count = 0
    for edge, path in paths:
        samples = []
        segments = []
        traversed = 0
        for source_index in range(edge["start"], edge["end"]):
            first, second = panel["points"][source_index:source_index + 2]
            source_segment = LineString([first, second])
            distances = {0.0, source_segment.length}
            for cell_index in tree.query(source_segment):
                intersection = cells[cell_index].intersection(source_segment)
                if intersection.is_empty:
                    continue
                geometries = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
                for geometry in geometries:
                    for coordinate in geometry.coords:
                        distances.add(source_segment.project(Point(coordinate)))
            ordered = []
            for distance in sorted(distances):
                if not ordered or distance - ordered[-1] > 1e-8:
                    ordered.append(distance)
            for first_distance, second_distance in zip(ordered, ordered[1:]):
                midpoint = source_segment.interpolate((first_distance + second_distance) / 2)
                embedding = _embedding(list(midpoint.coords)[0], positions, triangles, tree)
                endpoints = []
                for distance in (first_distance, second_distance):
                    position = list(source_segment.interpolate(distance).coords)[0]
                    sample = {"sourceSegment": source_index, "sourceFraction": distance / source_segment.length,
                              "arcMm": traversed + distance, "restPosition": list(position),
                              **_embedding(position, positions, triangles, tree)}
                    if samples and abs(samples[-1]["arcMm"] - sample["arcMm"]) < 1e-8:
                        endpoints.append(len(samples) - 1)
                    else:
                        sample_count += 1
                        if sample_count > MAX_PATH_SAMPLES:
                            raise ValueError("Interior stitching exceeds resource budget")
                        endpoints.append(len(samples))
                        samples.append(sample)
                segments.append({"samples": endpoints, "triangle": embedding["triangle"]})
            traversed += source_segment.length
        stitch_paths.append({"name": edge["name"], "sourceStart": edge["start"], "sourceEnd": edge["end"],
                             "lengthMm": path.length, "samples": samples, "segments": segments})
    mesh.update({
        "representation": "cut-line-cloth", "sourceDomain": "draft.cutLine",
        "sourcePanelDigest": hashlib.sha256(json.dumps(panel, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "stitchPaths": stitch_paths, "solverReady": False,
        "limitations": ["Interior stitching uses barycentric cloth embeddings; a weighted-constraint solver adapter is required.",
                        "Binding wraps, turning, folds, allowance contact and material bulk are not simulated.",
                        "Cut contours and stitch paths preserve the sampled source, not unsampled curves.",
                        "Element quality is not certified for a cloth solver."],
    })
    validate_cloth_domain(panel, mesh)
    return mesh


def validate_cloth_domain(panel, mesh):
    cut_shape = source_polygon({"points": panel.get("draft", {}).get("cutLine")})
    if not cut_shape.covers(source_polygon(panel)):
        raise ValueError("Cut domain must contain the complete source seam domain")
    edges = panel.get("draft", {}).get("edges", [])
    if not isinstance(edges, list) or len(edges) > MAX_POINTS:
        raise ValueError("Invalid stitching path budget")
    covered = []
    for edge in edges:
        if not isinstance(edge, dict) or type(edge.get("start")) is not int or type(edge.get("end")) is not int or not 0 <= edge["start"] < edge["end"] < len(panel["points"]):
            raise ValueError("Invalid source stitching edge")
        covered.extend(range(edge["start"], edge["end"]))
    if sorted(covered) != list(range(len(panel["points"]) - 1)):
        raise ValueError("Source stitching edges must cover every source segment exactly once")
    digest = hashlib.sha256(json.dumps(panel, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if mesh.get("sourcePanelDigest") != digest or mesh.get("sourceDomain") != "draft.cutLine" or mesh.get("representation") != "cut-line-cloth" or mesh.get("templateId") != panel["id"] or mesh.get("units") != "mm":
        raise ValueError("Cloth source identity mismatch")
    if mesh.get("solverReady") is not False:
        raise ValueError("Cloth foundation cannot claim solver readiness")
    positions, triangles = mesh.get("restPositions", []), mesh.get("triangles", [])
    if not positions or len(positions) > 60000 or not triangles or len(triangles) > 100000:
        raise ValueError("Invalid cloth geometry budget")
    for position in positions:
        if not isinstance(position, list) or len(position) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in position):
            raise ValueError("Invalid cloth vertex")
    cells = []
    incidence = {}
    used = set()
    for triangle in triangles:
        if not isinstance(triangle, list) or len(triangle) != 3 or len(set(triangle)) != 3 or any(type(index) is not int or not 0 <= index < len(positions) for index in triangle):
            raise ValueError("Invalid cloth triangle")
        cell = Polygon([positions[index] for index in triangle])
        first, second, third = [positions[index] for index in triangle]
        cross = (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])
        if cross <= 1e-12:
            raise ValueError("Degenerate or reversed cloth triangle")
        used.update(triangle)
        for first_index, second_index in zip(triangle, triangle[1:] + triangle[:1]):
            incidence.setdefault(tuple(sorted((first_index, second_index))), []).append((first_index, second_index))
        cells.append(cell)
    if used != set(range(len(positions))):
        raise ValueError("Unused cloth vertices")
    for occurrences in incidence.values():
        if len(occurrences) > 2 or len(occurrences) == 2 and occurrences[0] != occurrences[1][::-1]:
            raise ValueError("Nonmanifold cloth edge")
        if len(occurrences) == 1:
            boundary = LineString([positions[index] for index in occurrences[0]])
            if boundary.difference(cut_shape.boundary.buffer(1e-7)).length > 1e-7:
                raise ValueError("Unexpected cloth interior boundary")
    union = unary_union(cells)
    if union.symmetric_difference(cut_shape).area > 1e-6 or abs(sum(cell.area for cell in cells) - cut_shape.area) > 1e-6:
        raise ValueError("Cloth does not cover source cut domain exactly once")
    weights = mesh.get("sourceWeights", [])
    if len(weights) != len(positions):
        raise ValueError("Missing cut source correspondence")
    for vertex, (position, entries) in enumerate(zip(positions, weights)):
        if vertex < len(panel["draft"]["cutLine"]) - 1 and entries != [{"point": vertex, "weight": 1.0}]:
            raise ValueError("Original cut vertex identity changed")
        if not entries or len(entries) > 3 or any(type(entry.get("point")) is not int or not 0 <= entry["point"] < len(panel["draft"]["cutLine"]) - 1 or type(entry.get("weight")) not in (int, float) or not math.isfinite(entry["weight"]) or entry["weight"] < 0 for entry in entries):
            raise ValueError("Invalid cut source weights")
        if abs(sum(entry["weight"] for entry in entries) - 1) > 1e-9:
            raise ValueError("Cut source weights do not sum to one")
        support_points = [panel["draft"]["cutLine"][entry["point"]] for entry in entries]
        if len(set(map(tuple, support_points))) != len(support_points):
            raise ValueError("Duplicate cut source weight support")
        support = Polygon(support_points) if len(support_points) == 3 else LineString(support_points) if len(support_points) == 2 else Point(support_points[0])
        if not cut_shape.buffer(1e-7).covers(support):
            raise ValueError("Cut source weight support leaves source domain")
        reconstructed = [sum(panel["draft"]["cutLine"][entry["point"]][axis] * entry["weight"] for entry in entries) for axis in range(2)]
        if math.dist(position, reconstructed) > 1e-7:
            raise ValueError("Cloth vertex differs from source correspondence")
    paths = mesh.get("stitchPaths", [])
    if len(paths) != len(edges) or len({path.get("name") for path in paths}) != len(paths) or {path.get("name") for path in paths} != {edge["name"] for edge in edges}:
        raise ValueError("Missing or duplicate stitching paths")
    if sum(len(path.get("samples", [])) for path in paths) > MAX_PATH_SAMPLES:
        raise ValueError("Interior stitching exceeds resource budget")
    for edge in edges:
        path = next(path for path in paths if path["name"] == edge["name"])
        source = LineString(panel["points"][edge["start"]:edge["end"] + 1])
        if type(edge.get("lengthMm")) not in (int, float) or not math.isfinite(edge["lengthMm"]) or abs(edge["lengthMm"] - source.length) > 1e-4:
            raise ValueError("Source stitching length mismatch")
        if path.get("sourceStart") != edge["start"] or path.get("sourceEnd") != edge["end"] or type(path.get("lengthMm")) not in (int, float) or not math.isfinite(path["lengthMm"]) or abs(path["lengthMm"] - source.length) > 1e-7:
            raise ValueError("Stitching source interval mismatch")
        samples = path.get("samples", [])
        segments = path.get("segments", [])
        if not 2 <= len(samples) <= MAX_PATH_SAMPLES or len(segments) != len(samples) - 1:
            raise ValueError("Incomplete stitching path")
        previous_arc = -1
        for sample in samples:
            position = sample.get("restPosition")
            if not isinstance(position, list) or len(position) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in position):
                raise ValueError("Invalid stitching rest position")
            source_index, fraction = sample.get("sourceSegment"), sample.get("sourceFraction")
            arc, triangle_index, weights = sample.get("arcMm"), sample.get("triangle"), sample.get("weights")
            if type(source_index) is not int or not edge["start"] <= source_index < edge["end"] or type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("Invalid stitch source correspondence")
            if type(arc) not in (int, float) or not math.isfinite(arc) or arc <= previous_arc or type(triangle_index) is not int or not 0 <= triangle_index < len(triangles):
                raise ValueError("Invalid stitching sample")
            if not isinstance(weights, list) or len(weights) != 3 or any(type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0 for weight in weights) or abs(sum(weights) - 1) > 1e-9:
                raise ValueError("Invalid stitching embedding weights")
            first, second = panel["points"][source_index:source_index + 2]
            source_position = [first[axis] + (second[axis] - first[axis]) * fraction for axis in range(2)]
            expected_arc = sum(math.dist(panel["points"][index], panel["points"][index + 1]) for index in range(edge["start"], source_index)) + math.dist(first, second) * fraction
            reconstructed = [sum(positions[index][axis] * weight for index, weight in zip(triangles[triangle_index], weights)) for axis in range(2)]
            if abs(arc - expected_arc) > 1e-7 or math.dist(reconstructed, source_position) > 1e-7 or math.dist(sample["restPosition"], source_position) > 1e-7:
                raise ValueError("Stitch embedding differs from source")
            previous_arc = arc
        if abs(samples[0]["arcMm"]) > 1e-7 or abs(samples[-1]["arcMm"] - source.length) > 1e-7:
            raise ValueError("Stitching path endpoints are missing")
        total_length = 0
        for index, segment in enumerate(segments):
            triangle_index = segment.get("triangle")
            if segment.get("samples") != [index, index + 1] or type(triangle_index) is not int or not 0 <= triangle_index < len(cells):
                raise ValueError("Stitching segment connectivity mismatch")
            line = LineString([samples[index]["restPosition"], samples[index + 1]["restPosition"]])
            if line.difference(cells[triangle_index].buffer(1e-7)).length > 1e-7 or line.difference(source.buffer(1e-7)).length > 1e-7:
                raise ValueError("Stitching segment leaves its source or triangle")
            total_length += line.length
        if abs(total_length - source.length) > 1e-6:
            raise ValueError("Stitching path does not preserve source length")
    return {"sourceFidelityAccepted": True, "solverReady": False, "cutAreaMm2": cut_shape.area, "stitchPathCount": len(paths)}
