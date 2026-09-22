from collections import Counter

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree


def split_straight_crease(vertices, triangles, line):
    points = np.asarray(vertices)
    faces = np.asarray(triangles)
    axis = np.asarray(line)
    if (points.dtype.kind not in "fiu" or points.ndim != 2 or points.shape[1] != 2
            or not 3 <= len(points) <= 10000 or not np.isfinite(points).all()
            or faces.dtype.kind not in "iu" or faces.ndim != 2 or faces.shape[1] != 3
            or not 1 <= len(faces) <= 20000 or np.any(faces < 0) or np.any(faces >= len(points))
            or axis.dtype.kind not in "fiu" or axis.shape != (2, 2) or not np.isfinite(axis).all()):
        raise ValueError("Finite bounded 2D source mesh, integer triangles and crease endpoints required")
    points, axis = points.astype(float), axis.astype(float)
    direction = axis[1] - axis[0]
    length = np.linalg.norm(direction)
    scale = np.linalg.norm(np.ptp(points, axis=0))
    tolerance = max(scale * 1e-12, 1e-14)
    if length <= tolerance or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Distinct source vertices and crease endpoints required")
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    distances = (points - axis[0]) @ normal
    distances[np.abs(distances) <= tolerance] = 0
    source_polygons = [Polygon(points[face]) for face in faces]
    areas = np.array([polygon.area for polygon in source_polygons])
    signed = np.array([np.linalg.det(np.stack((points[face[1]] - points[face[0]],
                                              points[face[2]] - points[face[0]]))) for face in faces])
    edges = Counter(tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
                    for face in faces for index in range(3))
    domain = unary_union(source_polygons)
    if (np.any(areas <= tolerance * scale) or np.any(signed * signed[0] <= 0)
            or max(edges.values()) > 2 or set(faces.ravel()) != set(range(len(points)))
            or domain.geom_type != "Polygon" or not domain.is_valid
            or abs(domain.area - areas.sum()) > tolerance * scale):
        raise ValueError("Nonoverlapping, consistently wound, connected source triangles required")
    tree = STRtree(source_polygons)
    for index, polygon in enumerate(source_polygons):
        for other in tree.query(polygon):
            if other <= index:
                continue
            shared = sorted(set(faces[index]) & set(faces[other]))
            intersection = polygon.intersection(source_polygons[other])
            expected = (LineString(points[shared]) if len(shared) == 2
                        else Point(points[shared[0]]) if len(shared) == 1 else None)
            if ((expected is None and not intersection.is_empty)
                    or (expected is not None and not intersection.equals(expected))):
                raise ValueError("Source triangles must meet at shared vertices or complete shared edges")
    output = points.tolist()
    weights = [{int(index): 1.} for index in range(len(points))]
    crossings = {}
    for first, second in sorted(edges):
        if distances[first] * distances[second] < 0:
            fraction = distances[first] / (distances[first] - distances[second])
            crossings[first, second] = len(output)
            output.append(((1 - fraction) * points[first] + fraction * points[second]).tolist())
            weights.append({first: float(1 - fraction), second: float(fraction)})
    children, parents = [], []
    for parent, face in enumerate(faces):
        if not (np.any(distances[face] < 0) and np.any(distances[face] > 0)):
            children.append(face.tolist())
            parents.append(parent)
            continue
        for side in (-1, 1):
            polygon = []
            for index, first in enumerate(face):
                second = face[(index + 1) % 3]
                if side * distances[first] >= 0:
                    polygon.append(int(first))
                if distances[first] * distances[second] < 0:
                    polygon.append(crossings[tuple(sorted((int(first), int(second))))])
            for index in range(1, len(polygon) - 1):
                children.append([polygon[0], polygon[index], polygon[index + 1]])
                parents.append(parent)
    output = np.asarray(output)
    children = np.asarray(children, dtype=int)
    child_polygons = [Polygon(output[face]) for face in children]
    for parent, source_polygon in enumerate(source_polygons):
        pieces = [polygon for index, polygon in enumerate(child_polygons) if parents[index] == parent]
        if (any(polygon.area <= tolerance * scale for polygon in pieces)
                or unary_union(pieces).symmetric_difference(source_polygon).area > tolerance * scale
                or abs(sum(polygon.area for polygon in pieces) - source_polygon.area) > tolerance * scale):
            raise ValueError("Crease subdivision did not preserve its parent triangle")
    child_edges = Counter(tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
                          for face in children for index in range(3))
    on_line = np.abs((output - axis[0]) @ normal) <= tolerance
    crease = sorted(edge for edge, count in child_edges.items() if count == 2 and all(on_line[list(edge)]))
    degree = Counter(vertex for edge in crease for vertex in edge)
    if not crease or sorted(degree.values()).count(1) != 2 or any(value > 2 for value in degree.values()):
        raise ValueError("Crease must form one continuous interior chain")
    ordered = sorted(degree, key=lambda vertex: float((output[vertex] - axis[0]) @ direction))
    if set(crease) != {tuple(sorted(pair)) for pair in zip(ordered[:-1], ordered[1:])}:
        raise ValueError("Disconnected crease rejected")
    for endpoint in (ordered[0], ordered[-1]):
        if not any(endpoint in edge and count == 1 for edge, count in child_edges.items()):
            raise ValueError("Crease must terminate at the cut boundary")
    return {"vertices": output.tolist(), "triangles": children.tolist(),
            "parentTriangles": parents, "sourceWeights": weights, "creaseEdges": crease,
            "creaseVertices": ordered, "classification": "straight-line subdivision, not physical assembly"}
