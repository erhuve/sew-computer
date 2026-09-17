import math

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import Point, Polygon


def quality_triangles(shape, positions, boundary, max_edge_mm, max_vertices=12000):
    lookup = {tuple(point): index for index, point in enumerate(positions)}
    xmin, ymin, xmax, ymax = shape.bounds
    spacing = max_edge_mm * 0.65
    rows = math.ceil((ymax - ymin) / spacing)
    columns = math.ceil((xmax - xmin) / spacing)
    if rows * columns + len(positions) > max_vertices:
        raise ValueError("Quality mesh grid exceeds resource budget")

    def insert(point):
        key = tuple(map(float, point))
        if key not in lookup:
            if len(positions) >= max_vertices:
                raise ValueError("Quality mesh refinement exceeds resource budget")
            lookup[key] = len(positions)
            positions.append(list(key))

    for row in range(1, rows):
        for column in range(1, columns):
            point = Point(xmin + column * (xmax - xmin) / columns, ymin + row * (ymax - ymin) / rows)
            if shape.contains(point) and shape.boundary.distance(point) > spacing * 0.2:
                insert(point.coords[0])
    segments = [(tuple(first), tuple(second)) for first, second in zip(boundary, boundary[1:] + boundary[:1])]
    tolerance = 1e-7
    padded = shape.buffer(tolerance)
    for iteration in range(32):
        coordinates = np.asarray(positions)
        triangulation = Delaunay(coordinates)
        triangles = [list(map(int, triangle)) for triangle in triangulation.simplices if padded.covers(Polygon(coordinates[triangle]))]
        triangle_edges = {tuple(sorted((first, second))) for face in triangles for first, second in zip(face, face[1:] + face[:1])}
        missing = [(first, second) for first, second in segments if tuple(sorted((lookup[first], lookup[second]))) not in triangle_edges]
        if missing:
            replaced = []
            for first, second in segments:
                if (first, second) in missing:
                    middle = tuple((first[axis] + second[axis]) / 2 for axis in range(2))
                    insert(middle)
                    replaced.extend([(first, middle), (middle, second)])
                else:
                    replaced.append((first, second))
            segments = replaced
            continue
        additions = []
        encroached = set()
        tree = cKDTree(coordinates)
        for face in triangles:
            first, second, third = coordinates[face]
            edge_lengths = [float(np.linalg.norm(second - first)), float(np.linalg.norm(third - second)), float(np.linalg.norm(first - third))]
            angles = [math.acos(max(-1, min(1, (edge_lengths[index] ** 2 + edge_lengths[(index + 1) % 3] ** 2 - edge_lengths[(index + 2) % 3] ** 2) / (2 * edge_lengths[index] * edge_lengths[(index + 1) % 3])))) for index in range(3)]
            if min(angles) >= math.radians(20) and max(edge_lengths) <= max_edge_mm:
                continue
            matrix = 2 * np.asarray([second - first, third - first])
            try:
                center = first + np.linalg.solve(matrix, np.asarray([np.dot(second - first, second - first), np.dot(third - first, third - first)]))
            except np.linalg.LinAlgError:
                continue
            point = Point(center)
            if shape.contains(point) and shape.boundary.distance(point) > tolerance and tree.query(center)[0] > max(tolerance, min(edge_lengths) * 0.1):
                blocked = False
                for segment_index, (start, end) in enumerate(segments):
                    if (center[0] - start[0]) * (center[0] - end[0]) + (center[1] - start[1]) * (center[1] - end[1]) < -tolerance:
                        encroached.add(segment_index)
                        blocked = True
                if blocked:
                    continue
                if not additions or min(math.dist(center, previous) for previous in additions) > min(edge_lengths) * 0.25:
                    additions.append(center.tolist())
        if encroached:
            replaced = []
            for segment_index, (first, second) in enumerate(segments):
                if segment_index in encroached:
                    middle = tuple((first[axis] + second[axis]) / 2 for axis in range(2))
                    insert(middle)
                    replaced.extend([(first, middle), (middle, second)])
                else:
                    replaced.append((first, second))
            segments = replaced
            continue
        if not additions:
            return triangles
        for point in additions:
            insert(point)
    triangulation = Delaunay(np.asarray(positions))
    return [list(map(int, triangle)) for triangle in triangulation.simplices if padded.covers(Polygon([positions[index] for index in triangle]))]
