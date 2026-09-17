from numbers import Integral


def refine_constraint_colors(particle_count, color_groups, constraints):
    if isinstance(particle_count, bool) or not isinstance(particle_count, Integral) or particle_count < 0:
        raise ValueError("Invalid particle count")
    membership = {}
    groups = []
    for color, group in enumerate(color_groups):
        vertices = list(group)
        for vertex in vertices:
            if isinstance(vertex, bool) or not isinstance(vertex, Integral) or not 0 <= vertex < particle_count or vertex in membership:
                raise ValueError("Color groups must partition the particles")
            membership[vertex] = color
        groups.append(vertices)
    if len(membership) != particle_count:
        raise ValueError("Color groups must cover every particle")
    neighbors = {vertex: set() for vertex in membership}
    for constraint in constraints:
        vertices = list(constraint)
        if len(vertices) < 2 or any(isinstance(vertex, bool) or not isinstance(vertex, Integral) or vertex not in membership for vertex in vertices):
            raise ValueError("Invalid constraint particles")
        if len(set(vertices)) != len(vertices):
            raise ValueError("Repeated constraint particle")
        for first in vertices:
            for second in vertices:
                if first != second and membership[first] == membership[second]:
                    neighbors[first].add(second)
    refined = []
    for group in groups:
        assigned = {}
        partitions = []
        for vertex in sorted(group, key=lambda item: (-len(neighbors[item]), item)):
            forbidden = {assigned[neighbor] for neighbor in neighbors[vertex] if neighbor in assigned}
            color = 0
            while color in forbidden:
                color += 1
            if color == len(partitions):
                partitions.append([])
            partitions[color].append(vertex)
            assigned[vertex] = color
        refined.extend(partitions)
    return refined
