import math

import numpy as np


def torso_equilibrium_placement(pattern, instances, rest_positions, front_gap_mm=0.0):
    expected = {f"{face}_{side}:shell" for face in ("front", "back") for side in ("left", "right")}
    if not isinstance(instances, list) or {item["id"] for item in instances} != expected or len(instances) != 4 or set(rest_positions) != expected:
        raise ValueError("Torso control requires exactly the four torso shell instances")
    if type(front_gap_mm) not in (int, float) or not math.isfinite(front_gap_mm) or not 0 <= front_gap_mm <= 100:
        raise ValueError("Torso control gap must be finite and between 0 and 100 mm")
    panels = {panel["id"]: panel for panel in pattern["panels"]}
    by_id = {item["id"]: item for item in instances}
    shifts = {}

    def edge_points(template, edge_name):
        panel = panels[template]
        edges = [edge for edge in panel["draft"]["edges"] if edge["name"] == edge_name]
        if len(edges) != 1:
            raise ValueError("Torso control requires unique source edges")
        edge = edges[0]
        return np.asarray(panel["points"][edge["start"]:edge["end"] + 1], dtype=float)

    for side in ("left", "right"):
        front, back = f"front_{side}", f"back_{side}"
        displacement = edge_points(back, "shoulder")[0] - edge_points(front, "shoulder")[0]
        for edge_name in ("shoulder", "side"):
            front_edge, back_edge = edge_points(front, edge_name), edge_points(back, edge_name)
            if front_edge.shape != back_edge.shape or not np.allclose(front_edge + displacement, back_edge, rtol=0, atol=1e-8):
                raise ValueError("Torso source seams have no shared translation witness")
        for face in ("front", "back"):
            identity = f"{face}_{side}:shell"
            instance = by_id[identity]
            if instance["templateId"] != f"{face}_{side}" or instance["role"] != "shell" or instance["mirrorX"] is not (side == "right"):
                raise ValueError("Torso physical source identity or mirroring differs")
            hand = -1 if instance["mirrorX"] else 1
            shifts[identity] = np.array([hand * displacement[0], displacement[1], front_gap_mm]) / 1000 if face == "front" else np.zeros(3)
    left_center = edge_points("back_left", "center")
    right_center = edge_points("back_right", "center") * [-1, 1]
    if left_center.shape != right_center.shape or not np.allclose(left_center, right_center, rtol=0, atol=1e-8):
        raise ValueError("Torso source back-center paths do not coincide")
    placed = {}
    for identity, values in rest_positions.items():
        vertices = np.asarray(values, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.isfinite(vertices).all() or np.any(vertices[:, 2] != 0):
            raise ValueError("Torso control requires finite flat physical rest positions in metres")
        placed[identity] = vertices + shifts[identity]
    return placed, {"recipe": "sew-torso-coplanar-control/1", "classification": "diagnostic-only", "units": "m",
                    "frontGapMm": front_gap_mm, "translationsMeters": {identity: shift.tolist() for identity, shift in shifts.items()},
                    "limitations": ["Overlapping coplanar fabric is intentional; disable contact for this equilibrium control.",
                                    "Not garment assembly, collision-free placement or production output.",
                                    "Positive front gap is a rigid perturbation; seam closure then requires a solver."]}
