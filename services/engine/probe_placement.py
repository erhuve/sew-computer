import math

import numpy as np


def _straight_path(mesh, name):
    if not isinstance(mesh, dict) or mesh.get("units") != "mm" or not isinstance(name, str):
        raise ValueError("Probe placement requires named millimetre source paths")
    paths = mesh.get("stitchPaths")
    if not isinstance(paths, list) or len(paths) > 2000:
        raise ValueError("Invalid probe source paths")
    matches = [path for path in paths if isinstance(path, dict) and path.get("name") == name]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate probe source path")
    path = matches[0]
    samples, length = path.get("samples"), path.get("lengthMm")
    if not isinstance(samples, list) or not 2 <= len(samples) <= 100000 or type(length) not in (int, float) or not math.isfinite(length) or not 1e-6 < length <= 100000:
        raise ValueError("Invalid probe path length or sample budget")
    positions, arcs = [], []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Invalid probe path sample")
        position, arc = sample.get("restPosition"), sample.get("arcMm")
        if not isinstance(position, list) or len(position) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 10000 for value in position):
            raise ValueError("Invalid probe source position")
        if type(arc) not in (int, float) or not math.isfinite(arc):
            raise ValueError("Invalid probe source arc")
        positions.append([*position, 0.0])
        arcs.append(arc)
    positions, arcs = np.asarray(positions), np.asarray(arcs)
    delta = positions[-1] - positions[0]
    endpoint_length = np.linalg.norm(delta)
    if endpoint_length <= 1e-6 or abs(endpoint_length - length) > 1e-7:
        raise ValueError("Probe placement only supports nonzero straight paths")
    tangent = delta / endpoint_length
    expected = positions[0] + arcs[:, None] * tangent
    if abs(arcs[0]) > 1e-7 or abs(arcs[-1] - length) > 1e-7 or np.any(np.diff(arcs) <= 0) or np.max(np.linalg.norm(positions - expected, axis=1)) > 1e-7:
        raise ValueError("Probe placement requires complete monotone straight source paths")
    normal = np.array([0.0, 0.0, 1.0])
    transverse = np.cross(tangent, normal)
    return {"basis": np.column_stack((transverse, tangent, normal)), "centerM": (positions[0] + positions[-1]) / 2000,
            "lengthMm": length}


def opposed_path_frame(first_mesh, second_mesh, first_path_name="right", second_path_name="right", gap_m=0.005):
    if type(gap_m) not in (int, float) or not math.isfinite(gap_m) or not 0 <= gap_m <= 0.1:
        raise ValueError("Probe placement gap must be between zero and 0.1 metres")
    first = _straight_path(first_mesh, first_path_name)
    second = _straight_path(second_mesh, second_path_name)
    rotation = first["basis"] @ np.diag([-1.0, 1.0, -1.0]) @ second["basis"].T
    translation = first["centerM"] + np.array([0.0, 0.0, gap_m]) - rotation @ second["centerM"]
    return {"recipe": "source-straight-opposed-probe/1", "units": "m", "rotation": rotation.tolist(),
            "translationM": translation.tolist(), "firstPathName": first_path_name, "secondPathName": second_path_name,
            "firstLengthMm": first["lengthMm"], "secondLengthMm": second["lengthMm"], "gapM": gap_m,
            "classification": "unvalidated-probe-placement",
            "limitations": ["Requires straight source paths with matching interior-side orientation for opposing panel bulk.",
                            "Rigid initial placement only; allowance overlap, contact and assembled garment quality are not validated."]}


def transform_probe_positions(positions_m, frame):
    if not isinstance(frame, dict) or frame.get("units") != "m" or frame.get("recipe") != "source-straight-opposed-probe/1":
        raise ValueError("Invalid probe rigid frame")
    rotation, translation = np.asarray(frame.get("rotation"), dtype=float), np.asarray(frame.get("translationM"), dtype=float)
    positions = np.asarray(positions_m, dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all() or np.max(np.abs(translation)) > 100:
        raise ValueError("Invalid probe rigid transform")
    if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-10 or abs(np.linalg.det(rotation) - 1) > 1e-10:
        raise ValueError("Probe transform must be a proper rigid rotation")
    if positions.ndim != 2 or positions.shape[1] != 3 or not 1 <= positions.shape[0] <= 60000 or not np.isfinite(positions).all() or np.max(np.abs(positions)) > 100:
        raise ValueError("Invalid probe positions in metres")
    return positions @ rotation.T + translation
