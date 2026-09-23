"""Bind explicit gripper controls to captured canonical cloth identity."""

import copy
import hashlib
import json
import math

import numpy as np

from solver_material_grippers import MaterialGrippers, MaterialGripperSchedule


PROFILE = "captured-material-grippers-v1"


def mesh_identity(source):
    """A non-self-referential digest of exact source mesh/instance JSON values."""
    payload = {key: source[key] for key in ("restMeters", "triangles", "instanceOffsets")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def bind_material_grippers(source, subdivisions):
    # Validate captured JSON before coercion: mixed Boolean/integer arrays
    # otherwise erase the Boolean and can impersonate canonical indices.
    raw_rest, raw_faces = source.get("restMeters"), source.get("triangles")
    if (type(raw_rest) is not list or not 3 <= len(raw_rest) <= 25000
            or any(type(row) is not list or len(row) != 3
                   or any(type(value) not in (int, float) or abs(value) > 100
                          or not math.isfinite(value) for value in row) for row in raw_rest)
            or type(raw_faces) is not list or not raw_faces):
        raise ValueError("Bounded finite canonical gripper geometry required")
    nested_faces = type(raw_faces[0]) is list
    if nested_faces:
        if (len(raw_faces) > 50000 or any(type(face) is not list or len(face) != 3
                or any(type(index) is not int or not 0 <= index < len(raw_rest) for index in face)
                for face in raw_faces)):
            raise ValueError("Bounded integer canonical gripper triangles required")
    elif (len(raw_faces) > 150000 or len(raw_faces) % 3
            or any(type(index) is not int or not 0 <= index < len(raw_rest) for index in raw_faces)):
        raise ValueError("Bounded integer canonical gripper triangles required")
    recipe = source.get("gripperActuation")
    if (not isinstance(recipe, dict)
            or set(recipe) != {"profile", "accepted", "meshSha256", "anchors", "schedule"}
            or recipe["profile"] != PROFILE or recipe["accepted"] is not False
            or recipe["meshSha256"] != mesh_identity(source)):
        raise ValueError("Explicit unaccepted gripper recipe must match the captured source mesh identity")
    rest, raw_faces = np.asarray(raw_rest, dtype=float), np.asarray(raw_faces)
    if (rest.dtype.kind not in "fiu" or rest.ndim != 2 or rest.shape[1] != 3
            or not 3 <= len(rest) <= 25000 or not np.isfinite(rest).all()
            or np.max(np.abs(rest)) > 100 or raw_faces.dtype.kind not in "iu"
            or raw_faces.size % 3 or raw_faces.ndim not in (1, 2)):
        raise ValueError("Bounded finite canonical gripper geometry required")
    if raw_faces.ndim == 2 and raw_faces.shape[1] != 3:
        raise ValueError("Canonical gripper faces must be triangles")
    faces = raw_faces.reshape((-1, 3))
    offsets = source["instanceOffsets"]
    if (not isinstance(offsets, dict) or not 1 <= len(offsets) <= 64
            or any(type(identity) is not str or not 1 <= len(identity) <= 160
                   or type(offset) is not int or not 0 <= offset < len(rest)
                   for identity, offset in offsets.items())):
        raise ValueError("Bounded gripper physical instance offsets required")
    ordered = sorted(offsets.items(), key=lambda item: item[1])
    if ordered[0][1] != 0 or len({offset for _, offset in ordered}) != len(ordered):
        raise ValueError("Gripper physical instances must partition the canonical mesh")
    vertex_ids = []
    for index, (identity, offset) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(rest)
        if end - offset < 3:
            raise ValueError("Each gripper physical instance needs at least three vertices")
        vertex_ids.extend([identity] * (end - offset))
    grippers = MaterialGrippers(vertex_ids, faces, recipe["anchors"])
    schedule = MaterialGripperSchedule(recipe["schedule"], subdivisions, gripper_ids=grippers.gripper_ids)
    bindings = []
    for anchor in recipe["anchors"]:
        vertices = faces[anchor["triangleIndex"]]
        bindings.append({**copy.deepcopy(anchor), "canonicalVertices": vertices.tolist(),
                         "restAnchorMeters": (np.asarray(anchor["weights"]) @ rest[vertices]).tolist()})
    return grippers, schedule, {
        "profile": PROFILE, "accepted": False, "meshSha256": recipe["meshSha256"],
        "anchorBindings": bindings,
        "scope": "Canonical material-triangle anchors; complete canonical digest and captured source lineage remain in the enclosing run manifest. Targets are virtual controls, not collision geometry or construction proof."}
