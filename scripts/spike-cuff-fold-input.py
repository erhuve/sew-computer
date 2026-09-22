import argparse
import hashlib
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp

from solver_process_budget import read_regular
from solver_crease_mesh import split_straight_crease


parser = argparse.ArgumentParser(description="Extract a source-cuff centerline fold diagnostic; not turning or an allowance recipe")
parser.add_argument("--canonical", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--target-angle-radians", type=float, default=2.6)
parser.add_argument("--stiffness-joules", type=float, default=.02)
parser.add_argument("--crease", choices=("centerline", "outer-allowance"), default="centerline")
args = parser.parse_args()
if not np.isfinite(args.target_angle_radians) or not 0 < abs(args.target_angle_radians) < np.pi - 1e-8:
    parser.error("Nonzero signed target angle strictly inside the principal branch required")
if not np.isfinite(args.stiffness_joules) or not 0 < args.stiffness_joules <= 10:
    parser.error("Positive actuator stiffness at most 10 joules/radian squared required")
source_path = args.canonical.resolve()
content = read_regular(source_path, 50 * 1024 ** 2)
source = json.loads(content)
rest = np.asarray(source["restMeters"], dtype=float)
faces = np.asarray(source["triangles"]).reshape((-1, 3))
ordered = sorted(source["instanceOffsets"].items(), key=lambda entry: entry[1])
ranges = {identity: (start, ordered[index + 1][1] if index + 1 < len(ordered) else len(rest))
          for index, (identity, start) in enumerate(ordered)}
identity = "cuff_left:shell"
start, end = ranges[identity]
panel = rest[start:end].copy()
triangles = faces[np.all((faces >= start) & (faces < end), axis=1)] - start
if panel.shape != (20, 3) or triangles.shape != (24, 3) or np.any(panel[:, 2] != 0):
    raise ValueError("This recipe requires the reviewed planar 20-vertex source-cuff shell")
subdivision = None
if args.crease == "outer-allowance":
    template = source["sourceTemplates"]["cuff_left"]
    if (template.get("units") != "mm" or template.get("sourceDomain") != "draft.cutLine"
            or not np.allclose(np.asarray(template["restPositions"]) * .001, panel[:, :2], rtol=0, atol=1e-12)
            or not np.array_equal(template["triangles"], triangles)):
        raise ValueError("Cuff template must match the canonical cut mesh")
    paths = [path for path in template["stitchPaths"] if path["name"] == "outer"]
    if len(paths) != 1:
        raise ValueError("Exactly one recorded outer stitch path required")
    path = paths[0]
    samples = path["samples"]
    coordinates = np.asarray([sample["restPosition"] for sample in samples], dtype=float) * .001
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2
            or not np.isfinite(coordinates).all()):
        raise ValueError("Finite outer stitch samples required")
    for sample, coordinate in zip(samples, coordinates):
        weights = np.asarray(sample["weights"])
        triangle = sample["triangle"]
        if (type(triangle) is not int or not 0 <= triangle < len(triangles)
                or weights.shape != (3,) or weights.dtype.kind not in "fiu"
                or not np.isfinite(weights).all() or np.any(weights < -1e-12)
                or abs(weights.sum() - 1) > 1e-12
                or not np.allclose(weights @ panel[triangles[triangle], :2], coordinate, rtol=0, atol=1e-12)):
            raise ValueError("Outer stitch sample does not reconstruct from its source triangle")
    line = coordinates[[0, -1]]
    direction = line[1] - line[0]
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        raise ValueError("Nonzero outer stitch path required")
    normal = np.array([-direction[1], direction[0]]) / length
    progress = (coordinates - line[0]) @ direction / length
    if (np.max(np.abs((coordinates - line[0]) @ normal)) > 1e-12
            or np.any(np.diff(progress) <= 0)
            or not np.isclose(length * 1000, path["lengthMm"], rtol=0, atol=1e-8)
            or not np.allclose([sample["arcMm"] for sample in samples], progress * 1000, rtol=0, atol=1e-8)):
        raise ValueError("Only a straight, ordered, source-bound outer stitch path is supported")
    subdivision = split_straight_crease(panel[:, :2], triangles, line)
    subdivision["sourceStitchPath"] = path
    subdivision["extensionPolicy"] = "Extend the outer stitch line through both side allowances to the cut boundary"
    panel = np.column_stack((subdivision["vertices"], np.zeros(len(subdivision["vertices"]))))
    triangles = np.asarray(subdivision["triangles"])
builder = newton.ModelBuilder(gravity=(0, 0, 0))
builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
    vel=wp.vec3(0, 0, 0), vertices=panel.tolist(), indices=triangles.ravel().tolist(),
    density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.01, edge_kd=0)
builder.set_coloring([[vertex] for vertex in range(len(panel))])
model = builder.finalize(device="cpu")
hinges = model.edge_indices.numpy()
interior = hinges[np.all(hinges >= 0, axis=1)]
if subdivision is None:
    center_y = .5 * (panel[:, 1].min() + panel[:, 1].max())
    crease_vertices = np.flatnonzero(np.abs(panel[:, 1] - center_y) < 1e-12)
    ordered_vertices = crease_vertices[np.argsort(panel[crease_vertices, 0])]
    expected_edges = {tuple(sorted(pair)) for pair in zip(ordered_vertices[:-1], ordered_vertices[1:])}
    if (len(expected_edges) != 7
            or not np.isclose(panel[ordered_vertices[0], 0], panel[:, 0].min(), rtol=0, atol=1e-12)
            or not np.isclose(panel[ordered_vertices[-1], 0], panel[:, 0].max(), rtol=0, atol=1e-12)):
        raise ValueError("The source mesh must contain the complete straight centerline")
else:
    expected_edges = set(subdivision["creaseEdges"])
crease = np.asarray([row for row in interior if tuple(sorted(row[2:])) in expected_edges])
if len(crease) != len(expected_edges):
    raise ValueError("Every declared crease edge must have a source hinge")
control = {"restMeters": panel.tolist(), "placedMeters": panel.tolist(),
    "triangles": triangles.ravel().tolist(), "instanceOffsets": {identity: 0},
    "embeddedConstraints": {"constraints": []},
    "foldActuation": {"hinges": crease.tolist(), "stiffnessJoules": [args.stiffness_joules] * len(crease),
        "initialAnglesRadians": [0.] * len(crease),
        "targetAnglesRadians": [args.target_angle_radians] * len(crease),
        "recipe": "source-cuff-shell-" + args.crease + "-diagnostic-v1", "accepted": False,
        "limitations": "Experimental single-shell fold, not sewn cuff turning or assembled garment"},
    "provenance": {"sourcePath": str(source_path), "sourceSha256": hashlib.sha256(content).hexdigest(),
        "selectedInstance": identity, "originalRange": [start, end],
        "classification": "source-preserving isolated shell fold diagnostic", "accepted": False}}
if subdivision is not None:
    control["provenance"]["subdivision"] = subdivision
encoded = (json.dumps(control, indent=2, allow_nan=False) + "\n").encode()
output = args.output.resolve()
output.mkdir(mode=0o700)
(output / "canonical.json").write_bytes(encoded)
(output / "placement.json").write_text(json.dumps({"placedMeters": panel.tolist(),
    "canonicalDigest": hashlib.sha256(encoded).hexdigest()}, allow_nan=False) + "\n")
print(output)
