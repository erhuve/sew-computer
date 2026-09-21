import argparse
import hashlib
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp

from solver_process_budget import read_regular


parser = argparse.ArgumentParser(description="Extract a source-cuff centerline fold diagnostic; not turning or an allowance recipe")
parser.add_argument("--canonical", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--target-angle-radians", type=float, default=2.6)
parser.add_argument("--stiffness-joules", type=float, default=.02)
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
builder = newton.ModelBuilder(gravity=(0, 0, 0))
builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
    vel=wp.vec3(0, 0, 0), vertices=panel.tolist(), indices=triangles.ravel().tolist(),
    density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.01, edge_kd=0)
builder.set_coloring([[vertex] for vertex in range(len(panel))])
model = builder.finalize(device="cpu")
hinges = model.edge_indices.numpy()
interior = hinges[np.all(hinges >= 0, axis=1)]
center_y = .5 * (panel[:, 1].min() + panel[:, 1].max())
crease = interior[np.all(np.abs(panel[interior[:, 2:], 1] - center_y) < 1e-12, axis=1)]
crease_vertices = np.flatnonzero(np.abs(panel[:, 1] - center_y) < 1e-12)
ordered_vertices = crease_vertices[np.argsort(panel[crease_vertices, 0])]
expected_edges = {tuple(sorted(pair)) for pair in zip(ordered_vertices[:-1], ordered_vertices[1:])}
if (len(crease) != 7 or {tuple(sorted(row[2:])) for row in crease} != expected_edges
        or not np.isclose(panel[ordered_vertices[0], 0], panel[:, 0].min(), rtol=0, atol=1e-12)
        or not np.isclose(panel[ordered_vertices[-1], 0], panel[:, 0].max(), rtol=0, atol=1e-12)):
    raise ValueError("The source mesh must contain the complete straight centerline; no remeshing or invented crease")
control = {"restMeters": panel.tolist(), "placedMeters": panel.tolist(),
    "triangles": triangles.ravel().tolist(), "instanceOffsets": {identity: 0},
    "embeddedConstraints": {"constraints": []},
    "foldActuation": {"hinges": crease.tolist(), "stiffnessJoules": [args.stiffness_joules] * len(crease),
        "initialAnglesRadians": [0.] * len(crease),
        "targetAnglesRadians": [args.target_angle_radians] * len(crease),
        "recipe": "source-cuff-shell-centerline-diagnostic-v1", "accepted": False,
        "limitations": "Experimental centerline fold, not an original construction fold, allowance turn or assembled cuff"},
    "provenance": {"sourcePath": str(source_path), "sourceSha256": hashlib.sha256(content).hexdigest(),
        "selectedInstance": identity, "originalRange": [start, end],
        "classification": "source-preserving isolated shell fold diagnostic", "accepted": False}}
encoded = (json.dumps(control, indent=2, allow_nan=False) + "\n").encode()
output = args.output.resolve()
output.mkdir(mode=0o700)
(output / "canonical.json").write_bytes(encoded)
(output / "placement.json").write_text(json.dumps({"placedMeters": panel.tolist(),
    "canonicalDigest": hashlib.sha256(encoded).hexdigest()}, allow_nan=False) + "\n")
print(output)
