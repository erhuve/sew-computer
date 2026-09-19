import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from solver_process_budget import read_regular

parser = argparse.ArgumentParser(description="Extract the saved-source left cuff for a parallel-layer sewing diagnostic; not garment assembly")
parser.add_argument("--canonical", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source_path = args.canonical.resolve()
content = read_regular(source_path, 50 * 1024 ** 2)
source = json.loads(content)
rest = np.asarray(source["restMeters"])
faces = np.asarray(source["triangles"]).reshape((-1, 3))
ordered = sorted(source["instanceOffsets"].items(), key=lambda entry: entry[1])
ranges = {identity: (start, ordered[index + 1][1] if index + 1 < len(ordered) else len(rest))
          for index, (identity, start) in enumerate(ordered)}
selected = ("cuff_left:shell", "cuff_left:facing")
panels, triangles, offsets = [], [], {}
for identity in selected:
    start, end = ranges[identity]
    offset = sum(len(panel) for panel in panels)
    offsets[identity] = offset
    panels.append(rest[start:end])
    triangles.append(faces[np.all((faces >= start) & (faces < end), axis=1)] - start + offset)
np.testing.assert_array_equal(panels[0], panels[1])
np.testing.assert_array_equal(triangles[0], triangles[1] - len(panels[0]))
subset = np.concatenate(panels)
placed = subset.copy()
normal = np.cross(panels[0][triangles[0][0, 1]] - panels[0][triangles[0][0, 0]],
                  panels[0][triangles[0][0, 2]] - panels[0][triangles[0][0, 0]])
normal /= np.linalg.norm(normal)
placed[len(panels[0]):] += .002 * normal
constraints = copy.deepcopy([row for row in source["embeddedConstraints"]["constraints"]
    if {term["instanceId"] for term in row["terms"]} == set(selected)])
if len(constraints) != 20:
    raise ValueError("This diagnostic requires the reviewed 20-registration cuff fixture")
for row in constraints:
    anchors = []
    for sample in row["sourceSamples"]:
        panel = panels[selected.index(sample["instanceId"])]
        anchors.append(sum(weight["weight"] * panel[weight["vertex"]] for weight in sample["weights"]))
    np.testing.assert_allclose(anchors[0], anchors[1], rtol=0, atol=1e-14)
control = {"restMeters": subset.tolist(), "placedMeters": placed.tolist(),
           "triangles": np.concatenate(triangles).ravel().tolist(), "instanceOffsets": offsets,
           "embeddedConstraints": {"constraints": constraints},
           "provenance": {"sourcePath": str(source_path), "sourceSha256": hashlib.sha256(content).hexdigest(),
               "selectedInstances": selected, "originalRanges": {identity: ranges[identity] for identity in selected},
               "initialOffsetM": .002, "normal": normal.tolist(),
               "classification": "isolated source-cuff parallel-layer registration; no turning or full garment",
               "accepted": False}}
output = args.output.resolve()
output.mkdir(mode=0o700)
encoded = (json.dumps(control, indent=2, allow_nan=False) + "\n").encode()
(output / "canonical.json").write_bytes(encoded)
(output / "placement.json").write_text(json.dumps({"placedMeters": placed.tolist(),
    "canonicalDigest": hashlib.sha256(encoded).hexdigest()}, allow_nan=False) + "\n")
print(output)
