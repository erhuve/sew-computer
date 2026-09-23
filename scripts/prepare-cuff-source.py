"""Regenerate a synthetic source cuff for research on a fresh checkout.

This creates new input identity. It cannot reproduce missing historical captured
bytes or certify a garment. No body assets, private measurements or model calls
are used. The normal compiler and cut-domain validator remain authoritative.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def generate(output):
    # Capture local source before importing it. Digests use repository-relative
    # names and the output contains no checkout or private-artifact paths.
    paths = [Path(__file__), ROOT / "scripts/spike-full-shirt.py",
             ROOT / "scripts/solver_process_budget.py", ROOT / "scripts/solver-spike.requirements.txt"]
    paths += [ROOT / "services/engine" / name for name in (
        "shirt.py", "assembly.py", "cloth_domain.py", "meshing.py", "quality_meshing.py",
        "simulation_validation.py", "embedded_constraints.py", "meshing_test.py", "inspection_gltf.py")]
    captured = {str(path.relative_to(ROOT)): path.read_bytes() for path in paths}
    sys.path.insert(0, str(ROOT / "services/engine"))
    from assembly import compile_assembly, compile_inventory
    from cloth_domain import mesh_cloth_domain
    from embedded_constraints import build_embedded_constraints, validate_embedded_constraints
    from meshing_test import CONSTRUCTION, shirt_pattern
    from solver_process_budget import atomic_json

    spec = importlib.util.spec_from_file_location("cuff_source_registrations", ROOT / "scripts/spike-full-shirt.py")
    full_shirt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(full_shirt)
    pattern_bytes = json.dumps(shirt_pattern(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    pattern, inventory = compile_inventory(pattern_bytes, CONSTRUCTION)
    graph = compile_assembly(pattern, inventory)
    identities = ("cuff_left:shell", "cuff_left:facing")
    operations = [operation for operation in graph["operations"]
                  if {member["instanceId"] for member in operation["participants"]} == set(identities)]
    if len(operations) != 4 or {operation["id"] for operation in operations} != {
            "perimeter:cuff_left:" + name for name in ("extension", "end", "outer", "start")}:
        raise ValueError("Expected the source cuff's four perimeter operations and free attachment opening")
    panel = next(panel for panel in pattern["panels"] if panel["id"] == "cuff_left")
    mesh = mesh_cloth_domain(panel, 60, quality_refinement=True)
    sources = {identity: {"panel": panel, "mesh": mesh} for identity in identities}
    bundle = build_embedded_constraints(sources, full_shirt.shirt_embedded_registrations(operations, sources))
    validate_embedded_constraints(sources, bundle)
    rest = np.column_stack((np.asarray(mesh["restPositions"]) * .001, np.zeros(len(mesh["restPositions"]))))
    faces, count = np.asarray(mesh["triangles"]), len(rest)
    canonical = {
        "restMeters": np.concatenate((rest, rest)).tolist(),
        "placedMeters": np.concatenate((rest, rest + [0., 0., .002])).tolist(),
        "triangles": np.concatenate((faces, faces + count)).ravel().tolist(),
        "instanceOffsets": {identities[0]: 0, identities[1]: count},
        "sourceTemplates": {"cuff_left": mesh}, "embeddedConstraints": bundle,
        "sourcePattern": pattern, "sourceConstruction": CONSTRUCTION, "sourceAssemblyOperations": operations,
        "provenance": {"classification": "regenerated synthetic source cuff; not historical captured bytes",
                       "patternSha256": inventory["patternDigest"], "accepted": False,
                       "patternEncoding": "UTF-8 JSON, sorted keys, compact separators, no trailing newline",
                       "limitations": ["The sleeve attachment is intentionally omitted; its cuff opening remains unsewn.",
                                       "Selected perimeter operations retain their source dependencies; this isolated subset is not an executable assembly recipe.",
                                       "No turned cuff, accepted dynamics or garment fit is demonstrated."],
                       "runtime": {"python": platform.python_version(),
                                   **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "shapely")}},
                       "sourceDigests": {name: hashlib.sha256(content).hexdigest() for name, content in captured.items()}}}
    if any(path.read_bytes() != captured[str(path.relative_to(ROOT))] for path in paths):
        raise ValueError("Source changed during synthetic input generation")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    atomic_json(output / "canonical.json", canonical)
    print(json.dumps({"output": str(output.resolve()), "verticesPerLayer": count,
                      "trianglesPerLayer": len(faces), "registrations": len(bundle["constraints"]),
                      "accepted": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
