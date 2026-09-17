import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys


def main():
    parser = argparse.ArgumentParser(description="Create private source-bound flat 3D inspection artifacts. Does not assemble or simulate cloth.")
    parser.add_argument("--pattern", type=Path, required=True)
    parser.add_argument("--construction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-edge-mm", type=float, default=40)
    arguments = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 ** 2, 32 * 1024 ** 2))
    with arguments.pattern.open("rb") as source:
        pattern = source.read(4 * 1024 ** 2 + 1)
    with arguments.construction.open("rb") as source:
        construction_bytes = source.read(32769)
    if len(construction_bytes) > 32768:
        raise ValueError("Construction input exceeds budget")
    construction = json.loads(construction_bytes)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/engine"))
    from assembly import build_inspection
    from inspection_gltf import inspection_glb

    inspection = build_inspection(pattern, construction, arguments.max_edge_mm)
    artifact = inspection_glb(inspection)
    inspection["displayArtifact"] = {"filename": "inspection.glb", "sha256": hashlib.sha256(artifact).hexdigest(), "bytes": len(artifact)}
    encoded = json.dumps(inspection, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > 32 * 1024 ** 2:
        raise ValueError("Canonical mesh artifact exceeds budget")
    arguments.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (arguments.output / "inspection.glb").write_bytes(artifact)
    (arguments.output / "inspection.json").write_bytes(encoded)
    print(json.dumps({"classification": "placement-inspection", "output": str(arguments.output.resolve()), "fabricInstances": len(inspection["instances"]), "unresolvedInterfacing": len(inspection["unresolvedPhysicalRoles"])}))


if __name__ == "__main__":
    main()
