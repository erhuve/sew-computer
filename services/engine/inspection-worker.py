import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard import constrain


def main():
    for line in (Path(__file__).parent / "requirements.lock").read_text().splitlines():
        name, version = line.split("==")
        if importlib.metadata.version(name) != version:
            raise RuntimeError("Pinned inspection dependency mismatch")
    controls = constrain()
    payload = sys.stdin.buffer.read(5 * 1024 * 1024 + 1)
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError("Inspection input exceeds budget")
    inputs = json.loads(payload)
    from assembly import build_inspection
    from inspection_gltf import inspection_glb
    pattern = inputs["pattern"].encode("utf-8")
    inspection = build_inspection(pattern, inputs["construction"])
    artifact = inspection_glb(inspection)
    inspection["displayArtifact"] = {"filename": "inspection.glb", "sha256": hashlib.sha256(artifact).hexdigest(), "bytes": len(artifact)}
    inspection["executionControls"] = controls
    encoded = json.dumps(inspection, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > 16 * 1024 * 1024:
        raise ValueError("Inspection artifact exceeds budget")
    Path("inspection.glb").write_bytes(artifact)
    Path("inspection.json").write_bytes(encoded)


if __name__ == "__main__":
    main()
