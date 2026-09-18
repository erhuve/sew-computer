import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import numpy as np

from solver_ipc_contact import IpcSurfaceContact
from solver_spike_geometry import surface_intersections


def inspect_contact_placement(canonical, *, activation_distance_m, minimum_distance_m, stiffness):
    import ipctk

    rest = np.asarray(canonical["restMeters"], dtype=float)
    placed = np.asarray(canonical["placedMeters"], dtype=float)
    raw_triangles = np.asarray(canonical["triangles"])
    if raw_triangles.ndim == 1 and len(raw_triangles) % 3 == 0:
        raw_triangles = raw_triangles.reshape((-1, 3))
    if len(rest) > 25000 or len(raw_triangles) > 50000:
        raise ValueError("Contact preflight mesh budget exceeded")
    contact = IpcSurfaceContact(rest, raw_triangles, activation_distance_m=activation_distance_m,
                                minimum_distance_m=minimum_distance_m, stiffness=stiffness)
    placed = contact._positions(placed)
    oracle = surface_intersections(placed, contact.faces)
    intersections = bool(ipctk.has_intersections(contact.mesh, placed))
    collisions = ipctk.NormalCollisions()
    collisions.build(contact.mesh, placed, activation_distance_m, minimum_distance_m)
    distance_squared = float(collisions.compute_minimum_distance(contact.mesh, placed))
    if np.isnan(distance_squared) or distance_squared < 0:
        raise ValueError("Invalid toolkit candidate distance")
    rejection = None
    try:
        contact.validate_state(placed)
    except ValueError as error:
        rejection = str(error)
    admissible = rejection is None and oracle["intersectingPairCount"] == 0 and not intersections
    return {"classification": "experimental-contact-placement-preflight", "accepted": False,
            "contactAdmissible": admissible, "rejection": rejection,
            "vertices": len(rest), "triangles": len(contact.faces), "contactProfile": contact.profile(),
            "independentSurfaceOracle": oracle, "toolkitHasIntersections": intersections,
            "minimumActiveCandidateDistanceM": float(np.sqrt(distance_squared)) if np.isfinite(distance_squared) else None,
            "distanceScope": "Minimum over toolkit active candidates only; null means no active candidates, not infinite global clearance.",
            "limitations": ["Placement admission only, not assembled geometry, material calibration or drape acceptance.",
                            "No seam exclusions; a zero-distance sewing target is incompatible with positive separation at that target.",
                            "Topologically adjacent contact, turning, layer order and continuous motion are not certified by this preflight."]}


def main():
    parser = argparse.ArgumentParser(description="Reject inadmissible saved cloth placements before contact simulation")
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-distance-m", type=float, required=True)
    parser.add_argument("--minimum-distance-m", type=float, required=True)
    parser.add_argument("--stiffness", type=float, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3, 4 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 ** 2, 64 * 1024 ** 2))
    if args.canonical.stat().st_size > 50 * 1024 ** 2:
        parser.error("Canonical geometry exceeds 50 MiB")
    captured = args.canonical.read_bytes()
    sources = {name: Path(__file__).with_name(name).read_bytes() for name in
               ("solver_contact_preflight.py", "solver_ipc_contact.py", "solver_spike_geometry.py",
                "solver-contact.requirements.txt", "solver-spike.requirements.txt")}
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir(mode=0o700)
    for name, content in sources.items():
        (snapshot / name).write_bytes(content)
    started = time.monotonic()
    try:
        report = inspect_contact_placement(json.loads(captured), activation_distance_m=args.activation_distance_m,
                                           minimum_distance_m=args.minimum_distance_m, stiffness=args.stiffness)
    except (KeyError, ValueError, TypeError, OverflowError) as error:
        report = {"classification": "rejected-contact-placement-preflight", "accepted": False,
                  "contactAdmissible": False, "rejection": str(error)}
    report.update({"canonicalDigest": hashlib.sha256(captured).hexdigest(),
                   "sourceDigests": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
                   "runtime": {name: importlib.metadata.version(name) for name in ("ipctk", "numpy", "scipy")},
                   "wallSeconds": time.monotonic() - started})
    if any(Path(__file__).with_name(name).read_bytes() != content for name, content in sources.items()):
        raise ValueError("Source changed during contact preflight")
    destination = args.output / "report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(destination.resolve())
    return 0 if report["contactAdmissible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
