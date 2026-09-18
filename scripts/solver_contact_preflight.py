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


def staging_seam_gaps(canonical, original, staged):
    bundle = canonical.get("embeddedConstraints")
    if bundle is None:
        return None
    if not isinstance(bundle, dict) or not isinstance(bundle.get("constraints"), list):
        raise ValueError("Embedded constraints must contain a constraint list")
    constraints = bundle["constraints"]
    if len(constraints) > 32768:
        raise ValueError("Staging seam constraint budget exceeded")
    ordered = sorted(canonical["instanceOffsets"].items(), key=lambda entry: entry[1])
    ranges = {identity: (start, ordered[index + 1][1] if index + 1 < len(ordered) else len(original))
              for index, (identity, start) in enumerate(ordered)}
    rows = []
    for constraint in constraints:
        terms = constraint.get("terms") if isinstance(constraint, dict) else None
        if not isinstance(terms, list) or not 2 <= len(terms) <= 6:
            raise ValueError("Each staging seam requires two to six terms")
        row = {}
        for term in terms:
            if not isinstance(term, dict) or not isinstance(term.get("instanceId"), str):
                raise ValueError("Staging seam term requires a physical instance")
            identity, vertex, coefficient = term["instanceId"], term.get("vertex"), term.get("coefficient")
            if identity not in ranges:
                raise ValueError("Unknown staging seam instance")
            start, end = ranges[identity]
            if type(vertex) is not int or not 0 <= vertex < end - start:
                raise ValueError("Staging seam vertex must be an instance-local integer")
            if (type(coefficient) not in (int, float) or not np.isfinite(coefficient)
                    or not 0 < abs(coefficient) <= 1):
                raise ValueError("Staging seam coefficient must be finite and normalized")
            if start + vertex in row:
                raise ValueError("Duplicate staging seam vertex")
            row[start + vertex] = coefficient
        if (abs(sum(row.values())) > 1e-7
                or abs(sum(value for value in row.values() if value > 0) - 1) > 1e-7):
            raise ValueError("Staging seams must subtract two normalized anchors")
        rows.append(row)
    if not rows:
        return None
    residuals = [np.asarray([np.linalg.norm(sum((weight * positions[vertex] for vertex, weight in row.items()),
                                               np.zeros(3))) * 1000 for row in rows])
                 for positions in (original, staged)]
    if not all(np.isfinite(values).all() for values in residuals):
        raise ValueError("Nonfinite staging seam diagnostics")
    return {"constraintCount": len(rows), "beforeMax": float(residuals[0].max()),
            "afterMax": float(residuals[1].max()), "beforeP95": float(np.percentile(residuals[0], 95)),
            "afterP95": float(np.percentile(residuals[1], 95)),
            "maximumIndividualIncrease": float(np.max(residuals[1] - residuals[0]))}


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
    parser.add_argument("--rigid-clearance-m", type=float)
    parser.add_argument("--max-translation-m", type=float, default=.25)
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
    if args.rigid_clearance_m is not None:
        sources["solver_rigid_staging.py"] = Path(__file__).with_name("solver_rigid_staging.py").read_bytes()
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir(mode=0o700)
    for name, content in sources.items():
        (snapshot / name).write_bytes(content)
    started = time.monotonic()
    staging = None
    try:
        canonical = json.loads(captured)
        if args.rigid_clearance_m is not None:
            from solver_rigid_staging import separate_rigid_instances

            if args.rigid_clearance_m <= args.minimum_distance_m:
                raise ValueError("Rigid clearance must exceed contact minimum separation")
            original = np.asarray(canonical["placedMeters"], dtype=float)
            rest = np.asarray(canonical["restMeters"], dtype=float)
            staged, staging = separate_rigid_instances(original, np.asarray(canonical["triangles"]).reshape((-1, 3)),
                canonical["instanceOffsets"], args.rigid_clearance_m, args.max_translation_m)
            if rest.shape != original.shape or not np.isfinite(rest).all() or np.max(np.abs(rest)) > 100:
                raise ValueError("Matching finite source rest geometry required")
            ordered = sorted(canonical["instanceOffsets"].items(), key=lambda entry: entry[1])
            for index, (_, start) in enumerate(ordered):
                end = ordered[index + 1][1] if index + 1 < len(ordered) else len(rest)
                source = rest[start:end] - rest[start:end].mean(axis=0)
                placed = original[start:end] - original[start:end].mean(axis=0)
                left, _, right = np.linalg.svd(source.T @ placed)
                rotation = left @ right
                if np.linalg.det(rotation) < 0:
                    left[:, -1] *= -1
                    rotation = left @ right
                if not np.allclose(source @ rotation, placed, rtol=0, atol=1e-10):
                    raise ValueError("Original placement is not rigid relative to source rest geometry")
            staging["sourceRigidCheckPassed"] = True
            seam_gaps = staging_seam_gaps(canonical, original, staged)
            if seam_gaps is not None:
                staging["seamGapsMm"] = seam_gaps
            canonical["placedMeters"] = staged.tolist()
            placement_bytes = json.dumps({"placedMeters": staged.tolist(), "staging": staging,
                                         "canonicalDigest": hashlib.sha256(captured).hexdigest()},
                                        allow_nan=False).encode()
            (args.output / "staged-placement.json").write_bytes(placement_bytes)
            staging["artifactSha256"] = hashlib.sha256(placement_bytes).hexdigest()
        report = inspect_contact_placement(canonical, activation_distance_m=args.activation_distance_m,
                                           minimum_distance_m=args.minimum_distance_m, stiffness=args.stiffness)
    except (KeyError, ValueError, TypeError, OverflowError, IndexError, np.linalg.LinAlgError) as error:
        report = {"classification": "rejected-contact-placement-preflight", "accepted": False,
                  "contactAdmissible": False, "rejection": str(error)}
    report.update({"rigidStaging": staging, "canonicalDigest": hashlib.sha256(captured).hexdigest(),
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
