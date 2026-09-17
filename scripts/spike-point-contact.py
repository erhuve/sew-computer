import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

import newton
import numpy as np
import warp as wp

from solver_point_contact import install_point_contact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--quality-refinement", action="store_true")
    parser.add_argument("--disable-contact", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.steps <= 120:
        parser.error("steps must be 1..120")
    resource.setrlimit(resource.RLIMIT_CPU, (150, 155))
    arguments.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    engine = Path(__file__).resolve().parents[1] / "services/engine"
    source_paths = [Path(__file__), Path(__file__).with_name("solver_point_contact.py"), *sorted(engine.glob("*.py"))]
    source_snapshot = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    snapshot_directory = arguments.output / "source-snapshot"
    snapshot_directory.mkdir()
    for path in source_paths:
        captured_source = path.read_bytes()
        if hashlib.sha256(captured_source).hexdigest() != source_snapshot[str(path)]:
            raise ValueError("Source changed while capturing experiment")
        (snapshot_directory / path.name).write_bytes(captured_source)
    (arguments.output / "source-snapshot.json").write_text(json.dumps(source_snapshot, indent=2) + "\n")
    canonical_bytes = arguments.canonical.read_bytes()
    (arguments.output / "input-canonical.json").write_bytes(canonical_bytes)
    request = {"canonicalDigest": hashlib.sha256(canonical_bytes).hexdigest(), "steps": arguments.steps, "qualityRefinement": arguments.quality_refinement, "contactEnabled": not arguments.disable_contact, "accepted": False,
               "versions": {name: importlib.metadata.version(name) for name in ("newton", "warp-lang", "numpy", "shapely", "scipy")}}
    (arguments.output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    captured = json.loads(canonical_bytes)
    rest = captured["restMeters"]
    faces = np.asarray(captured["triangles"]).reshape((-1, 3))
    placed = captured["placedMeters"]
    if arguments.quality_refinement:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/engine"))
        from meshing import mesh_panel
        from meshing_test import shirt_pattern
        from placement import place_rest_positions

        pattern = shirt_pattern()
        if hashlib.sha256(json.dumps(pattern, separators=(",", ":")).encode()).hexdigest() != captured["inventory"]["patternDigest"]:
            raise ValueError("Synthetic source pattern no longer matches captured control")
        if set(captured["instanceOffsets"]) != {"front_left:shell"}:
            raise ValueError("Quality control is scoped to original front-left fixture")
        panel = next(panel for panel in pattern["panels"] if panel["id"] == "front_left")
        mesh = mesh_panel(panel, 60, {}, quality_refinement=True)
        rest = [[point[0] / 1000, point[1] / 1000, 0] for point in mesh["restPositions"]]
        faces = np.asarray(mesh["triangles"])
        frame = next(frame for frame in captured["placement"]["frames"] if frame["instanceId"] == "front_left:shell")
        placed = [[value / 1000 for value in point] for point in place_rest_positions([[value * 1000 for value in point] for point in rest], frame)]
    wp.init()
    wp.set_device("cpu")
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0), vertices=rest, indices=faces.flatten().tolist(), density=0.2, tri_ke=10000, tri_ka=10000, tri_kd=0.01, edge_ke=0.01, edge_kd=0.001, particle_radius=0.001)
    builder.particle_q[:] = [wp.vec3(*point) for point in placed]
    builder.particle_mass[0] = 0
    builder.color(include_bending=True)
    model = builder.finalize(device="cpu")
    np.savez_compressed(arguments.output / "input-geometry.npz", rest=np.asarray(rest), placed=np.asarray(placed), triangles=faces)
    solver = newton.solvers.SolverVBD(model, iterations=10, particle_enable_self_contact=not arguments.disable_contact, particle_collision_detection_interval=1, particle_topological_contact_filter_threshold=0, particle_self_contact_margin=0.003, particle_self_contact_gap=0.001)
    profile = {"profile": "contact-disabled-control", "accepted": False} if arguments.disable_contact else install_point_contact(solver, rest, faces, ["panel"] * len(rest), arguments.output, 0.003)
    (arguments.output / "run-input.json").write_text(json.dumps({**profile, "steps": arguments.steps, "qualityRefinement": arguments.quality_refinement, "contactEnabled": not arguments.disable_contact, "vertices": len(rest), "triangles": len(faces)}, indent=2) + "\n")
    state, next_state = model.state(), model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()
    for _ in range(arguments.steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1 / 240)
        state, next_state = next_state, state
    positions = state.particle_q.numpy()
    rest = np.asarray(rest)
    ratios = np.concatenate([np.linalg.norm(positions[faces[:, corner]] - positions[faces[:, (corner + 1) % 3]], axis=1) / np.linalg.norm(rest[faces[:, corner]] - rest[faces[:, (corner + 1) % 3]], axis=1) for corner in range(3)])
    if source_snapshot != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}:
        raise ValueError("Source changed during point-contact experiment")
    report = {**profile, **request, "capturedSourceDigests": source_snapshot, "vertices": len(rest), "triangles": len(faces), "maxDisplacementMm": float(np.linalg.norm(positions - np.asarray(placed), axis=1).max() * 1000), "edgeRatioMin": float(ratios.min()), "edgeRatioMax": float(ratios.max()), "finite": bool(np.isfinite(positions).all()), "wallSeconds": time.monotonic() - started}
    np.savez_compressed(arguments.output / "geometry.npz", rest=rest, placed=np.asarray(placed), positions=positions, triangles=faces)
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
