import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Full physical shirt numerical experiment; never accepted garment output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--quality-refinement", action="store_true")
    parser.add_argument("--shirt-placement", action="store_true")
    parser.add_argument("--fixture", choices=("full-shirt", "torso", "front-panel"), default="full-shirt")
    parser.add_argument("--no-sewing", action="store_true")
    parser.add_argument("--disable-contact", action="store_true")
    parser.add_argument("--rest-neighbor-filters", action="store_true")
    parser.add_argument("--pointwise-contact", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.steps <= 1000:
        parser.error("steps must be 1..1000")
    if arguments.disable_contact and arguments.rest_neighbor_filters:
        parser.error("rest-neighbor filters require contact")
    if arguments.pointwise_contact and (arguments.disable_contact or arguments.rest_neighbor_filters):
        parser.error("pointwise contact requires contact without static filters")
    arguments.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.environ["WARP_CACHE_PATH"] = str(arguments.output.resolve() / "kernel-cache")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3, 4 * 1024 ** 3))
    engine = Path(__file__).resolve().parents[1] / "services/engine"
    sources = {name: engine / name for name in ("assembly.py", "meshing.py", "shirt.py", "simulation_validation.py", "meshing_test.py", "placement.py")}
    sources[Path(__file__).name] = Path(__file__)
    sources["solver_spike_geometry.py"] = Path(__file__).with_name("solver_spike_geometry.py")
    if arguments.quality_refinement:
        sources["quality_meshing.py"] = engine / "quality_meshing.py"
    if arguments.rest_neighbor_filters:
        sources["solver_contact_filters.py"] = Path(__file__).with_name("solver_contact_filters.py")
    if arguments.pointwise_contact:
        sources["solver_point_contact.py"] = Path(__file__).with_name("solver_point_contact.py")
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    snapshot = arguments.output / "source-snapshot"
    snapshot.mkdir(mode=0o700)
    for name, path in sources.items():
        captured = path.read_bytes()
        if hashlib.sha256(captured).hexdigest() != digests[name]:
            raise ValueError("Source changed while capturing experiment")
        (snapshot / name).write_bytes(captured)
    (arguments.output / "run-input.json").write_text(json.dumps({"sourceDigests": digests, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}, "cpuLimitSeconds": 240, "memoryLimitBytes": 4 * 1024 ** 3}, indent=2) + "\n")
    sys.path.insert(0, str(engine))
    from assembly import compile_inventory, compile_assembly
    from meshing import mesh_panel
    from meshing_test import shirt_pattern, CONSTRUCTION
    from simulation_validation import validate_rest_mesh
    import newton
    import numpy as np
    import warp as wp
    from solver_spike_geometry import state_finiteness

    wp.init()
    wp.set_device("cpu")
    started = time.monotonic()
    source = json.dumps(shirt_pattern(), separators=(",", ":")).encode()
    pattern, inventory = compile_inventory(source, CONSTRUCTION)
    graph = compile_assembly(pattern, inventory)
    selected_instances = [instance for instance in inventory["instances"] if arguments.fixture == "full-shirt" or
                          (arguments.fixture == "torso" and instance["templateId"].startswith(("front_", "back_"))) or
                          (arguments.fixture == "front-panel" and instance["templateId"] == "front_left")]
    selected_ids = {instance["id"] for instance in selected_instances}
    operations = [operation for operation in graph["operations"] if all(participant["instanceId"] in selected_ids for participant in operation["participants"])]
    from placement import shirt_placement, place_rest_positions
    placement = shirt_placement(pattern, inventory["instances"])
    frames = {frame["instanceId"]: frame for frame in placement["frames"]}
    templates = {}
    for panel in pattern["panels"]:
        if panel["id"] not in {instance["templateId"] for instance in selected_instances}:
            continue
        print(json.dumps({"stage": "meshing", "templateId": panel["id"]}), flush=True)
        registrations = {}
        for operation in operations:
            for participant in operation["participants"]:
                if participant["instanceId"].split(":")[0] == panel["id"]:
                    registrations[participant["edgeName"]] = [0, 0.25, 0.5, 0.75, 1]
        mesh = mesh_panel(panel, 60, registrations, quality_refinement=arguments.quality_refinement)
        validate_rest_mesh(panel, mesh)
        templates[panel["id"]] = mesh
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    offsets = {}
    rest_vertices = []
    vertex_instance_ids = []
    placed_vertices = []
    indices = []
    for ordinal, instance in enumerate(selected_instances):
        mesh = templates[instance["templateId"]]
        hand = -1 if instance["mirrorX"] else 1
        rest = [[hand * point[0] / 1000, point[1] / 1000, 0] for point in mesh["restPositions"]]
        faces = [vertex for face in mesh["triangles"] for vertex in (reversed(face) if hand == -1 else face)]
        offset = len(builder.particle_q)
        offsets[instance["id"]] = offset
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0), vertices=rest, indices=faces, density=0.2, tri_ke=10000, tri_ka=10000, tri_kd=0.01, edge_ke=0.01, edge_kd=0.001, particle_radius=0.001, validate_mesh=True)
        angle = ordinal * 2 * math.pi / len(inventory["instances"])
        placed = [[point[0] * math.cos(angle) + 0.5 * math.cos(angle), point[0] * math.sin(angle) + 0.5 * math.sin(angle), 1 - point[1]] for point in rest]
        if arguments.shirt_placement:
            placed = [[value / 1000 for value in point] for point in place_rest_positions([[value * 1000 for value in point] for point in rest], frames[instance["id"]])]
        builder.particle_q[offset:offset + len(rest)] = [wp.vec3(*point) for point in placed]
        rest_vertices.extend(rest)
        vertex_instance_ids.extend([instance["id"]] * len(rest))
        placed_vertices.extend(placed)
        indices.extend(offset + vertex for vertex in faces)
    seams = []
    for operation in ([] if arguments.no_sewing else operations):
        anchors = []
        for participant in operation["participants"]:
            template = participant["instanceId"].split(":")[0]
            boundary = next(item for item in templates[template]["boundaries"] if item["name"] == participant["edgeName"])
            vertices = []
            for fraction in (0, 0.25, 0.5, 0.75, 1):
                selected = min(boundary["samples"], key=lambda sample: abs(sample["arcMm"] - fraction * boundary["lengthMm"]))
                if abs(selected["arcMm"] - fraction * boundary["lengthMm"]) > 1e-6:
                    raise ValueError("Registration anchor absent; refusing nearest-vertex substitution")
                vertices.append(offsets[participant["instanceId"]] + selected["vertex"])
            anchors.append(vertices)
        for vertices in anchors[1:]:
            if arguments.shirt_placement and operation["id"].startswith(("sleeve_back_", "underarm_")):
                vertices = list(reversed(vertices))
            for first, second in zip(anchors[0], vertices):
                if first != second:
                    seams.append((first, second))
    for first, second in sorted(set(seams)):
        builder.add_spring(first, second, ke=1000 if arguments.shirt_placement else 1000000, kd=0.01, control=0)
        builder.spring_rest_length[-1] = 0
    builder.particle_mass[0] = 0
    builder.color(include_bending=True)
    model = builder.finalize(device="cpu")
    contact_options = {}
    contact_report = None
    if arguments.rest_neighbor_filters:
        from solver_contact_filters import build_rest_neighbor_filters
        vertex_filters, edge_filters, contact_report = build_rest_neighbor_filters(rest_vertices, model.tri_indices.numpy(), model.edge_indices.numpy(), vertex_instance_ids, 0.003)
        contact_options = {"particle_external_vertex_contact_filtering_map": vertex_filters, "particle_external_edge_contact_filtering_map": edge_filters}
    if arguments.pointwise_contact:
        contact_options.update({"particle_topological_contact_filter_threshold": 0, "particle_collision_detection_interval": 1})
    solver = newton.solvers.SolverVBD(model, iterations=10, particle_enable_self_contact=not arguments.disable_contact, particle_self_contact_margin=0.003, particle_self_contact_gap=0.001, **contact_options)
    if arguments.pointwise_contact:
        from solver_point_contact import install_point_contact
        contact_report = install_point_contact(solver, rest_vertices, model.tri_indices.numpy(), vertex_instance_ids, arguments.output, 0.003)
    state = model.state()
    next_state = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    tensors = model.tri_poses.numpy().copy()
    for step in range(arguments.steps):
        if arguments.shirt_placement and seams:
            stiffness = 1000 + 999000 * min(1, step / max(1, arguments.steps - 1))
            model.spring_stiffness.assign(np.full(len(model.spring_stiffness), stiffness, dtype=np.float32))
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1 / 240)
        state, next_state = next_state, state
        positions, velocities = state.particle_q.numpy(), state.particle_qd.numpy()
        diagnostic = state_finiteness(positions, velocities)
        if not diagnostic["finite"]:
            if digests != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}:
                raise ValueError("Source changed during numerical experiment")
            failed_state = arguments.output / "failed-state.npz"
            np.savez_compressed(failed_state, positions=positions, velocities=velocities, rest=np.asarray(rest_vertices), triangles=np.asarray(indices).reshape((-1, 3)))
            failure = {"classification": "rejected-experimental-assembly", "accepted": False,
                       "error": "nonfinite-solver-state", "failedStep": step + 1, "requestedSteps": arguments.steps,
                       "state": diagnostic, "sourceDigests": digests, "patternDigest": inventory["patternDigest"],
                       "versions": {name: importlib.metadata.version(name) for name in (("newton", "warp-lang", "numpy", "shapely", "scipy") if arguments.quality_refinement else ("newton", "warp-lang", "numpy", "shapely"))},
                       "fixture": arguments.fixture, "qualityRefinement": arguments.quality_refinement,
                       "instances": len(selected_instances), "vertices": len(rest_vertices), "triangles": len(indices) // 3,
                       "instanceOffsets": offsets,
                       "contactFilterReport": contact_report,
                       "failedStateSha256": hashlib.sha256(failed_state.read_bytes()).hexdigest(),
                       "wallSeconds": time.monotonic() - started,
                       "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            (arguments.output / "report.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
            raise ValueError(f"Nonfinite solver state at step {step + 1}; rejected diagnostic saved")
    positions = state.particle_q.numpy().astype(np.float64)
    rest = np.asarray(rest_vertices)
    faces = np.asarray(indices).reshape((-1, 3))
    ratios = []
    for local in range(3):
        first, second = faces[:, local], faces[:, (local + 1) % 3]
        ratios.extend((np.linalg.norm(positions[first] - positions[second], axis=1) / np.linalg.norm(rest[first] - rest[second], axis=1)).tolist())
    residuals = [float(np.linalg.norm(positions[first] - positions[second]) * 1000) for first, second in seams]
    if digests != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}:
        raise ValueError("Source changed during numerical experiment")
    report = {"classification": "rejected-experimental-assembly", "accepted": False, "patternDigest": inventory["patternDigest"], "sourceDigests": digests,
              "versions": {name: importlib.metadata.version(name) for name in (("newton", "warp-lang", "numpy", "shapely", "scipy") if arguments.quality_refinement else ("newton", "warp-lang", "numpy", "shapely"))},
              "fixture": arguments.fixture, "selectedInstanceIds": sorted(selected_ids), "sewingEnabled": not arguments.no_sewing, "contactEnabled": not arguments.disable_contact,
              "contactFilterReport": contact_report,
              "instances": len(selected_instances), "vertices": len(positions), "triangles": len(faces), "constraints": len(set(seams)), "steps": arguments.steps, "qualityRefinement": arguments.quality_refinement, "placementRecipe": placement if arguments.shirt_placement else "radial-stress",
              "finite": bool(np.isfinite(positions).all()), "restTensorsUnchanged": bool(np.array_equal(tensors, model.tri_poses.numpy())),
              "edgeRatioMin": min(ratios), "edgeRatioMax": max(ratios), "seamGapMaxMm": max(residuals) if residuals else None, "seamGapP95Mm": float(np.percentile(residuals, 95)) if residuals else None,
              "maxDisplacementMm": float(np.linalg.norm(positions - np.asarray(placed_vertices), axis=1).max() * 1000),
              "maxSpeedMetersPerSecond": float(np.linalg.norm(state.particle_qd.numpy().astype(np.float64), axis=1).max()), "wallSeconds": time.monotonic() - started,
              "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "limitations": ["Rigid staging is unvalidated; no collision-free placement claim.", "Registration directions are experimental, not validated seam semantics.", "Localized button constraints, binding wraps, layer turning and interfacing are not implemented.", "No body, gravity, calibrated material, collision oracle or convergence acceptance."]}
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (arguments.output / "canonical.json").write_text(json.dumps({"restMeters": rest_vertices, "placedMeters": placed_vertices, "positionsMeters": positions.tolist(), "triangles": indices, "instanceOffsets": offsets, "inventory": inventory, "placement": placement if arguments.shirt_placement else "radial-stress", "sourceTemplates": templates, "assembly": graph}, separators=(",", ":"), allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
