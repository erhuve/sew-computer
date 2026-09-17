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
    parser = argparse.ArgumentParser(description="Source cut-cloth interior sewing coupling experiment")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--ramp-steps", type=int, default=0)
    parser.add_argument("--stitch-samples", type=int, default=5)
    parser.add_argument("--opposed-placement", action="store_true")
    parser.add_argument("--gather", action="store_true")
    parser.add_argument("--contact", action="store_true")
    parser.add_argument("--pointwise-contact", action="store_true")
    parser.add_argument("--seam-contact", action="store_true")
    parser.add_argument("--translated", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.steps <= 1000 or not 1 <= arguments.substeps <= 16:
        parser.error("steps must be 1..1000 and substeps 1..16")
    if not 2 <= arguments.stitch_samples <= 256:
        parser.error("stitch samples must be 2..256")
    if arguments.pointwise_contact and not arguments.contact:
        parser.error("pointwise contact requires --contact")
    if arguments.seam_contact and not arguments.pointwise_contact:
        parser.error("seam contact requires pointwise contact")
    if not 0 <= arguments.ramp_steps < arguments.steps:
        parser.error("ramp steps must be nonnegative and leave time for settling")
    arguments.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.environ["WARP_CACHE_PATH"] = str(arguments.output.resolve() / "kernel-cache")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3, 4 * 1024 ** 3))
    engine = Path(__file__).resolve().parents[1] / "services/engine"
    sources = {name: engine / name for name in ("embedded_constraints.py", "cloth_domain.py", "meshing.py", "shirt.py", "meshing_test.py", "simulation_validation.py", "assembly.py")}
    sources[Path(__file__).name] = Path(__file__)
    sources["solver_spike_geometry.py"] = Path(__file__).with_name("solver_spike_geometry.py")
    if arguments.pointwise_contact:
        sources["solver_point_contact.py"] = Path(__file__).with_name("solver_point_contact.py")
    if arguments.opposed_placement:
        sources["probe_placement.py"] = engine / "probe_placement.py"
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
    from cloth_domain import mesh_cloth_domain
    from embedded_constraints import build_embedded_constraints, validate_embedded_constraints, project_embedded_constraints, constraint_residuals
    from meshing_test import shirt_pattern
    from solver_spike_geometry import snapshot_particle_positions, state_finiteness, surface_intersections
    import newton
    import numpy as np
    import warp as wp

    started = time.monotonic()
    wp.init()
    wp.set_device("cpu")
    panels = {panel["id"]: panel for panel in shirt_pattern()["panels"]}
    templates = ("placket_left", "frill_left" if arguments.gather else "placket_left")
    instances = {}
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    slices = {}
    rest = {}
    placed = {}
    triangles = {}
    placement_frame = None
    for ordinal, template in enumerate(templates):
        identity = f"probe-{ordinal}"
        panel = panels[template]
        mesh = mesh_cloth_domain(panel, 30)
        instances[identity] = {"panel": panel, "mesh": mesh}
        local = np.array([[point[0] / 1000, point[1] / 1000, 0] for point in mesh["restPositions"]])
        faces = np.array(mesh["triangles"])
        offset = len(builder.particle_q)
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
                               vel=wp.vec3(0, 0, 0), vertices=local.tolist(), indices=faces.flatten().tolist(),
                               density=0.2, tri_ke=100000, tri_ka=100000, tri_kd=0.01,
                               edge_ke=0.01, edge_kd=0.001, particle_radius=0.0005)
        transformed = local.copy()
        if ordinal and arguments.opposed_placement:
            from probe_placement import opposed_path_frame, transform_probe_positions
            placement_frame = opposed_path_frame(instances["probe-0"]["mesh"], mesh)
            transformed = transform_probe_positions(local, placement_frame)
        elif ordinal:
            angle = 0.1
            transformed[:, 1] = local[:, 1] * math.cos(angle)
            transformed[:, 2] = local[:, 1] * math.sin(angle) + 0.005
        if arguments.translated:
            transformed += [0.3, -0.2, 0.4]
        builder.particle_q[offset:offset + len(local)] = [wp.vec3(*point) for point in transformed]
        rest[identity], placed[identity], triangles[identity] = local, transformed, faces
        slices[identity] = slice(offset, offset + len(local))
    registration = {"id": "interior-right-seam", "members": [], "sampleCount": arguments.stitch_samples, "complianceMPerN": 1e-8}
    for identity, source in instances.items():
        path = next(path for path in source["mesh"]["stitchPaths"] if path["name"] == "right")
        registration["members"].append({"instanceId": identity, "pathName": "right", "startArcMm": 0,
                                          "endArcMm": path["lengthMm"], "direction": "forward"})
    bundle = build_embedded_constraints(instances, [registration])
    validate_embedded_constraints(instances, bundle)
    builder.color(include_bending=True)
    model = builder.finalize(device="cpu")
    contact_options = {"particle_topological_contact_filter_threshold": 0, "particle_collision_detection_interval": 1} if arguments.pointwise_contact else {}
    solver = newton.solvers.SolverVBD(model, iterations=10, particle_enable_self_contact=arguments.contact,
                                    particle_self_contact_margin=0.0015, particle_self_contact_gap=0.0005, **contact_options)
    contact_report = None
    if arguments.pointwise_contact:
        from solver_point_contact import install_point_contact
        seam_options = {}
        if arguments.seam_contact:
            pairs = [{side: {"instanceId": sample["instanceId"], "restMeters": [value / 1000 for value in sample["restPositionMm"]]}
                      for side, sample in zip(("first", "second"), constraint["sourceSamples"])} for constraint in bundle["constraints"]]
            seam_options = {"seam_contact_pairs": pairs, "seam_radius": 0.0015}
        contact_report = install_point_contact(solver, np.concatenate(list(rest.values())), model.tri_indices.numpy(),
                                               [identity for identity, local in rest.items() for _ in local], arguments.output, 0.0015, **seam_options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state, next_state = model.state(), model.state()
    tensors = model.tri_poses.numpy().copy()
    inverse_masses = {identity: model.particle_inv_mass.numpy()[section] for identity, section in slices.items()}
    timestep = 1 / (240 * arguments.substeps)
    initial_residuals = constraint_residuals(bundle, placed)
    max_projection = 0.0
    for step in range(arguments.steps * arguments.substeps):
        previous = snapshot_particle_positions(state)
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, model.control(), contacts, timestep)
        candidate = next_state.particle_q.numpy()
        fraction = min(1.0, (step + 1) / (arguments.ramp_steps * arguments.substeps)) if arguments.ramp_steps else 1.0
        closure = fraction * fraction * (3 - 2 * fraction)
        projected = project_embedded_constraints(bundle, {identity: candidate[section] for identity, section in slices.items()},
                                                 inverse_masses, timestep, iterations=4,
                                                 target_offsets_m=initial_residuals * (1 - closure))
        max_projection = max(max_projection, projected["maxCorrectionM"])
        for identity, section in slices.items():
            candidate[section] = projected["positions"][identity]
        if not np.isfinite(candidate).all():
            raise ValueError("Nonfinite coupled sewing state")
        next_state.particle_q.assign(candidate)
        next_state.particle_qd.assign((candidate - previous) / timestep)
        if not state_finiteness(next_state.particle_q.numpy(), next_state.particle_qd.numpy())["finite"]:
            raise ValueError("Nonfinite coupled sewing positions or velocities")
        state, next_state = next_state, state
    final = {identity: state.particle_q.numpy()[section] for identity, section in slices.items()}
    ratios = []
    final_triangles = []
    for identity, faces in triangles.items():
        final_triangles.extend((faces + slices[identity].start).tolist())
        for edge in range(3):
            first, second = faces[:, edge], faces[:, (edge + 1) % 3]
            ratios.extend((np.linalg.norm(final[identity][first] - final[identity][second], axis=1) /
                           np.linalg.norm(rest[identity][first] - rest[identity][second], axis=1)).tolist())
    surface_report = surface_intersections(state.particle_q.numpy(), final_triangles)
    if digests != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}:
        raise ValueError("Source changed during experiment")
    report = {"classification": "experimental-embedded-sewing", "accepted": False, "sourceDigests": digests,
              "versions": {name: importlib.metadata.version(name) for name in ("newton", "warp-lang", "numpy", "shapely")},
              "steps": arguments.steps, "substeps": arguments.substeps, "timestepSeconds": timestep,
              "stitchSamples": arguments.stitch_samples, "opposedPlacement": placement_frame,
              "rampSteps": arguments.ramp_steps, "rampSchedule": "initial registration offsets to zero via smoothstep; cloth rest unchanged",
              "gather": arguments.gather, "contact": arguments.contact, "translated": arguments.translated,
              "pointwiseContact": contact_report,
              "seamContactExemptions": arguments.seam_contact,
              "initialResidualMaxMm": float(np.linalg.norm(initial_residuals, axis=1).max() * 1000),
              "finalResidualMaxMm": float(np.linalg.norm(constraint_residuals(bundle, final), axis=1).max() * 1000),
              "edgeRatioMin": min(ratios), "edgeRatioMax": max(ratios), "maxProjectionMm": max_projection * 1000,
              "restTensorsUnchanged": bool(np.array_equal(tensors, model.tri_poses.numpy())),
              "surfaceIntersections": surface_report,
              "maxSpeedMPerSecond": float(np.linalg.norm(state.particle_qd.numpy().astype(np.float64), axis=1).max()),
              "wallSeconds": time.monotonic() - started, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "limitations": ["Operator-split VBD cloth and XPBD sewing require timestep/convergence validation.",
                              "Projection follows collision handling and may reintroduce penetration.",
                              "Two actual source templates in a synthetic probe, not executable garment assembly.",
                              "No body, gravity, calibrated materials, binding wrap or turning acceptance."]}
    canonical = {"sources": instances, "constraints": bundle, "restMeters": {key: value.tolist() for key, value in rest.items()},
                 "positionsMeters": {key: value.tolist() for key, value in final.items()},
                 "previousPositionsMeters": {identity: previous[section].tolist() for identity, section in slices.items()},
                 "velocitiesMetersPerSecond": {identity: state.particle_qd.numpy()[section].tolist() for identity, section in slices.items()}}
    (arguments.output / "canonical.json").write_text(json.dumps(canonical, allow_nan=False))
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
