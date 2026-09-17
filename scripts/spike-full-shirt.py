import argparse
import atexit
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time


def shirt_registration_direction(operation, participant):
    identity = operation["id"]
    template = participant["instanceId"].split(":")[0]
    edge = participant["edgeName"]
    reverse = (
        (identity.startswith("sleeve_back_") and template.startswith("back_"))
        or (identity.startswith("underarm_") and edge == "underarm_right")
        or (identity in ("collar_neck_1", "collar_neck_3", "collar_neck_5") and template != "collar_stand")
        or (identity == "collar_fall_attach" and template == "collar_fall")
    )
    return "reverse" if reverse else "forward"


def global_cpu_limit(signum, frame):
    raise FloatingPointError("Global reference CPU time limit exhausted")


def shirt_embedded_registrations(operations, sources):
    registrations = []
    for operation in operations:
        members = []
        for participant in operation["participants"]:
            source = sources[participant["instanceId"]]
            mesh = source["mesh"]
            edge = next(edge for edge in source["panel"]["draft"]["edges"] if edge["name"] == participant["edgeName"])
            path = next(path for path in mesh["stitchPaths"] if path["name"] == participant["edgeName"])
            if participant["sourcePointInterval"] != [edge["start"], edge["end"]] or participant["sourcePointInterval"] != [path["sourceStart"], path["sourceEnd"]]:
                raise ValueError("Assembly registration differs from source point interval")
            start, end = participant["intervalMm"]
            if type(start) not in (int, float) or type(end) not in (int, float) or participant["intervalMm"] != [0, edge["lengthMm"]] or not math.isfinite(end) or abs(end - path["lengthMm"]) > 1e-6:
                raise ValueError("Assembly registration is not the complete source path")
            members.append({"instanceId": participant["instanceId"], "pathName": participant["edgeName"],
                            "startArcMm": 0, "endArcMm": path["lengthMm"],
                            "direction": shirt_registration_direction(operation, participant)})
        registrations.append({"id": operation["id"], "members": members, "sampleCount": 5, "complianceMPerN": 1e-8})
    return registrations


def main():
    parser = argparse.ArgumentParser(description="Full physical shirt numerical experiment; never accepted garment output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--embedded-sewing", action="store_true")
    parser.add_argument("--coupled-sewing", action="store_true")
    parser.add_argument("--augmented-sewing", action="store_true")
    parser.add_argument("--global-reference", action="store_true")
    parser.add_argument("--membrane-only-control", action="store_true")
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--stable-membrane", action="store_true")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--ramp-steps", type=int, default=0)
    parser.add_argument("--quality-refinement", action="store_true")
    parser.add_argument("--shirt-placement", action="store_true")
    parser.add_argument("--torso-equilibrium-control", action="store_true")
    parser.add_argument("--torso-front-gap-mm", type=float, default=0)
    parser.add_argument("--fixture", choices=("full-shirt", "torso", "front-panel"), default="full-shirt")
    parser.add_argument("--no-sewing", action="store_true")
    parser.add_argument("--disable-contact", action="store_true")
    parser.add_argument("--rest-neighbor-filters", action="store_true")
    parser.add_argument("--pointwise-contact", action="store_true")
    parser.add_argument("--pin-first-vertex", action="store_true")
    arguments = parser.parse_args()
    if arguments.membrane_only_control and not arguments.disable_contact:
        parser.error("membrane-only diagnostic requires disabled contact")
    if arguments.global_reference and (not arguments.membrane_only_control or not arguments.embedded_sewing or arguments.no_sewing or arguments.coupled_sewing or arguments.fixture == "front-panel"):
        parser.error("global reference requires membrane-only embedded sewing without coupled sewing")
    if arguments.global_reference and arguments.stable_membrane:
        parser.error("global reference uses its own membrane evaluation")
    if arguments.torso_equilibrium_control and (arguments.fixture != "torso" or not arguments.disable_contact or not arguments.embedded_sewing or arguments.shirt_placement):
        parser.error("torso equilibrium control requires torso, embedded sewing, disabled contact and no shirt placement")
    if not math.isfinite(arguments.torso_front_gap_mm) or not 0 <= arguments.torso_front_gap_mm <= 100 or (arguments.torso_front_gap_mm and not arguments.torso_equilibrium_control):
        parser.error("torso front gap requires the equilibrium control and must be 0..100 mm")
    if not 1 <= arguments.solver_iterations <= 200:
        parser.error("solver iterations must be 1..200")
    if arguments.augmented_sewing and not arguments.coupled_sewing:
        parser.error("augmented sewing requires coupled sewing")
    if arguments.coupled_sewing and (not arguments.embedded_sewing or arguments.no_sewing or arguments.fixture == "front-panel"):
        parser.error("coupled sewing requires embedded sewing on a sewn fixture")
    if not 1 <= arguments.steps <= 1000:
        parser.error("steps must be 1..1000")
    if not 1 <= arguments.substeps <= 16:
        parser.error("substeps must be 1..16")
    if not 0 <= arguments.ramp_steps < arguments.steps:
        parser.error("ramp steps must be nonnegative and leave time for settling")
    if arguments.ramp_steps and not arguments.embedded_sewing:
        parser.error("staged offsets require embedded sewing")
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
    sources = {name: engine / name for name in ("assembly.py", "meshing.py", "shirt.py", "simulation_validation.py", "meshing_test.py", "placement.py", "constraint_coloring.py")}
    sources[Path(__file__).name] = Path(__file__)
    sources["solver_spike_geometry.py"] = Path(__file__).with_name("solver_spike_geometry.py")
    sources["solver_strain_diagnostics.py"] = Path(__file__).with_name("solver_strain_diagnostics.py")
    if arguments.torso_equilibrium_control:
        sources["solver_torso_control.py"] = Path(__file__).with_name("solver_torso_control.py")
    if arguments.embedded_sewing:
        sources.update({name: engine / name for name in ("cloth_domain.py", "embedded_constraints.py")})
    if arguments.stable_membrane or arguments.coupled_sewing:
        sources["solver_membrane_stability.py"] = Path(__file__).with_name("solver_membrane_stability.py")
    if arguments.coupled_sewing:
        sources["solver_embedded_sewing.py"] = Path(__file__).with_name("solver_embedded_sewing.py")
    if arguments.global_reference:
        for name in ("solver_global_sewing.py", "solver_membrane_hessian.py", "solver_embedded_sewing.py", "solver_membrane_stability.py"):
            sources[name] = Path(__file__).with_name(name)
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
    from solver_spike_geometry import snapshot_particle_positions, state_finiteness
    from solver_strain_diagnostics import edge_strain_report, mass_motion_report
    if arguments.embedded_sewing:
        from cloth_domain import mesh_cloth_domain, validate_cloth_domain
        from embedded_constraints import build_embedded_constraints, validate_embedded_constraints, project_embedded_constraints, constraint_residuals

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
        try:
            if arguments.embedded_sewing:
                mesh = mesh_cloth_domain(panel, 60, quality_refinement=arguments.quality_refinement)
                validate_cloth_domain(panel, mesh)
            else:
                mesh = mesh_panel(panel, 60, registrations, quality_refinement=arguments.quality_refinement)
                validate_rest_mesh(panel, mesh)
        except ValueError as error:
            if digests != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}:
                raise ValueError("Source changed during meshing experiment") from error
            failure = {"classification": "rejected-experimental-assembly", "accepted": False,
                       "error": "source-meshing-failed", "failureStage": "meshing", "failureDetail": str(error),
                       "failedTemplateId": panel["id"], "completedTemplates": sorted(templates),
                       "sourceDigests": digests, "patternDigest": inventory["patternDigest"],
                       "fixture": arguments.fixture, "selectedInstanceIds": sorted(selected_ids),
                       "qualityRefinement": arguments.quality_refinement, "embeddedSewing": arguments.embedded_sewing,
                       "wallSeconds": time.monotonic() - started, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            (arguments.output / "report.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
            raise
        templates[panel["id"]] = mesh
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    offsets = {}
    sections = {}
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
        sections[instance["id"]] = slice(offset, offset + len(rest))
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0), vertices=rest, indices=faces, density=0.2, tri_ke=10000, tri_ka=10000, tri_kd=0 if arguments.membrane_only_control else 0.01, edge_ke=0 if arguments.membrane_only_control else 0.01, edge_kd=0 if arguments.membrane_only_control else 0.001, particle_radius=0.001, validate_mesh=True)
        angle = ordinal * 2 * math.pi / len(inventory["instances"])
        placed = [[point[0] * math.cos(angle) + 0.5 * math.cos(angle), point[0] * math.sin(angle) + 0.5 * math.sin(angle), 1 - point[1]] for point in rest]
        if arguments.shirt_placement:
            placed = [[value / 1000 for value in point] for point in place_rest_positions([[value * 1000 for value in point] for point in rest], frames[instance["id"]])]
        builder.particle_q[offset:offset + len(rest)] = [wp.vec3(*point) for point in placed]
        rest_vertices.extend(rest)
        vertex_instance_ids.extend([instance["id"]] * len(rest))
        placed_vertices.extend(placed)
        indices.extend(offset + vertex for vertex in faces)
    if arguments.torso_equilibrium_control:
        from solver_torso_control import torso_equilibrium_placement
        controlled, placement = torso_equilibrium_placement(pattern, selected_instances, {identity: rest_vertices[section] for identity, section in sections.items()}, arguments.torso_front_gap_mm)
        for identity, section in sections.items():
            placed_vertices[section] = controlled[identity].tolist()
            builder.particle_q[section] = [wp.vec3(*point) for point in controlled[identity]]
    seams = []
    bundle = None
    embedded_sources = None
    if arguments.embedded_sewing:
        panels = {panel["id"]: panel for panel in pattern["panels"]}
        embedded_sources = {instance["id"]: {"panel": panels[instance["templateId"]], "mesh": templates[instance["templateId"]]} for instance in selected_instances}
        registrations = shirt_embedded_registrations([] if arguments.no_sewing else operations, embedded_sources)
        if registrations:
            bundle = build_embedded_constraints(embedded_sources, registrations)
            validate_embedded_constraints(embedded_sources, bundle)
    for operation in ([] if arguments.no_sewing or arguments.embedded_sewing else operations):
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
            if shirt_registration_direction(operation, participant) == "reverse":
                vertices.reverse()
            anchors.append(vertices)
        for vertices in anchors[1:]:
            for first, second in zip(anchors[0], vertices):
                if first != second:
                    seams.append((first, second))
    for first, second in sorted(set(seams)):
        builder.add_spring(first, second, ke=1000 if arguments.shirt_placement else 1000000, kd=0.01, control=0)
        builder.spring_rest_length[-1] = 0
    if arguments.pin_first_vertex:
        builder.particle_mass[0] = 0
    builder.color(include_bending=True)
    from constraint_coloring import refine_constraint_colors
    original_colors = {int(vertex): color for color, group in enumerate(builder.particle_color_groups) for vertex in group}
    coloring_report = {"originalColors": len(builder.particle_color_groups),
                       "originalSpringConflicts": sum(original_colors[first] == original_colors[second] for first, second in set(seams))}
    sewing_rows = [{offsets[term["instanceId"]] + term["vertex"]: term["coefficient"] for term in constraint["terms"]}
                   for constraint in bundle["constraints"]] if arguments.coupled_sewing or arguments.global_reference else []
    builder.set_coloring(refine_constraint_colors(len(rest_vertices), builder.particle_color_groups, [*sorted(set(seams)), *sewing_rows]))
    coloring_report["refinedColors"] = len(builder.particle_color_groups)
    model = builder.finalize(device="cpu")
    final_colors = model.particle_colors.numpy()
    if any(final_colors[first] == final_colors[second] for first, second in set(seams)):
        raise ValueError("Sewing constraints conflict with particle update colors")
    contact_options = {}
    contact_report = None
    if arguments.rest_neighbor_filters:
        from solver_contact_filters import build_rest_neighbor_filters
        vertex_filters, edge_filters, contact_report = build_rest_neighbor_filters(rest_vertices, model.tri_indices.numpy(), model.edge_indices.numpy(), vertex_instance_ids, 0.003)
        contact_options = {"particle_external_vertex_contact_filtering_map": vertex_filters, "particle_external_edge_contact_filtering_map": edge_filters}
    if arguments.pointwise_contact:
        contact_options.update({"particle_topological_contact_filter_threshold": 0, "particle_collision_detection_interval": 1})
    solver = newton.solvers.SolverVBD(model, iterations=arguments.solver_iterations, particle_enable_self_contact=not arguments.disable_contact, particle_self_contact_margin=0.003, particle_self_contact_gap=0.001, **contact_options)
    membrane_report = None
    if arguments.stable_membrane and not arguments.coupled_sewing:
        from solver_membrane_stability import install_stable_membrane, uninstall_stable_membrane
        membrane_report = install_stable_membrane(solver, arguments.output)
        atexit.register(uninstall_stable_membrane, solver)
    if arguments.pointwise_contact:
        from solver_point_contact import install_point_contact
        contact_report = install_point_contact(solver, rest_vertices, model.tri_indices.numpy(), vertex_instance_ids, arguments.output, 0.003)
    sewing_report = None
    global_solver = None
    global_step_report = None
    if arguments.global_reference:
        from solver_global_sewing import GlobalSewingSolver
        compliances = {constraint["complianceMPerN"] for constraint in bundle["constraints"]}
        if len(compliances) != 1:
            raise ValueError("Global reference requires uniform sewing compliance")
        global_solver = GlobalSewingSolver(model, sewing_rows, compliances.pop())
        global_positions = np.asarray(placed_vertices, dtype=float)
        global_velocities = np.zeros_like(global_positions)
    if arguments.coupled_sewing:
        from solver_embedded_sewing import install_embedded_sewing, set_embedded_targets, uninstall_embedded_sewing
        compliances = {constraint["complianceMPerN"] for constraint in bundle["constraints"]}
        if len(compliances) != 1:
            raise ValueError("Coupled research adapter requires uniform captured sewing compliance")
        sewing_report = install_embedded_sewing(solver, sewing_rows, compliances.pop(), arguments.output, augmented=arguments.augmented_sewing)
        atexit.register(uninstall_embedded_sewing, solver)
    state = model.state()
    next_state = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    tensors = model.tri_poses.numpy().copy()
    timestep = 1 / (240 * arguments.substeps)
    inverse_masses = {identity: model.particle_inv_mass.numpy()[section] for identity, section in sections.items()}
    initial_residuals = constraint_residuals(bundle, {identity: np.asarray(placed_vertices)[section] for identity, section in sections.items()}) if bundle else None
    max_projection = 0.0
    coupling_report = {"mode": "global-membrane-reference" if arguments.global_reference else "coupled-embedded-cut-cloth" if arguments.coupled_sewing else "embedded-cut-cloth" if arguments.embedded_sewing else "boundary-springs", "substeps": arguments.substeps,
                       "membraneOnlyControl": arguments.membrane_only_control,
                       "globalReference": arguments.global_reference,
                       "solverIterations": arguments.solver_iterations, "sewingSolver": sewing_report,
                       "pinnedVertices": np.flatnonzero(model.particle_inv_mass.numpy() == 0).tolist(),
                       "registrationRecipe": "sew-shirt-source-endpoints/1",
                       "timestepSeconds": timestep, "rampSteps": arguments.ramp_steps,
                       "sourceEndpointPolicy": "Exact source edge metadata and point intervals; metadata arc discrepancy limited to 1e-6 mm before using source-derived path length" if arguments.embedded_sewing else None,
                       "rampSchedule": "initial registration offsets to zero via smoothstep; cloth rest unchanged" if bundle else None,
                       "constraintCount": len(bundle["constraints"]) if bundle else len(set(seams))}
    from solver_strain_diagnostics import membrane_energy_report
    trajectory_path = arguments.output / "trajectory.jsonl"
    membrane_inputs = (model.tri_indices.numpy().copy(), tensors, model.tri_areas.numpy().copy(), model.tri_materials.numpy().copy())
    particle_masses = model.particle_mass.numpy().astype(float)
    source_geometry = arguments.output / "source-geometry.json"
    source_geometry.write_text(json.dumps({"restMeters": rest_vertices, "placedMeters": placed_vertices,
        "triangles": indices, "instanceOffsets": offsets, "inventory": inventory, "assembly": graph,
        "sourceTemplates": templates, "embeddedConstraints": bundle, "coupling": coupling_report}, separators=(",", ":"), allow_nan=False))
    geometry_digest = hashlib.sha256(source_geometry.read_bytes()).hexdigest()
    unconverged_substeps = []
    if arguments.global_reference:
        signal.signal(signal.SIGXCPU, global_cpu_limit)
    for step in range(arguments.steps * arguments.substeps):
        if arguments.shirt_placement and seams:
            stiffness = 1000 + 999000 * min(1, step / max(1, arguments.steps * arguments.substeps - 1))
            model.spring_stiffness.assign(np.full(len(model.spring_stiffness), stiffness, dtype=np.float32))
        previous = global_positions.copy() if arguments.global_reference else snapshot_particle_positions(state) if bundle else None
        fraction = min(1.0, (step + 1) / (arguments.ramp_steps * arguments.substeps)) if arguments.ramp_steps else 1.0
        closure = fraction * fraction * (3 - 2 * fraction)
        if arguments.coupled_sewing:
            set_embedded_targets(solver, initial_residuals * (1 - closure))
        state.clear_forces()
        pipeline.collide(state, contacts)
        global_error = None
        if arguments.global_reference:
            try:
                global_positions, global_velocities, global_step_report = global_solver.step(global_positions, global_velocities, initial_residuals * (1 - closure), timestep)
                if not global_step_report["converged"]:
                    unconverged_substeps.append(step + 1)
            except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
                global_error = str(error)
                global_step_report = None
                unconverged_substeps.append(step + 1)
            next_state.particle_q.assign(global_positions.astype(np.float32))
            next_state.particle_qd.assign(global_velocities.astype(np.float32))
        else:
            solver.step(state, next_state, control, contacts, timestep)
        projection_error = None
        failure_stage = "cloth-step"
        if bundle and not arguments.coupled_sewing and not arguments.global_reference and state_finiteness(next_state.particle_q.numpy(), next_state.particle_qd.numpy())["finite"]:
            candidate = next_state.particle_q.numpy()
            failure_stage = "embedded-projection"
            try:
                projected = project_embedded_constraints(bundle, {identity: candidate[section] for identity, section in sections.items()},
                    inverse_masses, timestep, iterations=4, target_offsets_m=initial_residuals * (1 - closure))
                max_projection = max(max_projection, projected["maxCorrectionM"])
                for identity, section in sections.items():
                    candidate[section] = projected["positions"][identity]
                next_state.particle_q.assign(candidate)
                next_state.particle_qd.assign((candidate - previous) / timestep)
            except (ValueError, FloatingPointError) as error:
                projection_error = str(error)
        state, next_state = next_state, state
        positions, velocities = (global_positions, global_velocities) if arguments.global_reference else (state.particle_q.numpy(), state.particle_qd.numpy())
        diagnostic = state_finiteness(positions, velocities)
        if diagnostic["finite"] and not global_error:
            motion = mass_motion_report(placed_vertices, positions, velocities, particle_masses)
            energy = membrane_energy_report(positions, *membrane_inputs)
            target_residuals = constraint_residuals(bundle, {identity: positions[section] for identity, section in sections.items()}) - initial_residuals * (1 - closure) if bundle else None
            entry = {"substep": step + 1, "timeSeconds": (step + 1) * timestep, "closure": closure,
                     "globalSolve": global_step_report,
                     "membraneJoules": energy["joules"], "kineticJoules": float(np.sum(particle_masses[:, None] * velocities.astype(float) ** 2) / 2),
                     "centerOfMassDisplacementMm": motion["centerOfMassDisplacementMm"], "centerOfMassSpeedMetersPerSecond": motion["centerOfMassSpeedMetersPerSecond"],
                     "targetResidualMaxMm": float(np.linalg.norm(target_residuals, axis=1).max() * 1000) if bundle else None,
                     "scope": "Partial energy diagnostics; excludes bending, damping, contact and sewing energy. Moving sewing targets perform work."}
            with trajectory_path.open("a") as trajectory:
                trajectory.write(json.dumps(entry, allow_nan=False) + "\n")
        if not diagnostic["finite"] or projection_error or global_error:
            if digests != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}:
                raise ValueError("Source changed during numerical experiment")
            failed_state = arguments.output / "failed-state.npz"
            np.savez_compressed(failed_state, positions=positions, velocities=velocities, rest=np.asarray(rest_vertices), triangles=np.asarray(indices).reshape((-1, 3)))
            failure = {"classification": "rejected-experimental-assembly", "accepted": False,
                       "error": "global-solve-failed" if global_error else "embedded-projection-failed" if projection_error else "nonfinite-solver-state",
                       "failureDetail": global_error or projection_error, "failureStage": "global-solve" if global_error else failure_stage,
                       "failedStateMeaning": "previous valid state before failed global solve" if global_error else "failed candidate state",
                       "globalReferenceAllStepsConverged": False if arguments.global_reference else None,
                       "globalReferenceUnconvergedSubsteps": unconverged_substeps if arguments.global_reference else None,
                       "failedStep": step // arguments.substeps + 1, "failedSubstep": step % arguments.substeps + 1, "requestedSteps": arguments.steps,
                       "state": diagnostic, "sourceDigests": digests, "patternDigest": inventory["patternDigest"],
                       "versions": {name: importlib.metadata.version(name) for name in (("newton", "warp-lang", "numpy", "shapely", "scipy") if arguments.quality_refinement or arguments.global_reference else ("newton", "warp-lang", "numpy", "shapely"))},
                       "fixture": arguments.fixture, "qualityRefinement": arguments.quality_refinement,
                       "instances": len(selected_instances), "vertices": len(rest_vertices), "triangles": len(indices) // 3,
                       "instanceOffsets": offsets,
                       "contactFilterReport": contact_report,
                       "constraintColoring": coloring_report,
                       "membraneStability": membrane_report,
                       "coupling": coupling_report, "restTensorsUnchanged": bool(np.array_equal(tensors, model.tri_poses.numpy())),
                       "sourceGeometrySha256": geometry_digest,
                       "failedStateSha256": hashlib.sha256(failed_state.read_bytes()).hexdigest(),
                       "wallSeconds": time.monotonic() - started,
                       "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            (arguments.output / "report.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
            raise ValueError(f"Solver rejected at step {step // arguments.substeps + 1}, substep {step % arguments.substeps + 1}; diagnostic saved")
    positions = global_positions if arguments.global_reference else state.particle_q.numpy().astype(np.float64)
    rest = np.asarray(rest_vertices)
    faces = np.asarray(indices).reshape((-1, 3))
    ratios = []
    for local in range(3):
        first, second = faces[:, local], faces[:, (local + 1) % 3]
        ratios.extend((np.linalg.norm(positions[first] - positions[second], axis=1) / np.linalg.norm(rest[first] - rest[second], axis=1)).tolist())
    residuals = [float(np.linalg.norm(positions[first] - positions[second]) * 1000) for first, second in seams]
    if bundle:
        residuals = (np.linalg.norm(constraint_residuals(bundle, {identity: positions[section] for identity, section in sections.items()}), axis=1) * 1000).tolist()
        coupling_report.update({"initialResidualMaxMm": float(np.linalg.norm(initial_residuals, axis=1).max() * 1000), "maxProjectionMm": max_projection * 1000})
    if digests != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}:
        raise ValueError("Source changed during numerical experiment")
    report = {"classification": "rejected-experimental-assembly", "accepted": False, "patternDigest": inventory["patternDigest"], "sourceDigests": digests,
              "versions": {name: importlib.metadata.version(name) for name in (("newton", "warp-lang", "numpy", "shapely", "scipy") if arguments.quality_refinement or arguments.global_reference else ("newton", "warp-lang", "numpy", "shapely"))},
              "fixture": arguments.fixture, "selectedInstanceIds": sorted(selected_ids), "sewingEnabled": not arguments.no_sewing, "contactEnabled": not arguments.disable_contact,
              "contactFilterReport": contact_report,
              "constraintColoring": coloring_report,
              "membraneStability": membrane_report,
              "coupling": coupling_report,
              "sourceGeometrySha256": geometry_digest,
              "trajectorySha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
              "instances": len(selected_instances), "vertices": len(positions), "triangles": len(faces), "constraints": coupling_report["constraintCount"], "steps": arguments.steps, "qualityRefinement": arguments.quality_refinement, "placementRecipe": placement if arguments.shirt_placement or arguments.torso_equilibrium_control else "radial-stress",
              "finite": bool(np.isfinite(positions).all()), "restTensorsUnchanged": bool(np.array_equal(tensors, model.tri_poses.numpy())),
              "edgeRatioMin": min(ratios), "edgeRatioMax": max(ratios), "seamGapMaxMm": max(residuals) if residuals else None, "seamGapP95Mm": float(np.percentile(residuals, 95)) if residuals else None,
              "edgeStrain": edge_strain_report(rest, positions, faces, vertex_instance_ids, coupling_report["pinnedVertices"]),
              "massMotion": mass_motion_report(placed_vertices, positions, velocities, model.particle_mass.numpy()),
              "maxDisplacementMm": float(np.linalg.norm(positions - np.asarray(placed_vertices), axis=1).max() * 1000),
              "maxSpeedMetersPerSecond": float(np.linalg.norm(velocities.astype(np.float64), axis=1).max()), "wallSeconds": time.monotonic() - started,
              "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "limitations": ["Rigid staging is unvalidated; no collision-free placement claim.", "Registration directions are experimental, not validated seam semantics.", "Localized button constraints, binding wraps, layer turning and interfacing are not implemented.", "No body, gravity, calibrated material, collision oracle or convergence acceptance.", "All sewing operations ramp simultaneously; graph dependency sequence is not executed.", "Embedded projection follows collision handling and may reintroduce penetration; operator splitting needs convergence validation."]}
    if arguments.coupled_sewing:
        report["limitations"][-1] = "Sewing energy shares cloth/contact updates; convergence, contact thickness and assembly semantics remain unvalidated."
    if arguments.global_reference:
        report["limitations"][-1] = "Global membrane reference excludes bending, damping and all contact; it is not a garment solver."
        report["globalReferenceFinalStep"] = global_step_report
        report["globalReferenceAllStepsConverged"] = not unconverged_substeps
        report["globalReferenceUnconvergedSubsteps"] = unconverged_substeps
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (arguments.output / "canonical.json").write_text(json.dumps({"restMeters": rest_vertices, "placedMeters": placed_vertices, "positionsMeters": positions.tolist(), "previousPositionsMeters": previous.tolist() if bundle else None, "velocitiesMetersPerSecond": velocities.tolist(), "particleMassesKg": model.particle_mass.numpy().tolist(), "triangles": indices, "instanceOffsets": offsets, "inventory": inventory, "placement": placement if arguments.shirt_placement or arguments.torso_equilibrium_control else "radial-stress", "sourceTemplates": templates, "assembly": graph, "embeddedConstraints": bundle}, separators=(",", ":"), allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
