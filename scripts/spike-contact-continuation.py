import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from solver_process_budget import (ProgressStore, arm_parent_death, atomic_bytes, atomic_json,
                                   captured_source_path, prepare_worker, read_regular, recover_progress, supervise)
from solver_attempt_journal import strict_loads


def parse_arguments():
    parser = argparse.ArgumentParser(description="Bounded saved-source contact continuation research; never garment acceptance")
    for name in ("canonical", "placement", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("activation-distance-m", "minimum-distance-m", "pressure-pa", "target-fraction"):
        parser.add_argument(f"--{name}", type=float, required=True)
    parser.add_argument("--subdivisions", type=int, default=8)
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--cpu-limit-seconds", type=int, default=240)
    parser.add_argument("--wall-limit-seconds", type=int, default=480,
                        help="Parent-enforced elapsed worker budget (default: 480; range: 1–7200)")
    parser.add_argument("--step-seconds", type=float, default=1 / 240)
    parser.add_argument("--sewing-mode", choices=("vector", "distance", "normal-offset"), default="vector",
                        help="Scalar distance, declared source-normal offset, or legacy world-space vector sewing")
    parser.add_argument("--fold-actuation", action="store_true",
                        help="Execute the explicit captured source-hinge angle schedule alongside sewing")
    parser.add_argument("--assembly-schedule", action="store_true",
                        help="Execute captured piecewise sewing/fold progress without resetting cloth state")
    parser.add_argument("--sewing-activation", action="store_true",
                        help="Execute explicit source-bound per-row seam engagement and target controls")
    parser.add_argument("--material-grippers", action="store_true",
                        help="Execute captured compliant material-point targets with engagement/release work accounting")
    parser.add_argument("--contact-model", choices=("area-improved-max", "rest-filtered"),
                        default="area-improved-max")
    parser.add_argument("--ccd-profile", choices=("tight-inclusion", "swept-plane-tight-inclusion",
                                                   "temporal-separation-tight-inclusion"),
                        default="tight-inclusion")
    args = parser.parse_args()
    if args.contact_model == "rest-filtered" and args.ccd_profile == "swept-plane-tight-inclusion":
        parser.error("Rest-filtered research supports tight-inclusion or temporal-separation-tight-inclusion")
    if (not all(math.isfinite(value) and value > 0 for value in
                (args.activation_distance_m, args.minimum_distance_m, args.pressure_pa, args.step_seconds))
            or not math.isfinite(args.target_fraction) or not 0 < args.target_fraction <= 1):
        parser.error("Positive finite contact parameters and duration; target fraction in (0, 1] required")
    if (not 2 <= args.max_evaluations <= 10000 or not 1 <= args.max_attempts <= 4096
            or not 1 <= args.cpu_limit_seconds <= 3600 or not 1 <= args.wall_limit_seconds <= 7200
            or not 0 <= args.max_depth <= 30
            or not 1 <= args.subdivisions <= args.max_attempts
            or args.subdivisions & (args.subdivisions - 1)):
        parser.error("Bounded evaluation, attempt, CPU, wall and depth budgets and dyadic subdivisions required")
    if (args.material_grippers or args.sewing_activation) and args.subdivisions.bit_length() - 1 + args.max_depth > 40:
        parser.error("Captured control subdivision fractions must remain within the 2^40 dyadic bound")
    if args.sewing_activation and args.target_fraction != 1:
        parser.error("Explicit sewing controls require target-fraction 1; targets come from the captured recipe")
    return args


def bounded_input(path):
    return read_regular(path, 50 * 1024 ** 2)


def run_worker(output, parent_pid):
    arm_parent_death(parent_pid)
    payload, _, errors = recover_progress(output)
    if errors:
        raise ValueError("Worker requires intact initial capture")
    report = payload["report"]
    args = argparse.Namespace(**report["arguments"])
    args.output = output
    prepare_worker(args.cpu_limit_seconds, parent_pid)
    progress = ProgressStore(output)
    started, cpu_started = time.monotonic(), time.process_time()

    def cpu_timeout(signum, frame):
        raise TimeoutError("Contact continuation CPU budget exhausted")

    signal.signal(signal.SIGXCPU, cpu_timeout)
    try:
        for name, digest in report["sourceDigests"].items():
            if hashlib.sha256(read_regular(captured_source_path(output, name))).hexdigest() != digest:
                raise ValueError(f"Captured source hash mismatch: {name}")
        canonical_bytes = bounded_input(output / "canonical.json")
        placement_bytes = bounded_input(output / "placement.json")
        canonical_digest = hashlib.sha256(canonical_bytes).hexdigest()
        if (canonical_digest != report["canonicalSha256"]
                or hashlib.sha256(placement_bytes).hexdigest() != report["placementSha256"]):
            raise ValueError("Captured input hash mismatch")
        progress.save(report, "validating-inputs")
        source, placement = strict_loads(canonical_bytes), strict_loads(placement_bytes)
        fold_recipe = source.get("foldActuation")
        if args.fold_actuation != isinstance(fold_recipe, dict):
            raise ValueError("Fold actuation requires both explicit opt-in and a captured recipe")
        gripper_input = source.get("gripperActuation")
        if args.material_grippers != ("gripperActuation" in source):
            raise ValueError("Material grippers require both explicit opt-in and a captured recipe")
        sewing_input = source.get("sewingActuation")
        if args.sewing_activation != ("sewingActuation" in source):
            raise ValueError("Sewing activation requires both explicit opt-in and a captured recipe")
        schedule_recipe = source.get("assemblySchedule")
        if args.assembly_schedule != isinstance(schedule_recipe, dict):
            raise ValueError("Assembly schedule requires both explicit opt-in and a captured recipe")
        schedule = None
        if args.assembly_schedule:
            from solver_assembly_schedule import AssemblySchedule
            if not args.fold_actuation:
                raise ValueError("Assembly schedule requires explicit fold actuation")
            schedule = AssemblySchedule(schedule_recipe, args.subdivisions)
            report["assemblySchedule"] = schedule_recipe
        if placement.get("canonicalDigest") != canonical_digest:
            raise ValueError("Staged placement canonicalDigest does not match captured canonical bytes")
        import newton
        import numpy as np
        import warp as wp
        import ipctk
        from solver_adaptive_contact import adaptive_contact_step
        from solver_attempt_journal import AttemptJournal, recover_attempt_journal
        from solver_contact_preflight import staging_seam_gaps
        from solver_global_sewing import GlobalSewingSolver
        from solver_ipc_contact import IpcSurfaceContact
        from solver_ipc_broad_phase import contact_broad_phase
        from solver_spike_geometry import surface_intersections
        from solver_strain_diagnostics import edge_strain_report, mass_motion_report

        progress.save(report, "contact-preflight")
        ipctk.set_num_threads(1)
        rest = np.asarray(source["restMeters"], dtype=float)
        faces = np.asarray(source["triangles"])
        if faces.ndim == 1 and faces.size % 3 == 0:
            faces = faces.reshape((-1, 3))
        positions = np.asarray(placement["placedMeters"], dtype=float)
        if (rest.ndim != 2 or rest.shape[1] != 3 or not 3 <= len(rest) <= 25000
                or positions.shape != rest.shape or not np.isfinite(positions).all()
                or np.max(np.abs(positions)) > 100 or faces.ndim != 2 or faces.shape[1] != 3
                or not 1 <= len(faces) <= 50000):
            raise ValueError("Bounded matching source and staged geometry required")
        contact_parameters = dict(activation_distance_m=args.activation_distance_m,
                                  minimum_distance_m=args.minimum_distance_m, stiffness=args.pressure_pa)
        if args.contact_model == "rest-filtered":
            from solver_rest_filtered_contact import RestFilteredSurfaceContact
            contact = RestFilteredSurfaceContact(rest, faces, **contact_parameters,
                                                  ccd_profile=args.ccd_profile)
        else:
            contact = IpcSurfaceContact(rest, faces, **contact_parameters,
                                       energy_profile="area-improved-max", ccd_profile=args.ccd_profile)
        offsets = source["instanceOffsets"]
        if (not isinstance(offsets, dict) or not (1 if args.fold_actuation or args.material_grippers else 2) <= len(offsets) <= 64
                or any(not isinstance(identity, str) or not identity or type(offset) is not int
                       or not 0 <= offset < len(rest) for identity, offset in offsets.items())):
            raise ValueError("Bounded physical instance offsets required")
        ordered = sorted(offsets.items(), key=lambda entry: entry[1])
        starts = [entry[1] for entry in ordered]
        if starts[0] != 0 or len(set(starts)) != len(starts):
            raise ValueError("Physical instances must partition all vertices")
        owners = np.empty(len(rest), dtype=int)
        identities = []
        for index, (identity, start) in enumerate(ordered):
            end = ordered[index + 1][1] if index + 1 < len(ordered) else len(rest)
            if end - start < 3:
                raise ValueError("Each physical instance requires three vertices")
            owners[start:end] = index
            identities.extend([identity] * (end - start))
            source_piece = rest[start:end] - rest[start:end].mean(axis=0)
            placed_piece = positions[start:end] - positions[start:end].mean(axis=0)
            left, _, right = np.linalg.svd(source_piece.T @ placed_piece)
            if np.linalg.det(left @ right) < 0:
                left[:, -1] *= -1
            if not np.allclose(source_piece @ (left @ right), placed_piece, rtol=0, atol=1e-10):
                raise ValueError("Staged placement must remain rigid relative to source rest geometry")
        if np.any(owners[faces] != owners[faces[:, :1]]):
            raise ValueError("Faces must remain within their physical instance")
        original_positions = np.asarray(source["placedMeters"], dtype=float)
        if original_positions.shape != rest.shape or not np.isfinite(original_positions).all():
            raise ValueError("Matching finite original placement required")
        report["seamGapsMm"] = staging_seam_gaps(source, original_positions, positions)
        if report["seamGapsMm"] is None and not ((args.fold_actuation or args.material_grippers)
                and source.get("embeddedConstraints", {}).get("constraints") == []):
            raise ValueError("Nonempty source-validated embedded sewing constraints required")
        report["contactProfile"] = contact.profile()
        contact.validate_state(positions)
        report["initialContactEnergyJ"] = contact.energy(positions)
        report["initialPeakContactForceN"] = float(np.max(np.abs(contact.gradient(positions))))
        report["inactiveReferenceGatePassed"] = (report["initialContactEnergyJ"] == 0.
                                                  and report["initialPeakContactForceN"] == 0.)
        if not report["inactiveReferenceGatePassed"]:
            raise ValueError("Inactive-reference diagnostic rejects nonzero initial contact energy or force")
        report["initialIndependentSurfaceOracle"] = surface_intersections(positions, faces)
        report["initialToolkitHasIntersections"] = bool(ipctk.has_intersections(
            contact.mesh, positions, broad_phase=contact_broad_phase()))
        if (report["initialIndependentSurfaceOracle"]["intersectingPairCount"] != 0
                or report["initialToolkitHasIntersections"]):
            raise ValueError("Initial placement fails an intersection oracle")
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for index, (_, start) in enumerate(ordered):
            end = ordered[index + 1][1] if index + 1 < len(ordered) else len(rest)
            selected = faces[np.all((faces >= start) & (faces < end), axis=1)] - start
            builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
                vel=wp.vec3(0, 0, 0), vertices=rest[start:end].tolist(), indices=selected.ravel().tolist(),
                density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.01, edge_kd=0)
        builder.set_coloring([[vertex] for vertex in range(len(rest))])
        model = builder.finalize(device="cpu")
        model.particle_q.assign(positions.astype(np.float32))
        np.testing.assert_array_equal(faces, model.tri_indices.numpy())
        rows = [{offsets[term["instanceId"]] + term["vertex"]: term["coefficient"]
                 for term in constraint["terms"]} for constraint in source["embeddedConstraints"]["constraints"]]
        material_grippers, gripper_controls = None, None
        if args.material_grippers:
            from solver_gripper_input import bind_material_grippers
            material_grippers, gripper_controls, binding = bind_material_grippers(source, args.subdivisions)
            report["gripperActuation"] = gripper_input
            report["gripperBinding"] = binding
            start_targets, start_activation = gripper_controls.parameters(0.)
            start_grippers = material_grippers.potential(start_targets, start_activation)
            report["initialGripperEnergyJoules"] = start_grippers.energy(positions)
            report["initialGripperTargetsMeters"] = start_targets.tolist()
            report["initialGripperActivation"] = start_activation.tolist()
        sewing_controls = None
        if args.sewing_activation:
            from solver_sewing_input import bind_sewing_activation
            sewing_controls, binding = bind_sewing_activation(source, args.subdivisions, sewing_mode=args.sewing_mode)
            if rows != sewing_controls.rows:
                raise ValueError("Bound sewing rows must match every captured canonical row")
            report["sewingActuation"] = sewing_input
            report["sewingBinding"] = binding
        solver = GlobalSewingSolver(model, rows, sewing_controls.compliance if sewing_controls else 1e-8, contact=contact, fold_barrier_joules=1e-5,
                                   sewing_mode=args.sewing_mode,
                                   sewing_frame_faces=source.get("sewingFrames", {}).get("faces") if args.sewing_mode == "normal-offset" else None,
                                   sewing_sides=source.get("sewingFrames", {}).get("sides") if args.sewing_mode == "normal-offset" else None,
                                   fold_hinges=fold_recipe.get("hinges") if fold_recipe else None,
                                   fold_stiffness_joules=fold_recipe.get("stiffnessJoules") if fold_recipe else None,
                                   material_grippers=material_grippers)
        fold_options = {}
        if args.fold_actuation:
            if solver.fold_actuation is None:
                raise ValueError("Fold recipe must declare source hinges and stiffness")
            initial_fold = solver.fold_actuation.potential(fold_recipe.get("initialAnglesRadians"))
            final_fold = solver.fold_actuation.potential(fold_recipe.get("targetAnglesRadians"))
            if not np.allclose(initial_fold.angles(positions), initial_fold.rest_angles, rtol=0, atol=1e-10):
                raise ValueError("Initial fold targets must match the declared rigid placement")
            fold_options = {"initial_fold_targets": initial_fold.rest_angles,
                            "fold_targets": final_fold.rest_angles}
            report["foldActuation"] = fold_recipe
        if sewing_controls is not None:
            initial_targets, final_targets = sewing_controls.initial_targets, sewing_controls.final_targets
            initial_activation = sewing_controls.parameters(0.)
            active = initial_activation > 0
            if args.sewing_mode in ("distance", "normal-offset"):
                potential = solver.sewing_potential(initial_targets, activation=initial_activation)
                initial_sewing_residual = potential.residual(positions)
                geometry = potential.geometry(positions)
                initial_errors = (np.abs(geometry[1][active] - initial_targets[active])
                    if args.sewing_mode == "distance" else
                    np.max(np.abs(solver.sewing[active] @ positions - (initial_targets[active]
                        * solver.sewing_sides[active])[:, None] * geometry[2][active]), axis=1))
            else:
                initial_error_vectors = solver.sewing[active] @ positions - initial_targets[active]
                initial_errors = np.max(np.abs(initial_error_vectors), axis=1)
                initial_sewing_residual = (initial_error_vectors
                    * np.sqrt(initial_activation[active, None] / solver.compliance)).ravel()
            report["initialSewingEnergyJoules"] = float(.5 * np.sum(initial_sewing_residual ** 2))
            if not math.isfinite(report["initialSewingEnergyJoules"]):
                raise ValueError("Finite initial activated sewing energy required")
            report["initialSewingTargetsMeters"] = initial_targets.tolist()
            report["initialSewingActivation"] = initial_activation.tolist()
            initial_row_errors = [None] * len(initial_activation)
            for index, error in zip(np.flatnonzero(active), initial_errors):
                initial_row_errors[index] = float(error)
            report["initialSewingRowTargetErrorsM"] = initial_row_errors
            report["initialSewingTargetErrorMetric"] = "Unweighted maximum absolute Cartesian component per active vector/normal row; absolute scalar-distance error in distance mode; pending rows excluded"
        else:
            initial_targets = solver.sewing @ positions
            if args.sewing_mode in ("distance", "normal-offset"):
                initial_targets = np.linalg.norm(initial_targets, axis=1)
            if args.sewing_mode == "normal-offset":
                initial_error = solver.sewing_potential(initial_targets).residual(positions) * np.sqrt(solver.compliance)
                if np.max(np.abs(initial_error), initial=0) > 1e-10:
                    raise ValueError("Initial anchors must match the declared material-normal sides and offsets")
            final_targets = args.target_fraction * initial_targets
        report["sewingMode"] = args.sewing_mode
        report["attemptJournalRequired"] = True
        progress.save(report, "adaptive-journal-initializing")
        journal = AttemptJournal(output, report, positions, np.zeros_like(positions), dt=args.step_seconds,
                                 initial_subdivisions=args.subdivisions, max_depth=args.max_depth,
                                 max_attempts=args.max_attempts)
        progress.save(report, "adaptive-journal-ready")

        final, velocity, report["adaptive"] = adaptive_contact_step(
            solver, positions, np.zeros_like(positions), initial_targets, final_targets,
            args.step_seconds, initial_subdivisions=args.subdivisions, max_attempts=args.max_attempts,
            max_depth=args.max_depth, max_evaluations=args.max_evaluations, attempt_journal=journal,
            assembly_schedule=schedule_recipe,
            sewing_schedule=sewing_controls.schedule_recipe if sewing_controls else None,
            sewing_row_ids=sewing_controls.row_ids if sewing_controls else None,
            gripper_schedule=gripper_input["schedule"] if args.material_grippers else None, **fold_options)
        recovered = recover_attempt_journal(output, report)
        if recovered["attemptJournal"]["errors"]:
            raise ValueError("Attempt journal verification failed")
        report.update(recovered)
        if args.material_grippers:
            accepted_steps = report["adaptive"]["acceptedSteps"]
            energies = [record["step"]["energyBalance"] for record in accepted_steps]
            report["gripperWorkSummary"] = {
                "acceptedSteps": len(accepted_steps),
                **{key: math.fsum(energy[key] for energy in energies) for key in (
                    "gripperParameterWorkJoules", "gripperTargetParameterWorkJoules",
                    "gripperActivationParameterWorkJoules", "gripperReleaseEnergyRemovedJoules",
                    "externalParameterWorkJoules", "mechanicalChangeJoules",
                    "mechanicalChangeMinusParameterWorkJoules")},
                "scope": "Sum over accepted transitions only; discrete target-first parameter changes at prior positions, not continuous tool work or physical release dissipation"}
        if sewing_controls is not None:
            accepted_steps = report["adaptive"]["acceptedSteps"]
            energies = [record["step"]["energyBalance"] for record in accepted_steps]
            report["sewingWorkSummary"] = {
                "acceptedSteps": len(accepted_steps),
                **{key: math.fsum(energy[key] for energy in energies) for key in (
                    "sewingParameterWorkJoules", "sewingTargetParameterWorkJoules",
                    "sewingActivationParameterWorkJoules", "sewingActivationIncreaseWorkJoules",
                    "sewingReleaseEnergyRemovedJoules", "sewingFixedParameterChangeJoules",
                    "externalParameterWorkJoules", "mechanicalChangeJoules",
                    "mechanicalChangeMinusParameterWorkJoules")},
                "scope": "Accepted transitions only; target changes at old weights then activation at new targets, at prior positions. Numerical controls do not complete construction phases."}
        report["stateArtifact"] = atomic_json(args.output / "state.json", {"positionsMeters": final.tolist(),
            "velocitiesMetersPerSecond": velocity.tolist(),
            "completedDurationSeconds": report["adaptive"]["completedDurationSeconds"], "accepted": False})
        progress.save(report, "final-validation")
        report["edgeStrain"] = edge_strain_report(rest, final, faces, identities)
        report["massMotion"] = mass_motion_report(positions, final, velocity, solver.mass)
        report["finalIndependentSurfaceOracle"] = surface_intersections(final, faces)
        report["finalToolkitHasIntersections"] = bool(ipctk.has_intersections(
            contact.mesh, final, broad_phase=contact_broad_phase()))
        report["finalContactEnergyJ"] = contact.energy(final)
        report["finalPeakContactForceN"] = float(np.max(np.abs(contact.gradient(final))))
        final_anchors = solver.sewing @ final
        completed_fraction = report["adaptive"]["completedFraction"]
        sewing_progress, fold_progress = (schedule.progress(completed_fraction) if schedule else
                                          (completed_fraction, completed_fraction))
        completed_targets = (final_targets if sewing_progress == 1 else
                             initial_targets + sewing_progress * (final_targets - initial_targets))
        if sewing_controls is not None:
            completed_activation = sewing_controls.parameters(completed_fraction)
            active = completed_activation > 0
            active_anchors = solver.sewing[active] @ final
            if args.sewing_mode == "distance":
                target_errors = np.abs(np.linalg.norm(active_anchors, axis=1) - completed_targets[active])
            elif args.sewing_mode == "normal-offset":
                potential = solver.sewing_potential(completed_targets, activation=active.astype(float))
                normals = potential.geometry(final)[2][active]
                target_errors = np.max(np.abs(active_anchors - (completed_targets[active]
                    * solver.sewing_sides[active])[:, None] * normals), axis=1)
            else:
                target_errors = np.max(np.abs(active_anchors - completed_targets[active]), axis=1)
            report["finalSewingActivation"] = completed_activation.tolist()
            report["finalActiveSewingRows"] = np.flatnonzero(active).tolist()
            report["finalPendingSewingRows"] = np.flatnonzero(~active).tolist()
            report["finalSewingTargetErrorMetric"] = "Unweighted active rows only; maximum absolute Cartesian component for vector/normal, absolute length error for distance"
            report["finalMaximumAnchorGapM"] = float(np.max(np.linalg.norm(final_anchors, axis=1), initial=0))
            report["finalMaximumCompletedTargetErrorM"] = float(np.max(target_errors, initial=0))
        else:
            target_errors = (np.abs(np.linalg.norm(final_anchors, axis=1) - completed_targets)
                             if args.sewing_mode == "distance" else
                             np.linalg.norm(solver.sewing_potential(completed_targets).residual(final).reshape((-1, 3)), axis=1) * np.sqrt(solver.compliance)
                             if args.sewing_mode == "normal-offset" else
                             np.linalg.norm(final_anchors - completed_targets, axis=1))
            report["finalMaximumAnchorGapM"] = float(np.linalg.norm(final_anchors, axis=1).max()) if len(rows) else None
            report["finalMaximumCompletedTargetErrorM"] = float(target_errors.max()) if len(rows) else None
        if args.fold_actuation:
            completed_fold_targets = (final_fold.rest_angles if fold_progress == 1 else
                initial_fold.rest_angles + fold_progress * (final_fold.rest_angles - initial_fold.rest_angles))
            report["finalFoldAnglesRadians"] = final_fold.angles(final).tolist()
            report["finalFoldTargetErrorRadians"] = float(np.max(np.abs(
                final_fold.angles(final) - completed_fold_targets)))
        if (report["finalIndependentSurfaceOracle"]["intersectingPairCount"] != 0
                or report["finalToolkitHasIntersections"]):
            raise ValueError("Final state fails an intersection oracle")
        report["classification"] = ("completed-research-interval" if report["adaptive"]["complete"]
                                    else "partial-research-interval")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report.update({"workerReportedWallSeconds": time.monotonic() - started,
                       "workerReportedCpuSeconds": time.process_time() - cpu_started})
        exit_code = 0 if report.get("adaptive", {}).get("complete") and "failure" not in report else 1
        progress.save(report, "worker-finished", exit_code=exit_code)
    return exit_code


def main():
    if sys.argv[1:2] == ["--_worker-output"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--_worker-output", type=Path, required=True)
        parser.add_argument("--_parent-pid", type=int, required=True)
        worker = parser.parse_args()
        return run_worker(worker._worker_output, worker._parent_pid)
    args = parse_arguments()
    args.output = args.output.absolute()
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = time.monotonic()
    report = {"profile": ("experimental-sewing-activation-contact-continuation-v1" if args.sewing_activation else
                          "experimental-material-gripper-contact-continuation-v1" if args.material_grippers else
                          "experimental-inactive-reference-contact-continuation-v1"), "accepted": False,
              "terminal": False, "completed": False, "acceptedStateArtifacts": [], "sourceDigests": {},
              "arguments": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              "cpuLimitSeconds": args.cpu_limit_seconds, "wallLimitSeconds": args.wall_limit_seconds,
              "memoryLimitBytes": 4 * 1024 ** 3, "fileSizeLimitBytes": 64 * 1024 ** 2,
              "inputSizeLimitBytes": 50 * 1024 ** 2, "classification": "uncompleted-research-diagnostic",
              "limitations": ["Trusted saved-source research CLI; not an untrusted-input service boundary.",
                  "Completing one physical interval is not assembled garment or drape acceptance.",
                  "Adaptive subdivisions alter time discretization, not total duration, target ramp or tolerances.",
                  ("Explicit captured seam targets/activation; source phase declarations are not executed construction." if args.sewing_activation else
                   "Fraction-scaled diagnostic seam targets; no layer-side, turning or binding execution."),
                  "No body contact, damping or calibrated material model.",
                  "Linux same-process-group supervision; not a sandbox for escaping descendants.",
                  "Supervisor SIGKILL or host loss cannot publish a terminal report; progress remains incomplete.",
                  "Worker budgets exclude initial regular-file capture and allow bounded termination/reaping grace."]}
    atomic_json(args.output / "report.json", report)
    try:
        snapshot = args.output / "source-snapshot"
        snapshot.mkdir(mode=0o700)
        sources = {path.name: path for path in [*sorted(Path(__file__).parent.glob("solver_*.py")), Path(__file__),
                   Path(__file__).with_name("solver-contact.requirements.txt"),
                   Path(__file__).with_name("solver-spike.requirements.txt")]}
        if args.sewing_activation:
            from solver_engine_source_namespace import ENGINE_FILES, engine_source_root
            engine_root = engine_source_root(__file__)
            sources.update({"services/engine/" + name: engine_root / name for name in ENGINE_FILES})
            # Registration construction is captured even for a generic control,
            # so a cuff input can never import this helper from a live checkout.
            sources["spike-full-shirt.py"] = Path(__file__).with_name("spike-full-shirt.py")
        for name, source_path in sources.items():
            content = read_regular(source_path)
            destination = snapshot / name
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            report["sourceDigests"][name] = atomic_bytes(destination, content)["sha256"]
        if (snapshot / "services").exists():
            (snapshot / "services/engine").chmod(0o500)
            (snapshot / "services").chmod(0o500)
        snapshot.chmod(0o500)
        for name, source_path, digest_key in (("canonical.json", args.canonical, "canonicalSha256"),
                                              ("placement.json", args.placement, "placementSha256")):
            report[digest_key] = atomic_bytes(args.output / name, bounded_input(source_path))["sha256"]
        report["captureWallSeconds"] = time.monotonic() - started
        ProgressStore(args.output).save(report, "captured")
        atomic_json(args.output / "report.json", report, replace=True)
    except Exception as error:
        report.update({"terminal": True, "failure": {"type": type(error).__name__, "message": str(error)},
                       "captureWallSeconds": time.monotonic() - started})
        atomic_json(args.output / "report.json", report, replace=True)
        print(str(args.output / "report.json"))
        return 1
    report = supervise([sys.executable, "-B", str(snapshot / Path(__file__).name),
                        "--_worker-output", str(args.output), "--_parent-pid", str(os.getpid())],
                       args.output, report, cpu_limit_seconds=args.cpu_limit_seconds,
                       wall_limit_seconds=args.wall_limit_seconds)
    print(str(args.output / "report.json"))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
