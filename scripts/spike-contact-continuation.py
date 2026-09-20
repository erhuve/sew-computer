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
                                   prepare_worker, read_regular, recover_progress, supervise)


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
    parser.add_argument("--sewing-mode", choices=("vector", "distance"), default="vector",
                        help="Experimental scalar anchor lengths or legacy world-space vectors")
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
            if hashlib.sha256(read_regular(output / "source-snapshot" / name)).hexdigest() != digest:
                raise ValueError(f"Captured source hash mismatch: {name}")
        canonical_bytes = bounded_input(output / "canonical.json")
        placement_bytes = bounded_input(output / "placement.json")
        canonical_digest = hashlib.sha256(canonical_bytes).hexdigest()
        if (canonical_digest != report["canonicalSha256"]
                or hashlib.sha256(placement_bytes).hexdigest() != report["placementSha256"]):
            raise ValueError("Captured input hash mismatch")
        progress.save(report, "validating-inputs")
        source, placement = json.loads(canonical_bytes), json.loads(placement_bytes)
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
        if (not isinstance(offsets, dict) or not 2 <= len(offsets) <= 64
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
        if report["seamGapsMm"] is None:
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
        report["initialToolkitHasIntersections"] = bool(ipctk.has_intersections(contact.mesh, positions))
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
        solver = GlobalSewingSolver(model, rows, 1e-8, contact=contact, fold_barrier_joules=1e-5,
                                   sewing_mode=args.sewing_mode)
        initial_targets = solver.sewing @ positions
        if args.sewing_mode == "distance":
            initial_targets = np.linalg.norm(initial_targets, axis=1)
        report["sewingMode"] = args.sewing_mode
        report["attemptJournalRequired"] = True
        progress.save(report, "adaptive-journal-initializing")
        journal = AttemptJournal(output, report, positions, np.zeros_like(positions), dt=args.step_seconds,
                                 initial_subdivisions=args.subdivisions, max_depth=args.max_depth,
                                 max_attempts=args.max_attempts)
        progress.save(report, "adaptive-journal-ready")

        final, velocity, report["adaptive"] = adaptive_contact_step(
            solver, positions, np.zeros_like(positions), initial_targets, args.target_fraction * initial_targets,
            args.step_seconds, initial_subdivisions=args.subdivisions, max_attempts=args.max_attempts,
            max_depth=args.max_depth, max_evaluations=args.max_evaluations, attempt_journal=journal)
        recovered = recover_attempt_journal(output, report)
        if recovered["attemptJournal"]["errors"]:
            raise ValueError("Attempt journal verification failed")
        report.update(recovered)
        report["stateArtifact"] = atomic_json(args.output / "state.json", {"positionsMeters": final.tolist(),
            "velocitiesMetersPerSecond": velocity.tolist(),
            "completedDurationSeconds": report["adaptive"]["completedDurationSeconds"], "accepted": False})
        progress.save(report, "final-validation")
        report["edgeStrain"] = edge_strain_report(rest, final, faces, identities)
        report["massMotion"] = mass_motion_report(positions, final, velocity, solver.mass)
        report["finalIndependentSurfaceOracle"] = surface_intersections(final, faces)
        report["finalToolkitHasIntersections"] = bool(ipctk.has_intersections(contact.mesh, final))
        report["finalContactEnergyJ"] = contact.energy(final)
        report["finalPeakContactForceN"] = float(np.max(np.abs(contact.gradient(final))))
        final_anchors = solver.sewing @ final
        completed_targets = initial_targets * (1 + report["adaptive"]["completedFraction"]
                                               * (args.target_fraction - 1))
        target_errors = (np.abs(np.linalg.norm(final_anchors, axis=1) - completed_targets)
                         if args.sewing_mode == "distance" else
                         np.linalg.norm(final_anchors - completed_targets, axis=1))
        report["finalMaximumAnchorGapM"] = float(np.linalg.norm(final_anchors, axis=1).max())
        report["finalMaximumCompletedTargetErrorM"] = float(target_errors.max())
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
    report = {"profile": "experimental-inactive-reference-contact-continuation-v1", "accepted": False,
              "terminal": False, "completed": False, "acceptedStateArtifacts": [], "sourceDigests": {},
              "arguments": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              "cpuLimitSeconds": args.cpu_limit_seconds, "wallLimitSeconds": args.wall_limit_seconds,
              "memoryLimitBytes": 4 * 1024 ** 3, "fileSizeLimitBytes": 64 * 1024 ** 2,
              "inputSizeLimitBytes": 50 * 1024 ** 2, "classification": "uncompleted-research-diagnostic",
              "limitations": ["Trusted saved-source research CLI; not an untrusted-input service boundary.",
                  "Completing one physical interval is not assembled garment or drape acceptance.",
                  "Adaptive subdivisions alter time discretization, not total duration, target ramp or tolerances.",
                  "Fraction-scaled diagnostic seam targets; no layer-side, turning or binding execution.",
                  "No body contact, damping or calibrated material model.",
                  "Linux same-process-group supervision; not a sandbox for escaping descendants.",
                  "Supervisor SIGKILL or host loss cannot publish a terminal report; progress remains incomplete.",
                  "Worker budgets exclude initial regular-file capture and allow bounded termination/reaping grace."]}
    atomic_json(args.output / "report.json", report)
    try:
        snapshot = args.output / "source-snapshot"
        snapshot.mkdir(mode=0o700)
        sources = [*sorted(Path(__file__).parent.glob("solver_*.py")), Path(__file__),
                   Path(__file__).with_name("solver-contact.requirements.txt"),
                   Path(__file__).with_name("solver-spike.requirements.txt")]
        for source_path in sources:
            content = read_regular(source_path)
            report["sourceDigests"][source_path.name] = atomic_bytes(snapshot / source_path.name, content)["sha256"]
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
