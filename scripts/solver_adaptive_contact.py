import copy
from fractions import Fraction

import numpy as np

from solver_attempt_journal import diagnostic_json
from solver_assembly_schedule import AssemblySchedule


def adaptive_contact_step(solver, positions, velocities, initial_targets, targets, dt, *,
                          max_depth=8, max_attempts=256, initial_subdivisions=1, on_accept=None,
                          attempt_journal=None, initial_fold_targets=None, fold_targets=None,
                          assembly_schedule=None, gripper_schedule=None, **step_options):
    if on_accept is not None and not callable(on_accept):
        raise ValueError("Accepted-state callback must be callable")
    if "sewing_activation" in step_options:
        raise ValueError("Adaptive sewing activation requires an explicit captured activation schedule")
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    initial_targets = np.asarray(initial_targets, dtype=float)
    targets = np.asarray(targets, dtype=float)
    distance_mode = getattr(solver, "sewing_mode", "vector") in ("distance", "normal-offset")
    valid_targets = ((targets.ndim == 1 and np.all(targets > 0)
                      and np.all(initial_targets > 0)) if distance_mode else
                     (targets.ndim == 2 and targets.shape[1] == 3))
    if (positions.ndim != 2 or positions.shape[1] != 3 or not positions.shape[0]
            or velocities.shape != positions.shape or not valid_targets
            or initial_targets.shape != targets.shape
            or not all(np.isfinite(value).all() for value in
                       (positions, velocities, initial_targets, targets))
            or isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt)
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("Finite matching states, targets and positive physical duration required")
    if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in (max_depth, max_attempts, initial_subdivisions))
            or not 0 <= max_depth <= 30 or not 1 <= max_attempts <= 4096 or initial_subdivisions < 1
            or initial_subdivisions & (initial_subdivisions - 1)
            or initial_subdivisions > max_attempts):
        raise ValueError("Bounded depth, positive attempt budget and dyadic initial subdivisions required")
    current_positions, current_velocities = positions.copy(), velocities.copy()
    initial_targets, targets = initial_targets.copy(), targets.copy()
    fold_recipe = getattr(solver, "fold_actuation", None)
    if fold_recipe is None:
        if initial_fold_targets is not None or fold_targets is not None:
            raise ValueError("Fold schedule requires an actuator recipe")
    else:
        initial_fold_targets = fold_recipe.potential(initial_fold_targets).rest_angles.copy()
        fold_targets = fold_recipe.potential(fold_targets).rest_angles.copy()
    schedule = None
    if assembly_schedule is not None:
        if fold_recipe is None:
            raise ValueError("Assembly schedule requires explicit fold actuation")
        schedule = AssemblySchedule(assembly_schedule, initial_subdivisions)
    gripper_recipe = getattr(solver, "material_grippers", None)
    if any(key in step_options for key in ("gripper_targets", "gripper_activation")):
        raise ValueError("Adaptive gripper targets must come from the captured schedule")
    if (gripper_recipe is None) != (gripper_schedule is None):
        raise ValueError("Material-gripper recipe and captured schedule must be supplied together")
    gripper_controls = None
    if gripper_recipe is not None:
        if int(initial_subdivisions).bit_length() - 1 + max_depth > 40:
            raise ValueError("Material-gripper subdivision fractions must remain within the 2^40 dyadic bound")
        from solver_material_grippers import MaterialGripperSchedule
        gripper_controls = MaterialGripperSchedule(gripper_schedule, initial_subdivisions,
                                                 gripper_ids=gripper_recipe.gripper_ids)
    attempts, accepted, rejected = [], [], []
    completed_fraction = 0.
    reason = "attempt-budget-exhausted"
    pending = []
    next_initial_interval = 0
    while len(attempts) < max_attempts:
        if not pending:
            if next_initial_interval == initial_subdivisions:
                reason = "complete"
                break
            interval = next_initial_interval
            pending.append((interval / initial_subdivisions, (interval + 1) / initial_subdivisions, 0, None, interval))
            next_initial_interval += 1
        start_fraction, end_fraction, depth, parent_attempt, initial_interval = pending.pop()
        duration = float(dt * (end_fraction - start_fraction))
        if duration <= 0 or not np.isfinite(duration):
            reason = "substep-duration-underflow"
            break
        sewing_progress, fold_progress = (schedule.progress(end_fraction) if schedule else
                                          (end_fraction, end_fraction))
        substep_targets = (targets.copy() if sewing_progress == 1 else
                           initial_targets + sewing_progress * (targets - initial_targets))
        record = {"attemptId": len(attempts) + 1, "parentAttemptId": parent_attempt,
                  "initialInterval": initial_interval, "startFraction": start_fraction,
                  "endFraction": end_fraction, "durationSeconds": duration, "depth": depth, "converged": False}
        if attempt_journal is not None:
            attempt_journal.start(record)
        fatal = False
        propagate = None
        try:
            options = dict(step_options)
            if fold_recipe is not None:
                options["fold_targets"] = (fold_targets.copy() if fold_progress == 1 else
                    initial_fold_targets + fold_progress * (fold_targets - initial_fold_targets))
            if gripper_controls is not None:
                options["gripper_targets"], options["gripper_activation"] = gripper_controls.parameters(end_fraction)
            candidate_positions, candidate_velocities, step_report = solver.step(
                current_positions.copy(), current_velocities.copy(), substep_targets, duration, **options)
            if not isinstance(step_report, dict):
                raise ValueError("Solver diagnostic report must be an object")
            record["step"], nonfinite = diagnostic_json(step_report)
            record["nonfiniteDiagnostics"] = nonfinite
            candidate_positions = np.asarray(candidate_positions, dtype=float)
            candidate_velocities = np.asarray(candidate_velocities, dtype=float)
            for label, array in (("candidatePositions", candidate_positions), ("candidateVelocities", candidate_velocities)):
                for index in np.argwhere(~np.isfinite(array)):
                    pointer = "/" + label + "".join(f"/{int(part)}" for part in index)
                    _, tags = diagnostic_json(array[tuple(index)], pointer)
                    nonfinite.extend(tags)
            record["nonfiniteDiagnostics"] = nonfinite
            residual = step_report.get("gradientInfinityNorm")
            valid = (not nonfinite and candidate_positions.shape == positions.shape
                     and candidate_velocities.shape == velocities.shape
                     and np.isfinite(candidate_positions).all() and np.isfinite(candidate_velocities).all()
                     and isinstance(residual, (float, int, np.floating, np.integer))
                     and not isinstance(residual, (bool, np.bool_))
                     and np.isfinite(residual) and 0 <= residual <= 1e-6
                     and step_report.get("converged") is True)
            if valid and gripper_controls is not None:
                # Work belongs to the accepted transition and must be checked
                # before its immutable journal outcome is written. Trial
                # controls always use original fractions; retries do not
                # advance the state or accumulate rejected work.
                from solver_energy_balance import global_energy_transition
                old_sewing_progress, old_fold_progress = (schedule.progress(start_fraction) if schedule else
                                                          (start_fraction, start_fraction))
                old_targets = (targets.copy() if old_sewing_progress == 1 else
                               initial_targets + old_sewing_progress * (targets - initial_targets))
                old_grip_targets, old_grip_activation = gripper_controls.parameters(start_fraction)
                energy_options = {"previous_gripper_targets": old_grip_targets,
                                  "previous_gripper_activation": old_grip_activation,
                                  "gripper_targets": options["gripper_targets"],
                                  "gripper_activation": options["gripper_activation"]}
                if fold_recipe is not None:
                    old_fold = (fold_targets.copy() if old_fold_progress == 1 else
                                initial_fold_targets + old_fold_progress * (fold_targets - initial_fold_targets))
                    energy_options.update(previous_fold_targets=old_fold,
                                          fold_targets=options["fold_targets"])
                energy = global_energy_transition(solver, current_positions, candidate_positions,
                    current_velocities, candidate_velocities, old_targets, substep_targets, duration,
                    **energy_options)
                potential = gripper_recipe.potential(options["gripper_targets"], options["gripper_activation"])
                momentum_exact = [sum((Fraction(float(mass)) * (Fraction(float(new[axis])) - Fraction(float(old[axis])))
                                      for mass, new, old in zip(solver.mass, candidate_velocities, current_velocities)),
                                     Fraction()) for axis in range(3)]
                # The potential's aggregate retains cancellation across
                # anchors before individual force reports are rounded.
                total_force = potential.diagnostics(candidate_positions)["totalClothForceNewtons"]
                impulse_exact = [Fraction(duration) * Fraction(float(force)) for force in total_force]
                momentum, impulse = [np.array([float(value) for value in vector])
                                     for vector in (momentum_exact, impulse_exact)]
                error = np.array([float(change - applied) for change, applied in zip(momentum_exact, impulse_exact)])
                tolerance = len(solver.mass) * duration * 1e-6 + 64 * np.finfo(float).eps * max(
                    1., float(np.max(np.abs(momentum))), float(np.max(np.abs(impulse))))
                if (not np.all(solver.active) or not np.isfinite(error).all()
                        or np.max(np.abs(error)) > tolerance):
                    raise ValueError("Material-gripper transition fails free-cloth linear momentum accounting")
                step_report = dict(step_report, energyBalance=energy, gripperMomentum={
                    "changeKgMPerS": momentum.tolist(), "externalImpulseNs": impulse.tolist(),
                    "residualNs": error.tolist(), "toleranceNs": float(tolerance),
                    "scope": "Backward-Euler force at the new state on free cloth; virtual gripper impulse, not isolated-cloth momentum conservation"})
                record["step"], nonfinite = diagnostic_json(step_report)
                record["nonfiniteDiagnostics"] = nonfinite
                if nonfinite:
                    raise ValueError("Material-gripper transition has nonfinite work or momentum diagnostics")
            record["converged"] = bool(valid)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            valid = False
        except (TimeoutError, RuntimeError, MemoryError) as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            valid, fatal = False, True
        except BaseException as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            valid, fatal, propagate = False, True, error
        record["outcome"] = "accepted" if valid else ("interrupted" if propagate is not None else "rejected")
        record["fatal"] = fatal
        if valid:
            record["completedDurationSeconds"] = float(dt * end_fraction)
        if attempt_journal is not None:
            attempt_journal.outcome(record, candidate_positions if valid else None,
                                    candidate_velocities if valid else None)
        if propagate is not None:
            raise propagate
        attempts.append(record)
        if valid:
            current_positions, current_velocities = candidate_positions.copy(), candidate_velocities.copy()
            completed_fraction = end_fraction
            accepted.append(record)
            if on_accept is not None:
                on_accept(current_positions.copy(), current_velocities.copy(), copy.deepcopy(record))
        else:
            rejected.append(record)
            if fatal:
                reason = "solver-resource-or-runtime-failure"
                break
            if depth == max_depth:
                reason = "subdivision-depth-exhausted"
                break
            midpoint = .5 * (start_fraction + end_fraction)
            pending.extend(((midpoint, end_fraction, depth + 1, record["attemptId"], initial_interval),
                            (start_fraction, midpoint, depth + 1, record["attemptId"], initial_interval)))
    complete = completed_fraction == 1.
    if complete:
        reason = "complete"
    if attempt_journal is not None:
        attempt_journal.finish(reason, complete)
    return current_positions, current_velocities, {
        "profile": "experimental-adaptive-contact-time-subdivision-v1", "accepted": False,
        "complete": complete, "reason": reason, "requestedDurationSeconds": float(dt),
        "completedDurationSeconds": float(dt * completed_fraction), "completedFraction": completed_fraction,
        "initialSubdivisions": int(initial_subdivisions), "maxDepth": int(max_depth),
        "maxAttempts": int(max_attempts), "stationarityToleranceN": 1e-6,
        "attempts": attempts, "acceptedSteps": accepted, "rejectedSteps": rejected, "interruptedSteps": [],
        "targetInterpolation": ("captured piecewise-linear sewing/fold progress" if schedule else
                                "linear sewing progress over the original physical interval" if gripper_controls is not None else
                                "linear over the original physical interval"),
        **({"gripperInterpolation": "captured piecewise-linear material-point targets and activation over the original physical interval"}
           if gripper_controls is not None else {}),
    }
