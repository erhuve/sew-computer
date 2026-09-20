import copy

import numpy as np

from solver_attempt_journal import diagnostic_json


def adaptive_contact_step(solver, positions, velocities, initial_targets, targets, dt, *,
                          max_depth=8, max_attempts=256, initial_subdivisions=1, on_accept=None,
                          attempt_journal=None, **step_options):
    if on_accept is not None and not callable(on_accept):
        raise ValueError("Accepted-state callback must be callable")
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
        substep_targets = (targets.copy() if end_fraction == 1 else
                           initial_targets + end_fraction * (targets - initial_targets))
        record = {"attemptId": len(attempts) + 1, "parentAttemptId": parent_attempt,
                  "initialInterval": initial_interval, "startFraction": start_fraction,
                  "endFraction": end_fraction, "durationSeconds": duration, "depth": depth, "converged": False}
        if attempt_journal is not None:
            attempt_journal.start(record)
        fatal = False
        propagate = None
        try:
            candidate_positions, candidate_velocities, step_report = solver.step(
                current_positions.copy(), current_velocities.copy(), substep_targets, duration, **step_options)
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
        "targetInterpolation": "linear over the original physical interval",
    }
