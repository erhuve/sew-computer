import copy
from fractions import Fraction

import numpy as np

from solver_attempt_journal import diagnostic_json, same
from solver_assembly_schedule import AssemblySchedule


CONTROLLED_FOLD_SWEEP_POLICY = "all declared controlled hinges, including inactive; optimizer and physical affine paths"


def _same_control_array(actual, expected):
    return (type(actual) is np.ndarray and actual.dtype == np.dtype(np.float64)
            and actual.shape == expected.shape and actual.tobytes() == expected.tobytes())


def _validate_fold_step(solver, recipe, controls, fraction, options, positions, report):
    if getattr(solver, "controlled_fold_actuation", None) is not recipe or getattr(solver, "fold_actuation", None) is not None:
        raise ValueError("Controlled fold recipe identity changed during the adaptive transition")
    targets, activation = controls.parameters(fraction)
    if (not _same_control_array(options["fold_targets"], targets)
            or not _same_control_array(options["fold_activation"], activation)):
        raise ValueError("Step mutated the original-fraction controlled fold parameters")
    expected = recipe.potential(targets, activation).diagnostics(positions)
    for field, value in (("foldActuation", True), ("foldActuationJoules", expected["energyJoules"]),
                         ("foldTargetsRadians", targets.tolist()),
                         ("foldAnglesRadians", expected["sampledActiveAnglesRadians"]),
                         ("foldControls", expected), ("foldHingeSweepPolicy", CONTROLLED_FOLD_SWEEP_POLICY)):
        if field not in report or not same(report[field], value):
            raise ValueError("Step controlled fold diagnostics differ from fresh complete control evaluation: " + field)


def _validate_fold_energy(energy):
    fields = ("foldActuationBeforeJoules", "foldActuationAfterJoules", "foldActuationChangeJoules",
              "foldFixedPositionAfterJoules", "foldFixedParameterChangeJoules", "foldTargetParameterWorkJoules",
              "foldParameterWorkJoules", "foldActivationParameterWorkJoules", "foldActivationIncreaseWorkJoules",
              "foldReleaseEnergyRemovedJoules", "foldParameterWorkComponentSumErrorBoundJoules",
              "mechanicalChangeJoules", "targetParameterWorkJoules", "externalParameterWorkJoules",
              "mechanicalChangeMinusTargetWorkJoules", "mechanicalChangeMinusParameterWorkJoules")
    if (type(energy) is not dict or energy.get("accepted") is not False
            or any(isinstance(energy.get(field), (bool, np.bool_))
                   or not isinstance(energy.get(field), (int, float, np.integer, np.floating))
                   or not np.isfinite(energy[field]) for field in fields)
            or any(energy[field] < 0 for field in ("foldActuationBeforeJoules", "foldActuationAfterJoules",
                "foldFixedPositionAfterJoules", "foldActivationIncreaseWorkJoules", "foldReleaseEnergyRemovedJoules",
                "foldParameterWorkComponentSumErrorBoundJoules"))):
        raise ValueError("Complete finite unaccepted controlled-fold work accounting required")


def adaptive_contact_step(solver, positions, velocities, initial_targets, targets, dt, *,
                          max_depth=8, max_attempts=256, initial_subdivisions=1, on_accept=None,
                          attempt_journal=None, initial_fold_targets=None, fold_targets=None,
                          assembly_schedule=None, gripper_schedule=None, sewing_schedule=None, sewing_row_ids=None,
                          fold_control_schedule=None, **step_options):
    if on_accept is not None and not callable(on_accept):
        raise ValueError("Accepted-state callback must be callable")
    if "sewing_activation" in step_options:
        raise ValueError("Adaptive sewing activation requires an explicit captured activation schedule")
    if "fold_activation" in step_options:
        raise ValueError("Adaptive fold activation requires an explicit per-hinge control schedule")
    controlled_fold_recipe = getattr(solver, "controlled_fold_actuation", None)
    if controlled_fold_recipe is not None:
        from solver_controlled_fold import ControlledFoldActuation, _binary64
        if not isinstance(controlled_fold_recipe, ControlledFoldActuation):
            raise ValueError("An admitted controlled fold recipe is required")
        # Preserve original scalar admission: an outer float conversion must
        # not conceal Boolean values or unrepresentable integer state inputs.
        positions = controlled_fold_recipe._positions(positions)
        velocities = controlled_fold_recipe._positions(velocities)
        dt = _binary64(dt)
        if step_options.get("linear_solver", "direct") != "direct":
            raise ValueError("Controlled fold activation requires guarded direct search")
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
    if fold_recipe is not None and controlled_fold_recipe is not None:
        raise ValueError("Legacy and explicit controlled fold actuation cannot be combined")
    if (controlled_fold_recipe is None) != (fold_control_schedule is None):
        raise ValueError("Controlled fold recipe and explicit per-hinge schedule must be supplied together")
    fold_controls = None
    if controlled_fold_recipe is not None:
        from solver_controlled_fold import ControlledFoldActuation
        from solver_fold_control_schedule import FoldControlSchedule
        if (not isinstance(controlled_fold_recipe, ControlledFoldActuation)
                or initial_fold_targets is not None or fold_targets is not None or assembly_schedule is not None):
            raise ValueError("Controlled fold schedule requires its own recipe and cannot mix legacy fold or assembly controls")
        if int(initial_subdivisions).bit_length()-1 + max_depth > 40:
            raise ValueError("Controlled fold subdivision fractions must remain within the 2^40 dyadic bound")
        fold_controls = FoldControlSchedule(fold_control_schedule, int(initial_subdivisions), hinges=controlled_fold_recipe.hinges)
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
    sewing_controls = None
    if (sewing_schedule is None) != (sewing_row_ids is None):
        raise ValueError("Captured sewing schedule and bound ordered row identities must be supplied together")
    if sewing_schedule is not None:
        if int(initial_subdivisions).bit_length() - 1 + max_depth > 40:
            raise ValueError("Sewing subdivision fractions must remain within the 2^40 dyadic bound")
        from solver_sewing_activation_schedule import SewingActivationSchedule
        sewing_controls = SewingActivationSchedule(sewing_schedule, initial_subdivisions, row_ids=sewing_row_ids)
        if len(sewing_controls.row_ids) != solver.sewing.shape[0] or targets.shape[0] != solver.sewing.shape[0]:
            raise ValueError("Captured sewing schedule must retain every canonical solver row")
        if step_options.get("linear_solver", "direct") != "direct":
            raise ValueError("Captured sewing activation requires guarded direct search")
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
            if fold_controls is not None:
                options["fold_targets"], options["fold_activation"] = fold_controls.parameters(end_fraction)
            if gripper_controls is not None:
                options["gripper_targets"], options["gripper_activation"] = gripper_controls.parameters(end_fraction)
            if sewing_controls is not None:
                options["sewing_activation"] = sewing_controls.parameters(end_fraction)
            candidate_positions, candidate_velocities, step_report = solver.step(
                current_positions.copy(), current_velocities.copy(), substep_targets, duration, **options)
            if not isinstance(step_report, dict):
                raise ValueError("Solver diagnostic report must be an object")
            record["step"], nonfinite = diagnostic_json(step_report)
            record["nonfiniteDiagnostics"] = nonfinite
            if fold_controls is not None:
                candidate_positions = controlled_fold_recipe._positions(candidate_positions)
                candidate_velocities = controlled_fold_recipe._positions(candidate_velocities)
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
            if valid and fold_controls is not None:
                expected_targets = (targets.copy() if sewing_progress == 1 else
                                    initial_targets + sewing_progress * (targets-initial_targets))
                if not _same_control_array(substep_targets, expected_targets):
                    raise ValueError("Controlled step mutated the original sewing target interpolation")
                _validate_fold_step(solver, controlled_fold_recipe, fold_controls, end_fraction, options,
                                    candidate_positions, record["step"])
            if valid and sewing_controls is not None:
                from solver_sewing_activation import validate_sewing_activation
                expected_activation = sewing_controls.parameters(end_fraction)
                expected_targets = (targets.copy() if sewing_progress == 1 else
                                    initial_targets + sewing_progress * (targets - initial_targets))
                reported_activation = step_report.get("sewingActivation")
                if reported_activation is None:
                    raise ValueError("Step sewing activation diagnostics are required")
                reported_activation = validate_sewing_activation(reported_activation, len(expected_activation))
                active_rows = np.flatnonzero(expected_activation > 0).tolist()
                pending_rows = np.flatnonzero(expected_activation == 0).tolist()
                reported_active, reported_pending = step_report.get("activeSewingRows"), step_report.get("pendingSewingRows")
                if (not np.array_equal(options["sewing_activation"], expected_activation)
                        or not np.array_equal(substep_targets, expected_targets)
                        or step_report.get("sewingActivationExplicit") is not True
                        or not np.array_equal(reported_activation, expected_activation)
                        or type(reported_active) is not list or type(reported_pending) is not list
                        or any(type(row) is not int for row in [*reported_active, *reported_pending])
                        or reported_active != active_rows or reported_pending != pending_rows):
                    raise ValueError("Step sewing controls or row diagnostics differ from the captured schedule")
            if valid and (gripper_controls is not None or sewing_controls is not None or fold_controls is not None):
                # Work belongs to the accepted transition and must be checked
                # before its immutable journal outcome is written. Trial
                # controls always use original fractions; retries do not
                # advance the state or accumulate rejected work.
                from solver_energy_balance import global_energy_transition
                old_sewing_progress, old_fold_progress = (schedule.progress(start_fraction) if schedule else
                                                          (start_fraction, start_fraction))
                old_targets = (targets.copy() if old_sewing_progress == 1 else
                               initial_targets + old_sewing_progress * (targets - initial_targets))
                energy_options = {}
                if gripper_controls is not None:
                    old_grip_targets, old_grip_activation = gripper_controls.parameters(start_fraction)
                    energy_options.update(previous_gripper_targets=old_grip_targets,
                                          previous_gripper_activation=old_grip_activation,
                                          gripper_targets=options["gripper_targets"],
                                          gripper_activation=options["gripper_activation"])
                if sewing_controls is not None:
                    energy_options.update(previous_sewing_activation=sewing_controls.parameters(start_fraction),
                                          sewing_activation=sewing_controls.parameters(end_fraction))
                if fold_recipe is not None:
                    old_fold = (fold_targets.copy() if old_fold_progress == 1 else
                                initial_fold_targets + old_fold_progress * (fold_targets - initial_fold_targets))
                    energy_options.update(previous_fold_targets=old_fold,
                                          fold_targets=options["fold_targets"])
                if fold_controls is not None:
                    old_fold, old_activation = fold_controls.parameters(start_fraction)
                    new_fold, new_activation = fold_controls.parameters(end_fraction)
                    energy_options.update(previous_fold_targets=old_fold, previous_fold_activation=old_activation,
                                          fold_targets=new_fold, fold_activation=new_activation)
                    # Work is computed before publication. Isolated inputs
                    # preserve the last accepted state even if a helper fails
                    # after mutation; successful mutation also rejects.
                    work_states = [array.copy() for array in (current_positions, candidate_positions,
                        current_velocities, candidate_velocities, old_targets, substep_targets)]
                    work_options = {key: value.copy() for key, value in energy_options.items()}
                    observed_arrays = [*work_states, *work_options.values()]
                    snapshots = [array.copy() for array in observed_arrays]
                    energy = global_energy_transition(solver, *work_states, duration, **work_options)
                    if any(not _same_control_array(actual, expected) for actual, expected in zip(observed_arrays, snapshots)):
                        raise ValueError("Controlled-fold work helper mutated transition inputs")
                    _validate_fold_energy(energy)
                else:
                    energy = global_energy_transition(solver, current_positions, candidate_positions,
                        current_velocities, candidate_velocities, old_targets, substep_targets, duration,
                        **energy_options)
                step_report = dict(step_report, energyBalance=energy)
                if gripper_controls is not None:
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
                    step_report = dict(step_report, gripperMomentum={
                        "changeKgMPerS": momentum.tolist(), "externalImpulseNs": impulse.tolist(),
                        "residualNs": error.tolist(), "toleranceNs": float(tolerance),
                        "scope": "Backward-Euler force at the new state on free cloth; virtual gripper impulse, not isolated-cloth momentum conservation"})
                record["step"], nonfinite = diagnostic_json(step_report)
                record["nonfiniteDiagnostics"] = nonfinite
                if nonfinite:
                    raise ValueError("Controlled transition has nonfinite work or momentum diagnostics")
                if fold_controls is not None:
                    _validate_fold_step(solver, controlled_fold_recipe, fold_controls, end_fraction, options,
                                        candidate_positions, record["step"])
                    final_residual = record["step"].get("gradientInfinityNorm")
                    if (record["step"].get("converged") is not True
                            or isinstance(final_residual, bool)
                            or not isinstance(final_residual, (int, float))
                            or not 0 <= final_residual <= 1e-6):
                        raise ValueError("Controlled-fold final convergence diagnostics changed before publication")
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
                                "linear sewing progress over the original physical interval" if (gripper_controls is not None or sewing_controls is not None) else
                                "linear over the original physical interval"),
        **({"sewingActivationInterpolation": "captured monotone piecewise-linear canonical-row activation over the original physical interval"}
           if sewing_controls is not None else {}),
        **({"gripperInterpolation": "captured piecewise-linear material-point targets and activation over the original physical interval"}
           if gripper_controls is not None else {}),
        **({"foldControlInterpolation": "explicit per-hinge targets and activation sampled by exact interpolation at original dyadic fractions; no captured source or construction admission"}
           if fold_controls is not None else {}),
    }
