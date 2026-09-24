import math
from fractions import Fraction

import numpy as np

from solver_energy_change import membrane_energy_change


def _weighted_sewing_transition(solver, previous, positions, previous_targets, targets,
                                previous_activation, activation):
    """Stable discrete work, conditional on the existing sampled geometry.

    Vector anchors are accumulated from the binary source coefficients and
    positions. Distance lengths and normal frame normals retain the existing
    validated binary64 geometry convention; rational work accumulation does
    not certify exact real-valued distances or normals.
    """
    from solver_sewing_activation import validate_sewing_activation
    count = solver.sewing.shape[0]
    before_weights = validate_sewing_activation(previous_activation, count)
    weights = validate_sewing_activation(activation, count)
    if (isinstance(solver.compliance, (bool, np.bool_)) or not np.isscalar(solver.compliance)
            or not np.isfinite(solver.compliance) or solver.compliance <= 0
            or not np.isfinite(solver.sewing.data).all()):
        raise ValueError("Finite sewing coefficients and positive compliance required")
    mode = getattr(solver, "sewing_mode", "vector")
    union = (before_weights > 0) | (weights > 0)
    old_geometry = new_geometry = None
    if mode in ("distance", "normal-offset"):
        # Validate static targets/frames for all rows, but only evaluate
        # dynamic geometry where an endpoint or target-first change needs it.
        old_potential = solver.sewing_potential(previous_targets, activation=union.astype(float))
        new_potential = solver.sewing_potential(targets, activation=weights)
        old_geometry = old_potential.geometry(previous)
        new_geometry = new_potential.geometry(positions)
    elif mode != "vector":
        raise ValueError("Unsupported weighted sewing mode")
    zero = Fraction(0)
    compliance = Fraction(float(solver.compliance))
    coefficient = 1 / (2 * compliance)
    totals = {key: zero for key in ("before", "after", "fixed", "motion", "change", "target", "activation",
                                     "parameter", "increase", "release")}
    matrix = solver.sewing.tocsr()
    before_coordinates, after_coordinates = {}, {}

    def coordinate(cache, values, vertex, axis):
        key = vertex, axis
        if key not in cache:
            cache[key] = Fraction(float(values[vertex, axis]))
        return cache[key]

    def anchor(row, after=False):
        cache, values = (after_coordinates, positions) if after else (before_coordinates, previous)
        lower, upper = matrix.indptr[row:row + 2]
        return [sum((Fraction(float(weight)) * coordinate(cache, values, vertex, axis)
                     for vertex, weight in zip(matrix.indices[lower:upper], matrix.data[lower:upper])), zero)
                for axis in range(3)]

    def squared(values):
        return sum((value * value for value in values), zero)

    for row in np.flatnonzero(union):
        old, new = Fraction(float(before_weights[row])), Fraction(float(weights[row]))
        if mode == "distance":
            length = Fraction(float(old_geometry[1][row]))
            old_error = [length - Fraction(float(previous_targets[row]))] if old else [zero]
            fixed_error = [length - Fraction(float(targets[row]))]
        else:
            start_anchor = anchor(row)
            if mode == "normal-offset":
                normal = [Fraction(float(value)) for value in old_geometry[2][row]]
                side = int(solver.sewing_sides[row])
                old_target = [Fraction(float(previous_targets[row])) * side * value for value in normal]
                new_target = [Fraction(float(targets[row])) * side * value for value in normal]
            else:
                old_target = [Fraction(float(value)) for value in previous_targets[row]]
                new_target = [Fraction(float(value)) for value in targets[row]]
            old_error = [value - target for value, target in zip(start_anchor, old_target)] if old else [zero] * 3
            fixed_error = [value - target for value, target in zip(start_anchor, new_target)]
        old_norm, fixed_norm = squared(old_error), squared(fixed_error)
        before = coefficient * old * old_norm
        fixed = coefficient * new * fixed_norm
        target_work = coefficient * old * (fixed_norm - old_norm)
        activation_work = coefficient * (new - old) * fixed_norm
        parameter_work = coefficient * (new * fixed_norm - old * old_norm)
        motion = after = zero
        if new:
            start_anchor, end_anchor = anchor(row), anchor(row, after=True)
            delta = [end - start for start, end in zip(start_anchor, end_anchor)]
            if mode == "distance":
                # Rationalized length change retains motion below the ulp of a
                # rounded endpoint length. No norm difference or squared
                # floating endpoint-energy subtraction is used.
                initial_vector = [Fraction(float(value)) for value in old_geometry[0][row]]
                final_length = Fraction(float(new_geometry[1][row]))
                length_change = sum((change * (2 * value + change)
                                     for value, change in zip(initial_vector, delta)), zero) / (length + final_length)
                error_change = [length_change]
                after_norm = (final_length - Fraction(float(targets[row]))) ** 2
            elif mode == "normal-offset":
                final_normal = [Fraction(float(value)) for value in new_geometry[2][row]]
                signed_target = Fraction(float(targets[row])) * side
                error_change = [change - signed_target * (end - start)
                                for change, start, end in zip(delta, normal, final_normal)]
                after_norm = squared([value + change for value, change in zip(fixed_error, error_change)])
            else:
                error_change = delta
                after_norm = squared([value + change for value, change in zip(fixed_error, delta)])
            motion = coefficient * new * sum((change * (2 * value + change)
                                              for value, change in zip(fixed_error, error_change)), zero)
            after = coefficient * new * after_norm
        for key, value in (("before", before), ("fixed", fixed), ("after", after), ("motion", motion),
                           ("change", parameter_work + motion), ("target", target_work),
                           ("activation", activation_work), ("parameter", parameter_work),
                           ("increase", max(activation_work, zero)), ("release", max(-activation_work, zero))):
            totals[key] += value
    try:
        values = {key: float(value) for key, value in totals.items()}
        fixed_plus_activation = float(totals["motion"] + totals["activation"])
        rounded_sum = math.fsum((values["target"], values["activation"]))
        bound = math.fsum(math.ulp(values[key]) for key in ("parameter", "target", "activation"))
        bound += math.ulp(rounded_sum)
    except (OverflowError, ValueError) as error:
        raise ValueError("Finite weighted sewing energy and parameter-work fields required") from error
    return {
        "sewingBeforeJoules": values["before"], "sewingAfterJoules": values["after"],
        "sewingFixedPositionAfterJoules": values["fixed"], "sewingFixedParameterChangeJoules": values["motion"],
        "sewingChangeJoules": values["change"], "sewingTargetParameterWorkJoules": values["target"],
        "sewingActivationParameterWorkJoules": values["activation"], "sewingParameterWorkJoules": values["parameter"],
        "sewingActivationIncreaseWorkJoules": values["increase"], "sewingReleaseEnergyRemovedJoules": values["release"],
        "sewingParameterWorkComponentSumErrorBoundJoules": bound,
    }, fixed_plus_activation


def global_energy_transition(solver, previous, positions, previous_velocities, velocities,
                             previous_targets, targets, dt, *, previous_fold_targets=None, fold_targets=None,
                             previous_gripper_targets=None, gripper_targets=None,
                             previous_gripper_activation=None, gripper_activation=None,
                             previous_sewing_activation=None, sewing_activation=None,
                             previous_fold_activation=None, fold_activation=None):
    if (previous_sewing_activation is None) != (sewing_activation is None):
        raise ValueError("Both endpoint sewing activations are required for energy accounting")
    weighted_sewing = previous_sewing_activation is not None
    fold_recipe = getattr(solver, "fold_actuation", None)
    controlled_fold_recipe = getattr(solver, "controlled_fold_actuation", None)
    controlled_old_fold = controlled_new_fold = None
    if controlled_fold_recipe is not None:
        from solver_controlled_fold import ControlledFoldActuation, _binary64
        if not isinstance(controlled_fold_recipe, ControlledFoldActuation) or fold_recipe is not None:
            raise ValueError("One validated controlled-fold recipe, exclusive of legacy fold actuation, required")
        if any(value is None for value in (previous_fold_targets, fold_targets,
                                          previous_fold_activation, fold_activation)):
            raise ValueError("Both endpoint controlled-fold targets and activations are required for energy accounting")
        # Preserve raw numeric admission before the legacy global coercion can
        # hide Boolean or inexact integer coordinates. Targets and activation
        # likewise enter the primitive without any intermediate conversion.
        previous = controlled_fold_recipe._positions(previous)
        positions = controlled_fold_recipe._positions(positions)
        previous_velocities = controlled_fold_recipe._positions(previous_velocities)
        velocities = controlled_fold_recipe._positions(velocities)
        dt = _binary64(dt)
        controlled_old_fold = controlled_fold_recipe.potential(previous_fold_targets, previous_fold_activation)
        controlled_new_fold = controlled_fold_recipe.potential(fold_targets, fold_activation)
    elif previous_fold_activation is not None or fold_activation is not None:
        raise ValueError("Fold activation energy parameters require a controlled-fold recipe")
    gripper_recipe = getattr(solver, "material_grippers", None)
    gripper_parameters = (previous_gripper_targets, gripper_targets,
                          previous_gripper_activation, gripper_activation)
    if gripper_recipe is None:
        if any(value is not None for value in gripper_parameters):
            raise ValueError("Gripper energy parameters require a material gripper recipe")
    elif any(value is None for value in gripper_parameters):
        raise ValueError("Both endpoint gripper targets and activation are required for energy accounting")
    previous, positions, previous_velocities, velocities, previous_targets, targets = [
        np.asarray(value, dtype=float) for value in
        (previous, positions, previous_velocities, velocities, previous_targets, targets)]
    state_shape = (len(solver.mass), 3)
    normal_mode = getattr(solver, "sewing_mode", "vector") == "normal-offset"
    distance_mode = getattr(solver, "sewing_mode", "vector") == "distance"
    target_shape = (solver.sewing.shape[0],) if distance_mode or normal_mode else (solver.sewing.shape[0], 3)
    if (any(value.shape != state_shape for value in (previous, positions, previous_velocities, velocities))
            or any(value.shape != target_shape for value in (previous_targets, targets))
            or not all(np.isfinite(value).all() for value in
                       (previous, positions, previous_velocities, velocities, previous_targets, targets))
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("Finite correctly shaped energy states and positive timestep required")
    if (np.any(previous_velocities[~solver.active] != 0)
            or np.any(velocities[~solver.active] != 0)
            or not np.array_equal(previous[~solver.active], positions[~solver.active])):
        raise ValueError("Energy accounting requires stationary fixed vertices")
    displacement = positions - previous
    if not np.allclose(velocities, displacement / dt, rtol=1e-10, atol=1e-12):
        raise ValueError("Energy accounting requires reconstructed step velocities")
    coefficients = np.concatenate((-solver.poses.sum(axis=1)[:, None, :], solver.poses), axis=1)
    deformation = np.einsum("fvc,fva->fca", coefficients, previous[solver.faces])
    delta_deformation = np.einsum("fvc,fva->fca", coefficients, displacement[solver.faces])
    membrane_change = membrane_energy_change(deformation, delta_deformation, solver.areas, solver.materials[:, :3])
    bending_change = solver.bending.energy_change(previous, positions)
    barrier = getattr(solver, "fold_barrier", None)
    barrier_change = barrier.energy_change(previous, positions) if barrier is not None else 0.
    contact = getattr(solver, "contact", None)
    contact_change = contact.energy_change(previous, positions) if contact is not None else 0.
    velocity_change = velocities - previous_velocities
    kinetic_change = float(np.sum(solver.mass[:, None] *
                                 (previous_velocities + .5 * velocity_change) * velocity_change))
    sewing_accounting = {}
    if weighted_sewing:
        sewing_accounting, sewing_fixed_plus_activation = _weighted_sewing_transition(
            solver, previous, positions, previous_targets, targets, previous_sewing_activation, sewing_activation)
        target_work = sewing_accounting["sewingTargetParameterWorkJoules"]
        sewing_change = sewing_accounting["sewingChangeJoules"]
        fixed_target_change = sewing_accounting["sewingFixedParameterChangeJoules"]
        sewing_before, sewing_after = sewing_accounting["sewingBeforeJoules"], sewing_accounting["sewingAfterJoules"]
    elif normal_mode:
        old_potential = solver.sewing_potential(previous_targets)
        new_potential = solver.sewing_potential(targets)
        previous_residual = old_potential.residual(previous) * np.sqrt(solver.compliance)
        fixed_residual = new_potential.residual(previous) * np.sqrt(solver.compliance)
        sewn_displacement = new_potential.residual(positions) * np.sqrt(solver.compliance) - fixed_residual
    elif distance_mode:
        from solver_distance_sewing import DistanceSewing
        sewing = DistanceSewing(solver.sewing, targets, solver.compliance)
        DistanceSewing(solver.sewing, previous_targets, solver.compliance)
        vectors, lengths = sewing.geometry(previous)
        _, final_lengths = sewing.geometry(positions)
        delta_vectors = solver.sewing @ displacement
        sewn_displacement = np.sum(delta_vectors * (2 * vectors + delta_vectors), axis=1) / (lengths + final_lengths)
        previous_residual = lengths - previous_targets
    else:
        previous_residual = solver.sewing @ previous - previous_targets
        sewn_displacement = solver.sewing @ displacement
    if not weighted_sewing:
        target_change = previous_residual - fixed_residual if normal_mode else targets - previous_targets
        target_work = float(np.sum((-previous_residual + .5 * target_change) * target_change) / solver.compliance)
        residual_change = sewn_displacement - target_change
        sewing_change = float(np.sum((previous_residual + .5 * residual_change) * residual_change) / solver.compliance)
        fixed_target_residual = previous_residual - target_change
        fixed_target_change = float(np.sum((fixed_target_residual + .5 * sewn_displacement) * sewn_displacement)
                                    / solver.compliance)
        sewing_before = float(np.sum(previous_residual ** 2) / (2 * solver.compliance))
        sewing_after = float(np.sum((previous_residual + residual_change) ** 2) / (2 * solver.compliance))
    fold_change, fold_work, fold_fixed_change, fold_before, fold_after = 0., 0., 0., 0., 0.
    fold_accounting = {}
    fold_parameter_work = fold_activation_work = 0.
    if controlled_fold_recipe is not None:
        work = controlled_fold_recipe.parameter_energy_change(previous, previous_fold_targets,
            previous_fold_activation, fold_targets, fold_activation)
        try:
            if (work["parameterOrder"] != "target-first-at-old-activation-then-activation-at-new-target"
                    or work["coefficientPolicy"] != "rounded-binary64-stiffness-times-activation-v1"):
                raise ValueError("Controlled-fold energy accounting requires the declared coefficient and target-first work policies")
            keys = ("totalParameterWorkJoules", "targetParameterWorkJoules", "activationParameterWorkJoules",
                    "activationIncreaseWorkJoules", "releaseEnergyRemovedJoules", "roundedComponentSumErrorBoundJoules")
            if any(type(work[key]) not in (int, float) for key in keys):
                raise ValueError("Non-Boolean controlled-fold work values required")
            (fold_parameter_work, fold_work, fold_activation_work, fold_increase, fold_release, fold_rounding_bound) = (
                float(work[key]) for key in keys)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("Complete finite controlled-fold parameter-work accounting required") from error
        if (not all(math.isfinite(value) for value in (fold_parameter_work, fold_work, fold_activation_work,
                                                      fold_increase, fold_release, fold_rounding_bound))
                or min(fold_increase, fold_release, fold_rounding_bound) < 0):
            raise ValueError("Finite controlled-fold work and nonnegative release/rounding quantities required")
        fold_before, fold_after = controlled_old_fold.energy(previous), controlled_new_fold.energy(positions)
        fold_fixed_position_after = controlled_new_fold.energy(previous)
        fold_fixed_change = controlled_new_fold.energy_change(previous, positions)
        # The direct primitive total retains cancellation that separately
        # rounded target/activation components cannot necessarily reproduce.
        fold_change = math.fsum((fold_parameter_work, fold_fixed_change))
        fold_accounting = {
            "foldFixedPositionAfterJoules": fold_fixed_position_after,
            "foldFixedParameterChangeJoules": fold_fixed_change,
            "foldParameterWorkJoules": fold_parameter_work,
            "foldActivationParameterWorkJoules": fold_activation_work,
            "foldActivationIncreaseWorkJoules": fold_increase,
            "foldReleaseEnergyRemovedJoules": fold_release,
            "foldParameterWorkComponentSumErrorBoundJoules": fold_rounding_bound,
        }
    elif fold_recipe is None:
        if previous_fold_targets is not None or fold_targets is not None:
            raise ValueError("Fold energy targets require an actuator recipe")
    else:
        old_fold = fold_recipe.potential(previous_fold_targets)
        new_fold = fold_recipe.potential(fold_targets)
        angle_change = new_fold.rest_angles - old_fold.rest_angles
        old_error = old_fold.angles(previous) - old_fold.rest_angles
        fold_work = float(np.sum(old_fold.weights * angle_change * (-old_error + .5 * angle_change)))
        fold_fixed_change = new_fold.energy_change(previous, positions)
        fold_change = fold_work + fold_fixed_change
        fold_before, fold_after = old_fold.energy(previous), new_fold.energy(positions)
    gripper_before, gripper_after, gripper_fixed_position_after, gripper_fixed_change = 0., 0., 0., 0.
    gripper_work, gripper_target_work, gripper_activation_work = 0., 0., 0.
    gripper_activation_increase, gripper_release, gripper_rounding_bound = 0., 0., 0.
    if gripper_recipe is not None:
        old_gripper = gripper_recipe.potential(previous_gripper_targets, previous_gripper_activation)
        new_gripper = gripper_recipe.potential(gripper_targets, gripper_activation)
        gripper_before = old_gripper.energy(previous)
        gripper_after = new_gripper.energy(positions)
        gripper_fixed_position_after = new_gripper.energy(previous)
        gripper_fixed_change = new_gripper.energy_change(previous, positions)
        work = gripper_recipe.parameter_energy_change(previous, previous_gripper_targets,
            previous_gripper_activation, gripper_targets, gripper_activation)
        try:
            if work["parameterOrder"] != "target-first-at-old-activation-then-activation-at-new-target":
                raise ValueError("Gripper energy accounting requires target-first parameter work")
            (gripper_work, gripper_target_work, gripper_activation_work,
             gripper_activation_increase, gripper_release, gripper_rounding_bound) = [float(work[key]) for key in (
                "totalParameterWorkJoules", "targetParameterWorkJoules", "activationParameterWorkJoules",
                "activationIncreaseWorkJoules", "releaseEnergyRemovedJoules", "roundedComponentSumErrorBoundJoules")]
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("Complete finite gripper parameter-work accounting required") from error
        if (not all(np.isfinite(value) for value in (gripper_work, gripper_target_work, gripper_activation_work,
                gripper_activation_increase, gripper_release, gripper_rounding_bound))
                or min(gripper_activation_increase, gripper_release, gripper_rounding_bound) < 0):
            raise ValueError("Finite gripper parameter work and nonnegative release/rounding quantities required")
    gripper_change = math.fsum((gripper_work, gripper_fixed_change))
    if controlled_fold_recipe is not None:
        sewing_parameter_work = sewing_accounting["sewingParameterWorkJoules"] if weighted_sewing else target_work
        sewing_motion_plus_activation = sewing_fixed_plus_activation if weighted_sewing else fixed_target_change
        other_motion = (membrane_change, bending_change, barrier_change, contact_change, kinetic_change,
                        fold_fixed_change, gripper_fixed_change)
        mechanical_change = math.fsum((membrane_change, bending_change, barrier_change, contact_change,
                                      kinetic_change, sewing_change, fold_change, gripper_work, gripper_fixed_change))
        target_parameter_work = math.fsum((target_work, fold_work, gripper_target_work))
        external_parameter_work = math.fsum((sewing_parameter_work, fold_parameter_work, gripper_work))
        minus_parameter_work = math.fsum((*other_motion, fixed_target_change))
        minus_target_work = math.fsum((*other_motion, sewing_motion_plus_activation,
                                       fold_activation_work, gripper_activation_work))
    elif weighted_sewing:
        other_motion = (membrane_change, bending_change, barrier_change, contact_change, kinetic_change,
                        fold_fixed_change, gripper_fixed_change)
        mechanical_change = math.fsum((membrane_change, bending_change, barrier_change, contact_change,
                                      kinetic_change, sewing_change, fold_change, gripper_work, gripper_fixed_change))
        target_parameter_work = math.fsum((target_work, fold_work, gripper_target_work))
        external_parameter_work = math.fsum((sewing_accounting["sewingParameterWorkJoules"], fold_work, gripper_work))
        minus_parameter_work = math.fsum((*other_motion, fixed_target_change))
        minus_target_work = math.fsum((*other_motion, sewing_fixed_plus_activation, gripper_activation_work))
    else:
        # Retain the exact legacy arithmetic when sewing activation is omitted.
        original_mechanical_change = membrane_change + bending_change + barrier_change + contact_change + kinetic_change + sewing_change + fold_change
        original_fixed_target_change = membrane_change + bending_change + barrier_change + contact_change + kinetic_change + fixed_target_change + fold_fixed_change
        mechanical_change = (math.fsum((original_mechanical_change, gripper_work, gripper_fixed_change))
                             if gripper_recipe is not None else original_mechanical_change)
        target_parameter_work = (math.fsum((target_work, fold_work, gripper_target_work))
                                 if gripper_recipe is not None else target_work + fold_work)
        external_parameter_work = (math.fsum((target_work, fold_work, gripper_work))
                                   if gripper_recipe is not None else target_work + fold_work)
        minus_target_work = (math.fsum((original_fixed_target_change, gripper_fixed_change, gripper_activation_work))
                             if gripper_recipe is not None else original_fixed_target_change)
        minus_parameter_work = (math.fsum((original_fixed_target_change, gripper_fixed_change))
                                if gripper_recipe is not None else original_fixed_target_change)
    report = {
        "membraneChangeJoules": membrane_change,
        "bendingChangeJoules": bending_change,
        "bendingBeforeJoules": solver.bending.energy(previous),
        "bendingAfterJoules": solver.bending.energy(positions),
        "foldBarrierChangeJoules": barrier_change,
        "foldBarrierBeforeJoules": barrier.energy(previous) if barrier is not None else 0.,
        "foldBarrierAfterJoules": barrier.energy(positions) if barrier is not None else 0.,
        "contactChangeJoules": contact_change,
        "contactBeforeJoules": contact.energy(previous) if contact is not None else 0.,
        "contactAfterJoules": contact.energy(positions) if contact is not None else 0.,
        "kineticChangeJoules": kinetic_change,
        "sewingChangeJoules": sewing_change,
        "sewingBeforeJoules": sewing_before,
        "sewingAfterJoules": sewing_after,
        "foldActuationChangeJoules": fold_change,
        "foldActuationBeforeJoules": fold_before,
        "foldActuationAfterJoules": fold_after,
        "foldTargetParameterWorkJoules": fold_work,
        "gripperBeforeJoules": gripper_before,
        "gripperAfterJoules": gripper_after,
        "gripperFixedPositionAfterJoules": gripper_fixed_position_after,
        "gripperFixedParameterChangeJoules": gripper_fixed_change,
        "gripperChangeJoules": gripper_change,
        "gripperParameterWorkJoules": gripper_work,
        "gripperTargetParameterWorkJoules": gripper_target_work,
        "gripperActivationParameterWorkJoules": gripper_activation_work,
        "gripperActivationIncreaseWorkJoules": gripper_activation_increase,
        "gripperReleaseEnergyRemovedJoules": gripper_release,
        "gripperParameterWorkComponentSumErrorBoundJoules": gripper_rounding_bound,
        "targetParameterWorkJoules": target_parameter_work,
        "externalParameterWorkJoules": external_parameter_work,
        "mechanicalChangeJoules": mechanical_change,
        "mechanicalChangeMinusTargetWorkJoules": minus_target_work,
        "mechanicalChangeMinusParameterWorkJoules": minus_parameter_work,
        **sewing_accounting,
        **fold_accounting,
    }
    if not all(np.isfinite(value) for value in report.values()):
        raise ValueError("Finite energy balance required")
    return {
        **report,
        "accepted": False,
        "scope": ("Global membrane/elastic-bending/sewing dynamics with experimental frictionless surface contact. " if contact is not None else "Contact-disabled global membrane/elastic-bending/sewing dynamics. ") + "Includes optional local angular fold barriers, prescribed fold actuation and compliant material grippers. Target work counts target changes only. External parameter work also includes gripper activation/release at the previous positions, with target changes first at old activation and then activation changes at new targets. Release energy removed is a nonnegative discrete potential reduction, not claimed physical dissipation. Fixed-parameter changes use the new parameters during motion. Rounded parameter-work components may differ from the directly evaluated total within the reported component-sum error bound. These are discrete potential changes, not continuous actuator work. The signed remainder includes numerical dissipation or gain, not calibrated material damping or garment acceptance."
                 + (" Sewing target-first parameter work also includes signed sewing activation and release at the previous positions; pending rows are skipped. Rational work accumulation is conditional on the existing sampled binary64 distance lengths and frame normals, not an exact real-geometry proof. The source construction schedule may forbid release even though this mathematical accounting supports it." if weighted_sewing else "")
                 + (" Controlled-fold external parameter work uses the same rounded binary64 stiffness-times-activation coefficient as its potential: target changes first at the old coefficient, then activation changes at the new target. It includes signed engagement/release work; target work alone excludes it. Inactive hinges skip actuator angle evaluation without waiving any independent cloth, triangle, contact or hinge-path guard. Parameter work is exact quadratic arithmetic conditional on sampled binary64 angles; stable fixed-parameter angle increments and endpoint diagnostics have distinct rounding. No continuous work, calibrated damping, phase completion or refined-source execution is certified." if controlled_fold_recipe is not None else ""),
    }
