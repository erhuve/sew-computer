import math

import numpy as np

from solver_energy_change import membrane_energy_change


def global_energy_transition(solver, previous, positions, previous_velocities, velocities,
                             previous_targets, targets, dt, *, previous_fold_targets=None, fold_targets=None,
                             previous_gripper_targets=None, gripper_targets=None,
                             previous_gripper_activation=None, gripper_activation=None):
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
    if normal_mode:
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
    target_change = previous_residual - fixed_residual if normal_mode else targets - previous_targets
    target_work = float(np.sum((-previous_residual + .5 * target_change) * target_change) / solver.compliance)
    residual_change = sewn_displacement - target_change
    sewing_change = float(np.sum((previous_residual + .5 * residual_change) * residual_change) / solver.compliance)
    fixed_target_residual = previous_residual - target_change
    fixed_target_change = float(np.sum((fixed_target_residual + .5 * sewn_displacement) * sewn_displacement)
                                / solver.compliance)
    fold_recipe = getattr(solver, "fold_actuation", None)
    fold_change, fold_work, fold_fixed_change, fold_before, fold_after = 0., 0., 0., 0., 0.
    if fold_recipe is None:
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
    original_mechanical_change = membrane_change + bending_change + barrier_change + contact_change + kinetic_change + sewing_change + fold_change
    original_fixed_target_change = membrane_change + bending_change + barrier_change + contact_change + kinetic_change + fixed_target_change + fold_fixed_change
    # Preserve the previous no-gripper values. With grippers, sum independent
    # changes instead of subtracting potentially huge rounded endpoint energies.
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
        "sewingBeforeJoules": float(np.sum(previous_residual ** 2) / (2 * solver.compliance)),
        "sewingAfterJoules": float(np.sum((previous_residual + residual_change) ** 2) / (2 * solver.compliance)),
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
    }
    if not all(np.isfinite(value) for value in report.values()):
        raise ValueError("Finite energy balance required")
    return {
        **report,
        "accepted": False,
        "scope": ("Global membrane/elastic-bending/sewing dynamics with experimental frictionless surface contact. " if contact is not None else "Contact-disabled global membrane/elastic-bending/sewing dynamics. ") + "Includes optional local angular fold barriers, prescribed fold actuation and compliant material grippers. Target work counts target changes only. External parameter work also includes gripper activation/release at the previous positions, with target changes first at old activation and then activation changes at new targets. Release energy removed is a nonnegative discrete potential reduction, not claimed physical dissipation. Fixed-parameter changes use the new parameters during motion. Rounded parameter-work components may differ from the directly evaluated total within the reported component-sum error bound. These are discrete potential changes, not continuous actuator work. The signed remainder includes numerical dissipation or gain, not calibrated material damping or garment acceptance.",
    }
