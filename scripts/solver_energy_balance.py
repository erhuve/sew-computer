import numpy as np

from solver_energy_change import membrane_energy_change


def global_energy_transition(solver, previous, positions, previous_velocities, velocities,
                             previous_targets, targets, dt):
    previous, positions, previous_velocities, velocities, previous_targets, targets = [
        np.asarray(value, dtype=float) for value in
        (previous, positions, previous_velocities, velocities, previous_targets, targets)]
    state_shape = (len(solver.mass), 3)
    target_shape = (solver.sewing.shape[0], 3)
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
    velocity_change = velocities - previous_velocities
    kinetic_change = float(np.sum(solver.mass[:, None] *
                                 (previous_velocities + .5 * velocity_change) * velocity_change))
    previous_residual = solver.sewing @ previous - previous_targets
    target_change = targets - previous_targets
    target_work = float(np.sum((-previous_residual + .5 * target_change) * target_change) / solver.compliance)
    residual_change = solver.sewing @ displacement - target_change
    sewing_change = float(np.sum((previous_residual + .5 * residual_change) * residual_change) / solver.compliance)
    fixed_target_residual = previous_residual - target_change
    sewn_displacement = solver.sewing @ displacement
    fixed_target_change = float(np.sum((fixed_target_residual + .5 * sewn_displacement) * sewn_displacement)
                                / solver.compliance)
    report = {
        "membraneChangeJoules": membrane_change,
        "bendingChangeJoules": bending_change,
        "bendingBeforeJoules": solver.bending.energy(previous),
        "bendingAfterJoules": solver.bending.energy(positions),
        "foldBarrierChangeJoules": barrier_change,
        "foldBarrierBeforeJoules": barrier.energy(previous) if barrier is not None else 0.,
        "foldBarrierAfterJoules": barrier.energy(positions) if barrier is not None else 0.,
        "kineticChangeJoules": kinetic_change,
        "sewingChangeJoules": sewing_change,
        "sewingBeforeJoules": float(np.sum(previous_residual ** 2) / (2 * solver.compliance)),
        "sewingAfterJoules": float(np.sum((previous_residual + residual_change) ** 2) / (2 * solver.compliance)),
        "targetParameterWorkJoules": target_work,
        "mechanicalChangeJoules": membrane_change + bending_change + barrier_change + kinetic_change + sewing_change,
        "mechanicalChangeMinusTargetWorkJoules": membrane_change + bending_change + barrier_change + kinetic_change + fixed_target_change,
    }
    if not all(np.isfinite(value) for value in report.values()):
        raise ValueError("Finite energy balance required")
    return {
        **report,
        "accepted": False,
        "scope": "Contact-disabled global membrane/elastic-bending/sewing dynamics, including the optional experimental local angular fold barrier. Target work is the discrete potential change at the previous positions; it is not continuous actuator work. The signed remainder includes numerical dissipation or gain, not calibrated material damping or garment acceptance.",
    }
