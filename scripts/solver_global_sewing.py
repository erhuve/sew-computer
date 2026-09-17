import numpy as np
import warnings
from scipy.optimize import OptimizeResult, least_squares
from scipy.sparse import coo_matrix, diags, kron
from scipy.sparse.linalg import MatrixRankWarning, splu, spsolve


def _positive_definite_direction(matrix, gradient):
    matrix = matrix.tocsc()
    if not np.isfinite(matrix.data).all():
        return None
    asymmetry = matrix - matrix.T
    magnitude = max(1.0, float(np.max(np.abs(matrix.data), initial=0)))
    if np.max(np.abs(asymmetry.data), initial=0) > 1e-12 * magnitude:
        return None
    try:
        factor = splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0,
                      options={"SymmetricMode": True, "Equil": False})
    except RuntimeError:
        return None
    pivots = factor.U.diagonal()
    if (not np.array_equal(factor.perm_r, factor.perm_c) or not np.isfinite(pivots).all()
            or np.any(pivots <= 1e-12 * magnitude)):
        return None
    normalized_upper = diags(1 / pivots) @ factor.U
    error = normalized_upper - factor.L.T
    scale = max(1.0, float(np.max(np.abs(factor.L.data), initial=0)))
    if np.max(np.abs(error.data), initial=0) > 1e-10 * scale:
        return None
    direction = factor.solve(-gradient)
    residual = matrix @ direction + gradient
    if (not np.isfinite(direction).all()
            or np.max(np.abs(residual), initial=0) > 1e-8 * max(1e-6, float(np.max(np.abs(gradient))))):
        return None
    return direction


def _direct_descent(evaluate, start, max_evaluations, hessian, objective, exact_hessian=None,
                    inertia_diagonal=None, gradient_function=None, energy_change_function=None):
    positions = start.copy()
    residual = evaluate(positions)
    energy = objective(positions)
    evaluations = 1
    history = [energy]
    direction_steps = {"exact": 0, "projected": 0, "shifted": 0}
    direction_history = []
    status, message = 0, "Residual evaluation budget exhausted"
    while evaluations < max_evaluations:
        gradient = gradient_function(positions) if gradient_function else evaluate(positions, True).T @ residual
        if np.max(np.abs(gradient)) <= 1e-6:
            status, message = 1, "Stationarity tolerance satisfied"
            break
        accepted = False
        metrics = [("exact", exact_hessian), ("projected", hessian)] if exact_hessian else [("projected", hessian)]
        for name, metric in metrics:
            matrix = metric(positions)
            shift_report = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    if name == "exact" and inertia_diagonal is not None:
                        from solver_global_shift import shifted_positive_definite_direction
                        direction, shift_report = shifted_positive_definite_direction(matrix, gradient, inertia_diagonal)
                        if shift_report["lambda"]:
                            name = "shifted"
                    else:
                        direction = (_positive_definite_direction(matrix, gradient) if name == "exact"
                                     else spsolve(matrix.tocsc(), -gradient))
            except (MatrixRankWarning, np.linalg.LinAlgError):
                continue
            if direction is None:
                continue
            slope = float(np.dot(gradient, direction))
            if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
                continue
            scale = 1.0
            for _ in range(24):
                if evaluations >= max_evaluations:
                    break
                candidate = positions + scale * direction
                candidate_residual = evaluate(candidate)
                candidate_energy = objective(candidate)
                evaluations += 1
                finite_candidate = np.isfinite(candidate_residual).all() and np.isfinite(candidate_energy)
                change = (energy_change_function(positions, candidate) if energy_change_function
                          and finite_candidate else candidate_energy - energy)
                if (finite_candidate and np.isfinite(change) and not np.array_equal(candidate, positions)
                        and change <= 1e-4 * scale * slope):
                    positions, residual, energy = candidate, candidate_residual, candidate_energy
                    history.append(energy)
                    direction_steps[name] += 1
                    direction_history.append({"method": name, "scale": scale, "shift": shift_report})
                    accepted = True
                    break
                scale *= .5
            if accepted or evaluations >= max_evaluations:
                break
        if not accepted:
            status, message = -2, "Line search failed or evaluation budget exhausted"
            break
    gradient = gradient_function(positions) if gradient_function else evaluate(positions, True).T @ residual
    if np.max(np.abs(gradient)) <= 1e-6:
        status, message = 1, "Stationarity tolerance satisfied"
    return OptimizeResult(x=positions, cost=energy, nfev=evaluations, success=status == 1,
                          status=status, message=message, energy_history=history,
                          exact_steps=direction_steps["exact"], projected_steps=direction_steps["projected"],
                          shifted_steps=direction_steps["shifted"], direction_history=direction_history)


class GlobalSewingSolver:
    def __init__(self, model, rows, compliance):
        self.mass = model.particle_mass.numpy().astype(float)
        self.active = (self.mass > 0) & ((model.particle_flags.numpy() & 1) != 0)
        self.faces = model.tri_indices.numpy().astype(int) if model.tri_indices is not None else np.empty((0, 3), dtype=int)
        self.poses = model.tri_poses.numpy().astype(float) if model.tri_poses is not None else np.empty((0, 2, 2))
        self.areas = model.tri_areas.numpy().astype(float) if model.tri_areas is not None else np.empty(0)
        self.materials = model.tri_materials.numpy().astype(float) if model.tri_materials is not None else np.empty((0, 3))
        if (not all(np.isfinite(value).all() for value in (self.mass, self.poses, self.areas, self.materials))
                or np.any(self.mass < 0) or np.any(self.areas <= 0) or not np.any(self.active)):
            raise ValueError("Finite physical inputs and at least one active positive-mass vertex required")
        if not np.isfinite(compliance) or compliance <= 0:
            raise ValueError("Positive finite physical compliance required")
        if model.edge_bending_properties is not None and np.any(model.edge_bending_properties.numpy() != 0):
            raise ValueError("Global reference does not implement bending; explicitly disable it in diagnostic fixtures")
        if np.any(self.materials[:, 2] != 0):
            raise ValueError("Global reference does not implement membrane damping")
        if np.any(self.materials[:, :2] < 0):
            raise ValueError("Nonnegative membrane materials required")
        if np.any(model.gravity.numpy() != 0):
            raise ValueError("Global reference requires zero gravity")
        if model.spring_count or model.tet_count:
            raise ValueError("Global reference does not implement springs or volumetric elements")
        from solver_embedded_sewing import validate_rows
        rows = validate_rows(rows, len(self.mass), model.particle_colors.numpy())
        if any(abs(sum(row.values())) > 1e-12 for row in rows):
            raise ValueError("Global sewing rows must preserve translation to float64 precision")
        entries = [(index, vertex, coefficient) for index, row in enumerate(rows) for vertex, coefficient in row.items()]
        self.sewing = coo_matrix(([entry[2] for entry in entries],
                                  ([entry[0] for entry in entries], [entry[1] for entry in entries])),
                                 shape=(len(rows), len(self.mass))).tocsr()
        self.compliance = compliance
        self.free = np.flatnonzero(np.repeat(self.active, 3))
        self.sewing_xyz = kron(self.sewing, np.eye(3), format="csr")

    def step(self, previous_positions, previous_velocities, targets, dt, max_evaluations=300, linear_solver="direct"):
        previous = np.asarray(previous_positions, dtype=float)
        velocities = np.asarray(previous_velocities, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if linear_solver not in ("direct", "shifted", "lsmr") or type(max_evaluations) is not int or not 1 <= max_evaluations <= 10000:
            raise ValueError("Supported linear solver and bounded positive evaluation budget required")
        if (previous.shape != (len(self.mass), 3) or velocities.shape != previous.shape
                or targets.shape != (self.sewing.shape[0], 3)
                or not all(np.isfinite(value).all() for value in (previous, velocities, targets))
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError("Finite correctly shaped state and positive timestep required")
        predicted = previous + dt * velocities
        predicted[~self.active] = previous[~self.active]
        inertia_weights = np.repeat(np.sqrt(self.mass) / dt, 3)
        linear_jacobian = diags(inertia_weights, format="csr")
        sewn_jacobian = self.sewing_xyz / np.sqrt(self.compliance)
        fixed = previous.ravel().copy()
        coefficients = np.concatenate((-self.poses.sum(axis=1)[:, None, :], self.poses), axis=1)
        previous_deformation = np.einsum("fvc,fva->fca", coefficients, previous[self.faces])
        if np.any(np.linalg.norm(np.cross(previous_deformation[:, 0], previous_deformation[:, 1]), axis=1) <= 1e-10):
            raise ValueError("Degenerate membrane state cannot initialize the global reference")
        sqrt_mu = np.sqrt(self.areas * self.materials[:, 0])
        lame = self.materials[:, 0] + self.materials[:, 1]
        sqrt_lame = np.sqrt(self.areas * lame)
        alpha = 1 + self.materials[:, 0] / np.maximum(lame, 1e-6)

        def evaluate(free_positions, jacobian=False):
            flat = fixed.copy()
            flat[self.free] = free_positions
            positions = flat.reshape((-1, 3))
            deformation = np.einsum("fvc,fva->fca", coefficients, positions[self.faces])
            first, second = deformation[:, 0], deformation[:, 1]
            area_vectors = np.cross(first, second)
            raw_area_ratios = np.linalg.norm(area_vectors, axis=1)
            if not np.isfinite(deformation).all() or np.any(raw_area_ratios <= 1e-10):
                if jacobian:
                    raise ValueError("Degenerate membrane state cannot supply derivatives")
                return np.full(len(flat) + targets.size + 7 * len(self.faces), np.inf)
            area_ratios = np.maximum(raw_area_ratios, 1e-10)
            membrane = np.concatenate((sqrt_mu[:, None] * deformation.reshape((-1, 6)),
                                       (sqrt_lame * (area_ratios - alpha))[:, None]), axis=1)
            if not jacobian:
                return np.concatenate((inertia_weights * (flat - predicted.ravel()),
                                       (self.sewing @ positions - targets).ravel() / np.sqrt(self.compliance),
                                       membrane.ravel()))
            from scipy.sparse import vstack
            row_indices, column_indices, values = [], [], []
            gradients = np.stack((np.cross(second, area_vectors), np.cross(area_vectors, first)), axis=1)
            gradients /= area_ratios[:, None, None]
            gradients[raw_area_ratios <= 1e-10] = 0
            for local_vertex in range(3):
                for axis in range(3):
                    columns = self.faces[:, local_vertex] * 3 + axis
                    for component in range(2):
                        row_indices.extend(np.arange(len(self.faces)) * 7 + component * 3 + axis)
                        column_indices.extend(columns)
                        values.extend(sqrt_mu * coefficients[:, local_vertex, component])
                    row_indices.extend(np.arange(len(self.faces)) * 7 + 6)
                    column_indices.extend(columns)
                    values.extend(sqrt_lame * np.sum(gradients[:, :, axis] * coefficients[:, local_vertex], axis=1))
            membrane_jacobian = coo_matrix((values, (row_indices, column_indices)),
                                           shape=(7 * len(self.faces), len(flat))).tocsr()
            return vstack((linear_jacobian, sewn_jacobian, membrane_jacobian), format="csr")[:, self.free]

        def objective(free_positions):
            flat = fixed.copy()
            flat[self.free] = free_positions
            positions = flat.reshape((-1, 3))
            deformation = np.einsum("fvc,fva->fca", coefficients, positions[self.faces])
            first, second = deformation[:, 0], deformation[:, 1]
            area_ratio = np.linalg.norm(np.cross(first, second), axis=1)
            if not np.isfinite(deformation).all() or np.any(area_ratio <= 1e-10):
                return float("inf")
            first_norm, second_norm = np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1)
            shear_shape = ((first_norm - second_norm) ** 2
                           + 2 * np.sum(first * second, axis=1) ** 2 / (first_norm * second_norm + area_ratio))
            shear = self.materials[:, 0]
            strain = area_ratio - 1
            membrane_energy = self.areas * (shear * shear_shape / 2 + lame * strain ** 2 / 2
                                           + (shear + lame * (1 - alpha)) * strain)
            inertial = inertia_weights * (flat - predicted.ravel())
            sewing = (self.sewing @ positions - targets).ravel() / np.sqrt(self.compliance)
            return float((inertial @ inertial + sewing @ sewing) / 2 + membrane_energy.sum())

        def gradient_function(free_positions):
            flat = fixed.copy()
            flat[self.free] = free_positions
            positions = flat.reshape((-1, 3))
            deformation = np.einsum("fvc,fva->fca", coefficients, positions[self.faces])
            first, second = deformation[:, 0], deformation[:, 1]
            area_vectors = np.cross(first, second)
            area_ratios = np.linalg.norm(area_vectors, axis=1)
            if not np.isfinite(deformation).all() or np.any(area_ratios <= 1e-10):
                raise ValueError("Degenerate membrane state cannot supply derivatives")
            normals = area_vectors / area_ratios[:, None]
            area_gradients = np.stack((np.cross(second, normals), np.cross(normals, first)), axis=1)
            stress = (self.materials[:, 0, None, None] * deformation
                      + (lame * (area_ratios - alpha))[:, None, None] * area_gradients)
            element_gradient = self.areas[:, None, None] * np.einsum("fvc,fca->fva", coefficients, stress)
            gradient = (inertia_weights ** 2 * (flat - predicted.ravel())
                        + self.sewing_xyz.T @ ((self.sewing @ positions - targets).ravel() / self.compliance))
            np.add.at(gradient.reshape((-1, 3)), self.faces, element_gradient)
            return gradient[self.free]

        def energy_change_function(start_positions, end_positions):
            from solver_energy_change import membrane_energy_change
            flat = fixed.copy()
            flat[self.free] = start_positions
            delta = np.zeros_like(flat)
            delta[self.free] = end_positions - start_positions
            positions, displacement = flat.reshape((-1, 3)), delta.reshape((-1, 3))
            deformation = np.einsum("fvc,fva->fca", coefficients, positions[self.faces])
            delta_deformation = np.einsum("fvc,fva->fca", coefficients, displacement[self.faces])
            try:
                membrane_change = membrane_energy_change(deformation, delta_deformation, self.areas, self.materials[:, :3])
            except ValueError:
                return float("inf")
            inertial = inertia_weights * (flat - predicted.ravel())
            delta_inertial = inertia_weights * delta
            sewing = (self.sewing @ positions - targets).ravel() / np.sqrt(self.compliance)
            delta_sewing = (self.sewing @ displacement).ravel() / np.sqrt(self.compliance)
            return float((inertial + .5 * delta_inertial) @ delta_inertial
                         + (sewing + .5 * delta_sewing) @ delta_sewing + membrane_change)

        linear_hessian = linear_jacobian.T @ linear_jacobian + sewn_jacobian.T @ sewn_jacobian
        element_dofs = (self.faces[:, :, None] * 3 + np.arange(3)).reshape((-1, 9))
        cached_positions, cached_elements = None, None

        def assembled_hessian(free_positions, project_psd):
            nonlocal cached_positions, cached_elements
            from solver_membrane_hessian import membrane_element_derivatives
            if cached_positions is None or not np.array_equal(cached_positions, free_positions):
                flat = fixed.copy()
                flat[self.free] = free_positions
                _, _, cached_elements = membrane_element_derivatives(
                    flat.reshape((-1, 3))[self.faces], self.poses, self.areas, self.materials[:, :3])
                cached_positions = free_positions.copy()
            elements = cached_elements
            if project_psd:
                eigenvalues, eigenvectors = np.linalg.eigh(elements)
                elements = (eigenvectors * np.maximum(eigenvalues, 0)[:, None, :]) @ eigenvectors.transpose(0, 2, 1)
            membrane_hessian = coo_matrix((elements.ravel(),
                (np.broadcast_to(element_dofs[:, :, None], elements.shape).ravel(),
                 np.broadcast_to(element_dofs[:, None, :], elements.shape).ravel())),
                shape=(len(fixed), len(fixed))).tocsr()
            return (linear_hessian + membrane_hessian)[self.free][:, self.free]

        diagonal = diags(self.mass / dt ** 2)
        matrix = diagonal + self.sewing.T @ self.sewing / self.compliance
        rhs = diagonal @ predicted + self.sewing.T @ targets / self.compliance
        active_indices = np.flatnonzero(self.active)
        fixed_indices = np.flatnonzero(~self.active)
        rhs = rhs[active_indices] - matrix[active_indices][:, fixed_indices] @ previous[fixed_indices]
        linear_start = spsolve(matrix[active_indices][:, active_indices].tocsc(), rhs).reshape((-1, 3)).ravel()
        predicted_start = predicted.ravel()[self.free]
        start = min((linear_start, predicted_start, previous.ravel()[self.free]), key=objective)
        initial_energy = objective(start)
        result = _direct_descent(evaluate, start, max_evaluations,
                                lambda positions: assembled_hessian(positions, True), objective,
                                exact_hessian=lambda positions: assembled_hessian(positions, False),
                                inertia_diagonal=inertia_weights[self.free] ** 2 if linear_solver == "shifted" else None,
                                gradient_function=gradient_function,
                                energy_change_function=energy_change_function) if linear_solver in ("direct", "shifted") else least_squares(evaluate, start, jac=lambda positions: evaluate(positions, True),
                               method="trf", tr_solver="lsmr", x_scale="jac", ftol=1e-12, xtol=1e-12,
                               gtol=1e-9, max_nfev=max_evaluations,
                               tr_options={"atol": 1e-12, "btol": 1e-12, "maxiter": max(100, 3 * len(self.free))})
        final = fixed.copy()
        final[self.free] = result.x
        final = final.reshape((-1, 3))
        final_deformation = np.einsum("fvc,fva->fca", coefficients, final[self.faces])
        if np.any(np.linalg.norm(np.cross(final_deformation[:, 0], final_deformation[:, 1]), axis=1) <= 1e-10):
            raise ValueError("Degenerate membrane state rejected by global reference")
        gradient = gradient_function(result.x)
        gradient_norm = float(np.max(np.abs(gradient)))
        return final, (final - previous) / dt, {
            "profile": "experimental-global-membrane-sewing-reference-v6", "accepted": False,
            "converged": bool(result.success and gradient_norm <= 1e-6),
            "optimizerSuccess": bool(result.success), "stationarityToleranceN": 1e-6,
            "linearSolver": linear_solver, "energyHistory": getattr(result, "energy_history", None),
            "exactSteps": getattr(result, "exact_steps", None), "projectedSteps": getattr(result, "projected_steps", None),
            "shiftedSteps": getattr(result, "shifted_steps", None),
            "directionHistory": getattr(result, "direction_history", None),
            "status": int(result.status), "message": result.message,
            "evaluations": int(result.nfev), "initialEnergy": initial_energy, "finalEnergy": objective(result.x),
            "gradientInfinityNorm": gradient_norm,
            "limitations": ["No contact, external forces, bending or membrane damping; diagnostic reference only.",
                            "Stationarity does not certify a local energy minimum or dynamic stability."],
        }
