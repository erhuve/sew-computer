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
                    inertia_diagonal=None, gradient_function=None, energy_change_function=None,
                    coupled_hessian=None, step_limiter=None, guard_assembled_metrics=False):
    if exact_hessian is not None and coupled_hessian is not None:
        raise ValueError("Choose one safeguarded primary search metric")
    if guard_assembled_metrics and inertia_diagonal is None:
        raise ValueError("Guarded assembled search metrics require physical inertia")
    positions = start.copy()
    residual = evaluate(positions)
    energy = objective(positions)
    evaluations = 1
    history = [energy]
    direction_steps = {"exact": 0, "projected": 0, "shifted": 0, "coupled": 0}
    direction_history = []
    status, message = 0, "Residual evaluation budget exhausted"
    while evaluations < max_evaluations:
        gradient = gradient_function(positions) if gradient_function else evaluate(positions, True).T @ residual
        if np.max(np.abs(gradient)) <= 1e-6:
            status, message = 1, "Stationarity tolerance satisfied"
            break
        accepted = False
        primary = ([("coupled", coupled_hessian)] if coupled_hessian else
                   [("exact", exact_hessian)] if exact_hessian else [])
        metrics = primary + [("projected", hessian)]
        for name, metric in metrics:
            metric_name = name
            matrix = metric(positions)
            shift_report = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    if ((guard_assembled_metrics and name == "projected")
                            or (not guard_assembled_metrics and name == "exact" and inertia_diagonal is not None)):
                        from solver_global_shift import shifted_positive_definite_direction
                        direction, shift_report = shifted_positive_definite_direction(matrix, gradient, inertia_diagonal)
                        if shift_report["lambda"]:
                            name = "shifted"
                    else:
                        direction = (_positive_definite_direction(matrix, gradient) if name in ("exact", "coupled")
                                     else spsolve(matrix.tocsc(), -gradient))
            except (MatrixRankWarning, np.linalg.LinAlgError):
                continue
            if direction is None:
                continue
            slope = float(np.dot(gradient, direction))
            if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
                continue
            scale = float(step_limiter(positions, positions + direction)) if step_limiter else 1.0
            if not np.isfinite(scale) or not 0 <= scale <= 1:
                raise ValueError("Invalid search step bound")
            if scale == 0:
                continue
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
                    if guard_assembled_metrics:
                        direction_history[-1]["metric"] = metric_name
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
                          coupled_steps=direction_steps["coupled"],
                          shifted_steps=direction_steps["shifted"], direction_history=direction_history)


class GlobalSewingSolver:
    def __init__(self, model, rows, compliance, *, fold_barrier_joules=None, fold_activation_angle=np.pi / 2,
                 contact=None, sewing_mode="vector", sewing_frame_faces=None, sewing_sides=None,
                 fold_hinges=None, fold_stiffness_joules=None):
        if sewing_mode not in ("vector", "distance", "normal-offset"):
            raise ValueError("Unsupported sewing mode")
        if sewing_mode != "normal-offset" and (sewing_frame_faces is not None or sewing_sides is not None):
            raise ValueError("Material frames require normal-offset sewing")
        self.sewing_mode = sewing_mode
        self.mass = model.particle_mass.numpy().astype(float)
        self.active = (self.mass > 0) & ((model.particle_flags.numpy() & 1) != 0)
        self.faces = model.tri_indices.numpy().astype(int) if model.tri_indices is not None else np.empty((0, 3), dtype=int)
        self.contact = contact
        self.poses = model.tri_poses.numpy().astype(float) if model.tri_poses is not None else np.empty((0, 2, 2))
        self.areas = model.tri_areas.numpy().astype(float) if model.tri_areas is not None else np.empty(0)
        self.materials = model.tri_materials.numpy().astype(float) if model.tri_materials is not None else np.empty((0, 3))
        if (not all(np.isfinite(value).all() for value in (self.mass, self.poses, self.areas, self.materials))
                or np.any(self.mass < 0) or np.any(self.areas <= 0) or not np.any(self.active)):
            raise ValueError("Finite physical inputs and at least one active positive-mass vertex required")
        if not np.isfinite(compliance) or compliance <= 0:
            raise ValueError("Positive finite physical compliance required")
        self.contact_rest_metric_tolerance = 64 * np.finfo(np.float32).eps
        if contact is not None:
            if (contact.rest_positions.shape != (len(self.mass), 3)
                    or not np.array_equal(contact.faces, self.faces)):
                raise ValueError("Contact vertex count and ordered faces must match the solver model exactly")
            rest_triangles = contact.rest_positions[self.faces]
            rest_edges = rest_triangles[:, 1:] - rest_triangles[:, :1]
            rest_deformation = np.einsum("fvc,fva->fca", self.poses, rest_edges)
            rest_metric = np.einsum("fca,fda->fcd", rest_deformation, rest_deformation)
            rest_areas = .5 * np.linalg.norm(np.cross(rest_edges[:, 0], rest_edges[:, 1]), axis=1)
            if (not np.allclose(rest_metric, np.eye(2), rtol=0, atol=self.contact_rest_metric_tolerance)
                    or not np.allclose(rest_areas, self.areas, rtol=self.contact_rest_metric_tolerance, atol=0)):
                raise ValueError("Contact rest material metrics must match the solver model within float32 tolerance")
        from solver_bending import ElasticDihedralBending
        self.bending = ElasticDihedralBending.from_model(model)
        self.has_bending = bool(self.bending.residual(model.particle_q.numpy()).size)
        self.fold_actuation = None
        if fold_hinges is not None or fold_stiffness_joules is not None:
            from solver_fold_actuation import FoldActuation
            self.fold_actuation = FoldActuation(model, fold_hinges, fold_stiffness_joules)
        self.fold_barrier = None
        if fold_barrier_joules is not None:
            from solver_fold_barrier import LocalAngularFoldBarrier
            indices = (np.asarray(model.edge_indices.numpy()) if model.edge_indices is not None
                       else np.empty((0, 4), dtype=int))
            if (indices.ndim != 2 or indices.shape[1] != 4 or not np.isfinite(indices).all()
                    or np.any(indices != np.floor(indices)) or np.any(indices[:, :2] < -1)
                    or np.any(indices[:, 2:] < 0) or np.any(indices >= len(self.mass))):
                raise ValueError("Invalid local fold-barrier topology")
            interior = indices[np.all(indices >= 0, axis=1)]
            self.fold_barrier = LocalAngularFoldBarrier(len(self.mass), interior,
                activation_angle=fold_activation_angle, stiffness_joules=fold_barrier_joules)
            self.fold_barrier.energy(model.particle_q.numpy())
        if np.any(self.materials[:, 2] != 0):
            raise ValueError("Global reference does not implement membrane damping")
        if np.any(self.materials[:, :2] < 0):
            raise ValueError("Nonnegative membrane materials required")
        if np.any(model.gravity.numpy() != 0):
            raise ValueError("Global reference requires zero gravity")
        if model.spring_count or model.tet_count:
            raise ValueError("Global reference does not implement springs or volumetric elements")
        from solver_embedded_sewing import validate_rows
        if not (self.fold_actuation is not None and isinstance(rows, list) and not rows):
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
        self.sewing_frame_faces = None
        self.sewing_sides = None
        if sewing_mode == "normal-offset":
            from solver_normal_sewing import NormalOffsetSewing
            potential = NormalOffsetSewing(self.sewing, np.ones(len(rows)), compliance,
                                           sewing_frame_faces, sewing_sides)
            model_faces = {tuple(face) for face in self.faces}
            if any(tuple(face) not in model_faces for face in potential.faces):
                raise ValueError("Sewing frames must match ordered source model faces")
            potential.geometry(model.particle_q.numpy())
            self.sewing_frame_faces = potential.faces
            self.sewing_sides = potential.sides

    def sewing_potential(self, targets):
        if self.sewing_mode == "normal-offset":
            from solver_normal_sewing import NormalOffsetSewing
            return NormalOffsetSewing(self.sewing, targets, self.compliance,
                                      self.sewing_frame_faces, self.sewing_sides)
        from solver_distance_sewing import DistanceSewing
        return DistanceSewing(self.sewing, targets, self.compliance)

    def step(self, previous_positions, previous_velocities, targets, dt, max_evaluations=300, linear_solver="direct",
             *, fold_targets=None):
        previous = np.asarray(previous_positions, dtype=float)
        velocities = np.asarray(previous_velocities, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if (self.fold_actuation is None) != (fold_targets is None):
            raise ValueError("Fold recipe and explicit step targets must be supplied together")
        actuator = self.fold_actuation.potential(fold_targets) if self.fold_actuation is not None else None
        if actuator is not None and linear_solver != "direct":
            raise ValueError("Fold actuation requires guarded direct search")
        distance_sewing = None
        if self.sewing_mode in ("distance", "normal-offset"):
            distance_sewing = self.sewing_potential(targets)
            if linear_solver != "direct":
                raise ValueError("Distance sewing requires safeguarded direct search")
        if linear_solver not in ("direct", "shifted", "lsmr") or type(max_evaluations) is not int or not 1 <= max_evaluations <= 10000:
            raise ValueError("Supported linear solver and bounded positive evaluation budget required")
        if (self.has_bending or self.fold_barrier is not None) and linear_solver != "direct":
            raise ValueError("Bending reference requires direct search with branch-aware line search")
        if self.contact is not None and linear_solver != "direct":
            raise ValueError("Contact requires direct search with continuous collision guards")
        if len(self.faces) and linear_solver == "lsmr":
            raise ValueError("Cloth triangles require a search with swept nondegeneracy guards")
        if (previous.shape != (len(self.mass), 3) or velocities.shape != previous.shape
                or targets.shape != ((self.sewing.shape[0],) if distance_sewing is not None
                                     else (self.sewing.shape[0], 3))
                or not all(np.isfinite(value).all() for value in (previous, velocities, targets))
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError("Finite correctly shaped state and positive timestep required")
        predicted = previous + dt * velocities
        if self.contact is not None:
            self.contact.validate_state(previous)
        self.bending.energy(previous)
        if actuator is not None:
            actuator.energy(previous)
        predicted[~self.active] = previous[~self.active]
        inertia_weights = np.repeat(np.sqrt(self.mass) / dt, 3)
        linear_jacobian = diags(inertia_weights, format="csr")
        sewn_jacobian = self.sewing_xyz / np.sqrt(self.compliance)
        if distance_sewing is not None:
            distance_sewing.geometry(previous)

        def sewing_residual(positions):
            return (distance_sewing.residual(positions) if distance_sewing is not None else
                    (self.sewing @ positions - targets).ravel() / np.sqrt(self.compliance))
        fixed = previous.ravel().copy()
        coefficients = np.concatenate((-self.poses.sum(axis=1)[:, None, :], self.poses), axis=1)
        previous_deformation = np.einsum("fvc,fva->fca", coefficients, previous[self.faces])
        if np.any(np.linalg.norm(np.cross(previous_deformation[:, 0], previous_deformation[:, 1]), axis=1) <= 1e-10):
            raise ValueError("Degenerate membrane state cannot initialize the global reference")
        sqrt_mu = np.sqrt(self.areas * self.materials[:, 0])
        lame = self.materials[:, 0] + self.materials[:, 1]
        sqrt_lame = np.sqrt(self.areas * lame)
        alpha = 1 + self.materials[:, 0] / np.maximum(lame, 1e-6)
        residual_count = len(fixed) + targets.size + 7 * len(self.faces) + len(self.bending.indices)
        if self.sewing_mode == "normal-offset":
            residual_count += 2 * targets.size
        if self.fold_barrier is not None:
            residual_count += len(self.fold_barrier.indices)
        if actuator is not None:
            residual_count += len(actuator.indices)

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
                return np.full(residual_count, np.inf)
            area_ratios = np.maximum(raw_area_ratios, 1e-10)
            membrane = np.concatenate((sqrt_mu[:, None] * deformation.reshape((-1, 6)),
                                       (sqrt_lame * (area_ratios - alpha))[:, None]), axis=1)
            if not jacobian:
                try:
                    bending = self.bending.residual(positions)
                    barrier = self.fold_barrier.residual(positions) if self.fold_barrier is not None else np.empty(0)
                    sewing = sewing_residual(positions)
                    fold = actuator.residual(positions) if actuator is not None else np.empty(0)
                except ValueError:
                    return np.full(residual_count, np.inf)
                return np.concatenate((inertia_weights * (flat - predicted.ravel()),
                                       sewing,
                                       membrane.ravel(), bending, barrier, fold))
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
            sewing_jacobian = (distance_sewing.jacobian(positions) if distance_sewing is not None
                               else sewn_jacobian)
            blocks = [linear_jacobian, sewing_jacobian, membrane_jacobian, self.bending.jacobian(positions)]
            if actuator is not None:
                blocks.append(actuator.jacobian(positions))
            if self.fold_barrier is not None:
                blocks.append(self.fold_barrier.jacobian(positions))
            return vstack(blocks, format="csr")[:, self.free]

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
            try:
                sewing = sewing_residual(positions)
                bending_energy = self.bending.energy(positions)
                barrier_energy = self.fold_barrier.energy(positions) if self.fold_barrier is not None else 0.
                contact_energy = self.contact.energy(positions) if self.contact is not None else 0.
                fold_energy = actuator.energy(positions) if actuator is not None else 0.
            except ValueError:
                return float("inf")
            return float((inertial @ inertial + sewing @ sewing) / 2 + membrane_energy.sum()
                         + bending_energy + barrier_energy + contact_energy + fold_energy)

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
            sewing_gradient = (distance_sewing.gradient(positions) if distance_sewing is not None else
                self.sewing_xyz.T @ ((self.sewing @ positions - targets).ravel() / self.compliance))
            gradient = inertia_weights ** 2 * (flat - predicted.ravel()) + sewing_gradient
            np.add.at(gradient.reshape((-1, 3)), self.faces, element_gradient)
            gradient += self.bending.gradient(positions).ravel()
            if actuator is not None:
                gradient += actuator.gradient(positions).ravel()
            if self.fold_barrier is not None:
                gradient += self.fold_barrier.gradient(positions).ravel()
            if self.contact is not None:
                gradient += self.contact.gradient(positions).ravel()
            return gradient[self.free]

        def energy_change_function(start_positions, end_positions):
            from solver_energy_change import membrane_energy_change
            from solver_triangle_sweep import triangle_sweep_safe
            flat = fixed.copy()
            flat[self.free] = start_positions
            delta = np.zeros_like(flat)
            delta[self.free] = end_positions - start_positions
            positions, displacement = flat.reshape((-1, 3)), delta.reshape((-1, 3))
            if (not triangle_sweep_safe(positions, positions + displacement, self.faces)
                    or not triangle_sweep_safe(previous, positions + displacement, self.faces)):
                return float("inf")
            deformation = np.einsum("fvc,fva->fca", coefficients, positions[self.faces])
            delta_deformation = np.einsum("fvc,fva->fca", coefficients, displacement[self.faces])
            try:
                sewing_change = (distance_sewing.energy_change(positions, positions + displacement)
                                 if distance_sewing is not None else None)
                membrane_change = membrane_energy_change(deformation, delta_deformation, self.areas, self.materials[:, :3])
                bending_change = self.bending.energy_change(positions, positions + displacement)
                barrier_change = 0.
                contact_change = 0.
                fold_change = 0.
                if actuator is not None:
                    from solver_hinge_sweep import hinge_sweep_safe
                    if (not hinge_sweep_safe(positions, positions + displacement, actuator.indices)
                            or not hinge_sweep_safe(previous, positions + displacement, actuator.indices)):
                        return float("inf")
                    fold_change = actuator.energy_change(positions, positions + displacement)
                if self.contact is not None:
                    if (not self.contact.path_safe(positions, positions + displacement)
                            or not self.contact.path_safe(previous, positions + displacement)):
                        return float("inf")
                    contact_change = self.contact.energy_change(positions, positions + displacement)
                if self.fold_barrier is not None:
                    from solver_hinge_sweep import hinge_sweep_safe
                    if not hinge_sweep_safe(positions, positions + displacement, self.fold_barrier.indices):
                        return float("inf")
                    if not hinge_sweep_safe(previous, positions + displacement, self.fold_barrier.indices):
                        return float("inf")
                    barrier_change = self.fold_barrier.energy_change(positions, positions + displacement)
            except ValueError:
                return float("inf")
            inertial = inertia_weights * (flat - predicted.ravel())
            delta_inertial = inertia_weights * delta
            if sewing_change is None:
                sewing = sewing_residual(positions)
                delta_sewing = (self.sewing @ displacement).ravel() / np.sqrt(self.compliance)
                sewing_change = (sewing + .5 * delta_sewing) @ delta_sewing
            return float((inertial + .5 * delta_inertial) @ delta_inertial
                         + sewing_change + membrane_change
                         + bending_change + barrier_change + contact_change + fold_change)

        def step_limiter(start_positions, end_positions):
            start_flat, end_flat = fixed.copy(), fixed.copy()
            start_flat[self.free], end_flat[self.free] = start_positions, end_positions
            return self.contact.step_limit(start_flat.reshape((-1, 3)), end_flat.reshape((-1, 3)))

        linear_hessian = linear_jacobian.T @ linear_jacobian
        if distance_sewing is None:
            linear_hessian += sewn_jacobian.T @ sewn_jacobian
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
            flat = fixed.copy()
            flat[self.free] = free_positions
            bending_hessian = self.bending.hessian(flat.reshape((-1, 3)))
            if actuator is not None:
                bending_hessian += actuator.hessian(flat.reshape((-1, 3)))
            if distance_sewing is not None:
                bending_hessian += (distance_sewing.exact_hessian(flat.reshape((-1, 3)))
                    if self.sewing_mode == "normal-offset" and not project_psd else
                    distance_sewing.hessian(flat.reshape((-1, 3)), project_psd))
            if self.fold_barrier is not None:
                bending_hessian += self.fold_barrier.hessian(flat.reshape((-1, 3)))
            if self.contact is not None:
                bending_hessian += self.contact.hessian(flat.reshape((-1, 3)))
            return (linear_hessian + membrane_hessian + bending_hessian)[self.free][:, self.free]

        diagonal = diags(self.mass / dt ** 2)
        matrix = diagonal + self.sewing.T @ self.sewing / self.compliance
        linear_targets = self.sewing @ previous if distance_sewing is not None else targets
        rhs = diagonal @ predicted + self.sewing.T @ linear_targets / self.compliance
        active_indices = np.flatnonzero(self.active)
        fixed_indices = np.flatnonzero(~self.active)
        rhs = rhs[active_indices] - matrix[active_indices][:, fixed_indices] @ previous[fixed_indices]
        linear_start = spsolve(matrix[active_indices][:, active_indices].tocsc(), rhs).reshape((-1, 3)).ravel()
        predicted_start = predicted.ravel()[self.free]
        guarded = (self.has_bending or self.fold_barrier is not None or self.contact is not None
                   or distance_sewing is not None or actuator is not None)
        start = (previous.ravel()[self.free].copy() if guarded or len(self.faces) else
                 min((linear_start, predicted_start, previous.ravel()[self.free]), key=objective))
        initial_energy = objective(start)
        guard_assembled_metrics = bool(getattr(self.contact, "requires_guarded_metric", False))
        result = _direct_descent(evaluate, start, max_evaluations,
                                lambda positions: assembled_hessian(positions, True), objective,
                                exact_hessian=None if guarded else lambda positions: assembled_hessian(positions, False),
                                inertia_diagonal=inertia_weights[self.free] ** 2 if linear_solver == "shifted" or guard_assembled_metrics else None,
                                guard_assembled_metrics=guard_assembled_metrics,
                                gradient_function=gradient_function,
                                coupled_hessian=(lambda positions: assembled_hessian(positions, False)) if guarded else None,
                                energy_change_function=energy_change_function,
                                step_limiter=step_limiter if self.contact is not None else None) if linear_solver in ("direct", "shifted") else least_squares(evaluate, start, jac=lambda positions: evaluate(positions, True),
                               method="trf", tr_solver="lsmr", x_scale="jac", ftol=1e-12, xtol=1e-12,
                               gtol=1e-9, max_nfev=max_evaluations,
                               tr_options={"atol": 1e-12, "btol": 1e-12, "maxiter": max(100, 3 * len(self.free))})
        final = fixed.copy()
        final[self.free] = result.x
        final = final.reshape((-1, 3))
        from solver_triangle_sweep import triangle_sweep_safe
        if not triangle_sweep_safe(previous, final, self.faces):
            raise ValueError("Physical cloth step crosses a degenerate or unresolved triangle path")
        if actuator is not None:
            from solver_hinge_sweep import hinge_sweep_safe
            if not hinge_sweep_safe(previous, final, actuator.indices):
                raise ValueError("Physical fold step crosses an invalid hinge path")
        if self.contact is not None:
            self.contact.validate_state(final)
            if not self.contact.path_safe(previous, final):
                raise ValueError("Physical contact step crosses a collision")
        final_deformation = np.einsum("fvc,fva->fca", coefficients, final[self.faces])
        if np.any(np.linalg.norm(np.cross(final_deformation[:, 0], final_deformation[:, 1]), axis=1) <= 1e-10):
            raise ValueError("Degenerate membrane state rejected by global reference")
        gradient = gradient_function(result.x)
        gradient_norm = float(np.max(np.abs(gradient)))
        return final, (final - previous) / dt, {
            "profile": "experimental-global-ipc-guarded-contact-reference-v1" if guard_assembled_metrics else "experimental-global-ipc-contact-reference-v1" if self.contact is not None else "experimental-global-local-fold-barrier-v1" if self.fold_barrier is not None else "experimental-global-elastic-bending-reference-v2" if self.has_bending else "experimental-global-membrane-sewing-reference-v6", "accepted": False,
            "contact": self.contact.profile() if self.contact is not None else None,
            "sewingMode": self.sewing_mode,
            "triangleSweep": "all source faces; numerical Bernstein guard on optimizer and physical affine paths; v1",
            "foldActuation": actuator is not None,
            "foldActuationJoules": actuator.energy(final) if actuator is not None else 0.,
            "foldTargetsRadians": actuator.rest_angles.tolist() if actuator is not None else None,
            "foldAnglesRadians": actuator.angles(final).tolist() if actuator is not None else None,
            "foldActuationLimitations": "External angle penalty, not a change of cloth rest shape or a turning/binding recipe; Gauss-Newton search" if actuator is not None else None,
            "sewingJoules": float(.5 * np.sum(sewing_residual(final) ** 2)),
            "sewingTargetErrorM": float(np.max(np.abs(sewing_residual(final)), initial=0)
                                         * np.sqrt(self.compliance)),
            "sewingLimitations": ("Source-normal offset with full frame reactions and exact sewing curvature; Gauss-Newton fallback; swept triangles guarded, no turning or seam tangent alignment"
                                  if self.sewing_mode == "normal-offset" else
                                  "Scalar anchor distance does not prescribe layer side, seam tangent or turning"
                                  if distance_sewing is not None else "World-space vector registration"),
            "contactJoules": self.contact.energy(final) if self.contact is not None else 0.,
            "contactSearchMetric": "Signed-weight contact Hessian; both assembled metrics safeguarded: SPD primary or physical-inertia-shifted projected fallback; not exact total Hessian" if guard_assembled_metrics else "PSD-projected contact Hessian; not exact total Hessian" if self.contact is not None else None,
            "contactRestMetricTolerance": float(self.contact_rest_metric_tolerance) if self.contact is not None else None,
            "localFoldBarrier": self.fold_barrier is not None,
            "foldBarrierJoules": self.fold_barrier.energy(final) if self.fold_barrier is not None else 0.,
            "bendingHinges": len(self.bending.indices),
            "bendingSearchMetric": "Exact membrane + Gauss-Newton bending; projected membrane fallback, not exact total Hessian" if self.has_bending else None,
            "converged": bool(result.success and gradient_norm <= 1e-6),
            "optimizerSuccess": bool(result.success), "stationarityToleranceN": 1e-6,
            "linearSolver": linear_solver, "energyHistory": getattr(result, "energy_history", None),
            "exactSteps": getattr(result, "exact_steps", None), "projectedSteps": getattr(result, "projected_steps", None),
            "coupledSteps": getattr(result, "coupled_steps", None),
            "shiftedSteps": getattr(result, "shifted_steps", None),
            "directionHistory": getattr(result, "direction_history", None),
            "status": int(result.status), "message": result.message,
            "evaluations": int(result.nfev), "initialEnergy": initial_energy, "finalEnergy": objective(result.x),
            "gradientInfinityNorm": gradient_norm,
            "limitations": ["Experimental frictionless surface contact; no body contact, seam exclusions, external forces or material damping. Diagnostic reference only." if self.contact is not None else "No contact, external forces or material damping; diagnostic reference only.",
                            "Optional local angular fold barrier changes the energy model; it is not finite-thickness or nonadjacent cloth contact.",
                            "Elastic bending is uncalibrated; the numerical triangle guard requires independent saved-path verification.",
                            "Stationarity does not certify a local energy minimum or dynamic stability."],
        }
