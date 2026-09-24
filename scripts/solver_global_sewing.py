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
                    coupled_hessian=None, step_limiter=None, guard_assembled_metrics=False,
                    gradient_error_function=None, energy_change_interval_function=None):
    if exact_hessian is not None and coupled_hessian is not None:
        raise ValueError("Choose one safeguarded primary search metric")
    if guard_assembled_metrics and inertia_diagonal is None:
        raise ValueError("Guarded assembled search metrics require physical inertia")
    bounded = gradient_error_function is not None
    if bounded != (energy_change_interval_function is not None) or bounded and gradient_function is None:
        raise ValueError("Bounded search requires gradient and fixed-work enclosures together")
    if bounded:
        from fractions import Fraction as F
        from solver_cable_integration import directional_interval, _rat

    def gradient_error(positions):
        if not bounded:
            return 0.
        error = gradient_error_function(positions)
        if type(error) is not F or error < 0:
            raise ValueError("Nonnegative exact conditional gradient bound required")
        return error

    def stationary(gradient, error):
        norm = float(np.max(np.abs(gradient)))
        return (np.isfinite(gradient).all() and
                (F(norm)+error <= F(1e-6) if bounded else norm <= 1e-6))
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
        error = gradient_error(positions)
        if stationary(gradient, error):
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
            if not np.isfinite(direction).all():
                continue
            if bounded:
                if directional_interval(gradient, error, np.zeros_like(direction), direction)[1] >= 0:
                    continue
            else:
                slope = float(np.dot(gradient, direction))
                if not np.isfinite(slope) or slope >= 0:
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
                decision = None
                if bounded:
                    interval = energy_change_interval_function(positions, candidate) if finite_candidate else None
                    slope_interval = (directional_interval(gradient, error, positions, candidate)
                                      if finite_candidate else (F(), F()))
                    if interval is not None:
                        if (type(interval) is not tuple or len(interval) != 2
                                or any(type(value) is not F for value in interval) or interval[0] > interval[1]):
                            raise ValueError("Ordered exact conditional work interval required")
                        decision = {'conditionalSlopeLowerJoules': _rat(slope_interval[0]),
                                    'conditionalSlopeUpperJoules': _rat(slope_interval[1]),
                                    'conditionalChangeLowerJoules': _rat(interval[0]),
                                    'conditionalChangeUpperJoules': _rat(interval[1])}
                    descent = (interval is not None and slope_interval[1] < 0
                               and interval[1] <= F(1e-4)*slope_interval[0])
                else:
                    change = (energy_change_function(positions, candidate) if energy_change_function
                              and finite_candidate else candidate_energy - energy)
                    descent = np.isfinite(change) and change <= 1e-4 * scale * slope
                if finite_candidate and not np.array_equal(candidate, positions) and descent:
                    positions, residual, energy = candidate, candidate_residual, candidate_energy
                    history.append(energy)
                    direction_steps[name] += 1
                    direction_history.append({"method": name, "scale": scale, "shift": shift_report})
                    if guard_assembled_metrics:
                        direction_history[-1]["metric"] = metric_name
                    if bounded:
                        direction_history[-1]['cableAwareDecision'] = decision
                    accepted = True
                    break
                scale *= .5
            if accepted or evaluations >= max_evaluations:
                break
        if not accepted:
            status, message = -2, "Line search failed or evaluation budget exhausted"
            break
    gradient = gradient_function(positions) if gradient_function else evaluate(positions, True).T @ residual
    if stationary(gradient, gradient_error(positions)):
        status, message = 1, "Stationarity tolerance satisfied"
    return OptimizeResult(x=positions, cost=energy, nfev=evaluations, success=status == 1,
                          status=status, message=message, energy_history=history,
                          exact_steps=direction_steps["exact"], projected_steps=direction_steps["projected"],
                          coupled_steps=direction_steps["coupled"],
                          shifted_steps=direction_steps["shifted"], direction_history=direction_history)


class GlobalSewingSolver:
    def __init__(self, model, rows, compliance, *, fold_barrier_joules=None, fold_activation_angle=np.pi / 2,
                 contact=None, sewing_mode="vector", sewing_frame_faces=None, sewing_sides=None,
                 fold_hinges=None, fold_stiffness_joules=None, material_grippers=None,
                 controlled_fold_actuation=None, continuous_cable=None, cable_precision=None):
        if sewing_mode not in ("vector", "distance", "normal-offset"):
            raise ValueError("Unsupported sewing mode")
        if sewing_mode != "normal-offset" and (sewing_frame_faces is not None or sewing_sides is not None):
            raise ValueError("Material frames require normal-offset sewing")
        self.sewing_mode = sewing_mode
        self.mass = model.particle_mass.numpy().astype(float)
        self.active = (self.mass > 0) & ((model.particle_flags.numpy() & 1) != 0)
        self.continuous_cable = None
        if (continuous_cable is None) != (cable_precision is None):
            raise ValueError("Continuous cable recipe and explicit precision must be supplied together")
        if continuous_cable is not None:
            from solver_cable_integration import CableControl
            control = CableControl(continuous_cable, cable_precision)
            if control.vertex_count != len(self.mass) or not np.all(self.active):
                raise ValueError("Continuous cable integration requires matching free positive-mass cloth")
            self.continuous_cable = control
        self.faces = model.tri_indices.numpy().astype(int) if model.tri_indices is not None else np.empty((0, 3), dtype=int)
        self.material_grippers = material_grippers
        if material_grippers is not None:
            from solver_material_grippers import MaterialGrippers
            if (not isinstance(material_grippers, MaterialGrippers)
                    or material_grippers.vertex_count != len(self.mass)
                    or not np.array_equal(material_grippers.faces, self.faces)):
                raise ValueError("Material grippers must match the solver's canonical vertex count and ordered faces")
            if not np.all(self.active):
                raise ValueError("Material-gripper reference currently requires free positive-mass cloth; fixed reactions are not implemented")
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
        self.controlled_fold_actuation = None
        if controlled_fold_actuation is not None:
            from solver_controlled_fold import ControlledFoldActuation
            if (fold_hinges is not None or fold_stiffness_joules is not None
                    or not isinstance(controlled_fold_actuation, ControlledFoldActuation)
                    or controlled_fold_actuation.vertex_count != len(self.mass)):
                raise ValueError("Choose a controlled fold recipe matching this model, without legacy fold parameters")
            # Rebind the immutable declaration to this actual model's ordered
            # native hinges. Generic model membership is not source-profile
            # or captured-input admission; those remain caller obligations.
            self.controlled_fold_actuation = ControlledFoldActuation(model,
                controlled_fold_actuation.hinges, controlled_fold_actuation.stiffness)
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
        if not ((self.fold_actuation is not None or self.controlled_fold_actuation is not None
                 or self.material_grippers is not None or self.continuous_cable is not None)
                and isinstance(rows, list) and not rows):
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

    def sewing_potential(self, targets, *, activation=None):
        if self.sewing_mode == "normal-offset":
            from solver_normal_sewing import NormalOffsetSewing
            return NormalOffsetSewing(self.sewing, targets, self.compliance,
                                      self.sewing_frame_faces, self.sewing_sides, activation=activation)
        from solver_distance_sewing import DistanceSewing
        return DistanceSewing(self.sewing, targets, self.compliance, activation=activation)

    def step(self, previous_positions, previous_velocities, targets, dt, max_evaluations=300, linear_solver="direct",
             *, fold_targets=None, fold_activation=None, gripper_targets=None, gripper_activation=None, sewing_activation=None):
        cable = self.continuous_cable
        if cable is not None:
            from fractions import Fraction as F
            from solver_cable_integration import CableControl, assemble_gradient, round_sum, stationarity, _encoded, _rational
            from solver_controlled_fold import _binary64
            if type(cable) is not CableControl or linear_solver != 'direct':
                raise ValueError('Validated continuous cable controls require guarded direct search')
            previous_positions, previous_velocities = cable.positions(previous_positions), cable.positions(previous_velocities)
            dt = _binary64(dt)
            cable_definition = _encoded(cable.description())
            cable_cached_positions, cable_cached_response = None, None
            cable_gradient_positions, cable_gradient_error, cable_assembly_error = None, F(), F()
            cable_failures = {'count': 0, 'lastReason': None}

            def cable_identity():
                if self.continuous_cable is not cable or _encoded(cable.description()) != cable_definition:
                    raise ValueError('Continuous cable identity or precision changed during the step')

            def cable_response(positions):
                nonlocal cable_cached_positions, cable_cached_response
                cable_identity()
                if cable_cached_positions is None or positions.tobytes() != cable_cached_positions.tobytes():
                    snapshot = positions.copy()
                    response = cable.validate_response(positions, cable.evaluate(positions))
                    if positions.tobytes() != snapshot.tobytes():
                        raise ValueError('Cable helper mutated its supplied state')
                    cable_identity()
                    cable_cached_positions, cable_cached_response = snapshot, response
                return cable_cached_response
        controlled_fold = self.controlled_fold_actuation
        if controlled_fold is not None:
            if self.fold_actuation is not None or fold_targets is None or fold_activation is None:
                raise ValueError("Controlled fold targets and activation must be supplied together without legacy actuation")
            previous = controlled_fold._positions(previous_positions)
            velocities = controlled_fold._positions(previous_velocities)
            from solver_controlled_fold import _binary64
            dt = _binary64(dt)
            actuator = controlled_fold.potential(fold_targets, fold_activation)
            fold_path_hinges = controlled_fold.hinges
        else:
            if fold_activation is not None:
                raise ValueError("Explicit fold activation requires a controlled fold recipe")
            if (self.fold_actuation is None) != (fold_targets is None):
                raise ValueError("Fold recipe and explicit step targets must be supplied together")
            previous = np.asarray(previous_positions, dtype=float)
            velocities = np.asarray(previous_velocities, dtype=float)
            actuator = self.fold_actuation.potential(fold_targets) if self.fold_actuation is not None else None
            fold_path_hinges = actuator.indices if actuator is not None else None
        targets = np.asarray(targets, dtype=float)
        from solver_sewing_activation import validate_sewing_activation
        sewing_weights = validate_sewing_activation(sewing_activation, self.sewing.shape[0])
        sewing_positive = sewing_weights > 0
        active_sewing = self.sewing if np.all(sewing_positive) else self.sewing[sewing_positive]
        active_weights = sewing_weights[sewing_positive]
        sqrt_sewing_weights = np.sqrt(sewing_weights)
        if sewing_activation is not None and linear_solver != "direct":
            raise ValueError("Explicit sewing activation requires guarded direct search")
        if self.material_grippers is None:
            if gripper_targets is not None or gripper_activation is not None:
                raise ValueError("Explicit gripper parameters require a material-gripper recipe")
            grippers = None
        else:
            if gripper_targets is None or gripper_activation is None:
                raise ValueError("Material-gripper recipe requires explicit targets and activation")
            grippers = self.material_grippers.potential(gripper_targets, gripper_activation)
            if linear_solver != "direct":
                raise ValueError("Material grippers require guarded direct search")
        if actuator is not None and linear_solver != "direct":
            raise ValueError("Fold actuation requires guarded direct search")
        distance_sewing = None
        if self.sewing_mode in ("distance", "normal-offset"):
            distance_sewing = self.sewing_potential(targets, activation=sewing_activation)
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
        if controlled_fold is not None:
            from solver_hinge_sweep import hinge_sweep_safe
            if not hinge_sweep_safe(previous, previous, fold_path_hinges):
                raise ValueError("Initial controlled fold geometry fails the complete declared hinge guard")
        if grippers is not None:
            grippers.energy(previous)
        predicted[~self.active] = previous[~self.active]
        inertia_weights = np.repeat(np.sqrt(self.mass) / dt, 3)
        linear_jacobian = diags(inertia_weights, format="csr")
        sewn_jacobian = self.sewing_xyz / np.sqrt(self.compliance)
        if sewing_activation is not None:
            sewn_jacobian = diags(np.repeat(sqrt_sewing_weights, 3),
                                  shape=(3 * len(sewing_weights),) * 2) @ sewn_jacobian
        if distance_sewing is not None:
            distance_sewing.geometry(previous)

        def vector_sewing_error(positions):
            # Pending rows must not perform an overflowing subtraction only
            # to multiply it by zero afterward. Their source rows stay intact.
            result = np.zeros((len(sewing_weights), 3))
            result[sewing_positive] = active_sewing @ positions - targets[sewing_positive]
            return result

        def sewing_residual(positions):
            return (distance_sewing.residual(positions) if distance_sewing is not None else
                    (sqrt_sewing_weights[:, None] * vector_sewing_error(positions)).ravel() / np.sqrt(self.compliance))
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
            residual_count += (len(actuator.active_hinge_indices) if controlled_fold is not None else len(actuator.indices))
        if grippers is not None:
            residual_count += 3 * len(self.material_grippers.gripper_ids)

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
                    grip = grippers.residual(positions) if grippers is not None else np.empty(0)
                except ValueError:
                    return np.full(residual_count, np.inf)
                return np.concatenate((inertia_weights * (flat - predicted.ravel()),
                                       sewing,
                                       membrane.ravel(), bending, barrier, fold, grip))
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
            if self.fold_barrier is not None:
                blocks.append(self.fold_barrier.jacobian(positions))
            if actuator is not None:
                blocks.append(actuator.jacobian(positions))
            if grippers is not None:
                blocks.append(grippers.jacobian(positions))
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
                gripper_energy = grippers.energy(positions) if grippers is not None else 0.
            except ValueError:
                return float("inf")
            baseline = float((inertial @ inertial + sewing @ sewing) / 2 + membrane_energy.sum()
                             + bending_energy + barrier_energy + contact_energy + fold_energy + gripper_energy)
            if cable is not None:
                try:
                    return round_sum((baseline, cable_response(positions)['energy']))[0]
                except ValueError as failure:
                    cable_failures['count'] += 1
                    cable_failures['lastReason'] = str(failure)
                    return float('inf')
            return baseline

        def gradient_function(free_positions):
            nonlocal cable_gradient_positions, cable_gradient_error, cable_assembly_error
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
                (active_sewing.T @ (active_weights[:, None] *
                    (active_sewing @ positions - targets[sewing_positive]) / self.compliance)).ravel())
            gradient = inertia_weights ** 2 * (flat - predicted.ravel()) + sewing_gradient
            np.add.at(gradient.reshape((-1, 3)), self.faces, element_gradient)
            gradient += self.bending.gradient(positions).ravel()
            if actuator is not None:
                gradient += actuator.gradient(positions).ravel()
            if grippers is not None:
                gradient += grippers.gradient(positions).ravel()
            if self.fold_barrier is not None:
                gradient += self.fold_barrier.gradient(positions).ravel()
            if self.contact is not None:
                gradient += self.contact.gradient(positions).ravel()
            if cable is not None:
                response = cable_response(positions)
                gradient, cable_gradient_error, cable_assembly_error = assemble_gradient(
                    gradient, response['gradient'],
                    _rational(response['certificate']['gradientMaxAbsoluteErrorBoundNewtons']))
                cable_gradient_positions = free_positions.copy()
            return gradient[self.free]

        def gradient_error_function(free_positions):
            if cable_gradient_positions is None or cable_gradient_positions.tobytes() != free_positions.tobytes():
                gradient_function(free_positions)
            return cable_gradient_error

        def energy_change_function(start_positions, end_positions):
            from solver_energy_change import membrane_energy_change
            from solver_triangle_sweep import triangle_sweep_safe
            flat = fixed.copy()
            flat[self.free] = start_positions
            delta = np.zeros_like(flat)
            delta[self.free] = end_positions - start_positions
            positions, displacement = flat.reshape((-1, 3)), delta.reshape((-1, 3))
            if cable is not None:
                end_flat = fixed.copy()
                end_flat[self.free] = end_positions
                next_positions = end_flat.reshape((-1, 3))
            else:
                next_positions = positions + displacement
            if (not triangle_sweep_safe(positions, next_positions, self.faces)
                    or not triangle_sweep_safe(previous, next_positions, self.faces)):
                return float("inf")
            deformation = np.einsum("fvc,fva->fca", coefficients, positions[self.faces])
            delta_deformation = np.einsum("fvc,fva->fca", coefficients, displacement[self.faces])
            try:
                sewing_change = (distance_sewing.energy_change(positions, next_positions)
                                 if distance_sewing is not None else None)
                membrane_change = membrane_energy_change(deformation, delta_deformation, self.areas, self.materials[:, :3])
                bending_change = self.bending.energy_change(positions, next_positions)
                barrier_change = 0.
                contact_change = 0.
                fold_change = 0.
                gripper_change = grippers.energy_change(positions, next_positions) if grippers is not None else 0.
                if actuator is not None:
                    from solver_hinge_sweep import hinge_sweep_safe
                    if (not hinge_sweep_safe(positions, next_positions, fold_path_hinges)
                            or not hinge_sweep_safe(previous, next_positions, fold_path_hinges)):
                        return float("inf")
                    fold_change = actuator.energy_change(positions, next_positions)
                if self.contact is not None:
                    if (not self.contact.path_safe(positions, next_positions)
                            or not self.contact.path_safe(previous, next_positions)):
                        return float("inf")
                    contact_change = self.contact.energy_change(positions, next_positions)
                if self.fold_barrier is not None:
                    from solver_hinge_sweep import hinge_sweep_safe
                    if not hinge_sweep_safe(positions, next_positions, self.fold_barrier.indices):
                        return float("inf")
                    if not hinge_sweep_safe(previous, next_positions, self.fold_barrier.indices):
                        return float("inf")
                    barrier_change = self.fold_barrier.energy_change(positions, next_positions)
            except ValueError:
                return float("inf")
            inertial = inertia_weights * (flat - predicted.ravel())
            delta_inertial = inertia_weights * delta
            if sewing_change is None:
                sewing = sewing_residual(positions)
                delta_sewing = np.zeros((len(sewing_weights), 3))
                delta_sewing[sewing_positive] = (np.sqrt(active_weights)[:, None] *
                                                (active_sewing @ displacement) / np.sqrt(self.compliance))
                delta_sewing = delta_sewing.ravel()
                sewing_change = (sewing + .5 * delta_sewing) @ delta_sewing
            return float((inertial + .5 * delta_inertial) @ delta_inertial
                         + sewing_change + membrane_change
                         + bending_change + barrier_change + contact_change + fold_change + gripper_change)

        def energy_change_interval_function(start_positions, end_positions):
            baseline = energy_change_function(start_positions, end_positions)
            if not np.isfinite(baseline):
                return None
            first, last = fixed.copy(), fixed.copy()
            first[self.free], last[self.free] = start_positions, end_positions
            first, last = first.reshape((-1, 3)), last.reshape((-1, 3))
            snapshots = first.tobytes(), last.tobytes()
            try:
                cable_identity()
                work = cable.validate_change(first, last, cable.energy_change(first, last))
                if (first.tobytes(), last.tobytes()) != snapshots:
                    raise ValueError('Cable work helper mutated its supplied states')
                cable_identity()
                value, error = round_sum((baseline, work['changeJoules']),
                                        _rational(work['certificate']['changeErrorBoundJoules']))
                return F(value)-error, F(value)+error
            except ValueError as failure:
                cable_failures['count'] += 1
                cable_failures['lastReason'] = str(failure)
                return None

        def step_limiter(start_positions, end_positions):
            start_flat, end_flat = fixed.copy(), fixed.copy()
            start_flat[self.free], end_flat[self.free] = start_positions, end_positions
            return self.contact.step_limit(start_flat.reshape((-1, 3)), end_flat.reshape((-1, 3)))

        linear_hessian = linear_jacobian.T @ linear_jacobian
        if distance_sewing is None:
            linear_hessian += sewn_jacobian.T @ sewn_jacobian
        if grippers is not None:
            linear_hessian += grippers.hessian()
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
            if cable is not None:
                bending_hessian += cable_response(flat.reshape((-1, 3)))['hessian']
            return (linear_hessian + membrane_hessian + bending_hessian)[self.free][:, self.free]

        diagonal = diags(self.mass / dt ** 2)
        weighted_sewing = diags(active_weights, shape=(len(active_weights),) * 2) @ active_sewing
        matrix = diagonal + active_sewing.T @ weighted_sewing / self.compliance
        linear_targets = active_sewing @ previous if distance_sewing is not None else targets[sewing_positive]
        rhs = diagonal @ predicted + active_sewing.T @ (active_weights[:, None] * linear_targets) / self.compliance
        active_indices = np.flatnonzero(self.active)
        fixed_indices = np.flatnonzero(~self.active)
        rhs = rhs[active_indices] - matrix[active_indices][:, fixed_indices] @ previous[fixed_indices]
        linear_start = spsolve(matrix[active_indices][:, active_indices].tocsc(), rhs).reshape((-1, 3)).ravel()
        predicted_start = predicted.ravel()[self.free]
        guarded = (self.has_bending or self.fold_barrier is not None or self.contact is not None
                   or distance_sewing is not None or actuator is not None or grippers is not None or cable is not None)
        start = (previous.ravel()[self.free].copy() if guarded or len(self.faces) else
                 min((linear_start, predicted_start, previous.ravel()[self.free]), key=objective))
        initial_energy = objective(start)
        contact_guard_assembled_metrics = bool(getattr(self.contact, "requires_guarded_metric", False))
        guard_assembled_metrics = contact_guard_assembled_metrics or cable is not None
        result = _direct_descent(evaluate, start, max_evaluations,
                                lambda positions: assembled_hessian(positions, True), objective,
                                exact_hessian=None if guarded else lambda positions: assembled_hessian(positions, False),
                                inertia_diagonal=inertia_weights[self.free] ** 2 if linear_solver == "shifted" or guard_assembled_metrics else None,
                                guard_assembled_metrics=guard_assembled_metrics,
                                gradient_function=gradient_function,
                                coupled_hessian=(lambda positions: assembled_hessian(positions, False)) if guarded else None,
                                energy_change_function=energy_change_function,
                                step_limiter=step_limiter if self.contact is not None else None,
                                **({'gradient_error_function': gradient_error_function,
                                    'energy_change_interval_function': energy_change_interval_function}
                                   if cable is not None else {})) if linear_solver in ("direct", "shifted") else least_squares(evaluate, start, jac=lambda positions: evaluate(positions, True),
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
            if not hinge_sweep_safe(previous, final, fold_path_hinges):
                raise ValueError("Physical fold step crosses an invalid hinge path")
        if cable is not None and self.fold_barrier is not None:
            from solver_hinge_sweep import hinge_sweep_safe
            if not hinge_sweep_safe(previous, final, self.fold_barrier.indices):
                raise ValueError('Physical cable step crosses a declared fold-barrier hinge path')
        if self.contact is not None:
            self.contact.validate_state(final)
            if not self.contact.path_safe(previous, final):
                raise ValueError("Physical contact step crosses a collision")
        final_deformation = np.einsum("fvc,fva->fca", coefficients, final[self.faces])
        if np.any(np.linalg.norm(np.cross(final_deformation[:, 0], final_deformation[:, 1]), axis=1) <= 1e-10):
            raise ValueError("Degenerate membrane state rejected by global reference")
        gradient = gradient_function(result.x)
        gradient_norm = float(np.max(np.abs(gradient)))
        cable_report = cable_stationarity = None
        cable_converged = True
        if cable is not None:
            response = cable_response(final)
            cable_report = cable.validate_diagnostics(final, {
                'definition': cable.description(), 'energyJoules': response['energy'],
                'certificate': response['certificate']})
            cable_stationarity = stationarity(gradient, cable_gradient_error,
                _rational(response['certificate']['gradientMaxAbsoluteErrorBoundNewtons']), cable_assembly_error)
            cable_converged = _rational(cable_stationarity['stationarityUpperBoundNewtons']) <= F(1e-6)
        row_errors = np.zeros(len(sewing_weights))
        if self.sewing_mode == "distance":
            _, lengths = distance_sewing.geometry(final)
            row_errors[sewing_positive] = np.abs(lengths[sewing_positive] - targets[sewing_positive])
        elif self.sewing_mode == "normal-offset":
            _, _, normals, _ = distance_sewing.geometry(final)
            errors = active_sewing @ final - (targets[sewing_positive] * self.sewing_sides[sewing_positive])[:, None] * normals[sewing_positive]
            row_errors[sewing_positive] = np.max(np.abs(errors), axis=1)
        else:
            row_errors = np.max(np.abs(vector_sewing_error(final)), axis=1)
        controlled_diagnostic = actuator.diagnostics(final) if controlled_fold is not None else None
        report = {
            "profile": "experimental-global-fixed-cable-reference-v1" if cable is not None else "experimental-global-ipc-guarded-contact-reference-v1" if contact_guard_assembled_metrics else "experimental-global-ipc-contact-reference-v1" if self.contact is not None else "experimental-global-local-fold-barrier-v1" if self.fold_barrier is not None else "experimental-global-elastic-bending-reference-v2" if self.has_bending else "experimental-global-membrane-sewing-reference-v6", "accepted": False,
            "contact": self.contact.profile() if self.contact is not None else None,
            "sewingMode": self.sewing_mode,
            "sewingActivationExplicit": sewing_activation is not None,
            "sewingActivation": sewing_weights.tolist(),
            "sewingActivationLimitations": "Per-row energy weights, not completed construction phases or continuous seam coverage; parameter work requires adjacent controls" if sewing_activation is not None else None,
            "activeSewingRows": np.flatnonzero(sewing_positive).tolist(),
            "pendingSewingRows": np.flatnonzero(~sewing_positive).tolist(),
            "sewingRowTargetErrorsM": [float(error) if active else None for error, active in zip(row_errors, sewing_positive)],
            "sewingTargetErrorMetric": "Unweighted maximum absolute Cartesian component per active vector/normal row; absolute scalar-distance error in distance mode; pending rows excluded",
            "triangleSweep": "all source faces; numerical Bernstein guard on optimizer and physical affine paths; v1",
            "foldActuation": actuator is not None,
            "foldActuationJoules": actuator.energy(final) if actuator is not None else 0.,
            "foldTargetsRadians": (controlled_diagnostic["targetsRadians"] if controlled_fold is not None else
                                   actuator.rest_angles.tolist() if actuator is not None else None),
            "foldAnglesRadians": (controlled_diagnostic["sampledActiveAnglesRadians"] if controlled_fold is not None else
                                  actuator.angles(final).tolist() if actuator is not None else None),
            **({"foldControls": controlled_diagnostic,
                "foldHingeSweepPolicy": "all declared controlled hinges, including inactive; optimizer and physical affine paths"}
               if controlled_fold is not None else {}),
            "foldActuationLimitations": "External angle penalty, not a change of cloth rest shape or a turning/binding recipe; Gauss-Newton search" if actuator is not None else None,
            "materialGrippers": grippers is not None,
            "gripperDiagnostics": grippers.diagnostics(final) if grippers is not None else None,
            "gripperEnergyJoules": grippers.energy(final) if grippers is not None else 0.,
            "gripperTargetsMeters": grippers.targets.tolist() if grippers is not None else None,
            "gripperActivation": grippers.activation.tolist() if grippers is not None else None,
            "gripperLimitations": "Prescribed compliant source-material anchors on free cloth; targets are not collision geometry or a verified binding/turning recipe. Parameter work requires adjacent states." if grippers is not None else None,
            "sewingJoules": float(.5 * np.sum(sewing_residual(final) ** 2)),
            "sewingTargetErrorM": float(np.max(row_errors, initial=0)),
            "sewingLimitations": ("Source-normal offset with full frame reactions and exact sewing curvature; Gauss-Newton fallback; swept triangles guarded, no turning or seam tangent alignment"
                                  if self.sewing_mode == "normal-offset" else
                                  "Scalar anchor distance does not prescribe layer side, seam tangent or turning"
                                  if distance_sewing is not None else "World-space vector registration"),
            "contactJoules": self.contact.energy(final) if self.contact is not None else 0.,
            "contactSearchMetric": "Signed-weight contact Hessian; both assembled metrics safeguarded: SPD primary or physical-inertia-shifted projected fallback; not exact total Hessian" if contact_guard_assembled_metrics else "PSD-projected contact Hessian; not exact total Hessian" if self.contact is not None else None,
            "contactRestMetricTolerance": float(self.contact_rest_metric_tolerance) if self.contact is not None else None,
            "localFoldBarrier": self.fold_barrier is not None,
            "foldBarrierJoules": self.fold_barrier.energy(final) if self.fold_barrier is not None else 0.,
            "bendingHinges": len(self.bending.indices),
            "bendingSearchMetric": "Exact membrane + Gauss-Newton bending; projected membrane fallback, not exact total Hessian" if self.has_bending else None,
            "converged": bool(result.success and gradient_norm <= 1e-6 and cable_converged),
            "optimizerSuccess": bool(result.success), "stationarityToleranceN": 1e-6,
            "linearSolver": linear_solver, "energyHistory": getattr(result, "energy_history", None),
            "exactSteps": getattr(result, "exact_steps", None), "projectedSteps": getattr(result, "projected_steps", None),
            "coupledSteps": getattr(result, "coupled_steps", None),
            "shiftedSteps": getattr(result, "shifted_steps", None),
            "directionHistory": getattr(result, "direction_history", None),
            "status": int(result.status), "message": result.message,
            "evaluations": int(result.nfev), "initialEnergy": initial_energy, "finalEnergy": objective(result.x),
            "gradientInfinityNorm": gradient_norm,
            **({'continuousCable': cable_report, 'cableStationarity': cable_stationarity,
                'cableEvaluationFailures': dict(cable_failures),
                'cableSearchMetric': 'Additive slack-sided generalized cable curvature; rounded PSD not certified; both assembled metrics guarded with physical-inertia fallback',
                'cableLineSearchPolicy': 'Conditional interval Armijo on actual binary endpoint displacement: slope upper<0 and change upper<=binary64(1e-4)*slope lower; non-cable terms retain existing numerical scope'}
               if cable is not None else {}),
            "limitations": [("Experimental frictionless surface contact; no body contact or seam exclusions. " if self.contact is not None else "No contact. ")
                            + ("Prescribed compliant material grippers provide external forces. " if grippers is not None else "No external translational forces. ")
                            + ("Cable anchor conversion defects remain explicit; the numerical joint does not certify exact translation invariance or force balance. " if cable is not None else "")
                            + "No material damping; diagnostic reference only.",
                            "Optional local angular fold barrier changes the energy model; it is not finite-thickness or nonadjacent cloth contact.",
                            "Elastic bending is uncalibrated; the numerical triangle guard requires independent saved-path verification.",
                            "Stationarity does not certify a local energy minimum or dynamic stability."],
        }
        if cable is not None:
            cable_identity()
            cable.validate_diagnostics(final, report['continuousCable'])
        return final, (final - previous) / dt, report
