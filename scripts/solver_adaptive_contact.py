import copy
from fractions import Fraction

import numpy as np

from solver_attempt_journal import diagnostic_json, same
from solver_assembly_schedule import AssemblySchedule
from solver_stationarity import stationarity_tolerance, validate_tightened_report


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


def _sewing_array_identity(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("Finite numeric sewing model identity required")
    return type(value), array.dtype.str, array.shape, array.tobytes()


def _sewing_model_identity(solver):
    """Snapshot coefficient storage and caches without normalizing either."""
    result = []
    for field in ("sewing", "sewing_xyz"):
        if not hasattr(solver, field):
            result.append((field, False))
            continue
        matrix = getattr(solver, field)
        if getattr(matrix, "format", None) != "csr":
            raise ValueError("Stable CSR sewing operators required")
        result.append((field, True, type(matrix), matrix.shape,
                       *(_sewing_array_identity(getattr(matrix, key)) for key in ("data", "indices", "indptr"))))
    result.append(("compliance", _sewing_array_identity(solver.compliance)))
    mode = getattr(solver, "sewing_mode", "vector")
    if type(mode) is not str or mode not in ("vector", "distance", "normal-offset"):
        raise ValueError("Stable explicit sewing mode required")
    result.append(("sewing_mode", mode))
    for field in ("sewing_frame_faces", "sewing_frames", "sewing_sides"):
        present = hasattr(solver, field)
        result.append((field, present, _sewing_array_identity(getattr(solver, field)) if present else None))
    return tuple(result)


def _validate_sewing_step(solver, model_identity, controls, fraction, options,
                          substep_targets, expected_targets, report):
    from solver_sewing_activation import validate_sewing_activation
    if _sewing_model_identity(solver) != model_identity:
        raise ValueError("Sewing model identity changed during the adaptive transition")
    expected_activation = controls.parameters(fraction)
    if report.get("sewingActivation") is None:
        raise ValueError("Step sewing activation diagnostics are required")
    reported_activation = validate_sewing_activation(report["sewingActivation"], len(expected_activation))
    active, pending = report.get("activeSewingRows"), report.get("pendingSewingRows")
    if (not _same_control_array(options["sewing_activation"], expected_activation)
            or not _same_control_array(substep_targets, expected_targets)
            or report.get("sewingActivationExplicit") is not True
            or not _same_control_array(reported_activation, expected_activation)
            or type(active) is not list or type(pending) is not list
            or any(type(row) is not int for row in [*active, *pending])
            or active != np.flatnonzero(expected_activation > 0).tolist()
            or pending != np.flatnonzero(expected_activation == 0).tolist()):
        raise ValueError("Step sewing controls or row diagnostics differ from the captured schedule")


def _validate_sewing_energy(energy):
    fields = ("sewingBeforeJoules", "sewingAfterJoules", "sewingFixedPositionAfterJoules",
              "sewingFixedParameterChangeJoules", "sewingChangeJoules", "sewingTargetParameterWorkJoules",
              "sewingActivationParameterWorkJoules", "sewingParameterWorkJoules",
              "sewingActivationIncreaseWorkJoules", "sewingReleaseEnergyRemovedJoules",
              "sewingParameterWorkComponentSumErrorBoundJoules", "mechanicalChangeJoules",
              "targetParameterWorkJoules", "externalParameterWorkJoules",
              "mechanicalChangeMinusTargetWorkJoules", "mechanicalChangeMinusParameterWorkJoules")
    if (type(energy) is not dict or energy.get("accepted") is not False
            or any(isinstance(energy.get(field), (bool, np.bool_))
                   or not isinstance(energy.get(field), (int, float, np.integer, np.floating))
                   or not np.isfinite(energy[field]) for field in fields)
            or any(energy[field] < 0 for field in ("sewingBeforeJoules", "sewingAfterJoules",
                "sewingFixedPositionAfterJoules", "sewingActivationIncreaseWorkJoules",
                "sewingReleaseEnergyRemovedJoules", "sewingParameterWorkComponentSumErrorBoundJoules"))):
        raise ValueError("Complete finite unaccepted sewing work accounting required")


def _validate_cable_identity(solver, control, definition):
    if (getattr(solver, "continuous_cable", None) is not control
            or getattr(solver, "cable_parameter_control", None) is not None
            or not same(control.description(), definition)):
        raise ValueError("Fixed cable control identity or precision changed during the adaptive transition")


def _validate_cable_step(solver, control, definition, positions, report, *, tolerance_newtons=1e-6):
    from solver_cable_integration import CableControl, _rational, stationarity
    _validate_cable_identity(solver, control, definition)
    isolated = positions.copy()
    expected_diagnostics = control.diagnostics(isolated)
    if not _same_control_array(isolated, positions):
        raise ValueError("Cable diagnostic evaluation mutated its candidate position input")
    expected_diagnostics = CableControl.validate_diagnostics(control, positions, expected_diagnostics)
    _validate_cable_identity(solver, control, definition)
    if not same(report.get("continuousCable"), expected_diagnostics):
        raise ValueError("Step cable diagnostics differ from fresh supplied-state evaluation")
    bounds = report.get("cableStationarity")
    norm = report.get("gradientInfinityNorm")
    if (type(bounds) is not dict or isinstance(norm, bool)
            or not isinstance(norm, (int, float)) or not np.isfinite(norm) or norm < 0
            or "assemblyRoundingBoundNewtons" not in bounds):
        raise ValueError("Complete finite cable stationarity diagnostics required")
    cable_error = _rational(expected_diagnostics["certificate"]["gradientMaxAbsoluteErrorBoundNewtons"])
    assembly_error = _rational(bounds["assemblyRoundingBoundNewtons"])
    expected_bounds = stationarity(np.array([norm]), cable_error+assembly_error, cable_error, assembly_error,
                                   tolerance_newtons=tolerance_newtons)
    if (not same(bounds, expected_bounds) or report.get("converged") is not True
            or _rational(expected_bounds["stationarityUpperBoundNewtons"]) > Fraction(tolerance_newtons)):
        raise ValueError("Cable uncertainty-aware stationarity is unresolved or inconsistent")
    return expected_diagnostics


def _validate_cable_energy(control, previous, positions, report, after):
    from solver_cable_integration import CableControl
    from solver_energy_balance import validate_continuous_cable_energy
    if (type(report) is not dict or report.get("accepted") is not False
            or any(type(report.get(field)) is not float or not np.isfinite(report[field])
                   for field in ("targetParameterWorkJoules", "externalParameterWorkJoules"))):
        raise ValueError("Complete finite unaccepted cable transition accounting required")
    original_report = copy.deepcopy(report)
    payload = validate_continuous_cable_energy(control, previous, positions, report)
    old = previous.copy()
    before = control.diagnostics(old)
    if not _same_control_array(old, previous):
        raise ValueError("Cable before-energy validation mutated its isolated state")
    before = CableControl.validate_diagnostics(control, previous, before)
    start, end = previous.copy(), positions.copy()
    work = control.energy_change(start, end)
    if not _same_control_array(start, previous) or not _same_control_array(end, positions):
        raise ValueError("Cable fixed-work validation mutated its isolated states")
    work = CableControl.validate_change(control, previous, positions, work)
    for field, expected in (("before", before), ("after", after), ("work", work)):
        if not same(payload[field], expected):
            raise ValueError("Cable work accounting differs from fresh immutable-control evaluation: " + field)
    if not same(report, original_report):
        raise ValueError("Cable energy report changed during fresh numerical validation")


def _validate_varying_identity(solver, control, definition, schedule, schedule_definition):
    if (getattr(solver, "cable_parameter_control", None) is not control
            or getattr(solver, "continuous_cable", None) is not None
            or not same(control.description(), definition)
            or not same(schedule.description(), schedule_definition)):
        raise ValueError("Varying cable recipe, schedule or precision changed during the adaptive transition")


def _isolated_cable_call(function, *arrays):
    """Keep helper mutation separate from accepted/candidate states and controls."""
    copies = [value.copy() for value in arrays]
    result = function(*copies)
    if any(not _same_control_array(actual, expected) for actual, expected in zip(copies, arrays)):
        raise ValueError("Varying cable helper mutated isolated state or control inputs")
    return result


def _varying_cable_step(control, schedule, fraction, options, positions, report, *, tolerance_newtons=1e-6):
    from solver_cable_integration import CableControl, _rational, stationarity
    values, weights = schedule.parameters(fraction)
    if (not _same_control_array(options["cable_targets"], values)
            or not _same_control_array(options["cable_activation"], weights)):
        raise ValueError("Step mutated original-fraction varying cable controls")
    expected_record = _isolated_cable_call(control.parameter_record, values, weights)
    if (report.get("profile") != "experimental-global-varying-cable-reference-v1"
            or not same(report.get("varyingCable"), expected_record)):
        raise ValueError("Step varying cable parameter record differs from original-fraction controls")
    effective = _isolated_cable_call(control.effective, values, weights)
    if type(effective) is not CableControl:
        raise ValueError("An admitted immutable effective cable control is required")
    diagnostic = _isolated_cable_call(effective.diagnostics, positions)
    diagnostic = _isolated_cable_call(
        lambda q: CableControl.validate_diagnostics(effective, q, diagnostic), positions)
    if not same(report.get("continuousCable"), diagnostic):
        raise ValueError("Varying cable response differs from fresh end-control evaluation")
    bounds, norm = report.get("cableStationarity"), report.get("gradientInfinityNorm")
    if (type(bounds) is not dict or type(norm) not in (int, float)
            or not np.isfinite(norm) or norm < 0 or "assemblyRoundingBoundNewtons" not in bounds):
        raise ValueError("Complete finite varying cable stationarity diagnostics required")
    cable_error = _rational(diagnostic["certificate"]["gradientMaxAbsoluteErrorBoundNewtons"])
    assembly_error = _rational(bounds["assemblyRoundingBoundNewtons"])
    expected = stationarity(np.array([norm]), cable_error+assembly_error, cable_error, assembly_error,
                           tolerance_newtons=tolerance_newtons)
    if (not same(bounds, expected) or report.get("converged") is not True
            or _rational(expected["stationarityUpperBoundNewtons"]) > Fraction(tolerance_newtons)):
        raise ValueError("Varying cable uncertainty-aware stationarity is unresolved or inconsistent")
    return diagnostic


def _validate_varying_cable_energy(control, schedule, start, end, previous, positions, report, after):
    from solver_cable_integration import CableControl
    from solver_cable_varying import VaryingCableControl
    from solver_energy_balance import validate_varying_cable_energy
    if type(report) is not dict or report.get("accepted") is not False:
        raise ValueError("Complete unaccepted varying cable energy accounting required")
    original = copy.deepcopy(report)
    d0, a0 = schedule.parameters(start)
    d1, a1 = schedule.parameters(end)
    payload = _isolated_cable_call(
        lambda q0, q1, t0, w0, t1, w1: validate_varying_cable_energy(control, q0, q1, t0, w0, t1, w1, report),
        previous, positions, d0, a0, d1, a1)
    old = _isolated_cable_call(control.effective, d0, a0)
    new = _isolated_cable_call(control.effective, d1, a1)
    if type(old) is not CableControl or type(new) is not CableControl:
        raise ValueError("Admitted immutable old and new effective cable controls required")
    before = _isolated_cable_call(old.diagnostics, previous)
    before = _isolated_cable_call(lambda q: CableControl.validate_diagnostics(old, q, before), previous)
    parameter_work = _isolated_cable_call(control.parameter_work, previous, d0, a0, d1, a1)
    parameter_work = _isolated_cable_call(
        lambda q, t0, w0, t1, w1: VaryingCableControl.validate_parameter_work(control, q, t0, w0, t1, w1, parameter_work),
        previous, d0, a0, d1, a1)
    motion_work = _isolated_cable_call(new.energy_change, previous, positions)
    motion_work = _isolated_cable_call(
        lambda q0, q1: CableControl.validate_change(new, q0, q1, motion_work), previous, positions)
    for key, expected in (("before", before), ("after", after),
                          ("parameterWork", parameter_work), ("motionWork", motion_work),
                          ("beforeParameters", _isolated_cable_call(control.parameter_record, d0, a0)),
                          ("afterParameters", _isolated_cable_call(control.parameter_record, d1, a1))):
        if not same(payload[key], expected):
            raise ValueError("Varying cable accounting differs from fresh numerical evaluation: " + key)
    if not same(report, original):
        raise ValueError("Varying cable energy report changed during final numerical validation")


def _step_core(report, *, grippers):
    authored = {"energyBalance", "gripperMomentum"} if grippers else {"energyBalance"}
    return {key: value for key, value in report.items() if key not in authored}


def _varying_cable_totals(accepted):
    from solver_cable_integration import _rat, _rational, round_sum
    fields = ("cableTargetParameterWorkJoules", "cableActivationParameterWorkJoules",
              "cableParameterWorkJoules", "cableActivationIncreaseWorkJoules",
              "cableReleaseEnergyRemovedJoules", "cableFixedParameterChangeJoules", "cableChangeJoules",
              "targetParameterWorkJoules", "externalParameterWorkJoules", "mechanicalChangeJoules",
              "mechanicalChangeMinusTargetWorkJoules", "mechanicalChangeMinusParameterWorkJoules")
    values, bounds, rounding = {}, {}, {}
    for field in fields:
        reports = [row["step"]["energyBalance"] for row in accepted]
        radius = sum((_rational(report["varyingCableEnergy"]["errorBoundsJoules"][field])
                      for report in reports), Fraction())
        value, bound = round_sum((report[field] for report in reports), radius)
        values[field], bounds[field], rounding[field] = value, _rat(bound), _rat(bound-radius)
    return {"valuesJoules": values, "errorBoundsJoules": bounds, "assemblyRoundingBoundsJoules": rounding,
            "acceptedStepCount": len(accepted),
            "scope": "One rounding of exact sums of accepted-step binary64 work and mechanical values, with summed certified cable/assembly bounds plus final rounding. Rejected and interrupted attempts and endpoint energy diagnostics are excluded. Non-cable constitutive errors remain uncertified; no continuous actuator-work, physical dissipation or source acceptance claim."}


def adaptive_contact_step(solver, positions, velocities, initial_targets, targets, dt, *,
                          max_depth=8, max_attempts=256, initial_subdivisions=1, on_accept=None,
                          attempt_journal=None, initial_fold_targets=None, fold_targets=None,
                          assembly_schedule=None, gripper_schedule=None, sewing_schedule=None, sewing_row_ids=None,
                          fold_control_schedule=None, cable_parameter_schedule=None,
                          stationarity_tolerance_newtons=1e-6, **step_options):
    tolerance = stationarity_tolerance(stationarity_tolerance_newtons)
    tightened = tolerance != 1e-6
    if tightened:
        from solver_attempt_journal import AttemptJournal
        if isinstance(attempt_journal, AttemptJournal):
            raise ValueError("Legacy captured journals declare only 1e-6 N; tightened research needs its own explicit journal")
        if step_options.get("linear_solver", "direct") != "direct":
            raise ValueError("Tightened stationarity requires guarded direct search")
        step_options["stationarity_tolerance_newtons"] = tolerance
    if on_accept is not None and not callable(on_accept):
        raise ValueError("Accepted-state callback must be callable")
    if "sewing_activation" in step_options:
        raise ValueError("Adaptive sewing activation requires an explicit captured activation schedule")
    if "fold_activation" in step_options:
        raise ValueError("Adaptive fold activation requires an explicit per-hinge control schedule")
    cable_control = getattr(solver, "continuous_cable", None)
    cable_definition = None
    varying_control = getattr(solver, "cable_parameter_control", None)
    varying_definition = cable_schedule_definition = cable_preflight = cable_controls = None
    if (varying_control is None) != (cable_parameter_schedule is None):
        raise ValueError("Varying cable control and explicit parameter schedule must be supplied together")
    if any(key in step_options for key in ("cable_targets", "cable_activation")):
        raise ValueError("Adaptive cable parameters must come from the original captured schedule")
    if varying_control is not None:
        from solver_cable_varying import VaryingCableControl
        from solver_controlled_fold import _binary64
        if (type(varying_control) is not VaryingCableControl or cable_control is not None
                or varying_control.vertex_count != len(solver.mass)):
            raise ValueError("An exclusive admitted immutable varying cable control is required")
        varying_definition = copy.deepcopy(varying_control.description())
        positions, velocities = varying_control.positions(positions), varying_control.positions(velocities)
        dt = _binary64(dt)
        if step_options.get("linear_solver", "direct") != "direct":
            raise ValueError("Varying cable controls require guarded direct search")
        if any(key == "continuous_cable" or key.startswith("cable_") for key in step_options):
            raise ValueError("Varying cable controls and precision cannot be overridden per adaptive step")
    if cable_control is not None:
        from solver_cable_integration import CableControl
        from solver_controlled_fold import _binary64
        if type(cable_control) is not CableControl or cable_control.vertex_count != len(solver.mass):
            raise ValueError("An admitted immutable fixed cable control is required")
        cable_definition = copy.deepcopy(cable_control.description())
        positions = cable_control.positions(positions)
        velocities = cable_control.positions(velocities)
        dt = _binary64(dt)
        if step_options.get("linear_solver", "direct") != "direct":
            raise ValueError("Continuous cable controls require guarded direct search")
        if any(key == "continuous_cable" or key.startswith("cable_") for key in step_options):
            raise ValueError("Fixed cable controls and precision cannot be overridden per adaptive step")
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
    if varying_control is not None:
        from solver_cable_parameter_schedule import CableParameterSchedule
        cable_controls = CableParameterSchedule(cable_parameter_schedule, int(initial_subdivisions),
                                                cable_recipe=varying_control.recipe)
        cable_schedule_definition = copy.deepcopy(cable_controls.description())
        cable_preflight = cable_controls.preflight(int(max_depth))
        initial_parameters = cable_controls.parameters(0)
        if varying_control.parameter_sha256(*initial_parameters) != varying_control.parameter_sha256(*varying_control.recipe.initial_parameters):
            raise ValueError("Initial cable schedule parameters must exactly match the declared base parameters")
        _validate_varying_identity(solver, varying_control, varying_definition, cable_controls, cable_schedule_definition)
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
    sewing_controls = sewing_model_identity = None
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
        sewing_model_identity = _sewing_model_identity(solver)
    if varying_control is not None:
        varying_sewing_identity = _sewing_model_identity(solver)
        varying_other_controls = (fold_recipe, controlled_fold_recipe, gripper_recipe)
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
            attempt_journal.start(copy.deepcopy(record) if varying_control is not None or tightened else record)
        fatal = False
        propagate = None
        try:
            options = dict(step_options)
            if cable_control is not None:
                _validate_cable_identity(solver, cable_control, cable_definition)
            if varying_control is not None:
                _validate_varying_identity(solver, varying_control, varying_definition, cable_controls, cable_schedule_definition)
                if (_sewing_model_identity(solver) != varying_sewing_identity
                        or any(getattr(solver, field, None) is not expected for field, expected in zip(
                            ("fold_actuation", "controlled_fold_actuation", "material_grippers"), varying_other_controls))):
                    raise ValueError("Existing sewing or controller identity changed during varying cable continuation")
                options["cable_targets"], options["cable_activation"] = cable_controls.parameters(end_fraction)
                before_values, before_weights = cable_controls.parameters(start_fraction)
                record["cableParameterInterval"] = {
                    "before": _isolated_cable_call(varying_control.parameter_record, before_values, before_weights),
                    "after": _isolated_cable_call(varying_control.parameter_record,
                                                 options["cable_targets"], options["cable_activation"])}
                varying_interval_snapshot = copy.deepcopy(record["cableParameterInterval"])
            if fold_recipe is not None:
                options["fold_targets"] = (fold_targets.copy() if fold_progress == 1 else
                    initial_fold_targets + fold_progress * (fold_targets - initial_fold_targets))
            if fold_controls is not None:
                options["fold_targets"], options["fold_activation"] = fold_controls.parameters(end_fraction)
            if gripper_controls is not None:
                options["gripper_targets"], options["gripper_activation"] = gripper_controls.parameters(end_fraction)
            if sewing_controls is not None:
                options["sewing_activation"] = sewing_controls.parameters(end_fraction)
                if _sewing_model_identity(solver) != sewing_model_identity:
                    raise ValueError("Sewing model identity changed before the adaptive step")
            step_positions, step_velocities = current_positions.copy(), current_velocities.copy()
            if varying_control is not None:
                # Retain every live state/control passed through publication;
                # late callbacks in a helper must not defeat an earlier check.
                varying_arrays = [current_positions, current_velocities, step_positions, step_velocities,
                                  substep_targets, *(value for value in options.values() if type(value) is np.ndarray)]
                varying_snapshots = [array.copy() for array in varying_arrays]
            candidate_positions, candidate_velocities, step_report = solver.step(
                step_positions, step_velocities, substep_targets, duration, **options)
            if not isinstance(step_report, dict):
                raise ValueError("Solver diagnostic report must be an object")
            record["step"], nonfinite = diagnostic_json(step_report)
            record["nonfiniteDiagnostics"] = nonfinite
            if fold_controls is not None:
                candidate_positions = controlled_fold_recipe._positions(candidate_positions)
                candidate_velocities = controlled_fold_recipe._positions(candidate_velocities)
            if cable_control is not None:
                candidate_positions = cable_control.positions(candidate_positions)
                candidate_velocities = cable_control.positions(candidate_velocities)
            if varying_control is not None:
                candidate_positions = varying_control.positions(candidate_positions)
                candidate_velocities = varying_control.positions(candidate_velocities)
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
                     and np.isfinite(residual) and 0 <= residual <= tolerance
                     and step_report.get("converged") is True)
            if valid:
                # Validate original scalars before diagnostic_json can narrow
                # a wider real or Fraction to the requested binary64 value.
                validate_tightened_report(step_report, tolerance)
            if valid and fold_controls is not None:
                expected_targets = (targets.copy() if sewing_progress == 1 else
                                    initial_targets + sewing_progress * (targets-initial_targets))
                if not _same_control_array(substep_targets, expected_targets):
                    raise ValueError("Controlled step mutated the original sewing target interpolation")
                _validate_fold_step(solver, controlled_fold_recipe, fold_controls, end_fraction, options,
                                    candidate_positions, record["step"])
            if valid and sewing_controls is not None:
                expected_targets = (targets.copy() if sewing_progress == 1 else
                                    initial_targets + sewing_progress * (targets - initial_targets))
                _validate_sewing_step(solver, sewing_model_identity, sewing_controls, end_fraction,
                    options, substep_targets, expected_targets, record["step"])
                original_step_core = copy.deepcopy(_step_core(record["step"], grippers=gripper_controls is not None))
            if valid and cable_control is not None:
                expected_targets = (targets.copy() if sewing_progress == 1 else
                                    initial_targets + sewing_progress * (targets-initial_targets))
                if not _same_control_array(substep_targets, expected_targets):
                    raise ValueError("Cable step mutated the original sewing target interpolation")
                _validate_cable_step(solver, cable_control, cable_definition, candidate_positions, record["step"],
                                     tolerance_newtons=tolerance)
                original_step_core = copy.deepcopy(_step_core(record["step"], grippers=gripper_controls is not None))
            if valid and varying_control is not None:
                varying_arrays.extend((candidate_positions, candidate_velocities))
                varying_snapshots.extend((candidate_positions.copy(), candidate_velocities.copy()))
                original_step_core = copy.deepcopy(_step_core(record["step"], grippers=gripper_controls is not None))
                if any(not _same_control_array(actual, expected) for actual, expected in zip(varying_arrays, varying_snapshots)):
                    raise ValueError("Varying cable solver mutated supplied transition inputs")
                _varying_cable_step(varying_control, cable_controls, end_fraction, options, candidate_positions, record["step"],
                                   tolerance_newtons=tolerance)
                _validate_varying_identity(solver, varying_control, varying_definition, cable_controls, cable_schedule_definition)
            if valid and (gripper_controls is not None or sewing_controls is not None or fold_controls is not None
                          or cable_control is not None or varying_control is not None):
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
                if varying_control is not None:
                    old_cable_targets, old_cable_activation = cable_controls.parameters(start_fraction)
                    new_cable_targets, new_cable_activation = cable_controls.parameters(end_fraction)
                    energy_options.update(previous_cable_targets=old_cable_targets,
                                          previous_cable_activation=old_cable_activation,
                                          cable_targets=new_cable_targets, cable_activation=new_cable_activation)
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
                if (fold_controls is not None or sewing_controls is not None
                        or cable_control is not None or varying_control is not None):
                    # Work is computed before publication. Isolated inputs
                    # preserve the last accepted state even if a helper fails
                    # after mutation; successful mutation also rejects.
                    work_states = [array.copy() for array in (current_positions, candidate_positions,
                        current_velocities, candidate_velocities, old_targets, substep_targets)]
                    work_options = {key: value.copy() for key, value in energy_options.items()}
                    observed_arrays = [*work_states, *work_options.values()]
                    snapshots = [array.copy() for array in observed_arrays]
                    if varying_control is not None:
                        varying_arrays.extend(observed_arrays)
                        varying_snapshots.extend(snapshots)
                    energy = global_energy_transition(solver, *work_states, duration, **work_options)
                    if any(not _same_control_array(actual, expected) for actual, expected in zip(observed_arrays, snapshots)):
                        raise ValueError("Controlled work helper mutated transition inputs")
                    if fold_controls is not None:
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
                    momentum_tolerance = len(solver.mass) * duration * tolerance + 64 * np.finfo(float).eps * max(
                        1., float(np.max(np.abs(momentum))), float(np.max(np.abs(impulse))))
                    if (not np.all(solver.active) or not np.isfinite(error).all()
                            or np.max(np.abs(error)) > momentum_tolerance):
                        raise ValueError("Material-gripper transition fails free-cloth linear momentum accounting")
                    step_report = dict(step_report, gripperMomentum={
                        "changeKgMPerS": momentum.tolist(), "externalImpulseNs": impulse.tolist(),
                        "residualNs": error.tolist(), "toleranceNs": float(momentum_tolerance),
                        "scope": "Backward-Euler force at the new state on free cloth; virtual gripper impulse, not isolated-cloth momentum conservation"})
                validate_tightened_report(step_report, tolerance)
                record["step"], nonfinite = diagnostic_json(step_report)
                record["nonfiniteDiagnostics"] = nonfinite
                if nonfinite:
                    raise ValueError("Controlled transition has nonfinite work or momentum diagnostics")
                if cable_control is not None:
                    cable_publication_snapshot = copy.deepcopy(record["step"])
                if varying_control is not None:
                    varying_publication_snapshot = copy.deepcopy(record["step"])
                if fold_controls is not None:
                    _validate_fold_step(solver, controlled_fold_recipe, fold_controls, end_fraction, options,
                                        candidate_positions, record["step"])
                    final_residual = record["step"].get("gradientInfinityNorm")
                    if (record["step"].get("converged") is not True
                            or isinstance(final_residual, bool)
                            or not isinstance(final_residual, (int, float))
                            or not 0 <= final_residual <= tolerance):
                        raise ValueError("Controlled-fold final convergence diagnostics changed before publication")
                if sewing_controls is not None:
                    _validate_sewing_step(solver, sewing_model_identity, sewing_controls, end_fraction,
                        options, substep_targets, expected_targets, record["step"])
                    if not same(_step_core(record["step"], grippers=gripper_controls is not None), original_step_core):
                        raise ValueError("Sewing step diagnostics changed during work before publication")
                    _validate_sewing_energy(record["step"].get("energyBalance"))
                if cable_control is not None:
                    fresh_cable_after = _validate_cable_step(solver, cable_control, cable_definition, candidate_positions, record["step"],
                                                             tolerance_newtons=tolerance)
                    _validate_cable_energy(cable_control, current_positions, candidate_positions,
                                           record["step"].get("energyBalance"), fresh_cable_after)
                    if not same(record["step"], cable_publication_snapshot):
                        raise ValueError("Cable publication diagnostics changed during final numerical validation")
                    if not same(_step_core(record["step"], grippers=gripper_controls is not None), original_step_core):
                        raise ValueError("Cable step diagnostics changed during work before publication")
                    _validate_cable_identity(solver, cable_control, cable_definition)
                if varying_control is not None:
                    fresh_after = _varying_cable_step(varying_control, cable_controls, end_fraction,
                                                     options, candidate_positions, record["step"], tolerance_newtons=tolerance)
                    _validate_varying_cable_energy(varying_control, cable_controls, start_fraction, end_fraction,
                        current_positions, candidate_positions, record["step"].get("energyBalance"), fresh_after)
                    final_targets, final_activation = cable_controls.parameters(end_fraction)
                    _validate_varying_identity(solver, varying_control, varying_definition, cable_controls, cable_schedule_definition)
                    if any(not _same_control_array(actual, expected) for actual, expected in zip(varying_arrays, varying_snapshots)):
                        raise ValueError("Varying cable transition state or controls changed before publication")
                    if (_sewing_model_identity(solver) != varying_sewing_identity
                            or any(getattr(solver, field, None) is not expected for field, expected in zip(
                                ("fold_actuation", "controlled_fold_actuation", "material_grippers"), varying_other_controls))
                            or not _same_control_array(options["cable_targets"], final_targets)
                            or not _same_control_array(options["cable_activation"], final_activation)
                            or not same(record.get("cableParameterInterval"), varying_interval_snapshot)
                            or not same(record["step"], varying_publication_snapshot)
                            or not same(_step_core(record["step"], grippers=gripper_controls is not None), original_step_core)):
                        raise ValueError("Varying cable step or accounting changed during final numerical validation")
            if valid:
                validate_tightened_report(record["step"], tolerance)
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
            if varying_control is not None or tightened:
                attempt_journal.outcome(copy.deepcopy(record), candidate_positions.copy() if valid else None,
                                        candidate_velocities.copy() if valid else None)
            else:
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
        "maxAttempts": int(max_attempts), "stationarityToleranceN": tolerance,
        "attempts": attempts, "acceptedSteps": accepted, "rejectedSteps": rejected, "interruptedSteps": [],
        "targetInterpolation": ("captured piecewise-linear sewing/fold progress" if schedule else
                                "linear sewing progress over the original physical interval" if (gripper_controls is not None or sewing_controls is not None) else
                                "linear over the original physical interval"),
        **({"sewingActivationInterpolation": "captured monotone piecewise-linear canonical-row activation over the original physical interval"}
           if sewing_controls is not None else {}),
        **({"gripperInterpolation": "captured piecewise-linear material-point targets and activation over the original physical interval"}
           if gripper_controls is not None else {}),
        **({"continuousCable": copy.deepcopy(cable_definition),
            "continuousCableScope": "Fixed supplied-cell control; cable uncertainty is checked before publication. Other numerical terms and their assembly diagnostics remain trusted global-solver terms; no independent non-cable error certificate, source admission or construction acceptance."}
           if cable_control is not None else {}),
        **({"varyingCableControl": copy.deepcopy(varying_definition),
            "cableParameterSchedule": copy.deepcopy(cable_schedule_definition),
            "cableParameterPreflight": copy.deepcopy(cable_preflight),
            "finalCableParameters": varying_control.parameter_record(*cable_controls.parameters(completed_fraction)),
            "varyingCableAcceptedWorkTotals": _varying_cable_totals(accepted),
            "cableParameterInterpolation": "Original dyadic fractions with exact binary64-endpoint interpolation and one rounding; rejected attempts advance neither state nor controls",
            "varyingCableScope": "Generic supplied-cell controls with fixed precision policies; parameter work and new-control motion work are freshly checked before publication. Other numerical terms remain trusted global-solver terms. No captured source admission, continuous actuator-work certificate, construction acceptance or resume-at-fraction API."}
           if varying_control is not None else {}),
        **({"foldControlInterpolation": "explicit per-hinge targets and activation sampled by exact interpolation at original dyadic fractions; no captured source or construction admission"}
           if fold_controls is not None else {}),
    }
