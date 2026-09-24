"""Conditional step-doubling control, with synchronous in-memory pair commits.

These indicators do not certify continuous-time error, physical dissipation,
or garment acceptance. Trial evaluation supplies the existing mechanical and
path checks. This module never extrapolates or publishes a coarse trial.
"""
import copy
from fractions import Fraction as F
import hashlib
import math
from types import MappingProxyType

import numpy as np


PROFILE = "conditional-step-doubling-v1"
MOTION_FIELDS = ("membraneChangeJoules", "bendingChangeJoules", "foldBarrierChangeJoules",
                 "contactChangeJoules", "sewingFixedParameterChangeJoules",
                 "foldFixedParameterChangeJoules", "gripperFixedParameterChangeJoules",
                 "cableFixedParameterChangeJoules")


class _ObjectIdentity:
    """Retain native wrappers so repeated property access has stable identity."""
    __slots__ = ("value",)

    def __init__(self, value): self.value = value
    def __hash__(self): return id(self.value)
    def __eq__(self, other): return type(other) is _ObjectIdentity and self.value is other.value


def strict_array(value):
    from solver_controlled_fold import _binary64
    # Object construction preserves mixed Python integer/float inputs before
    # checking each original scalar for exact binary64 representability.
    raw = np.asarray(value, dtype=object)
    return np.array([_binary64(item) for item in raw.flat], dtype=np.float64).reshape(raw.shape)


def problem_identity(solver, controls=()):
    """Snapshot numerical Python configuration, excluding only contact caches.

    Native object identities and exposed contact parameters are included;
    this does not purport to inspect arbitrary hidden native implementation.
    """
    caches = {"_cached_positions", "_cached_collisions", "_bucket_positions", "_bucket_cache",
              "_exact_certificates_json"}

    def capture(value):
        if isinstance(value, np.ndarray):
            return type(value), value.dtype.str, value.shape, value.strides, value.tobytes()
        if isinstance(value, np.generic):
            return type(value), value.tobytes()
        if value is None or type(value) in (bool, int, str, bytes):
            return type(value), value
        if type(value) is float:
            return float, value.hex()
        if type(value) is F:
            return F, value.numerator, value.denominator
        if type(value) in (list, tuple):
            return type(value), tuple(capture(item) for item in value)
        if type(value) in (set, frozenset):
            return type(value), frozenset(capture(item) for item in value)
        if type(value) in (dict, MappingProxyType):
            return type(value), frozenset((capture(key), capture(item)) for key, item in value.items())
        if getattr(value, "format", None) in ("csr", "csc"):
            return type(value), value.shape, *(capture(getattr(value, key)) for key in ("data", "indices", "indptr"))
        if type(value).__module__.startswith("solver_"):
            fields = dict(getattr(value, "__dict__", {}))
            for cls in type(value).__mro__:
                for name in getattr(cls, "__slots__", ()):
                    if hasattr(value, name):
                        fields[name] = getattr(value, name)
            is_contact = type(value).__module__ in ("solver_ipc_contact", "solver_rest_filtered_contact")
            return type(value), id(value), tuple((name, capture(item)) for name, item in sorted(fields.items())
                                                  if not (is_contact and name in caches))
        native_parameters = tuple((name, capture(getattr(value, name))) for name in
                                  ("dhat", "barrier", "tolerance", "max_iterations", "conservative_rescaling")
                                  if hasattr(value, name))
        return type(value), _ObjectIdentity(value), native_parameters

    fields = ("mass", "active", "free", "faces", "poses", "areas", "materials", "has_bending",
              "contact_rest_metric_tolerance", "sewing", "sewing_xyz", "compliance", "sewing_mode",
              "sewing_frame_faces", "sewing_sides", "bending", "contact", "fold_barrier",
              "fold_actuation", "controlled_fold_actuation", "material_grippers",
              "continuous_cable", "cable_parameter_control", "contact_work_control")
    return tuple((name, capture(getattr(solver, name, None))) for name in fields) + (("controls", capture(controls)),)


def rational(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _fraction(value):
    if (type(value) is not dict or set(value) != {"numerator", "denominator"}
            or any(type(part) is not str or len(part) > 4096 for part in value.values())):
        raise ValueError("Bounded rational certificate required")
    try:
        result = F(int(value["numerator"]), int(value["denominator"]))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Valid rational certificate required") from error
    if result < 0 or rational(result) != value:
        raise ValueError("Canonical nonnegative rational certificate required")
    return result


def policy(value):
    """Copy a strictly declared policy; no mutable user policy enters trials."""
    from solver_controlled_fold import _binary64
    fields = {"profile", "positionToleranceM", "velocityToleranceMPerS",
              "numericalEnergyBudgetJ", "maxEvaluationBudget"}
    if type(value) is not dict or set(value) != fields or value["profile"] != PROFILE:
        raise ValueError("Complete explicit conditional step-doubling policy required")
    result = {"profile": PROFILE}
    for field in ("positionToleranceM", "velocityToleranceMPerS", "numericalEnergyBudgetJ"):
        number = _binary64(value[field])
        if number <= 0:
            raise ValueError("Positive finite absolute temporal thresholds required")
        result[field] = number
    budget = value["maxEvaluationBudget"]
    if type(budget) is not int or not 1 <= budget <= 4096 * 10000:
        raise ValueError("Bounded explicit temporal evaluation budget required")
    result["maxEvaluationBudget"] = budget
    return result


def _state(value, shape=None):
    if (type(value) is not np.ndarray or value.dtype != np.dtype(np.float64)
            or value.ndim != 2 or value.shape[1] != 3 or not len(value)
            or (shape is not None and value.shape != shape) or not np.isfinite(value).all()):
        raise ValueError("Finite binary64 temporal state required")
    # Immutable bytes sever aliases retained by a solver/helper between trials.
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def state_record(positions, velocities):
    positions, velocities = _state(positions), _state(velocities, positions.shape)
    identity = hashlib.sha256(str(positions.shape).encode("ascii") + b"\0" +
                              positions.tobytes() + velocities.tobytes()).hexdigest()
    return {"sha256": identity, "positionsMeters": positions.tolist(), "velocitiesMPerS": velocities.tolist()}


def discrepancy(first, second, tolerance):
    """Exact stored-value squared norms decide; the square root is display only."""
    first, second = _state(first), _state(second, first.shape)
    squares = [sum(((F(float(a))-F(float(b)))**2 for a, b in zip(row, other)), F())
               for row, other in zip(first, second)]
    maximum = max(squares)
    limit = F(tolerance)**2
    # Avoid overflow/underflow affecting the decision or JSON diagnostics.
    approximate = math.sqrt(float(maximum)) if maximum <= F(np.finfo(float).max) else None
    return {"maximumSquared": rational(maximum), "toleranceSquared": rational(limit),
            "maximumApproximate": approximate, "maximumVertex": squares.index(maximum),
            "withinThreshold": maximum <= limit}


def energy_defect(mass, before, after, energy, allocation):
    before, after = _state(before), _state(after, before.shape)
    if (type(mass) is not np.ndarray or mass.dtype != np.dtype(np.float64)
            or mass.shape != (len(before),) or not np.isfinite(mass).all() or np.any(mass <= 0)):
        raise ValueError("Free positive binary64 masses required for temporal energy accounting")
    if type(energy) is not dict or energy.get("accepted") is not False:
        raise ValueError("An unaccepted complete mechanical energy record is required")
    from solver_energy_balance import _VARYING_BASE_SCALARS
    # This set is unconditional in global_energy_transition; varying cable
    # validation already requires it. Require it for every temporal route too.
    if any(type(energy.get(key)) is not float or not math.isfinite(energy[key]) for key in _VARYING_BASE_SCALARS):
        raise ValueError("Complete finite public mechanical and parameter-work accounting required")
    payload = energy.get("temporalMotion")
    if (type(payload) is not dict or set(payload) != {"termsJoules", "knownErrorBoundJoules"}
            or type(payload["termsJoules"]) is not dict or set(payload["termsJoules"]) != set(MOTION_FIELDS)):
        raise ValueError("Complete separate fixed-control motion terms required")
    terms = payload["termsJoules"]
    if any(type(value) is not float or not math.isfinite(value) for value in terms.values()):
        raise ValueError("Finite binary64 motion terms required")
    if any(type(energy.get(key)) is not float or energy[key].hex() != value.hex() for key, value in terms.items()):
        raise ValueError("Temporal motion terms differ from complete public fixed-motion accounting")
    cable = energy.get("varyingCableEnergy", energy.get("continuousCableEnergy"))
    from solver_contact_work_control import contact_error
    from solver_energy_balance import validate_contact_mechanical
    validate_contact_mechanical(energy)
    expected_radius = contact_error(energy)
    if cable is not None:
        from solver_cable_integration import _rational
        work = cable["motionWork"] if "varyingCableEnergy" in energy else cable["work"]
        expected_radius += _rational(work["certificate"]["changeErrorBoundJoules"])
        reference = cable["aggregationTermsJoules"]["mechanicalChangeMinusParameterWorkJoules"]
        if any(type(reference.get(key)) is not float or reference[key].hex() != value.hex() for key, value in terms.items()):
            raise ValueError("Temporal motion terms differ from validated cable accounting summands")
    delta = sum((F(float(m))/2 * sum((F(float(b))**2-F(float(a))**2
                  for a, b in zip(old, new)), F()) for m, old, new in zip(mass, before, after)), F())
    nominal = delta + sum((F(value) for value in terms.values()), F())
    radius = _fraction(payload["knownErrorBoundJoules"])
    if radius != expected_radius:
        raise ValueError("Temporal uncertainty differs from validated contact and cable fixed-motion certificates")
    upper, lower = abs(nominal)+radius, max(F(), abs(nominal)-radius)
    outcome = "within-budget" if upper <= allocation else "exceeds-budget" if lower > allocation else "uncertainty-overlap"
    return {"exactStoredKineticChangeJoules": rational(delta), "nominalDefectJoules": rational(nominal),
            "knownErrorBoundJoules": rational(radius), "absoluteUpperBoundJoules": rational(upper),
            "allocationJoules": rational(allocation), "outcome": outcome}


def run_trials(evaluate, positions, velocities, mass, dt, *, declaration, max_depth,
               max_attempts, initial_subdivisions, evaluation_limit):
    """Private orchestration. No external callbacks or persistent transaction API.

    Each solve reserves its entire unchanged nonlinear evaluation allowance,
    including failures with no result. Reported optimizer counts are separate
    and do not count every constitutive/guard helper call. Existing per-trial
    primitive limits and an external process budget remain necessary.
    """
    declaration = policy(declaration)
    positions, velocities = _state(positions), _state(velocities, positions.shape)
    mass = np.frombuffer(mass.tobytes(), dtype=np.float64)
    attempts, assessments = [], []
    # One immutable-prefix reference is replaced only after constructing a pair.
    committed = (positions, velocities, 0., (), (), F())
    charged = 0
    pending, next_initial = [], 0
    reason = "complete"
    propagated = None

    def trial(q, v, a, b, depth, parent, initial, role):
        nonlocal charged, reason, propagated
        if len(attempts) >= max_attempts:
            reason = "attempt-budget-exhausted"
            return None
        if charged + evaluation_limit > declaration["maxEvaluationBudget"]:
            reason = "evaluation-budget-exhausted"
            return None
        duration = float(dt * (b-a))
        if duration <= 0 or not math.isfinite(duration):
            reason = "substep-duration-underflow"
            return None
        if F(duration) != F(dt)*(F(b)-F(a)):
            reason = "substep-duration-unrepresentable"
            return None
        record = {"attemptId": len(attempts)+1, "parentAttemptId": parent, "initialInterval": initial,
                  "startFraction": a, "endFraction": b, "durationSeconds": duration, "depth": depth,
                  "role": role, "converged": False, "evaluationAllowanceCharged": evaluation_limit}
        record["startStateSha256"] = state_record(q, v)["sha256"]
        charged += evaluation_limit
        output = None
        try:
            nq, nv, valid, fatal, propagate = evaluate(q.copy(), v.copy(), a, b, duration, record)
            record["numericallyValid"], record["fatal"] = bool(valid), bool(fatal)
            record["outcome"] = "provisional" if valid else "interrupted" if propagate else "numerically-rejected"
            if valid:
                nq, nv = _state(nq, positions.shape), _state(nv, positions.shape)
                record["state"] = state_record(nq, nv)
                output = (nq, nv, copy.deepcopy(record))
            if fatal:
                reason = "solver-resource-or-runtime-failure"
            if propagate is not None:
                propagated = propagate
                reason = "interrupted"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            record.update(numericallyValid=False, outcome="numerically-rejected", fatal=False,
                          error={"type": type(error).__name__, "message": str(error)})
        except (TimeoutError, RuntimeError, MemoryError) as error:
            record.update(numericallyValid=False, outcome="interrupted", fatal=True,
                          error={"type": type(error).__name__, "message": str(error)})
            reason = "solver-resource-or-runtime-failure"
        except BaseException as error:
            record.update(numericallyValid=False, outcome="interrupted", fatal=True,
                          error={"type": type(error).__name__, "message": str(error)})
            reason = "interrupted"
            propagated = error
        attempts.append(copy.deepcopy(record))
        return output

    try:
        while True:
            if not pending:
                if next_initial == initial_subdivisions:
                    break
                i = next_initial
                pending.append((i/initial_subdivisions, (i+1)/initial_subdivisions, 0, None, i))
                next_initial += 1
            a, b, depth, parent, initial = pending.pop()
            if depth >= max_depth:
                reason = "temporal-depth-exhausted"
                break
            q, v = committed[:2]
            midpoint = (a+b)/2
            ids_before = len(attempts)
            row = {"assessmentId": len(assessments)+1, "startFraction": a, "endFraction": b,
                   "depth": depth, "trialIds": [], "outcome": "incomplete"}
            coarse = trial(q, v, a, b, depth, parent, initial, "coarse")
            half1 = half2 = None
            if coarse is not None and reason == "complete":
                half1 = trial(q, v, a, midpoint, depth+1, coarse[2]["attemptId"], initial, "fine-first")
            if half1 is not None and reason == "complete":
                half2 = trial(half1[0], half1[1], midpoint, b, depth+1, coarse[2]["attemptId"], initial, "fine-second")
            row["trialIds"] = list(range(ids_before+1, len(attempts)+1))
            valid = coarse is not None and half1 is not None and half2 is not None
            if valid:
                try:
                    row["position"] = discrepancy(coarse[0], half2[0], declaration["positionToleranceM"])
                    row["velocity"] = discrepancy(coarse[1], half2[1], declaration["velocityToleranceMPerS"])
                    budget = F(declaration["numericalEnergyBudgetJ"])
                    row["fineEnergy"] = [energy_defect(mass, start, end[1], end[2]["step"]["energyBalance"],
                                                      budget*(F(right)-F(left)))
                        for start, end, left, right in ((v, half1, a, midpoint), (half1[1], half2, midpoint, b))]
                    valid = (row["position"]["withinThreshold"] and row["velocity"]["withinThreshold"]
                             and all(part["outcome"] == "within-budget" for part in row["fineEnergy"]))
                    row["outcome"] = "committed" if valid else "temporally-rejected"
                except (ValueError, KeyError, FloatingPointError) as error:
                    valid = False
                    row.update(outcome="invalid-temporal-evidence", error={"type": type(error).__name__, "message": str(error)})
            elif reason == "complete":
                row["outcome"] = "numerically-rejected"
            assessments.append(row)
            if reason != "complete":
                break
            if valid:
                fine = []
                for item in (half1, half2):
                    record = copy.deepcopy(item[2])
                    record.update(outcome="committed", transactionId=row["assessmentId"],
                                  completedDurationSeconds=float(dt*record["endFraction"]))
                    fine.append(record)
                increment = sum((_fraction(part["absoluteUpperBoundJoules"]) for part in row["fineEnergy"]), F())
                total = committed[5] + increment
                if total > F(declaration["numericalEnergyBudgetJ"])*F(b):
                    raise RuntimeError("Committed conditional energy budget failed exact prefix accounting")
                transaction = {"transactionId": row["assessmentId"], "fineTrialIds": [item["attemptId"] for item in fine],
                               "startFraction": a, "endFraction": b}
                committed = (half2[0], half2[1], b, committed[3]+tuple(fine),
                             committed[4]+(transaction,), total)
            else:
                if depth+1 >= max_depth:
                    reason = "temporal-depth-exhausted"
                    break
                parent = row["trialIds"][0] if row["trialIds"] else parent
                pending.extend(((midpoint, b, depth+1, parent, initial), (a, midpoint, depth+1, parent, initial)))
    except BaseException as error:
        # Preserve the prior atomic prefix for exceptions outside a solve too.
        # Hard process termination/OOM cannot promise an allocated report.
        propagated = error
        reason = "interrupted"
    q, v, fraction, accepted, transactions, total = committed
    complete = fraction == 1.
    committed_transactions = {item["transactionId"] for item in transactions}
    for assessment in assessments:
        if assessment["assessmentId"] in committed_transactions:
            assessment["outcome"] = "committed"
        elif assessment["outcome"] == "committed":
            assessment["outcome"] = "interrupted-before-commit"
    committed_ids = {record["attemptId"] for record in accepted}
    for record in attempts:
        if record["outcome"] == "provisional":
            record["outcome"] = "committed" if record["attemptId"] in committed_ids else "discarded"
    observed = [record.get("step", {}).get("evaluations") for record in attempts]
    known = [value for value in observed if type(value) is int and 0 <= value <= evaluation_limit]
    report = {"profile": PROFILE, "accepted": False, "complete": complete,
              "reason": "complete" if complete else reason, "policy": declaration,
              "requestedDurationSeconds": float(dt), "completedDurationSeconds": float(dt*fraction),
              "completedFraction": fraction, "initialSubdivisions": initial_subdivisions,
              "initialState": state_record(positions, velocities),
              "maxDepth": max_depth, "maxAttempts": max_attempts, "attempts": attempts,
              "acceptedSteps": list(accepted), "temporalAssessments": assessments,
              "transactions": list(transactions), "conditionalAbsoluteEnergyBoundJoules": rational(total),
              "resources": {"mechanicalTrials": len(attempts), "chargedEvaluationAllowance": charged,
                            "reportedOptimizerEvaluations": sum(known), "trialsWithoutEvaluationCount": len(observed)-len(known),
                            "energyTransitionCalls": sum(record.get("energyTransitionCalls", 0) for record in attempts)},
              "scope": "Conditional local maximum-vertex state indicators and absolute per-fine-interval numerical mechanical-energy budget. Exact reductions of stored values; non-cable work remains numerical. No global error, phase accuracy, continuous actuator work, calibrated damping, source admission, garment acceptance, persistent journal or resume API. Pair commitment is synchronous and in memory. External process limits and fixed per-trial primitive limits remain required."}
    result = (q.copy(), v.copy(), report)
    if propagated is not None:
        propagated.temporal_result = result
        raise propagated
    return result
