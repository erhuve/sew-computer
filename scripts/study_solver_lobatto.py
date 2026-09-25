"""Coupled fourth-order analytic diagnostic; never a native cloth integrator.

All potentials are mechanical energies with an explicit mass. Newton residuals
have force units, but no equivalence to the native solver's stationarity test or
certified force enclosure is claimed. Full primary accepted-stage histories and
failed-trial counters are retained. Decimal precision checks are not intervals.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
import hashlib
import json
from pathlib import Path


METHODS = ("gauss4", "lobatto3a")
CONFIG = {
    "durationSeconds": "0.064", "massKilograms": "1",
    "precisionDigits": [80, 120], "precisionAbsoluteTolerance": "1e-28",
    "positionGoalMetres": "0.000001", "nativeVelocityGoalMPerSecond": "0.00025",
    "decisionGuardAbsolute": "1e-24", "maximumHypotheticalTrials": 4096,
    "newton": {"forceResidualNewtons": "1e-35", "maximumNewtonUpdates": 32,
               "maximumBacktracks": 20, "maximumPotentialCalls": 300,
               "armijoFraction": "0.0001"},
    "polynomial": {"omegaRadPerSecond": "500", "amplitudeMetres": "0.0001",
                   "fineIntervals": [16, 32, 64, 128, 256, 512]},
    "inverse": {"aMetres": ["0.0001", "0.00001"], "bMetresPerSecond": "0.05",
                "fineIntervals": [16, 32, 64, 128, 256, 512, 1024, 2048, 2730]},
    "maximumResultBytes": 67108864,
    "maximumPrimaryCheckpointBytes": 67108864,
}


def identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def encode(value):
    if isinstance(value, D):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Model:
    kind: str
    mass: D
    duration: D
    omega: D = D(0)
    amplitude: D = D(0)
    a: D = D(0)
    b: D = D(0)

    def __post_init__(self):
        if self.kind not in ("polynomial", "inverse"):
            raise ValueError("unknown analytic potential")
        if any(type(v) is not D or not v.is_finite() for v in
               (self.mass, self.duration, self.omega, self.amplitude, self.a, self.b)):
            raise ValueError("finite Decimal parameters required")
        if self.mass <= 0 or self.duration <= 0:
            raise ValueError("positive mass and duration required")
        if self.kind == "inverse" and (self.a <= 0 or self.b <= 0):
            raise ValueError("positive inverse-potential scales required")
        if self.kind == "polynomial" and (self.omega <= 0 or self.amplitude <= 0):
            raise ValueError("positive manufactured scales required")

    def sample(self, x, t):
        """Energy J, gradient N, Hessian N/m, parameter power J/s."""
        if type(x) is not D or type(t) is not D or not x.is_finite() or not t.is_finite():
            raise ValueError("nonfinite analytic state")
        if self.kind == "inverse":
            if x <= 0:
                raise ValueError("inverse-potential domain requires x>0")
            k = self.mass*self.a**2*self.b**2
            return k/(2*x*x), -k/x**3, 3*k/x**4, D(0)
        s = t/self.duration
        f = self.omega**2*self.amplitude*s**5 + 20*self.amplitude*s**3/self.duration**2
        fp = 5*self.omega**2*self.amplitude*s**4/self.duration + 60*self.amplitude*s**2/self.duration**3
        return (self.mass*(self.omega**2*x*x/2-f*x),
                self.mass*(self.omega**2*x-f), self.mass*self.omega**2, -self.mass*fp*x)

    def exact(self, t):
        if type(t) is not D or not t.is_finite():
            raise ValueError("finite Decimal reference time required")
        if self.kind == "inverse":
            x = (self.a*self.a+self.b*self.b*t*t).sqrt()
            return x, self.b*self.b*t/x
        s = t/self.duration
        return self.amplitude*s**5, 5*self.amplitude*s**4/self.duration

    def exact_work(self):
        if self.kind == "inverse":
            return D(0)
        return -self.mass*(self.omega**2*self.amplitude**2/2+D("7.5")*self.amplitude**2/self.duration**2)


def tableau(method):
    if method == "gauss4":
        r = D(3).sqrt()/6
        return (((D(1)/4, D(1)/4-r), (D(1)/4+r, D(1)/4)),
                (D(1)/2-r, D(1)/2+r), (D(1)/2, D(1)/2), (0, 1))
    if method == "lobatto3a":
        return (((D(0), D(0), D(0)), (D(5)/24, D(1)/3, -D(1)/24),
                 (D(1)/6, D(2)/3, D(1)/6)),
                (D(0), D(1)/2, D(1)), (D(1)/6, D(2)/3, D(1)/6), (1, 2))
    raise ValueError("unsupported fourth-order method")


def inverse2(a):
    determinant = a[0][0]*a[1][1]-a[0][1]*a[1][0]
    if not determinant.is_finite() or determinant == 0:
        raise ArithmeticError("singular coupled matrix")
    return ((a[1][1]/determinant, -a[0][1]/determinant),
            (-a[1][0]/determinant, a[0][0]/determinant))


class TrialFailure(Exception):
    def __init__(self, reason, counts, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.counts = dict(counts)
        self.detail = detail


def checked_policy(value=None):
    result = dict(CONFIG["newton"] if value is None else value)
    if result.keys() != CONFIG["newton"].keys():
        raise ValueError("complete fixed-shape Newton policy required")
    for key, maximum in (("maximumNewtonUpdates", 32), ("maximumBacktracks", 20), ("maximumPotentialCalls", 300)):
        if type(result[key]) is not int or not 1 <= result[key] <= maximum:
            raise ValueError("invalid bounded Newton policy")
    residual = D(result["forceResidualNewtons"])
    armijo = D(result["armijoFraction"])
    if not residual.is_finite() or not 0 < residual <= D("1e-35"):
        raise ValueError("diagnostic force threshold may only tighten")
    if not armijo.is_finite() or not 0 < armijo < 1:
        raise ValueError("Armijo fraction must be in (0,1)")
    return result


def trial(method, model, x0, v0, t0, h, *, policy=None):
    """Solve a single coupled trial; never changes the supplied initial state."""
    policy = checked_policy(policy)
    if any(type(v) is not D or not v.is_finite() for v in (x0, v0, t0, h)) or h <= 0:
        raise ValueError("finite Decimal initial state and positive step required")
    a, c, b, unknown = tableau(method)
    n = len(b)
    a2 = tuple(tuple(sum(a[i][k]*a[k][j] for k in range(n)) for j in range(n)) for i in range(n))
    block = tuple(tuple(a2[i][j] for j in unknown) for i in unknown)
    inv = inverse2(block)
    known = tuple(i for i in range(n) if i not in unknown)
    times = tuple(t0+h*ci for ci in c)
    predicted = tuple(x0+h*c[i]*v0 for i in unknown)
    scale = model.mass/(h*h)
    counts = {"potentialCalls": 0, "residualEvaluations": 0, "linearSolves": 0,
              "lineSearchTrials": 0, "domainRejectedCandidates": 0,
              "residualRejectedCandidates": 0, "budgetDeniedCalls": 0,
              "newtonUpdates": 0, "initialStateCalls": 0, "freshStageCalls": 0,
              "freshResidualValidations": 0, "freshEndpointCalls": 0}

    def sample(x, t, purpose=None):
        if counts["potentialCalls"] >= policy["maximumPotentialCalls"]:
            counts["budgetDeniedCalls"] += 1
            raise TrialFailure("potential-call-budget", counts)
        counts["potentialCalls"] += 1
        if purpose:
            counts[purpose] += 1
        return model.sample(x, t)

    try:
        # Known initial stage, when present, is fresh within this physical trial.
        initial_sample = sample(x0, t0, "initialStateCalls")
        known_samples = {i: initial_sample for i in known}

        def evaluate(positions):
            counts["residualEvaluations"] += 1
            samples = dict(known_samples)
            for j, x in zip(unknown, positions):
                samples[j] = sample(x, times[j])
            forcing_known = tuple(sum(a2[i][j]*samples[j][1] for j in known) for i in unknown)
            residual = tuple(scale*sum(inv[k][j]*(positions[j]-predicted[j]) for j in range(2))
                             + samples[unknown[k]][1]
                             + sum(inv[k][j]*forcing_known[j] for j in range(2)) for k in range(2))
            jacobian = tuple(tuple(scale*inv[i][j] + (samples[unknown[i]][2] if i == j else 0)
                                   for j in range(2)) for i in range(2))
            return residual, jacobian, samples

        positions = predicted
        current = evaluate(positions)
        threshold = D(policy["forceResidualNewtons"])
        armijo = D(policy["armijoFraction"])
        while max(map(abs, current[0])) > threshold:
            if counts["newtonUpdates"] >= policy["maximumNewtonUpdates"]:
                raise TrialFailure("newton-update-budget", counts, encode(current[0]))
            counts["linearSolves"] += 1
            ji = inverse2(current[1])
            direction = tuple(-sum(ji[i][j]*current[0][j] for j in range(2)) for i in range(2))
            old_norm = max(map(abs, current[0]))
            accepted = False
            amount = D(1)
            for _ in range(policy["maximumBacktracks"]):
                counts["lineSearchTrials"] += 1
                candidate = tuple(positions[i]+amount*direction[i] for i in range(2))
                try:
                    candidate_state = evaluate(candidate)
                except ValueError:
                    counts["domainRejectedCandidates"] += 1
                else:
                    if max(map(abs, candidate_state[0])) <= (1-armijo*amount)*old_norm:
                        positions, current = candidate, candidate_state
                        counts["newtonUpdates"] += 1
                        accepted = True
                        break
                    counts["residualRejectedCandidates"] += 1
                amount /= 2
            if not accepted:
                raise TrialFailure("backtracking-budget", counts, encode(current[0]))

        # Reconstruct every stage with fresh mechanical samples for publication.
        stages_x = tuple(x0 if i in known else positions[unknown.index(i)] for i in range(n))
        samples = tuple(sample(x, t, "freshStageCalls") for x, t in zip(stages_x, times))
        gradients = tuple(s[1] for s in samples)
        counts["freshResidualValidations"] += 1
        forcing_known = tuple(sum(a2[i][j]*gradients[j] for j in known) for i in unknown)
        force_residual = tuple(scale*sum(inv[k][j]*(positions[j]-predicted[j]) for j in range(2))
                               + gradients[unknown[k]]
                               + sum(inv[k][j]*forcing_known[j] for j in range(2)) for k in range(2))
        if max(map(abs, force_residual)) > threshold:
            raise TrialFailure("fresh-force-residual", counts, encode(force_residual))
        stages_v = tuple(v0-h/model.mass*sum(a[i][j]*gradients[j] for j in range(n)) for i in range(n))
        formula_x = x0+h*sum(b[i]*stages_v[i] for i in range(n))
        formula_v = v0-h/model.mass*sum(b[i]*gradients[i] for i in range(n))
        x1, v1 = (stages_x[-1], stages_v[-1]) if method == "lobatto3a" else (formula_x, formula_v)
        final = sample(x1, t0+h, "freshEndpointCalls")
        stage_x_residual = tuple(stages_x[i]-x0-h*sum(a[i][j]*stages_v[j] for j in range(n)) for i in range(n))
        stage_v_residual = tuple(stages_v[i]-v0+h/model.mass*sum(a[i][j]*gradients[j] for j in range(n)) for i in range(n))
        return {"xMetres": x1, "nativeVelocityMPerSecond": v1,
                "stagePositionsMetres": stages_x, "stageVelocitiesMPerSecond": stages_v,
                "stageTimesSeconds": times, "stageGradientsNewtons": gradients,
                "forceResidualNewtons": force_residual,
                "stagePositionResidualMetres": stage_x_residual,
                "stageVelocityResidualMPerSecond": stage_v_residual,
                "endpointPositionFormulaDefectMetres": x1-formula_x,
                "endpointVelocityFormulaDefectMPerSecond": v1-formula_v,
                "parameterWorkJoules": h*sum(b[i]*samples[i][3] for i in range(n)),
                "mechanicalEnergyJoules": model.mass*v1*v1/2+final[0],
                "counts": dict(counts)}
    except TrialFailure:
        raise
    except (ValueError, ArithmeticError) as failure:
        raise TrialFailure("invalid-or-singular-trial", counts, str(failure)) from failure


def diagnostic_pass(value, target):
    if abs(value-target) <= D(CONFIG["decisionGuardAbsolute"]):
        return None
    return value < target


def compare_history(actual, expected, comparison):
    """Compare same-stage values; counters are reported separately from physics."""
    if actual.keys() != expected.keys():
        comparison["sameHistorySchema"] = False
        return
    for key, value in actual.items():
        old = expected[key]
        if key == "counts":
            if value != old:
                comparison["stepsWithDifferentCounters"] += 1
            continue
        values = value if isinstance(value, list) else [value]
        prior = old if isinstance(old, list) else [old]
        if len(values) != len(prior):
            comparison["sameHistorySchema"] = False
            continue
        for index, (item, original) in enumerate(zip(values, prior)):
            difference = abs(D(item)-D(original))
            if difference > comparison["maximumAbsoluteDifference"]:
                comparison["maximumAbsoluteDifference"] = difference
                comparison["maximumDifferenceWitness"] = {"field": key, "element": index,
                    "actual": str(item), "expected": str(original)}
            comparison["numericFieldsCompared"] += 1


def trajectory(method, model, intervals, *, expected=None):
    if type(intervals) is not int or not 2 <= intervals <= 2730 or intervals % 2:
        raise ValueError("bounded positive even interval count required")
    h = model.duration/intervals
    x, v = model.exact(D(0))
    initial_energy = model.mass*model.b**2/2 if model.kind == "inverse" else D(0)
    work = D(0)
    max_x = max_v = max_energy = max_force = D(0)
    position_peak_index = velocity_peak_index = 0
    counts = {}
    history = []
    completed = 0
    failed = None
    comparison = {"maximumAbsoluteDifference": D(0), "numericFieldsCompared": 0,
                  "stepsWithDifferentCounters": 0, "sameHistorySchema": True,
                  "unmatchedSuccessfulSteps": []}
    for k in range(intervals):
        try:
            step = trial(method, model, x, v, h*k, h)
        except TrialFailure as failure:
            failed = {"intervalIndex": k+1, "reason": failure.reason, "detail": failure.detail,
                      "counts": failure.counts, "uncommitted": True}
            for key, value in failure.counts.items():
                counts[key] = counts.get(key, 0)+value
            break
        x, v = step["xMetres"], step["nativeVelocityMPerSecond"]
        completed += 1
        work += step["parameterWorkJoules"]
        exact_x, exact_v = model.exact(h*(k+1))
        error_x, error_v = abs(x-exact_x), abs(v-exact_v)
        if error_x > max_x:
            max_x, position_peak_index = error_x, k+1
        if error_v > max_v:
            max_v, velocity_peak_index = error_v, k+1
        max_energy = max(max_energy, abs(step["mechanicalEnergyJoules"]-initial_energy-work))
        max_force = max(max_force, *map(abs, step["forceResidualNewtons"]))
        for key, value in step["counts"].items():
            counts[key] = counts.get(key, 0)+value
        encoded_step = encode(step)
        if expected is None:
            history.append(encoded_step)
        elif k >= len(expected["history"]):
            comparison["unmatchedSuccessfulSteps"].append({"intervalIndex": k+1, "record": encoded_step})
        else:
            compare_history(encoded_step, expected["history"][k], comparison)
    complete = completed == intervals
    row = {"method": method, "model": encode(model.__dict__), "fineIntervals": intervals,
           "stepSeconds": str(h), "completedIntervals": completed, "complete": complete,
           "committedTimeSeconds": str(h*completed), "failedTrial": failed,
           "hypotheticalCoarseFineTrialsIfComplete": 3*intervals//2,
           "withinHypotheticalTrialBudget": 3*intervals//2 <= CONFIG["maximumHypotheticalTrials"],
           "maximumPositionErrorMetres": str(max_x), "maximumNativeVelocityErrorMPerSecond": str(max_v),
           "positionPeakIndex": position_peak_index, "velocityPeakIndex": velocity_peak_index,
           "maximumMechanicalEnergyBalanceDefectJoules": str(max_energy),
           "maximumStageForceResidualNewtons": str(max_force),
           "parameterWorkJoules": str(work),
           "exactFullDurationParameterWorkJoules": str(model.exact_work()),
           "fullDurationWorkErrorJoules": str(abs(work-model.exact_work())) if complete else None,
           "positionGoalPass": diagnostic_pass(max_x, D(CONFIG["positionGoalMetres"])) if complete else False,
           "nativeVelocityGoalPass": diagnostic_pass(max_v, D(CONFIG["nativeVelocityGoalMPerSecond"])) if complete else False,
           "counts": counts}
    if expected is None:
        row["history"] = history
    else:
        comparison["sameCompletedPrefix"] = completed == expected["completedIntervals"] and complete == expected["complete"]
        comparison["sameDecisions"] = all(row[key] == expected[key] for key in ("positionGoalPass", "nativeVelocityGoalPass"))
        prior_failure = expected["failedTrial"]
        comparison["sameFailureReason"] = ((failed is None and prior_failure is None)
            or (failed is not None and prior_failure is not None and failed["reason"] == prior_failure["reason"]))
        row["historyComparison"] = encode(comparison)
    return row


def study(precision, expected=None):
    with localcontext() as ctx:
        ctx.prec = precision
        models = [Model("polynomial", D(CONFIG["massKilograms"]), D(CONFIG["durationSeconds"]),
                        omega=D(CONFIG["polynomial"]["omegaRadPerSecond"]), amplitude=D(CONFIG["polynomial"]["amplitudeMetres"]))]
        models += [Model("inverse", D(CONFIG["massKilograms"]), D(CONFIG["durationSeconds"]), a=D(a), b=D(CONFIG["inverse"]["bMetresPerSecond"])) for a in CONFIG["inverse"]["aMetres"]]
        rows = []
        for model in models:
            grid = CONFIG[model.kind]["fineIntervals"]
            for method in METHODS:
                for n in grid:
                    previous = None if expected is None else expected[len(rows)]
                    rows.append(trajectory(method, model, n, expected=previous))
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    declaration = json.loads(args.declaration.read_text())
    if declaration.get("configuration") != CONFIG or declaration.get("source") != identity(__file__):
        raise ValueError("source or fixed prospective configuration changed")
    primary = study(CONFIG["precisionDigits"][0])
    # Retain the completed primary histories before the comparison can fail or
    # be stopped. This checkpoint is not a resume promise or verified result.
    checkpoint_path = args.output.with_name(args.output.stem+"-primary.json")
    checkpoint = {"schema": "sew-coupled-primary-checkpoint-v1", "configuration": CONFIG,
                  "source": identity(__file__), "declaration": identity(args.declaration),
                  "precisionDigits": CONFIG["precisionDigits"][0], "primary": primary}
    checkpoint_bytes = (json.dumps(checkpoint, sort_keys=True, separators=(",", ":"))+"\n").encode()
    if len(checkpoint_bytes) > CONFIG["maximumPrimaryCheckpointBytes"]:
        raise ValueError("primary checkpoint exceeds declared byte limit")
    with checkpoint_path.open("xb") as output:
        output.write(checkpoint_bytes)
    print(json.dumps({"primaryCheckpoint": str(checkpoint_path), **identity(checkpoint_path)}), flush=True)
    repeated = study(CONFIG["precisionDigits"][1], primary)
    maximum = max(D(row["historyComparison"]["maximumAbsoluteDifference"]) for row in repeated)
    precision = {"maximumAbsoluteDifferenceAcrossStoredNumericFields": str(maximum),
                 "numericFieldsCompared": sum(row["historyComparison"]["numericFieldsCompared"] for row in repeated),
                 "stepsWithDifferentCounters": sum(row["historyComparison"]["stepsWithDifferentCounters"] for row in repeated),
                 "withinDeclaredTolerance": maximum <= D(CONFIG["precisionAbsoluteTolerance"]),
                 "samePrefixesAndDecisions": all(row["historyComparison"][key] for row in repeated
                    for key in ("sameCompletedPrefix", "sameDecisions", "sameFailureReason", "sameHistorySchema")),
                 "certifiedIntervalArithmetic": False}
    result = {"schema": "sew-coupled-fourth-order-diagnostic-v1", "configuration": CONFIG,
              "source": identity(__file__), "declaration": identity(args.declaration),
              "scope": "Smooth analytic controls only; no native cloth motion, physical contact, nonlinear native cost or solver acceptance.",
              "primaryCheckpoint": {"path": str(checkpoint_path), **identity(checkpoint_path)},
              "precisionComparison": precision, "primary": primary, "repeatedSummaries": repeated}
    encoded = (json.dumps(result, sort_keys=True, separators=(",", ":"))+"\n").encode()
    if len(encoded) > CONFIG["maximumResultBytes"]:
        raise ValueError("result exceeds declared byte limit")
    with args.output.open("xb") as output:
        output.write(encoded)
    print(json.dumps({"output": str(args.output), **identity(args.output), "rows": len(primary),
                      "completeRows": sum(row["complete"] for row in primary), "precisionComparison": precision}))
    return 0 if precision["withinDeclaredTolerance"] and precision["samePrefixesAndDecisions"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
