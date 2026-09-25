"""Bounded analytic time-integrator comparison; never a cloth solver or gate.

The autonomous maps are rational functions, evaluated with Decimal arithmetic.
Reference rotations use a Taylor series with argument halving. Two declared
precisions are compared; this is not an interval-arithmetic error certificate.
The forced example solves the coupled Runge--Kutta stages directly and has an
independent polynomial exact solution and continuous parameter-work integral.
"""
from __future__ import annotations

import argparse
from decimal import Decimal as D, localcontext, getcontext
import hashlib
import json
from pathlib import Path


METHODS = ("midpoint", "gauss4")
CONFIG = {
    "durationSeconds": "0.064",
    "frequenciesRadPerSecond": ["50", "5000", "85991"],
    "syntheticVelocityAmplitudeMPerSecond": "0.010",
    "fineIntervals": [128, 256, 512, 1024, 2048, 2730, 4096, 8192, 16384, 32768],
    "positionGoalMetres": "0.000001",
    "nativeVelocityGoalMPerSecond": "0.00025",
    "localPositionIndicatorMetres": "0.0000001",
    "localVelocityIndicatorMPerSecond": "0.0001",
    "maximumTrials": 4096,
    "precisionDigits": [80, 120],
    "precisionAgreementAbsolute": "1e-55",
    "decisionGuardAbsolute": "1e-45",
    "referenceClosureAbsolute": "1e-65",
    "forced": {
        "frequencyRadPerSecond": "500",
        "finalPositionMetres": "0.0001",
        "unitMassKilograms": "1",
        "fineIntervals": [16, 32, 64, 128, 256, 512],
        "polynomialDegree": 5,
    },
}


def identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def multiply(z, w):
    a, b = z
    c, d = w
    return a*c-b*d, a*d+b*c


def sincos(angle):
    """Return (cos, sin); no pi constant or phase-unwrapping decision."""
    precision = getcontext().prec
    with localcontext() as ctx:
        ctx.prec = precision + 16
        x = +angle
        halvings = 0
        while abs(x) > D("0.25"):
            x /= 2
            halvings += 1
            if halvings > 32:
                raise ValueError("angle outside declared diagnostic range")
        cosine, sine = D(1), x
        cosine_term, sine_term = D(1), x
        threshold = D(10) ** (-ctx.prec - 2)
        for k in range(1, 256):
            cosine_term *= -x*x / ((2*k-1)*(2*k))
            sine_term *= -x*x / ((2*k)*(2*k+1))
            cosine += cosine_term
            sine += sine_term
            if max(abs(cosine_term), abs(sine_term)) < threshold:
                break
        else:
            raise ArithmeticError("Taylor series did not terminate")
        for _ in range(halvings):
            cosine, sine = cosine*cosine-sine*sine, 2*cosine*sine
    return +cosine, +sine


def rotation(method, z):
    if method not in METHODS:
        raise ValueError("unsupported method")
    real = D(1) if method == "midpoint" else 1-z*z/12
    imag = z/2
    denominator = real*real+imag*imag
    return (real*real-imag*imag)/denominator, 2*real*imag/denominator


def distance_squared(z, w):
    return (z[0]-w[0])**2 + (z[1]-w[1])**2


def diagnostic_pass(values_and_goals):
    """Near-threshold values stay unresolved; this is not a rigorous bound."""
    statuses = [None if abs(value-goal) <= D(CONFIG["decisionGuardAbsolute"])
                else value < goal for value, goal in values_and_goals]
    return False if False in statuses else None if None in statuses else True


def oscillator(method, omega, intervals):
    if type(intervals) is not int or intervals <= 0 or intervals % 2 or intervals > max(CONFIG["fineIntervals"]):
        raise ValueError("oscillator grid must be a bounded positive even integer")
    if not isinstance(omega, D) or not omega.is_finite() or omega <= 0:
        raise ValueError("frequency must be a finite positive Decimal")
    duration = D(CONFIG["durationSeconds"])
    amplitude = D(CONFIG["syntheticVelocityAmplitudeMPerSecond"])
    h = duration / intervals
    step = rotation(method, omega*h)
    reference_step = sincos(omega*h)
    candidate = reference = (D(1), D(0))
    peak_squared = D(0)
    peak_index = 0
    scalar_x = scalar_v = energy_drift = D(0)
    for k in range(1, intervals+1):
        candidate = multiply(candidate, step)
        reference = multiply(reference, reference_step)
        error_squared = distance_squared(candidate, reference)
        if error_squared > peak_squared:
            peak_squared, peak_index = error_squared, k
        scalar_x = max(scalar_x, abs(candidate[1]-reference[1])*amplitude/omega)
        scalar_v = max(scalar_v, abs(candidate[0]-reference[0])*amplitude)
        energy_drift = max(energy_drift, abs(candidate[0]**2+candidate[1]**2-1))
    velocity_error = amplitude*peak_squared.sqrt()
    position_error = velocity_error/omega
    reference_closure = distance_squared(reference, sincos(omega*duration)).sqrt()
    if reference_closure > D(CONFIG["referenceClosureAbsolute"]):
        raise ArithmeticError("reference recurrence exceeds declared closure tolerance")
    # One coarse trial spans TWO accepted fine intervals. No Richardson scaling.
    local_difference = distance_squared(rotation(method, 2*omega*h), multiply(step, step)).sqrt()
    local_v = amplitude*local_difference
    local_x = local_v/omega
    trials = 3*(intervals//2)
    return {
        "method": method, "frequencyRadPerSecond": str(omega),
        "fineIntervals": intervals, "timeStepSeconds": str(h),
        "hypotheticalDoublingTransactions": intervals//2,
        "hypotheticalPhysicalTrials": trials,
        "minimumStageForceEvaluations": trials*(1 if method == "midpoint" else 2),
        "coupledPositionUnknownsPerMechanicalDof": 1 if method == "midpoint" else 2,
        "withinTrialBudget": trials <= CONFIG["maximumTrials"],
        "maximumErrorSampleIndex": peak_index,
        "maximumWorstPhasePositionErrorMetres": str(position_error),
        "maximumWorstPhaseNativeVelocityErrorMPerSecond": str(velocity_error),
        "finalWorstPhaseNativeVelocityErrorMPerSecond": str(amplitude*distance_squared(candidate, reference).sqrt()),
        "zeroPositionInitialPhaseMaximumPositionErrorMetres": str(scalar_x),
        "zeroPositionInitialPhaseMaximumVelocityErrorMPerSecond": str(scalar_v),
        "maximumArithmeticRelativeEnergyDefect": str(energy_drift),
        "referenceFinalRotationClosure": str(reference_closure),
        "localDoublingWorstPhasePositionIndicatorMetres": str(local_x),
        "localDoublingWorstPhaseVelocityIndicatorMPerSecond": str(local_v),
        "localIndicatorsPass": diagnostic_pass((
            (local_x, D(CONFIG["localPositionIndicatorMetres"])),
            (local_v, D(CONFIG["localVelocityIndicatorMPerSecond"])),
        )),
        "sampledAccuracyPass": diagnostic_pass((
            (position_error, D(CONFIG["positionGoalMetres"])),
            (velocity_error, D(CONFIG["nativeVelocityGoalMPerSecond"])),
        )),
    }


def polynomial_motion(t, duration, amplitude):
    s = t/duration
    return amplitude*s**5, 5*amplitude*s**4/duration


def forcing(t, omega, duration, amplitude):
    s = t/duration
    return omega**2*amplitude*s**5 + 20*amplitude*s**3/duration**2


def forcing_derivative(t, omega, duration, amplitude):
    s = t/duration
    return 5*omega**2*amplitude*s**4/duration + 60*amplitude*s**2/duration**3


def tableau(method):
    if method == "midpoint":
        return ((D("0.5"),),), (D("0.5"),), (D(1),)
    if method != "gauss4":
        raise ValueError("unsupported method")
    r = D(3).sqrt()/6
    return ((D("0.25"), D("0.25")-r),
            (D("0.25")+r, D("0.25"))), (D("0.5")-r, D("0.5")+r), (D("0.5"), D("0.5"))


def forced_step(method, x0, v0, t0, h, omega, duration, amplitude):
    """Direct coupled linear stages, with unit mass; not nonlinear cloth work."""
    a, c, b = tableau(method)
    count = len(b)
    a2 = tuple(tuple(sum(a[i][k]*a[k][j] for k in range(count)) for j in range(count)) for i in range(count))
    times = tuple(t0+ci*h for ci in c)
    loads = tuple(forcing(t, omega, duration, amplitude) for t in times)
    matrix = tuple(tuple(D(i == j)+h*h*omega*omega*a2[i][j] for j in range(count)) for i in range(count))
    rhs = tuple(x0+h*c[i]*v0+h*h*sum(a2[i][j]*loads[j] for j in range(count)) for i in range(count))
    if count == 1:
        positions = (rhs[0]/matrix[0][0],)
    else:
        determinant = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
        positions = ((rhs[0]*matrix[1][1]-matrix[0][1]*rhs[1])/determinant,
                     (matrix[0][0]*rhs[1]-rhs[0]*matrix[1][0])/determinant)
    accelerations = tuple(-omega*omega*positions[i]+loads[i] for i in range(count))
    velocities = tuple(v0+h*sum(a[i][j]*accelerations[j] for j in range(count)) for i in range(count))
    x1 = x0+h*sum(b[i]*velocities[i] for i in range(count))
    v1 = v0+h*sum(b[i]*accelerations[i] for i in range(count))
    stage_x_defect = max(abs(positions[i]-x0-h*sum(a[i][j]*velocities[j] for j in range(count))) for i in range(count))
    stage_v_defect = max(abs(velocities[i]-v0-h*sum(a[i][j]*accelerations[j] for j in range(count))) for i in range(count))
    work = -h*sum(b[i]*forcing_derivative(times[i], omega, duration, amplitude)*positions[i] for i in range(count))
    split_work = None
    if method == "midpoint":
        f0 = forcing(t0, omega, duration, amplitude)
        f1 = forcing(t0+h, omega, duration, amplitude)
        split_work = -(loads[0]-f0)*x0-(f1-loads[0])*x1
    return {"x": x1, "v": v1, "stagePositions": positions,
            "stageVelocities": velocities, "stageTimes": times,
            "stagePositionDefect": stage_x_defect, "stageVelocityDefect": stage_v_defect,
            "quadratureWork": work, "symmetricSplitWork": split_work}


def forced(method, intervals):
    configuration = CONFIG["forced"]
    omega = D(configuration["frequencyRadPerSecond"])
    amplitude = D(configuration["finalPositionMetres"])
    duration = D(CONFIG["durationSeconds"])
    h = duration/intervals
    x = v = work = split_work = D(0)
    position_error = velocity_error = stage_x = stage_v = D(0)
    for k in range(intervals):
        step = forced_step(method, x, v, h*k, h, omega, duration, amplitude)
        x, v = step["x"], step["v"]
        work += step["quadratureWork"]
        if step["symmetricSplitWork"] is not None:
            split_work += step["symmetricSplitWork"]
        exact_x, exact_v = polynomial_motion(h*(k+1), duration, amplitude)
        position_error = max(position_error, abs(x-exact_x))
        velocity_error = max(velocity_error, abs(v-exact_v))
        stage_x = max(stage_x, step["stagePositionDefect"])
        stage_v = max(stage_v, step["stageVelocityDefect"])
    exact_work = -omega*omega*amplitude*amplitude/2-D("7.5")*amplitude*amplitude/duration**2
    final_energy = v*v/2+omega*omega*x*x/2-forcing(duration, omega, duration, amplitude)*x
    row = {
        "method": method, "fineIntervals": intervals,
        "maximumPositionErrorMetres": str(position_error),
        "maximumNativeVelocityErrorMPerSecond": str(velocity_error),
        "exactContinuousParameterWorkJoules": str(exact_work),
        "quadratureWorkJoules": str(work), "quadratureWorkErrorJoules": str(abs(work-exact_work)),
        "finalMechanicalEnergyJoules": str(final_energy),
        "quadratureEnergyBalanceDefectJoules": str(final_energy-work),
        "maximumStagePositionDefectMetres": str(stage_x),
        "maximumStageVelocityDefectMPerSecond": str(stage_v),
    }
    if method == "midpoint":
        row.update(symmetricSplitWorkJoules=str(split_work),
                   symmetricSplitWorkErrorJoules=str(abs(split_work-exact_work)),
                   symmetricSplitEnergyBalanceDefectJoules=str(final_energy-split_work))
    return row


def study(precision):
    with localcontext() as ctx:
        ctx.prec = precision
        return {
            "oscillators": [oscillator(method, D(omega), n) for omega in CONFIG["frequenciesRadPerSecond"] for method in METHODS for n in CONFIG["fineIntervals"]],
            "forced": [forced(method, n) for method in METHODS for n in CONFIG["forced"]["fineIntervals"]],
        }


def compare_precision(primary, repeated):
    peak = D(0)
    checked = 0
    for family in ("oscillators", "forced"):
        if len(primary[family]) != len(repeated[family]):
            raise ValueError("precision run row mismatch")
        for low, high in zip(primary[family], repeated[family]):
            if low.keys() != high.keys():
                raise ValueError("precision run field mismatch")
            for key in low:
                value, other = low[key], high[key]
                if isinstance(value, str) and key != "method":
                    difference = abs(D(value)-D(other))
                    peak = max(peak, difference)
                    checked += 1
                elif value != other:
                    raise ArithmeticError("precision changes decision or row identity: "+key)
    if peak > D(CONFIG["precisionAgreementAbsolute"]):
        raise ArithmeticError("precision comparison exceeds declared tolerance")
    return {"comparedNumericFields": checked, "maximumAbsoluteDifference": str(peak),
            "decisionsAndIdentitiesAgree": True, "certifiedIntervalArithmetic": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    declaration = json.loads(args.declaration.read_text())
    if declaration.get("configuration") != CONFIG:
        raise ValueError("declaration does not match the fixed prospective configuration")
    if declaration.get("source") != identity(__file__):
        raise ValueError("study source differs from its declaration")
    primary, repeated = (study(precision) for precision in CONFIG["precisionDigits"])
    comparison = compare_precision(primary, repeated)
    result = {"schema": "sew-time-integrator-analytic-study-v1", "configuration": CONFIG,
              "scope": "Synthetic smooth linear diagnostic; no cloth execution, nonlinear timing, path certification, or solver acceptance.",
              "source": identity(__file__), "declaration": identity(args.declaration),
              "precisionComparison": comparison, "results": primary}
    encoded = (json.dumps(result, sort_keys=True, indent=2)+"\n").encode()
    if len(encoded) > 1024*1024:
        raise ValueError("result exceeds prospective byte limit")
    with args.output.open("xb") as output:
        output.write(encoded)
    print(json.dumps({"output": str(args.output), **identity(args.output), "precisionComparison": comparison}))


if __name__ == "__main__":
    main()
