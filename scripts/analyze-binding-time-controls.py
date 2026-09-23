"""Compare fixed-timestep first-turn controls under a declared fourfold dilation.

The current first-turn reporter admits each zero/driven pair and observes every
saved state. This layer checks the one-factor comparison and reports fixed
windows, without declaring convergence, settling or construction acceptance.
"""

import argparse
import copy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path

from solver_process_budget import atomic_bytes, read_regular


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("binding_time_reference", SCRIPTS / "analyze-binding-first-turn.py")
FIRST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIRST)
require, encoded, digest = FIRST.require, FIRST.encoded, FIRST.digest
TIMING_ARGUMENTS = {"step_seconds", "subdivisions"}
BUDGET_ARGUMENTS = {"max_attempts", "cpu_limit_seconds", "wall_limit_seconds"}


def time_independent_source(source):
    """Remove only admitted time settings and the changed generating entrypoint."""
    metadata = source["bindingFirstTurnDiagnostic"]
    FIRST.validate_time_policy(metadata)
    common = copy.deepcopy(source)
    common["sewingActuation"].pop("sourceSha256")
    declaration = common["bindingFirstTurnDiagnostic"]
    declaration.pop("timePolicy", None)
    declaration.pop("subdivisions")
    # The recorded generator necessarily changes to expose the new policy.
    # Every dependency, runtime, actual target and source field stays compared.
    declaration["codeDigests"].pop("scripts/prepare-binding-first-turn.py")
    return digest(encoded(common))


def validate_time_pair(original, slower, original_source, slower_source):
    """Require identical physical inputs and exact nominal grids across durations."""
    for record, source, identity, subdivisions in (
            (original, original_source, "original-128ms-v1", 64),
            (slower, slower_source, "fourfold-512ms-v1", 256)):
        policy = FIRST.validate_time_policy(source["bindingFirstTurnDiagnostic"], record["physicsArguments"])
        require(policy["id"] == identity and encoded(record["timePolicy"]) == encoded(policy),
                "Original and fourfold controls must use their declared time policies")
        require(record["nominalTimeStepPreserved"] is True and len(record["states"]) == subdivisions + 1,
                "Time-dilation comparison requires the complete unchanged two-millisecond grid")
        for index, state in enumerate(record["states"]):
            fields = ("fraction", "timeSeconds") + (("stepDurationSeconds",) if index else ())
            require(all(type(state[field]) in (int, float) and math.isfinite(state[field]) for field in fields),
                    "Saved fractions, times and timesteps must be finite numbers, not Boolean values")
            require(state["fraction"] == index / subdivisions
                    and state["timeSeconds"] == state["fraction"] * policy["durationSeconds"]
                    and (index == 0 or state["stepDurationSeconds"] == .002),
                    "Saved fractions, times or actual timesteps differ from the declared grid")
    original_common = time_independent_source(original_source)
    require(original_common == time_independent_source(slower_source),
            "Time controls change source geometry, target path or a non-timing declaration")
    for key in ("angleDegrees", "sourceUnitSha256", "initialPositionsSha256", "sewingControls",
                "gripperAnchors", "gripperActivationSchedule", "initialGripperTargets",
                "sourceDigests", "sewingPathToleranceMeters"):
        require(encoded(original[key]) == encoded(slower[key]), "Time controls differ in " + key)
    physical = [{key: value for key, value in record["physicsArguments"].items()
                 if key not in TIMING_ARGUMENTS | BUDGET_ARGUMENTS} for record in (original, slower)]
    require(encoded(physical[0]) == encoded(physical[1]), "Time controls change other numerical/physical settings")
    return {"timeIndependentSourceSha256": original_common,
        "originalCanonicalSha256": original["canonicalSha256"], "slowerCanonicalSha256": slower["canonicalSha256"],
        "declaredTiming": [record["timePolicy"] for record in (original, slower)],
        "supervisionArguments": [{key: record["physicsArguments"][key] for key in sorted(BUDGET_ARGUMENTS)}
                                 for record in (original, slower)],
        "generatorSha256": [source["bindingFirstTurnDiagnostic"]["codeDigests"]["scripts/prepare-binding-first-turn.py"]
                            for source in (original_source, slower_source)],
        "scope": "Only declared duration/subdivision and explicitly reported supervision/generator identity may differ; actual source geometry, placements, 12-knot control recipes and other numerical settings match."}


def summarize(record):
    """Saved-state metrics with a fixed observation horizon, never a stop rule."""
    states = record["states"]
    command = record["angleDegrees"]

    def angles(state):
        return [row["relativeTurnDegrees"] for row in state["bindingGeometry"]["rows"]]

    def window(lower, upper):
        selected = [state for state in states if lower <= Fraction(state["fraction"]) <= upper]
        require(selected and Fraction(selected[0]["fraction"]) == lower
                and Fraction(selected[-1]["fraction"]) == upper, "Exact saved window endpoints required")
        by_row = list(zip(*(angles(state) for state in selected)))
        return {"startFraction": float(lower), "endFraction": float(upper),
            "startTimeSeconds": selected[0]["timeSeconds"], "endTimeSeconds": selected[-1]["timeSeconds"],
            "savedStates": len(selected),
            "maximumAbsoluteCommandErrorDegrees": max(abs(value - command) for row in by_row for value in row),
            "maximumSameRowAngleRangeDegrees": max(max(row) - min(row) for row in by_row),
            "maximumSameRowEndpointChangeDegrees": max(abs(row[-1] - row[0]) for row in by_row),
            "maximumSpeedMetersPerSecond": max(state["maximumSpeedMetersPerSecond"] for state in selected),
            "maximumGripperTrackingErrorMeters": max(grip["trackingErrorMeters"] for state in selected
                                                       for grip in state["gripperGeometry"]["grippers"])}

    duration = record["timePolicy"]["durationSeconds"]
    last_window_start = 1 - Fraction(.016) / Fraction(duration)
    require(last_window_start in (Fraction(7, 8), Fraction(31, 32)), "Declared fixed sixteen-millisecond window required")
    rows = [row for state in states for row in state["bindingGeometry"]["rows"]]
    strains = [item["current"] for state in states for item in state["bindingGeometry"]["strainByInstance"].values()]
    peak = max((value, state["fraction"], row) for state in states for row, value in enumerate(angles(state)))
    minimum = min(states, key=lambda state: state["surfaceSeparation"]["distanceMeters"])
    final = states[-1]
    sleeve_fits = [state["bindingGeometry"]["rigidFits"][record["declaration"]["sleeveInstanceId"]] for state in states]
    return {"directory": record["directory"], "accepted": False, "commandDegrees": command,
        "timePolicy": record["timePolicy"], "savedStates": len(states),
        "reportSha256": record["reportSha256"], "canonicalSha256": record["canonicalSha256"],
        "replaySha256": record["replaySha256"], "exactContactLeaves": record["exactContactLeaves"],
        "sewingPathToleranceMeters": record["sewingPathToleranceMeters"],
        "peakRelativeTurn": {"degrees": peak[0], "fraction": peak[1], "rowIndex": peak[2]},
        "finalRelativeTurnsDegrees": angles(final),
        "finalRelativeRigidFitDegrees": final["bindingGeometry"]["relativeRigidMotion"]["rotationDegrees"],
        "finalMaximumSpeedMetersPerSecond": final["maximumSpeedMetersPerSecond"],
        "minimumSavedSurfaceGap": {"meters": minimum["surfaceSeparation"]["distanceMeters"],
                                   "fraction": minimum["fraction"]},
        "maximumSampledDistanceErrorMeters": max(row["current"]["absoluteDistanceErrorMeters"] for row in rows),
        "maximumAbsoluteTangentOffsetMeters": max(abs(row["current"]["tangentOffsetMeters"]) for row in rows),
        "maximumAbsoluteCrossTangentOffsetMeters": max(abs(row["current"]["crossTangentOffsetMeters"]) for row in rows),
        "principalStretchMinimum": min(item["principalStretchMinimum"] for item in strains),
        "principalStretchMaximum": max(item["principalStretchMaximum"] for item in strains),
        "maximumSleeveFitRotationDegrees": max(fit["rotationDegrees"] for fit in sleeve_fits),
        "maximumSleeveFitTranslationMeters": max(math.hypot(*fit["translationMeters"]) for fit in sleeve_fits),
        "maximumSleeveFitResidualMeters": max(fit["maximumResidualMeters"] for fit in sleeve_fits),
        "maximumSavedContactEnergyJoules": max(state.get("contactEnergyJoules", 0.) for state in states),
        "allTime": window(Fraction(0), Fraction(1)), "hold": window(Fraction(1, 2), Fraction(3, 4)),
        "passiveTail": window(Fraction(7, 8), Fraction(1)),
        "finalSixteenMilliseconds": window(last_window_start, Fraction(1)),
        "sewingWorkSummary": record["sewingWorkSummary"], "gripperWorkSummary": record["gripperWorkSummary"],
        "scope": "Saved-state ranges and exact saved window endpoints; no continuous motion extrema or settled-pose threshold."}


def compare(original_control, original_driven, slower_control, slower_driven):
    original = FIRST.compare(original_control, original_driven)
    slower = FIRST.compare(slower_control, slower_driven)
    identities = {}
    for role, old_path, slow_path in (("control", original_control, slower_control),
                                     ("driven", original_driven, slower_driven)):
        old_source, old_digest = FIRST.document(Path(old_path) / "canonical.json")
        slow_source, slow_digest = FIRST.document(Path(slow_path) / "canonical.json")
        require(old_digest == original[role]["canonicalSha256"] and slow_digest == slower[role]["canonicalSha256"],
                "Canonical source changed after observation")
        identities[role] = validate_time_pair(original[role], slower[role], old_source, slow_source)
    return {"profile": "source-binding-fourfold-time-control-v1", "accepted": False,
        "scope": "Predeclared fourfold duration comparison at unchanged nominal two-millisecond steps, without easing, stiffness or source changes.",
        "identities": identities,
        "summaries": {name: {role: summarize(pair[role]) for role in ("control", "driven")}
                      for name, pair in (("original", original), ("fourfold", slower))},
        "original": original, "fourfold": slower,
        "codeDigests": {**original["codeDigests"], "analyze-binding-time-controls.py": digest(read_regular(Path(__file__)))},
        "limitations": ["Longer evolution permits more existing backward-Euler numerical dissipation; it adds no calibrated physical damping.",
            "Both trajectories retain piecewise-linear target chords and their start/stop velocity discontinuities, with fourfold lower prescribed velocities.",
            "Equal fractions across durations are equal control progress, not equal elapsed physical time.",
            "A lower terminal speed may be an oscillation phase; hold, whole passive tail and final fixed sixteen-millisecond windows remain separate.",
            "No accepted settled pose, binding wrap, source phase, spatial stitching, temporal/spatial convergence, calibrated material or full-garment result."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("original-control", "original-driven", "slower-control", "slower-driven", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.original_control, args.original_driven, args.slower_control, args.slower_driven)
    atomic_bytes(args.output, json.dumps(result, indent=2, allow_nan=False).encode() + b"\n")
    print(json.dumps({"accepted": False, "summaries": result["summaries"]}))


if __name__ == "__main__":
    main()
