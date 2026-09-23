"""Compare a longer fixed hold with the previous 512 ms binding control.

All four runs must pass current reporter admission. The extended experiment
must reproduce every position and velocity in the initial 384 ms exactly.
This is a control diagnostic, not a material damping or settled-pose claim.
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
SPEC = importlib.util.spec_from_file_location("binding_hold_time_report", SCRIPTS / "analyze-binding-time-controls.py")
TIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIME)
FIRST = TIME.FIRST
encoded, digest, require = FIRST.encoded, FIRST.digest, FIRST.require


def hold_independent_source(source):
    FIRST.validate_time_policy(source["bindingFirstTurnDiagnostic"])
    common = copy.deepcopy(source)
    common["sewingActuation"].pop("sourceSha256")
    declared = common["bindingFirstTurnDiagnostic"]
    declared.pop("timePolicy")
    declared.pop("subdivisions")
    declared["codeDigests"].pop("scripts/prepare-binding-first-turn.py")
    for key in ("turnFractions", "angularSampleFractions", "holdFractions", "releaseFractions", "passiveTailFractions"):
        declared["schedule"].pop(key)
    for knot in common["gripperActuation"]["schedule"]["knots"]:
        knot.pop("fraction")
    return digest(encoded(common))


def validate_hold_pair(previous, held, previous_source, held_source):
    for record, source, identity, count in (
            (previous, previous_source, "fourfold-512ms-v1", 256),
            (held, held_source, "extended-hold-1024ms-v1", 512)):
        policy = FIRST.validate_time_policy(source["bindingFirstTurnDiagnostic"], record["physicsArguments"])
        require(policy["id"] == identity and encoded(record["timePolicy"]) == encoded(policy),
                "Previous and extended-hold time policies required")
        require(record["nominalTimeStepPreserved"] is True and len(record["states"]) == count + 1,
                "Complete two-millisecond control grid required")
        for index, state in enumerate(record["states"]):
            fields = ("fraction", "timeSeconds") + (("stepDurationSeconds",) if index else ())
            require(all(type(state[key]) in (int, float) and math.isfinite(state[key]) for key in fields)
                    and state["fraction"] == index / count
                    and state["timeSeconds"] == state["fraction"] * policy["durationSeconds"]
                    and (index == 0 or state["stepDurationSeconds"] == .002), "Actual saved time grid differs")
        # Bound every normalized-away time field to the actual physical policy.
        expected = [index * .032 for index in range(9)] + ([.384, .448, .512] if count == 256 else [.896, .960, 1.024])
        knots = source["gripperActuation"]["schedule"]["knots"]
        turn_end, hold_end, release_end = (.5, .75, .875) if count == 256 else (.25, .875, .9375)
        fractions = [index * turn_end / 8 for index in range(9)] + [hold_end, release_end, 1.]
        require(encoded([knot["fraction"] for knot in knots]) == encoded(fractions)
                and [knot["fraction"] * policy["durationSeconds"] for knot in knots] == expected,
                "Actual target/release times differ from the fixed-hold experiment")
        phases = {"turnFractions": [0., turn_end], "angularSampleFractions": fractions[:9],
                  "holdFractions": [turn_end, hold_end], "releaseFractions": [hold_end, release_end],
                  "passiveTailFractions": [release_end, 1.]}
        require(all(encoded(source["bindingFirstTurnDiagnostic"]["schedule"][key]) == encoded(value)
                    for key, value in phases.items()), "Declared phase boundaries differ from actual hold policy")
    common = hold_independent_source(previous_source)
    require(common == hold_independent_source(held_source), "Extended hold changes non-timing source/control fields")
    for key in ("angleDegrees", "sourceUnitSha256", "initialPositionsSha256", "sewingControls", "gripperAnchors",
                "initialGripperTargets", "sourceDigests", "sewingPathToleranceMeters"):
        require(encoded(previous[key]) == encoded(held[key]), "Extended hold changes " + key)
    physical = [{key: value for key, value in record["physicsArguments"].items()
                 if key not in TIME.TIMING_ARGUMENTS | TIME.BUDGET_ARGUMENTS} for record in (previous, held)]
    require(encoded(physical[0]) == encoded(physical[1]), "Extended hold changes other numerical/physical settings")
    return {"holdIndependentSourceSha256": common,
        "canonicalSha256": [record["canonicalSha256"] for record in (previous, held)],
        "timing": [record["timePolicy"] for record in (previous, held)],
        "supervisionArguments": [{key: record["physicsArguments"][key] for key in sorted(TIME.BUDGET_ARGUMENTS)}
                                 for record in (previous, held)],
        "generatorSha256": [source["bindingFirstTurnDiagnostic"]["codeDigests"]["scripts/prepare-binding-first-turn.py"]
                            for source in (previous_source, held_source)]}


def verify_prefix(previous_path, held_path, previous, held):
    reports = []
    for path, record in ((previous_path, previous), (held_path, held)):
        report, report_hash = FIRST.document(Path(path) / "report.json")
        require(report_hash == record["reportSha256"], "Run report changed after observation")
        reports.append(report)
    witnesses = []
    for index in range(192):
        values, hashes = [], []
        for path, report, record in zip((previous_path, held_path), reports, (previous, held)):
            artifact = report["acceptedStateArtifacts"][index]
            state, state_hash = FIRST.document(Path(path) / artifact["path"])
            require(state_hash == artifact["sha256"] == record["states"][index + 1]["stateSha256"],
                    "Accepted prefix state changed after observation")
            values.append({key: state[key] for key in ("positionsMeters", "velocitiesMetersPerSecond")})
            hashes.append(state_hash)
        require(encoded(values[0]) == encoded(values[1]), "Initial 384 ms position/velocity prefix differs")
        witnesses.append({"step": index + 1, "timeSeconds": (index + 1) * .002,
            "stateSha256": hashes, "positionsAndVelocitiesSha256": digest(encoded(values[0]))})
    return {"durationSeconds": .384, "transitions": 192,
        "sameInitialPositionsSha256": previous["initialPositionsSha256"], "witnesses": witnesses,
        "scope": "Exact canonical JSON equality of every saved position and velocity at unchanged physical times, separately from original fraction labels and state artifact identities."}


def compare(previous_control, previous_driven, held_control, held_driven):
    previous = FIRST.compare(previous_control, previous_driven)
    held = FIRST.compare(held_control, held_driven)
    identities, prefixes = {}, {}
    for role, previous_path, held_path in (("control", previous_control, held_control),
                                          ("driven", previous_driven, held_driven)):
        first_source, first_hash = FIRST.document(Path(previous_path) / "canonical.json")
        second_source, second_hash = FIRST.document(Path(held_path) / "canonical.json")
        require(first_hash == previous[role]["canonicalSha256"] and second_hash == held[role]["canonicalSha256"],
                "Canonical source changed after observation")
        identities[role] = validate_hold_pair(previous[role], held[role], first_source, second_source)
        prefixes[role] = verify_prefix(previous_path, held_path, previous[role], held[role])
    return {"profile": "source-binding-extended-hold-comparison-v1", "accepted": False,
        "scope": "Same 256 ms turn and 64 ms release/tail, with hold extended from128 to640 ms; fixed2 ms steps and unchanged physical source/settings.",
        "identities": identities, "prefixVerification": prefixes,
        "summaries": {"previous": {role: TIME.summarize(previous[role]) for role in ("control", "driven")},
            "extendedHold": {role: TIME.summarize(held[role], hold_fractions=(Fraction(1, 4), Fraction(7, 8)),
                               passive_start=Fraction(15, 16)) for role in ("control", "driven")}},
        "extendedHold": held,
        "codeDigests": {**held["codeDigests"],
            **{name: digest(read_regular(SCRIPTS / name)) for name in (
                "analyze-binding-time-controls.py", "analyze-binding-hold-control.py")}},
        "limitations": ["Longer hold permits more existing numerical dissipation and is not calibrated material damping.",
            "Release occurs at a fixed predeclared time, not when a chosen endpoint happens to appear settled.",
            "Saved-window ranges do not prove continuous extrema or settled equilibrium.",
            "No binding wrap, stitch-down, source phase completion, convergence or garment acceptance."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("previous-control", "previous-driven", "held-control", "held-driven", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.previous_control, args.previous_driven, args.held_control, args.held_driven)
    atomic_bytes(args.output, json.dumps(result, indent=2, allow_nan=False).encode() + b"\n")
    print(json.dumps({"accepted": False, "summaries": result["summaries"]}))


if __name__ == "__main__":
    main()
