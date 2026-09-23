"""Compare replayed matched binding controls without promoting construction status.

This trusted offline reporter binds existing current replay artifacts to their
captured inputs and state bytes. It does not rerun those path/force proofs.
New geometry observations use current independent diagnostic helpers.
"""

import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from solver_attempt_journal import strict_loads
from solver_binding_diagnostics import analyze_binding_diagnostic
from solver_gripper_replay import derive_grippers, parameters as gripper_parameters
from solver_process_budget import atomic_bytes, read_regular, verify_capture
from solver_sewing_replay import derive_sewing
from solver_surface_distance_diagnostics import minimum_surface_distance


SCRIPTS = Path(__file__).resolve().parent


def validate_time_policy(metadata, arguments=None):
    """Admit only declared fixed-duration controls, including historical v1."""
    policies = {"original-128ms-v1": (.128, 64), "fourfold-512ms-v1": (.512, 256)}
    if "timePolicy" in metadata:
        declaration = metadata["timePolicy"]
        require(type(declaration) is dict and type(declaration.get("id")) is str
                and declaration["id"] in policies, "Known explicit binding time policy required")
        identity = declaration["id"]
    else:
        # Historical snapshots predate this field. Their original physical
        # duration and grid are fixed, not inferred from supplied run arguments.
        identity = "original-128ms-v1"
        declaration = None
    duration, subdivisions = policies[identity]
    expected = {"profile": "binding-first-turn-time-v1", "id": identity,
                "durationSeconds": duration, "nominalStepSeconds": .002}
    if declaration is not None:
        require(encoded(declaration) == encoded(expected), "Declared binding timing constants differ")
    require(type(metadata.get("subdivisions")) is int and metadata["subdivisions"] == subdivisions,
            "Declared binding grid differs from time policy")
    if arguments is not None:
        require(type(arguments.get("subdivisions")) is int and arguments["subdivisions"] == subdivisions
                and type(arguments.get("step_seconds")) in (int, float)
                and arguments["step_seconds"] == duration
                and arguments["step_seconds"] / subdivisions == .002,
                "Actual duration/grid differs from binding time policy")
    return expected


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def document(path):
    content = read_regular(path, 64 * 1024 ** 2)
    return strict_loads(content), digest(content)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def grip_observations(record, positions, fraction):
    targets, activation = gripper_parameters(record, fraction)
    observations, energy = [], Fraction()
    total_force = [Fraction()] * 3
    for index, (identity, instance, triangle, weights, stiffness) in enumerate(record.anchors):
        anchor = [sum((Fraction(float(weight)) * Fraction(float(positions[vertex, axis]))
                       for vertex, weight in zip(record.faces[triangle], weights)), Fraction()) for axis in range(3)]
        error = [value - Fraction(float(goal)) for value, goal in zip(anchor, targets[index])]
        coefficient = Fraction(stiffness) * Fraction(float(activation[index]))
        force = [-coefficient * value for value in error]
        energy += coefficient * sum((value * value for value in error), Fraction()) / 2
        weight_sum = sum((Fraction(float(weight)) for weight in weights), Fraction())
        total_force = [a + weight_sum * b for a, b in zip(total_force, force)]
        observations.append({"id": identity, "instanceId": instance, "triangleIndex": triangle,
            "positionMeters": [float(value) for value in anchor], "targetMeters": targets[index].tolist(),
            "activation": float(activation[index]), "trackingErrorMeters": math.hypot(*(float(value) for value in error)),
            "forceNewtons": [float(value) for value in force]})
    if fraction >= .875:
        require(not np.any(activation) and energy == 0 and all(value == 0 for value in total_force),
                "Released grippers must have exactly zero activation, force and stored energy")
    return {"grippers": observations, "energyJoules": float(energy),
            "totalClothForceNewtons": [float(value) for value in total_force],
            "scope": "Exact binary-input material anchors and quadratic virtual-tool diagnostics rounded for reporting; no solid-tool geometry"}


def validate_reference(source, initial, sewing, grippers):
    """Check the declared reference against source paths and actual controls."""
    metadata = source["bindingFirstTurnDiagnostic"]
    validate_time_policy(metadata)
    require(metadata.get("profile") == "source-left-binding-first-turn-v1"
            and metadata["bindingInstanceId"] == "opening_binding_left_left:shell"
            and metadata["sleeveInstanceId"] == "sleeve_left:shell", "Bounded left-binding reference required")
    panels = {panel["id"]: panel for panel in source["sourcePattern"]["panels"]}
    axes = [np.column_stack((np.asarray(panels[name]["points"])[indices] * .001, np.zeros(2)))
            for name, indices in (("opening_binding_left_left", [1, 2]), ("sleeve_left", [5, 6]))]
    for field, axis in zip(("sourceAxisEndpointsMeters", "receiverAxisEndpointsMeters"), axes):
        np.testing.assert_array_equal(metadata[field], axis)
    tangents = [(axis[1] - axis[0]) / np.linalg.norm(axis[1] - axis[0]) for axis in axes]
    for field, tangent in zip(("bindingTangentsRest", "sleeveTangentsRest"), tangents):
        np.testing.assert_array_equal(metadata[field], np.tile(tangent, (5, 1)))
    require(metadata["nominalSeamOffsetMeters"] == .001, "Explicit one-millimetre research stand-off required")
    origin = axes[1].mean(axis=0) + [0., 0., .001]
    tangent = tangents[1]
    np.testing.assert_array_equal(metadata["axisOriginMeters"], origin)
    np.testing.assert_array_equal(metadata["axisDirection"], tangent)
    ordered = sorted(source["instanceOffsets"], key=source["instanceOffsets"].get)
    ranges = {name: [source["instanceOffsets"][name], source["instanceOffsets"][ordered[index + 1]]
                    if index + 1 < len(ordered) else len(initial)] for index, name in enumerate(ordered)}
    require(encoded(metadata["instanceRanges"]) == encoded(ranges), "Instance ranges differ from canonical source")
    inward = np.array([-tangent[1], tangent[0], 0.])
    rotation = np.column_stack((-inward, tangent, [0., 0., 1.]))
    translation = origin - rotation @ axes[0].mean(axis=0)
    expected_placement = np.asarray(source["restMeters"]).copy()
    parking = {"cuff_left:shell": .05, "cuff_left:facing": .10,
               "opening_binding_left_right:shell": .15, "sleeve_left:shell": 0.}
    for name, (lower, upper) in ranges.items():
        if name == metadata["bindingInstanceId"]:
            expected_placement[lower:upper] = expected_placement[lower:upper] @ rotation.T + translation
        else:
            expected_placement[lower:upper] += [0., 0., parking[name]]
    np.testing.assert_array_equal(initial, expected_placement)
    require(list(grippers.ids) == metadata["gripperIds"], "Declared gripper identities differ")
    require(all(anchor[-1] == metadata["gripperStiffnessNPerM"] for anchor in grippers.anchors),
            "Declared tool stiffness differs from actual anchors")
    fractions = [index / 16 for index in range(9)] + [.75, .875, 1.]
    knots = source["gripperActuation"]["schedule"]["knots"]
    require([knot["fraction"] for knot in knots] == fractions, "Eight angular samples, hold, release and passive tail required")
    angle = metadata["angleDegrees"]
    require(type(angle) in (int, float) and math.isfinite(angle) and 0 <= angle <= 3,
            "Bounded declared angle required")
    theta = math.radians(angle)
    require(metadata["angleRadians"] == theta, "Declared angular units differ")
    require(metadata["schedule"]["maximumChordRelativeContraction"] == 1 - math.cos(theta / 16),
            "Declared chord contraction differs from angular samples")
    clearance = metadata["rigidReferenceClearanceMeters"]
    require(clearance["minimumToSleevePlane"] == .001 - .010 * math.sin(theta)
            and clearance["maximumAboveSleevePlane"] == .001 + .030 * math.sin(theta),
            "Declared nominal rigid-reference clearance differs from the bounded allowance policy")
    cross = np.array([[0., -tangent[2], tangent[1]], [tangent[2], 0., -tangent[0]],
                      [-tangent[1], tangent[0], 0.]])
    lower, upper = ranges[metadata["bindingInstanceId"]]
    for index, knot in enumerate(knots):
        target_angle = theta * min(index, 8) / 8
        reference = initial.copy()
        if target_angle != 0:
            rotation = np.eye(3) + math.sin(target_angle) * cross + (1 - math.cos(target_angle)) * (cross @ cross)
            reference[lower:upper] = (initial[lower:upper] - origin) @ rotation.T + origin
        expected = [[float(sum((Fraction(float(weight)) * Fraction(float(reference[vertex, axis]))
                               for vertex, weight in zip(grippers.faces[triangle], weights)), Fraction()))
                     for axis in range(3)] for _, _, triangle, weights, _ in grippers.anchors]
        require(encoded(knot["targetsMeters"]) == encoded(expected),
                "Actual gripper target differs from declared source-axis rotation reference")
        require(knot["activation"] == [0. if knot["fraction"] >= .875 else 1.] * 3,
                "Actual gripper knot activation differs from hold/release policy")
    # Compare every actual source/control/declaration field except precisely
    # these rederived angle-dependent values and the resulting source hash.
    common = copy.deepcopy(source)
    common["sewingActuation"].pop("sourceSha256")
    for knot in common["gripperActuation"]["schedule"]["knots"]:
        knot.pop("targetsMeters")
    declared = common["bindingFirstTurnDiagnostic"]
    for key in ("angleDegrees", "angleRadians"):
        declared.pop(key)
    declared["schedule"].pop("maximumChordRelativeContraction")
    for key in ("minimumToSleevePlane", "maximumAboveSleevePlane"):
        declared["rigidReferenceClearanceMeters"].pop(key)
    return digest(encoded(common))


def load_control(directory):
    directory = Path(directory).resolve()
    report, report_digest = document(directory / "report.json")
    source, canonical_digest = document(directory / "canonical.json")
    placement, placement_digest = document(directory / "placement.json")
    replay, replay_digest = document(directory / "verified-replay.json")
    require(report.get("terminal") is True and report.get("accepted") is False
            and report.get("completed") is True and replay.get("accepted") is False
            and replay.get("completed") is True and replay.get("completedFraction") == 1.
            and replay.get("finalStateVerified") is True, "Complete unaccepted research and current replay required")
    verify_capture(directory, report)
    require(replay.get("reportSha256") == report_digest
            and encoded(replay.get("sourceDigests")) == encoded(report["sourceDigests"]),
            "Replay evidence differs from captured report/source identity")
    for key, filename in (("replayScriptSha256", "replay-rest-filtered-continuation.py"),
                          ("sewingVerifierSha256", "solver_sewing_replay.py"),
                          ("sewingPathVerifierSha256", "solver_sewing_sweep.py"),
                          ("gripperVerifierSha256", "solver_gripper_replay.py"),
                          ("triangleVerifierSha256", "solver_triangle_sweep.py"),
                          ("broadPhaseVerifierSha256", "solver_ipc_broad_phase.py"),
                          ("candidateCoverageVerifierSha256", "solver_candidate_coverage.py"),
                          ("captureVerifierSha256", "solver_process_budget.py")):
        require(replay.get(key) == digest(read_regular(SCRIPTS / filename)),
                "Current replay and independent control/path verifiers required")
    tolerance = replay.get("sewingPathToleranceM")
    require(type(tolerance) in (int, float) and math.isfinite(tolerance) and tolerance > 0,
            "Explicit sampled-distance path tolerance required")
    metadata = source.get("bindingFirstTurnDiagnostic")
    require(type(metadata) is dict and metadata.get("accepted") is False,
            "Explicit unaccepted binding diagnostic declaration required")
    require(metadata.get("heldRowIndices") == list(range(5)) and metadata.get("pendingRowIndices") == list(range(5, 40)),
            "Complete five-held/thirty-five-pending source-row partition required")
    original = {key: source[key] for key in metadata["preservedSourceFields"]}
    require(digest(encoded(original)) == metadata["sourceUnitCanonicalSha256"],
            "Original cuff source fields changed")
    arguments = report["arguments"]
    time_policy = validate_time_policy(metadata, arguments)
    require(arguments.get("sewing_activation") is True and arguments.get("material_grippers") is True
            and arguments.get("sewing_mode") == "distance" and arguments["subdivisions"] == metadata["subdivisions"],
            "Matched captured sewing and material-gripper controls required")
    sewing = derive_sewing(source, arguments["subdivisions"], sewing_mode="distance")
    grippers = derive_grippers(source, arguments["subdivisions"])
    indices = metadata["heldRowIndices"]
    require(list(sewing.row_ids[:5]) == metadata["heldRowIds"], "Held row source identities changed")
    require(encoded(list(sewing.initial_targets[:5])) == encoded(metadata["heldSeamTargetsMeters"]),
            "Declared held target distances differ from actual controls")
    require(all(all(value == (1. if row < 5 else 0.) for row, value in enumerate(weights))
                for weights in sewing.activation), "This comparison requires constant held/pending activation")
    require(sewing.initial_targets == sewing.final_targets, "Held sewing targets cannot change")
    rest, initial = np.asarray(source["restMeters"]), np.asarray(placement["placedMeters"])
    faces = np.asarray(source["triangles"]).reshape((-1, 3))
    np.testing.assert_array_equal(initial, source["placedMeters"])
    matched_source_digest = validate_reference(source, initial, sewing, grippers)
    require(encoded(replay["contactProfile"]) == encoded(report["contactProfile"]),
            "Replay contact profile differs from captured physics")
    geometry_options = {key: metadata[key] for key in (
        "sleeveFrameFaces", "bindingFrameFaces", "sleeveTangentsRest", "bindingTangentsRest")}
    options = {"instance_ranges": metadata["instanceRanges"], "row_ids": metadata["heldRowIds"],
        "rows": [sewing.rows[index] for index in indices],
        "sleeve_instance_id": metadata["sleeveInstanceId"], "binding_instance_id": metadata["bindingInstanceId"],
        "sleeve_frame_faces": geometry_options["sleeveFrameFaces"],
        "binding_frame_faces": geometry_options["bindingFrameFaces"],
        "sleeve_tangents_rest": geometry_options["sleeveTangentsRest"],
        "binding_tangents_rest": geometry_options["bindingTangentsRest"],
        "target_distances": metadata["heldSeamTargetsMeters"]}
    groups = []
    for name in (metadata["bindingInstanceId"], metadata["sleeveInstanceId"]):
        lower, upper = metadata["instanceRanges"][name]
        groups.append(np.flatnonzero(np.all((faces >= lower) & (faces < upper), axis=1)).tolist())

    def observe(positions, fraction, speed):
        return {"fraction": fraction, "timeSeconds": fraction * time_policy["durationSeconds"],
            "maximumSpeedMetersPerSecond": speed,
            "bindingGeometry": analyze_binding_diagnostic(rest, initial, positions, faces, **options),
            "surfaceSeparation": minimum_surface_distance(positions, faces, *groups),
            "gripperGeometry": grip_observations(grippers, positions, fraction)}

    states = [observe(initial, 0., 0.)]
    artifacts, proofs = report["acceptedStateArtifacts"], replay["states"]
    require(len(artifacts) == len(proofs), "Replay must cover every accepted state")
    previous, previous_fraction = initial, 0.
    for artifact, proof in zip(artifacts, proofs):
        state, state_digest = document(directory / artifact["path"])
        require(state_digest == artifact["sha256"] and state.get("accepted") is False,
                "Accepted numerical state artifact changed")
        record = state["record"]
        fraction = record["endFraction"]
        require(record["startFraction"] == previous_fraction and fraction > previous_fraction
                and proof["endFraction"] == fraction, "Continuous original state fractions required")
        require(record["durationSeconds"] == time_policy["durationSeconds"] * (fraction - previous_fraction),
                "Actual transition duration differs from original control fractions")
        require(proof.get("physicalPathPass") is True and proof.get("endpointIntersections") == 0
                and type(proof.get("recomputedResidualN")) in (int, float)
                and 0 <= proof["recomputedResidualN"] <= 1e-6,
                "Replay physical path, endpoint and stationarity checks must pass")
        triangle = proof.get("trianglePathVerification", {})
        coverage = proof.get("candidateCoverageVerification", {})
        certificate = proof.get("pathCertificate", {})
        require(triangle.get("profile") == "exact-rational-affine-triangle-nondegeneracy-v1"
                and triangle.get("triangles") == len(faces) and triangle.get("verifiedLeaves", 0) >= len(faces)
                and coverage.get("verified") is True and coverage.get("status") == "complete"
                and coverage["counts"]["vertices"] == len(rest) and coverage["counts"]["faces"] == len(faces)
                and coverage["counts"]["observedCandidates"] >= coverage["counts"]["requiredCandidates"]
                and certificate.get("safe") is True and certificate.get("unresolvedCount") == 0
                and certificate.get("candidateCount") == certificate.get("certifiedCount")
                and certificate.get("broadPhase") == replay["verificationBroadPhase"] == "ipctk-HashGrid-explicit-v1",
                "Complete triangle and contact-candidate path evidence required")
        positions, velocities = np.asarray(state["positionsMeters"]), np.asarray(state["velocitiesMetersPerSecond"])
        np.testing.assert_array_equal(velocities, (positions - previous) / record["durationSeconds"])
        path = proof.get("sewingPathVerification", {})
        require(path.get("verified") is True and path.get("checkedRows") == 5 and path.get("pendingRows") == 35
                and path.get("sourceSha256") == sewing.source_sha256 and path.get("toleranceM") == tolerance,
                "Complete held sampled-seam path evidence required")
        step = record["step"]
        require(step["pendingSewingRows"] == list(range(5, 40))
                and all(value is None for value in step["sewingRowTargetErrorsM"][5:]),
                "Pending rows cannot acquire hidden active diagnostics")
        require(step["energyBalance"]["sewingParameterWorkJoules"] == 0.,
                "Constant sewing controls must perform zero parameter work")
        observed = observe(positions, fraction, float(np.max(np.linalg.norm(velocities, axis=1))))
        observed.update(stateSha256=state_digest, recomputedResidualNewtons=proof["recomputedResidualN"],
            stepDurationSeconds=record["durationSeconds"],
            contactEnergyJoules=proof["contactEnergyJ"], sewingPathVerification=path,
            gripperMomentum=step["gripperMomentum"])
        states.append(observed)
        previous, previous_fraction = positions, fraction
    require(previous_fraction == report["adaptive"]["completedFraction"], "Accepted prefix completion differs")
    require(previous_fraction == 1., "Full matched control intervals required")
    final_state, final_digest = document(directory / report["stateArtifact"]["path"])
    require(final_digest == report["stateArtifact"]["sha256"], "Final state artifact changed")
    np.testing.assert_array_equal(final_state["positionsMeters"], previous)
    np.testing.assert_array_equal(final_state["velocitiesMetersPerSecond"], velocities)
    require(final_state["completedDurationSeconds"] == report["arguments"]["step_seconds"],
            "Final state duration differs from matched interval")
    return {"directory": directory.name, "accepted": False, "completed": report["completed"],
        "completedFraction": previous_fraction, "angleDegrees": metadata["angleDegrees"],
        "reportSha256": report_digest, "canonicalSha256": canonical_digest, "placementSha256": placement_digest,
        "replaySha256": replay_digest, "sourceUnitSha256": metadata["sourceUnitCanonicalSha256"],
        "physicsArguments": {key: value for key, value in arguments.items() if key not in ("canonical", "placement", "output")},
        "declaration": metadata, "states": states,
        "timePolicy": time_policy,
        "nominalTimeStepPreserved": all(state["stepDurationSeconds"] == .002 for state in states[1:]),
        "sewingWorkSummary": replay["verifiedSewingWorkSummary"], "gripperWorkSummary": replay["verifiedGripperWorkSummary"],
        "exactContactLeaves": replay["exactCertificateLeaves"], "sewingPathToleranceMeters": tolerance,
        "sourceDigests": report["sourceDigests"], "matchedSourceSha256": matched_source_digest,
        "initialPositionsSha256": digest(encoded(initial.tolist())),
        "sewingControls": {key: value for key, value in source["sewingActuation"].items() if key != "sourceSha256"},
        "gripperAnchors": source["gripperActuation"]["anchors"],
        "gripperActivationSchedule": [{"fraction": knot["fraction"], "activation": knot["activation"]}
            for knot in source["gripperActuation"]["schedule"]["knots"]],
        "initialGripperTargets": source["gripperActuation"]["schedule"]["knots"][0]["targetsMeters"]}


def compare(control_path, driven_path):
    control, driven = load_control(control_path), load_control(driven_path)
    require(control["angleDegrees"] == 0. and 0 < driven["angleDegrees"] <= 3.,
            "A zero-angle control and a positive at-most-three-degree diagnostic are required")
    for key in ("sourceUnitSha256", "physicsArguments", "sourceDigests", "sewingPathToleranceMeters",
                "initialPositionsSha256", "sewingControls", "gripperAnchors", "gripperActivationSchedule",
                "initialGripperTargets", "matchedSourceSha256"):
        require(encoded(control[key]) == encoded(driven[key]), "Matched controls differ in " + key)
    for key in ("rigidPlacements", "heldSeamTargetsMeters", "gripperStiffnessNPerM", "gripperSourceSupports",
                "textileSidePolicy", "frameSelectionBindings"):
        require(encoded(control["declaration"][key]) == encoded(driven["declaration"][key]),
                "Matched source/placement/tool declaration differs in " + key)
    shared = sorted(set(item["fraction"] for item in control["states"]) & set(item["fraction"] for item in driven["states"]))
    return {"profile": "source-binding-first-turn-comparison-v1", "accepted": False,
        "scope": "Current endpoint geometry observations of matched prepared/held sampled attachments. Existing replay artifacts are hash-bound and checked for current verifier identity; this reporter does not rerun their force or path proofs.",
        "control": control, "driven": driven, "exactCommonFractions": shared,
        "commonTimePolicy": "Compare only identical saved original fractions; adaptive states are not interpolated into new observations.",
        "codeDigests": {name: digest(read_regular(SCRIPTS / name)) for name in (
            "analyze-binding-first-turn.py", "solver_binding_diagnostics.py", "solver_surface_distance_diagnostics.py",
            "solver_gripper_replay.py", "solver_sewing_replay.py", "solver_process_budget.py")},
        "limitations": ["Geometric normals and material-correspondence offsets are not inferred textile sides or measured physical thread slip.",
            "Recorded contact and momentum/work evidence retains the current replay's numerical/runtime boundaries.",
            "Five sampled scalar rows do not establish continuous spatial stitching or a finite-thickness joint.",
            "No binding wrap, stitch-down, apex securing, turning passage, construction milestone, calibrated material or garment acceptance."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--driven", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.control, args.driven)
    atomic_bytes(args.output, json.dumps(result, indent=2, allow_nan=False).encode() + b"\n")
    print(json.dumps({"accepted": False, "controlStates": len(result["control"]["states"]),
        "drivenStates": len(result["driven"]["states"]), "commonFractions": len(result["exactCommonFractions"])}))


if __name__ == "__main__":
    main()
