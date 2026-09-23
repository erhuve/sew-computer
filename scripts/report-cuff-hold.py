"""Summarize trusted, replay-verified synthetic cuff baseline/hold artifacts.

Plotting-only environment: matplotlib==3.10.7. This does not run the solver or
assign garment acceptance. Raw captured research artifacts remain private.
"""

import argparse
import hashlib
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError("Evidence reporting requires enabled verification assertions")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path):
    report = json.loads((path / "report.json").read_text())
    verified = json.loads((path / "verified-replay.json").read_text())
    source = json.loads((path / "canonical.json").read_text())
    assert report["accepted"] is verified["accepted"] is False
    assert report["terminal"] is report["completed"] is verified["completed"] is verified["finalStateVerified"] is True
    assert report["captureIntegrityVerified"] is True and not report["recoveryErrors"]
    assert report["adaptive"]["completedFraction"] == verified["completedFraction"] == 1.
    assert verified["reportSha256"] == digest(path / "report.json")
    assert report["canonicalSha256"] == digest(path / "canonical.json")
    assert report["placementSha256"] == digest(path / "placement.json")
    assert report["sourceDigests"] == verified["sourceDigests"]
    assert report["contactProfile"] == verified["contactProfile"]
    assert report["contactProfile"]["broadPhase"] == verified["verificationBroadPhase"] == "ipctk-HashGrid-explicit-v1"
    assert source["assemblySchedule"] == report["assemblySchedule"] == verified["assemblySchedule"]
    assert source["foldActuation"] == report["foldActuation"] == verified["foldActuation"]
    mode = report["arguments"]["sewing_mode"]
    assert mode in ("normal-offset", "distance")
    assert report["sewingMode"] == verified["sewingMode"] == mode
    for name, expected in report["sourceDigests"].items():
        assert digest(path / "source-snapshot" / name) == expected
    states = []
    for artifact in report["acceptedStateArtifacts"]:
        assert digest(path / artifact["path"]) == artifact["sha256"]
        states.append(json.loads((path / artifact["path"]).read_text()))
    assert 0 < len(states) == len(verified["states"])
    assert source["instanceOffsets"] == {"cuff_left:shell": 0, "cuff_left:facing": 33}
    faces = np.asarray(source["triangles"]).reshape((-1, 3))
    hinges = np.asarray(source["foldActuation"]["hinges"])
    assert np.asarray(source["restMeters"]).shape == (66, 3) and faces.shape == (96, 3)
    assert np.all((0 <= faces[:48]) & (faces[:48] < 33))
    assert np.array_equal(faces[48:] - 33, faces[:48])
    assert hinges.shape == (24, 4) and np.all((0 <= hinges[:12]) & (hinges[:12] < 33))
    assert np.array_equal(hinges[12:] - 33, hinges[:12])
    times = np.array([state["record"]["endFraction"] for state in states]) * report["arguments"]["step_seconds"]
    angles = np.degrees([state["record"]["step"]["foldAnglesRadians"] for state in states])
    speeds = [float(np.linalg.norm(state["velocitiesMetersPerSecond"], axis=1).max()) for state in states]
    targets = np.asarray([state["record"]["step"]["foldTargetsRadians"] for state in states])
    assert angles.shape == targets.shape == (len(states), 24)
    assert np.isfinite(angles).all() and np.isfinite(speeds).all() and np.isfinite(targets).all()
    assert np.all(targets == targets[:, :1])  # One plotted target must represent every hinge.
    assert np.all(np.diff(times) > 0) and times[-1] == report["arguments"]["step_seconds"]
    final_descriptor = report["stateArtifact"]
    assert digest(path / final_descriptor["path"]) == final_descriptor["sha256"]
    final_state = json.loads((path / final_descriptor["path"]).read_text())
    assert final_state["accepted"] is False and final_state["completedDurationSeconds"] == times[-1]
    for field in ("positionsMeters", "velocitiesMetersPerSecond"):
        assert final_state[field] == states[-1][field]
    for state, proof in zip(states, verified["states"]):
        assert state["accepted"] is False and state["record"]["converged"] is True
        assert state["record"]["step"]["sewingMode"] == mode
        assert state["record"]["endFraction"] == proof["endFraction"]
        assert state["record"]["step"]["foldAnglesRadians"] == proof["foldAnglesRadians"]
        assert np.isfinite(proof["recomputedResidualN"]) and 0 <= proof["recomputedResidualN"] <= 1e-6
        assert proof["physicalPathPass"] is True and proof["endpointIntersections"] == 0
        triangle, coverage, contact = (proof[key] for key in
            ("trianglePathVerification", "candidateCoverageVerification", "pathCertificate"))
        assert triangle["profile"] == "exact-rational-affine-triangle-nondegeneracy-v1"
        assert triangle["triangles"] == 96 and triangle["verifiedLeaves"] >= 96
        assert coverage["verified"] is True and coverage["status"] == "complete"
        assert coverage["profile"] == "exact-rational-swept-aabb-v1"
        assert coverage["counts"]["vertices"] == 66 and coverage["counts"]["faces"] == 96
        assert contact["safe"] is True and contact["unresolvedCount"] == 0
        assert contact["broadPhase"] == "ipctk-HashGrid-explicit-v1"
        assert contact["candidateCount"] == contact["certifiedCount"] == coverage["counts"]["observedCandidates"]
        assert coverage["counts"]["requiredCandidates"] <= contact["candidateCount"]
    assert sum(row["pathCertificate"]["leafCount"] for row in verified["states"]) == verified["exactCertificateLeaves"]
    durations = np.diff(np.concatenate(([0.], times)))
    summary = {"accepted": False, "completed": True, "runDirectory": path.name,
        "canonicalSha256": report["canonicalSha256"], "placementSha256": report["placementSha256"],
        "reportSha256": digest(path / "report.json"), "replaySha256": digest(path / "verified-replay.json"),
        "finalState": final_descriptor, "lastAcceptedState": report["acceptedStateArtifacts"][-1],
        "arguments": {key: value for key, value in report["arguments"].items()
                      if key not in ("canonical", "placement", "output")},
        "assemblySchedule": source["assemblySchedule"], "sourceProvenance": source["provenance"],
        "acceptedSteps": len(states), "rejectedAttempts": len(report["adaptive"]["rejectedSteps"]),
        "supervisedCpuSeconds": report["cpuSeconds"], "workerReportedCpuSeconds": report["workerReportedCpuSeconds"],
        "contactProfile": report["contactProfile"], "edgeStrain": verified["edgeStrain"],
        "sewingMode": mode,
        "seamTargetErrorDefinition": ("Norm of anchor-displacement minus the prescribed material-normal offset vector."
            if mode == "normal-offset" else "Absolute difference between anchor distance and the prescribed scalar separation."),
        "finalMaximumSeamTargetErrorMm": report["finalMaximumCompletedTargetErrorM"] * 1000,
        "finalMaximumSpeedMps": speeds[-1], "finalAnglesDegrees": angles[-1].tolist(),
        "finalLayerAnglesDegrees": {layer: {"minimum": float(angles[-1, first:last].min()),
            "mean": float(angles[-1, first:last].mean()), "maximum": float(angles[-1, first:last].max())}
            for layer, first, last in (("shell", 0, 12), ("facing", 12, 24))},
        "acceptedTimePartition": {"endTimesSha256": hashlib.sha256(json.dumps(times.tolist(),
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "initialStepSeconds": report["arguments"]["step_seconds"] / report["arguments"]["subdivisions"],
            "minimumStepSeconds": float(durations.min()), "maximumStepSeconds": float(durations.max())},
        "exactContactLeaves": verified["exactCertificateLeaves"],
        "exactTriangleLeaves": sum(row["trianglePathVerification"]["verifiedLeaves"] for row in verified["states"]),
        "coveredCandidateOccurrences": sum(row["candidateCoverageVerification"]["counts"]["requiredCandidates"]
                                            for row in verified["states"]),
        "coverageComparisons": sum(row["candidateCoverageVerification"]["counts"]["comparisons"]
                                   for row in verified["states"]),
        "maximumReconstructedResidualN": max(row["recomputedResidualN"] for row in verified["states"]),
        "verificationDigests": {key: value for key, value in verified.items()
                                 if key.endswith("VerifierSha256") or key == "replayScriptSha256"}}
    return report, source, states, times, angles, speeds, summary


def distance_comparison(normal, distance):
    """Admit one seam-law substitution with otherwise identical captured inputs."""
    assert normal[0]["arguments"]["sewing_mode"] == "normal-offset"
    assert distance[0]["arguments"]["sewing_mode"] == "distance"
    for field in ("canonicalSha256", "placementSha256", "sourceDigests", "contactProfile"):
        assert normal[0][field] == distance[0][field]
    assert normal[1] == distance[1]
    ignored_paths_and_mode = {"canonical", "placement", "output", "sewing_mode"}
    assert {key: value for key, value in normal[0]["arguments"].items() if key not in ignored_paths_and_mode} == {
        key: value for key, value in distance[0]["arguments"].items() if key not in ignored_paths_and_mode}
    assert normal[0]["arguments"]["step_seconds"] == distance[0]["arguments"]["step_seconds"] == .256
    assert normal[-1]["verificationDigests"] == distance[-1]["verificationDigests"]
    return {"accepted": False, "normalRun": normal[-1]["runDirectory"], "distanceRun": distance[-1]["runDirectory"],
        "sharedCanonicalSha256": normal[0]["canonicalSha256"],
        "sharedPlacementSha256": normal[0]["placementSha256"],
        "sameAcceptedTimePartition": bool(np.array_equal(normal[3], distance[3])),
        "comparisonSeconds": .256,
        "limitations": ["Only the seam potential changes; captured geometry, placement, targets, contact, code and other arguments agree.",
            "Adaptive time partitions differ in these runs; this is not a comparison on identical physical substeps.",
            "Final states retain motion; this finite-duration control does not establish equilibrium or timestep/refinement convergence.",
            "Normal-offset seam error is a vector residual; distance-only seam error is a scalar distance residual. Their magnitudes are not the same metric.",
            "Distance-only sewing does not prescribe the material-normal side or orientation. It is a diagnostic substitution, not an accepted cuff construction."]}


def plot_distance_comparison(normal, distance, destination):
    figure = plt.figure(figsize=(12, 11), layout="constrained")
    figure.suptitle("Seam-law comparison · different adaptive histories", fontsize=17)
    grid = figure.add_gridspec(3, 2, height_ratios=[1.15, 1., .8])
    runs, names = (normal, distance), ("Normal-offset seam", "Distance-only seam")
    colors = ("#c17a25", "#267692")
    positions = [np.asarray(run[2][-1]["positionsMeters"]) * 1000 for run in runs]
    lower, upper = np.vstack(positions).min(axis=0) - 3, np.vstack(positions).max(axis=0) + 3
    faces = np.asarray(normal[1]["triangles"]).reshape((-1, 3))
    all_angles = np.concatenate([run[4].ravel() for run in runs])
    targets = [np.degrees([state["record"]["step"]["foldTargetsRadians"][0] for state in run[2]]) for run in runs]
    angle_lower, angle_upper = min(all_angles.min(), *(target.min() for target in targets)), max(
        all_angles.max(), *(target.max() for target in targets))
    padding = max(2., .05 * (angle_upper - angle_lower))
    for column, (run, points, name, target) in enumerate(zip(runs, positions, names, targets)):
        geometry = figure.add_subplot(grid[0, column], projection="3d")
        for first, last, color in ((0, 48, colors[0]), (48, 96, colors[1])):
            geometry.add_collection3d(Poly3DCollection(points[faces[first:last]], facecolor=color,
                edgecolor=color, linewidth=.4, alpha=.55))
        geometry.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
                     xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title=name + " · 256 ms")
        geometry.set_box_aspect(upper - lower)
        geometry.view_init(elev=25, azim=-63)
        geometry.tick_params(labelsize=8, pad=0)
        axis = figure.add_subplot(grid[1, column])
        times = run[3] * 1000
        for first, last, color, layer in ((0, 12, colors[0], "Shell"), (12, 24, colors[1], "Facing")):
            subset = run[4][:, first:last]
            axis.fill_between(times, subset.min(axis=1), subset.max(axis=1), color=color, alpha=.18)
            axis.plot(times, subset.mean(axis=1), color=color, label=layer + " mean / range")
        axis.plot(times, target, "--", color="#222222", label="Prescribed target")
        axis.axvline(64, color="#777777", linewidth=1, linestyle=":")
        axis.set(xlim=(0, 256), ylim=(angle_lower - padding, angle_upper + padding),
                 xlabel="Physical time (ms)", ylabel="Allowance hinge angle (degrees)")
        axis.legend(loc="lower right", frameon=False, fontsize=8)
        axis.grid(alpha=.18)
    speed_axis = figure.add_subplot(grid[2, :])
    for run, name, color in zip(runs, names, ("#344352", "#89559b")):
        speed_axis.semilogy(run[3] * 1000, run[5], color=color,
                           label=f"{name}: {run[-1]['acceptedSteps']} accepted / {run[-1]['rejectedAttempts']} rejected integration steps")
    speed_axis.axvline(64, color="#777777", linewidth=1, linestyle=":")
    speed_axis.set(xlim=(0, 256), xlabel="Physical time (ms)", ylabel="Maximum vertex speed (m/s)")
    speed_axis.legend(frameon=False, fontsize=9)
    speed_axis.grid(alpha=.18)
    figure.supxlabel("Same source, target schedule and contact · residual motion remains · accepted: false\n"
        "Distance-only sewing changes the constraint law and does not prescribe layer side; neither run demonstrates cuff turning", fontsize=9)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hold", type=Path, required=True)
    parser.add_argument("--distance-hold", type=Path,
                        help="Optional same-input 256 ms distance-seam diagnostic, separately compared with the normal-offset hold")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    baseline, hold = load_run(args.baseline), load_run(args.hold)
    assert baseline[0]["sourceDigests"] == hold[0]["sourceDigests"]
    assert baseline[0]["arguments"]["step_seconds"] == .064
    assert hold[0]["arguments"]["step_seconds"] == .256
    allowed_differences = {"canonical", "placement", "output", "step_seconds", "subdivisions",
                           "cpu_limit_seconds", "wall_limit_seconds"}
    assert {key: value for key, value in baseline[0]["arguments"].items() if key not in allowed_differences} == {
        key: value for key, value in hold[0]["arguments"].items() if key not in allowed_differences}
    assert baseline[0]["arguments"]["subdivisions"] == 64 and hold[0]["arguments"]["subdivisions"] == 256
    assert baseline[0]["contactProfile"] == hold[0]["contactProfile"]
    assert baseline[0]["arguments"]["sewing_mode"] == "normal-offset"
    for run, fractions, sewing, folding in ((baseline, [0., .25, 1.], [0., 1., 1.], [0., 0., 1.]),
                                          (hold, [0., .0625, .25, 1.], [0., 1., 1., 1.], [0., 0., 1., 1.])):
        assert run[1]["assemblySchedule"] == {"profile": "sewing-fold-progress-v1", "knots": [
            {"fraction": fraction, "sewingProgress": sew, "foldProgress": fold}
            for fraction, sew, fold in zip(fractions, sewing, folding)]}
    hold_provenance = hold[1]["provenance"].copy()
    hold_derivation = hold_provenance.pop("holdDiagnostic")
    assert hold_provenance == baseline[1]["provenance"]
    assert hold_derivation["parentCanonicalSha256"] == baseline[0]["canonicalSha256"]
    source_base, source_hold = baseline[1].copy(), hold[1].copy()
    for source in (source_base, source_hold):
        source.pop("assemblySchedule")
        source.pop("provenance")
    assert source_base == source_hold
    assert len(hold[2]) >= len(baseline[2])
    for before, after in zip(baseline[2], hold[2]):
        assert before["record"]["endFraction"] == 4 * after["record"]["endFraction"]
        for field in ("positionsMeters", "velocitiesMetersPerSecond"):
            assert before[field] == after[field]
    distance = load_run(args.distance_hold) if args.distance_hold is not None else None
    comparison = distance_comparison(hold, distance) if distance is not None else None
    result = {"profile": "synthetic-cuff-normal-offset-hold-evidence-v1", "accepted": False,
        "runtime": json.loads(args.runtime.read_text()), "sourceDigests": baseline[0]["sourceDigests"],
        "runtimeManifestSha256": digest(args.runtime),
        "runtimeEvidence": "Separately captured runtime metadata; the run reports do not cryptographically bind this manifest.",
        "verificationScope": "Checks captured hashes and completed replay results; does not re-execute the numerical or exact proofs.",
        "sourceProvenance": baseline[1]["provenance"],
        "sharedPrefix": {"states": len(baseline[2]), "seconds": .064,
                         "maximumPositionDifferenceM": 0., "maximumVelocityDifferenceMps": 0.},
        "runs": {"baseline": baseline[-1], "hold": hold[-1]},
        "limitations": ["Synthetic coarse cuff control; no turning, sleeve attachment or garment acceptance.",
            "Targets hold after the unchanged sewing/fold ramps; no physical damping is added.",
            "Motion decay includes implicit integration; this is not a temporal/refinement convergence study.",
            "Candidate coverage and nonzero triangle area do not establish layer order or physical thickness at incident primitives.",
            "Raw source snapshots, journals and trajectories remain in ignored research storage; hashes cannot recreate missing bytes."],
        "reportGeneratorSha256": digest(Path(__file__))}
    if distance is not None:
        result["runs"]["distanceHold"] = distance[-1]
        result["seamLawComparison"] = comparison
    args.output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure = plt.figure(figsize=(12, 9), layout="constrained")
    figure.suptitle("Source cuff at the end of the ramp and after a target hold", fontsize=17)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.2, 1])
    colors = ("#c17a25", "#267692")
    final_positions = [np.asarray(run[2][-1]["positionsMeters"]) * 1000 for run in (baseline, hold)]
    bounds = np.vstack(final_positions)
    lower, upper = bounds.min(axis=0) - 3, bounds.max(axis=0) + 3
    faces = np.asarray(hold[1]["triangles"]).reshape((-1, 3))
    for index, (points, title) in enumerate(zip(final_positions, ("64 ms · end of target ramp", "256 ms · after 192 ms hold"))):
        axis = figure.add_subplot(grid[0, index], projection="3d")
        for first, last, color in ((0, 48, colors[0]), (48, 96, colors[1])):
            axis.add_collection3d(Poly3DCollection(points[faces[first:last]], facecolor=color,
                edgecolor=color, linewidth=.4, alpha=.55))
        axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
                 xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title=title)
        axis.set_box_aspect(upper - lower)
        axis.view_init(elev=25, azim=-63)
        axis.tick_params(labelsize=8, pad=0)
    times, angles, speeds = hold[3] * 1000, hold[4], hold[5]
    angle_axis, speed_axis = figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])
    for first, last, color, label in ((0, 12, colors[0], "Shell"), (12, 24, colors[1], "Facing")):
        subset = angles[:, first:last]
        angle_axis.fill_between(times, subset.min(axis=1), subset.max(axis=1), color=color, alpha=.18)
        angle_axis.plot(times, subset.mean(axis=1), color=color, label=label + " mean / range")
    targets = np.degrees([row["record"]["step"]["foldTargetsRadians"][0] for row in hold[2]])
    angle_axis.plot(times, targets, "--", color="#222222", label="Prescribed target")
    angle_lower, angle_upper = min(angles.min(), targets.min()), max(angles.max(), targets.max())
    angle_padding = max(2., .05 * (angle_upper - angle_lower))
    angle_axis.set(xlabel="Physical time (ms)", ylabel="Allowance hinge angle (degrees)",
                   ylim=(angle_lower - angle_padding, angle_upper + angle_padding))
    angle_axis.legend(loc="lower right", frameon=False, fontsize=9)
    speed_axis.semilogy(times, speeds, color="#344352")
    speed_axis.set(xlabel="Physical time (ms)", ylabel="Maximum vertex speed (m/s)")
    for axis in (angle_axis, speed_axis):
        axis.axvline(64, color="#777777", linewidth=1, linestyle=":")
        axis.set_xlim(0, 256)
        axis.grid(alpha=.18)
    figure.supxlabel("Actual saved geometry and measurements · 66 vertices / 96 triangles · accepted: false\n"
                      "Uncalibrated parallel-layer diagnostic; no turning, sleeve attachment or fit claim", fontsize=10)
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=160)
    plt.close(figure)
    output = {"ledger": str(args.output_prefix.with_suffix('.json')),
              "figure": str(args.output_prefix.with_suffix('.png')), "accepted": False}
    if distance is not None:
        comparison_path = args.output_prefix.with_name(args.output_prefix.stem + "-seams.png")
        plot_distance_comparison(hold, distance, comparison_path)
        output["seamComparisonFigure"] = str(comparison_path)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
