"""Report replay-verified cuff frame/contact sensitivity without garment acceptance.

Uses the plotting-only matplotlib==3.10.7 environment. Repeat --body-hold for
distinct activation distances; the 1 mm body-frame control is mandatory.
The older three-run hold report and its figures are not changed by this script.
"""

import argparse
import difflib
import importlib.util
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError("Evidence reporting requires enabled verification assertions")

HELPER = Path(__file__).with_name("report-cuff-hold.py")
spec = importlib.util.spec_from_file_location("cuff_hold_report", HELPER)
hold_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hold_report)
np, plt = hold_report.np, hold_report.plt
Poly3DCollection = hold_report.Poly3DCollection
digest = hold_report.digest
REVIEWED_REPLAY_VERSIONS = {
    "cf51e0cd2dd7027c753728c65bd4da1fcc2ad8f147c1e7f98efc3cbd242dd48a": "Historical fixed 100 s soft / 105 s hard CPU budget; no recorded replay CPU fields.",
    "c6fc6a2434c11d1e8df4e6d08540696c4cd5503ac30fd8162467c65041f6e96a": "Reviewed bounded CPU CLI/reporting revision (commit 753673e); exact replay mathematics unchanged.",
}


def differences(before, after, path=""):
    """Exact JSON-value differences; absent members are distinguished from null."""
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before:
                result.append({"path": child, "after": after[key]})
            elif key not in after:
                result.append({"path": child, "before": before[key]})
            else:
                result.extend(differences(before[key], after[key], child))
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [entry for index, (left, right) in enumerate(zip(before, after))
                for entry in differences(left, right, path + "/" + str(index))]
    return [] if type(before) is type(after) and before == after else [{
        "path": path, "before": before, "after": after}]


def without(value, *fields):
    return {key: item for key, item in value.items() if key not in fields}


def body_frame_policy(source):
    """Check the declared body region against unchanged source triangles/anchors."""
    frames = source["sewingFrames"]
    assert frames["accepted"] is False
    assert frames["recipe"] == "explicit-facing-source-parent-region-v2"
    assert frames["creaseFrameRegion"] == "body"
    assert frames["selectionPolicy"] == "requested crease region, then lexicographic source-parent and child vertex identities"
    constraints = source["embeddedConstraints"]["constraints"]
    assert len(frames["faces"]) == len(frames["bindings"]) == len(constraints) == 20
    assert frames["sides"] == [-1] * 20
    subdivision = source["provenance"]["subdivision"]
    vertices = np.asarray(subdivision["vertices"])
    triangles = subdivision["triangles"]
    assert np.array_equal(np.asarray(source["triangles"]).reshape((-1, 3))[:48], triangles)
    assert np.array_equal(np.asarray(source["restMeters"])[:33, :2], vertices)
    halfspace = frames["bodyHalfspace"]
    assert halfspace["sourcePathName"] == "outer"
    line = np.asarray(halfspace["creaseEndpointsMeters"])
    samples = subdivision["sourceStitchPath"]["samples"]
    assert np.allclose(line, np.asarray([samples[0]["restPosition"], samples[-1]["restPosition"]]) / 1000,
                       rtol=0, atol=1e-15)
    direction = line[1] - line[0]
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    assert np.allclose(normal, halfspace["leftNormal"], rtol=0, atol=1e-15)
    tolerance = halfspace["toleranceMeters"]
    assert np.isfinite(tolerance) and 0 < tolerance < 1e-10
    sign = halfspace["sign"]
    assert sign in (-1, 1)
    off_crease_signs, crease_rows = set(), []
    for index, (face, binding, constraint) in enumerate(zip(frames["faces"], frames["bindings"], constraints)):
        assert binding["instanceId"] == "cuff_left:facing" and binding["region"] == "body"
        child = [vertex - 33 for vertex in face]
        assert child == binding["childVertices"] and child in triangles
        parent = subdivision["parentTriangles"][triangles.index(child)]
        assert parent == binding["sourceParentTriangle"]
        parent_vertices = {int(vertex) for triangle, ancestor in zip(triangles, subdivision["parentTriangles"])
                           if ancestor == parent for child_vertex in triangle
                           for vertex, weight in subdivision["sourceWeights"][child_vertex].items() if weight > 0}
        assert sorted(parent_vertices) == binding["sourceParentVertices"] and len(parent_vertices) == 3
        negative = [term for term in constraint["terms"] if term["coefficient"] < 0]
        assert negative and all(term["instanceId"] == "cuff_left:facing" for term in negative)
        support = sorted(term["vertex"] for term in negative)
        assert support == binding["negativeAnchorVertices"] and set(support).issubset(child)
        anchor = sum(-term["coefficient"] * vertices[term["vertex"]] for term in negative)
        distance = float((anchor - line[0]) @ normal)
        on_crease = abs(distance) <= tolerance
        assert binding["onCrease"] is on_crease
        if on_crease:
            crease_rows.append(index)
        else:
            off_crease_signs.add(1 if distance > 0 else -1)
        child_distances = (vertices[child] - line[0]) @ normal
        strict_sides = child_distances[np.abs(child_distances) > tolerance] * sign
        assert len(strict_sides) > 0 and np.all(strict_sides > 0)
    assert off_crease_signs == {sign}
    assert crease_rows == list(range(9, 16))
    return {"metadata": frames, "creaseRegistrationIndices": crease_rows,
            "validation": "Every facing frame is an oriented source child of its stated parent, contains its negative anchor support, and lies on the declared body side. Off-crease anchors independently identify that side."}


def load_run(path):
    run = hold_report.load_run(path)
    report, source, states, times, angles, speeds, summary = run
    args = report["arguments"]
    assert args["sewing_mode"] == "normal-offset"
    assert args["step_seconds"] == .256 and args["subdivisions"] == 256
    assert args["minimum_distance_m"] == .0001 and args["pressure_pa"] == 10000.
    assert np.isfinite(args["activation_distance_m"]) and args["activation_distance_m"] > 0
    assert args["assembly_schedule"] is args["fold_actuation"] is True
    assert source["assemblySchedule"] == {"profile": "sewing-fold-progress-v1", "knots": [
        {"fraction": fraction, "sewingProgress": sewing, "foldProgress": fold}
        for fraction, sewing, fold in zip([0., .0625, .25, 1.], [0., 1., 1., 1.], [0., 0., 1., 1.])]}
    profile = report["contactProfile"]
    assert profile["fullActivationM"] == args["activation_distance_m"]
    assert profile["fullMinimumM"] == args["minimum_distance_m"]
    assert profile["pressurePa"] == args["pressure_pa"]
    assert profile["filteredPrimitivePairs"] == 0
    placement = json.loads((path / "placement.json").read_text())
    assert set(placement) == {"canonicalDigest", "placedMeters"}
    assert placement["canonicalDigest"] == report["canonicalSha256"]
    assert placement["placedMeters"] == source["placedMeters"]
    for field, module in (("triangleVerifierSha256", "solver_triangle_sweep.py"),
                          ("broadPhaseVerifierSha256", "solver_ipc_broad_phase.py"),
                          ("candidateCoverageVerifierSha256", "solver_candidate_coverage.py")):
        assert summary["verificationDigests"][field] == report["sourceDigests"][module]
    for state in states:
        assert np.asarray(state["positionsMeters"]).shape == (66, 3)
        assert np.asarray(state["velocitiesMetersPerSecond"]).shape == (66, 3)
        assert np.isfinite(state["positionsMeters"]).all()
        assert np.isfinite(state["velocitiesMetersPerSecond"]).all()
    summary["finalContactEnergyJ"] = report["finalContactEnergyJ"]
    summary["finalPeakContactForceN"] = report["finalPeakContactForceN"]
    summary["sourceDigests"] = report["sourceDigests"]
    summary["sewingFrameRecipe"] = source["sewingFrames"]["recipe"]
    replay = json.loads((path / "verified-replay.json").read_text())
    assert replay["replayScriptSha256"] in REVIEWED_REPLAY_VERSIONS
    recorded_limit = replay.get("verificationCpuLimitSeconds")
    recorded_cpu = replay.get("verificationCpuSeconds")
    if recorded_limit is not None or recorded_cpu is not None:
        assert type(recorded_limit) is int and 1 <= recorded_limit <= 3600
        assert np.isfinite(recorded_cpu) and 0 <= recorded_cpu <= recorded_limit + 5
    summary["replayBudget"] = {"recordedCpuLimitSeconds": recorded_limit, "recordedCpuSeconds": recorded_cpu,
        "interpretation": "Null means the historical replay artifact did not record this quantity; it is not zero. Replay verification budgets are separate from captured simulation budgets."}
    return run


def time_comparison(before, after):
    left, right = before[3], after[3]
    prefix = 0
    for earlier, later in zip(left, right):
        if earlier != later:
            break
        prefix += 1
    return {"sameAcceptedTimePartition": bool(np.array_equal(left, right)),
            "sameEndTimesPrefixCount": prefix,
            "sameEndTimesPrefixLastSeconds": float(left[prefix - 1]) if prefix else 0.,
            "commonEndTimeCount": len(set(left) & set(right)),
            "before": before[-1]["acceptedTimePartition"], "after": after[-1]["acceptedTimePartition"],
            "interpretation": "Matching times do not imply matching states. Adaptive integration and implicit numerical dissipation can differ; this is not a fixed-substep or timestep-convergence comparison."}


def frame_comparison(historical, body):
    old_frames = historical[1]["sewingFrames"]
    assert set(old_frames) == {"accepted", "recipe", "faces", "sides"}
    assert old_frames["accepted"] is False and old_frames["recipe"] == "remapped-facing-source-parent-v1"
    assert len(old_frames["faces"]) == 20 and old_frames["sides"] == [-1] * 20
    old_triangles = np.asarray(historical[1]["triangles"]).reshape((-1, 3)).tolist()
    assert all(face in old_triangles[48:] for face in old_frames["faces"])
    assert historical[0]["arguments"]["activation_distance_m"] == body[0]["arguments"]["activation_distance_m"] == .001
    assert without(historical[1], "sewingFrames", "provenance") == without(body[1], "sewingFrames", "provenance")
    assert without(historical[1]["provenance"], "holdDiagnostic") == without(body[1]["provenance"], "holdControl")
    assert historical[1]["sewingFrames"]["sides"] == body[1]["sewingFrames"]["sides"]
    assert without(historical[0]["arguments"], "canonical", "placement", "output") == without(
        body[0]["arguments"], "canonical", "placement", "output")
    assert historical[0]["contactProfile"] == body[0]["contactProfile"]
    verifier_changes = verifier_comparison(historical, body)
    code_changes = differences(historical[0]["sourceDigests"], body[0]["sourceDigests"])
    assert [change["path"] for change in code_changes] == ["/solver_cuff_sequence.py"]
    policy = body_frame_policy(body[1])
    changed_rows = [index for index, (before, after) in enumerate(zip(
        historical[1]["sewingFrames"]["faces"], body[1]["sewingFrames"]["faces"])) if before != after]
    assert changed_rows == policy["creaseRegistrationIndices"]
    return {"accepted": False, "beforeRun": historical[-1]["runDirectory"], "afterRun": body[-1]["runDirectory"],
        "canonicalDifferences": differences(historical[1], body[1]),
        "argumentDifferencesExcludingFilePaths": [], "contactProfileDifferences": [],
        "sourceDigestDifferences": code_changes, "changedFrameRegistrationIndices": changed_rows,
        "replayVerificationComparison": verifier_changes,
        "sourceFramePolicy": policy, "timePartition": time_comparison(historical, body),
        "interpretation": "Seven on-crease material-normal reference triangles move from allowance children to body children. Source metrics, mesh, sewing anchors/compliance, fold targets, initial positions and contact are identical. Only the input-construction helper changes in captured code; every solver/contact module and dependency requirement is identical."}


def activation_comparison(reference, alternate):
    left, right = reference[0], alternate[0]
    assert reference[1] == alternate[1]
    for field in ("canonicalSha256", "placementSha256", "sourceDigests"):
        assert left[field] == right[field]
    verifier_changes = verifier_comparison(reference, alternate)
    assert without(left["arguments"], "canonical", "placement", "output", "activation_distance_m") == without(
        right["arguments"], "canonical", "placement", "output", "activation_distance_m")
    assert without(left["contactProfile"], "fullActivationM") == without(right["contactProfile"], "fullActivationM")
    assert left["arguments"]["activation_distance_m"] != right["arguments"]["activation_distance_m"]
    return {"accepted": False, "beforeRun": reference[-1]["runDirectory"], "afterRun": alternate[-1]["runDirectory"],
        "canonicalDifferences": [], "sourceDigestDifferences": [],
        "argumentDifferencesExcludingFilePaths": differences(without(left["arguments"], "canonical", "placement", "output"),
                                                              without(right["arguments"], "canonical", "placement", "output")),
        "contactProfileDifferences": differences(left["contactProfile"], right["contactProfile"]),
        "replayVerificationComparison": verifier_changes,
        "timePartition": time_comparison(reference, alternate),
        "interpretation": "Only the contact activation width beyond the exclusion core changes (outer force-support distance = core + activation). This changes the constitutive contact energy/force model; the 0.1 mm exclusion core, 10000 Pa pressure, source frame policy and all other inputs remain fixed. This is model sensitivity, not calibration, tolerance relaxation or an accepted construction."}


def verifier_comparison(before, after):
    left, right = before[-1], after[-1]
    assert all(run["verificationDigests"]["replayScriptSha256"] in REVIEWED_REPLAY_VERSIONS for run in (left, right))
    assert without(left["verificationDigests"], "replayScriptSha256") == without(right["verificationDigests"], "replayScriptSha256")
    return {"digestDifferences": differences(left["verificationDigests"], right["verificationDigests"]),
        "beforeBudget": left["replayBudget"], "afterBudget": right["replayBudget"],
        "interpretation": "The triangle, broad-phase and candidate-coverage verifier modules are identical. The standalone replay script hash and verification budget may differ and are recorded explicitly; identical replayer bytes are not claimed."}


def replay_attempt_history(runs, paths):
    result = []
    for run in runs:
        path = paths[run[-1]["runDirectory"]]
        previous_log = path.with_name(path.name + "-replay.log")
        completed_log = path.with_name(path.name + "-replay-600.log")
        if not (previous_log.exists() and completed_log.exists()):
            continue
        previous_text = previous_log.read_text()
        resource_rows = [json.loads(line) for line in completed_log.read_text().splitlines() if line.startswith("{")]
        resources = next(row for row in resource_rows if "maximumResidentKiB" in row)
        completion = next(row for row in resource_rows if "verifiedStates" in row)
        assert '"verifiedStates"' not in previous_text
        assert resources["exitCode"] == 0 and completion["verifiedStates"] == run[-1]["acceptedSteps"]
        assert completion["exactCertificateLeaves"] == run[-1]["exactContactLeaves"]
        assert run[-1]["replayBudget"]["recordedCpuLimitSeconds"] == 600
        assert 105 < run[-1]["replayBudget"]["recordedCpuSeconds"] <= resources["cpuSeconds"]
        result.append({"run": path.name, "initialAttempt": {
            "stdoutLog": previous_log.name, "stdoutLogSha256": digest(previous_log),
            "cpuSoftLimitSeconds": 100, "cpuHardLimitSeconds": 105,
            "observation": "The task supervisor reported exit 137 with no completed verification artifact. The saved stdout log contains no completed-proof record and does not itself record the exit status or signal cause."},
            "completedAttempt": {"log": completed_log.name, "logSha256": digest(completed_log),
                "wrapperResources": resources, "recordedVerifierBudget": run[-1]["replayBudget"]},
            "interpretation": "The successful verifier CPU time exceeds the earlier 105 s hard limit. A larger verification budget was needed; no memory failure is evidenced. Solver captures and exact proof mathematics were unchanged. The standalone replay CLI gained an explicit bounded CPU option and recorded CPU metadata; its changed digest is preserved separately."})
    return result


def plot_runs(runs, destination):
    count = len(runs)
    figure = plt.figure(figsize=(5 * count, 11), layout="constrained")
    figure.suptitle("Cuff frame and contact sensitivity · captured 256 ms controls", fontsize=17)
    grid = figure.add_gridspec(3, count, height_ratios=[1.15, 1., .85])
    colors = ("#c17a25", "#267692")
    positions = [np.asarray(run[2][-1]["positionsMeters"]) * 1000 for run in runs]
    bounds = np.vstack(positions)
    lower, upper = bounds.min(axis=0) - 3, bounds.max(axis=0) + 3
    faces = np.asarray(runs[0][1]["triangles"]).reshape((-1, 3))
    target_curves = [np.degrees([state["record"]["step"]["foldTargetsRadians"][0] for state in run[2]]) for run in runs]
    all_angles = np.concatenate([run[4].ravel() for run in runs] + target_curves)
    bottom, top = all_angles.min(), all_angles.max()
    padding = max(2., .05 * (top - bottom))
    for index, (run, points, target) in enumerate(zip(runs, positions, target_curves)):
        activation = run[0]["arguments"]["activation_distance_m"] * 1000
        name = ("Allowance-side crease frames" if index == 0 else "Body-side crease frames") + f"\n{activation:g} mm activation buffer"
        geometry = figure.add_subplot(grid[0, index], projection="3d")
        for first, last, color in ((0, 48, colors[0]), (48, 96, colors[1])):
            geometry.add_collection3d(Poly3DCollection(points[faces[first:last]], facecolor=color,
                edgecolor=color, linewidth=.4, alpha=.55))
        geometry.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
                     xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title=name)
        geometry.set_box_aspect(upper - lower)
        geometry.view_init(elev=25, azim=-63)
        geometry.tick_params(labelsize=8, pad=0)
        angle_axis = figure.add_subplot(grid[1, index])
        times = run[3] * 1000
        for first, last, color, layer in ((0, 12, colors[0], "Shell"), (12, 24, colors[1], "Facing")):
            subset = run[4][:, first:last]
            angle_axis.fill_between(times, subset.min(axis=1), subset.max(axis=1), color=color, alpha=.18)
            angle_axis.plot(times, subset.mean(axis=1), color=color, label=layer + " mean / range")
        angle_axis.plot(times, target, "--", color="#222222", label="Prescribed target")
        angle_axis.set(ylim=(bottom - padding, top + padding), ylabel="Allowance hinge angle (degrees)")
        angle_axis.legend(loc="lower right", frameon=False, fontsize=8)
        speed_axis = figure.add_subplot(grid[2, index])
        # Linear scaling displays zero speeds without silently dropping samples.
        speed_axis.plot(times, run[5], color="#344352")
        speed_axis.set(ylabel="Maximum vertex speed (m/s)", ylim=(0, 1.05 * max(max(item[5]) for item in runs)))
        speed_axis.set_title(f"{run[-1]['acceptedSteps']} accepted / {run[-1]['rejectedAttempts']} rejected steps\n"
                            f"Final maximum speed: {run[-1]['finalMaximumSpeedMps']:.5f} m/s", fontsize=9)
        for axis in (angle_axis, speed_axis):
            axis.axvline(64, color="#777777", linewidth=1, linestyle=":")
            axis.set(xlim=(0, 256), xlabel="Physical time (ms)")
            axis.grid(alpha=.18)
    figure.supxlabel("Actual saved geometry and trajectories · same 0.1 mm core / 10000 Pa pressure · accepted: false\n"
        "Activation width is beyond the core; force support ends at core + width\n"
        "Finite-duration, uncalibrated model sensitivity; adaptive histories may differ; no equilibrium, turning or fit claim", fontsize=10)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-hold", type=Path, required=True)
    parser.add_argument("--body-hold", type=Path, action="append", required=True,
                        help="Repeat for distinct activation distances; include the 1 mm body-frame control")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    assert len({path.resolve() for path in [args.historical_hold, *args.body_hold]}) == 1 + len(args.body_hold)
    historical = load_run(args.historical_hold)
    bodies = sorted((load_run(path) for path in args.body_hold), key=lambda run: -run[0]["arguments"]["activation_distance_m"])
    activations = [run[0]["arguments"]["activation_distance_m"] for run in bodies]
    assert len(set(activations)) == len(activations) and .001 in activations
    reference = bodies[activations.index(.001)]
    frame = frame_comparison(historical, reference)
    activation_comparisons = [activation_comparison(reference, run) for run in bodies if run is not reference]
    paths = {path.name: path for path in [args.historical_hold, *args.body_hold]}
    assert len(paths) == 1 + len(bodies)
    helper_name = "solver_cuff_sequence.py"
    frame["inputHelperUnifiedDiff"] = "".join(difflib.unified_diff(
        (args.historical_hold / "source-snapshot" / helper_name).read_text().splitlines(keepends=True),
        (paths[reference[-1]["runDirectory"]] / "source-snapshot" / helper_name).read_text().splitlines(keepends=True),
        fromfile="historical/" + helper_name, tofile="body-frame/" + helper_name))
    result = {"profile": "synthetic-cuff-frame-contact-sensitivity-evidence-v1", "accepted": False,
        "runtime": json.loads(args.runtime.read_text()), "runtimeManifestSha256": digest(args.runtime),
        "runtimeEvidence": "Separately captured runtime metadata; run reports do not cryptographically bind this manifest.",
        "verificationScope": "Checks captured source, input, state and replay hashes, completed replay proofs and source-frame metadata. Does not re-execute numerical or exact path proofs.",
        "reportGeneratorSha256": digest(Path(__file__)), "reportHelperSha256": digest(HELPER),
        "reviewedReplayVersions": REVIEWED_REPLAY_VERSIONS,
        "runs": {"historicalHold": historical[-1], "bodyFrameHolds": [run[-1] for run in bodies]},
        "frameComparison": frame, "activationComparisons": activation_comparisons,
        "replayAttemptHistory": replay_attempt_history(bodies, paths),
        "limitations": ["Synthetic coarse cuff with parallel shell/facing; no turning, sleeve attachment, layer-order acceptance or physical fit claim.",
            "The activation distance is an uncalibrated model parameter. A smaller value changes contact energy and forces; it does not validate the material or establish a preferred setting.",
            "The unchanged sewing/fold ramps end at 64 ms, followed by a 192 ms target hold. No physical damping is added.",
            "Motion at the final sample is reported, not assumed zero. Finite-duration controls do not establish equilibrium, temporal convergence or mesh convergence.",
            "Accepted integration steps are numerically admitted states, not accepted garment constructions. All runs retain accepted: false.",
            "Exact path/candidate/triangle checks do not guarantee layer order or thickness at incident primitives.",
            "Raw source snapshots and trajectories remain in ignored research storage. Recorded hashes cannot reconstruct missing bytes."]}
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_prefix.with_suffix(".json").write_text(encoded)
    plot_runs([historical, *bodies], args.output_prefix.with_suffix(".png"))
    print(json.dumps({"ledger": str(args.output_prefix.with_suffix(".json")),
                      "figure": str(args.output_prefix.with_suffix(".png")), "accepted": False}))


if __name__ == "__main__":
    main()
