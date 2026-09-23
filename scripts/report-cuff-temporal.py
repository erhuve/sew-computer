"""Report two replay-verified cuff timesteps at exact shared saved times.

Plotting environment: matplotlib==3.10.7. This reports temporal sensitivity,
not asymptotic convergence, equilibrium, calibrated material or acceptance.
"""

import argparse
from collections import Counter
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError("Evidence reporting requires enabled verification assertions")

HELPER = Path(__file__).with_name("report-cuff-hold.py")
spec = importlib.util.spec_from_file_location("cuff_hold_report", HELPER)
hold_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hold_report)
np, plt, digest = hold_report.np, hold_report.plt, hold_report.digest
REPLAY_SHA256 = "c6fc6a2434c11d1e8df4e6d08540696c4cd5503ac30fd8162467c65041f6e96a"


def without(value, *fields):
    return {key: item for key, item in value.items() if key not in fields}


def hinge_angles(points, hinges):
    p = np.asarray(points)[np.asarray(hinges)]
    first = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])
    second = np.cross(p[:, 3] - p[:, 1], p[:, 2] - p[:, 1])
    edge = p[:, 3] - p[:, 2]
    for vectors in (first, second, edge):
        lengths = np.linalg.norm(vectors, axis=1)
        assert np.isfinite(lengths).all() and np.all(lengths > 0)
        vectors /= lengths[:, None]
    return np.arctan2(np.sum(np.cross(first, second) * edge, axis=1), np.sum(first * second, axis=1))


def load_run(directory, replay_log):
    # This mandatory audited loader verifies every captured source/state digest
    # and completed replay proof. A missing replay artifact is always an error.
    run = hold_report.load_run(directory)
    report, source, states, times, recorded_angles, speeds, summary = run
    replay = json.loads((directory / "verified-replay.json").read_text())
    assert replay["replayScriptSha256"] == REPLAY_SHA256
    for field, module in (("triangleVerifierSha256", "solver_triangle_sweep.py"),
                          ("broadPhaseVerifierSha256", "solver_ipc_broad_phase.py"),
                          ("candidateCoverageVerifierSha256", "solver_candidate_coverage.py")):
        assert replay[field] == report["sourceDigests"][module]
    assert type(replay["verificationCpuLimitSeconds"]) is int
    assert 1 <= replay["verificationCpuLimitSeconds"] <= 3600
    assert np.isfinite(replay["verificationCpuSeconds"])
    assert 0 <= replay["verificationCpuSeconds"] <= replay["verificationCpuLimitSeconds"] + 5
    placement = json.loads((directory / "placement.json").read_text())
    assert set(placement) == {"placedMeters", "canonicalDigest"}
    assert placement["canonicalDigest"] == report["canonicalSha256"]
    assert placement["placedMeters"] == source["placedMeters"]
    descriptor = report["attemptJournal"]["initialState"]
    assert digest(directory / descriptor["path"]) == descriptor["sha256"]
    initial = json.loads((directory / descriptor["path"]).read_text())
    assert initial["accepted"] is False
    assert initial["positionsMeters"] == placement["placedMeters"]
    assert np.array_equal(initial["velocitiesMetersPerSecond"], np.zeros((66, 3)))
    hinges = source["foldActuation"]["hinges"]
    trajectory = {}
    for state in [initial, *states]:
        fraction = Fraction(state["record"]["endFraction"]) if "record" in state else Fraction(0)
        assert fraction not in trajectory and 0 <= fraction <= 1
        q = np.asarray(state["positionsMeters"])
        v = np.asarray(state["velocitiesMetersPerSecond"])
        assert q.shape == v.shape == (66, 3) and np.isfinite(q).all() and np.isfinite(v).all()
        angles = hinge_angles(q, hinges)
        assert angles.shape == (24,) and np.isfinite(angles).all()
        if "record" in state:
            np.testing.assert_allclose(angles, state["record"]["step"]["foldAnglesRadians"], rtol=0, atol=2e-14)
        trajectory[fraction] = {"q": q, "v": v, "a": angles}
    fractions = list(trajectory)
    assert fractions == sorted(fractions) and fractions[0] == 0 and fractions[-1] == 1
    step_counts = Counter(after - before for before, after in zip(fractions, fractions[1:]))
    final = trajectory[Fraction(1)]
    faces = np.asarray(source["triangles"]).reshape((-1, 3))
    edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1), axis=0)
    rest = np.asarray(source["restMeters"])
    ratios = np.linalg.norm(final["q"][edges[:, 1]] - final["q"][edges[:, 0]], axis=1) / np.linalg.norm(
        rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    assert np.isfinite(ratios).all()
    for strain in (report["edgeStrain"], replay["edgeStrain"]):
        assert strain["edgeCount"] == len(edges)
        np.testing.assert_allclose([ratios.min(), ratios.max()], [strain["min"], strain["max"]], rtol=0, atol=1e-14)
    rows = [json.loads(line) for line in replay_log.read_text().splitlines() if line.startswith("{")]
    proofs = [row for row in rows if "verifiedStates" in row]
    resources = [row for row in rows if "maximumResidentKiB" in row]
    assert len(proofs) == len(resources) == 1
    proof, resource = proofs[0], resources[0]
    assert proof["verifiedStates"] == len(states)
    assert proof["exactCertificateLeaves"] == summary["exactContactLeaves"]
    assert proof["maxResidualN"] == summary["maximumReconstructedResidualN"]
    assert resource["exitCode"] == 0
    assert resource["cpuSeconds"] >= replay["verificationCpuSeconds"]
    assert resource["wallSeconds"] > 0 and resource["maximumResidentKiB"] > 0
    summary = without(summary, "sourceProvenance", "assemblySchedule", "arguments")
    summary.update({"initialState": descriptor,
        "acceptedStepHistogram": [{"durationFraction": str(step),
            "durationSeconds": float(step) * report["arguments"]["step_seconds"], "count": count}
            for step, count in sorted(step_counts.items())],
        "recomputedFinalEdgeLengthRatioRange": [float(ratios.min()), float(ratios.max())],
        "finalContactEnergyJ": report["finalContactEnergyJ"], "finalPeakContactForceN": report["finalPeakContactForceN"],
        "replayResources": {"recordedVerifierCpuLimitSeconds": replay["verificationCpuLimitSeconds"],
            "recordedVerifierCpuSeconds": replay["verificationCpuSeconds"],
            "wrapper": resource, "log": replay_log.name, "logSha256": digest(replay_log)},
        "angleReconstruction": "Recomputed normalized face-normal atan2 angles from every saved geometry; agreement with captured angles within 2e-14 radians."})
    return {"run": run, "trajectory": trajectory, "summary": summary}


def validate_pair(coarse, fine):
    left, right = coarse["run"], fine["run"]
    assert left[1] == right[1]
    for field in ("canonicalSha256", "placementSha256", "sourceDigests", "contactProfile", "assemblySchedule", "foldActuation"):
        assert left[0][field] == right[0][field]
    assert left[-1]["verificationDigests"] == right[-1]["verificationDigests"]
    args_left, args_right = left[0]["arguments"], right[0]["arguments"]
    assert set(args_left) == set(args_right)
    assert {key for key in args_left if args_left[key] != args_right[key]} == {"subdivisions", "output"}
    assert without(args_left, "subdivisions", "output") == without(args_right, "subdivisions", "output")
    assert args_left["subdivisions"] == 256 and args_right["subdivisions"] == 512
    assert args_left["step_seconds"] == .256 and args_left["sewing_mode"] == "normal-offset"
    assert args_left["activation_distance_m"] == args_left["minimum_distance_m"] == .0001
    assert args_left["pressure_pa"] == 10000.
    assert args_left["assembly_schedule"] is args_left["fold_actuation"] is True
    assert left[1]["sewingFrames"]["recipe"] == "explicit-facing-source-parent-region-v2"
    assert left[1]["sewingFrames"]["creaseFrameRegion"] == "body"
    assert left[1]["assemblySchedule"] == {"profile": "sewing-fold-progress-v1", "knots": [
        {"fraction": fraction, "sewingProgress": sewing, "foldProgress": fold}
        for fraction, sewing, fold in zip([0., .0625, .25, 1.], [0., 1., 1., 1.], [0., 0., 1., 1.])]}
    profile = left[0]["contactProfile"]
    assert profile["filteredPrimitivePairs"] == 0
    assert profile["fullActivationM"] == profile["fullMinimumM"] == .0001
    assert profile["pressurePa"] == 10000.
    assert set(coarse["trajectory"]) == {Fraction(index, 256) for index in range(257)}
    assert all(fraction in fine["trajectory"] for fraction in coarse["trajectory"])


def state_difference(coarse, fine, fraction, duration):
    dp = np.linalg.norm(coarse["q"] - fine["q"], axis=1) * 1000
    dv = np.linalg.norm(coarse["v"] - fine["v"], axis=1)
    da = np.abs(np.degrees(coarse["a"] - fine["a"]))
    speeds = [np.linalg.norm(state["v"], axis=1) for state in (coarse, fine)]
    return {"fraction": str(fraction), "timeMs": float(fraction) * duration * 1000,
        "maximumPositionDifferenceMm": float(dp.max()), "rmsPositionDifferenceMm": float(np.sqrt(np.mean(dp ** 2))),
        "maximumVelocityVectorDifferenceMPerS": float(dv.max()), "rmsVelocityVectorDifferenceMPerS": float(np.sqrt(np.mean(dv ** 2))),
        "maximumSpeedDifferenceMPerS": float(np.abs(speeds[0] - speeds[1]).max()),
        "coarseMaximumSpeedMPerS": float(speeds[0].max()), "fineMaximumSpeedMPerS": float(speeds[1].max()),
        "maximumAngleDifferenceDegrees": float(da.max()), "rmsAngleDifferenceDegrees": float(np.sqrt(np.mean(da ** 2))),
        "shellMaximumAngleDifferenceDegrees": float(da[:12].max()), "facingMaximumAngleDifferenceDegrees": float(da[12:].max()),
        "worstPositionVertex": int(dp.argmax()), "worstAngleHinge": int(da.argmax())}


def compare(coarse, fine):
    validate_pair(coarse, fine)
    fractions = sorted(coarse["trajectory"])
    return [state_difference(coarse["trajectory"][fraction], fine["trajectory"][fraction], fraction,
                             coarse["run"][0]["arguments"]["step_seconds"]) for fraction in fractions]


def plot(comparisons, destination):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    times = [row["timeMs"] for row in comparisons]
    specifications = [
        (axes[0, 0], [("maximumPositionDifferenceMm", "Maximum"), ("rmsPositionDifferenceMm", "RMS")], "World-position difference (mm)"),
        (axes[0, 1], [("shellMaximumAngleDifferenceDegrees", "Shell maximum"), ("facingMaximumAngleDifferenceDegrees", "Facing maximum")], "Hinge-angle difference (degrees)"),
        (axes[1, 0], [("maximumVelocityVectorDifferenceMPerS", "Maximum vector difference"), ("rmsVelocityVectorDifferenceMPerS", "RMS vector difference")], "Velocity-vector difference (m/s)"),
        (axes[1, 1], [("coarseMaximumSpeedMPerS", "1 ms initial steps"), ("fineMaximumSpeedMPerS", "0.5 ms initial steps")], "Maximum vertex speed (m/s)"),
    ]
    for axis, fields, ylabel in specifications:
        for field, label in fields:
            axis.plot(times, [row[field] for row in comparisons], label=label)
        axis.set(xlim=(0, 256), ylim=(0, None), xlabel="Physical time (ms)", ylabel=ylabel)
        axis.axvline(64, color="#777777", linestyle=":", linewidth=1)
        axis.grid(alpha=.18)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Cuff timestep sensitivity · exact common saved times", fontsize=17)
    final = comparisons[-1]
    figure.supxlabel(f"Final maximum differences: {final['maximumPositionDifferenceMm']:.4f} mm position · "
        f"{final['maximumAngleDifferenceDegrees']:.5f}° hinge angle · {final['maximumVelocityVectorDifferenceMPerS']:.5f} m/s velocity\n"
        "Unaligned world coordinates; no interpolation · 256 shared positive samples · accepted: false\n"
        "Two temporal discretizations do not establish convergence, equilibrium, material calibration, turning or fit", fontsize=10)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("coarse", "fine", "coarse-replay-log", "fine-replay-log", "runtime", "output-prefix"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    assert args.coarse.resolve() != args.fine.resolve()
    coarse, fine = load_run(args.coarse, args.coarse_replay_log), load_run(args.fine, args.fine_replay_log)
    comparisons = compare(coarse, fine)
    report, source = coarse["run"][:2]
    metrics = [key for key in comparisons[0] if key not in ("fraction", "timeMs", "worstPositionVertex", "worstAngleHinge")]
    result = {"profile": "synthetic-cuff-two-timestep-sensitivity-evidence-v1", "accepted": False,
        "scope": "Two time discretizations at exact common saved fractions; no asymptotic convergence, settled-state, material calibration, turning or garment acceptance claim.",
        "verificationScope": "Mandatory audited capture/replay hash and proof-result checks; independent saved-geometry angle/edge reconstruction. Does not rerun solver or exact path proofs.",
        "positionComparison": "Unaligned world coordinates from identical source and placement; no interpolation or temporal resampling.",
        "metricDefinitions": {"RMS": "Square root of the mean squared per-vertex Euclidean difference (positions/velocities) or per-hinge angle difference.",
            "sampling": "Comparison rows, trajectoryMaxima and plotted curves use only the 257 exact common saved times, including the initial state. They do not bound continuous-time differences or summarize fine-only samples.",
            "angle": "Absolute difference of independently reconstructed signed principal hinge angles, in degrees.",
            "speed": "Per-vertex Euclidean velocity norm; speed difference and velocity-vector difference are distinct metrics.",
            "indices": "Zero-based source vertex/hinge indices; shell hinges 0–11, facing hinges 12–23."},
        "runtime": json.loads(args.runtime.read_text()), "runtimeManifestSha256": digest(args.runtime),
        "runtimeEvidence": "Separately captured runtime metadata; run reports do not cryptographically bind this manifest.",
        "reportGeneratorSha256": digest(Path(__file__)), "reportHelperSha256": digest(HELPER),
        "sourceCodeAndPhysicsInputsIdentical": True,
        "sharedInputs": {"canonicalSha256": report["canonicalSha256"], "placementSha256": report["placementSha256"],
            "sourceDigests": report["sourceDigests"], "sourceProvenance": source["provenance"],
            "sewingFrames": source["sewingFrames"], "foldActuation": source["foldActuation"],
            "assemblySchedule": source["assemblySchedule"], "contactProfile": report["contactProfile"],
            "arguments": without(report["arguments"], "canonical", "placement", "output", "subdivisions")},
        "argumentDifferences": {"subdivisions": [256, 512], "outputRunNames": [args.coarse.name, args.fine.name]},
        "commonSavedFractionsIncludingInitial": len(comparisons), "commonPositiveFractions": len(comparisons) - 1,
        "allCoarseSavedFractionsPresentExactlyInFine": True,
        "fineOnlySavedPositiveFractions": [str(fraction) for fraction in fine["trajectory"] if fraction not in coarse["trajectory"]],
        "runs": {"coarse": coarse["summary"], "fine": fine["summary"]},
        "finalComparison": comparisons[-1],
        "trajectoryMaxima": {key: max(comparisons, key=lambda row: row[key]) for key in metrics},
        "milestones": [row for row in comparisons if Fraction(row["fraction"]) in [Fraction(1, 16), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1)]],
        "comparisons": comparisons,
        "limitations": ["Only one halving of the initial timestep, with adaptive subdivisions retained; no asymptotic convergence order or error bound follows.",
            "Both runs retain final motion. Implicit numerical dissipation and different integration histories remain part of this comparison.",
            "The 0.1 mm activation width beyond the fixed 0.1 mm core is an uncalibrated contact-law choice, not a physically validated material setting.",
            "Accepted numerical steps do not imply accepted garment construction. Exact path checks do not establish layer order or thickness at incident primitives.",
            "Raw captures and logs remain in ignored research storage; hashes cannot reconstruct missing bytes."]}
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_prefix.with_suffix(".json").write_text(encoded)
    if args.plot:
        plot(comparisons, args.output_prefix.with_suffix(".png"))
    print(json.dumps({"ledger": str(args.output_prefix.with_suffix(".json")),
        "figure": str(args.output_prefix.with_suffix(".png")) if args.plot else None, "accepted": False}))


if __name__ == "__main__":
    main()
