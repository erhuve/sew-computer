import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
from fractions import Fraction

# Keep current verification outside the captured numerical module namespace.
# An unlisted snapshot file cannot replace it, and loading must not cache
# current hinge/contact implementations used by historical solver replay.
def load_verifier(filename, alias):
    path = Path(__file__).with_name(filename)
    content = path.read_bytes()
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(content).hexdigest()


triangle_verifier, triangle_verifier_digest = load_verifier("solver_triangle_sweep.py", "replay_triangle_verifier")
broad_phase_verifier, broad_phase_verifier_digest = load_verifier("solver_ipc_broad_phase.py", "replay_broad_phase_verifier")
coverage_verifier, coverage_verifier_digest = load_verifier("solver_candidate_coverage.py", "replay_coverage_verifier")

if not __debug__:
    raise RuntimeError("Replay requires enabled verification assertions")

parser = argparse.ArgumentParser(description="Replay trusted saved rest-filtered research states and exact path proofs")
parser.add_argument("run", type=Path)
parser.add_argument("--cpu-limit-seconds", type=int, default=100,
                    help="Bounded replay CPU budget, independent of simulation budgets (1–3600 seconds)")
replay_arguments = parser.parse_args()
if not 1 <= replay_arguments.cpu_limit_seconds <= 3600:
    parser.error("Replay CPU limit must be between 1 and 3600 seconds")
resource.setrlimit(resource.RLIMIT_CPU, (replay_arguments.cpu_limit_seconds,
                                      replay_arguments.cpu_limit_seconds + 5))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
run = replay_arguments.run.resolve()
report = json.loads((run / "report.json").read_text())
assert report["terminal"]
assert report["accepted"] is False
assert report["arguments"]["contact_model"] == "rest-filtered"
for name, expected in report["sourceDigests"].items():
    assert hashlib.sha256((run / "source-snapshot" / name).read_bytes()).hexdigest() == expected
for name, field in (("canonical.json", "canonicalSha256"), ("placement.json", "placementSha256")):
    assert hashlib.sha256((run / name).read_bytes()).hexdigest() == report[field]
sys.path.insert(0, str(run / "source-snapshot"))
import ipctk
import newton
import numpy as np
import warp as wp
from solver_rest_filtered_contact import RestFilteredSurfaceContact
from solver_global_sewing import GlobalSewingSolver
from solver_spike_geometry import surface_intersections
from solver_contact_preflight import staging_seam_gaps
from solver_strain_diagnostics import edge_strain_report
from solver_attempt_journal import recover_attempt_journal

recovered = recover_attempt_journal(run, report)
assert not recovered["attemptJournal"]["errors"]
assert recovered["acceptedStateArtifacts"] == report["acceptedStateArtifacts"]

ipctk.set_num_threads(1)
source = json.loads((run / "canonical.json").read_text())
placement = json.loads((run / "placement.json").read_text())
rest = np.asarray(source["restMeters"])
faces = np.asarray(source["triangles"]).reshape((-1, 3))
initial = np.asarray(placement["placedMeters"])
arguments = report["arguments"]
schedule_knots = None
if arguments.get("assembly_schedule"):
    from solver_assembly_schedule import AssemblySchedule
    recipe = source["assemblySchedule"]
    AssemblySchedule(recipe, arguments["subdivisions"])
    assert report["assemblySchedule"] == recipe
    schedule_knots = recipe["knots"]


def operation_progress(fraction, key):
    if schedule_knots is None:
        return fraction
    upper_index = next(index for index in range(1, len(schedule_knots))
                       if fraction <= schedule_knots[index]["fraction"])
    lower, upper = schedule_knots[upper_index - 1], schedule_knots[upper_index]
    remaining = (upper["fraction"] - fraction) / (upper["fraction"] - lower["fraction"])
    return remaining * lower[key] + (1 - remaining) * upper[key]

contact = RestFilteredSurfaceContact(rest, faces, activation_distance_m=arguments["activation_distance_m"],
    minimum_distance_m=arguments["minimum_distance_m"], stiffness=arguments["pressure_pa"],
    ccd_profile=arguments["ccd_profile"])
assert contact.profile() == report["contactProfile"]
builder = newton.ModelBuilder(gravity=(0, 0, 0))
ordered = sorted(source["instanceOffsets"].items(), key=lambda pair: pair[1])
identities = []
for index, (identity, start) in enumerate(ordered):
    end = ordered[index + 1][1] if index + 1 < len(ordered) else len(rest)
    triangles = faces[np.all((faces >= start) & (faces < end), axis=1)] - start
    builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1, vel=wp.vec3(0, 0, 0),
        vertices=rest[start:end].tolist(), indices=triangles.ravel().tolist(), density=.2,
        tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.01, edge_kd=0)
    identities.extend([identity] * (end - start))
builder.set_coloring([[vertex] for vertex in range(len(rest))])
model = builder.finalize(device="cpu")
model.particle_q.assign(initial.astype(np.float32))
rows = [{source["instanceOffsets"][term["instanceId"]] + term["vertex"]: term["coefficient"]
         for term in row["terms"]} for row in source["embeddedConstraints"]["constraints"]]
fold_recipe = source.get("foldActuation") if arguments.get("fold_actuation") else None
solver = GlobalSewingSolver(model, rows, 1e-8, contact=contact, fold_barrier_joules=1e-5,
                           sewing_mode=arguments.get("sewing_mode", "vector"),
                           sewing_frame_faces=source.get("sewingFrames", {}).get("faces") if arguments.get("sewing_mode") == "normal-offset" else None,
                           sewing_sides=source.get("sewingFrames", {}).get("sides") if arguments.get("sewing_mode") == "normal-offset" else None,
                           fold_hinges=fold_recipe["hinges"] if fold_recipe else None,
                           fold_stiffness_joules=fold_recipe["stiffnessJoules"] if fold_recipe else None)
if fold_recipe:
    initial_fold = solver.fold_actuation.potential(fold_recipe["initialAnglesRadians"])
    final_fold = solver.fold_actuation.potential(fold_recipe["targetAnglesRadians"])
    np.testing.assert_allclose(initial_fold.angles(initial), initial_fold.rest_angles, rtol=0, atol=1e-10)
initial_targets = solver.sewing @ initial
if solver.sewing_mode in ("distance", "normal-offset"):
    initial_targets = np.linalg.norm(initial_targets, axis=1)
previous, previous_velocity = initial.copy(), np.zeros_like(initial)
previous_fraction = 0.
results = []
total_leaves = 0


def verify_certificate(start, end, proof):
    from solver_temporal_separation import _GROUPS, _primitive_ids

    def identity(group, first, second):
        first, second = tuple(sorted(first)), tuple(sorted(second))
        if group in ("vv_candidates", "ee_candidates") and first > second:
            first, second = second, first
        return group, first, second

    candidates = ipctk.Candidates()
    candidates.build(contact.mesh, start, end,
                     inflation_radius=np.nextafter(contact.minimum_distance_m / 2, np.inf),
                     broad_phase=broad_phase_verifier.contact_broad_phase())
    observed = [("fv", int(candidate.face_id), int(candidate.vertex_id)) for candidate in candidates.fv_candidates]
    observed += [("ee", int(candidate.edge0_id), int(candidate.edge1_id)) for candidate in candidates.ee_candidates]
    coverage = coverage_verifier.verify_candidate_coverage(start, end, contact._edges, faces,
        float(np.nextafter(contact.minimum_distance_m / 2, np.inf)), observed)
    assert coverage["verified"], coverage
    assert len(observed) == len(candidates)  # This replay's cloth has no isolated edges or vertices.
    expected = {}
    for group in _GROUPS:
        for index, candidate in enumerate(getattr(candidates, group)):
            minimum = contact.minimum_distance_m
            if contact._key(group, candidate) in contact._filtered:
                if contact.profile()["adapter"] == "experimental-rest-filtered-contact-v2":
                    # Independently reconstruct the fixed primitive-pair core
                    # from source coordinates, rather than reuse assignment.
                    rest_distance = float(np.sqrt(candidate.compute_distance(
                        candidate.dof(rest, contact._edges, contact.faces))))
                    minimum = min(contact.minimum_distance_m, rest_distance / 4)
                else:
                    minimum = contact._local_minimum
            first, second = _primitive_ids(group, candidate, contact._edges, contact.faces)
            key = identity(group, first, second)
            assert key not in expected
            expected[key] = minimum
    assert len(expected) == len(candidates) == proof["candidateCount"]
    intervals, indexed_identities = {}, {}
    for leaf in proof["certificateLeaves"]:
        key = identity(leaf["group"], leaf["first"], leaf["second"])
        minimum = expected[key]
        index_key = leaf["group"], leaf["candidate"]
        assert type(leaf["candidate"]) is int and 0 <= leaf["candidate"] < len(getattr(candidates, leaf["group"]))
        assert indexed_identities.setdefault(index_key, key) == key
        first, second = leaf["first"], leaf["second"]
        assert minimum == leaf["minimumDistanceM"]
        normal = [Fraction(value) for value in leaf["normal"]]
        projections = []
        for fraction in (leaf["t0"], leaf["t1"]):
            fraction = Fraction(fraction)
            for first_id in first:
                for second_id in second:
                    projections.append(sum(component * ((1 - fraction) * (
                        Fraction(float(start[first_id, axis])) - Fraction(float(start[second_id, axis])))
                        + fraction * (Fraction(float(end[first_id, axis]))
                                      - Fraction(float(end[second_id, axis]))))
                        for axis, component in enumerate(normal)))
        gap = max(min(projections), -max(projections))
        assert gap > 0 and gap ** 2 > Fraction(minimum) ** 2 * sum(value ** 2 for value in normal)
        assert Fraction(leaf["lowerBoundM"]) ** 2 * sum(value ** 2 for value in normal) <= gap ** 2
        intervals.setdefault(key, []).append((leaf["t0"], leaf["t1"]))
    assert set(intervals) == set(expected)
    assert len(indexed_identities) == len(expected)
    for spans in intervals.values():
        cursor = 0.
        for lower, upper in sorted(spans):
            assert lower == cursor and upper > lower
            cursor = upper
        assert cursor == 1.
    return coverage


for artifact in report["acceptedStateArtifacts"]:
    content = (run / artifact["path"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
    state = json.loads(content)
    assert state["accepted"] is False
    record = state["record"]
    positions = np.asarray(state["positionsMeters"])
    velocity = np.asarray(state["velocitiesMetersPerSecond"])
    assert record["startFraction"] == previous_fraction
    duration = record["durationSeconds"]
    assert duration == arguments["step_seconds"] * (record["endFraction"] - previous_fraction)
    np.testing.assert_array_equal(velocity, (positions - previous) / duration)
    final_targets = initial_targets * arguments["target_fraction"]
    sewing_progress = operation_progress(record["endFraction"], "sewingProgress")
    targets = (final_targets if sewing_progress == 1 else
               initial_targets + sewing_progress * (final_targets - initial_targets))
    coefficients = np.concatenate((-solver.poses.sum(axis=1)[:, None], solver.poses), axis=1)
    deformation = np.einsum("fvc,fva->fca", coefficients, positions[faces])
    normals = np.cross(deformation[:, 0], deformation[:, 1])
    ratio = np.linalg.norm(normals, axis=1)
    normals /= ratio[:, None]
    area_gradient = np.stack((np.cross(deformation[:, 1], normals),
                              np.cross(normals, deformation[:, 0])), axis=1)
    lame = solver.materials[:, 0] + solver.materials[:, 1]
    alpha = 1 + solver.materials[:, 0] / np.maximum(lame, 1e-6)
    stress = solver.materials[:, 0, None, None] * deformation + (lame * (ratio - alpha))[:, None, None] * area_gradient
    elements = solver.areas[:, None, None] * np.einsum("fvc,fca->fva", coefficients, stress)
    gradient = (np.sqrt(solver.mass) / duration)[:, None] ** 2 * (positions - previous - duration * previous_velocity)
    anchors = solver.sewing @ positions
    if solver.sewing_mode == "normal-offset":
        frame = positions[solver.sewing_frame_faces]
        first_edge, second_edge = frame[:, 1] - frame[:, 0], frame[:, 2] - frame[:, 0]
        area_vector = np.cross(first_edge, second_edge)
        area = np.linalg.norm(area_vector, axis=1)
        normal = area_vector / area[:, None]
        signed_offset = targets * solver.sewing_sides
        offset_residual = anchors - signed_offset[:, None] * normal
        gradient += solver.sewing.T @ (offset_residual / solver.compliance)
        tangent_residual = offset_residual - np.sum(offset_residual * normal, axis=1)[:, None] * normal
        area_force = -signed_offset[:, None] * tangent_residual / (solver.compliance * area[:, None])
        first_reaction = np.cross(second_edge, area_force)
        second_reaction = np.cross(area_force, first_edge)
        for local, reaction in enumerate((-first_reaction - second_reaction, first_reaction, second_reaction)):
            np.add.at(gradient, solver.sewing_frame_faces[:, local], reaction)
    elif solver.sewing_mode == "distance":
        lengths = np.linalg.norm(anchors, axis=1)
        gradient += solver.sewing.T @ (anchors / lengths[:, None]
                                      * ((lengths - targets) / solver.compliance)[:, None])
    else:
        gradient += solver.sewing.T @ ((anchors - targets) / solver.compliance)
    np.add.at(gradient, faces, elements)
    gradient += solver.bending.gradient(positions) + solver.fold_barrier.gradient(positions) + contact.gradient(positions)
    fold_diagnostics = {}
    if fold_recipe:
        fold_progress = operation_progress(record["endFraction"], "foldProgress")
        fold_targets = (final_fold.rest_angles if fold_progress == 1 else
            initial_fold.rest_angles + fold_progress * (final_fold.rest_angles - initial_fold.rest_angles))
        actuator = solver.fold_actuation.potential(fold_targets)
        gradient += actuator.gradient(positions)
        np.testing.assert_allclose(record["step"]["foldTargetsRadians"], fold_targets, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(record["step"]["foldAnglesRadians"], actuator.angles(positions))
        fold_diagnostics = {"foldAnglesRadians": actuator.angles(positions).tolist(),
                            "foldTargetErrorRadians": float(np.max(np.abs(actuator.angles(positions) - fold_targets)))}
    residual = float(np.max(np.abs(gradient.ravel()[solver.free])))
    assert residual <= 1e-6
    assert surface_intersections(positions, faces)["intersectingPairCount"] == 0
    assert not ipctk.has_intersections(contact.mesh, positions,
                                     broad_phase=broad_phase_verifier.contact_broad_phase())
    triangle_proof = triangle_verifier.verify_triangle_sweep_exact(previous, positions, faces)
    assert contact.path_safe(previous, positions)
    from solver_hinge_sweep import hinge_sweep_safe
    assert hinge_sweep_safe(previous, positions, solver.fold_barrier.indices)
    proof = contact.path_certificate(previous, positions, keep_leaves=True)
    assert proof["safe"]
    candidate_coverage = verify_certificate(previous, positions, proof)
    total_leaves += len(proof.pop("certificateLeaves"))
    np.testing.assert_array_equal(contact.rest_positions, rest)
    results.append({"endFraction": record["endFraction"], "recomputedResidualN": residual,
        "recordedResidualN": record["step"]["gradientInfinityNorm"], "endpointIntersections": 0,
        "physicalPathPass": True, "pathCertificate": proof, "contactEnergyJ": contact.energy(positions),
        "trianglePathVerification": triangle_proof,
        "candidateCoverageVerification": candidate_coverage,
        **fold_diagnostics})
    previous, previous_velocity, previous_fraction = positions, velocity, record["endFraction"]
assert report["adaptive"]["completedFraction"] == previous_fraction
assert report["adaptive"]["completedDurationSeconds"] == previous_fraction * arguments["step_seconds"]
if report.get("stateArtifact"):
    final_content = (run / report["stateArtifact"]["path"]).read_bytes()
    assert hashlib.sha256(final_content).hexdigest() == report["stateArtifact"]["sha256"]
    final_state = json.loads(final_content)
    np.testing.assert_array_equal(final_state["positionsMeters"], previous)
    np.testing.assert_array_equal(final_state["velocitiesMetersPerSecond"], previous_velocity)
    assert final_state["completedDurationSeconds"] == previous_fraction * arguments["step_seconds"]
else:
    assert not report["completed"]
result = {"accepted": False, "states": results, "contactProfile": contact.profile(),
          "completed": report["completed"], "completedFraction": previous_fraction,
          "finalStateVerified": bool(report.get("stateArtifact")),
          "sewingMode": solver.sewing_mode, "exactCertificateLeaves": total_leaves,
          "foldActuation": fold_recipe,
          "assemblySchedule": source.get("assemblySchedule") if arguments.get("assembly_schedule") else None,
          "replayScriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "triangleVerifierSha256": triangle_verifier_digest,
          "broadPhaseVerifierSha256": broad_phase_verifier_digest,
          "candidateCoverageVerifierSha256": coverage_verifier_digest,
          "verificationBroadPhase": broad_phase_verifier.CONTACT_BROAD_PHASE_PROFILE,
          "verificationCpuLimitSeconds": replay_arguments.cpu_limit_seconds,
          "verificationCpuSeconds": (resource.getrusage(resource.RUSAGE_SELF).ru_utime
                                     + resource.getrusage(resource.RUSAGE_SELF).ru_stime),
          "sourceDigests": report["sourceDigests"], "reportSha256": hashlib.sha256((run / "report.json").read_bytes()).hexdigest(),
          "seamGaps": staging_seam_gaps(source, initial, previous),
          "edgeStrain": edge_strain_report(rest, previous, faces, identities),
          "review": "implementer reconstruction; same collision and fold-angle derivative modules, separate assembled residual formula and endpoint oracle"}
destination = run / "verified-replay.json"
with destination.open("x") as handle:
    handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"verifiedStates": len(results), "exactCertificateLeaves": total_leaves,
                  "maxResidualN": max((item["recomputedResidualN"] for item in results), default=None)}))
