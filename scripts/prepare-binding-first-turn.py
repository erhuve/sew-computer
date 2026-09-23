"""Prepare unaccepted left-binding first-turn/control inputs; never run a solver.

This bounded research recipe preserves the full five-fabric source unit. Five existing
left-binding rows start held at their sampled positive initial distances. Three
material grippers request a small whole-strip turn, hold, release and passive
tail. No seam engagement phase, wrap, stitch-down, apex or construction
milestone is supplied or completed. Use the source-compatible pinned Linux
runtime; exact source coefficient rederivation is intentionally required.
The selected time policy declares the total physical interval and initial
subdivision count; the runner must use those values. Target knots are unchanged.
"""

import argparse
import copy
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from scipy.sparse import coo_matrix

from solver_attempt_journal import strict_loads
from solver_cuff_source_binding import ENGINE_FILES, SOURCE_HELPER_FILES, validate_cuff_source_binding
from solver_distance_sewing import DistanceSewing
from solver_gripper_input import PROFILE as GRIPPER_PROFILE, bind_material_grippers, mesh_identity
from solver_material_grippers import SCHEDULE_PROFILE
from solver_process_budget import atomic_bytes, read_regular
from solver_sewing_input import PROFILE as SEWING_PROFILE, bind_sewing_activation, derive_sewing_row_ids, sewing_source_identity


ACTIVE_INSTANCE = "opening_binding_left_left:shell"
SLEEVE_INSTANCE = "sleeve_left:shell"
TIME_POLICIES = {"original-128ms-v1": (.128, 64), "fourfold-512ms-v1": (.512, 256)}
GAP_M = .001
EXPECTED_FACES = [[1, 10, 2], [5, 10, 1], [10, 5, 9], [10, 6, 2],
                  [6, 10, 9], [3, 8, 0], [8, 4, 0], [5, 4, 9],
                  [4, 8, 9], [7, 8, 3], [7, 6, 9], [8, 7, 9]]
ANCHORS = (("binding-wrist-body", 6, [.5, .25, .25]),
           ("binding-apex-body", 1, [.25, .5, .25]),
           ("binding-allowance", 10, [.375, .375, .25]))


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def canonical_encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def sampled_anchors(positions, faces, anchors):
    """Correctly round exact binary source-barycentric positions once.

    An exact binary rational weighted point need not itself fit in binary64.
    Unavoidable rounding is retained and its initial stored energy is measured;
    this function does not move a cloth vertex to manufacture zero energy.
    """
    return np.asarray([[float(sum((Fraction(weight) * Fraction(float(positions[vertex, axis]))
                                  for vertex, weight in zip(faces[anchor["triangleIndex"]], anchor["weights"])),
                                 Fraction())) for axis in range(3)] for anchor in anchors])


def generate(unit_path, output, *, angle_degrees=3., gripper_stiffness_n_per_m=1.,
             time_policy="original-128ms-v1"):
    if not sys.platform.startswith("linux"):
        raise ValueError("Use the compatible pinned Linux source runtime; no cross-runtime coefficient tolerance")
    if type(time_policy) is not str or time_policy not in TIME_POLICIES:
        raise ValueError("Explicit supported binding first-turn time policy required")
    duration, subdivisions = TIME_POLICIES[time_policy]
    time_declaration = {"profile": "binding-first-turn-time-v1", "id": time_policy,
                        "durationSeconds": duration, "nominalStepSeconds": duration / subdivisions}
    if (type(angle_degrees) not in (int, float) or not math.isfinite(angle_degrees)
            or not 0 <= angle_degrees <= 3
            or type(gripper_stiffness_n_per_m) not in (int, float)
            or not math.isfinite(gripper_stiffness_n_per_m) or not 0 < gripper_stiffness_n_per_m <= 100):
        raise ValueError("Angle must be finite within 0..3 degrees; gripper stiffness must be within (0,100] N/m")
    helper_names = {*SOURCE_HELPER_FILES, "solver_attempt_journal.py", "solver_process_budget.py",
                    "solver_gripper_input.py", "solver_material_grippers.py", "solver_sewing_input.py",
                    "solver_sewing_activation_schedule.py", "solver_sewing_activation.py", "solver_distance_sewing.py"}
    paths = {"scripts/" + name: ROOT / "scripts" / name for name in sorted(helper_names)}
    paths.update({"services/engine/" + name: ROOT / "services/engine" / name for name in ENGINE_FILES})
    paths["scripts/prepare-binding-first-turn.py"] = Path(__file__).resolve()
    code = {name: read_regular(path) for name, path in paths.items()}
    unit_bytes = read_regular(unit_path, 8 * 1024 ** 2)
    unit = strict_loads(unit_bytes)
    binding = validate_cuff_source_binding(unit)
    additions = {"placedMeters", "bindingFirstTurnDiagnostic", "sewingActuation", "gripperActuation"}
    if (unit["side"] != "left" or additions.intersection(unit)
            or {"foldActuation", "assemblySchedule", "sewingFrames"}.intersection(unit)
            or any(instance["mirrorX"] is not False for instance in unit["instances"])):
        raise ValueError("Unplaced, uncontrolled, unmirrored left five-fabric source unit required")
    if unit["sourcePattern"]["drafting"]["seamAllowanceMm"] != 10:
        raise ValueError("This bounded first-turn policy requires the explicit 10 mm source allowance")
    panels = {panel["id"]: panel for panel in unit["sourcePattern"]["panels"]}
    strip_panel, sleeve_panel = panels["opening_binding_left_left"], panels["sleeve_left"]
    strip = unit["sourceTemplates"]["opening_binding_left_left"]
    if len(strip["restPositions"]) != 11 or strip["triangles"] != EXPECTED_FACES:
        raise ValueError("Unsupported binding mesh layout; explicit source anchor migration required")
    strip_edge = next(edge for edge in strip_panel["draft"]["edges"] if edge["name"] == "right")
    sleeve_edge = next(edge for edge in sleeve_panel["draft"]["edges"] if edge["name"] == "opening_left")
    if ([strip_edge["start"], strip_edge["end"]] != [1, 2]
            or [sleeve_edge["start"], sleeve_edge["end"]] != [5, 6]):
        raise ValueError("Unsupported source stitch-axis endpoint layout")
    strip_axis = np.column_stack((np.asarray(strip_panel["points"])[[1, 2]] * .001, np.zeros(2)))
    sleeve_axis = np.column_stack((np.asarray(sleeve_panel["points"])[[5, 6]] * .001, np.zeros(2)))
    strip_length = float(np.linalg.norm(strip_axis[1] - strip_axis[0]))
    if (not np.array_equal(strip_axis[:, 0], [.020, .020]) or strip_axis[0, 1] != 0.
            or strip_length <= 0 or not np.array_equal(strip_axis[:, 2], [0., 0.])):
        raise ValueError("Expected straight 20 mm-wide source binding with forward longitudinal attachment")
    cut = np.asarray(strip_panel["draft"]["cutLine"]) * .001
    if (np.max(np.abs(cut.min(axis=0) - [-.010, -.010])) > 1e-12
            or np.max(np.abs(cut.max(axis=0) - [.030, strip_length + .010])) > 1e-12):
        raise ValueError("Full unchanged 10 mm binding end/side allowances required")
    rest = np.asarray(unit["restMeters"], dtype=float)
    faces = np.asarray(unit["triangles"], dtype=int).reshape(-1, 3)
    area_normals = np.cross(rest[faces[:, 1]] - rest[faces[:, 0]], rest[faces[:, 2]] - rest[faces[:, 0]])
    if (np.any(area_normals[:, 2] <= 0) or np.any(area_normals[:, :2] != 0)):
        raise ValueError("This research side policy requires verified +z canonical winding on every source face")
    source_midpoint = strip_axis.mean(axis=0)
    axis_origin = sleeve_axis.mean(axis=0) + [0., 0., GAP_M]
    tangent = sleeve_axis[1] - sleeve_axis[0]
    tangent /= np.linalg.norm(tangent)
    inward = np.array([-tangent[1], tangent[0], 0.])
    rotation = np.column_stack((-inward, tangent, [0., 0., 1.]))
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-14)
            or not math.isclose(float(np.linalg.det(rotation)), 1., rel_tol=0, abs_tol=1e-14)):
        raise ValueError("Proper source-derived rotation required")
    translation = axis_origin - rotation @ source_midpoint
    placed = rest.copy()
    poses, instance_ranges = [], {}
    parking = {"cuff_left:shell": .05, "cuff_left:facing": .10,
               "opening_binding_left_right:shell": .15, "sleeve_left:shell": 0.}
    lo = unit["instanceOffsets"][ACTIVE_INSTANCE]
    hi = lo + len(strip["restPositions"])
    for instance in unit["instances"]:
        identity = instance["id"]
        start = unit["instanceOffsets"][identity]
        end = start + len(unit["sourceTemplates"][instance["templateId"]]["restPositions"])
        instance_ranges[identity] = [start, end]
        r, shift = (rotation, translation) if identity == ACTIVE_INSTANCE else (np.eye(3), np.array([0., 0., parking[identity]]))
        placed[start:end] = rest[start:end] @ r.T + shift
        poses.append({"instanceId": identity, "rotation": r.tolist(), "translationMeters": shift.tolist(),
                      "determinant": float(np.linalg.det(r)), "sourceMirrorX": False,
                      "canonicalNormal": [0., 0., 1.], "initialPlacedNormal": (r @ np.array([0., 0., 1.])).tolist()})
    theta = math.radians(angle_degrees)
    cross = np.array([[0., -tangent[2], tangent[1]], [tangent[2], 0., -tangent[0]],
                      [-tangent[1], tangent[0], 0.]])

    def pose(angle):
        result = placed.copy()
        if angle != 0:
            turn = np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)
            result[lo:hi] = (placed[lo:hi] - axis_origin) @ turn.T + axis_origin
        return result

    active_groups = [group for group in binding["rowGroups"]
                     if group["registrationId"] == "bind_opening_left_left" and group["memberIndex"] == 1]
    if len(active_groups) != 1 or active_groups[0]["rowIndices"] != list(range(5)):
        raise ValueError("Unchanged five left-binding rows at canonical indices 0..4 required")
    constraints = unit["embeddedConstraints"]["constraints"]
    if len(constraints) != 40 or len({row["complianceMPerN"] for row in constraints}) != 1:
        raise ValueError("All 40 original rows and homogeneous original compliance required")
    activation = np.array([1.] * 5 + [0.] * 35)
    entries = [(index, unit["instanceOffsets"][term["instanceId"]] + term["vertex"], term["coefficient"])
               for index, row in enumerate(constraints) for term in row["terms"]]
    coupling = coo_matrix(([entry[2] for entry in entries],
                         ([entry[0] for entry in entries], [entry[1] for entry in entries])),
                        shape=(40, len(rest))).tocsr()
    nominal_targets = np.full(40, GAP_M)
    initial_sewing = DistanceSewing(coupling, nominal_targets, constraints[0]["complianceMPerN"], activation=activation)
    _, lengths = initial_sewing.geometry(placed)
    targets = nominal_targets.copy()
    targets[:5] = lengths[:5]
    if np.max(np.abs(targets[:5] - GAP_M)) > 1e-12:
        raise ValueError("Rigid source registration exceeds the disclosed tiny source-length mismatch")
    face_indices = {tuple(face): index for index, face in enumerate(faces.tolist())}
    row_ids = list(derive_sewing_row_ids(unit))
    instances_by_id = {instance["id"]: instance for instance in unit["instances"]}
    frame_faces = {SLEEVE_INSTANCE: [], ACTIVE_INSTANCE: []}
    frame_bindings = []
    frame_policy = ("Use unchanged oriented source triangles containing every nonzero source-anchor support vertex; "
                    "if more than one exists, choose the lexicographically smallest oriented local vertex tuple, "
                    "then source triangle index. This explicit diagnostic choice does not infer a material director "
                    "or merge edge/corner samples.")
    for row_index, row in enumerate(constraints[:5]):
        samples = {sample["instanceId"]: sample for sample in row["sourceSamples"]}
        if set(samples) != {SLEEVE_INSTANCE, ACTIVE_INSTANCE} or len(row["sourceSamples"]) != 2:
            raise ValueError("Expected one positive sleeve and one negative binding anchor per held row")
        for identity, sign in ((SLEEVE_INSTANCE, 1), (ACTIVE_INSTANCE, -1)):
            sample = samples[identity]
            weighted_support = {entry["vertex"]: entry["weight"] for entry in sample["weights"] if entry["weight"] != 0}
            signed_support = {entry["vertex"]: entry["coefficient"] for entry in row["terms"]
                              if entry["instanceId"] == identity and entry["coefficient"] != 0}
            if signed_support != {vertex: sign * weight for vertex, weight in weighted_support.items()}:
                raise ValueError("Held row does not preserve its signed source-anchor support")
            template_id = instances_by_id[identity]["templateId"]
            template_faces = unit["sourceTemplates"][template_id]["triangles"]
            candidates = [(tuple(face), index) for index, face in enumerate(template_faces)
                          if set(weighted_support).issubset(face)]
            if not candidates:
                raise ValueError("Held source-anchor support has no original oriented triangle frame")
            local_face, local_index = min(candidates)
            canonical_face = [unit["instanceOffsets"][identity] + vertex for vertex in local_face]
            canonical_index = face_indices[tuple(canonical_face)]
            frame_faces[identity].append(canonical_face)
            frame_bindings.append({"rowIndex": row_index, "rowId": row_ids[row_index],
                "instanceId": identity, "sourceTemplateId": template_id, "coefficientSign": sign,
                "sourceAnchorSupport": copy.deepcopy(sample["weights"]),
                "sourceTriangleIndex": local_index, "instanceLocalVertices": list(local_face),
                "canonicalTriangleIndex": canonical_index, "canonicalVertices": canonical_face,
                "candidateSourceTriangleIndices": sorted(index for _, index in candidates),
                "ambiguousFrameChoice": len(candidates) > 1})
    strip_tangent = (strip_axis[1] - strip_axis[0]) / strip_length
    anchors, supports = [], []
    for identity, local_triangle, weights in ANCHORS:
        local_vertices = strip["triangles"][local_triangle]
        vertices = [lo + vertex for vertex in local_vertices]
        index = face_indices[tuple(vertices)]
        anchors.append({"id": identity, "instanceId": ACTIVE_INSTANCE, "triangleIndex": index,
                        "weights": weights, "stiffnessNPerM": float(gripper_stiffness_n_per_m)})
        supports.append({"id": identity, "sourceTemplateId": "opening_binding_left_left",
                         "sourceTriangleIndex": local_triangle, "instanceLocalVertices": local_vertices,
                         "canonicalTriangleIndex": index, "canonicalVertices": vertices,
                         "weights": weights, "weightFractions": [str(Fraction(weight)) for weight in weights]})
    initial_anchors = sampled_anchors(placed, faces, anchors)
    if np.linalg.norm(np.cross(initial_anchors[1] - initial_anchors[0], initial_anchors[2] - initial_anchors[0])) <= 1e-10:
        raise ValueError("Three noncollinear source-barycentric grippers required")
    knots = [{"fraction": index / 16, "targetsMeters": sampled_anchors(pose(theta * index / 8), faces, anchors).tolist(),
              "activation": [1., 1., 1.]} for index in range(9)]
    final_targets = knots[-1]["targetsMeters"]
    knots.extend({"fraction": fraction, "targetsMeters": copy.deepcopy(final_targets), "activation": [active] * 3}
                 for fraction, active in ((.75, 1.), (.875, 0.), (1., 0.)))
    source = copy.deepcopy(unit)
    source["placedMeters"] = placed.tolist()
    source["gripperActuation"] = {"profile": GRIPPER_PROFILE, "accepted": False, "meshSha256": mesh_identity(source),
        "anchors": anchors, "schedule": {"profile": SCHEDULE_PROFILE,
            "gripperIds": [anchor["id"] for anchor in anchors], "knots": knots}}
    grippers, gripper_schedule, _ = bind_material_grippers(source, subdivisions)
    initial_targets, initial_activation = gripper_schedule.parameters(0)
    initial_gripper_energy = grippers.potential(initial_targets, initial_activation).energy(placed)
    lower_gap = GAP_M - .010 * math.sin(theta)
    if lower_gap <= .0002:
        raise ValueError("Declared rigid reference must clear the 0.1 mm core plus 0.1 mm activation buffer")
    source["bindingFirstTurnDiagnostic"] = {
        "profile": "source-left-binding-first-turn-v1", "accepted": False, "solverReady": False,
        "scope": "Initial held offset attachment plus requested whole-strip first turn, hold, gripper release and passive tail. No engagement phase, fold crease, binding wrap, stitch-down, apex completion or construction milestone.",
        "sourceUnitBytesSha256": sha(unit_bytes), "sourceUnitCanonicalSha256": sha(canonical_encoded(unit)),
        "preservedSourceFields": sorted(unit), "sourceBinding": binding,
        "codeDigests": {name: sha(content) for name, content in code.items()},
        "runtime": {"python": platform.python_version(), "platform": sys.platform,
                    **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "shapely")}},
        "subdivisions": subdivisions, "timePolicy": time_declaration,
        "angleDegrees": float(angle_degrees), "angleRadians": theta,
        "nominalSeamOffsetMeters": GAP_M, "heldSeamTargetsMeters": targets[:5].tolist(),
        "targetPolicy": "Held positive distances sampled by the declared CSR/binary64 distance geometry at rigid initial placement; tiny original source-length mismatch retained. Pending targets are unused positive 1 mm placeholders.",
        "sourceArcLengthsMeters": {"strip": strip_length, "sleeve": float(np.linalg.norm(sleeve_axis[1] - sleeve_axis[0]))},
        "bindingInstanceId": ACTIVE_INSTANCE, "sleeveInstanceId": SLEEVE_INSTANCE,
        "heldRowIds": row_ids[:5], "heldRowIndices": list(range(5)), "instanceRanges": instance_ranges,
        "sleeveFrameFaces": frame_faces[SLEEVE_INSTANCE], "bindingFrameFaces": frame_faces[ACTIVE_INSTANCE],
        "sleeveTangentsRest": [tangent.tolist() for _ in range(5)],
        "bindingTangentsRest": [strip_tangent.tolist() for _ in range(5)],
        "frameSelectionPolicy": frame_policy, "frameSelectionBindings": frame_bindings,
        "activeRowIndices": list(range(5)), "pendingRowIndices": list(range(5, 40)),
        "initialAttachmentCondition": "Rows 0..4 begin at activation one and stay held; this initial numerical condition does not execute an attachment phase.",
        "rigidPlacements": poses, "axisOriginMeters": axis_origin.tolist(), "axisDirection": tangent.tolist(),
        "sourceAxisEndpointsMeters": strip_axis.tolist(), "receiverAxisEndpointsMeters": sleeve_axis.tolist(),
        "rotationSignPolicy": "Positive rotation lifts the 30 mm source body wing and lowers the 10 mm source allowance wing; this is an explicit research choice.",
        "textileSidePolicy": {"classification": "explicit research declaration, not inferred source material semantics",
            "rightSideRelativeToCanonicalNormal": {"sleeve_left:shell": 1, ACTIVE_INSTANCE: -1,
                "cuff_left:shell": "unresolved", "cuff_left:facing": "unresolved", "opening_binding_left_right:shell": "unresolved"},
            "initialActiveRightSideNormals": {"sleeve_left:shell": [0., 0., 1.], ACTIVE_INSTANCE: [0., 0., -1.]},
            "initialRelation": "Declared right sides face each other across the positive 1 mm stack offset; canonical normals both remain +z."},
        "gripperIds": [anchor["id"] for anchor in anchors],
        "gripperSourceSupports": supports, "gripperStiffnessNPerM": float(gripper_stiffness_n_per_m),
        "initialGripperTargetPolicy": "Correctly rounded exact binary source-barycentric positions; no cloth vertex adjustment. Any unavoidable initial binary64 rounding energy is retained.",
        "initialGripperEnergyJoules": initial_gripper_energy,
        "schedule": {"turnFractions": [0., .5], "angularSampleFractions": [index / 16 for index in range(9)],
            "holdFractions": [.5, .75], "releaseFractions": [.75, .875], "passiveTailFractions": [.875, 1.],
            "betweenKnotTargetMotion": "Piecewise linear material-point target chords; not a continuous rigid rotation.",
            "maximumChordRelativeContraction": 1 - math.cos(theta / 16)},
        "rigidReferenceClearanceMeters": {"minimumToSleevePlane": lower_gap,
            "maximumAboveSleevePlane": GAP_M + .030 * math.sin(theta),
            "assumedCore": .0001, "assumedActivationWidthBeyondCore": .0001,
            "scope": "Geometric reference only; three compliant grippers do not constrain all actual cloth vertices to this motion."},
        "limitations": ["Zero angle is the matched held-spring/passive-cloth control with identical schedule, offset and stiffness.",
            "All fabrics remain free; sleeve response can change relative rotation. World strip angle alone is not an observable of relative turning.",
            "Contact, strain, material frames, forces, discrete parameter work and release behavior require actual guarded simulation and independent replay.",
            "Full cut allowances, every original row and the unexecuted source phase plan remain unchanged. No trimming, rescaling, rest reset or continuous seam proof."]}
    source["sewingActuation"] = {"profile": SEWING_PROFILE, "accepted": False,
        "sourceSha256": sewing_source_identity(source), "mode": "distance",
        "initialTargetsMeters": targets.tolist(), "finalTargetsMeters": targets.tolist(),
        "schedule": {"profile": "sewing-row-activation-v1", "rowIds": row_ids,
            "knots": [{"fraction": fraction, "activation": activation.tolist()} for fraction in (0., 1.)]}}
    bind_sewing_activation(source, subdivisions, sewing_mode="distance")
    if any(canonical_encoded(source[key]) != canonical_encoded(unit[key]) for key in unit):
        raise ValueError("Original source unit changed while preparing controls")
    if read_regular(unit_path, 8 * 1024 ** 2) != unit_bytes or any(read_regular(path) != code[name] for name, path in paths.items()):
        raise ValueError("Source input or generator dependencies changed during preparation")
    canonical = encoded(source)
    placement = {"placedMeters": placed.tolist(), "canonicalDigest": sha(canonical), "accepted": False,
                 "intent": source["bindingFirstTurnDiagnostic"]["scope"], "rigidPlacements": poses}
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    snapshot = output / "source-snapshot"
    snapshot.mkdir(mode=0o700)
    for name, content in code.items():
        target = snapshot / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_bytes(target, content)
    atomic_bytes(output / "source-unit.json", unit_bytes)
    atomic_bytes(output / "canonical.json", canonical)
    atomic_bytes(output / "placement.json", encoded(placement))
    print(json.dumps({"prepared": True, "accepted": False, "angleDegrees": angle_degrees,
        "vertices": len(rest), "triangles": len(faces), "fabricInstances": len(unit["instances"]),
        "activeSeamRows": 5, "pendingSeamRows": 35, "grippers": 3, "subdivisions": subdivisions,
        "timePolicy": time_declaration,
        "canonicalSha256": sha(canonical), "minimumRigidReferenceGapMeters": lower_gap,
        "initialGripperEnergyJoules": initial_gripper_energy, "solverRun": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-unit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angle-degrees", type=float, default=3.)
    parser.add_argument("--gripper-stiffness-n-per-m", type=float, default=1.)
    parser.add_argument("--time-policy", choices=tuple(TIME_POLICIES), default="original-128ms-v1",
                        help="Declared total duration and subdivision count, both with a 2 ms nominal step")
    args = parser.parse_args()
    generate(args.source_unit, args.output, angle_degrees=args.angle_degrees,
             gripper_stiffness_n_per_m=args.gripper_stiffness_n_per_m, time_policy=args.time_policy)


if __name__ == "__main__":
    main()
