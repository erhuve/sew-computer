import copy
import json

import numpy as np

from solver_crease_mesh import split_straight_crease


def combine_cuff_controls(sewing, fold, *, crease_frame_region=None):
    if crease_frame_region not in (None, "body", "allowance"):
        raise ValueError("Crease frame region must be explicitly body or allowance")
    identities = ("cuff_left:shell", "cuff_left:facing")
    original = np.asarray(sewing["restMeters"], dtype=float)
    original_faces = np.asarray(sewing["triangles"]).reshape((-1, 3))
    panel = np.asarray(fold["restMeters"], dtype=float)
    faces = np.asarray(fold["triangles"]).reshape((-1, 3))
    if (sewing["instanceOffsets"] != {identities[0]: 0, identities[1]: 20}
            or fold["instanceOffsets"] != {identities[0]: 0}
            or original.shape != (40, 3) or original_faces.shape != (48, 3)
            or panel.shape != (33, 3) or faces.shape != (48, 3)
            or sewing["provenance"]["sourceSha256"] != fold["provenance"]["sourceSha256"]):
        raise ValueError("Matching reviewed source-cuff sewing and allowance controls required")
    np.testing.assert_array_equal(original[:20], original[20:])
    np.testing.assert_array_equal(original[:20], panel[:20])
    np.testing.assert_array_equal(original_faces[:24], original_faces[24:] - 20)
    np.testing.assert_array_equal(panel[:, 2], 0.)
    subdivision = fold["provenance"]["subdivision"]
    path = subdivision["sourceStitchPath"]
    if path["name"] != "outer":
        raise ValueError("Explicit outer allowance crease required")
    line = np.asarray([path["samples"][index]["restPosition"] for index in (0, -1)]) * .001
    expected = json.loads(json.dumps(split_straight_crease(original[:20, :2], original_faces[:24], line)))
    for key, value in expected.items():
        if key in ("triangles", "parentTriangles"):
            continue
        if subdivision[key] != value:
            raise ValueError("Subdivision differs from reconstructed source parents")
    # Child enumeration is not a material-frame declaration. Verify each exact
    # oriented child and its parent, but permit a consistent enumeration change.
    if (len(subdivision["parentTriangles"]) != len(faces)
            or any(type(parent) is not int for parent in subdivision["parentTriangles"])
            or sorted((parent, tuple(triangle)) for parent, triangle in
                      zip(subdivision["parentTriangles"], subdivision["triangles"]))
            != sorted((parent, tuple(triangle)) for parent, triangle in
                      zip(expected["parentTriangles"], expected["triangles"]))):
        raise ValueError("Subdivision differs from reconstructed source parents")
    np.testing.assert_array_equal(panel[:, :2], expected["vertices"])
    np.testing.assert_array_equal(faces, subdivision["triangles"])
    np.testing.assert_array_equal(fold["placedMeters"], panel)
    placed_original = np.asarray(sewing["placedMeters"])
    np.testing.assert_array_equal(placed_original[:20], original[:20])
    normal = np.cross(original[original_faces[0, 1]] - original[original_faces[0, 0]],
                      original[original_faces[0, 2]] - original[original_faces[0, 0]])
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(placed_original[20:], original[20:] + .002 * normal, rtol=0, atol=1e-14)
    rows = copy.deepcopy(sewing["embeddedConstraints"]["constraints"])
    if len(rows) != 20:
        raise ValueError("All twenty source cuff registrations required")
    direction = line[1] - line[0]
    crease_normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    tolerance = max(np.linalg.norm(np.ptp(panel[:, :2], axis=0)) * 1e-12, 1e-14)
    distances = (panel[:, :2] - line[0]) @ crease_normal
    distances[np.abs(distances) <= tolerance] = 0.
    off_crease_sides, frame_requests = set(), []
    for row in rows:
        samples = row["sourceSamples"]
        if [sample["instanceId"] for sample in samples] != list(identities):
            raise ValueError("Ordered shell/facing source anchors required")
        if samples[0]["pathName"] != samples[1]["pathName"]:
            raise ValueError("Paired source registrations must name the same cuff path")
        expected_terms = sorted((sample["instanceId"], weight["vertex"], sign * weight["weight"])
            for sign, sample in zip((1, -1), samples) for weight in sample["weights"] if weight["weight"] != 0)
        if sorted((term["instanceId"], term["vertex"], term["coefficient"]) for term in row["terms"]) != expected_terms:
            raise ValueError("Seam terms differ from source anchors")
        remapped_terms, coordinates = [], []
        for sign, sample in zip((1, -1), samples):
            weights = sample["weights"]
            if (not 1 <= len(weights) <= 3 or len({weight["vertex"] for weight in weights}) != len(weights)
                    or any(type(weight["vertex"]) is not int or not 0 <= weight["vertex"] < 20
                           or type(weight["weight"]) not in (float, int) or not np.isfinite(weight["weight"])
                           or not 0 <= weight["weight"] <= 1 for weight in weights)
                    or abs(sum(weight["weight"] for weight in weights) - 1) > 1e-12):
                raise ValueError("Normalized source-triangle anchor required")
            coordinate = sum(weight["weight"] * original[weight["vertex"], :2] for weight in weights)
            np.testing.assert_allclose(coordinate * 1000, sample["restPositionMm"], rtol=0, atol=1e-9)
            coordinates.append(coordinate)
            signed_distance = float((coordinate - line[0]) @ crease_normal)
            on_crease = bool(abs(signed_distance) <= tolerance)
            if sample["pathName"] == "outer" and not on_crease:
                raise ValueError("Outer registrations must lie on the source outer crease")
            if not on_crease:
                off_crease_sides.add(1 if signed_distance > 0 else -1)
            parents = [index for index, triangle in enumerate(original_faces[:24])
                       if {weight["vertex"] for weight in weights}.issubset(set(triangle))]
            match = None
            for child, parent in enumerate(expected["parentTriangles"]):
                if parent not in parents:
                    continue
                # Preserve the original source interpolation independently of
                # the explicitly chosen frame or the input child enumeration.
                triangle = np.asarray(expected["triangles"][child])
                barycentric = np.linalg.solve(np.vstack((panel[triangle, :2].T, np.ones(3))), [*coordinate, 1.])
                if np.min(barycentric) >= -1e-12:
                    barycentric[np.abs(barycentric) < 1e-12] = 0.
                    barycentric /= barycentric.sum()
                    np.testing.assert_allclose(barycentric @ panel[triangle, :2], coordinate, rtol=0, atol=1e-12)
                    match = triangle, barycentric, parent
                    break
            if match is None:
                raise ValueError("No source-parent child triangle contains the seam anchor")
            triangle, barycentric, parent = match
            sample["originalWeights"] = weights
            sample["parentTriangle"] = parent
            sample["weights"] = [{"vertex": int(vertex), "weight": float(weight)}
                                 for vertex, weight in zip(triangle, barycentric) if weight != 0]
            remapped_terms.extend({"instanceId": sample["instanceId"], "vertex": weight["vertex"],
                                   "coefficient": sign * weight["weight"]} for weight in sample["weights"])
            if sign == -1:
                frame_requests.append((parents, {weight["vertex"] for weight in sample["weights"]}, on_crease))
        if not np.allclose(coordinates[0], coordinates[1], rtol=0., atol=1e-12):
            raise ValueError("Paired source registrations must preserve coincident material anchors")
        row["terms"] = sorted(remapped_terms, key=lambda term: (term["instanceId"], term["vertex"]))
    if len(off_crease_sides) != 1:
        raise ValueError("Off-crease source registrations must unambiguously identify one body halfspace")
    body_side = off_crease_sides.pop()
    frame_faces, frame_bindings = [], []
    for parents, support, on_crease in frame_requests:
        matches = []
        for triangle, parent in zip(faces, subdivision["parentTriangles"]):
            if parent not in parents or not support.issubset(set(triangle)):
                continue
            sides = set(np.sign(distances[triangle]).astype(int)) - {0}
            if len(sides) != 1:
                raise ValueError("Frame child must lie strictly on one side of the source crease")
            region = "body" if sides.pop() == body_side else "allowance"
            matches.append((parent, triangle, region))
        if not matches:
            raise ValueError("No source-parent frame child contains the negative anchor support")
        regions = {match[2] for match in matches}
        if len(regions) > 1 and crease_frame_region is None:
            raise ValueError("Ambiguous crease frame: explicitly choose body or allowance with --crease-frame-region")
        if on_crease and crease_frame_region is not None:
            matches = [match for match in matches if match[2] == crease_frame_region]
        if not matches:
            raise ValueError("Requested crease frame region has no source-parent child containing the negative anchor support")
        parent, triangle, region = min(matches, key=lambda match: (
            tuple(sorted(original_faces[match[0]].tolist())), tuple(sorted(match[1].tolist()))))
        frame_faces.append((triangle + len(panel)).tolist())
        frame_bindings.append({"instanceId": identities[1], "sourceParentTriangle": parent,
            "sourceParentVertices": sorted(original_faces[parent].tolist()),
            "childVertices": triangle.tolist(), "negativeAnchorVertices": sorted(support),
            "onCrease": on_crease, "region": region})
    actuator = copy.deepcopy(fold["foldActuation"])
    hinges = np.asarray(actuator["hinges"])
    for key in ("stiffnessJoules", "initialAnglesRadians", "targetAnglesRadians"):
        actuator[key] = actuator[key] * 2
    actuator["hinges"] = np.concatenate((hinges, hinges + len(panel))).tolist()
    actuator["recipe"] = "two-layer-source-cuff-outer-allowance-v1"
    actuator["limitations"] = "Both parallel layers fold in the same signed direction; not cuff turning"
    rest = np.concatenate((panel, panel))
    placed = np.concatenate((panel, panel + .002 * normal))
    return {"restMeters": rest.tolist(), "placedMeters": placed.tolist(),
        "triangles": np.concatenate((faces, faces + len(panel))).ravel().tolist(),
        "instanceOffsets": {identities[0]: 0, identities[1]: len(panel)},
        "embeddedConstraints": {"constraints": rows},
        "sewingFrames": {"faces": frame_faces, "sides": [-1] * len(rows),
            "recipe": "explicit-facing-source-parent-region-v2", "accepted": False,
            "creaseFrameRegion": crease_frame_region,
            "bodyHalfspace": {"sourcePathName": "outer", "creaseEndpointsMeters": line.tolist(),
                "leftNormal": crease_normal.tolist(), "sign": body_side,
                "basis": "all off-crease source registrations occupy one strict halfspace",
                "toleranceMeters": float(tolerance)},
            "selectionPolicy": "requested crease region, then lexicographic source-parent and child vertex identities",
            "bindings": frame_bindings,
            "limitations": "Explicit normal-seam frame choice for this parallel source-cuff diagnostic; not turning, layer order or accepted construction"},
        "foldActuation": actuator,
        "assemblySchedule": {"profile": "sewing-fold-progress-v1", "knots": [
            {"fraction": 0., "sewingProgress": 0., "foldProgress": 0.},
            {"fraction": .25, "sewingProgress": 1., "foldProgress": 0.},
            {"fraction": 1., "sewingProgress": 1., "foldProgress": 1.}]},
        "provenance": {"sourceSha256": sewing["provenance"]["sourceSha256"],
            "subdivision": subdivision, "accepted": False,
            "classification": "sequential cuff closure and two-layer allowance fold diagnostic; no turning"}}
