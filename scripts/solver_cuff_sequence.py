import copy
import json

import numpy as np

from solver_crease_mesh import split_straight_crease


def combine_cuff_controls(sewing, fold):
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
        if subdivision[key] != value:
            raise ValueError("Subdivision differs from reconstructed source parents")
    np.testing.assert_array_equal(panel[:, :2], expected["vertices"])
    np.testing.assert_array_equal(faces, expected["triangles"])
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
    frame_faces = []
    for row in rows:
        samples = row["sourceSamples"]
        if [sample["instanceId"] for sample in samples] != list(identities):
            raise ValueError("Ordered shell/facing source anchors required")
        expected_terms = sorted((sample["instanceId"], weight["vertex"], sign * weight["weight"])
            for sign, sample in zip((1, -1), samples) for weight in sample["weights"] if weight["weight"] != 0)
        if sorted((term["instanceId"], term["vertex"], term["coefficient"]) for term in row["terms"]) != expected_terms:
            raise ValueError("Seam terms differ from source anchors")
        remapped_terms = []
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
            parents = [index for index, triangle in enumerate(original_faces[:24])
                       if {weight["vertex"] for weight in weights}.issubset(set(triangle))]
            match = None
            for child, parent in enumerate(expected["parentTriangles"]):
                if parent not in parents:
                    continue
                triangle = faces[child]
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
                frame_faces.append((triangle + len(panel)).tolist())
        row["terms"] = sorted(remapped_terms, key=lambda term: (term["instanceId"], term["vertex"]))
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
            "recipe": "remapped-facing-source-parent-v1", "accepted": False},
        "foldActuation": actuator,
        "assemblySchedule": {"profile": "sewing-fold-progress-v1", "knots": [
            {"fraction": 0., "sewingProgress": 0., "foldProgress": 0.},
            {"fraction": .25, "sewingProgress": 1., "foldProgress": 0.},
            {"fraction": 1., "sewingProgress": 1., "foldProgress": 1.}]},
        "provenance": {"sourceSha256": sewing["provenance"]["sourceSha256"],
            "subdivision": subdivision, "accepted": False,
            "classification": "sequential cuff closure and two-layer allowance fold diagnostic; no turning"}}
