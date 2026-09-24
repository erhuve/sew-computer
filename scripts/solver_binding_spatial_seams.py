"""Continuous source-material curves, distinct from preserved point sewing rows.

The exact spatial geometry policy is explicit. No stitching forces, captured
admission, motion, contact acceptance or garment construction are installed.
"""
import copy
from fractions import Fraction as F
import hashlib
import math

from solver_binding_remap import _basis
from solver_binding_sewing_schedule import _phases, _reference
from solver_binding_source import CONTROL_FIELDS, INSTANCE, _bounded, _encoded, validate_binding_source
from solver_sewing_input import derive_sewing_row_ids

PROFILE = "source-binding-spatial-seam-map-v1"
CURVE_POLICY = "stored-arc-position-polyline-exact-v1"
MATERIAL_POLICY = "raw-lineage-original-mm-material-coordinates-v1"
TARGET_POLICY = "piecewise-affine-preserved-row-target-diagnostic-v1"
LIMITATIONS = [
    "Static source-bound material-curve mapping only; no controls installed, refined motion executed or captured source admitted.",
    "The curve follows captured arc/position polyline metadata, not exact Euclidean arclength or an unsampled source curve; recorded point discrepancies remain explicit.",
    "Exact lineage material coordinates and stored numerical rest coordinates remain distinct; no vertex, coefficient, original row or target is repaired or normalized.",
    "Material-coordinate pullback does not equate deformed refined vertices with interpolation of the original vertex prefix.",
    "Equivalent point operators on shared material edges do not select equivalent normal directors or resolve textile sides.",
    "Spatial target interpolation is an explicit diagnostic reference through preserved point targets, not a continuous stitching force law or stiffness density.",
    "Five held samples and thirty-five pending rows remain unchanged; fixed-state spatial distance bounds do not prove physical stitching, temporal paths, contact or construction.",
    "Stable motion, materials, body interaction, full-shirt integration and garment acceptance remain unresolved.",
]


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _rat(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _read(value):
    return F(int(value["numerator"]), int(value["denominator"]))


def _weights(values):
    return [{"vertex": vertex, "weight": _rat(weight)} for vertex, weight in sorted(values.items()) if weight]


def _unweights(values):
    return {item["vertex"]: _read(item["weight"]) for item in values}


def _cross(a, b):
    return a[0]*b[1] - a[1]*b[0]


def _minus(a, b):
    return [x-y for x, y in zip(a, b)]


def _area(points, face):
    a, b, c = (points[v] for v in face)
    return _cross(_minus(b, a), _minus(c, a))


def _bary(point, points, face):
    a, b, c = (points[v] for v in face)
    u, v, p = _minus(b, a), _minus(c, a), _minus(point, a)
    determinant = _cross(u, v)
    if determinant <= 0:
        raise ValueError("Positive exact material triangle orientation required")
    second, third = _cross(p, v)/determinant, _cross(u, p)/determinant
    return [1-second-third, second, third]


def _interpolate(a, b, fraction):
    return {vertex: value for vertex in set(a) | set(b)
            if (value := (1-fraction)*a.get(vertex, F()) + fraction*b.get(vertex, F()))}


def _pullback(weights, basis):
    result = {}
    for vertex, weight in weights.items():
        for original, amount in basis[vertex].items():
            result[original] = result.get(original, F()) + weight*amount
    return {v: w for v, w in result.items() if w}


def _tiling(original, original_faces, points, faces, parents, basis):
    proofs = []
    for parent, original_face in enumerate(original_faces):
        children = [i for i, value in enumerate(parents) if value == parent]
        edges, area = {}, F()
        for index in children:
            face = faces[index]
            if any(not set(basis[v]).issubset(original_face) for v in face):
                raise ValueError("Every material child must retain its original parent support")
            child_area = _area(points, face)
            if child_area <= 0:
                raise ValueError("Every exact material child must have positive area")
            area += child_area
            for a, b in zip(face, face[1:] + face[:1]):
                edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
        boundaries = []
        for entries in edges.values():
            if len(entries) == 1:
                boundaries.append(entries[0])
            elif len(entries) != 2 or entries[0] != entries[1][::-1]:
                raise ValueError("Material child interior edges must pair once with opposite orientation")
        sides, assigned = [], set()
        for first, second in zip(original_face, original_face[1:] + original_face[:1]):
            a, b = original[first], original[second]
            direction = _minus(b, a)
            axis = next(i for i in range(2) if direction[i])
            pieces = []
            for edge in boundaries:
                u, v = (points[i] for i in edge)
                if _cross(direction, _minus(u, a)) or _cross(direction, _minus(v, a)):
                    continue
                t0, t1 = (point[axis]-a[axis] for point in (u, v))
                t0, t1 = t0/direction[axis], t1/direction[axis]
                if not 0 <= t0 < t1 <= 1 or edge in assigned:
                    raise ValueError("Boundary material edges must follow one original side monotonically")
                assigned.add(edge)
                pieces.append((t0, t1, edge))
            pieces.sort()
            if (not pieces or pieces[0][0] != 0 or pieces[-1][1] != 1
                    or any(a[1] != b[0] for a, b in zip(pieces, pieces[1:]))):
                raise ValueError("Every original parent side must be exactly covered without overlap or gaps")
            sides.append({"originalVertices": [first, second], "pieces": [
                {"localVertices": list(edge), "interval": [_rat(t0), _rat(t1)]} for t0, t1, edge in pieces]})
        parent_area = _area(original, original_face)
        if assigned != set(boundaries) or parent_area <= 0 or area != parent_area:
            raise ValueError("Exact child area and complete original parent boundary must agree")
        proofs.append({"originalTriangleIndex": parent, "childTriangleIndices": children,
            "originalSignedDoubleAreaMmSquared": _rat(parent_area),
            "childSumSignedDoubleAreaMmSquared": _rat(area), "boundarySides": sides})
    return proofs


def _material(source):
    base, contexts, records, old_face_offset = source["baseUnit"], {}, [], 0
    refined_basis = _basis(source["bindingRefinement"])
    for instance in source["instances"]:
        name, template_id = instance["id"], instance["templateId"]
        template = base["sourceTemplates"][template_id]
        original = [[F(value) for value in point] for point in template["restPositions"]]
        mesh = source["numericalMeshes"][name]
        basis = ([{i: value for i, value in enumerate(row) if value} for row in refined_basis]
                 if name == INSTANCE else [{i: F(1)} for i in range(len(original))])
        if any(not row or any(v < 0 for v in row.values()) or sum(row.values(), F()) != 1 for row in basis):
            raise ValueError("Exactly unit, nonnegative material lineage required")
        points = [[sum((weight*original[v][axis] for v, weight in row.items()), F()) for axis in range(2)] for row in basis]
        faces, original_faces = mesh["triangles"], template["triangles"]
        parents = (source["bindingRefinement"]["ultimateOriginalTriangleIndices"] if name == INSTANCE
                   else list(range(len(faces))))
        context = {"name": name, "templateId": template_id, "template": template, "original": original,
            "points": points, "originalFaces": original_faces, "faces": faces, "parents": parents, "basis": basis,
            "oldFaceOffset": old_face_offset, "faceOffset": source["instanceTriangleOffsets"][name]}
        if len(points) != len(mesh["verticesMeters"]) or any(_area(points, face) <= 0 for face in faces):
            raise ValueError("Complete positive exact material mesh required")
        contexts[name] = context
        records.append({"instanceId": name, "templateId": template_id,
            "originalMeshSha256": _sha({"verticesMm": template["restPositions"], "triangles": original_faces}),
            "numericalMeshSha256": _sha(mesh), "materialCoordinatesSha256": _sha([[_rat(v) for v in point] for point in points]),
            "vertexLineageSha256": _sha([_weights(row) for row in basis]),
            "vertexOffset": source["instanceOffsets"][name], "triangleOffset": context["faceOffset"],
            "originalVertexOffset": base["instanceOffsets"][name], "originalTriangleOffset": old_face_offset,
            "vertexCount": len(points), "triangleCount": len(faces), "originalVertexCount": len(original),
            "originalTriangleCount": len(original_faces),
            "storedRestMinusMaterialMeters": [[_rat(F(stored[axis])-(point[axis]/1000 if axis < 2 else 0))
                for axis in range(3)] for stored, point in zip(mesh["verticesMeters"], points)],
            "originalParentTiling": _tiling(original, original_faces, points, faces, parents, basis) if name == INSTANCE else []})
        old_face_offset += len(original_faces)
    return contexts, records


def _clip(first, last):
    lower, upper = F(), F(1)
    for start, end in zip(first, last):
        slope = end-start
        if slope > 0:
            lower = max(lower, -start/slope)
        elif slope < 0:
            upper = min(upper, -start/slope)
        elif start < 0:
            return None
    return (lower, upper) if lower < upper else None


def _traversals(context, path, member, original):
    points, faces = ((context["original"], context["originalFaces"]) if original else (context["points"], context["faces"]))
    start, end = F(member["startArcMm"]), F(member["endArcMm"])
    if start >= end or member["direction"] not in ("forward", "reverse"):
        raise ValueError("Positive member arc interval and explicit direction required")
    origin, slope = (start, end-start) if member["direction"] == "forward" else (end, start-end)
    result = []
    for segment_index, segment in enumerate(path["segments"]):
        first, last = (path["samples"][i] for i in segment["samples"])
        s0, s1 = (F(item["arcMm"])-origin for item in (first, last))
        s0, s1 = s0/slope, s1/slope
        if s0 == s1:
            raise ValueError("Strictly positive stored path arc spans required")
        if max(s0, s1) <= 0 or min(s0, s1) >= 1:
            continue
        p0, p1 = ([F(v) for v in item["restPosition"]] for item in (first, last))
        for index, face in enumerate(faces):
            w0, w1 = _bary(p0, points, face), _bary(p1, points, face)
            interval = _clip(w0, w1)
            if interval is None:
                continue
            ends = [s0+(s1-s0)*value for value in interval]
            lower, upper = max(F(), min(ends)), min(F(1), max(ends))
            if lower >= upper:
                continue
            operators = []
            for s in (lower, upper):
                u = (s-s0)/(s1-s0)
                weights = {vertex: value for vertex, a, b in zip(face, w0, w1) if (value := (1-u)*a+u*b)}
                if any(v < 0 for v in weights.values()) or sum(weights.values(), F()) != 1:
                    raise ValueError("Exact traversal anchor must stay in its material face")
                operators.append(weights)
            result.append({"lower": lower, "upper": upper, "start": operators[0], "end": operators[1],
                           "face": index, "segment": segment_index})
    return result


def _operator(piece, fraction):
    return _interpolate(piece["start"], piece["end"], (fraction-piece["lower"])/(piece["upper"]-piece["lower"]))


def _member(source, context, member):
    paths = [path for path in context["template"]["stitchPaths"] if path["name"] == member["pathName"]]
    if len(paths) != 1:
        raise ValueError("Exactly one complete stored material stitch path required")
    path = paths[0]
    original = _traversals(context, path, member, True)
    numerical = _traversals(context, path, member, False)
    breaks = sorted({F(), F(1), *(p[key] for pieces in (original, numerical) for p in pieces for key in ("lower", "upper"))})
    if len(breaks) > 4097:
        raise ValueError("Bounded exact member traversal required")
    cells = []
    for lower, upper in zip(breaks, breaks[1:]):
        groups = [[p for p in pieces if p["lower"] <= lower and p["upper"] >= upper] for pieces in (original, numerical)]
        if any(not group for group in groups):
            raise ValueError("Stored material curve has an uncovered exact interval")
        segments = {p["segment"] for group in groups for p in group}
        if len(segments) != 1:
            raise ValueError("One original path segment must cover each exact traversal cell")
        cell = {"fractionInterval": [_rat(lower), _rat(upper)], "sourcePathSegmentIndex": next(iter(segments))}
        endpoints = []
        for is_original, label, group in zip((True, False), ("original", "numerical"), groups):
            group.sort(key=lambda p: p["face"])
            first = [_operator(group[0], s) for s in (lower, upper)]
            if any([_operator(p, s) for s in (lower, upper)] != first for p in group[1:]):
                raise ValueError("Overlapping material faces define different point operators")
            faces = context["originalFaces"] if is_original else context["faces"]
            offset = context["oldFaceOffset"] if is_original else context["faceOffset"]
            cell[label+"Candidates"] = [{"localTriangleIndex": p["face"], "canonicalTriangleIndex": offset+p["face"],
                "localVertices": list(faces[p["face"]]),
                "originalTriangleIndex": p["face"] if is_original else context["parents"][p["face"]]} for p in group]
            for ending, weights in zip(("StartWeights", "EndWeights"), first):
                cell[label+ending] = _weights(weights)
            endpoints.append(first)
        if any(_pullback(new, context["basis"]) != old for old, new in zip(*endpoints)):
            raise ValueError("Complete numerical material curve must pull back to its original point operator")
        if cells and any(cells[-1][label+"EndWeights"] != cell[label+"StartWeights"] for label in ("original", "numerical")):
            raise ValueError("Material point operators must be exactly continuous across traversal cells")
        cell["pullbackVerified"] = True
        cells.append(cell)
    panel = next(panel for panel in source["baseUnit"]["sourcePattern"]["panels"] if panel["id"] == context["templateId"])
    witnesses = []
    for index, sample in enumerate(path["samples"]):
        a, b = panel["points"][sample["sourceSegment"]:sample["sourceSegment"]+2]
        fraction = F(sample["sourceFraction"])
        residual = [F(point)-((1-fraction)*F(first)+fraction*F(last)) for point, first, last in zip(sample["restPosition"], a, b)]
        witnesses.append({"sampleIndex": index, "sourceSegment": sample["sourceSegment"], "sourceFraction": sample["sourceFraction"],
            "arcMm": sample["arcMm"], "sourceSampleSha256": _sha(sample),
            "storedPointMinusSourceInterpolationMm": [_rat(v) for v in residual]})
    return {"registrationMember": copy.deepcopy(member), "templateId": context["templateId"], "pathSha256": _sha(path),
            "sourcePathSamples": witnesses, "cells": cells}


def _at(member, fraction, version="numerical"):
    found = []
    for cell in member["cells"]:
        lower, upper = map(_read, cell["fractionInterval"])
        if lower <= fraction <= upper:
            found.append(_interpolate(_unweights(cell[version+"StartWeights"]), _unweights(cell[version+"EndWeights"]),
                                      (fraction-lower)/(upper-lower)))
    if not found or any(value != found[0] for value in found[1:]):
        raise ValueError("Unique continuous material point operator required at every declared fraction")
    return found[0]


def _combined(selector, fraction, version):
    result = {}
    for sign, member in zip((1, -1), selector["members"]):
        name = member["registrationMember"]["instanceId"]
        for vertex, weight in _at(member, fraction, version).items():
            result[name, vertex] = result.get((name, vertex), F()) + sign*weight
    return {key: value for key, value in result.items() if value}


def _raw(row):
    return {(term["instanceId"], term["vertex"]): F(term["coefficient"]) for term in row["terms"]}


def _residual(continuum, raw):
    return {key: value for key in set(continuum) | set(raw) if (value := continuum.get(key, F())-raw.get(key, F()))}


def _terms(values):
    return [{"instanceId": name, "vertex": vertex, "coefficient": _rat(value)} for (name, vertex), value in sorted(values.items())]


def build_binding_spatial_seams(source, reference_source):
    inputs = (source, reference_source)
    before = [_bounded(value) for value in inputs]
    try:
        if type(source) is not dict or CONTROL_FIELDS.intersection(source):
            raise ValueError("Bare refined source required for static material seam mapping")
        validate_binding_source(source)
        base = source["baseUnit"]
        recipe, row_ids = _reference(base, reference_source, 1)
        if list(derive_sewing_row_ids(source)) != row_ids:
            raise ValueError("Original and numerical row identities must match")
        contexts, meshes = _material(source)
        rows, originals = source["embeddedConstraints"]["constraints"], base["embeddedConstraints"]["constraints"]
        phases, groups = _phases(base), {}
        for index, row in enumerate(rows):
            groups.setdefault((row["registrationId"], row["memberIndex"]), []).append(index)
        selectors, mapping = [], {}
        for (registration_id, member_index), indices in groups.items():
            if [F(rows[i]["fraction"]) for i in indices] != [F(i, 4) for i in range(5)]:
                raise ValueError("Complete five-row source selector sampling required")
            registration = next(r for r in base["embeddedConstraints"]["registrations"] if r["id"] == registration_id)
            members = [registration["members"][i] for i in (0, member_index)]
            phase_ids = {phases[i] for i in indices}
            if len(phase_ids) != 1:
                raise ValueError("Each spatial registration retains one source construction phase")
            selector = {"registrationId": registration_id, "memberIndex": member_index, "rowIndices": indices,
                "rowIds": [row_ids[i] for i in indices], "phaseId": next(iter(phase_ids)), "held": indices == list(range(5)),
                "members": [_member(source, contexts[m["instanceId"]], m) for m in members],
                "targetKnots": [{"fraction": _rat(F(rows[i]["fraction"])), "targetMeters": recipe["initialTargetsMeters"][i]} for i in indices]}
            selectors.append(selector)
            mapping.update({i: selector for i in indices})
        if len(selectors) != 8 or set(mapping) != set(range(40)) or sum(s["held"] for s in selectors) != 1:
            raise ValueError("All eight selectors and forty rows required")
        comparisons = []
        for index, (old, new) in enumerate(zip(originals, rows)):
            selector, fraction = mapping[index], F(new["fraction"])
            old_raw, new_raw = _raw(old), _raw(new)
            old_diff, new_diff = (_residual(_combined(selector, fraction, version), raw)
                                  for version, raw in (("original", old_raw), ("numerical", new_raw)))
            anchors = []
            for sign, member in zip((1, -1), selector["members"]):
                name = member["registrationMember"]["instanceId"]
                anchors.append({"instanceId": name, "sign": sign,
                    "originalStoredCoefficientSum": _rat(sum((v for (n, _), v in old_raw.items() if n == name), F())),
                    "numericalStoredCoefficientSum": _rat(sum((v for (n, _), v in new_raw.items() if n == name), F())),
                    "originalResidualL1": _rat(sum((abs(v) for (n, _), v in old_diff.items() if n == name), F())),
                    "numericalResidualL1": _rat(sum((abs(v) for (n, _), v in new_diff.items() if n == name), F()))})
            comparisons.append({"rowIndex": index, "rowId": row_ids[index], "originalRowSha256": _sha(old),
                "numericalRowSha256": _sha(new), "originalResidualTerms": _terms(old_diff), "numericalResidualTerms": _terms(new_diff),
                "perAnchor": anchors})
        result = {"profile": PROFILE, "curvePolicy": CURVE_POLICY, "materialPolicy": MATERIAL_POLICY, "targetPolicy": TARGET_POLICY,
            "sourceSha256": _sha(source), "baseUnitSha256": _sha(base),
            "refinementDescriptorSha256": _sha(source["bindingRefinement"]), "seamRemapDescriptorSha256": _sha(source["bindingSeamRemap"]),
            "referenceSourceSha256": _sha(reference_source), "referenceSewingActuationSha256": _sha(recipe),
            "originalBundleSha256": _sha(base["embeddedConstraints"]), "numericalBundleSha256": _sha(source["embeddedConstraints"]),
            "sourcePhasePlanSha256": _sha(base["phasePlan"]), "sourcePhaseConstraintRowsSha256": _sha(base["phaseConstraintRows"]),
            "materialMeshes": meshes, "selectors": selectors, "rowComparisons": comparisons,
            "initialTargetsMeters": copy.deepcopy(recipe["initialTargetsMeters"]),
            "finalTargetsMeters": copy.deepcopy(recipe["finalTargetsMeters"]), "complianceMPerN": 1e-8,
            "sewingControlSchedule": copy.deepcopy(recipe["schedule"]), "heldRowIndices": list(range(5)), "pendingRowIndices": list(range(5, 40)),
            "materialCurveCoverageVerified": True,
            **{key: False for key in ("accepted", "solverReady", "executable", "controlsInstalled", "continuousSpatialStitchingVerified",
                                      "constructionPhaseCompleted", "historicalExtrasInherited")}, "limitations": list(LIMITATIONS)}
        if before != [_bounded(value) for value in inputs]:
            raise ValueError("Material seam inputs changed during static derivation")
        _bounded(result)
        return copy.deepcopy(result)
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Malformed source-bound spatial seam input") from error


def validate_binding_spatial_seams(source, reference_source, descriptor):
    claimed = _bounded(descriptor)
    expected = build_binding_spatial_seams(source, reference_source)
    if claimed != _encoded(expected):
        raise ValueError("Material seam mapping differs from complete source rederivation")
    return expected


def _target(selector, fraction):
    knots = selector["targetKnots"]
    for first, last in zip(knots, knots[1:]):
        lower, upper = _read(first["fraction"]), _read(last["fraction"])
        if lower <= fraction <= upper:
            u = (fraction-lower)/(upper-lower)
            return (1-u)*F(first["targetMeters"])+u*F(last["targetMeters"])
    raise ValueError("Spatial diagnostic target is not defined at this fraction")


def _vector(terms, positions, offsets):
    return [sum((coefficient*F(positions[offsets[name]+vertex][axis]) for (name, vertex), coefficient in terms.items()), F())
            for axis in range(3)]


def evaluate_binding_spatial_seams(source, reference_source, descriptor, positions, *, tolerance_m):
    """Bound declared spatial distances at one supplied state; no trajectory."""
    from solver_spatial_sewing import verify_spatial_distance_exact
    inputs = (source, reference_source, descriptor, positions, tolerance_m)
    before = [_bounded(value) for value in inputs]
    validated = validate_binding_spatial_seams(source, reference_source, descriptor)
    if (type(positions) is not list or len(positions) != len(source["restMeters"])
            or any(type(point) is not list or len(point) != 3 for point in positions)):
        raise ValueError("Complete current numerical vertex positions required")
    for point in positions:
        for value in point:
            if (type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e6
                    or F(value) != F(float(value))):
                raise ValueError("Bounded finite exactly binary64 raw positions required")
    results, rows, offsets = [], [], source["instanceOffsets"]
    for selector in validated["selectors"]:
        breaks = sorted({_read(item) for member in selector["members"] for cell in member["cells"] for item in cell["fractionInterval"]}
                        | {_read(knot["fraction"]) for knot in selector["targetKnots"]})
        cells = []
        for lower, upper in zip(breaks, breaks[1:]):
            vectors = [_vector(_combined(selector, fraction, "numerical"), positions, offsets) for fraction in (lower, upper)]
            cells.append({"interval": [_rat(lower), _rat(upper)],
                "startDifferenceMeters": [_rat(v) for v in vectors[0]], "endDifferenceMeters": [_rat(v) for v in vectors[1]],
                "startTargetMeters": _rat(_target(selector, lower)), "endTargetMeters": _rat(_target(selector, upper))})
        certificate = verify_spatial_distance_exact(cells, tolerance_m=tolerance_m)
        results.append({"registrationId": selector["registrationId"], "memberIndex": selector["memberIndex"],
            "held": selector["held"], "rowIndices": list(selector["rowIndices"]), "certificate": certificate})
        for index in selector["rowIndices"]:
            row = source["embeddedConstraints"]["constraints"][index]
            fraction = F(row["fraction"])
            raw = _vector(_raw(row), positions, offsets)
            continuous = _vector(_combined(selector, fraction, "numerical"), positions, offsets)
            rows.append({"rowIndex": index, "rowId": validated["rowComparisons"][index]["rowId"], "held": selector["held"],
                "rawNumericalVectorMeters": [_rat(v) for v in raw], "continuumVectorMeters": [_rat(v) for v in continuous],
                "continuumMinusRawMeters": [_rat(v-u) for u, v in zip(raw, continuous)],
                "rawSquaredGapMetersSquared": _rat(sum((v*v for v in raw), F())),
                "continuumSquaredGapMetersSquared": _rat(sum((v*v for v in continuous), F())),
                "targetMeters": validated["initialTargetsMeters"][index]})
    report = {"profile": "source-binding-spatial-seam-evaluation-v1", "verified": True, "accepted": False,
        "sourceSha256": validated["sourceSha256"], "referenceSourceSha256": validated["referenceSourceSha256"],
        "mapDescriptorSha256": _sha(validated), "positionsSha256": _sha(positions), "toleranceMeters": tolerance_m,
        "selectors": results, "rowDiagnostics": sorted(rows, key=lambda row: row["rowIndex"]),
        "activeSpatialDistanceBoundSatisfied": all(item["certificate"]["withinTolerance"] for item in results if item["held"]),
        "controlsInstalled": False, "constructionPhaseCompleted": False, "continuousSpatialStitchingVerified": False,
        "limitations": list(LIMITATIONS),
        "scope": "Exact fixed-state spatial bounds for the declared material curves and target interpolation. Pending curves remain pending; no temporal, contact, force-law or construction acceptance."}
    if before != [_bounded(value) for value in inputs]:
        raise ValueError("Spatial diagnostic inputs changed during evaluation")
    _bounded(report)
    return copy.deepcopy(report)
