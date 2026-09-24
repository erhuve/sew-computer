"""Independent exact static material-curve mapping; no solver or producer.

Pattern/split rederivation remains a separate source-validator prerequisite.
This verifies supplied-source topology, lineage, clipping and point operators;
it neither installs those operators nor gives five samples spatial authority.
"""

from fractions import Fraction as F
import hashlib

from solver_binding_remap_replay import verify_binding_remap
from solver_binding_sewing_schedule_replay import (
    _encoded, _number, _phase_membership, _reference_recipe, _row_identity,
)


PROFILE = "source-binding-spatial-seam-map-v1"
CURVE_POLICY = "stored-arc-position-polyline-exact-v1"
MATERIAL_POLICY = "raw-lineage-original-mm-material-coordinates-v1"
TARGET_POLICY = "piecewise-affine-preserved-row-target-diagnostic-v1"
INSTANCE = "opening_binding_left_left:shell"
REQUIRED_HELPER_FILES = ("solver_binding_remap_replay.py", "solver_binding_sewing_schedule_replay.py",
    "solver_binding_control_schedule_replay.py", "solver_binding_fold_replay.py", "solver_binding_placement_replay.py")
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


def _same(a, b, label):
    if _encoded(a) != _encoded(b):
        raise ValueError("Independent spatial mapping mismatch: " + label)


def _rat(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _sparse(weights):
    return [{"vertex": index, "weight": _rat(weight)} for index, weight in sorted(weights.items()) if weight]


def _read_sparse(weights):
    return {v["vertex"]: F(int(v["weight"]["numerator"]), int(v["weight"]["denominator"])) for v in weights}


def _index(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError("Bounded non-Boolean spatial index required")
    return value


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _basis(source):
    refinement = source["bindingRefinement"]
    basis = [{i: F(1)} for i in range(len(refinement["originalLocalMesh"]["verticesMeters"]))]
    faces = refinement["originalLocalMesh"]["triangles"]
    parents = list(range(len(faces)))
    for stage in refinement["stages"]:
        output = stage["output"]
        following, supports = [], []
        for support in output["sourceWeights"]:
            row, intermediate = {}, set()
            for key, raw in support.items():
                if type(key) is not str or not key.isascii() or not key.isdigit() or str(int(key)) != key:
                    raise ValueError("Canonical raw lineage index required")
                index, weight = _index(int(key), len(basis)), _number(raw, 0, 1)
                if weight:
                    intermediate.add(index)
                    for original, coefficient in basis[index].items():
                        row[original] = row.get(original, F()) + weight*coefficient
            if not row or sum(row.values(), F()) != 1 or any(v < 0 for v in row.values()):
                raise ValueError("Exact nonnegative unit material lineage required")
            following.append(row)
            supports.append(intermediate)
        next_parents = []
        for face, parent in zip(output["triangles"], output["parentTriangles"]):
            parent = _index(parent, len(faces))
            if any(not supports[v] <= set(faces[parent]) for v in face):
                raise ValueError("Material child escapes its immediate parent")
            next_parents.append(parents[parent])
        basis, faces, parents = following, output["triangles"], next_parents
    return basis, parents


def _tiling(original_points, original_faces, points, faces, lineage, parents):
    """Exact oriented cell boundary proof, separately for every original face."""
    result = []
    if len(faces) != len(parents):
        raise ValueError("Every material child requires an original parent")
    for parent, original in enumerate(original_faces):
        children = [i for i, p in enumerate(parents) if p == parent]
        if not children:
            raise ValueError("Original material triangle omitted")
        edges, total = {}, F()
        for child in children:
            face = faces[child]
            area = _cross(*(points[v] for v in face))
            if area <= 0:
                raise ValueError("Strict positive material child area required")
            total += area
            for vertex in face:
                row = lineage[vertex]
                if (sum(row.values(), F()) != 1 or any(w < 0 for w in row.values())
                        or not set(row) <= set(original)):
                    raise ValueError("Child material basis must remain inside its original parent")
            for a, b in zip(face, face[1:]+face[:1]):
                edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
        sides = [[] for _ in range(3)]
        for incident in edges.values():
            if len(incident) == 2:
                if incident[0] != incident[1][::-1]:
                    raise ValueError("Interior material edges must pair oppositely")
                continue
            if len(incident) != 1:
                raise ValueError("Material edge incidence must be one or two")
            a, b = incident[0]
            matches = []
            for side, (first, last) in enumerate(zip(original, original[1:]+original[:1])):
                if set(lineage[a]) <= {first, last} and set(lineage[b]) <= {first, last}:
                    lower, upper = lineage[a].get(last, F()), lineage[b].get(last, F())
                    if not 0 <= lower < upper <= 1:
                        raise ValueError("Boundary material interval must follow original winding")
                    matches.append(side)
                    sides[side].append((lower, upper, a, b))
            if len(matches) != 1:
                raise ValueError("Every material boundary piece must lie on exactly one original side")
        boundary = []
        for (a, b), pieces in zip(zip(original, original[1:]+original[:1]), sides):
            pieces.sort()
            if (not pieces or pieces[0][0] != 0 or pieces[-1][1] != 1
                    or any(left[1] != right[0] for left, right in zip(pieces, pieces[1:]))):
                raise ValueError("Material boundary intervals must tile the complete original side exactly")
            boundary.append({"originalVertices": [a, b], "pieces": [
                {"localVertices": [u, v], "interval": [_rat(lo), _rat(hi)]} for lo, hi, u, v in pieces]})
        original_area = _cross(*(original_points[v] for v in original))
        if original_area <= 0 or total != original_area:
            raise ValueError("Material child signed areas must exactly equal their original parent")
        result.append({"originalTriangleIndex": parent, "childTriangleIndices": children,
            "originalSignedDoubleAreaMmSquared": _rat(original_area), "childSumSignedDoubleAreaMmSquared": _rat(total),
            "boundarySides": boundary})
    return result


def _meshes(source):
    base = source["baseUnit"]
    refined_basis, refined_parents = _basis(source)
    objects, evidence, old_vertex, old_triangle = {}, [], 0, 0
    for instance in source["instances"]:
        name, template_id = instance["id"], instance["templateId"]
        template = base["sourceTemplates"][template_id]
        original_points = [[_number(x, -100000, 100000) for x in p] for p in template["restPositions"]]
        if any(len(p) != 2 for p in original_points):
            raise ValueError("Original two-dimensional material coordinates required")
        original_faces = template["triangles"]
        mesh = source["numericalMeshes"][name]
        lineage = refined_basis if name == INSTANCE else [{i: F(1)} for i in range(len(original_points))]
        parents = refined_parents if name == INSTANCE else list(range(len(original_faces)))
        points = [[sum((weight*original_points[v][axis] for v, weight in row.items()), F()) for axis in (0, 1)] for row in lineage]
        faces = mesh["triangles"]
        if len(points) != len(mesh["verticesMeters"]) or any(_cross(*(points[v] for v in f)) <= 0 for f in faces):
            raise ValueError("Complete positive material mesh required")
        tiling = _tiling(original_points, original_faces, points, faces, lineage, parents) if name == INSTANCE else []
        _same(base["instanceOffsets"][name], old_vertex, "original cumulative vertex offsets")
        difference = [[_rat(_number(value, -100, 100)-(point[axis]/1000 if axis < 2 else F()))
                       for axis, value in enumerate(stored)] for point, stored in zip(points, mesh["verticesMeters"])]
        evidence.append({"instanceId": name, "templateId": template_id,
            "originalMeshSha256": _sha({"verticesMm": template["restPositions"], "triangles": original_faces}),
            "numericalMeshSha256": _sha(mesh), "materialCoordinatesSha256": _sha([[_rat(x) for x in p] for p in points]),
            "vertexLineageSha256": _sha([_sparse(row) for row in lineage]),
            "vertexOffset": source["instanceOffsets"][name], "triangleOffset": source["instanceTriangleOffsets"][name],
            "originalVertexOffset": old_vertex, "originalTriangleOffset": old_triangle,
            "vertexCount": len(points), "triangleCount": len(faces), "originalVertexCount": len(original_points),
            "originalTriangleCount": len(original_faces), "storedRestMinusMaterialMeters": difference,
            "originalParentTiling": tiling})
        objects[name] = {"templateId": template_id, "originalPoints": original_points, "originalFaces": original_faces,
            "points": points, "faces": faces, "lineage": lineage, "parents": parents,
            "originalTriangleOffset": old_triangle, "triangleOffset": source["instanceTriangleOffsets"][name]}
        old_vertex += len(original_points)
        old_triangle += len(original_faces)
    return objects, evidence


def _barycentric(point, points, face):
    a, b, c = [points[v] for v in face]
    determinant = _cross(a, b, c)
    if determinant <= 0:
        raise ValueError("Positive oriented material face required")
    second, third = _cross(a, point, c)/determinant, _cross(a, b, point)/determinant
    return [1-second-third, second, third]


def _clip(start, end, lower=F(), upper=F(1)):
    """Clip a barycentric affine field to its nonnegative closed triangle."""
    lo, hi = lower, upper
    for a, b in zip(start, end):
        slope = (b-a)/(upper-lower)
        if slope > 0:
            lo = max(lo, lower-a/slope)
        elif slope < 0:
            hi = min(hi, lower-a/slope)
        elif a < 0:
            return None
    return (lo, hi) if lo < hi else None


def _at(candidate, fraction):
    _, _, _, domain_lo, domain_hi, first, last = candidate
    t = (fraction-domain_lo)/(domain_hi-domain_lo)
    return [a+t*(b-a) for a, b in zip(first, last)]


def _operator(face, weights):
    if min(weights) < 0 or sum(weights, F()) != 1:
        raise ValueError("Exact nonnegative unit spatial point operator required")
    return {vertex: weight for vertex, weight in zip(face, weights) if weight}


def _clipped_faces(points, faces, start, end, lower, upper):
    result = []
    for index, face in enumerate(faces):
        first, last = _barycentric(start, points, face), _barycentric(end, points, face)
        interval = _clip(first, last, lower, upper)
        if interval is not None:
            result.append((*interval, index, lower, upper, first, last))
    return result


def _pullback(weights, lineage):
    result = {}
    for vertex, weight in weights.items():
        for original, coefficient in lineage[vertex].items():
            result[original] = result.get(original, F()) + weight*coefficient
    return {i: value for i, value in result.items() if value}


def _member_map(source, member, mesh):
    if (type(member) is not dict or set(member) != {"instanceId", "pathName", "startArcMm", "endArcMm", "direction"}
            or member["direction"] not in ("forward", "reverse")):
        raise ValueError("Explicit bounded source registration member required")
    template = source["baseUnit"]["sourceTemplates"][mesh["templateId"]]
    paths = [p for p in template["stitchPaths"] if p["name"] == member["pathName"]]
    panels = [p for p in source["baseUnit"]["sourcePattern"]["panels"] if p["id"] == mesh["templateId"]]
    if len(paths) != 1 or len(panels) != 1:
        raise ValueError("Unique source panel and stored stitch path required")
    path, panel = paths[0], panels[0]
    length = _number(path["lengthMm"], 0, 100000, positive=True)
    start = _number(member["startArcMm"], 0, float(length))
    end = _number(member["endArcMm"], 0, float(length))
    if start >= end:
        raise ValueError("Positive registration arc interval required")
    arc0, arc1 = (start, end) if member["direction"] == "forward" else (end, start)
    samples, segments = path["samples"], path["segments"]
    if (type(samples) is not list or not 2 <= len(samples) <= 100000
            or type(segments) is not list or len(segments) != len(samples)-1):
        raise ValueError("Complete bounded stored path required")
    sample_evidence, arcs, positions = [], [], []
    for index, sample in enumerate(samples):
        source_segment = _index(sample["sourceSegment"], len(panel["points"])-1)
        u = _number(sample["sourceFraction"], 0, 1)
        arc = _number(sample["arcMm"], 0, float(length))
        point = [_number(x, -100000, 100000) for x in sample["restPosition"]]
        if len(point) != 2 or arcs and arc <= arcs[-1]:
            raise ValueError("Strictly increasing path arcs and two-dimensional points required")
        a, b = [[_number(x, -100000, 100000) for x in p] for p in panel["points"][source_segment:source_segment+2]]
        sample_evidence.append({"sampleIndex": index, "sourceSegment": source_segment,
            "sourceFraction": sample["sourceFraction"], "arcMm": sample["arcMm"], "sourceSampleSha256": _sha(sample),
            "storedPointMinusSourceInterpolationMm": [_rat(p-(x+u*(y-x))) for p, x, y in zip(point, a, b)]})
        arcs.append(arc)
        positions.append(point)
    if arcs[0] != 0 or arcs[-1] != length:
        raise ValueError("Stored path must retain its complete arc endpoints")
    cells = []
    for index, segment in enumerate(segments):
        _same(segment["samples"], [index, index+1], "ordered stored path segment endpoints")
        mapped = [(arcs[i]-arc0)/(arc1-arc0) for i in (index, index+1)]
        lower, upper = max(F(), min(mapped)), min(F(1), max(mapped))
        if lower >= upper:
            continue
        def point_at(fraction):
            t = (arc0+fraction*(arc1-arc0)-arcs[index])/(arcs[index+1]-arcs[index])
            return [a+t*(b-a) for a, b in zip(positions[index], positions[index+1])]
        first, last = point_at(lower), point_at(upper)
        old = _clipped_faces(mesh["originalPoints"], mesh["originalFaces"], first, last, lower, upper)
        new = _clipped_faces(mesh["points"], mesh["faces"], first, last, lower, upper)
        cuts = sorted({lower, upper, *(x for candidate in old+new for x in candidate[:2])})
        for lo, hi in zip(cuts, cuts[1:]):
            midpoint = (lo+hi)/2
            domains = [[c for c in candidates if c[0] <= midpoint <= c[1]] for candidates in (old, new)]
            if not all(domains):
                raise ValueError("Stored material curve has an exact coverage gap")
            faces_evidence, endpoints = [], []
            for numerical, candidates in enumerate(domains):
                faces = mesh["faces"] if numerical else mesh["originalFaces"]
                offset = mesh["triangleOffset"] if numerical else mesh["originalTriangleOffset"]
                operators = [[_operator(faces[c[2]], _at(c, t)) for t in (lo, hi)] for c in candidates]
                if any(value != operators[0] for value in operators[1:]):
                    raise ValueError("Coincident material candidates have different point operators")
                endpoints.append(operators[0])
                faces_evidence.append([{"localTriangleIndex": c[2], "canonicalTriangleIndex": offset+c[2],
                    "localVertices": faces[c[2]], "originalTriangleIndex": mesh["parents"][c[2]] if numerical else c[2]} for c in candidates])
            if any(_pullback(new_operator, mesh["lineage"]) != original_operator
                   for new_operator, original_operator in zip(endpoints[1], endpoints[0])):
                raise ValueError("Numerical material curve differs from its exact original operator pullback")
            cells.append({"fractionInterval": [_rat(lo), _rat(hi)], "sourcePathSegmentIndex": index,
                "originalCandidates": faces_evidence[0], "numericalCandidates": faces_evidence[1],
                "originalStartWeights": _sparse(endpoints[0][0]), "originalEndWeights": _sparse(endpoints[0][1]),
                "numericalStartWeights": _sparse(endpoints[1][0]), "numericalEndWeights": _sparse(endpoints[1][1]),
                "pullbackVerified": True})
    cells.sort(key=lambda cell: _fraction(cell["fractionInterval"][0]))
    if not cells or _fraction(cells[0]["fractionInterval"][0]) != 0 or _fraction(cells[-1]["fractionInterval"][1]) != 1:
        raise ValueError("Material curve must cover the entire registration")
    for a, b in zip(cells, cells[1:]):
        if _fraction(a["fractionInterval"][1]) != _fraction(b["fractionInterval"][0]):
            raise ValueError("Material cells must form an exact gap-free nonoverlapping partition")
        for domain in ("original", "numerical"):
            if _read_sparse(a[domain+"EndWeights"]) != _read_sparse(b[domain+"StartWeights"]):
                raise ValueError("Material point operators must be continuous across every cell boundary")
    return {"registrationMember": member, "templateId": mesh["templateId"], "pathSha256": _sha(path),
            "sourcePathSamples": sample_evidence, "cells": cells}


def _fraction(value):
    return F(int(value["numerator"]), int(value["denominator"]))


def _sample(member, fraction, domain):
    candidates = []
    for cell in member["cells"]:
        a, b = [_fraction(v) for v in cell["fractionInterval"]]
        if a <= fraction <= b:
            u = (fraction-a)/(b-a)
            start, end = _read_sparse(cell[domain+"StartWeights"]), _read_sparse(cell[domain+"EndWeights"])
            weights = {i: (1-u)*start.get(i, F())+u*end.get(i, F()) for i in start.keys() | end.keys()}
            candidates.append({i: w for i, w in weights.items() if w})
    if not candidates or any(c != candidates[0] for c in candidates):
        raise ValueError("Unique continuous sampled material point operator required")
    return candidates[0]


def _comparisons(source, selectors, row_ids):
    old_rows = source["baseUnit"]["embeddedConstraints"]["constraints"]
    rows = source["embeddedConstraints"]["constraints"]
    result = [None]*40
    for selector in selectors:
        for index in selector["rowIndices"]:
            original, numerical = old_rows[index], rows[index]
            fraction = F(numerical["fraction"])
            residuals, anchors = [[], []], []
            owners = [m["registrationMember"]["instanceId"] for m in selector["members"]]
            if len(set(owners)) != 2:
                raise ValueError("Two distinct registered spatial anchor owners required")
            for domain, row in (("original", original), ("numerical", numerical)):
                if type(row["terms"]) is not list or not 2 <= len(row["terms"]) <= 6:
                    raise ValueError("Complete bounded preserved spatial row terms required")
                seen, present = set(), set()
                for term in row["terms"]:
                    if type(term) is not dict or set(term) != {"instanceId", "vertex", "coefficient"}:
                        raise ValueError("Exact preserved spatial row term fields required")
                    name = term["instanceId"]
                    if type(name) is not str or name not in owners:
                        raise ValueError("Preserved spatial term owner is outside its registered pair")
                    if domain == "original":
                        template_id = next(i["templateId"] for i in source["instances"] if i["id"] == name)
                        count = len(source["baseUnit"]["sourceTemplates"][template_id]["restPositions"])
                    else:
                        count = len(source["numericalMeshes"][name]["verticesMeters"])
                    vertex = _index(term["vertex"], count)
                    coefficient = _number(term["coefficient"], -1, 1)
                    if ((name, vertex) in seen or not coefficient
                            or (coefficient > 0) != (name == owners[0])):
                        raise ValueError("Unique nonzero correctly signed spatial anchor terms required")
                    seen.add((name, vertex))
                    present.add(name)
                if present != set(owners):
                    raise ValueError("Both preserved spatial anchors must be present")
            for sign, member in zip((1, -1), selector["members"]):
                name = member["registrationMember"]["instanceId"]
                sums, errors = [], []
                for slot, (domain, row) in enumerate(zip(("original", "numerical"), (original, numerical))):
                    continuum = {i: sign*w for i, w in _sample(member, fraction, domain).items()}
                    stored = {}
                    for term in row["terms"]:
                        if term["instanceId"] == name:
                            vertex = term["vertex"]
                            if vertex in stored:
                                raise ValueError("Duplicate preserved spatial row term")
                            stored[vertex] = _number(term["coefficient"], -1, 1)
                    differences = {i: continuum.get(i,F())-stored.get(i,F()) for i in continuum.keys() | stored.keys()}
                    sums.append(sum(stored.values(),F()))
                    errors.append(sum((abs(v) for v in differences.values()),F()))
                    residuals[slot].extend({"instanceId": name, "vertex": i, "coefficient": _rat(v)} for i, v in differences.items() if v)
                anchors.append({"instanceId": name, "sign": sign,
                    "originalStoredCoefficientSum": _rat(sums[0]), "numericalStoredCoefficientSum": _rat(sums[1]),
                    "originalResidualL1": _rat(errors[0]), "numericalResidualL1": _rat(errors[1])})
            for terms in residuals:
                terms.sort(key=lambda t: (t["instanceId"], t["vertex"]))
            result[index] = {"rowIndex": index, "rowId": row_ids[index], "originalRowSha256": _sha(original),
                "numericalRowSha256": _sha(numerical), "originalResidualTerms": residuals[0],
                "numericalResidualTerms": residuals[1], "perAnchor": anchors}
    if any(row is None for row in result):
        raise ValueError("All forty preserved row discrepancy reports required")
    return result


def verify_binding_spatial_seams(source, reference_source, descriptor):
    """Independently reconstruct the complete static mapping, without state evaluation."""
    inputs = source, reference_source, descriptor
    before = [_encoded(value) for value in inputs]
    if max(len(before[0]), len(before[1])) > 8*1024**2:
        raise ValueError("Source/reference exceed original eight-MiB budget")
    try:
        if type(source) is not dict or {"placedMeters", "sewingActuation", "gripperActuation", "foldActuation",
                "assemblySchedule", "bindingRefinedDiagnostic", "sewingFrames"}.intersection(source):
            raise ValueError("Bare refined source required for spatial mapping")
        remap_audit = verify_binding_remap(source)
        base = source["baseUnit"]
        old_rows, rows = base["embeddedConstraints"]["constraints"], source["embeddedConstraints"]["constraints"]
        if len(old_rows) != 40 or len(rows) != 40:
            raise ValueError("Exactly forty preserved sewing rows required")
        row_ids, groups = [], {}
        for index, (old, new) in enumerate(zip(old_rows, rows)):
            a, b = _row_identity(old), _row_identity(new)
            _same([old[k] for k in ("registrationId", "memberIndex", "fraction")],
                  [new[k] for k in ("registrationId", "memberIndex", "fraction")], "original/numerical row selector")
            if a != b or a[3] in row_ids:
                raise ValueError("Ordered unique unchanged spatial row identities required")
            row_ids.append(a[3])
            groups.setdefault(a[:2], []).append((index, a[2]))
        if (len(groups) != 8 or any([f for _, f in group] != [F(i,4) for i in range(5)] for group in groups.values())
                or [i for i,_ in groups.get(("bind_opening_left_left",1),[])] != list(range(5))):
            raise ValueError("Complete eight-selector five-fraction partition required")
        recipe = _reference_recipe(reference_source, base, row_ids)
        phases = _phase_membership(base)
        meshes, material_evidence = _meshes(source)
        registrations = base["embeddedConstraints"]["registrations"]
        if type(registrations) is not list or not 1 <= len(registrations) <= 512:
            raise ValueError("Bounded source registrations required")
        selectors, cache = [], {}
        for (registration_id, member_index), group in groups.items():
            matches = [r for r in registrations if r["id"] == registration_id]
            if len(matches) != 1:
                raise ValueError("Unique original registration required")
            members = matches[0]["members"]
            _index(member_index, len(members))
            indices = [i for i,_ in group]
            phase_ids = {phases[i] for i in indices}
            if len(phase_ids) != 1:
                raise ValueError("All five samples of a selector must share a source phase")
            mapped = []
            for member in (members[0], members[member_index]):
                key = _sha(member)
                if key not in cache:
                    cache[key] = _member_map(source, member, meshes[member["instanceId"]])
                mapped.append(cache[key])
            selectors.append({"registrationId": registration_id, "memberIndex": member_index,
                "rowIndices": indices, "rowIds": [row_ids[i] for i in indices], "phaseId": next(iter(phase_ids)),
                "held": indices == list(range(5)), "members": mapped,
                "targetKnots": [{"fraction": _rat(f), "targetMeters": recipe["initialTargetsMeters"][i]} for i,f in group]})
        comparisons = _comparisons(source, selectors, row_ids)
        expected = {"profile": PROFILE, "curvePolicy": CURVE_POLICY, "materialPolicy": MATERIAL_POLICY, "targetPolicy": TARGET_POLICY,
            "sourceSha256": _sha(source), "baseUnitSha256": _sha(base), "refinementDescriptorSha256": _sha(source["bindingRefinement"]),
            "seamRemapDescriptorSha256": _sha(source["bindingSeamRemap"]), "referenceSourceSha256": _sha(reference_source),
            "referenceSewingActuationSha256": _sha(recipe), "originalBundleSha256": _sha(base["embeddedConstraints"]),
            "numericalBundleSha256": _sha(source["embeddedConstraints"]), "sourcePhasePlanSha256": _sha(base["phasePlan"]),
            "sourcePhaseConstraintRowsSha256": _sha(base["phaseConstraintRows"]), "materialMeshes": material_evidence,
            "selectors": selectors, "rowComparisons": comparisons, "initialTargetsMeters": recipe["initialTargetsMeters"],
            "finalTargetsMeters": recipe["finalTargetsMeters"], "complianceMPerN": 1e-8,
            "sewingControlSchedule": recipe["schedule"], "heldRowIndices": list(range(5)), "pendingRowIndices": list(range(5,40)),
            "materialCurveCoverageVerified": True,
            **{k:False for k in ("accepted", "solverReady", "executable", "controlsInstalled", "continuousSpatialStitchingVerified",
                                "constructionPhaseCompleted", "historicalExtrasInherited")}, "limitations": LIMITATIONS}
        _same(descriptor, expected, "complete descriptor reconstruction")
        if before != [_encoded(value) for value in inputs]:
            raise ValueError("Spatial mapping inputs changed during verification")
        return {"profile": "independent-binding-spatial-seam-verification-v1", "verified": True, "accepted": False,
            "descriptorSha256": _sha(descriptor), "sourceSha256": _sha(source), "referenceSourceSha256": _sha(reference_source),
            "selectorCount": 8, "rowCount": 40, "heldRowCount": 5, "pendingRowCount": 35,
            "memberCellCounts": [[len(member["cells"]) for member in selector["members"]] for selector in selectors],
            "materialMeshesSha256": _sha(material_evidence), "selectorsSha256": _sha(selectors),
            "rowComparisonsSha256": _sha(comparisons), "remapAudit": remap_audit,
            "requiredHelperFiles": list(REQUIRED_HELPER_FILES),
            "scope": "Independent supplied-source exact raw-lineage material coordinates, positive oriented parent tilings, stored arc/position polyline clipping against every original/numerical face, complete rational partitions, shared-edge point-operator equality, exact pullbacks and all eight selectors/forty preserved-row discrepancies. Original pattern and split-geometry rederivation remain a separate strict validator prerequisite. No producer, pattern engine, solver or force model is imported. This audit does not evaluate a posed state, install continuum constraints, choose normal directors/textile sides, certify temporal/contact paths or grant physical stitching/construction acceptance."}
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, ZeroDivisionError, StopIteration) as error:
        raise ValueError("Malformed independent material spatial-seam input") from error
