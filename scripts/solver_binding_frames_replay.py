"""Independent static support/frame correspondence; no force model or executor.

The supplied source's coefficient lineage and topology are audited here.
Original pattern and split-geometry rederivation remain a separate validator's
responsibility. No candidate normal is a textile-side or joint-model decision.
"""

from fractions import Fraction as F
import hashlib

from solver_binding_remap_replay import verify_binding_remap
from solver_binding_sewing_schedule_replay import (
    _encoded, _number, _phase_membership, _reference_recipe, _row_identity,
)


PROFILE = "source-binding-normal-frame-correspondence-v1"
REQUEST_PROFILE = "source-binding-normal-frame-request-v1"
FRAME_MODEL = "oriented-triangle-containing-negative-anchor-v1"
INSTANCE = "opening_binding_left_left:shell"
REQUIRED_HELPER_FILES = (
    "solver_binding_remap_replay.py", "solver_binding_sewing_schedule_replay.py",
    "solver_binding_control_schedule_replay.py", "solver_binding_fold_replay.py",
    "solver_binding_placement_replay.py",
)
CONTROL_FIELDS = {"placedMeters", "sewingActuation", "gripperActuation", "foldActuation",
                  "assemblySchedule", "bindingRefinedDiagnostic", "sewingFrames"}
LIMITATIONS = [
    "Static frame correspondence only; no controls installed, refined motion executed or captured source admitted.",
    "All forty source rows, targets and compliance remain explicit; five held samples and thirty-five pending rows do not establish continuous spatial stitching.",
    "Every nonzero negative coefficient determines support, including tiny allowance terms; original and numerical anchors remain distinct.",
    "Support-containing faces define candidate directors, not equivalent joints; candidates sharing an anchor can develop different normals under deformation.",
    "Offset signs, textile right/wrong sides and topological body/allowance regions are separate declarations; geometric winding supplies no textile-side authority.",
    "A normal-offset joint adds frame reactions beyond the preserved scalar-distance reference; complete declarations do not validate that mechanical model.",
    "Construction, swept frame validity, contact paths, settling, material calibration, refinement convergence and full-shirt acceptance remain unresolved.",
]


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _same(actual, expected, label):
    if _encoded(actual) != _encoded(expected):
        raise ValueError("Independent frame correspondence mismatch: " + label)


def _index(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError("Bounded raw integer frame index required")
    return value


def _rat(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _lineage(source):
    """Compose raw binary coefficients, checking each immediate parent too."""
    refinement = source["bindingRefinement"]
    original = refinement["originalLocalMesh"]
    basis = [{i: F(1)} for i in range(len(original["verticesMeters"]))]
    faces = original["triangles"]
    parents = list(range(len(faces)))
    for stage in refinement["stages"]:
        output = stage["output"]
        next_basis = []
        raw_supports = []
        for support in output["sourceWeights"]:
            terms = {}
            raw = {}
            for key, value in support.items():
                if type(key) is not str or not key.isascii() or not key.isdigit() or str(int(key)) != key:
                    raise ValueError("Canonical lineage vertex key required")
                vertex = _index(int(key), len(basis))
                weight = _number(value, 0, 1)
                if weight:
                    raw[vertex] = weight
                    for original_vertex, coefficient in basis[vertex].items():
                        terms[original_vertex] = terms.get(original_vertex, F()) + weight * coefficient
            if not terms or sum(terms.values(), F()) != 1 or any(value < 0 for value in terms.values()):
                raise ValueError("Exactly normalized nonnegative frame vertex lineage required")
            next_basis.append(terms)
            raw_supports.append(raw)
        next_parents = []
        for face, parent in zip(output["triangles"], output["parentTriangles"]):
            parent = _index(parent, len(faces))
            if any(index not in faces[parent] for vertex in face for index in raw_supports[vertex]):
                raise ValueError("Frame child lineage escapes its immediate parent")
            next_parents.append(parents[parent])
        basis, faces, parents = next_basis, output["triangles"], next_parents
    return basis, parents


def _regions(faces, chain):
    """Flood the dual graph with the complete directed rail removed."""
    if type(chain) is not list or len(chain) < 2 or len(set(chain)) != len(chain):
        raise ValueError("Simple complete source-directed crease chain required")
    edges = {}
    for triangle, face in enumerate(faces):
        for a, b in zip(face, face[1:] + face[:1]):
            incidents = edges.setdefault(tuple(sorted((a, b))), [])
            if len(incidents) >= 2 or incidents and incidents[0][1:] != (b, a):
                raise ValueError("Consistently oriented manifold frame mesh required")
            incidents.append((triangle, a, b))
    removed = {tuple(sorted(pair)) for pair in zip(chain, chain[1:])}
    boundary = {vertex for edge, incidents in edges.items() if len(incidents) == 1 for vertex in edge}
    if chain[0] not in boundary or chain[-1] not in boundary:
        raise ValueError("Frame crease must reach both full cut boundaries")
    body, allowance = set(), set()
    for a, b in zip(chain, chain[1:]):
        incidents = edges.get(tuple(sorted((a, b))), [])
        if len(incidents) != 2:
            raise ValueError("Every directed crease segment must have two incident faces")
        body.add(next(item[0] for item in incidents if item[1:] == (a, b)))
        allowance.add(next(item[0] for item in incidents if item[1:] == (b, a)))
    neighbors = [set() for _ in faces]
    for edge, incidents in edges.items():
        if edge not in removed and len(incidents) == 2:
            a, b = (item[0] for item in incidents)
            neighbors[a].add(b)
            neighbors[b].add(a)
    unseen, components = set(range(len(faces))), []
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            triangle = pending.pop()
            if triangle not in component:
                component.add(triangle)
                pending.extend(neighbors[triangle] - component)
        unseen -= component
        components.append(component)
    if len(components) != 2:
        raise ValueError("Full directed rail must separate two complete face regions")
    left = next((part for part in components if body <= part), None)
    right = next((part for part in components if allowance <= part), None)
    if left is None or right is None or left == right:
        raise ValueError("Source winding must consistently distinguish body and allowance")
    left_vertices = {v for i in left for v in faces[i]}
    right_vertices = {v for i in right for v in faces[i]}
    if left_vertices & right_vertices != set(chain):
        raise ValueError("Frame regions may share only their entire crease chain")
    return {i: "body" if i in left else "allowance" for i in range(len(faces))}


def _source_rails(source):
    refinement = source["bindingRefinement"]
    creases, stages = refinement["finalCreases"], refinement["stages"]
    if (type(creases) is not list or len(creases) != 2
            or [c["sourcePathName"] for c in creases] != ["right", "left"]):
        raise ValueError("Both ordered source-directed frame rails required")
    mesh = source["numericalMeshes"][INSTANCE]
    points, faces = mesh["verticesMeters"], mesh["triangles"]
    panels = [p for p in source["baseUnit"]["sourcePattern"]["panels"] if p["id"] == "opening_binding_left_left"]
    if len(panels) != 1:
        raise ValueError("Unique original binding panel required")
    panel = panels[0]
    result = {}
    for name, first, last, x, direction, crease, stage in zip(
            ("right", "left"), (1, 3), (2, 4), (.02, 0.), (1, -1), creases, stages):
        paths = [p for p in panel["draft"]["edges"] if p["name"] == name]
        if len(paths) != 1:
            raise ValueError("Unique named original rail required")
        path = paths[0]
        _same([path["start"], path["end"]], [first, last], "directed source path endpoints")
        line = [[float(_number(value, -100000, 100000))*.001 for value in panel["points"][i]] for i in (first, last)]
        if any(len(point) != 2 or point[0] != x for point in line) or direction*(F(line[1][1])-F(line[0][1])) <= 0:
            raise ValueError("Declared right/left source-line directions required")
        _same(stage["sourcePathSha256"], _sha(path), "named source path digest")
        _same([stage["sourceStart"], stage["sourceEnd"]], [first, last], "split source endpoint indices")
        _same(stage["sourceLineEndpointsMeters"], line, "split source line")
        _same(crease["sourceLineEndpointsMeters"], line, "final source line")
        chain = crease["vertices"]
        if type(chain) is not list or not 2 <= len(chain) <= len(points):
            raise ValueError("Bounded source crease chain required")
        for vertex in chain:
            _index(vertex, len(points))
        _same(chain, stage["output"]["creaseVertices"], "retained stage chain")
        edges = sorted(sorted((a, b)) for a, b in zip(chain, chain[1:]))
        _same(sorted(crease["edges"]), edges, "complete crease edge inventory")
        _same(sorted(stage["output"]["creaseEdges"]), edges, "split crease edge inventory")
        if any(direction*(F(points[b][1])-F(points[a][1])) <= 0 for a, b in zip(chain, chain[1:])):
            raise ValueError("Rail chain must follow its original named source direction")
        _same(crease["cutBoundaryEndpointsMeters"], [points[chain[0]][:2], points[chain[-1]][:2]], "full cut-boundary endpoints")
        result[name] = _regions(faces, chain)
    return result


def _meshes(source):
    base = source["baseUnit"]
    result, originals = [], {}
    old_vertex_offset = old_face_offset = 0
    for instance in source["instances"]:
        name, template = instance["id"], instance["templateId"]
        original_faces = base["sourceTemplates"][template]["triangles"]
        count = len(base["sourceTemplates"][template]["restPositions"])
        _same(base["instanceOffsets"][name], old_vertex_offset, "original instance offsets")
        original = {"verticesMeters": base["restMeters"][old_vertex_offset:old_vertex_offset + count],
                    "triangles": original_faces}
        numerical = source["numericalMeshes"][name]
        originals[name] = original
        result.append({"instanceId": name, "templateId": template,
            "originalMeshSha256": _sha(original), "numericalMeshSha256": _sha(numerical),
            "originalVertexOffset": old_vertex_offset, "vertexOffset": source["instanceOffsets"][name],
            "originalTriangleOffset": old_face_offset, "triangleOffset": source["instanceTriangleOffsets"][name],
            "originalVertexCount": count, "vertexCount": len(numerical["verticesMeters"]),
            "originalTriangleCount": len(original_faces), "triangleCount": len(numerical["triangles"])})
        old_vertex_offset += count
        old_face_offset += len(original_faces)
    return result, originals


def _terms(row, meshes, offsets):
    result, negative, seen = [], {}, set()
    signs = {1: set(), -1: set()}
    if type(row["terms"]) is not list or not 2 <= len(row["terms"]) <= 6:
        raise ValueError("Bounded two-anchor frame row required")
    for term in row["terms"]:
        if type(term) is not dict or set(term) != {"instanceId", "vertex", "coefficient"}:
            raise ValueError("Exact source frame term fields required")
        instance = term["instanceId"]
        if type(instance) is not str or instance not in meshes:
            raise ValueError("Known source frame instance required")
        vertex = _index(term["vertex"], len(meshes[instance]["verticesMeters"]))
        coefficient = _number(term["coefficient"], -1, 1)
        if not coefficient or (instance, vertex) in seen:
            raise ValueError("Unique exactly nonzero source frame terms required")
        seen.add((instance, vertex))
        signs[1 if coefficient > 0 else -1].add(instance)
        if coefficient < 0:
            negative.setdefault(instance, set()).add(vertex)
        result.append({"instanceId": instance, "localVertex": vertex,
                       "canonicalVertex": offsets[instance] + vertex, "coefficient": term["coefficient"]})
    if len(signs[1]) != 1 or len(signs[-1]) != 1 or signs[1] == signs[-1]:
        raise ValueError("Distinct positive and negative anchor instances required")
    instance = next(iter(negative))
    return result, instance, negative[instance]


def _candidates(source, originals, old_instance, old_support, instance, support, basis, parents, regions):
    if instance != old_instance:
        raise ValueError("Original and numerical negative anchor owner must match")
    mesh = source["numericalMeshes"][instance]
    original_faces = originals[instance]["triangles"]
    result = []
    for index, face in enumerate(mesh["triangles"]):
        if not support <= set(face):
            continue
        parent = parents[index] if instance == INSTANCE else index
        original_face = original_faces[parent]
        if not old_support <= set(original_face):
            continue
        lineage = []
        for vertex in face:
            coefficients = basis[vertex] if instance == INSTANCE else {vertex: F(1)}
            if (sum(coefficients.values(), F()) != 1 or any(weight < 0 for weight in coefficients.values())
                    or any(weight and i not in original_face for i, weight in coefficients.items())):
                raise ValueError("Frame lineage must remain normalized inside its original parent")
            lineage.append({"localVertex": vertex, "originalWeights": [
                {"vertex": i, "coefficient": _rat(weight)} for i, weight in sorted(coefficients.items()) if weight]})
        a, b, c = [[_number(value, -100, 100) for value in mesh["verticesMeters"][vertex]] for vertex in face]
        u, v = [y-x for x,y in zip(a,b)], [y-x for x,y in zip(a,c)]
        cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        squared = sum((x*x for x in cross), F())
        if squared <= 0:
            raise ValueError("Nondegenerate exact binary-rest frame required")
        result.append({"candidateId": "face:" + _sha([instance, index, face]), "instanceId": instance,
            "localTriangleIndex": index, "canonicalTriangleIndex": source["instanceTriangleOffsets"][instance]+index,
            "localVertices": face, "canonicalVertices": [source["instanceOffsets"][instance]+v for v in face],
            "originalTriangleIndex": parent, "originalVertices": original_face, "vertexLineage": lineage,
            "restCrossProductMetersSquared": [_rat(value) for value in cross],
            "restCrossProductSquaredNormMetersFourth": _rat(squared),
            "creaseRegions": {name: mapping[index] for name, mapping in regions.items()} if instance == INSTANCE else {}})
    if not result:
        raise ValueError("Every source row needs a complete support-containing frame candidate")
    return result


def _request(value, source, reference, bindings):
    if (type(value) is not dict or set(value) != {"profile", "sourceSha256", "referenceSourceSha256", "rows", "instanceTextileSides"}
            or value["profile"] != REQUEST_PROFILE or value["sourceSha256"] != _sha(source)
            or value["referenceSourceSha256"] != _sha(reference)):
        raise ValueError("Complete source/reference-bound frame request required")
    rows, sides = value["rows"], value["instanceTextileSides"]
    if type(rows) is not list or len(rows) != 40 or type(sides) is not list or len(sides) != 5:
        raise ValueError("All forty frame choices and five textile declarations required")
    for choice, binding in zip(rows, bindings):
        if type(choice) is not dict or set(choice) != {"rowId", "originalRowSha256", "numericalRowSha256", "candidateId", "offsetSide", "attachmentRegion"}:
            raise ValueError("Exact row frame choice fields required")
        for key in ("rowId", "originalRowSha256", "numericalRowSha256"):
            _same(choice[key], binding[key], "request row identity")
        side, region, candidate_id = choice["offsetSide"], choice["attachmentRegion"], choice["candidateId"]
        if side is not None and (type(side) is not int or side not in (-1, 1)):
            raise ValueError("Explicit raw integer offset sign or unresolved null required")
        if region is not None and (type(region) is not str or region not in ("body", "allowance")):
            raise ValueError("Explicit topological attachment region or null required")
        candidates = binding["candidates"]
        if candidate_id is not None and (type(candidate_id) is not str or candidate_id not in {c["candidateId"] for c in candidates}):
            raise ValueError("Selected frame must be a complete source-support candidate")
        if region is not None and (candidates[0]["instanceId"] != INSTANCE
                or not any(c["creaseRegions"]["right"] == region for c in candidates)):
            raise ValueError("Requested attachment region has no source-support candidate")
        selected = next((c for c in candidates if c["candidateId"] == candidate_id), None)
        if selected is not None and region is not None and selected["creaseRegions"]["right"] != region:
            raise ValueError("Selected frame conflicts with explicit attachment region")
    for instance, side in zip(source["instances"], sides):
        if (type(side) is not dict or set(side) != {"instanceId", "positiveNormalTextileSide"}
                or side["instanceId"] != instance["id"] or side["positiveNormalTextileSide"] is not None
                and (type(side["positiveNormalTextileSide"]) is not str or side["positiveNormalTextileSide"] not in ("right", "wrong"))):
            raise ValueError("Five ordered explicit textile-side choices or nulls required")
    return value


def verify_binding_frames(source, reference_source, descriptor):
    """Verify a static descriptor, never installing frames or executing joints."""
    inputs = source, reference_source, descriptor
    before = [_encoded(value) for value in inputs]
    if len(before[0]) > 8*1024**2 or len(before[1]) > 8*1024**2:
        raise ValueError("Source and reference exceed original eight-MiB budgets")
    try:
        if type(source) is not dict or CONTROL_FIELDS.intersection(source):
            raise ValueError("Bare refined source required for static frame correspondence")
        remap_audit = verify_binding_remap(source)
        base = source["baseUnit"]
        original_rows, rows = base["embeddedConstraints"]["constraints"], source["embeddedConstraints"]["constraints"]
        if len(original_rows) != 40 or len(rows) != 40:
            raise ValueError("All forty frame rows required")
        identities, groups, row_ids = [], {}, []
        for index, (old, new) in enumerate(zip(original_rows, rows)):
            old_id, new_id = _row_identity(old), _row_identity(new)
            _same([old[key] for key in ("registrationId", "memberIndex", "fraction")],
                  [new[key] for key in ("registrationId", "memberIndex", "fraction")], "source/numerical selector")
            if old_id != new_id or old_id[3] in row_ids:
                raise ValueError("Unique unchanged ordered source row identities required")
            identities.append(old_id)
            row_ids.append(old_id[3])
            groups.setdefault(old_id[:2], []).append((index, old_id[2]))
        if (len(groups) != 8 or any([fraction for _, fraction in group] != [F(i, 4) for i in range(5)] for group in groups.values())
                or [i for i, _ in groups.get(("bind_opening_left_left", 1), [])] != list(range(5))):
            raise ValueError("Eight complete five-sample source selectors required")
        recipe = _reference_recipe(reference_source, base, row_ids)
        phases = _phase_membership(base)
        meshes, originals = _meshes(source)
        basis, parents = _lineage(source)
        regions = _source_rails(source)
        bindings = []
        for index, (old, new, identity, target) in enumerate(zip(original_rows, rows, identities, recipe["initialTargetsMeters"])):
            old_terms, old_instance, old_support = _terms(old, originals, base["instanceOffsets"])
            new_terms, instance, support = _terms(new, source["numericalMeshes"], source["instanceOffsets"])
            candidates = _candidates(source, originals, old_instance, old_support, instance, support, basis, parents, regions)
            registration, member, fraction, row_id = identity
            bindings.append({"rowIndex": index, "rowId": row_id, "registrationId": registration, "memberIndex": member,
                "fractionNumerator": fraction.numerator, "fractionDenominator": fraction.denominator,
                "originalRowSha256": _sha(old), "numericalRowSha256": _sha(new), "phaseId": phases[index], "held": index < 5,
                "targetMeters": target, "complianceMPerN": new["complianceMPerN"], "originalTerms": old_terms,
                "numericalTerms": new_terms, "candidates": candidates})
        request = _request(descriptor["request"], source, reference_source, bindings)
        for binding, choice in zip(bindings, request["rows"]):
            chosen = choice["candidateId"]
            authority = "explicit-request" if chosen is not None else "unique-support" if len(binding["candidates"]) == 1 else "unresolved"
            if chosen is None and len(binding["candidates"]) == 1:
                chosen = binding["candidates"][0]["candidateId"]
            binding.update(selectedCandidateId=chosen, selectionAuthority=authority,
                           offsetSide=choice["offsetSide"], attachmentRegion=choice["attachmentRegion"])
        unresolved_frames = [b["rowIndex"] for b in bindings if b["selectedCandidateId"] is None]
        unresolved_offsets = [b["rowIndex"] for b in bindings if b["offsetSide"] is None]
        unresolved_textile = [s["instanceId"] for s in request["instanceTextileSides"] if s["positiveNormalTextileSide"] is None]
        expected = {"profile": PROFILE, "frameModel": FRAME_MODEL, "sourceSha256": _sha(source), "baseUnitSha256": _sha(base),
            "refinementDescriptorSha256": _sha(source["bindingRefinement"]), "seamRemapDescriptorSha256": _sha(source["bindingSeamRemap"]),
            "originalBundleSha256": _sha(base["embeddedConstraints"]), "numericalBundleSha256": _sha(source["embeddedConstraints"]),
            "referenceSourceSha256": _sha(reference_source), "referenceSewingActuationSha256": _sha(recipe),
            "sourcePhasePlanSha256": _sha(base["phasePlan"]), "sourcePhaseConstraintRowsSha256": _sha(base["phaseConstraintRows"]),
            "requestSha256": _sha(request), "request": request, "meshes": meshes, "rowBindings": bindings,
            "instanceTextileSides": request["instanceTextileSides"], "referenceSewingMode": "distance",
            "sewingControlSchedule": recipe["schedule"], "initialTargetsMeters": recipe["initialTargetsMeters"],
            "finalTargetsMeters": recipe["finalTargetsMeters"], "complianceMPerN": 1e-8,
            "heldRowIndices": list(range(5)), "pendingRowIndices": list(range(5, 40)),
            "frameChoicesComplete": not unresolved_frames, "offsetSidesComplete": not unresolved_offsets,
            "textileSidesComplete": not unresolved_textile, "choicesComplete": not (unresolved_frames or unresolved_offsets or unresolved_textile),
            "unresolvedFrameRowIndices": unresolved_frames, "unresolvedOffsetSideRowIndices": unresolved_offsets,
            "unresolvedTextileInstanceIds": unresolved_textile,
            **{key: False for key in ("accepted", "solverReady", "executable", "controlsInstalled", "normalOffsetMechanicsExecuted",
                "constructionPhaseCompleted", "continuousSpatialSeamsVerified", "historicalExtrasInherited")}, "limitations": LIMITATIONS}
        _same(descriptor, expected, "complete descriptor reconstruction")
        if before != [_encoded(value) for value in inputs]:
            raise ValueError("Frame inputs changed during independent verification")
        return {"profile": "independent-binding-frame-correspondence-verification-v1", "verified": True, "accepted": False,
            "descriptorSha256": _sha(descriptor), "sourceSha256": _sha(source), "referenceSourceSha256": _sha(reference_source),
            "rowCount": 40, "heldRowCount": 5, "pendingRowCount": 35, "selectorCount": 8, "instanceCount": 5,
            "candidateCount": sum(len(b["candidates"]) for b in bindings),
            "ambiguousFrameRowIndices": [b["rowIndex"] for b in bindings if len(b["candidates"]) > 1],
            "choicesComplete": expected["choicesComplete"], "frameBindingsSha256": _sha(bindings), "remapAudit": remap_audit,
            "requiredHelperFiles": list(REQUIRED_HELPER_FILES),
            "scope": "Independent supplied-source exact coefficient lineage, complete support-containing oriented face candidates, original-parent containment, binary-rest cross products, directed dual-graph regions, all forty reference targets/compliance/held-pending rows and explicit choices. Transitive helpers are stdlib-only independent auditors; no pattern engine, producer, force model or executor is imported. Original pattern and split-geometry rederivation require the separate strict source validator. This does not install frames, infer textile sides, validate normal-offset mechanics, prove swept geometry/contact, execute construction or grant garment acceptance."}
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Malformed independent static frame correspondence input") from error
