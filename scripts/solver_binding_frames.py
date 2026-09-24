"""Static, source-bound candidate directors for refined sewing anchors.

This descriptor preserves the original scalar-distance recipe. It neither
installs normal-offset controls nor admits a refined source to any runner.
"""

import copy
from fractions import Fraction
import hashlib

from solver_binding_fold import _rail
from solver_binding_remap import _basis
from solver_binding_sewing_schedule import _phases, _reference
from solver_binding_source import CONTROL_FIELDS, INSTANCE, _bounded, _encoded, validate_binding_source
from solver_sewing_input import derive_sewing_row_ids


PROFILE = "source-binding-normal-frame-correspondence-v1"
REQUEST_PROFILE = "source-binding-normal-frame-request-v1"
FRAME_MODEL = "oriented-triangle-containing-negative-anchor-v1"
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


def _rat(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _terms(row, offsets):
    return [{"instanceId": term["instanceId"], "localVertex": term["vertex"],
             "canonicalVertex": offsets[term["instanceId"]] + term["vertex"],
             "coefficient": term["coefficient"]}
            for term in row["terms"] if term["coefficient"] != 0]


def _negative(terms):
    negative = [term for term in terms if term["coefficient"] < 0]
    names = {term["instanceId"] for term in negative}
    if len(names) != 1:
        raise ValueError("Exactly one instance must support the complete negative anchor")
    return next(iter(names)), {term["localVertex"] for term in negative}


def _meshes(source):
    base, result, old_face_offset = source["baseUnit"], [], 0
    for instance in source["instances"]:
        name, template_id = instance["id"], instance["templateId"]
        template = base["sourceTemplates"][template_id]
        start, count = base["instanceOffsets"][name], len(template["restPositions"])
        original = {"verticesMeters": base["restMeters"][start:start + count], "triangles": template["triangles"]}
        numerical = source["numericalMeshes"][name]
        result.append({"instanceId": name, "templateId": template_id,
            "originalMeshSha256": _sha(original), "numericalMeshSha256": _sha(numerical),
            "originalVertexOffset": start, "vertexOffset": source["instanceOffsets"][name],
            "originalTriangleOffset": old_face_offset, "triangleOffset": source["instanceTriangleOffsets"][name],
            "originalVertexCount": count, "vertexCount": len(numerical["verticesMeters"]),
            "originalTriangleCount": len(original["triangles"]), "triangleCount": len(numerical["triangles"])})
        old_face_offset += len(original["triangles"])
    return result


def _candidates(source, original_terms, numerical_terms, basis, rails):
    name, support = _negative(numerical_terms)
    original_name, original_support = _negative(original_terms)
    if name != original_name:
        raise ValueError("Negative anchor instance identity must remain unchanged")
    positive = {term["canonicalVertex"] for term in numerical_terms if term["coefficient"] > 0}
    mesh = source["numericalMeshes"][name]
    template_id = next(instance["templateId"] for instance in source["instances"] if instance["id"] == name)
    original_faces = source["baseUnit"]["sourceTemplates"][template_id]["triangles"]
    offset, face_offset = source["instanceOffsets"][name], source["instanceTriangleOffsets"][name]
    result = []
    for index, face in enumerate(mesh["triangles"]):
        if not support.issubset(face) or positive.intersection(offset + vertex for vertex in face):
            continue
        parent = source["bindingRefinement"]["ultimateOriginalTriangleIndices"][index] if name == INSTANCE else index
        original_face = original_faces[parent]
        if not original_support.issubset(original_face):
            continue
        lineage = []
        for vertex in face:
            weights = ({i: value for i, value in enumerate(basis[vertex]) if value != 0}
                       if name == INSTANCE else {vertex: Fraction(1)})
            if (not set(weights).issubset(original_face) or any(value < 0 for value in weights.values())
                    or sum(weights.values(), Fraction()) != 1):
                raise ValueError("Candidate lineage must remain inside its complete original parent")
            lineage.append({"localVertex": vertex, "originalWeights": [
                {"vertex": original, "coefficient": _rat(value)} for original, value in sorted(weights.items())]})
        a, b, c = [[Fraction(value) for value in mesh["verticesMeters"][vertex]] for vertex in face]
        u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        cross = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        norm = sum((value * value for value in cross), Fraction())
        if norm <= 0:
            raise ValueError("Every candidate director requires an exactly nonzero rest cross product")
        regions = {}
        if name == INSTANCE:
            for rail_name, rail in rails.items():
                if index in rail["bodyFacesLocal"]:
                    regions[rail_name] = "body"
                elif index in rail["allowanceFacesLocal"]:
                    regions[rail_name] = "allowance"
                else:
                    raise ValueError("Complete candidate membership in both source rail partitions required")
        result.append({"candidateId": "face:" + _sha([name, index, face]), "instanceId": name,
            "localTriangleIndex": index, "canonicalTriangleIndex": face_offset + index,
            "localVertices": list(face), "canonicalVertices": [offset + vertex for vertex in face],
            "originalTriangleIndex": parent, "originalVertices": list(original_face), "vertexLineage": lineage,
            "restCrossProductMetersSquared": [_rat(value) for value in cross],
            "restCrossProductSquaredNormMetersFourth": _rat(norm), "creaseRegions": regions})
    if not result:
        raise ValueError("No source-parent-preserving triangle contains the complete negative anchor")
    return result


def _request(source, reference, rows, request):
    identity = {"profile": REQUEST_PROFILE, "sourceSha256": _sha(source), "referenceSourceSha256": _sha(reference)}
    row_keys = {"rowId", "originalRowSha256", "numericalRowSha256", "candidateId", "offsetSide", "attachmentRegion"}
    if request is None:
        request = {**identity, "rows": [{**{key: row[key] for key in
            ("rowId", "originalRowSha256", "numericalRowSha256")},
            "candidateId": None, "offsetSide": None, "attachmentRegion": None} for row in rows],
            "instanceTextileSides": [{"instanceId": instance["id"], "positiveNormalTextileSide": None}
                                     for instance in source["instances"]]}
    if (type(request) is not dict or set(request) != set(identity) | {"rows", "instanceTextileSides"}
            or any(_encoded(request[key]) != _encoded(value) for key, value in identity.items())
            or type(request["rows"]) is not list or len(request["rows"]) != len(rows)):
        raise ValueError("Complete source-bound frame choice request required")
    for row, choice in zip(rows, request["rows"]):
        if (type(choice) is not dict or set(choice) != row_keys
                or any(_encoded(choice[key]) != _encoded(row[key]) for key in
                       ("rowId", "originalRowSha256", "numericalRowSha256"))):
            raise ValueError("All ordered original and numerical row identities must bind frame choices")
        selected, side, region = choice["candidateId"], choice["offsetSide"], choice["attachmentRegion"]
        if ((selected is not None and type(selected) is not str)
                or (side is not None and (type(side) is not int or side not in (-1, 1)))
                or (region is not None and (type(region) is not str or region not in ("body", "allowance")))):
            raise ValueError("Explicit candidate identity, raw integer offset side and named topological region required")
        candidates = row["candidates"]
        if region is not None and not any(candidate["creaseRegions"].get("right") == region for candidate in candidates):
            raise ValueError("Requested attachment region cannot contain the complete negative anchor")
        if selected is not None:
            if selected not in {candidate["candidateId"] for candidate in candidates}:
                raise ValueError("Requested frame is not a complete source-bound support candidate")
            authority = "explicit-request"
        elif len(candidates) == 1:
            selected, authority = candidates[0]["candidateId"], "unique-support"
        else:
            authority = "unresolved"
        if selected is not None and region is not None:
            candidate = next(candidate for candidate in candidates if candidate["candidateId"] == selected)
            if candidate["creaseRegions"].get("right") != region:
                raise ValueError("Requested frame and attachment region disagree")
        row.update(selectedCandidateId=selected, selectionAuthority=authority, offsetSide=side, attachmentRegion=region)
    textiles = request["instanceTextileSides"]
    if type(textiles) is not list or len(textiles) != len(source["instances"]):
        raise ValueError("Every source instance requires an explicit textile-side declaration or null")
    for instance, textile in zip(source["instances"], textiles):
        if (type(textile) is not dict or set(textile) != {"instanceId", "positiveNormalTextileSide"}
                or textile["instanceId"] != instance["id"] or type(textile["instanceId"]) is not str
                or textile["positiveNormalTextileSide"] is not None and (
                    type(textile["positiveNormalTextileSide"]) is not str
                    or textile["positiveNormalTextileSide"] not in ("right", "wrong"))):
            raise ValueError("Ordered instance identity and independently declared textile side required")
    return copy.deepcopy(request)


def build_binding_frames(source, reference_source, request=None):
    """Enumerate every source-support candidate and preserve unresolved choices."""
    inputs = (source, reference_source, request)
    before = [_bounded(value) for value in inputs]
    try:
        if type(source) is not dict or CONTROL_FIELDS.intersection(source):
            raise ValueError("Bare refined source required for static frame correspondence")
        validate_binding_source(source)
        base = source["baseUnit"]
        recipe, row_ids = _reference(base, reference_source, 1)
        if list(derive_sewing_row_ids(source)) != row_ids:
            raise ValueError("All ordered original and numerical row identities must match")
        original_rows, numerical_rows = base["embeddedConstraints"]["constraints"], source["embeddedConstraints"]["constraints"]
        if len(original_rows) != 40 or len(numerical_rows) != 40:
            raise ValueError("All forty original and numerical rows required")
        basis = _basis(source["bindingRefinement"])
        rails = {name: _rail(source, name)[0] for name in ("right", "left")}
        phases, rows = _phases(base), []
        for index, (original, numerical) in enumerate(zip(original_rows, numerical_rows)):
            original_terms = _terms(original, base["instanceOffsets"])
            numerical_terms = _terms(numerical, source["instanceOffsets"])
            fraction = Fraction(numerical["fraction"])
            rows.append({"rowIndex": index, "rowId": row_ids[index], "registrationId": numerical["registrationId"],
                "memberIndex": numerical["memberIndex"], "fractionNumerator": fraction.numerator,
                "fractionDenominator": fraction.denominator, "originalRowSha256": _sha(original),
                "numericalRowSha256": _sha(numerical), "phaseId": phases[index], "held": index < 5,
                "targetMeters": recipe["initialTargetsMeters"][index], "complianceMPerN": numerical["complianceMPerN"],
                "originalTerms": original_terms, "numericalTerms": numerical_terms,
                "candidates": _candidates(source, original_terms, numerical_terms, basis, rails)})
        normalized = _request(source, reference_source, rows, request)
        missing_frames = [row["rowIndex"] for row in rows if row["selectedCandidateId"] is None]
        missing_offsets = [row["rowIndex"] for row in rows if row["offsetSide"] is None]
        missing_textiles = [item["instanceId"] for item in normalized["instanceTextileSides"]
                            if item["positiveNormalTextileSide"] is None]
        result = {"profile": PROFILE, "frameModel": FRAME_MODEL,
            "sourceSha256": _sha(source), "baseUnitSha256": _sha(base),
            "refinementDescriptorSha256": _sha(source["bindingRefinement"]),
            "seamRemapDescriptorSha256": _sha(source["bindingSeamRemap"]),
            "originalBundleSha256": _sha(base["embeddedConstraints"]), "numericalBundleSha256": _sha(source["embeddedConstraints"]),
            "referenceSourceSha256": _sha(reference_source), "referenceSewingActuationSha256": _sha(recipe),
            "sourcePhasePlanSha256": _sha(base["phasePlan"]), "sourcePhaseConstraintRowsSha256": _sha(base["phaseConstraintRows"]),
            "requestSha256": _sha(normalized), "request": normalized, "meshes": _meshes(source), "rowBindings": rows,
            "instanceTextileSides": copy.deepcopy(normalized["instanceTextileSides"]),
            "referenceSewingMode": "distance", "sewingControlSchedule": copy.deepcopy(recipe["schedule"]),
            "initialTargetsMeters": copy.deepcopy(recipe["initialTargetsMeters"]),
            "finalTargetsMeters": copy.deepcopy(recipe["finalTargetsMeters"]), "complianceMPerN": 1e-8,
            "heldRowIndices": list(range(5)), "pendingRowIndices": list(range(5, 40)),
            "frameChoicesComplete": not missing_frames, "offsetSidesComplete": not missing_offsets,
            "textileSidesComplete": not missing_textiles,
            "choicesComplete": not (missing_frames or missing_offsets or missing_textiles),
            "unresolvedFrameRowIndices": missing_frames, "unresolvedOffsetSideRowIndices": missing_offsets,
            "unresolvedTextileInstanceIds": missing_textiles,
            "accepted": False, "solverReady": False, "executable": False, "controlsInstalled": False,
            "normalOffsetMechanicsExecuted": False, "constructionPhaseCompleted": False,
            "continuousSpatialSeamsVerified": False, "historicalExtrasInherited": False, "limitations": list(LIMITATIONS)}
        if before != [_bounded(value) for value in inputs]:
            raise ValueError("Static frame inputs changed during derivation")
        _bounded(result)
        return copy.deepcopy(result)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Malformed or incomplete static frame correspondence input") from error


def validate_binding_frames(source, reference_source, descriptor):
    """Rebuild every candidate, declaration and scope flag, byte-sensitively."""
    claimed = _bounded(descriptor)
    if type(descriptor) is not dict or "request" not in descriptor:
        raise ValueError("Complete static frame descriptor required")
    expected = build_binding_frames(source, reference_source, descriptor["request"])
    if claimed != _encoded(expected):
        raise ValueError("Static frame correspondence differs from complete rederivation")
    return expected
