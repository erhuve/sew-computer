"""Source-directed binding fold controls; no placement or dynamics execution.

The declared reference region defines a relative angle, not a fixed opposite
region. Density describes an external line actuator, never textile bending.
"""

import copy
from fractions import Fraction
import hashlib
import math
import struct

from solver_binding_source import CONTROL_FIELDS, INSTANCE, _bounded, _encoded, validate_binding_source
from solver_binding_remap import _rational


PROFILE = "source-binding-fold-control-v1"
REQUEST_PROFILE = "source-binding-fold-request-v1"
ANGLE_CONVENTION = "source-directed-right-hand-reference-region-v1"
STIFFNESS_MEASURE = "stored-crease-edge-length-v1"
MAX_DENSITY = 1e6


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _cycles(face):
    return [face[index:] + face[:index] for index in range(3)]


def _edges(faces):
    edges = {}
    for index, face in enumerate(faces):
        for shift in range(3):
            first, second, opposite = face[shift:] + face[:shift]
            edges.setdefault(tuple(sorted((first, second))), []).append((index, first, second, opposite))
    if any(len(entries) not in (1, 2) for entries in edges.values()):
        raise ValueError("Manifold source edges required")
    return edges


def _request(request):
    if (type(request) is not dict or set(request) != {"profile", "accepted", "angleConvention", "stiffnessMeasure", "folds"}
            or request["profile"] != REQUEST_PROFILE or request["accepted"] is not False
            or request["angleConvention"] != ANGLE_CONVENTION or request["stiffnessMeasure"] != STIFFNESS_MEASURE):
        raise ValueError("Explicit unaccepted source-directed line-actuator request required")
    folds = request["folds"]
    if type(folds) is not list or not 1 <= len(folds) <= 2:
        raise ValueError("One or two explicitly declared binding rails required")
    names = set()
    for fold in folds:
        if (type(fold) is not dict or set(fold) != {"sourcePathName", "referenceRotationRegion", "targetRightHandAngleRadians", "stiffnessJoulesPerMeter"}
                or type(fold["sourcePathName"]) is not str or fold["sourcePathName"] not in ("right", "left")
                or fold["sourcePathName"] in names or fold["referenceRotationRegion"] not in ("body", "allowance")):
            raise ValueError("Unique named source rail and explicit reference rotation region required")
        angle, density = fold["targetRightHandAngleRadians"], fold["stiffnessJoulesPerMeter"]
        if (type(angle) not in (int, float) or not math.isfinite(angle) or abs(angle) >= math.pi - 1e-8
                or type(density) not in (int, float) or not math.isfinite(density) or not 0 < density <= MAX_DENSITY):
            raise ValueError("Finite principal-branch angle and positive bounded line-actuator stiffness required")
        names.add(fold["sourcePathName"])
    return folds


def _native(source, topology):
    faces = [source["triangles"][index:index + 3] for index in range(0, len(source["triangles"]), 3)]
    if (type(topology) is not dict or set(topology) != {"vertexCount", "triangles", "hinges"}
            or type(topology["vertexCount"]) is not int or topology["vertexCount"] != len(source["restMeters"])
            or _encoded(topology["triangles"]) != _encoded(faces)):
        raise ValueError("Complete native vertex and oriented source triangle identity required")
    edges = _edges(faces)
    rows = topology["hinges"]
    if type(rows) is not list or len(rows) != len(edges) or len(rows) > 150000:
        raise ValueError("Complete native edge topology required")
    mapping = {}
    for index, row in enumerate(rows):
        if (type(row) is not list or len(row) != 4 or any(type(v) is not int for v in row)
                or any(not 0 <= v < topology["vertexCount"] for v in row[2:])
                or any(not -1 <= v < topology["vertexCount"] for v in row[:2])):
            raise ValueError("Bounded raw native hinge indices required")
        first, second = row[2:]
        edge = tuple(sorted((first, second)))
        if edge not in edges or edge in mapping:
            raise ValueError("Unique complete source edge binding required")
        incident = edges[edge]
        expected = sorted([entry[3] for entry in incident] + ([-1] if len(incident) == 1 else []))
        if sorted(row[:2]) != expected:
            raise ValueError("Native opposite vertices differ from source incident faces")
        for opposite, a, b in ((row[0], first, second), (row[1], second, first)):
            if opposite == -1:
                continue
            face = next(faces[item[0]] for item in incident if item[3] == opposite)
            if [opposite, a, b] not in _cycles(face):
                raise ValueError("Native ordered hinge reverses an oriented source face")
        mapping[edge] = (index, row)
    if set(mapping) != set(edges):
        raise ValueError("Native source edge coverage incomplete")
    return mapping


def _length(first, second):
    squared = sum(((Fraction(b) - Fraction(a)) ** 2 for a, b in zip(first, second)), Fraction())
    if squared <= 0:
        raise ValueError("Positive source crease segment required")
    approximate = float(squared)
    if not math.isfinite(approximate) or approximate <= 0:
        raise ValueError("Source crease length outside bounded binary64 square range")
    rounded = math.sqrt(approximate)
    for _ in range(4):
        lower = (Fraction(math.nextafter(rounded, -math.inf)) + Fraction(rounded)) / 2
        upper = (Fraction(math.nextafter(rounded, math.inf)) + Fraction(rounded)) / 2
        even = struct.unpack(">Q", struct.pack(">d", rounded))[0] % 2 == 0
        if squared < lower * lower or squared == lower * lower and not even:
            rounded = math.nextafter(rounded, -math.inf)
        elif squared > upper * upper or squared == upper * upper and not even:
            rounded = math.nextafter(rounded, math.inf)
        else:
            if not math.isfinite(rounded) or rounded <= 0:
                raise ValueError("Positive finite rounded crease length required")
            return {"exactSquaredLengthMetersSquared": _rational(squared), "roundedLengthMeters": rounded,
                "roundedLengthSquaredMinusExactMetersSquared": _rational(Fraction(rounded) ** 2 - squared),
                "lowerMidpointSquaredMetersSquared": _rational(lower ** 2),
                "upperMidpointSquaredMetersSquared": _rational(upper ** 2)}
    raise ValueError("Crease length rounding could not be independently bounded")


def _rail(source, name):
    refinement = source["bindingRefinement"]
    crease = next(item for item in refinement["finalCreases"] if item["sourcePathName"] == name)
    stage = next(item for item in refinement["stages"] if item["sourcePathName"] == name)
    points = source["numericalMeshes"][INSTANCE]["verticesMeters"]
    faces = source["numericalMeshes"][INSTANCE]["triangles"]
    chain = crease["vertices"]
    if len(chain) < 2 or len(set(chain)) != len(chain):
        raise ValueError("Simple ordered source crease chain required")
    edges = _edges(faces)
    cuts = {tuple(sorted(pair)) for pair in zip(chain, chain[1:])}
    if any(edge not in edges or len(edges[edge]) != 2 for edge in cuts):
        raise ValueError("Two incident source faces per crease segment required")
    if any(not any(vertex in edge and len(entries) == 1 for edge, entries in edges.items()) for vertex in (chain[0], chain[-1])):
        raise ValueError("Source crease must reach both full cut boundaries")
    graph = [set() for _ in faces]
    for edge, entries in edges.items():
        if len(entries) == 2 and edge not in cuts:
            a, b = entries[0][0], entries[1][0]
            graph[a].add(b)
            graph[b].add(a)
    remaining, components = set(range(len(faces))), []
    while remaining:
        pending, component = [min(remaining)], set()
        while pending:
            face = pending.pop()
            if face not in component:
                component.add(face)
                pending.extend(graph[face] - component)
        components.append(component)
        remaining -= component
    if len(components) != 2:
        raise ValueError("A full source crease must partition the strip into exactly two regions")
    pairs = []
    for first, second in zip(chain, chain[1:]):
        adjacent = edges[tuple(sorted((first, second)))]
        body = [item for item in adjacent if (item[1], item[2]) == (first, second)]
        allowance = [item for item in adjacent if (item[1], item[2]) == (second, first)]
        if len(body) != 1 or len(allowance) != 1:
            raise ValueError("Consistent +z source winding across the directed crease required")
        pairs.append((first, second, body[0], allowance[0]))
    body_faces = next(component for component in components if pairs[0][2][0] in component)
    allowance_faces = next(component for component in components if pairs[0][3][0] in component)
    if body_faces == allowance_faces or any(body[0] not in body_faces or allowance[0] not in allowance_faces for _, _, body, allowance in pairs):
        raise ValueError("Source chain has inconsistent body/allowance region adjacency")
    body_vertices = {vertex for face in body_faces for vertex in faces[face]}
    allowance_vertices = {vertex for face in allowance_faces for vertex in faces[face]}
    if body_vertices & allowance_vertices != set(chain) or body_vertices | allowance_vertices != set(range(len(points))):
        raise ValueError("Both full source regions must meet only on the complete crease chain")
    line = crease["sourceLineEndpointsMeters"]
    direction = 1 if name == "right" else -1
    if line[0][0] != line[1][0] or direction * (line[1][1] - line[0][1]) <= 0:
        raise ValueError("Expected forward attachment and reverse finish source paths")
    if any(direction * (Fraction(points[b][1]) - Fraction(points[a][1])) <= 0 for a, b in zip(chain, chain[1:])):
        raise ValueError("Crease chain must follow the declared source path direction")
    rail = {"sourcePathName": name, "sourcePathSha256": stage["sourcePathSha256"],
        "sourceLineEndpointsMeters": copy.deepcopy(line), "sourceTangent": [0., float(direction), 0.],
        "geometricSourceNormal": [0., 0., 1.], "bodyLeftNormal": [float(-direction), 0., 0.],
        "chainVerticesLocal": list(chain), "chainVerticesCanonical": [source["instanceOffsets"][INSTANCE] + vertex for vertex in chain],
        "cutBoundaryEndpointsMeters": copy.deepcopy(crease["cutBoundaryEndpointsMeters"]),
        "bodyFacesLocal": sorted(body_faces), "allowanceFacesLocal": sorted(allowance_faces),
        "bodyVerticesOffChainLocal": sorted(body_vertices - set(chain)),
        "allowanceVerticesOffChainLocal": sorted(allowance_vertices - set(chain)),
        "storedRailOffsetsMeters": [_rational(Fraction(points[vertex][0]) - Fraction(line[0][0])) for vertex in chain]}
    return rail, pairs


def build_binding_fold_control(source, request, native_topology):
    """Derive complete source/native hinge bindings without installing controls."""
    before = [_bounded(value) for value in (source, request, native_topology)]
    validate_binding_source(source)
    if CONTROL_FIELDS.intersection(source):
        raise ValueError("Bare refined core required for source-bound fold derivation")
    folds, native = _request(request), _native(source, native_topology)
    offset, face_offset = source["instanceOffsets"][INSTANCE], source["instanceTriangleOffsets"][INSTANCE]
    points = source["restMeters"]
    rails, hinges, stiffness, targets = [], [], [], []
    for fold in folds:
        rail, pairs = _rail(source, fold["sourcePathName"])
        region_sign = -1 if fold["referenceRotationRegion"] == "body" else 1
        source_angle = float(fold["targetRightHandAngleRadians"]) * region_sign
        rail.update(referenceRotationRegion=fold["referenceRotationRegion"], sourceRelativeTargetAngleRadians=source_angle,
            stiffnessJoulesPerMeter=fold["stiffnessJoulesPerMeter"], segments=[])
        ideal_total, rounded_total = Fraction(), Fraction()
        for first, second, body, allowance in pairs:
            source_hinge = [offset + body[3], offset + allowance[3], offset + first, offset + second]
            native_index, native_hinge = native[tuple(sorted(source_hinge[2:]))]
            edge_sign = 1 if native_hinge[2:] == source_hinge[2:] else -1
            opposite_sign = 1 if native_hinge[:2] == source_hinge[:2] else -1
            native_sign = edge_sign * opposite_sign
            if native_sign != 1:
                raise ValueError("Source/native winding must preserve the directed relative angle")
            length = _length(points[offset + first], points[offset + second])
            ideal = Fraction(fold["stiffnessJoulesPerMeter"]) * Fraction(length["roundedLengthMeters"])
            numerical = float(ideal)
            if not math.isfinite(numerical) or numerical <= 0:
                raise ValueError("Derived line stiffness must remain positive finite binary64")
            ideal_total += ideal
            rounded_total += Fraction(numerical)
            target = source_angle * native_sign
            rail["segments"].append({"actuatorIndex": len(hinges), "orientedEdgeLocal": [first, second],
                "orientedEdgeCanonical": [offset + first, offset + second],
                "bodyFaceLocal": body[0], "bodyFaceCanonical": face_offset + body[0],
                "allowanceFaceLocal": allowance[0], "allowanceFaceCanonical": face_offset + allowance[0],
                "sourceDirectedHingeCanonical": source_hinge, "nativeHingeIndex": native_index,
                "nativeOrderedHingeCanonical": list(native_hinge), "nativeEdgeDirectionSign": edge_sign,
                "nativeOppositeOrderSign": opposite_sign, "nativeAngleSign": native_sign,
                "length": length, "idealStiffnessFromRoundedLengthJoules": _rational(ideal),
                "numericalStiffnessJoules": numerical, "stiffnessRoundingResidualJoules": _rational(Fraction(numerical) - ideal),
                "targetNativeAngleRadians": target})
            hinges.append(list(native_hinge))
            stiffness.append(numerical)
            targets.append(target)
        rail.update(idealTotalStiffnessFromRoundedLengthsJoules=_rational(ideal_total),
            numericalTotalStiffnessJoules=_rational(rounded_total),
            totalStiffnessRoundingResidualJoules=_rational(rounded_total - ideal_total))
        rails.append(rail)
    recipe = {"hinges": hinges, "stiffnessJoules": stiffness, "initialAnglesRadians": [0.] * len(hinges),
        "targetAnglesRadians": targets, "recipe": PROFILE, "accepted": False,
        "limitations": "External relative-angle line actuators; both cloth regions remain free. No textile-side, fixed-region, material, path or construction acceptance."}
    result = {"profile": PROFILE, "accepted": False, "solverReady": False, "executable": False,
        "sourceSha256": hashlib.sha256(before[0]).hexdigest(), "baseUnitSha256": _sha(source["baseUnit"]),
        "requestSha256": hashlib.sha256(before[1]).hexdigest(), "nativeTopologySha256": hashlib.sha256(before[2]).hexdigest(),
        "instanceId": INSTANCE, "request": copy.deepcopy(request), "nativeTopology": copy.deepcopy(native_topology),
        "rails": rails, "foldActuation": recipe,
        "units": {"length": "m", "angle": "rad", "stiffnessDensity": "J/(m rad^2)", "hingeStiffness": "J/rad^2"},
        "limitations": [
            "Source-bound research fold control derivation only; no controls are installed, no placement is generated and no trajectory is executed.",
            "Reference rotation region defines relative hinge sign; balanced internal actuator torques do not hold the other region fixed.",
            "Body and allowance are topological sides of the directed source path, not inferred textile right sides or final construction layer order.",
            "Actual ordered native hinges and all native/source triangles are bound explicitly; edge and opposite-vertex ordering cannot silently change fold direction.",
            "External stiffness density is explicit research actuation, not calibrated textile bending. Segment-length weighting does not establish mesh-convergent cloth mechanics.",
            "Stored crease coordinates and their source-line residuals remain unchanged. Squared-length and binary64 conversion witnesses do not claim exact irrational edge lengths.",
            "This control supplies no normal-offset sewing frame, attachment side override, solid tool, continuous path proof, wrap, stitch-down, apex or garment acceptance."]}
    if [_bounded(value) for value in (source, request, native_topology)] != before:
        raise ValueError("Fold derivation inputs changed")
    _bounded(result)
    return copy.deepcopy(result)


def validate_binding_fold_control(source, descriptor):
    claimed = _bounded(descriptor)
    if type(descriptor) is not dict or "request" not in descriptor or "nativeTopology" not in descriptor:
        raise ValueError("Complete fold descriptor required")
    expected = build_binding_fold_control(source, descriptor["request"], descriptor["nativeTopology"])
    if claimed != _encoded(expected):
        raise ValueError("Fold descriptor differs from complete source/control rederivation")
    return expected


def bind_binding_fold_control(source, descriptor, model):
    """Check exact current native topology and construct the existing primitive."""
    from solver_fold_actuation import FoldActuation
    checked = validate_binding_fold_control(source, descriptor)
    native = {"vertexCount": len(model.particle_mass.numpy()), "triangles": model.tri_indices.numpy().tolist(),
              "hinges": model.edge_indices.numpy().tolist()}
    if _bounded(native) != _encoded(checked["nativeTopology"]):
        raise ValueError("Fold descriptor does not match the current native model topology")
    recipe = checked["foldActuation"]
    primitive = FoldActuation(model, recipe["hinges"], recipe["stiffnessJoules"])
    return primitive, {"profile": "source-binding-fold-primitive-binding-v1", "verified": True, "accepted": False,
        "sourceSha256": checked["sourceSha256"], "descriptorSha256": _sha(checked),
        "nativeTopologySha256": checked["nativeTopologySha256"], "hingeCount": len(recipe["hinges"]),
        "scope": "Current native topology and existing fold primitive only; no numerical integration or construction execution."}
