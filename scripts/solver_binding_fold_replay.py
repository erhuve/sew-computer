"""Independent standard-library audit of a declared binding fold control.

This checks topology, relative-angle signs and prescribed binary64 arithmetic.
Pattern drafting and split-geometry validity remain a separate source-validator
responsibility. A verified descriptor is not an executed or accepted fold.
"""

from fractions import Fraction
import hashlib
import json
import math
import struct


PROFILE = "independent-binding-fold-verification-v1"
REQUIRED_HELPER_FILES = ()
INSTANCE = "opening_binding_left_left:shell"
TEMPLATE = "opening_binding_left_left"
MAX_BYTES = 24 * 1024 ** 2
CORE_FIELDS = {"profile", "accepted", "solverReady", "units", "sourceArcUnits", "baseUnit", "bindingRefinement",
    "bindingSeamRemap", "instances", "numericalMeshes", "instanceOffsets", "instanceTriangleOffsets", "restMeters",
    "triangles", "embeddedConstraints", "sourceRowCorrespondence", "derivationCodeDigests", "limitations"}


def _encoded(value):
    pending, remaining, text_bytes = [(value, 0)], 1000000, 0
    while pending:
        item, depth = pending.pop()
        remaining -= 1
        if remaining < 0 or depth > 40:
            raise ValueError("Independent fold structure budget exceeded")
        if type(item) is dict:
            if any(type(key) is not str for key in item) or len(item) * 2 > remaining:
                raise ValueError("Bounded raw JSON objects required")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif type(item) is list:
            if len(item) > remaining:
                raise ValueError("Independent fold array budget exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > MAX_BYTES:
                raise ValueError("Independent fold string budget exceeded")
            try:
                text_bytes += len(item.encode())
            except UnicodeEncodeError as error:
                raise ValueError("UTF-8 JSON strings required") from error
            if text_bytes > MAX_BYTES:
                raise ValueError("Independent fold text budget exceeded")
        elif type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Independent fold integer budget exceeded")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Finite raw JSON numbers required")
        elif item is not None and type(item) is not bool:
            raise ValueError("Raw JSON fold inputs required")
    result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(result) > MAX_BYTES:
        raise ValueError("Independent fold input byte budget exceeded")
    return result


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _same(actual, expected, label):
    if _encoded(actual) != _encoded(expected):
        raise ValueError("Independent fold mismatch: " + label)


def _number(value, maximum=100.):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > maximum:
        raise ValueError("Finite bounded non-Boolean binary64 number required")
    converted = float(value)
    if converted != value:
        raise ValueError("Exact binary64 input conversion required")
    return Fraction(converted)


def _index(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError("Bounded non-Boolean vertex or face index required")
    return value


def _rat(value):
    if value.numerator.bit_length() > 16384 or value.denominator.bit_length() > 16384:
        raise ValueError("Fold rational arithmetic budget exceeded")
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "roundedBinary64": float(value)}


def _float_bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _from_bits(bits):
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _sqrt_round(square):
    """Correct nearest-even binary64 sqrt, independently via exact bisection."""
    if type(square) is not Fraction or square <= 0:
        raise ValueError("Strictly positive exact squared edge length required")
    if square.numerator.bit_length() > 16384 or square.denominator.bit_length() > 16384:
        raise ValueError("Squared-length arithmetic budget exceeded")
    # Largest binary64 not exceeding the exact root. This never evaluates a
    # floating square root, so it is independent of the producer's seed/repair.
    low, high = 0, 0x7fefffffffffffff
    if Fraction(_from_bits(high)) ** 2 < square:
        raise ValueError("Edge length overflows binary64")
    while low < high:
        middle = (low + high + 1) // 2
        if Fraction(_from_bits(middle)) ** 2 <= square:
            low = middle
        else:
            high = middle - 1
    below = _from_bits(low)
    if Fraction(below) ** 2 == square:
        result = below
    else:
        above = _from_bits(low + 1)
        midpoint_square = ((Fraction(below) + Fraction(above)) / 2) ** 2
        result = below if square < midpoint_square or square == midpoint_square and low % 2 == 0 else above
    if result <= 0 or not math.isfinite(result):
        raise ValueError("Positive edge length becomes zero or nonfinite in binary64")
    return result


def _faces(value, count):
    if type(value) is not list or not 1 <= len(value) <= 50000:
        raise ValueError("Bounded nonempty source face array required")
    seen = set()
    for face in value:
        if type(face) is not list or len(face) != 3:
            raise ValueError("Three oriented triangle vertices required")
        for vertex in face:
            _index(vertex, count)
        key = tuple(sorted(face))
        if len(set(face)) != 3 or key in seen:
            raise ValueError("Unique nonrepeated source triangles required")
        seen.add(key)
    return value


def _mesh(source):
    if (type(source) is not dict or set(source) != CORE_FIELDS
            or source["profile"] != "source-left-binding-refined-unit-v1"
            or source["accepted"] is not False or source["solverReady"] is not False
            or source["units"] != "m" or source["sourceArcUnits"] != "mm"):
        raise ValueError("Bare explicit refined source core required")
    instances = source["instances"]
    if type(instances) is not list or len(instances) != 5:
        raise ValueError("Five source fabric instances required")
    names = [entry["id"] for entry in instances]
    if (len(set(names)) != 5 or INSTANCE not in names
            or any(type(name) is not str or not name or len(name) > 128 for name in names)
            or any(entry["mirrorX"] is not False for entry in instances)
            or any(set(source[field]) != set(names) for field in
                ("numericalMeshes", "instanceOffsets", "instanceTriangleOffsets"))):
        raise ValueError("Explicit ordered unmirrored source instance identities required")
    _same(instances, source["baseUnit"]["instances"], "immutable instance identities")
    positions, faces, owners, offsets, face_offsets = [], [], [], {}, {}
    for name in names:
        mesh = source["numericalMeshes"][name]
        if type(mesh) is not dict or set(mesh) != {"verticesMeters", "triangles"}:
            raise ValueError("Exact local numerical mesh fields required")
        points = mesh["verticesMeters"]
        if type(points) is not list or not 3 <= len(points) <= 25000 or len(positions) + len(points) > 25000:
            raise ValueError("Bounded complete material vertex arrays required")
        for point in points:
            if type(point) is not list or len(point) != 3:
                raise ValueError("Three explicit material coordinates required")
            for coordinate in point:
                _number(coordinate)
        local = _faces(mesh["triangles"], len(points))
        if len(faces) + len(local) > 50000:
            raise ValueError("Source face budget exceeded")
        offsets[name], face_offsets[name] = len(positions), len(faces)
        faces.extend([[v + len(positions) for v in face] for face in local])
        positions.extend(points)
        owners.extend([name] * len(points))
    _same(source["instanceOffsets"], offsets, "canonical vertex offsets")
    _same(source["instanceTriangleOffsets"], face_offsets, "canonical triangle offsets")
    _same(source["restMeters"], positions, "complete canonical rest positions")
    _same(source["triangles"], [v for face in faces for v in face], "complete canonical faces")
    _faces(faces, len(positions))
    return positions, faces, owners, offsets, face_offsets


def _edges(faces):
    result = {}
    for triangle, face in enumerate(faces):
        for i in range(3):
            a, b, opposite = face[i], face[(i + 1) % 3], face[(i + 2) % 3]
            incident = result.setdefault(tuple(sorted((a, b))), [])
            if len(incident) == 2 or incident and incident[0][1:3] != (b, a):
                raise ValueError("Oriented manifold source edges required")
            incident.append((triangle, a, b, opposite))
    return result


def _native(topology, positions, faces, owners, edges):
    if type(topology) is not dict or set(topology) != {"vertexCount", "triangles", "hinges"}:
        raise ValueError("Complete raw native topology fields required")
    _same(topology["vertexCount"], len(positions), "native vertex count")
    _same(topology["triangles"], faces, "native oriented source triangles")
    hinges = topology["hinges"]
    if type(hinges) is not list or len(hinges) != len(edges):
        raise ValueError("One native row for every source edge required")
    by_edge = {}
    for index, hinge in enumerate(hinges):
        if type(hinge) is not list or len(hinge) != 4:
            raise ValueError("Four native hinge indices required")
        for slot, value in enumerate(hinge):
            if type(value) is not int or not (-1 if slot < 2 else 0) <= value < len(positions):
                raise ValueError("Bounded non-Boolean native hinge indices required")
        first, second, a, b = hinge
        vertices = [v for v in hinge if v >= 0]
        key = tuple(sorted((a, b)))
        if (len(set(vertices)) != len(vertices) or len(set(owners[v] for v in vertices)) != 1
                or key not in edges or key in by_edge):
            raise ValueError("Unique native source edge with one material owner required")
        expected = {(edge_start, edge_end, opposite) for _, edge_start, edge_end, opposite in edges[key]}
        actual = {(a, b, first)} if first >= 0 else set()
        if second >= 0:
            actual.add((b, a, second))
        if actual != expected:
            raise ValueError("Ordered native hinge differs from oriented source face incidence")
        by_edge[key] = (index, hinge)
    if set(by_edge) != set(edges):
        raise ValueError("Native topology omits source edges")
    return by_edge


def _components(faces, edges, removed):
    neighbors = [set() for _ in faces]
    for edge, adjacent in edges.items():
        if edge not in removed and len(adjacent) == 2:
            a, b = adjacent[0][0], adjacent[1][0]
            neighbors[a].add(b)
            neighbors[b].add(a)
    remaining, components = set(range(len(faces))), []
    while remaining:
        pending, found = [min(remaining)], set()
        while pending:
            triangle = pending.pop()
            if triangle in found:
                continue
            found.add(triangle)
            pending.extend(neighbors[triangle] - found)
        remaining -= found
        components.append(found)
    return components


def _rail_partition(local_faces, chain):
    edges = _edges(local_faces)
    removed = {tuple(sorted((a, b))) for a, b in zip(chain, chain[1:])}
    if len(removed) != len(chain) - 1 or any(len(edges.get(edge, [])) != 2 for edge in removed):
        raise ValueError("Distinct internal source-directed chain edges required")
    boundary = {vertex for edge, incident in edges.items() if len(incident) == 1 for vertex in edge}
    if chain[0] not in boundary or chain[-1] not in boundary:
        raise ValueError("Full cut-boundary crease endpoints required")
    components = _components(local_faces, edges, removed)
    if len(components) != 2:
        raise ValueError("Complete crease cut must produce exactly two face components")
    body, allowance = set(), set()
    pairs = []
    for a, b in zip(chain, chain[1:]):
        incident = edges[tuple(sorted((a, b)))]
        left = next(item for item in incident if item[1:3] == (a, b))
        right = next(item for item in incident if item[1:3] == (b, a))
        body.add(left[0])
        allowance.add(right[0])
        pairs.append((left, right))
    body_component = next((component for component in components if body <= component), None)
    allowance_component = next((component for component in components if allowance <= component), None)
    if body_component is None or allowance_component is None or body_component == allowance_component:
        raise ValueError("Chain orientation must consistently separate body and allowance")
    return sorted(body_component), sorted(allowance_component), pairs


def _request(value):
    if (type(value) is not dict or set(value) != {"profile", "accepted", "angleConvention", "stiffnessMeasure", "folds"}
            or value["profile"] != "source-binding-fold-request-v1" or value["accepted"] is not False
            or value["angleConvention"] != "source-directed-right-hand-reference-region-v1"
            or value["stiffnessMeasure"] != "stored-crease-edge-length-v1"):
        raise ValueError("Explicit source-directed fold request required")
    folds = value["folds"]
    if type(folds) is not list or not 1 <= len(folds) <= 2:
        raise ValueError("One or two declared binding rails required")
    names = set()
    for fold in folds:
        if (type(fold) is not dict or set(fold) != {"sourcePathName", "referenceRotationRegion", "targetRightHandAngleRadians", "stiffnessJoulesPerMeter"}
                or type(fold["sourcePathName"]) is not str or fold["sourcePathName"] not in ("right", "left")
                or fold["sourcePathName"] in names or type(fold["referenceRotationRegion"]) is not str
                or fold["referenceRotationRegion"] not in ("body", "allowance")):
            raise ValueError("Unique source rail names and explicit reference regions required")
        angle = _number(fold["targetRightHandAngleRadians"], math.pi)
        density = _number(fold["stiffnessJoulesPerMeter"], 1e6)
        if abs(angle) >= Fraction(math.pi - 1e-8) or density <= 0:
            raise ValueError("Bounded principal angle and positive stiffness density required")
        names.add(fold["sourcePathName"])
    return folds


def _length(first, second):
    squared = sum(((Fraction(b) - Fraction(a)) ** 2 for a, b in zip(first, second)), Fraction())
    rounded = _sqrt_round(squared)
    lower = (Fraction(math.nextafter(rounded, -math.inf)) + Fraction(rounded)) / 2
    upper = (Fraction(math.nextafter(rounded, math.inf)) + Fraction(rounded)) / 2
    return {"exactSquaredLengthMetersSquared": _rat(squared), "roundedLengthMeters": rounded,
        "roundedLengthSquaredMinusExactMetersSquared": _rat(Fraction(rounded) ** 2 - squared),
        "lowerMidpointSquaredMetersSquared": _rat(lower ** 2),
        "upperMidpointSquaredMetersSquared": _rat(upper ** 2)}


def _rail(source, name, offset):
    refinement = source["bindingRefinement"]
    if (refinement["profile"] != "source-left-binding-refinement-descriptor-v1"
            or any(refinement[flag] is not False for flag in ("accepted", "solverReady", "executable"))
            or refinement["instanceId"] != INSTANCE or refinement["sourceTemplateId"] != TEMPLATE
            or refinement["sourceUnitSha256"] != _sha(source["baseUnit"])):
        raise ValueError("Explicit immutable-base refinement identity required")
    stages, creases = refinement["stages"], refinement["finalCreases"]
    if (type(stages) is not list or type(creases) is not list or len(stages) != 2 or len(creases) != 2
            or [stage["sourcePathName"] for stage in stages] != ["right", "left"]
            or [crease["sourcePathName"] for crease in creases] != ["right", "left"]):
        raise ValueError("Complete ordered two-source-line refinement required")
    stage, crease = [(stages[i], creases[i]) for i in range(2) if stages[i]["sourcePathName"] == name][0]
    mesh = source["numericalMeshes"][INSTANCE]
    points, faces = mesh["verticesMeters"], mesh["triangles"]
    if len(points) != 29 or len(faces) != 44 or any(point[2] != 0 for point in points):
        raise ValueError("Unchanged planar 29-vertex/44-face refined binding required")
    _same(refinement["finalLocalMesh"], {"verticesMeters": [point[:2] for point in points], "triangles": faces},
          "refinement and numerical binding mesh")
    for a, b, c in faces:
        first, second, third = [[Fraction(v) for v in points[index][:2]] for index in (a, b, c)]
        if (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0]) <= 0:
            raise ValueError("Nondegenerate +z source triangle winding required")
    panels = [panel for panel in source["baseUnit"]["sourcePattern"]["panels"] if panel["id"] == TEMPLATE]
    if len(panels) != 1:
        raise ValueError("Unique immutable binding source panel required")
    panel = panels[0]
    paths = [path for path in panel["draft"]["edges"] if path["name"] == name]
    if len(paths) != 1:
        raise ValueError("Unique named source path required")
    path = paths[0]
    source_start, source_end, x, direction = (1, 2, .020, 1) if name == "right" else (3, 4, 0., -1)
    _same([path["start"], path["end"]], [source_start, source_end], "oriented original source endpoints")
    line = [[float(_number(coordinate, 100000.)) * .001 for coordinate in panel["points"][index]]
            for index in (source_start, source_end)]
    if (any(len(point) != 2 or point[0] != x for point in line)
            or direction * (Fraction(line[1][1]) - Fraction(line[0][1])) <= 0):
        raise ValueError("Declared right/left straight source path geometry required")
    _same(stage["sourcePathSha256"], _sha(path), "immutable named source path hash")
    _same([stage["sourceStart"], stage["sourceEnd"]], [source_start, source_end], "split source endpoint identity")
    _same(stage["sourceLineEndpointsMeters"], line, "split source line")
    _same(crease["sourceLineEndpointsMeters"], line, "final source line")
    chain = crease["vertices"]
    if type(chain) is not list or not 2 <= len(chain) <= 29:
        raise ValueError("Bounded full crease chain required")
    for vertex in chain:
        _index(vertex, len(points))
    if len(set(chain)) != len(chain):
        raise ValueError("Simple source crease chain required")
    _same(stage["output"]["creaseVertices"], chain, "retained ordered split chain")
    chain_edges = sorted([sorted(pair) for pair in zip(chain, chain[1:])])
    _same(sorted(crease["edges"]), chain_edges, "complete final crease edge set")
    _same(sorted(stage["output"]["creaseEdges"]), chain_edges, "retained split crease edges")
    if any(direction * (Fraction(points[b][1]) - Fraction(points[a][1])) <= 0 for a, b in zip(chain, chain[1:])):
        raise ValueError("Chain direction must match the immutable source path")
    ends = [points[chain[0]][:2], points[chain[-1]][:2]]
    _same(crease["cutBoundaryEndpointsMeters"], ends, "unchanged full cut-boundary endpoints")
    body, allowance, pairs = _rail_partition(faces, chain)
    body_vertices = {vertex for face in body for vertex in faces[face]}
    allowance_vertices = {vertex for face in allowance for vertex in faces[face]}
    if body_vertices & allowance_vertices != set(chain) or body_vertices | allowance_vertices != set(range(len(points))):
        raise ValueError("Source regions must cover the strip and meet only on the chain")
    return {"sourcePathName": name, "sourcePathSha256": _sha(path),
        "sourceLineEndpointsMeters": line, "sourceTangent": [0., float(direction), 0.],
        "geometricSourceNormal": [0., 0., 1.], "bodyLeftNormal": [float(-direction), 0., 0.],
        "chainVerticesLocal": list(chain), "chainVerticesCanonical": [offset + vertex for vertex in chain],
        "cutBoundaryEndpointsMeters": ends, "bodyFacesLocal": body, "allowanceFacesLocal": allowance,
        "bodyVerticesOffChainLocal": sorted(body_vertices - set(chain)),
        "allowanceVerticesOffChainLocal": sorted(allowance_vertices - set(chain)),
        "storedRailOffsetsMeters": [_rat(Fraction(points[v][0]) - Fraction(line[0][0])) for v in chain]}, pairs


def verify_binding_fold_control(source, descriptor):
    """Rederive the declared control from supplied source/topology, never execute."""
    before = [_encoded(value) for value in (source, descriptor)]
    try:
        positions, faces, owners, offsets, face_offsets = _mesh(source)
        fields = {"profile", "accepted", "solverReady", "executable", "sourceSha256", "baseUnitSha256",
            "requestSha256", "nativeTopologySha256", "instanceId", "request", "nativeTopology", "rails",
            "foldActuation", "units", "limitations"}
        if (type(descriptor) is not dict or set(descriptor) != fields
                or descriptor["profile"] != "source-binding-fold-control-v1"
                or any(descriptor[flag] is not False for flag in ("accepted", "solverReady", "executable"))
                or descriptor["instanceId"] != INSTANCE):
            raise ValueError("Exact unaccepted fold descriptor fields required")
        request, topology = descriptor["request"], descriptor["nativeTopology"]
        for field, value in (("sourceSha256", source), ("baseUnitSha256", source["baseUnit"]),
                             ("requestSha256", request), ("nativeTopologySha256", topology)):
            _same(descriptor[field], _sha(value), field)
        folds = _request(request)
        edges = _edges(faces)
        native = _native(topology, positions, faces, owners, edges)
        offset, face_offset = offsets[INSTANCE], face_offsets[INSTANCE]
        rails, hinges, stiffness, targets = [], [], [], []
        for fold in folds:
            rail, pairs = _rail(source, fold["sourcePathName"], offset)
            source_angle = float(fold["targetRightHandAngleRadians"]) * (-1 if fold["referenceRotationRegion"] == "body" else 1)
            rail.update(referenceRotationRegion=fold["referenceRotationRegion"], sourceRelativeTargetAngleRadians=source_angle,
                        stiffnessJoulesPerMeter=fold["stiffnessJoulesPerMeter"], segments=[])
            ideal_total, rounded_total = Fraction(), Fraction()
            for left, right in pairs:
                a, b = left[1:3]
                source_hinge = [offset + left[3], offset + right[3], offset + a, offset + b]
                native_index, native_hinge = native[tuple(sorted(source_hinge[2:]))]
                # Only the two oriented-face-preserving permutations are valid.
                if native_hinge == source_hinge:
                    edge_sign = opposite_sign = 1
                elif native_hinge == [source_hinge[1], source_hinge[0], source_hinge[3], source_hinge[2]]:
                    edge_sign = opposite_sign = -1
                else:
                    raise ValueError("Native hinge cannot preserve the source relative-angle convention")
                length = _length(positions[offset + a], positions[offset + b])
                ideal = Fraction(fold["stiffnessJoulesPerMeter"]) * Fraction(length["roundedLengthMeters"])
                numerical = float(ideal)
                if numerical <= 0 or not math.isfinite(numerical):
                    raise ValueError("Positive finite once-rounded segment stiffness required")
                ideal_total += ideal
                rounded_total += Fraction(numerical)
                rail["segments"].append({"actuatorIndex": len(hinges), "orientedEdgeLocal": [a, b],
                    "orientedEdgeCanonical": [offset + a, offset + b],
                    "bodyFaceLocal": left[0], "bodyFaceCanonical": face_offset + left[0],
                    "allowanceFaceLocal": right[0], "allowanceFaceCanonical": face_offset + right[0],
                    "sourceDirectedHingeCanonical": source_hinge, "nativeHingeIndex": native_index,
                    "nativeOrderedHingeCanonical": list(native_hinge), "nativeEdgeDirectionSign": edge_sign,
                    "nativeOppositeOrderSign": opposite_sign, "nativeAngleSign": 1,
                    "length": length, "idealStiffnessFromRoundedLengthJoules": _rat(ideal),
                    "numericalStiffnessJoules": numerical, "stiffnessRoundingResidualJoules": _rat(Fraction(numerical) - ideal),
                    "targetNativeAngleRadians": source_angle})
                hinges.append(list(native_hinge))
                stiffness.append(numerical)
                targets.append(source_angle)
            rail.update(idealTotalStiffnessFromRoundedLengthsJoules=_rat(ideal_total),
                numericalTotalStiffnessJoules=_rat(rounded_total),
                totalStiffnessRoundingResidualJoules=_rat(rounded_total - ideal_total))
            rails.append(rail)
        _same(descriptor["rails"], rails, "complete ordered rails and arithmetic witnesses")
        expected_recipe = {"hinges": hinges, "stiffnessJoules": stiffness, "initialAnglesRadians": [0.] * len(hinges),
            "targetAnglesRadians": targets, "recipe": "source-binding-fold-control-v1", "accepted": False,
            "limitations": "External relative-angle line actuators; both cloth regions remain free. No textile-side, fixed-region, material, path or construction acceptance."}
        _same(descriptor["foldActuation"], expected_recipe, "complete fold actuator recipe")
        _same(descriptor["units"], {"length": "m", "angle": "rad", "stiffnessDensity": "J/(m rad^2)", "hingeStiffness": "J/rad^2"}, "declared units")
        if (type(descriptor["limitations"]) is not list or not 1 <= len(descriptor["limitations"]) <= 16
                or any(type(item) is not str or not item or len(item) > 4096 for item in descriptor["limitations"])):
            raise ValueError("Bounded explicit limitation statements required")
        if before != [_encoded(value) for value in (source, descriptor)]:
            raise ValueError("Fold verification inputs changed")
        return {"profile": PROFILE, "verified": True, "accepted": False,
            "sourceSha256": _sha(source), "descriptorSha256": _sha(descriptor),
            "baseUnitSha256": _sha(source["baseUnit"]), "requestSha256": _sha(request),
            "nativeTopologySha256": _sha(topology), "foldActuationSha256": _sha(expected_recipe),
            "sourceVertexCount": len(positions), "sourceTriangleCount": len(faces), "nativeEdgeCount": len(edges),
            "railCount": len(rails), "hingeCount": len(hinges),
            "railNames": [rail["sourcePathName"] for rail in rails],
            "sourceScope": "Independent supplied-source topology, full native edge incidence, oriented dual-graph region partition, relative-angle signs, exact squared lengths and prescribed nearest binary64 length/stiffness arithmetic. Pattern and split-geometry validity require the separately captured source validator. Textual limitations are not mathematical evidence. No fixed-region motion, textile-side assignment, material calibration, stability, trajectory, wrap or construction acceptance."}
    except (KeyError, TypeError, IndexError, StopIteration, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Malformed or incomplete independent fold audit input") from error
