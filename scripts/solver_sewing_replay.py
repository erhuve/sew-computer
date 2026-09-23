"""Independent current verifier for captured sewing activation and work.

Only stdlib and NumPy are imported. This verifies captured canonical JSON,
sampled geometry, forces and discrete work. Cuff pattern/engine rederivation
remains the separately checked captured source-binder boundary. Neither seam
activation nor a matching phase plan proves construction completion.
"""

from fractions import Fraction
import hashlib
import json
import math

import numpy as np


PROFILE = "independent-captured-sewing-replay-v1"
INPUT_PROFILE = "captured-sewing-activation-v1"
REFINED_PROFILE = "source-left-binding-refined-unit-v1"
GEOMETRY_SAMPLER = "binary64 sorted CSR fused-multiply-add sampled anchors; verified on pinned macOS ARM and Linux ARM, not a universal sparse-backend guarantee"
METRIC = "Unweighted maximum absolute Cartesian component per active vector/normal row; absolute scalar-distance error in distance mode; pending rows excluded"
LIMITATIONS = "Per-row energy weights, not completed construction phases or continuous seam coverage; parameter work requires adjacent controls"
CONSTRUCTION = "Source phase plan remains declarative and unexecuted; numerical knot times do not establish binding, turning, gate completion or any construction milestone."
BINDING_SCOPE = "Explicit virtual seam controls preserve all canonical source rows. Activation is force weighting, not executed construction or completed seam evidence. Normal sides are declared mechanical frame offsets, not textile right-side assignments."
_EPS, _TINY = np.finfo(float).eps, float(np.nextafter(0., 1.))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _source_json(value):
    remaining, byte_count = 1000000, 0

    def walk(item, depth):
        nonlocal remaining, byte_count
        remaining -= 1
        if remaining < 0 or depth > 40:
            raise ValueError("Sewing source exceeds structural budget")
        if item is None or type(item) is bool:
            return
        if type(item) is int and item.bit_length() <= 63:
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is str:
            byte_count += len(item.encode())
            if byte_count > 8 * 1024 ** 2:
                raise ValueError("Sewing source exceeds string budget")
            return
        if type(item) is list and len(item) <= remaining:
            for child in item:
                walk(child, depth + 1)
            return
        if type(item) is dict and len(item) * 2 <= remaining and all(type(key) is str for key in item):
            for key, child in item.items():
                walk(key, depth + 1)
                walk(child, depth + 1)
            return
        raise ValueError("Finite bounded raw JSON sewing source required")

    try:
        walk(value, 0)
        result = _json(value)
        if len(result.encode()) > 8 * 1024 ** 2:
            raise ValueError("Sewing source exceeds byte budget")
        return result
    except (TypeError, OverflowError, UnicodeError, RecursionError) as error:
        raise ValueError("Invalid bounded sewing source") from error


def _identity(value):
    return type(value) is str and 1 <= len(value) <= 160 and all(ord(char) >= 32 for char in value)


def _number(value, lower=-math.inf, upper=math.inf, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper
            or positive and value <= 0):
        raise ValueError("Invalid finite non-Boolean sewing scalar")
    return float(value)


def _bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b" or value.dtype.kind == "O" and any(_bool(item) for item in value.flat)
    return isinstance(value, (list, tuple)) and any(_bool(item) for item in value)


def _array(value, shape, label):
    try:
        raw = np.asarray(value)
        if _bool(value) or raw.shape != shape or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
            raise ValueError("Invalid " + label)
        return raw.astype(float, copy=True)
    except (TypeError, OverflowError) as error:
        raise ValueError("Invalid " + label) from error


def _fraction(value):
    if type(value) not in (int, float, Fraction):
        raise ValueError("Explicit dyadic sewing fraction required")
    try:
        result = Fraction(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Finite sewing fraction required") from error
    if not 0 <= result <= 1 or result.denominator > 2 ** 40 or result.denominator & (result.denominator - 1):
        raise ValueError("Sewing fraction exceeds bounded dyadic grid")
    return result


class SewingRecord:
    __slots__ = ("vertex_count", "faces", "rows", "row_ids", "mode", "compliance", "initial_targets",
                 "final_targets", "fractions", "activation", "frames", "sides", "source_sha256",
                 "recipe_json", "binding_json", "cuff", "refined_verification_json")

    def __setattr__(self, name, value):
        raise AttributeError("Independent sewing bindings are immutable")

    def __delattr__(self, name):
        raise AttributeError("Independent sewing bindings are immutable")


def derive_sewing(source, subdivisions, *, sewing_mode, refined_source_verifier=None):
    """Independently bind complete source JSON, ordered rows and all controls."""
    if (type(source) is not dict or type(subdivisions) is not int or not 1 <= subdivisions <= 4096
            or subdivisions & (subdivisions - 1) or sewing_mode not in ("vector", "distance", "normal-offset")):
        raise ValueError("Canonical sewing source, mode and bounded subdivisions required")
    source_json = _source_json(source)
    digest = _sha({key: value for key, value in source.items() if key != "sewingActuation"})
    recipe = source.get("sewingActuation")
    if (type(recipe) is not dict or set(recipe) != {"profile", "accepted", "sourceSha256", "mode",
            "initialTargetsMeters", "finalTargetsMeters", "schedule"} or recipe["profile"] != INPUT_PROFILE
            or recipe["accepted"] is not False or recipe["sourceSha256"] != digest or recipe["mode"] != sewing_mode):
        raise ValueError("Sewing recipe differs from full captured source identity")
    profile = source.get("profile")
    refined_claim = (type(profile) is str and profile.startswith("source-left-binding-")
                     or bool({"baseUnit", "bindingRefinement", "bindingSeamRemap"}.intersection(source)))
    refined = profile == REFINED_PROFILE
    if refined_claim and not refined:
        raise ValueError("Refined cuff source cannot downgrade to another source profile")
    if refined and sewing_mode == "normal-offset":
        raise ValueError("Refined cuff normal-offset sewing has no declared crease-side frame policy")
    refined_evidence = None
    if refined:
        if not callable(refined_source_verifier):
            raise ValueError("Current independent refined-source remap verifier required")
        refined_evidence = refined_source_verifier(source)
        if _source_json(source) != source_json:
            raise ValueError("Independent refined-source audit changed its input")
        if (type(refined_evidence) is not dict or refined_evidence.get("verified") is not True
                or refined_evidence.get("accepted") is not False or refined_evidence.get("sourceSha256") != digest):
            raise ValueError("Independent refined-source evidence must bind this complete source")
        _source_json(refined_evidence)
    cuff = profile == "source-cuff-construction-unit-v1" or refined
    cuff_keys = {"sourcePattern", "sourceConstruction", "sourceInventory", "sourceAssembly", "phasePlan",
                 "phaseConstraintRows", "selectedOperationIds", "excludedOperationIds",
                 "unexecutedOtherOperationsTouchingUnit", "unexecutedClosuresTouchingUnit"}
    if not cuff and (cuff_keys.intersection(source) or type(source.get("provenance")) is dict
                     and "phasePlanSha256" in source["provenance"]):
        raise ValueError("Cuff source cannot downgrade to generic JSON binding")
    rest, face_data, offsets = source.get("restMeters"), source.get("triangles"), source.get("instanceOffsets")
    if type(rest) is not list or not 3 <= len(rest) <= 25000:
        raise ValueError("Bounded sewing rest mesh required")
    for point in rest:
        if type(point) is not list or len(point) != 3:
            raise ValueError("Three rest coordinates required")
        for component in point:
            _number(component, -100, 100)
    if type(face_data) is not list or not face_data:
        raise ValueError("Canonical sewing faces required")
    if type(face_data[0]) is list:
        faces = face_data
    else:
        if len(face_data) % 3 or len(face_data) > 150000:
            raise ValueError("Canonical triangle index shape invalid")
        faces = [face_data[index:index + 3] for index in range(0, len(face_data), 3)]
    if (not 1 <= len(faces) <= 50000 or any(type(face) is not list or len(face) != 3
            or any(type(vertex) is not int or not 0 <= vertex < len(rest) for vertex in face)
            or len(set(face)) != 3 for face in faces)
            or len({tuple(sorted(face)) for face in faces}) != len(faces)):
        raise ValueError("Distinct bounded oriented triangle identities required")
    if (type(offsets) is not dict or not 1 <= len(offsets) <= 64
            or any(not _identity(key) or type(value) is not int or not 0 <= value < len(rest)
                   for key, value in offsets.items())):
        raise ValueError("Physical sewing instance offsets required")
    ordered = sorted(offsets, key=offsets.get)
    if offsets[ordered[0]] != 0 or len(set(offsets.values())) != len(ordered):
        raise ValueError("Instance offsets must partition canonical cloth")
    owners, counts = [], {}
    for index, name in enumerate(ordered):
        end = offsets[ordered[index + 1]] if index + 1 < len(ordered) else len(rest)
        counts[name] = end - offsets[name]
        if counts[name] < 3:
            raise ValueError("Every physical instance requires a cloth triangle")
        owners.extend([name] * counts[name])
    adjacency = [set() for _ in rest]
    for index, face in enumerate(faces):
        if len({owners[vertex] for vertex in face}) != 1:
            raise ValueError("Triangle crosses physical instances")
        for vertex in face:
            adjacency[vertex].add(index)
    if any(not incident for incident in adjacency):
        raise ValueError("All canonical vertices require triangle membership")
    bundle = source.get("embeddedConstraints")
    source_rows = bundle.get("constraints") if type(bundle) is dict else None
    if type(source_rows) is not list or not 1 <= len(source_rows) <= 4096:
        raise ValueError("Complete bounded canonical sewing rows required")
    rows, ids, bindings, selectors = [], [], [], {}
    registration_order, last_key, compliance = [], None, None
    for index, row in enumerate(source_rows):
        if type(row) is not dict:
            raise ValueError("Source sewing row object required")
        registration, member = row.get("registrationId"), row.get("memberIndex")
        if not _identity(registration) or type(member) is not int or not 1 <= member <= 7:
            raise ValueError("Registration and star member identities required")
        _number(row.get("fraction"), 0, 1)
        fraction = Fraction(row["fraction"])
        key = fraction, member
        if not registration_order or registration_order[-1] != registration:
            if registration in registration_order:
                raise ValueError("Source registration blocks cannot be interleaved")
            registration_order.append(registration)
            last_key = None
        if last_key is not None and key <= last_key:
            raise ValueError("Source sample/member order must be strictly preserved")
        last_key = key
        identity = "row:" + _sha([registration, member, fraction.numerator, fraction.denominator])
        if identity in ids:
            raise ValueError("Duplicate source row identity")
        value = _number(row.get("complianceMPerN"), 0, 1000, positive=True)
        if compliance is not None and value != compliance:
            raise ValueError("Heterogeneous source compliance is unsupported")
        compliance = value
        terms = row.get("terms")
        if type(terms) is not list or not 2 <= len(terms) <= 6:
            raise ValueError("Complete source anchor coefficients required")
        sparse, anchors, names = {}, {1: {}, -1: {}}, {}
        for term in terms:
            if type(term) is not dict or set(term) != {"instanceId", "vertex", "coefficient"}:
                raise ValueError("Exact sparse source term fields required")
            name, vertex = term["instanceId"], term["vertex"]
            if type(name) is not str or name not in counts or type(vertex) is not int or not 0 <= vertex < counts[name]:
                raise ValueError("Unknown source anchor support")
            coefficient = _number(term["coefficient"], -1, 1)
            if not coefficient or offsets[name] + vertex in sparse:
                raise ValueError("Duplicate or zero sparse sewing term")
            sign = 1 if coefficient > 0 else -1
            if sign in names and names[sign] != name:
                raise ValueError("One instance per material anchor required")
            names[sign] = name
            anchors[sign][vertex] = coefficient
            sparse[offsets[name] + vertex] = coefficient
        if set(names) != {-1, 1} or names[1] == names[-1]:
            raise ValueError("Two different source instances required")
        for sign, anchor in anchors.items():
            support = {offsets[names[sign]] + vertex for vertex in anchor}
            if (not 1 <= len(anchor) <= 3 or abs(math.fsum(anchor.values()) - sign) > 1e-12
                    or not any(support.issubset(faces[face]) for face in adjacency[min(support)])):
                raise ValueError("Normalized source triangle anchor required")
        if "sourceSamples" in row:
            samples = row["sourceSamples"]
            if type(samples) is not list or len(samples) != 2:
                raise ValueError("Two signed source samples required")
            for sign, sample in zip((1, -1), samples):
                if type(sample) is not dict or sample.get("instanceId") != names[sign]:
                    raise ValueError("Source sample instance/sign order differs")
                weights = sample.get("weights")
                if type(weights) is not list or not 1 <= len(weights) <= 3:
                    raise ValueError("Explicit bounded sample weights required")
                expected, seen = {}, set()
                for item in weights:
                    if (type(item) is not dict or set(item) != {"vertex", "weight"}
                            or type(item["vertex"]) is not int or not 0 <= item["vertex"] < counts[names[sign]]
                            or item["vertex"] in seen):
                        raise ValueError("Distinct source sample vertices required")
                    seen.add(item["vertex"])
                    amount = _number(item["weight"], 0, 1)
                    if amount:
                        expected[item["vertex"]] = sign * amount
                if expected != anchors[sign]:
                    raise ValueError("Source samples differ from sparse coefficients")
        rows.append(tuple(sorted(sparse.items())))
        ids.append(identity)
        selectors.setdefault((registration, member), []).append(index)
        bindings.append({"rowIndex": index, "rowId": identity, "registrationId": registration, "memberIndex": member,
            "fractionNumerator": fraction.numerator, "fractionDenominator": fraction.denominator,
            "positiveInstanceId": names[1], "negativeInstanceId": names[-1],
            "canonicalTerms": [{"vertex": vertex, "coefficient": weight} for vertex, weight in sparse.items()],
            "sourceRowSha256": _sha(row)})
    count = len(ids)
    target_shape = (count, 3) if sewing_mode == "vector" else (count,)
    initial = _array(recipe["initialTargetsMeters"], target_shape, "initial sewing targets")
    final = _array(recipe["finalTargetsMeters"], target_shape, "final sewing targets")
    if any(np.any(np.abs(target) > 100) or sewing_mode != "vector" and np.any(target <= 0) for target in (initial, final)):
        raise ValueError("Bounded explicit positive scalar sewing targets required")
    schedule = recipe["schedule"]
    if (type(schedule) is not dict or set(schedule) != {"profile", "rowIds", "knots"}
            or schedule["profile"] != "sewing-row-activation-v1" or schedule["rowIds"] != ids
            or type(schedule["knots"]) is not list or not 2 <= len(schedule["knots"]) <= 65):
        raise ValueError("Explicit complete ordered sewing schedule required")
    fractions, activation = [], []
    for knot in schedule["knots"]:
        if type(knot) is not dict or set(knot) != {"fraction", "activation"}:
            raise ValueError("Exact sewing knot fields required")
        fraction = _fraction(knot["fraction"])
        weights = _array(knot["activation"], (count,), "sewing activation")
        if (fraction * subdivisions).denominator != 1 or fractions and fraction <= fractions[-1]:
            raise ValueError("Ordered sewing knots must lie on initial interval boundaries")
        if np.any(weights < 0) or np.any(weights > 1) or activation and np.any(weights < activation[-1]):
            raise ValueError("Sewing weights must be monotone without release")
        fractions.append(fraction)
        activation.append(tuple(weights))
    if fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError("Sewing schedule must cover the original full interval")
    if cuff:
        if count != 40 or len(selectors) != 8:
            raise ValueError("Cuff controls must retain all 40 rows and eight selectors")
        for indices in selectors.values():
            if len(indices) != 5 or [Fraction(source_rows[i]["fraction"]) for i in indices] != [Fraction(i, 4) for i in range(5)]:
                raise ValueError("Cuff selector must retain its five source fractions")
            if any(len({knot[index] for index in indices}) != 1 for knot in activation):
                raise ValueError("All samples of a cuff selector activate together")
    frame_faces, frame_sides, frame_bindings = [], [], []
    if sewing_mode == "normal-offset":
        declaration = source.get("sewingFrames")
        if (type(declaration) is not dict or set(declaration) != {"faces", "sides", "bindings"}
                or any(type(declaration[key]) is not list or len(declaration[key]) != count for key in declaration)):
            raise ValueError("Explicit complete source normal frames required")
        for index, row in enumerate(rows):
            binding, face, side = (declaration[key][index] for key in ("bindings", "faces", "sides"))
            if (type(binding) is not dict or set(binding) != {"rowId", "instanceId", "triangleIndex", "side"}
                    or binding["rowId"] != ids[index] or type(binding["triangleIndex"]) is not int
                    or not 0 <= binding["triangleIndex"] < len(faces) or type(side) is not int or side not in (-1, 1)
                    or type(binding["side"]) is not int or binding["side"] != side
                    or type(face) is not list or len(face) != 3 or any(type(v) is not int for v in face)
                    or face != faces[binding["triangleIndex"]]):
                raise ValueError("Normal frame winding, side or source identity mismatch")
            name = binding["instanceId"]
            if (name != bindings[index]["negativeInstanceId"] or any(owners[v] != name for v in face)
                    or not {v for v, w in row if w < 0}.issubset(face) or {v for v, w in row if w > 0}.intersection(face)):
                raise ValueError("Normal frame must contain the negative source anchor only")
            evidence = dict(binding, canonicalVertices=face.copy(), instanceLocalVertices=[v - offsets[name] for v in face])
            if cuff:
                instances = {item["id"]: item for item in source["instances"]}
                template = instances[name]["templateId"]
                matches = [i for i, source_face in enumerate(source["sourceTemplates"][template]["triangles"])
                           if source_face == evidence["instanceLocalVertices"]]
                if len(matches) != 1:
                    raise ValueError("Cuff normal frame lacks unique oriented source triangle")
                evidence.update(templateId=template, sourceTriangleIndex=matches[0])
            frame_bindings.append(evidence)
            frame_faces.append(tuple(face))
            frame_sides.append(side)
    expected_binding = {"profile": INPUT_PROFILE, "accepted": False, "sourceSha256": digest,
        "mode": sewing_mode, "rowIds": ids, "rowBindings": bindings, "complianceMPerN": compliance,
        "sourceBundleSha256": _sha(bundle), "frameBindings": frame_bindings,
        "sourceFrameMetadataUsed": sewing_mode == "normal-offset",
        "sourceBindingScope": ("rederived refined cuff source with recorded coefficient approximation" if refined else
                               "rederived cuff construction source" if cuff else "captured canonical JSON only; no pattern-source proof"),
        "constructionStatus": CONSTRUCTION, "scope": BINDING_SCOPE}
    record = SewingRecord()
    freeze_target = lambda values: tuple(tuple(row) for row in values) if values.ndim == 2 else tuple(values)
    for key, value in dict(vertex_count=len(rest), faces=tuple(tuple(face) for face in faces), rows=tuple(rows),
            row_ids=tuple(ids), mode=sewing_mode, compliance=compliance, initial_targets=freeze_target(initial),
            final_targets=freeze_target(final), fractions=tuple(fractions), activation=tuple(activation),
            frames=tuple(frame_faces), sides=tuple(frame_sides), source_sha256=digest, recipe_json=_json(recipe),
            binding_json=_json(expected_binding), cuff=cuff, refined_verification_json=_json(refined_evidence)).items():
        object.__setattr__(record, key, value)
    return record


def parameters(record, fraction):
    fraction = _fraction(fraction)
    if fraction in record.fractions:
        return np.array(record.activation[record.fractions.index(fraction)])
    upper = next(i for i, knot in enumerate(record.fractions) if knot > fraction)
    amount = float((fraction - record.fractions[upper - 1]) / (record.fractions[upper] - record.fractions[upper - 1]))
    return np.array([a + amount * (b - a) for a, b in zip(record.activation[upper - 1], record.activation[upper])])


def targets(record, progress):
    progress = _number(progress, 0, 1)
    start, end = np.array(record.initial_targets), np.array(record.final_targets)
    return end if progress == 1 else start + progress * (end - start)


def _cross(first, second):
    return [first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0]]


def _geometry(record, positions, active):
    q = _array(positions, (record.vertex_count, 3), "sewing positions")
    geometry = {}
    for index in np.flatnonzero(active):
        row = record.rows[index]
        exact = [sum((Fraction(w) * Fraction(float(q[v, axis])) for v, w in row), Fraction()) for axis in range(3)]
        # The pinned ARM SciPy CSR kernel performs binary64 fused multiply-add
        # in sorted sparse index order. Sample it explicitly without importing
        # SciPy; separate NumPy multiply/add changes endpoint norms by an ulp.
        sampled = np.zeros(3)
        for vertex, weight in row:
            for axis in range(3):
                sampled[axis] = math.fma(weight, float(q[vertex, axis]), float(sampled[axis]))
        result = {"exact": exact, "sampled": sampled}
        if record.mode == "distance":
            length = float(np.sqrt(np.sum(sampled * sampled)))
            if not math.isfinite(length) or length <= 0:
                raise ValueError("Active distance anchor is coincident or nonfinite")
            result["length"] = length
        elif record.mode == "normal-offset":
            a, b, c = q[list(record.frames[index])]
            first, second = b - a, c - a
            cross = np.cross(first, second)
            magnitude = float(np.sqrt(np.sum(cross * cross)))
            scale = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)))
            if not math.isfinite(magnitude) or not math.isfinite(scale) or scale <= 0 or magnitude <= 1e-12 * scale ** 2:
                raise ValueError("Active source normal frame is degenerate")
            result.update(normal=cross / magnitude, first=first, second=second, magnitude=magnitude)
        geometry[index] = result
    return q, geometry


def _error(record, index, geometry, target):
    if record.mode == "distance":
        return [Fraction(geometry["length"]) - Fraction(float(target))]
    if record.mode == "normal-offset":
        offset = Fraction(float(target)) * record.sides[index]
        return [coordinate - offset * Fraction(float(normal)) for coordinate, normal in zip(geometry["exact"], geometry["normal"])]
    return [coordinate - Fraction(float(goal)) for coordinate, goal in zip(geometry["exact"], target)]


def _energy(record, geometry, target, activation):
    return sum((Fraction(float(activation[i])) * sum((e * e for e in _error(record, i, g, target[i])), Fraction())
                / (2 * Fraction(record.compliance)) for i, g in geometry.items() if activation[i] > 0), Fraction())


def _row_errors(record, geometry, target):
    errors, scales = [None] * len(record.rows), np.zeros(len(record.rows))
    for i, g in geometry.items():
        if record.mode == "distance":
            error = abs(g["length"] - target[i])
            scale = g["length"] + abs(target[i])
        else:
            offset = target[i] if record.mode == "vector" else target[i] * record.sides[i] * g["normal"]
            error = float(np.max(np.abs(g["sampled"] - offset)))
            scale = float(np.max(np.abs(g["sampled"]) + np.abs(offset)))
        errors[i], scales[i] = float(error), scale
    return errors, scales


def _sampled_potential(record, geometry, target, activation, *, initial=False):
    """Independent reported residual-energy arithmetic, separate from work.

    Rounded sparse anchors can nearly cancel their rounded targets. Comparing
    this diagnostic to the rational original-input potential by an energy-
    relative tolerance is therefore unsound. Reconstruct the declared sampled
    residual first; retain rational energy/work checks independently below.
    """
    scalar = record.mode == "distance"
    residual = np.zeros(len(record.rows) if scalar else (len(record.rows), 3))
    root_compliance = np.sqrt(record.compliance)
    for i, g in geometry.items():
        if activation[i] <= 0:
            continue
        if scalar:
            error = g["length"] - target[i]
        elif record.mode == "normal-offset":
            error = g["sampled"] - (target[i] * record.sides[i]) * g["normal"]
        else:
            error = g["sampled"] - target[i]
        if record.mode == "vector":
            residual[i] = (error * np.sqrt(activation[i] / record.compliance) if initial else
                           (np.sqrt(activation[i]) * error) / root_compliance)
        else:
            residual[i] = error / root_compliance * np.sqrt(activation[i])
    if initial and record.mode == "vector":
        residual = residual[activation > 0]
    result = float(.5 * np.sum(residual.ravel() ** 2))
    if not math.isfinite(result):
        raise ValueError("Finite sampled sewing potential required")
    return result


def _close(value, expected, label, *, scale=None, operations=128):
    observed = _number(value)
    expected = float(expected)
    scale = abs(expected) if scale is None else scale
    bound = operations * _EPS * scale + operations * _TINY
    if not math.isfinite(expected) or not math.isfinite(bound) or abs(observed - expected) > bound:
        raise ValueError("Independent sewing mismatch: " + label)


def _exact(value, expected, label):
    if _number(value) != expected:
        raise ValueError("Independent exact sewing mismatch: " + label)


def _reported_errors(values, expected, scales, label):
    if type(values) is not list or len(values) != len(expected):
        raise ValueError("Complete active/pending row errors required")
    for i, (value, error) in enumerate(zip(values, expected)):
        if error is None:
            if value is not None:
                raise ValueError("Pending sewing row error must be null")
        else:
            _close(value, error, label, scale=scales[i])


def verify_initial(record, initial, report):
    if type(report) is not dict or _json(report.get("sewingActuation")) != record.recipe_json:
        raise ValueError("Complete captured sewing recipe required in initial report")
    binding = report.get("sewingBinding")
    expected = json.loads(record.binding_json)
    if type(binding) is not dict or set(binding) != set(expected) | {"cuffSourceBinding"}:
        raise ValueError("Complete initial sewing binding report required")
    if _json({key: value for key, value in binding.items() if key != "cuffSourceBinding"}) != _json(expected):
        raise ValueError("Recorded sewing binding differs from independent source identity")
    if (not record.cuff and binding["cuffSourceBinding"] is not None
            or record.cuff and type(binding["cuffSourceBinding"]) is not dict):
        raise ValueError("Cuff source-validator evidence must retain its separate scope")
    activation, target = parameters(record, 0), targets(record, 0.)
    if not np.array_equal(_array(report.get("initialSewingActivation"), activation.shape, "initial activation"), activation):
        raise ValueError("Initial sewing activation mismatch")
    if not np.array_equal(_array(report.get("initialSewingTargetsMeters"), target.shape, "initial targets"), target):
        raise ValueError("Initial sewing targets mismatch")
    _, geometry = _geometry(record, initial, activation > 0)
    energy = float(_energy(record, geometry, target, activation))
    sampled_energy = _sampled_potential(record, geometry, target, activation, initial=True)
    if np.any(activation > 0):
        _close(report.get("initialSewingEnergyJoules"), sampled_energy, "initial sampled sewing energy")
    else:
        _exact(report.get("initialSewingEnergyJoules"), 0., "all-pending initial sewing energy")
    errors, scales = _row_errors(record, geometry, target)
    _reported_errors(report.get("initialSewingRowTargetErrorsM"), errors, scales, "initial active error")
    if report.get("initialSewingTargetErrorMetric") != METRIC:
        raise ValueError("Initial sewing errors require unweighted active-row metric")
    evidence = {"profile": PROFILE, "accepted": False, "verified": True, "sourceSha256": record.source_sha256,
        "rowCount": len(record.rows), "initialEnergyJoules": sampled_energy,
        "initialOriginalInputEnergyJoules": energy, "geometrySampler": GEOMETRY_SAMPLER,
        "sourceScope": "Canonical JSON/source rows independently bound; cuff pattern/engine rederivation is checked separately by the captured source binder"}
    if record.refined_verification_json != "null":
        evidence["refinedSourceVerification"] = json.loads(record.refined_verification_json)
    return evidence


def verify_sewing_step(record, previous, positions, start_fraction, end_fraction, old_targets, new_targets, step):
    start, end = _fraction(start_fraction), _fraction(end_fraction)
    if end <= start or type(step) is not dict:
        raise ValueError("Ordered sewing transition and complete diagnostics required")
    before_a, after_a = parameters(record, start), parameters(record, end)
    shape = (len(record.rows), 3) if record.mode == "vector" else (len(record.rows),)
    before_t, after_t = (_array(t, shape, "step sewing targets") for t in (old_targets, new_targets))
    if any(np.any(np.abs(t) > 100) or record.mode != "vector" and np.any(t <= 0) for t in (before_t, after_t)):
        raise ValueError("Bounded sewing target values required")
    observed_a = _array(step.get("sewingActivation"), after_a.shape, "reported activation")
    active, pending = np.flatnonzero(after_a > 0).tolist(), np.flatnonzero(after_a == 0).tolist()
    if (step.get("sewingMode") != record.mode or step.get("sewingActivationExplicit") is not True
            or not np.array_equal(observed_a, after_a) or step.get("sewingActivationLimitations") != LIMITATIONS
            or any(type(step.get(key)) is not list or any(type(i) is not int for i in step[key])
                   or step[key] != expected for key, expected in (("activeSewingRows", active), ("pendingSewingRows", pending)))):
        raise ValueError("Step activation or canonical active/pending identity mismatch")
    q0, g0 = _geometry(record, previous, (before_a > 0) | (after_a > 0))
    q1, g1 = _geometry(record, positions, after_a > 0)
    errors, scales = _row_errors(record, g1, after_t)
    _reported_errors(step.get("sewingRowTargetErrorsM"), errors, scales, "active target error")
    if step.get("sewingTargetErrorMetric") != METRIC:
        raise ValueError("Step errors require unweighted active-row metric")
    _close(step.get("sewingTargetErrorM"), max((e for e in errors if e is not None), default=0.),
           "maximum active target error", scale=float(np.max(scales, initial=0)))
    endpoint_before = _energy(record, g0, before_t, before_a)
    endpoint_after = _energy(record, g1, after_t, after_a)
    fixed_before_activation = _energy(record, g0, after_t, before_a)
    fixed_after_activation = _energy(record, g0, after_t, after_a)
    target_work = fixed_before_activation - endpoint_before
    activation_work = fixed_after_activation - fixed_before_activation
    parameter_work = fixed_after_activation - endpoint_before
    motion = endpoint_after - fixed_after_activation
    if record.mode == "distance":
        # Sampled endpoint norms round away tiny displacement. Reconstruct the
        # norm increment from the exact source-anchor displacement and sampled
        # endpoint lengths, then compare potentials in rational arithmetic.
        motion = Fraction()
        for i, last in g1.items():
            first = g0[i]
            delta = [b - a for a, b in zip(first["exact"], last["exact"])]
            numerator = sum((d * (2 * Fraction(float(v)) + d) for d, v in zip(delta, first["sampled"])), Fraction())
            length_delta = numerator / (Fraction(first["length"]) + Fraction(last["length"]))
            old_error = Fraction(first["length"]) - Fraction(float(after_t[i]))
            changed_error = old_error + length_delta
            motion += Fraction(float(after_a[i])) * (changed_error ** 2 - old_error ** 2) / (2 * Fraction(record.compliance))
    # The public schedule forbids release, but retained accounting still states
    # the signed activation term and its nonnegative positive/negative parts.
    expected_energy = {"sewingBeforeJoules": float(endpoint_before), "sewingAfterJoules": float(endpoint_after),
        "sewingFixedPositionAfterJoules": float(fixed_after_activation), "sewingFixedParameterChangeJoules": float(motion),
        "sewingChangeJoules": float(parameter_work + motion), "sewingTargetParameterWorkJoules": float(target_work),
        "sewingActivationParameterWorkJoules": float(activation_work), "sewingParameterWorkJoules": float(parameter_work),
        "sewingActivationIncreaseWorkJoules": float(activation_work), "sewingReleaseEnergyRemovedJoules": 0.}
    rounded_sum = math.fsum((float(target_work), float(activation_work)))
    expected_energy["sewingParameterWorkComponentSumErrorBoundJoules"] = math.fsum(
        math.ulp(expected_energy[key]) for key in ("sewingParameterWorkJoules", "sewingTargetParameterWorkJoules",
                                                   "sewingActivationParameterWorkJoules")) + math.ulp(rounded_sum)
    energy_report = step.get("energyBalance")
    if type(energy_report) is not dict:
        raise ValueError("Captured sewing step requires complete energy accounting")
    for key, value in expected_energy.items():
        _exact(energy_report.get(key), value, key)
    sampled_energy = _sampled_potential(record, g1, after_t, after_a)
    if active:
        _close(step.get("sewingJoules"), sampled_energy, "sampled sewing potential")
    else:
        _exact(step.get("sewingJoules"), 0., "all-pending sewing potential")
    nodal = [[Fraction()] * 3 for _ in range(record.vertex_count)]
    for i, g in g1.items():
        stiffness = Fraction(float(after_a[i])) / Fraction(record.compliance)
        error = _error(record, i, g, after_t[i])
        if record.mode == "distance":
            anchor_gradient = [stiffness * error[0] * Fraction(float(v)) / Fraction(g["length"]) for v in g["sampled"]]
        else:
            anchor_gradient = [stiffness * e for e in error]
        for vertex, coefficient in record.rows[i]:
            for axis in range(3):
                nodal[vertex][axis] += Fraction(coefficient) * anchor_gradient[axis]
        if record.mode == "normal-offset":
            normal = [Fraction(float(x)) for x in g["normal"]]
            dot = sum((a * b for a, b in zip(error, normal)), Fraction())
            factor = -stiffness * Fraction(float(after_t[i])) * record.sides[i] / Fraction(g["magnitude"])
            area_gradient = [factor * (e - dot * n) for e, n in zip(error, normal)]
            first = _cross([Fraction(float(x)) for x in g["second"]], area_gradient)
            second = _cross(area_gradient, [Fraction(float(x)) for x in g["first"]])
            reactions = [[-a - b for a, b in zip(first, second)], first, second]
            for vertex, reaction in zip(record.frames[i], reactions):
                for axis in range(3):
                    nodal[vertex][axis] += reaction[axis]
    gradient = np.array([[float(value) for value in row] for row in nodal])
    if not np.isfinite(gradient).all():
        raise ValueError("Finite independent sewing force required")
    return gradient, {"profile": PROFILE, "accepted": False, "verified": True, "sourceSha256": record.source_sha256,
        "rowCount": len(record.rows), "activeRows": active, "pendingRows": pending,
        "geometrySampler": GEOMETRY_SAMPLER, "sampledSewingJoules": sampled_energy, **expected_energy,
        "scope": "Independent canonical-row controls, weighted sewing forces, frame reactions and discrete target-first work. Distance/normal geometry uses declared binary64 samples. Cuff source-engine proof remains separate; no construction or continuous-seam acceptance."}
