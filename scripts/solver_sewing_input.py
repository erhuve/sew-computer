"""Bind explicit sewing controls to complete captured canonical source identity.

Generic inputs establish captured-JSON correspondence only. The dedicated cuff
profile additionally requires rederivation by the current source verifier.
Neither binding executes a construction phase or proves a textile right side.
"""

import copy
from fractions import Fraction
import hashlib
import json
import math

import numpy as np

from solver_sewing_activation_schedule import MAX_ROWS, SewingActivationSchedule


PROFILE = "captured-sewing-activation-v1"
CUFF_PROFILE = "source-cuff-construction-unit-v1"
REFINED_PROFILE = "source-left-binding-refined-unit-v1"
MAX_SOURCE_BYTES = 8 * 1024 ** 2
_CUFF_FIELDS = {"sourcePattern", "sourceConstruction", "sourceInventory", "sourceAssembly",
                "phasePlan", "phaseConstraintRows", "selectedOperationIds", "excludedOperationIds",
                "unexecutedOtherOperationsTouchingUnit", "unexecutedClosuresTouchingUnit"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _bounded_json(value):
    # Bound traversal before encoding/copying caller-controlled nested payloads.
    remaining = 1000000
    text_bytes = 0

    def visit(item, depth):
        nonlocal remaining, text_bytes
        remaining -= 1
        if remaining < 0 or depth > 40:
            raise ValueError("Captured sewing source exceeds bounded JSON structure")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Captured sewing source integer exceeds JSON bound")
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Captured sewing source requires finite JSON numbers")
            return
        if type(item) is str:
            if len(item) > MAX_SOURCE_BYTES:
                raise ValueError("Captured sewing source string exceeds byte bound")
            try:
                text_bytes += len(item.encode())
            except UnicodeEncodeError as error:
                raise ValueError("Captured sewing source requires valid UTF-8 strings") from error
            if text_bytes > MAX_SOURCE_BYTES:
                raise ValueError("Captured sewing source exceeds byte bound")
            return
        if type(item) is list:
            if len(item) > remaining:
                raise ValueError("Captured sewing source exceeds bounded JSON structure")
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict:
            if len(item) * 2 > remaining or any(type(key) is not str for key in item):
                raise ValueError("Captured sewing source requires bounded string-keyed JSON objects")
            for key, child in item.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        raise ValueError("Captured sewing source must contain raw JSON values only")

    visit(value, 0)
    if len(_encoded(value)) > MAX_SOURCE_BYTES:
        raise ValueError("Captured sewing source exceeds byte bound")


def sewing_source_identity(source):
    """Hash canonical entire source excluding only its non-self-referential recipe."""
    if type(source) is not dict:
        raise ValueError("Captured sewing source must be an object")
    _bounded_json(source)
    return hashlib.sha256(_encoded({key: value for key, value in source.items()
                                   if key != "sewingActuation"})).hexdigest()


def _identity(value):
    return type(value) is str and 1 <= len(value) <= 160 and all(ord(char) >= 32 for char in value)


def _number(value, lower, upper, label, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not lower <= value <= upper or positive and value <= 0):
        raise ValueError("Invalid " + label)
    return float(value)


def _mesh(source):
    rest, raw_faces, offsets = source.get("restMeters"), source.get("triangles"), source.get("instanceOffsets")
    if (type(rest) is not list or not 3 <= len(rest) <= 25000
            or any(type(row) is not list or len(row) != 3 for row in rest)):
        raise ValueError("Bounded canonical sewing rest positions required")
    for row in rest:
        for value in row:
            _number(value, -100, 100, "canonical sewing rest coordinate")
    if type(raw_faces) is not list or not raw_faces:
        raise ValueError("Bounded canonical sewing triangles required")
    if type(raw_faces[0]) is list:
        if len(raw_faces) > 50000 or any(type(face) is not list or len(face) != 3 for face in raw_faces):
            raise ValueError("Canonical sewing triangles require exactly three vertices")
        faces = raw_faces
    else:
        if len(raw_faces) > 150000 or len(raw_faces) % 3:
            raise ValueError("Canonical sewing triangles require exactly three vertices")
        faces = [raw_faces[index:index + 3] for index in range(0, len(raw_faces), 3)]
    if any(any(type(vertex) is not int or not 0 <= vertex < len(rest) for vertex in face)
           or len(set(face)) != 3 for face in faces):
        raise ValueError("Canonical sewing triangle vertices must be distinct integer indices")
    if len({tuple(sorted(face)) for face in faces}) != len(faces):
        raise ValueError("Duplicate canonical sewing triangle")
    if (type(offsets) is not dict or not 1 <= len(offsets) <= 64
            or any(not _identity(identity) or type(offset) is not int or not 0 <= offset < len(rest)
                   for identity, offset in offsets.items())):
        raise ValueError("Bounded physical sewing instance offsets required")
    ordered = sorted(offsets.items(), key=lambda item: item[1])
    if ordered[0][1] != 0 or len({offset for _, offset in ordered}) != len(ordered):
        raise ValueError("Physical sewing instances must partition canonical vertices")
    counts, owners = {}, []
    for index, (identity, offset) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(rest)
        if end - offset < 3:
            raise ValueError("Every sewing instance requires at least three vertices")
        counts[identity] = end - offset
        owners.extend([identity] * (end - offset))
    adjacent = [[] for _ in rest]
    for index, face in enumerate(faces):
        if len({owners[vertex] for vertex in face}) != 1:
            raise ValueError("Sewing triangles cannot cross physical instances")
        for vertex in face:
            adjacent[vertex].append(index)
    if any(not items for items in adjacent):
        raise ValueError("Every canonical sewing vertex must belong to its cloth triangles")
    return faces, offsets, counts, owners, adjacent


def _rows(source, mesh):
    faces, offsets, counts, owners, adjacent = mesh
    bundle = source.get("embeddedConstraints")
    constraints = bundle.get("constraints") if type(bundle) is dict else None
    if type(constraints) is not list or not 1 <= len(constraints) <= MAX_ROWS:
        raise ValueError("Complete bounded nonempty canonical sewing rows required")
    rows, bindings, identities = [], [], []
    completed_registrations, previous_registration, previous_order = set(), None, None
    compliance = None
    for index, constraint in enumerate(constraints):
        if type(constraint) is not dict:
            raise ValueError("Canonical sewing row must be an object")
        registration, member, fraction = (constraint.get(key) for key in ("registrationId", "memberIndex", "fraction"))
        if not _identity(registration) or type(member) is not int or not 1 <= member <= 7:
            raise ValueError("Explicit bounded sewing registration and star member identity required")
        _number(fraction, 0, 1, "source sewing fraction")
        fraction = Fraction(fraction)
        order = fraction, member
        if registration != previous_registration:
            if registration in completed_registrations:
                raise ValueError("Canonical sewing registrations must retain contiguous source order")
            completed_registrations.add(registration)
            previous_order = None
        if previous_order is not None and order <= previous_order:
            raise ValueError("Canonical sewing fractions and star members must retain strict source order")
        previous_registration, previous_order = registration, order
        row_id = "row:" + hashlib.sha256(_encoded([registration, member, fraction.numerator, fraction.denominator])).hexdigest()
        if row_id in identities:
            raise ValueError("Duplicate canonical sewing row identity")
        value = _number(constraint.get("complianceMPerN"), 0, 1000, "positive sewing compliance", positive=True)
        if compliance is not None and value != compliance:
            raise ValueError("Captured sewing requires one unchanged positive compliance for all rows")
        compliance = value
        terms = constraint.get("terms")
        if type(terms) is not list or not 2 <= len(terms) <= 6:
            raise ValueError("Bounded nonempty positive and negative material anchor terms required")
        row, per_sign, term_keys = {}, {1: [], -1: []}, set()
        for term in terms:
            if type(term) is not dict or set(term) != {"instanceId", "vertex", "coefficient"}:
                raise ValueError("Canonical sewing terms require only instance, vertex and coefficient")
            identity, vertex = term["instanceId"], term["vertex"]
            if (type(identity) is not str or identity not in counts or type(vertex) is not int
                    or not 0 <= vertex < counts[identity] or (identity, vertex) in term_keys):
                raise ValueError("Unknown or duplicate canonical sewing support")
            coefficient = _number(term["coefficient"], -1, 1, "sewing coefficient")
            if coefficient == 0:
                raise ValueError("Sparse sewing source terms must be nonzero")
            term_keys.add((identity, vertex))
            row[offsets[identity] + vertex] = coefficient
            per_sign[1 if coefficient > 0 else -1].append((identity, vertex, coefficient))
        anchor_instances = []
        for sign in (1, -1):
            anchor = per_sign[sign]
            if (not 1 <= len(anchor) <= 3 or len({item[0] for item in anchor}) != 1
                    or abs(math.fsum(item[2] for item in anchor) - sign) > 1e-12):
                raise ValueError("Each source sewing anchor must be normalized within one physical instance")
            identity = anchor[0][0]
            anchor_instances.append(identity)
            support = {offsets[identity] + item[1] for item in anchor}
            if not any(support.issubset(faces[face]) for face in adjacent[min(support)]):
                raise ValueError("Every source sewing anchor support must lie on a canonical triangle")
        if anchor_instances[0] == anchor_instances[1]:
            raise ValueError("Sewing anchors must name two distinct physical instances")
        if "sourceSamples" in constraint:
            samples = constraint["sourceSamples"]
            if type(samples) is not list or len(samples) != 2:
                raise ValueError("Paired source sewing samples required")
            for sign, sample, identity in zip((1, -1), samples, anchor_instances):
                if type(sample) is not dict or sample.get("instanceId") != identity:
                    raise ValueError("Source sewing samples must retain signed instance order")
                weights = sample.get("weights")
                if type(weights) is not list or not 1 <= len(weights) <= 3:
                    raise ValueError("Bounded source sewing sample weights required")
                expected, seen = {}, set()
                for weight in weights:
                    if (type(weight) is not dict or set(weight) != {"vertex", "weight"}
                            or type(weight["vertex"]) is not int or not 0 <= weight["vertex"] < counts[identity]
                            or weight["vertex"] in seen):
                        raise ValueError("Distinct source sewing sample vertices required")
                    seen.add(weight["vertex"])
                    amount = _number(weight["weight"], 0, 1, "source sewing sample weight")
                    if amount:
                        expected[weight["vertex"]] = sign * amount
                if expected != {vertex: coefficient for _, vertex, coefficient in per_sign[sign]}:
                    raise ValueError("Sewing sparse coefficients differ from declared source sample weights")
        identities.append(row_id)
        rows.append(row)
        bindings.append({"rowIndex": index, "rowId": row_id, "registrationId": registration,
                         "memberIndex": member, "fractionNumerator": fraction.numerator,
                         "fractionDenominator": fraction.denominator,
                         "positiveInstanceId": anchor_instances[0], "negativeInstanceId": anchor_instances[1],
                         "canonicalTerms": [{"vertex": vertex, "coefficient": coefficient} for vertex, coefficient in row.items()],
                         "sourceRowSha256": hashlib.sha256(_encoded(constraint)).hexdigest()})
    return rows, bindings, tuple(identities), compliance


def derive_sewing_row_ids(source):
    """Derive complete ordered row IDs for authoring a captured recipe."""
    sewing_source_identity(source)
    return _rows(source, _mesh(source))[2]


def _targets(value, count, mode):
    if type(value) is not list or len(value) != count:
        raise ValueError("Explicit sewing target for every canonical row required")
    if mode == "vector":
        if any(type(row) is not list or len(row) != 3 for row in value):
            raise ValueError("Vector sewing targets require one three-vector per canonical row")
        for row in value:
            for component in row:
                _number(component, -100, 100, "explicit vector sewing target")
    else:
        for item in value:
            _number(item, 0, 100, "explicit positive scalar sewing target", positive=True)
    return np.asarray(value, dtype=float)


def _frames(source, rows, bindings, row_ids, mesh):
    faces, offsets, _, owners, _ = mesh
    frames = source.get("sewingFrames")
    if (type(frames) is not dict or set(frames) != {"faces", "sides", "bindings"}
            or any(type(frames[key]) is not list or len(frames[key]) != len(rows) for key in frames)):
        raise ValueError("Explicit complete canonical normal-frame faces, sides and row bindings required")
    result = []
    for index, (row, declaration) in enumerate(zip(rows, frames["bindings"])):
        face, side = frames["faces"][index], frames["sides"][index]
        if (type(declaration) is not dict or set(declaration) != {"rowId", "instanceId", "triangleIndex", "side"}
                or declaration["rowId"] != row_ids[index] or type(declaration["triangleIndex"]) is not int
                or not 0 <= declaration["triangleIndex"] < len(faces)
                or type(side) is not int or side not in (-1, 1)
                or type(declaration["side"]) is not int or declaration["side"] != side
                or type(face) is not list or len(face) != 3 or any(type(vertex) is not int for vertex in face)
                or face != faces[declaration["triangleIndex"]]):
            raise ValueError("Normal frame must preserve explicit canonical triangle winding, row and side")
        identity = declaration["instanceId"]
        negative = {vertex for vertex, coefficient in row.items() if coefficient < 0}
        positive = {vertex for vertex, coefficient in row.items() if coefficient > 0}
        if (identity != bindings[index]["negativeInstanceId"] or any(owners[vertex] != identity for vertex in face)
                or not negative.issubset(face) or positive.intersection(face)):
            raise ValueError("Declared normal frame must contain only its negative material anchor support")
        result.append({**copy.deepcopy(declaration), "canonicalVertices": face.copy(),
                       "instanceLocalVertices": [vertex - offsets[identity] for vertex in face]})
    return np.asarray(frames["faces"], dtype=int), np.asarray(frames["sides"], dtype=int), result


class CapturedSewingControls:
    """Immutable captured targets/topology; activation samples remain stateless."""

    __slots__ = ("_row_ids", "_initial", "_final", "_compliance", "_mode", "_schedule",
                 "_schedule_bytes", "_rows", "_faces", "_sides")

    def __setattr__(self, name, value):
        raise AttributeError("Captured sewing controls are immutable")

    def __delattr__(self, name):
        raise AttributeError("Captured sewing controls are immutable")

    def __init__(self, row_ids, initial, final, compliance, mode, schedule, recipe, rows, faces, sides):
        def freeze(array):
            if array is None:
                return None
            return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        for name, value in (("_row_ids", row_ids), ("_initial", freeze(initial)), ("_final", freeze(final)),
                            ("_compliance", compliance), ("_mode", mode), ("_schedule", schedule),
                            ("_schedule_bytes", _encoded(recipe)), ("_rows", tuple(tuple(row.items()) for row in rows)),
                            ("_faces", freeze(faces)), ("_sides", freeze(sides))):
            object.__setattr__(self, name, value)

    row_ids = property(lambda self: self._row_ids)
    initial_targets = property(lambda self: self._initial.copy())
    final_targets = property(lambda self: self._final.copy())
    compliance = property(lambda self: self._compliance)
    mode = property(lambda self: self._mode)
    activation_schedule = property(lambda self: self._schedule)
    schedule_recipe = property(lambda self: json.loads(self._schedule_bytes))
    rows = property(lambda self: [dict(row) for row in self._rows])
    frame_faces = property(lambda self: None if self._faces is None else self._faces.copy())
    sides = property(lambda self: None if self._sides is None else self._sides.copy())

    def parameters(self, fraction):
        return self._schedule.parameters(fraction)


def bind_sewing_activation(source, subdivisions, *, sewing_mode):
    source_digest = sewing_source_identity(source)
    profile = source.get("profile")
    refined_claim = (type(profile) is str and profile.startswith("source-left-binding-")
                     or bool({"baseUnit", "bindingRefinement", "bindingSeamRemap"}.intersection(source)))
    refined = profile == REFINED_PROFILE
    if refined_claim and not refined:
        raise ValueError("Refined cuff source cannot downgrade to another source profile")
    cuff = profile == CUFF_PROFILE or refined
    provenance = source.get("provenance")
    cuff_provenance = type(provenance) is dict and "phasePlanSha256" in provenance
    if not cuff and (_CUFF_FIELDS.intersection(source) or cuff_provenance):
        raise ValueError("Cuff construction source cannot downgrade to generic sewing binding")
    if type(sewing_mode) is not str or sewing_mode not in ("vector", "distance", "normal-offset"):
        raise ValueError("Explicit supported sewing mode required")
    if refined and sewing_mode == "normal-offset":
        raise ValueError("Refined cuff normal-offset sewing requires an unimplemented explicit crease-side frame policy")
    recipe = source.get("sewingActuation")
    if (type(recipe) is not dict or set(recipe) != {"profile", "accepted", "sourceSha256", "mode",
                                                  "initialTargetsMeters", "finalTargetsMeters", "schedule"}
            or recipe["profile"] != PROFILE or recipe["accepted"] is not False
            or recipe["sourceSha256"] != source_digest or recipe["mode"] != sewing_mode):
        raise ValueError("Explicit unaccepted sewing recipe must match complete captured source and mode")
    mesh = _mesh(source)
    rows, row_bindings, row_ids, compliance = _rows(source, mesh)
    initial = _targets(recipe["initialTargetsMeters"], len(rows), sewing_mode)
    final = _targets(recipe["finalTargetsMeters"], len(rows), sewing_mode)
    schedule = SewingActivationSchedule(recipe["schedule"], subdivisions, row_ids=row_ids)
    cuff_binding = None
    if cuff:
        if refined:
            from solver_binding_source import validate_binding_source
            cuff_binding = validate_binding_source(source)
        else:
            from solver_cuff_source_binding import validate_cuff_source_binding
            cuff_binding = validate_cuff_source_binding(source)
        selectors = {}
        for index, row in enumerate(row_bindings):
            selectors.setdefault((row["registrationId"], row["memberIndex"]), []).append(index)
        if len(row_bindings) != 40 or len(selectors) != 8:
            raise ValueError("Cuff controls must retain all 40 rows and eight selectors")
        for indices in selectors.values():
            if (len(indices) != 5 or [Fraction(row_bindings[index]["fractionNumerator"],
                                             row_bindings[index]["fractionDenominator"]) for index in indices]
                    != [Fraction(index, 4) for index in range(5)]):
                raise ValueError("Cuff selectors require all five original source sample rows")
            for knot in recipe["schedule"]["knots"]:
                if len({knot["activation"][index] for index in indices}) != 1:
                    raise ValueError("Every cuff selector's five sample rows must share activation at every knot")
    frame_faces, sides, frame_bindings = None, None, []
    if sewing_mode == "normal-offset":
        frame_faces, sides, frame_bindings = _frames(source, rows, row_bindings, row_ids, mesh)
        if cuff:
            instances = {item["id"]: item for item in source["instances"]}
            for binding in frame_bindings:
                template = instances[binding["instanceId"]]["templateId"]
                triangles = source["sourceTemplates"][template]["triangles"]
                matches = [index for index, face in enumerate(triangles) if face == binding["instanceLocalVertices"]]
                if len(matches) != 1:
                    raise ValueError("Validated cuff normal frame lacks unique oriented source template triangle")
                binding.update(templateId=template, sourceTriangleIndex=matches[0])
    controls = CapturedSewingControls(row_ids, initial, final, compliance, sewing_mode, schedule,
                                     recipe["schedule"], rows, frame_faces, sides)
    manifest = {"profile": PROFILE, "accepted": False, "sourceSha256": source_digest,
                "mode": sewing_mode, "rowIds": list(row_ids), "rowBindings": row_bindings,
                "complianceMPerN": compliance, "sourceBundleSha256": hashlib.sha256(_encoded(source["embeddedConstraints"])).hexdigest(),
                "frameBindings": frame_bindings, "sourceFrameMetadataUsed": sewing_mode == "normal-offset",
                "sourceBindingScope": ("rederived refined cuff source with recorded coefficient approximation" if refined else
                                       "rederived cuff construction source" if cuff else "captured canonical JSON only; no pattern-source proof"),
                "cuffSourceBinding": cuff_binding,
                "constructionStatus": "Source phase plan remains declarative and unexecuted; numerical knot times do not establish binding, turning, gate completion or any construction milestone.",
                "scope": "Explicit virtual seam controls preserve all canonical source rows. Activation is force weighting, not executed construction or completed seam evidence. Normal sides are declared mechanical frame offsets, not textile right-side assignments."}
    return controls, manifest
