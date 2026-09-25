"""Independent retained-prefix evidence checks for bounded research audits.

This module imports no mechanics or producer validators. Conditional temporal
accounting is not constitutive/contact/path validation, process success, source
garment admission, or a resume protocol. Large decode/exact-reduction operations
must run under separately declared process memory/CPU/wall budgets.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct


PROFILE = "retained-temporal-prefix-audit-v1"
TRANSPORT = "conditional-temporal-process-transport-v1"
RETAINED = "conditional-temporal-retained-run-v1"
CONTROLLER = "conditional-step-doubling-v1"
EVALUATOR = "controller-callback-binary64-le-v1"

# These are public wire fields, deliberately not imported from the producer.
MOTION = ("membraneChangeJoules", "bendingChangeJoules", "foldBarrierChangeJoules",
          "contactChangeJoules", "sewingFixedParameterChangeJoules",
          "foldFixedParameterChangeJoules", "gripperFixedParameterChangeJoules",
          "cableFixedParameterChangeJoules")
PUBLIC_ENERGY = (
    "membraneChangeJoules", "bendingChangeJoules", "bendingBeforeJoules", "bendingAfterJoules",
    "foldBarrierChangeJoules", "foldBarrierBeforeJoules", "foldBarrierAfterJoules",
    "contactChangeJoules", "contactBeforeJoules", "contactAfterJoules", "kineticChangeJoules",
    "sewingChangeJoules", "sewingBeforeJoules", "sewingAfterJoules",
    "foldActuationChangeJoules", "foldActuationBeforeJoules", "foldActuationAfterJoules",
    "foldTargetParameterWorkJoules", "gripperBeforeJoules", "gripperAfterJoules",
    "gripperFixedPositionAfterJoules", "gripperFixedParameterChangeJoules", "gripperChangeJoules",
    "gripperParameterWorkJoules", "gripperTargetParameterWorkJoules", "gripperActivationParameterWorkJoules",
    "gripperActivationIncreaseWorkJoules", "gripperReleaseEnergyRemovedJoules",
    "gripperParameterWorkComponentSumErrorBoundJoules", "targetParameterWorkJoules", "externalParameterWorkJoules",
    "mechanicalChangeJoules", "mechanicalChangeMinusTargetWorkJoules", "mechanicalChangeMinusParameterWorkJoules")
ADAPTIVE_FIELDS = (
    "stationarityToleranceN", "sewingMode", "initialTargets", "targets", "initialFoldTargets", "foldTargets",
    "assemblySchedule", "defaultProgress", "sewingSchedule", "gripperSchedule", "foldSchedule",
    "fixedCableControl", "varyingCableControl", "cableParameterSchedule", "cableParameterPreflight",
    "boundedContactControl", "temporalSchedulePreflight", "stepOptions")
CONTACT_SCOPE = (
    "Bounded declared contact scalar work on complete trusted native endpoint inventories; "
    "exact binary-input geometry and captured native-rounded constants. Native energy observations, "
    "gradient/Hessian arithmetic, other constitutive work, source authenticity and physical "
    "contact are not certified. No path, garment or temporal-accuracy acceptance.")


@dataclass(frozen=True)
class AuditLimits:
    input_bytes: int
    nesting_depth: int
    json_values: int
    string_characters: int
    vertices: int
    trials: int
    assessments: int

    def __post_init__(self):
        for value, maximum in ((self.input_bytes, 2**31), (self.nesting_depth, 128),
                               (self.json_values, 100_000_000), (self.string_characters, 2**20),
                               (self.vertices, 1_000_000), (self.trials, 4096),
                               (self.assessments, 4096)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Explicit bounded audit limits required")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_text(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Canonical SHA256 required")
    return value


def integer(value, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum),
            "Bounded exact integer required")
    return value


def number(value, minimum=None):
    require(type(value) is float and math.isfinite(value) and (minimum is None or value >= minimum),
            "Finite binary64 value required")
    return F(value)


def flag(value):
    require(type(value) is bool, "Exact Boolean required")
    return value


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected), "Exact record fields required")


def rational(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def fraction(value, *, nonnegative=False):
    keys(value, ("numerator", "denominator"))
    for item in value.values():
        require(type(item) is str and 1 <= len(item) <= 4096, "Bounded rational strings required")
    try:
        result = F(int(value["numerator"]), int(value["denominator"]))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Valid rational required") from error
    require(rational(result) == value and (not nonnegative or result >= 0), "Canonical rational required")
    return result


def same(first, second):
    """Typed tree equality, including signed-zero bits; inputs are bounded trees."""
    pending = [(first, second)]
    while pending:
        a, b = pending.pop()
        require(type(a) is type(b), "Evidence value types differ")
        if type(a) is dict:
            require(a.keys() == b.keys(), "Evidence fields differ")
            pending.extend((a[key], b[key]) for key in a)
        elif type(a) is list:
            require(len(a) == len(b), "Evidence array lengths differ")
            pending.extend(zip(a, b))
        elif type(a) is float:
            require(a.hex() == b.hex(), "Evidence binary64 bits differ")
        else:
            require(a == b, "Evidence values differ")


def canonical_identity(value, *, newline=False):
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(",", ":"), check_circular=True)
    digest, count = hashlib.sha256(), 0
    for text in encoder.iterencode(value):
        content = text.encode("ascii")
        digest.update(content); count += len(content)
    if newline:
        digest.update(b"\n"); count += 1
    return {"bytes": count, "sha256": digest.hexdigest()}


def bounded_tree(value, limits):
    """Post-decode shape limits; process limits must bound decode allocation too."""
    pending, active, count = [(value, 0, False)], set(), 0
    while pending:
        item, depth, leaving = pending.pop()
        if leaving:
            active.remove(id(item))
            continue
        count += 1
        require(count <= limits.json_values and depth+int(type(item) in (dict, list)) <= limits.nesting_depth,
                "JSON value/depth budget exceeded")
        if type(item) in (dict, list):
            require(id(item) not in active, "Cyclic evidence is not JSON")
            active.add(id(item)); pending.append((item, depth, True))
            if type(item) is dict:
                require(all(type(key) is str for key in item), "String JSON keys required")
                for key, part in item.items():
                    pending.extend(((key, depth+1, False), (part, depth+1, False)))
            else:
                pending.extend((part, depth+1, False) for part in item)
        elif type(item) is str:
            require(len(item) <= limits.string_characters, "JSON string budget exceeded")
        elif type(item) is float:
            require(math.isfinite(item), "Nonfinite JSON value")
        else:
            require(item is None or type(item) in (bool, int), "Unsupported JSON value")
            if type(item) is int:
                require(item.bit_length() <= 13607, "JSON integer budget exceeded")
    return count


def read_snapshot(path, *, expected_bytes, expected_sha256, limits):
    """Read exact canonical snapshot bytes; no mechanical/numerical admission."""
    require(type(limits) is AuditLimits, "Exact audit policy required")
    limits.__post_init__()
    integer(expected_bytes, 1, limits.input_bytes); digest_text(expected_sha256)
    path = Path(path)
    require(path.name == "snapshot.json", "Only a final snapshot basename is readable")
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size == expected_bytes,
                "Expected bounded regular snapshot required")
        raw, digest = bytearray(), hashlib.sha256()
        depth = 0
        quoted = escaped = False
        while True:
            chunk = os.read(fd, min(65536, expected_bytes-len(raw)+1))
            if not chunk: break
            require(len(raw)+len(chunk) <= expected_bytes, "Snapshot grew beyond declared bytes")
            digest.update(chunk); raw.extend(chunk)
            # Producer canonical JSON is ASCII. This bounds recursive nesting
            # before object decoding, including strings spanning read chunks.
            for byte in chunk:
                require(byte < 128, "Canonical ASCII JSON required")
                if quoted:
                    if escaped: escaped = False
                    elif byte == 92: escaped = True
                    elif byte == 34: quoted = False
                elif byte == 34:
                    quoted = True
                elif byte in (91, 123):
                    depth += 1
                    require(depth <= limits.nesting_depth, "JSON nesting budget exceeded")
                elif byte in (93, 125):
                    depth -= 1
                    require(depth >= 0, "Malformed JSON nesting")
        after = os.fstat(fd)
        named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(all(getattr(before, key) == getattr(after, key) == getattr(named, key) for key in fields),
                "Snapshot changed during read")
        require(len(raw) == expected_bytes and digest.hexdigest() == expected_sha256,
                "Snapshot identity differs from external expectation")
    finally:
        try:
            if fd is not None: os.close(fd)
        finally:
            os.close(parent)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def parse_float(text):
        value = float(text)
        require(math.isfinite(value), "JSON floating point overflow")
        return value

    def parse_constant(text):
        raise ValueError("Nonfinite JSON constant")

    value = json.loads(raw, object_pairs_hook=pairs, parse_float=parse_float, parse_constant=parse_constant)
    bounded_tree(value, limits)
    same(canonical_identity(value, newline=True), {"bytes": expected_bytes, "sha256": expected_sha256})
    return value


def state(value, limits, count=None):
    """Recompute the declared native little-endian, row-major binary64 hash."""
    keys(value, ("sha256", "positionsMeters", "velocitiesMPerS"))
    digest_text(value["sha256"])
    q, v = value["positionsMeters"], value["velocitiesMPerS"]
    require(type(q) is list and type(v) is list, "State arrays required")
    integer(len(q), 1, limits.vertices)
    require(len(v) == len(q) and (count is None or len(q) == count), "State shape changed")
    digest = hashlib.sha256(f"({len(q)}, 3)".encode("ascii")+b"\0")
    for array in (q, v):
        for row in array:
            require(type(row) is list and len(row) == 3, "Three state components required")
            for item in row:
                number(item)
                digest.update(struct.pack("<d", item))
    require(digest.hexdigest() == value["sha256"], "State bits do not match hash")
    return value


def context(value, limits):
    keys(value, ("initialState", "massKg", "requestedDurationSeconds", "policy", "maxDepth",
                 "maxAttempts", "initialSubdivisions", "evaluationLimit", "adaptiveContext"))
    initial = state(value["initialState"], limits)
    require(type(value["massKg"]) is list and len(value["massKg"]) == len(initial["positionsMeters"]),
            "One mass per vertex required")
    require(all(number(item) > 0 for item in value["massKg"]), "Positive masses required")
    require(number(value["requestedDurationSeconds"]) > 0, "Positive duration required")
    policy = value["policy"]
    keys(policy, ("profile", "positionToleranceM", "velocityToleranceMPerS", "numericalEnergyBudgetJ",
                  "maxEvaluationBudget"))
    require(policy["profile"] == CONTROLLER, "Unsupported temporal controller")
    for name in ("positionToleranceM", "velocityToleranceMPerS", "numericalEnergyBudgetJ"):
        require(number(policy[name]) > 0, "Positive temporal thresholds required")
    integer(policy["maxEvaluationBudget"], 1, 4096*10000)
    depth = integer(value["maxDepth"], 1, 30)
    attempts = integer(value["maxAttempts"], 1, min(4096, limits.trials))
    divisions = integer(value["initialSubdivisions"], 1, attempts)
    require(divisions & (divisions-1) == 0 and divisions.bit_length()-1+depth <= 40,
            "Supported dyadic scheduler domain required")
    integer(value["evaluationLimit"], 1, 10000)
    adaptive = value["adaptiveContext"]
    if adaptive is not None:
        # Identity binds the complete context externally. This profile checks
        # presence of work controls, not their interpolation or physical law.
        keys(adaptive, ADAPTIVE_FIELDS)
        require(number(adaptive["stationarityToleranceN"]) > 0, "Positive stationarity criterion required")
        require(type(adaptive["sewingMode"]) is str, "Declared sewing mode required")
        require(adaptive["fixedCableControl"] is None or adaptive["varyingCableControl"] is None,
                "Fixed and varying cable modes are mutually exclusive")
        for name in ("fixedCableControl", "varyingCableControl", "boundedContactControl"):
            require(adaptive[name] is None or type(adaptive[name]) is dict, "Control definition required")
    return value


def metric(first, second, tolerance):
    squares = [sum(((F(a)-F(b))**2 for a, b in zip(left, right)), F())
               for left, right in zip(first, second)]
    maximum = max(squares)
    threshold = F(tolerance)**2
    # The display square root has no authority over the exact comparison.
    largest = F(float.fromhex("0x1.fffffffffffffp+1023"))
    approximate = math.sqrt(float(maximum)) if maximum <= largest else None
    return {"maximumSquared": rational(maximum), "toleranceSquared": rational(threshold),
            "maximumApproximate": approximate, "maximumVertex": squares.index(maximum),
            "withinThreshold": maximum <= threshold}


def _contact_radius(energy):
    if "boundedContactWork" not in energy:
        require("boundedContactMechanical" not in energy, "Missing contact work")
        return F()
    work = energy["boundedContactWork"]
    keys(work, ("profile", "definition", "changeJoules", "changeErrorBoundJoules", "start", "end",
                "unionTerms", "logCalls", "logTerms", "scope", "accepted"))
    require(work["profile"] == "bounded-native-contact-work-record-v1" and
            work["scope"] == CONTACT_SCOPE and work["accepted"] is False, "Contact work profile mismatch")
    definition = work["definition"]
    keys(definition, ("profile", "scalarProfile", "nativeIdentitySha256", "vertexCount", "policy", "scope"))
    require(definition["profile"] == "guarded-native-contact-work-control-v1" and
            definition["scalarProfile"] == "captured-ipc-contact-work-v1" and
            definition["scope"] == CONTACT_SCOPE, "Contact definition mismatch")
    integer(definition["vertexCount"], 3)
    digest_text(definition["nativeIdentitySha256"])
    policy = definition["policy"]
    bounds = {"bits": (64, 512), "max_log_terms": (1, 512), "max_total_log_terms": (1, 1000000),
              "max_fraction_bits": (4096, 131072), "max_endpoint_terms": (1, 100000)}
    keys(policy, bounds)
    for name, (lo, hi) in bounds.items(): integer(policy[name], lo, hi)
    counts = []
    for label, public in (("start", "contactBeforeJoules"), ("end", "contactAfterJoules")):
        endpoint = work[label]
        keys(endpoint, ("positionsSha256", "inventorySha256", "termCount", "nativeEnergyJoules"))
        digest_text(endpoint["positionsSha256"]); digest_text(endpoint["inventorySha256"])
        counts.append(integer(endpoint["termCount"], 0, policy["max_endpoint_terms"]))
        number(endpoint["nativeEnergyJoules"], 0)
        same(endpoint["nativeEnergyJoules"], energy[public])
    number(work["changeJoules"]); same(work["changeJoules"], energy["contactChangeJoules"])
    radius = number(work["changeErrorBoundJoules"], 0)
    require(sum(counts) <= policy["max_endpoint_terms"], "Contact endpoint budget exceeded")
    union = integer(work["unionTerms"], max(counts), sum(counts))
    integer(work["logCalls"], 0, 3*union)
    integer(work["logTerms"], 0, policy["max_total_log_terms"])
    return radius


def _control_presence(energy, adaptive):
    require(not ("continuousCableEnergy" in energy and "varyingCableEnergy" in energy),
            "Two cable accounting owners")
    if adaptive is None:
        # Generic callbacks have no externally declared primitive work control.
        # Certificates would require the corresponding full bound context.
        require(not any(key in energy for key in ("boundedContactWork", "boundedContactMechanical",
                                                 "continuousCableEnergy", "varyingCableEnergy")),
                "Work certificates require externally bound adaptive control context")
        return
    for field, payload in (("boundedContactControl", "boundedContactWork"),
                           ("fixedCableControl", "continuousCableEnergy"),
                           ("varyingCableControl", "varyingCableEnergy")):
        require((adaptive[field] is not None) == (payload in energy), "Bound work control presence changed")
        if payload in energy:
            require(type(energy[payload]) is dict, "Work record required")
            same(energy[payload].get("definition"), adaptive[field])


def _contact_sums(energy, contact_radius, cable, varying, cable_radius):
    if "boundedContactWork" not in energy: return
    payload = energy.get("boundedContactMechanical")
    keys(payload, ("profile", "aggregationTermsJoules", "errorBoundsJoules", "assemblyRoundingBoundsJoules"))
    require(payload["profile"] == "bounded-contact-conditional-sums-v1", "Contact sum profile mismatch")
    common = set(MOTION[:4]) | {"kineticChangeJoules", "gripperFixedParameterChangeJoules"}
    names = {
        "mechanicalChangeJoules": common | {"sewingChangeJoules", "foldActuationChangeJoules", "gripperParameterWorkJoules"},
        "mechanicalChangeMinusTargetWorkJoules": common | {"sewingMotionAndActivationJoules", "foldFixedParameterChangeJoules",
                                                         "foldActivationParameterWorkJoules", "gripperActivationParameterWorkJoules"},
        "mechanicalChangeMinusParameterWorkJoules": common | {"sewingFixedParameterChangeJoules", "foldFixedParameterChangeJoules"},
        "targetParameterWorkJoules": {"sewingTargetParameterWorkJoules", "foldTargetParameterWorkJoules", "gripperTargetParameterWorkJoules"},
        "externalParameterWorkJoules": {"sewingParameterWorkJoules", "foldParameterWorkJoules", "gripperParameterWorkJoules"}}
    bases = {name: contact_radius if i < 3 else F() for i, name in enumerate(names)}
    if cable is not None:
        for name in list(names)[:3]:
            names[name].add("cableFixedParameterChangeJoules")
            bases[name] += cable_radius
    if varying:
        extras = ("cableParameterWorkJoules", "cableActivationParameterWorkJoules", None,
                  "cableTargetParameterWorkJoules", "cableParameterWorkJoules")
        errors = cable["parameterWork"]["certificate"]["errorsJoules"]
        parameters = {"cableParameterWorkJoules": "totalWorkJoules", "cableActivationParameterWorkJoules": "activationWorkJoules",
                      "cableTargetParameterWorkJoules": "targetWorkJoules"}
        for name, extra in zip(names, extras):
            if extra is not None:
                names[name].add(extra)
                bases[name] += fraction(errors[parameters[extra]], nonnegative=True)
        # The producer requires these endpoint/scalar error fields as well.
        for endpoint in ("before", "after"):
            fraction(cable[endpoint]["certificate"]["energyErrorBoundJoules"], nonnegative=True)
        for label in ("activationIncreaseWorkJoules", "releaseEnergyRemovedJoules"):
            fraction(errors[label], nonnegative=True)
    for label in ("aggregationTermsJoules", "errorBoundsJoules", "assemblyRoundingBoundsJoules"):
        keys(payload[label], names)
    for name, fields in names.items():
        terms = payload["aggregationTermsJoules"][name]
        keys(terms, fields)
        for field, value in terms.items():
            number(value); same(value, energy[field])
        exact = sum((F(item) for item in terms.values()), F())
        rounded = float(exact)
        number(rounded); same(rounded, energy[name])
        rounding = abs(F(rounded)-exact)
        same(rational(rounding), payload["assemblyRoundingBoundsJoules"][name])
        same(rational(rounding+bases[name]), payload["errorBoundsJoules"][name])
        if cable is not None and name in cable["aggregationTermsJoules"]:
            for label in ("aggregationTermsJoules", "errorBoundsJoules", "assemblyRoundingBoundsJoules"):
                same(payload[label][name], cable[label][name])


def energy_reduction(mass, before, after, energy, allocation, adaptive):
    """Conditional exact stored-value reduction; primitive enclosures stay inputs."""
    require(type(energy) is dict and energy.get("accepted") is False, "Unaccepted energy record required")
    for name in PUBLIC_ENERGY: number(energy.get(name))
    _control_presence(energy, adaptive)
    motion = energy.get("temporalMotion")
    keys(motion, ("termsJoules", "knownErrorBoundJoules"))
    terms = motion["termsJoules"]
    keys(terms, MOTION)
    for name, item in terms.items(): number(item); same(item, energy.get(name))
    contact_radius = _contact_radius(energy)
    varying = "varyingCableEnergy" in energy
    cable = energy.get("varyingCableEnergy", energy.get("continuousCableEnergy"))
    cable_radius = F()
    if cable is not None:
        work = cable["motionWork" if varying else "work"]
        number(work["changeJoules"])
        same(work["changeJoules"], terms["cableFixedParameterChangeJoules"])
        cable_radius = fraction(work["certificate"]["changeErrorBoundJoules"], nonnegative=True)
        reference = cable["aggregationTermsJoules"]["mechanicalChangeMinusParameterWorkJoules"]
        for name, item in terms.items(): same(item, reference.get(name))
    _contact_sums(energy, contact_radius, cable, varying, cable_radius)
    radius = fraction(motion["knownErrorBoundJoules"], nonnegative=True)
    require(radius == contact_radius+cable_radius, "Fixed-motion uncertainty changed")
    kinetic = sum((F(m)/2 * sum((F(y)**2-F(x)**2 for x, y in zip(old, new)), F())
                   for m, old, new in zip(mass, before, after)), F())
    nominal = kinetic + sum((F(item) for item in terms.values()), F())
    upper, lower = abs(nominal)+radius, max(F(), abs(nominal)-radius)
    outcome = "within-budget" if upper <= allocation else "exceeds-budget" if lower > allocation else "uncertainty-overlap"
    return {"exactStoredKineticChangeJoules": rational(kinetic), "nominalDefectJoules": rational(nominal),
            "knownErrorBoundJoules": rational(radius), "absoluteUpperBoundJoules": rational(upper),
            "allocationJoules": rational(allocation), "outcome": outcome}


def _work_endpoints(energy, start, end):
    """Bind stored endpoint hashes, without claiming their enclosures are true."""
    q0, q1 = start["positionsMeters"], end["positionsMeters"]
    if "boundedContactWork" in energy:
        record = energy["boundedContactWork"]
        same(record["definition"]["vertexCount"], len(q0))
        for label, q in (("start", q0), ("end", q1)):
            digest = hashlib.sha256()
            for row in q:
                for value in row: digest.update(struct.pack("<d", value))
            same(record[label]["positionsSha256"], digest.hexdigest())
    cable = energy.get("varyingCableEnergy", energy.get("continuousCableEnergy"))
    if cable is None: return
    varying = "varyingCableEnergy" in energy
    first, last = canonical_identity(q0)["sha256"], canonical_identity(q1)["sha256"]
    for endpoint, digest in (("before", first), ("after", last)):
        same(cable[endpoint]["certificate"]["positionsSha256"], digest)
        number(cable[endpoint]["energyJoules"], 0)
        same(cable[endpoint]["energyJoules"], energy["cableBeforeJoules" if endpoint == "before" else "cableAfterJoules"])
    work = cable["motionWork" if varying else "work"]["certificate"]
    same(work["startPositionsSha256"], first); same(work["endPositionsSha256"], last)
    require(work["profile"] == "continuous-cable-fixed-parameter-work-enclosure-v1" and
            work["verified"] is True and work["accepted"] is False, "Cable motion certificate profile mismatch")
    if varying:
        same(cable["parameterWork"]["certificate"]["positionsSha256"], first)


def _error_record(value, *, stage=False):
    keys(value, ("type", "message", "stage") if stage else ("type", "message"))
    require(all(type(part) is str for part in value.values()), "String diagnostic fields required")


def _phase(snapshot):
    phase, reason = snapshot["phase"], snapshot["controllerReason"]
    reasons = {"complete", "interrupted", "attempt-budget-exhausted", "evaluation-budget-exhausted",
               "substep-duration-underflow", "substep-duration-unrepresentable", "temporal-depth-exhausted",
               "solver-resource-or-runtime-failure"}
    require(phase in ("running", "controller-finished", "report-ready"), "Unsupported controller phase")
    require((reason is None if phase == "running" else type(reason) is str and reason in reasons),
            "Controller phase/reason mismatch")
    same(snapshot["reportReady"], phase == "report-ready")
    require(snapshot["enrichment"] in ("not-requested", "pending", "ready", "failed"), "Unknown enrichment status")
    require(snapshot["enrichment"] == "not-requested" or phase == "report-ready", "Enrichment precedes report")
    require(type(snapshot["failures"]) is list, "Failure observations required")
    for row in snapshot["failures"]: _error_record(row, stage=True)


def _replay(snapshot, limits):
    c = context(snapshot["context"], limits)
    initial = c["initialState"]
    count = len(initial["positionsMeters"])
    attempts, reservations = snapshot["attempts"], snapshot["reservations"]
    assessments, transactions, accepted = (snapshot[name] for name in ("temporalAssessments", "transactions", "acceptedSteps"))
    for rows in (attempts, reservations, assessments, transactions, accepted):
        require(type(rows) is list, "History arrays required")
    require(len(attempts) <= len(reservations) <= len(attempts)+1, "Only one final unpublished trial is permitted")
    require(len(reservations) <= min(limits.trials, c["maxAttempts"]) and
            len(assessments) <= limits.assessments and len(accepted) == 2*len(transactions), "History budget or pair count mismatch")
    allowance, evaluation_budget = c["evaluationLimit"], c["policy"]["maxEvaluationBudget"]
    require(len(reservations)*allowance <= evaluation_budget, "Reserved evaluation budget exceeded")
    same(snapshot["resources"], {"reservedTrials": len(reservations), "recordedTrials": len(attempts),
                                 "chargedEvaluationAllowance": len(reservations)*allowance})
    dt, policy = c["requestedDurationSeconds"], c["policy"]
    budget = F(policy["numericalEnergyBudgetJ"])
    current, completed, total = initial, 0., F()
    ri = ai = ti = 0
    pending, next_initial = [], 0
    terminal, missing, tail = None, None, "none"
    fatal_seen = False

    def unavailable_trial(a, b):
        if ri >= c["maxAttempts"]: return "attempt-budget-exhausted"
        if (ri+1)*allowance > evaluation_budget: return "evaluation-budget-exhausted"
        duration = float(dt*(b-a))
        if not math.isfinite(duration) or duration <= 0: return "substep-duration-underflow"
        if F(duration) != F(dt)*(F(b)-F(a)): return "substep-duration-unrepresentable"
        return None

    while ri < len(reservations) or ai < len(assessments):
        require(terminal is None, "Work follows a terminal or uncommitted observation")
        if not pending:
            require(next_initial < c["initialSubdivisions"], "Work exceeds requested duration")
            i = next_initial
            pending.append((i/c["initialSubdivisions"], (i+1)/c["initialSubdivisions"], 0, None, i))
            next_initial += 1
        a, b, depth, parent, original = pending.pop()
        require(depth < c["maxDepth"], "Trial exceeds temporal depth")
        midpoint = (a+b)/2
        row = assessments[ai] if ai < len(assessments) else None
        if row is not None:
            require(type(row) is dict, "Assessment record required")
            for name, expected in (("assessmentId", ai+1), ("startFraction", a), ("endFraction", b), ("depth", depth)):
                same(row.get(name), expected)
            ids = row.get("trialIds")
            require(type(ids) is list and len(ids) <= 3, "Bounded assessment trial IDs required")
            same(ids, list(range(ri+1, ri+1+len(ids))))
            n = len(ids)
            require(ri+n <= len(attempts), "Assessment references unpublished trial")
        else:
            n = len(reservations)-ri
            require(1 <= n <= 3, "Bounded orphan trial suffix required")
        node, starts = [], []
        first_id = ri+1
        missing = None
        for role_index in range(n):
            if role_index:
                require(node[-1] is not None and node[-1]["numericallyValid"] and not node[-1]["fatal"],
                        "Rejected or fatal trial cannot authorize next role")
            start = node[1]["state"] if role_index == 2 else current
            left, right = ((a, b), (a, midpoint), (midpoint, b))[role_index]
            require(unavailable_trial(left, right) is None, "A reserved trial violates pre-dispatch budget/duration")
            base = {"attemptId": ri+1, "parentAttemptId": parent if role_index == 0 else first_id,
                    "initialInterval": original, "startFraction": left, "endFraction": right,
                    "durationSeconds": float(dt*(right-left)), "depth": depth+(role_index != 0),
                    "role": ("coarse", "fine-first", "fine-second")[role_index], "converged": False,
                    "evaluationAllowanceCharged": allowance, "startStateSha256": start["sha256"]}
            reservation = reservations[ri]
            published = ri < len(attempts)
            keys(reservation, ("trial", "observation", "recordPublished") if published else ("trial", "observation"))
            same(reservation["trial"], base)
            require(reservation["observation"] in ("reserved", "dispatch-authorized", "return-observed"),
                    "Unknown reservation observation")
            starts.append(start)
            if published:
                require(reservation["recordPublished"] is True, "Published trial marker required")
                trial = attempts[ri]
                require(type(trial) is dict, "Trial record required")
                for name, expected in base.items():
                    if name != "converged": same(trial.get(name), expected)
                flag(trial.get("converged")); valid = flag(trial.get("numericallyValid")); fatal = flag(trial.get("fatal"))
                require("transactionId" not in trial and "completedDurationSeconds" not in trial,
                        "Raw trials cannot carry commit metadata")
                if valid:
                    require(trial.get("outcome") == "provisional", "Valid raw trial must stay provisional")
                    require(reservation["observation"] == "return-observed", "Valid trial lacks observed callback return")
                    state(trial.get("state"), limits, count)
                    # Generic callbacks need not set converged or satisfy a
                    # specific velocity law. Those are evaluator audit gates.
                else:
                    require(trial.get("outcome") in ("numerically-rejected", "interrupted"), "Invalid trial outcome mismatch")
                    # A post-return exception may leave q/v and converged as
                    # diagnostic fields. They never authorize a next role.
                if "error" in trial: _error_record(trial["error"])
                fatal_seen |= fatal
                node.append(trial)
            else:
                require(row is None and role_index == n-1, "Unrecorded reservation must end the history")
                node.append(None)
            ri += 1
        if row is None:
            tail = "unpublished-assessment-or-trial"
            terminal = "unresolved-interruption"
            break
        ai += 1
        base_fields = {"assessmentId", "startFraction", "endFraction", "depth", "trialIds", "outcome"}
        all_valid = len(node) == 3 and all(part["numericallyValid"] for part in node)
        passed = False
        if all_valid:
            position = metric(node[0]["state"]["positionsMeters"], node[2]["state"]["positionsMeters"], policy["positionToleranceM"])
            velocity = metric(node[0]["state"]["velocitiesMPerS"], node[2]["state"]["velocitiesMPerS"], policy["velocityToleranceMPerS"])
            if row["outcome"] == "invalid-temporal-evidence":
                # Keep partial reductions only in their assignment order.
                require(set(row) in (base_fields | {"error"}, base_fields | {"error", "position"},
                                     base_fields | {"error", "position", "velocity"}), "Unsupported partial reduction shape")
                _error_record(row["error"])
                if "position" in row: same(row["position"], position)
                if "velocity" in row: same(row["velocity"], velocity)
                # This profile does not infer a thrown validation exception
                # from arbitrary diagnostic text. No later work is admitted.
                terminal, tail = "unresolved-invalid-temporal-evidence", "invalid-temporal-evidence"
            else:
                keys(row, base_fields | {"position", "velocity", "fineEnergy"})
                same(row["position"], position); same(row["velocity"], velocity)
                reduced = []
                for k in (1, 2):
                    trial = node[k]
                    require(type(trial.get("step")) is dict, "Fine trial energy step required")
                    energy = trial["step"].get("energyBalance")
                    require(type(energy) is dict, "Fine trial energy record required")
                    _work_endpoints(energy, starts[k], trial["state"])
                    reduced.append(energy_reduction(c["massKg"], starts[k]["velocitiesMPerS"],
                        trial["state"]["velocitiesMPerS"], energy,
                        budget*(F(trial["endFraction"])-F(trial["startFraction"])), c["adaptiveContext"]))
                same(row["fineEnergy"], reduced)
                passed = position["withinThreshold"] and velocity["withinThreshold"] and all(x["outcome"] == "within-budget" for x in reduced)
                same(row["outcome"], "indicators-passed-awaiting-commit" if passed else "temporally-rejected")
        else:
            keys(row, base_fields)
            if len(node) < 3 and (not node or node[-1]["numericallyValid"] and not node[-1]["fatal"]):
                left, right = ((a, b), (a, midpoint), (midpoint, b))[len(node)]
                missing = unavailable_trial(left, right)
            if row["outcome"] == "numerically-rejected":
                require(bool(node) and not node[-1]["numericallyValid"] and not node[-1]["fatal"] and
                        node[-1]["outcome"] == "numerically-rejected", "Numerical rejection lacks a nonfatal rejecting trial")
            else:
                require(row["outcome"] == "incomplete", "Incomplete node outcome mismatch")
                terminal, tail = missing or "unresolved-interruption", "incomplete-assessment"
        if passed and ti < len(transactions) and transactions[ti].get("transactionId") == row["assessmentId"]:
            require(not any(part["fatal"] for part in node), "Fatal trial cannot commit")
            same(a, completed)
            expected_transaction = {"transactionId": row["assessmentId"], "fineTrialIds": [node[1]["attemptId"], node[2]["attemptId"]],
                                    "startFraction": a, "endFraction": b}
            same(transactions[ti], expected_transaction)
            for index, trial in enumerate(node[1:]):
                expected = {**trial, "outcome": "committed", "transactionId": row["assessmentId"],
                            "completedDurationSeconds": float(dt*trial["endFraction"])}
                same(accepted[2*ti+index], expected)
            total += sum((fraction(part["absoluteUpperBoundJoules"], nonnegative=True) for part in row["fineEnergy"]), F())
            require(total <= budget*F(b), "Committed energy prefix exceeds allocation")
            current, completed = node[2]["state"], b
            ti += 1
        elif passed:
            terminal, tail = "awaiting-commit", "passed-assessment-without-commit"
        elif terminal is None:
            if any(part["fatal"] for part in node):
                terminal, tail = "fatal-trial", "failed-assessment"
            elif depth+1 >= c["maxDepth"]:
                terminal, tail = "temporal-depth-exhausted", "failed-assessment"
            else:
                pending.extend(((midpoint, b, depth+1, first_id, original), (a, midpoint, depth+1, first_id, original)))
    require(ri == len(reservations) and ai == len(assessments) and ti == len(transactions),
            "Unconsumed history or transaction")
    same(snapshot["state"], current)
    same(snapshot["completedFraction"], completed)
    same(snapshot["complete"], completed == 1.)
    same(snapshot["conditionalAbsoluteEnergyBoundJoules"], rational(total))
    reason = snapshot["controllerReason"]
    if completed == 1.: require(reason in (None, "complete", "interrupted"), "Full prefix has an impossible terminal reason")
    if reason == "solver-resource-or-runtime-failure": require(fatal_seen, "Resource reason lacks a published fatal trial")
    if reason == "complete": require(completed == 1. and tail == "none", "Successful controller report has an incomplete prefix")
    if reason in ("attempt-budget-exhausted", "evaluation-budget-exhausted"):
        require(missing == reason, "Budget failure did not reach the corresponding dispatch gate")
    if reason in ("substep-duration-underflow", "substep-duration-unrepresentable"):
        require(missing == reason, "Duration failure is not reproduced")
    if reason == "temporal-depth-exhausted": require(terminal == reason, "Depth failure is not reproduced")
    if fatal_seen and reason is not None:
        require(reason in ("interrupted", "solver-resource-or-runtime-failure"), "Fatal trial disagrees with controller reason")
    return {"schedulerPrefixConsistent": True, "committedPrefixIndicatorsReproduced": True,
            "numericalPrefixComplete": completed == 1., "completedFraction": completed,
            "stateSha256": current["sha256"], "committedTransactions": ti, "committedFineTrials": 2*ti,
            "reservedTrials": ri, "recordedTrials": len(attempts), "chargedEvaluationAllowance": ri*allowance,
            "conditionalAbsoluteEnergyBoundJoules": rational(total), "tailObservation": tail,
            "causalStoppingExplanation": "complete-prefix" if completed == 1. else "not-independently-established"}


def audit_snapshot(path, *, expected_bytes, expected_sha256, expected_run_identity,
                   expected_context_sha256, evaluator_profile, limits):
    """Admit a bounded raw observation and reproduce its committed pair prefix.

    Expectations must originate in a separate frozen run declaration. This
    profile supports little-endian binary64 controller callbacks and full bound
    adaptive contexts for work-control presence only. It does not authenticate
    source/model inputs, prove a callback's physics, establish process outcome,
    or admit a resume. Unsupported/malformed histories raise without a result.
    """
    require(evaluator_profile == EVALUATOR, "Unsupported evaluator/endian contract")
    digest_text(expected_run_identity)
    if expected_context_sha256 is not None: digest_text(expected_context_sha256)
    envelope = read_snapshot(path, expected_bytes=expected_bytes, expected_sha256=expected_sha256, limits=limits)
    keys(envelope, ("profile", "accepted", "runIdentity", "snapshot"))
    require(envelope["profile"] == TRANSPORT and envelope["accepted"] is False, "Transport profile mismatch")
    same(envelope["runIdentity"], expected_run_identity)
    snapshot = envelope["snapshot"]
    common = {"profile", "accepted", "bound", "stopRequested", "stopReason", "available"}
    require(type(snapshot) is dict, "Retained snapshot required")
    available = flag(snapshot.get("available"))
    keys(snapshot, common | ({"context", "phase", "controllerReason", "reportReady", "enrichment", "failures",
            "complete", "completedFraction", "state", "acceptedSteps", "transactions", "conditionalAbsoluteEnergyBoundJoules",
            "reservations", "attempts", "temporalAssessments", "resources"} if available else {"admissionFailure"}))
    require(snapshot["profile"] == RETAINED and snapshot["accepted"] is False, "Retained profile mismatch")
    flag(snapshot["bound"])
    if flag(snapshot["stopRequested"]):
        require(type(snapshot["stopReason"]) is str and 1 <= len(snapshot["stopReason"]) <= 128, "Bounded stop reason required")
    else:
        require(snapshot["stopReason"] is None, "Unrequested stop reason")
    result = {"profile": PROFILE, "accepted": False, "available": available, "evaluatorProfile": EVALUATOR,
              "artifactIdentityVerified": True, "contextIdentityVerified": available,
              "envelopeWellFormed": True, "artifact": {"bytes": expected_bytes, "sha256": expected_sha256},
              "runIdentity": expected_run_identity, "contextSha256": expected_context_sha256,
              "scope": "Raw controller observations and conditional exact stored-value pair accounting only.",
              "notEvaluated": ["source-model-authenticity", "adaptive-response-and-control-interpolation",
                  "primitive-work-enclosures", "constitutive-and-force-accuracy", "contact-and-triangle-paths",
                  "global-temporal-and-spatial-accuracy", "actual-process-outcome", "resume", "garment-acceptance"]}
    if not available:
        require(expected_context_sha256 is None, "Unavailable snapshot cannot verify expected context")
        if snapshot["admissionFailure"] is not None: _error_record(snapshot["admissionFailure"], stage=True)
        result.update(schedulerPrefixConsistent=None, committedPrefixIndicatorsReproduced=None,
                      numericalPrefixComplete=None, causalStoppingExplanation="unavailable")
        return result
    require(snapshot["bound"] is True and expected_context_sha256 is not None, "Available snapshot needs bound external context")
    same(canonical_identity(snapshot["context"])["sha256"], expected_context_sha256)
    _phase(snapshot)
    try:
        result.update(_replay(snapshot, limits))
    except (KeyError, TypeError, OverflowError, IndexError) as error:
        raise ValueError("Incomplete or unsupported retained temporal evidence") from error
    return result
