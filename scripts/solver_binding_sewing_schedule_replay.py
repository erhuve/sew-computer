"""Independent static composition of held sewing and declared fold controls.

Only supplied-source arithmetic and identity are checked. Pattern and split
rederivation remain the separate strict source validator's responsibility.
No numerical sampler, force model, engine, capture or motion is imported.
"""

from fractions import Fraction as F
import hashlib
import math

from solver_binding_control_schedule_replay import _encoded, verify_binding_control_schedule
from solver_binding_remap_replay import verify_binding_remap


PROFILE = "independent-binding-sewing-fold-schedule-verification-v1"
REQUIRED_HELPER_FILES = ("solver_binding_control_schedule_replay.py", "solver_binding_remap_replay.py",
                         "solver_binding_fold_replay.py", "solver_binding_placement_replay.py")
REFERENCE_EXTRAS = {"sewingActuation", "placedMeters", "bindingFirstTurnDiagnostic", "gripperActuation"}
LIMITATIONS = [
    "Static composition only; no controls installed, motion executed, captured source admitted or construction phase completed.",
    "All forty source rows, targets and compliance remain explicit. Five held samples are an initial numerical condition; thirty-five pending rows remain inactive.",
    "Original and refined coefficient operators remain distinct. Exact initial squared-gap witnesses retain their differences without retargeting or coefficient normalization.",
    "One original fraction and retry grid serves both controls. Positive parameter timesteps do not establish executable solver timesteps or continuous actuator work.",
    "Reference placement, grippers and textile-side declarations are provenance only and are not inherited. Scalar distance controls supply no material-normal frame or rotational joint.",
    "Releasing fold actuation leaves five sewing samples held; it is not an all-controls-free tail or a settling certificate.",
    "Continuous spatial seams, binding wrap/stitch-down/apex, open-edge and turning gates, material calibration, contact paths and garment acceptance remain unresolved.",
]


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _same(actual, expected, label):
    if _encoded(actual) != _encoded(expected):
        raise ValueError("Independent combined schedule mismatch: " + label)


def _number(value, lower, upper, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not lower <= value <= upper or positive and value <= 0):
        raise ValueError("Finite bounded non-Boolean binary64 scalar required")
    numerical = float(value)
    if F(numerical) != F(value):
        raise ValueError("Exact binary64 input representation required")
    return F(numerical)


def _rational(value):
    # Squared binary-input values are retained exactly; no float projection.
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _phase_membership(base):
    phases = base["phaseConstraintRows"]
    if type(phases) is not dict or not 1 <= len(phases) <= 128:
        raise ValueError("Complete bounded phase-to-row partition required")
    membership = {}
    for phase_id, rows in phases.items():
        if (type(phase_id) is not str or not 1 <= len(phase_id) <= 160
                or type(rows) is not list or len(rows) > 40):
            raise ValueError("Explicit phase identity and bounded row list required")
        for row in rows:
            if type(row) is not int or not 0 <= row < 40 or row in membership:
                raise ValueError("Unique raw integer phase rows covering all forty rows required")
            membership[row] = phase_id
    if set(membership) != set(range(40)):
        raise ValueError("Phase partition must cover all forty rows")
    return membership


def _row_identity(row):
    registration, member = row["registrationId"], row["memberIndex"]
    if (type(registration) is not str or not 1 <= len(registration) <= 160
            or any(ord(character) < 32 for character in registration)
            or type(member) is not int or not 1 <= member <= 7):
        raise ValueError("Raw bounded sewing selector identity required")
    fraction = _number(row["fraction"], 0, 1)
    if _number(row["complianceMPerN"], 0, 1, positive=True) != F(1e-8):
        raise ValueError("Every original and numerical row must retain exactly 1e-8 compliance")
    identity = "row:" + _sha([registration, member, fraction.numerator, fraction.denominator])
    return registration, member, fraction, identity


def _reference_recipe(reference, base, row_ids):
    if (type(reference) is not dict or not set(base) <= set(reference)
            or set(reference) - set(base) - REFERENCE_EXTRAS or "sewingActuation" not in reference):
        raise ValueError("Exact original base and explicit original sewing reference required")
    _same({key: reference[key] for key in base}, base, "every immutable original base field")
    recipe = reference["sewingActuation"]
    fields = {"profile", "accepted", "sourceSha256", "mode", "initialTargetsMeters", "finalTargetsMeters", "schedule"}
    if (type(recipe) is not dict or set(recipe) != fields or recipe["profile"] != "captured-sewing-activation-v1"
            or recipe["accepted"] is not False or recipe["mode"] != "distance"
            or recipe["sourceSha256"] != _sha({key:value for key,value in reference.items() if key != "sewingActuation"})):
        raise ValueError("Original distance recipe must bind the complete reference excluding only itself")
    for name in ("initialTargetsMeters", "finalTargetsMeters"):
        values = recipe[name]
        if type(values) is not list or len(values) != 40:
            raise ValueError("All forty explicit original targets required")
        for value in values:
            _number(value, 0, 100, positive=True)
    _same(recipe["initialTargetsMeters"], recipe["finalTargetsMeters"], "unchanged initial and final targets")
    schedule = recipe["schedule"]
    if (type(schedule) is not dict or set(schedule) != {"profile", "rowIds", "knots"}
            or schedule["profile"] != "sewing-row-activation-v1"):
        raise ValueError("Exact original row activation schedule required")
    _same(schedule["rowIds"], row_ids, "all ordered original row IDs")
    knots = schedule["knots"]
    if type(knots) is not list or len(knots) != 2:
        raise ValueError("Exactly two constant held/pending sewing knots required")
    for index, knot in enumerate(knots):
        if (type(knot) is not dict or set(knot) != {"fraction", "activation"}
                or _number(knot["fraction"], 0, 1) != index
                or type(knot["activation"]) is not list or len(knot["activation"]) != 40):
            raise ValueError("Complete sewing endpoint knot required")
        for row, value in enumerate(knot["activation"]):
            if _number(value, 0, 1) != (1 if row < 5 else 0):
                raise ValueError("Five original held rows and thirty-five pending rows required")
    return recipe


def _operator(row, source, placement, target_squared, *, original):
    terms = row["terms"]
    if type(terms) is not list or not 2 <= len(terms) <= 6:
        raise ValueError("Bounded signed source coefficient operator required")
    base = source["baseUnit"]
    original_counts = {item["id"]: len(base["sourceTemplates"][item["templateId"]]["restPositions"])
                       for item in base["instances"]}
    numerical_counts = {key: len(mesh["verticesMeters"]) for key,mesh in source["numericalMeshes"].items()}
    counts = original_counts if original else numerical_counts
    vector, supports, signs = [F(), F(), F()], set(), {1:[], -1:[]}
    for term in terms:
        if type(term) is not dict or set(term) != {"instanceId", "vertex", "coefficient"}:
            raise ValueError("Explicit instance/local vertex/coefficient source term required")
        identity, local = term["instanceId"], term["vertex"]
        if (type(identity) is not str or identity not in counts or type(local) is not int
                or not 0 <= local < counts[identity] or (identity,local) in supports):
            raise ValueError("Distinct valid original-prefix or numerical local supports required")
        coefficient = _number(term["coefficient"], -1, 1)
        if coefficient == 0:
            raise ValueError("Sparse source coefficient must be nonzero")
        supports.add((identity,local))
        signs[1 if coefficient > 0 else -1].append((identity,coefficient))
        # Both operators use CURRENT canonical offsets. Original local indices
        # identify the retained vertex prefix, never historical global indices.
        point = placement["placedMeters"][source["instanceOffsets"][identity]+local]
        for axis in range(3):
            vector[axis] += coefficient * _number(point[axis], -100, 100)
    for sign, anchor in signs.items():
        if (not 1 <= len(anchor) <= 3 or len({identity for identity,_ in anchor}) != 1
                or abs(sum((weight for _,weight in anchor),F())-sign) > F(1e-12)):
            raise ValueError("Two bounded normalized source anchors required")
    if signs[1][0][0] == signs[-1][0][0]:
        raise ValueError("Two distinct physical source instances required")
    squared = sum((component*component for component in vector),F())
    return {"vectorMeters": [_rational(component) for component in vector],
            "squaredGapMetersSquared": _rational(squared),
            "squaredGapMinusTargetSquaredMetersSquared": _rational(squared-target_squared)}, squared


def verify_binding_sewing_schedule(source, fold, placement, fold_schedule, reference_source, descriptor):
    """Rederive static combined controls and exact squared-gap evidence only."""
    inputs = (source, fold, placement, fold_schedule, reference_source, descriptor)
    before = [_encoded(value) for value in inputs]
    # Match the original reference binder's raw JSON ceiling without importing
    # it or confusing that binder with independent source-pattern validation.
    if len(before[4]) > 8 * 1024**2:
        raise ValueError("Original sewing reference exceeds its bounded JSON byte budget")
    try:
        fold_audit = verify_binding_control_schedule(source, fold, placement, fold_schedule)
        remap_audit = verify_binding_remap(source)
        base = source["baseUnit"]
        original_rows = base["embeddedConstraints"]["constraints"]
        numerical_rows = source["embeddedConstraints"]["constraints"]
        if (type(original_rows) is not list or type(numerical_rows) is not list
                or len(original_rows) != 40 or len(numerical_rows) != 40):
            raise ValueError("Exactly forty original and numerical rows required")
        identities, groups, row_ids = [], {}, []
        for index,(old,new) in enumerate(zip(original_rows,numerical_rows)):
            original = _row_identity(old)
            numerical = _row_identity(new)
            _same([old[key] for key in ("registrationId","memberIndex","fraction")],
                  [new[key] for key in ("registrationId","memberIndex","fraction")], "original/numerical row selector")
            if original != numerical or original[3] in row_ids:
                raise ValueError("Complete unique original/numerical row identities required")
            identities.append(original)
            row_ids.append(original[3])
            groups.setdefault(original[:2], []).append((index,original[2]))
        fractions = [F(0),F(1,4),F(1,2),F(3,4),F(1)]
        if (len(groups) != 8 or any([fraction for _,fraction in rows] != fractions for rows in groups.values())
                or [index for index,_ in groups.get(("bind_opening_left_left",1),[])] != list(range(5))):
            raise ValueError("Eight five-fraction selectors with the first five left-binding rows required")
        phase_ids = _phase_membership(base)
        recipe = _reference_recipe(reference_source, base, row_ids)
        bindings, witnesses = [], []
        for index,(old,new,identity,target) in enumerate(zip(original_rows,numerical_rows,identities,recipe["initialTargetsMeters"])):
            registration,member,fraction,row_id = identity
            bindings.append({"rowIndex":index,"rowId":row_id,"registrationId":registration,"memberIndex":member,
                "fractionNumerator":fraction.numerator,"fractionDenominator":fraction.denominator,
                "originalRowSha256":_sha(old),"numericalRowSha256":_sha(new),"phaseId":phase_ids[index],
                "held":index<5,"targetMeters":target,"complianceMPerN":new["complianceMPerN"]})
            if index < 5:
                target_squared = F(target)**2
                old_operator,old_squared = _operator(old,source,placement,target_squared,original=True)
                new_operator,new_squared = _operator(new,source,placement,target_squared,original=False)
                witnesses.append({"rowIndex":index,"originalOperator":old_operator,"numericalOperator":new_operator,
                    "targetSquaredMetersSquared":_rational(target_squared),
                    "numericalMinusOriginalSquaredGapMetersSquared":_rational(new_squared-old_squared)})
        expected = {"profile":"source-binding-sewing-fold-schedule-v1","accepted":False,"solverReady":False,"executable":False,
            "sourceSha256":_sha(source),"baseUnitSha256":_sha(base),"foldDescriptorSha256":_sha(fold),
            "placementDescriptorSha256":_sha(placement),"foldScheduleDescriptorSha256":_sha(fold_schedule),
            "referenceSourceSha256":_sha(reference_source),"referenceSewingActuationSha256":_sha(recipe),
            "originalBundleSha256":_sha(base["embeddedConstraints"]),"numericalBundleSha256":_sha(source["embeddedConstraints"]),
            "nativeTopologySha256":fold_schedule["nativeTopologySha256"],
            **{key:fold_schedule[key] for key in ("durationSeconds","initialSubdivisions","maxDepth","retryGridDenominator","minimumStepSeconds")},
            "timePolicy":"one-original-fraction-grid-for-all-controls-v1",
            "foldControlSchedule":fold_schedule["foldControlSchedule"],"foldStiffnessJoules":fold_schedule["foldStiffnessJoules"],
            "sewingMode":"distance","sewingControlSchedule":recipe["schedule"],
            "initialTargetsMeters":recipe["initialTargetsMeters"],"finalTargetsMeters":recipe["finalTargetsMeters"],"complianceMPerN":1e-8,
            "heldSelector":{"registrationId":"bind_opening_left_left","memberIndex":1},
            "heldRowIndices":list(range(5)),"pendingRowIndices":list(range(5,40)),"rowBindings":bindings,
            "initialHeldGeometryWitnesses":witnesses,"sourcePhasePlanSha256":_sha(base["phasePlan"]),
            "sourcePhaseConstraintRowsSha256":_sha(base["phaseConstraintRows"]),"historicalExtrasInherited":False,
            "controlsInstalled":False,"constructionPhaseCompleted":False,"continuousSpatialSeamsVerified":False,"materialSidesResolved":False,
            "releasedFoldTailPolicy":"sewing-remains-held-when-folds-release","limitations":LIMITATIONS}
        if before[5] != _encoded(expected):
            raise ValueError("Independent combined schedule differs from complete reconstruction")
        if before != [_encoded(value) for value in inputs]:
            raise ValueError("Combined schedule inputs changed during verification")
        return {"profile":PROFILE,"verified":True,"accepted":False,"descriptorSha256":_sha(descriptor),
            "sourceSha256":_sha(source),"baseUnitSha256":_sha(base),"referenceSourceSha256":_sha(reference_source),
            "rowCount":40,"heldRowCount":5,"pendingRowCount":35,"selectorCount":8,
            "geometryWitnessesSha256":_sha(witnesses),"foldScheduleAudit":fold_audit,"remapAudit":remap_audit,
            "sourceScope":"Independent supplied-source combined declaration: every original/numerical row, reference target and compliance, held/pending activation, unique phase-row membership and exact coefficient vectors/squared gaps at CURRENT placement. Original local prefixes use current canonical offsets; no square root, normalization or target reseeding. Fold/remap prerequisite scopes remain in their audit summaries; pattern and split-geometry rederivation require the separate strict source validator. Historical extras are hashed provenance only. No controls installed, source admission, numerical motion, continuous spatial seam, frame/material side, contact path, all-controls-free tail, settling or construction acceptance."}
    except (KeyError, TypeError, IndexError, StopIteration, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Malformed or incomplete independent combined schedule input") from error
