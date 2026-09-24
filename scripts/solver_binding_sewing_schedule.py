"""Static composition of source-bound folds and preserved distance sewing.

Only the original reference sewing recipe is bound. The refined source remains
bare: no force model, integration, captured admission or construction runs.
"""

import copy
from fractions import Fraction
import hashlib
import math

from solver_binding_control_schedule import validate_binding_control_schedule
from solver_binding_source import _bounded, _encoded
from solver_sewing_input import bind_sewing_activation, derive_sewing_row_ids


PROFILE = "source-binding-sewing-fold-schedule-v1"
HELD_SELECTOR = {"registrationId": "bind_opening_left_left", "memberIndex": 1}
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


def _rat(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _reference(base, reference, subdivisions):
    if (type(reference) is not dict or not set(base).issubset(reference)
            or not set(reference).difference(base).issubset(REFERENCE_EXTRAS)
            or "sewingActuation" not in reference
            or any(_encoded(reference[key]) != _encoded(value) for key, value in base.items())):
        raise ValueError("Reference must preserve the complete original base and only declared provenance extras")
    # This validates only the preexisting ORIGINAL cuff recipe; no refined
    # sewingActuation object is created or passed to an admission path.
    controls, manifest = bind_sewing_activation(reference, subdivisions, sewing_mode="distance")
    recipe = reference["sewingActuation"]
    initial, final = recipe["initialTargetsMeters"], recipe["finalTargetsMeters"]
    if len(initial) != 40 or _encoded(initial) != _encoded(final):
        raise ValueError("All forty unchanged initial/final reference targets required")
    for value in initial:
        if (type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 100
                or Fraction(float(value)) != Fraction(value)):
            raise ValueError("Finite positive exactly representable reference distances required")
    knots = recipe["schedule"]["knots"]
    if len(knots) != 2:
        raise ValueError("Constant five-held/thirty-five-pending reference schedule required")
    for index, knot in enumerate(knots):
        if type(knot["fraction"]) not in (int, float) or knot["fraction"] != index:
            raise ValueError("Reference sewing knots must span exactly zero and one")
        if (len(knot["activation"]) != 40
                or any(type(value) not in (int, float) or value != (1 if row < 5 else 0)
                       for row, value in enumerate(knot["activation"]))):
            raise ValueError("Exactly five held and thirty-five pending sewing samples required")
    if controls.compliance != 1e-8 or manifest["sourceFrameMetadataUsed"] is not False:
        raise ValueError("Original distance-only compliance without material-normal frames required")
    return recipe, list(controls.row_ids)


def _phases(base):
    result = {}
    for phase, indices in base["phaseConstraintRows"].items():
        if type(phase) is not str or type(indices) is not list:
            raise ValueError("Explicit source phase row membership required")
        for index in indices:
            if type(index) is not int or not 0 <= index < 40 or index in result:
                raise ValueError("Every source row must have exactly one declared phase")
            result[index] = phase
    if set(result) != set(range(40)):
        raise ValueError("All forty source phase row memberships required")
    return result


def _operator(row, offsets, positions, target_squared):
    # Prefix preservation is checked by the static refined-source validator.
    # Keep original coefficients exactly as binary input, including tiny terms.
    vector = [sum((Fraction(term["coefficient"]) * Fraction(
        positions[offsets[term["instanceId"]] + term["vertex"]][axis])
        for term in row["terms"]), Fraction()) for axis in range(3)]
    squared = sum((coordinate * coordinate for coordinate in vector), Fraction())
    return {"vectorMeters": [_rat(coordinate) for coordinate in vector],
            "squaredGapMetersSquared": _rat(squared),
            "squaredGapMinusTargetSquaredMetersSquared": _rat(squared - target_squared)}, squared


def build_binding_sewing_schedule(source, fold, placement, fold_schedule, reference_source):
    """Declare one shared control clock without installing either control."""
    inputs = (source, fold, placement, fold_schedule, reference_source)
    before = [_bounded(value) for value in inputs]
    try:
        schedule = validate_binding_control_schedule(source, fold, placement, fold_schedule)
        base = source["baseUnit"]
        recipe, row_ids = _reference(base, reference_source, schedule["initialSubdivisions"])
        numerical_ids = list(derive_sewing_row_ids(source))
        if numerical_ids != row_ids:
            raise ValueError("All ordered original and numerical row identities must match")
        original_rows = base["embeddedConstraints"]["constraints"]
        rows = source["embeddedConstraints"]["constraints"]
        if len(original_rows) != 40 or len(rows) != 40:
            raise ValueError("All forty original and numerical source rows required")
        phases, bindings, witnesses, selectors, held_indices = _phases(base), [], [], {}, []
        for index, (original, row) in enumerate(zip(original_rows, rows)):
            fraction = Fraction(row["fraction"])
            selector = (row["registrationId"], row["memberIndex"])
            if (selector != (original["registrationId"], original["memberIndex"])
                    or fraction != Fraction(original["fraction"])
                    or row["complianceMPerN"] != 1e-8 or original["complianceMPerN"] != 1e-8):
                raise ValueError("Unchanged selector, source fraction and compliance required")
            selectors.setdefault(selector, []).append(fraction)
            held = selector == (HELD_SELECTOR["registrationId"], HELD_SELECTOR["memberIndex"])
            if held:
                held_indices.append(index)
            target = recipe["initialTargetsMeters"][index]
            bindings.append({"rowIndex": index, "rowId": row_ids[index], "registrationId": selector[0],
                "memberIndex": selector[1], "fractionNumerator": fraction.numerator,
                "fractionDenominator": fraction.denominator, "originalRowSha256": _sha(original),
                "numericalRowSha256": _sha(row), "phaseId": phases[index], "held": held,
                "targetMeters": target, "complianceMPerN": row["complianceMPerN"]})
            if held:
                target_squared = Fraction(target)**2
                old, old_squared = _operator(original, source["instanceOffsets"], placement["placedMeters"], target_squared)
                new, new_squared = _operator(row, source["instanceOffsets"], placement["placedMeters"], target_squared)
                witnesses.append({"rowIndex": index, "originalOperator": old, "numericalOperator": new,
                    "targetSquaredMetersSquared": _rat(target_squared),
                    "numericalMinusOriginalSquaredGapMetersSquared": _rat(new_squared - old_squared)})
        if (held_indices != list(range(5)) or len(selectors) != 8
                or any(fractions != [Fraction(index, 4) for index in range(5)] for fractions in selectors.values())):
            raise ValueError("Complete eight-selector sampling with the declared five held rows required")
        result = {"profile": PROFILE, "accepted": False, "solverReady": False, "executable": False,
            "sourceSha256": hashlib.sha256(before[0]).hexdigest(), "baseUnitSha256": _sha(base),
            "foldDescriptorSha256": hashlib.sha256(before[1]).hexdigest(),
            "placementDescriptorSha256": hashlib.sha256(before[2]).hexdigest(),
            "foldScheduleDescriptorSha256": hashlib.sha256(before[3]).hexdigest(),
            "referenceSourceSha256": hashlib.sha256(before[4]).hexdigest(),
            "referenceSewingActuationSha256": _sha(recipe),
            "originalBundleSha256": _sha(base["embeddedConstraints"]),
            "numericalBundleSha256": _sha(source["embeddedConstraints"]),
            "nativeTopologySha256": schedule["nativeTopologySha256"],
            **{key: copy.deepcopy(schedule[key]) for key in (
                "durationSeconds", "initialSubdivisions", "maxDepth", "retryGridDenominator", "minimumStepSeconds",
                "foldControlSchedule", "foldStiffnessJoules")},
            "timePolicy": "one-original-fraction-grid-for-all-controls-v1",
            "sewingMode": "distance", "sewingControlSchedule": copy.deepcopy(recipe["schedule"]),
            "initialTargetsMeters": copy.deepcopy(recipe["initialTargetsMeters"]),
            "finalTargetsMeters": copy.deepcopy(recipe["finalTargetsMeters"]), "complianceMPerN": 1e-8,
            "heldSelector": dict(HELD_SELECTOR), "heldRowIndices": held_indices,
            "pendingRowIndices": list(range(5, 40)), "rowBindings": bindings,
            "initialHeldGeometryWitnesses": witnesses, "sourcePhasePlanSha256": _sha(base["phasePlan"]),
            "sourcePhaseConstraintRowsSha256": _sha(base["phaseConstraintRows"]),
            "historicalExtrasInherited": False, "controlsInstalled": False, "constructionPhaseCompleted": False,
            "continuousSpatialSeamsVerified": False, "materialSidesResolved": False,
            "releasedFoldTailPolicy": "sewing-remains-held-when-folds-release", "limitations": list(LIMITATIONS)}
        if before != [_bounded(value) for value in inputs]:
            raise ValueError("Combined static declaration inputs changed during derivation")
        _bounded(result)
        return copy.deepcopy(result)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Malformed or incomplete combined static control input") from error


def validate_binding_sewing_schedule(source, fold, placement, fold_schedule, reference_source, descriptor):
    """Require complete byte-sensitive rederivation, including all scope flags."""
    claimed = _bounded(descriptor)
    expected = build_binding_sewing_schedule(source, fold, placement, fold_schedule, reference_source)
    if claimed != _encoded(expected):
        raise ValueError("Combined static controls differ from complete rederivation")
    return expected
