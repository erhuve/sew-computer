"""Source/placement-bound temporal fold declarations; no motion or admission.

The static fold descriptor supplies native identities and stiffness, while this
request supplies new source-relative angles. Its old reference targets do not
certify this schedule. No sewing, gripper or construction phase is installed.
"""

import copy
from fractions import Fraction
import hashlib
import math

from solver_binding_fold import validate_binding_fold_control
from solver_binding_placement import validate_binding_placement
from solver_binding_source import _bounded, _encoded
from solver_fold_control_schedule import FoldControlSchedule


PROFILE = "source-binding-fold-schedule-v1"
REQUEST_PROFILE = "source-binding-fold-schedule-request-v1"
GRIPPER_POLICY = "none-balanced-relative-folds-v1"
ANGLE_LIMIT = math.pi - 1e-8
MAX_DURATION_SECONDS = 3600
LIMITATIONS = [
    "Static source/placement-bound fold declaration only; no controls installed, motion executed, captured input admitted or construction completed.",
    "Static fold reference targets supply neither this trajectory nor its envelope. This request independently declares every source-relative angle and activation.",
    "Relative hinge actuation follows actual local geometry with balanced cloth reactions. No world-space finish axis or fixed opposite region is assumed.",
    "No sewing or material gripper recipe is installed. A future combined request must separately bind all source rows, held/pending activation and any physical control anchors.",
    "Placement identity binds the entire declared starting state; its static clearance does not certify future contact, frame, triangle or hinge paths.",
    "The coefficient lower-bound witnesses cover only the declared finite dyadic retry grid and the rounded binary64 stiffness-times-activation policy.",
    "External actuator stiffness and control time are research inputs, not calibrated textile mechanics, continuous actuator work, settling or material acceptance.",
    "No material-side inference, binding wrap/stitch-down/apex completion, cuff turning, garment or application acceptance is granted.",
]


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _number(value, *, minimum, maximum, strict_minimum=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < minimum or value > maximum or strict_minimum and value == minimum):
        raise ValueError("Finite bounded raw non-Boolean control number required")
    converted = float(value)
    if Fraction(converted) != Fraction(value):
        raise ValueError("Control number must be exactly representable as binary64")
    return converted


def _rational(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "roundedBinary64": float(value)}


def _request(request, rails):
    fields = {"profile", "accepted", "durationSeconds", "initialSubdivisions", "maxDepth",
              "railOrder", "knots", "gripperPolicy"}
    if (type(request) is not dict or set(request) != fields or request["profile"] != REQUEST_PROFILE
            or request["accepted"] is not False or request["gripperPolicy"] != GRIPPER_POLICY):
        raise ValueError("Explicit unaccepted fold-only source schedule request required")
    duration = _number(request["durationSeconds"], minimum=0, maximum=MAX_DURATION_SECONDS, strict_minimum=True)
    subdivisions, depth = request["initialSubdivisions"], request["maxDepth"]
    if (type(subdivisions) is not int or not 1 <= subdivisions <= 4096 or subdivisions & (subdivisions - 1)
            or type(depth) is not int or not 0 <= depth <= 30 or subdivisions.bit_length() - 1 + depth > 40):
        raise ValueError("Explicit dyadic subdivisions and retry depth within the 2^40 grid required")
    denominator = subdivisions * 2**depth
    minimum_step = Fraction(duration) / denominator
    if float(minimum_step) <= 0:
        raise ValueError("Positive minimum physical timestep must not underflow")
    expected_order = [{key: rail[key] for key in ("sourcePathName", "referenceRotationRegion")} for rail in rails]
    if _encoded(request["railOrder"]) != _encoded(expected_order):
        raise ValueError("Complete ordered source rail and reference-region identity required")
    knots = request["knots"]
    if type(knots) is not list or not 2 <= len(knots) <= 65:
        raise ValueError("Two through 65 complete source control knots required")
    fractions, angles, activations = [], [], []
    for knot in knots:
        if (type(knot) is not dict or set(knot) != {"fraction", "sourceRightHandAnglesRadians", "activation"}
                or type(knot["fraction"]) not in (int, float)):
            raise ValueError("Explicit raw fraction, source-relative angles and activation required")
        fraction = Fraction(_number(knot["fraction"], minimum=0, maximum=1))
        if (fraction * subdivisions).denominator != 1 or fractions and fraction <= fractions[-1]:
            raise ValueError("Strictly increasing source knots on the initial dyadic grid required")
        values = []
        for key, is_activation in (("sourceRightHandAnglesRadians", False), ("activation", True)):
            raw = knot[key]
            if type(raw) is not list or len(raw) != len(rails):
                raise ValueError("Exactly one explicit value per ordered source rail required")
            row = [_number(value, minimum=0 if is_activation else -ANGLE_LIMIT,
                           maximum=1 if is_activation else ANGLE_LIMIT) for value in raw]
            if not is_activation and any(abs(value) >= ANGLE_LIMIT for value in row):
                raise ValueError("Source angles must remain strictly inside the principal branch")
            values.append(row)
        fractions.append(fraction)
        angles.append(values[0])
        activations.append(values[1])
    if fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError("Complete source schedule must span exactly [0,1]")
    if any(angles[0]) or any(activations[0]):
        raise ValueError("Initial flat reference targets and disengaged fold controls required")
    return duration, subdivisions, depth, denominator, minimum_step, fractions, angles, activations


def _minimum_activation(fractions, activation, column, denominator):
    """Enumerate monotone interval extrema on the declared discrete grid."""
    candidates = []
    increment = Fraction(1, denominator)
    for lower, upper, first, last in zip(fractions, fractions[1:], activation, activation[1:]):
        a, b = Fraction(first[column]), Fraction(last[column])
        if not a and not b:
            continue
        probes = [lower, upper]
        if upper - lower > increment:
            if not a:
                probes.append(lower + increment)
            if not b:
                probes.append(upper - increment)
        for fraction in probes:
            amount = (fraction - lower) / (upper - lower)
            exact = (1 - amount) * a + amount * b
            if exact > 0:
                numerical = float(exact)
                if numerical <= 0:
                    raise ValueError("Positive interpolated activation underflows on the declared retry grid")
                candidates.append((numerical, fraction, exact))
    return min(candidates, key=lambda item: (item[0], item[1])) if candidates else None


def build_binding_control_schedule(source, fold_descriptor, placement_descriptor, request):
    """Expand source rail controls without installing or executing them."""
    inputs = (source, fold_descriptor, placement_descriptor, request)
    before = [_bounded(value) for value in inputs]
    fold = validate_binding_fold_control(source, fold_descriptor)
    placement = validate_binding_placement(source, placement_descriptor)
    rails = fold["rails"]
    duration, subdivisions, depth, denominator, minimum_step, fractions, angles, activation = _request(request, rails)
    hinges, stiffness = fold["foldActuation"]["hinges"], fold["foldActuation"]["stiffnessJoules"]
    bindings, seen, minimum_witnesses = [], [], []
    knots = [{"fraction": float(fraction), "targetsRadians": [None] * len(hinges),
              "activation": [None] * len(hinges)} for fraction in fractions]
    for column, rail in enumerate(rails):
        region_sign = -1 if rail["referenceRotationRegion"] == "body" else 1
        minimum = _minimum_activation(fractions, activation, column, denominator)
        indices, signs = [], []
        for segment in rail["segments"]:
            index = segment["actuatorIndex"]
            sign = region_sign * segment["nativeAngleSign"]
            if index in seen or hinges[index] != segment["nativeOrderedHingeCanonical"]:
                raise ValueError("Unique complete source-to-native actuator incidence required")
            seen.append(index)
            indices.append(index)
            signs.append(sign)
            for knot, targets, weights in zip(knots, angles, activation):
                knot["targetsRadians"][index] = targets[column] * sign
                knot["activation"][index] = weights[column]
            witness = {"actuatorIndex": index, "everActive": minimum is not None}
            if minimum is not None:
                minimum_value, minimum_fraction, exact_activation = minimum
                exact_coefficient = Fraction(stiffness[index]) * Fraction(minimum_value)
                coefficient = float(exact_coefficient)
                if not math.isfinite(coefficient) or coefficient <= 0:
                    raise ValueError("Positive active coefficient underflows on the declared retry grid")
                witness.update(fraction=_rational(minimum_fraction), exactActivation=_rational(exact_activation),
                    numericalActivation=minimum_value, exactCoefficientFromRoundedActivationJoules=_rational(exact_coefficient),
                    numericalCoefficientJoules=coefficient,
                    coefficientRoundingResidualJoules=_rational(Fraction(coefficient) - exact_coefficient))
            minimum_witnesses.append(witness)
        bindings.append({"sourcePathName": rail["sourcePathName"], "sourcePathSha256": rail["sourcePathSha256"],
            "referenceRotationRegion": rail["referenceRotationRegion"], "actuatorIndices": indices,
            "sourceRightHandToNativeSigns": signs,
            "initialPlacedChainMeters": [copy.deepcopy(placement["placedMeters"][v]) for v in rail["chainVerticesCanonical"]]})
    if seen != list(range(len(hinges))):
        raise ValueError("Complete ordered source-to-native actuator coverage required")
    schedule = {"profile": "fold-angle-activation-v1", "hinges": copy.deepcopy(hinges), "knots": knots}
    # This constructs only an immutable parameter sampler, never a force model.
    FoldControlSchedule(schedule, subdivisions, hinges=hinges)
    result = {"profile": PROFILE, "accepted": False, "solverReady": False, "executable": False,
        "sourceSha256": hashlib.sha256(before[0]).hexdigest(), "baseUnitSha256": fold["baseUnitSha256"],
        "foldDescriptorSha256": hashlib.sha256(before[1]).hexdigest(),
        "placementDescriptorSha256": hashlib.sha256(before[2]).hexdigest(),
        "nativeTopologySha256": fold["nativeTopologySha256"], "requestSha256": hashlib.sha256(before[3]).hexdigest(),
        "instanceId": fold["instanceId"], "request": copy.deepcopy(request), "durationSeconds": duration,
        "initialSubdivisions": subdivisions, "maxDepth": depth, "retryGridDenominator": denominator,
        "minimumStepSeconds": _rational(minimum_step), "gripperPolicy": GRIPPER_POLICY,
        "sewingControlsInstalled": False, "gripperControlsInstalled": False, "constructionPhaseCompleted": False,
        "staticFoldReferenceTargetsUsedForMotion": False,
        "staticFoldReferenceTargetsRadians": copy.deepcopy(fold["foldActuation"]["targetAnglesRadians"]),
        "railBindings": bindings, "foldStiffnessJoules": copy.deepcopy(stiffness), "foldControlSchedule": schedule,
        "knotTimesSeconds": [_rational(Fraction(duration) * fraction) for fraction in fractions],
        "minimumActiveCoefficientWitnesses": minimum_witnesses,
        "coefficientPolicy": "rounded-binary64-stiffness-times-activation-v1",
        "parameterOrder": "target-first-at-old-activation-then-activation-at-new-target",
        "limitations": list(LIMITATIONS)}
    if [_bounded(value) for value in inputs] != before:
        raise ValueError("Source fold schedule inputs changed during derivation")
    _bounded(result)
    return copy.deepcopy(result)


def validate_binding_control_schedule(source, fold_descriptor, placement_descriptor, descriptor):
    """Require exact complete static rederivation, including all identities."""
    claimed = _bounded(descriptor)
    if type(descriptor) is not dict or "request" not in descriptor:
        raise ValueError("Complete source fold schedule descriptor required")
    expected = build_binding_control_schedule(source, fold_descriptor, placement_descriptor, descriptor["request"])
    if claimed != _encoded(expected):
        raise ValueError("Source fold schedule differs from complete rederivation")
    return expected
