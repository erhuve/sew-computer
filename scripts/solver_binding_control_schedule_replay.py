"""Independent static audit of source-bound fold targets and activation.

Only the standard library and the two independently authored static auditors
are used. This does not import a parameter sampler, force model or executor.
Pattern/split geometry still requires the separate strict source validator.
"""

from fractions import Fraction as F
import hashlib
import json
import math

from solver_binding_fold_replay import verify_binding_fold_control
from solver_binding_placement_replay import verify_binding_placement


PROFILE = "independent-binding-fold-schedule-verification-v1"
REQUIRED_HELPER_FILES = ("solver_binding_fold_replay.py", "solver_binding_placement_replay.py")
MAX_BYTES = 24 * 1024**2
ANGLE_LIMIT = math.pi - 1e-8
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


def _encoded(value):
    pending, remaining, text_bytes = [(value, 0)], 1000000, 0
    while pending:
        item, depth = pending.pop()
        remaining -= 1
        if remaining < 0 or depth > 40:
            raise ValueError("Independent schedule structure budget exceeded")
        if type(item) is dict:
            if any(type(key) is not str for key in item) or len(item) * 2 > remaining:
                raise ValueError("Bounded raw string-keyed JSON objects required")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif type(item) is list:
            if len(item) > remaining:
                raise ValueError("Independent schedule array budget exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > MAX_BYTES:
                raise ValueError("Independent schedule text budget exceeded")
            try:
                text_bytes += len(item.encode())
            except UnicodeEncodeError as error:
                raise ValueError("UTF-8 JSON strings required") from error
            if text_bytes > MAX_BYTES:
                raise ValueError("Independent schedule text budget exceeded")
        elif type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Independent schedule integer budget exceeded")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Finite raw JSON numbers required")
        elif item is not None and type(item) is not bool:
            raise ValueError("Raw JSON schedule values required")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_BYTES:
        raise ValueError("Independent schedule byte budget exceeded")
    return encoded


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Bounded finite non-Boolean control number required")
    numerical = float(value)
    if F(numerical) != F(value):
        raise ValueError("Exactly representable binary64 input required")
    return numerical


def _rat(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "roundedBinary64": float(value)}


def _request(request, rails):
    fields = {"profile", "accepted", "durationSeconds", "initialSubdivisions", "maxDepth",
              "railOrder", "knots", "gripperPolicy"}
    if (type(request) is not dict or set(request) != fields
            or request["profile"] != "source-binding-fold-schedule-request-v1"
            or request["accepted"] is not False
            or request["gripperPolicy"] != "none-balanced-relative-folds-v1"):
        raise ValueError("Complete unaccepted fold-only schedule request required")
    duration = _number(request["durationSeconds"], 0, 3600)
    subdivisions, depth = request["initialSubdivisions"], request["maxDepth"]
    if (duration <= 0 or type(subdivisions) is not int or not 1 <= subdivisions <= 4096
            or subdivisions & (subdivisions - 1) or type(depth) is not int or not 0 <= depth <= 30
            or subdivisions.bit_length() - 1 + depth > 40):
        raise ValueError("Positive duration and bounded dyadic retry grid required")
    denominator = subdivisions << depth
    minimum_time = F(duration) / denominator
    if float(minimum_time) <= 0:
        raise ValueError("Minimum positive physical step underflows binary64")
    order = [{"sourcePathName": rail["sourcePathName"],
              "referenceRotationRegion": rail["referenceRotationRegion"]} for rail in rails]
    if _encoded(request["railOrder"]) != _encoded(order):
        raise ValueError("Complete source rail order and reference-region binding required")
    raw_knots = request["knots"]
    if type(raw_knots) is not list or not 2 <= len(raw_knots) <= 65:
        raise ValueError("Two through 65 complete knots required")
    indices, angles, activation = [], [], []
    for knot in raw_knots:
        if type(knot) is not dict or set(knot) != {"fraction", "sourceRightHandAnglesRadians", "activation"}:
            raise ValueError("Complete raw source knot required")
        fraction = F(_number(knot["fraction"], 0, 1))
        initial_index = fraction * subdivisions
        if initial_index.denominator != 1:
            raise ValueError("Knots must lie on the initial dyadic grid")
        grid_index = int(initial_index) << depth
        if indices and grid_index <= indices[-1]:
            raise ValueError("Strictly ordered knot fractions required")
        indices.append(grid_index)
        for key, output, limits in (("sourceRightHandAnglesRadians", angles, (-ANGLE_LIMIT, ANGLE_LIMIT)),
                                    ("activation", activation, (0, 1))):
            values = knot[key]
            if type(values) is not list or len(values) != len(rails):
                raise ValueError("Exactly one control per source rail required")
            row = [_number(value, *limits) for value in values]
            if key == "sourceRightHandAnglesRadians" and any(abs(value) >= ANGLE_LIMIT for value in row):
                raise ValueError("Targets must be strictly inside the principal branch")
            output.append(row)
    if indices[0] != 0 or indices[-1] != denominator or any(angles[0]) or any(activation[0]):
        raise ValueError("Complete [0,1] schedule must start flat and disengaged")
    return duration, subdivisions, depth, denominator, minimum_time, indices, angles, activation


def _minimum_on_grid(indices, activation, column, denominator):
    """Audit a linear sequence by its integer endpoints and zero neighbours.

    Exact interpolation is monotone on each integer interval; nearest-even
    conversion is monotone too. Thus these at most four candidates per interval
    prove the minimum positive value without traversing up to 2**40 samples.
    Canonical witness ties use the earliest of these extremal candidates.
    """
    candidates = []
    for segment in range(len(indices) - 1):
        lo, hi = indices[segment:segment + 2]
        first, last = F(activation[segment][column]), F(activation[segment + 1][column])
        if first == last == 0:
            continue
        points = {lo, hi}
        if hi - lo > 1:
            if first == 0:
                points.add(lo + 1)
            if last == 0:
                points.add(hi - 1)
        for index in points:
            exact = ((hi-index)*first + (index-lo)*last) / (hi-lo)
            if exact > 0:
                rounded = float(exact)
                if rounded <= 0:
                    raise ValueError("Positive sampled activation underflows binary64")
                candidates.append((rounded, index, exact))
    if not candidates:
        return None
    numerical, index, exact = min(candidates, key=lambda item: (item[0], item[1]))
    return numerical, F(index, denominator), exact


def verify_binding_control_schedule(source, fold_descriptor, placement_descriptor, descriptor):
    """Require complete independent static reconstruction; never run motion."""
    inputs = (source, fold_descriptor, placement_descriptor, descriptor)
    before = [_encoded(value) for value in inputs]
    try:
        fold_audit = verify_binding_fold_control(source, fold_descriptor)
        placement_audit = verify_binding_placement(source, placement_descriptor)
        if type(descriptor) is not dict or "request" not in descriptor:
            raise ValueError("Complete static schedule descriptor required")
        rails = fold_descriptor["rails"]
        request = descriptor["request"]
        duration, subdivisions, depth, denominator, minimum_time, indices, angles, activation = _request(request, rails)
        hinges = fold_descriptor["foldActuation"]["hinges"]
        stiffness = fold_descriptor["foldActuation"]["stiffnessJoules"]
        count = len(hinges)
        knots = [{"fraction": float(F(index, denominator)), "targetsRadians": [None]*count,
                  "activation": [None]*count} for index in indices]
        bindings, witnesses, seen = [], [], []
        for column, rail in enumerate(rails):
            minimum = _minimum_on_grid(indices, activation, column, denominator)
            actuator_indices, signs = [], []
            for segment in rail["segments"]:
                index = segment["actuatorIndex"]
                source_hinge = segment["sourceDirectedHingeCanonical"]
                native_hinge = fold_descriptor["nativeTopology"]["hinges"][segment["nativeHingeIndex"]]
                # Derive parity from both ordered pairs, rather than trusting
                # the producer's nativeAngleSign or expanded target values.
                opposite_sign = 1 if native_hinge[:2] == source_hinge[:2] else -1
                edge_sign = 1 if native_hinge[2:] == source_hinge[2:] else -1
                sign = opposite_sign * edge_sign * (-1 if rail["referenceRotationRegion"] == "body" else 1)
                if index != len(seen) or hinges[index] != native_hinge:
                    raise ValueError("Complete ordered native actuator incidence required")
                seen.append(index)
                actuator_indices.append(index)
                signs.append(sign)
                for knot, target_row, activation_row in zip(knots, angles, activation):
                    knot["targetsRadians"][index] = sign * target_row[column]
                    knot["activation"][index] = activation_row[column]
                witness = {"actuatorIndex": index, "everActive": minimum is not None}
                if minimum is not None:
                    numerical_activation, fraction, exact_activation = minimum
                    exact_coefficient = F(stiffness[index]) * F(numerical_activation)
                    numerical_coefficient = float(exact_coefficient)
                    if not math.isfinite(numerical_coefficient) or numerical_coefficient <= 0:
                        raise ValueError("Positive sampled actuator coefficient underflows binary64")
                    witness.update(fraction=_rat(fraction), exactActivation=_rat(exact_activation),
                        numericalActivation=numerical_activation,
                        exactCoefficientFromRoundedActivationJoules=_rat(exact_coefficient),
                        numericalCoefficientJoules=numerical_coefficient,
                        coefficientRoundingResidualJoules=_rat(F(numerical_coefficient)-exact_coefficient))
                witnesses.append(witness)
            bindings.append({"sourcePathName": rail["sourcePathName"], "sourcePathSha256": rail["sourcePathSha256"],
                "referenceRotationRegion": rail["referenceRotationRegion"], "actuatorIndices": actuator_indices,
                "sourceRightHandToNativeSigns": signs,
                "initialPlacedChainMeters": [placement_descriptor["placedMeters"][v] for v in rail["chainVerticesCanonical"]]})
        if seen != list(range(count)):
            raise ValueError("Missing native controlled hinge")
        expected = {"profile": "source-binding-fold-schedule-v1", "accepted": False, "solverReady": False, "executable": False,
            "sourceSha256": hashlib.sha256(before[0]).hexdigest(), "baseUnitSha256": fold_audit["baseUnitSha256"],
            "foldDescriptorSha256": hashlib.sha256(before[1]).hexdigest(),
            "placementDescriptorSha256": hashlib.sha256(before[2]).hexdigest(),
            "nativeTopologySha256": fold_audit["nativeTopologySha256"], "requestSha256": _sha(request),
            "instanceId": fold_descriptor["instanceId"], "request": request, "durationSeconds": duration,
            "initialSubdivisions": subdivisions, "maxDepth": depth, "retryGridDenominator": denominator,
            "minimumStepSeconds": _rat(minimum_time), "gripperPolicy": "none-balanced-relative-folds-v1",
            "sewingControlsInstalled": False, "gripperControlsInstalled": False, "constructionPhaseCompleted": False,
            "staticFoldReferenceTargetsUsedForMotion": False,
            "staticFoldReferenceTargetsRadians": fold_descriptor["foldActuation"]["targetAnglesRadians"],
            "railBindings": bindings, "foldStiffnessJoules": stiffness,
            "foldControlSchedule": {"profile": "fold-angle-activation-v1", "hinges": hinges, "knots": knots},
            "knotTimesSeconds": [_rat(F(duration)*F(index, denominator)) for index in indices],
            "minimumActiveCoefficientWitnesses": witnesses,
            "coefficientPolicy": "rounded-binary64-stiffness-times-activation-v1",
            "parameterOrder": "target-first-at-old-activation-then-activation-at-new-target", "limitations": LIMITATIONS}
        if before[3] != _encoded(expected):
            raise ValueError("Independent source fold schedule differs from complete reconstruction")
        if before != [_encoded(value) for value in inputs]:
            raise ValueError("Independent schedule audit inputs changed")
        return {"profile": PROFILE, "verified": True, "accepted": False,
            "sourceSha256": expected["sourceSha256"], "baseUnitSha256": expected["baseUnitSha256"],
            "descriptorSha256": hashlib.sha256(before[3]).hexdigest(), "requestSha256": expected["requestSha256"],
            "foldDescriptorSha256": expected["foldDescriptorSha256"],
            "placementDescriptorSha256": expected["placementDescriptorSha256"],
            "nativeTopologySha256": expected["nativeTopologySha256"], "railCount": len(rails), "hingeCount": count,
            "knotCount": len(knots), "retryGridDenominator": denominator, "minimumStepSeconds": _rat(minimum_time),
            "everActiveHingeCount": sum(witness["everActive"] for witness in witnesses),
            "coefficientWitnessesSha256": _sha(witnesses), "foldAudit": fold_audit, "placementAudit": placement_audit,
            "sourceScope": "Independent complete source-rail/native/placement/request binding and exact physical-time witnesses. Minimum positive activation and RN(k*RN(alpha)) coefficients cover only the declared finite dyadic retry grid, using integer-interval extrema; no grid enumeration. Prerequisite audit scopes remain in foldAudit and placementAudit. Pattern/split validity requires the separate source validator. No controls installed, motion, captured admission, sewing/gripper recipe, continuous work, contact/hinge path, settling, material or construction acceptance."}
    except (KeyError, TypeError, IndexError, StopIteration, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Malformed or incomplete independent source schedule input") from error
