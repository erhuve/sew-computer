import copy
import hashlib
import json
import math

import numpy as np

from cloth_domain import validate_cloth_domain


MAX_CONSTRAINTS = 32768
MAX_INSTANCES = 128


def _number(value, lower, upper, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"Invalid {label}")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sample_stitch_path(mesh, path_name, arc_mm):
    paths = [path for path in mesh.get("stitchPaths", []) if path.get("name") == path_name]
    if len(paths) != 1:
        raise ValueError("Unknown or duplicate stitching path")
    path = paths[0]
    _number(arc_mm, 0, path["lengthMm"], "stitch arc")
    samples = path["samples"]
    for segment in path["segments"]:
        first_sample, second_sample = [samples[index] for index in segment["samples"]]
        first_arc, second_arc = first_sample["arcMm"], second_sample["arcMm"]
        if first_arc - 1e-9 <= arc_mm <= second_arc + 1e-9:
            fraction = min(1, max(0, (arc_mm - first_arc) / (second_arc - first_arc)))
            position = np.asarray(first_sample["restPosition"], dtype=float) * (1 - fraction) + np.asarray(second_sample["restPosition"], dtype=float) * fraction
            triangle = mesh["triangles"][segment["triangle"]]
            vertices = np.asarray([mesh["restPositions"][index] for index in triangle], dtype=float)
            system = np.vstack((vertices.T, np.ones(3)))
            weights = np.linalg.solve(system, np.append(position, 1.0))
            if not np.isfinite(weights).all() or min(weights) < -1e-8:
                raise ValueError("Stitch sample is outside its cloth triangle")
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            return {"pathName": path_name, "arcMm": arc_mm, "restPositionMm": position.tolist(),
                    "weights": [{"vertex": index, "weight": float(weight)} for index, weight in zip(triangle, weights) if weight > 0]}
    raise ValueError("Stitch arc is not covered by cloth path segments")


def _derive(sources, registrations):
    if not isinstance(sources, dict) or not 1 <= len(sources) <= MAX_INSTANCES:
        raise ValueError("Invalid physical instance budget")
    identities = {}
    for instance_id, source in sources.items():
        if not isinstance(instance_id, str) or not 1 <= len(instance_id) <= 160 or not isinstance(source, dict):
            raise ValueError("Invalid physical source identity")
        validate_cloth_domain(source["panel"], source["mesh"])
        identities[instance_id] = {"templateId": source["mesh"]["templateId"], "panelDigest": source["mesh"]["sourcePanelDigest"],
                                   "meshDigest": _digest(source["mesh"]), "vertexCount": len(source["mesh"]["restPositions"])}
    if not isinstance(registrations, list) or not 1 <= len(registrations) <= 512:
        raise ValueError("Invalid sewing registration budget")
    identifiers = set()
    physical_registrations = set()
    constraints = []
    for registration in registrations:
        if not isinstance(registration, dict) or set(registration) != {"id", "members", "sampleCount", "complianceMPerN"}:
            raise ValueError("Invalid sewing registration fields")
        identifier, members, count = registration["id"], registration["members"], registration["sampleCount"]
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 160 or identifier in identifiers:
            raise ValueError("Invalid sewing registration identity")
        identifiers.add(identifier)
        compliance = _number(registration["complianceMPerN"], 0, 1000, "sewing compliance")
        if type(count) is not int or not 2 <= count <= 256 or not isinstance(members, list) or not 2 <= len(members) <= 8:
            raise ValueError("Invalid sewing registration sampling")
        if len(constraints) + count * (len(members) - 1) > MAX_CONSTRAINTS:
            raise ValueError("Sewing constraint budget exceeded")
        member_ids = set()
        for member in members:
            if not isinstance(member, dict) or set(member) != {"instanceId", "pathName", "startArcMm", "endArcMm", "direction"}:
                raise ValueError("Invalid sewing member fields")
            if member["instanceId"] not in sources or member["direction"] not in ("forward", "reverse"):
                raise ValueError("Unknown sewing instance or direction")
            path = next((path for path in sources[member["instanceId"]]["mesh"]["stitchPaths"] if path["name"] == member["pathName"]), None)
            if path is None:
                raise ValueError("Unknown sewing source path")
            start = _number(member["startArcMm"], 0, path["lengthMm"], "sewing interval start")
            end = _number(member["endArcMm"], 0, path["lengthMm"], "sewing interval end")
            if end <= start:
                raise ValueError("Sewing interval must have positive source length")
            identity = (member["instanceId"], member["pathName"], start, end)
            if identity in member_ids:
                raise ValueError("Duplicate physical sewing member")
            member_ids.add(identity)
        direct = tuple(sorted((member["instanceId"], member["pathName"], member["startArcMm"], member["endArcMm"], member["direction"]) for member in members))
        reversed_registration = tuple(sorted((member["instanceId"], member["pathName"], member["startArcMm"], member["endArcMm"], "reverse" if member["direction"] == "forward" else "forward") for member in members))
        physical_identity = min(direct, reversed_registration)
        if physical_identity in physical_registrations:
            raise ValueError("Duplicate physical sewing registration")
        physical_registrations.add(physical_identity)
        for sample_index in range(count):
            fraction = sample_index / (count - 1)
            samples = []
            for member in members:
                local_fraction = fraction if member["direction"] == "forward" else 1 - fraction
                arc = member["startArcMm"] * (1 - local_fraction) + member["endArcMm"] * local_fraction
                sample = sample_stitch_path(sources[member["instanceId"]]["mesh"], member["pathName"], arc)
                samples.append({"instanceId": member["instanceId"], **sample})
            for member_index in range(1, len(samples)):
                terms = {}
                for sign, sample in ((1, samples[0]), (-1, samples[member_index])):
                    for weight in sample["weights"]:
                        key = (sample["instanceId"], weight["vertex"])
                        terms[key] = terms.get(key, 0.0) + sign * weight["weight"]
                constraints.append({"registrationId": identifier, "fraction": fraction, "memberIndex": member_index,
                    "sourceSamples": [samples[0], samples[member_index]], "complianceMPerN": compliance,
                    "terms": [{"instanceId": key[0], "vertex": key[1], "coefficient": value} for key, value in sorted(terms.items()) if value != 0]})
    return {"schemaVersion": 1, "units": "m", "sourceArcUnits": "mm", "kind": "embedded-sewing-coupling",
            "topology": "independent-star", "sourceIdentities": identities, "registrations": copy.deepcopy(registrations),
            "constraints": constraints, "solverReady": False,
            "limitations": ["Only numerical sewing coupling; no cloth elasticity, bend, contact, fold or material simulation.",
                            "Uniform normalized source-arc registration preserves unequal rest lengths; gather settling is not validated.",
                            "Physical instance inventory, placement and assembly sequence require separate validation."]}


def build_embedded_constraints(sources, registrations):
    return _derive(sources, registrations)


def validate_embedded_constraints(sources, bundle):
    if not isinstance(bundle, dict):
        raise ValueError("Invalid embedded sewing bundle")
    expected = _derive(sources, bundle.get("registrations"))
    if _digest(bundle) != _digest(expected):
        raise ValueError("Embedded sewing differs from captured source registrations")
    return {"sourceCorrespondenceAccepted": True, "constraintCount": len(expected["constraints"]), "solverReady": False}


def _state(bundle, positions, inverse_masses=None):
    if not isinstance(bundle, dict) or bundle.get("units") != "m" or bundle.get("kind") != "embedded-sewing-coupling" or bundle.get("solverReady") is not False:
        raise ValueError("Invalid embedded coupling classification")
    identities, constraints = bundle.get("sourceIdentities"), bundle.get("constraints")
    if not isinstance(identities, dict) or not 1 <= len(identities) <= MAX_INSTANCES or not isinstance(constraints, list) or not 1 <= len(constraints) <= MAX_CONSTRAINTS:
        raise ValueError("Invalid coupling resource budget")
    if not isinstance(positions, dict) or set(positions) != set(identities) or inverse_masses is not None and (not isinstance(inverse_masses, dict) or set(inverse_masses) != set(identities)):
        raise ValueError("Coupling instance state mismatch")
    copied, masses = {}, {}
    for identifier, identity in identities.items():
        count = identity["vertexCount"]
        if type(count) is not int or not 3 <= count <= 60000:
            raise ValueError("Invalid coupling vertex budget")
        array = np.asarray(positions[identifier], dtype=float)
        if array.shape != (count, 3) or not np.isfinite(array).all() or np.max(np.abs(array)) > 1e6:
            raise ValueError("Invalid coupling positions in metres")
        copied[identifier] = array.copy()
        if inverse_masses is not None:
            inverse = np.asarray(inverse_masses[identifier], dtype=float)
            if inverse.shape != (count,) or not np.isfinite(inverse).all() or np.any(inverse < 0) or np.any(inverse > 1e12):
                raise ValueError("Invalid inverse masses")
            masses[identifier] = inverse.copy()
    for constraint in constraints:
        if not isinstance(constraint, dict):
            raise ValueError("Invalid sparse sewing constraint")
        _number(constraint.get("complianceMPerN"), 0, 1000, "sewing compliance")
        terms = constraint.get("terms")
        if not isinstance(terms, list) or len(terms) > 6:
            raise ValueError("Invalid sparse sewing terms")
        keys = set()
        for term in terms:
            if not isinstance(term, dict) or term.get("instanceId") not in copied:
                raise ValueError("Unknown sparse sewing instance")
            vertex, identifier = term.get("vertex"), term["instanceId"]
            if type(vertex) is not int or not 0 <= vertex < len(copied[identifier]):
                raise ValueError("Invalid sparse sewing vertex")
            key = (identifier, vertex)
            if key in keys:
                raise ValueError("Duplicate sparse sewing vertex")
            keys.add(key)
            _number(term.get("coefficient"), -1, 1, "sparse sewing coefficient")
        if abs(sum(term["coefficient"] for term in terms)) > 1e-9:
            raise ValueError("Sewing coefficients must preserve translation")
    return copied, masses


def _residual(constraint, positions):
    return sum((term["coefficient"] * positions[term["instanceId"]][term["vertex"]] for term in constraint["terms"]), start=np.zeros(3))


def constraint_residuals(bundle, positions):
    copied, _ = _state(bundle, positions)
    return np.asarray([_residual(constraint, copied) for constraint in bundle["constraints"]])


def project_embedded_constraints(bundle, positions, inverse_masses, timestep_seconds, iterations=1, multipliers=None, target_offsets_m=None):
    copied, masses = _state(bundle, positions, inverse_masses)
    _number(timestep_seconds, 1e-8, 1, "coupling timestep")
    if type(iterations) is not int or not 1 <= iterations <= 1000 or iterations * len(bundle["constraints"]) > 2000000:
        raise ValueError("Invalid coupling iteration budget")
    constraints = bundle["constraints"]
    lambdas = np.zeros((len(constraints), 3)) if multipliers is None else np.asarray(multipliers, dtype=float).copy()
    if lambdas.shape != (len(constraints), 3) or not np.isfinite(lambdas).all():
        raise ValueError("Invalid coupling multipliers")
    targets = np.zeros((len(constraints), 3)) if target_offsets_m is None else np.asarray(target_offsets_m, dtype=float).copy()
    if targets.shape != (len(constraints), 3) or not np.isfinite(targets).all() or np.max(np.abs(targets)) > 100:
        raise ValueError("Invalid staged coupling targets in metres")
    originals = {identifier: array.copy() for identifier, array in copied.items()}
    immovable = set()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for iteration in range(iterations):
            for constraint_index, constraint in enumerate(constraints):
                compliance = constraint["complianceMPerN"] / timestep_seconds ** 2
                effective_mass = sum(term["coefficient"] ** 2 * masses[term["instanceId"]][term["vertex"]] for term in constraint["terms"])
                residual = _residual(constraint, copied) - targets[constraint_index]
                if effective_mass == 0:
                    if np.linalg.norm(residual) > 1e-12:
                        immovable.add(constraint_index)
                    continue
                increment = -(residual + compliance * lambdas[constraint_index]) / (effective_mass + compliance)
                lambdas[constraint_index] += increment
                for term in constraint["terms"]:
                    copied[term["instanceId"]][term["vertex"]] += masses[term["instanceId"]][term["vertex"]] * term["coefficient"] * increment
    if not all(np.isfinite(array).all() for array in copied.values()) or not np.isfinite(lambdas).all():
        raise ValueError("Nonfinite embedded coupling state")
    residuals = np.asarray([_residual(constraint, copied) for constraint in constraints])
    return {"positions": copied, "multipliers": lambdas, "residualsM": residuals, "targetResidualsM": residuals - targets,
            "maxCorrectionM": max(float(np.max(np.linalg.norm(array - originals[identifier], axis=1))) for identifier, array in copied.items()),
            "immovableConstraints": sorted(immovable), "solverReady": False}
