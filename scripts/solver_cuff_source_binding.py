"""Strict correspondence for the declared five-fabric source cuff unit.

This verifies captured source geometry, registrations and the phase-row
partition. It does not execute phases, certify a placement, supply gate
witnesses, or turn a declaration into an accepted garment simulation.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

from solver_engine_source_namespace import ENGINE_FILES, engine_source_root, load_engine_modules
from solver_cuff_construction import POLICY, build_cuff_phase_plan


PROFILE = "source-cuff-construction-unit-v1"
SOURCE_HELPER_FILES = ("solver_cuff_source_binding.py", "solver_engine_source_namespace.py",
                       "solver_cuff_construction.py", "spike-full-shirt.py")
MAX_SOURCE_BYTES = 16 * 1024 ** 2


def _encoded(value):
    try:
        content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("Finite bounded JSON cuff source required") from error
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError("Cuff source JSON budget exceeded")
    return content


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _equal(actual, expected, label):
    # Canonical bytes distinguish booleans from integer indices and preserve
    # every row, sample and coefficient; Python container equality does not.
    if _encoded(actual) != _encoded(expected):
        raise ValueError(f"Cuff {label} differs from source rederivation")


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cuff_source_binding(source):
    """Return fresh row groups and phase rows after complete source checks.

    The ordered local triangles in ``sourceTemplates`` and ``instanceOffsets``
    are validated here, so callers may bind frame identities to them. Callers
    must still bind activation row IDs, control schedules and numerical states.
    """
    if not isinstance(source, dict) or source.get("profile") != PROFILE:
        raise ValueError("Exact source-cuff-construction-unit-v1 profile required")
    _encoded(source)
    try:
        return _validate(source)
    except (KeyError, TypeError, IndexError, StopIteration, AttributeError, OverflowError) as error:
        raise ValueError("Incomplete or malformed source cuff unit") from error


def _validate(source):
    if (source.get("accepted") is not False or source.get("solverReady") is not False
            or source.get("units") != "m" or source.get("sourceArcUnits") != "mm"
            or source.get("side") not in ("left", "right")):
        raise ValueError("Unaccepted cuff declaration in metre/mm source units required")
    pattern, construction = source["sourcePattern"], source["sourceConstruction"]
    if not isinstance(pattern, dict) or not isinstance(construction, dict):
        raise ValueError("Captured pattern and construction objects required")
    # Bound canonical arrays before any geometry, hashing of individual mesh
    # objects, or numerical allocation in the trusted source validators.
    if (not isinstance(source["restMeters"], list) or not 3 <= len(source["restMeters"]) <= 25000
            or not isinstance(source["triangles"], list) or not 3 <= len(source["triangles"]) <= 150000
            or len(source["triangles"]) % 3):
        raise ValueError("Cuff canonical geometry budget exceeded")
    provenance = source["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("Captured source provenance required")
    engine = engine_source_root(__file__)
    helper = Path(__file__).resolve().with_name("spike-full-shirt.py")
    if not helper.is_file() or helper.resolve().parent != Path(__file__).resolve().parent:
        raise ValueError("Adjacent captured registration helper required")
    dependencies = {"services/engine/" + name: engine / name for name in ENGINE_FILES}
    dependencies["scripts/spike-full-shirt.py"] = helper
    checked_hashes = {name: _file_digest(path) for name, path in dependencies.items()}
    captured_hashes = provenance.get("sourceDigests")
    if (not isinstance(captured_hashes, dict)
            or any(captured_hashes.get(name) != value for name, value in checked_hashes.items())):
        raise ValueError("Captured cuff engine or registration helper differs; explicit migration required")
    assembly_module, cloth_module, embedded_module = load_engine_modules(
        __file__, "assembly", "cloth_domain", "embedded_constraints")
    rebuilt_pattern, inventory = assembly_module.compile_inventory(_encoded(pattern), construction)
    _equal(rebuilt_pattern, pattern, "pattern")
    _equal(source["sourceInventory"], inventory, "inventory")
    assembly = assembly_module.compile_assembly(pattern, inventory)
    _equal(source["sourceAssembly"], assembly, "assembly")
    side = source["side"]
    plan = build_cuff_phase_plan(pattern, inventory, assembly, side=side, policy=POLICY)
    _equal(source["phasePlan"], plan, "phase plan")
    selected = set(plan["requiredInstanceIds"])
    instances = [item for item in inventory["instances"] if item["id"] in selected]
    if len(instances) != 5:
        raise ValueError("All five source fabric instances required")
    _equal(source["instances"], instances, "physical instance order")
    operation_ids = {item["id"] for item in plan["sourceOperations"]}
    operations = [item for item in assembly["operations"] if item["id"] in operation_ids]
    if len(operations) != 7:
        raise ValueError("All seven source cuff operations required")
    panels = {item["id"]: item for item in pattern["panels"]}
    templates = source["sourceTemplates"]
    if not isinstance(templates, dict) or set(templates) != {item["templateId"] for item in instances}:
        raise ValueError("Exact cuff source template set required")
    validations = {name: cloth_module.validate_cloth_domain(panels[name], mesh)
                   for name, mesh in templates.items()}
    _equal(source["meshValidation"], validations, "mesh validation")
    rest, triangles, offsets = [], [], {}
    for instance in instances:
        mesh = templates[instance["templateId"]]
        offset = len(rest)
        offsets[instance["id"]] = offset
        rest.extend([[point[0] * .001, point[1] * .001, 0.0] for point in mesh["restPositions"]])
        triangles.extend(vertex + offset for face in mesh["triangles"] for vertex in face)
    _equal(source["instanceOffsets"], offsets, "instance offsets")
    _equal(source["restMeters"], rest, "unchanged template rest positions")
    _equal(source["triangles"], triangles, "unchanged template triangles")
    sources = {item["id"]: {"panel": panels[item["templateId"]], "mesh": templates[item["templateId"]]}
               for item in instances}
    specification = importlib.util.spec_from_file_location("_bound_cuff_registration_helper", helper)
    registration_helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(registration_helper)
    registrations = registration_helper.shirt_embedded_registrations(operations, sources)
    bundle = embedded_module.build_embedded_constraints(sources, registrations)
    _equal(source["embeddedConstraints"], bundle, "full embedded sewing bundle")
    rows = bundle["constraints"]
    if len(rows) != 40 or len({row["complianceMPerN"] for row in rows}) != 1:
        raise ValueError("Exactly 40 source rows with homogeneous compliance required")
    groups = {}
    for index, row in enumerate(rows):
        key = row["registrationId"], row["memberIndex"]
        groups.setdefault(key, []).append(index)
    expected_selectors = {(item["registrationId"], item["memberIndex"]) for item in plan["starRowSelectors"]}
    if len(groups) != 8 or set(groups) != expected_selectors:
        raise ValueError("Exact eight source star selectors required")
    for indices in groups.values():
        _equal([rows[index]["fraction"] for index in indices], [0.0, .25, .5, .75, 1.0],
               "five distinct source fractions per selector")
    phase_rows, claimed = {}, set()
    for phase in plan["phases"]:
        indices = [index for selector in phase["activatesStarRows"]
                   for index in groups[(selector["registrationId"], selector["memberIndex"])]]
        if claimed.intersection(indices) or len(set(indices)) != len(indices):
            raise ValueError("Cuff phase rows overlap")
        claimed.update(indices)
        phase_rows[phase["id"]] = indices
    if claimed != set(range(40)):
        raise ValueError("Cuff phase rows omit source samples")
    _equal(source["phaseConstraintRows"], phase_rows, "complete phase row partition")
    excluded = [item for item in assembly["operations"] if item["id"] not in operation_ids]
    expected_fields = {
        "selectedOperationIds": [item["id"] for item in operations],
        "excludedOperationIds": [item["id"] for item in excluded],
        "unexecutedOtherOperationsTouchingUnit": [item["id"] for item in excluded
            if any(member["instanceId"] in selected for member in item["participants"])],
        "unexecutedClosuresTouchingUnit": [item["id"] for item in assembly["closures"]
            if any(member["instanceId"] in selected for member in item["participants"])],
        "sourceFreeBoundaries": [item for item in assembly["freeBoundaries"] if item["instanceId"] in selected]}
    for name, value in expected_fields.items():
        _equal(source[name], value, name)
    provenance_fields = {
        "classification": "declarative cuff construction research input; no executed assembly",
        "patternSha256": inventory["patternDigest"], "constructionSha256": inventory["constructionDigest"],
        "assemblySha256": _digest(assembly), "phasePlanSha256": _digest(plan),
        "consumedParentFields": ["sourcePattern", "sourceConstruction", "provenance"],
        "parentNumericalGeometryUsed": False, "qualityRefinement": True}
    for name, value in provenance_fields.items():
        _equal(provenance[name], value, "provenance " + name)
    if (not isinstance(provenance.get("sourceInputSha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", provenance["sourceInputSha256"]) is None
            or type(provenance.get("maxEdgeMm")) not in (int, float)
            or not math.isfinite(provenance["maxEdgeMm"]) or not 10 <= provenance["maxEdgeMm"] <= 120):
        raise ValueError("Bounded captured parent and meshing metadata required")
    if any(_file_digest(path) != checked_hashes[name] for name, path in dependencies.items()):
        raise ValueError("Engine or registration helper changed during source binding")
    return {"profile": "cuff-source-binding-v1", "sourceProfile": PROFILE,
            "accepted": False, "solverReady": False, "sourceCorrespondenceValidated": True, "side": side,
            "scope": "Source geometry, unchanged registrations and declarative phase-row correspondence only; no executed phases, gate witnesses, placement or physical acceptance.",
            "rowGroups": [{"registrationId": key[0], "memberIndex": key[1], "rowIndices": indices}
                          for key, indices in groups.items()],
            "phaseConstraintRows": phase_rows,
            "counts": {"fabricInstances": 5, "sourceOperations": 7, "starRowSelectors": 8,
                       "constraints": 40, "phases": len(plan["phases"]), "vertices": len(rest),
                       "triangles": len(triangles) // 3},
            "patternSha256": inventory["patternDigest"], "constructionSha256": inventory["constructionDigest"],
            "assemblySha256": _digest(assembly), "phasePlanSha256": _digest(plan),
            "embeddedConstraintsSha256": _digest(bundle), "checkedSourceDigests": checked_hashes,
            "provenanceLimitations": [
                "The parent input byte digest, other historical generator/phase-helper digests and runtime are captured metadata; the parent bytes are not provided to this validator.",
                "Engine and registration-helper bytes are checked against captured digests. Geometry is checked by the source cloth-domain validator; no remeshing or new exact-real geometry proof is claimed.",
                "Exact embedded coefficient rederivation requires a compatible numerical runtime. A different BLAS/platform may reject a valid capture; coefficient differences are never silently tolerated or rewritten.",
                "An activation schedule and numerical phase completion require separate validation; this declaration does not provide either."]}
