"""Build a source-derived five-fabric cuff research unit and explicit phase plan.

This consumes only the captured pattern and construction, not the parent's
numerical mesh or placement. It includes the sleeve, both cuff layers and both
opening bindings. No phase is executed and no binding/turning evidence is
invented. A fresh private output directory is required.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from solver_attempt_journal import strict_loads
from solver_process_budget import atomic_bytes, atomic_json, read_regular


ROOT = Path(__file__).resolve().parents[1]
ENGINE_FILES = ("shirt.py", "assembly.py", "cloth_domain.py", "meshing.py",
                "quality_meshing.py", "simulation_validation.py",
                "embedded_constraints.py", "inspection_gltf.py")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def generate(source_path, output, *, side, policy, max_edge_mm=60.):
    if type(max_edge_mm) not in (int, float) or not math.isfinite(max_edge_mm) or not 10 <= max_edge_mm <= 120:
        raise ValueError("Cuff unit mesh edge length must be finite and within 10..120 mm")
    paths = [Path(__file__), *(ROOT / "services/engine" / name for name in ENGINE_FILES),
             *(ROOT / "scripts" / name for name in (
                 "solver_cuff_construction.py", "solver_engine_source_namespace.py", "spike-full-shirt.py", "solver_attempt_journal.py",
                 "solver_process_budget.py", "solver-spike.requirements.txt"))]
    captured = {str(path.relative_to(ROOT)): path.read_bytes() for path in paths}
    source_bytes = read_regular(source_path, 8 * 1024 ** 2)
    source = strict_loads(source_bytes)
    if not isinstance(source, dict) or not isinstance(source.get("provenance"), dict):
        raise ValueError("Captured pattern and construction provenance required")
    pattern, construction = source.get("sourcePattern"), source.get("sourceConstruction")
    if not isinstance(pattern, dict) or not isinstance(construction, dict):
        raise ValueError("Captured pattern and construction required")
    pattern_bytes = encoded(pattern)
    if source["provenance"].get("patternSha256") != digest(pattern_bytes):
        raise ValueError("Captured pattern digest mismatch")
    previous_code = source["provenance"].get("sourceDigests", {})
    if not isinstance(previous_code, dict) or any(
            previous_code.get("services/engine/" + name) != digest(captured["services/engine/" + name])
            for name in ENGINE_FILES):
        raise ValueError("Captured source engine differs; explicit source migration required")
    sys.path.insert(0, str(ROOT / "services/engine"))
    from assembly import compile_assembly, compile_inventory
    from cloth_domain import mesh_cloth_domain, validate_cloth_domain
    from embedded_constraints import build_embedded_constraints, validate_embedded_constraints
    from solver_cuff_construction import build_cuff_phase_plan, validate_cuff_phase_plan

    pattern, inventory = compile_inventory(pattern_bytes, construction)
    assembly = compile_assembly(pattern, inventory)
    phase_plan = build_cuff_phase_plan(pattern, inventory, assembly, side=side, policy=policy)
    validate_cuff_phase_plan(pattern, inventory, assembly, phase_plan, side=side, policy=policy)
    cuff, sleeve = f"cuff_{side}", f"sleeve_{side}"
    selected = {f"{cuff}:shell", f"{cuff}:facing", f"{sleeve}:shell",
                *(f"opening_binding_{side}_{edge}:shell" for edge in ("left", "right"))}
    instances = [item for item in inventory["instances"] if item["id"] in selected]
    if len(instances) != 5:
        raise ValueError("Expected all five physical fabric instances in cuff construction unit")
    operation_ids = {f"cuff_gather_{side}",
                     *(f"bind_opening_{side}_{edge}" for edge in ("left", "right")),
                     *(f"perimeter:{cuff}:{edge}" for edge in ("extension", "end", "outer", "start"))}
    operations = [item for item in assembly["operations"] if item["id"] in operation_ids]
    if len(operations) != 7 or any(member["instanceId"] not in selected
                                  for item in operations for member in item["participants"]):
        raise ValueError("Cuff operation subset does not reconcile with selected physical instances")
    if any(dependency not in operation_ids for item in operations for dependency in item["dependsOn"]):
        raise ValueError("Cuff operation dependency is outside captured unit")
    panels = {panel["id"]: panel for panel in pattern["panels"]}
    templates, validation = {}, {}
    for template in sorted({instance["templateId"] for instance in instances}):
        mesh = mesh_cloth_domain(panels[template], max_edge_mm, quality_refinement=True)
        validation[template] = validate_cloth_domain(panels[template], mesh)
        templates[template] = mesh
    sources = {item["id"]: {"panel": panels[item["templateId"]], "mesh": templates[item["templateId"]]}
               for item in instances}
    spec = importlib.util.spec_from_file_location("cuff_unit_registrations", ROOT / "scripts/spike-full-shirt.py")
    registration_helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registration_helper)
    bundle = build_embedded_constraints(sources, registration_helper.shirt_embedded_registrations(operations, sources))
    validate_embedded_constraints(sources, bundle)
    phase_rows, claimed_rows = {}, set()
    for phase in phase_plan["phases"]:
        rows = []
        for selector in phase["activatesStarRows"]:
            matches = [index for index, row in enumerate(bundle["constraints"])
                       if row["registrationId"] == selector["registrationId"]
                       and row["memberIndex"] == selector["memberIndex"]]
            if not matches or claimed_rows.intersection(matches):
                raise ValueError("Phase activation must partition unchanged source star rows exactly once")
            rows.extend(matches)
            claimed_rows.update(matches)
        phase_rows[phase["id"]] = rows
    if claimed_rows != set(range(len(bundle["constraints"]))):
        raise ValueError("Cuff phase plan omits source star rows")
    rest, triangles, offsets = [], [], {}
    for instance in instances:
        mesh = templates[instance["templateId"]]
        offsets[instance["id"]] = len(rest)
        triangles.extend((np.asarray(mesh["triangles"]) + len(rest)).tolist())
        points = np.asarray(mesh["restPositions"]) * .001
        rest.extend(np.column_stack((points, np.zeros(len(points)))).tolist())
    if len(rest) > 25000 or len(triangles) > 50000:
        raise ValueError("Cuff construction mesh budget exceeded")
    excluded = [item for item in assembly["operations"] if item["id"] not in operation_ids]
    unit = {
        "profile": "source-cuff-construction-unit-v1", "accepted": False, "solverReady": False,
        "units": "m", "sourceArcUnits": "mm", "side": side,
        "sourcePattern": pattern, "sourceConstruction": construction,
        "sourceInventory": inventory, "sourceAssembly": assembly,
        "instances": instances, "instanceOffsets": offsets, "sourceTemplates": templates,
        "restMeters": rest, "triangles": np.asarray(triangles).ravel().tolist(),
        "embeddedConstraints": bundle, "phasePlan": phase_plan, "phaseConstraintRows": phase_rows,
        "selectedOperationIds": [item["id"] for item in operations],
        "excludedOperationIds": [item["id"] for item in excluded],
        "unexecutedOtherOperationsTouchingUnit": [item["id"] for item in excluded
            if any(member["instanceId"] in selected for member in item["participants"])],
        "unexecutedClosuresTouchingUnit": [item["id"] for item in assembly["closures"]
            if any(member["instanceId"] in selected for member in item["participants"])],
        "sourceFreeBoundaries": [item for item in assembly["freeBoundaries"] if item["instanceId"] in selected],
        "meshValidation": validation,
        "provenance": {
            "classification": "declarative cuff construction research input; no executed assembly",
            "sourceInputSha256": digest(source_bytes), "patternSha256": inventory["patternDigest"],
            "constructionSha256": inventory["constructionDigest"],
            "assemblySha256": digest(encoded(assembly)), "phasePlanSha256": digest(encoded(phase_plan)),
            "consumedParentFields": ["sourcePattern", "sourceConstruction", "provenance"],
            "parentNumericalGeometryUsed": False, "maxEdgeMm": max_edge_mm, "qualityRefinement": True,
            "runtime": {"python": platform.python_version(),
                        **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "shapely")}},
            "sourceDigests": {name: digest(content) for name, content in captured.items()}},
        "limitations": [
            "All five fabric instances and seven cuff operations are represented; no operation has been executed.",
            "Rest coordinates remain in each source template's local frame. No initial placement or material-side orientation is supplied.",
            "Binding wraps, gathered attachment, allowance/corner treatment, turning passage and final closure require numerical execution and evidence.",
            "Interfacing remains an unresolved physical role; including fabric alone does not complete cuff construction.",
            "Sleeve-to-body and underarm operations and localized closures remain explicitly unexecuted.",
            "Discrete sewing samples do not seal a continuous pocket or prove passage through the declared opening."]}
    if any(path.read_bytes() != captured[str(path.relative_to(ROOT))] for path in paths):
        raise ValueError("Source changed during cuff construction generation")
    if read_regular(source_path, 8 * 1024 ** 2) != source_bytes:
        raise ValueError("Captured parent input changed during cuff construction generation")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    snapshot = output / "source-snapshot"
    snapshot.mkdir(mode=0o700)
    for name, content in captured.items():
        destination = snapshot / name
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_bytes(destination, content)
    atomic_bytes(output / "parent-canonical.json", source_bytes)
    atomic_json(output / "unit.json", unit)
    print(json.dumps({"output": str(output.resolve()), "instances": len(instances),
                      "vertices": len(rest), "triangles": len(triangles),
                      "constraints": len(bundle["constraints"]), "accepted": False, "solverReady": False}))
    return unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--attachment-policy", choices=("inner-facing-first-outer-shell-last-v1",), required=True)
    parser.add_argument("--max-edge-mm", type=float, default=60.)
    args = parser.parse_args()
    generate(args.source_canonical, args.output, side=args.side,
             policy=args.attachment_policy, max_edge_mm=args.max_edge_mm)


if __name__ == "__main__":
    main()
