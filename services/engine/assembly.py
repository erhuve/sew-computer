import hashlib
import json
import math
import platform
from pathlib import Path

import shapely
from shapely.geometry import LineString, Point

from meshing import mesh_panel, source_polygon
from simulation_validation import validate_rest_mesh


def compile_inventory(pattern_bytes, construction):
    if not isinstance(pattern_bytes, bytes) or len(pattern_bytes) > 4 * 1024 * 1024:
        raise ValueError("Pattern input budget exceeded")
    pattern = json.loads(pattern_bytes)
    if type(pattern.get("schemaVersion")) is not int or pattern["schemaVersion"] != 1 or pattern.get("units") != "mm" or (pattern.get("family"), pattern.get("drafting", {}).get("compiler")) not in (("shirt", "sew-relaxed-shirt/1"), ("dress", "sew-relaxed-dress/1"), ("skirt", "sew-elastic-skirt/1")):
        raise ValueError("Only explicit supported component patterns are supported")
    panels = pattern.get("panels")
    if not isinstance(panels, list) or not 4 <= len(panels) <= 18:
        raise ValueError("Invalid template inventory")
    by_id = {panel["id"]: panel for panel in panels}
    if len(by_id) != len(panels):
        raise ValueError("Duplicate template identity")
    skirt = pattern['family'] == 'skirt'
    block = {'shirt': 'relaxed-drop-shoulder', 'dress': 'relaxed-dress', 'skirt': 'elastic-waist-skirt'}[pattern['family']]
    if not isinstance(construction, dict) or construction.get('block') != block:
        raise ValueError('Construction does not match the source garment family')
    if skirt:
        from skirt import QUARTERS
        expected = {f'{prefix}_{quarter}': quantity for quarter in QUARTERS for prefix, quantity in (('skirt', 1), ('waistband', 2))}
        if pattern['drafting'].get('components') != ['body', 'waistband']:
            raise ValueError('Skirt components differ from captured construction')
        if any(abs(by_id.get(f'waistband_{quarter}', {}).get('heightMm', -1) - construction['waistbandDepthMm']) > 1e-6 for quarter in QUARTERS):
            raise ValueError('Skirt waistband depth differs from construction')
    else:
        expected = {f"{face}_{side}": 1 for face in ("front", "back") for side in ("left", "right")}
        components = pattern["drafting"].get("components", [])
        if not isinstance(construction, dict) or construction.get("collar") not in ("none", "stand", "stand-and-fall"):
            raise ValueError("Explicit captured construction is required")
        selected = {"body"}
        for key, values, component in [("sleeves", ("none", "short", "long"), "sleeves"), ("cuff", ("none", "button"), "cuffs"), ("collar", ("none", "stand", "stand-and-fall"), "collar"), ("opening", ("none", "buttons"), "front-opening"), ("frill", ("none", "front-opening"), "frills"), ("hem", ("straight", "curved-back-tail"), "tails")]:
            if construction.get(key) not in values:
                raise ValueError("Unsupported captured construction")
            if construction[key] != values[0]:
                selected.add(component)
        if set(components) != selected:
            raise ValueError("Source components differ from captured construction")
        if len(set(components)) != len(components) or "body" not in components or set(components) - {"body", "sleeves", "cuffs", "collar", "front-opening", "tails", "frills"}:
            raise ValueError("Invalid component declaration")
        for component, prefix, quantity in [("sleeves", "sleeve", 1), ("cuffs", "cuff", 2), ("front-opening", "placket", 2), ("frills", "frill", 1)]:
            if component in components:
                expected.update({f"{prefix}_{side}": quantity for side in ("left", "right")})
        if "cuffs" in components:
            if "sleeves" not in components or construction["sleeves"] != "long":
                raise ValueError("Cuffs require sleeves")
            expected.update({f"opening_binding_{side}_{edge}": 1 for side in ("left", "right") for edge in ("left", "right")})
        if "collar" in components:
            expected["collar_stand"] = 2
            if construction["collar"] == "stand-and-fall":
                expected["collar_fall"] = 2
        if ("collar" in components or "frills" in components) and "front-opening" not in components:
            raise ValueError("Selected component requires front opening")
    if set(expected) != set(by_id):
        raise ValueError("Selected components do not reconcile with template inventory")
    instances = []
    unresolved = []
    for template, quantity in sorted(expected.items()):
        panel = by_id[template]
        shape = source_polygon(panel)
        draft = panel.get("draft", {})
        if isinstance(panel.get("cutQuantity"), bool) or isinstance(draft.get("cutQuantity"), bool) or panel.get("cutQuantity") != quantity or draft.get("cutQuantity") != quantity or draft.get("material") != "shell":
            raise ValueError("Physical cut count or material mismatch")
        component = "waistband" if template.startswith("waistband_") else "body" if template.startswith(("skirt_", "front_", "back_")) else "cuffs" if template.startswith(("cuff_", "opening_binding_")) else "collar" if template.startswith("collar_") else "sleeves" if template.startswith("sleeve_") else "frills" if template.startswith("frill_") else "front-opening"
        if draft.get("component") != component:
            raise ValueError("Source component does not match template role")
        edges = draft.get("edges")
        if not isinstance(edges, list) or not 3 <= len(edges) <= 30:
            raise ValueError("Missing source edge semantics")
        names = set()
        cursor = 0
        for edge in edges:
            if not isinstance(edge.get("name"), str) or not edge["name"] or edge["name"] in names or edge.get("start") != cursor or isinstance(edge.get("end"), bool) or not isinstance(edge.get("end"), int) or not cursor < edge["end"] < len(panel["points"]):
                raise ValueError("Invalid source edge identity or interval")
            names.add(edge["name"])
            measured = LineString(panel["points"][cursor:edge["end"] + 1]).length
            length = edge.get("lengthMm")
            if isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length) or abs(measured - length) > 1e-4:
                raise ValueError("Invalid source edge metric")
            cursor = edge["end"]
        if cursor != len(panel["points"]) - 1:
            raise ValueError("Source edges do not cover the boundary")
        grain = draft.get("grainline")
        if not isinstance(grain, list) or len(grain) != 2 or any(not isinstance(point, list) or len(point) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 10000 for value in point) for point in grain) or math.dist(*grain) < 0.01 or not shape.covers(LineString(grain)):
            raise ValueError("Invalid source grain")
        mirror = template.endswith("_right") if not template.startswith("opening_binding_") else template.startswith("opening_binding_right_")
        for role in (["shell", "facing"] if quantity == 2 else ["shell"]):
            instances.append({"id": f"{template}:{role}", "templateId": template, "role": role, "mirrorX": mirror, "sourceGrainline": grain})
        if quantity == 2 and not skirt:
            unresolved.append({"templateId": template, "role": "interfacing", "reason": "Trimming and material properties are not machine-defined; no physical mesh claimed."})
    return pattern, {
        "schemaVersion": 1, "compiler": f"sew-{pattern['family']}-inventory/1", "patternDigest": hashlib.sha256(pattern_bytes).hexdigest(),
        "constructionDigest": hashlib.sha256(json.dumps(construction, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "units": "mm", "classification": "placement-inspection", "instances": instances,
        "unresolvedPhysicalRoles": unresolved,
        "capabilityGaps": ["Seam orientation and registration require explicit assembly semantics.", "Facing turning, binding wraps, multi-way gathers and local closures are not implemented.", "No assembly, contact, material response or drape has been computed."],
    }


def build_inspection(pattern_bytes, construction, max_edge_mm=40):
    module_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ("assembly.py", "meshing.py", "simulation_validation.py", "inspection_gltf.py", "guard.py")}
    pattern, inventory = compile_inventory(pattern_bytes, construction)
    assembly = compile_assembly(pattern, inventory)
    meshes = []
    reports = []
    for panel in pattern["panels"]:
        mesh = mesh_panel(panel, max_edge_mm)
        reports.append({"templateId": panel["id"], **validate_rest_mesh(panel, mesh)})
        meshes.append(mesh)
        if sum(len(item["triangles"]) for item in meshes) > 150000 or sum(len(item["restPositions"]) for item in meshes) > 100000:
            raise ValueError("Garment mesh budget exceeded")
    if any(hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest for name, digest in module_hashes.items()):
        raise ValueError("Engine source changed during inspection")
    return {**inventory, "assembly": assembly, "mesher": "sew-constrained-mesh/2", "runtime": {"python": platform.python_version(), "shapely": shapely.__version__, "geos": shapely.geos_version_string, "sourceDigests": module_hashes}, "maxEdgeMm": max_edge_mm, "templates": meshes, "validation": reports}


def compile_assembly(pattern, inventory):
    if pattern["family"] == "skirt":
        from skirt_assembly import compile_skirt_assembly
        return compile_skirt_assembly(pattern, inventory)
    panels = {panel["id"]: panel for panel in pattern["panels"]}
    instances = {instance["id"]: instance for instance in inventory["instances"]}
    expected = {}

    def expect(identity, first, first_edge, second, second_edge, treatment="plain"):
        expected[identity] = (first, first_edge, second, second_edge, treatment)

    expect("center_back", "back_left", "center", "back_right", "center")
    if "placket_left" not in panels:
        expect("center_front", "front_left", "center", "front_right", "center")
    for side in ("left", "right"):
        for edge in ("shoulder", "side"):
            expect(f"{edge}_{side}", f"front_{side}", edge, f"back_{side}", edge)
        if f"placket_{side}" in panels:
            expect(f"placket_attach_{side}", f"front_{side}", "center", f"placket_{side}", "right")
        if f"frill_{side}" in panels:
            expect(f"frill_gather_{side}", f"frill_{side}", "right", f"placket_{side}", "right", "gather")
        if f"sleeve_{side}" in panels:
            for face in ("front", "back"):
                expect(f"sleeve_{face}_{side}", f"sleeve_{side}", f"cap_{face}", f"{face}_{side}", "armhole")
            expect(f"underarm_{side}", f"sleeve_{side}", "underarm_left", f"sleeve_{side}", "underarm_right")
        if f"cuff_{side}" in panels:
            expect(f"cuff_gather_{side}", f"sleeve_{side}", "wrist", f"cuff_{side}", "attachment", "gather")
            for edge in ("left", "right"):
                expect(f"bind_opening_{side}_{edge}", f"sleeve_{side}", f"opening_{edge}", f"opening_binding_{side}_{edge}", "right", "bind")
    if "collar_stand" in panels:
        chain = [("placket_left", "top"), ("front_left", "neck"), ("back_left", "neck"), ("back_right", "neck"), ("front_right", "neck"), ("placket_right", "top")]
        for index, (template, edge) in enumerate(chain):
            expect(f"collar_neck_{index}", "collar_stand", f"neck_{index}", template, edge)
        if "collar_fall" in panels:
            expect("collar_fall_attach", "collar_stand", "fall", "collar_fall", "top")
    source = pattern["drafting"].get("assembly")
    if not isinstance(source, list) or len(source) != len(expected) or {item.get("id") for item in source} != set(expected):
        raise ValueError("Assembly source operations do not match selected construction")
    for seam in source:
        first, first_edge, second, second_edge, treatment = expected[seam["id"]]
        sides = seam.get("sides")
        expected_sides = [{"panel": template, "edge": next(index for index, edge in enumerate(panels[template]["draft"]["edges"]) if edge["name"] == name)} for template, name in ((first, first_edge), (second, second_edge))]
        if sides != expected_sides or any(type(side.get("edge")) is not int for side in sides) or seam.get("treatment") != treatment:
            raise ValueError("Assembly source attachment differs from trusted recipe")
        lengths = [panels[side["panel"]]["draft"]["edges"][side["edge"]]["lengthMm"] for side in sides]
        ratio = seam.get("ratio")
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or abs(ratio - lengths[0] / lengths[1]) > 1e-8 or (treatment != "gather" and abs(lengths[0] - lengths[1]) > 0.1) or (treatment == "gather" and lengths[0] < lengths[1]):
            raise ValueError("Assembly source length or gather ratio mismatch")
    stitches = [{"panelA": seam["sides"][0]["panel"], "edgeA": seam["sides"][0]["edge"], "panelB": seam["sides"][1]["panel"], "edgeB": seam["sides"][1]["edge"]} for seam in source]
    if pattern.get("stitches") != stitches:
        raise ValueError("Stitch graph differs from construction graph")
    operations = []
    occupied = set()

    def participant(template, edge, role="shell"):
        boundary = next(item for item in panels[template]["draft"]["edges"] if item["name"] == edge)
        return {"instanceId": f"{template}:{role}", "edgeName": edge, "intervalMm": [0, boundary["lengthMm"]], "sourcePointInterval": [boundary["start"], boundary["end"]]}

    def operation(identity, kind, participants, dependencies=(), **details):
        for item in participants:
            key = (item["instanceId"], item["edgeName"])
            if key in occupied:
                raise ValueError("Undeclared duplicate physical boundary attachment")
            occupied.add(key)
        operations.append({"id": identity, "kind": kind, "participants": participants, "dependsOn": list(dependencies), **details})

    for identity, (first, first_edge, second, second_edge, treatment) in expected.items():
        if identity.startswith("frill_gather"):
            continue
        members = [participant(first, first_edge), participant(second, second_edge)]
        dependencies = []
        if identity.startswith("placket_attach"):
            members.append(participant(second, second_edge, "facing"))
            side = identity.rsplit("_", 1)[1]
            if f"frill_{side}" in panels:
                members.append(participant(f"frill_{side}", "right"))
                treatment = "gather"
        elif identity.startswith(("cuff_gather", "collar_neck", "collar_fall_attach")):
            members += [participant(template, edge, "facing") for template, edge in ((first, first_edge), (second, second_edge)) if f"{template}:facing" in instances]
        if identity.startswith("collar_neck"):
            dependencies += [f"placket_attach_{side}" for side in ("left", "right")]
            if "collar_fall" in panels:
                dependencies.append("collar_fall_attach")
        if identity == "collar_fall_attach":
            dependencies += [f"perimeter:collar_fall:{edge}" for edge in ("right", "bottom", "left")]
        if identity.startswith(("underarm", "side_")) and "sleeve_left" in panels:
            side = identity.rsplit("_", 1)[1]
            dependencies += [f"sleeve_{face}_{side}" for face in ("front", "back")]
        if identity.startswith("cuff_gather"):
            side = identity.rsplit("_", 1)[1]
            dependencies += [f"bind_opening_{side}_{edge}" for edge in ("left", "right")]
        operation(identity, "binding" if treatment == "bind" else "gather" if treatment == "gather" else "seam", members, dependencies,
                  registrationFractions=[0, 0.5, 1], distribution="uniform" if treatment == "gather" else "equal-arc",
                  orientationStatus="requires-validated-placement", executionStatus="unsupported-binding-wrap" if treatment == "bind" else "not-simulated")
    for template in panels:
        if f"{template}:facing" not in instances:
            continue
        dependencies = [item["id"] for item in operations if any(member["instanceId"].split(":")[0] == template for member in item["participants"])]
        if template == "collar_fall":
            dependencies = []
        for edge in panels[template]["draft"]["edges"]:
            if (f"{template}:shell", edge["name"]) not in occupied:
                operation(f"perimeter:{template}:{edge['name']}", "layer-perimeter", [participant(template, edge["name"]), participant(template, edge["name"], "facing")], dependencies,
                          orientationStatus="requires-validated-turning", executionStatus="unsupported-turning")
    free = []
    for instance in inventory["instances"]:
        for edge in panels[instance["templateId"]]["draft"]["edges"]:
            if (instance["id"], edge["name"]) not in occupied:
                free.append({**participant(instance["templateId"], edge["name"], instance["role"]), "finish": edge["finish"], "classification": "free-boundary"})
    closures = []
    for template in panels:
        if template == "placket_right":
            continue
        if template == "placket_left" or template.startswith("cuff_") or template == "collar_stand":
            opposite = "placket_right" if template == "placket_left" else template
            buttons = [(index, mark) for index, mark in enumerate(panels[template]["draft"]["marks"]) if mark["kind"] == "button"]
            holes = [(index, mark) for index, mark in enumerate(panels[opposite]["draft"]["marks"]) if mark["kind"] == "buttonhole"]
            if not buttons or len(buttons) != len(holes):
                raise ValueError("Localized closure marks missing or unmatched")
            for index, (button, hole) in enumerate(zip(buttons, holes)):
                points = []
                for identity, (mark_index, mark) in ((template, button), (opposite, hole)):
                    point = mark.get("point")
                    if not isinstance(point, list) or len(point) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in point) or not source_polygon(panels[identity]).covers(Point(point)):
                        raise ValueError("Closure mark is outside source fabric")
                    points.append({"instanceId": f"{identity}:shell", "sourceMarkIndex": mark_index, "pointMm": point})
                closures.append({"id": f"closure:{template}:{index}", "kind": "localized-button", "participants": points, "state": "closed", "executionStatus": "not-simulated"})
    pending = list(operations)
    ordered = []
    while pending:
        ready = [item for item in pending if set(item["dependsOn"]) <= {item["id"] for item in ordered}]
        if not ready:
            raise ValueError("Assembly dependency cycle or missing operation")
        ordered.extend(ready)
        pending = [item for item in pending if item not in ready]
    return {"schemaVersion": 1, "compiler": "sew-shirt-assembly/1", "units": "mm", "operations": ordered, "closures": closures,
            "freeBoundaries": free, "sourceOperationCount": len(source), "physicalInstanceCount": len(instances),
            "semanticCoverage": "explicit-attachments-and-unresolved-execution", "solverReady": False,
            "limitations": ["Boundary orientation requires validated placement.", "Layer turning and binding wraps are not simulated.", "Interfacing remains an unresolved physical role.", "Seam allowances and their bulk are omitted."]}


def expand_instance_meshes(inspection):
    templates = {mesh["templateId"]: mesh for mesh in inspection["templates"]}
    expanded = []
    total_vertices = 0
    total_triangles = 0
    for instance in inspection["instances"]:
        template = templates[instance["templateId"]]
        total_vertices += len(template["restPositions"])
        total_triangles += len(template["triangles"])
        if total_vertices > 150000 or total_triangles > 250000:
            raise ValueError("Physical instance mesh budget exceeded")
    for instance in inspection["instances"]:
        template = templates[instance["templateId"]]
        handedness = -1 if instance["mirrorX"] else 1
        expanded.append({
            "instanceId": instance["id"], "templateId": instance["templateId"], "role": instance["role"],
            "restPositions": [[handedness * point[0], point[1], 0] for point in template["restPositions"]],
            "triangles": [list(reversed(face)) if handedness == -1 else list(face) for face in template["triangles"]],
            "sourceVertexIndices": list(range(len(template["restPositions"]))),
            "grainline": [[handedness * point[0], point[1], 0] for point in instance["sourceGrainline"]],
            "sourceToPhysical": {"mirrorX": instance["mirrorX"], "units": "mm"},
        })
    return expanded
