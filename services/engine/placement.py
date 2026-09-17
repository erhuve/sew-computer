import math


def shirt_placement(pattern, instances):
    panels = {panel["id"]: panel for panel in pattern["panels"]}
    torso = panels["back_left"]
    shoulder = next(edge for edge in torso["draft"]["edges"] if edge["name"] == "shoulder")
    shoulder_x = torso["points"][shoulder["start"]][0]
    radius = shoulder_x / math.sqrt(2)
    top = 1200.0
    frames = []
    for instance in instances:
        template = instance["templateId"]
        panel = panels[template]
        hand = -1 if instance["mirrorX"] else 1
        layer = 2.0 if instance["role"] == "facing" else 0.0
        tangent = [hand, 0.0, 0.0]
        down = [0.0, 0.0, -1.0]
        origin = [0.0, radius + layer, top]
        if template.startswith(("front_", "back_")):
            front = 1 if template.startswith("front_") else -1
            tangent = [hand / math.sqrt(2), -front / math.sqrt(2), 0.0]
            origin = [0.0, front * radius, top]
        elif template.startswith("sleeve_"):
            cap = next(edge for edge in panel["draft"]["edges"] if edge["name"] == "cap_front")
            tangent = [0.0, -hand, 0.0]
            down = [hand, 0.0, 0.0]
            origin = [hand * radius, hand * cap["lengthMm"], top]
        elif template.startswith("cuff_"):
            sleeve = panels[f"sleeve_{'right' if hand == -1 else 'left'}"]
            sleeve_length = max(point[1] for point in sleeve["points"])
            tangent = [0.0, -hand, 0.0]
            down = [hand, 0.0, 0.0]
            origin = [hand * (radius + sleeve_length), panel["widthMm"] / 2, top - 20 - layer]
        elif template.startswith("collar_"):
            origin = [-panel["widthMm"] / 2, radius / 2 + layer, top + (20 if template == "collar_fall" else 0)]
        elif template.startswith(("placket_", "frill_")):
            front = panels[f"front_{'right' if hand == -1 else 'left'}"]
            center = next(edge for edge in front["draft"]["edges"] if edge["name"] == "center")
            origin = [-hand * panel["widthMm"], radius + layer + (10 if template.startswith("frill_") else 0), top - front["points"][center["start"]][1]]
        tangent = [value * hand for value in tangent]
        normal = [tangent[1] * down[2] - tangent[2] * down[1], tangent[2] * down[0] - tangent[0] * down[2], tangent[0] * down[1] - tangent[1] * down[0]]
        frames.append({"instanceId": instance["id"], "originMm": origin, "basis": [tangent, down, normal]})
    frame_by_instance = {frame["instanceId"]: frame for frame in frames}
    for instance in instances:
        if not instance["templateId"].startswith("opening_binding_"):
            continue
        _, _, side, opening = instance["templateId"].split("_")
        sleeves = [candidate for candidate in instances if candidate["templateId"] == f"sleeve_{side}"]
        if len(sleeves) != 1:
            raise ValueError("Opening binding placement requires exactly one matching sleeve")
        sleeve = sleeves[0]
        sleeve_frame = frame_by_instance[sleeve["id"]]
        sleeve_panel = panels[sleeve["templateId"]]
        sleeve_edge = next(edge for edge in sleeve_panel["draft"]["edges"] if edge["name"] == f"opening_{opening}")
        binding_panel = panels[instance["templateId"]]
        binding_edge = next(edge for edge in binding_panel["draft"]["edges"] if edge["name"] == "right")
        sleeve_hand = -1 if sleeve["mirrorX"] else 1
        binding_hand = -1 if instance["mirrorX"] else 1
        sleeve_rest = [[sleeve_hand * sleeve_panel["points"][index][0], sleeve_panel["points"][index][1], 0.0] for index in (sleeve_edge["start"], sleeve_edge["end"])]
        binding_rest = [[binding_hand * binding_panel["points"][index][0], binding_panel["points"][index][1], 0.0] for index in (binding_edge["start"], binding_edge["end"])]
        sleeve_length, binding_length = math.dist(*sleeve_rest), math.dist(*binding_rest)
        if min(sleeve_length, binding_length) <= 0 or abs(sleeve_length - binding_length) > 1e-6:
            raise ValueError("Opening binding placement requires matching straight source edges")
        if sleeve_edge["end"] != sleeve_edge["start"] + 1 or binding_edge["end"] != binding_edge["start"] + 1:
            raise ValueError("Opening binding placement requires straight source intervals")
        target_start, target_end = place_rest_positions(sleeve_rest, sleeve_frame)
        direction = [(target_end[axis] - target_start[axis]) / sleeve_length for axis in range(3)]
        normal = sleeve_frame["basis"][2]
        perpendicular = [direction[1] * normal[2] - direction[2] * normal[1], direction[2] * normal[0] - direction[0] * normal[2], direction[0] * normal[1] - direction[1] * normal[0]]
        local_direction = [(binding_rest[1][axis] - binding_rest[0][axis]) / binding_length for axis in range(2)]
        tangent = [local_direction[0] * direction[axis] + local_direction[1] * perpendicular[axis] for axis in range(3)]
        down = [local_direction[1] * direction[axis] - local_direction[0] * perpendicular[axis] for axis in range(3)]
        spacing = -5.0 if opening == "left" else 5.0
        origin = [target_start[axis] + spacing * normal[axis] - binding_rest[0][0] * tangent[axis] - binding_rest[0][1] * down[axis] for axis in range(3)]
        frame_by_instance[instance["id"]].update({"originMm": origin, "basis": [tangent, down, list(normal)]})
    return {"recipe": "sew-shirt-rigid-staging/2", "units": "mm", "inputCoordinates": "physical-instance-rest-after-declared-mirroring", "classification": "unvalidated-placement", "frames": frames,
            "assumptions": ["Body-free four-flat-panel torso staging from drafted shoulder width.", "Straight horizontal sleeve staging; no posed avatar.", "Facing offsets are 2 mm initialization spacing, not material thickness.", "Opening bindings follow their source sleeve edges with opposite 5 mm sheet-normal initialization spacing, not material thickness.", "Turning, binding wrap and collision-free placement are not established."]}


def place_rest_positions(rest_positions, frame):
    return [[frame["originMm"][axis] + sum(point[local] * frame["basis"][local][axis] for local in range(3)) for axis in range(3)] for point in rest_positions]
