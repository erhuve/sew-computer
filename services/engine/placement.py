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
        elif template.startswith(("cuff_", "opening_binding_")):
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
    return {"recipe": "sew-shirt-rigid-staging/1", "units": "mm", "inputCoordinates": "physical-instance-rest-after-declared-mirroring", "classification": "unvalidated-placement", "frames": frames,
            "assumptions": ["Body-free four-flat-panel torso staging from drafted shoulder width.", "Straight horizontal sleeve staging; no posed avatar.", "Layer offsets are 2 mm initialization spacing, not material thickness.", "Turning, binding wrap and collision-free placement are not established."]}


def place_rest_positions(rest_positions, frame):
    return [[frame["originMm"][axis] + sum(point[local] * frame["basis"][local][axis] for local in range(3)) for axis in range(3)] for point in rest_positions]
