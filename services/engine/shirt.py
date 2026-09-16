import math
from shapely.geometry import Polygon


def distance(start, end):
    return math.hypot(end[0] - start[0], end[1] - start[1])


def compile_shirt(inputs, commit):
    design = inputs["design"]
    body = inputs["bodyMm"]
    allowance = design["seamAllowanceMm"]
    length = inputs["lengthMm"]
    width = (max(body["bust"], body["hip"]) + inputs["easeMm"]) / 4
    hem_width = width * inputs["flare"]
    arm_depth = body["bust"] / 10 + 110
    neck_width = body["bust"] / 12
    front_depth = neck_width
    back_depth = 25
    placket = design["placketWidthMm"] if design["opening"] == "buttons" else 0
    tail = design["tailExtensionMm"] if design["hem"] == "curved-back-tail" else 0
    if width * 2 < body["shoulder"] + 20:
        raise ValueError("DESIGN_INPUT: This relaxed drop-shoulder block needs finished upper-body width at least 20 mm wider than shoulder width. Increase garment ease or choose another construction; do not alter accurate body measurements.")
    if length < arm_depth + 150 or hem_width < width * 0.9:
        raise ValueError("DESIGN_INPUT: This relaxed shirt requires at least 150 mm below the armhole and flare of at least 0.9.")
    panels = []
    assembly = []
    operations = []
    materials = ["Shell: woven shirting; fabric/color remain the reviewed design specification. Yardage depends on fabric width and cutting layout and is not calculated."]

    def panel(name, component, boundaries, quantity=1, material="shell", marks=None):
        points = []
        edges = []
        for edge_name, path in boundaries:
            if points and distance(points[-1], path[0]) > 0.00001:
                raise ValueError("Disconnected drafted edge")
            if not points:
                points.append(path[0])
            start = len(points) - 1
            points.extend(path[1:])
            edge_length = sum(distance(first, second) for first, second in zip(path, path[1:]))
            if edge_length < 0.01:
                raise ValueError("Degenerate drafted edge")
            finish = "bagged-facing" if quantity == 2 else "single-turn-overlocked"
            edges.append({"name": edge_name, "start": start, "end": len(points) - 1, "lengthMm": edge_length, "finish": finish})
        if distance(points[0], points[-1]) > 0.00001:
            raise ValueError("Open drafted boundary")
        shape = Polygon(points)
        if not shape.is_valid or shape.area < 1:
            raise ValueError("DESIGN_INPUT: This construction produces overlapping panel edges. Revise the garment proportions; saved measurements are unchanged.")
        cutting = shape.buffer(allowance, join_style="mitre", mitre_limit=3)
        if cutting.geom_type != "Polygon" or not cutting.is_valid or cutting.interiors or not cutting.covers(shape):
            raise ValueError("DESIGN_INPUT: Unable to construct a single valid seam-allowance contour.")
        xmin, ymin, xmax, ymax = shape.bounds
        shift = lambda point: [round(point[0] - xmin, 6), round(point[1] - ymin, 6)]
        middle = shape.representative_point()
        grain_x, grain_y = middle.x, middle.y
        grain_half = min(30, (ymax - ymin) / 4)
        item = {"id": name, "name": name.replace("_", " "), "points": [shift(point) for point in points], "widthMm": round(xmax - xmin, 6), "heightMm": round(ymax - ymin, 6), "cutQuantity": quantity,
                "draft": {"component": component, "material": material, "cutQuantity": quantity, "edges": edges, "cutLine": [shift(point) for point in cutting.exterior.coords], "grainline": [shift([grain_x, grain_y - grain_half]), shift([grain_x, grain_y + grain_half])], "marks": [{**mark, "point": shift(mark["point"])} for mark in (marks or [])]}}
        panels.append(item)
        return item

    def rectangle(name, component, wide, high, quantity=1, marks=None):
        corners = [[0, 0], [wide, 0], [wide, high], [0, high], [0, 0]]
        return panel(name, component, [(label, corners[index:index + 2]) for index, label in enumerate(["top", "right", "bottom", "left"])], quantity, marks=marks)

    def edge(piece, name):
        return next(index for index, item in enumerate(piece["draft"]["edges"]) if item["name"] == name)

    def join(name, first, first_edge, second, second_edge, instruction, treatment="plain"):
        first_index = edge(first, first_edge)
        second_index = edge(second, second_edge)
        first_length = first["draft"]["edges"][first_index]["lengthMm"]
        second_length = second["draft"]["edges"][second_index]["lengthMm"]
        if treatment in ("plain", "layer") and abs(first_length - second_length) > 0.1:
            raise ValueError(f"Incompatible drafted seam: {name}")
        assembly.append({"id": name, "sides": [{"panel": first["id"], "edge": first_index}, {"panel": second["id"], "edge": second_index}], "treatment": treatment, "ratio": first_length / second_length, "instruction": instruction})
        for piece, edge_index in [(first, first_index), (second, second_index)]:
            boundary = piece["draft"]["edges"][edge_index]
            boundary["finish"] = "assembly"
            path = piece["points"][boundary["start"]:boundary["end"] + 1]
            target = boundary["lengthMm"] / 2
            traversed = 0
            for start, end in zip(path, path[1:]):
                segment = distance(start, end)
                if traversed + segment >= target:
                    fraction = (target - traversed) / segment
                    piece["draft"]["marks"].append({"kind": "notch", "point": [start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])], "label": name})
                    break
                traversed += segment

    torsos = {}
    for side in ("left", "right"):
        for face in ("front", "back"):
            offset = placket / 2 if face == "front" else 0
            depth = front_depth if face == "front" else back_depth
            extension = tail if face == "back" else 0
            neckline = [[offset + (neck_width - offset) * math.cos(math.pi * step / 128), depth * math.sin(math.pi * step / 128)] for step in range(65)]
            neckline[0] = [neck_width, 0]
            neckline[-1] = [offset, depth]
            hem = [[offset + (hem_width - offset) * step / 64, length + extension * (1 + math.cos(math.pi * step / 64)) / 2] for step in range(65)]
            boundaries = [("center", [[offset, depth], [offset, length + extension]]), ("hem", hem), ("side", [[hem_width, length], [width, arm_depth]]), ("armhole", [[width, arm_depth], [width, 0]]), ("shoulder", [[width, 0], [neck_width, 0]]), ("neck", neckline)]
            piece = panel(f"{face}_{side}", "body", boundaries)
            torsos[(face, side)] = piece
        join(f"shoulder_{side}", torsos[("front", side)], "shoulder", torsos[("back", side)], "shoulder", "Sew the shoulder seam with right sides together; finish and press toward back.")
        join(f"side_{side}", torsos[("front", side)], "side", torsos[("back", side)], "side", "Sew side seam; front and back meet at the same side-hem height.")
    join("center_back", torsos[("back", "left")], "center", torsos[("back", "right")], "center", "Sew center back, including the tail extension, and finish the seam.")
    operations.append("Cut each explicitly named left/right torso once, as a mirrored pair on fabric. All outlines are seam lines; the separate offset contours are cut lines. Transfer grainlines and registration marks.")
    operations.append("Finish curved hems with a narrow single-turn overlocked hem using the stated allowance; no extra double-turn hem allowance is included.")

    if placket:
        count = max(2, math.floor((length - front_depth - 50) / design["buttonSpacingMm"]) + 1)
        positions = [25 + index * design["buttonSpacingMm"] for index in range(count)]
        plackets = {}
        for side in ("left", "right"):
            kind = "buttonhole" if side == "right" else "button"
            piece = rectangle(f"placket_{side}", "front-opening", placket, length - front_depth, 2, [{"kind": kind, "point": [placket / 2, value], "label": f"{kind} {index + 1}; size to chosen button"} for index, value in enumerate(positions)])
            plackets[side] = piece
            join(f"placket_attach_{side}", torsos[("front", side)], "center", piece, "right", "Interface one placket layer. Sandwich torso edge between shell and facing; sew attachment, turn outer edge and understitch. Full placket width overlaps its opposite when closed.")
            if design["frill"] != "none":
                frill = rectangle(f"frill_{side}", "frills", design["frillWidthMm"], (length - front_depth) * design["frillFullness"])
                join(f"frill_gather_{side}", frill, "right", piece, "right", "Finish the free edge and ends with a narrow single-turn overlocked hem. Gather to the placket attachment length, match midpoint marks, and insert into the torso/placket seam before closing the facing.", "gather")
        materials.append(f'Front closure: {count} buttons with matching buttonholes, spaced {design["buttonSpacingMm"]} mm apart starting 25 mm below the neckline. Choose button diameter before sewing buttonholes. Interfacing: one layer of each placket template, trim seam allowance to reduce bulk.')
        operations.append("Make and attach plackets before the collar; right placket overlaps left by one full finished placket width. Button axes define the same finished center-front line.")
    else:
        join("center_front", torsos[("front", "left")], "center", torsos[("front", "right")], "center", "Sew center front. Neck opening has no closure; verify head passage with the toile.")

    if design["sleeves"] != "none":
        sleeve_length = design["sleeveLengthMm"]
        cuff_depth = design["cuffDepthMm"] if design["cuff"] == "button" else 0
        if sleeve_length <= cuff_depth + 120:
            raise ValueError("DESIGN_INPUT: Sleeve length must leave at least 120 mm above the cuff.")
        sleeve_length -= cuff_depth
        wrist = max(design["cuffCircumferenceMm"] * 1.35, arm_depth * 1.1) if cuff_depth else arm_depth * 1.6
        if wrist >= arm_depth * 2:
            raise ValueError("DESIGN_INPUT: Cuff and armhole proportions leave no usable sleeve taper. Revise sleeve/cuff dimensions.")
        taper = (arm_depth * 2 - wrist) / 2
        opening_height = min(100, sleeve_length * 0.3) if cuff_depth else 0
        underarm_length = sleeve_length - opening_height
        opening_x = taper * underarm_length / sleeve_length
        for side in ("left", "right"):
            boundaries = [("cap_front", [[0, 0], [arm_depth, 0]]), ("cap_back", [[arm_depth, 0], [2 * arm_depth, 0]]), ("underarm_right", [[2 * arm_depth, 0], [2 * arm_depth - opening_x, underarm_length]])]
            if cuff_depth:
                boundaries.append(("opening_right", [[2 * arm_depth - opening_x, underarm_length], [2 * arm_depth - taper, sleeve_length]]))
            boundaries.append(("wrist", [[2 * arm_depth - taper, sleeve_length], [taper, sleeve_length]]))
            if cuff_depth:
                boundaries.append(("opening_left", [[taper, sleeve_length], [opening_x, underarm_length]]))
            boundaries.append(("underarm_left", [[opening_x, underarm_length], [0, 0]]))
            sleeve = panel(f"sleeve_{side}", "sleeves", boundaries)
            join(f"sleeve_front_{side}", sleeve, "cap_front", torsos[("front", side)], "armhole", "Attach the flat sleeve cap to the straight dropped armhole; align center cap with shoulder seam.")
            join(f"sleeve_back_{side}", sleeve, "cap_back", torsos[("back", side)], "armhole", "Attach back half of sleeve cap; this is a relaxed drop-shoulder sleeve, not a set-in sleeve.")
            join(f"underarm_{side}", sleeve, "underarm_left", sleeve, "underarm_right", "Sew underarm only to the opening endpoints. Leave both separately bound cuff-opening edges unsewn." if cuff_depth else "Close sleeve underarm seam; finish sleeve hem with a single-turn overlocked hem.")
            if cuff_depth:
                cuff_width = design["cuffCircumferenceMm"]
                overlap = 20
                cuff = panel(f"cuff_{side}", "cuffs", [("attachment", [[0, 0], [cuff_width, 0]]), ("extension", [[cuff_width, 0], [cuff_width + overlap, 0]]), ("end", [[cuff_width + overlap, 0], [cuff_width + overlap, cuff_depth]]), ("outer", [[cuff_width + overlap, cuff_depth], [0, cuff_depth]]), ("start", [[0, cuff_depth], [0, 0]])], 2, marks=[{"kind": "button", "point": [10, cuff_depth / 2], "label": "cuff button"}, {"kind": "buttonhole", "point": [cuff_width + 10, cuff_depth / 2], "label": "cuff buttonhole; size to button"}])
                join(f"cuff_gather_{side}", sleeve, "wrist", cuff, "attachment", "Gather sleeve wrist to cuff attachment edge; keep 20 mm cuff extension free. Interface outer cuff; attach inner cuff, turn ends and outer edge, close facing. Button axes are one finished cuff circumference apart.", "gather")
                for position in ("left", "right"):
                    opening_edge = sleeve["draft"]["edges"][edge(sleeve, f"opening_{position}")]
                    binding = rectangle(f"opening_binding_{side}_{position}", "cuffs", allowance * 2, opening_edge["lengthMm"])
                    join(f"bind_opening_{side}_{position}", sleeve, f"opening_{position}", binding, "right", "Bind this open edge separately; turn strip around seam allowance and stitch down. Secure opening apex without sewing the opening closed.", "bind")
        if cuff_depth:
            materials.append("Cuffs: two additional buttons; interfacing one layer per cuff template. Two fabric layers per named cuff are already included in cut quantities.")
        operations.append("Set sleeves flat before closing underarm and side seams. The selected sleeve length runs from dropped shoulder to cuff edge, including cuff depth; it is not an anatomical arm measurement.")
    else:
        operations.append("Finish armholes with a single-turn overlocked edge using the provided allowance; verify comfort in a toile.")

    if design["collar"] != "none":
        neck_chain = [(plackets["left"], "top"), (torsos[("front", "left")], "neck"), (torsos[("back", "left")], "neck"), (torsos[("back", "right")], "neck"), (torsos[("front", "right")], "neck"), (plackets["right"], "top")]
        spans = [piece["draft"]["edges"][edge(piece, name)]["lengthMm"] for piece, name in neck_chain]
        total = sum(spans)
        stand_depth = design["collarStandMm"]
        boundaries = []
        cursor = 0
        for index, span in enumerate(spans):
            boundaries.append((f"neck_{index}", [[cursor, 0], [cursor + span, 0]]))
            cursor += span
        boundaries.extend([("right", [[total, 0], [total, stand_depth]]), ("upper_end", [[total, stand_depth], [total - placket / 2, stand_depth]]), ("fall", [[total - placket / 2, stand_depth], [placket / 2, stand_depth]]), ("upper_start", [[placket / 2, stand_depth], [0, stand_depth]]), ("left", [[0, stand_depth], [0, 0]])])
        stand = panel("collar_stand", "collar", boundaries, 2, marks=[{"kind": "button", "point": [placket / 2, stand_depth / 2], "label": "collar button"}, {"kind": "buttonhole", "point": [total - placket / 2, stand_depth / 2], "label": "collar buttonhole; size to button"}])
        for index, (piece, name) in enumerate(neck_chain):
            join(f"collar_neck_{index}", stand, f"neck_{index}", piece, name, "Attach collar stand along the specified neckline/placket segment, matching registration marks. Sandwich between outer stand and facing.")
        if design["collar"] == "stand-and-fall":
            fall = rectangle("collar_fall", "collar", total - placket, design["collarFallMm"], 2)
            join("collar_fall_attach", stand, "fall", fall, "top", "Sew two collar-fall layers around free edges, turn and press, then insert into stand upper edge. Stand ends remain free of fall; baste and check roll/turn-of-cloth in a toile.")
        materials.append("Collar: one additional button; interfacing one stand and one fall layer if present, using corresponding templates with allowance trimmed. Collar layers use identical templates; turn-of-cloth is not compensated.")
        operations.append("Construct collar after plackets. The stand lower edge includes both placket tops; its closed circumference subtracts one placket overlap. Check neck comfort before final buttonholes.")
    else:
        operations.append("Finish neckline with a single-turn overlocked edge using the provided allowance; test curvature and head passage in the toile.")

    components = ["body"]
    for condition, feature in [(design["sleeves"] != "none", "sleeves"), (design["cuff"] != "none", "cuffs"), (design["collar"] != "none", "collar"), (bool(placket), "front-opening"), (bool(tail), "tails"), (design["frill"] != "none", "frills")]:
        if condition:
            components.append(feature)
    warnings = ["Digital construction draft, not a physically fitted or sewing-validated pattern. Make a toile and check wearing ease, neck/head passage, cuff passage, collar roll and mobility before cutting final fabric.", "All drafting dimensions are reviewed design assumptions. Neck width and armhole depth use disclosed synthetic proportions: bust/12 and bust/10 + 110 mm. Torso quarter width = (max(bust, hip) + ease)/4. No anatomical arm, wrist or neck measurements are inferred.", "Each named left/right torso and sleeve is cut once as a mirrored pair. Plackets, cuffs and collar templates are cut twice for shell/facing; same-template interfacing is listed separately in derived materials. No fold-cut torso edges.", f"Uniform {allowance} mm outward cut-line offset includes seam and single-turn overlocked edge finishing. Offsets use sampled curves, miter limit 3, and polygon validity/containment checks. Double-turn hems, grading, fabric shrinkage, turn-of-cloth and physical print calibration are not validated."]
    warnings.extend(inputs["provenance"])
    measurements = [{"name": "Closed chest at underarm", "valueMm": width * 4, "method": "Four torso quarter widths, adjusted for front placket replacement and overlap; horizontal seam-line circumference, no fabric simulation."}, {"name": "Side length from shoulder baseline", "valueMm": length, "method": "Vertical drafted baseline-to-side-hem distance; not a worn front length."}, {"name": "Back center length", "valueMm": length + tail - back_depth, "method": "Center-back neck seam to center-back hem seam."}]
    if design["sleeves"] != "none":
        measurements.append({"name": "Sleeve including cuff", "valueMm": design["sleeveLengthMm"], "method": "Dropped armhole baseline to cuff/hem seam-line, includes cuff depth."})
    if design["cuff"] != "none":
        measurements.append({"name": "Closed cuff circumference", "valueMm": design["cuffCircumferenceMm"], "method": "Distance between button and buttonhole axes; excludes 20 mm extension."})
    if design["collar"] != "none":
        measurements.append({"name": "Closed collar stand circumference", "valueMm": total - placket, "method": "Stand neckline seam-line length minus one full placket overlap; not a body neck measurement."})
    def stage(seam):
        for prefix, order in [("center_", 5), ("shoulder_", 10), ("frill_", 15), ("placket_", 20), ("collar_fall", 25), ("collar_neck", 30), ("bind_opening", 35), ("sleeve_", 40), ("side_", 45), ("underarm_", 50), ("cuff_", 55)]:
            if seam["id"].startswith(prefix):
                return order
        raise ValueError("Unscheduled assembly operation")
    assembly.sort(key=lambda seam: (stage(seam), seam["id"]))
    operations = [operations[0], "Interface one layer of each placket, cuff and collar template that is present; test interfacing compatibility and shrinkage on scraps."]
    operations.extend(f'{seam["id"]}: {seam["instruction"]}' for seam in assembly)
    operations.append(f"Finish unattached garment hems and frill edges with a single-turn overlocked hem using {allowance} mm allowance. Finish a collar-free neckline and sleeve-free armholes with the same allowance; verify curved finishes in a toile.")
    if placket or design["cuff"] != "none" or design["collar"] != "none":
        operations.append("Work buttonholes only after choosing button diameter and testing closure placement. Sew buttons, press, then verify the toile against the design and body measurements.")
    else:
        operations.append("Press and verify head passage, mobility, wearing ease and silhouette in a toile.")
    return {"schemaVersion": 1, "units": "mm", "inputDigest": inputs["inputDigest"], "engineVersion": f"Sew relaxed shirt compiler 1 / runtime pin {commit}", "family": "shirt", "panels": panels, "stitches": [{"panelA": seam["sides"][0]["panel"], "edgeA": seam["sides"][0]["edge"], "panelB": seam["sides"][1]["panel"], "edgeB": seam["sides"][1]["edge"]} for seam in assembly], "warnings": warnings, "classification": "printable-reference", "assumptions": warnings[:4], "drafting": {"compiler": "sew-relaxed-shirt/1", "seamAllowanceMm": allowance, "components": components, "assembly": assembly, "measurements": measurements, "materials": materials, "operations": list(dict.fromkeys(operations))}}
