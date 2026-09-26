import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard import COMMIT, ROOT, constrain, verify_runtime, verify_dependencies


def build_body(inputs):
    from assets.bodies.body_params import BodyParameters
    fixture = json.loads((ROOT / "synthetic-body.json").read_text())
    mm = inputs["bodyMm"]
    body = {"height": mm["height"] / 10, "bust": mm["bust"] / 10, "waist": mm["waist"] / 10, "hips": mm["hip"] / 10, "shoulder_w": mm["shoulder"] / 10}
    warnings = [fixture["purpose"]]
    if inputs["family"] != "shirt":
        body["waist"] += inputs["easeMm"] / 10
        body["hips"] += inputs["easeMm"] / 10
        warnings.append("Lower-garment ease is added to both waist and hip circumference in the construction body, before synthetic dependent proportions; not a verified finished-garment POM.")
    for name, (source, ratio) in fixture["derived"].items():
        body[name] = body[source] * ratio
        warnings.append(f"Assumed {name} = {ratio} * construction {source} = {body[name] * 10:.4f} mm; original synthetic fixture, not inferred anatomy.")
    for name, degrees in fixture["fixedDegrees"].items():
        body[name] = degrees
        warnings.append(f"Assumed {name} = {degrees} degrees; original synthetic fixture.")
    Path("construction-body.json").write_text(json.dumps({"body": body}), encoding="utf-8")
    params = BodyParameters("construction-body.json")
    warnings.extend([
        "Upstream derived assumptions: waist level = height - head_l - waist_line; leg length = waist level - hips_line; sleeve balance = shoulder_w - 20 mm; bust line = 2/3 vert_bust_line + 1/3 bust_line; hip inclination is halved; armscye depth includes 25 mm of upstream ease.",
        "Canonical inputs are converted mm to cm exactly once; upstream pattern units_in_meter=100 is checked; curve coordinates and lengths are converted cm to mm exactly once. Angles and ratios are not scaled.",
    ])
    return params, warnings


def compile_piece(inputs, upstream, body):
    import yaml
    from assets.garment_programs.bodice import Shirt
    from assets.garment_programs.circle_skirt import SkirtCircle
    from assets.garment_programs.pants import Pants
    design = yaml.safe_load((upstream / "assets/design_params/default_template.yaml").read_text())["design"]
    family, length, flare = inputs["family"], inputs["lengthMm"] / 10, inputs["flare"]
    if family == "shirt":
        design["shirt"]["length"]["v"] = length / body["waist_line"]
        design["shirt"]["width"]["v"] = 1 + inputs["easeMm"] / 10 / body["bust"]
        design["shirt"]["flare"]["v"] = flare
        design["sleeve"]["sleeveless"]["v"] = True
        design["left"]["enable_asym"]["v"] = False
        piece = Shirt(body, design)
        description = "Supported subset: symmetric sleeveless Shirt with curved armholes and circle neck openings; no sleeves or collar panel. Length sets the upstream torso construction length from shoulder baseline; width = (bust + ease)/bust; flare is upstream hem-width ratio. Shoulder inclination can increase panel bounding height; length is not a validated finished POM."
    else:
        section = "flare-skirt" if family == "skirt" else "pants"
        design[section]["rise"]["v"] = 1.0
        design[section]["length"]["v"] = (length - body["hips_line"]) / body["_leg_length"]
        if family == "skirt":
            design[section]["suns"]["v"] = 0.5 * flare
            piece = SkirtCircle(body, design)
            description = "Supported subset: two-panel circular skirt, natural-waist rise, no waistband/closure. Length is radial waist-to-hem length; flare maps to circle fullness suns=0.5*flare, so 1 means half-circle and 2 full-circle. Ease expands the construction waist and hip; no closure, allowance or worn-fit validation."
        else:
            design[section]["flare"]["v"] = flare
            design[section]["width"]["v"] = 1.0
            design[section]["cuff"]["type"]["v"] = None
            piece = Pants(body, design)
            description = "Supported subset: four-panel darted trousers, natural-waist rise, no waistband/cuffs/closure. Length sets base leg length plus synthetic hip depth; upstream back panels add 10% hip depth. Flare is upstream leg-width ratio. Upstream crotch extension retains its fixed +50 mm construction ease and a -20 mm hem heuristic; entered ease additionally expands waist/hip. No finished-POM or seam-ease correctness claim."
    mapped = {key: {k: v["v"] for k, v in design[key].items() if isinstance(v, dict) and "v" in v} for key in (["shirt", "collar", "sleeve"] if family == "shirt" else [section])}
    return piece, [description, "Pinned default-template settings not exposed in this slice: " + json.dumps(mapped, sort_keys=True), "Trusted fixed family classes are selected directly; MetaGarment's implicit waistband insertion/0.7 waist scaling is deliberately not used."]


def normalize(piece, digest, warnings, family):
    pattern = piece.assembly()
    if pattern.spec["properties"]["units_in_meter"] != 100:
        raise ValueError("Upstream coordinate units changed")
    if piece.is_self_intersecting():
        raise ValueError("Upstream reports self-intersecting geometry; this input combination is unsupported")
    raw_panels = pattern.pattern["panels"]
    if not 1 <= len(raw_panels) <= 32:
        raise ValueError("Invalid panel count")
    panels, edge_lengths = [], {}
    for name in sorted(raw_panels):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", name):
            raise ValueError("Unexpected panel name")
        path, _, _ = pattern._draw_a_panel(name, apply_transform=False)
        xmin, xmax, ymin, ymax = path.bbox()
        width, height = (xmax - xmin) * 10, (ymax - ymin) * 10
        if not all(math.isfinite(n) and 1 < n <= 5000 for n in (width, height)):
            raise ValueError("Invalid curve bounds")
        points, annotations = [], ["Printable reference only; NOT cutting ready.", "No seam allowances, grainlines, notches, cut quantities, printer calibration or physical fit verification."]
        edge_lengths[name] = []
        previous = None
        for index, curve in enumerate(path):
            if previous is not None and abs(previous - curve.start) > 1e-6:
                raise ValueError("Disconnected panel edges")
            length = float(curve.length()) * 10
            if not math.isfinite(length) or length <= 0 or length > 20000:
                raise ValueError("Invalid edge length")
            edge_lengths[name].append(length)
            is_curve = "curvature" in raw_panels[name]["edges"][index]
            segments = min(8192, max(16, math.ceil(length / 2))) if is_curve else 1
            start_index = len(points)
            for step in range(segments):
                p = curve.point(step / segments)
                points.append([round((p.real - xmin) * 10, 6), round((p.imag - ymin) * 10, 6)])
            annotations.append(f"Upstream edge {index}: {type(curve).__name__}, length {length:.4f} mm, sampled points {start_index}..{len(points)}.")
            previous = curve.end
        if previous is None or abs(previous - path[0].start) > 1e-6:
            raise ValueError("Open panel boundary")
        points.append(points[0])
        if len(points) > 20001:
            raise ValueError("Panel sample budget exceeded")
        area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:]))) / 2
        if not math.isfinite(area) or area < 1:
            raise ValueError("Degenerate panel area")
        panels.append({"id": name, "name": name, "points": points, "widthMm": round(width, 6), "heightMm": round(height, 6)})
        warnings.extend(f"Panel {name}: {annotation}" for annotation in annotations)
    stitches = []
    for pair in pattern.pattern["stitches"]:
        if len(pair) < 2:
            raise ValueError("Invalid stitch pair")
        a, b = pair[:2]
        lengths = []
        for side in (a, b):
            if side["panel"] not in edge_lengths or not isinstance(side["edge"], int) or not 0 <= side["edge"] < len(edge_lengths[side["panel"]]):
                raise ValueError("Invalid stitch edge reference")
            lengths.append(edge_lengths[side["panel"]][side["edge"]])
        stitches.append({"panelA": a["panel"], "edgeA": a["edge"], "panelB": b["panel"], "edgeB": b["edge"]})
        warnings.append(f"Stitch {a['panel']} upstream edge {a['edge']} ({lengths[0]:.4f} mm) <-> {b['panel']} upstream edge {b['edge']} ({lengths[1]:.4f} mm); flags {json.dumps(pair[2:])}; seam compatibility unverified.")
    stitches.sort(key=lambda item: (item["panelA"], item["edgeA"], item["panelB"], item["edgeB"]))
    if not stitches:
        raise ValueError("Missing assembly links")
    warnings += [
        "Printable reference only; NOT cutting ready. No allowances, grain, notches, cut counts, closure engineering, physical print calibration, seam-compatibility or fit validation.",
        "Actual upstream circular/quadratic/cubic curves are sampled, not replaced with vertex-only polygons. Nominal curve sample spacing 2 mm of arc length with uniform parameter steps (not a guaranteed chord-error tolerance); exact analytical curve bounds in mm; coordinates rounded to 0.000001 mm. Straight edges retain endpoints.",
        "Upstream panel self-intersection check returned false. This is a limited geometric check, not certification of construction or robust coverage of all parameter combinations.",
        "Geometry can reveal body dimensions, even if original body inputs are excluded from later exports. Keep artifacts private by default.",
    ]
    warnings.append("Stitch edge indices identify upstream edges, not sampled-polyline segment indices; point ranges are recorded in the panel warnings.")
    return {"schemaVersion": 1, "units": "mm", "panels": panels, "stitches": stitches, "warnings": warnings, "engineVersion": f"Sew Computer CPU adapter 1 / GarmentCode {COMMIT} / synthetic-body-v1", "inputDigest": digest, "family": family, "classification": "printable-reference", "assumptions": [item for item in warnings if "assum" in item.lower() or "synthetic" in item.lower() or "subset" in item.lower()]}


def render(geometry):
    from xml.sax.saxutils import escape
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    panels = geometry["panels"]
    pad = 10
    layouts = []
    for panel in panels:
        points = list(panel["points"])
        if panel.get("draft"):
            draft = panel["draft"]
            points.extend(draft["cutLine"])
            points.extend(draft["grainline"])
            for mark in draft["marks"]:
                horizontal, vertical = mark["point"]
                points.extend([(horizontal - 1.5, vertical - 1.5), (horizontal + 1.5, vertical + 1.5)])
        min_horizontal = min(0, *(point[0] for point in points))
        min_vertical = min(0, *(point[1] for point in points))
        max_horizontal = max(panel["widthMm"], *(point[0] for point in points))
        max_vertical = max(panel["heightMm"], *(point[1] for point in points))
        layouts.append((min_horizontal, min_vertical, max_horizontal - min_horizontal, max_vertical - min_vertical))
    page_sizes = []
    svg_width = max(layout[2] for layout in layouts) + pad * 2
    svg_height = sum(layout[3] + 35 + pad * 2 for layout in layouts) + 20
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width:.6f}mm" height="{svg_height:.6f}mm" viewBox="0 0 {svg_width:.6f} {svg_height:.6f}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="10" y="8" font-size="4">PRINTABLE REFERENCE - NOT CUTTING READY</text>']
    offset = 15
    with PdfPages("pattern.pdf", metadata={"Title": "Sew Computer pattern reference", "Creator": "Sew Computer CPU adapter 1", "CreationDate": None, "ModDate": None}) as pdf:
        for number, panel in enumerate(panels, 1):
            min_horizontal, min_vertical, content_width, content_height = layouts[number - 1]
            shift_horizontal, shift_vertical = pad - min_horizontal, pad - min_vertical
            width = max(content_width + pad * 2, 240 if panel.get("draft") else 0)
            height = content_height + 20 + pad * 2
            page_sizes.append(f"{panel['id']}: {width:.6f} x {height:.6f} mm")
            elements.append(f'<text x="10" y="{offset + 5:.6f}" font-size="4">{escape(panel["id"])} - reference only</text>')
            coords = ' '.join(f'{x + shift_horizontal:.6f},{y + offset + shift_vertical + 12:.6f}' for x, y in panel["points"])
            elements.append(f'<polyline points="{coords}" fill="#f2f2f2" stroke="#222222" stroke-width="0.3"/>')
            fig = plt.figure(figsize=(width / 25.4, height / 25.4))
            ax = fig.add_axes((0, 0, 1, 1))
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.axis("off")
            ax.text(10, 6, "PRINTABLE REFERENCE - NOT CUTTING READY", fontsize=8, va="top")
            ax.text(10, 12, f"{panel['id']} | {number}/{len(panels)} | page {width:.2f} x {height:.2f} mm", fontsize=8, va="top")
            ax.text(10, 18, "Input SHA-256: " + geometry["inputDigest"], fontsize=6, va="top")
            ax.text(10, height - 6, "Draft annotations supplied; toile required." if panel.get("draft") else "No seam allowances / grain / notches. No printer calibration or fit verification.", fontsize=7, va="top")
            xs = [p[0] + shift_horizontal for p in panel["points"]]
            ys = [p[1] + 16 + shift_vertical for p in panel["points"]]
            ax.plot(xs, ys, color="#222222", linewidth=0.6)
            if panel.get("draft"):
                draft = panel["draft"]
                contours = [(draft["cutLine"], "#8b4538", "-"), (draft["grainline"], "#457361", "--")]
                for contour, color, style in contours:
                    coordinates = ' '.join(f'{point[0] + shift_horizontal:.6f},{point[1] + offset + shift_vertical + 12:.6f}' for point in contour)
                    elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="0.4"/>')
                    ax.plot([point[0] + shift_horizontal for point in contour], [point[1] + shift_vertical + 16 for point in contour], color=color, linestyle=style, linewidth=0.6)
                for mark in draft["marks"]:
                    point = mark["point"]
                    elements.append(f'<circle cx="{point[0] + shift_horizontal:.6f}" cy="{point[1] + offset + shift_vertical + 12:.6f}" r="1.2" fill="none" stroke="#457361"><title>{escape(mark["kind"] + ": " + mark["label"])}</title></circle>')
                    ax.plot(point[0] + shift_horizontal, point[1] + shift_vertical + 16, marker="+" if mark["kind"] == "notch" else "o", markersize=3, color="#457361")
                label = f'Cut {draft["cutQuantity"]} {draft["material"]}; red=cut line; black=seam line; green=grain/marks. Transfer labels from JSON.'
                elements.append(f'<text x="10" y="{offset + 10:.6f}" font-size="3">{escape(label)}</text>')
                ax.text(10, 20, label, fontsize=6, va="top")
            offset += content_height + 35 + pad * 2
            pdf.savefig(fig)
            plt.close(fig)
    elements.append('</svg>')
    Path("pattern.svg").write_text(''.join(elements), encoding="utf-8")
    geometry["warnings"].append("Pattern PDF custom paper sizes from actual page geometry (not A4/Letter, not tiled): " + "; ".join(page_sizes))
    geometry["warnings"].append(f"SVG physical canvas {svg_width:.6f} x {svg_height:.6f} mm. PDF and SVG use sampled normalized paths at nominal 1:1 digital scale, not physically calibrated. No source revision/artifact ID is available to this engine API; input SHA-256 is the immutable document identity on sheets.")
    Path("pattern.json").write_text(json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def main():
    security = constrain()
    if sys.argv[1:] == ["--probe"]:
        print(json.dumps(security))
        return
    upstream = Path(sys.argv[1]).resolve()
    data = sys.stdin.buffer.read(512 * 1024 + 1)
    if len(data) > 512 * 1024:
        raise ValueError("Input budget exceeded")
    inputs = json.loads(data)
    if inputs["family"] not in ("shirt", "skirt", "trousers") or not re.fullmatch("[a-f0-9]{64}", inputs["inputDigest"]):
        raise ValueError("Invalid engine input")
    if inputs.get("design"):
        verify_dependencies()
        from shirt import compile_shirt
        geometry = compile_shirt(inputs, COMMIT)
    else:
        verify_runtime(upstream)
        sys.path.insert(0, str(upstream))
        body, warnings = build_body(inputs)
        piece, lowering = compile_piece(inputs, upstream, body)
        geometry = normalize(piece, inputs["inputDigest"], warnings + lowering + inputs["provenance"], inputs["family"])
    geometry["warnings"] += [f"Execution: non-root uid {security['uid']}; CPU 60 s; address space 2 GiB; output file 16 MiB; wall budget 90 s enforced by parent; {security['network']}.", "No filesystem/network-namespace sandbox is available on this host. Trusted pinned code only, clean environment without API credentials; private single-owner prototype, not public/multiuser or arbitrary-code execution."]
    render(geometry)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
