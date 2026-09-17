import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


def panel(width, height, origin, rotation=0.0, divisions=4, rows=None):
    rows = rows or divisions
    rest = [[width * column / divisions, height * row / rows, 0.0] for row in range(rows + 1) for column in range(divisions + 1)]
    placed = [[point[0] + origin[0], point[1] * math.cos(rotation) + origin[1], point[1] * math.sin(rotation) + origin[2]] for point in rest]
    triangles = []
    for row in range(rows):
        for column in range(divisions):
            index = row * (divisions + 1) + column
            triangles.extend([index, index + 1, index + divisions + 1, index + 1, index + divisions + 2, index + divisions + 1])
    return rest, placed, triangles


def fixture(name, profile):
    if name == "gather" and profile == "gather-refinement":
        panels = [panel(0.2, 0.2, (0, 0, 0.4)), panel(0.2, 0.3, (0.22, 0, 0.4), rotation=0.2, rows=12)]
        return panels, [(row * 5 + 4, 25 + row * 15) for row in range(5)]
    if name == "equal-seam":
        panels = [panel(0.2, 0.2, (0, 0, 0.4)), panel(0.2, 0.2, (0.22, 0, 0.4))]
        seams = [(row * 5 + 4, 25 + row * 5) for row in range(5)]
    elif name == "gather":
        panels = [panel(0.2, 0.2, (0, 0, 0.4)), panel(0.2, 0.3, (0.22, 0, 0.4))]
        seams = [(row * 5 + 4, 25 + row * 5) for row in range(5)]
    elif name == "layered-fold":
        panels = [panel(0.2, 0.08, (0, 0, 0.4)), panel(0.2, 0.08, (0, 0.02, 0.405), math.pi * 0.75)]
        seams = [(column, 25 + column) for column in range(5)]
    else:
        panels = [panel(0.2, 0.2, (0, 0, 0.4)), panel(0.2, 0.2, (0.16, 0, 0.41))]
        seams = [(9, 30), (19, 40)]
    return panels, seams


def source_fixture():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/engine"))
    from meshing import mesh_panel
    from shirt import compile_shirt
    from simulation_validation import validate_rest_mesh

    inputs = {"bodyMm": {"bust": 960, "hip": 1000, "shoulder": 400}, "lengthMm": 650, "easeMm": 100, "flare": 1, "inputDigest": "synthetic-solver-spike", "provenance": [], "design": {"seamAllowanceMm": 10, "opening": "buttons", "placketWidthMm": 30, "hem": "curved-back-tail", "tailExtensionMm": 100, "buttonSpacingMm": 80, "frill": "front-opening", "frillWidthMm": 35, "frillFullness": 1.8, "sleeves": "long", "sleeveLengthMm": 550, "cuff": "button", "cuffDepthMm": 55, "cuffCircumferenceMm": 220, "collar": "stand-and-fall", "collarStandMm": 30, "collarFallMm": 60}}
    pattern = compile_shirt(inputs, "synthetic-solver-spike")
    panels = []
    boundaries = []
    metadata = {"patternSha256": hashlib.sha256(json.dumps(pattern, sort_keys=True).encode()).hexdigest(), "sourceTemplates": [], "seamPaths": [], "pins": []}
    offset = 0
    for template_id, resolution in (("placket_left", 47.5), ("frill_left", 42.75)):
        template = next(item for item in pattern["panels"] if item["id"] == template_id)
        mesh = mesh_panel(template, resolution)
        validation = validate_rest_mesh(template, mesh)
        boundary = next(item for item in mesh["boundaries"] if item["name"] == "right")
        samples = boundary["samples"]
        anchors = []
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            sample = min(samples, key=lambda item: abs(item["arcMm"] - fraction * boundary["lengthMm"]))
            if abs(sample["arcMm"] - fraction * boundary["lengthMm"]) > 1e-6:
                raise ValueError("Source mesh lacks exact required registration anchor")
            anchors.append(offset + sample["vertex"])
        boundaries.append(anchors)
        metadata["seamPaths"].append([offset + sample["vertex"] for sample in samples])
        rest = [[point[0] / 1000, point[1] / 1000, 0] for point in mesh["restPositions"]]
        width = max(point[0] for point in rest)
        if template_id == "placket_left":
            placed = [[point[0] - width, point[1], 0.4] for point in rest]
            metadata["pins"] = [index for index, point in enumerate(rest) if point[0] == 0 and point[1] in (0, max(vertex[1] for vertex in rest))]
        else:
            placed = [[width - point[0] + 0.02, point[1] * math.cos(0.2), 0.4 + point[1] * math.sin(0.2)] for point in rest]
        panels.append((rest, placed, [vertex for face in mesh["triangles"] for vertex in face]))
        metadata["sourceTemplates"].append({"templateId": template_id, "mesh": mesh, "restValidation": validation})
        offset += len(rest)
    return panels, list(zip(*boundaries)), metadata


def write_gltf(path, positions, indices, label):
    import numpy as np

    source = np.asarray(positions, dtype="<f4")
    vertices = np.stack((source[:, 0], source[:, 2], -source[:, 1]), axis=1)
    faces = np.asarray(indices, dtype="<u4")
    vertex_bytes = vertices.tobytes()
    index_bytes = faces.tobytes()
    data = vertex_bytes + index_bytes
    document = {
        "asset": {"version": "2.0", "generator": "Sew Computer solver feasibility; synthetic microfixture only"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": label}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode(), "byteLength": len(data)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(vertex_bytes)}, {"buffer": 0, "byteOffset": len(vertex_bytes), "byteLength": len(index_bytes)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3", "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()}, {"bufferView": 1, "componentType": 5125, "count": len(faces), "type": "SCALAR"}],
    }
    path.write_text(json.dumps(document))


def run_fixture(name, steps, output, solver_name, profile, timestep, translated):
    import newton
    import numpy as np
    import warp as wp

    started = time.monotonic()
    from solver_spike_geometry import inspect_geometry

    source_metadata = None
    if name == "source-gather":
        panels, seams, source_metadata = source_fixture()
    else:
        panels, seams = fixture(name, profile)
    if translated:
        panels = [(rest, [[point[0] + 0.1, point[1] - 0.2, point[2] + 0.3] for point in placed], triangles) for rest, placed, triangles in panels]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    all_indices = []
    mappings = []
    rest_vertices = []
    placed_vertices = []
    for panel_index, (rest, placed, triangles) in enumerate(panels):
        offset = len(builder.particle_q)
        stiffness = 10000 if profile == "gather-refinement" else 1000
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=rest, indices=triangles, density=0.2, tri_ke=stiffness, tri_ka=stiffness, tri_kd=0.01, edge_ke=0.01, edge_kd=0.001, add_springs=solver_name == "xpbd", spring_ke=stiffness, spring_kd=0.01, particle_radius=0.001, validate_mesh=True)
        builder.particle_q[offset:offset + len(rest)] = [wp.vec3(*point) for point in placed]
        all_indices.extend(offset + vertex for vertex in triangles)
        mappings.extend({"panel": panel_index, "sourceVertex": vertex, "restMeters": rest[vertex], "grain": [0, 1]} for vertex in range(len(rest)))
        rest_vertices.extend(rest)
        placed_vertices.extend(placed)
    for first, second in seams:
        builder.add_spring(first, second, ke=1000000 if profile == "gather-refinement" else 100000, kd=0.01, control=0)
        builder.spring_rest_length[-1] = 0.0
    pins = source_metadata["pins"] if source_metadata else [0, len(panels[0][0]) - int(math.sqrt(len(panels[0][0])))]
    for pin in pins:
        builder.particle_mass[pin] = 0.0
    if solver_name == "vbd":
        builder.color(include_bending=True)
    model = builder.finalize(device="cpu")
    if solver_name == "vbd":
        solver = newton.solvers.SolverVBD(model, iterations=10, particle_enable_self_contact=True, particle_self_contact_margin=0.003, particle_self_contact_gap=0.001)
    else:
        solver = newton.solvers.SolverXPBD(model, iterations=10)
    state = model.state()
    next_state = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    initial_rest = model.tri_poses.numpy().copy()
    for _ in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, timestep)
        state, next_state = next_state, state
    positions = state.particle_q.numpy()
    if not np.isfinite(positions).all():
        raise ValueError("Solver produced nonfinite positions")
    residuals = [float(np.linalg.norm(positions[first] - positions[second]) * 1000) for first, second in seams]
    edge_ratios = []
    for offset in range(0, len(all_indices), 3):
        triangle = all_indices[offset:offset + 3]
        for local in range(3):
            first, second = triangle[local], triangle[(local + 1) % 3]
            rest_length = np.linalg.norm(np.asarray(rest_vertices[first]) - rest_vertices[second])
            edge_ratios.append(float(np.linalg.norm(positions[first] - positions[second]) / rest_length))
    canonical = {"fixture": name, "units": "m", "rest": rest_vertices, "placement": placed_vertices, "positions": positions.tolist(), "triangles": all_indices, "sourceMappings": mappings, "seamPairs": seams}
    if source_metadata:
        canonical["sourcePattern"] = source_metadata
    (output / f"{name}.json").write_text(json.dumps(canonical))
    write_gltf(output / f"{name}.gltf", positions, all_indices, name)
    simulation_seconds = time.monotonic() - started
    seam_paths = [[pair[side] for pair in seams] for side in (0, 1)]
    if name in ("equal-seam", "gather"):
        seam_paths = [list(range(4, len(panels[0][0]), 5)), list(range(len(panels[0][0]), len(rest_vertices), 5))]
    if source_metadata:
        seam_paths = source_metadata["seamPaths"]
    geometry = inspect_geometry(rest_vertices, placed_vertices, positions, all_indices, mappings, seam_paths)
    return {"fixture": name, "profile": profile, "translated": translated, "solver": solver_name, "particles": len(positions), "triangles": len(all_indices) // 3, "seamConstraints": len(seams), "steps": steps, "dtSeconds": timestep, "iterations": 10, "simulationWallSeconds": simulation_seconds, "wallSeconds": time.monotonic() - started, "seamResidualMaxMm": max(residuals), "seamResidualP95Mm": float(np.percentile(residuals, 95)), "seamResidualMeanMm": float(np.mean(residuals)), "edgeStretchMin": min(edge_ratios), "edgeStretchMax": max(edge_ratios), "restTensorUnchanged": bool(np.array_equal(initial_rest, model.tri_poses.numpy())), "finite": True, "sourceMappingCount": len(mappings), "positionsSha256": hashlib.sha256(positions.tobytes()).hexdigest(), "geometry": geometry, "classification": "experimental microfixture; not accepted assembly"}


def main():
    parser = argparse.ArgumentParser(description="Bounded synthetic Newton CPU solver feasibility; no garment or fit claims")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--solver", choices=["vbd", "xpbd"], default="vbd")
    parser.add_argument("--profile", choices=["baseline", "gather-refinement"], default="baseline")
    parser.add_argument("--fixture", choices=["equal-seam", "gather", "layered-fold", "localized-overlap", "source-gather"])
    parser.add_argument("--timestep-denominator", type=int, choices=[240, 480], default=240)
    parser.add_argument("--translated", action="store_true")
    parser.add_argument("--verify-profile", choices=["refined-gather-v1"])
    args = parser.parse_args()
    source_files = {name: Path(__file__).with_name(name) for name in ("spike-3d-solver.py", "solver_spike_geometry.py", "solver-spike-profiles.json")}
    if args.fixture == "source-gather":
        source_files.update({name: Path(__file__).resolve().parents[1] / "services/engine" / name for name in ("shirt.py", "meshing.py", "simulation_validation.py")})
    source_digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files.items()}
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be 1..1000")
    if args.verify_profile and (args.fixture != "gather" or args.profile != "gather-refinement"):
        parser.error("refined-gather-v1 requires the gather fixture and gather-refinement inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").unlink(missing_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(args.output.resolve() / "kernel-cache")
    resource.setrlimit(resource.RLIMIT_CPU, (240, 250))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    import warp as wp

    wp.init()
    wp.set_device("cpu")
    results = []
    for name in ([args.fixture] if args.fixture else ("equal-seam", "gather", "layered-fold", "localized-overlap")):
        result = run_fixture(name, args.steps, args.output, args.solver, args.profile, 1 / args.timestep_denominator, args.translated)
        results.append(result)
        print(json.dumps(result), flush=True)
    if source_digests != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files.items()}:
        raise RuntimeError("Experiment source changed during execution; rerun before recording evidence")
    report = {"versions": {name: importlib.metadata.version(name) for name in ("newton", "warp-lang", "numpy", "shapely")}, "sourceDigests": source_digests, "devices": [str(device) for device in wp.get_devices()], "maxResidentKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "fixtures": results, "limitations": ["Discrete nonadjacent surface collision checks only; no swept or thickness oracle", "No calibrated material, anisotropy, body, turning or allowance validation", "Layered-fold fixture starts from rigidly rotated flat panels; not a validated turning operation", "Point springs only; continuous interval gather constraints remain to be implemented", "No garment acceptance thresholds established"]}
    if args.verify_profile:
        frozen = json.loads(Path(__file__).with_name("solver-spike-profiles.json").read_text())[args.verify_profile]
        result = results[0]
        geometry = result["geometry"]
        checks = {"seam": result["seamResidualMaxMm"] <= frozen["maxSeamResidualMm"], "strain": frozen["minEdgeStretch"] <= result["edgeStretchMin"] <= result["edgeStretchMax"] <= frozen["maxEdgeStretch"], "area": frozen["minAreaRatio"] <= geometry["deformedAreaRatioMin"] <= geometry["deformedAreaRatioMax"] <= frozen["maxAreaRatio"], "arcLength": all(abs(side["ratio"] - 1) <= frozen["maxArcLengthRelativeError"] for side in geometry["seamPolylineLengths"]), "surfaceIntersection": geometry["intersectingPairCount"] <= frozen["maxNonadjacentSurfaceIntersections"], "degenerate": geometry["collapsedTriangleCount"] <= frozen["maxCollapsedTriangles"], "rest": result["restTensorUnchanged"] == frozen["restTensorUnchanged"]}
        report["profileAssessment"] = {"profile": args.verify_profile, "frozenContents": frozen, "checks": checks, "passed": all(checks.values())}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.verify_profile and not report["profileAssessment"]["passed"]:
        raise SystemExit("Frozen microfixture profile failed; see report.json")


if __name__ == "__main__":
    main()
