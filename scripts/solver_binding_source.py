"""Strict derived numerical cuff source; original source remains immutable.

This profile admits a declared binary64 approximation of five binding anchors.
It does not execute a construction phase, select crease-side material frames,
or grant garment/material acceptance.
"""

import hashlib
import importlib
import json
from pathlib import Path
import stat
import sys


PROFILE = "source-left-binding-refined-unit-v1"
PROFILE_PREFIX = "source-left-binding-"
INSTANCE = "opening_binding_left_left:shell"
RESERVED_FIELDS = frozenset(("baseUnit", "bindingRefinement", "bindingSeamRemap"))
CONTROL_FIELDS = frozenset(("placedMeters", "sewingActuation", "gripperActuation", "foldActuation",
                            "assemblySchedule", "bindingRefinedDiagnostic"))
MAX_SOURCE_BYTES = 8 * 1024 ** 2
REQUIRED_HELPER_FILES = ("solver_binding_source.py", "solver_binding_remap.py", "solver_binding_refinement.py",
    "solver_crease_mesh.py", "solver_cuff_source_binding.py", "solver_engine_source_namespace.py",
    "solver_cuff_construction.py", "spike-full-shirt.py", "solver_sewing_input.py",
    "solver_sewing_activation_schedule.py", "solver_attempt_journal.py", "solver_process_budget.py")
REQUIRED_ENGINE_FILES = ("shirt.py", "assembly.py", "cloth_domain.py", "meshing.py", "quality_meshing.py",
                        "simulation_validation.py", "embedded_constraints.py", "inspection_gltf.py")


def is_refined_source(source):
    """Reserve both the profile namespace and its fields against downgrade."""
    return (type(source) is dict and (bool(RESERVED_FIELDS.intersection(source))
        or type(source.get("profile")) is str and source["profile"].startswith(PROFILE_PREFIX)))


def _encoded(value):
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeEncodeError) as error:
        raise ValueError("Finite raw JSON refined source required") from error
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("Refined source exceeds byte budget")
    return data


def _bounded(value):
    pending, remaining = [(value, 0)], 1000000
    while pending:
        item, depth = pending.pop()
        remaining -= 1
        if remaining < 0 or depth > 40:
            raise ValueError("Refined source exceeds structure budget")
        if type(item) is dict:
            if any(type(key) is not str for key in item) or len(item) * 2 > remaining:
                raise ValueError("Bounded string-keyed source objects required")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif type(item) is list:
            if len(item) > remaining:
                raise ValueError("Refined source exceeds array budget")
            pending.extend((child, depth + 1) for child in item)
        elif item is None or type(item) in (bool, float):
            pass
        elif type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Refined source integer exceeds bound")
        elif type(item) is str:
            if len(item) > MAX_SOURCE_BYTES:
                raise ValueError("Refined source string exceeds bound")
        else:
            raise ValueError("Refined source must contain only raw JSON values")
    return _encoded(value)


def _copy(value):
    return json.loads(_encoded(value))


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _regular_digest(path):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Complete regular source dependencies required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_binding_source_namespace(source_root=None, *, source_digests=None):
    """Reject missing, unlisted or mixed captured helpers before importing.

    Captured callers must pass their snapshot directory and complete manifest.
    Repository callers may omit both. Engine namespace isolation remains in
    the existing engine helper; this extends the same rule to local helpers.
    """
    origin = Path(__file__)
    if origin.is_symlink():
        raise ValueError("Refined source helper cannot be linked")
    root = origin.resolve().parent
    if (source_root is None) != (source_digests is None):
        raise ValueError("Captured namespace and digest manifest must be supplied together")
    if source_root is not None:
        requested = Path(source_root)
        if requested.is_symlink() or requested.resolve() != root:
            raise ValueError("Refined source verifier belongs to another helper namespace")
    if source_digests is not None and type(source_digests) is not dict:
        raise ValueError("Captured source digest manifest required")
    try:
        checked = {name: _regular_digest(root / name) for name in REQUIRED_HELPER_FILES}
        for name in REQUIRED_HELPER_FILES:
            module_name = name[:-3]
            if "-" in module_name or module_name not in sys.modules:
                continue
            cached = getattr(sys.modules[module_name], "__file__", None)
            if type(cached) is not str or Path(cached).resolve() != root / name:
                raise ValueError("Cached refined source helper belongs to another namespace: " + module_name)
        if source_digests is not None and any(source_digests.get(name) != value for name, value in checked.items()):
            raise ValueError("Required refined source helper missing or changed in capture manifest")
        namespace = importlib.import_module("solver_engine_source_namespace")
        if Path(namespace.__file__).resolve() != root / "solver_engine_source_namespace.py":
            raise ValueError("Engine namespace helper came from a different source root")
        if tuple(namespace.ENGINE_FILES) != REQUIRED_ENGINE_FILES:
            raise ValueError("Refined engine dependency closure requires explicit migration")
        engine_root = namespace.engine_source_root(__file__)
        if source_digests is not None and engine_root != root / "services/engine":
            raise ValueError("Captured refined source requires its adjacent captured engine tree")
        checked.update({"services/engine/" + name: _regular_digest(engine_root / name) for name in REQUIRED_ENGINE_FILES})
        if source_digests is not None and any(source_digests.get(name) != value for name, value in checked.items()):
            raise ValueError("Required refined engine dependency missing or changed in capture manifest")
    except OSError as error:
        raise ValueError("Complete regular refined source dependency closure required") from error
    return checked


def _helpers():
    hashes = assert_binding_source_namespace()
    modules = [importlib.import_module(name) for name in
               ("solver_binding_refinement", "solver_binding_remap")]
    # Imports must neither select another cached module nor escape the checked
    # directory through a fallback in sys.path.
    if hashes != assert_binding_source_namespace():
        raise ValueError("Refined source dependencies changed during import")
    return hashes, modules


def build_binding_source(base_unit):
    """Rebuild the complete numerical mesh/rows under a new explicit profile."""
    before = _bounded(base_unit)
    if type(base_unit) is not dict or is_refined_source(base_unit):
        raise ValueError("An immutable original v1 cuff unit is required")
    hashes, (refinement_module, remap_module) = _helpers()
    refinement = refinement_module.build_binding_refinement(base_unit)
    remap = remap_module.build_binding_seam_remap_descriptor(base_unit, refinement)
    remaps = {row["rowIndex"]: row for row in remap["rows"]}
    if len(remaps) != 5:
        raise ValueError("Exactly five separately derived binding anchors required")
    meshes, offsets, face_offsets, rest, triangles = {}, {}, {}, [], []
    for instance in base_unit["instances"]:
        name = instance["id"]
        original = base_unit["sourceTemplates"][instance["templateId"]]
        start = base_unit["instanceOffsets"][name]
        if name == INSTANCE:
            local = refinement["finalLocalMesh"]
            mesh = {"verticesMeters": [point + [0.] for point in local["verticesMeters"]],
                    "triangles": _copy(local["triangles"])}
        else:
            mesh = {"verticesMeters": _copy(base_unit["restMeters"][start:start + len(original["restPositions"])]),
                    "triangles": _copy(original["triangles"])}
        offsets[name], face_offsets[name], meshes[name] = len(rest), len(triangles) // 3, mesh
        rest.extend(mesh["verticesMeters"])
        triangles.extend(vertex + offsets[name] for face in mesh["triangles"] for vertex in face)
    original_bundle = base_unit["embeddedConstraints"]
    rows, correspondence = [], []
    for index, original_row in enumerate(original_bundle["constraints"]):
        row = _copy(original_row)
        if index in remaps:
            evidence = remaps[index]
            if evidence["originalRowSha256"] != _digest(original_row) or _encoded(evidence["originalRow"]) != _encoded(original_row):
                raise ValueError("Remapped numerical row differs from its immutable source row")
            weights = _copy(evidence["selectedNumericalWeights"])
            samples = [sample for sample in row["sourceSamples"] if sample["instanceId"] == INSTANCE]
            if len(samples) != 1:
                raise ValueError("Unique original binding-side source sample required")
            samples[0]["weights"] = weights
            row["terms"] = sorted([term for term in row["terms"] if term["instanceId"] != INSTANCE]
                + [{"instanceId": INSTANCE, "vertex": item["vertex"], "coefficient": -item["weight"]} for item in weights],
                key=lambda term: (term["instanceId"], term["vertex"]))
        rows.append(row)
        correspondence.append({"rowIndex": index, "registrationId": original_row["registrationId"],
            "memberIndex": original_row["memberIndex"], "fraction": original_row["fraction"],
            "originalRowSha256": _digest(original_row), "numericalRowSha256": _digest(row),
            "bindingAnchorRemapped": index in remaps})
    bundle = {"kind": "derived-embedded-sewing-coupling", "solverReady": False,
        **{key: _copy(original_bundle[key]) for key in ("units", "sourceArcUnits", "topology", "registrations")},
        "constraints": rows, "originalBundleSha256": _digest(original_bundle),
        "sourceIdentities": {name: {"originalPanelDigest": original_bundle["sourceIdentities"][name]["panelDigest"],
            "originalTemplateId": original_bundle["sourceIdentities"][name]["templateId"],
            "numericalMeshSha256": _digest(mesh), "vertexCount": len(mesh["verticesMeters"])} for name, mesh in meshes.items()},
        "limitations": ["Original sampled source rows remain immutable in baseUnit; five derived negative anchors use explicitly rounded child coefficients.",
            "Recorded rational pullback residuals distinguish this numerical approximation from exact original material-operator equality.",
            "This coupling does not establish continuous stitching, crease-side material frames or construction phase completion."]}
    result = {"profile": PROFILE, "accepted": False, "solverReady": False, "units": "m", "sourceArcUnits": "mm",
        "baseUnit": _copy(base_unit), "bindingRefinement": refinement, "bindingSeamRemap": remap,
        "instances": _copy(base_unit["instances"]), "numericalMeshes": meshes,
        "instanceOffsets": offsets, "instanceTriangleOffsets": face_offsets, "restMeters": rest, "triangles": triangles,
        "embeddedConstraints": bundle, "sourceRowCorrespondence": correspondence,
        "derivationCodeDigests": hashes,
        "limitations": ["Explicit derived numerical research source; no construction phase or garment acceptance.",
            "Original templates, cut boundaries, source annotations and all forty original rows are retained only in immutable baseUnit.",
            "Refinement retains the original vertex prefix and full cut domain under the separately recorded splitter and binary64 limits.",
            "Five numerical anchors use prescribed nearest-binary64 conversion; thirty-five local rows remain byte-identical by canonical JSON.",
            "Crease-side material frames and original-gripper migration are not supplied by this profile; normal-offset sewing is unsupported.",
            "Placement and numerical controls require separate captured input validation; no material calibration, settling or convergence is implied."]}
    if _bounded(base_unit) != before or hashes != assert_binding_source_namespace():
        raise ValueError("Original unit or derivation dependencies changed during rebuilding")
    return _copy(result)


def validate_binding_source(source):
    """Strictly rederive the entire source core while controls bind separately."""
    _bounded(source)
    if type(source) is not dict or source.get("profile") != PROFILE or not RESERVED_FIELDS.issubset(source):
        raise ValueError("Exact complete source-left-binding-refined-unit-v1 profile required")
    expected = build_binding_source(source["baseUnit"])
    actual = {key: value for key, value in source.items() if key not in CONTROL_FIELDS}
    if _encoded(actual) != _encoded(expected):
        raise ValueError("Refined numerical source differs from complete base/lineage/row rederivation")
    binding = expected["bindingRefinement"]["sourceBinding"]
    return {"profile": "binding-refined-source-binding-v1", "sourceProfile": PROFILE,
        "accepted": False, "solverReady": False, "sourceCorrespondenceValidated": True,
        "sourceUnitSha256": _digest(expected["baseUnit"]), "refinementDescriptorSha256": _digest(expected["bindingRefinement"]),
        "seamRemapDescriptorSha256": _digest(expected["bindingSeamRemap"]), "numericalSourceCoreSha256": _digest(expected),
        "originalEmbeddedConstraintsSha256": _digest(expected["baseUnit"]["embeddedConstraints"]),
        "numericalEmbeddedConstraintsSha256": _digest(expected["embeddedConstraints"]),
        "rowGroups": _copy(binding["rowGroups"]), "phaseConstraintRows": _copy(binding["phaseConstraintRows"]),
        "counts": {**binding["counts"], "vertices": len(expected["restMeters"]),
            "triangles": len(expected["triangles"]) // 3, "remappedBindingAnchors": 5, "unchangedLocalRows": 35},
        "checkedSourceDigests": expected["derivationCodeDigests"],
        "scope": "Strict original source and declared numerical-refinement rederivation with recorded coefficient approximation; source phases remain declarative and unexecuted.",
        "limitations": expected["limitations"]}
