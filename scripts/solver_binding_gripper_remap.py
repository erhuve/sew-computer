"""Preserve declared source gripper controls across binding subdivision.

This derives a numerical recipe and explicit interpolation witnesses. It does
not install controls, place cloth, run a solver or infer crease-side frames.
"""

import copy
from fractions import Fraction
import hashlib

from solver_binding_source import (CONTROL_FIELDS, INSTANCE, _bounded, _encoded,
    is_refined_source, validate_binding_source)
from solver_binding_remap import _audit_selected, _basis, _equivalent_selection, _rational, _solve_three
from solver_cuff_source_binding import validate_cuff_source_binding
from solver_gripper_input import bind_material_grippers, mesh_identity


PROFILE = "source-binding-gripper-remap-v1"
POLICY = "same-control-exact-coefficient-nearest-binary64-v1"
MAX_MIGRATED_GRIPPERS = 16


def _sha(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _faces(source):
    return [source["triangles"][index:index + 3] for index in range(0, len(source["triangles"]), 3)]


def _rest_anchor(source, anchor):
    face = _faces(source)[anchor["triangleIndex"]]
    return [sum((Fraction(weight) * Fraction(source["restMeters"][vertex][axis])
                 for vertex, weight in zip(face, anchor["weights"])), Fraction()) for axis in range(3)]


def build_binding_gripper_remap(original_source, refined_source, *, subdivisions):
    """Derive an unchanged-control recipe on a validated bare numerical core."""
    original_bytes, refined_bytes = _bounded(original_source), _bounded(refined_source)
    if (type(subdivisions) is not int or not 1 <= subdivisions <= 4096
            or subdivisions & (subdivisions - 1)):
        raise ValueError("Explicit bounded dyadic gripper subdivisions required")
    if type(original_source) is not dict or is_refined_source(original_source):
        raise ValueError("Original v1 cuff gripper source required")
    validate_cuff_source_binding(original_source)
    validate_binding_source(refined_source)
    if CONTROL_FIELDS.intersection(refined_source):
        raise ValueError("Bare refined source core required before gripper migration")
    base = refined_source["baseUnit"]
    if any(key not in original_source for key in base) or _encoded(
            {key: original_source[key] for key in base}) != _encoded(base):
        raise ValueError("Original gripper source differs from immutable refined base unit")
    original_recipe = original_source.get("gripperActuation")
    if (type(original_recipe) is not dict or type(original_recipe.get("anchors")) is not list
            or not 1 <= len(original_recipe["anchors"]) <= MAX_MIGRATED_GRIPPERS):
        raise ValueError("One through sixteen explicitly captured source grippers required")
    bind_material_grippers(original_source, subdivisions)
    refinement = refined_source["bindingRefinement"]
    basis = _basis(refinement)
    source_faces = _faces(original_source)
    original_instances = {instance["id"]: instance for instance in base["instances"]}
    recipe = copy.deepcopy(original_recipe)
    recipe["meshSha256"] = mesh_identity(refined_source)
    bindings = []
    for index, (original_anchor, numerical_anchor) in enumerate(zip(original_recipe["anchors"], recipe["anchors"])):
        identity = original_anchor["instanceId"]
        template = base["sourceTemplates"][original_instances[identity]["templateId"]]
        offset = base["instanceOffsets"][identity]
        original_face = [vertex - offset for vertex in source_faces[original_anchor["triangleIndex"]]]
        matches = [i for i, face in enumerate(template["triangles"]) if face == original_face]
        if len(matches) != 1:
            raise ValueError("Unique unchanged oriented original gripper triangle required")
        original_parent = matches[0]
        item = {"gripperIndex": index, "id": original_anchor["id"], "instanceId": identity,
            "migrationKind": "refined-binding" if identity == INSTANCE else "unchanged-instance",
            "originalAnchor": copy.deepcopy(original_anchor), "originalLocalTriangleIndex": original_parent,
            "originalLocalVertices": original_face}
        if identity == INSTANCE:
            source_weights = [Fraction()] * len(basis[0])
            for vertex, weight in zip(original_face, original_anchor["weights"]):
                source_weights[vertex] = Fraction(weight)
            candidates = []
            final = refinement["finalLocalMesh"]
            for child, (face, parent) in enumerate(zip(final["triangles"], refinement["ultimateOriginalTriangleIndices"])):
                if parent != original_parent:
                    continue
                weights = _solve_three([[basis[vertex][original] for vertex in face] for original in original_face],
                                       [source_weights[vertex] for vertex in original_face])
                if any(sum((value * basis[vertex][old] for vertex, value in zip(face, weights)), Fraction()) != weight
                       for old, weight in enumerate(source_weights)):
                    raise ValueError("Gripper child leaves the original source material operator")
                candidates.append({"child": child, "parent": parent, "face": face, "weights": weights,
                                   "nonnegative": all(value >= 0 for value in weights)})
            selected, equivalent = _equivalent_selection(candidates)
            audit = _audit_selected(refinement["originalLocalMesh"]["verticesMeters"], final["verticesMeters"],
                                    basis, source_weights, selected)
            by_vertex = {entry["vertex"]: entry["weight"] for entry in audit["selectedNumericalWeights"]}
            numerical_anchor["weights"] = [by_vertex.get(vertex, 0.) for vertex in selected["face"]]
            local_index = selected["child"]
            item.update(candidates=[{"childTriangleIndex": candidate["child"], "sourceParentTriangle": candidate["parent"],
                "orientedVertices": candidate["face"], "coefficientNonnegative": candidate["nonnegative"],
                "exactWeights": [_rational(weight) for weight in candidate["weights"]]} for candidate in candidates],
                coefficientAudit=audit)
        else:
            mesh = refined_source["numericalMeshes"][identity]
            if mesh["triangles"] != template["triangles"]:
                raise ValueError("Other gripper instance must retain every original local triangle")
            local_index, equivalent = original_parent, [original_parent]
        canonical_index = refined_source["instanceTriangleOffsets"][identity] + local_index
        numerical_anchor["triangleIndex"] = canonical_index
        item.update(selectedLocalTriangleIndex=local_index, equivalentLocalTriangleIndices=equivalent,
            selectedCanonicalTriangleIndex=canonical_index, numericalAnchor=copy.deepcopy(numerical_anchor),
            originalWeightSum=_rational(sum(map(Fraction, original_anchor["weights"]), Fraction())),
            storedCoordinateResidualMeters=[_rational(new - old) for new, old in
                zip(_rest_anchor(refined_source, numerical_anchor), _rest_anchor(original_source, original_anchor))])
        bindings.append(item)
    # The established primitive must also admit the complete derived recipe;
    # this does not establish original-source correspondence by itself.
    numerical_source = {**refined_source, "gripperActuation": recipe}
    bind_material_grippers(numerical_source, subdivisions)
    result = {"profile": PROFILE, "policy": POLICY, "accepted": False, "solverReady": False, "executable": False,
        "subdivisions": subdivisions, "originalSourceSha256": hashlib.sha256(original_bytes).hexdigest(),
        "refinedSourceSha256": hashlib.sha256(refined_bytes).hexdigest(), "baseUnitSha256": _sha(base),
        "originalRecipeSha256": _sha(original_recipe), "originalMeshSha256": mesh_identity(original_source),
        "refinedMeshSha256": mesh_identity(refined_source), "originalRecipe": copy.deepcopy(original_recipe),
        "numericalRecipe": recipe, "anchors": bindings,
        "limitations": [
            "Source-bound gripper recipe derivation only; no controls are installed, no placement is generated and no solver trajectory is executed.",
            "Targets, activation, stiffness, identities and order remain exactly as captured. Only mesh identity and source-derived triangle/weight bindings change.",
            "Once-rounded numerical coefficients follow the explicit seam-remap conversion policy; exact pullback and stored-coordinate reconstruction residuals are separate observations.",
            "Equivalent child triangles name the same point-grip operator and select no material normal, crease side, solid tool or construction phase.",
            "Unchanged instances retain original local ordered triangles and raw weights while their canonical indices are rebuilt.",
            "Rest-coordinate residuals can change spring extension even with exact coefficient pullback. World placement, force/work comparison and captured replay require separate validation.",
            "This bounded migration accepts one through sixteen original grippers. Source-pattern and refined geometry validity use the existing strict source validators."]}
    if _bounded(original_source) != original_bytes or _bounded(refined_source) != refined_bytes:
        raise ValueError("Gripper source changed during migration")
    return copy.deepcopy(result)


def validate_binding_gripper_remap(original_source, refined_source, descriptor, *, subdivisions):
    """Require complete strict source/control/arithmetic rederivation."""
    claimed = _bounded(descriptor)
    expected = build_binding_gripper_remap(original_source, refined_source, subdivisions=subdivisions)
    if claimed != _encoded(expected):
        raise ValueError("Gripper migration differs from complete source rederivation")
    return expected
