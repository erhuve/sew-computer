"""Source-bound declarative cuff phases; no numerical execution or acceptance.

The inspection assembly graph records final membership, not a turning program.
This explicitly selected research policy refines its cuff dependency milestones
without modifying that graph or claiming its prose uniquely fixes layer order.
"""

import copy
import hashlib
import json
from pathlib import Path
import re
import sys

PROFILE = "sew-cuff-construction-phases/1"
POLICY = "inner-facing-first-outer-shell-last-v1"


def _encoded(value):
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Finite JSON construction source required") from error
    if len(data) > 4 * 1024 ** 2:
        raise ValueError("Construction source or plan exceeds bounded JSON budget")
    return data


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _validated_sources(pattern, inventory, assembly):
    if not all(isinstance(value, dict) for value in (pattern, inventory, assembly)):
        raise ValueError("Pattern, inventory and assembly objects required")
    for value in (pattern, inventory, assembly):
        _encoded(value)
    engine = str(Path(__file__).resolve().parents[1] / "services/engine")
    if engine not in sys.path:
        sys.path.insert(0, engine)
    from assembly import compile_assembly, compile_inventory

    try:
        templates = {panel["id"] for panel in pattern["panels"]}
        components = set(pattern["drafting"]["components"])
        if not {"cuff_left", "cuff_right", "sleeve_left", "sleeve_right"} <= templates:
            raise ValueError("The explicit long-sleeve button-cuff source is required")
        # Cuffs require long sleeves in this trusted compiler. Every other
        # construction selector is recoverable from the full source inventory.
        construction = {"sleeves": "long", "cuff": "button",
            "collar": "stand-and-fall" if "collar_fall" in templates else "stand" if "collar_stand" in templates else "none",
            "opening": "buttons" if "placket_left" in templates else "none",
            "frill": "front-opening" if "frill_left" in templates else "none",
            "hem": "curved-back-tail" if "tails" in components else "straight"}
        _, expected_inventory = compile_inventory(_encoded(pattern), construction)
        captured_digest = inventory["patternDigest"]
        if not isinstance(captured_digest, str) or re.fullmatch(r"[a-f0-9]{64}", captured_digest) is None:
            raise ValueError("Captured inventory pattern digest required")
        # The API receives a parsed pattern, not its original whitespace bytes.
        # Preserve that captured byte digest and separately bind canonical JSON.
        expected_inventory["patternDigest"] = captured_digest
        if _encoded(inventory) != _encoded(expected_inventory):
            raise ValueError("Inventory differs from reconstructed source inventory")
        expected_assembly = compile_assembly(pattern, inventory)
        if _encoded(assembly) != _encoded(expected_assembly):
            raise ValueError("Assembly differs from reconstructed source membership or dependencies")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Incomplete or malformed construction source") from error


def build_cuff_phase_plan(pattern, inventory, assembly, *, side, policy):
    """Derive one cuff's research phase plan from the complete trusted source.

    Star selectors name all fractions of the existing registration/member pair.
    A pending selector is absent from the numerical force law, not a spring at
    its initial target. This module defines no implementation of that behavior.
    """
    if type(side) is not str or side not in ("left", "right"):
        raise ValueError("Explicit left or right cuff side required")
    if type(policy) is not str or policy != POLICY:
        raise ValueError("Explicit inner-facing-first/outer-shell-last research policy required")
    _validated_sources(pattern, inventory, assembly)
    cuff, sleeve = f"cuff_{side}", f"sleeve_{side}"
    prefix, gather = f"cuff:{side}", f"cuff_gather_{side}"
    binding_ids = [f"bind_opening_{side}_{edge}" for edge in ("left", "right")]
    perimeter_ids = [f"perimeter:{cuff}:{edge}" for edge in ("extension", "end", "outer", "start")]
    source_ids = [*binding_ids, gather, *perimeter_ids]
    operations = {operation["id"]: operation for operation in assembly["operations"]}
    sources = [copy.deepcopy(operations[identifier]) for identifier in source_ids]
    members = operations[gather]["participants"]
    if [(member["instanceId"], member["edgeName"]) for member in members] != [
            (f"{sleeve}:shell", "wrist"), (f"{cuff}:shell", "attachment"), (f"{cuff}:facing", "attachment")]:
        raise ValueError("Exact three-member wrist/shell/facing source group required")
    required_instances = sorted({member["instanceId"] for operation in sources for member in operation["participants"]})
    if len(required_instances) != 5:
        raise ValueError("Cuff, sleeve and both binding fabric instances required")

    def selector(registration, member=1):
        return {"registrationId": registration, "memberIndex": member}

    binding_rows = [selector(identifier) for identifier in binding_ids]
    inner_row, outer_row = selector(gather, 2), selector(gather, 1)
    perimeter_rows = [selector(identifier) for identifier in perimeter_ids]
    initial_milestone = f"{gather}:inner-attached"
    turned_milestone = f"{prefix}:turned"
    allowance_milestone = f"{prefix}:shell-attachment-allowance-turned-under"
    opening_id = f"{cuff}:turn-opening:attachment-shell"
    attach_phase, turn_phase = f"{prefix}:attach-facing", f"{prefix}:turn"
    allowance_phase, close_phase = f"{prefix}:turn-under-shell-attachment", f"{prefix}:close-shell"
    phases, completion_rules, lifecycle = [], [], []

    def phase(identifier, kind, source_operations, requires, produces, activates, held, gates, openings=()):
        phases.append({"id": identifier, "kind": kind, "sourceOperationIds": source_operations,
            "requiresMilestones": requires, "producesMilestones": produces,
            "activatesStarRows": copy.deepcopy(activates), "requiresHeldStarRows": copy.deepcopy(held),
            "requiresGates": gates, "requiresReservedOpenings": list(openings)})

    def completion(operation, starting_phase, milestones):
        completion_rules.append({"milestone": f"{operation}:complete", "allOfMilestones": milestones})
        lifecycle.append({"sourceOperationId": operation, "startedByPhase": starting_phase,
            "completeOnlyAtMilestone": f"{operation}:complete",
            "betweenStartAndCompletion": "active; source operation remains incomplete"})

    for edge, identifier, row in zip(("left", "right"), binding_ids, binding_rows):
        phase_id, milestone = f"{prefix}:bind-opening-{edge}", f"{identifier}:wrapped-and-finished"
        phase(phase_id, "binding-wrap", [identifier], [], [milestone], [row], [],
              [f"binding-wrap:{identifier}", "material-side-assignment", "continuous-path-verification"])
        completion(identifier, phase_id, [milestone])
    phase(attach_phase, "gather-inner-attachment", [gather], [f"{identifier}:complete" for identifier in binding_ids],
          [initial_milestone], [inner_row], binding_rows,
          ["material-side-assignment", "interfacing-treatment", "gather-distribution-and-stitch-joint", "continuous-path-verification"])
    stitched = []
    for identifier, row in zip(perimeter_ids, perimeter_rows):
        phase_id, milestone = f"{prefix}:stitch-{identifier.rsplit(':', 1)[1]}", f"{identifier}:stitched"
        phase(phase_id, "perimeter-stitch", [identifier], [initial_milestone], [milestone], [row],
              [*binding_rows, inner_row], ["material-side-assignment", "turn-compatible-stitch-joint", "continuous-path-verification"])
        stitched.append(milestone)
        completion(identifier, phase_id, [milestone, turned_milestone])
    held = [*binding_rows, inner_row, *perimeter_rows]
    phase(turn_phase, "turn-through-reserved-opening", perimeter_ids, [initial_milestone, *stitched],
          [turned_milestone], [], held,
          ["material-side-assignment", "perimeter-allowance-and-corner-treatment", "turning-passage-and-layer-observables", "source-grippers-and-work", "continuous-path-verification"], [opening_id])
    phase(allowance_phase, "turn-under-attachment-allowance", [gather], [turned_milestone],
          [allowance_milestone], [], held,
          ["attachment-allowance-fold", "source-grippers-and-work", "continuous-path-verification"], [opening_id])
    phase(close_phase, "close-outer-attachment", [gather],
          [allowance_milestone, *[f"{identifier}:complete" for identifier in perimeter_ids]],
          [f"{gather}:outer-attached"], [outer_row], held,
          ["material-side-assignment", "gather-distribution-and-stitch-joint", "continuous-path-verification"], [opening_id])
    completion(gather, attach_phase, [initial_milestone, turned_milestone, f"{gather}:outer-attached"])

    expected_selectors = {(operation["id"], member) for operation in sources
                          for member in range(1, len(operation["participants"]))}
    activated = [(row["registrationId"], row["memberIndex"]) for item in phases for row in item["activatesStarRows"]]
    if len(activated) != len(set(activated)) or set(activated) != expected_selectors:
        raise ValueError("Phase selectors must exactly partition every original source star member")
    gate_descriptions = {
        **{f"binding-wrap:{identifier}": "Source strip must wrap its own sleeve opening seam allowance, be stitched down and secure its apex without closing the opening; both opening edges remain distinct and unsewn to each other. Wrap, stitch-down and apex treatment are unresolved." for identifier in binding_ids},
        "material-side-assignment": "Declare each fabric right side relative to expanded rest winding and its mirror provenance; role names and geometric winding do not supply this assignment.",
        "interfacing-treatment": "Resolve the source interfacing role as an explicit layer or disclosed effective-material approximation.",
        "gather-distribution-and-stitch-joint": "Preserve unequal wrist/cuff rest arcs and all normalized-arc registration members; numerical gather and stitch coverage require validation.",
        "turn-compatible-stitch-joint": "A finite-offset normal frame is not automatically a freely rotating sewn joint during eversion.",
        "perimeter-allowance-and-corner-treatment": "Resolve allowance and corner treatment for the extension, end, outer and start perimeter before turning. Preserve the full source cut domain and rest metric; no undeclared grading, clipping or trimming.",
        "turning-passage-and-layer-observables": "Verify passage through the declared opening, material-side reversal, inward allowance placement and continued seam/contact constraints; hinge angle alone is insufficient.",
        "source-grippers-and-work": "Require source-derived weighted anchors, bounded compliant trajectories, force/reaction accounting and target/release work; the present solver lacks this control.",
        "attachment-allowance-fold": "Specify the shell attachment allowance fold from unchanged cut/stitch geometry; no undeclared trimming or rest reset.",
        "continuous-path-verification": "Saved physical motion must retain contact separation, material-frame nondegeneracy and declared seam constraints with independent replay.",
    }
    selected_instance_set = set(required_instances)
    source_instruction = next(operation for operation in pattern["drafting"]["assembly"] if operation["id"] == gather)
    plan = {
        "schemaVersion": 1, "profile": PROFILE, "kind": "declarative-cuff-construction-phases",
        "accepted": False, "solverReady": False, "side": side, "policy": policy,
        "policyBasis": "Explicit research selection; source wording about inner cuff and closing facing does not uniquely establish these layer roles.",
        "sourceBindings": {"canonicalPatternSha256": _digest(pattern), "inventorySha256": _digest(inventory),
            "assemblySha256": _digest(assembly), "inventoryPatternDigest": inventory["patternDigest"],
            "constructionDigest": inventory["constructionDigest"], "assemblyCompiler": assembly["compiler"],
            "inventoryPatternDigestScope": "Captured original byte digest; parsed pattern is independently bound by canonicalPatternSha256."},
        "sourceOperations": sources, "sourceInstruction": copy.deepcopy(source_instruction),
        "sourceInstructions": [copy.deepcopy(operation) for operation in pattern["drafting"]["assembly"] if operation["id"] in {*binding_ids, gather}],
        "requiredInstanceIds": required_instances,
        "sourceInstances": [copy.deepcopy(instance) for instance in inventory["instances"] if instance["id"] in selected_instance_set],
        "excludedSourceOperationIds": [operation["id"] for operation in assembly["operations"] if operation["id"] not in source_ids],
        "unitExternalOperationIds": [operation["id"] for operation in assembly["operations"] if operation["id"] not in source_ids
            and any(member["instanceId"] in selected_instance_set for member in operation["participants"])],
        "deferredClosures": [{"source": copy.deepcopy(closure), "execution": "deferred until after turning and full cuff attachment"}
            for closure in assembly["closures"] if any(member["instanceId"].startswith(cuff + ":") for member in closure["participants"])],
        "dependencyRefinements": [{"sourceOperationId": identifier, "sourceDependsOn": gather,
            "originalRequirement": "complete source operation before dependent operation",
            "phaseRequirement": initial_milestone, "fullSourceCompletionRemains": f"{gather}:complete",
            "reason": "Initial inner attachment precedes perimeter stitching; the outer attachment remains reserved until after turning."}
            for identifier in perimeter_ids],
        "phases": phases, "completionRules": completion_rules, "sourceOperationLifecycle": lifecycle,
        "constraintLifetime": "All fractions of a selected registration/member are activated together and held thereafter; pending selectors exert no force. No source seam is released for turning.",
        "starRowSelectors": [selector(registration, member) for registration, member in sorted(expected_selectors)],
        "openings": [{"id": opening_id, "classification": "reserved-unsewn-attachment",
            "sourceOperationId": gather, "sourceMemberIndex": 1, **copy.deepcopy(members[1]),
            "reservedDuringPhases": [item["id"] for item in phases if item["id"] != close_phase],
            "passagePhase": turn_phase, "closurePhase": close_phase, "closingStarRows": [outer_row],
            "counterpartSourceMembers": [copy.deepcopy(members[index]) for index in (0, 2)],
            "finalBoundaryDisposition": "All original source group members are attached; this is not a final free boundary."}],
        "materialSides": [{"instanceId": instance["id"], "sourceRole": instance["role"], "sourceMirrorX": instance["mirrorX"],
            "rightSideRelativeToExpandedRestNormal": "unresolved",
            "policyFinishedRole": "outer cuff" if instance["id"] == cuff + ":shell" else "inner cuff" if instance["id"] == cuff + ":facing" else "source role retained"}
            for instance in inventory["instances"] if instance["id"] in selected_instance_set],
        "unresolvedPhysicalRoles": [copy.deepcopy(role) for role in inventory["unresolvedPhysicalRoles"] if role["templateId"] == cuff],
        "unresolvedGates": [{"id": identifier, "status": "unresolved", "requirement": description}
            for identifier, description in gate_descriptions.items()],
        "counts": {"sourceOperations": len(sources), "fabricInstances": len(required_instances),
            "phases": len(phases), "starRowSelectors": len(expected_selectors)},
        "limitations": ["Declarative dependency refinement only; no phase has been executed or completed.",
            "Both sleeve-opening binding prerequisites are retained and unresolved.",
            "Sleeve-cap, underarm and other excluded garment operations remain outside this unit; their source obligations are retained.",
            "A sparse star registration is not a watertight continuous seam or proof of a usable turning opening.",
            "No material calibration, gripper implementation, turning passage, rest-shape change or garment acceptance is supplied."],
    }
    return json.loads(_encoded(plan))


def validate_cuff_phase_plan(pattern, inventory, assembly, plan, *, side, policy):
    """Strict rederivation; metadata match never means physical completion."""
    expected = build_cuff_phase_plan(pattern, inventory, assembly, side=side, policy=policy)
    if _encoded(plan) != _encoded(expected):
        raise ValueError("Cuff phase plan differs from strict source and policy rederivation")
    return {"accepted": False, "solverReady": False,
            "sourceCorrespondence": "matches rederived declarative phase plan", **expected["counts"]}
