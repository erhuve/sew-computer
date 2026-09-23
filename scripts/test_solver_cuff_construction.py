"""Independent source, phase dependency and tamper checks; no execution proof."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/engine"))
from assembly import compile_assembly, compile_inventory
from meshing_test import CONSTRUCTION, shirt_pattern
from solver_cuff_construction import POLICY, PROFILE, build_cuff_phase_plan, validate_cuff_phase_plan


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def selector_key(value):
    return value["registrationId"], value["memberIndex"]


def closure_of_milestones(milestones, rules):
    """Test-only logical closure, independent of phase builder; no physical claim."""
    result = set(milestones)
    while True:
        previous = set(result)
        result.update(rule["milestone"] for rule in rules if set(rule["allOfMilestones"]) <= previous)
        if result == previous:
            return result


class CuffConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pattern, cls.inventory = compile_inventory(encoded(shirt_pattern()), CONSTRUCTION)
        cls.assembly = compile_assembly(cls.pattern, cls.inventory)
        cls.plans = {side: cls.build(side=side) for side in ("left", "right")}

    @classmethod
    def build(cls, *, side="left", policy=POLICY, pattern=None, inventory=None, assembly=None):
        return build_cuff_phase_plan(cls.pattern if pattern is None else pattern,
                                    cls.inventory if inventory is None else inventory,
                                    cls.assembly if assembly is None else assembly, side=side, policy=policy)

    def validate(self, plan, *, side="left"):
        return validate_cuff_phase_plan(self.pattern, self.inventory, self.assembly, plan, side=side, policy=POLICY)

    def test_exact_source_is_retained_without_mutation_or_shared_references(self):
        before = encoded([self.pattern, self.inventory, self.assembly])
        plan = self.build()
        self.assertEqual(encoded([self.pattern, self.inventory, self.assembly]), before)
        self.assertEqual(plan["profile"], PROFILE)
        self.assertEqual(plan["counts"], {"sourceOperations": 7, "fabricInstances": 5, "phases": 10, "starRowSelectors": 8})
        self.assertIs(plan["accepted"], False)
        self.assertIs(plan["solverReady"], False)
        by_id = {item["id"]: item for item in self.assembly["operations"]}
        for operation in plan["sourceOperations"]:
            self.assertEqual(operation, by_id[operation["id"]])
        instructions = {item["id"]: item for item in self.pattern["drafting"]["assembly"]}
        self.assertEqual({item["id"] for item in plan["sourceInstructions"]},
                         {"cuff_gather_left", "bind_opening_left_left", "bind_opening_left_right"})
        for instruction in plan["sourceInstructions"]:
            self.assertEqual(instruction, instructions[instruction["id"]])
        self.assertEqual(plan["sourceBindings"]["canonicalPatternSha256"], hashlib.sha256(encoded(self.pattern)).hexdigest())
        self.assertEqual(plan["sourceBindings"]["assemblySha256"], hashlib.sha256(encoded(self.assembly)).hexdigest())
        result = self.validate(plan)
        self.assertIs(result["accepted"], False)
        self.assertIs(result["solverReady"], False)
        plan["sourceOperations"][0]["participants"][0]["instanceId"] = "changed"
        plan["sourceInstructions"][0]["id"] = "changed"
        self.assertEqual(encoded([self.pattern, self.inventory, self.assembly]), before)

    def test_mirrored_side_keeps_source_provenance_and_material_side_unresolved(self):
        for side, plan in self.plans.items():
            self.validate(plan, side=side)
            self.assertEqual(set(plan["requiredInstanceIds"]),
                             {f"sleeve_{side}:shell", f"cuff_{side}:shell", f"cuff_{side}:facing",
                              f"opening_binding_{side}_left:shell", f"opening_binding_{side}_right:shell"})
            self.assertTrue(all(item["sourceMirrorX"] is (side == "right") for item in plan["materialSides"]))
            self.assertTrue(all(item["rightSideRelativeToExpandedRestNormal"] == "unresolved" for item in plan["materialSides"]))
            roles = {item["instanceId"]: item["policyFinishedRole"] for item in plan["materialSides"]}
            self.assertEqual(roles[f"cuff_{side}:facing"], "inner cuff")
            self.assertEqual(roles[f"cuff_{side}:shell"], "outer cuff")
            self.assertIn("does not uniquely", plan["policyBasis"])
            self.assertEqual(plan["openings"][0]["instanceId"], f"cuff_{side}:shell")
        with self.assertRaises(ValueError):
            self.validate(self.plans["right"], side="left")

    def test_full_star_partition_retains_all_three_gather_members_and_rest_arcs(self):
        for side, plan in self.plans.items():
            expected = {(operation["id"], member) for operation in plan["sourceOperations"]
                        for member in range(1, len(operation["participants"]))}
            actual = Counter(selector_key(row) for phase in plan["phases"] for row in phase["activatesStarRows"])
            self.assertEqual(set(actual), expected)
            self.assertEqual(set(actual.values()), {1})
            self.assertEqual({selector_key(row) for row in plan["starRowSelectors"]}, expected)
            gather = next(item for item in plan["sourceOperations"] if item["id"] == f"cuff_gather_{side}")
            self.assertEqual([item["instanceId"] for item in gather["participants"]],
                             [f"sleeve_{side}:shell", f"cuff_{side}:shell", f"cuff_{side}:facing"])
            self.assertEqual([item["intervalMm"] for item in gather["participants"]], [[0., 297.], [0., 220.], [0., 220.]])
            self.assertIn("pending selectors exert no force", plan["constraintLifetime"])
            self.assertIn("No source seam is released", plan["constraintLifetime"])

    def test_dependency_refinement_changes_only_the_named_completion_requirement(self):
        for side, plan in self.plans.items():
            gather = f"cuff_gather_{side}"
            operations = {item["id"]: item for item in plan["sourceOperations"]}
            self.assertEqual(operations[gather]["dependsOn"], [f"bind_opening_{side}_left", f"bind_opening_{side}_right"])
            self.assertEqual(len(plan["dependencyRefinements"]), 4)
            for refinement in plan["dependencyRefinements"]:
                original = operations[refinement["sourceOperationId"]]
                self.assertEqual(original["dependsOn"], [gather])
                self.assertEqual(refinement["sourceDependsOn"], gather)
                self.assertEqual(refinement["phaseRequirement"], f"{gather}:inner-attached")
                self.assertEqual(refinement["fullSourceCompletionRemains"], f"{gather}:complete")

    def test_logical_schedule_keeps_opening_and_gather_incomplete_through_turning(self):
        for side, plan in self.plans.items():
            gather = f"cuff_gather_{side}"
            rules = plan["completionRules"]
            opening = plan["openings"][0]
            close_row = (gather, 1)
            self.assertEqual(opening["sourceMemberIndex"], 1)
            self.assertEqual(opening["edgeName"], "attachment")
            self.assertEqual(opening["intervalMm"], [0., 220.])
            self.assertEqual([selector_key(item) for item in opening["closingStarRows"]], [close_row])
            self.assertEqual(set(opening["reservedDuringPhases"]), {item["id"] for item in plan["phases"][:-1]})
            milestones, active = set(), set()
            ids = [item["id"] for item in plan["phases"]]
            self.assertEqual(len(ids), len(set(ids)))
            for phase in plan["phases"]:
                self.assertLessEqual(set(phase["requiresMilestones"]), milestones)
                self.assertLessEqual({selector_key(row) for row in phase["requiresHeldStarRows"]}, active)
                self.assertNotIn(f"{gather}:complete", milestones)
                if phase["id"] != opening["closurePhase"]:
                    self.assertNotIn(close_row, active)
                    self.assertNotIn(close_row, {selector_key(row) for row in phase["activatesStarRows"]})
                if phase["kind"] == "gather-inner-attachment":
                    self.assertEqual(phase["requiresMilestones"], [f"bind_opening_{side}_left:complete", f"bind_opening_{side}_right:complete"])
                if phase["id"] == opening["passagePhase"]:
                    self.assertNotIn(f"cuff:{side}:turned", milestones)
                    self.assertFalse(any(item.endswith(":complete") and item.startswith("perimeter:") for item in milestones))
                    self.assertEqual(len(active), 7)
                if phase["id"] == opening["closurePhase"]:
                    self.assertIn(f"cuff:{side}:turned", milestones)
                    self.assertIn(f"cuff:{side}:shell-attachment-allowance-turned-under", milestones)
                    self.assertTrue(all(f"perimeter:cuff_{side}:{edge}:complete" in milestones for edge in ("extension", "end", "outer", "start")))
                active.update(selector_key(row) for row in phase["activatesStarRows"])
                milestones = closure_of_milestones(milestones | set(phase["producesMilestones"]), rules)
            self.assertIn(f"{gather}:complete", milestones)
            self.assertEqual(len(active), 8)
            # This is a test of logical implications only. The real gates have
            # no witnesses and the returned plan does not claim these events.
            self.assertTrue(all(gate["status"] == "unresolved" for gate in plan["unresolvedGates"]))
            self.assertIs(plan["accepted"], False)

    def test_gates_cover_source_bindings_corner_treatment_and_missing_physics(self):
        plan = self.plans["left"]
        gates = {item["id"]: item for item in plan["unresolvedGates"]}
        referenced = {identifier for phase in plan["phases"] for identifier in phase["requiresGates"]}
        self.assertEqual(set(gates), referenced)
        self.assertTrue(all(item["status"] == "unresolved" for item in gates.values()))
        for identifier in ("bind_opening_left_left", "bind_opening_left_right"):
            text = gates[f"binding-wrap:{identifier}"]["requirement"]
            for phrase in ("stitched down", "apex", "without closing", "unresolved"):
                self.assertIn(phrase, text)
        turn = next(item for item in plan["phases"] if item["kind"] == "turn-through-reserved-opening")
        self.assertIn("perimeter-allowance-and-corner-treatment", turn["requiresGates"])
        self.assertIn("no undeclared grading, clipping or trimming", gates["perimeter-allowance-and-corner-treatment"]["requirement"])
        for identifier in ("source-grippers-and-work", "material-side-assignment", "interfacing-treatment",
                           "turning-passage-and-layer-observables", "turn-compatible-stitch-joint", "continuous-path-verification"):
            self.assertIn(identifier, gates)
        self.assertEqual(len(plan["unresolvedPhysicalRoles"]), 1)
        self.assertEqual(plan["unresolvedPhysicalRoles"][0]["role"], "interfacing")

    def test_whole_garment_obligations_and_unexecuted_closure_are_preserved(self):
        for side, plan in self.plans.items():
            selected = {item["id"] for item in plan["sourceOperations"]}
            excluded = set(plan["excludedSourceOperationIds"])
            self.assertEqual(selected | excluded, {item["id"] for item in self.assembly["operations"]})
            self.assertFalse(selected & excluded)
            self.assertEqual(set(plan["unitExternalOperationIds"]), {f"sleeve_front_{side}", f"sleeve_back_{side}", f"underarm_{side}"})
            self.assertEqual(len(plan["deferredClosures"]), 1)
            self.assertEqual(plan["deferredClosures"][0]["source"], next(item for item in self.assembly["closures"] if item["id"] == f"closure:cuff_{side}:0"))
            self.assertEqual(plan["deferredClosures"][0]["execution"], "deferred until after turning and full cuff attachment")

    def test_corrupted_source_memberships_dependencies_inventory_and_boundaries_reject(self):
        for case in ("drop facing", "member order", "short wrist", "dependency", "perimeter dependency", "remove binding",
                     "duplicate operation", "free boundary", "inventory role", "inventory mirror", "digest shape", "unknown field"):
            pattern, inventory, assembly = copy.deepcopy((self.pattern, self.inventory, self.assembly))
            gather = next(item for item in assembly["operations"] if item["id"] == "cuff_gather_left")
            if case == "drop facing":
                gather["participants"].pop()
            elif case == "member order":
                gather["participants"][1:] = gather["participants"][:0:-1]
            elif case == "short wrist":
                gather["participants"][0]["intervalMm"][1] = 220.
            elif case == "dependency":
                gather["dependsOn"] = []
            elif case == "perimeter dependency":
                next(item for item in assembly["operations"] if item["id"] == "perimeter:cuff_left:outer")["dependsOn"] = []
            elif case == "remove binding":
                assembly["operations"] = [item for item in assembly["operations"] if item["id"] != "bind_opening_left_right"]
            elif case == "duplicate operation":
                assembly["operations"].append(copy.deepcopy(gather))
            elif case == "free boundary":
                assembly["freeBoundaries"] = []
            elif case == "inventory role":
                next(item for item in inventory["instances"] if item["id"] == "cuff_left:facing")["role"] = "shell"
            elif case == "inventory mirror":
                next(item for item in inventory["instances"] if item["id"] == "cuff_left:shell")["mirrorX"] = True
            elif case == "digest shape":
                inventory["patternDigest"] = "trusted"
            else:
                assembly["physicallyComplete"] = True
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.build(pattern=pattern, inventory=inventory, assembly=assembly)

    def test_tampered_plan_including_premature_completion_or_closure_rejects(self):
        for case in ("drop selector", "duplicate selector", "duplicate phase", "early shell", "drop prerequisite", "early gather complete",
                     "early perimeter complete", "opening wrong member", "opening early closure", "open after closure", "gate proved",
                     "material side invented", "claim accepted", "claim solver ready", "unknown proof", "drop refinement", "source instruction"):
            plan = copy.deepcopy(self.plans["left"])
            if case == "drop selector":
                plan["phases"][-1]["activatesStarRows"] = []
            elif case == "duplicate selector":
                plan["phases"][2]["activatesStarRows"] *= 2
            elif case == "duplicate phase":
                plan["phases"][1]["id"] = plan["phases"][0]["id"]
            elif case == "early shell":
                plan["phases"][2]["activatesStarRows"] += plan["phases"][-1]["activatesStarRows"]
                plan["phases"][-1]["activatesStarRows"] = []
            elif case == "drop prerequisite":
                plan["phases"][2]["requiresMilestones"] = []
            elif case in ("early gather complete", "early perimeter complete"):
                key = "cuff_gather_left:complete" if case == "early gather complete" else "perimeter:cuff_left:outer:complete"
                next(item for item in plan["completionRules"] if item["milestone"] == key)["allOfMilestones"] = ["cuff_gather_left:inner-attached"]
            elif case == "opening wrong member":
                plan["openings"][0]["sourceMemberIndex"] = 2
            elif case == "opening early closure":
                plan["openings"][0]["closurePhase"] = plan["phases"][2]["id"]
            elif case == "open after closure":
                plan["openings"][0]["reservedDuringPhases"].append(plan["phases"][-1]["id"])
            elif case == "gate proved":
                plan["unresolvedGates"][0]["status"] = "verified"
            elif case == "material side invented":
                plan["materialSides"][0]["rightSideRelativeToExpandedRestNormal"] = 1
            elif case == "claim accepted":
                plan["accepted"] = True
            elif case == "claim solver ready":
                plan["solverReady"] = True
            elif case == "unknown proof":
                plan["proof"] = True
            elif case == "drop refinement":
                plan["dependencyRefinements"].pop()
            else:
                plan["sourceInstructions"][0]["id"] = "invented"
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.validate(plan)

    def test_explicit_policy_finite_json_and_bounded_payload_required(self):
        for policy in (None, "", "source-default", "outer-first", 1, True):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.build(policy=policy)
        for side in (None, "", "both", 0, True):
            with self.subTest(side=side), self.assertRaises(ValueError):
                self.build(side=side)
        for value in (None, [], {"bad": float("nan")}, {"bad": "x" * (4 * 1024 ** 2)}):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                build_cuff_phase_plan(value, self.inventory, self.assembly, side="left", policy=POLICY)
        plan = copy.deepcopy(self.plans["left"])
        plan["counts"]["phases"] = float("inf")
        with self.assertRaises(ValueError):
            self.validate(plan)


if __name__ == "__main__":
    unittest.main()
