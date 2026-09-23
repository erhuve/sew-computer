"""Independent fresh synthetic source and repaired-hash cuff binding attacks."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from solver_cuff_source_binding import ENGINE_FILES, SOURCE_HELPER_FILES, validate_cuff_source_binding
from solver_cuff_construction import POLICY, build_cuff_phase_plan
from solver_engine_source_namespace import load_engine_modules


ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def repair_hashes(source):
    """Repair outer metadata without repairing the attacked source semantics."""
    provenance = source["provenance"]
    provenance["patternSha256"] = digest(source["sourcePattern"])
    provenance["constructionSha256"] = digest(source["sourceConstruction"])
    provenance["assemblySha256"] = digest(source["sourceAssembly"])
    provenance["phasePlanSha256"] = digest(source["phasePlan"])
    source["sewingActuation"] = {"sourceSha256": digest(
        {key: value for key, value in source.items() if key != "sewingActuation"})}


class CuffSourceBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        cls.cli("prepare-cuff-source.py", "--output", cls.directory / "parent")
        cls.units = {}
        for side in ("left", "right"):
            cls.cli("prepare-cuff-construction.py", "--source-canonical", cls.directory / "parent/canonical.json",
                    "--side", side, "--attachment-policy", POLICY, "--output", cls.directory / side)
            cls.units[side] = json.loads((cls.directory / side / "unit.json").read_bytes())

    @classmethod
    def cli(cls, name, *arguments):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
                                env=cls.environment, cwd=cls.directory, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    def reject(self, changed, pattern=None):
        repair_hashes(changed)
        with self.assertRaisesRegex(ValueError, pattern or ".+"):
            validate_cuff_source_binding(changed)

    def test_fresh_mirrored_source_has_all_rows_and_no_execution_claim(self):
        for side, source in self.units.items():
            with self.subTest(side=side):
                before = encoded(source)
                manifest = validate_cuff_source_binding(source)
                self.assertEqual(encoded(source), before)
                self.assertIs(manifest["accepted"], False)
                self.assertIs(manifest["solverReady"], False)
                self.assertIn("no executed phases", manifest["scope"])
                self.assertEqual(manifest["side"], side)
                expected = {(f"bind_opening_{side}_{edge}", 1) for edge in ("left", "right")}
                expected |= {(f"perimeter:cuff_{side}:{edge}", 1) for edge in ("extension", "end", "outer", "start")}
                expected |= {(f"cuff_gather_{side}", 1), (f"cuff_gather_{side}", 2)}
                groups = {(item["registrationId"], item["memberIndex"]): item["rowIndices"]
                          for item in manifest["rowGroups"]}
                self.assertEqual(set(groups), expected)
                self.assertEqual(sorted(index for group in groups.values() for index in group), list(range(40)))
                for indices in groups.values():
                    self.assertEqual([source["embeddedConstraints"]["constraints"][index]["fraction"]
                                      for index in indices], [0., .25, .5, .75, 1.])
                self.assertEqual(groups[(f"cuff_gather_{side}", 1)], [10, 12, 14, 16, 18])
                self.assertEqual(groups[(f"cuff_gather_{side}", 2)], [11, 13, 15, 17, 19])
                self.assertEqual(manifest["phaseConstraintRows"], source["phaseConstraintRows"])
                manifest["rowGroups"][0]["rowIndices"].clear()
                manifest["phaseConstraintRows"].clear()
                self.assertEqual(encoded(source), before)

    def test_original_corner_samples_remain_distinct_and_pending_shell_has_own_rows(self):
        source = self.units["left"]
        result = validate_cuff_source_binding(source)
        rows = source["embeddedConstraints"]["constraints"]
        # Source corners may share material coordinates, but belong to separate
        # complete registrations. They must not be deduplicated into one row.
        extension = [index for index, row in enumerate(rows) if row["registrationId"] == "perimeter:cuff_left:extension"]
        end = [index for index, row in enumerate(rows) if row["registrationId"] == "perimeter:cuff_left:end"]
        self.assertEqual(rows[extension[-1]]["sourceSamples"][0]["restPositionMm"],
                         rows[end[0]]["sourceSamples"][0]["restPositionMm"])
        self.assertNotEqual(extension[-1], end[0])
        shell_rows = {10, 12, 14, 16, 18}
        phases = list(result["phaseConstraintRows"].values())
        self.assertEqual(set(phases[-1]), shell_rows)
        self.assertFalse(shell_rows.intersection(index for phase in phases[:-1] for index in phase))

    def test_canonical_geometry_and_instance_maps_cannot_drift_from_templates(self):
        mutations = [
            lambda s: s["restMeters"][0].__setitem__(0, math.nextafter(s["restMeters"][0][0], math.inf)),
            lambda s: s["restMeters"][0].__setitem__(2, False),
            lambda s: s["triangles"].__setitem__(0, s["triangles"][1]),
            lambda s: s["instanceOffsets"].__setitem__(s["instances"][1]["id"], 0),
            lambda s: s["instances"].reverse(),
            lambda s: s["sourceTemplates"].pop("opening_binding_left_left"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                changed = copy.deepcopy(self.units["left"])
                mutate(changed)
                self.reject(changed)

    def test_repaired_source_panel_hash_does_not_hide_stale_template_geometry(self):
        changed = copy.deepcopy(self.units["left"])
        panel = next(item for item in changed["sourcePattern"]["panels"] if item["id"] == "cuff_left")
        panel["draft"]["cutLine"][1][0] += .1
        assembly_module, = load_engine_modules(__file__, "assembly")
        _, changed["sourceInventory"] = assembly_module.compile_inventory(encoded(changed["sourcePattern"]),
                                                                          changed["sourceConstruction"])
        changed["sourceAssembly"] = assembly_module.compile_assembly(changed["sourcePattern"], changed["sourceInventory"])
        changed["phasePlan"] = build_cuff_phase_plan(changed["sourcePattern"], changed["sourceInventory"],
                                                     changed["sourceAssembly"], side="left", policy=POLICY)
        changed["sourceTemplates"]["cuff_left"]["sourcePanelDigest"] = digest(panel)
        # Even rebuilding every inventory/plan hash and claiming the updated
        # source panel cannot turn the old cloth domain into that new cut line.
        self.reject(changed)

    def test_repaired_coefficient_sample_registration_and_compliance_attacks(self):
        mutations = [
            lambda b: b["constraints"][0]["terms"][0].__setitem__("coefficient", math.nextafter(b["constraints"][0]["terms"][0]["coefficient"], math.inf)),
            lambda b: b["constraints"][0]["sourceSamples"][0]["weights"][0].__setitem__("weight", .5),
            lambda b: b["constraints"][0].__setitem__("complianceMPerN", 2e-8),
            lambda b: b["registrations"][0].__setitem__("sampleCount", 4),
            lambda b: b["registrations"][0]["members"][0].__setitem__("direction", "reverse"),
            lambda b: b["constraints"][0].__setitem__("fraction", False),
            lambda b: b["constraints"].__setitem__(1, copy.deepcopy(b["constraints"][0])),
            lambda b: b["constraints"].pop(),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                changed = copy.deepcopy(self.units["left"])
                mutate(changed["embeddedConstraints"])
                self.reject(changed, "full embedded sewing bundle")

    def test_repaired_partition_gate_and_source_obligation_attacks(self):
        mutations = [
            lambda s: next(iter(s["phaseConstraintRows"].values())).pop(),
            lambda s: s["phaseConstraintRows"].__setitem__(s["phasePlan"]["phases"][0]["id"], [10, 12, 14, 16, 18]),
            lambda s: s["phasePlan"]["unresolvedGates"][0].__setitem__("status", "verified"),
            lambda s: s["phasePlan"]["openings"][0].__setitem__("sourceMemberIndex", 2),
            lambda s: s["phasePlan"]["unresolvedPhysicalRoles"].clear(),
            lambda s: s["unexecutedOtherOperationsTouchingUnit"].clear(),
            lambda s: s["sourceFreeBoundaries"].clear(),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                changed = copy.deepcopy(self.units["left"])
                mutate(changed)
                self.reject(changed)

    def test_profile_provenance_and_raw_budget_errors_fail_closed(self):
        mutations = [lambda s: s.__setitem__("profile", "generic"),
                     lambda s: s.__setitem__("accepted", True),
                     lambda s: s.__setitem__("side", "other"),
                     lambda s: s["provenance"].__setitem__("parentNumericalGeometryUsed", True),
                     lambda s: s["provenance"]["sourceDigests"].__setitem__("services/engine/assembly.py", "0" * 64),
                     lambda s: s["provenance"].__setitem__("maxEdgeMm", True),
                     lambda s: s.__setitem__("restMeters", [[0., 0., 0.]] * 25001),
                     lambda s: s.__setitem__("triangles", [0] * 150003),
                     lambda s: s["sourceConstruction"].__setitem__("sleeves", "none")]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                changed = copy.deepcopy(self.units["left"])
                mutate(changed)
                self.reject(changed)
        for value in (None, [], {}, {"profile": "source-cuff-construction-unit-v1"}):
            with self.assertRaises(ValueError):
                validate_cuff_source_binding(value)

    def test_control_extensions_do_not_replace_original_source_checks(self):
        changed = copy.deepcopy(self.units["left"])
        changed.update({"sewingFrames": {"callerMustValidate": True}, "placedMeters": [],
                        "gripperActuation": {"callerMustValidate": True}})
        repair_hashes(changed)
        result = validate_cuff_source_binding(changed)
        self.assertEqual(result, validate_cuff_source_binding(self.units["left"]))

    def snapshot(self, directory, *, shaped=False):
        scripts = directory / "scripts" if shaped else directory
        scripts.mkdir(parents=True)
        engine = directory / "services/engine"
        engine.mkdir(parents=True)
        for name in SOURCE_HELPER_FILES:
            shutil.copyfile(ROOT / "scripts" / name, scripts / name)
        for name in ENGINE_FILES:
            shutil.copyfile(ROOT / "services/engine" / name, engine / name)
        return scripts

    def snapshot_check(self, scripts, *, preload=False):
        code = ("import sys,json; "
                + (f"sys.path.insert(0,{str(ROOT / 'services/engine')!r}); import assembly; " if preload else "")
                + "sys.path.insert(0,sys.argv[1]); from solver_cuff_source_binding import validate_cuff_source_binding; "
                  "source=json.load(open(sys.argv[2])); print(json.dumps(validate_cuff_source_binding(source)))")
        return subprocess.run([sys.executable, "-I", "-c", code, str(scripts), str(self.directory / "left/unit.json")],
                              cwd=self.directory, env=self.environment, capture_output=True, text=True, timeout=30)

    def test_complete_flat_and_repository_shaped_snapshots_are_self_contained(self):
        expected = validate_cuff_source_binding(self.units["left"])
        for shaped in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                scripts = self.snapshot(Path(directory) / "snapshot", shaped=shaped)
                result = self.snapshot_check(scripts)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), expected)

    def test_incomplete_snapshot_does_not_fall_back_to_complete_parent_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            scripts = self.snapshot(Path(directory) / "snapshot", shaped=True)
            (scripts / "services").mkdir()  # Parent has a complete engine; adjacent one is incomplete.
            result = self.snapshot_check(scripts)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Complete captured or repository engine", result.stderr)

    def test_adjacent_linked_services_or_engine_never_falls_back(self):
        for entry in ("services", "engine"):
            for dangling in (False, True):
                with self.subTest(entry=entry, dangling=dangling), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "snapshot"
                    scripts = self.snapshot(root, shaped=True)
                    # A complete fallback remains available one level above.
                    # Both a live link to it and a dangling link must reject.
                    target = root / "absent" if dangling else root / "services"
                    link = scripts / "services"
                    if entry == "engine":
                        link.mkdir()
                        link /= "engine"
                        if not dangling:
                            target /= "engine"
                    link.symlink_to(target, target_is_directory=True)
                    result = self.snapshot_check(scripts)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("Linked engine source directories", result.stderr)

    def test_repository_shaped_linked_services_or_engine_is_rejected(self):
        for entry in ("services", "engine"):
            for dangling in (False, True):
                with self.subTest(entry=entry, dangling=dangling), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "snapshot"
                    scripts = self.snapshot(root, shaped=True)
                    link = root / "services"
                    if entry == "engine":
                        link /= "engine"
                    relocated = Path(directory) / "relocated"
                    link.rename(relocated)
                    target = Path(directory) / "absent" if dangling else relocated
                    link.symlink_to(target, target_is_directory=True)
                    result = self.snapshot_check(scripts)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("Linked engine source directories", result.stderr)

    def test_cached_engine_from_other_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            scripts = self.snapshot(Path(directory) / "snapshot")
            result = self.snapshot_check(scripts, preload=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Cached engine module", result.stderr)
            self.assertIn("different source root", result.stderr)

    def test_changed_snapshot_engine_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            scripts = self.snapshot(Path(directory) / "snapshot")
            with (scripts / "services/engine/assembly.py").open("a") as stream:
                stream.write("\n# changed captured dependency\n")
            result = self.snapshot_check(scripts)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("explicit migration required", result.stderr)


if __name__ == "__main__":
    unittest.main()
