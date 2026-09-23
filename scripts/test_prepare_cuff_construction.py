"""Source fidelity and failure checks for the declarative five-instance cuff unit."""

from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
POLICY = "inner-facing-first-outer-shell-last-v1"


class PrepareCuffConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.environment = {
            "PATH": os.environ.get("PATH", ""), "HOME": str(cls.directory), "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        cls.run_cli("prepare-cuff-source.py", "--output", cls.directory / "parent")
        cls.parent_path = cls.directory / "parent/canonical.json"
        cls.parent = json.loads(cls.parent_path.read_bytes())
        cls.units = {}
        for side in ("left", "right"):
            cls.generate(cls.directory / side, side=side)
            cls.units[side] = json.loads((cls.directory / side / "unit.json").read_bytes())

    @classmethod
    def run_cli(cls, name, *arguments, success=True):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
                                cwd=cls.directory, env=cls.environment, capture_output=True, text=True, timeout=60)
        if success and result.returncode:
            raise AssertionError(f"{name} failed: {result.stdout}\n{result.stderr}")
        return result

    @classmethod
    def generate(cls, output, *, source=None, side="left", success=True):
        return cls.run_cli("prepare-cuff-construction.py", "--source-canonical", source or cls.parent_path,
                           "--side", side, "--attachment-policy", POLICY, "--output", output, success=success)

    def test_reproduction_source_snapshots_and_private_output(self):
        output = self.directory / "repeat"
        self.generate(output)
        content = (output / "unit.json").read_bytes()
        self.assertEqual(content, (self.directory / "left/unit.json").read_bytes())
        self.assertNotIn(str(ROOT), content.decode())
        self.assertNotIn(str(self.directory), content.decode())
        unit = self.units["left"]
        for name, digest in unit["provenance"]["sourceDigests"].items():
            self.assertEqual(hashlib.sha256((output / "source-snapshot" / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        self.assertEqual((output / "parent-canonical.json").read_bytes(), self.parent_path.read_bytes())
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((output / "unit.json").stat().st_mode), 0o400)
        self.assertFalse(unit["accepted"])
        self.assertFalse(unit["solverReady"])
        self.assertNotIn("placedMeters", unit)

    def test_five_physical_instances_preserve_full_cut_domains(self):
        for side, unit in self.units.items():
            with self.subTest(side=side):
                expected = {f"cuff_{side}:shell", f"cuff_{side}:facing", f"sleeve_{side}:shell",
                            f"opening_binding_{side}_left:shell", f"opening_binding_{side}_right:shell"}
                self.assertEqual(set(unit["instanceOffsets"]), expected)
                self.assertEqual(len(unit["instances"]), 5)
                self.assertTrue(all(item["mirrorX"] is (side == "right") for item in unit["instances"]))
                rest = np.asarray(unit["restMeters"])
                faces = np.asarray(unit["triangles"]).reshape(-1, 3)
                self.assertTrue(np.isfinite(rest).all())
                self.assertTrue(np.all(rest[:, 2] == 0))
                panels = {panel["id"]: panel for panel in unit["sourcePattern"]["panels"]}
                for instance in unit["instances"]:
                    mesh = unit["sourceTemplates"][instance["templateId"]]
                    offset = unit["instanceOffsets"][instance["id"]]
                    count = len(mesh["restPositions"])
                    local = rest[offset:offset + count]
                    np.testing.assert_array_equal(local[:, :2], np.asarray(mesh["restPositions"]) * .001)
                    selected = faces[np.all((faces >= offset) & (faces < offset + count), axis=1)] - offset
                    np.testing.assert_array_equal(selected, mesh["triangles"])
                    actual = unary_union([Polygon(local[face, :2] * 1000) for face in selected])
                    expected_shape = Polygon(panels[instance["templateId"]]["draft"]["cutLine"])
                    self.assertLess(actual.symmetric_difference(expected_shape).area, 1e-7)

    def test_source_lengths_and_star_partition_remain_unshortened(self):
        for side, unit in self.units.items():
            bundle = unit["embeddedConstraints"]
            counts = Counter((row["registrationId"], row["memberIndex"]) for row in bundle["constraints"])
            self.assertEqual(len(bundle["constraints"]), 40)
            self.assertEqual(len(counts), 8)
            self.assertEqual(set(counts.values()), {5})
            gathered = [item for item in bundle["registrations"] if item["id"] == f"cuff_gather_{side}"][0]
            self.assertEqual([item["endArcMm"] for item in gathered["members"]], [297., 220., 220.])
            self.assertEqual([item["pathName"] for item in gathered["members"]], ["wrist", "attachment", "attachment"])
            self.assertEqual([item["instanceId"] for item in gathered["members"]],
                             [f"sleeve_{side}:shell", f"cuff_{side}:shell", f"cuff_{side}:facing"])
            all_rows = [index for rows in unit["phaseConstraintRows"].values() for index in rows]
            self.assertEqual(sorted(all_rows), list(range(40)))
            phases = unit["phasePlan"]["phases"]
            gather_phases = [(index, phase) for index, phase in enumerate(phases)
                             if any(selector["registrationId"] == f"cuff_gather_{side}" for selector in phase["activatesStarRows"])]
            self.assertEqual(len(gather_phases), 2)
            self.assertEqual(gather_phases[0][1]["activatesStarRows"], [{"registrationId": f"cuff_gather_{side}", "memberIndex": 2}])
            self.assertEqual(gather_phases[1][1]["activatesStarRows"], [{"registrationId": f"cuff_gather_{side}", "memberIndex": 1}])
            self.assertGreater(gather_phases[1][0] - gather_phases[0][0], 4)
            # Evaluate every material anchor directly in unmodified local metres.
            for row in bundle["constraints"]:
                for sample in row["sourceSamples"]:
                    instance_offset = unit["instanceOffsets"][sample["instanceId"]]
                    position = sum(weight["weight"] * np.asarray(unit["restMeters"][instance_offset + weight["vertex"]])
                                   for weight in sample["weights"])
                    np.testing.assert_allclose(position[:2] * 1000, sample["restPositionMm"], rtol=0, atol=1e-10)

    def test_unexecuted_garment_obligations_and_interfacing_remain_visible(self):
        for side, unit in self.units.items():
            assembly = unit["sourceAssembly"]
            self.assertEqual(len(unit["selectedOperationIds"]), 7)
            self.assertEqual(set(unit["selectedOperationIds"]) | set(unit["excludedOperationIds"]),
                             {item["id"] for item in assembly["operations"]})
            self.assertFalse(set(unit["selectedOperationIds"]) & set(unit["excludedOperationIds"]))
            self.assertEqual(set(unit["unexecutedOtherOperationsTouchingUnit"]),
                             {f"sleeve_front_{side}", f"sleeve_back_{side}", f"underarm_{side}"})
            self.assertEqual(unit["unexecutedClosuresTouchingUnit"], [f"closure:cuff_{side}:0"])
            self.assertTrue(any(item["templateId"] == f"cuff_{side}" and item["role"] == "interfacing"
                                for item in unit["sourceInventory"]["unresolvedPhysicalRoles"]))

    def test_parent_numerical_positions_are_explicitly_unused(self):
        parent = copy.deepcopy(self.parent)
        parent["restMeters"] = [[99, 99, 99]]
        parent["placedMeters"] = [[-99, -99, -99]]
        source = self.directory / "unused-numerical.json"
        source.write_text(json.dumps(parent))
        output = self.directory / "regenerated"
        self.generate(output, source=source)
        actual = json.loads((output / "unit.json").read_bytes())
        expected = self.units["left"]
        self.assertNotEqual(actual["provenance"]["sourceInputSha256"], expected["provenance"]["sourceInputSha256"])
        actual["provenance"]["sourceInputSha256"] = expected["provenance"]["sourceInputSha256"]
        self.assertEqual(actual, expected)

    def test_invalid_parent_digest_code_and_recipe_reject_before_output(self):
        for name, change in (
            ("digest", lambda source: source["provenance"].update(patternSha256="0" * 64)),
            ("code", lambda source: source["provenance"]["sourceDigests"].update({"services/engine/assembly.py": "0" * 64})),
            ("construction", lambda source: source["sourceConstruction"].update(cuff="none")),
        ):
            with self.subTest(name=name):
                source = copy.deepcopy(self.parent)
                change(source)
                path, output = self.directory / (name + ".json"), self.directory / (name + "-output")
                path.write_text(json.dumps(source))
                result = self.generate(output, source=path, success=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())

    def test_parent_change_during_meshing_rejects_before_publication(self):
        spec = importlib.util.spec_from_file_location("cuff_construction_generator", ROOT / "scripts/prepare-cuff-construction.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        sys.path.insert(0, str(ROOT / "services/engine"))
        import cloth_domain

        source, output = self.directory / "changing-parent.json", self.directory / "changing-output"
        original_bytes = self.parent_path.read_bytes()
        source.write_bytes(original_bytes)
        original_mesh = cloth_domain.mesh_cloth_domain

        def changed_input(*args, **kwargs):
            mesh = original_mesh(*args, **kwargs)
            source.write_bytes(original_bytes + b"\n")
            return mesh

        with patch.object(cloth_domain, "mesh_cloth_domain", side_effect=changed_input):
            with self.assertRaisesRegex(ValueError, "parent input changed"):
                generator.generate(source, output, side="left", policy=POLICY)
        self.assertFalse(output.exists())

    def test_required_policy_bounded_mesh_strict_json_and_existing_output(self):
        output = self.directory / "bad-cli-output"
        args = ["--source-canonical", self.parent_path, "--side", "left", "--output", output]
        missing = self.run_cli("prepare-cuff-construction.py", *args, success=False)
        self.assertNotEqual(missing.returncode, 0)
        self.assertFalse(output.exists())
        for bad in ("nan", "0", "121"):
            result = self.run_cli("prepare-cuff-construction.py", *args, "--attachment-policy", POLICY,
                                  "--max-edge-mm", bad, success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
        duplicate = self.directory / "duplicate.json"
        duplicate.write_text('{"sourcePattern":{},"sourcePattern":{}}')
        self.assertNotEqual(self.generate(output, source=duplicate, success=False).returncode, 0)
        self.assertFalse(output.exists())
        existing = self.directory / "left"
        before = (existing / "unit.json").read_bytes()
        self.assertNotEqual(self.generate(existing, success=False).returncode, 0)
        self.assertEqual(before, (existing / "unit.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
