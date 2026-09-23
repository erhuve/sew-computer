"""Fresh immutable-source reconstruction and captured namespace regressions."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from solver_binding_source import (PROFILE, REQUIRED_HELPER_FILES, REQUIRED_ENGINE_FILES,
    assert_binding_source_namespace, build_binding_source, is_refined_source, validate_binding_source)


SCRIPTS = Path(__file__).resolve().parent


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


class BindingSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        for script, arguments in (("prepare-cuff-source.py", ["--output", cls.root / "parent"]),
                ("prepare-cuff-construction.py", ["--source-canonical", cls.root / "parent/canonical.json",
                    "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1", "--output", cls.root / "unit"])):
            completed = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)],
                env=environment, cwd=cls.root, capture_output=True, text=True, timeout=60)
            if completed.returncode:
                raise AssertionError(completed.stdout + completed.stderr)
        cls.base = json.loads((cls.root / "unit/unit.json").read_bytes())
        cls.source = build_binding_source(cls.base)

    def test_complete_mesh_rebuild_preserves_all_original_local_vertices_and_other_meshes(self):
        source, base = self.source, self.base
        self.assertEqual(source["profile"], PROFILE)
        self.assertEqual(encoded(source["baseUnit"]), encoded(base))
        self.assertEqual(source["instanceOffsets"], {"cuff_left:shell": 0, "cuff_left:facing": 20,
            "opening_binding_left_left:shell": 40, "opening_binding_left_right:shell": 69, "sleeve_left:shell": 80})
        self.assertEqual(source["instanceTriangleOffsets"], {"cuff_left:shell": 0, "cuff_left:facing": 24,
            "opening_binding_left_left:shell": 48, "opening_binding_left_right:shell": 92, "sleeve_left:shell": 104})
        self.assertEqual((len(source["restMeters"]), len(source["triangles"]) // 3), (245, 398))
        faces = [source["triangles"][i:i + 3] for i in range(0, len(source["triangles"]), 3)]
        for instance in base["instances"]:
            name = instance["id"]
            old = base["sourceTemplates"][instance["templateId"]]
            count, before, after = len(old["restPositions"]), base["instanceOffsets"][name], source["instanceOffsets"][name]
            self.assertEqual(source["restMeters"][after:after + count], base["restMeters"][before:before + count])
            mesh = source["numericalMeshes"][name]
            face_start = source["instanceTriangleOffsets"][name]
            self.assertEqual(faces[face_start:face_start + len(mesh["triangles"])],
                             [[v + after for v in face] for face in mesh["triangles"]])
            if name != "opening_binding_left_left:shell":
                self.assertEqual(mesh["triangles"], old["triangles"])
        self.assertNotIn("sourceTemplates", source)
        self.assertIs(source["accepted"], False)
        self.assertIs(source["solverReady"], False)

    def test_only_five_negative_material_samples_change_and_all_row_identifiers_remain(self):
        source, base = self.source, self.base
        original = base["embeddedConstraints"]["constraints"]
        derived = source["embeddedConstraints"]["constraints"]
        self.assertEqual(len(derived), 40)
        name = "opening_binding_left_left:shell"
        mapped = {row["rowIndex"]: row for row in source["bindingSeamRemap"]["rows"]}
        self.assertEqual(set(mapped), set(range(5)))
        for index, (old, new, correspondence) in enumerate(zip(original, derived, source["sourceRowCorrespondence"])):
            self.assertEqual(correspondence["originalRowSha256"], digest(old))
            self.assertEqual(correspondence["numericalRowSha256"], digest(new))
            self.assertEqual(correspondence["bindingAnchorRemapped"], index < 5)
            for key in ("registrationId", "memberIndex", "fraction", "complianceMPerN"):
                self.assertEqual(encoded(new[key]), encoded(old[key]))
            if index >= 5:
                self.assertEqual(encoded(new), encoded(old))
            else:
                self.assertEqual([term for term in old["terms"] if term["instanceId"] != name],
                                 [term for term in new["terms"] if term["instanceId"] != name])
                self.assertEqual(new["sourceSamples"][0], old["sourceSamples"][0])
                self.assertEqual(new["sourceSamples"][1]["weights"], mapped[index]["selectedNumericalWeights"])
                self.assertEqual({item["vertex"]: item["coefficient"] for item in new["terms"] if item["instanceId"] == name},
                                 {item["vertex"]: -item["weight"] for item in mapped[index]["selectedNumericalWeights"]})
        self.assertEqual(source["embeddedConstraints"]["registrations"], base["embeddedConstraints"]["registrations"])
        self.assertEqual(source["embeddedConstraints"]["originalBundleSha256"], digest(base["embeddedConstraints"]))
        self.assertEqual(source["embeddedConstraints"]["sourceIdentities"][name]["vertexCount"], 29)

    def test_strict_validator_rederives_core_and_does_not_mutate_source(self):
        before = encoded(self.base)
        source = build_binding_source(self.base)
        report = validate_binding_source(source)
        self.assertEqual(encoded(self.base), before)
        self.assertEqual(report["counts"]["vertices"], 245)
        self.assertEqual(report["counts"]["unchangedLocalRows"], 35)
        self.assertEqual(report["counts"]["constraints"], 40)
        self.assertEqual(len(report["rowGroups"]), 8)
        self.assertEqual(report["sourceUnitSha256"], digest(self.base))
        self.assertEqual(report["numericalSourceCoreSha256"], digest(source))
        self.assertEqual(report["phaseConstraintRows"], self.base["phaseConstraintRows"])
        report["rowGroups"].clear()
        source["baseUnit"]["embeddedConstraints"]["constraints"].clear()
        self.assertEqual(encoded(self.base), before)
        self.assertEqual(len(self.source["baseUnit"]["embeddedConstraints"]["constraints"]), 40)

    def test_repaired_derived_hashes_cannot_authorize_geometry_rows_or_lineage_mutations(self):
        def rest(data):
            data["restMeters"][40][0] += 1e-6
        def offset(data):
            data["instanceOffsets"]["sleeve_left:shell"] = 62
        def face(data):
            data["triangles"][144:147] = data["triangles"][144:147][::-1]
        def row(data):
            value = data["embeddedConstraints"]["constraints"][0]
            value["terms"][0]["coefficient"] *= .99
            data["sourceRowCorrespondence"][0]["numericalRowSha256"] = digest(value)
        def base(data):
            data["baseUnit"]["embeddedConstraints"]["constraints"][0]["terms"][0]["coefficient"] *= .99
        def lineage(data):
            data["bindingRefinement"]["finalLocalMesh"]["verticesMeters"][11][0] += 1e-9
        def provenance(data):
            data["derivationCodeDigests"]["solver_binding_source.py"] = "0" * 64
        def stale_source(data):
            data["sourceTemplates"] = copy.deepcopy(data["baseUnit"]["sourceTemplates"])
        def undeclared_frame(data):
            data["sewingFrames"] = {"faces": []}
        for mutation in (rest, offset, face, row, base, lineage, provenance, stale_source, undeclared_frame):
            with self.subTest(mutation=mutation.__name__):
                changed = copy.deepcopy(self.source)
                mutation(changed)
                with self.assertRaises(ValueError):
                    validate_binding_source(changed)

    def test_reserved_namespace_and_raw_json_downgrades_reject(self):
        for profile in (PROFILE, "source-left-binding-refined-unit-v0", "source-left-binding-unknown"):
            self.assertTrue(is_refined_source({"profile": profile}))
        for key in ("baseUnit", "bindingRefinement", "bindingSeamRemap"):
            self.assertTrue(is_refined_source({"profile": "generic", key: None}))
        self.assertFalse(is_refined_source(self.base))
        for mutation in (lambda x: x.update(profile="generic"), lambda x: x.update(profile=True),
                         lambda x: x.update(accepted=0), lambda x: x.update(instanceOffsets={}),
                         lambda x: x["instanceOffsets"].update({"cuff_left:shell": False}),
                         lambda x: x.update(restMeters=tuple(x["restMeters"]))):
            changed = copy.deepcopy(self.source)
            mutation(changed)
            with self.assertRaises(ValueError):
                validate_binding_source(changed)
        with self.assertRaises(ValueError):
            build_binding_source(self.source)

    def test_placement_and_controls_are_separate_from_source_core_validation(self):
        original = validate_binding_source(self.source)
        changed = copy.deepcopy(self.source)
        changed["placedMeters"] = [[0, 0, 0]]
        changed["sewingActuation"] = {"intentionallyInvalidControl": True}
        # This validator deliberately establishes only the source core. The
        # dedicated input binder must separately reject these malformed controls.
        self.assertEqual(validate_binding_source(changed), original)
        changed["unrecognizedControl"] = {}
        with self.assertRaises(ValueError):
            validate_binding_source(changed)

    def test_complete_captured_helper_closure_and_manifest_are_required_before_import(self):
        manifest = assert_binding_source_namespace()
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "services/engine").mkdir(parents=True)
        for name in REQUIRED_HELPER_FILES:
            shutil.copyfile(SCRIPTS / name, snapshot / name)
        for name in REQUIRED_ENGINE_FILES:
            shutil.copyfile(SCRIPTS.parent / "services/engine" / name, snapshot / "services/engine" / name)
        manifest_path = self.root / "snapshot-manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        program = ("import json,sys; from pathlib import Path; "
            "sys.path.insert(0,sys.argv[1]); sys.path.append(sys.argv[3]); "
            "from solver_binding_source import assert_binding_source_namespace; "
            "assert_binding_source_namespace(Path(sys.argv[1]), source_digests=json.load(open(sys.argv[2])))")
        def run():
            return subprocess.run([sys.executable, "-B", "-c", program, str(snapshot), str(manifest_path), str(SCRIPTS)],
                cwd=self.root, capture_output=True, text=True, timeout=30)
        completed = run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for name in REQUIRED_HELPER_FILES:
            with self.subTest(missing=name):
                path = snapshot / name
                saved = path.read_bytes()
                path.unlink()
                changed = dict(manifest)
                changed.pop(name)
                manifest_path.write_text(json.dumps(changed))
                self.assertNotEqual(run().returncode, 0)
                path.write_bytes(saved)
                manifest_path.write_text(json.dumps(manifest))
        changed = dict(manifest)
        changed.pop("solver_binding_remap.py")
        manifest_path.write_text(json.dumps(changed))
        self.assertNotEqual(run().returncode, 0)

    def test_cached_helper_and_wrong_root_cannot_mix_with_source_namespace(self):
        with self.assertRaises(ValueError):
            assert_binding_source_namespace(self.root)
        program = ("import sys,types; sys.path.insert(0,sys.argv[1]); "
            "from solver_binding_source import assert_binding_source_namespace; "
            "fake=types.ModuleType('solver_crease_mesh'); fake.__file__='/wrong/solver_crease_mesh.py'; "
            "sys.modules['solver_crease_mesh']=fake; assert_binding_source_namespace()")
        completed = subprocess.run([sys.executable, "-B", "-c", program, str(SCRIPTS)],
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("another namespace", completed.stderr)


if __name__ == "__main__":
    unittest.main()
