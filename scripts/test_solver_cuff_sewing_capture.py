"""Linux source-cuff capture/replay with all seams pending; no phase execution.

The five unchanged source fabrics receive only separate flat translations.
This tests source binding and inactive force semantics, not assembly placement,
textile-side orientation, binding, turning, or completed construction.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from solver_cuff_source_binding import ENGINE_FILES
from solver_sewing_input import PROFILE, derive_sewing_row_ids, sewing_source_identity


SCRIPTS = Path(__file__).resolve().parent
MISSING_DEPENDENCY = "services/engine/embedded_constraints.py"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@unittest.skipUnless(sys.platform.startswith("linux"), "Captured worker supervision requires Linux")
class CuffSewingCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        cls.environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        for name, arguments in (
                ("prepare-cuff-source.py", ["--output", root / "parent"]),
                ("prepare-cuff-construction.py", ["--source-canonical", root / "parent/canonical.json",
                    "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1",
                    "--output", root / "unit"])):
            result = subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, arguments)],
                                    env=cls.environment, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
        cls.unit = json.loads((root / "unit/unit.json").read_bytes())

    @staticmethod
    def pending_source(unit):
        """Retain the full source unit and add explicit diagnostic controls."""
        source = copy.deepcopy(unit)
        if source["side"] != "left" or any(item["mirrorX"] is not False for item in source["instances"]):
            raise ValueError("This translation-only fixture explicitly uses the unmirrored left source unit")
        rest = np.asarray(source["restMeters"])
        placed = rest.copy()
        translations = []
        for index, instance in enumerate(source["instances"]):
            start = source["instanceOffsets"][instance["id"]]
            count = len(source["sourceTemplates"][instance["templateId"]]["restPositions"])
            translation = [0., 0., index / 32.]
            placed[start:start + count] += translation
            translations.append({"instanceId": instance["id"], "translationMeters": translation,
                                 "sourceMirrorX": instance["mirrorX"]})
        source["placedMeters"] = placed.tolist()
        source["diagnosticPlacement"] = {
            "classification": "separated-flat source-local translations; no assembly placement",
            "accepted": False, "translations": translations,
            "sourceGeometryPolicy": "Canonical source rest coordinates, triangle winding and mirror metadata retained; no additional reflection or rest-shape change.",
            "materialSideOrientation": "unresolved; no textile right-side or construction orientation claim",
            "constructionStatus": "All 40 sewing rows remain pending; no source phase is executed or completed."}
        row_ids = list(derive_sewing_row_ids(source))
        source["sewingActuation"] = {"profile": PROFILE, "accepted": False,
            "sourceSha256": sewing_source_identity(source), "mode": "vector",
            "initialTargetsMeters": [[0., 0., 0.] for _ in row_ids],
            "finalTargetsMeters": [[0., 0., 0.] for _ in row_ids],
            "schedule": {"profile": "sewing-row-activation-v1", "rowIds": row_ids,
                "knots": [{"fraction": fraction, "activation": [0.] * len(row_ids)} for fraction in (0., 1.)]}}
        return source

    def run_control(self, root, *, remove_dependency=False):
        source = self.pending_source(self.unit)
        canonical, placement, output = root / "canonical.json", root / "placement.json", root / "run"
        canonical.write_bytes(encoded(source))
        placement.write_bytes(encoded({"placedMeters": source["placedMeters"],
            "canonicalDigest": hashlib.sha256(canonical.read_bytes()).hexdigest(),
            "intent": source["diagnosticPlacement"]}))
        arguments = ["--canonical", str(canonical), "--placement", str(placement), "--output", str(output),
            "--contact-model", "rest-filtered", "--ccd-profile", "temporal-separation-tight-inclusion",
            "--activation-distance-m", ".0001", "--minimum-distance-m", ".0001", "--pressure-pa", "10000",
            "--target-fraction", "1", "--subdivisions", "1", "--step-seconds", str(1 / 1024),
            "--sewing-mode", "vector", "--sewing-activation", "--max-attempts", "8", "--max-depth", "2",
            "--cpu-limit-seconds", "45", "--wall-limit-seconds", "60"]
        entry = SCRIPTS / "spike-contact-continuation.py"
        if remove_dependency:
            # Change only this temporary run after capture, before the worker.
            # Also remove its listed digest and republish progress, so the
            # namespace's mandatory-dependency check must fail independently of
            # a trivial listed-file hash failure. The live engine remains intact.
            entry = root / "missing-dependency-harness.py"
            entry.write_text(
                "import importlib.util,sys\n"
                f"sys.path.insert(0,{str(SCRIPTS)!r})\n"
                f"spec=importlib.util.spec_from_file_location('cuff_capture_cli',{str(SCRIPTS / 'spike-contact-continuation.py')!r})\n"
                "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
                "original=module.supervise\n"
                "def remove(command, output, report, **options):\n"
                f"    name={MISSING_DEPENDENCY!r}\n"
                "    path=output/'source-snapshot'/name\n"
                "    path.parent.chmod(0o700); path.unlink(); path.parent.chmod(0o500)\n"
                "    report['sourceDigests'].pop(name)\n"
                "    module.ProgressStore(output).save(report,'captured')\n"
                "    module.atomic_json(output/'report.json',report,replace=True)\n"
                "    return original(command,output,report,**options)\n"
                "module.supervise=remove\n"
                "raise SystemExit(module.main())\n")
        process = subprocess.run([sys.executable, str(entry), *arguments],
                                 env=self.environment, capture_output=True, text=True, timeout=75)
        report = json.loads((output / "report.json").read_bytes())
        return source, process, output, report

    def replay(self, output):
        return subprocess.run([sys.executable, str(SCRIPTS / "replay-rest-filtered-continuation.py"),
                               str(output), "--cpu-limit-seconds", "45"],
                              env=self.environment, capture_output=True, text=True, timeout=55)

    def test_full_pending_source_unit_captured_worker_and_independent_replay(self):
        from solver_sewing_replay import derive_sewing, verify_sewing_step

        with tempfile.TemporaryDirectory() as directory:
            source, process, output, report = self.run_control(Path(directory))
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr + str(report.get("failure")))
            self.assertTrue(report["completed"])
            self.assertFalse(report["accepted"])
            self.assertEqual((output / "canonical.json").read_bytes(), encoded(source))
            # Every original source field is unchanged by this control fixture.
            for key, value in self.unit.items():
                self.assertEqual(encoded(source[key]), encoded(value), key)
            rest = np.asarray(source["restMeters"])
            placed = np.asarray(source["placedMeters"])
            faces = np.asarray(source["triangles"]).reshape(-1, 3)
            np.testing.assert_array_equal(placed[faces[:, 1:]] - placed[faces[:, :1]],
                                          rest[faces[:, 1:]] - rest[faces[:, :1]])
            self.assertEqual(report["sewingBinding"]["sourceSha256"], source["sewingActuation"]["sourceSha256"])
            self.assertEqual(report["sewingBinding"]["sourceBundleSha256"],
                             hashlib.sha256(encoded(self.unit["embeddedConstraints"])).hexdigest())
            binding = report["sewingBinding"]["cuffSourceBinding"]
            self.assertEqual(binding["counts"]["fabricInstances"], 5)
            self.assertEqual(binding["counts"]["starRowSelectors"], 8)
            self.assertEqual(binding["counts"]["constraints"], 40)
            self.assertEqual(binding["phaseConstraintRows"], self.unit["phaseConstraintRows"])
            self.assertFalse(binding["accepted"])
            self.assertFalse(binding["solverReady"])
            self.assertIn("no executed phases", binding["scope"])
            for name in (*("services/engine/" + item for item in ENGINE_FILES), "spike-full-shirt.py"):
                content = (output / "source-snapshot" / name).read_bytes()
                self.assertEqual(report["sourceDigests"][name], hashlib.sha256(content).hexdigest())
            self.assertEqual(report["initialSewingEnergyJoules"], 0.)
            self.assertEqual(report["initialSewingActivation"], [0.] * 40)
            self.assertEqual(report["initialSewingRowTargetErrorsM"], [None] * 40)
            self.assertGreaterEqual(len(report["acceptedStateArtifacts"]), 1)
            independent = derive_sewing(source, 1, sewing_mode="vector")
            previous = placed
            for descriptor in report["acceptedStateArtifacts"]:
                state = json.loads((output / descriptor["path"]).read_bytes())
                record, positions = state["record"], np.asarray(state["positionsMeters"])
                step = record["step"]
                self.assertEqual(step["activeSewingRows"], [])
                self.assertEqual(step["pendingSewingRows"], list(range(40)))
                self.assertEqual(step["sewingActivation"], [0.] * 40)
                self.assertEqual(step["sewingJoules"], 0.)
                self.assertEqual(step["sewingRowTargetErrorsM"], [None] * 40)
                gradient, evidence = verify_sewing_step(independent, previous, positions,
                    record["startFraction"], record["endFraction"], np.zeros((40, 3)), np.zeros((40, 3)), step)
                np.testing.assert_array_equal(gradient, np.zeros_like(positions))
                for name in ("sewingBeforeJoules", "sewingAfterJoules", "sewingFixedParameterChangeJoules",
                             "sewingTargetParameterWorkJoules", "sewingActivationParameterWorkJoules",
                             "sewingParameterWorkJoules"):
                    self.assertEqual(evidence[name], 0., name)
                previous = positions
            replay = self.replay(output)
            self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
            proof_path = output / "verified-replay.json"
            proof = json.loads(proof_path.read_bytes())
            self.assertFalse(proof["accepted"])
            self.assertTrue(proof["completed"])
            self.assertTrue(proof["finalStateVerified"])
            self.assertTrue(proof["initialSewingVerification"]["verified"])
            self.assertEqual(len(proof["states"]), len(report["acceptedStateArtifacts"]))
            for state in proof["states"]:
                self.assertTrue(state["physicalPathPass"])
                self.assertTrue(state["pathCertificate"]["safe"])
                self.assertTrue(state["candidateCoverageVerification"]["verified"])
                self.assertEqual(state["trianglePathVerification"]["triangles"], len(faces))
                self.assertGreaterEqual(state["trianglePathVerification"]["verifiedLeaves"], len(faces))
                self.assertEqual(state["sewingVerification"]["activeRows"], [])
                self.assertEqual(state["sewingVerification"]["pendingRows"], list(range(40)))
                self.assertLessEqual(state["recomputedResidualN"], 1e-6)
            self.assertEqual(proof["verifiedSewingWorkSummary"]["sewingParameterWorkJoules"], 0.)
            proof_path.unlink()
            for label, mutation in (
                    ("false is not integer zero", lambda value: value["sewingBinding"]["cuffSourceBinding"].__setitem__("accepted", 0)),
                    ("member index is not true", lambda value: value["sewingBinding"]["cuffSourceBinding"]["rowGroups"][0].__setitem__("memberIndex", True))):
                with self.subTest(label=label):
                    altered = copy.deepcopy(report)
                    mutation(altered)
                    (output / "report.json").write_bytes(encoded(altered))
                    rejected = self.replay(output)
                    self.assertNotEqual(rejected.returncode, 0, rejected.stdout + rejected.stderr)
                    self.assertNotIn("FileExistsError", rejected.stderr)
                    self.assertFalse(proof_path.exists())
            (output / "report.json").write_bytes(encoded(report))
            # A repaired report cannot omit a required captured engine module:
            # the immutable journal still binds the original dependency list.
            # The separate worker attack below reaches the namespace check
            # using self-consistent pre-journal capture metadata.
            missing = output / "source-snapshot" / MISSING_DEPENDENCY
            missing.parent.chmod(0o700)
            missing.unlink()
            missing.parent.chmod(0o500)
            report["sourceDigests"].pop(MISSING_DEPENDENCY)
            (output / "report.json").write_bytes(encoded(report))
            rejected = self.replay(output)
            self.assertNotEqual(rejected.returncode, 0, rejected.stdout + rejected.stderr)
            self.assertIn("Journal changed captured provenance", rejected.stderr)
            self.assertFalse(proof_path.exists())

    def test_worker_missing_captured_engine_rejects_even_with_repaired_capture_metadata(self):
        live = SCRIPTS.parent / MISSING_DEPENDENCY
        before = hashlib.sha256(live.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            _, process, output, report = self.run_control(Path(directory), remove_dependency=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(report["completed"])
            self.assertFalse(report["accepted"])
            self.assertIn("Complete captured or repository engine source tree required", report["failure"]["message"])
            self.assertEqual(report.get("acceptedStateArtifacts", []), [])
            self.assertFalse((output / "verified-replay.json").exists())
        self.assertEqual(hashlib.sha256(live.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
