"""Independent refined-source admission tests; no cloth dynamics or private fixtures."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

from solver_sewing_input import bind_sewing_activation, derive_sewing_row_ids, sewing_source_identity
from solver_sewing_replay import derive_sewing, verify_initial, METRIC
from test_solver_sewing_input import fixture


SCRIPTS = Path(__file__).resolve().parent
PROFILE = "source-left-binding-refined-unit-v1"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def recipe(source, mode="distance", activation=None):
    source = copy.deepcopy(source)
    identities = list(derive_sewing_row_ids(source))
    weights = [0.] * len(identities) if activation is None else activation
    targets = [[0., 0., .001] for _ in identities] if mode == "vector" else [.001] * len(identities)
    source["sewingActuation"] = {"profile": "captured-sewing-activation-v1", "accepted": False,
        "sourceSha256": sewing_source_identity(source), "mode": mode,
        "initialTargetsMeters": targets, "finalTargetsMeters": copy.deepcopy(targets),
        "schedule": {"profile": "sewing-row-activation-v1", "rowIds": identities,
            "knots": [{"fraction": 0., "activation": weights.copy()},
                      {"fraction": 1., "activation": weights.copy()}]}}
    return source


def rehash(source):
    source["sewingActuation"]["sourceSha256"] = sewing_source_identity(source)
    return source


class RefinedSourceDowngradeTests(unittest.TestCase):
    def test_reserved_profiles_and_fields_never_use_generic_or_original_binding(self):
        base = fixture("distance")
        bind_sewing_activation(base, 4, sewing_mode="distance")
        derive_sewing(base, 4, sewing_mode="distance")
        claims = [{"profile": "source-left-binding-refined-unit-v2"},
                  {"profile": "source-left-binding-refinement-descriptor-v1"}]
        for key in ("baseUnit", "bindingRefinement", "bindingSeamRemap"):
            claims.extend(({key: {}, "profile": profile} for profile in
                           (None, "synthetic", "source-cuff-construction-unit-v1", False, 1)))
        for claim in claims:
            source = rehash(dict(copy.deepcopy(base), **claim))
            callback = Mock(side_effect=AssertionError("Downgrade must reject before external audit"))
            with self.subTest(claim=claim):
                with self.assertRaisesRegex(ValueError, "cannot downgrade"):
                    bind_sewing_activation(source, 4, sewing_mode="distance")
                with self.assertRaisesRegex(ValueError, "cannot downgrade"):
                    derive_sewing(source, 4, sewing_mode="distance", refined_source_verifier=callback)
                callback.assert_not_called()

    def test_refined_independent_derivation_requires_explicit_current_audit(self):
        source = fixture("distance")
        source.update(profile=PROFILE, baseUnit={}, bindingRefinement={}, bindingSeamRemap={})
        rehash(source)
        with self.assertRaisesRegex(ValueError, "independent refined-source"):
            derive_sewing(source, 4, sewing_mode="distance")
        invalid = (None, {}, {"verified": 1, "accepted": False, "sourceSha256": sewing_source_identity(source)},
                   {"verified": True, "accepted": 0, "sourceSha256": sewing_source_identity(source)},
                   {"verified": True, "accepted": False, "sourceSha256": "0" * 64})
        for evidence in invalid:
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, "evidence must bind"):
                derive_sewing(source, 4, sewing_mode="distance", refined_source_verifier=lambda _: evidence)
        digest = sewing_source_identity(source)

        def mutating_audit(value):
            value["sewingActuation"]["finalTargetsMeters"][0] = .5
            return {"verified": True, "accepted": False, "sourceSha256": digest}

        with self.assertRaisesRegex(ValueError, "changed its input"):
            derive_sewing(source, 4, sewing_mode="distance", refined_source_verifier=mutating_audit)

    def test_refined_normal_offset_is_rejected_before_frame_or_source_inference(self):
        source = fixture("normal-offset")
        source.update(profile=PROFILE, baseUnit={}, bindingRefinement={}, bindingSeamRemap={})
        rehash(source)
        callback = Mock(side_effect=AssertionError("Normal frames are unsupported"))
        with self.assertRaisesRegex(ValueError, "crease-side"):
            bind_sewing_activation(source, 4, sewing_mode="normal-offset")
        with self.assertRaisesRegex(ValueError, "crease-side"):
            derive_sewing(source, 4, sewing_mode="normal-offset", refined_source_verifier=callback)
        callback.assert_not_called()


class RefinedSourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from solver_binding_source import build_binding_source
        from solver_binding_remap_replay import verify_binding_remap

        cls.audit = staticmethod(verify_binding_remap)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        commands = [("prepare-cuff-source.py", ["--output", root / "parent"]),
                    ("prepare-cuff-construction.py", ["--source-canonical", root / "parent/canonical.json",
                     "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1",
                     "--output", root / "unit"])]
        for name, arguments in commands:
            completed = subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, arguments)],
                env=environment, cwd=root, capture_output=True, text=True, timeout=60)
            if completed.returncode:
                raise AssertionError(completed.stdout + completed.stderr)
        cls.base = json.loads((root / "unit/unit.json").read_bytes())
        before = encoded(cls.base)
        cls.source = build_binding_source(cls.base)
        if encoded(cls.base) != before:
            raise AssertionError("Refined source generation mutated immutable base")

    def bind_both(self, source, mode="distance"):
        controls, manifest = bind_sewing_activation(source, 4, sewing_mode=mode)
        record = derive_sewing(source, 4, sewing_mode=mode, refined_source_verifier=self.audit)
        self.assertEqual(controls.row_ids, record.row_ids)
        self.assertEqual([sorted(row.items()) for row in controls.rows], [list(row) for row in record.rows])
        return controls, manifest, record

    def test_vector_and_distance_bind_complete_derived_mesh_without_changing_base(self):
        original_ids = derive_sewing_row_ids(self.base)
        for mode in ("vector", "distance"):
            with self.subTest(mode=mode):
                source = recipe(self.source, mode)
                before = encoded(source)
                controls, manifest, record = self.bind_both(source, mode)
                self.assertEqual(encoded(source), before)
                self.assertEqual(encoded(source["baseUnit"]), encoded(self.base))
                self.assertEqual(controls.row_ids, original_ids)
                self.assertEqual(len(record.rows), 40)
                self.assertEqual(record.vertex_count, 245)
                self.assertEqual(len(record.faces), 398)
                self.assertTrue(record.cuff)
                self.assertNotIn("sourceTemplates", source)
                self.assertEqual(manifest["sourceBindingScope"],
                                 "rederived refined cuff source with recorded coefficient approximation")
                self.assertIs(manifest["cuffSourceBinding"]["accepted"], False)
                self.assertIs(manifest["sourceFrameMetadataUsed"], False)
                self.assertEqual(manifest["frameBindings"], [])
                self.assertEqual(source["instanceOffsets"], {"cuff_left:shell": 0, "cuff_left:facing": 20,
                    "opening_binding_left_left:shell": 40, "opening_binding_left_right:shell": 69,
                    "sleeve_left:shell": 80})
                self.assertEqual(manifest["sourceBundleSha256"],
                                 hashlib.sha256(encoded(source["embeddedConstraints"])).hexdigest())

    def test_all_pending_initial_report_retains_current_lineage_evidence(self):
        source = recipe(self.source)
        controls, manifest, record = self.bind_both(source)
        report = {"sewingActuation": source["sewingActuation"], "sewingBinding": manifest,
                  "initialSewingActivation": [0.] * 40,
                  "initialSewingTargetsMeters": controls.initial_targets.tolist(),
                  "initialSewingEnergyJoules": 0., "initialSewingRowTargetErrorsM": [None] * 40,
                  "initialSewingTargetErrorMetric": METRIC}
        evidence = verify_initial(record, source["restMeters"], report)
        self.assertTrue(evidence["verified"])
        self.assertEqual(evidence["initialEnergyJoules"], 0.)
        self.assertEqual(evidence["refinedSourceVerification"], self.audit(source))
        evidence["refinedSourceVerification"]["verified"] = False
        self.assertTrue(verify_initial(record, source["restMeters"], report)["refinedSourceVerification"]["verified"])

    def test_repaired_hash_does_not_hide_original_or_derived_geometry_changes(self):
        base = recipe(self.source)
        self.bind_both(base)
        attacks = [lambda s: s["baseUnit"]["restMeters"][0].__setitem__(0, .125),
                   lambda s: s["restMeters"][40].__setitem__(0, .125),
                   lambda s: s["instanceOffsets"].__setitem__("sleeve_left:shell", 62),
                   lambda s: s["triangles"].__setitem__(144, s["triangles"][145]),
                   lambda s: s["embeddedConstraints"]["constraints"][0].__setitem__("complianceMPerN", 2e-8),
                   lambda s: s["baseUnit"]["embeddedConstraints"]["constraints"].reverse(),
                   lambda s: s.__setitem__("bindingRefinement", {}),
                   lambda s: s.__setitem__("bindingSeamRemap", {})]
        for index, attack in enumerate(attacks):
            source = copy.deepcopy(base)
            attack(source)
            rehash(source)
            with self.subTest(attack=index):
                with self.assertRaises(ValueError):
                    bind_sewing_activation(source, 4, sewing_mode="distance")
                with self.assertRaises(ValueError):
                    derive_sewing(source, 4, sewing_mode="distance", refined_source_verifier=self.audit)

    def test_partial_selector_activation_is_rejected_even_with_valid_remap(self):
        self.bind_both(recipe(self.source, activation=[1.] * 5 + [0.] * 35))
        source = recipe(self.source, activation=[1.] + [0.] * 39)
        with self.assertRaisesRegex(ValueError, "share activation"):
            bind_sewing_activation(source, 4, sewing_mode="distance")
        with self.assertRaisesRegex(ValueError, "activate together"):
            derive_sewing(source, 4, sewing_mode="distance", refined_source_verifier=self.audit)

    def test_original_v1_and_generic_binding_scope_remain_distinct(self):
        source = recipe(self.base)
        _, original = bind_sewing_activation(source, 4, sewing_mode="distance")
        record = derive_sewing(source, 4, sewing_mode="distance")
        self.assertEqual(original["sourceBindingScope"], "rederived cuff construction source")
        self.assertEqual(record.refined_verification_json, "null")
        generic = fixture("distance")
        _, manifest = bind_sewing_activation(generic, 4, sewing_mode="distance")
        self.assertIsNone(manifest["cuffSourceBinding"])
        self.assertIn("no pattern-source proof", manifest["sourceBindingScope"])


if __name__ == "__main__":
    unittest.main()
