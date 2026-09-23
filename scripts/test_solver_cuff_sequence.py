import copy
import json
import tempfile
import unittest

import numpy as np

from solver_cuff_sequence import combine_cuff_controls
import test_solver_cuff_crease_cli as cuff_crease_cli


class CuffSequenceTests(unittest.TestCase):
    @staticmethod
    def set_anchor(row, coordinate, panel, faces):
        for triangle in faces:
            weights = np.linalg.solve(np.vstack((panel[triangle, :2].T, np.ones(3))), [*coordinate, 1.])
            if np.min(weights) >= -1e-12:
                weights[np.abs(weights) < 1e-12] = 0.
                weights /= weights.sum()
                break
        else:
            raise AssertionError("Test anchor outside source mesh")
        row["terms"] = []
        for sign, sample in zip((1, -1), row["sourceSamples"]):
            sample["weights"] = [{"vertex": int(vertex), "weight": float(weight)}
                                 for vertex, weight in zip(triangle, weights) if weight != 0.]
            sample["restPositionMm"] = (np.asarray(coordinate) * 1000).tolist()
            row["terms"].extend({"instanceId": sample["instanceId"], "vertex": weight["vertex"],
                                 "coefficient": sign * weight["weight"]} for weight in sample["weights"])

    @classmethod
    def setUpClass(cls):
        fixture = cuff_crease_cli.CuffCreaseCliTests()
        source = fixture.fixture()
        with tempfile.TemporaryDirectory() as directory:
            result, output = fixture.run_extract(source, directory)
            if result.returncode:
                raise RuntimeError(result.stderr)
            cls.fold = json.loads((output / "canonical.json").read_text())
        panel = np.asarray(source["restMeters"])
        faces = np.asarray(source["triangles"]).reshape((-1, 3))
        normal = np.cross(panel[faces[0, 1]] - panel[faces[0, 0]], panel[faces[0, 2]] - panel[faces[0, 0]])
        normal /= np.linalg.norm(normal)
        rows = []
        for index, face in enumerate(faces[:20]):
            samples, terms = [], []
            for sign, identity in ((1, "cuff_left:shell"), (-1, "cuff_left:facing")):
                samples.append({"instanceId": identity, "pathName": "synthetic", "arcMm": index,
                    "restPositionMm": (panel[face, :2].mean(axis=0) * 1000).tolist(),
                    "weights": [{"vertex": int(vertex), "weight": 1 / 3} for vertex in face]})
                terms.extend({"instanceId": identity, "vertex": int(vertex), "coefficient": sign / 3}
                             for vertex in face)
            rows.append({"registrationId": "synthetic", "fraction": index / 19, "memberIndex": 1,
                         "sourceSamples": samples, "terms": terms})
        cls.set_anchor(rows[0], [.12, .055], panel, faces)
        cls.sewing = {"restMeters": np.concatenate((panel, panel)).tolist(),
            "placedMeters": np.concatenate((panel, panel + .002 * normal)).tolist(),
            "triangles": np.concatenate((faces, faces + 20)).ravel().tolist(),
            "instanceOffsets": {"cuff_left:shell": 0, "cuff_left:facing": 20},
            "embeddedConstraints": {"constraints": rows},
            "provenance": {"sourceSha256": cls.fold["provenance"]["sourceSha256"]}}

    def test_remapped_anchors_preserve_source_and_use_child_faces(self):
        sewing, fold = copy.deepcopy(self.sewing), copy.deepcopy(self.fold)
        result = combine_cuff_controls(sewing, fold, crease_frame_region="body")
        self.assertEqual(sewing, self.sewing)
        self.assertEqual(fold, self.fold)
        rest = np.asarray(result["restMeters"])
        faces = np.asarray(result["triangles"]).reshape((-1, 3))
        np.testing.assert_array_equal(rest[:33], rest[33:])
        np.testing.assert_array_equal(rest[:20], np.asarray(sewing["restMeters"])[:20])
        self.assertEqual(len(result["foldActuation"]["hinges"]), 24)
        changed = 0
        for row in result["embeddedConstraints"]["constraints"]:
            for sample in row["sourceSamples"]:
                weights = sample["weights"]
                coordinate = sum(weight["weight"] * rest[weight["vertex"], :2] for weight in weights)
                np.testing.assert_allclose(coordinate * 1000, sample["restPositionMm"], rtol=0, atol=1e-9)
                child_faces = [faces[index] for index, parent in enumerate(result["provenance"]["subdivision"]["parentTriangles"])
                               if parent == sample["parentTriangle"]]
                self.assertTrue(any({weight["vertex"] for weight in weights}.issubset(set(face)) for face in child_faces))
                changed += weights != sample["originalWeights"]
        self.assertGreater(changed, 0)
        self.assertFalse(result["provenance"]["accepted"])

    def test_explicit_frame_region_preserves_every_other_control(self):
        body = combine_cuff_controls(self.sewing, self.fold, crease_frame_region="body")
        allowance = combine_cuff_controls(self.sewing, self.fold, crease_frame_region="allowance")
        for key in body.keys() - {"sewingFrames"}:
            self.assertEqual(body[key], allowance[key], key)
        self.assertNotEqual(body["sewingFrames"]["faces"][0], allowance["sewingFrames"]["faces"][0])
        panel = np.asarray(body["restMeters"])
        source_faces = np.asarray(self.sewing["triangles"]).reshape((-1, 3))[:24]
        actual_faces = {tuple(face) for face in np.asarray(body["triangles"]).reshape((-1, 3))}
        for region, control in (("body", body), ("allowance", allowance)):
            frames = control["sewingFrames"]
            self.assertEqual(frames["creaseFrameRegion"], region)
            for face, binding, row in zip(frames["faces"], frames["bindings"],
                                          control["embeddedConstraints"]["constraints"]):
                self.assertIn(tuple(face), actual_faces)
                self.assertEqual(binding["childVertices"], (np.asarray(face) - 33).tolist())
                negative = {term["vertex"] for term in row["terms"] if term["coefficient"] < 0}
                self.assertEqual(negative, set(binding["negativeAnchorVertices"]))
                self.assertLessEqual(negative, set(binding["childVertices"]))
                self.assertEqual(binding["sourceParentVertices"], sorted(source_faces[binding["sourceParentTriangle"]]))
                self.assertEqual(binding["region"], region if binding["onCrease"] else "body")
                if binding["onCrease"]:
                    signed = panel[face, 1] - .055
                    self.assertTrue(np.all(signed <= 1e-14) if region == "body" else np.all(signed >= -1e-14))

    def test_child_enumeration_cannot_select_frame_or_change_anchor_coefficients(self):
        reordered = copy.deepcopy(self.fold)
        subdivision = reordered["provenance"]["subdivision"]
        order = np.arange(len(subdivision["triangles"]))[::-1]
        for key in ("triangles", "parentTriangles"):
            subdivision[key] = [subdivision[key][index] for index in order]
        reordered["triangles"] = np.asarray(subdivision["triangles"]).ravel().tolist()
        for region in ("body", "allowance"):
            before = combine_cuff_controls(self.sewing, self.fold, crease_frame_region=region)
            after = combine_cuff_controls(self.sewing, reordered, crease_frame_region=region)
            for key in ("restMeters", "placedMeters", "embeddedConstraints", "foldActuation", "sewingFrames"):
                self.assertEqual(before[key], after[key], key)
        # Reordering parents independently is not an enumeration change.
        subdivision["parentTriangles"] = self.fold["provenance"]["subdivision"]["parentTriangles"]
        with self.assertRaisesRegex(ValueError, "source parents"):
            combine_cuff_controls(self.sewing, reordered, crease_frame_region="body")

    def test_no_choice_or_ambiguous_body_semantics_reject(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous crease frame"):
            combine_cuff_controls(self.sewing, self.fold)
        for region in ("nearest", "", False, 0):
            with self.subTest(region=region), self.assertRaises(ValueError):
                combine_cuff_controls(self.sewing, self.fold, crease_frame_region=region)
        panel = np.asarray(self.sewing["restMeters"])
        faces = np.asarray(self.sewing["triangles"]).reshape((-1, 3))[:24]
        for case in ("mixed", "all on crease", "false outer"):
            sewing = copy.deepcopy(self.sewing)
            rows = sewing["embeddedConstraints"]["constraints"]
            if case == "mixed":
                self.set_anchor(rows[1], [.12, .06], panel, faces)
            elif case == "all on crease":
                for row in rows:
                    self.set_anchor(row, [.12, .055], panel, faces)
            else:
                for sample in rows[1]["sourceSamples"]:
                    sample["pathName"] = "outer"
            with self.subTest(case=case), self.assertRaises(ValueError):
                combine_cuff_controls(sewing, self.fold, crease_frame_region="body")

    def test_unambiguous_body_anchors_do_not_require_a_crease_frame_choice(self):
        sewing = copy.deepcopy(self.sewing)
        self.set_anchor(sewing["embeddedConstraints"]["constraints"][0], [.12, .03],
                        np.asarray(sewing["restMeters"]), np.asarray(sewing["triangles"]).reshape((-1, 3))[:24])
        result = combine_cuff_controls(sewing, self.fold)
        self.assertIsNone(result["sewingFrames"]["creaseFrameRegion"])
        self.assertTrue(all(binding["region"] == "body" for binding in result["sewingFrames"]["bindings"]))

    def test_mismatched_source_and_false_correspondence_reject(self):
        for mutation in ("source", "parent", "rest", "terms", "coordinate", "weights", "placement"):
            sewing, fold = copy.deepcopy(self.sewing), copy.deepcopy(self.fold)
            sample = sewing["embeddedConstraints"]["constraints"][0]["sourceSamples"][0]
            if mutation == "source":
                sewing["provenance"]["sourceSha256"] = "different"
            elif mutation == "parent":
                fold["provenance"]["subdivision"]["parentTriangles"][0] += 1
            elif mutation == "rest":
                fold["restMeters"][25][0] += .001
            elif mutation == "terms":
                sewing["embeddedConstraints"]["constraints"][0]["terms"][0]["coefficient"] *= -1
            elif mutation == "coordinate":
                sample["restPositionMm"][0] += 1
            elif mutation == "weights":
                sample["weights"][0]["vertex"] = 99
            else:
                sewing["placedMeters"][25][0] += .001
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, AssertionError)):
                combine_cuff_controls(sewing, fold)


if __name__ == "__main__":
    unittest.main()
