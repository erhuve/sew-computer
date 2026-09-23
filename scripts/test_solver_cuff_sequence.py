import copy
import json
import tempfile
import unittest

import numpy as np

from solver_cuff_sequence import combine_cuff_controls
import test_solver_cuff_crease_cli as cuff_crease_cli


class CuffSequenceTests(unittest.TestCase):
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
        cls.sewing = {"restMeters": np.concatenate((panel, panel)).tolist(),
            "placedMeters": np.concatenate((panel, panel + .002 * normal)).tolist(),
            "triangles": np.concatenate((faces, faces + 20)).ravel().tolist(),
            "instanceOffsets": {"cuff_left:shell": 0, "cuff_left:facing": 20},
            "embeddedConstraints": {"constraints": rows},
            "provenance": {"sourceSha256": cls.fold["provenance"]["sourceSha256"]}}

    def test_remapped_anchors_preserve_source_and_use_child_faces(self):
        sewing, fold = copy.deepcopy(self.sewing), copy.deepcopy(self.fold)
        result = combine_cuff_controls(sewing, fold)
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
