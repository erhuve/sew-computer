import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class CuffCreaseCliTests(unittest.TestCase):
    def fixture(self):
        points = [[-.01, -.01], [-.01, .065], [.25, .065], [.25, -.01], [-.01, .0275]]
        points += [[coordinate, .065] for coordinate in (.042, .094, .146, .198)]
        points += [[.25, .0275]]
        points += [[coordinate, -.01] for coordinate in (.198, .146, .094, .042)]
        points += [[-.01 + .26 * index / 7, .0275] for index in range(1, 7)]
        faces = [[14,4,0],[4,14,1],[13,14,0],[18,8,7],[6,15,16],[13,15,14],
                 [15,13,12],[15,12,16],[17,18,7],[17,11,18],[15,5,14],[5,15,6],
                 [14,5,1],[8,19,2],[19,9,2],[18,19,8],[9,19,3],[19,10,3],
                 [11,10,18],[10,19,18],[6,17,7],[17,6,16],[11,17,16],[12,11,16]]
        points = np.asarray(points)
        samples = []
        for coordinate, arc in (([.24, .055], 0.), ([0., .055], 240.)):
            for triangle, face in enumerate(faces):
                weights = np.linalg.solve(np.vstack((points[face].T, np.ones(3))), [*coordinate, 1.])
                if min(weights) >= -1e-12:
                    samples.append({"restPosition": (np.asarray(coordinate) * 1000).tolist(),
                                    "triangle": triangle, "weights": weights.tolist(), "arcMm": arc})
                    break
        return {"restMeters": np.column_stack((points, np.zeros(20))).tolist(),
                "triangles": np.asarray(faces).ravel().tolist(), "instanceOffsets": {"cuff_left:shell": 0},
                "sourceTemplates": {"cuff_left": {"units": "mm", "sourceDomain": "draft.cutLine",
                    "restPositions": (points * 1000).tolist(), "triangles": faces,
                    "stitchPaths": [{"name": "outer", "lengthMm": 240., "samples": samples}]}}}

    def run_extract(self, source, directory):
        root = Path(directory)
        canonical = root / "source.json"
        canonical.write_text(json.dumps(source))
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("spike-cuff-fold-input.py")),
            "--canonical", str(canonical), "--output", str(root / "result"), "--crease", "outer-allowance"],
            capture_output=True, text=True, timeout=30)
        return result, root / "result"

    def test_allowance_extraction_and_false_source_correspondence(self):
        source = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            result, output = self.run_extract(source, directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            control = json.loads((output / "canonical.json").read_text())
            self.assertGreater(len(control["restMeters"]), 20)
            self.assertTrue(control["provenance"]["subdivision"]["extensionPolicy"])
            self.assertEqual(len(control["foldActuation"]["hinges"]), 12)
        for mutation in ("weights", "length", "mesh", "duplicate"):
            changed = copy.deepcopy(source)
            template = changed["sourceTemplates"]["cuff_left"]
            path = template["stitchPaths"][0]
            if mutation == "weights":
                path["samples"][0]["weights"] = [1., 0., 0.]
            elif mutation == "length":
                path["lengthMm"] += 1
            elif mutation == "mesh":
                template["restPositions"][0][0] += 1
            else:
                template["stitchPaths"].append(copy.deepcopy(path))
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                result, output = self.run_extract(changed, directory)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
