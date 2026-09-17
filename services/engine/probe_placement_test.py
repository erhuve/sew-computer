import copy
import unittest

import numpy as np

from probe_placement import opposed_path_frame, transform_probe_positions


def straight_mesh(length=100, origin=(20, 0), direction=(0, 1)):
    samples = [{"restPosition": [origin[axis] + direction[axis] * length * fraction for axis in range(2)], "arcMm": length * fraction} for fraction in (0, 0.2, 0.7, 1)]
    return {"units": "mm", "stitchPaths": [{"name": "right", "lengthMm": length, "samples": samples}]}


class ProbePlacementTests(unittest.TestCase):
    def test_opposed_equal_panels_preserve_winding_and_distances(self):
        mesh = straight_mesh()
        snapshot = copy.deepcopy(mesh)
        frame = opposed_path_frame(mesh, mesh)
        np.testing.assert_allclose(frame["rotation"], np.diag([-1, 1, -1]), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(frame["rotation"]), 1)
        panel = np.array([[0, 0, 0], [0.02, 0, 0], [0.02, 0.1, 0], [0, 0.1, 0]])
        original = panel.copy()
        transformed = transform_probe_positions(panel, frame)
        np.testing.assert_allclose(transformed[1:3], [[0.02, 0, 0.005], [0.02, 0.1, 0.005]], atol=1e-14)
        self.assertLess(panel[:, 0].mean(), 0.02)
        self.assertGreater(transformed[:, 0].mean(), 0.02)
        for first in range(4):
            for second in range(4):
                self.assertAlmostEqual(np.linalg.norm(panel[first] - panel[second]), np.linalg.norm(transformed[first] - transformed[second]))
        normal = np.cross(panel[1] - panel[0], panel[2] - panel[0])
        transformed_normal = np.cross(transformed[1] - transformed[0], transformed[2] - transformed[0])
        np.testing.assert_allclose(transformed_normal, np.asarray(frame["rotation"]) @ normal, atol=1e-14)
        np.testing.assert_array_equal(panel, original)
        self.assertEqual(mesh, snapshot)

    def test_unequal_lengths_center_without_stretching(self):
        frame = opposed_path_frame(straight_mesh(), straight_mesh(200))
        second = np.array([[0.02, 0, 0], [0.02, 0.2, 0]])
        placed = transform_probe_positions(second, frame)
        np.testing.assert_allclose(placed.mean(axis=0), [0.02, 0.05, 0.005], atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(placed[1] - placed[0]), 0.2)

    def test_arbitrary_straight_tangents_align_with_proper_rotation(self):
        first = straight_mesh(origin=(12, -33), direction=(0.6, 0.8))
        second = straight_mesh(200, origin=(-44, 71), direction=(-0.8, 0.6))
        frame = opposed_path_frame(first, second)
        endpoints = np.asarray([[*sample["restPosition"], 0] for sample in second["stitchPaths"][0]["samples"]])[::3] / 1000
        placed = transform_probe_positions(endpoints, frame)
        np.testing.assert_allclose((placed[1] - placed[0]) / 0.2, [0.6, 0.8, 0], atol=1e-14)
        np.testing.assert_allclose(placed.mean(axis=0), [0.042, 0.007, 0.005], atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(frame["rotation"]), 1)

    def test_translation_covariance(self):
        first, second = straight_mesh(), straight_mesh(200)
        frame = opposed_path_frame(first, second)
        shift = np.array([0.071, -0.093, 0])
        for mesh in (first, second):
            for sample in mesh["stitchPaths"][0]["samples"]:
                sample["restPosition"] = (np.asarray(sample["restPosition"]) + shift[:2] * 1000).tolist()
        shifted = opposed_path_frame(first, second)
        positions = np.array([[0, 0, 0], [0.02, 0.2, 0]])
        np.testing.assert_allclose(transform_probe_positions(positions + shift, shifted), transform_probe_positions(positions, frame) + shift, atol=1e-14)

    def test_malformed_curved_and_zero_paths_fail(self):
        for mutation in ("curve", "zero", "nan", "reverse_arc", "missing", "duplicate", "length"):
            mesh = straight_mesh()
            path = mesh["stitchPaths"][0]
            if mutation == "curve":
                path["samples"][1]["restPosition"][0] += 1
            elif mutation == "zero":
                path["samples"][-1]["restPosition"] = path["samples"][0]["restPosition"][:]
            elif mutation == "nan":
                path["samples"][1]["restPosition"][0] = float("nan")
            elif mutation == "reverse_arc":
                path["samples"][2]["arcMm"] = 1
            elif mutation == "missing":
                path["samples"].pop(0)
            elif mutation == "duplicate":
                mesh["stitchPaths"].append(copy.deepcopy(path))
            else:
                path["lengthMm"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                opposed_path_frame(straight_mesh(), mesh)
        for gap in (-1, True, float("nan"), 0.2):
            with self.assertRaises(ValueError):
                opposed_path_frame(straight_mesh(), straight_mesh(), gap_m=gap)

    def test_reflection_and_scaling_transforms_fail(self):
        for diagonal in ([-1, 1, 1], [2, 1, 1]):
            frame = opposed_path_frame(straight_mesh(), straight_mesh())
            frame["rotation"] = np.diag(diagonal).tolist()
            with self.assertRaisesRegex(ValueError, "proper rigid rotation"):
                transform_probe_positions([[0, 0, 0]], frame)


if __name__ == "__main__":
    unittest.main()
