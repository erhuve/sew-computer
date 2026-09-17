import unittest

import numpy as np

from solver_strain_diagnostics import edge_strain_report, mass_motion_report


class StrainDiagnosticsTests(unittest.TestCase):
    def test_closed_seam_does_not_hide_free_system_drift(self):
        initial = [[0, 0, 0], [1, 0, 0]]
        report = mass_motion_report(initial, [[1, 0, 0], [1, 0, 0]], [[10, 0, 0], [0, 0, 0]], [1, 1])
        self.assertEqual(report["centerOfMassDisplacementMm"], 500)
        self.assertEqual(report["centerOfMassSpeedMetersPerSecond"], 5)
        balanced = mass_motion_report(initial, [[.5, 0, 0], [.5, 0, 0]], [[5, 0, 0], [-5, 0, 0]], [1, 1])
        self.assertEqual(balanced["centerOfMassDisplacementMm"], 0)
        self.assertEqual(balanced["centerOfMassSpeedMetersPerSecond"], 0)
        pinned = mass_motion_report(initial, initial, [[0, 0, 0], [0, 0, 0]], [0, 1])
        self.assertEqual(pinned["pinnedVertexCount"], 1)
        with self.assertRaises(ValueError):
            mass_motion_report(initial, initial, initial, [0, 0])

    def test_unique_edges_source_identity_and_pin_localization(self):
        rest = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        positions = rest * [2, 1, 1]
        report = edge_strain_report(rest, positions, [[0, 1, 2], [0, 2, 3]], ["panel"] * 4, [0])
        self.assertEqual(report["edgeCount"], 5)
        self.assertEqual(report["min"], 1)
        self.assertEqual(report["max"], 2)
        self.assertEqual(report["worstEdges"][0]["vertices"], [0, 1])
        self.assertTrue(report["worstEdges"][0]["touchesPin"])
        self.assertEqual(report["worstEdges"][0]["instanceId"], "panel")
        np.testing.assert_array_equal(rest, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    def test_rigid_motion_and_multiple_instances(self):
        triangle = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        rest = np.concatenate([triangle, triangle])
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        moved = np.concatenate([triangle @ rotation + 7, triangle * 3 - 8])
        report = edge_strain_report(rest, moved, [[0, 1, 2], [3, 4, 5]], ["first"] * 3 + ["second"] * 3)
        self.assertEqual(report["perInstance"]["first"]["max"], 1)
        self.assertAlmostEqual(report["perInstance"]["second"]["max"], 3)

    def test_invalid_geometry_rejects(self):
        rest = [[0., 0, 0], [1, 0, 0], [0, 1, 0]]
        for faces, identities, pins in (([[0, 0, 2]], ["a"] * 3, []), ([[0, 1, 2]], ["a", "b", "a"], []),
                                        ([[0, 1, 3]], ["a"] * 3, []), ([[0, 1, 2]], ["a"] * 3, [True])):
            with self.subTest(faces=faces, identities=identities, pins=pins), self.assertRaises(ValueError):
                edge_strain_report(rest, rest, faces, identities, pins)
        with self.assertRaises(ValueError):
            edge_strain_report(rest, np.asarray(rest) * np.nan, [[0, 1, 2]], ["a"] * 3)


if __name__ == "__main__":
    unittest.main()
