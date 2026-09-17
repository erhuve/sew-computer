import unittest

import numpy as np

from solver_membrane_hessian import membrane_element_derivatives


class MembraneHessianTests(unittest.TestCase):
    def test_analytic_derivatives(self):
        positions = np.array([[[.2, -.1, .3], [.31, -.08, .34], [.18, .02, .28]]])
        poses = np.array([[[10, -2], [0, 8]]], dtype=float)
        areas, materials = np.array([.006]), np.array([[10000, 7000, 0]])
        energy, gradient, hessian = membrane_element_derivatives(positions, poses, areas, materials)
        difference = 1e-6
        numerical_gradient = np.zeros(9)
        numerical_hessian = np.zeros((9, 9))
        for coordinate in range(9):
            offset = np.zeros_like(positions)
            offset.ravel()[coordinate] = difference
            plus = membrane_element_derivatives(positions + offset, poses, areas, materials)
            minus = membrane_element_derivatives(positions - offset, poses, areas, materials)
            numerical_gradient[coordinate] = (plus[0][0] - minus[0][0]) / (2 * difference)
            numerical_hessian[:, coordinate] = (plus[1][0] - minus[1][0]) / (2 * difference)
        np.testing.assert_allclose(gradient[0], numerical_gradient, atol=2e-6, rtol=1e-7)
        np.testing.assert_allclose(hessian[0], numerical_hessian, atol=2e-4, rtol=1e-7)
        np.testing.assert_allclose(hessian, hessian.transpose(0, 2, 1), atol=1e-12)

    def test_rest_rigid_modes(self):
        positions = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
        poses, areas, materials = np.eye(2)[None], np.array([.5]), np.array([[10000, 10000, 0]])
        for project in (False, True):
            _, gradient, hessian = membrane_element_derivatives(positions, poses, areas, materials, project)
            np.testing.assert_allclose(gradient, 0, atol=1e-10)
            for axis in np.eye(3):
                translation = np.tile(axis, 3)
                rotation = np.cross(axis, positions[0]).ravel()
                np.testing.assert_allclose(hessian[0] @ translation, 0, atol=1e-10)
                np.testing.assert_allclose(hessian[0] @ rotation, 0, atol=1e-10)

    def test_projection_only_changes_search_metric(self):
        positions = np.array([[[0, 0, 0], [.6, 0, 0], [0, .7, .1]]])
        arguments = (positions, np.eye(2)[None], np.array([.5]), np.array([[10000, 10000, 0]]))
        raw = membrane_element_derivatives(*arguments)
        projected = membrane_element_derivatives(*arguments, project_psd=True)
        np.testing.assert_array_equal(raw[0], projected[0])
        np.testing.assert_array_equal(raw[1], projected[1])
        self.assertLess(np.linalg.eigvalsh(raw[2][0]).min(), -1)
        self.assertGreater(np.linalg.eigvalsh(projected[2][0]).min(), -1e-9)

    def test_degenerate_rejected(self):
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            membrane_element_derivatives(np.zeros((1, 3, 3)), np.eye(2)[None], np.array([.5]), np.array([[1, 1, 0]]))


if __name__ == "__main__":
    unittest.main()
