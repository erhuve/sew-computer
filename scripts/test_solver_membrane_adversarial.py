import unittest

import numpy as np

from solver_membrane_hessian import membrane_element_derivatives


class MembraneAdversarialTests(unittest.TestCase):
    def test_randomized_derivatives_and_rigid_covariance(self):
        generator = np.random.default_rng(9231)
        for _ in range(20):
            positions = generator.normal(size=(1, 3, 3)) * .1
            poses = generator.normal(size=(1, 2, 2)) * 5
            if abs(np.linalg.det(poses[0])) < .2:
                continue
            areas = np.array([.005])
            materials = np.array([[10000., 7000., 0.]])
            arguments = (positions, poses, areas, materials)
            _, _, hessian = membrane_element_derivatives(*arguments)
            numerical = np.zeros((9, 9))
            for coordinate in range(9):
                offset = np.zeros_like(positions)
                offset.ravel()[coordinate] = 1e-7
                positive = membrane_element_derivatives(positions + offset, poses, areas, materials)[1]
                negative = membrane_element_derivatives(positions - offset, poses, areas, materials)[1]
                numerical[:, coordinate] = (positive - negative)[0] / 2e-7
            self.assertLess(np.linalg.norm(numerical - hessian[0]) / np.linalg.norm(numerical), 1e-7)
            rotation = np.linalg.qr(generator.normal(size=(3, 3)))[0]
            transform = np.kron(np.eye(3), rotation.T)
            for project in (False, True):
                energy, gradient, hessian = membrane_element_derivatives(*arguments, project_psd=project)
                moved = membrane_element_derivatives(
                    positions @ rotation + np.array([3, -4, 2]), poses, areas, materials,
                    project_psd=project,
                )
                np.testing.assert_allclose(moved[0], energy, rtol=1e-9, atol=1e-8)
                np.testing.assert_allclose(moved[1][0], transform @ gradient[0], rtol=1e-8, atol=1e-7)
                np.testing.assert_allclose(moved[2][0], transform @ hessian[0] @ transform.T, rtol=1e-8, atol=1e-6)
                np.testing.assert_allclose(hessian[0] @ np.tile(np.eye(3), (3, 1)), 0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
